"""Compare the original Qwen3-VL model and its Mode 1 language ONNX on CPU."""
import argparse
import gc
import hashlib
import json
import platform
import random
import string
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import transformers
from PIL import Image, ImageDraw
from transformers import AutoProcessor, GenerationConfig, Qwen3VLForConditionalGeneration

HERE = Path(__file__).resolve().parent
SEEDS = (0, 2026)
LIMITS = {"max_abs": 0.002, "rmse": 0.0002, "min_cosine": 0.999999}
MAX_TOKENS = 8


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    ort.set_seed(seed)


def sample(processor, case):
    images = []
    content = []
    if "shapes" in case:
        image = Image.new("RGB", (384, 384), "white")
        draw = ImageDraw.Draw(image)
        for shape, box, color in case["shapes"]:
            getattr(draw, shape)(box, fill=color)
        images.append(image)
        content.append({"type": "image"})
    content.append({"type": "text", "text": case["prompt"]})
    prompt = processor.apply_chat_template([{"role": "user", "content": content}],
                                           tokenize=False, add_generation_prompt=True)
    return processor(text=[prompt], images=images or None, return_tensors="pt")


def capture_prefill(feed, kwargs):
    # Capture only the inputs supplied by the original visual tower and embeddings.
    # ORT computes every language layer, the LM head and its own 36 KV pairs.
    emb = kwargs["inputs_embeds"]
    feed["inputs_embeds"] = emb.numpy().copy()
    feed["position_ids"] = kwargs["position_ids"].numpy().copy()
    feed["attention_mask"] = kwargs["attention_mask"].numpy().copy()
    deep = kwargs.get("deepstack_visual_embeds")
    for i in range(3):
        dense = torch.zeros_like(emb)
        if deep is not None:
            dense[kwargs["visual_pos_masks"]] = deep[i]
        feed[f"deepstack_{i}"] = dense.numpy()


@torch.inference_mode()
def reference(model, processor, cases):
    net = Qwen3VLForConditionalGeneration.from_pretrained(
        model, local_files_only=True, dtype=torch.float32, attn_implementation="eager").eval()
    cfg = GenerationConfig(do_sample=False, num_beams=1, max_new_tokens=MAX_TOKENS,
                           use_cache=True, eos_token_id=net.generation_config.eos_token_id,
                           pad_token_id=net.generation_config.pad_token_id)
    refs = []
    for case in cases:
        batch = sample(processor, case)
        feed = {}

        def hook(module, args, kwargs):
            if not feed:
                capture_prefill(feed, kwargs)

        handle = net.model.language_model.register_forward_pre_hook(hook, with_kwargs=True)
        runs = []
        try:
            for seed in SEEDS:
                seed_all(seed)
                gen = net.generate(**batch, generation_config=cfg, use_model_defaults=False,
                                   do_sample=False, return_dict_in_generate=True,
                                   output_logits=True)
                ids = gen.sequences[0, batch.input_ids.shape[1]:].tolist()
                logits = np.stack([step[0].numpy() for step in gen.logits])
                assert ids == logits.argmax(1).tolist(), "Expected unprocessed greedy generation"
                runs.append((ids, logits))
        finally:
            handle.remove()
        ids, scores = runs[0]
        refs.append(dict(feed=feed, ids=ids, logits=scores, repeat_ids=runs[1][0],
                         seed_max_abs=float(np.max(np.abs(scores-runs[1][1])))
                         if ids == runs[1][0] else None,
                         next_pos=int(feed["position_ids"].max()) + 1,
                         grid=batch.image_grid_thw.tolist() if "image_grid_thw" in batch else None))
        print("Transformers", case["id"], ids, flush=True)
    return refs, net.get_input_embeddings().weight.detach(), cfg.eos_token_id


def new_session(path, threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.use_deterministic_compute = True
    opts.add_session_config_entry("session.disable_prepacking", "1")
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def ort_generate(sess, ref, embedding, eos, teacher=None):
    feed = ref["feed"].copy()
    for inp in sess.get_inputs():
        if inp.name.startswith("past_key_values."):
            feed[inp.name] = np.empty((1, inp.shape[1], 0, inp.shape[3]), np.float32)
    outputs = [o.name for o in sess.get_outputs()]
    assert outputs[0] == "logits" and all(o.startswith("present.") for o in outputs[1:])
    scores, ids = [], []
    length = feed["attention_mask"].shape[1]
    for step in range(len(teacher) if teacher is not None else MAX_TOKENS):
        logits, *kv = sess.run(outputs, feed)
        last = logits[0, -1].copy()
        token = int(last.argmax())
        ids.append(token)
        scores.append(last)
        if teacher is None and token in eos:
            break
        token_in = teacher[step] if teacher is not None else token
        emb = embedding[token_in].numpy().reshape(1, 1, -1)
        feed = dict(inputs_embeds=emb, attention_mask=np.ones((1, length+step+1), np.int64),
                    position_ids=np.full((3, 1, 1), ref["next_pos"] + step, np.int64))
        for i in range(3):
            feed[f"deepstack_{i}"] = np.zeros_like(emb)
        for name, value in zip(outputs[1:], kv):
            feed[name.replace("present.", "past_key_values.", 1)] = value
    return ids, np.stack(scores)


def metrics(ref, actual):
    a, b = ref.astype(np.float64), actual.astype(np.float64)
    diff = a-b
    return dict(top1_equal=bool(a.argmax() == b.argmax()),
                max_abs=float(np.abs(diff).max()), rmse=float(np.sqrt(np.mean(diff**2))),
                cosine=float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))))


def compare_case(case, ref, sess, embedding, eos, tokenizer):
    ort_runs = []
    for seed in SEEDS:
        seed_all(seed)
        ort_runs.append(ort_generate(sess, ref, embedding, eos))
    ids, logits = ort_runs[0]
    # If the paths diverge, replay HF tokens to keep all numerical comparisons aligned.
    if ids != ref["ids"]:
        _, logits = ort_generate(sess, ref, embedding, eos, teacher=ref["ids"])
    steps = [dict(step=i, phase="decode" if i else "prefill", **metrics(a, b))
             for i, (a, b) in enumerate(zip(ref["logits"], logits, strict=True))]
    hf_text, ort_text = [tokenizer.decode(x, skip_special_tokens=True) for x in (ref["ids"], ids)]
    def correct(text):
        return text.strip().strip(string.punctuation).strip().casefold() == case["answer"]
    ort_gap = (float(np.max(np.abs(ort_runs[0][1]-ort_runs[1][1])))
               if ids == ort_runs[1][0] else None)
    return dict(id=case["id"], expected=case["answer"], prompt_tokens=ref["feed"]["inputs_embeds"].shape[1],
                image_grid_thw=ref["grid"], hf_text=hf_text, onnx_text=ort_text,
                hf_correct=correct(hf_text), onnx_correct=correct(ort_text),
                hf_ids=ref["ids"], onnx_ids=ids, sequence_equal=ids == ref["ids"],
                hf_repeat_ids=ref["repeat_ids"], onnx_repeat_ids=ort_runs[1][0],
                hf_seed_max_abs=ref["seed_max_abs"], onnx_seed_max_abs=ort_gap, steps=steps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("onnx_model", type=Path)
    parser.add_argument("--results", type=Path, default=HERE / "results.json")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cases_path = HERE / "cases.json"
    cases = json.loads(cases_path.read_text())
    model = args.checkpoint.resolve()
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    refs, embedding, eos = reference(model, processor, cases)
    gc.collect()
    sess = new_session(args.onnx_model, args.threads)
    rows = []
    for case, ref in zip(cases, refs, strict=True):
        row = compare_case(case, ref, sess, embedding, eos, processor.tokenizer)
        rows.append(row)
        print("ONNX", row["id"], row["onnx_ids"], row["steps"][0], flush=True)
    steps = [step for row in rows for step in row["steps"]]
    summary = dict(cases=len(rows), positions=len(steps), top1_matches=sum(s["top1_equal"] for s in steps),
                   identical_sequences=sum(r["sequence_equal"] for r in rows),
                   hf_correct=sum(r["hf_correct"] for r in rows), onnx_correct=sum(r["onnx_correct"] for r in rows),
                   worst_max_abs=max(s["max_abs"] for s in steps), worst_rmse=max(s["rmse"] for s in steps),
                   lowest_cosine=min(s["cosine"] for s in steps),
                   seed_invariant=all(r["hf_seed_max_abs"] == r["onnx_seed_max_abs"] == 0 for r in rows))
    passed = (summary["seed_invariant"] and summary["identical_sequences"] == len(rows)
              and summary["top1_matches"] == len(steps)
              and all(s["max_abs"] <= LIMITS["max_abs"] and s["rmse"] <= LIMITS["rmse"]
                      and s["cosine"] >= LIMITS["min_cosine"] for s in steps))
    graph = onnx.load(args.onnx_model, load_external_data=False)
    ext = {p.value for t in graph.graph.initializer for p in t.external_data if p.key == "location"}
    files = [args.onnx_model, *(args.onnx_model.parent / f for f in sorted(ext)),
             cases_path, *sorted(model.glob("*.json")), *sorted(model.glob("*.safetensors"))]
    result = dict(passed=passed, thresholds=LIMITS, summary=summary, seeds=SEEDS,
                  max_new_tokens=MAX_TOKENS, eos_token_ids=eos, dtype="float32",
                  decoding="greedy", torch_attention="eager", threads=args.threads,
                  platform=platform.platform(), python=platform.python_version(), numpy=np.__version__,
                  torch=torch.__version__, transformers=transformers.__version__, onnx=onnx.__version__,
                  onnxruntime=ort.__version__, providers=sess.get_providers(),
                  sha256={p.name: digest(p) for p in files}, samples=rows)
    args.results.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print("PASS" if passed else "FAIL", summary)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
