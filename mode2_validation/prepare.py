"""Build identical inputs and the two full-vocabulary FP32 references for the five cases."""
import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen3VLConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mode1_validation"))
from compare import HERE, new_session, ort_generate, reference, sample

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("onnx", type=Path)
    p.add_argument("fixtures", type=Path)
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    cases = json.loads((HERE / "cases.json").read_text())
    proc = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    refs, embedding, eos = reference(args.model, proc, cases)
    gc.collect()
    sess = new_session(args.onnx, 4)
    cfg = Qwen3VLConfig.from_pretrained(args.model, local_files_only=True)
    rotary = Qwen3VLTextRotaryEmbedding(cfg.text_config)
    args.fixtures.mkdir(parents=True, exist_ok=True)
    for case, ref in zip(cases, refs, strict=True):
        ids, logits = ort_generate(sess, ref, embedding, eos, teacher=ref["ids"])
        tokens = sample(proc, case).input_ids.numpy().ravel()
        positions = ref["feed"]["position_ids"]
        # Extra rows allow checking the fixed IMAGE table during decode as well.
        tail = np.broadcast_to(np.arange(ref["next_pos"], ref["next_pos"]+8), (3,1,8))
        cos, sin = rotary(torch.empty(1), torch.from_numpy(np.concatenate([positions, tail], axis=2)))
        np.savez(args.fixtures / (case["id"] + ".npz"), **ref["feed"], tokens=tokens,
                 hf_ids=ref["ids"], ort_ids=ids, hf_logits=ref["logits"], ort_logits=logits,
                 expected_rope=torch.cat([cos, sin], -1)[0].numpy(), image_token_id=cfg.image_token_id)
        print("wrote", case["id"], flush=True)
