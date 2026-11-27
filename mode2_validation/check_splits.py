"""Run the emitted Caffe graphs on the Mode 1 fixtures, on matched token prefixes."""
import argparse
import json
from pathlib import Path

import caffe
import numpy as np

MAX_ABS, RMSE, MIN_COS = 0.002, 0.0002, 0.999999


def scale(a, axis):
    shape = list(a.shape)
    shape[axis:axis + 1] = [shape[axis] // 64, 64]
    maxima = np.abs(a.reshape(shape)).max(axis=axis + 1)
    return np.where(maxima > 0, 6 - np.floor(np.log2(np.maximum(maxima, 1e-30))), 0)


def data(net, name):
    return net.blobs[name].data


def load(root, stage, i):
    base = "lmhead" if i == 36 else f"split_{i}"
    path = root / stage / f"split_{i}"
    weights = path / "lmhead.caffemodel" if i == 36 else root / "caffemodels" / (base + ".caffemodel")
    return caffe.Net(str(path / (base + ".prototxt")), str(weights), caffe.TEST)


def interface(net):
    return {role: {name: list(data(net, name).shape) for name in names}
            for role, names in (("inputs", net.inputs), ("outputs", net.outputs))}


def put(net, name, rows):
    target = data(net, name)
    target.fill(0)
    target.reshape(-1, target.shape[-1])[:len(rows)] = rows
    data(net, name + "_scale")[:] = scale(target, 3)


def offset_softmax(prompt, decode, start, count):
    """Evaluate second-chunk query rows with the emitted token softmax and absolute KV length."""
    names = list(decode.bottom_names["attention_softmax"])
    scores, scales, length = [data(decode, name) for name in names]
    capacity = scores.shape[-1]
    assert start + count <= capacity
    for name in ("attention_softmax", "attention_softmax_scale"):
        data(prompt, name).fill(0)
    for q in range(count):
        scores[:] = data(prompt, "QxKt")[:, :, q:q+1, :capacity]
        scales[:] = data(prompt, "QxKt_scale")[:, :, q:q+1, :scales.shape[-1]]
        length.fill(start+q+1)
        decode.forward(start="attention_softmax", end="attention_softmax")
        for name in ("attention_softmax", "attention_softmax_scale"):
            output = data(decode, name)
            data(prompt, name)[:, :, q:q+1, :output.shape[-1]] = output


def block(net, x, rope, deep, start, keys, values, prompt, mask_net=None):
    length = len(x)
    end = start + length
    put(net, "input_ids_embedded", x)
    if "deepstack_input_ids_embedded" in net.inputs:
        put(net, "deepstack_input_ids_embedded", deep)
    pos = data(net, "position_ids_embedded").reshape(-1, 256)
    pos.fill(0)
    pos[:length] = rope
    data(net, "valid_till").fill(end)
    if not prompt:
        base = start // 64 * 64
        v = data(net, "past_values_requant")
        v[:] = values[base:base+64].reshape(v.shape)
        data(net, "past_values_requant_scale")[:] = scale(v, 1)

    # The split exposes the current K/V before its attention reads the external cache.
    # Perform the host-side cache write between those two parts of the same graph.
    net.forward(end="v_proj_scale_reshape_permute" if prompt else "v_proj_scale_reshape")
    if prompt:
        k = data(net, "k_norm_reshape_permute_rope_reshape_permute")
        v = data(net, "v_proj_reshape_permute")
        values[start:end] = v.reshape(-1, 1024)[:length]
    else:
        k = data(net, "k_norm_permute_rope")
        # In the SDK float executor Requant passes its value block through.
        # Stage the new V in the correct slot of the host-managed 64-row block.
        v.reshape(64, 1024)[start % 64] = data(net, "v_proj_reshape").reshape(1024)
        data(net, "past_values_requant_scale")[:] = scale(v, 1)
        names = list(net._layer_names)
        net.forward(start=names[names.index("v_proj_scale_reshape")+1], end="requant")
        values[base:base+64] = data(net, "requant").reshape(64, 1024)
    keys[start:end] = k.reshape(-1, 1024)[:length]
    for name, cache, axis in [("past_keys", keys, 3), ("past_values", values, 2)]:
        target = data(net, name)
        target.fill(0)
        target.reshape(-1, 1024)[:end] = cache[:end]
        data(net, name + "_scale")[:] = scale(target, axis)
    if mask_net is not None and start:
        net.forward(start="q_proj", end="QxKt")
        offset_softmax(net, mask_net, start, length)
        net.forward(start="attention_softmax_permute")
    else:
        net.forward(start="q_proj")
    name = "deepstack_residual" if "deepstack_input_ids_embedded" in net.inputs else "mlp_residual"
    return data(net, name).reshape(-1, 2560)[:length].copy()


def measure(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    return dict(max_abs=float(np.max(np.abs(a-b))), rmse=float(np.sqrt(np.mean((a-b)**2))),
                cosine=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),
                top1=bool(a.argmax() == b.argmax()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("llm", type=Path)
    p.add_argument("fixtures", type=Path)
    p.add_argument("results", type=Path)
    p.add_argument("--position-aware", action="store_true",
                   help="Diagnostic with prompt-specific RoPE and absolute-position causal softmax for later chunks")
    args = p.parse_args()
    root = args.llm
    table = np.memmap(root / "assets/embedding_table_float.bin", np.float32, "r").reshape(-1,2560)
    rope = np.fromfile(root / "assets/rope_table_float.bin", np.float32).reshape(-1,256)
    cases = []
    for f in sorted(args.fixtures.glob("*.npz")):
        with np.load(f) as archive:
            case = dict(archive)
        ids = case["tokens"]
        x = table[ids].copy()
        mask = ids == case["image_token_id"]
        x[mask] = case["inputs_embeds"][0, mask]
        np.testing.assert_array_equal(x, case["inputs_embeds"][0])
        case.update(name=f.stem, prefill=x, decode=table[case["hf_ids"][:-1]].copy(), masks=[])
        cases.append(case)
    assert len(cases) == 5
    interfaces = {}
    for i in range(36):
        prompt = load(root, "prmpt_prc_128", i)
        decode = load(root, "tkn_gen_1024", i)
        assert data(prompt, "past_keys").shape == data(decode, "past_keys").shape == (1,1,1024,1024)
        for tag, net in [("prefill_0-2" if i < 3 else "prefill_3-35", prompt), ("decode_0-35", decode)]:
            actual = interface(net)
            assert interfaces.setdefault(tag, actual) == actual
        for case in cases:
            x, d = case["prefill"], case["decode"]
            positions = case["expected_rope"] if args.position_aware else rope
            slots = ((len(x) + len(d) + 63) // 64) * 64
            keys = np.zeros((slots,1024), np.float32)
            values = np.zeros_like(keys)
            for start in range(0,len(x),128):
                chunk = slice(start,start+128)
                deep = case[f"deepstack_{i}"][0,chunk] if i < 3 else None
                x[chunk] = block(prompt,x[chunk],positions[start:start+len(x[chunk])],deep,start,keys,values,True,
                                 mask_net=decode if args.position_aware else None)
                if i == 0:
                    softmax = data(prompt, "attention_softmax")[0, 0]
                    for q in sorted({0, len(x[chunk])-1}):
                        active = np.flatnonzero(softmax[q])
                        case["masks"].append(dict(query=start+q, last_nonzero_key=int(active[-1]) if active.size else None,
                                                  visible_keys=int(active.size)))
            for t in range(len(d)):
                start = len(x)+t
                d[t:t+1] = block(decode,d[t:t+1],positions[start:start+1],None,start,keys,values,False)
        del prompt, decode
        print("completed layer", i, flush=True)
    head = load(root, "lmhead", 36)
    interfaces["lmhead_36"] = interface(head)
    rows = []
    for case in cases:
        steps = []
        for j, x in enumerate(np.vstack((case["prefill"][-1:], case["decode"]))):
            put(head, "input_ids_embedded", x[None])
            head.forward()
            logits = data(head, "lm_head").ravel()
            steps.append(dict(phase="decode" if j else "prefill", prefix_generated=case["hf_ids"][:j].tolist(),
                              caffe_id=int(logits.argmax()), hf_id=int(case["hf_ids"][j]), ort_id=int(case["ort_ids"][j]),
                              vs_hf=measure(case["hf_logits"][j],logits), vs_onnx=measure(case["ort_logits"][j],logits)))
        length = len(case["prefill"]) + len(case["decode"])
        rows.append(dict(id=case["name"], prompt_tokens=len(case["tokens"]), steps=steps,
                         layer0_head0_mask=case["masks"],
                         fixed_table_max_rope_error=float(np.max(np.abs(rope[:length]-case["expected_rope"][:length])))))
    metrics = [s["vs_onnx"] for row in rows for s in row["steps"]]
    against_both = [s[ref] for row in rows for s in row["steps"] for ref in ("vs_hf","vs_onnx")]
    passed = all(m["top1"] and m["max_abs"] <= MAX_ABS and m["rmse"] <= RMSE and
                 m["cosine"] >= MIN_COS for m in against_both)
    result = dict(passed=passed, position_aware_diagnostic=args.position_aware,
                  mode="Caffe CPU FP32 qdq=0, reference-token prefixes, split-native KV",
                  prefill_size=128, prefill_capacity=1024, decode_capacity=1024, interfaces=interfaces,
                  thresholds=dict(max_abs=MAX_ABS,rmse=RMSE,min_cosine=MIN_COS),
                  top1_matches_onnx=sum(m["top1"] for m in metrics), positions=len(metrics), samples=rows)
    args.results.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("PASS" if passed else "FAIL", result["top1_matches_onnx"], "/", len(metrics))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
