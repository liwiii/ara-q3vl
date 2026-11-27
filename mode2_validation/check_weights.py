"""Check all emitted language parameters against the ONNX initializers, including layout."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import onnx


def array(tensor, directory):
    ext = {x.key: x.value for x in tensor.external_data}
    if ext:
        return np.memmap(directory / ext["location"], "<f4", "r",
                         offset=int(ext.get("offset", 0)), shape=tuple(tensor.dims))
    return onnx.numpy_helper.to_array(tensor)


def source_name(i, layer):
    prefix = f"model.layers.{i}."
    if i == 36:
        return "lm_head.MatMul.weight" if layer == "lm_head" else prefix + "final_norm_layernorm.weight"
    if layer in ("input_layernorm", "post_attention_layernorm"):
        return prefix + layer + ".weight"
    if layer in ("q_norm", "k_norm"):
        return prefix + "attn." + layer + ".layernorm.weight"
    group = "mlp." if layer in ("up_proj", "down_proj", "gate_proj") else "attn."
    return prefix + group + layer + ".MatMul.weight"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for a in ("onnx", "llm", "result"):
        p.add_argument(a, type=Path)
    args = p.parse_args()
    graph = onnx.load(args.onnx, load_external_data=False).graph
    weights = {t.name: t for t in graph.initializer}
    checked, zero_biases, differences = [], 0, []
    for i in range(37):
        path = args.llm / ("lmhead/split_36/lmhead.caffemodel" if i == 36 else f"caffemodels/split_{i}.caffemodel")
        with h5py.File(path) as h:
            for layer, params in h["data"].items():
                if "0" not in params:
                    continue
                name = source_name(i, layer)
                expected = array(weights[name], args.onnx.parent).T
                target = params["0"]
                assert expected.size == target.size
                for start in range(0,len(target),2048):
                    chunk = expected[start:start+2048]
                    if not np.array_equal(chunk, target[start:start+2048].reshape(chunk.shape)):
                        differences.append(name)
                        break
                checked.append(name)
                if "1" in params:
                    zero_biases += 1
                    if np.count_nonzero(params["1"][:]):
                        differences.append(name + ": nonzero extra bias")
    expected_names = {t for t in weights if t.startswith("model.layers.") or t == "lm_head.MatMul.weight"}
    missing = sorted(expected_names - set(checked))
    result = dict(passed=not (missing or differences), files=37, parameters=len(checked), zero_biases_checked=zero_biases,
                  missing=missing, differences=differences)
    args.result.write_text(json.dumps(result,indent=2)+"\n")
    print(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
