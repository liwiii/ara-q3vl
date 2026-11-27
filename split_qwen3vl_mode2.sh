#!/usr/bin/env bash
# Usage: ./split_qwen3vl_mode2.sh MODE1_CONFIG OUTPUT_DIR
set -euo pipefail
config=$(realpath "${1:?Pass the Mode 1 kinara_run_config.yaml}")
mkdir -p "${2:?Pass an output directory for the 1024 language splits}"
out=$(cd -- "$2" && pwd -P)
root=$(cd -- "$(dirname -- "$0")" && pwd -P)
paths=$(python3 - "$config" "$out" <<'PY'
import sys
from pathlib import Path
import yaml
source, out = map(Path, sys.argv[1:])
cfg = yaml.safe_load(source.read_text())
cfg.update(out=str(out), prmpt_context_size=1024, prmpt_prc_sizes=[128], tkn_gen_sizes=[1024])
cfg.pop("vision_out", None)
dest = out / "kinara_run_config.yaml"
if dest == source:
    raise SystemExit("Choose an output directory separate from the Mode 1 configuration.")
dest.write_text(yaml.safe_dump(cfg))
for key in ("model_repo", "onnx_model_repo"):
    print(Path(cfg[key]).resolve())
PY
)
{ read -r model; read -r onnx; } <<< "$paths"
tool=/dv2/utils/qwen3_vl_4b
exec docker run --rm --platform linux/amd64 --user "$(id -u):$(id -g)" --group-add 0 \
  -e HOME=/tmp -e PYTHONPATH=/dv2/python:/dv2/python/caffe/proto:/dv2/utils/llm_compiler:/dv2/utils \
  -v "$model:$model:ro" -v "$onnx:$onnx:ro" -v "$out:$out" -v "$root/ara_docker$tool:$tool:ro" \
  aracompiler:0.2 python "$tool/language/split.py" "$out/kinara_run_config.yaml"
