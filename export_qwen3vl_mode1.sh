#!/usr/bin/env bash
# Export the Qwen3-VL-4B-Instruct language graph for Ara Mode 1.
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 CHECKPOINT_DIR OUTPUT_DIR" >&2
  exit 2
fi

sdk=$(cd -- "$(dirname -- "$0")" && pwd -P)
model=$(cd -- "$1" && pwd -P)
mkdir -p -- "$2"
out=$(cd -- "$2" && pwd -P)
tool=/dv2/utils/qwen3_vl_4b

# Keep host paths in the generated kinara_run_config.yaml for Mode 2/3.
# Group 0 has read access to /dv2; write the export as the host user.
exec docker run --rm --platform linux/amd64 --user "$(id -u):$(id -g)" --group-add 0 \
  -e HOME=/tmp \
  -v "$model:$model:ro" -v "$out:$out" -v "$sdk/ara_docker$tool:$tool:ro" \
  aracompiler:0.2 bash "$tool/language/export.sh" "$model" "$out" IMAGE 1
