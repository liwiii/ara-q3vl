#!/bin/bash
set -eu
export USE_TF=0
# Use the compiler image's torch, ORT and transformers. Install just the
# ONNX IR builder dependency into this disposable container.
python -m pip install --disable-pip-version-check --no-deps --target /tmp/qwen3-export-deps onnx-ir==1.1.0
export PYTHONPATH="/tmp/qwen3-export-deps:/dv2/utils:/dv2/utils/llm_compiler/convert_to_onnx_llm"
exec python /dv2/utils/qwen3_vl_4b/language/export.py "$@"
