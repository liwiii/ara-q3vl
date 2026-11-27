#!/bin/bash
set -euo pipefail
echo STARTED
MODEL_DIR=""
OUT_DIR=""


while getopts ":m:o:t:f:g:r:n::" opt; do
  case "$opt" in
    m) MODEL_DIR="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    :) echo "Option -$OPTARG requires an argument" >&2; exit 2 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

# Validate required args
: "${MODEL_DIR:?Missing -m}"
: "${OUT_DIR:?Missing -o}"
/usr/bin/python3.12 -m venv /dv2/utils/qwen2_5vl_3b/vision_encoder/llm_env 
source /dv2/utils/qwen2_5vl_3b/vision_encoder/llm_env/bin/activate
export PYTHONPATH=/dv2/utils/qwen2_5vl_3b/vision_encoder/llm_env/lib/python3.12/site-packages
export C_INCLUDE_PATH=/usr/local/include/python3.12
export CPLUS_INCLUDE_PATH=/usr/local/include/python3.12
/usr/bin/python3.12 -V
pip install -r /dv2/utils/llm_compiler/convert_to_onnx_llm/requirements.txt

# Note: gptqmodel versions beyond 4.2.5 refer to later versions of pypcre,
# which may refer to 32 bit libraries on 64 bit machines.
# if customers face this issue, we should freeze gptqmodel version in the
# line below.
# pip install -v gptqmodel --no-build-isolation 
echo 'LLM venv ready at /dv2/utils/qwen2_5vl_3b/vision_encoder/llm_env'

cd /dv2/utils/qwen2_5vl_3b/vision_encoder/
/usr/bin/python3.12 extract_embed_tokens_safetensors_to_pt.py --model_path "$MODEL_DIR" --output-path "$OUT_DIR"