#!/bin/bash
set -euo pipefail
echo STARTED
MODEL_DIR=""
OUT_DIR=""
TYPE=""
SPECD_SIZE=""
NUM_MCS_SHARDS=""
FORMAT_DIR=""
GGUF_FILE=""
VIDEO_OR_IMAGE=""
MULTIMODAL_STANDARD_COMPILATION=""

while getopts ":m:o:t:s:f:g:r:n:v:" opt; do
  case "$opt" in
    m) MODEL_DIR="$OPTARG" ;;
    o) OUT_DIR="$OPTARG" ;;
    t) TYPE="$OPTARG" ;;
    s) SPECD_SIZE="$OPTARG" ;;    
    n) NUM_MCS_SHARDS="$OPTARG" ;;
    f) FLOAT_DIR="$OPTARG" ;;
    g) GGUF_FILE="$OPTARG" ;;
    r) VIDEO_OR_IMAGE="$OPTARG" ;;
    v) MULTIMODAL_STANDARD_COMPILATION="$OPTARG" ;;
    :) echo "Option -$OPTARG requires an argument" >&2; exit 2 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; exit 2 ;;
  esac
done
shift $((OPTIND - 1))

# Validate required args
: "${MODEL_DIR:?Missing -m}"
: "${OUT_DIR:?Missing -o}"
: "${TYPE:?Missing -t}"
: "${SPECD_SIZE:?Missing -t}"
: "${NUM_MCS_SHARDS:?Missing -n}"
: "${FLOAT_DIR:?Missing -f}"
: "${GGUF_FILE:?Missing -g}"
: "${MULTIMODAL_STANDARD_COMPILATION:?Missing -v}"

if [[ -z "$MULTIMODAL_STANDARD_COMPILATION" || "$MULTIMODAL_STANDARD_COMPILATION" =~ ^([Tt][Rr][Uu][Ee]|[Ff][Aa][Ll][Ss][Ee])$ ]]; then
    :   # no-op (do nothing)
else
    echo "Invalid MULTIMODAL_STANDARD_COMPILATION (-v): $MULTIMODAL_STANDARD_COMPILATION. Expected empty or True/False case insensitive"
    exit 1
fi
if [[ ! "$MULTIMODAL_STANDARD_COMPILATION" =~ ^([Tr][Rr][Uu][Ee])$ ]]; then
    # MULTIMODAL_STANDARD_COMPILATION: Anything other than True/true (case insensitive) considered False".
    MULTIMODAL_STANDARD_COMPILATION=""  # Clear it for optimal arg setting to parse_model
fi

/usr/bin/python3.12 -m venv /dv2/utils/llm_compiler/convert_to_onnx_llm/llm_env 
source /dv2/utils/llm_compiler/convert_to_onnx_llm/llm_env/bin/activate
export PYTHONPATH=/dv2/utils/llm_compiler/convert_to_onnx_llm/llm_env/lib/python3.12/site-packages
export C_INCLUDE_PATH=/usr/local/include/python3.12
export CPLUS_INCLUDE_PATH=/usr/local/include/python3.12
/usr/bin/python3.12  -V
pip install -U setuptools
pip install -r /dv2/utils/llm_compiler/convert_to_onnx_llm/requirements.txt

# Note: gptqmodel versions beyond 4.2.5 refer to later versions of pypcre,
# which may refer to 32 bit libraries on 64 bit machines.
# if customers face this issue, we should freeze gptqmodel version in the
# line below.
pip install -v "gptqmodel==5.4.2" "torchao==0.17.0" --no-build-isolation
echo 'LLM venv ready at /dv2/utils/llm_compiler/convert_to_onnx_llm/llm_env'

cd /dv2/utils/llm_compiler/convert_to_onnx_llm/

OPTIONAL_ARGS=()
if [[ -n "$MULTIMODAL_STANDARD_COMPILATION" ]]; then
    OPTIONAL_ARGS+=("-v")
fi

echo "/usr/bin/python3.12 parse_model.py -m $MODEL_DIR -o $OUT_DIR -t $TYPE -f $FLOAT_DIR -g $GGUF_FILE -r $VIDEO_OR_IMAGE -s $SPECD_SIZE -n $NUM_MCS_SHARDS ${OPTIONAL_ARGS[@]}"
/usr/bin/python3.12 parse_model.py -m "$MODEL_DIR" -o "$OUT_DIR" -t "$TYPE" -f "$FLOAT_DIR" -g "$GGUF_FILE" -r "$VIDEO_OR_IMAGE" -s "$SPECD_SIZE" -n "$NUM_MCS_SHARDS" ${OPTIONAL_ARGS[@]}