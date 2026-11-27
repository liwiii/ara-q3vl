#!/bin/bash
set -euo pipefail
while getopts 'm:o:' opt; do
  case "$opt" in
    m) model=$OPTARG ;;
    o) output=$OPTARG ;;
    *) exit 2 ;;
  esac
done
cd "$(dirname "$0")"
# aracompiler:0.2 ships torch, safetensors and its matching Caffe bindings.
python extract_embed_tokens_safetensors_to_pt.py -i "${model:?missing -m}" -o "${output:?missing -o}"
exec python safetensors_to_caffe_dvconfig.py -i "$model" -o "$output"
