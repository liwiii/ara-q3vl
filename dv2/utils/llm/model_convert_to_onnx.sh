#!/bin/bash

if [ -z "$1" ]; then
    echo "First argument model path is missing"
else
    echo "First argument is: $1"
fi

if [ -z "$2" ]; then
    echo "Second argument is output directory for saving converted 256 onnx model"
else
    echo "Second argument is: $2"
fi

if [ -z "$3" ]; then
    echo "Third argument is output directory for saving simplfied onnx"
else
    echo "Third argument is: $3"
fi

optimum-cli export onnx --model $1 $2 --trust-remote-code --task=text-generation --sequence_length 256 --batch_size 1

python /opt/onnxsim_large_model/simplify_large_onnx.py -m $2/model.onnx -o $3 --input_shape '{"input_ids":[1, 256], "attention_mask":[1, 256], "position_ids":[1,256]}'
