#!/usr/bin/env python3

'''
Copyright (c) 2025, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''


import onnx
import onnxruntime
import numpy as np
import os, argparse

def adjust_and_store(wt_array, scale, offset, output_file, outputdir):
    adjusted_wt_array = np.floor((wt_array / scale) + offset + 0.5)
    clamped_wt_array = np.clip(adjusted_wt_array, -128, 127)
    dir = outputdir
    if not os.path.exists(dir):
        # Create the directory
        os.makedirs(dir)
    ofile = os.path.join(dir,output_file)
    clamped_wt_array.astype(np.int8).tofile(ofile)

def run_inference(model_path, input_data):
    ort_session = onnxruntime.InferenceSession(model_path)
    output_names = [n.name for n in ort_session.get_outputs()]
    outputs = ort_session.run(output_names, input_data)
    return dict(zip(output_names, outputs))


def run(args):
    try:
        model_path = args.inet
        model = onnx.load(model_path)


        position_ids_input = np.arange(2048).reshape((1, 2048)).astype(np.int64)
        dummy_input_ids = np.zeros((1, 2048)).astype(np.int64)
        dummy_attention_mask = np.ones((1, 2048)).astype(np.int64)


        input_data = {
            "position_ids": position_ids_input,
            "input_ids": dummy_input_ids,
            "attention_mask": dummy_attention_mask
        }
        print("Starting inference")
        sin_val_info = [i for i in model.graph.value_info if i.name =='/model/rotary_emb/Sin_output_0']
        cos_val_info = [i for i in model.graph.value_info if i.name =='/model/rotary_emb/Cos_output_0']
        model.graph.output.extend(sin_val_info)
        model.graph.output.extend(cos_val_info)
        tmp_onnx = os.path.join(args.output_dir,"tmp.onnx")
        onnx.save(model, tmp_onnx,all_tensors_to_one_file=False, save_as_external_data=True)
        outputs = run_inference(tmp_onnx, input_data)
        print("Inference done!!!")
        # breakpoint()
        sin_output = outputs["/model/rotary_emb/Sin_output_0"]
        cos_output = outputs["/model/rotary_emb/Cos_output_0"]
        rope_output = np.concatenate((cos_output, sin_output), axis=2)
        scale = args.scale
        offset = args.offset

        adjust_and_store(rope_output, scale, offset, args.rope_file_name, os.path.join(args.output_dir))

        print("Done")
    except Exception as e:
        raise Exception("sin_cos failed with " + str(e))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inet', required=True, help='path to onnx file')
    parser.add_argument('--output_dir', required=True, help='path to llama output dir')
    parser.add_argument('--scale', type=float,required=False, help='scale value')
    parser.add_argument('--offset', type=float, required=False, help='offset value')
    parser.add_argument('--rope_file_name',required=False, help='rope file to dump with')
    # global args
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
