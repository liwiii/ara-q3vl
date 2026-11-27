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

import os
import argparse
import glob
import json
import shutil

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/dv2/python/caffe')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/dv2/python/caffe/proto')))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../llm_compiler')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../llm_compiler/proto_utils')))

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


print(f"sys.path: {sys.path}")

import caffe

from llm_compiler.caffe_loader import CaffeLoader
from llm_compiler.utils import create_sdk_config, check_for_existence

from pathlib import Path
from safetensors import safe_open

import vision_model_creator
import get_caffemodel

GENERATED_CAFFEMODEL_FILENAME = 'vision.caffemodel'
GENERATED_DVRUN_YAML_CONFIG_NAME = 'dvrun_config.yaml'

PROTOTXT = {
    '3B' : {'rotated' : 'vision_encoder/qwen2_5_vl_vision_3b_rotated_hadamard.prototxt',
            'not_rotated' : 'vision_encoder/qwen2_5_vl_vision_3b.prototxt'},
    '7B' : {'rotated' : 'vision_encoder/qwen2_5_vl_vision_7b_rotated_hadamard.prototxt',
            'not_rotated' : 'vision_encoder/qwen2_5_vl_vision_7b.prototxt'},
}

SO_YAML = {'rotated' : 'vision_encoder/qwen2_5_vl_vision_3b_rotated_hadamard_so.yaml',
           'not_rotated' : 'vision_encoder/qwen2_5_vl_vision_so_no_rotate.yaml'}

def parse_args():
    parser = argparse.ArgumentParser(description="Safetensors to Caffemodel arguments")
    parser.add_argument("-b", '--build', default = '/', required=False, help="build directory")
    parser.add_argument("-i","--input_safetensors_dir", type=str, required=True, help="Directory containing the safetensors")
    parser.add_argument("-o","--output_dir", type=str, required=True, help="Output directory for saving the caffemodel and dvrun_config file")
    return parser.parse_args()


def generate_safetensor_to_kinara_weight_name_map(
    safetensors_to_kinara_weight_names_map_file: str, model_is_rotated:bool):
  vision_model = vision_model_creator.make_vision_model(is_rotated=model_is_rotated)
  vision_model.update_layer_names()

  # Port vision_model.get_dict() into a .json so that it is both
  # human readable and also python/c++ readable later
 
  safetensor_to_kinara_weight_name_map = vision_model.get_dict()

  print(f"Writing safetensors-->Kinara weight name map to {safetensors_to_kinara_weight_names_map_file}")
  with open(safetensors_to_kinara_weight_names_map_file, "w") as json_file:
    json.dump(safetensor_to_kinara_weight_name_map, json_file, indent=4)

def obtain_safetensor_to_kinara_weight_name_map(
    safetensors_to_kinara_weight_names_map_file: str, model_is_rotated:bool):
    check_for_existence(safetensors_to_kinara_weight_names_map_file)
    # Write to JSON file
    generate_safetensor_to_kinara_weight_name_map(safetensors_to_kinara_weight_names_map_file, model_is_rotated)
    weights_name_map = []
    with open(safetensors_to_kinara_weight_names_map_file, "r") as file:
        weights_name_map = json.load(file)

    return weights_name_map


def convert_safetensors_to_caffe(input_safetensors_path:str, build_dir:str, output_dir:str, model_is_rotated:bool, param_size_str:str):
    output_caffemodel_path=os.path.join(output_dir, GENERATED_CAFFEMODEL_FILENAME)
    print(f"output_caffemodel_path: {output_caffemodel_path}")
    print(f"convert_safetensors_to_caffe called for {input_safetensors_path}->{output_caffemodel_path}")
    check_for_existence(output_caffemodel_path)

    safetensors_to_kinara_weight_names_map_file = \
        os.path.join(output_dir, "safetensor_to_kinara_weight_name_map.json")
    weights_name_map = obtain_safetensor_to_kinara_weight_name_map(safetensors_to_kinara_weight_names_map_file, model_is_rotated)

    print(f"weights_name_map loaded with {len(weights_name_map)} weight keys")

    safetensors_fileset = glob.glob(os.path.join(input_safetensors_path, "*.safetensors"))
    print(f"safetensors_fileset:{safetensors_fileset}")

    # Assumes Kinara .prototxt is generated already.
    caffe_loader = CaffeLoader(build_dir)
    caffe = caffe_loader.get_caffe()
    caffe_prototxt_file = PROTOTXT[param_size_str]['rotated' if model_is_rotated else 'not_rotated']

    get_caffemodel.generate_caffe_model(caffe,
                                        weights_name_map,
                                        safetensors_fileset,
                                        os.path.join(output_dir, caffe_prototxt_file),
                                        output_caffemodel_path)

def create_random_analyzer_inputs(build_dir:str, network_file:str, img_path:str):
    caffe_loader = CaffeLoader(build_dir)
    caffe = caffe_loader.get_caffe()
    net = caffe.Net(network_file, caffe.TEST)
    input_dict = {}
    os.makedirs(img_path, exist_ok=True)

    import numpy as np
    for layer_name in net.inputs:
        shape = net.blobs[layer_name].data.shape
        random_analyzer_input_filename = os.path.join(img_path, layer_name + '.bin')
        print(f"create_random_analyzer_inputs: {shape} {random_analyzer_input_filename}")
        if layer_name.endswith('scale'):
            data = np.ones(shape)*3
        elif layer_name == 'rope_table':
            # This will be enabled in future. This is because there is a specific way to grab
            # Rope table for vision encoder (different from Rope table for language component).
            # This python library only handles vision encoder.            
            print("Not generating rope_table, Using existing")
            continue
        else:
            data = np.random.rand(*shape)
        data.astype(dtype=np.float32).tofile(random_analyzer_input_filename)

def create_dvrun_config_for_qwen2_5vl_vision(build_dir:str, output_dir:str, model_is_rotated:bool, param_size_str:str):
    caffe_loader = CaffeLoader(build_dir)
    caffe = caffe_loader.get_caffe()
    caffe_prototxt_file = PROTOTXT[param_size_str]['rotated' if model_is_rotated else 'not_rotated']
    so_yaml_file = SO_YAML['rotated' if model_is_rotated else 'not_rotated']

    # Note: 'create_sdk_config' won't run on internal build.
    create_sdk_config(caffe=caffe,
                      # Path after '/dv2/utils/qwen2_5vl_3b_assets/'
                      network_file=os.path.join(output_dir, caffe_prototxt_file),
                      model_file=os.path.join(output_dir, GENERATED_CAFFEMODEL_FILENAME),
                      img_path=os.path.join(output_dir, 'vision_encoder/inputs'),
                      scale_offset_file=os.path.join(output_dir, so_yaml_file),
                      out_file=os.path.join(output_dir, GENERATED_DVRUN_YAML_CONFIG_NAME))
    print(f"Generated {os.path.join(output_dir, GENERATED_DVRUN_YAML_CONFIG_NAME)}")

def detect_if_safetensors_rotated(input_safetensors_dir:str):
    safetensors_fileset = glob.glob(os.path.join(input_safetensors_dir, '*.safetensors'))
    print(f"safetensors_fileset:{safetensors_fileset}")
        

    for filename in safetensors_fileset:
        with safe_open(filename, framework="pt", device="cpu") as f:
            if 'visual.patch_embed.rotation.weight' in f.keys():
                print(f"visual.patch_embed.rotation present in {filename}")
                return True

    print(f"visual.patch_embed.rotation.weight not found in {safetensors_fileset}")
    return False

def detect_if_safetensors_3b_or_7b(input_safetensors_dir:str):
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(input_safetensors_dir)
    if config.vision_config.out_hidden_size == 3584:
        return '7B'
    elif config.vision_config.out_hidden_size == 2048:
        return '3B'
    else:
        raise ValueError("fUnsupported Qwen 2.5VL model with config.vision_config.out_hidden_size {config.vision_config.out_hidden_size}")

    # Alternative approach is to actually count the number of parameters
    # using model.parameters() and see if they are closed to 3B or 7B.

if __name__ == "__main__":
   args = parse_args()
   print(f"Writing to {args.output_dir}")
   Path(args.output_dir).mkdir(parents=True, exist_ok=True)
   print(f"Copying /dv2/utils/qwen2_5vl_3b_assets to {args.output_dir}")
   shutil.copytree('/dv2/utils/qwen2_5vl_3b_assets', args.output_dir, dirs_exist_ok=True)
   print(f"Copying /dv2/utils/qwen2_5vl_7b_assets to {args.output_dir}")
   shutil.copytree('/dv2/utils/qwen2_5vl_7b_assets', args.output_dir, dirs_exist_ok=True)

   # This will be enabled in future.
   #    create_random_analyzer_inputs(args.build,
   #                                  network_file=os.path.join(args.output_dir, 'vision_encoder/qwen2_5_vl_vision_3b.prototxt'),
   #                                  img_path=os.path.join(args.output_dir, 'vision_encoder/inputs/input0'))

   is_rotated = detect_if_safetensors_rotated(args.input_safetensors_dir)
   param_size_str = detect_if_safetensors_3b_or_7b(args.input_safetensors_dir)

   print(f"safetensors_to_caffe_dvconfig.py: is_rotated: {is_rotated}, param_size_str: {param_size_str}")
   convert_safetensors_to_caffe(args.input_safetensors_dir, args.build, args.output_dir, is_rotated, param_size_str)
   create_dvrun_config_for_qwen2_5vl_vision(args.build, args.output_dir, is_rotated, param_size_str)
   # Now invoke dvrun