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

import numpy as np
from typing import Tuple
import yaml
import os

def quantize(tensor: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        numpy_array = tensor.astype(np.float32)
        # calculate scale per each group of 64 elements as 7 - log2(max(abs(group)))
        grouped = numpy_array.reshape((int(numpy_array.size / 64)), 64)
        absolute_values = np.abs(grouped)
        max_values_group = np.max(absolute_values, axis=1)
        log_max_group = np.log2(max_values_group)
        scales = np.floor(log_max_group) + 1
        scales = 7 - scales
        scales = scales.astype(np.int8)

        # scale each group by 2**scale
        scales = scales.reshape(-1, 1)
        scaled_array = np.floor(grouped * (np.float_power(2, scales.astype(np.int32))) + 0.5)
        scaled_array = np.clip(scaled_array, -128, 127).astype(np.int8)

        return (scaled_array.flatten(), scales.flatten())

def get_scale_tensor_float(tensor: np.ndarray, shape, scale_shape, group_size: int, group_axis: int):
        numpy_array = tensor.astype(np.float32).reshape(shape)
        # calculate scale per each group of 64 elements as 7 - log2(max(abs(group)))
        accumulated_dim = 1
        for idx in range(group_axis+1, len(shape)):
                accumulated_dim*=shape[idx]
        new_shape = (-1, group_size, accumulated_dim)
        grouped = numpy_array.reshape(new_shape)
        absolute_values = np.abs(grouped)
        max_values_group = np.max(absolute_values, axis=1)
        log_max_group = np.log2(max_values_group)
        scales = np.floor(log_max_group) + 1
        scales = 7 - scales
        return scales.reshape(scale_shape)



def dequantize(quantized : np.ndarray, scales : np.ndarray, shape : tuple)->np.ndarray:
        # divide by 2**scale in order to  dequantize
        q_new = quantized.reshape(int(quantized.size / 64), 64)
        out = scales.reshape(-1, 1)
        q_new = q_new / np.float_power(2, out.astype(np.float16))
        return q_new.reshape(shape)

def create_config(caffe, network_file, model_file, img_path, scale_offset_file, out_file):
        net = caffe.Net(network_file, caffe.TEST)
        input_dict = {}
        for k in net.inputs:  # k is the name of the layer
                shape = net.blobs[k].data.shape
                input_dict[k] = shape
        # Convert tuples to comma-separated strings
        input_shapes = ' '.join([','.join(map(str, shape)) for shape in input_dict.values()])
        # Join keys with spaces for your YAML
        input_node_name = ' '.join(input_dict.keys())
        layer_types_str = ''
        for inp in net.inputs:
                if 'scale' in inp:
                       layer_types_str += f'{inp}:int8,'
                if 'valid_till' in inp:
                       layer_types_str += f'{inp}:uint16,'
        yaml_dict = {
                "network": {
                        "dataset": {
                        "labels": "/auto/share/sw/common/data/classification/imagenet/groundtruth/tags.txt",
                        "quantize": img_path,
                        "raw_image": True,
                        "raw_image_dir": True,
                        "verify": img_path
                        },
                        "ignore_checksums": False,
                        "metric_enabled": False,
                        "model_file": model_file,
                        "network_file": network_file,
                        "src_model_type": "caffe"
                },
                "tools": {
                        "analyzer": {
                        "input_layer_types": layer_types_str,
                        "qmode": 13,
                        "scale_offset_adjust_file": scale_offset_file
                        },
                        "nnconvert": {
                        "dryrun_count": 0,
                        "dummy": True,
                        "input_node_name": input_node_name,
                        "input_shape": input_shapes,
                        "output_node_name": None,
                        "platform_target": "Ara2"
                        }
                }
        }

        with open(out_file, 'w') as yaml_file:
                yaml.dump(yaml_dict, yaml_file, default_flow_style=False, sort_keys=False)
def deep_merge(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1:
            if isinstance(value, dict) and isinstance(dict1[key], dict):
                deep_merge(dict1[key], value)
            else:
                dict1[key] = value
        else:
            dict1[key] = value
    return dict1

def check_for_existence(file_path:str):
    if os.path.exists(file_path):
        timestamp = os.path.getmtime(file_path)
        print(f"Target file already exists: {file_path} from {timestamp}")
        print(f"Renaming it to {file_path + '.old.' + str(timestamp)}")
        os.rename(file_path, file_path + '.old.' + str(timestamp))

def create_sdk_config(caffe, network_file, model_file, img_path, scale_offset_file, out_file, license_key=None):
        net = caffe.Net(network_file, caffe.TEST)
        input_dict = {}
        for k in net.inputs:  # k is the name of the layer
                shape = net.blobs[k].data.shape
                input_dict[k] = shape
        # Convert tuples to comma-separated strings
        input_shapes = ':'.join([','.join(map(str, shape)) for shape in input_dict.values()])
        # Join keys with spaces for your YAML
        input_node_name = ':'.join(input_dict.keys())
        layer_types_str = ''
        for inp in net.inputs:
                if 'scale' in inp:
                       layer_types_str += f'{inp}:int8,'
                if 'valid_till' in inp:
                       layer_types_str += f'{inp}:uint16,'
        yaml_dict = {
                "network": {
                        "srcfw" : "caffe",
                        "images": {
                        "tags": "/dv2/utils/llm/tags.txt",
                        "quantize": img_path,
                        "verify": img_path
                        },
                        "iwt": model_file,
                        "inet": network_file,
                        "dim": input_shapes 
                },
                "dvconvert": {
                        "inode": input_node_name
                },
                "dvnc": {
                        "input_type": layer_types_str,
                        "qmode": 13,
                        "scale_offset_adjust_file": scale_offset_file  
                },
                "out" : os.path.dirname(out_file),
                "license_key": license_key
        }
        # Future improvement: This should set key 'model_name' also for
        # consistency in debugging. Right now it is picked from default
        # /dv2/utils/llm/sample.yaml        

        with open("/dv2/utils/llm/sample.yaml", 'r') as sfile:
                template_yaml_dict = yaml.safe_load(sfile)
        merged_dict = deep_merge(template_yaml_dict.copy(), yaml_dict)
        # template_yaml_dict = {**template_yaml_dict, **config_yaml}
        # dv.sw.utils.utils.write_yaml_file(curr_path+'/config_ara2.yaml', config_yaml)
        check_for_existence(out_file)
        with open(out_file, 'w') as file:
                yaml.dump(merged_dict, file, default_flow_style=False)

def make_inputs(caffe, proto, img_out_dir, token_size=4096):
        from constants import INPUT, PAST_KEYS, PAST_VALUES, PAST_VALUES_REQUANT
        img_dir = os.path.join(img_out_dir, 'chksum0')
        os.makedirs(img_dir, exist_ok=True)
        net = caffe.Net(proto, caffe.TEST)
        input_dict = {}
        for k in net.inputs:
                if 'valid_till' in k:
                        input_dict[k] = np.array([token_size-1], dtype=np.float32)
                        continue
                if 'scale' not in k:
                        input_dict[k] = (np.random.rand(*(net.blobs[k].data.shape)).astype(np.float32) * 2) - 1
                        continue
        for k in net.inputs:
                if INPUT + '_scale' in k:
                        input_dict[k] = get_scale_tensor_float(input_dict[INPUT], input_dict[INPUT].shape, net.blobs[k].data.shape, 64, 3)
                if PAST_KEYS + '_scale' in k:
                        input_dict[k] = get_scale_tensor_float(input_dict[PAST_KEYS], input_dict[PAST_KEYS].shape, net.blobs[k].data.shape, 64, 3)
                if PAST_VALUES + '_scale' in k:
                        input_dict[k] = get_scale_tensor_float(input_dict[PAST_VALUES], input_dict[PAST_VALUES].shape, net.blobs[k].data.shape, 64, 2)
                if PAST_VALUES_REQUANT + '_scale' in k:
                        input_dict[k] = get_scale_tensor_float(input_dict[PAST_VALUES_REQUANT], input_dict[PAST_VALUES_REQUANT].shape, net.blobs[k].data.shape, 64, 2)
        for k in input_dict.keys():
                input_dict[k].astype(np.float32).tofile(os.path.join(img_dir, k + '.bin'))

def create_run_py(run_dir):
        script_content = '''#!/bin/env python3

import logging, sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../../../../../src')))
import dv.sw.runner.runner
import dv.sw.utils.sys

logger = logging.getLogger("run")


def test():
  r = dv.sw.runner.runner.ara_runner()
  r.run()


if __name__ == '__main__':
  test()
        '''
        full_path = os.path.join(run_dir, "run.py")
        with open(full_path, 'w') as f:
                f.write(script_content)
        # Optionally, make the file executable
        os.chmod(full_path,  0o777)
        print(f"Written script to: {full_path}")