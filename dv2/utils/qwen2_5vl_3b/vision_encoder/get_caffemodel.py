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
from safetensors import safe_open
from pathlib import Path
import torch
from typing import List


def generate_caffe_model(caffe, final_torch_to_caffe: dict, filenames: List[str], source_proto: Path, target_caffemodel: Path):
    tensors = {}
    
    print(f"filenames: {filenames}")

    for filename in filenames:
        with safe_open(filename, framework="pt", device="cpu"
        ) as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
                print(key)

    source_net = caffe.Net(source_proto, caffe.TEST)

    # caffe_to_torch = {v: k for k, v in final_torch_to_caffe.items()}
    caffe_to_torch = {}
    for k, v in final_torch_to_caffe.items():
        if isinstance(v, list):
            for v_1 in v:
                caffe_to_torch[v_1] = k
        else:
            caffe_to_torch[v] = k
    for param_name in source_net.params.keys():
        if param_name in caffe_to_torch.keys():
            print(f"{param_name} : {caffe_to_torch[param_name]}")
            print("Number of blobs for the node :", len(source_net.params[param_name]))

            torch_name = caffe_to_torch[param_name]
            if "up_proj" in torch_name or "gate_proj" in torch_name:
                tensors[torch_name + ".weight"] = torch.nn.functional.pad(tensors[torch_name + ".weight"], (0, 0, 0, 36))
                tensors[torch_name + ".bias"] = torch.nn.functional.pad(tensors[torch_name + ".bias"], (0, 36))
            if "down_proj" in torch_name:
                tensors[torch_name + ".weight"] = torch.torch.nn.functional.pad(tensors[torch_name + ".weight"], (0, 36))
            source_shape = source_net.params[param_name][0].data.shape
            if "qkv" in torch_name:
                if "q_proj" in param_name:
                    idx = 0
                if "k_proj" in param_name:
                    idx = 1
                if "v_proj" in param_name:
                    idx = 2
                print(torch_name)
                print(50 * "*")
                print(source_net.params[param_name][0].data.flatten()[-10:])
                print(source_net.params[param_name][1].data.flatten()[-10:])
                print(50 * "*")
                source_shape = source_net.params[param_name][1].data.shape
                source_net.params[param_name][1].data[...] = (
                    tensors[torch_name + ".bias"][idx * 1280 : (idx + 1) * 1280]
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(source_shape)
                )
                source_shape = source_net.params[param_name][0].data.shape
                source_net.params[param_name][0].data[...] = (
                    tensors[torch_name + ".weight"][idx * 1280 : (idx + 1) * 1280]
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(source_shape)
                )
                print(source_net.params[param_name][0].data.flatten()[-10:])
                print(source_net.params[param_name][1].data.flatten()[-10:])
                print(50 * "*")
            else:
                print(torch_name)
                print(50 * "*")
                print(source_net.params[param_name][0].data.flatten()[-10:])
                print(50 * "*")
                if torch_name + ".bias" in tensors.keys():
                    source_shape = source_net.params[param_name][1].data.shape
                    source_net.params[param_name][1].data[...] = (
                        tensors[torch_name + ".bias"]
                        .float()
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(source_shape)
                    )
                    print(source_net.params[param_name][1].data.flatten()[-10:])
                    print(50 * "*")
                source_shape = source_net.params[param_name][0].data.shape
                source_net.params[param_name][0].data[...] = (
                    tensors[torch_name + ".weight"]
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(source_shape)
                )
                print(source_net.params[param_name][0].data.flatten()[-10:])
                print(50 * "*")
                # breakpoint()
    for param_name in source_net.params.keys():
        print(50 * "*")
        print(param_name)
        print(source_net.params[param_name][0].data.flatten()[-10:])
        print(50 * "*")

    source_net.save_hdf5(target_caffemodel)
