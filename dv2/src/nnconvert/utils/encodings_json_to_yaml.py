'''
Copyright (c) 2022-25, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

import json
import yaml
import csv
import os
import logging
import numpy as np 
from dv_logging import get_logger
import copy

logger = get_logger(__name__)

def generate_precision_adjust_yaml(args, output_adjust_precision_yaml_path):
    try:
        # Force YAML to output True/False instead of true/false
        yaml.add_representer(
            bool,
            lambda dumper, value: dumper.represent_scalar(
                'tag:yaml.org,2002:bool',
                'True' if value else 'False'
            )
        )

        with open(args.encodings_json_file, "r") as f:
            data = json.load(f)

        layer_dict = {"Layers": {}}

        for layer_name, layer_data in data.items():

            # Skip layers that are not a dict or don't follow the nested tensor structure
            # (e.g., "input_image" which is a flat encoding dict without "output.0")
            if not isinstance(layer_data, dict) or "output.0" not in layer_data:
                logger.info("Layer '%s' does not have output.0, skipping.", layer_name)
                continue

            # Only check bitwidth of output.0
            output_bitwidth = layer_data["output.0"].get("bitwidth")

            if output_bitwidth == 16.0:
                sanitized_name = layer_name.replace('-', '_') \
                                           .replace('\\', '_') \
                                           .replace('/', '_') \
                                           .replace('[', '_') \
                                           .replace(']', '_') \
                                           .replace(':', '_') \
                                           .replace('.', '_') \
                                           .replace(' ', '_')

                layer_entry = {"Precision": "INT16"}

                # Check if mean_quantizer or variance_quantizer is present in layer_data
                if "mean_quantizer" in layer_data or "variance_quantizer" in layer_data:
                    layer_entry["isCaffeName"] = True

                layer_dict["Layers"][sanitized_name] = layer_entry

        with open(output_adjust_precision_yaml_path, "w") as f:
            yaml.dump(layer_dict, f, default_flow_style=False, sort_keys=False)

        logger.info('adjust yaml file saved at : %s ', output_adjust_precision_yaml_path)
        return 0

    except Exception as e:
        logger.error("dumping adjust_precision.yaml failed: %s", e)
    return -1

def get_input_layers(prototxt_layers):
    for name, info in prototxt_layers.items():
        if info.get("base_type") == "Input":
            return name  # single input as of now
    return None

def parse_prototxt(proto_path):
    layers = {}
    with open(proto_path, "r") as f:
        lines = f.readlines()

    inside_layer, current_name, current_type, current_base_type, current_pool, current_bottom = False, None, None, None, None, list()

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("layer {"):
            inside_layer, current_name, current_type, current_base_type, current_pool, current_bottom = True, None, None, None, None, list()
            continue
        if inside_layer and stripped.startswith("name:"):
            current_name = stripped.split(":", 1)[1].strip().strip('"')
        if inside_layer and stripped.startswith("type:"):
            current_type = stripped.split(":", 1)[1].strip().strip('"')
        if inside_layer and stripped.startswith("base_type:"):
            current_base_type = stripped.split(":", 1)[1].strip().strip('"')
        if inside_layer and stripped.startswith("pool:"):
            current_pool = stripped.split(":", 1)[1].strip()
        if inside_layer and stripped.startswith("bottom:"):
            current_bottom.append(stripped.split(":", 1)[1].strip().strip('"'))
        if stripped == "}":
            if current_name:
                layers[current_name] = {"type": current_type, "base_type": current_base_type, "inputs": current_bottom}
                if current_base_type == "Pooling":
                    layers[current_name]["pool"] = current_pool
            inside_layer = False

    return layers

def convert_json_to_yaml(args, caffe_prototxt, layer_name_map_csv_path, output_scale_offet_yaml_path):
    try:
        with open(args.encodings_json_file,  "r") as f:
            data = json.load(f)

        name_map = {}
        type_map = {}
        with open(layer_name_map_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = row["Src_Name"]
                aranames = [a.strip() for a in row["ARANet_name"].split(",")]
                aratype = row["ARANetType"]
                preferred = next((a for a in aranames if a.endswith("_inrp")), aranames if aranames else None)
                if preferred:
                    name_map[src] = preferred
                    type_map[src] = aratype
        layer_dict = {"LayerNames": {}}
        prototxt_layers = parse_prototxt(caffe_prototxt)
        input_layer_name = get_input_layers(prototxt_layers)

        # Add input layer entry from "input.0"
        for l_name, tensors in data.items():
            if "input.0" in tensors:
                enc = tensors["input.0"]["encodings"][0]
                if input_layer_name:
                    multiplier, shift = enc.get("multiplier"), enc.get("shift")
                    if multiplier is None or shift is None:
                      continue
                    scale = float(np.float32(multiplier * pow(2,shift)))
                    layer_dict["LayerNames"][input_layer_name] = {
                        "scale": scale,
                        "offset": enc.get("offset")
                    }
                break

        for layer_name, tensors in data.items():
            if layer_name not in name_map:
                logger.info('Layer %s is not present in prototxt but present in encoding json', layer_name)
                continue
            # Exclude MAX pooling
            if type_map.get(layer_name) == "Pooling":
                caffeName = name_map[layer_name][0]
                if caffeName in prototxt_layers:
                    info = prototxt_layers[caffeName]
                    if info.get("base_type") == "Pooling" and info.get("pool") == "MAX":
                        continue
                else:
                    logger.warning(f"Warning: {caffeName} not found in prototxt layers")
            
            if type_map.get(layer_name) == "SpacialDist" or type_map.get(layer_name) == "MeanRatio":
                caffeName = name_map[layer_name][0]
                if caffeName in prototxt_layers:
                    continue
                
            layer_entry = {}
            if "output.0" in tensors:
                enc = tensors["output.0"]["encodings"][0]
                multiplier, shift = enc.get("multiplier"), enc.get("shift")
                layer_entry["scale"] = float(np.float32(multiplier * pow(2,shift)))
                layer_entry["offset"] = enc.get("offset")
            if "param.weight" in tensors:
                weight_encs = tensors["param.weight"]["encodings"]
                multiplier, shift = [we.get("multiplier") for we in weight_encs], [we.get("shift") for we in weight_encs]
                scale = [float(np.float32(m * (2 ** s))) for m, s in zip(multiplier, shift)]
                # layer_entry["weightScales"] = [we.get("scale") for we in weight_encs]
                layer_entry["weightScales"] = scale
                layer_entry["weightOffset"] = [we.get("offset") for we in weight_encs]
                if "is_symmetric" in tensors["param.weight"]:
                    layer_entry["is_symmetric"] = tensors["param.weight"]["is_symmetric"]
            if type_map.get(layer_name) == "LayerNorm":
                # adding intermediate multipliers for layernorm
                if "mean_quantizer" in tensors:
                    mean_encs = tensors["mean_quantizer"]["encodings"][0]
                    meanMin, meanMax = mean_encs.get("min"), mean_encs.get("max")
                    layer_entry["meanMin"] = meanMin
                    layer_entry["meanMax"] = meanMax

                if "variance_quantizer" in tensors:
                    var_encs = tensors["variance_quantizer"]["encodings"][0]
                    varMultiplier, varShift = var_encs.get("multiplier"), var_encs.get("shift")
                    varMin, varMax = var_encs.get("min"), var_encs.get("max")
                    layer_entry["varMin"] = varMin
                    layer_entry["varMax"] = varMax
                if "input_sub_mean_quantizer" in tensors:
                    meanDev_encs = tensors["input_sub_mean_quantizer"]["encodings"][0]
                    meanDevMin, meanDevMax = meanDev_encs.get("min"), meanDev_encs.get("max")
                    absMeanDevMax = max(abs(meanDevMax), abs(meanDevMin))
                    layer_entry["absMeanDevMax"] = absMeanDevMax
                
                if "norm_with_shape_quantizer" in tensors:
                    norm_encs = tensors["norm_with_shape_quantizer"]["encodings"][0]
                    normMin, normMax = norm_encs.get("min"), norm_encs.get("max")
                    absNormMax = max(abs(normMin), abs(normMax))
                    layer_entry["absNormMax"] = absNormMax
            if layer_entry:
                if name_map[layer_name]:
                    for new_caffe_name in name_map[layer_name]:
                        layer_dict["LayerNames"][new_caffe_name] = copy.deepcopy(layer_entry)

        # Define the layers that we are allowed to look through
        PASS_THROUGH_LAYERS = {"Reshape", "Permute", "PermuteNnp"}
        for layer_name, tensors in data.items():
            if layer_name not in name_map:
                logger.info('Layer %s is not present in prototxt but present in encoding json', layer_name)
                continue
            caffeName = name_map[layer_name]
            if name_map[layer_name]:
                for caffeName in name_map[layer_name]:
                    if caffeName not in prototxt_layers:
                        continue
                        
                    info = prototxt_layers[caffeName]
                    
                    for idx, input_name in enumerate(info["inputs"]):
                        if f"input.{idx}" not in tensors:
                            continue
                            
                        # Start tracing backward from the immediate input
                        current_node = input_name
                        dvshift_name = None
                        
                        while current_node in prototxt_layers:
                            base_type = prototxt_layers[current_node].get("base_type")
                            if base_type == "DVShift":
                                # We found the DVShift layer, save its name and stop searching
                                dvshift_name = current_node
                                break
                            elif base_type in PASS_THROUGH_LAYERS:
                                # It's a pass-through layer, look at its input to keep going backward
                                node_inputs = prototxt_layers[current_node].get("inputs", [])
                                if node_inputs:
                                    current_node = node_inputs[0]
                                else:
                                    # Dead end (no inputs to trace back)
                                    break
                            else:
                                # Reached a layer that is NOT DVShift and NOT a pass-through layer
                                break
                        
                        # If the trace successfully found a DVShift upstream
                        if dvshift_name:
                            layer_entry = {}
                            enc = tensors[f"input.{idx}"]["encodings"][0]
                            multiplier = enc.get("multiplier")
                            shift = enc.get("shift")
                            
                            layer_entry["scale"] = float(np.float32(multiplier * pow(2, shift)))
                            layer_entry["offset"] = enc.get("offset")
                            
                            # Use the actual DVShift layer's name as the dictionary key
                            layer_dict["LayerNames"][dvshift_name] = layer_entry

                        

        # Define the layers that we are allowed to look through
        PASS_THROUGH_LAYERS = {"Reshape", "Permute", "PermuteNnp"}

        for layer_name, tensors in data.items():
            if layer_name not in name_map:
                logger.info('Layer %s is not present in prototxt but present in encoding json', layer_name)
                continue
            caffeNames = name_map[layer_name]
            for caffeName in caffeNames:
                if caffeName not in prototxt_layers:
                    continue
                    
                info = prototxt_layers[caffeName]
                
                for idx, input_name in enumerate(info["inputs"]):
                    if f"input.{idx}" not in tensors:
                        continue
                        
                    # Start tracing backward from the immediate input
                    current_node = input_name
                    dvshift_name = None
                    
                    while current_node in prototxt_layers:
                        base_type = prototxt_layers[current_node].get("base_type")
                        
                        if base_type == "DVShift":
                            # We found the DVShift layer, save its name and stop searching
                            dvshift_name = current_node
                            break
                        elif base_type in PASS_THROUGH_LAYERS:
                            # It's a pass-through layer, look at its input to keep going backward
                            node_inputs = prototxt_layers[current_node].get("inputs", [])
                            if node_inputs:
                                current_node = node_inputs[0]
                            else:
                                # Dead end (no inputs to trace back)
                                break
                        else:
                            # Reached a layer that is NOT DVShift and NOT a pass-through layer
                            break
                    
                    # If the trace successfully found a DVShift upstream
                    if dvshift_name:
                        layer_entry = {}
                        enc = tensors[f"input.{idx}"]["encodings"][0]
                        multiplier = enc.get("multiplier")
                        shift = enc.get("shift")
                        
                        layer_entry["scale"] = float(np.float32(multiplier * pow(2, shift)))
                        layer_entry["offset"] = enc.get("offset")
                        
                        # Use the actual DVShift layer's name as the dictionary key
                        layer_dict["LayerNames"][dvshift_name] = layer_entry

                        

        with open(output_scale_offet_yaml_path, "w") as f:
            yaml.dump(layer_dict, f, default_flow_style=False, sort_keys=False)

        logger.info('scale_offset_adjust yaml file saved at : %s ', output_scale_offet_yaml_path)
        return 0
    except Exception as e:
        logger.error("conversion of encodings json to scale_offset_adjust yaml failed")
    return -1

