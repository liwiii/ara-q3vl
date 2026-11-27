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

import yaml
import os
import yaml
import json
import numpy as np
import argparse
import sys
import onnx
from copy import deepcopy
from typing import Tuple
from pathlib import Path

from caffe_loader import CaffeLoader
from model import Model, Qwen2Model, Qwen2_5VLModel
from sharded_model import ShardedModel, Qwen2ShardedModel, Qwen2_5VLShardedModel

sys.path.append(str(Path(__file__).absolute().parent / "qwen3_vl"))

class LLMConverter:
    def __init__(self, build_dir, yaml_file, output_dir, sdk, num_shards):
        self.build_dir = build_dir
        self.yaml_file = yaml_file
        self.config = self.load_config()
        self.hf_config = self.load_hf_config()
        self.caffe_loader = CaffeLoader(self.build_dir)
        self.caffe = self.caffe_loader.get_caffe()
        self.caffe_pb2 = self.caffe_loader.get_caffe_pb2()
        self.proto_utils = self.caffe_loader.get_proto_utils()
        self.onnx_model = self.load_onnx_model()
        self.sdk = sdk

        self.output_dir = {}  # modality->output dir map
        # This arg output_dir should be phased out, and we should use self.config['out']
        # even for internal runs to make things less complicated as we expand modalities.
        self.output_dir['llm'] = self.create_output_dir(output_dir)
        self.num_shards = int(num_shards)

        if self.config.get('vision_out', None):
            self.output_dir['vision'] = self.create_output_dir(self.config['vision_out'])

        print(f"Loaded Caffe version: {self.caffe.__version__}")
     
    def load_config(self):
        with open(self.yaml_file, 'r') as file:
            return yaml.safe_load(file)
    def load_hf_config(self):
        config_path = os.path.join(self.config['model_repo'], "config.json")
        with open(config_path, 'r') as file:
            return json.load(file) 
    def create_output_dir(self, output_dir):
        if self.sdk: 
            output_dir = output_dir
        else:
            output_dir = os.path.join(output_dir,'__rand.'+self.config.get('run_suffix','output'))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        return output_dir
      
    def load_onnx_model(self):
        try:
            onnx_model_path = os.path.join(self.config['onnx_model_repo'], "model.onnx")
            onnx_model = onnx.load(self.config['onnx_model_repo'] + "/model.onnx")
            return onnx_model
        except Exception as e:
            raise RuntimeError(f"Failed to load ONNX model from {onnx_model_path}: {e}") 
        
    def get_onnx_model_info(self):
        graph = self.onnx_model.graph
        nodes = graph.node
        for node in nodes:
            print(f"Name: {node.name}")
            print(f"  OpType: {node.op_type}")
            print(f"  Inputs: {node.input}")
            print(f"  Outputs: {node.output}")
            print(f"  Attributes: {[attr.name for attr in node.attribute]}")
            print("-" * 40)
        for initializer in graph.initializer:
            print(f"Initializer: {initializer.name}")
            print(f"  Shape: {initializer.dims}")
            print(f"  Data Type: {onnx.TensorProto.DataType.Name(initializer.data_type)}")
            print("=" * 40)
    
    def convert_model(self):
        model_obj = None
        model_architecures = self.hf_config.get('architectures',[''])
        if self.num_shards > 1:
            if model_architecures[0] == "LlamaForCausalLM": 
                model_obj = ShardedModel(self.config, self.hf_config, self.proto_utils, self.onnx_model, self.caffe, self.output_dir['llm'], self.sdk, self.num_shards)
            elif model_architecures[0] == "Qwen2ForCausalLM":
                model_obj = Qwen2ShardedModel(self.config, self.hf_config, self.proto_utils, self.onnx_model, self.caffe, self.output_dir['llm'], self.sdk, self.num_shards)
            elif model_architecures[0] == "Qwen2_5_VLForConditionalGeneration":
                if self.output_dir.get('vision', None):
                    # New standard compilation.
                    model_obj = Qwen2_5VLShardedModel(self.config, self.hf_config, self.proto_utils, self.onnx_model, self.caffe, self.output_dir, self.sdk, self.num_shards)
                else:
                    # Legacy flow for Qwen2.5VL
                    model_obj = Qwen2ShardedModel(self.config, self.hf_config, self.proto_utils, self.onnx_model, self.caffe, self.output_dir['llm'], self.sdk, self.num_shards)
            else:
                raise RuntimeError(f"Unsupported Architecture for sharded llm: {model_architecures[0]}")

        elif model_architecures[0] == "LlamaForCausalLM":
            model_obj = Model(self.config, self.hf_config, self.proto_utils, self.onnx_model, self.caffe, self.output_dir['llm'], self.sdk)
        elif model_architecures[0] == "Qwen2ForCausalLM":
            model_obj = Qwen2Model(self.config, self.hf_config, self.proto_utils, self.onnx_model, self.caffe, self.output_dir['llm'], self.sdk) 
        elif model_architecures[0] == "Qwen2_5_VLForConditionalGeneration":
            if self.output_dir.get('vision', None):
                # New standard compilation.
                model_obj = Qwen2_5VLModel(self.config, self.hf_config, self.proto_utils, self.onnx_model, self.caffe, self.output_dir, self.sdk)
            else:
                # Legacy flow for Qwen2.5VL
                model_obj = Qwen2Model(self.config, self.hf_config, self.proto_utils, self.onnx_model, self.caffe, self.output_dir['llm'], self.sdk)
        elif model_architecures[0] == "Qwen3ForCausalLM":
                # Only LLM should work.
                from qwen3_vl.llm_model import Qwen3LLMModel  # imported here so that caffe_loader has been instantiated by now.
                model_obj = Qwen3LLMModel(self.config, self.hf_config, self.proto_utils, self.onnx_model, self.caffe, self.output_dir['llm'], self.sdk)
        elif model_architecures[0] == "Qwen3VLForConditionalGeneration":
                from qwen3_vl.vl_model import Qwen3VLModel   # imported here so that caffe_loader has been instantiated by now.
                model_obj = Qwen3VLModel(self.config, self.hf_config, self.proto_utils, self.onnx_model, self.caffe, self.output_dir, self.sdk)
        else:
            raise RuntimeError(f"Unsupported Architecture: {model_architecures[0]}")
        model_obj.dump_assets()
        model_obj.convert_model()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', required=True, help="build directory")
    parser.add_argument('--yaml_file', required=True, help="YAML file with model configuration")

    # This should be phased out and we should only use output directory from above --yaml_file.
    # It will help in streamlining multimodal models.
    parser.add_argument('--output_dir', required=True, help="Output directory for llm splits")

    parser.add_argument('--sdk', action='store_true', help="set to True for SDK run, default is False", default=False)
    parser.add_argument('--num_shards', required=False, help="num shards in the model, default is 1", default = 1)
    args = parser.parse_args()
    converter = LLMConverter(args.build, args.yaml_file, args.output_dir, args.sdk, args.num_shards)
    converter.convert_model()

    print("Conversion completed successfully.")


