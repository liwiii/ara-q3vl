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

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from gptqmodel import GPTQModel
from gguf.gguf_reader import GGUFReader
import re
import json
import sys
from pathlib import Path

import os
import argparse
import yaml

current_file = os.path.abspath(__file__)
current_dir = Path(os.path.dirname(current_file))
sys.path.extend([str(current_dir.parent), str(current_dir.parents[1])])

from constants import *
from builder import create_model
from ort_genai.May2026.models.builder import create_model as may2026_ort_genai_create_model

FLOAT = 'float'
GGUF = 'gguf'
GPTQ = 'gptq'

def parse_args():
    parser = argparse.ArgumentParser(description="Convert to ONNX arguments")
    parser.add_argument("-m","--model_dir", type=str, required=True, help="Directory containing the model")
    parser.add_argument("-o","--output_dir", type=str, required=True, help="Output directory for saving the onnx model")
    parser.add_argument("-t","--model_type", type=str, required=True, choices=[FLOAT, GGUF, GPTQ], help="Type of the model, accepted values: ['float', 'gguf', 'gptq']")
    parser.add_argument("-s","--specd_size", type=int, required=True, choices=[1, 4], help="batch size of tokens for speculative decoding, accepted values: [1, 4] based on kernel implementations as of Jan 2026")

    # In future allowed values of num_mcs_shards can be based on model architecture.
    parser.add_argument("-n","--num_mcs_shards", type=int, required=True, choices=[1, 2, 4], help="number of muti-chip shards, compiled dvm is one standalone shard, accepted values: [1, 2, 4], this will expand later.")

    parser.add_argument("-v","--multimodal_standard_compilation", action="store_true", default=False, required=False,help="Should all modalities be compiled alongside llm.")
    parser.add_argument("-f","--float_out_path", type=str, required=False,help="Output directory to store dequantized float model when input is gguf/gptq" )
    parser.add_argument("-g","--gguf_file", type=str, required=False,help="Path to GGUF file")
    parser.add_argument(
        "-r", "--vl_rope_type", type=str, required=False,  default="IMAGE", choices=["IMAGE", "VIDEO", "MULTI_ROPE"], help="VL Rope type need to be used"
    )
    parser.add_argument("-p","--use_8bit_prompt", action="store_true", default=False, required=False,help="Should prompt processing use 8bit quantization. Helpful when better accuracy is needed or special optimizations like LLM weight unpacking.")
    return parser.parse_args()

class GGUFModelExporter():
  def __init__(self, model_dir, output_dir, gguf_file,):
    self.model_dir = model_dir
    self.output_dir = output_dir
    self.gguf_file = gguf_file
  
  def load_model(self):
    try:
      self.model = AutoModelForCausalLM.from_pretrained(
                self.model_dir,
                gguf_file=self.gguf_file, 
                torch_dtype="float32"
              )
      self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, gguf_file=self.gguf_file)
    except Exception as e:
      raise RuntimeError(f"Unable to load model/tokenizer: {e}")
  
  def dump_model(self):
    try:
        os.makedirs(self.output_dir, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Unable to create directory: {self.output_dir}")
    self.model.save_pretrained(self.output_dir)
    self.tokenizer.save_pretrained(self.output_dir)
    print(f"Model and tokenizer Dumped to {self.output_dir}")
  
  def export_model(self):
     self.load_model()
     self.dump_model()
     return self.output_dir
  
  def normalize_bit_width(self, s):
      match = re.search(r'(\d+)', s)
      n = int(match.group(1)) if match else 4
      if n <= 2:
         return INT2
      elif n == 3:
         return INT3
      else:
         return INT4
  
  def generate_weight_config(self):
     yaml_dict = {}
     with open(self.gguf_file, "rb") as f:
         reader = GGUFReader(f)
     
     print("Tensors:") # noqa: NP100
     tensor_info_format = "{:<30} | Shape: {:<15} | Size: {:<12} | Quantization: {}"
     print(tensor_info_format.format("Tensor Name", "Shape", "Size", "Quantization")) # noqa: NP100
     print("-" * 80) # noqa: NP100
     gguf_quant_map = {}
     weight_layers_map = {K_PROJ_GGUF:K_PROJ, V_PROJ_GGUF: V_PROJ, Q_PROJ_GGUF:Q_PROJ, O_PROJ_GGUF:O_PROJ, UP_PROJ_GGUF:UP_PROJ, GATE_PROJ_GGUF:GATE_PROJ, DOWN_PROJ_GGUF:DOWN_PROJ}
     for tensor in reader.tensors:
        shape_str = "x".join(map(str, tensor.shape))
        size_str = str(tensor.n_elements)
        quantization_str = tensor.tensor_type.name
        print(tensor_info_format.format(tensor.name, shape_str, size_str, quantization_str))
        tensor_name = tensor.name.split(".")
        layer_idx = tensor_name[1]
        tensor_name = ".".join(tensor_name[2:])
        if tensor_name not in weight_layers_map.keys():
           continue
        layer_name = weight_layers_map[tensor_name]
        if layer_name not in gguf_quant_map.keys():
            gguf_quant_map[layer_name] = {}
        bit_width = self.normalize_bit_width(quantization_str)
        print("Mapping {0} to {1}".format(quantization_str, bit_width))
        if bit_width not in gguf_quant_map[layer_name].keys():
            gguf_quant_map[layer_name][bit_width] = [] 
        gguf_quant_map[layer_name][bit_width].append(int(layer_idx))
     config = None
     with open(os.path.join(self.output_dir,'config.json'), 'r') as file:
        config = json.load(file)
     num_hidden_layers = config.get('num_hidden_layers', 32)
     gguf_quant_map[LMHEAD] = {INT4: [num_hidden_layers]}
     return gguf_quant_map

class GPTQModelExporter():
  def __init__(self, model_dir, output_dir):
      self.model_dir = model_dir
      self.output_dir = output_dir
    
  def export_model(self):
    try:
        os.makedirs(self.output_dir, exist_ok=True)
        GPTQModel.export(self.model_dir, self.output_dir, 'hf', trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"Unable to export model: {e}")
    return self.output_dir

  def generate_weight_config(self):
     yaml_dict = {}
     config = None
     with open(os.path.join(self.model_dir,'config.json'), 'r') as file:
        config = json.load(file)
     num_hidden_layers = config.get('num_hidden_layers', 32)
     bit_width = config.get('quantization_config')['bits']
     bit_width_map = {4:INT4, 3:INT3, 2:INT2}
     if bit_width not in bit_width_map.keys():
        raise NotImplementedError(f"Bit width not implemented: {str(bit_width)}")

     proj_layers = [K_PROJ, Q_PROJ, V_PROJ, O_PROJ ,UP_PROJ, GATE_PROJ, DOWN_PROJ]
     for layer in proj_layers:
        yaml_dict[layer] = {bit_width_map[bit_width] : [i for i in range(num_hidden_layers)]}
     yaml_dict[LMHEAD] = {INT4: [num_hidden_layers]}
     return yaml_dict
    
     
     

class FloatModel():
   def __init__(self, model_dir):
      self.output_dir = model_dir
      self.model_dir = model_dir
   def export_model(self):
      return self.output_dir
   def generate_weight_config(self):
     yaml_dict = {}
     config = None
     with open(os.path.join(self.model_dir,'config.json'), 'r') as file:
        config = json.load(file)
     num_hidden_layers = config.get('num_hidden_layers') or config['text_config'].get('num_hidden_layers', 32)
     bit_width = 4 #Setting as default for float dequantized models
     bit_width_map = {4:INT4, 3:INT3, 2:INT2}
     if bit_width not in bit_width_map.keys():
        raise NotImplementedError(f"Bit width not implemented: {str(bit_width)}")

     proj_layers = [K_PROJ, Q_PROJ, V_PROJ, O_PROJ, UP_PROJ, GATE_PROJ, DOWN_PROJ]
     for layer in proj_layers:
        yaml_dict[layer] = {bit_width_map[bit_width] : [i for i in range(num_hidden_layers)]}
     yaml_dict[LMHEAD] = {INT4: [num_hidden_layers]}
     return yaml_dict


def convert_model_to_onnx():
  args = parse_args()
  if(args.model_type == GPTQ):
    assert(args.float_out_path is not None)
    exporter = GPTQModelExporter(args.model_dir, args.float_out_path)
  if(args.model_type == GGUF):
    assert(args.float_out_path is not None and args.gguf_file is not None)
    exporter = GGUFModelExporter(args.model_dir, args.float_out_path, args.gguf_file)
  if(args.model_type == FLOAT):
    exporter = FloatModel(args.model_dir)
  
  float_model_dir = exporter.export_model()
  weights_dict = exporter.generate_weight_config()
  extra_options = {}
  if args.vl_rope_type is not None:
     extra_options['vl_rope_type'] = args.vl_rope_type

  if (args.model_type == FLOAT):
    config = AutoConfig.from_pretrained(args.model_dir)
  else:
    config = AutoConfig.from_pretrained(args.float_out_path)
  if config.architectures[0] == "Qwen3VLForConditionalGeneration":
    create_model_output = may2026_ort_genai_create_model(None, float_model_dir, args.output_dir, "fp32", "cpu", os.path.join('.', 'cache_dir'), **extra_options)
  else:
    create_model_output = create_model(None, float_model_dir, args.output_dir, "fp32", "cpu", os.path.join('.', 'cache_dir'), **extra_options)
  
  yaml_dict = {}
  yaml_dict["model_repo"] = float_model_dir
  yaml_dict["original_model_repo"] = args.model_dir
  yaml_dict["onnx_model_repo"] = args.output_dir
  yaml_dict["out"] = os.path.join(args.output_dir, "kinara_llm")  # should become kinara_genai
  if args.multimodal_standard_compilation:
    # This is added because llm_model_gen.py invokes convert_llm.py with '/llm' in --output_dir
    # while for vision we soleley depend on the path in kinara_run_config without manipulation.
    # We plan to phase out this discrepency by making convert_llm.py only use exact paths from
    # kinara_run_config.yaml for all modalities.
    yaml_dict["vision_out"] =  os.path.join(yaml_dict["out"], "vision")
  yaml_dict["prmpt_prc_sizes"] = [128]
  yaml_dict["prmpt_context_size"] = 4096
  yaml_dict["tkn_gen_sizes"] = [i*yaml_dict["prmpt_context_size"]//4  for i in range(1,5)]
  yaml_dict["weight_config"] = weights_dict
  yaml_dict["use_8bit_prompt"] = args.use_8bit_prompt
  yaml_dict["tie_word_embeddings"] = config.tie_word_embeddings

  yaml_dict["specd_size"] = args.specd_size
  yaml_dict["num_mcs_shards"] = args.num_mcs_shards

  # Pass through all entries in 'create_model_output'.
  if create_model_output.keys() & yaml_dict.keys():
     raise KeyError(f"Key conflict: {create_model_output.keys() & yaml_dict.keys()}")
  yaml_dict.update(create_model_output)

  if "qxk_out_scale_32_layers" not in yaml_dict.keys():
     yaml_dict["qxk_out_scale_32_layers"] = []
  if "attention_scale_layers" not in yaml_dict.keys():
     yaml_dict["attention_scale_layers"] = []

  run_yaml_file = os.path.join(args.output_dir,"kinara_run_config.yaml")
  with open(run_yaml_file, "w") as file:
    yaml.dump(yaml_dict, file)

if __name__ == "__main__":
   convert_model_to_onnx()







 
