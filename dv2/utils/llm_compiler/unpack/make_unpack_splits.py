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

"""Module for generating unpack split artifacts from prompt processing and token generation splits.

This module processes pre-split model artifacts from `prmpt_prc_*` and `tkn_gen_*` directories,
creating corresponding unpack split directories with prototext, caffemodel, config, and asset
files required for the Ara2 platform. It handles dynamic layers (convolution, layer norm,
attention softmax) and modifies caffemodels with runtime weights from the split directories.
"""
import os
import shutil
import yaml
import numpy as np
import argparse
import sys
import json

from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "dv2/python/caffe"))
sys.path.append(str(Path(__file__).resolve().parent / "dv2/python/caffe/proto"))
sys.path.append(str(Path(__file__).resolve().parent))

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "llm_compiler"))

from llm_compiler.caffe_loader import CaffeLoader
from llm_compiler.utils import create_config, make_inputs
from unpack_decision import should_trigger_unpack_from_dict

parser = argparse.ArgumentParser()
parser.add_argument('--config', required=False, help="The path to config yaml file for unpack(e.g. kinara_run_config.yaml). It checks if the setup is supported.")
parser.add_argument('--splits_dir', required=False, help="The path containing prmpt_prc* and tkn_gen_* directories. No checks are done, unpack splits generation is attempted.")
parser.add_argument('--build', required=True, help="build directory")

build_path = Path(parser.parse_args().build) 
assert build_path.is_dir(), f"{build_path} is not a valid directory"
sys.path.append(build_path / "dv2/python/")
sys.path.append(build_path / "dv2/python/caffe/proto/")

import logging
logger = logging.getLogger(Path(__file__).name)
logging.basicConfig(level=logging.INFO)
logger.info(f"sys.path: {sys.path}")

caffe_loader = CaffeLoader(build_path)
caffe = caffe_loader.get_caffe()

from caffe.proto import caffe_pb2
import google.protobuf.text_format as text_format

from jinja2 import Environment, FileSystemLoader

def get_assets_dir(prmpt_prc_split_dir:Path):
  logger.debug(f"prmpt_prc_split_dir: {prmpt_prc_split_dir}")
  candid_yaml_file = prmpt_prc_split_dir / 'config_ara2.yaml'
  sdk_yaml_file = prmpt_prc_split_dir / 'dvrun_config.yaml'
  logger.debug(f"candid_yaml_file: {candid_yaml_file}")
  logger.debug(f"sdk_yaml_file: {sdk_yaml_file}")
  if candid_yaml_file.exists():
    return prmpt_prc_split_dir / '__run.default/cfg0_bringup_assets/'

  elif sdk_yaml_file.exists():
    raise NotImplementedError(f"Path for json file in sdk run not supported yet")

  else:
    raise FileNotFoundError(f"Neither SDK nor internal run")
  
def _replace_rendered_with_bin(name: str) -> str:
    if name.endswith("_rendered"):
        return  name.removesuffix("_rendered") + ".bin"
    else:
      raise ValueError(f"Layer name does not end with _rendered as expected.")
    
def modify_caffemodel(prmpt_prc_split_dir:Path, out_caffemodel:Path, out_prototxt:Path):
  net = caffe.Net(str(out_prototxt), str(out_caffemodel), caffe.TEST)

  with open(out_prototxt) as f:
    net_param = caffe_pb2.NetParameter()
    proto_text = f.read()
  text_format.Parse(proto_text, net_param)
  layer_list = net_param.layer

  for j in range(len(layer_list)):
    if layer_list[j].type == 'Ara2DVTensor':
      layer_name = layer_list[j].name
      rwts = np.fromfile(get_assets_dir(prmpt_prc_split_dir) / _replace_rendered_with_bin(layer_name), np.int8)
      rwts = rwts.astype(np.float32)
      rwts = rwts.reshape(-1,1,1,1)
      net.params[layer_name][0].data[...] = rwts

  net.save(str(out_caffemodel))

def render_prototxt(unpack_params:Path, tkn_so:Path, template_dir:Path):
  logger.info(f"unpack_params: {unpack_params}")
  logger.info(f"tkn_so: {tkn_so}")
  logger.info(f"template_dir: {template_dir}")
  env = Environment(loader=FileSystemLoader(template_dir))
  template_unpack_nnp = env.get_template("unpack_nnp_tmpl.j2")
  template_unpack_dv_tensor = env.get_template("unpack_dv_tensor_tmpl.j2")
  with open(unpack_params) as f:
    unpack_params_json = json.load(f)
  with open(tkn_so, "r") as f:
    so_data = yaml.safe_load(f)
  
  unpack_proto = 'is_converted: true\nplatform_target: "Ara2"\nlayer {\nname: "data"\ntype: "Input"\ntop: "data"\nbase_type: "Input"\nestimator_group: "Input"\ninput_param {\nshape{\ndim: 1\ndim: 4096\ndim: 1\ndim: 1\n}\nlayout: "NCHW"\n}\n}'

  for layer in unpack_params_json:
    logger.info(f"layer:{layer}")
    if layer['layer_type'] == 'Ara2DynamicConvolution':
      layer_name = layer['layer_name']
      w_prec = so_data["LayerNames"][layer_name]["weightPrecision"][0][layer_name][-1]
      # Note that w_prec is coming from token_gen so.yaml. For prmpt_prc, it will always be 8 bit.
      wnum = layer['num']
      wchannels = layer['channels']
      is_bias = layer['is_bias']
      if (is_bias):
        is_bias = "true"
      else:
        is_bias = "false"

      z = layer['z']
      d = layer['d']

      is_lut = layer['is_lut']
      unpack_proto = unpack_proto + template_unpack_nnp.render(layer_name = layer_name , w_prec = w_prec, wnum = wnum, wchannels = wchannels, is_bias = is_bias, z = z, d = d, is_lut = is_lut)
    elif layer['layer_type'] == 'Ara2DynamicLayerNorm' or layer['layer_type'] == 'Ara2DynamicAttentionSoftmax':
      layer_name = layer['layer_name'] + "_rendered"
      if layer['layer_type'] == 'Ara2DynamicLayerNorm':
        wnum = layer['num']
        wt_sz = wnum * 7
      else:
        wnum = layer['width']
        wt_sz = wnum
      unpack_proto = unpack_proto + template_unpack_dv_tensor.render(layer_name = layer_name, wt_sz = wt_sz)
  return unpack_proto

def get_unpack_params(prmpt_prc_split_dir:Path):
  logger.info(f"prmpt_prc_split_dir: {prmpt_prc_split_dir}")
  candid_yaml_file = prmpt_prc_split_dir / 'config_ara2.yaml'
  sdk_yaml_file = prmpt_prc_split_dir / 'dvrun_config.yaml'
  logger.debug(f"candid_yaml_file: {candid_yaml_file}")
  logger.debug(f"sdk_yaml_file: {sdk_yaml_file}")

  if candid_yaml_file.exists():
    return prmpt_prc_split_dir / '__run.default/unpack_chunk_params.json'

  elif sdk_yaml_file.exists():
    raise NotImplementedError(f"Path for json file in sdk run not supported yet")

  else:
    raise FileNotFoundError(f"Neither SDK nor internal run")
  

def get_tkn_so_caffemodel(tkn_gen_split_dir:Path):
  candid_yaml_file = tkn_gen_split_dir / "config_ara2.yaml"
  sdk_yaml_file = tkn_gen_split_dir / "dvrun_config.yaml"

  if candid_yaml_file.exists():
    with open(candid_yaml_file, 'r') as file:
      tkn_config_ara2 = yaml.safe_load(file)
      tkn_so = tkn_config_ara2["tools"]["analyzer"]["scale_offset_adjust_file"]
      tkn_caffemodel = tkn_config_ara2["network"]["model_file"]

  elif sdk_yaml_file.exists():
    with open(sdk_yaml_file, 'r') as file:
      sdk_dvrun_config = yaml.safe_load(file)
      tkn_so = sdk_dvrun_config["dvnc"]["scale_offset_adjust_file"]
      tkn_caffemodel = sdk_dvrun_config["network"]["iwt"]

  else:
    raise FileNotFoundError(f"Neither config_ara2.yaml nor dvrun_config.yaml was found in {tkn_gen_split_dir}")
  
  return tkn_so, tkn_caffemodel


def fill_directory(prmpt_prc_split_dir:Path, tkn_gen_split_dir:Path, unpack_split_dir:Path, template_dir:Path):
  logger.info(f"prmpt_prc_split_dir: {prmpt_prc_split_dir}")
  logger.info(f"tkn_gen_split_dir: {tkn_gen_split_dir}")
  logger.info(f"unpack_split_dir: {unpack_split_dir}")
  logger.info(f"template_dir: {template_dir}")

  unpack_params = get_unpack_params(prmpt_prc_split_dir)
  logger.info(f"unpack_params: {unpack_params}")

  tkn_so, tkn_caffemodel = get_tkn_so_caffemodel(tkn_gen_split_dir)  
  logger.info(f"tkn_so: {tkn_so}")
  logger.info(f"tkn_caffemodel: {tkn_caffemodel}")
  
  unpack_proto = render_prototxt(unpack_params, tkn_so, template_dir)
  out_caffemodel = unpack_split_dir / "unpack.caffemodel"
  out_prototxt = unpack_split_dir / "unpack.prototxt"
  out_config = unpack_split_dir / "config_ara2.yaml"
  offsets = unpack_split_dir / "offsets.json"
  img_path = unpack_split_dir / "imgs"
  os.makedirs(img_path, exist_ok=True)
  shutil.copy(get_assets_dir(prmpt_prc_split_dir) / "offsets.json", offsets)
  shutil.copy(tkn_caffemodel, out_caffemodel)
  shutil.copy(prmpt_prc_split_dir / "run.py", unpack_split_dir / "run.py")
  with open(out_prototxt, "w") as ptxt:
    ptxt.write(unpack_proto)
  make_inputs(caffe, str(out_prototxt), str(img_path))
  create_config(caffe, str(out_prototxt), str(out_caffemodel), str(img_path), str(offsets), str(out_config))
  modify_caffemodel(prmpt_prc_split_dir, out_caffemodel, out_prototxt)

def fill_directories(config:dict):
  splits_dir = Path(config['out'])
  num_decoders = sum(1 for p in splits_dir.rglob("prmpt_prc_*/split_*") if p.is_dir())

  prmpt_prc_dir =  str(list(splits_dir.rglob("prmpt_prc_*"))[0])  # Some nuiance of rglob
  ctx_size = config["tkn_gen_sizes"][0] # Just pick first token_gen

  template_dir = os.path.dirname(__file__)

  tkn_gen_dir = Path(prmpt_prc_dir).parent / ('tkn_gen_' + str(ctx_size))

  unpack_dir = Path(prmpt_prc_dir).parent / 'unpack'
  for i in range(num_decoders):
    prmpt_prc_split_dir = Path(prmpt_prc_dir) / ("split_" + str(i))
    tkn_gen_split_dir = tkn_gen_dir / ("split_" + str(i))
    unpack_split_dir = unpack_dir / ("split_" + str(i))
    os.makedirs(unpack_split_dir, exist_ok=True)
    fill_directory(prmpt_prc_split_dir, tkn_gen_split_dir, unpack_split_dir, template_dir)

def is_non_empty(param:str):
  return param is not None and param != ""  # True if non empty.

if __name__ == "__main__":
  args = parser.parse_args()
  
  # Either config or splits_dir should be set.
  if is_non_empty(args.config) ^ is_non_empty(args.splits_dir):
    # All good. Any one is set.
    pass
  else:
    raise ValueError(f"Exactly one between --config {args.config} or --splits_dir {args.splits_dir} must be set.")

  if is_non_empty(args.config):
    with open(args.config, 'r') as file:
      config = yaml.safe_load(file)
      if not should_trigger_unpack_from_dict(config):
        logger.info("Skipping unpack splits generation")
        sys.exit(0)
  else:
    config = {}
    config['out'] = args.splits_dir

    # Just first token_gen is needed.
    # Ideally our SDK should have this '1024' numeral at one place, but right now this is scattered.
    config['tkn_gen_sizes'] = [1024] 

  fill_directories(config)
