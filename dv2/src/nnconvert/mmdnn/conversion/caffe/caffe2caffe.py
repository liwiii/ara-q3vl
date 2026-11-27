'''
# Copyright (c) 2018-22, Deep Vision, Inc. All rights reserved.
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

import copy
import argparse
import logging
from collections import defaultdict
import json
import sys as _sys
import os
import math
import tempfile
import numpy as np
import yaml
import shutil
import pandas as pd

#os.environ['GLOG_minloglevel'] = '1'

os.environ['GLOG_minloglevel'] = '1'
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../../../python/caffe/proto')
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../../../python')
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../')
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../../../analyzer/python')
import caffe
from caffe import layers as L
import caffe.proto.caffe_pb2 as caffe_pb2
from caffe import params as P
from caffe import to_proto

from mmdnn.conversion.caffe.c2c.transforms import run_transformations
from mmdnn.conversion.caffe.utils import compare_outputs
from mmdnn.conversion.common.utils import AdjustFileConfigurator
from dv_logging import get_logger
from mmdnn.conversion.caffe.c2c import common
import google.protobuf as pb
import google.protobuf.text_format
import networkx as nx
from mmdnn.conversion.caffe import subgraph_util
from mmdnn.conversion.common.utils import caffe_to_graph

logger = get_logger(__name__, 'DVCTOC')
COPYRIGHT_MSG = "\n" + \
"*---------------------------------------------------------------*\n" +\
"*                        Ara Compiler                           *\n" +\
"* Copyright (c) 2018-22, Deep Vision, Inc. All rights reserved. *\n" +\
"* Copyright (c) 2018-25, Kinara, Inc. All rights reserved.      *\n" +\
"* Copyright 2025-2026 NXP                                       *\n" +\
"*---------------------------------------------------------------*\n "


def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


"""
randomly initialize weights in caffe
to validate
"""


def prepare_df_to_dump(data, col_names):
  df = pd.DataFrame(data)
  df['index'] = np.arange(len(df)).tolist()
  df = df[col_names]
  return df


def init_wts(net):
  wts = dict()
  for i in net.params:
    idx = list(net._layer_names).index(i)
    wts[i] = []
    if net.layers[idx].type == 'BatchNorm':
      for j in net.params[i]:
        j.data[:] = np.random.uniform(low=1, high=1, size=j.data.shape)
        wts[i].append(j.data)
    else:
      for j in net.params[i]:
        j.data[...] = np.random.uniform(low=-0.25, high=0.25, size=j.data.shape)
        wts[i].append(j.data)
  return wts


"""
get all the weights of caffe net
into a weight dictionary(wts)
"""


def get_wts(net):
  wts = dict()
  for i in net.params:
    wts[i] = []
    for j in net.params[i]:
      wts[i].append(j.data)
  return wts


"""
assign wts back to caffe net
"""


def set_wts(net, wts):
  for i in net.params:
    for idx, j in enumerate(net.params[i]):
      try:
        j.data.flat = wts[i][idx]
      except Exception as e:
        logger.error("weights copy error", exc_info=e)


def emit_name_map(output_dir: str, dv_graph, mutation_observer: dict):
  name_map_out_dir = os.path.join(output_dir, 'stats', "layer_name_map.csv")
  if not os.path.exists(name_map_out_dir):
    logger.debug("src -> caffe map does not exist %s", name_map_out_dir)
    return

  name_mapping = pd.read_csv(name_map_out_dir, sep=',')
  csv_names = ['', 'Src_Name', 'Intermediate_name', 'ARANet_name', 'SrcType', 'ARANetType']
  src_name = csv_names[1]
  aranet_name = csv_names[2]
  intermediate_name = csv_names[3]
  for idx, node in enumerate(name_mapping['Intermediate_name']):
    if node not in mutation_observer['family']:
      name_mapping.loc[idx, "ARANet_name"] = "Optimized Out"
      continue
    name_mapping.loc[idx, "ARANet_name"] = ', '.join(mutation_observer["family"][node])
  name_mapping.pop('Intermediate_name')
  col_names = ['Src_Name', 'ARANet_name', 'SrcType', 'ARANetType']
  df = prepare_df_to_dump(name_mapping, col_names)
  with open(name_map_out_dir, 'w') as f:
    df.to_csv(f, index=False)
  logger.info("name mapping done onnx2caffe")


'''
TODO: for all frameworks 
'''


def _main(_args=None):
  #print(COPYRIGHT_MSG)
  parser = argparse.ArgumentParser()
  from dv_logging import setup_logging
  required_args = parser.add_argument_group('required arguments')
  required_args.add_argument('-inet', "--inputPrototxt", help='Input Caffe Prototxt Path', required=True)
  required_args.add_argument('-onet', "--outputPrototxt", help='Output Caffe Prototxt Path', required=True)
  required_args.add_argument('-output_dir', "--output_dir", help='Output dir to save the output model files and intermediate files', required=True)
  required_args.add_argument('-src_framework', help='source framework', required=True)
  required_args.add_argument('-output_model_name', help='Output model name to save. .prototxt and .caffemodel will be saved in output_dir', required=True)
  optional_args = parser.add_argument_group('optional arguments')
  optional_args.add_argument('-iwt', "--inputWeight", help='Input Caffe Model Path', required=False)
  optional_args.add_argument('-owt', "--outputWeight", help='Input Caffe Model Path', required=False)
  optional_args.add_argument('-manifest_json', help='manifest json from fwgen', required=False)
  optional_args.add_argument('-data_format', required=False, help='data format for each input')
  optional_args.add_argument('-dformat', required=False, help='data type for each input', default='float')
  optional_args.add_argument('-images', required=False, help='image directory for output comparison')
  optional_args.add_argument('-inode', required=False, help='input node names in order of data format')
  optional_args.add_argument('-onode', required=False, help='output node names in order of data format')
  optional_args.add_argument('-input_shape', required=False, help='shape of the input node')
  optional_args.add_argument('-pp_yaml', help='preprocessing yaml file', required=False)
  optional_args.add_argument('-fork_sftmx_argmx', help='in (node) -> (sftmx) -> (agmx), make sftmx and agmx both as network outputs', type=str2bool, required=False)
  optional_args.add_argument('-dummy_json', help='meta for layers replaced by dummy layers', required=False)
  optional_args.add_argument('-name', help='name of the model', required=False)
  optional_args.add_argument('-dynamic', help='dynamic shape network like densepose', type=str2bool, required=False, default=False)
  optional_args.add_argument('-c2c_config', help='c2c config file path', required=False, default=os.path.join(os.path.dirname(__file__), "c2c", "TransformConfig.yaml"))
  optional_args.add_argument('-no_final_perm_opt', help="don't run PermuteToReshape c2c transform", required=False, type=str2bool, default=False)
  optional_args.add_argument('-dtype', help='input data type', required=False)
  optional_args.add_argument('-platform_target', help='platform_target', type=str, required=False, default='Ara1')
  optional_args.add_argument('-dynamic_input', required=False, type=str2bool, help='Network level dynamic inputs', default=False)
  optional_args.add_argument('-summary_csv', help='file path for dumping warnings/errors', required=False)
  optional_args.add_argument('-dry_run', required=False, type=int, help='Dry run mode', default=0)
  optional_args.add_argument('-tf2onnx_happened', required=False, type=str2bool, help='did tf model got converted to onnx model', default=False)
  optional_args.add_argument('-ignore_mse', required=False, type=str2bool, help='ignore mse', default=False)
  optional_args.add_argument('-adjust_file', required=False, type=str, help='file path of the adjust file', default='None')
  parser.add_argument('-L', '--logLevel', required=False, help='Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
  args = parser.parse_args() if _args == None else parser.parse_args(_args)
  # Extracting subgraph

  context_size = None # For Llama
  if args.src_framework == "caffe":
    layer_pb = caffe_pb2.NetParameter()
    with open(args.inputPrototxt, 'r') as f:
      pb.text_format.Merge(f.read(), layer_pb)
    if layer_pb.is_converted:
      outPrototxt = args.output_dir + '/params/' + args.output_model_name + '.prototxt'
      if os.path.isfile(outPrototxt) or os.path.islink(outPrototxt):
        os.unlink(outPrototxt)
      if layer_pb.platform_target:
        os.symlink(args.inputPrototxt, outPrototxt)
      else:
        layer_pb.platform_target = args.platform_target
        with open(outPrototxt, 'w') as f:
          f.write(pb.text_format.MessageToString(layer_pb))
      outCaffemodel = args.output_dir + '/params/' + args.output_model_name + '.caffemodel'
      if os.path.isfile(outCaffemodel) or os.path.islink(outCaffemodel):
        os.unlink(outCaffemodel)
      os.symlink(args.inputWeight, outCaffemodel)
      logger.info("skipping Caffe2Caffe since prototxt has 'is_converted: true'")
      logger.info("PASSED")
      return 0
    if hasattr(layer_pb, 'context_size'): # For Llama
      context_size = layer_pb.context_size
  try:
    if args.src_framework == 'caffe' and args.inode and args.onode:
      outputPrototxt = os.path.join(args.output_dir, "output.prototxt")
      extractor_args = ['-inet', args.inputPrototxt, '-onet', outputPrototxt, '-ishape', args.input_shape, '-inputnn', args.inode.split(), '-outputnn', args.onode.split()]
      subgraph_util._main(extractor_args)
      args.inputPrototxt = outputPrototxt
  except Exception as e:
    logger.warning(str(e))

  setup_logging(log_level=args.logLevel)
  logger.debug("Updating the input prototxt " + args.inputPrototxt)
  graph = caffe_to_graph(args.inputPrototxt)
  if args.inputWeight:
    caffe_input_net = caffe.Net(args.inputPrototxt, caffe.TEST, weights=args.inputWeight)
    inp_wts = get_wts(caffe_input_net)
  else:
    caffe_input_net = caffe.Net(args.inputPrototxt, caffe.TEST)
    inp_wts = init_wts(caffe_input_net)
  cp_only_list = set()
  skip_list = set()
  cp_precision = {}
  logger.info('meta file is %s', args.manifest_json)
  if args.manifest_json:
    #manifest_json = args.manifest_json.split()
    if len(args.manifest_json) > 1:
      all_sets = []

      for i in range(len(args.manifest_json)):
        cp_list = set()
        cp_layer_dict = dict()
        if os.path.exists(args.manifest_json[i]):
          with open(args.manifest_json[i]) as cp_file:
            cp_layer_dict = json.load(cp_file)
          for cp_layer in cp_layer_dict:
            failed_layer_list = cp_layer_dict[cp_layer]
            for layer in failed_layer_list[:-1]:
              cp_list.add(layer)
          all_sets.append(cp_list)
      skip_list = (set.union(*all_sets) - set.intersection(*all_sets)) if len(all_sets) else []

  if args.manifest_json and os.path.exists(args.manifest_json[-1]):
    with open(args.manifest_json[-1], 'r') as cp_file:
      cp_layer_dict = json.load(cp_file)
    logger.debug('cp_layer_dict %s', cp_layer_dict)
    for cp_layer in cp_layer_dict:
      logger.info("cp layer is found from estimator json %s", cp_layer)
      failed_layer_list = cp_layer_dict[cp_layer]
      for layer in failed_layer_list[:-1]:
        logger.debug("cp layer is being added fused %s", layer)
        idx_ = layer.find('_ara')
        if idx_ != -1:
          layer = layer[:idx_]
        cp_only_list.add(layer)
        cp_precision[layer] = "FP32"
  caffe_cp_only_list = set()
  for node in graph.get_topsort():
    layer = node.get_layer()
    if layer.type == "Normalize" and (layer.norm_param.across_spatial and (float(layer.norm_param.scale_before_clip) == float(1) and float(layer.norm_param.scale_after_div) == float(1))):
      caffe_cp_only_list.add(layer.name)
      cp_precision[layer.name] = "FP32"
  if args.pp_yaml:
    with open(args.pp_yaml, 'r') as stream:
      pp_yaml = yaml.safe_load(stream)
  else:
    pp_yaml = None

  adjust_file_obj = AdjustFileConfigurator(args.adjust_file)
  input_permuted, output_permuted = {}, {}
  input_node_name = []
  output_node_name = []
  if args.data_format:
    data_format = args.data_format.split()
    input_node_name = args.inode.split()
  
  if args.onode:
    output_node_name = args.onode.split()

  graph, mutation_observer, inp_wts = run_transformations(caffe_input_net, graph, pp_yaml, cp_only_list, caffe_cp_only_list, skip_list, args.fork_sftmx_argmx, cp_precision, inp_wts, args.dynamic, args.dynamic_input, args.platform_target, args.summary_csv, args.dry_run, args.c2c_config, args.no_final_perm_opt, input_permuted, output_permuted, args.output_dir, args.tf2onnx_happened, input_node_name, output_node_name, adjust_file_obj.cfg)

  emit_name_map(args.output_dir, graph, mutation_observer)
  layer_pb = caffe_pb2.NetParameter()
  topsort = graph.get_topsort()
  data_format = []

  if len(data_format) != len(input_node_name):
    
    logger.info("Warning!!! number of inode names do not match data format size; taking input shape to resolve")
    input_node_name = []
    input_shape = args.input_shape
    if len(input_shape) != len(data_format):
      logger.info("Warning!!! input shapes also do not match with number of data formats")
    for d in range(len(data_format)):
      for node in topsort:
        if node.type == 'Input':
          if node._layer.input_param.shape[0].dim == [int(x) for x in input_shape[d].split(',')]:
            input_node_name.append(node.name)
            break
  ind = 0
  if (topsort[0].type != 'Input'):
    for idx, node in enumerate(topsort):
      if node.type == 'Input':
        ind = idx
        break
  _t = topsort.pop(ind)
  topsort.insert(0, _t)
  for node in topsort:
    if node.type == 'Input':
      for d in range(len(data_format)):
        if node.name == input_node_name[d]:
          node._layer.input_param.layout.append(data_format[d])
          break
        elif len(input_node_name) == 1 and input_node_name[0] == 'data':  #Default case
          node._layer.input_param.layout.append(data_format[d])
          break
  for node in topsort:
    layer_pb.layer.append(node.get_layer())

  with open(args.outputPrototxt, 'w') as f:
    f.write(pb.text_format.MessageToString(layer_pb))
    f.flush()
    os.fsync(f.fileno())
    caffe_verif = caffe.Net(args.outputPrototxt, caffe.TEST)
    set_wts(caffe_verif, inp_wts)
    # net_1.save(args.outputWeight)
    total = 0
    for i in caffe_verif.params:
      for j in caffe_verif.params[i]:
        total += j.data.size * 4
    if total > (2 * 1024 * 1024 * 1024):
      caffe_verif.save_hdf5(args.outputWeight)
    else:
      caffe_verif.save(args.outputWeight)

  res = compare_outputs(caffe_input_net, caffe_verif, args.images, args.dformat.split(' '), input_permuted, output_permuted, args.dtype)
  #hack - argmax pass if layer occurs in between the network
  layer_pb = caffe_pb2.NetParameter()
  graph = caffe_to_graph(args.outputPrototxt, log=False)
  argmax = graph.get_nodes_of_base_type("ArgMax")
  for node in argmax:
    if len(node.successors()) == 0:
      node.argmax_param.out_max_val = True
  layer_pb.is_converted = True
  layer_pb.platform_target = args.platform_target
  if args.dynamic:
    layer_pb.is_dynamic = True
  if args.dynamic_input:
    layer_pb.is_dynamic_input = True
  if context_size is not None: # For Llama
    layer_pb.context_size = context_size
  topsort = graph.get_topsort()
  if (topsort[0].type != 'Input'):
    for idx, node in enumerate(topsort):
      if node.type == 'Input':
        ind = idx
        break
  _t = topsort.pop(ind)
  topsort.insert(0, _t)
  for node in topsort:
    layer_pb.layer.append(node.get_layer())
  if args.name:
    layer_pb.name = args.name
  with open(args.outputPrototxt, 'w') as f:
    f.write(pb.text_format.MessageToString(layer_pb))
    f.flush()
    os.fsync(f.fileno())
  ret = 1 if res == -1 else 0
  ret = 0 if args.ignore_mse else ret
  return ret


if __name__ == "__main__":
  ret = _main()
  if ret != 0:
    exit(int(ret))
