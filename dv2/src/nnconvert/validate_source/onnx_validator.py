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

import onnx
from onnx import helper
from onnx.utils import extract_model
from onnx.checker import MAXIMUM_PROTOBUF
import argparse
import os
import onnxsim
import onnxruntime
from utils.io_utils import *
import copy
from copy import deepcopy
import numpy as np
import pandas as pd
import glob
import shutil

from dv_logging import get_logger
from typing import List

_sess_options = onnxruntime.SessionOptions()
_sess_options.inter_op_num_threads = 1
_sess_options.intra_op_num_threads = 1
_sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
logger = get_logger(__name__, 'DVIPVAL')
SPECIAL_CHARS = [".", "/", ":", "[", "]", ",", "{", "}", "-", "\\", ";"]


def special(name):
  for char in SPECIAL_CHARS:
    if char in name:
      return True
  if name.isdigit():
    return True
  return False


def mk_node_name(name, name_dict, update_name_dict = True):
  # Scopes created in a nested scope may have initial characters
  # that are illegal as the initial character of an op name
  # (viz. '-', '\', '/', and '_').
  if name in name_dict:
    return name_dict[name]
  
  name2 = name.replace('-','_')\
              .replace('\\','_')\
              .replace('/','_')\
              .replace('[','_')\
              .replace(']','_')\
              .replace(':', '_')\
              .replace('.','_')\
              .replace(' ','_')
  
  name3 = name2
  i = 0
  while name3 in name_dict.values():
    name3 = name2 + f"_{i}"
    i = i + 1

  if update_name_dict:
    name_dict[name] = name3
  
  return '{}'.format(name3)

def remove_all_but_selected_files(dirpath, preserve_files_list):
  if not os.path.isdir(dirpath):
    logger.error(f"Not a directory: {dirpath}")
    raise
  try:
    files = glob.glob(os.path.join(dirpath, "*"))
    for f in files:
      if (os.path.isfile(f) or os.path.islink(f)) and f not in preserve_files_list:
        os.unlink(f)
      elif os.path.isdir(f):
        shutil.rmtree(f)
  except Exception as e:
    logger.error(f"Failed to remove specified files from {dirpath}: {e}")

def remove_dir(dirpath):
  if not os.path.isdir(dirpath):
    logger.error('Not a directory: %s', dirpath)
    raise
  try:
    shutil.rmtree(dirpath)
    logger.info(f"Removed existing directory: {dirpath}")
  except Exception as e:
    logger.error(f"Error removing directory {dirpath}: {e}")
    raise

def save_onnx_model(model, model_out):
  saved = False
  chkpt_path = os.path.join(os.path.dirname(os.path.dirname(model_out)), 'checkpoint_model')
  if os.path.isdir(chkpt_path):
    remove_dir(chkpt_path)

  chkpt_path = shutil.move(os.path.dirname(model_out), chkpt_path)
  os.makedirs(os.path.dirname(model_out))
  try:
    if model.ByteSize() > MAXIMUM_PROTOBUF:
      onnx.save_model(model, model_out, all_tensors_to_one_file=False, save_as_external_data=True, size_threshold=200, convert_attribute=True)
      saved = True
    else:
      onnx.save_model(model, model_out, save_as_external_data=False)
      saved = True
  except Exception as e:
    logger.error('Failed in onnx.save_model: %s', e)
  if saved:
    remove_dir(chkpt_path)
  else:
    remove_dir(os.path.dirname(model_out))
    os.rename(chkpt_path, os.path.dirname(model_out))


def _simplify_onnx_graph(args, model_out):
  model_chkpt = onnx.load_model(model_out, load_external_data=True)
  '''
    currently using onnx-simplifier tool
    use DV implemented logic to tackle this in future.
    TODO: Aditya
    '''
  try:
    inp_dict = dict()
    model = onnx.load_model(model_out, load_external_data=True)
    shapes_from_model = [[j.dim_value for j in i.type.tensor_type.shape.dim] for i in model.graph.input]
    dynamic_input_shape = any([0 in i for i in shapes_from_model])
    if dynamic_input_shape:
      shapes = [[int(j) for j in i.split(',')] for i in args.input_shape]
      input_shapes = {i: np.array(j) for i, j in zip(args.input_node_name, shapes)}
      model, check = onnxsim.simplify(model_out, skip_fuse_bn=True, skipped_optimizers=['fuse_pad_into_pool'], input_shapes=input_shapes)
    else:
      model, check = onnxsim.simplify(model_out, skip_fuse_bn=True, skipped_optimizers=['fuse_pad_into_pool'])
    logger.info("OnnxSimplifier check %s: ", str(check))
    if not check:
      logger.warn("[E]: OnnxSimplifier tool failed")
    else:
      save_onnx_model(model, model_out)
      logger.info("ONNX Model simplified!")
  except Exception as e:
    logger.warn("Onnx simplifier failed:  [%s]", str(e))
    logger.warn("... falling back to checkpoint model")
    save_onnx_model(model_chkpt, model_out)


def _is_node_present(model, name):
  '''
    Get the node in the model for a given name
    '''
  for node in model.graph.input:
    if name == node.name:
      return node
  for node in model.graph.node:
    if name in node.output:
      return node
    for inp in node.input:
      if inp == name:
        return node
    for node in model.graph.node:
      if node.name == name:
        return node
  for node in model.graph.output:
    if name == node.name:
      return node
  return None


def preprocess_output_nodes(args, model):
  model_chkpt = deepcopy(model)
  try:
    o_name_map = {}
    name_dict2 = dict()
    for node in model.graph.node:
      for idx, oname in enumerate(node.output):
        if oname.isdigit():
          o_name_map[oname] = node.name + ("_{}".format(oname))
          node.output[idx] = node.name + ("_{}".format(oname))
        elif oname is not mk_node_name(oname, name_dict2, False):
          o_name_map[oname] = mk_node_name(node.name, name_dict2)
          node.output[idx] = mk_node_name(oname, name_dict2)
    # update inputs
    for node in model.graph.node:
      for idx, iname in enumerate(node.input):
        if iname in o_name_map:
          node.input[idx] = o_name_map[iname]
    for idx, node in enumerate(model.graph.output):
      if node.name in o_name_map:
        model.graph.output[idx].name = o_name_map[node.name]
    # onnx.checker.check_model(model)
  except Exception as e:
    logger.info("Failed to preprocess output nodes. Please check.: %s" % e)
    logger.warn("... falling back to checkpoint model")
    model.graph.CopyFrom(model_chkpt.graph)
    return model, None
  logger.debug("Model is fine after updating model inp/out node names ...")
  return model, o_name_map

def handle_layer_and_output_name_matches(args, model):
  '''
  This function is added for the cases where an onnx model has an output tensor whose name matches with a layer name. tf2onnx can generate such models.
  '''
  all_layer_names = set(i.name for i in model.graph.node)
  all_output_names = set(i.name for i in model.graph.output)
  for i in model.graph.node:
    name = i.name
    index = 0
    check = name in all_output_names
    check = check and (name not in i.output)
    while check:
      name = i.op_type + '_index_' + str(index)
      index += 1
      check = name in all_output_names
      check = check and (name not in i.output)
    all_layer_names.remove(i.name)
    all_layer_names.add(name)
    i.name = name
  return model

def handle_layer_and_output_name_matches(args, model):
  '''
  This function is added for the cases where an onnx model has an output tensor whose name matches with a layer name. tf2onnx can generate such models.
  '''
  all_layer_names = set(i.name for i in model.graph.node)
  all_output_names = set(i.name for i in model.graph.output)
  for i in model.graph.node:
    name = i.name
    index = 0
    check = name in all_output_names
    check = check and (name not in i.output)
    while check:
      name = i.op_type + '_index_' + str(index)
      index += 1
      check = name in all_output_names
      check = check and (name not in i.output)
    all_layer_names.remove(i.name)
    all_layer_names.add(name)
    i.name = name
  return model


def _onnx_extract_subgraph(args, model_out):
  # Subgraph extraction
  try:
    extract_model(model_out, model_out, args.input_node_name, args.output_node_name)
    model = onnx.load(model_out)
    # onnx.checker.check_model(model)
    return True
  except Exception as e:
    logger.error("Onnx subgraph extractor failed. Err trace: [%s]", e)
    return False


def _node_name_preprocess(args, model, model_out):
  model_chkpt = deepcopy(model)
  try:
    name_mapping = dict()
    input_dict = dict()
    node_dict = dict()
    iname_map = {}
    name_dict = dict()
    if args.iname is not None and len(args.iname):
      for name in args.iname:
        new_inp_name = name.split('@')[1]
        if special(new_inp_name):
          raise ValueError(f"iname provided: {new_inp_name} contains invalid characters")
        iname_map[name.split('@')[0]] = new_inp_name
    model_input_name = [node.name for node in model.graph.input]
    for name in iname_map:
      if name not in model_input_name:
        raise ValueError(f"iname provided: {name} not found in model graph input, please provide valid inames")
    for idx, node in enumerate(model.graph.input):
      old_name = node.name
      if len(iname_map.keys()) or special(old_name):
        if old_name in iname_map:
          node.name = iname_map[old_name]
        else:
          if len(iname_map.keys()):
            logger.error("input not mentioned in iname")
          else:
            node.name = mk_node_name(old_name, name_dict)
            if node.name.isdigit():
              node.name = "input_{}".format(idx)
      input_dict[old_name] = node.name
      name_mapping[old_name] = (node.name, "Input")

    for idx, node in enumerate(model.graph.output):
      old_name = node.name
      node.name = mk_node_name(node.name, name_dict)
      name_mapping[old_name] = (old_name, "Output")
    outlayers = [node.name for node in model.graph.output]
    for idx, node in enumerate(model.graph.initializer):
      if node.name in input_dict:
        node.name = input_dict[node.name]
      else:
        node.name = mk_node_name(node.name, name_dict)
    for node in model.graph.node:
      old_name = node.name
      if node.name == '' or node.name[0].isdigit():
        if node.op_type in node_dict:
          node_dict[node.op_type] += 1
        else:
          node_dict[node.op_type] = 0
        node.name = node.op_type + '_{}'.format(str(node_dict[node.op_type]))
        # TODO: remove this hack after this issue is closed.
        # https://github.com/facebookarchive/caffe2/issues/1971
        node.name = node.name.lower()
      node.name = mk_node_name(node.name, name_dict)
      for _idx, name in enumerate(node.input):
        if node.input[_idx] in input_dict:
          node.input[_idx] = input_dict[node.input[_idx]]
        else:
          node.input[_idx] = mk_node_name(node.input[_idx], name_dict)
      for _idx, name in enumerate(node.output):
        node.output[_idx] = mk_node_name(node.output[_idx], name_dict)
      if (old_name == ''):
        old_name = node.name
      name_mapping[old_name] = (node.name, node.op_type)

    args.output_node_name = [mk_node_name(name, name_dict) for name in args.output_node_name]
    model, o_name_map = preprocess_output_nodes(args, model)
    model = handle_layer_and_output_name_matches(args, model)
    for node_name in outlayers:
      if node_name in o_name_map:
        name_mapping[node_name] = (o_name_map[node_name], "Output")
    if o_name_map is not None:
      for i in o_name_map:
        if i in args.input_node_name:
          args.input_node_name = list(map(lambda x: x.replace(i, o_name_map[i]), args.input_node_name))
        if i in args.output_node_name:

          args.output_node_name = list(map(lambda x: x.replace(i, o_name_map[i]), args.output_node_name))
    # Check model after update
    # onnx.checker.check_model(model)
    save_onnx_model(model, model_out)
  except Exception as e:
    logger.warn('Model updation failed. Missing name fields not properly added. Please check.: %s' % e)
    logger.warn("... falling back to checkpoint model")
    model.graph.CopyFrom(model_chkpt.graph)
    save_onnx_model(model, model_out)
    input_dict = dict()
    for idx, node in enumerate(model.graph.input):
      input_dict[node.name] = node.name
    return input_dict
  logger.info("DV preprocessing of ONNX modle. Added missing name fields. model_path = [%s]", model_out)
  return input_dict, name_mapping


def prepare_df_to_dump(data, col_names):
  df = pd.DataFrame(data)
  df['index'] = np.arange(len(df)).tolist()
  df = df[col_names]
  return df


def emit_name_map(name_map: dict, output_dir):
  name_mapping = {'Src_Name': [], 'Intermediate_name': [], 'SrcType': [], 'ARANetType': [], 'ARANet_name': []}
  name_mapping_col_names = ['index', 'Src_Name', 'Intermediate_name', 'ARANet_name', 'SrcType', 'ARANetType']
  for key, val in name_map.items():
    name_mapping['Src_Name'].append(key)
    name_mapping['Intermediate_name'].append(val[0])
    name_mapping['SrcType'].append(val[1])
    name_mapping['ARANetType'].append('')
    name_mapping['ARANet_name'].append(val[0])

  name_df = prepare_df_to_dump(name_mapping, name_mapping_col_names)
  name_map_out_dir = os.path.join(output_dir, 'stats', "layer_name_map.csv")
  with open(name_map_out_dir, 'w') as f:
    name_df.to_csv(f, index=False)
  logger.info("[A] dumped layer_name_map.csv %s", name_map_out_dir)

def get_model_size(model):
  total = 0
  import onnx.numpy_helper as nh
  for tensor in model.graph.initializer:
    if tensor.raw_data:
      total += len(tensor.raw_data)
    else:
      total += nh.to_array(tensor).nbytes
  return total

def _save_all_inference_graph(args, model, all_inf_graph):
  model_chkpt = deepcopy(model)
  logger.info("Saving all inference onnx model ...")
  try:
    if args.onnx_backend not in ["onnxruntime"]:
      pass
    else:
      for idx, node in enumerate(model.graph.node):
        if node.op_type in ['Constant', 'AliasWithName']:
          continue
        for onode in node.output:
          intermediate_tensor_name = onode
          intermediate_layer_value_info = helper.ValueInfoProto()
          intermediate_layer_value_info.name = intermediate_tensor_name
          model.graph.output.extend([intermediate_layer_value_info])
    try:
      if get_model_size(model)*1.1 > 2 * 1024**3:
        onnx.save_model(model, all_inf_graph, all_tensors_to_one_file=False, save_as_external_data=True, size_threshold=200, convert_attribute=True)
      else:
        onnx.save(model, all_inf_graph, all_tensors_to_one_file=False, save_as_external_data=False)
    except:
      try:
        onnx.save_model(model, all_inf_graph, all_tensors_to_one_file=False, save_as_external_data=True, size_threshold=200, convert_attribute=True)
      except Exception as e:
        logger.error('Failed saving model in _save_all_inference_graph: %s', e)
    # save_onnx_model(model, all_inf_graph, True)
    args.input_weight = all_inf_graph
    logger.info("Preprocessed ONNX model saved [%s]", all_inf_graph)
  except Exception as e:
    logger.error("Failed to save an all_inference model for the model. Please check. [%s]\n %s", args.input_weight, str(e))
    logger.warn("... falling back to checkpoint model")
    model.graph.CopyFrom(model_chkpt.graph)


  # Check if the given input and output names are in the model
def onnx_preprocess(args):
  try:
    model = onnx.load_model(args.input_network, load_external_data=True)
    if args.input_node_name == ['None'] and args.output_node_name == ['None']:
      args.input_node_name = [i.name for i in model.graph.input]
    model_out = args.tmp_onnx + '.onnx'
    save_onnx_model(model, model_out)
    if args.output_node_name != ['None']:
      rv = _onnx_extract_subgraph(args, model_out)
      if not rv:
        return -1
    model = onnx.load_model(model_out, load_external_data=True)
    if args.transform_graph:
      _simplify_onnx_graph(args, model_out)
    model = onnx.load_model(model_out, load_external_data=True)
    input_name_dict, name_map = _node_name_preprocess(args, model, model_out)
    emit_name_map(name_map, args.output_dir)
    input_all = [node.name for node in model.graph.input]
    input_initializer = set(node.name for node in model.graph.initializer)
    net_feed_input = [i for i in input_all if i not in input_initializer]
    old_name_shape_dict = {i: j for i, j in zip(args.input_node_name, args.input_shape)}
    old_name_format_dict = {i: j for i, j in zip(args.input_node_name, args.data_format)}
    old_name_dformat_dict = {i: j for i, j in zip(args.input_node_name, args.dformat)}
    new_old_name_dict = {i: j for i, j in zip(input_name_dict.values(), input_name_dict.keys())}
    args.input_node_name = net_feed_input
    args.input_shape = [old_name_shape_dict[new_old_name_dict[i]] for i in args.input_node_name]
    args.data_format = [old_name_format_dict[new_old_name_dict[i]] for i in args.input_node_name]
    args.dformat = [old_name_dformat_dict[new_old_name_dict[i]] for i in args.input_node_name]
    model = onnx.load_model(model_out, load_external_data=True)
    args.input_network = model_out
    args.input_weight = model_out
    if args.output_node_name == ['None']:
      args.output_node_name = []
      for output in model.graph.output:
        args.output_node_name.append(output.name)
    # onnx.checker.check_model(model)
    all_inf_graph = os.path.join(args.output_dir, 'onnx_assets', '__model.onnx')
    _save_all_inference_graph(args, model, all_inf_graph)
    try:
      # validating forward pass
      if args.onnx_backend == "onnxruntime":
        logger.info('Validating forward pass using onnxruntime..')
        input_image_dict = generate_input(args)
        session = onnxruntime.InferenceSession(args.input_network, sess_options=_sess_options)
        output_name = [n.name for n in session.get_outputs()]
        result = session.run(output_name, input_image_dict[list(input_image_dict.keys())[0]])
        logger.info('Forward pass validated.')
    except Exception as e:
      logger.error('The preprocessed model is invalid: %s' % e)
      return -1
  except Exception as e:
    logger.error('Failed while preprocessing the model: %s' % e)
    return -1
  return 0


class onnx_validator:

  def __init__(self):
    pass

  @staticmethod
  def run_model(args):
    rv = 0
    try:
      onnx_model = onnx.load_model(args.input_network, load_external_data=True)
      try:
        onnx.save(onnx_model, args.tmp_onnx + '.onnx', all_tensors_to_one_file=False, save_as_external_data=False)
      except:
        onnx.save_model(onnx_model, args.tmp_onnx + '.onnx', all_tensors_to_one_file=False, save_as_external_data=True, size_threshold=200, convert_attribute=True)
        try:
          onnx.shape_inference.infer_shapes_path(args.input_network, args.tmp_onnx + '.onnx')
        except Exception as e:
          logger.error('failed shape inference: %s' % e)
    except Exception as e:
      logger.error('The model is invalid: %s' % e)
      logger.error(args.input_network)
      return -1
    remove_all_but_selected_files(os.path.join(args.output_dir, 'onnx_assets'), [args.tmp_onnx + '.onnx'])
    logger.debug("Starting Preprocessing ...")
    rv = onnx_preprocess(args)
    return rv
