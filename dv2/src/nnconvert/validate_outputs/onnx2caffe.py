'''
Copyright (c) 2018-22, Deep Vision, Inc. All rights reserved.
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
import onnxruntime
import os
import re
import pandas as pd
import sys as _sys

_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../python/caffe/proto')
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../analyzer/python')
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../python')
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2

import numpy as np
from dv_logging import get_logger
from onnx import helper
from utils.io_utils import *
import copy
import argparse
import google.protobuf as pb

_sess_options = onnxruntime.SessionOptions()
_sess_options.inter_op_num_threads = 1
_sess_options.intra_op_num_threads = 1
_sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
logger = get_logger(__name__, 'DVOPVL')

def sanitize_name(name):
  # Match a leading number followed by underscore
  match = re.match(r'^(\d+)_([A-Za-z_].*)$', name)
  if match:
      number, rest = match.groups()
      return f"{rest}_{number}"
  return name # if name is "380_DequantizeLinear, then new name will be DequantizeLinear_380"

def _output_tensor_to_name_map(onnx_model):
  onnx_model = onnx.load(onnx_model)
  _output_tensor_map = dict()
  _output_name_map = dict()
  # output tensor to name map
  for nodes in onnx_model.graph.node:
    for _idx, tensors in enumerate(nodes.output):
      _output_tensor_map[tensors] = nodes.name
      _output_name_map[nodes.name] = tensors
  return _output_tensor_map, _output_name_map


def mk_node_name(name):
  # Scopes created in a nested scope may have initial characters
  # that are illegal as the initial character of an op name
  # (viz. '-', '\', '/', and '_').
  name = name.replace('-','_')\
              .replace('\\','_')\
              .replace('/','_')\
              .replace('_','_')\
              .replace('[','_')\
              .replace(']','_')
  return f'{name}'


def prepare_df_to_dump(data, col_names):
  df = pd.DataFrame(data)
  df['index'] = np.arange(len(df)).tolist()
  df = df[col_names]
  return df


def generate_mse(args, caffe_net, data_dict, tensor_name_map, image):
  logger.debug("Starting Layerwise MSE generation.")
  emax_max_tolerance = 0.0001  # Should be made dynamic, based on the network and data. [Appropriate metric needed.]
  data = {
      'onnxname': [],
      'caffename': [],
      'Shape_Match': [],
      'onnx_op': [],
      'caffe_op': [],
      'MSE': [],
      'EMAX': [],
      'LOW/HIGH': [],
      'diff_mean': [],
      'onnx_mean': [],
      'caffe_mean': [],
      'onnx': [],
      'caffe': [],
      'onnx_std': [],
      'caffe_std': [],
      'onnx_json': [],
  }

  onnx.shape_inference.infer_shapes_path(args.input_weight, args.output_dir + "/onnx_assets/model.onnx")
  onnx_model = onnx.load_model(args.output_dir + "/onnx_assets/model.onnx")
  onnx_out = dict()
  if args.onnx_backend == "onnxruntime":
    session = onnxruntime.InferenceSession(args.input_weight, sess_options=_sess_options)
    output_name = [n.name for n in session.get_outputs()]
    result = session.run(None, data_dict)
    for idx, node in enumerate(output_name):
      onnx_out[node] = result[idx]
  '''
    # DEPRECATED: context upgrade versions of torch, onnx, onnxruntime
    elif args.onnx_backend == "caffe2":
        # model = onnx.load(model_path)
        model = onnx.load(args.input_network) 
        prepared_backend = Caffe2Backend.prepare(model)
        inf = prepared_backend.run(data_dict)
        onnx_out = dict(inf._asdict())
    '''
  caffe_all_names = [name for name in caffe_net._layer_names]
  for nodes in onnx_model.graph.node:
    old_name = tensor_name_map[nodes.output[0]]
    node_name = mk_node_name(old_name)

    possible_matches = [node_name]
    for matches in possible_matches:
      if matches in caffe_net.blobs.keys():
        try:
          caffe_data = caffe_net.blobs[matches].data[...]
          onnx_data = onnx_out[nodes.output[0]]
          if onnx_data.shape == caffe_data.shape:
            idx = caffe_all_names.index(matches)
            caffe_op = caffe_net.layers[idx].type
            diff = abs(caffe_data - onnx_data)
            data['onnxname'].append(tensor_name_map[nodes.output[0]])
            data['caffename'].append(matches)
            data['Shape_Match'].append('YES')
            data['onnx_op'].append(nodes.op_type)
            data['caffe_op'].append(caffe_op)
            data['MSE'].append(np.square(diff).mean())
            onnx_json = nodes
            E_AMAX = diff.max()
            E_PAMX = E_AMAX / np.abs(onnx_data).mean()
            data['EMAX'].append(E_AMAX)
            data['diff_mean'].append(diff.mean())
            data['onnx_mean'].append(onnx_data.mean())
            data['onnx_std'].append(onnx_data.std())
            data['caffe_mean'].append(caffe_data.mean())
            data['caffe_std'].append(caffe_data.std())
            data['onnx'].append(onnx_data[np.unravel_index(np.argmax(diff), diff.shape)])
            data['caffe'].append(caffe_data[np.unravel_index(np.argmax(diff), diff.shape)])
            if E_AMAX > emax_max_tolerance and E_PAMX > emax_max_tolerance:
              data['LOW/HIGH'].append('HIGH')
              data['onnx_json'].append(onnx_json)
            else:
              data['LOW/HIGH'].append('LOW')
              data['onnx_json'].append(None)
          else:
            data['onnxname'].append(tensor_name_map[nodes.output[0]])
            data['caffename'].append(matches)
            data['Shape_Match'].append('NO')
            data['MSE'].append(None)
            data['onnx_op'].append(None)
            data['caffe_op'].append(None)
            data['EMAX'].append(None)
            data['diff_mean'].append(None)
            data['onnx_mean'].append(None)
            data['onnx_std'].append(None)
            data['caffe_mean'].append(None)
            data['onnx'].append(None)
            data['caffe'].append(None)
            data['caffe_std'].append(None)
            data['LOW/HIGH'].append(None)
            data['onnx_json'].append(None)
        except Exception as e:
          logger.error(str(e))
      else:
        logger.debug("O2C layer not present in caffe %s", matches)

  col_names = ['index', 'onnxname', 'caffename', 'Shape_Match', 'MSE', 'onnx_op', 'caffe_op', 'EMAX', 'diff_mean', 'onnx_mean', 'onnx_std', 'caffe_mean', 'caffe_std', 'LOW/HIGH', 'onnx_json']

  df = prepare_df_to_dump(data, col_names)

  try:
    output_model = os.path.join(args.output_dir, 'stats', image)
    output_dir = os.path.join(args.output_dir, 'stats')
    name_map_out_dir = os.path.join(args.output_dir, 'stats', "layer_name_map.csv")
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    with open(str(output_model) + '.csv', 'w') as f:
      df.to_csv(f, index=False)
    mse_path = str(output_model) + '.csv'
    logger.info("MSE output: %s", mse_path)
  except Exception as e:
    logger.error("Layer wise MSE calculation error seen :%s", str(e))


def print_mse(onnx_model, onnx_out, caffe_net):
  for i in onnx_model.graph.node:
    if i.name in caffe_net.blobs:
      o_data = onnx_out[i.output[0]]
      c_data = caffe_net.blobs[i.name].data
      if o_data.shape != c_data.shape:
        logger.info(f"Onnx node {i.name} has shape mismatch. O - {o_data.shape}, C - {c_data.shape}")
        continue
      diff = abs(o_data - c_data)
      MSE = np.square(diff).mean()
      E_AMAX = diff.max()
      E_PAMX = E_AMAX / np.abs(o_data).mean()
      logger.info(f"For node {i.name}\nShape :{o_data.shape}\nMSE   :{MSE}\nE_AMAX: {E_AMAX}\nE_PAMX: {E_PAMX}")
      if E_AMAX > 0.0001 and E_PAMX > 0.0001:
        logger.info(f"\t\t\t\tINCORRECT")
    else:
      logger.info(f"Onnx node {i.name} not present in Caffe")
    logger.info("\n\n")


def emit_name_map(output_dir: str, output_model_name):

  caffe_prototxt = os.path.join(output_dir, 'params', f"{output_model_name}.prototxt")
  name_map = os.path.join(output_dir, 'stats', 'layer_name_map.csv')

  layer_pb = caffe_pb2.NetParameter()
  with open(caffe_prototxt, 'r') as f:
    pb.text_format.Merge(f.read(), layer_pb)
  caffe_layer_names = [layer.name for layer in layer_pb.layer]
  caffe_layer_type_map = dict()
  for layer in layer_pb.layer:
    caffe_layer_type_map[layer.name] = layer.type

  if not os.path.exists(name_map):
    logger.warn("[E] error while generating layer_name_map")

  name_mapping = pd.read_csv(name_map, sep=',')
  name_mapping = name_mapping.iloc[:, 1:]
  csv_names = ['index', 'Src_Name', 'Intermediate_name', 'ARANet_name', 'SrcType', 'ARANetType']
  name_mapping["ARANetType"] = name_mapping["ARANetType"].astype("object")
  src_name = csv_names[1]
  aranet_name = csv_names[2]
  IntermediateName = csv_names[3]
  for idx, node in enumerate(name_mapping['Intermediate_name']):
    if node not in caffe_layer_names:
      name_mapping.loc[idx, "ARANet_name"] = "Optimized Out"
      continue

    r = re.compile(f"{node}_ara.*")
    node_family = list(filter(r.match, caffe_layer_names))
    node_family.append(node)
    name_mapping.loc[idx, "ARANet_name"] = ', '.join(node_family)
    name_mapping.loc[idx, "ARANetType"] = ', '.join([caffe_layer_type_map[layer] for layer in node_family])

  col_names = ['Src_Name', 'Intermediate_name', 'ARANet_name', 'SrcType', 'ARANetType']
  df = prepare_df_to_dump(name_mapping, col_names)
  with open(name_map, 'w') as f:
    df.to_csv(f, index=False)
  logger.debug("[A] name mapping done onnx2caffe")


class onnx2caffe:

  @staticmethod
  def compare_outputs(args, mse_logs):
    emit_name_map(args.output_dir, args.output_model_name)
    logger.info("Begining NW validation.")
    data_dict = dict()
    data_dict_onnx = dict()
    tensor_name_map, output_name_map = _output_tensor_to_name_map(args.input_network)
    output_name = []
    net_outs = False
    if args.output_node_name is None:
      args.output_node_name = 'None'
    for node in args.output_node_name:
      if node not in tensor_name_map.values():
        logger.info("No such node present in the graph. Running till net outs.")
        net_outs = True
        break
    output_model = os.path.join(args.output_dir, 'params', args.output_model_name)

    caffe_net = caffe.Net(output_model + '.prototxt', output_model + '.caffemodel', caffe.TEST)
    input_shape = dict()
    shape_list = list()
    input_layers = list()
    emax_max_tolerance = 0.0001
    for layer in caffe_net.layer_dict.keys():
      if caffe_net.layer_dict[layer].type == 'Input':
        input_layers.append(layer)
    for layer in input_layers:
      inp_shape = list()
      for i in caffe_net.blobs[layer].shape:
        inp_shape.append(i)
      input_shape[layer] = inp_shape
    try:
      onnx.shape_inference.infer_shapes_path(args.input_network, args.output_dir + "/onnx_assets/model.onnx")
      onnx_model = onnx.load_model(args.output_dir + "/onnx_assets/model.onnx")
    except Exception as e:
      logger.error("error in onnx.shape_inference.infer_shapes: %s,continuing", str(e))

    output_name = list()
    input_node_name = args.input_node_name
    session = None
    if args.onnx_backend == "onnxruntime":
      session = onnxruntime.InferenceSession(args.input_network, sess_options=_sess_options)
      # session = onnxruntime.InferenceSession(args.input_weight, sess_options=_sess_options)
      output_name = [n.name for n in session.get_outputs()]
      input_node_name = [n.name for n in session.get_inputs()]
      args.input_node_name = input_node_name
    # ''' TODO: Need to change this once multi input support is added.
    #     Proper name mapping b/w onnx input names and caffe input names needed.
    # '''
    else:
      input_all = [node.name for node in onnx_model.graph.input]
      input_initializer = [node.name for node in onnx_model.graph.initializer]
    new_args = argparse.Namespace(**vars(args))
    new_args.input_node_name = input_node_name
    input_image_dict = generate_input(args)
    rv = 0
    for image in input_image_dict:
      onnx_out = dict()
      result = list()
      flag = 0
      if args.onnx_backend == "onnxruntime":
        assert session is not None
        result = session.run(None, input_image_dict[image])
        for idx, node in enumerate(output_name):
          onnx_out[node] = result[idx]
      '''
            # Deprecated: context upgrdae versions for torch, onnx, onnxruntime
            elif args.onnx_backend == "caffe2": 
                model = onnx.load(args.input_network)
                prepared_backend = Caffe2Backend.prepare(model)
                external_outputs = list()
                for node in prepared_backend.predict_net.op: 
                    external_outputs.extend(node.output)
                if False:
                    # hack caffe2 net to tap all node output
                    prepared_backend.predict_net.external_output[:] = external_outputs                 
                inf = prepared_backend.run(input_image_dict[image])
                inf = inf._asdict()
                output_name = list(inf.keys())
                onnx_out = dict(inf)
                result = [value for key, value in onnx_out.items()]
            '''
      caffe_net = caffe.Net(output_model + '.prototxt', output_model + '.caffemodel', caffe.TEST)
      blobs_layer_map = {}
      for k, v in caffe_net.top_names.items():
        for i in v:
          blobs_layer_map[i] = k
      for i in range(len(input_layers)):
        caffe_net.blobs[input_layers[i]].data[...] = input_image_dict[image][input_layers[i]]
      caffe_out = caffe_net.forward()
      # print_mse(onnx_model,onnx_out,caffe_net)
      generate_mse(args, caffe_net, input_image_dict[image], tensor_name_map, image)
      if len(caffe_out) < len(result):
        mse_logs.append("Output nodes count mismatch.")
        return -1
      error_amax = list()
      error_pamx = list()
      error_ramx = list()
      caffe_out_layers = list(caffe_out.keys())
      for onode, o_out in onnx_out.items():
        onode = sanitize_name(onode)
        c_out = None
        """
        since we are adding an extra reshape to account for 
        different blob name compared to layer name, one 
        shouldn't rely on tensor_name_map for final layer 
        map. e.g. tvo_model_slot_attention.
        """
        onode_name = onode  # if onode not in tensor_name_map else tensor_name_map[onode]
        fork_prefix = '_splitOutput'
        caffe_layer = None
        # breakpoint()
        if (onode_name + fork_prefix) in caffe_net.blobs and any([node.startswith(onode_name) for node in caffe_out_layers]):
          c_out = caffe_net.blobs[onode_name + fork_prefix].data[...]
          caffe_layer = blobs_layer_map[onode_name + fork_prefix]
        elif onode_name in caffe_net.blobs and any([node.startswith(onode_name) for node in caffe_out_layers]):
          c_out = caffe_net.blobs[onode_name].data[...]
          caffe_layer = blobs_layer_map[onode_name]
        if c_out is None:
          logger.error("node corresponding to onnx not present in caffe %s", c_out)
          return -1
        if o_out.shape == c_out.shape:
          try:
            if caffe_net.layer_dict[caffe_layer].type == 'ArgMax':
              c_out = c_out[:, 0, ...]
          except:
            logger.error("layer not found in caffe %s", onode_name)
          diff = abs(o_out - c_out)
          if diff.size == 0:
            continue
          MSE = np.square(diff).mean()
          E_AMAX = diff.max()
          E_PAMX = E_AMAX / np.abs(o_out).mean()
          E_RAMX = (diff / abs(o_out)).max()
          error_amax.append(E_AMAX)
          error_pamx.append(E_PAMX)
          error_ramx.append(E_RAMX)
          mse_logs.append("\n\nOutput ONNX layer:    {} \nOutput Caffe Layer: {}\nImage: {}\n--------------------\nMSE: {:.8f}\nE_AMAX: {:.8f}\nE_PAMX: {:.8f}\nE_RAMX: {:.8f}\n".format(onode, caffe_layer, image, MSE, E_AMAX, E_PAMX, E_RAMX))
          logger.debug("Error stats: amean: %s, mean: %s, min: %s, max: %s, std: %s", np.abs(diff).mean(), diff.mean(), diff.min(), diff.max(), diff.std())
        else:
          mse_logs.append("Shapes Mismatch : {} :: {} || Layer_caffe  {}".format(c_out.shape, o_out.shape, caffe_layer))

      flag_count = 0
      for ind in range(len(error_amax)):
        if error_amax[ind] < emax_max_tolerance or error_pamx[ind] < emax_max_tolerance or error_ramx[ind] < emax_max_tolerance:
          flag_count += 1
      if flag_count < len(onnx_out):
        logger.error("Validation failed, EMAX > emax_max_tolerance, emax_max_tolerance = %s", emax_max_tolerance)
        rv = -1
    if args.ignore_mse:
      logger.warn("ignoring mse")
      return 0
    return rv
