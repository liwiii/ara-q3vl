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

import mxnet as mx
import mxnet.symbol as sym
import numpy as np
import warnings
import mxnet.ndarray as nd
import sys as _sys
import os
import shutil
import collections
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../python/caffe/proto')
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../analyzer/python')
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../python')
from utils.import_helpers import get_variable_name
from utils.import_helpers import dump_json
import caffe
import caffe.proto.caffe_pb2 as caffe_pb2
import google.protobuf as pb
import google.protobuf.text_format
from dv_logging import get_logger
import yaml
from utils.preprocessor import Preprocessor
from utils.io_utils import *
from utils.import_helpers import dump_np_data
from pathlib import Path
import pandas as pd
import json
import copy
logger = get_logger(__name__, 'DVOPVL')


def looks_like_weight(name):
  weight_like = ('_weight', '_bias', '_beta', '_gamma', '_moving_var', '_moving_mean', '_running_var', '_running_mean')
  return name.endswith(weight_like)


def dump_input(args, input_image_dict):
  out_dir = os.path.join(args.output_dir, 'inputs', 'debug')
  if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
  os.makedirs(out_dir, exist_ok=True)
  for key, val in input_image_dict.items():
    for k, v in val.items():
      _fname = k
      logger.info("Input %s saved: %s", _fname, os.path.join(out_dir, _fname))
      np.save(os.path.join(out_dir, _fname), v)


def dump_data(mx_lwise, mx_json, caffe_net, data_dict, output_dir, image):
  output_dir = os.path.join(output_dir, 'converter', 'debug')
  dvnet_outputs_dir = os.path.join(output_dir, image, 'outputs', 'dvnet_fp32')
  dvnet_params_dir = os.path.join(output_dir, 'dvnet_params')
  mxnet_outputs_dir = os.path.join(output_dir, image, 'outputs', 'mxnet_fp32')
  net_inputs_dir = os.path.join(output_dir, image, 'inputs')
  if os.path.exists(dvnet_outputs_dir):
    shutil.rmtree(dvnet_outputs_dir)
  if os.path.exists(dvnet_params_dir):
    shutil.rmtree(dvnet_params_dir)
  if os.path.exists(mxnet_outputs_dir):
    shutil.rmtree(mxnet_outputs_dir)
  if os.path.exists(net_inputs_dir):
    shutil.rmtree(net_inputs_dir)
  os.makedirs(dvnet_outputs_dir, exist_ok=True)
  os.makedirs(dvnet_params_dir, exist_ok=True)
  os.makedirs(mxnet_outputs_dir, exist_ok=True)
  os.makedirs(net_inputs_dir, exist_ok=True)
  all_layers = mx_json.get_internals()
  for lid, layer in enumerate(mx_lwise):
    fname = all_layers[lid].name + '.ndy'
    mx.nd.save(os.path.join(mxnet_outputs_dir, fname), layer)
    fname = all_layers[lid].name + '.npy'
    np.save(os.path.join(mxnet_outputs_dir, fname), layer.asnumpy())
  for bid, blob in enumerate(caffe_net.blobs):
    fname = blob + '.npy'
    np.save(os.path.join(dvnet_outputs_dir, fname), caffe_net.blobs[blob].data)
  for pid, param in enumerate(caffe_net.params):
    for idx in range(len(caffe_net.params[param])):
      fname = param + '__' + str(pid) + '__' + str(idx) + '.npy'
      np.save(os.path.join(dvnet_params_dir, fname), caffe_net.params[param][idx].data)
  np.save(os.path.join(net_inputs_dir, 'inputs.npy'), data_dict)


"""
fetches the protobuf layer object from 
proto net for debug purpose
"""


def fetch_layer_proto(layer_name, proto_input_net):
  for layer in proto_input_net.layer:
    if layer.name == layer_name:
      return str(layer)
  return ''


def find_states(sym_json, shape_dict):
  """
    states hold the name and shape
    of each rnn state in the model
    which needs to be passed
    explicitly in arg_params.
    Imp:Batch-size is determined from
    network's batch-size.
    """
  symbol = sym.load(sym_json)
  all_layers = symbol.get_internals()
  states = dict()
  souts = []
  with open(sym_json, 'r') as f:
    x = json.load(f)
  for i in x['nodes']:
    if i['op'] == 'RNN':
      inputs = i['inputs']
      if len(inputs) > 2:
        h0 = x['nodes'][inputs[2][0]]
        if h0['op'] == 'null':
          state_out = i['name'] + '_state_output'
          arg_shape, oshape, aux_shape = all_layers[state_out].infer_shape(**shape_dict)
          # assumption : state output has single blob
          assert oshape is not None
          states[h0['name']] = oshape[0]
          souts.append(state_out)
      if len(inputs) > 3:
        c0 = x['nodes'][inputs[3][0]]
        if c0['op'] == 'null':
          state_out = i['name'] + '_statecell_output'
          arg_shape, oshape, aux_shape = all_layers[state_out].infer_shape(**shape_dict)
          assert oshape is not None
          states[c0['name']] = oshape[0]
          souts.append(state_out)
  return states, souts


class mxnet2caffe:

  @staticmethod
  def compare_outputs(args, mse_logs):

    def compare_outputs_debug(args, proto_input_net):
      input_shape = get_input_shape(args.input_shape)
      state_shape = get_input_shape(args.state_shape)
      label_name = None
      label_shape = None
      if args.label_shape:
        if not args.label_name:
          logger.error("please provide both label name and shape")
          return -1
      if args.label_name:
        if not args.label_shape:
          logger.error("please provide both label name and shape")
          return -1
      input_label_shape = []
      if args.label_shape:
        input_label_shape = get_input_shape(args.label_shape)
      if args.label_name:
        if len(args.label_name) != len(args.label_shape):
          logger.error("number of label names and shapes didn't match")
          return -1
        label_name = tuple(args.label_name)
        label_shape = []
        for i in range(len(args.label_name)):
          label_shape.append((args.label_name[i], tuple(input_label_shape[i])))
      warnings.filterwarnings("ignore")
      symbol = sym.load(args.input_network)
      save_dict = nd.load(args.input_weight)
      arg_params = {}
      if save_dict:
        arg_params = save_dict
      all_layers = symbol.get_internals()
      output_model = os.path.join(args.output_dir, 'params', args.output_model_name)
      caffe_net = caffe.Net(output_model + '.prototxt', caffe.TEST, weights=output_model + '.caffemodel')
      data = {
          'mxname': [],
          'caffename': [],
          'mxtype': [],
          'MSE': [],
          'EMAX': [],
          'LOW/HIGH': [],
          'mxjson': [],
          'caffelayer': [],
          'diff_mean': [],
          'mx_mean': [],
          'mx_std': [],
          'caffe_mean': [],
          'caffe_std': [],
          'mx_emax_data': [],
          'caffe_emax_data': [],
      }
      data_shapes = []
      for i in range(len(args.input_node_name)):
        data_shapes.append((
            args.input_node_name[i],
            tuple(input_shape[i]),
        ))
      states = dict()
      inputs = [mx.sym.var(i) for i in args.input_node_name]
      model = mx.gluon.nn.SymbolBlock(outputs=list(all_layers), inputs=inputs)
      model.load_parameters(args.input_weight, ctx=mx.cpu(), allow_missing=True, ignore_extra=True, cast_dtype=True)
      for idx, i in enumerate(args.state_name):
        states[i] = state_shape[idx]
      for i in states:
        # arg_params[i] = mx.nd.array(np.random.random_sample(states[i]))
        arg_params[i] = model.params[i].data()
      for i in states:
        if i in caffe_net.blobs:
          logger.debug("state [%s] found in dvnet", i)
          caffe_net.blobs[i].data[...] = arg_params[i].asnumpy()
      input_image_dict = generate_input(args)
      possible_suffix = [
          '', '_crop', '_crop_crop', '_padding_crop', '_padding', '_scale', '_reshape1', '_splitOutput', '_output0', '_output1', '_output2', '_state_output', '_statecell_output',
          '_output0'
      ]
      emax_max_tolerance = 0.0001
      for image in input_image_dict:
        data_list = []
        index = 0
        for i in range(len(input_shape)):
          if args.input_node_name[i] in caffe_net.blobs:
            caffe_net.blobs[args.input_node_name[i]].data[...] = input_image_dict[image][args.input_node_name[i]]
          data_list.append(mx.nd.array(input_image_dict[image][args.input_node_name[i]]))
        caffe_out = caffe_net.forward()
        mx_out = model(*data_list)
        if not isinstance(mx_out, list):
          mx_out = [mx_out]
        for lindex, layer in enumerate(all_layers):
          layer_dict = json.loads(layer.tojson())
          layer_dict = layer_dict['nodes'][len(layer_dict['nodes']) - 1]
          if layer_dict["op"] == "null" or layer_dict["name"] == "data":
            index = index + 1
            continue
          else:
            min_diff = float("inf")
            counter = 0
            for suffix in possible_suffix:
              try:
                caffe_layer = layer_dict['name'] + suffix
                if caffe_layer in caffe_net.blobs:
                  caffe_data = caffe_net.blobs[caffe_layer].data
                  mx_data = mx_out[index].asnumpy()
                  if caffe_data.shape != mx_data.shape:
                    continue
                  diff = abs(caffe_data - mx_data)
                  counter = counter + 1
                  if layer_dict['op'] == 'RNN':
                    data['mxname'].append(all_layers[index].list_outputs()[0])
                  else:
                    data['mxname'].append(layer_dict['name'])
                  data['caffename'].append(caffe_layer)
                  data['mxtype'].append(layer_dict['op'])
                  data['MSE'].append(np.square(diff).mean())
                  new_diff = diff.max()
                  if counter == 0:
                    min_diff = new_diff
                  data['EMAX'].append(new_diff)
                  data['diff_mean'].append(diff.mean())
                  data['mx_mean'].append(mx_data.mean())
                  data['mx_std'].append(mx_data.std())
                  data['caffe_mean'].append(caffe_data.mean())
                  data['caffe_std'].append(caffe_data.std())
                  data['caffelayer'].append(fetch_layer_proto(caffe_layer, proto_input_net))
                  index_diff_max = np.unravel_index(np.argmax(diff, axis=None), caffe_data.shape)
                  data['mx_emax_data'].append(mx_data[index_diff_max])
                  data['caffe_emax_data'].append(caffe_data[index_diff_max])
                  if new_diff < min_diff:
                    min_diff = new_diff
              except Exception as e:
                logger.error(str(e))
            if min_diff >= emax_max_tolerance:
              data['mxjson'].extend([layer_dict] * counter)
              data['LOW/HIGH'].extend(['HIGH'] * counter)
            else:
              data['mxjson'].extend([''] * counter)
              data['LOW/HIGH'].extend(['LOW'] * counter)
              data['caffelayer'][-1] = ''
          index = index + 1
        df = pd.DataFrame(data)
        output_model = os.path.join(args.output_dir, 'stats', image)
        output_dir = os.path.join(args.output_dir, 'stats')
        if not os.path.exists(output_dir):
          os.makedirs(output_dir)
        if args.dump_debug:
          dump_data(mx_out, symbol, caffe_net, input_image_dict[image], args.output_dir, image)
        df.to_csv(str(output_model) + '.csv')
      return

    proto_input_net = caffe_pb2.NetParameter()
    output_model = os.path.join(args.output_dir, 'params', args.output_model_name)
    with open(output_model + '.prototxt', 'r') as f:
      pb.text_format.Merge(f.read(), proto_input_net)
    # if args.dump_debug:
    try:
      compare_outputs_debug(args, proto_input_net)
    except Exception as e:
      logger.error("couldn't generate mse file, %s", str(e))
    input_shape = get_input_shape(args.input_shape)
    state_shape = get_input_shape(args.state_shape)
    label_name = None
    label_shape = None
    if args.label_shape:
      if not args.label_name:
        logger.error("please provide both label name and shape")
        return -1
    if args.label_name:
      if not args.label_shape:
        logger.error("please provide both label name and shape")
        return -1
    input_label_shape = []
    if args.label_shape:
      for x in args.label_shape:
        shape = x.split(',')
        input_label_shape.append([int(x) for x in shape])
    if args.label_name:
      if len(args.label_name) != len(args.label_shape):
        logger.error("number of label names and shapes didn't match")
        return -1
      label_name = tuple(args.label_name)
      label_shape = []
      for i in range(len(args.label_name)):
        label_shape.append((args.label_name[i], tuple(input_label_shape[i])))
    data_shapes = []
    data_dict = generate_input_default(args)['default']
    data_list = []
    for i in range(len(args.input_node_name)):
      data_shapes.append((
          args.input_node_name[i],
          tuple(input_shape[i]),
      ))
      data_list.append(mx.nd.array(data_dict[args.input_node_name[i]]))
    warnings.filterwarnings("ignore")
    symbol = sym.load(args.input_network)
    save_dict = nd.load(args.input_weight)
    arg_params = {}
    if save_dict:
      arg_params = save_dict
    all_layers = symbol.get_internals()
    output_model = os.path.join(args.output_dir, 'params', args.output_model_name)
    caffe_net = caffe.Net(output_model + '.prototxt', caffe.TEST, weights=output_model + '.caffemodel')
    for i in range(len(input_shape)):
      if args.input_node_name[i] in caffe_net.blobs:
        caffe_net.blobs[args.input_node_name[i]].data[...] = data_dict[args.input_node_name[i]]
    caffe_out = caffe_net.forward()
    mx_all_outs = all_layers.list_outputs()
    mx_sym_group = []
    caffe_mapped_layers = []
    for i in caffe_out.keys():
      if i in mx_all_outs:
        mx_name = i
        caffe_mapped_layers.append(i)
        mx_sym_group.append(all_layers[mx_name])
      elif i + '_output0' in mx_all_outs:
        mx_name = i + '_output0'
        caffe_mapped_layers.append(i)
        mx_sym_group.append(all_layers[mx_name])
      elif i + '_output' in mx_all_outs:
        mx_name = i + '_output'
        caffe_mapped_layers.append(i)
        mx_sym_group.append(all_layers[mx_name])
      elif i.endswith('_padding') and (i[:-8] in mx_all_outs):
        mx_name = i[:-8]
        caffe_mapped_layers.append(i)
        mx_sym_group.append(all_layers[mx_name])
      elif i.endswith('_padding') and (i[:-8] + '_output' in mx_all_outs):
        mx_name = i[:-8] + '_output'
        caffe_mapped_layers.append(i)
        mx_sym_group.append(all_layers[mx_name])
      elif i.endswith('_crop') and (i[:-5] in mx_all_outs):
        mx_name = i[:-5]
        caffe_mapped_layers.append(i)
        mx_sym_group.append(all_layers[mx_name])
      elif i.endswith('_crop') and (i[:-5] + '_output' in mx_all_outs):
        mx_name = i[:-5] + '_output'
        caffe_mapped_layers.append(i)
        mx_sym_group.append(all_layers[mx_name])
      elif i.endswith('_crop_crop') and (i[:-10] + '_output' in mx_all_outs):
        mx_name = i[:-10] + '_output'
        caffe_mapped_layers.append(i)
        mx_sym_group.append(all_layers[mx_name])
      elif i.endswith('_reshape1') and (i[:-9] + '_output' in mx_all_outs):
        mx_name = i[:-9] + '_output'
        caffe_mapped_layers.append(i)
        mx_sym_group.append(all_layers[mx_name])
      elif i.endswith('_reshape2') and (i[:-9] + '_output' in mx_all_outs):
        mx_name = i[:-9] + '_output'
        caffe_mapped_layers.append(i)
        mx_sym_group.append(all_layers[mx_name])
      elif i.endswith('_scale') and (i[:-6] + '_output' in mx_all_outs):
        mx_name = i[:-6] + '_output'
        caffe_mapped_layers.append(i)
        mx_sym_group.append(all_layers[mx_name])
      elif i.endswith('_reshape3') and (i[:-9] + '_output' in mx_all_outs):
        mx_name = i[:-9] + '_output'
        caffe_mapped_layers.append(i)
        mx_sym_group.append(all_layers[mx_name])
      elif i.endswith('_splitOutput') and (i[:-12] + '_output' in mx_all_outs):
        mx_name = i[:-12] + '_output'
        caffe_mapped_layers.append(i)
        mx_sym_group.append(all_layers[mx_name])
      elif i.endswith('_perm2') and (i[:-6] + '_output' in mx_all_outs):
        mx_name = i[:-6] + '_output'
        caffe_mapped_layers.append(i)
        mx_sym_group.append(all_layers[mx_name])
      else:
        logger.warn("Couldn't map back node for output comparison: %s", i)
    mx_grp_names = []
    for i in mx_sym_group:
      mx_grp_names.append(i.name)
    for i in symbol:
      if i.name not in mx_grp_names:
        mx_sym_group = mx_sym_group + [i]
    inputs = [mx.sym.var(i) for i in args.input_node_name]
    model = mx.gluon.nn.SymbolBlock(outputs=mx_sym_group, inputs=inputs)
    model.load_parameters(args.input_weight, ctx=mx.cpu(), allow_missing=True, ignore_extra=True, cast_dtype=True)
    shape_dict = dict()
    for i in data_shapes:
      shape_dict[i[0]] = i[1]
    # states, souts = find_states(args.input_network, shape_dict)
    _, souts = find_states(args.input_network, shape_dict)
    states = dict()
    for idx, i in enumerate(args.state_name):
      states[i] = state_shape[idx]
    for i in states:
      # arg_params[i] = mx.nd.array(np.random.random_sample(states[i]))
      arg_params[i] = model.params[i].data()
    rv = 0
    input_image_dict = generate_input(args)
    for image in input_image_dict:
      for inp in input_image_dict[image]:
        preprocessed_file = os.path.join(args.output_dir, 'inputs', image, 'fp32')
        dump_np_data(input_image_dict[image][inp], preprocessed_file, inp)
      data_list = []
      caffe_net = caffe.Net(output_model + '.prototxt', caffe.TEST, weights=output_model + '.caffemodel')
      for i in range(len(input_shape)):
        if args.input_node_name[i] in caffe_net.blobs:
          caffe_net.blobs[args.input_node_name[i]].data[...] = input_image_dict[image][args.input_node_name[i]]
        data_list.append(mx.nd.array(input_image_dict[image][args.input_node_name[i]]))
      for i in states:
        if i in caffe_net.blobs:
          logger.debug("state [%s] found in dvnet", i)
          caffe_net.blobs[i].data[...] = arg_params[i].asnumpy()
        else:
          logger.error("state [%s] not found in dvnet", i)
          return -1
      caffe_out = caffe_net.forward()
      caffe_outputs = list(caffe_out.keys())
      temp_outputs = copy.deepcopy(caffe_outputs)
      for idx, i in enumerate(caffe_outputs):
        if (i in souts) and (i not in symbol.list_outputs()):
          temp_outputs.remove(i)
      caffe_outputs = temp_outputs
      symbol_outputs = []
      for name in symbol.list_outputs():
        if not looks_like_weight(name):
          symbol_outputs.append(name)
      if len(caffe_outputs) != len(symbol_outputs):
        logger.error('Unequal number of outputs')
      try:
        mx_out = model(*data_list)
        if not isinstance(mx_out, list):
          mx_out = [mx_out]
        one_output = mx_out[0].asnumpy()
      except Exception as e:
        logger.error("Conversion done.Failed in MXNet forward pass.", exc_info=e)
        return -1
      error_amax = []
      error_pamx = []
      error_ramx = [] #introducing this for handling et models very high values flowing through
      for i in range(len(caffe_mapped_layers)):
        caffe_final_out = caffe_out[caffe_mapped_layers[i]]
        for l in proto_input_net.layer:
          if caffe_mapped_layers[i] in l.top and l.type == 'ArgMax':
            caffe_final_out = caffe_final_out[:, 0, ...]
        mx_final_out = mx_out[i].asnumpy()
        diff = abs(mx_final_out - caffe_final_out)
        MSE = np.square(diff).mean()
        E_AMAX = diff.max()
        error_amax.append(E_AMAX)
        E_PAMX = E_AMAX / np.abs(mx_final_out).mean()
        error_pamx.append(E_PAMX)
        error_ramx.append((diff/abs(mx_final_out)).max())

        mxnet_out_file = os.path.join(args.output_dir, 'outputs', image, 'fp32', 'source')
        caffe_out_file = os.path.join(args.output_dir, 'outputs', image, 'fp32', 'dvnet')
        dump_np_data(mx_final_out, mxnet_out_file, caffe_mapped_layers[i])
        dump_np_data(caffe_final_out, caffe_out_file, caffe_mapped_layers[i])
        logger.info("Source net output saved to : %s, layer : %s", mxnet_out_file, caffe_mapped_layers[i])
        logger.info("DVNet net output saved to : %s, layer : %s", caffe_out_file, caffe_mapped_layers[i])
        logger.info("\n\nOutput: %s\nImage: %s\n--------------------\nMSE: %s\nE_AMAX: %s\nE_PAMX: %s\n", caffe_mapped_layers[i], image, MSE, E_AMAX, E_PAMX)
      emax_max_tolerance = 0.0001
      for i in range(len(error_amax)):
        if error_amax[i] > emax_max_tolerance and error_pamx[i] > emax_max_tolerance and error_ramx[i] > emax_max_tolerance:
          logger.error("validation failed, EMAX > emax_max_tolerance, emax_max_tolerance = %s", emax_max_tolerance)
          rv = -1
    if rv == -1:
      logger.debug("Saving input image: %s", os.path.join(args.output_dir, 'inputs', 'debug'))
      dump_input(args, input_image_dict)
    if args.ignore_mse:
      logger.warn("ignoring mse")
      return 0
    logger.info("validation ok")
    return rv
