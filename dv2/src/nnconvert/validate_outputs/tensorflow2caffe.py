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

import tensorflow as tf
import sys as _sys
import os
import numpy as np
import collections
import re
import warnings
import pandas as pd
import json
from utils.import_helpers import get_variable_name
from utils.import_helpers import dump_json
from utils.import_helpers import dump_np_data
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../python/caffe/proto')
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../analyzer/python')
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../python')
import caffe
from tensorflow.core.protobuf import meta_graph_pb2 as meta_graph_pb2, config_pb2, rewriter_config_pb2
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from dv_logging import get_logger
from pathlib import Path
import shutil
import yaml
from utils.io_utils import *
from utils.preprocessor import Preprocessor
#tf.disable_v2_behavior()
logger = get_logger(__name__, 'DVOPVL')


def dump_input(args, input_image_dict):
  out_dir = os.path.join(args.output_dir, 'inputs', 'debug')
  if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
  os.makedirs(out_dir, exist_ok=True)
  for key, val in input_image_dict.items():
    for k, v in val.items():
      _fname = k.replace('/', '_')
      logger.info("Input %s saved: %s", k, os.path.join(out_dir, _fname))
      np.save(os.path.join(out_dir, _fname), v)


def dump_data(tf_dict, caffe_net, data_dict, output_dir, image):
  output_dir = os.path.join(output_dir, 'converter', 'debug')
  dvnet_outputs_dir = os.path.join(output_dir, image, 'outputs', 'dvnet_fp32')
  dvnet_params_dir = os.path.join(output_dir, 'dvnet_params')
  tf_outputs_dir = os.path.join(output_dir, image, 'outputs', 'tf_fp32')
  net_inputs_dir = os.path.join(output_dir, image, 'inputs')
  if os.path.exists(dvnet_outputs_dir):
    shutil.rmtree(dvnet_outputs_dir)
  if os.path.exists(dvnet_params_dir):
    shutil.rmtree(dvnet_params_dir)
  if os.path.exists(tf_outputs_dir):
    shutil.rmtree(tf_outputs_dir)
  if os.path.exists(net_inputs_dir):
    shutil.rmtree(net_inputs_dir)
  os.makedirs(dvnet_outputs_dir, exist_ok=True)
  os.makedirs(dvnet_params_dir, exist_ok=True)
  os.makedirs(tf_outputs_dir, exist_ok=True)
  os.makedirs(net_inputs_dir, exist_ok=True)
  for lid, layer in enumerate(tf_dict):
    fname = layer + '.npy'
    np.save(os.path.join(tf_outputs_dir, fname), tf_dict[layer][0])
  for bid, blob in enumerate(caffe_net.blobs):
    fname = blob + '.npy'
    np.save(os.path.join(dvnet_outputs_dir, fname), caffe_net.blobs[blob].data)
  for pid, param in enumerate(caffe_net.params):
    for idx in range(len(caffe_net.params[param])):
      fname = param + '__' + str(pid) + '__' + str(idx) + '.npy'
      np.save(os.path.join(dvnet_params_dir, fname), caffe_net.params[param][idx].data)
  np.save(os.path.join(net_inputs_dir, 'inputs.npy'), data_dict)


def load_caffe_graph(prototxt_path, model_path):
  net = caffe.Net(prototxt_path, caffe.TEST, weights=model_path)
  return net


def generate_mse(args, tf_output, net, caffe_all_names, image):
  logger.debug("Starting Layerwise MSE generation.")
  emax_max_tolerance = 0.0001  # Should be made dynamic, based on the network and data. [Appropriate metric needed.]
  data = {
      'tfname': [],
      'caffename': [],
      'Shape_Match': [],
      'tf_op': [],
      'caffe_op': [],
      'MSE': [],
      'EMAX': [],
      'LOW/HIGH': [],
      'diff_mean': [],
      'tf_mean': [],
      'caffe_mean': [],
      'tf_std': [],
      'caffe_std': [],
      'tf_json': [],
  }
  for keys, _ in tf_output.items():
    r = re.compile(keys + '_.*')
    possible_matches = [line for line in caffe_all_names if (line == keys or r.match(line))]
    for matches in possible_matches:
      if matches in net.blobs:
        try:
          caffe_data = net.blobs[matches].data[...]
          tf_data = tf_output[keys][0]
          if len(tf_data.shape) == 4:
            tf_data = tf_data.transpose(0, 3, 1, 2)
          elif len(tf_data.shape) == 3:
            tf_data = tf_data.transpose(0, 2, 1)
          if (tf_data.shape == caffe_data.shape):
            idx = caffe_all_names.index(matches)
            caffe_op = net.layers[idx].type
            diff = abs(caffe_data - tf_data)
            data['tfname'].append(keys.replace('_', '/'))
            data['caffename'].append(matches)
            data['Shape_Match'].append('YES')
            data['tf_op'].append(tf_output[keys][1])
            data['caffe_op'].append(caffe_op)
            data['MSE'].append(np.square(diff).mean())
            tf_json = tf_output[keys][2]
            E_AMAX = diff.max()
            E_PAMX = E_AMAX / np.abs(tf_data).mean()
            data['EMAX'].append(E_AMAX)
            data['diff_mean'].append(diff.mean())
            data['tf_mean'].append(tf_data.mean())
            data['tf_std'].append(tf_data.std())
            data['caffe_mean'].append(caffe_data.mean())
            data['caffe_std'].append(caffe_data.std())
            if E_AMAX > emax_max_tolerance and E_PAMX > emax_max_tolerance:
              data['LOW/HIGH'].append('HIGH')
              data['tf_json'].append(tf_json)
            else:
              data['LOW/HIGH'].append('LOW')
              data['tf_json'].append(None)
          else:
            data['tfname'].append(keys.replace('_', '/'))
            data['caffename'].append(matches)
            data['Shape_Match'].append('NO')
            data['MSE'].append(None)
            data['tf_op'].append(None)
            data['caffe_op'].append(None)
            data['EMAX'].append(None)
            data['diff_mean'].append(None)
            data['tf_mean'].append(None)
            data['tf_std'].append(None)
            data['caffe_mean'].append(None)
            data['caffe_std'].append(None)
            data['LOW/HIGH'].append(None)
            data['tf_json'].append(None)
        except Exception as e:
          logger.error(str(e))
  col_names = [
      'index', 'tfname', 'caffename', 'Shape_Match', 'MSE', 'tf_op', 'caffe_op', 'EMAX', 'diff_mean', 'tf_mean', 'tf_std', 'caffe_mean', 'caffe_std', 'LOW/HIGH', 'tf_json'
  ]
  df = pd.DataFrame(data)
  df['index'] = np.arange(len(df)).tolist()
  df = df[col_names]
  try:
    output_model = os.path.join(args.output_dir, 'stats', image)
    output_dir = os.path.join(args.output_dir, 'stats')
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    with open(str(output_model) + '.csv', 'w') as f:
      df.to_csv(f, index=False)
    mse_path = str(output_model) + '.csv'
    logger.info("MSE output: %s", mse_path)
  except Exception as e:
    logger.error("Layer wise MSE calculation error seen :%s", str(e))


def compare_outputs_debug(args, generate_MSE_file=True):
  if not generate_MSE_file:
    return
  input_shape = list()
  tf_output = collections.OrderedDict()
  if args.input_shape != None:
    for x in args.input_shape:
      shape = x.split(',')
      input_shape.append([int(x) for x in shape])
  with tf.io.gfile.GFile(args.input_weight, 'rb') as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
  # Add comment here
  graph_def = strip_unused_lib.strip_unused(input_graph_def=graph_def,
                                            input_node_names=args.input_node_name,
                                            output_node_names=args.output_node_name,
                                            placeholder_type_enum=dtypes.float32.as_datatype_enum)
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")
    input_node_name = args.input_node_name[0]
    inp_tensor = graph.get_tensor_by_name(input_node_name + ':0')
    input_image_dict = generate_input(args)
    output_model = os.path.join(args.output_dir, 'params', args.output_model_name)
    net = load_caffe_graph(output_model + '.prototxt', output_model + '.caffemodel')
    config = config_pb2.ConfigProto()
    config.graph_options.rewrite_options.remapping = (rewriter_config_pb2.RewriterConfig.OFF)
    with tf.compat.v1.Session(config=config) as sess:
      layers = []
      tf_op = []
      for node in graph_def.node:
        try:
          # Ignoring Const, Identity and Assert for layerwise MSE.
          if node.op not in ['Const', 'Identity', 'Assert']:
            if "is_training" in node.attr and node.attr["is_training"].b:
              pass
            else:
              new_tensor = graph.get_tensor_by_name(node.name + ':0')
              layers.append(new_tensor)
              tf_op.append(node.op)
        except Exception as e:
          logger.info("Possible Error : %s", e)
      for image in input_image_dict:
        feed_dict = {}
        for i in args.input_node_name:
          feed_dict[graph.get_tensor_by_name(i + ':0')] = input_image_dict[image][i]
          caffe_data = clast2first(input_image_dict[image][i])
          net.blobs[i.replace('/', '_')].data[...] = caffe_data
        net.forward()
        caffe_all_names = [n for n in net._layer_names]
        output_hidden_layers = sess.run(layers, feed_dict=feed_dict)
        for ind in range(len(output_hidden_layers)):
          cur_layer = layers[ind].name
          converted_layer = cur_layer.replace('/', '_')
          converted_layer = re.sub(":0", '', converted_layer)
          tf_output[converted_layer] = [output_hidden_layers[ind], tf_op[ind], str(layers[ind].op)]
        generate_mse(args, tf_output, net, caffe_all_names, image)
        if args.dump_debug:
          dump_data(tf_output, net, input_image_dict[image], args.output_dir, image)


class tensorflow2caffe:

  @staticmethod
  def compare_outputs(args, mse_logs):
    warnings.filterwarnings("ignore")
    tf_output = collections.OrderedDict()
    input_shape = list()
    caffe_outputs = list()
    data_dict = generate_input_default(args)['default']
    feed_dict = dict()
    layer_map = collections.OrderedDict()
    layer_map["input_nodes"] = dict()
    layer_map["output_nodes"] = dict()
    # Calling "" compare_outputs_debug "" for layerwise logging
    # if args.dump_debug:
    compare_outputs_debug(args, generate_MSE_file=True)
    input_image_dict = generate_input(args)
    output_model = os.path.join(args.output_dir, 'params', args.output_model_name)
    caffe_net = caffe.Net(output_model + '.prototxt', caffe.TEST, weights=output_model + '.caffemodel')
    out_node_names = args.output_node_name
    if args.input_shape != None:
      for x in args.input_shape:
        shape = x.split(',')
        input_shape.append([int(x) for x in shape])
    for i in range(len(args.input_node_name)):
      layer_map["input_nodes"][args.input_node_name[i]] = get_variable_name(args.input_node_name[i])
    for i in args.input_node_name:
      caffe_net.blobs[i.replace('/', '_')].data[...] = clast2first(data_dict[i])
    caffe_out = caffe_net.forward()
    if len(caffe_out) != len(out_node_names):
      logger.error("Number of output nodes doesn't match.")
      return -1
    with tf.io.gfile.GFile(args.input_weight, 'rb') as f:
      graph_def = tf.compat.v1.GraphDef()
      graph_def.ParseFromString(f.read())
    # Add comment here
    graph_def = strip_unused_lib.strip_unused(input_graph_def=graph_def,
                                              input_node_names=args.input_node_name,
                                              output_node_names=args.output_node_name,
                                              placeholder_type_enum=dtypes.float32.as_datatype_enum)
    rv = 0
    with tf.Graph().as_default() as graph:
      tf.import_graph_def(graph_def, name="")
      net = load_caffe_graph(output_model + '.prototxt', output_model + '.caffemodel')
      with tf.compat.v1.Session() as sess:
        layers = list()
        tf_op = list()
        for node in out_node_names:
          try:
            # Fill up all the dict with all the nodes available.
            new_tensor = graph.get_tensor_by_name(node + ':0')
            layers.append(new_tensor)
            tf_op.append(new_tensor.op)
          except Exception as e:
            logger.info("Possible Error : %s", e)
        emax_max_tolerance = 0.0001  # Fix metric.
        for image in input_image_dict:
          for inp in input_image_dict[image]:
            preprocessed_file = os.path.join(args.output_dir, 'inputs', image, 'fp32')
            dump_np_data(input_image_dict[image][inp], preprocessed_file, inp.replace('/', '_'))
          for i in args.input_node_name:
            feed_dict[graph.get_tensor_by_name(i + ':0')] = input_image_dict[image][i]
            caffe_data = clast2first(input_image_dict[image][i])
            net.blobs[i.replace('/', '_')].data[...] = caffe_data
          net.forward()
          caffe_dict = net
          output_hidden_layers = sess.run(layers, feed_dict=feed_dict)
          for ind in range(len(output_hidden_layers)):
            cur_layer = layers[ind].name
            converted_layer = cur_layer.replace('/', '_')
            converted_layer = re.sub(":0", '', converted_layer)
            tf_output[converted_layer] = [output_hidden_layers[ind], tf_op[ind], str(layers[ind].op)]
          error_amax = list()
          error_pamx = list()
          caffe_ = list(caffe_out.keys())
          blobs_layer_map = {}
          for k, v in caffe_dict.top_names.items():
            for i in v:
              blobs_layer_map[i] = k
          for i in range(len(out_node_names)):
            tf_layer = out_node_names[i].replace('/', '_')
            for l in caffe_:
              tf_data = tf_output[tf_layer][0]
              caffe_data = caffe_dict.blobs[l].data[...]
              if len(tf_data.shape) == 5:
                tf_data = tf_data.transpose(0, 4, 1, 2, 3)
              elif len(tf_data.shape) == 4:
                if tf_output[tf_layer][1].type == 'ArgMin':
                  tf_data = tf_data.reshape(caffe_data.transpose(0, 2, 3, 4, 1).shape)
                  tf_data = tf_data.transpose(0, 4, 1, 2, 3)
                elif caffe_dict.layer_dict[l].type == 'Tf_Argmax':
                  tf_data = tf_data.reshape(caffe_data.transpose(0, 2, 3, 4, 1).shape)
                  tf_data = tf_data.transpose(0, 4, 1, 2, 3)
                elif tf_output[tf_layer][1].type != 'ArgMax':
                  tf_data = tf_data.transpose(0, 3, 1, 2)
              elif len(tf_data.shape) == 3:
                if tf_output[tf_layer][1].type == 'ArgMin':
                  tf_data = tf_data.reshape(caffe_data.transpose(0, 2, 3, 1).shape)
                  tf_data = tf_data.transpose(0, 3, 1, 2)
                elif caffe_dict.layer_dict[l].type == 'Tf_Argmax':
                  tf_data = tf_data.reshape(caffe_data.transpose(0, 2, 3, 1).shape)
                  tf_data = tf_data.transpose(0, 3, 1, 2)
                elif tf_output[tf_layer][1].type != 'ArgMax':
                  tf_data = tf_data.transpose(0, 2, 1)
              # else:
              #     tf_data = tf_data.transpose(0,2,1)
              if caffe_dict.layer_dict[blobs_layer_map[l]].type == 'ArgMax':
                caffe_data = caffe_dict.blobs[l].data[:, 0, ...]
              if caffe_data.shape == tf_data.shape:
                if np.all(np.isnan(tf_data) == np.isnan(caffe_data)):  #Hyperbolic Ops
                  tf_data = np.nan_to_num(tf_data)
                  caffe_data = np.nan_to_num(caffe_data)
                  diff = abs(tf_data - caffe_data)
                  MSE = np.square(diff).mean()
                  E_AMAX = diff.max()
                  E_PAMX = E_AMAX / np.abs(tf_data).mean()
                else:
                  logger.info("Error: np.NaN for a non np.NaN value present in data")
                  diff = np.ones(caffe_data.shape)
                  MSE = np.square(diff).mean()
                  E_AMAX = 1
                  E_PAMX = 1
                  rv = -1
                if E_AMAX < emax_max_tolerance or E_PAMX < emax_max_tolerance:
                  layer_map["output_nodes"][out_node_names[i]] = l
                  tf_out_file = os.path.join(args.output_dir, 'outputs', image, 'fp32', 'source')
                  caffe_out_file = os.path.join(args.output_dir, 'outputs', image, 'fp32', 'dvnet')
                  dump_np_data(tf_data, tf_out_file, tf_layer)
                  dump_np_data(caffe_data, caffe_out_file, l)
                  logger.info("Source net output saved to : %s, layer : %s", tf_out_file, tf_layer)
                  logger.info("DVNet net output saved to : %s, layer : %s", caffe_out_file, l)
                error_amax.append(E_AMAX)
                error_pamx.append(E_PAMX)
                logger.info("\n\nOutput TF layer:    %s \nOutput Caffe Layer: %s\nImage: %s\n--------------------\nMSE: %s\nE_AMAX: %s\nE_PAMX: %s\n", out_node_names[i], l, image,
                            MSE, E_AMAX, E_PAMX)
                logger.debug("Error stats: amean: %s, mean: %s, min: %s, max: %s, std: %s", np.abs(diff).mean(), diff.mean(), diff.min(), diff.max(), diff.std())
                logger.debug("TF blob stats: amean: %s, mean: %s, min: %s, max: %s, std: %s", np.abs(tf_data).mean(), tf_data.mean(), tf_data.min(), tf_data.max(), tf_data.std())
              else:
                logger.info("Shapes Mismatch : %s :: %s || Layer_caffe  %s :: Layer_tf :: %s", caffe_data.shape, tf_data.shape, l, out_node_names[i])
          flag_count = 0
          for ind in range(len(error_amax)):
            if error_amax[ind] < emax_max_tolerance or error_pamx[ind] < emax_max_tolerance:
              flag_count += 1
          if flag_count < len(out_node_names):
            logger.error("Validation failed, EMAX > emax_max_tolerance, emax_max_tolerance = %s", emax_max_tolerance)
            logger.debug("Saving input image: %s", os.path.join(args.output_dir, 'inputs', 'debug'))
            dump_input(args, input_image_dict)
            rv = -1
        od = os.path.join(args.output_dir, 'stats')
        if not os.path.exists(od):
          os.makedirs(od)
        out_file = os.path.join(args.output_dir, 'stats', 'layer_map.json')
        dump_json(layer_map, out_file)
        logger.info("Layer map data dumped to file : %s", out_file)
    if args.ignore_mse:
      logger.warn("ignoring mse")
      return 0
    logger.info("Validation Ok!")
    return rv
