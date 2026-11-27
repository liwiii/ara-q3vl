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

import onnx, numpy as np
import tensorflow as tf, tf2onnx
import os, sys
from validate_source.tensorflow_validator import get_frozen_model
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from dv_logging import get_logger
from utils.io_utils import *
import re
import onnxruntime
_sess_options = onnxruntime.SessionOptions()
_sess_options.inter_op_num_threads = 1
_sess_options.intra_op_num_threads = 1
_sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
logger = get_logger(__name__, 'DVONFB')


def run(args):
  if args.force_fallback:
    logger.info("args.force_fallback = True, running the model after converting it into onnx")
  else:
    logger.info("Error occured during " + args.src_framework + " run. Retrying after converting it to onnx")
  try:
    if args.src_framework == 'mxnet':
      logger.error("Mxnet support has been deprecated")
      input_shapes = [tuple(int(j) for j in i.split(',')) for i in args.input_shape]
      new_path = mxnet.onnx.export_model(args.input_network, args.input_weight, input_shapes, np.float32, os.path.join(args.output_dir, '__model.onnx'))
      onnx.checker.check_model(new_path)
      args.input_network = args.input_weight = new_path
      args.src_framework = 'onnx'
    elif args.src_framework == 'tensorflow':
      get_frozen_model(args)
      with tf.io.gfile.GFile(args.input_weight, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        input_names = [i + ':0' for i in args.input_node_name]
        output_names = [i + ':0' for i in args.output_node_name]
        tf_input_node_name = args.input_node_name
        tf_output_node_name = args.output_node_name
        tf_input_shape = args.input_shape
        tf_data_format = args.data_format
        input_image_dict = generate_input(args)
        onnx_model = tf2onnx.convert.from_graph_def(graph_def=graph_def, input_names=input_names, output_names=output_names)
        onnx_model = onnx_model[0]
        input_name_dict = {i: j for i, j in zip(input_names, args.input_node_name)}
        output_name_dict = {i: j for i, j in zip(output_names, args.output_node_name)}
        for i in onnx_model.graph.input:
          i.name = input_name_dict.get(i.name, i.name)
        for o in onnx_model.graph.output:
          o.name = output_name_dict.get(o.name, o.name)
        for n in onnx_model.graph.node:
          for i in range(len(n.input)):
            n.input[i] = input_name_dict.get(n.input[i], n.input[i])
          for o in range(len(n.output)):
            n.output[o] = output_name_dict.get(n.output[o], n.output[o])
        new_path = os.path.join(args.output_dir, '__model.onnx')
        onnx.save(onnx_model, new_path)
        args.input_network = args.input_weight = new_path
        input_shape = [i.split(',') for i in args.input_shape]
        input_shape = [i[:1] + i[2:] + i[1:2] for i in input_shape]
        input_shape = [','.join(i) for i in input_shape]
        args.input_shape = input_shape
        args.src_framework = 'onnx'
        formats = {'HWC': 'CHW', 'HC': 'CH', 'WC': 'CW', 'DHWC': 'CDHW', 'NHWC': 'NCHW'}
        args.data_format = [formats.get(i, i) for i in args.data_format]
        on_input_node_name = args.input_node_name
        on_output_node_name = args.output_node_name
        on_input_shape = args.input_shape
        on_data_format = args.data_format
      graph_def = strip_unused_lib.strip_unused(input_graph_def=graph_def,
                                            input_node_names=tf_input_node_name,
                                            output_node_names=tf_output_node_name,
                                            placeholder_type_enum=dtypes.float32.as_datatype_enum)
      with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        session = onnxruntime.InferenceSession(args.input_network, sess_options=_sess_options)
        with tf.compat.v1.Session() as sess:
          layers = list()
          tf_op = list()
          for node in tf_output_node_name:
            try:
              # Fill up all the dict with all the nodes available.
              new_tensor = graph.get_tensor_by_name(node + ':0')
              layers.append(new_tensor)
              tf_op.append(new_tensor.op)
            except Exception as e:
              logger.info("Possible Error : %s", e)
          for image in input_image_dict:
            feed_dict = dict()
            tf_output = dict()
            onnx_out = dict()
            for i in tf_input_node_name:
              feed_dict[graph.get_tensor_by_name(i + ':0')] = input_image_dict[image][i]
            output_hidden_layers = sess.run(layers, feed_dict=feed_dict)
            for ind in range(len(output_hidden_layers)):
              cur_layer = layers[ind].name
              converted_layer = cur_layer.replace('/', '_')
              converted_layer = re.sub(":0", '', converted_layer)
              tf_output[converted_layer] = [output_hidden_layers[ind], tf_op[ind], str(layers[ind].op)]
            result = session.run(None, input_image_dict[image])
            output_name = [n.name for n in session.get_outputs()]
            for idx, node in enumerate(output_name):
              onnx_out[node.replace('/', '_')] = result[idx]
            error_amax = list()
            error_pamx = list()
            for onode, o_out in onnx_out.items():
              for tnode, t_out in tf_output.items():
                if o_out.shape == t_out[0].shape:
                  diff = abs(o_out - t_out[0])
                  if diff.size == 0:
                    continue
                  MSE = np.square(diff).mean()
                  E_AMAX = diff.max()
                  E_PAMX = E_AMAX / np.abs(o_out).mean()
                  error_amax.append(E_AMAX)
                  error_pamx.append(E_PAMX)
                  logger.debug("Error stats: amean: %s, mean: %s, min: %s, max: %s, std: %s", np.abs(diff).mean(), diff.mean(), diff.min(), diff.max(), diff.std())
            flag_count = 0
            emax_max_tolerance = 0.0001 
            for ind in range(len(error_amax)):
              if error_amax[ind] < emax_max_tolerance or error_pamx[ind] < emax_max_tolerance:
                flag_count += 1
            if flag_count < len(tf_output_node_name):
               raise Exception("Validation failed for converted onnx vs original tf model, EMAX > emax_max_tolerance, emax_max_tolerance = "+ str(emax_max_tolerance))
      args.tf2onnx_happened = True
  except Exception as e:
    logger.info("Error occured while converting to onnx: ", str(e))
    return 1
  return 0
