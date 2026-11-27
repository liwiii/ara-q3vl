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

import subprocess
import tensorflow as tf
from tensorflow.python.framework import tensor_util
import sys as _sys
import os
import dv_logging
import pandas as pd
import re
logger = dv_logging.get_logger(__name__, 'DVTFSSD')


def preprocess(args, temp_dir):

  def _tf2litepb_cmd(args, temp_dir, add_postprocessing_op):
    CMD = [str(_sys.executable), os.path.dirname(os.path.abspath(__file__)) + '/tf_obj/object_detection/export_tflite_ssd_graph.py']
    CMD.append("--pipeline_config_path=" + args.config)
    CMD.append("--trained_checkpoint_prefix=" + args.input_weight)
    if add_postprocessing_op == 'true':
      new_dir = os.path.join(temp_dir, 'tflite')
    else:
      new_dir = os.path.join(temp_dir, 'tflite_pruned')
    CMD.append("--output_directory=" + new_dir)
    CMD.append("--add_postprocessing_op=" + add_postprocessing_op)
    logger.info("running command: %s", CMD)
    return CMD, new_dir

  def _litepb2lite(args, new_dir, input_shape):
    input_arrays = ["normalized_input_image_tensor"]
    input_shapes = {"normalized_input_image_tensor": [1] + input_shape[1:] + [input_shape[0]]}
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(os.path.join(new_dir, 'tflite_graph.pb'), input_arrays, output_arrays, input_shapes)
    converter.inference_type = tf.compat.v1.lite.constants.FLOAT
    converter.output_format = tf.compat.v1.lite.constants.TFLITE
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    open(os.path.join(new_dir, 'detect.tflite'), "wb").write(tflite_model)

  input_shape = []
  if args.input_shape != None:
    for x in args.input_shape:
      shape = x.split(',')
      shape = shape[1:]
      input_shape.append([int(x) for x in shape])
  add_postprocessing_op = 'true'
  cmd, new_dir = _tf2litepb_cmd(args, temp_dir, add_postprocessing_op)
  x = subprocess.run(cmd)
  output_arrays = ["TFLite_Detection_PostProcess", "TFLite_Detection_PostProcess:1", "TFLite_Detection_PostProcess:2", "TFLite_Detection_PostProcess:3"]
  _litepb2lite(args, new_dir, input_shape[0])
  add_postprocessing_op = 'false'
  cmd, new_dir2 = _tf2litepb_cmd(args, temp_dir, add_postprocessing_op)
  x = subprocess.run(cmd)
  output_arrays = ["raw_outputs/class_predictions", "raw_outputs/box_encodings", "anchors"]
  return {
      "pruned_graph": os.path.join(new_dir2, 'tflite_graph.pb'),
      "compat_graph": os.path.join(new_dir, 'tflite_graph.pb'),
      "tflite_graph": os.path.join(new_dir, 'detect.tflite'),
      "inode": ['normalized_input_image_tensor'],
      "onode": output_arrays
  }


def preprocess_names(args):
  #FIXME: Move the reusable code to a function
  # Special characters
  spl_char_string = r'[-.]'
  regex = re.compile(spl_char_string)
  csv_file = os.path.join(os.path.abspath(args.output_dir), "_model-name-map.csv")
  new_model_dir = os.path.abspath(args.output_dir) + "/model"
  new_model_file = os.path.abspath(args.output_dir) + "/__model.pb"
  try:
    if os.path.isdir(args.input_weight):
      logger.info("Renaming layer names in the keras model")
      model = tf.keras.models.load_model(args.input_weight)
      config = model.get_config()
      old_to_new = {}
      new_to_old = {}
      # Updating layers names and creating a mapping dictionary
      for layer in config['layers']:
        new_name = layer['name']
        if regex.search(layer['name']) is not None:
          for ch in spl_char_string:
            new_name = new_name.replace(ch, '_')
        old_to_new[layer['name']], new_to_old[new_name] = new_name, layer['name']
        layer['name'] = new_name
        layer['config']['name'] = new_name
        # Updating the parent names
        if len(layer['inbound_nodes']) > 0:
          for in_node in layer['inbound_nodes'][0]:
            in_node[0] = old_to_new[in_node[0]]
      # Updating the input and output layers
      for input_layer in config['input_layers']:
        input_layer[0] = old_to_new[input_layer[0]]
      for output_layer in config['output_layers']:
        output_layer[0] = old_to_new[output_layer[0]]
      new_model = tf.keras.Model().from_config(config)
      # Set the weights for the new model from old model
      for layer in new_model.layers:
        layer.set_weights(model.get_layer(new_to_old[layer.name]).get_weights())
      # Updating the input and output node names
      for i in range(len(args.input_node_name)):
        if args.input_node_name[i] in old_to_new:
          args.input_node_name[i] = old_to_new[args.input_node_name[i]]
      for i in range(len(args.output_node_name)):
        if args.output_node_name[i] in old_to_new:
          args.output_node_name[i] = old_to_new[args.output_node_name[i]]
      # Saving the new model
      model.save(new_model_dir)
      args.input_weight = new_model_dir
      # Creating a dictionary for the renamed layers
      map_dict = {}
      for name in old_to_new:
        if name != old_to_new[name]:
          map_dict[name] = old_to_new[name]
      # Export the dictionary to csv file
      if len(map_dict) > 0:
        df = pd.DataFrame(map_dict.items(), columns=["Original", "Renamed"])
        df.to_csv(csv_file)
        logger.info("Rename map exported to: {}".format(csv_file))
      return 0
    else:
      from tensorflow.python.platform import gfile
      logger.info("Renaming layer names in the frozen graph")
      # Get the graph def
      model = tf.compat.v1.GraphDef()
      with open(args.input_weight, 'rb') as f:
        serialized = f.read()
      model.ParseFromString(serialized)
      # Creating a mapping dictionary
      old_to_new = {}
      for node in model.node:
        new_name = node.name
        if regex.search(new_name) is not None:
          for ch in spl_char_string:
            new_name = new_name.replace(ch, '_')
        old_to_new[node.name] = new_name
      # Change the layer names and input names
      for node in model.node:
        node.name = old_to_new[node.name]
        new_inputs = []
        for n in node.input:
          split = n.split(':')
          if len(split) > 0 and split[-1].isdigit():
            split[0] = old_to_new[split[0]]
            new_inputs.append(':'.join(split))
          else:
            new_inputs.append(old_to_new[n])
        for i in range(len(node.input)):
          node.input.pop()
        node.input.extend(new_inputs)
      # Saving the new graph def
      with gfile.GFile(new_model_file, "wb") as f:
        f.write(model.SerializeToString())
      print("Serializing done!")
      # Updating the input weight path
      args.input_weight = new_model_file
      # Updating the input and output node names
      for i in range(len(args.input_node_name)):
        if args.input_node_name[i] in old_to_new:
          args.input_node_name[i] = old_to_new[args.input_node_name[i]]
      for i in range(len(args.output_node_name)):
        if args.output_node_name[i] in old_to_new:
          args.output_node_name[i] = old_to_new[args.output_node_name[i]]
      # Creating a dictionary for the renamed layers
      map_dict = {}
      for name in old_to_new:
        if name != old_to_new[name]:
          map_dict[name] = old_to_new[name]
      # Export the dictionary to csv file
      if len(map_dict) > 0:
        df = pd.DataFrame(map_dict.items(), columns=["Original", "Renamed"])
        df.to_csv(csv_file)
        logger.info("Rename map exported to: {}".format(csv_file))
      return 0
  except Exception as e:
    logger.error(e)
    return -1
