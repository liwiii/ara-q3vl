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

import tensorflow as tf
from tensorflow.python.keras.saving import saving_utils as _saving_utils
from utils.io_utils import *
from . import tf_loader
from . import rutils
import os
import uuid
import numpy as np
import re
from tensorflow.python.platform import gfile
import copy
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from dv_logging import get_logger
logger = get_logger(__name__, 'DVIPVL')
keras_class_to_op = {
    'BatchNormalization': 'FusedBatchNormV3',
    'ZeroPadding2D': 'Pad',
    'GlobalAveragePooling2D': 'Mean',
    'Dropout': 'Identity',
    'Concatenate': 'ConcatV2',
    'AveragePooling2D': 'AvgPool',
    'MaxPooling2D': 'MaxPool'
}
SPECIAL_CHARS = [".", "/", ":", "[", "]", ",", "{", "}", "-", "\\", ";"]


def special(name):
  for char in SPECIAL_CHARS:
    if char in name:
      return True
  if name.isdigit():
    return True
  return False


def mk_node_name(name):
  # Scopes created in a nested scope may have initial characters
  # that are illegal as the initial character of an op name
  # (viz. '-', '\', '/', and '_').
  name = name.replace('-','_')\
              .replace('\\','_')\
              .replace('/','_')\
              .replace('_','_')\
              .replace('[','_')\
              .replace(']','_')\
              .replace(':', '_')\
              .replace('.','_')
  # .split(':')[0]
  return '{}'.format(name)


# API to optimize grappler, provided  by tf2onnx
def tf_optimize_grappler(input_names, output_names, graph_def, fold_constant=None):
  from tensorflow.core.protobuf import meta_graph_pb2 as meta_graph_pb2, config_pb2, rewriter_config_pb2
  from tensorflow.python.grappler import tf_optimizer as tf_opt
  config = config_pb2.ConfigProto()
  rewrite_options = config.graph_options.rewrite_options
  config.graph_options.infer_shapes = True
  rewrite_options.optimizers[:] = [
      #  'pruning', 'constfold', 'arithmetic', 'dependency', 'function',
      'constfold',
      'function'
  ]
  meta_graph = tf.compat.v1.train.export_meta_graph(graph_def=graph_def)
  fetch_collection = meta_graph_pb2.CollectionDef()
  for t in input_names + output_names:
    fetch_collection.node_list.value.append(t)
  meta_graph.collection_def["train_op"].CopyFrom(fetch_collection)
  graph_def = tf_opt.OptimizeGraph(config, meta_graph)
  return graph_def


# API to optimize the graph. Performs constant folding, provided by tf2onnx
def tf_optimize(input_names, output_names, graph_def, fold_constant=True):
  """Extract inference subgraph and optimize graph."""
  needed_names = [rutils.node_name(i) for i in input_names] + \
                 [rutils.node_name(i) for i in output_names]
  graph_def = tf.compat.v1.graph_util.extract_sub_graph(graph_def, needed_names)
  graph_def = tf_optimize_grappler(input_names, output_names, graph_def, fold_constant)
  return graph_def


# API provided by tf2onnx
def _rename_duplicate_keras_model_names(model):
  old_out_names = None
  if model.output_names and len(set(model.output_names)) != len(model.output_names):
    old_out_names = model.output_names
    used_names = set()
    new_out_names = []
    for name in model.output_names:
      new_name = name
      i = 0
      while new_name in used_names:
        i += 1
        new_name = name + "_" + str(i)
      used_names.add(new_name)
      new_out_names.append(new_name)
    model.output_names = new_out_names
  return old_out_names


# API provided by tf2onnx
def tensor_names_from_structed(concrete_func, input_names, output_names):
  tensors_to_rename = {}
  flat_structured_inp = tf.nest.flatten(concrete_func.structured_input_signature)
  structured_inputs = [t.name for t in flat_structured_inp if isinstance(t, tf.TensorSpec)]
  tensors_to_rename.update(zip(input_names, structured_inputs))
  if isinstance(concrete_func.structured_outputs, dict):
    for k, v in concrete_func.structured_outputs.items():
      tensors_to_rename[v.name] = k
  return tensors_to_rename


# API provided by tf2onnx (Modified to capture the mapping between the names before and after freezing the model)
def freeze_model(model, args):
  # Get the model signature
  all_sigs = model.signatures.keys()
  valid_sigs = [s for s in all_sigs if not s.startswith("_")]
  # Generate the concrete function from the model
  if len(valid_sigs) > 0:
    concrete_func = model.signatures[valid_sigs[0]]
  else:
    input_shape = []
    input_tensors = []
    if args.input_shape != None:
      for x in args.input_shape:
        shape = x.split(',')
        input_shape.append([int(x) for x in shape])
    for i, sl in enumerate(input_shape):
      input_shape[i] = [sl[0]] + sl[2:] + [sl[1]]
    for shape in input_shape:
      input_tensors.append(tf.TensorSpec(shape=tuple(shape)))
    function = _saving_utils.trace_model_call(model, input_tensors)
    concrete_func = function.get_concrete_function()
  tensors_to_rename = {}
  inputs = [tensor.name for tensor in concrete_func.inputs if tensor.dtype != tf.dtypes.resource]
  graph_captures = concrete_func.graph.captures
  if isinstance(graph_captures, dict):
      # Old TF: dict mapping {Tensor -> Placeholder}
      captured_inputs = [t_name.name for _, t_name in graph_captures.values()]
  else:
      # New TF: list of (Tensor, Placeholder) tuples
      captured_inputs = [t_name.name for _, t_name in graph_captures]
  inputs = [inp for inp in inputs if inp not in captured_inputs]
  if len(args.output_node_name) > 0:
    outputs = [op + ':0' for op in args.output_node_name]
  else:
    outputs = [tensor.name for tensor in concrete_func.outputs if tensor.dtype != tf.dtypes.resource]
  # Freeze the graph
  frozen_graph, initialized_tables = tf_loader.from_trackable(model, concrete_func, inputs, outputs, False)
  # Optimize the frozen graph
  tf.compat.v1.reset_default_graph()
  with tf.compat.v1.Session() as sess:
    frozen_graph = tf_optimize(inputs, outputs, frozen_graph)
  tf.compat.v1.reset_default_graph()
  # Save the frozen graph
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(frozen_graph, name="")
    tf.io.write_graph(graph, os.getcwd(), 'frozen.pb', as_text=False)
  # Update the input_weight argument
  args.input_weight = os.path.join(os.getcwd(), 'frozen.pb')


def get_frozen_model(args):
  skip_freeze = False
  try:
    # Load the model
    try:
      # Load with standard keras API
      logger.info("Attempting to load the model using keras API")
      model = tf.keras.models.load_model(args.input_weight)
    except Exception as e:
      logger.info("Attempting to load the model using saved_model API")
      # Load the model with tensorflow saved_model API
      model = tf.saved_model.load(args.input_weight)
  except Exception as e:
    if os.path.isdir(args.input_weight):
      logger.error("The model directory path is not valid")
      return -1
    else:
      skip_freeze = True
      logger.info("The model might already be frozen")
  #Freeze the model
  if not skip_freeze:
    logger.info("Freezing the model")
    freeze_model(model, args)


class tensorflow_validator:

  @staticmethod
  def run_model(args):
    new_model_file = os.path.abspath(args.output_dir) + "/__model.pb"

    def _transform_model(args, data_dict, orig_tf_outputs, orig_graph_def):
      try:
        if args.transform_graph:
          return 0
          logger.info('running tf graph transformer')
          #from tensorflow_core.tools.graph_transforms import TransformGraph as tg
          #opt_graph= tg(orig_graph_def,args.input_node_name,args.output_node_name,['fold_constants(ignore_errors=true)','fold_batch_norms','fold_old_batch_norms','remove_nodes(op=Identity)'])
          opt_graph = orig_graph_def
          with tf.Graph().as_default() as new_graph:
            tf.import_graph_def(opt_graph, name="")
            feed_dict = {}
            for i in args.input_node_name:
              feed_dict[new_graph.get_tensor_by_name(i + ':0')] = data_dict[i]
            output_tensors = []
            for i in args.output_node_name:
              output_tensors.append(new_graph.get_tensor_by_name(i + ':0'))
            with tf.compat.v1.Session() as sess2:
              new_tf_outputs = sess2.run(output_tensors, feed_dict=feed_dict)
            if len(new_tf_outputs) != len(orig_tf_outputs):
              logger.error("number of out nodes changed after graph transformation")
              return -1
            for i in range(len(new_tf_outputs)):
              if (0 not in new_tf_outputs[i].shape) and (0 not in orig_tf_outputs[i].shape):
                logger.info("EMAX after graph transformation %s: %f", args.output_node_name[i], abs(new_tf_outputs[i] * 1 - orig_tf_outputs[i] * 1).max())
              else:
                logger.warn("Encountered 0 in shape")
            tg_filename = "dv_netspec_defn_" + uuid.uuid4().hex + "_tg.pb"
            logger.info("transformed graph:%s", os.path.join(args.output_dir, tg_filename))
            tf.io.write_graph(new_graph, args.output_dir, tg_filename, as_text=False)
            args.input_weight = os.path.join(args.output_dir, tg_filename)
        else:
          return 0
      except Exception as e:
        logger.error("exception occurred when transforming tensorflow network: ", exc_info=e)
        return -1

    get_frozen_model(args)
    try:
      # Get the graph def
      model = tf.compat.v1.GraphDef()
      with open(args.input_weight, 'rb') as f:
        serialized = f.read()
      model.ParseFromString(serialized)
      iname_map = {}
      if args.iname is not None and len(args.iname):
        for name in args.iname:
          new_inp_name = name.split('@')[1]
          if special(new_inp_name):
            raise ValueError(f"iname provided: {new_inp_name} contains invalid characters")
          iname_map[name.split('@')[0]] = new_inp_name
      input_list = []
      for node in model.node:
        if node.op == "Placeholder":
          input_list.append(node)
      model_input_name = [node.name for node in input_list]
      for name in iname_map:
        if name not in model_input_name:
          raise ValueError(f"iname provided: {name} not found in model graph input, please provide valid inames")
      # spl_char_string = r'[-.]'
      # regex = re.compile(spl_char_string)
      # Creating a mapping dictionary
      old_to_new = {}
      for node in model.node:
        new_name = node.name
        # if regex.search(new_name) is not None:
        #     for ch in spl_char_string:
        #         new_name = new_name.replace(ch, '_')
        old_to_new[node.name] = new_name
      for idx, node in enumerate(input_list):
        old_name = node.name
        new_name = node.name
        if len(iname_map.keys()) or special(old_name):
          if old_name in iname_map:
            new_name = iname_map[old_name]
          else:
            if len(iname_map.keys()):
              logger.error("input not mentioned in iname")
            else:
              new_name = mk_node_name(old_name)
              if new_name.isdigit():
                new_name = "input_{}".format(idx)
        old_to_new[old_name] = new_name
      # Change the layer names and input names
      for node in model.node:
        node.name = old_to_new[node.name]
        new_inputs = []
        for n in node.input:
          m = copy.deepcopy(n)
          #handle case where nodes like Assert appear as ^Assert in one of node.input
          if m is not None and len(m) and m[0] == '^':
            old_to_new[m] = m
            continue
          split = m.split(':')
          if len(split) > 0 and split[-1].isdigit():
            split[0] = old_to_new[split[0]]
            new_inputs.append(':'.join(split))
          else:
            new_inputs.append(old_to_new[m])
        for i in range(len(node.input)):
          node.input.pop()
        node.input.extend(new_inputs)
      # Saving the new graph def
      with gfile.GFile(new_model_file, "wb") as f:
        f.write(model.SerializeToString())
      # Updating the input weight path
      args.input_weight = new_model_file
      # Updating the input and output node names
      for i in range(len(args.input_node_name)):
        if args.input_node_name[i] in old_to_new:
          args.input_node_name[i] = old_to_new[args.input_node_name[i]]
      for i in range(len(args.output_node_name)):
        if args.output_node_name[i] in old_to_new:
          args.output_node_name[i] = old_to_new[args.output_node_name[i]]
      with tf.io.gfile.GFile(args.input_weight, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
      with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
        input_shape = []
        if args.input_shape != None:
          for x in args.input_shape:
            shape = x.split(',')
            input_shape.append([int(x) for x in shape])
        for i, sl in enumerate(input_shape):
          input_shape[i] = [sl[0]] + sl[2:] + [sl[1]]
        data_shapes = []
        data_dict = generate_input_default(args)['default']
        data_list = []
        for i in range(len(args.input_node_name)):
          data_shapes.append((
              args.input_node_name[i],
              tuple(input_shape[i]),
          ))
        feed_dict = {}
        for i in args.input_node_name:
          feed_dict[graph.get_tensor_by_name(i + ':0')] = data_dict[i]
        output_tensors = []
        for i in args.output_node_name:
          output_tensors.append(graph.get_tensor_by_name(i + ':0'))
        with tf.compat.v1.Session() as sess:
          tf_output = sess.run(output_tensors, feed_dict=feed_dict)
        if _transform_model(args, data_dict, tf_output, graph_def) == -1:
          return -1
      return 0
    except Exception as e:
      logger.error("exception occurred when validating tensorflow network", exc_info=e)
      return -1
