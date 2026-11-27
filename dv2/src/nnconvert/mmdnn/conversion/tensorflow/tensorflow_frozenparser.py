'''
Portions of this file are derived from https://github.com/microsoft/MMdnn and are subject to the original MIT License

Copyright (c) Microsoft Corporation. All rights reserved.
SPDX-License-Identifier: MIT

Copyright (c) 2018 - 2025, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Modifications:

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

import numpy as np
import copy
import argparse
import tensorflow.compat.v1 as tensorflow
tensorflow.disable_v2_behavior()
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import ops
from tensorflow.core.framework import attr_value_pb2
from mmdnn.conversion.tensorflow.tensorflow_graph import TensorflowGraph
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.utils import *
from mmdnn.conversion.common.DataStructure.parser import Parser
from distutils.version import LooseVersion
import utils.tf_obj.tf_config_reader as config_reader
import tempfile
import os
import shutil
import json
import math
from dv_logging import get_logger
import sys
import pandas as pd
from mmdnn.conversion.common.types import UPSP_TYPE
logger = get_logger(__name__, 'DVPRSR')


def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('true', '1'):
    return True
  elif v.lower() in ('false', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def clast2first(shape):
  if len(shape) > 1:
    return [shape[0], shape[-1]] + shape[1:-1]
  else:
    return shape

def _cfirst_to_clast(data):
    if len(data) == 1 or len(data) == 2:
        return data
    elif len(data) == 3:
        return [int(data[0]),int(data[2]),int(data[1])]
    elif len(data) == 5: 
        return [int(data[0]),int(data[2]),int(data[3]), int(data[4]),int(data[1])]
    else:
        return [int(data[0]),int(data[2]),int(data[3]),int(data[1])]


def _set_output_shape_from_list(IR_node, shape_list):
    shape = graph_pb2.TensorShape()
    for _s in shape_list:
        new_dim = shape.dim.add()
        new_dim.size = _s
    IR_node.attr["_output_shapes"].list.shape.extend([shape])


class TensorflowParser2(Parser):
  # skip_prefix = [
  #     "^",
  #     "train_op",
  #     "save",
  #     "gradients",
  #     "init",
  #     "global_step",
  #     "distort_image",
  #     "Adagrad",
  # ]
  skip_prefix = []
  leakyrelu_maximum = []
  skip_node_list = []
  skip_scope = ["random_uniform", "Initializer", "optimizer", "weight_loss", "parallel_read", "case"]
  skip_type = set([
      "L2Loss",
      "VariableV2",
      "Const",
      "Assign",
      "RandomUniform",
      "FIFOQueueV2",
      "Assert",
      "Unpack",
      "NextIteration",
      "TensorArrayV3",
      "Range",
      "TensorArrayScatterV3",
      "TensorArrayReadV3",
      "TensorArrayWriteV3",
      # "Switch"
      "Dequantize",
      # "RequantizationRange",
      # "Requantize",
      #"ExpandDims",
      # "Identity",
      # "Mean",
      # "Cast",
      # "Slice",
      # "StridedSlice",
      "Pack",
      "CheckNumerics",
      "Where"
      #"Conv2DBackpropInput"
  ])
  q_type = set(["Dequantize", "QuantizeV2", "QuantizedConv2D", "QuantizedReshape", "RequantizationRange"])
  dtype_map = {
      0: graph_pb2.DT_UNDEFINED,
      1: graph_pb2.DT_FLOAT32,
      2: graph_pb2.DT_FLOAT64,
      3: graph_pb2.DT_INT32,
      4: graph_pb2.DT_UINT8,
      5: graph_pb2.DT_INT16,
      6: graph_pb2.DT_INT8,
      7: graph_pb2.DT_STRING,
      9: graph_pb2.DT_INT64,
      10: graph_pb2.DT_BOOL
  }

  @property
  def src_graph(self):
    return self.tf_graph

  def __init__(self, frozen_file, inputshape, in_nodes, dest_nodes, dynamic_input=False, config=None, debug=None, dummy=None, dest_path=None):
    if LooseVersion(tensorflow.__version__) < LooseVersion('1.8.0'):
      raise ImportError('Your TensorFlow version %s is outdated. ' 'MMdnn requires tensorflow>=1.8.0' % tensorflow.__version__)
    super(TensorflowParser2, self).__init__()
    self.weight_loaded = True
    self.config = config
    self.debug = debug
    self.dest_path = dest_path
    self.dummy = dummy
    self.dummy_layers_json = dict()
    self.summary_csv = os.path.join(self.dest_path, 'stats', 'summary.csv')
    if self.dummy:
      logger.info("Enabling dummy layer flag")
      self.use_dummy_layer = True
    else:
      self.use_dummy_layer = False
      # if not (os.path.exists(self.dummy_file) and os.stat(self.dummy_file).st_size > 0):
      #     self.dummy_layers_json = dict()
      # else:
      #     with open(self.dummy_file, 'r') as dummy_file1:
      #         self.dummy_layers_json = json.load(dummy_file1)
    # load model files into TensorFlow graph
    with open(frozen_file, 'rb') as f:
      serialized = f.read()
    ops.reset_default_graph()
    #original_gdef = tensorflow.GraphDef()   ###for debug added by kalyan
    original_gdef1 = tensorflow.GraphDef()
    #original_gdef.ParseFromString(serialized) ###for debug added by kalyan
    original_gdef1.ParseFromString(serialized)
    in_type_list = {}
    for n in original_gdef1.node:
      if n.name in in_nodes:
        in_type_list[n.name] = n.attr['dtype'].type
    from tensorflow.python.tools import strip_unused_lib
    from tensorflow.python.framework import dtypes
    from tensorflow.python.platform import gfile
    original_gdef = strip_unused_lib.strip_unused(input_graph_def=original_gdef1,
                                                  input_node_names=in_nodes,
                                                  output_node_names=dest_nodes,
                                                  placeholder_type_enum=dtypes.float32.as_datatype_enum)
    ####for debug added by kalyan
    original_gdef.library.CopyFrom(original_gdef1.library)
    original_gdef.versions.CopyFrom(original_gdef1.versions)
    # Save it to an output file
    tempdir = tempfile.mkdtemp()
    frozen_model_file = os.path.join(tempdir, 'frozen.pb')
    with gfile.GFile(frozen_model_file, "wb") as f:
      f.write(original_gdef.SerializeToString())
    with open(frozen_model_file, 'rb') as f:
      serialized = f.read()
    shutil.rmtree(tempdir)
    ops.reset_default_graph()
    model = tensorflow.GraphDef()
    model.ParseFromString(serialized)
    output_shape_map = dict()
    input_shape_map = dict()
    dtype = tensorflow.float32
    with tensorflow.Graph().as_default() as g:
      input_map = {}
      for i in range(len(inputshape)):
        if in_type_list[in_nodes[i]] == 1 or in_type_list[in_nodes[i]] == 0:
          dtype = tensorflow.float32
          x = tensorflow.placeholder(dtype, shape=inputshape[i], name=in_nodes[i])
        elif in_type_list[in_nodes[i]] == 3:
          dtype = tensorflow.int32
          x = tensorflow.placeholder(dtype, shape=inputshape[i], name=in_nodes[i])
        elif in_type_list[in_nodes[i]] == 10:
          dtype = tensorflow.bool
          x = tensorflow.placeholder(dtype, name=in_nodes[i])
        elif in_type_list[in_nodes[i]] == 4:
          dtype = tensorflow.uint8
          x = tensorflow.placeholder(dtype, shape=inputshape[i], name=in_nodes[i])
        input_map[in_nodes[i] + ':0'] = x
      tensorflow.import_graph_def(model, name='', input_map=input_map)
    with tensorflow.Session(graph=g) as sess:
      tempdir = tempfile.mkdtemp()
      meta_graph_def = tensorflow.train.export_meta_graph(filename=os.path.join(tempdir, 'my-model.meta'))
      model = meta_graph_def.graph_def
      shutil.rmtree((tempdir))
    self.tf_graph = TensorflowGraph(model)
    self.tf_graph.build()

  @staticmethod
  def _get_scopes(layer_name):
    return layer_name.split('/')

  def check_const(self, node):
    while node:
      if node.type == "Const":
        return node
      elif node.type == "NoOp":
        return None
      else:
        node = self.get_parent(node.name, [0])

  def _convert_reduction_operators(self, source_node, new_op=None):
    IR_node = self._convert_identity_operation(source_node, 1, new_op)
    # keep dims
    IR_node.attr['keepdims'].b = source_node.layer.attr['keep_dims'].b
    # axes
    axes = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor
    axes = tensor_util.MakeNdarray(axes)
    IR_node.attr['axes'].list.i.extend(axes)

  def _convert_layers_batchnorm(self, source_node):
    IR_node = self.IR_graph.node.add()
    TensorflowParser2._copy_and_reop(source_node, IR_node, 'BatchNorm')
    is_transformed = False
    test = self.get_parent(source_node.name, [0])
    if test.type == 'Mul':
      is_transformed = True
    # ssd model is transformed
    if is_transformed:
      # Ax - (Au - b)
      # A
      input_mul_A = self.get_parent(source_node.name, [0, 1])
      tensor_content = input_mul_A.get_attr('value')
      A_content = tensor_util.MakeNdarray(tensor_content)
      self.set_weight(source_node.name, 'A', A_content)
      # b
      input_sub = self.get_parent(source_node.name, [1])
      tensor_content = input_sub.get_attr('value')
      sub_content = tensor_util.MakeNdarray(tensor_content)
      # logger.info(sub_content)
      self.set_weight(source_node.name, 'b', sub_content)
      input_node = self.get_parent(source_node.name, [0])
      IR_node.input.append(input_node.real_name)
      IR_node.attr["_output_shapes"].list.shape.pop()
      IR_node.attr["_output_shapes"].MergeFromString(input_node.layer.attr['_output_shapes'].SerializeToString())
    else:
      # epsilon
      epsilon = self.get_parent(source_node.name, [1])
      IR_node.attr['epsilon'].f = epsilon.layer.attr['value'].tensor.float_val[0]
      # moving variance (var) /read
      moving_variance = self.get_parent(source_node.name, [0])
      if moving_variance.type == 'Identity':
        moving_variance_read = self.src_graph.get_parent(moving_variance.name, [0])
        tensor_content = moving_variance_read.get_attr('value')
        moving_variance_content = tensor_util.MakeNdarray(tensor_content)
        self.set_weight(source_node.name, 'var', moving_variance_content)
      else:
        logger.info("moving variance: %s", moving_variance.layer)
        assert False
      # gamma (scale)
      Rsqrt = self.get_son(source_node.name, [0], True)
      # logger.info(Rsqrt.out_edges)
      if len(Rsqrt.out_edges) == 2:
        IR_node.attr['scale'].b = False
        output_node = self.get_son(Rsqrt.name, [0, 0], True)
        if output_node.type == 'Sub':
          output_node = self.get_son(Rsqrt.name, [1, 0], True)
          Mul = self.get_son(Rsqrt.name, [0], True)
        else:
          Mul = self.get_son(Rsqrt.name, [1], True)
      else:
        IR_node.attr['scale'].b = True
        son = self.get_son(Rsqrt.name, [0, 0], True)
        gamma_from = self.get_parent(son.name, [1, 1], True)
        gamma = self.check_const(gamma_from)
        gamma_tensor = gamma.get_attr('value')
        scale = tensor_util.MakeNdarray(gamma_tensor)
        self.set_weight(source_node.name, 'scale', scale)
        output_node = self.get_son(source_node.name, [0, 0, 0, 0], True)
        if output_node.type == 'Sub':
          output_node = self.get_son(source_node.name, [0, 0, 0, 0, 0], True)
          Mul = self.get_son(Rsqrt.name, [0, 0], True)
        else:
          Mul = self.get_son(Rsqrt.name, [0, 1], True)
      # beta  (bias)
      beta = self.get_parent(output_node.name, [1, 0, 0], True).get_attr('value')
      bias = tensor_util.MakeNdarray(beta)
      IR_node.attr['bias'].b = True
      self.set_weight(source_node.name, 'bias', bias)
      # moving mean (mean)
      moving_mean = self.get_parent(Mul.name, [0, 0]).get_attr('value')
      mean = tensor_util.MakeNdarray(moving_mean)
      self.set_weight(source_node.name, 'mean', mean)
      # input node
      assert output_node.type == 'Add' or output_node.type == 'AddV2'
      input_node = self.get_parent(output_node.name, [0, 0])
      IR_node.input.append(input_node.real_name)
      IR_node.attr["_output_shapes"].list.shape.pop()
      IR_node.attr["_output_shapes"].MergeFromString(input_node.layer.attr['_output_shapes'].SerializeToString())
      output_node.real_name = source_node.name

  def _convert_layers_instancenorm(self, source_node):
    IR_node = self.IR_graph.node.add()
    TensorflowParser2._copy_and_reop(source_node, IR_node, 'InstanceNorm')
    # epsilon
    epsilon = self.get_parent(source_node.name, [1])
    epsilon_value = epsilon.get_attr('value').float_val[0]
    IR_node.attr['epsilon'].f = epsilon_value
    # beta
    output_node = self.get_son(source_node.name, [0, 0, 0, 0], True)
    beta = self.get_parent(output_node.name, [1, 0, 0, 0, 0, 1], True)
    beta_tensor = beta.get_attr('value')
    beta = tensor_util.MakeNdarray(beta_tensor)
    self.set_weight(source_node.name, 'bias', beta)
    # gamma (scale)
    IR_node.attr['scale'].b = True
    son = self.get_son(source_node.name, [0, 0, 0], True)
    gamma = self.get_parent(son.name, [1, 1, 0, 0, 0, 1], True)
    gamma_tensor = gamma.get_attr('value')
    scale = tensor_util.MakeNdarray(gamma_tensor)
    self.set_weight(source_node.name, 'scale', scale)
    # output_node = self.get_son(source_node.name, [0, 0, 0, 0], True)
    assert output_node.type == 'Add' or output_node.type == 'AddV2'
    input_node = self.get_parent(output_node.name, [0, 0])
    IR_node.input.append(input_node.real_name)
    output_node.real_name = source_node.name
    # assert False
  @classmethod
  def _skip_node(cls, source_node):
    if source_node.covered:
      return True
    for prefix in cls.skip_prefix:
      if source_node.name.startswith(prefix):
        return True
    scopes = TensorflowParser2._get_scopes(source_node.name)
    for s in scopes:
      if s in cls.skip_scope:
        return True
    return False

  def _add_constant_node(self, source_node):
    parent_ids = range(len(source_node.in_edges))
    for idx in parent_ids:
      s = source_node.in_edges[idx]
      parent_node = self.tf_graph.get_node(s)
      if parent_node.type == 'Const':
        self._rename_Const(parent_node)

  def __stringify_attrs(self, source_node_attrs):
    attributes = []
    for attr in source_node_attrs:
      if isinstance(source_node_attrs[attr], list):
        attr_value = "".join(map(str, source_node_attrs[attr]))
      elif isinstance(source_node_attrs[attr], bytes):
        attr_value = source_node_attrs[attr].decode("utf-8")
      else:
        attr_value = str(source_node_attrs[attr])
      attributes.append(f"{attr}:{attr_value}")
    op_params = ', '.join(attributes)
    return op_params.replace('\n', '')

  def _rename_Const(self, source_node):
    IR_node = self._convert_identity_operation(source_node, end_idx=0, new_op='Constant')  # Constant
    value = source_node.get_attr('value')
    if value.float_val:
      value = value.float_val[0]
    elif value.int_val:
      value = value.int_val[0]
    else:
      value = tensor_util.MakeNdarray(value).tolist()
    kwargs = {'value': value}
    assign_IRnode_values(IR_node, kwargs)

  def gen_IR(self):
    self.dummy_layers = set()
    df = pd.DataFrame(columns=['op', 'issue', 'source'])
    if self.debug:
      with open(self.debug, 'r') as fd:
        _gen_dummy = json.load(fd)
      for dummy_layer in _gen_dummy:
        logger.info("cp layer is found from estimator json %s", dummy_layer)
        failed_layer_list = _gen_dummy[dummy_layer]
        for layer in failed_layer_list[:-1]:
          logger.info("adding dummy layer ot the list %s", layer)
          self.dummy_layers.add(layer)
    ignore_list = []
    for layer in self.src_graph.topological_sort:
      current_node = self.src_graph.get_node(layer)
      node_type = current_node.type
      if node_type == "Shape":
        for outnode in current_node.out_nodes:
          if outnode.type == 'StridedSlice':
            for grandoutnode in outnode.out_nodes:
              if grandoutnode.type == 'Pack' and len(grandoutnode.out_nodes) == 1 and grandoutnode.out_nodes[0].type == 'Reshape':
                ignore_list.extend([current_node.name, outnode.name, grandoutnode.name])
              if grandoutnode.type == 'Mul' and len(grandoutnode.out_nodes) == 1 and grandoutnode.out_nodes[0].type == 'ResizeNearestNeighbor':
                ignore_list.extend([current_node.name, outnode.name, grandoutnode.name])
              if grandoutnode.type == 'ResizeBilinear':
                ignore_list.extend([current_node.name, outnode.name])
      if current_node.name in ignore_list:
        continue
      if self._skip_node(current_node):
        continue
      if hasattr(self, "rename_" + node_type):
        func = getattr(self, "rename_" + node_type)
        try:
          old_nodes = len(self.IR_graph.node)
          func(current_node)
        except Exception as e:
          #remove new nodes if there is an exception in mapping a node
          removable_nodes = [i for i in range(old_nodes, len(self.IR_graph.node))]
          for i in removable_nodes:
            del self.IR_graph.node[i]
          if node_type not in ['Placeholder', 'NoOp']:
            self.rename_DUMMY(current_node)
          logger.exception("error while renaming node. node name:  %s node type: %s error msg: %s node info: %s", current_node.name, node_type, e, current_node.layer)
          row = {}
          row['op'] = current_node.name
          row['issue'] = str(e)
          row['source'] = current_node.layer
          df = pd.concat([df,pd.DataFrame(row,index=[0])], ignore_index=True)
      else:
        try:
          if node_type not in ['Const']:
            self.rename_DUMMY(current_node)
          else:
            continue
        except Exception as e:
          logger.exception("Exception occurred while generation IR for op:{%s}", node_type)
          logger.exception("Exception details: {%s}", str(e), exc_info=sys.exc_info())
          row = {}
          row['op'] = current_node.name
          row['issue'] = str(e)
          row['source'] = current_node.layer
          df = pd.concat([df,pd.DataFrame(row,index=[0])], ignore_index=True)
    df.to_csv(self.summary_csv, index=False)
    if self.use_dummy_layer:
      logger.info("Total dummy layers added to the network are %s", len(self.dummy_layers_json))
      for layer_name in self.dummy_layers_json.keys():
        logger.info("dummy layer added in place of %s", layer_name)
      dummy_json = os.path.join(self.dest_path, 'dummy_layers.json')
      with open(dummy_json, 'w') as dummy_file1:
        json.dump(self.dummy_layers_json, dummy_file1)
    if self.config:
      self.rename_DetectionOut()

  @staticmethod
  def tensor_shape_to_list(shapes):
    if isinstance(shapes, attr_value_pb2.AttrValue):
      return [dim.size for dim in shapes.shape.dim]
    else:
      ret = []
      for shape in shapes:
        this_one = [dim.size for dim in shape.dim]
        ret.append(this_one)
      return ret

  @staticmethod
  def _copy_and_reop(source_node, IR_node, new_op=None):
    if new_op == None: new_op = source_node.type
    # if source_node.name == "input_1":
    #    IR_node.name = "input"
    # else:
    IR_node.name = source_node.name
    IR_node.op = new_op
    kwargs = {}
    if 'data_format' in source_node.layer.attr:
      kwargs['data_format'] = source_node.get_attr('data_format')
    if 'T' in source_node.layer.attr:
      if source_node.type not in TensorflowParser2.q_type:
        if source_node.type == 'Enter':
          IR_node.attr["dtype"].type = TensorflowParser2.dtype_map[6]
        else:
          assert source_node.layer.attr['T'].type in TensorflowParser2.dtype_map, 'type [{}] is unknown.'.format(source_node.layer.attr['dtype'].type)
          IR_node.attr["dtype"].type = TensorflowParser2.dtype_map[source_node.layer.attr['T'].type]
      else:
        # Quantized model type
        IR_node.attr["dtype"].type = TensorflowParser2.dtype_map[6]
    if '_output_shapes' in source_node.layer.attr:
      IR_node.attr["_output_shapes"].MergeFromString(source_node.layer.attr['_output_shapes'].SerializeToString())
    if 'paddings' in source_node.layer.attr:
      IR_node.attr["paddings"].MergeFromString(source_node.layer.attr['paddings'].SerializeToString())
    assign_IRnode_values(IR_node, kwargs)

  def _convert_inedge(self, source_node, IR_node, start_idx=0, end_idx=None):
    if end_idx == None: end_idx = len(source_node.in_edges)
    for idx in range(start_idx, end_idx):
      if self.src_graph.get_node(source_node.in_edges[idx]).type in ['Split']:
        IR_node.input.append(self.src_graph.get_node(source_node.in_edges[idx]).real_name + '_output' + str(source_node.in_edges[idx]).split(':')[1])
      elif self.src_graph.get_node(source_node.in_edges[idx]).type in ['SplitV']:
        #IR_node.input.append(str(source_node.in_edges[idx]).replace(':', '_'))
        in_name = self.src_graph.get_node(source_node.in_edges[idx]).real_name + '_out'
        if len(str(source_node.in_edges[idx]).split(':')) == 1:
          in_name = in_name + '0'
        else:
          in_name = in_name + str(source_node.in_edges[idx]).split(':')[1]
        IR_node.input.append(in_name)
      else:
        IR_node.input.append(self.src_graph.get_node(source_node.in_edges[idx]).real_name)

  @staticmethod
  def _copy_shape(source_node, IR_node):
    assert 'shape' in source_node.layer.attr
    if source_node.layer.attr['shape'].list.shape:
      IR_node.attr['shape'].shape.MergeFromString(source_node.layer.attr['shape'].list.shape[0].SerializeToString())
    else:
      IR_node.attr['shape'].shape.MergeFromString(source_node.layer.attr['shape'].shape.SerializeToString())

  def rename_UNKNOWN(self, source_node):
    if source_node.type in self.skip_type:
      return
    if self.use_dummy_layer:
      self.rename_DUMMY(source_node)
      return
    else:
      self.rename_DUMMY(source_node)
      raise NotImplementedError("Tensorflow has unsupported operator {} with name {}.".format(source_node.type, source_node.name))

  def rename_NoOp(self, source_node):
    return

  def _convert_padding(self, source_node, IR_node, kernel_size):
    # TODO: Fused conv and pool with padding is different from defused operators
    input_node = self.get_parent(source_node.name, [0])
    input_shape = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    if source_node.get_attr('padding') == 'VALID':
      dims = len(input_shape)
      assign_IRnode_values(IR_node, {'auto_pad': "VALID", 'pads': [0, 0] * dims})
    elif source_node.get_attr('padding') == 'SAME':
      padding = compute_tf_same_padding(input_shape, kernel_size, source_node.get_attr('strides'))
      logger.debug("padding for layer is %s", str(padding))
      assign_IRnode_values(IR_node, {'auto_pad': "SAME_LOWER", 'pads': padding})
    elif source_node.get_attr('padding') == 'EXPLICIT' and source_node.get_attr('explicit_paddings') is not None:
      pads = source_node.get_attr('explicit_paddings')
      padding = compute_tf_explicit_padding(pads, source_node.get_attr('data_format'))
      assign_IRnode_values(IR_node, {'auto_pad': "EXPLICIT", 'pads': padding})
    else:
      assert False

  def _convert_padding_deconv(self, source_node, IR_node, kernel_size):
    # TODO: Fused conv and pool with padding is different from defused operators
    for edg in range(len(source_node.in_edges)):
      input_node = self.src_graph.get_parent(source_node.name, [edg])
      if 'read' not in input_node.name and 'stack' not in input_node.name and input_node.type != 'Const':
        break
    # input_node = self.get_parent(source_node.name, [1])
    input_shape = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    output_shape = self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]
    if source_node.get_attr('padding') == 'VALID':
      dims = len(input_shape)
      assign_IRnode_values(IR_node, {'auto_pad': "VALID", 'pads': [0, 0] * dims})
    elif source_node.get_attr('padding') == 'SAME':
      padding = compute_tf_deconv_same_padding(input_shape, output_shape, kernel_size, source_node.get_attr('strides'))
      assign_IRnode_values(IR_node, {'auto_pad': "SAME_LOWER", 'pads': padding})
    else:
      assert False

  def _get_bias(self, source_node, IR_node):
    if not source_node.out_edges:
      return
    add_node = self.tf_graph.get_node(source_node.out_edges[0])
    if add_node.type != "Add" and add_node.type != "BiasAdd" and add_node.type != "AddV2":
      return
    variable = self.check_const(self.tf_graph.get_node(add_node.in_edges[1]))  #add_bias node
    if not variable:
      return
    if variable.type != 'Const':
      return
    bias_value = variable.get_attr('value')
    bias = tensor_util.MakeNdarray(bias_value)
    # assert variable.get_attr('_output_shapes')[0].dim[0].size == IR_node.attr['kernel_shape'].list.i[-1]
    add_node.real_name = IR_node.name
    add_node.covered = True
    IR_node.attr['use_bias'].b = True
    current_layer = self.weights[source_node.name]
    current_layer['bias'] = bias

  def _convert_pooling(self, source_node, pool_type):
    IR_node = self._convert_identity_operation(source_node, new_op='Pool')
    kwargs = {}
    # strides
    kwargs['strides'] = source_node.get_attr('strides')
    # window_shape
    kwargs['kernel_shape'] = source_node.get_attr('ksize')
    # pool type
    kwargs['pooling_type'] = pool_type
    # padding
    self._convert_padding(source_node, IR_node, kwargs['kernel_shape'][1:-1])
    assign_IRnode_values(IR_node, kwargs)

  def _convert_identity_operation(self, source_node, start_idx=0, end_idx=None, new_op=None):
    IR_node = self.IR_graph.node.add()
    TensorflowParser2._copy_and_reop(source_node, IR_node, new_op)
    self._convert_inedge(source_node, IR_node, start_idx, end_idx)
    return IR_node

  def rename_DetectionOut(self):
    code_type = {"tf_center_size": 4, "tf_corner": 1}
    configs = config_reader.get_configs_from_pipeline_file(self.config)
    config_dict = config_reader.create_configs_from_pipeline_proto(configs)
    #Not a good check . TODO
    if config_dict['model'].ssd.box_coder.faster_rcnn_box_coder.ByteSize() == 0:
      logger.error("code type not as expected")
      raise NotImplementedError()
    IR_node = self.IR_graph.node.add()
    IR_node.name = 'detection_out'
    IR_node.op = 'DetectionOutput'
    IR_node.input.extend([self.src_graph.get_parent("raw_outputs/box_encodings", [0]).name, self.src_graph.get_parent("raw_outputs/class_predictions", [0]).name])
    IR_node.attr['classes'].i = config_dict['model'].ssd.num_classes + 1
    #TODO
    IR_node.attr['share_loc'].b = True
    #TODO
    IR_node.attr['var_encoded'].b = True
    #TODO
    IR_node.attr['code_type'].i = code_type['tf_center_size']
    #TODO
    IR_node.attr['bg_lid'].i = 0
    IR_node.attr['nms_threshold'].f = config_dict['model'].ssd.post_processing.batch_non_max_suppression.iou_threshold
    IR_node.attr['conf_threshold'].f = config_dict['model'].ssd.post_processing.batch_non_max_suppression.score_threshold
    IR_node.attr['top_k'].i = config_dict['model'].ssd.post_processing.batch_non_max_suppression.max_detections_per_class
    IR_node.attr['keep_top_k'].i = config_dict['model'].ssd.post_processing.batch_non_max_suppression.max_total_detections
    anchors = tensor_util.MakeNdarray(self.src_graph.model.node[-1].attr['value'].tensor)
    self.set_weight(IR_node.name, 'weights', anchors)

  def rename_PRelu(self, source_node):
    self._convert_identity_operation(source_node, new_op='PRelu')

  def rename_dvnet_Relu6(self, source_node):
    self._convert_identity_operation(source_node, new_op='Relu6')

  def rename_Relu6(self, source_node):
    if not source_node.variable_name in self.dummy_layers:
      self.rename_dvnet_Relu6(source_node)
      return
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Relu6')

  def rename_Softsign(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_SoftSign')

  def rename_Softplus(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_SoftPlus')

  def rename_Merge(self, source_node):
    # In facenet or other newtwork using slim.batch_norm,
    # There are two BN(train, test) skip switch and merge.
    source_node.real_name = self.src_graph.get_node(source_node.in_edges[0]).real_name

  def rename_dvnet_DepthwiseConv2dNative(self, source_node):
    IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op='DepthwiseConv')
    kwargs = {}
    kwargs['strides'] = source_node.get_attr('strides')
    kwargs['dilations'] = source_node.get_attr('dilations') or [1] * len(kwargs['strides'])
    input_node = self.src_graph.get_parent(source_node.name, [1])
    kwargs['kernel_shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    kwargs['padding'] = source_node.get_attr('padding')
    spatial_size = len(kwargs['kernel_shape'][:-2])
    kernel_size_dil = [1] * spatial_size
    for i in range(spatial_size):
      kernel_size_dil[i] = kwargs['kernel_shape'][i] + (kwargs['kernel_shape'][i] - 1) * (kwargs['dilations'][1 + i] - 1)
    self._convert_padding(source_node, IR_node, kernel_size_dil)
    weight_node = self.src_graph.get_parent(source_node.name, [1])
    weight = self.check_const(weight_node).get_attr('value')
    weight_content = tensor_util.MakeNdarray(weight)
    self.set_weight(source_node.name, 'weights', weight_content)
    assign_IRnode_values(IR_node, kwargs)
    self._get_bias(source_node, IR_node)

  def rename_DepthwiseConv2dNative(self, source_node):
    real_name = source_node.real_name.replace('/', '_').replace('-', '_')
    if not real_name in self.dummy_layers:
      self.rename_dvnet_DepthwiseConv2dNative(source_node)
      return
    IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op='Tf_DepthwiseConv2dNative')
    kwargs = {}
    kwargs['strides'] = source_node.get_attr('strides')
    kwargs['dilations'] = source_node.get_attr('dilations') or [1] * len(kwargs['strides'])
    input_node = self.src_graph.get_parent(source_node.name, [1])
    kwargs['kernel_shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    kwargs['padding'] = source_node.get_attr('padding')
    weight_node = self.src_graph.get_parent(source_node.name, [1])
    weight = self.check_const(weight_node).get_attr('value')
    weight_content = tensor_util.MakeNdarray(weight)
    self.set_weight(source_node.name, 'weights', weight_content)
    assign_IRnode_values(IR_node, kwargs)
    self._get_bias(source_node, IR_node)

  def rename_Conv2DBackpropInput(self, source_node):
    IR_node = self._convert_identity_operation(source_node)
    kwargs = {}
    kwargs['strides'] = source_node.get_attr('strides')
    kwargs['padding'] = source_node.get_attr('padding')
    kwargs['dilations'] = source_node.get_attr('dilations') or [1] * len(kwargs['strides'])
    kwargs['group'] = source_node.get_attr('group', 1)
    for edg in range(len(source_node.in_edges)):
      input_node = self.src_graph.get_parent(source_node.name, [edg])
      if 'read' not in input_node.name and 'stack' not in input_node.name and input_node.type != 'Const':
        break
    logger.info("input node for deconv node is %s", input_node.name)
    kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    weight_node = self.src_graph.get_parent(source_node.name, [1])
    weight = self.check_const(weight_node).get_attr('value')
    weight_content = tensor_util.MakeNdarray(weight)
    self.set_weight(source_node.name, 'weights', weight_content)
    kwargs['kernel_shape'] = self.tensor_shape_to_list(weight_node.get_attr('_output_shapes'))[0]
    kwargs['num_outputs'] = kwargs['kernel_shape'][-2]
    self._convert_padding_deconv(source_node, IR_node, kwargs['kernel_shape'][:-2])
    assign_IRnode_values(IR_node, kwargs)
    self._get_bias(source_node, IR_node)

  def rename_BatchNormWithGlobalNormalization(self, source_node):
    if source_node.real_variable_name not in self.dummy_layers:
      self.rename_dvnet_BatchNormWithGlobalNormalization(source_node)
      return
    IR_node = self._convert_identity_operation(source_node, new_op='Mxnet_Batchnorm')
    IR_node.attr['axis'].i = 1
    IR_node.attr['fix_gamma'].b = False
    IR_node.attr['output_mean_var'].b = False
    attr = source_node.attrs
    IR_node.attr['eps'].f = attr['variance_epsilon']
    IR_node.attr['momentum'].f = 1
    IR_node.attr['use_global_stats'].b = False
    IR_node.attr['bias'].b = True
    IR_node.attr['scale'].b = True
    gamma = self.get_parent(source_node.name, [4])
    gamma_value = gamma.get_attr('value')
    gamma = tensor_util.MakeNdarray(gamma_value)
    beta = self.get_parent(source_node.name, [3])
    beta_value = beta.get_attr('value')
    beta = tensor_util.MakeNdarray(beta_value)
    mean = self.get_parent(source_node.name, [1])
    mean_value = mean.get_attr('value')
    mean = tensor_util.MakeNdarray(mean_value)
    moving_variance = self.get_parent(source_node.name, [2])
    tensor_variance = moving_variance.get_attr('value')
    moving_variance_content = tensor_util.MakeNdarray(tensor_variance)
    self.set_weight(source_node.name, "scale", gamma)
    self.set_weight(source_node.name, "bias", beta)
    # mean
    self.set_weight(source_node.name, "mean", mean)
    # var
    self.set_weight(source_node.name, "var", moving_variance_content)

  def rename_dvnet_BatchNormWithGlobalNormalization(self, source_node):
    IR_node = self._convert_identity_operation(source_node, start_idx=0, end_idx=1, new_op='BatchNorm')
    # epsilon
    IR_node.attr['epsilon'].f = source_node.get_attr('variance_epsilon')
    # moving variance (var) /read
    moving_variance = self.get_parent(source_node.name, [2])
    tensor_variance = moving_variance.get_attr('value')
    moving_variance_content = tensor_util.MakeNdarray(tensor_variance)
    self.set_weight(source_node.name, 'var', moving_variance_content)
    # gamma (scale)
    gamma = self.get_parent(source_node.name, [4])
    gamma_value = gamma.get_attr('value')
    gamma = tensor_util.MakeNdarray(gamma_value)
    self.set_weight(source_node.name, 'scale', gamma)
    IR_node.attr['scale'].b = True
    # beta  (bias)
    beta = self.get_parent(source_node.name, [3])
    beta_value = beta.get_attr('value')
    beta = tensor_util.MakeNdarray(beta_value)
    self.set_weight(source_node.name, 'bias', beta)
    IR_node.attr['use_bias'].b = True
    # moving mean (mean)
    mean = self.get_parent(source_node.name, [1])
    mean_value = mean.get_attr('value')
    mean = tensor_util.MakeNdarray(mean_value)
    self.set_weight(source_node.name, 'mean', mean)

  def rename_Placeholder(self, source_node):
    if source_node.layer.attr.get("_output_shapes", False):
      if source_node.layer.attr.get("_output_shapes", False).shape.unknown_rank:
        return
    if source_node.layer.attr.get("shape", False):
      if source_node.layer.attr.get("shape", False).shape.unknown_rank:
        return
    IR_node = self._convert_identity_operation(source_node, new_op='DataInput')
    TensorflowParser2._copy_shape(source_node, IR_node)
    # IR_node.attr['shape'].shape.dim[0].size = 1 #-1
    # IR_node.attr['_output_shapes'].list.shape[0].dim[0].size = 1 #-1
  def rename_dvnet_Mean(self, source_node):
    input_node = self.src_graph.get_parent(source_node.name, [0])
    input_node_shape = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    # axes
    axes = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor
    axes = tensor_util.MakeNdarray(axes)
    if np.all(axes == list(range(1, len(input_node_shape) - 1))):
      IR_node = self._convert_identity_operation(source_node, new_op='PoolReduce')
      kwargs = {}
      # strides
      kwargs['strides'] = [1] * len(input_node_shape)
      # window_shape
      kwargs['kernel_shape'] = [1] * len(input_node_shape)
      # pool type
      kwargs['pooling_type'] = 'AVG'
      kwargs['auto_pad'] = "VALID"
      kwargs['pads'] = [0, 0] * len(input_node_shape)
      IR_node.attr['keepdims'].b = source_node.layer.attr['keep_dims'].b
      assign_IRnode_values(IR_node, kwargs)
    else:
      # ReduceMean
      IR_node = self._convert_identity_operation(source_node, start_idx=0, end_idx=1, new_op='Tf_Mean')
      # keep dims
      IR_node.attr['keep_dims'].b = source_node.layer.attr['keep_dims'].b
      # axes
      axis = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor
      axis = tensor_util.MakeNdarray(axis)
      IR_node.attr['axis'].list.i.extend(axis)

  def rename_Mean(self, source_node):
    real_name = source_node.real_name.replace('/', '_')
    if not real_name in self.dummy_layers:
      self.rename_dvnet_Mean(source_node)
      return
    # ReduceMean
    IR_node = self._convert_identity_operation(source_node, start_idx=0, end_idx=1, new_op='Tf_Mean')
    # keep dims
    IR_node.attr['keep_dims'].b = source_node.layer.attr['keep_dims'].b
    # axes
    axis = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor
    axis = tensor_util.MakeNdarray(axis)
    IR_node.attr['axis'].list.i.extend(axis)

  def rename_CropAndResize(self, source_node):
    shape = self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]
    kwargs = {}
    kwargs['shape'] = shape
    IR_node = self._convert_identity_operation(source_node, new_op="Tf_Cropandresize")
    input_node_weight = self.src_graph.get_parent(source_node.name, [1])
    tensor_content = self.check_const(input_node_weight).get_attr('value')
    W = tensor_util.MakeNdarray(tensor_content)
    self.set_weight(source_node.name, 'weights', W)
    kwargs['box_ind'] = self.src_graph.get_parent(source_node.name, [2]).get_attr("value").int_val[0]
    kwargs['method'] = source_node.get_attr('method')
    IR_node.attr['e_val'].f = source_node.get_attr('extrapolation_value')
    assign_IRnode_values(IR_node, kwargs)

  def rename_Reshape(self, source_node):
    IR_node = self._convert_identity_operation(source_node, end_idx=1)
    #logger.info("Number of in nodes = " + str(len(IR_node.in_edges())))
    # parent_node = self.src_graph.get_parent(source_node.name, [1])
    shape = self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]
    kwargs = {'shape': shape}
    if not source_node.get_attr('data_format'):
      input_node = self.src_graph.get_parent(source_node.name, [0])
      shape = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
      if len(shape) == 4:
        kwargs['data_format'] = "NHWC"
      elif len(shape) == 3:
        kwargs['data_format'] = "NWC"
      elif len(shape) == 2:
        kwargs['data_format'] = "NC"
    assign_IRnode_values(IR_node, kwargs)

  def rename_ExpandDims(self, source_node):
    IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op="Reshape")
    #logger.info("Number of in nodes = " + str(len(IR_node.in_edges())))
    # parent_node = self.src_graph.get_parent(source_node.name, [1])
    shape = self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]
    kwargs = {'shape': shape}
    if not source_node.get_attr('data_format'):
      input_node = self.src_graph.get_parent(source_node.name, [0])
      shape = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
      if len(shape) == 4:
        kwargs['data_format'] = "NHWC"
      elif len(shape) == 3:
        kwargs['data_format'] = "NWC"
      elif len(shape) == 2:
        kwargs['data_format'] = "NC"
    assign_IRnode_values(IR_node, kwargs)

  def rename_MirrorPad(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='MirrorPad')
    input_node = self.src_graph.get_parent(source_node.name, [1])
    tensor_content = tensor_util.MakeNdarray(input_node.get_attr('value')).reshape(-1)
    kwargs = {}
    kwargs['mode'] = source_node.get_attr('mode')
    kwargs['pads'] = tensor_content.tolist()
    assign_IRnode_values(IR_node, kwargs)

  def rename_Min(self, source_node):
    IR_node = self._convert_identity_operation(source_node, start_idx=0, end_idx=1, new_op='Min')
    kwargs = {}
    input_node = self.src_graph.get_parent(source_node.name, [0])
    kwargs['shape_0'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    input_node = self.src_graph.get_parent(source_node.name, [1])
    kwargs['shape_1'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    assign_IRnode_values(IR_node, kwargs)

  def rename_Max(self, source_node):
    IR_node = self._convert_identity_operation(source_node, start_idx=0, end_idx=1, new_op='Max')
    kwargs = {}
    input_node = self.src_graph.get_parent(source_node.name, [0])
    kwargs['shape_0'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    input_node = self.src_graph.get_parent(source_node.name, [1])
    kwargs['shape_1'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    assign_IRnode_values(IR_node, kwargs)

  def rename_Mul(self, source_node):
    real_name = source_node.real_name.replace('/', '_')
    if not real_name in self.dummy_layers:
      self.rename_dvnet_Mul(source_node)
      return
    const_idx = -1
    if source_node.in_nodes[0].type in ['Const', 'Identity']:
      const_idx = 0
    elif source_node.in_nodes[1].type in ['Const', 'Identity']:
      const_idx = 1
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Mul')
    if const_idx != -1 and (source_node.in_nodes[const_idx].type == 'Identity'):
      scale = tensor_util.MakeNdarray(source_node.in_nodes[const_idx].in_nodes[0].get_attr('value'))
      self.set_weight(source_node.name, 'weights', scale)
    elif const_idx != -1 and (source_node.in_nodes[const_idx].type == 'Const'):
      scale = tensor_util.MakeNdarray(source_node.in_nodes[const_idx].get_attr('value'))
      self.set_weight(source_node.name, 'weights', scale)

  def rename_dvnet_Mul(self, source_node):
    scopes = self._get_scopes(source_node.name)

    def is_candidate_for_scale():
      """ in some cases the constant input
            to the mul node can pass through an identity
            node, handle case accordingly
            """
      if len(source_node.in_nodes) == 2:
        input1 = source_node.in_nodes[0].type
        input2 = source_node.in_nodes[1].type
        if (input1 == 'Const' and input2 != 'Const') or ((input2 == 'Const' and input1 != 'Const')):
          return True
        if input1 == 'Identity':
          parent = source_node.in_nodes[0].in_nodes[0].type
          if parent == 'Const':
            return True
          if input2 == 'Const':
            return True
        if input2 == 'Identity':
          parent = source_node.in_nodes[1].in_nodes[0].type
          if parent == 'Const':
            return True
          if input1 == 'Const':
            return True
      return False

    #leakyrelu combo to be ignored here
    if len(source_node.in_nodes) == 2 and source_node.in_nodes[0].type == "Const":
      if len(source_node.out_nodes) == 1 and source_node.out_nodes[0].type == 'Maximum':
        if len(source_node.in_nodes[0].get_attr('value').float_val) == 1:
          self.leakyrelu_maximum.append(source_node.out_nodes[0].name)
          return
    # ignore the mul operator if the other input is scalar of value 1.0
    scale_value = source_node.in_nodes[1].get_attr('value')
    if scale_value:
      # if the other input is a layer, then scale_value will be None
      scale = tensor_util.MakeNdarray(source_node.in_nodes[1].get_attr('value'))
      if (isinstance(scale, int) and scale == 1) or (isinstance(scale, np.ndarray) and scale.size == 1 and scale.tolist() == 1):
        source_node.real_name = self.src_graph.get_node(source_node.in_edges[0]).real_name
        return
    is_bn_mul = False
    if is_candidate_for_scale():
      const_index = 1
      parent_index = 0
      if source_node.in_nodes[0].type in ["Const", "Identity"]:
        const_index = 0
        parent_index = 1
      logger.info("found mul node with one input const")
      current_node_shape = self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]
      parent_node_shape = self.tensor_shape_to_list(source_node.in_nodes[parent_index].get_attr('_output_shapes'))[0]
      const_node_shape = self.tensor_shape_to_list(source_node.in_nodes[const_index].get_attr('_output_shapes'))[0]
      if (source_node.in_nodes[const_index].type == 'Identity'):
        value = source_node.in_nodes[const_index].in_nodes[0].get_attr('value')
      else:
        value = source_node.in_nodes[const_index].get_attr('value')
      if value.float_val:
        value = value.float_val[0]
      elif value.int_val:
        value = value.int_val[0]
      else:
        value = tensor_util.MakeNdarray(value).tolist()
      if (parent_node_shape == current_node_shape and len(const_node_shape) > 0 and current_node_shape[-1] == const_node_shape[-1]):
        logger.info("found scale candidate")
        is_bn_mul = True
        IR_node = self._convert_identity_operation(source_node, new_op='BatchNorm')
        const_node = source_node.in_nodes[const_index]
        if const_node.type == 'Identity':
          const_node = const_node.in_nodes[0]
        scale = tensor_util.MakeNdarray(const_node.get_attr('value'))
        self.set_weight(source_node.name, 'scale', scale)
        IR_node.attr['scale'].b = True
        IR_node.attr['epsilon'].f = 0
        self.set_weight(source_node.name, 'bias', np.zeros(scale.shape))
        IR_node.attr['use_bias'].b = True
        IR_node.attr['bias'].b = True
        self.set_weight(source_node.name, 'mean', np.zeros(scale.shape))
        self.set_weight(source_node.name, 'var', np.ones(scale.shape))
        self.skip_node_list.append(source_node.out_nodes[0].name)
        if len(source_node.out_nodes) == 1:
          logger.info("only one output node for mul node")
          if len(source_node.out_nodes[0].in_nodes) > 1 and source_node.out_nodes[0].in_nodes[const_index].type == "Const":
            logger.info("fusing mul and add to scale")
          if (source_node.out_nodes[0].type == "Add" or source_node.out_nodes[0].type == "AddV2"):
            child_node_shape = self.tensor_shape_to_list(source_node.out_nodes[0].get_attr('_output_shapes'))[0]
            if (current_node_shape == child_node_shape and child_node_shape[-1] == current_node_shape[-1]):
              bias = tensor_util.MakeNdarray(source_node.out_nodes[0].in_nodes[const_index].get_attr('value'))
              self.set_weight(source_node.name, 'bias', bias)
              IR_node.attr['use_bias'].b = True
              IR_node.attr['bias'].b = True
              self.set_weight(source_node.name, 'mean', np.zeros(bias.shape))
              self.set_weight(source_node.name, 'var', np.ones(bias.shape))
              self.skip_node_list.append(source_node.out_nodes[0].name)
        else:
          self.rename_FusedBatchNorm(source_node)
          # IR_node = self.IR_graph.node.add()
          # TensorflowParser2._copy_and_reop(source_node, IR_node, 'BatchNorm')
      elif parent_node_shape == current_node_shape and (isinstance(value, float) or isinstance(value, int)):
        logger.info("found scale candidate")
        is_bn_mul = True
        IR_node = self._convert_identity_operation(source_node, new_op='BatchNorm')
        const_node = source_node.in_nodes[const_index]
        if const_node.type == 'Identity':
          const_node = const_node.in_nodes[0]
        self.set_weight(source_node.name, 'scale', np.array([value] * current_node_shape[-1]))
        IR_node.attr['scale'].b = True
        IR_node.attr['epsilon'].f = 0
        self.set_weight(source_node.name, 'bias', np.zeros(current_node_shape[-1]))
        IR_node.attr['use_bias'].b = True
        IR_node.attr['bias'].b = True
        self.set_weight(source_node.name, 'mean', np.zeros(current_node_shape[-1]))
        self.set_weight(source_node.name, 'var', np.ones(current_node_shape[-1]))
    if not is_bn_mul:
      if len(scopes) >= 2:
        if scopes[-2] == "batchnorm" or scopes[-2].startswith("Assign"):
          return
      self._add_constant_node(source_node)
      self._convert_identity_operation(source_node)

  def rename_Add(self, source_node):
    if source_node.name in self.skip_node_list:
      #skip node
      source_node.real_name = self.src_graph.get_node(source_node.in_edges[0]).real_name
      return
    scopes = self._get_scopes(source_node.name)
    if len(scopes) > 1:
      if scopes[-2] == 'batchnorm':
        # if scopes[-3] == 'BatchNorm' or scopes[-3] == 'batch_normalization':
        if True:
          self._convert_layers_batchnorm(source_node)
        elif scopes[-3] == 'InstanceNorm':
          self._convert_layers_instancenorm(source_node)
      elif self.get_parent(source_node.name, [1]).get_attr('value') is not None:
        # If the case is Tensor  + Scalar --> map it to Batchnorm.
        self._convert_hard_swish_batchnorm(source_node)
        return
      else:
        if source_node.in_nodes[0].type == "Const" or source_node.in_nodes[1].type == "Const":
          idx = 1
          if source_node.in_nodes[0].type == "Const":
            idx = 0
          if source_node.in_nodes[idx].get_attr('value').float_val:
            if source_node.in_nodes[idx].get_attr('value').float_val[0] == 0.0:
              source_node.real_name = self.src_graph.get_node(source_node.in_edges[1 - idx]).real_name
            else:
              IR_node = self._convert_identity_operation(source_node, new_op="Add")
              IR_node.attr['y'].f = source_node.in_nodes[idx].get_attr('value').float_val[0]
          else:
            if source_node.in_nodes[idx].get_attr('value').int_val[0] == 0:
              source_node.real_name = self.src_graph.get_node(source_node.in_edges[1 - idx]).real_name
            else:
              IR_node = self._convert_identity_operation(source_node, new_op="Add")
              IR_node.attr['y'].f = source_node.in_nodes[idx].get_attr('value').int_val[0]
        else:
          IR_node = self._convert_identity_operation(source_node, new_op="Add")
    else:
      IR_node = self._convert_identity_operation(source_node, new_op="Add")
      const_inputs = [node for node in source_node.in_nodes if node.type == "Const"]
      if len(const_inputs) == 1:
        value = self.get_parent(source_node.name, [1]).get_attr('value')
        if value.tensor_content:
          value = tensor_util.MakeNdarray(value)
        else:
          value = value.float_val[0]
        inp_shape = self.tensor_shape_to_list(self.get_parent(source_node.name, [0]).get_attr('_output_shapes'))[0]
        inp_shape = clast2first(inp_shape)
        if np.prod(value.shape) == np.prod(inp_shape) and value.shape != inp_shape:
          if len(value.shape) == 3:
            value = np.transpose(value, axes=[0, 2, 1])
          elif len(value.shape) == 4:
            value = np.transpose(value, axes=[0, 3, 1, 2])
        value = np.broadcast_to(value, inp_shape)
        IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
        while len(IR_node_0.input): IR_node_0.input.pop()
        IR_node_0.name = 'dvtensor_'+source_node.name + '_' + const_inputs[0].name
        out_shape = _cfirst_to_clast(inp_shape)
        IR_node_0.attr['_output_shapes'].list.shape.pop()
        _set_output_shape_from_list(IR_node_0, out_shape)
        self.set_weight(IR_node_0.name, "weights", value)
        IR_node.input[1] = IR_node_0.name

  def _convert_hard_swish_batchnorm(self, source_node):
    if self.get_parent(source_node.name, [1]).get_attr('value') is not None:
      value = self.get_parent(source_node.name, [1]).get_attr('value')
      if value.tensor_content:
        value = tensor_util.MakeNdarray(value)
      else:
        value = value.float_val[0]
    IR_node = self._convert_identity_operation(source_node, new_op='BatchNorm')
    self.set_weight(IR_node.name, 'var', 1.0)
    self.set_weight(IR_node.name, 'mean', value * -1.0)
    IR_node.attr['epsilon'].f = 0  #float('1e-10')

  def rename_AddV2(self, source_node):
    self.rename_Add(source_node)

  def rename_Fill(self, source_node):
    node_attr = self.src_graph.get_parent(source_node.name, [1]).get_attr('value')
    if len(node_attr.bool_val): value = node_attr.bool_val[0]
    elif len(node_attr.dcomplex_val): value = node_attr.dcomplex_val[0]
    elif len(node_attr.double_val): value = node_attr.double_val[0]
    elif len(node_attr.float_val): value = node_attr.float_val[0]
    elif len(node_attr.half_val): value = node_attr.half_val[0]
    elif len(node_attr.int64_val): value = node_attr.int64_val[0]
    elif len(node_attr.int_val): value = node_attr.int_val[0]
    elif len(node_attr.resource_handle_val): value = node_attr.resource_handle_val[0]
    elif len(node_attr.string_val): value = node_attr.string_val[0]
    elif len(node_attr.uint32_val): value = node_attr.uint32_val[0]
    elif len(node_attr.uint64_val): value = node_attr.uint64_val[0]
    elif len(node_attr.variant_val): value = node_attr.variant_val[0]
    dims = tensor_util.MakeNdarray(self.src_graph.get_parent(source_node.name, [0]).get_attr('value'))
    IR_node = self._convert_identity_operation(source_node, new_op="Tf_Fill")
    kwargs = {'dims': dims, 'value': value}
    assign_IRnode_values(IR_node, kwargs)

  def rename_Sub(self, source_node):
    scopes = self._get_scopes(source_node.name)
    if len(scopes) > 2:
        if scopes[-2].startswith('Assign') or scopes[-1].startswith('Assign'):
            return
    const_inputs = [node for node in source_node.in_nodes if node.type=="Const"]
    if len(const_inputs) == 1:
        IR_node = self._convert_identity_operation(source_node, new_op = "Add")
        value = self.get_parent(source_node.name, [1]).get_attr('value')
        if value.tensor_content:
            value = -1*tensor_util.MakeNdarray(value)
        else:
            value = np.array(-1*value.float_val[0])
        inp_shape = self.tensor_shape_to_list(self.get_parent(source_node.name, [0]).get_attr('_output_shapes'))[0]
        inp_shape = clast2first(inp_shape)
        if np.prod(value.shape) == np.prod(inp_shape) and value.shape != inp_shape:
            if len(value.shape) == 3:
                value = np.transpose(value, axes=[0,2,1])
            elif len(value.shape) == 4:
                value = np.transpose(value, axes=[0,3,1,2])
        value = np.broadcast_to(value, inp_shape)
        IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
        while len(IR_node_0.input): IR_node_0.input.pop()
        IR_node_0.name = 'dvtensor_'+source_node.name + '_' + const_inputs[0].name
        out_shape = _cfirst_to_clast(inp_shape)
        IR_node_0.attr['_output_shapes'].list.shape.pop()
        _set_output_shape_from_list(IR_node_0, out_shape)
        self.set_weight(IR_node_0.name, "weights", value)
        IR_node.input[1] = IR_node_0.name
        return
    IR_node = self._convert_identity_operation(source_node, end_idx=2, new_op = "Sub")
    self._add_constant_node(source_node)

  def rename_Sum(self, source_node):
    IR_node = self._convert_identity_operation(source_node, start_idx=0, end_idx=1, new_op='Sum')
    input_node = self.src_graph.get_parent(source_node.name, [0])
    kwargs = {}
    kwargs['cal_shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    input_node_indices = self.src_graph.get_parent(source_node.name, [1])
    indice_value = input_node_indices.get_attr('value')
    if indice_value.tensor_content:
      shapes = tensor_util.MakeNdarray(indice_value)
      c = shapes.tolist()
      kwargs['sum_indices'] = c
    else:
      kwargs['sum_indices'] = input_node_indices.get_attr('value').int_val[0]
    assign_IRnode_values(IR_node, kwargs)

  def rename_Rsqrt(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Rsqrt")
    kwargs = {}
    input_node = self.src_graph.get_parent(source_node.name, [0])
    kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    assign_IRnode_values(IR_node, kwargs)

  def rename_Square(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Square')
    input_node = self.src_graph.get_parent(source_node.name, [0])
    kwargs = {}
    kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    assign_IRnode_values(IR_node, kwargs)

  def rename_Sigmoid(self, source_node):
    real_name = source_node.real_name.replace('/', '_')
    real_name = real_name.replace('-', '_')
    if not real_name in self.dummy_layers:
      self.rename_dvnet_Sigmoid(source_node)
      return
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Sigmoid')

  def rename_dvnet_Sigmoid(self, source_node):
    IR_node = self._convert_identity_operation(source_node)
    kwargs = {}
    input_node = self.src_graph.get_parent(source_node.name, [0])
    kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    assign_IRnode_values(IR_node, kwargs)
    return IR_node

  def rename_Reciprocal(self, source_node):
    IR_node = self._convert_identity_operation(source_node, start_idx=0, end_idx=1, new_op='Reciprocal')

  def rename_Minimum(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Minimum')

  def rename_Maximum(self, source_node):
    if source_node.name in self.leakyrelu_maximum:
      IR_node = self._convert_identity_operation(source_node, new_op='LeakyRelu')
      alpha_node = self.src_graph.get_parent(source_node.name, [1])
      slope = alpha_node.in_nodes[0].get_attr('value').float_val[0]
      IR_node.attr["alpha"].f = slope
      return
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Maximum')

  def rename_RealDiv(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Div')
    self._add_constant_node(source_node)

  def rename_Enter(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Enter')

  def rename_Switch(self, source_node):
    # Skip the node as merge
    source_node.real_name = self.src_graph.get_node(source_node.in_edges[0]).real_name

  def rename_Identity(self, source_node):
    source_node.real_name = self.src_graph.get_node(source_node.in_edges[0]).real_name

  def rename_IdentityN(self, source_node):
    source_node.real_name = self.src_graph.get_node(source_node.in_edges[0]).real_name

  def rename_ResizeBilinear(self, source_node):
    input_node = self.src_graph.get_parent(source_node.name, [0])
    input_shape = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    output_shape = self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]
    '''
        if input_shape[1] == input_shape[2]and input_shape[1] == 1:
            kwargs = {}
            IR_node = self._convert_identity_operation(source_node,end_idx=1, new_op = 'ResizeBilinear')
            kwargs['shape'] = input_shape
            assign_IRnode_values(IR_node, kwargs)
            return
        '''
    if input_shape == output_shape:
      source_node.real_name = self.src_graph.get_node(source_node.in_edges[0]).real_name
    else:
      kwargs = {}
      scale = output_shape[1] / input_shape[1]
      mode = None
      input_node = self.src_graph.get_parent(source_node.name, [0])
      kwargs['shape'] = input_shape
      kwargs['align_corners'] = source_node.get_attr('align_corners')
      if (source_node.get_attr('align_corners')):
        mode = UPSP_TYPE.BILINEAR_TRUE.value
      else:
        IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op='ResizeBilinear')
        kwargs = {}
        input_node = self.src_graph.get_parent(source_node.name, [0])
        kwargs['shape'] = input_shape
        kwargs['align_corners'] = source_node.get_attr('align_corners')
        kwargs['half_pixel_centers'] = source_node.get_attr('half_pixel_centers')
        assign_IRnode_values(IR_node, kwargs)
        return
      kwargs['half_pixel_centers'] = source_node.get_attr('half_pixel_centers')
      if mode == UPSP_TYPE.BILINEAR_FALSE.value:
        kwargs['scale'] = scale
      else:
        s = []
        s.append(scale)
        s.append((input_shape[1] - 1) / (output_shape[1] - 1))
        s.append((input_shape[2] - 1) / (output_shape[2] - 1))
        s.append(output_shape[1])
        s.append(output_shape[2])
        kwargs['scale'] = s
      kwargs['mode'] = mode
      IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op='Upsample')
      assign_IRnode_values(IR_node, kwargs)

  def rename_Cast(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Cast')
    input_node = self.src_graph.get_parent(source_node.name, [0])
    kwargs = {}
    kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    assign_IRnode_values(IR_node, kwargs)

  def rename_Prod(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Prod')
    input_node = self.src_graph.get_parent(source_node.name, [0])
    kwargs = {}
    kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    input_node_const = self.src_graph.get_parent(source_node.name, [1])
    kwargs['const'] = input_node_const.get_attr('value').int_val[0]
    assign_IRnode_values(IR_node, kwargs)

  def rename_Shape(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Shape')
    input_node = self.src_graph.get_parent(source_node.name, [0])
    kwargs = {}
    kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    assign_IRnode_values(IR_node, kwargs)

  def rename_Squeeze(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Squeeze')
    if source_node.get_attr('_output_shapes')[0].unknown_rank or len(source_node.layer.attr['squeeze_dims'].list.i) == 0 or len(source_node.layer.attr['axis'].list.i) == 0:
      input_node = self.src_graph.get_parent(source_node.name, [0])
      kwargs = {}
      kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
      kwargs['shape'] = list(filter((1).__ne__, kwargs['shape']))
      kwargs['shape'] = list(filter((-1).__ne__, kwargs['shape']))
      assign_IRnode_values(IR_node, kwargs)

  def rename_Gather(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Embedding')
    W = self.src_graph.get_parent(source_node.name, [0])
    W = self.src_graph.get_parent(W.name, [0])
    self.set_weight(source_node.name, "weights", self.ckpt_data[W.name])
    kwargs = {'input_dim': self.ckpt_data[W.name].shape[0], 'output_dim': self.ckpt_data[W.name].shape[1], 'mask_zero': False}
    kwargs['axis'] = 0  # add default
    assign_IRnode_values(IR_node, kwargs)
    return IR_node

  def rename_GatherV2(self, source_node):
    IR_node = self.rename_Gather(source_node)
    kwargs = {}
    kwargs['axis'] = source_node.layer.attr['axis'].i
    assign_IRnode_values(IR_node, kwargs)

  def rename_StridedSlice(self, source_node):
    # if source_node.in_nodes[0].layer.op == 'Const':
    #     IR_node_tensor = self._convert_identity_operation(source_node.in_nodes[0],new_op = 'DVTensor')
    #     const_tensor = tensor_util.MakeNdarray(source_node.in_nodes[0].get_attr('value'))
    #     self.set_weight(IR_node_tensor.name, "weights", const_tensor)
    IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op='Slice')
    kwargs = {}
    kwargs = {
        'begin_mask': source_node.get_attr('begin_mask'),
        'end_mask': source_node.get_attr('end_mask'),
        'shrink_axis_mask': source_node.get_attr('shrink_axis_mask'),
        'new_axis_mask': source_node.get_attr('new_axis_mask'),
        'ellipsis_mask': source_node.get_attr('ellipsis_mask'),
    }
    begin_mask = []
    end_mask = []
    shrink_axis_mask = []
    begin_mask_value = kwargs['begin_mask']
    end_mask_value = kwargs['end_mask']
    shrink_axis_mask_value = kwargs['shrink_axis_mask']
    if kwargs['new_axis_mask']:
      raise NotImplementedError("Non zero new_axis_mask currently not implemented")
    if kwargs['ellipsis_mask']:
      raise NotImplementedError("Non zero ellipsis_mask currently not implemented")
    mask_length = len(self.src_graph.get_parent(source_node.name, [0]).get_attr('_output_shapes')[0].dim)
    for i in range(mask_length):
      begin_mask.append(begin_mask_value % 2)
      begin_mask_value = begin_mask_value // 2
      end_mask.append(end_mask_value % 2)
      end_mask_value = end_mask_value // 2
      shrink_axis_mask.append(shrink_axis_mask_value % 2)
      shrink_axis_mask_value = shrink_axis_mask_value // 2
    starts = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor
    starts = tensor_util.MakeNdarray(starts).tolist()
    starts = starts + [0] * (mask_length - len(starts))
    kwargs['starts'] = starts
    ends = self.get_parent(source_node.name, [2]).layer.attr['value'].tensor
    ends = tensor_util.MakeNdarray(ends).tolist()
    ends = ends + [0] * (mask_length - len(ends))
    kwargs['ends'] = ends
    kwargs['shape'] = self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]
    kwargs['shrink_axis_mask'] = shrink_axis_mask
    for i in range(mask_length):
      if begin_mask[i] == 1:
        kwargs['starts'][i] = 0
      if end_mask[i] == 1:
        kwargs['ends'][i] = kwargs['shape'][i]
    if self.get_parent(source_node.name, [3]) != None:
      if 'value' in self.get_parent(source_node.name, [3]).layer.attr:
        strides = self.get_parent(source_node.name, [3]).layer.attr['value'].tensor
        strides = tensor_util.MakeNdarray(strides).tolist()
      else:
        shape = self.get_parent(source_node.name, [3]).layer.attr['_output_shapes'].list.shape[0].dim[0].size
        np_type = tensor_util.dtypes.as_dtype(self.get_parent(source_node.name, [3]).layer.attr['index_type'].type).as_numpy_dtype
        const_node = self.get_parent(source_node.name, [3]).layer.input
        const_node = const_node[0] if 'Const' in const_node[0] else const_node[1]
        const_node = self.tf_graph.layer_map[const_node].layer
        node_attr = const_node.attr['value'].tensor
        if len(node_attr.bool_val): value = node_attr.bool_val[0]
        elif len(node_attr.dcomplex_val): value = node_attr.dcomplex_val[0]
        elif len(node_attr.double_val): value = node_attr.double_val[0]
        elif len(node_attr.float_val): value = node_attr.float_val[0]
        elif len(node_attr.half_val): value = node_attr.half_val[0]
        elif len(node_attr.int64_val): value = node_attr.int64_val[0]
        elif len(node_attr.int_val): value = node_attr.int_val[0]
        elif len(node_attr.resource_handle_val): value = node_attr.resource_handle_val[0]
        elif len(node_attr.string_val): value = node_attr.string_val[0]
        elif len(node_attr.uint32_val): value = node_attr.uint32_val[0]
        elif len(node_attr.uint64_val): value = node_attr.uint64_val[0]
        elif len(node_attr.variant_val): value = node_attr.variant_val[0]
        strides = np.full(shape, np_type(value)).tolist()
      kwargs['strides'] = strides
    for i in kwargs['strides']:
      if i != 1:
        raise NotImplementedError("non unity strides in stridedslice currently not supported")
    assign_IRnode_values(IR_node, kwargs)

  def rename_ResizeNearestNeighbor(self, source_node):
    real_name = source_node.real_name.replace('/', '_')
    input_node = self.src_graph.get_parent(source_node.name, [0])
    input_shape = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    output_shape = self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]
    mode = None
    kwargs = {}
    if source_node.get_attr('align_corners'):
      kwargs['align_corners'] = source_node.get_attr('align_corners')
    else:
      kwargs['align_corners'] = False
    if source_node.get_attr('half_pixel_centers'):
      kwargs['half_pixel_centers'] = source_node.get_attr('half_pixel_centers')
    else:
      kwargs['half_pixel_centers'] = False
    if (len(output_shape) == 4):
      mode = UPSP_TYPE.NEAREST_4D.value
    elif (len(output_shape) == 5):
      #import pdb; pdb.set_trace()
      if 'zero_mode' in source_node.layer.attr:
        kwargs['zero_mode'] = [_i for _i in source_node.layer.attr['zero_mode'].list.i]
      else:
        kwargs['zero_mode'] = [0] * (len(output_shape) - 2)
      mode = UPSP_TYPE.NEAREST_5D.value
      scale = [oo / ii for oo, ii in zip(output_shape[1:-1], input_shape[1:-1])]
      IR_node = self._convert_identity_operation(source_node, new_op="Upsample")
      kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
      kwargs['scale'] = scale
      kwargs['mode'] = mode
      kwargs['align_corners'] = "NearestNeighbor"
      assign_IRnode_values(IR_node, kwargs)
      return
    else:
      mode = UPSP_TYPE.BILINEAR_FALSE.value
    scale = [oo / ii for oo, ii in zip(output_shape[1:-1], input_shape[1:-1])]
    if len(output_shape) in [4, 5]:
      croppable = False
      if all([oo == (math.ceil(ss) * ii - 1) for ss, oo, ii in zip(scale, output_shape[1:-1], input_shape[1:-1])]):
        croppable = True
        scale = [float(math.ceil(ss)) for ss in scale]
      # for 2d allowed cases : scale_h = scale_w
      # for 3d allowed cases either : (scale_d = scale_h = scale_w) or (scale_d = 1, scale_h = scale_w)
      if (input_shape[-1] == output_shape[-1] and scale[-1] == scale[-2] and (len(scale) == 2 or (scale[0] in [1, scale[-1]])) and (scale[-1].is_integer() or croppable) and
          (not source_node.get_attr('align_corners') or (source_node.get_attr('align_corners') and scale[-1].is_integer() and int(scale[-1]) == 2))):
        if not real_name in self.dummy_layers:
          self.rename_dvnet_ResizeNearestNeighbor(source_node, scale)
        return
    IR_node = self._convert_identity_operation(source_node, new_op="Upsample")
    kwargs['shape'] = self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]
    kwargs['scale'] = scale
    kwargs['mode'] = mode
    if source_node.get_attr('align_corners'):
      IR_node = self._convert_identity_operation(source_node, new_op="Tf_ResizeNearestNeighbor")
      assign_IRnode_values(IR_node, kwargs)
      return
    assign_IRnode_values(IR_node, kwargs)

  def rename_dvnet_ResizeNearestNeighbor(self, source_node, scale):
    IR_node = self._convert_identity_operation(source_node)
    kwargs = {}
    input_node = self.src_graph.get_parent(source_node.name, [0])
    kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    if 'zero_mode' in source_node.layer.attr:
      kwargs['zero_mode'] = [_i for _i in source_node.layer.attr['zero_mode'].list.i]
    if source_node.get_attr('align_corners'):
      logger.error("align corners = true is currently unsupported")
    kwargs['scale'] = scale
    if (len(kwargs['shape']) == 5):
      kwargs['mode'] = UPSP_TYPE.NEAREST_5D.value
    else:
      kwargs['mode'] = UPSP_TYPE.NEAREST_4D.value
    assign_IRnode_values(IR_node, kwargs)

  def rename_dvnet_Conv2D(self, source_node):
    IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op='Conv')
    kwargs = {}
    kwargs['strides'] = source_node.get_attr('strides')
    logger.debug("dilation is %s", source_node.get_attr('dilations'))
    kwargs['dilations'] = source_node.get_attr('dilations') or [1] * len(kwargs['strides'])
    kwargs['padding'] = source_node.get_attr('padding')
    input_node = self.src_graph.get_parent(source_node.name, [0])
    kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    # weights
    input_node_weight = self.src_graph.get_parent(source_node.name, [1])
    tensor_content = self.check_const(input_node_weight).get_attr('value')
    W = tensor_util.MakeNdarray(tensor_content)
    kwargs['kernel_shape'] = self.tensor_shape_to_list(input_node_weight.get_attr('_output_shapes'))[0]
    self.set_weight(source_node.name, 'weights', W)
    spatial_size = len(kwargs['kernel_shape'][:-2])
    kernel_size_dil = [1] * spatial_size
    for i in range(spatial_size):
      kernel_size_dil[i] = kwargs['kernel_shape'][i] + (kwargs['kernel_shape'][i] - 1) * (kwargs['dilations'][1 + i] - 1)
    self._convert_padding(source_node, IR_node, kernel_size_dil)
    assign_IRnode_values(IR_node, kwargs)
    logger.debug("IR value for conv is %s", IR_node)
    self._get_bias(source_node, IR_node)

  def rename_dvnet_Conv3D(self, source_node):
    self.rename_dvnet_Conv2D(source_node)

  def rename_Relu(self, source_node):
    if source_node.real_variable_name not in self.dummy_layers:
      self.rename_dvnet_Relu(source_node)
      return
    IR_node = self._convert_identity_operation(source_node, new_op='Mxnet_Activation')
    IR_node.attr['act'].i = 0

  def rename_dvnet_Relu(self, source_node):
    IR_node = self._convert_identity_operation(source_node)
    kwargs = {'shape': self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]}
    assign_IRnode_values(IR_node, kwargs)

  def rename_MaxPool(self, source_node):
    real_name = source_node.name.replace('/', '_')
    if real_name in self.dummy_layers:
      return self.rename_Tf_Maxpool(source_node)
    self._convert_pooling(source_node, b'MAX')

  def rename_MaxPool3D(self, source_node):
    self.rename_MaxPool(source_node)

  def rename_Tf_Maxpool(self, source_node):
    mode_map = {'SAME': 0, 'VALID': 1}
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Maxpool')
    kwargs = dict()
    kwargs["ksize"] = source_node.get_attr("ksize")
    kwargs["strides"] = source_node.get_attr("strides")
    kwargs["padding"] = source_node.get_attr("padding")
    assign_IRnode_values(IR_node, kwargs)

  def rename_AvgPool(self, source_node):
    input_node = self.get_parent(source_node.name, [0])
    input_shape = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    if source_node.get_attr('padding') == 'VALID':
      dims = len(input_shape)
      padding = [0, 0] * dims
    elif source_node.get_attr('padding') == 'SAME':
      padding = compute_tf_same_padding(input_shape, source_node.get_attr('ksize')[1:-1], source_node.get_attr('strides'))
    else:
      logger.error("unknown padding type: layer=%s , padding=%s", source_node.name, source_node.get_attr('padding'))
      return
    real_name = source_node.name.replace('/', '_')
    if real_name in self.dummy_layers or sum(padding) > 0:
      return self.rename_Tf_Avgpool(source_node)
    self._convert_pooling(source_node, b'AVG')

  def rename_Tf_Avgpool(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Tf_AvgPool")
    kwargs = dict()
    kwargs['padding'] = source_node.get_attr('padding')
    kwargs['ksize'] = source_node.get_attr('ksize')
    kwargs['strides'] = source_node.get_attr('strides')
    assign_IRnode_values(IR_node, kwargs)

  def rename_dvnet_LRN(self, source_node):
    IR_node = self._convert_identity_operation(source_node)
    # alpha
    IR_node.attr["alpha"].f = float(source_node.get_attr("alpha", "0.0001"))
    # beta
    IR_node.attr["beta"].f = float(source_node.get_attr("beta", "0.75"))
    IR_node.attr["size"].i = source_node.get_attr("depth_radius")
    IR_node.attr["bias"].f = float(source_node.get_attr("bias"))

  def rename_LRN(self, source_node):
    # real_name = source_node.real_name.replace('/','_')
    # real_name = real_name.replace('-','_')
    # if not real_name in self.dummy_layers:
    #     self.rename_dvnet_LRN(source_node)
    #     return
    IR_node = self._convert_identity_operation(source_node, new_op="Tf_LRN")
    IR_node.attr["alpha"].f = float(source_node.get_attr("alpha", "1.0"))
    IR_node.attr["beta"].f = float(source_node.get_attr("beta", "0.5"))
    IR_node.attr["depth_radius"].i = source_node.get_attr("depth_radius", "5")
    IR_node.attr["bias"].f = float(source_node.get_attr("bias", "1.0"))

  def rename_Concat(self, source_node):
    n = len(source_node.in_edges)
    IR_node = self._convert_identity_operation(source_node, start_idx=1, end_idx=n, new_op='Concat')
    axis = self.tf_graph.get_parent(source_node.name, [n - 1]).get_attr('value').int_val[0]
    if axis < 0:
      axis = len(self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]) + axis
    IR_node.attr["axis"].i = axis

  def rename_ConcatV2(self, source_node):
    n = len(source_node.in_edges)
    IR_node = self._convert_identity_operation(source_node, start_idx=0, end_idx=n - 1, new_op='Concat')
    axis = self.tf_graph.get_parent(source_node.name, [n - 1]).get_attr('value').int_val[0]
    if axis < 0:
      axis = len(self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]) + axis
    IR_node.attr["axis"].i = axis

  def rename_Tf_Softmax(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Tf_Softmax")
    kwargs = {'shape': self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]}
    logger.warn("assuming last axis for softmax")
    axis = source_node.get_attr('axis')
    if axis is None or axis == -1:
      axis = len(kwargs['shape'])
      IR_node.attr["axis"].i = axis - 1
    if len(kwargs['shape']) == 4:
      IR_node.attr["axis"].i = 3
    elif len(kwargs['shape']) == 3:
      IR_node.attr["axis"].i = 2
    elif len(kwargs['shape']) == 2:
      IR_node.attr["axis"].i = 1
    assign_IRnode_values(IR_node, kwargs)

  def rename_MatMul(self, source_node):
    # Supporting MatMul using Conv2D
    input_weight_node = self.src_graph.get_parent(source_node.name, [1])
    weightnode = self.check_const(input_weight_node)
    if weightnode:  #or source_node.type == 'BatchMatMulV2':
      kwargs = {}
      kwargs['kernel_shape'] = self.tensor_shape_to_list(input_weight_node.get_attr('_output_shapes'))[0]
      # if reshape after matmul brings it back to 4D dims; then onnly it can be mapped to 1x1 Conv
      if self.src_graph.get_son(source_node.name, [0]).type == 'Reshape':
        logger.info("converting Matmul using Conv1x1")
        IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op='Conv')
        if len(kwargs['kernel_shape']) == 2:
          kwargs['kernel_shape'] = [1, 1] + kwargs['kernel_shape']
        kwargs['strides'] = [1] * 4
        kwargs['dilations'] = [1] * 4
        kwargs['pads'] = [0, 0] * 4
        # infer 4D input shapes; MatMul converts the inputs into Matrix (combining spatial dims into 1 dim x channels dim)
        reshape_out = self.src_graph.get_son(source_node.name, [0], set_flag=True)
        next_node = self.src_graph.get_son(reshape_out.name, [0])
        next_node.in_edges[0] = source_node.name
        final_shape = graph_pb2.TensorShape()
        for dim1 in reshape_out.get_attr('_output_shapes')[0].dim:
          new_dim = final_shape.dim.add()
        if dim1 == None:
          new_dim.size = -1
        else:
          new_dim.size = dim1.size
        IR_node.attr['_output_shapes'].list.shape.pop()
        IR_node.attr['_output_shapes'].list.shape.insert(0, final_shape)
        assign_IRnode_values(IR_node, kwargs)
      else:
        logger.info("converting Matmul not using Conv1x1")
        IR_node = self._convert_identity_operation(source_node, end_idx=1)
      weight_value = weightnode.get_attr('value')
      weight = tensor_util.MakeNdarray(weight_value).reshape(kwargs['kernel_shape'])
      self.set_weight(source_node.name, 'weights', weight)
      units = source_node.layer.attr['_output_shapes'].list.shape[-1].dim[-1].size
      IR_node.attr['units'].i = units
      if source_node.out_edges and self.tf_graph.get_node(source_node.out_edges[0]).type == 'BiasAdd':
        add_node = self.tf_graph.get_node(source_node.out_edges[0])
        add_node.covered = True
        add_node.real_name = source_node.real_name
        TensorflowParser2._copy_and_reop(source_node, IR_node, 'FullyConnected')
        variable = self.tf_graph.get_node(add_node.in_edges[1])  #add_bias node
        biasnode = self.check_const(variable)
        bias_value = biasnode.get_attr('value')
        bias = tensor_util.MakeNdarray(bias_value)
        self.set_weight(source_node.name, 'bias', bias)
        IR_node.attr['use_bias'].b = True
    else:
      IR_node = self._convert_identity_operation(source_node, new_op="Tf_Matmul")
      kwargs = {}
      kwargs['transpose_a'] = source_node.get_attr('transpose_a')
      kwargs['transpose_b'] = source_node.get_attr('transpose_b')
      assign_IRnode_values(IR_node, kwargs)

  def rename_Softmax(self, source_node):
    axis = source_node.get_attr('axis')
    real_name = source_node.name.replace('/', '_')
    if real_name in self.dummy_layers:
      self.rename_Tf_Softmax(source_node)
      return
    IR_node = self._convert_identity_operation(source_node, new_op="Softmax")
    kwargs = {'shape': self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]}
    logger.warn("assuming last axis for softmax")
    if axis is None or axis == -1:
      axis = len(kwargs['shape'])
      IR_node.attr["axis"].i = axis - 1
    if len(kwargs['shape']) == 4:
      IR_node.attr["axis"].i = 3
    elif len(kwargs['shape']) == 3:
      IR_node.attr["axis"].i = 2
    elif len(kwargs['shape']) == 2:
      IR_node.attr["axis"].i = 1
    assign_IRnode_values(IR_node, kwargs)

  def rename_BiasAdd(self, source_node):
    # Skip BiasAdd
    source_node.real_name = self.src_graph.get_node(source_node.in_edges[0]).real_name

  def rename_QuantizeV2(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='QuantizeV2')
    TensorflowParser2._copy_shape(source_node, IR_node)

  def rename_QuantizedRelu(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="QuantizedRelu")
    kwargs = {'shape': self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]}
    assign_IRnode_values(IR_node, kwargs)

  def rename_LeakyRelu(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="LeakyRelu")
    if isinstance(source_node.get_attr('alpha'), float):
      IR_node.attr['alpha'].f = float(source_node.get_attr('alpha'))
    else:
      IR_node.attr['alpha'].f = float(source_node.get_attr('alpha').float_val[0])

  def rename_L2Norm(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='L2Norm')
    kwargs = {'across_spatial': source_node.get_attr('across_spatial'), 'eps': source_node.get_attr('eps')}
    assign_IRnode_values(IR_node, kwargs)

  def rename_QuantizedReshape(self, source_node):
    IR_node = self._convert_identity_operation(source_node, end_idx=1)
    kwargs = {'shape': self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]}
    assign_IRnode_values(IR_node, kwargs)

  def rename_QuantizedConv2D(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='QConv')
    kwargs = {}
    kwargs['strides'] = source_node.get_attr('strides')
    kwargs['padding'] = source_node.get_attr('padding')
    # weights
    input_node = self.src_graph.get_parent(source_node.name, [1])
    tensor_content = input_node.get_attr('value')
    W = tensor_util.MakeNdarray(tensor_content)
    W = W.astype(np.uint8)
    kwargs['kernel_shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    input_node_minw = self.src_graph.get_parent(source_node.name, [4])
    min_W = input_node_minw.get_attr('value').float_val[0]
    input_node_maxw = self.src_graph.get_parent(source_node.name, [5])
    max_W = input_node_maxw.get_attr('value').float_val[0]
    if source_node.get_attr('Tfilter') == tensorflow.quint8:
      W = ((max_W - min_W) / 255.0) * W + min_W
    else:
      assert False, ('Only uint8 weights handled currently by the converter')
    self.set_weight(source_node.name, 'kernel_weights', W)
    assign_IRnode_values(IR_node, kwargs)

  def rename_Requantize(self, source_node):
    input_node = self.get_parent(source_node.name, [0])
    son_node = self.get_son(source_node.name, [0])
    son_node.real_name = source_node.name

  def rename_RequantizationRange(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='RequantizationRange')
    TensorflowParser2._copy_shape(source_node, IR_node)

  def rename_ZerosLike(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='ZerosLike')

  def rename_Rank(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Rank')

  def rename_Transpose(self, source_node):
    IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op='Transpose')
    input_node_perm = self.get_parent(source_node.name, [1])
    # input_node_perm = self.check_const(self.get_parent(source_node.name, [1], True))
    tensor_content = input_node_perm.get_attr('value')
    perm = tensor_util.MakeNdarray(tensor_content).tolist()
    assign_IRnode_values(IR_node, {'perm': perm, 'data_format': 'NHWC'})

  def rename_Greater(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Greater')

  def rename_All(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='All')

  def rename_LogicalAnd(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Mul')

  def rename_Pack(self, source_node):
    #source_node.real_name =  self.src_graph.get_node(source_node.in_edges[0]).real_name
    IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op='Pack')
    kwargs = {}
    kwargs['axis'] = source_node.layer.attr['axis'].i
    assign_IRnode_values(IR_node, kwargs)

  def rename_BiFPN(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='BiFPN')
    IR_node.attr['w1'].f = source_node.get_attr(source_node.in_edges[0]).float_val[0]
    IR_node.attr['w2'].f = source_node.get_attr(source_node.in_edges[1]).float_val[0]
    IR_node.attr['w3'].f = source_node.get_attr(source_node.in_edges[2]).float_val[0]

  def rename_BiFPNinterim(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='BiFPN')
    IR_node.attr['w1'].f = source_node.get_attr(source_node.in_edges[0]).float_val[0]
    IR_node.attr['w2'].f = source_node.get_attr(source_node.in_edges[1]).float_val[0]

  def rename_Pad(self, source_node):
    IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op='Pad')
    kwargs = {}
    kwargs['mode'] = 'constant'
    # paddings
    padding = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor
    shapes = tensor_util.MakeNdarray(padding)
    kwargs['pads'] = convert_tf_pad_to_onnx(shapes)
    assign_IRnode_values(IR_node, kwargs)

  def rename_FusedBatchNormV3(self, source_node):
    if source_node.real_variable_name not in self.dummy_layers:
      self.rename_dvnet_FusedBatchNormV3(source_node)
      return
    scalenode = self.check_const(self.get_parent(source_node.name, [1], True))
    if ':' in source_node.in_edges[1]:  # ?
      scalenode = None
    if scalenode:
      scale_value = scalenode.get_attr('value')
      IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op='Mxnet_Batchnorm')
      # for attr.shape >= 2
      for i in range(len(IR_node.attr["_output_shapes"].list.shape) - 1):
        IR_node.attr["_output_shapes"].list.shape.pop()
    else:
      # For models built by slim.batch_norm, remove duplicate BN (eg.facenet)
      return
    IR_node.attr['axis'].i = 1
    IR_node.attr['fix_gamma'].b = False
    IR_node.attr['output_mean_var'].b = False
    IR_node.attr['eps'].f = source_node.get_attr('epsilon', 0)
    IR_node.attr['momentum'].f = 1
    IR_node.attr['use_global_stats'].b = False
    IR_node.attr['bias'].b = True
    IR_node.attr['scale'].b = True
    scale = tensor_util.MakeNdarray(scale_value)
    self.set_weight(source_node.name, 'scale', scale)
    biasnode = self.check_const(self.get_parent(source_node.name, [2], True))
    if biasnode:
      bias_value = biasnode.get_attr('value')
    else:
      innode = self.get_parent(source_node.name, [2], True)
      name = innode.name.split(':')[0]
      bias_value = self.check_const(self.src_graph.layer_map[name]).get_attr('value')
    bias = tensor_util.MakeNdarray(bias_value)
    self.set_weight(source_node.name, 'bias', bias)
    meannode = self.check_const(self.get_parent(source_node.name, [3], True))
    mean_value = meannode.get_attr('value')
    mean = tensor_util.MakeNdarray(mean_value)
    self.set_weight(source_node.name, 'mean', mean)
    variancenode = self.check_const(self.get_parent(source_node.name, [4], True))
    variance_value = variancenode.get_attr('value')
    variance = tensor_util.MakeNdarray(variance_value)
    self.set_weight(source_node.name, 'var', variance)

  def rename_FusedBatchNorm(self, source_node):
    scalenode = self.check_const(self.get_parent(source_node.name, [1], True))
    if ':' in source_node.in_edges[1]:  # ?
      scalenode = None
    if scalenode:
      scale_value = scalenode.get_attr('value')
      IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op='BatchNorm')
      # for attr.shape >= 2
      for i in range(len(IR_node.attr["_output_shapes"].list.shape) - 1):
        IR_node.attr["_output_shapes"].list.shape.pop()
    else:
      # For models built by slim.batch_norm, remove duplicate BN (eg.facenet)
      return
    scale = tensor_util.MakeNdarray(scale_value)
    self.set_weight(source_node.name, 'scale', scale)
    IR_node.attr['scale'].b = True
    IR_node.attr['epsilon'].f = source_node.get_attr('epsilon', 0)
    biasnode = self.check_const(self.get_parent(source_node.name, [2], True))
    if biasnode:
      bias_value = biasnode.get_attr('value')
    else:
      innode = self.get_parent(source_node.name, [2], True)
      name = innode.name.split(':')[0]
      bias_value = self.check_const(self.src_graph.layer_map[name]).get_attr('value')
    bias = tensor_util.MakeNdarray(bias_value)
    self.set_weight(source_node.name, 'bias', bias)
    IR_node.attr['bias'].b = True
    meannode = self.check_const(self.get_parent(source_node.name, [3], True))
    mean_value = meannode.get_attr('value')
    mean = tensor_util.MakeNdarray(mean_value)
    self.set_weight(source_node.name, 'mean', mean)
    variancenode = self.check_const(self.get_parent(source_node.name, [4], True))
    variance_value = variancenode.get_attr('value')
    variance = tensor_util.MakeNdarray(variance_value)
    # if len(variance) == 0:
    # variance = np.array([1] * IR_node.attr["_output_shapes"].list.shape[0].dim[-1].size, dtype = np.float32)
    self.set_weight(source_node.name, 'var', variance)

  def rename_dvnet_FusedBatchNormV3(self, source_node):
    self.rename_FusedBatchNorm(source_node)

  def rename_SpaceToBatchND(self, source_node):
    padding = tensor_util.MakeNdarray(self.src_graph.get_parent(source_node.name, [2]).get_attr('value'))
    block_shape = tensor_util.MakeNdarray(self.src_graph.get_parent(source_node.name, [1]).get_attr('value'))
    padding = padding.tolist()
    padding = [padding[0][0], padding[0][1], padding[1][0], padding[1][1]]
    block_shape = block_shape.tolist()
    IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op='Tf_SpaceToBatchND')
    kwargs = {'block_shape': block_shape, 'padding': padding}
    assign_IRnode_values(IR_node, kwargs)

  def rename_BatchToSpaceND(self, source_node):
    IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op='BatchToSpaceND')

  def rename_ArgMax(self, source_node):
    axis = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor.int_val[0]
    real_name = source_node.name.replace('/', '_')
    if real_name in self.dummy_layers or axis != 3:
      return self.rename_Tf_ArgMax(source_node)
    IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op='argmax')
    if axis < 0:
      input_node = self.src_graph.get_parent(source_node.name, [0])
      axis = len(self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]) + axis
    kwargs = {'axis': axis, 'keepdims': True}
    assign_IRnode_values(IR_node, kwargs)

  def rename_Tf_ArgMax(self, source_node):
    IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op='Tf_ArgMax')
    axis = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor.int_val[0]
    if axis < 0:
      input_node = self.src_graph.get_parent(source_node.name, [0])
      axis = len(self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]) - 1
    kwargs = {'axis': axis, 'keepdims': True, 'select_last_index':0}
    assign_IRnode_values(IR_node, kwargs)

  # def rename_Slice(self, source_node):
  #     # IR_node = self._convert_identity_operation(source_node, new_op = 'Slice')
  #     # begin_node = self.get_parent(source_node.name, [1]).layer
  #     # kwargs = {
  #     #     'axis' : begin_node.attr['axis'].i,
  #     #     'offset' : source_node.get_attr('num_split')
  #     # }
  #     # assign_IRnode_values(IR_node, kwargs)
  ### Adding custom splitV2 ###
  def rename_SplitV(self, source_node):
    if source_node.get_attr('num_split') == 1:
      source_node.real_name = self.get_parent(source_node.name, [1]).real_name
    else:
      temp = self.get_parent(source_node.name, [2]).layer.attr['value'].tensor
      axis = int(tensor_util.MakeNdarray(temp))
      temp = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor
      size_splits = tensor_util.MakeNdarray(temp)
      IR_node = self._convert_identity_operation(source_node, start_idx=0, new_op='Tf_SplitV')
      kwargs = {
          # 'axis':self.get_parent(source_node.name, [0]).layer.attr['value'].tensor.int_val[0],
          'axis': axis,
          'split': source_node.get_attr('num_split'),
          'split_size': size_splits
      }
      assign_IRnode_values(IR_node, kwargs)
      for i in range(0, kwargs['split']):
        base_name = source_node.name
        IR_node1 = self._convert_identity_operation(source_node, start_idx=0, new_op='Tf_SplitV')
        '''if i==0:
                    out1 = base_name + '_out' + str(i)
                else:
                    out1 = base_name + '_out' + str(i)'''
        out1 = base_name + '_out' + str(i)
        IR_node1.name = out1
        IR_node1.attr["ignore_node"].b = True

  ### Adding swish_f32
  def rename_swish_f32(self, source_node):
    if '_output_shapes' not in source_node.layer.attr:
      source_node.layer.attr['_output_shapes'] = pr_node.layer.attr['_output_shapes']
    self._convert_identity_operation(source_node, new_op='Tf_swish_f32')
    '''pr_node = self.get_parent(source_node.name, [0])
        if '_output_shapes' not in source_node.layer.attr:
            source_node.layer.attr['_output_shapes'] = pr_node.layer.attr['_output_shapes']
        real_name = source_node.real_name.replace('/','_')
        real_name = real_name.replace('-','_')
        if not real_name in self.dummy_layers:
            IR_node1 = self.rename_dvnet_Sigmoid(source_node)
        else: 
            IR_node1 = self._convert_identity_operation(source_node, new_op='Tf_Sigmoid')
        IR_node = self.IR_graph.node.add()
        IR_node.name = IR_node1.name
        IR_node1.name = IR_node1.name + "_sigmoid"
        IR_node1.op = 'Sigmoid'
        IR_node.op = "Mul"
        IR_node.attr["_output_shapes"].copyfrom(IR_node1.attr['_output_shapes'])
        IR_node.input.append(self.src_graph.get_node(source_node.in_edges[0]).real_name)
        IR_node.input.append(IR_node1.name)'''

  def rename_Split(self, source_node):
    if source_node.get_attr('num_split') == 1:
      source_node.real_name = self.get_parent(source_node.name, [1]).real_name
    else:
      IR_node = self._convert_identity_operation(source_node, start_idx=1, new_op='Split')
      kwargs = {'axis': self.get_parent(source_node.name, [0]).layer.attr['value'].tensor.int_val[0], 'num_split': source_node.get_attr('num_split')}
      assign_IRnode_values(IR_node, kwargs)
      for i in range(0, kwargs['num_split']):
        base_name = source_node.name
        IR_node1 = self.IR_graph.node.add()
        IR_node1.CopyFrom(IR_node)
        out1 = base_name + '_output' + str(i)
        IR_node1.name = out1
        IR_node1.attr["ignore_node"].b = True

  def rename_Tile(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tile')

  def rename_Sqrt(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Sqrt')

  def rename_Tanh(self, source_node):
    IR_node = self._convert_identity_operation(source_node)
    kwargs = {}
    input_node = self.src_graph.get_parent(source_node.name, [0])
    kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    assign_IRnode_values(IR_node, kwargs)

  def rename_Log(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Log')

  def rename_Acos(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Acos')

  def rename_Atanh(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Atanh')

  def rename_Erf(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Erf')

  def rename_Floor(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Floor')

  def rename_ArgMin(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Argmin')
    axis = self.get_parent(source_node.name, [1]).layer.attr['value'].tensor.int_val[0]
    if axis < 0:
      input_node = self.src_graph.get_parent(source_node.name, [0])
      axis = len(self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]) + axis
    kwargs = {'axis': axis}
    assign_IRnode_values(IR_node, kwargs)

  def rename_BatchNorm(self, source_node):
    if source_node.real_variable_name not in self.dummy_layers:
      self.rename_dvnet_BatchNorm(source_node)
      return
    IR_node = self._convert_identity_operation(source_node, new_op='Mxnet_Batchnorm')
    IR_node.attr['axis'].i = 1
    IR_node.attr['fix_gamma'].b = False
    IR_node.attr['output_mean_var'].b = False
    attr = source_node.attrs
    IR_node.attr['eps'].f = attr['epsilon'].float_val[0]
    IR_node.attr['momentum'].f = 1
    IR_node.attr['use_global_stats'].b = False
    IR_node.attr['bias'].b = False
    IR_node.attr['scale'].b = False
    if source_node.get_attr('scale') is not None:
      scale = tensor_util.MakeNdarray(source_node.get_attr('scale'))
      self.set_weight(source_node.name, 'scale', scale)
      IR_node.attr['scale'].b = True
    if source_node.get_attr('bias') is not None:
      bias = tensor_util.MakeNdarray(source_node.get_attr('bias'))
      self.set_weight(source_node.name, 'bias', bias)
      IR_node.attr['bias'].b = True
    if source_node.get_attr('mean') is not None:
      mean = tensor_util.MakeNdarray(source_node.get_attr('mean'))
      self.set_weight(source_node.name, 'mean', mean)
    if source_node.get_attr('variance') is not None:
      var = tensor_util.MakeNdarray(source_node.get_attr('variance'))
      self.set_weight(source_node.name, 'var', var)

  def rename_dvnet_BatchNorm(self, source_node):
    IR_node = self.IR_graph.node.add()
    TensorflowParser2._copy_and_reop(source_node, IR_node, 'BatchNorm')
    IR_node.input.append(self.get_parent(source_node.name, [-1]).real_name)
    IR_node.attr['epsilon'].f = source_node.get_attr('epsilon').float_val[0]
    if source_node.get_attr('variance') is not None:
      var = tensor_util.MakeNdarray(source_node.get_attr('variance'))
      self.set_weight(source_node.name, 'var', var)
    if source_node.get_attr('scale') is not None:
      scale = tensor_util.MakeNdarray(source_node.get_attr('scale'))
      self.set_weight(source_node.name, 'scale', scale)
    if source_node.get_attr('bias') is not None:
      bias = tensor_util.MakeNdarray(source_node.get_attr('bias'))
      self.set_weight(source_node.name, 'bias', bias)
    if source_node.get_attr('mean') is not None:
      mean = tensor_util.MakeNdarray(source_node.get_attr('mean'))
      self.set_weight(source_node.name, 'mean', mean)

  def rename_Atan2(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Atan2')

  def rename_Equal(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Equal')

  def rename_GreaterEqual(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Greaterequal')

  def rename_Expm1(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Expm1')

  def rename_Acosh(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Tf_Acosh")

  def rename_Asin(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Asin')

  def rename_Asinh(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Asinh')

  # def rename_Bessel_i0(self, source_node):
  #     IR_node = self._convert_identity_operation(source_node, new_op = 'Tf_bessel_i0')
  # def rename_cumsum(self, source_node):
  #     IR_node = self._convert_identity_operation(source_node, new_op = 'Tf_cumsum')
  #     # input_node = self.src_graph.get_parent(source_node.name, [0])
  #     # input_node['exclusive'].b = source_node.layer.attr['exclusive'].b
  #     # input_node['reverse'].b = source_node.layer.attr['reverse'].b
  #     input_node['axis'].i = source_node.layer.attr['axis'].i
  def rename_DivNoNan(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_divnonan')

  def rename_Less(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_less')

  def rename_FloorDiv(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_floordiv')

  def rename_FloorMod(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Floormod')

  def rename_Abs(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Abs')

  def rename_Atan(self, source_node):
    self._convert_identity_operation(source_node, new_op='Tf_Atan')

  def rename_Cosh(self, source_node):
    self._convert_identity_operation(source_node, new_op='Tf_Cosh')

  def rename_Cos(self, source_node):
    self._convert_identity_operation(source_node, new_op='Tf_Cos')

  def rename_Exp(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Exp')

  def rename_AddN(self, source_node):
    self._convert_identity_operation(source_node, new_op='Tf_Add_N')

  def _compare(self, inp, out):
    if len(inp) != len(out):
      return 0
    for idx, val in enumerate(inp):
      if inp[idx] < out[idx]:
        return 1
    return -1

  def rename_DUMMY(self, source_node):
    '''
        if source_node.type == "Identity": 
            self.rename_Identity(source_node)
            return
        input_node = self.get_parent(source_node.name, [0])
        input_shape = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
        output_shape = self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]
        kwargs = {}
        kwargs['dummy_node'] = True
        self.dummy_layers_json[source_node.name] = {'type' : source_node.type, 'input_shape' : input_shape, 'output_shape' : output_shape}
        if input_shape == output_shape:
            input_node_name = self.get_parent(source_node.name, [0]).name
            logger.debug("Node: %s, op: %s", source_node.name, source_node.type)
            IR_node = self._convert_identity_operation(source_node, end_idx = 1 ,new_op="Reshape")
            if not source_node.get_attr('data_format'):
                if len(output_shape) == 4:
                    kwargs['data_format'] = "NHWC"
                elif len(output_shape) == 3:
                    kwargs['data_format'] = "NWC"
                elif len(output_shape) == 2:
                    kwargs['data_format'] = "NC"
            assign_IRnode_values(IR_node, kwargs)
            return
            #return  self.rename_Identity(source_node)
        if self._compare(input_shape, output_shape) == 1: 
            padding = np.subtract(output_shape , input_shape).tolist() + [0]*( 8 - len(input_shape))
            if padding[3] > 0 or padding[7] > 0:
                # hacking using permute and pad to do channel pad; need to add channel padding to the stack -- moved to emitter
                logger.warn(" ... channel padding")
            IR_node = self._convert_identity_operation(source_node, new_op='Pad')
            logger.debug("adding pad layer with pad_value: " + str(padding))
            kwargs['pads'] = padding
            assign_IRnode_values(IR_node, kwargs)
            return 
        elif self._compare(input_shape, output_shape) == -1:
            IR_node = self._convert_identity_operation(source_node, new_op='Slice')
            logger.debug("adding crop layer as dummy")
            mask_length = len(self.src_graph.get_parent(source_node.name,[0]).get_attr('_output_shapes')[0].dim)
            kwargs['starts'] = [0]*(mask_length)
            kwargs['shape'] = self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0]
            kwargs['strides'] = [1]*(mask_length) 
            assign_IRnode_values(IR_node, kwargs)
            return
        else : 
            logger.info("Flatten logic : Inp Shape : %s , Output Shape : %s", len(input_shape), len(output_shape))
            self.rename_Identity(source_node)
            return
        '''
    output_layers = []
    output_shapes = []
    output_count = 0
    base_name = source_node.name
    IR_node = self._convert_identity_operation(source_node, new_op="Aten")
    op_params = self.__stringify_attrs(source_node.layer.attr)
    kwargs = {}
    if source_node.type in ['Split']:
      for idx, i in enumerate(source_node.out_edges):
        for jdx, j in enumerate(self.src_graph.layer_map[i].in_nodes):
          if j == source_node:
            out_edge_name = self.src_graph.layer_map[i].in_edges[jdx]
            out_edge_name = source_node.real_name + '_output' + out_edge_name.split(':')[1]
            output_layers.insert(0, out_edge_name)
            output_shapes.insert(0, self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[idx])
            output_count += 1
    elif source_node.type in ['SplitV']:
      for idx, i in enumerate(source_node.out_edges):
        for jdx, j in enumerate(self.src_graph.layer_map[i].in_nodes):
          if j == source_node:
            tmp_name = self.src_graph.layer_map[i].in_edges[jdx]
            out_edge_name = source_node.real_name + '_out'
            if len(str(tmp_name).split(':')) == 1:
              out_edge_name = out_edge_name + '0'
            else:
              out_edge_name = out_edge_name + str(tmp_name).split(':')[1]
            output_layers.insert(0, out_edge_name)
            output_shapes.insert(0, self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[idx])
            output_count += 1
    else:
      output_layers.append(source_node.real_name)
      output_shapes.insert(0, self.tensor_shape_to_list(source_node.get_attr('_output_shapes'))[0])
      output_count += 1
    output_shapes_cfirst = [clast2first(i) for i in output_shapes]
    if source_node.type in ['ArgMin']:
      output_shapes_cfirst[0] = output_shapes_cfirst[0] + [1]
    if source_node.type in ['ArgMax']:
      if len(output_shapes_cfirst[0]) >= 2:
        output_shapes_cfirst[0] = output_shapes_cfirst[0][:1] + output_shapes_cfirst[2:]
    kwargs["out_layers"] = output_layers
    kwargs["out_count"] = output_count
    shape_str = ';'.join(','.join(map(str, i)) for i in output_shapes_cfirst)
    kwargs["out_shape"] = shape_str
    kwargs["op_params"] = op_params
    kwargs["original_layer_type"] = source_node.type
    assign_IRnode_values(IR_node, kwargs)
    if len(output_layers) > 1:
      for idx, node in enumerate(output_layers):
        node_name = node if isinstance(node, str) else node.decode()
        IR_node1 = self._convert_identity_operation(source_node, new_op="Aten")
        IR_node1.name = node_name
        IR_node1.attr["ignore_node"].b = True

  def rename_Conv2D(self, source_node):
    if not source_node.variable_name in self.dummy_layers:
      self.rename_dvnet_Conv2D(source_node)
      return
    IR_node = self._convert_identity_operation(source_node, end_idx=1, new_op='Tf_Conv2D')
    kwargs = {}
    kwargs['strides'] = source_node.get_attr('strides')
    kwargs['dilations'] = source_node.get_attr('dilations') or [1] * len(kwargs['strides'])
    kwargs['padding'] = source_node.get_attr('padding')
    input_node = self.src_graph.get_parent(source_node.name, [0])
    kwargs['shape'] = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    # weights
    input_node_weight = self.src_graph.get_parent(source_node.name, [1])
    tensor_content = self.check_const(input_node_weight).get_attr('value')
    W = tensor_util.MakeNdarray(tensor_content)
    kwargs['kernel_shape'] = self.tensor_shape_to_list(input_node_weight.get_attr('_output_shapes'))[0]
    self.set_weight(source_node.name, 'weights', W)
    self._convert_padding(source_node, IR_node, kwargs['kernel_shape'][:-2])
    assign_IRnode_values(IR_node, kwargs)
    self._get_bias(source_node, IR_node)

  def rename_Conv3D(self, source_node):
    if not source_node.variable_name in self.dummy_layers:
      self.rename_dvnet_Conv3D(source_node)
      return
    raise NotImplementedError("Conv3D is in dummy_layer. Tf_Conv3D not implemented")

  def rename_BatchMatMulV2(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="BatchDot")
    kwargs = {}
    kwargs['transpose_a'] = str(not (str2bool(source_node.get_attr('adj_a', "False"))))
    kwargs['transpose_b'] = str(not (str2bool(source_node.get_attr('adj_b', "False"))))
    kwargs['data_format'] = "NHWC"
    assign_IRnode_values(IR_node, kwargs)

  def rename_LessEqual(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Lessequal')

  def rename_Erfc(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Erfc')

  def rename_Mish(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Mish')
    kwargs = dict()
    kwargs["alpha"] = float(source_node.get_attr("alpha").float_val[0])
    kwargs["beta"] = float(source_node.get_attr("beta").float_val[0])
    kwargs["gamma"] = float(source_node.get_attr("gamma").float_val[0])
    kwargs["delta"] = float(source_node.get_attr("delta").float_val[0])
    assign_IRnode_values(IR_node, kwargs)

  def rename_Yolov4_IR1(self, source_node):
    input_node = self.get_parent(source_node.name, [0])
    input_shape = self.tensor_shape_to_list(input_node.get_attr('_output_shapes'))[0]
    output_shape = []
    output_shape.append(input_shape[0])
    output_shape.append(input_shape[1] * 2)
    output_shape.append(input_shape[2] * 2)
    output_shape.append(input_shape[3])
    kwargs = dict()
    IR_node = self._convert_identity_operation(source_node, new_op='ResizeNearestNeighbor')
    kwargs['shape'] = input_shape
    kwargs['shape_out'] = output_shape
    IR_node.attr["_output_shapes"].Clear()
    kwargs["_output_shapes"] = [list_to_shape(output_shape)]
    assign_IRnode_values(IR_node, kwargs)

  def rename_Relu6_Scaled(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Relu6')

  def rename_HardSwish(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='HardSwish')

  def rename_Swish(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Swish')
