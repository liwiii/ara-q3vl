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

import os
import json
import numpy as np
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.utils import *
from mmdnn.conversion.common.DataStructure.parser import Parser
from mmdnn.conversion.pytorch.pytorch_graph import PytorchGraph
import torch
from torch import nn
import sys
from collections import defaultdict
from dv_logging import get_logger
logger = get_logger(__name__, 'DVGRPH')


class PytorchParser(Parser):
  _layer_map = {
      'onnx::Conv': 'Conv',
      'onnx::Flatten': 'Flatten',
      'onnx::Gemm': 'FullyConnected',
      'onnx::MaxPool': 'Maxpool',
      'onnx::AveragePool': 'Avgpool',
      'onnx::Dropout': 'Dropout',
      'onnx::BatchNormalization': 'BatchNormalization',
      'onnx::Add': 'Add',
      'onnx::Concat': 'Concat',
      'onnx::Relu': 'Relu',
      'onnx::Tanh': 'Tanh',
      'onnx::Sigmoid': 'Sigmoid',
      'onnx::Mul': 'Mul',
      'onnx::Constant': 'Constant',
      'onnx::Clip': 'Clip',
      'onnx::GlobalAveragePool': 'GlobalAvgpool',
      'onnx::Shape': 'Shape',
      'onnx::Reshape': 'Reshape',
      'onnx::Unsqueeze': 'Unsqueeze',
      'onnx::Pad': 'Pad',
      'onnx::Resize': 'Resize',
      'onnx::MatMul': 'MatMul',
      'dv::L2Normalize': 'L2Norm',
      'onnx::ConvTranspose': 'ConvTranspose',
      'onnx::InstanceNormalization': 'InstanceNormalization',
      'onnx::Gather': 'gather',
  }
  layer_map = defaultdict(str)
  for k, v in _layer_map.items():
    layer_map[k] = v
  # TODO
  # 'max_pool2d': convert_maxpool,
  # 'onnx::Mul': convert_elementwise_mul,
  # 'onnx::Sub': convert_elementwise_sub,
  # 'onnx::ConvTranspose': convert_convtranspose,
  # 'onnx::LeakyRelu': convert_lrelu,
  # 'onnx::Sigmoid': convert_sigmoid,
  # 'onnx::Softmax': convert_softmax,
  # 'onnx::Selu': convert_selu,
  # 'onnx::Transpose': convert_transpose,
  # 'onnx::Reshape': convert_reshape,
  # 'onnx::MatMul': convert_matmul,
  # 'onnx::Gather': convert_gather,
  # 'onnx::ReduceSum': convert_reduce_sum,
  # 'onnx::Constant': convert_constant,
  # 'onnx::Upsample': convert_upsample,
  # 'onnx::Pad': convert_padding,
  ############
  # property #
  ############
  @property
  def src_graph(self):
    return self.pytorch_graph

  ####################
  # Public Functions #
  ####################
  def __init__(self, model_class_absolute, model_state_dict, input_shape, debug):
    from utils.import_helpers import get_cls_and_pkg
    super(PytorchParser, self).__init__()
    if not os.path.exists(model_state_dict):
      logger.error("Pytorch model file [{}] is not found.".format(model_state_dict))
      assert False
    # test
    # cpu: https://github.com/pytorch/pytorch/issues/5286
    model_class = get_cls_and_pkg(model_class_absolute).cls
    model = model_class()
    model.load_state_dict(torch.load(model_state_dict, map_location='cpu'))
    self.weight_loaded = True
    self.debug = debug
    from pprint import pprint as pp
    # Build network graph
    self.pytorch_graph = PytorchGraph(model)
    self.input_shape = tuple(input_shape)
    self.pytorch_graph.build(self.input_shape)
    self.state_dict = self.pytorch_graph.state_dict
    self.shape_dict = self.pytorch_graph.shape_dict
    self.pad_value_nodes = set()

  def _get_from_runtime_params(self, gnode, attr_key):
    #args_idx_mapping = self.pytorch_graph.layer_args_order_map
    #args_idx_mapping = args_idx_mapping[gnode.name]
    # if attr_key in ['conv2.0.layers.0.conv1.weight']:
    #     import pdb; pdb.set_trace()
    if attr_key not in self.state_dict:
      for v in gnode.runtime_params:
        if v.endswith(attr_key):
          return v
    return attr_key

  def gen_IR(self):
    for layer in self.src_graph.topological_sort:
      current_node = self.src_graph.get_node(layer)
      onnx_node_type = current_node.type
      node_type = PytorchParser.layer_map[onnx_node_type]
      if hasattr(self, "rename_" + node_type):
        func = getattr(self, "rename_" + node_type)
        func(current_node)
      else:
        self.rename_UNKNOWN(current_node)
    self.gen_Input()

  def _set_output_shape(self, source_node, IR_node, preserve_dims=False):
    shape = graph_pb2.TensorShape()
    layer_name = source_node.name
    shape_pytorch = self.shape_dict[layer_name]
    new_dim = shape.dim.add()
    # (batch, C, H, W)  & NHWC
    if len(shape_pytorch) == 4:
      if shape_pytorch[0] == 1:
        new_dim.size = shape_pytorch[0]
      else:
        new_dim.size = shape_pytorch[0]
      for index in [2, 3, 1]:
        new_dim = shape.dim.add()
        dim = shape_pytorch[index]
        new_dim.size = dim if dim else -1
    elif len(shape_pytorch) == 2 and (not preserve_dims):
      if shape_pytorch[0] == 1:
        new_dim.size = shape_pytorch[0]
      else:
        new_dim.size = shape_pytorch[0]
      for _ in range(2):
        new_dim = shape.dim.add()
        new_dim.size = 1
      new_dim = shape.dim.add()
      dim = shape_pytorch[1]
      new_dim.size = dim if dim else -1
    elif len(shape_pytorch) == 2 and preserve_dims:
      if shape_pytorch[0] == 1:
        new_dim.size = -1
      new_dim = shape.dim.add()
      dim = shape_pytorch[1]
      new_dim.size = dim if dim else -1
    IR_node.attr["_output_shapes"].list.shape.extend([shape])

  ##########
  # Layers #
  ##########
  def rename_UNKNOWN(self, source_node):
    logger.error("PyTorch parser has not supported operator [%s] with name [%s]. " % (source_node.type, source_node.name))

  def gen_Input(self):
    IR_node = self.IR_graph.node.add()
    IR_node.name = 'input'
    IR_node.op = "DataInput"
    for node in self.IR_graph.node:
      if node.name in self.src_graph.input_layers:
        node.input.append('input')
    assert len(self.input_shape) == 4
    new_dim = IR_node.attr["shape"].shape.dim.add()
    if self.input_shape[0] == 1:
      new_dim.size = -1
    else:
      new_dim.size = self.input_shape[0]
    for index in [2, 3, 1]:
      new_dim = IR_node.attr["shape"].shape.dim.add()
      new_dim.size = self.input_shape[index]
    shape = graph_pb2.TensorShape()
    new_dim = shape.dim.add()
    shape_pytorch = self.input_shape
    if len(shape_pytorch) == 4:
      if shape_pytorch[0] == 1:
        new_dim.size = -1
      else:
        new_dim.size = shape_pytorch[0]
      for index in [2, 3, 1]:
        new_dim = shape.dim.add()
        dim = shape_pytorch[index]
        new_dim.size = dim if dim else -1
    elif len(shape_pytorch) == 2:
      if shape_pytorch[0] == 1:
        new_dim.size = -1
      else:
        new_dim.size = shape_pytorch[0]
      for _ in range(2):
        new_dim = shape.dim.add()
        new_dim.size = 1
      new_dim = shape.dim.add()
      dim = shape_pytorch[1]
      new_dim.size = dim if dim else -1
    IR_node.attr["_output_shapes"].list.shape.extend([shape])

  def rename_Pad(self, source_node):
    attrs = source_node.attrs
    IR_node = self._convert_identity_operation(source_node, new_op='Pad')
    kwargs = {}
    padding = attrs['padding_attrs']['value'].numpy()
    kwargs['mode'] = 'CONSTANT'
    kwargs['pads'] = padding
    kwargs['constant_values'] = attrs['value']
    assign_IRnode_values(IR_node, kwargs)

  def rename_Conv(self, source_node):
    import re
    attr = source_node.attrs
    kwargs = dict()
    # dilation
    if 'dilations' in attr:
      kwargs['dilations'] = [1] + attr['dilations'] + [1]
    else:
      kwargs['dilations'] = [1] + [1, 1] + [1]
    if len(attr['pads']) == 4:
      kwargs['pads'] = [0] + attr['pads'][0:2] + [0, 0] + attr['pads'][2:] + [0]
    elif len(attr['pads']) == 2:
      kwargs['pads'] = ([0] + attr['pads'][0:2] + [0]) * 2
    if 'strides' not in attr:
      kwargs['strides'] = [1] + [1, 1] + [1]
    else:
      kwargs['strides'] = [1] + attr['strides'] + [1]
    kwargs['group'] = attr['group']
    _st = source_node._st
    if torch.__version__ < '1.6.0':
      # if source_node.name in ['conv2.0.layers.0.conv1.weight']:
      bias_name = '{0}.bias'.format(source_node.weights_name)
      weights_name = '{0}.weight'.format(source_node.weights_name)
    else:
      bias_name = '{0}.bias'.format(re.search("%([0-9\.]*)\.bias", _st).group(1))
      weights_name = '{0}.weight'.format(re.search("%([0-9\.]*)\.weight", _st).group(1))
    weights_name = self._get_from_runtime_params(source_node, weights_name)
    bias_name = self._get_from_runtime_params(source_node, bias_name)
    # bias_name = '{0}.bias'.format(source_node.weights_name)
    # weights_name = '{0}.weight'.format(source_node.weights_name)
    # weights_name = self._get_from_runtime_params(source_node, weights_name)
    weight = self.state_dict[weights_name]
    weight = weight.numpy()
    dim = weight.ndim - 2
    IR_node = self._convert_identity_operation(source_node, new_op="Conv")
    weight = np.transpose(weight, list(range(2, dim + 2)) + [1, 0])
    self.set_weight(source_node.name, 'weights', weight)
    kwargs['kernel_shape'] = list(weight.shape)
    # handle bias
    if bias_name in self.state_dict:
      bias = self.state_dict[bias_name].numpy()
      self.set_weight(source_node.name, 'bias', bias)
      kwargs['use_bias'] = True
    else:
      kwargs['use_bias'] = False
    assign_IRnode_values(IR_node, kwargs)

  def rename_ConvTranspose(self, source_node):
    attr = source_node.attrs
    kwargs = dict()
    # dilation
    if 'dilations' in attr:
      kwargs['dilations'] = [1] + attr['dilations'] + [1]
    else:
      kwargs['dilations'] = [1] + [1, 1] + [1]
    if len(attr['pads']) == 4:
      kwargs['pads'] = [0] + attr['pads'][0:2] + [0, 0] + attr['pads'][2:] + [0]
    elif len(attr['pads']) == 2:
      kwargs['pads'] = ([0] + attr['pads'][0:2] + [0]) * 2
    if 'strides' not in attr:
      kwargs['strides'] = [1] + [1, 1] + [1]
    else:
      kwargs['strides'] = [1] + attr['strides'] + [1]
    kwargs['group'] = attr['group']
    weights_name = '{0}.weight'.format(source_node.weights_name)
    weights_name = self._get_from_runtime_params(source_node, weights_name)
    weight = self.state_dict[weights_name]
    weight = weight.numpy()
    dim = weight.ndim - 2
    IR_node = self._convert_identity_operation(source_node, new_op="Conv2DBackpropInput")
    weight = np.transpose(weight, list(range(2, dim + 2)) + [1, 0])
    self.set_weight(source_node.name, 'weights', weight)
    kwargs['kernel_shape'] = list(weight.shape)
    bias_name = '{0}.bias'.format(source_node.weights_name)
    if bias_name in self.state_dict:
      bias = self.state_dict[bias_name].numpy()
      self.set_weight(source_node.name, 'bias', bias)
      kwargs['use_bias'] = True
    else:
      kwargs['use_bias'] = False
    assign_IRnode_values(IR_node, kwargs)

  def rename_MatMul(self, source_node):
    i1_shape = self.src_graph.get_parent(source_node.name, [0])
    i1_shape = self.shape_dict[i1_shape.name]
    i2_shape = self.src_graph.get_parent(source_node.name, [1])
    if i2_shape is None:
      weights = source_node.runtime_params[0]
      weights = self.state_dict[weights]
      weights = weights.detach().numpy()
      logger.info("got weights for matmul - emitting innerproduct, dims: inp(%s), wt(%s)", i1_shape, weights.shape)
      IR_node = self._convert_identity_operation(source_node, new_op="FullyConnected")
      IR_node.attr['units'].i = weights.shape[1]
      self.set_weight(source_node.name, 'weights', weights)
    else:
      logger.warn("generic matmul not supported")

  def rename_BatchNormalization(self, source_node):
    # TODO
    # output_shape
    import re
    IR_node = self._convert_identity_operation(source_node, new_op="BatchNorm")
    attr = source_node.attrs
    # epsilon
    IR_node.attr['epsilon'].f = attr['epsilon']
    _st = source_node._st
    if torch.__version__ < '1.6.0':
      bias_name = '{0}.bias'.format(source_node.weights_name)
      weights_name = '{0}.weight'.format(source_node.weights_name)
      mean_name = '{0}.running_mean'.format(source_node.weights_name)
      var_name = '{0}.running_var'.format(source_node.weights_name)
    else:
      bias_name = '{0}.bias'.format(re.search("%([0-9\.]*)\.bias", _st).group(1))
      weights_name = '{0}.weight'.format(re.search("%([0-9\.]*)\.weight", _st).group(1))
      mean_name = '{0}.running_mean'.format(re.search("%([0-9\.]*)\.running_mean", _st).group(1))
      var_name = '{0}.running_var'.format(re.search("%([0-9\.]*)\.running_var", _st).group(1))
    weights_name = self._get_from_runtime_params(source_node, weights_name)
    var_name = self._get_from_runtime_params(source_node, var_name)
    mean_name = self._get_from_runtime_params(source_node, mean_name)
    bias_name = self._get_from_runtime_params(source_node, bias_name)
    if bias_name in self.state_dict:
      beta = self.state_dict[bias_name].numpy()
      IR_node.attr['bias'].b = True
    else:
      IR_node.attr['bias'].b = False
    if weights_name in self.state_dict:
      gamma = self.state_dict[weights_name].numpy()
      IR_node.attr['scale'].b = True
    else:
      IR_node.attr['scale'].b = False
    mean = self.state_dict[mean_name].numpy()
    variance = self.state_dict[var_name].numpy()
    if IR_node.attr['scale'].b:
      self.set_weight(source_node.name, "scale", gamma)
    if IR_node.attr['bias'].b:
      self.set_weight(source_node.name, "bias", beta)
    # mean
    self.set_weight(source_node.name, "mean", mean)
    # var
    self.set_weight(source_node.name, "var", variance)

  def rename_Clip(self, source_node):
    attrs = source_node.attrs
    if 'max' not in attrs:
      attrs['max'] = 6
    if 'min' not in attrs:
      attrs['min'] = 0
    IR_node = self._convert_identity_operation(source_node, new_op='clip')
    assign_IRnode_values(IR_node, {'a_max': attrs['max'], 'a_min': attrs['min']})

  def rename_Constant(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Constant")
    # python if else will treat ndarray of 0 as false
    tensor_value = source_node.attrs['value']
    if type(tensor_value) is not list:
      if type(tensor_value) is not np.ndarray:
        tensor_value = tensor_value.numpy()
      if tensor_value.shape == tuple():
        tensor_value = str(tensor_value) if not tensor_value else tensor_value
    assign_IRnode_values(IR_node, {'value': tensor_value})

  def rename_Shape(self, source_node):
    # get the bottom of this node and capture its's output size and
    # expose it as a DummyData
    IR_node = self._convert_identity_operation(source_node, new_op="DummyData")
    parent = self.src_graph.get_parent(source_node.name, [0])
    parent_output_shape = self.shape_dict[parent.name]

  def rename_Resize(self, source_node):
    resize_node = self.src_graph.get_parent(source_node.name, [3])
    resize_shape = self.src_graph\
                       .get_parent(source_node.name, [3])\
                       .attrs['value']
    scale_factors = self.src_graph\
                        .get_parent(source_node.name, [2])\
                        .attrs['value']
    rois = self.src_graph\
               .get_parent(source_node.name, [1])\
               .attrs['value']
    if len(scale_factors) > 0:
      raise Exception("Scale factors not supported for resize yet")
    logger.warn(('pytorch emitter does not support Resize yet, using the ' 'python_shims.PytorchInterpolate shim layer'))
    IR_node = self._convert_identity_operation(source_node, new_op="PythonLayer")
    kwargs = dict(class_name='PytorchInterpolate', layer_args=json.dumps(dict(resize_shape=list(resize_shape.tolist()))))
    assign_IRnode_values(IR_node, kwargs)

  def rename_Reshape(self, source_node):
    from copy import deepcopy
    my_input_shape = self.src_graph.get_parent(source_node.name, [0])
    my_input_shape = self.shape_dict[my_input_shape.name]
    my_output_shape = deepcopy(my_input_shape)
    for _idx, reshape_dim in enumerate(source_node.attrs['shape']):
      if reshape_dim == -1:
        continue
      my_output_shape[_idx] = reshape_dim
    self.shape_dict[source_node.name] = my_output_shape
    # do not remove from state dict and graph
    # as we will process later
    IR_node = self._convert_identity_operation(source_node, new_op="Reshape")
    self._set_output_shape(source_node, IR_node, True)
    return
    if len(my_input_shape) == 4 and len(shape_data_node.attrs['value']) == 2:
      IR_node = self._convert_identity_operation(source_node, new_op="Flatten")
    else:
      IR_node = self._convert_identity_operation(source_node, new_op="Reshape")

  def rename_Gather(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Gather")

  def rename_Unsqueeze(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Unsqueeze")

  def rename_Relu(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Relu")

  def rename_Tanh(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Tanh")

  def rename_Sigmoid(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Sigmoid")

  def rename_Mul(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Mul")

  def rename_Maxpool(self, source_node):
    attr = source_node.attrs
    kwargs = dict()
    kwargs['strides'] = [1] + attr['strides'] + [1]
    if 'dilations' not in attr:
      kwargs['dilations'] = [1] + [1, 1] + [1]
    else:
      kwargs['dilations'] = [1] + attr['dilations'] + [1]
    if 'pads' in attr:
      kwargs['pads'] = [0] + attr['pads'][0:2] + [0, 0] + attr['pads'][2:] + [0]
    else:
      kwargs['pads'] = [0] * 8
    kwargs['kernel_shape'] = [1] + attr['kernel_shape'] + [1]
    IR_node = self._convert_identity_operation(source_node, new_op="Pool")
    kwargs['pooling_type'] = 'MAX'
    assign_IRnode_values(IR_node, kwargs)

  def rename_GlobalAvgpool(self, source_node):
    attr = source_node.attrs
    kwargs = dict()
    kwargs['strides'] = [1] + attr['strides'] + [1]
    if 'dilations' not in attr:
      kwargs['dilations'] = [1] + [1, 1] + [1]
    else:
      kwargs['dilations'] = [1] + attr['dilations'] + [1]
    if 'pad' in attr:
      kwargs['pads'] = [0] + attr['pads'][0:2] + [0, 0] + attr['pads'][2:] + [0]
    else:
      kwargs['pads'] = [0] * 8
    parent = self.src_graph.get_parent(source_node.name, [0])
    parent_shape = self.shape_dict[parent.name]
    my_shape = self.shape_dict[source_node.name]
    kernel_shapes = []
    for par_ax, my_ax in zip(parent_shape, my_shape):
      if par_ax != my_ax:
        kernel_shapes.append(par_ax)
    kwargs['kernel_shape'] = [1] + kernel_shapes + [1]
    IR_node = self._convert_identity_operation(source_node, new_op="Pool")
    kwargs['pooling_type'] = 'AVG'
    assign_IRnode_values(IR_node, kwargs)
    IR_node.attr['global_pooling'].b = True

  def rename_Avgpool(self, source_node):
    attr = source_node.attrs
    kwargs = dict()
    kwargs['strides'] = [1] + attr['strides'] + [1]
    if 'dilations' not in attr:
      kwargs['dilations'] = [1] + [1, 1] + [1]
    else:
      kwargs['dilations'] = [1] + attr['dilations'] + [1]
    if 'pad' in attr:
      kwargs['pads'] = [0] + attr['pads'][0:2] + [0, 0] + attr['pads'][2:] + [0]
    else:
      kwargs['pads'] = [0] * 8
    kwargs['kernel_shape'] = [1] + attr['kernel_shape'] + [1]
    IR_node = self._convert_identity_operation(source_node, new_op="Pool")
    kwargs['pooling_type'] = 'AVG'
    assign_IRnode_values(IR_node, kwargs)

  def rename_Flatten(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Flatten")

  def rename_FullyConnected(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="FullyConnected")
    weights_name = '{0}.weight'.format(source_node.weights_name)
    weights_name = self._get_from_runtime_params(source_node, weights_name)
    bias_name = '{0}.bias'.format(source_node.weights_name)
    W = self.state_dict[weights_name].numpy().transpose()
    input_channels, output_channels = W.shape
    # Kit weight tranpose
    # weight: N x M -> C x H x W x M -> H x W x C x M -> N x M
    if self.weight_loaded:
      parent = self.src_graph.get_parent(source_node.name, [0])
      while parent.type == 'onnx::Flatten' or parent.type == 'onnx::Dropout':
        parent = self.src_graph.get_parent(parent.name, [0])
      if len(self.shape_dict[parent.name]) == 4:
        #
        original_shape = W.shape
        channel_first_list = self.shape_dict[parent.name][1:]
        dim = len(channel_first_list) + 1
        weight = W.reshape(channel_first_list + [original_shape[1]])
        assert dim > 2
        weight = weight.transpose(list(range(1, dim - 1)) + [0, dim - 1])
        W = weight.reshape(original_shape)
    # weights
    self.set_weight(source_node.name, 'weights', W)
    # use_bias
    if bias_name in self.state_dict:
      IR_node.attr['use_bias'].b = True
      bias = self.state_dict[bias_name].numpy()
      self.set_weight(source_node.name, 'bias', bias)
    else:
      IR_node.attr['use_bias'].b = False
    # units
    IR_node.attr['units'].i = output_channels
    self._set_output_shape(source_node, IR_node, True)

  def rename_Dropout(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Dropout')
    IR_node.attr['keep_prob'].f = source_node.attrs['ratio']

  def rename_Concat(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Concat')
    if source_node.attrs['axis'] == 1:
      IR_node.attr['axis'].i = len(self.shape_dict[source_node.name]) - 1
    else:
      IR_node.attr['axis'].i = source_node.attrs['axis']

  def rename_Add(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Add')

  def rename_MaxPool2d(self, source_node):
    self._convert_pooling(source_node)

  def rename_View(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Reshape')
    assign_IRnode_values(IR_node, {'shape': list(source_node.get_attr('new_sizes'))[1:]})

  def rename_Addmm(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='FullyConnected')
    kwargs = dict()
    # handle weight
    weight = source_node.get_attr('next_functions')[2][0].next_functions[0][0].variable.data.numpy()
    weight = np.transpose(weight)
    kwargs['units'] = weight.shape[1]
    self.set_weight(source_node.name, 'weights', weight)
    # handle bias
    if source_node.get_attr('next_functions')[0][0]:
      bias = source_node.get_attr('next_functions')[0][0].variable.data.numpy()
      kwargs['use_bias'] = True
      self.set_weight(source_node.name, 'bias', weight)
    assign_IRnode_values(IR_node, kwargs)

  def rename_L2Norm(self, source_node):
    """Currently we only support l2 norm where the input
        is a vector - i.e, only 1 dimension has more that
        one element. The eps value for l2 normalization is 
        also different from the default offered by caffe.
        """

    def _is_vector_like(shape):
      return len(list(filter(lambda x: x > 1, shape))) == 1

    parent = self.src_graph.get_parent(source_node.name, [0])
    parent_shape = self.shape_dict[parent.name]
    if _is_vector_like(parent_shape):
      IR_node = self._convert_identity_operation(source_node, new_op='L2Norm')
      assign_IRnode_values(IR_node, {})
    else:
      self.rename_UNKNOWN(source_node)

  def rename_InstanceNormalization(self, source_node):
    import re
    attr = source_node.attrs
    kwargs = dict()
    IR_node = self._convert_identity_operation(source_node, new_op='Mxnet_InstanceNorm')
    IR_node.attr['epsilon'].f = attr['epsilon']
    _st = source_node._st
    if torch.__version__ < '1.6.0':
      bias_name = '{0}.bias'.format(source_node.weights_name)
      weights_name = '{0}.weight'.format(source_node.weights_name)
    else:
      bias_name = '{0}.bias'.format(re.search("%([0-9\.]*)\.bias", _st).group(1))
      weights_name = '{0}.weight'.format(re.search("%([0-9\.]*)\.weight", _st).group(1))
    weights_name = self._get_from_runtime_params(source_node, weights_name)
    bias_name = self._get_from_runtime_params(source_node, bias_name)
    if bias_name in self.state_dict:
      beta = self.state_dict[bias_name].numpy()
      IR_node.attr['bias'].b = True
    else:
      IR_node.attr['bias'].b = False
    if weights_name in self.state_dict:
      gamma = self.state_dict[weights_name].numpy()
      IR_node.attr['scale'].b = True
    else:
      IR_node.attr['scale'].b = False
    if IR_node.attr['scale'].b:
      self.set_weight(source_node.name, "scale", gamma)
    if IR_node.attr['bias'].b:
      self.set_weight(source_node.name, "bias", beta)

  ####################
  # Helper Functions #
  ####################
  @staticmethod
  def _copy_and_reop(source_node, IR_node, new_op=None):
    if new_op == None: new_op = source_node.type
    IR_node.name = source_node.name
    IR_node.op = new_op

  def _convert_identity_operation(self, source_node, in_edge_count=None, new_op=None):
    IR_node = self.IR_graph.node.add()
    PytorchParser._copy_and_reop(source_node, IR_node, new_op)
    self.convert_inedge(source_node, IR_node, 0, in_edge_count)
    self._set_output_shape(source_node, IR_node, True)
    return IR_node

  def _convert_pooling(self, source_node):
    kwargs = dict()
    kwargs['strides'] = [1] + list(source_node.get_attr('stride')) + [1]
    kwargs['dilations'] = [1] + list(source_node.get_attr('dilation')) + [1]
    kwargs['pads'] = ([0] + list(source_node.get_attr('padding')) + [0]) * 2
    kwargs['kernel_shape'] = [1] + list(source_node.get_attr('kernel_size')) + [1]
    IR_node = self._convert_identity_operation(source_node, new_op="Pool")
    if source_node.name.startswith('Max'):
      kwargs['pooling_type'] = 'MAX'
    elif source_node.name.startswith('Avg'):
      kwargs['pooling_type'] = 'AVG'
    else:
      raise ValueError('Unknown pooling type')
    assign_IRnode_values(IR_node, kwargs)
