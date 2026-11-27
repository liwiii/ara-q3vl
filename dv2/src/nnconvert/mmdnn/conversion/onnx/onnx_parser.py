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
import math
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.utils import *
from mmdnn.conversion.common.DataStructure.parser import Parser
from mmdnn.conversion.onnx.onnx_graph import OnnxGraph, OnnxGraphNode
from onnx import numpy_helper
import torch
import onnx
from torch import nn
import copy
import mmdnn.conversion.onnx.onnx_graph_utils as netx
import sys
from collections import defaultdict
import pickle
from dv_logging import get_logger
import pandas as pd
from mmdnn.conversion.common.types import UPSP_TYPE


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
              .replace('.','_')\
              .replace(' ','_')
  return '{}'.format(name)


logger = get_logger(__name__, 'DVGRPH')


class OnnxParser(Parser):
  _layer_map = {
      "ATen_group_norm": "ATen_group_norm",
      "Abs": "Abs",
      "Add": "Sum",
      "Advertima_IR": "Advertima_IR",
      "AliasWithName": "Identity",
      "ArgMax": "ArgMax",
      "AveragePool": "Avgpool",
      "BBoxTransform": "BBoxTransform",
      "BBoxTransform_detectron2": "BBoxTransform_detectron2",
      "BBoxTransform_static_detectron2": "BBoxTransform_static_detectron2",
      "BatchNormalization": "BatchNormalization",
      "BatchPermutation": "BatchPermutation",
      "BoxWithNMSLimit": "BoxWithNMSLimit",
      "Cast": "Identity",
      "Centernet_IR1": "Centernet_IR1",
      "Centernet_IR4": "Centernet_IR4",
      "Clip": "Clip",
      "CollectRpnProposals": "CollectRpnProposals",
      "Concat": "Concat",
      "Constant": "Constant",
      "Conv": "Conv",
      "ConvTranspose": "ConvTranspose",
      "DVTensor": "DVTensor",
      "DetectionBoxOutput": "DetectionBoxOutput",
      "DistributeFpnProposals": "DistributeFpnProposals",
      "Div": "Div",
      "Dropout": "Dropout",
      "Einsum": "Einsum",
      "Elu": "Elu",
      "Equal": "Equal",
      "Expand": "Expand",
      "Flatten": "Flatten",
      "Floor": "Floor",
      "Focus": "Focus",
      "FullyConnected": "FullyConnected",
      "Gather": "Gather",
      "GatherElements": "GatherElements",
      "GatherND": "GatherND",
      "Gelu": "Gelu",
      "Gemm": "FullyConnected",
      "GenerateProposals": "GenerateProposals",
      "GlobalAveragePool": "GlobalAvgpool",
      "GlobalMaxPool": "GlobalMaxPool",
      "GreaterOrEqual": "GreaterOrEqual",
      "GridSample": "GridSample",
      "GroupNorm": "GroupNorm",
      "HardSigmoid": "HardSigmoid",
      "IBN": "IBN",
      "Identity": "Identity",
      "Input": "Input",
      "InstanceNormalization": "InstanceNormalization",
      "L1Norm": "L1Norm",
      "L2Norm": "L2Norm",
      "L2Norm_AC": "L2Norm_AC",
      "L2Norm_SSD": "L2Norm_SSD",
      'L2Norm_scale': 'L2Norm_scale',
      "LRN": "LRN",
      "LayerNorm": "LayerNorm",
      "LayerNormalization": "LayerNorm",
      "LeakyRelu": "LeakyRelu",
      "LpNormalization": "LpNormalization",
      "MatMul": "MatMul",
      "MaxPool": "Maxpool",
      "MaxRoiPool": "MaxRoiPool",
      "Min": "Min",
      "Mish": "Mish",
      "Mul": "Mul",
      "Neg": "Neg",
      "NoOp": "NoOp",
      "Not": "Not",
      "Pad": "Pad",
      "Pow": "Pow",
      "PytorchOnnxSoftmax": "PytorchOnnxSoftmax",
      "ReduceL2": "ReduceL2",
      "ReduceMax": "ReduceMax",
      "ReduceMean": "ReduceMean",
      "ReduceSum": "ReduceSum",
      "ReduceSum_To_ELTS2": "ReduceSum_To_ELTS2",
      "ReduceSum_To_ELTS3": "ReduceSum_To_ELTS3",
      "Relu": "Relu",
      "Reshape": "Reshape",
      "Resize": "Resize",
      "ResizeNearest": "ResizeNearest",
      "RoIAlign": "ROIAlign",
      "RoiAlign": "ROIAlign",
      "RoiAlignMax": "RoiAlignMax",
      "ScatterND": "ScatterND",
      "Shape": "Shape",
      "Sigmoid": "Sigmoid",
      "Slice": "Slice",
      "Softmax": "Softmax",
      "Softplus": "Softplus",
      "Split": "Split",
      "Sqrt": "Sqrt",
      "Squeeze": "Unsqueeze",
      "Sub": "Sub",
      "Sum": "Sum",
      "Swish": "Swish",
      "Tanh": "Tanh",
      "Tile": "Tile",
      "Transpose": "Transpose",
      "Unsqueeze": "Unsqueeze",
      "Upsample": "Upsample",
      "Where": "Where",
      "Rope": "Rope",
      "Sin": "Sin",
      "Cos": "Cos",
      "NonMaxSuppression": "NMS",
      "NMSPose": "NMSPose",
      "TopK": "topk",
      "NMSRtmdet": "NMSRtmdet",
      "QLinearConv": "QLinearConv",
      "QLinearAdd": "QLinearAdd",
      "QLinearMul": "QLinearMul",
      "QLinearSigmoid": "QLinearSigmoid",
      "QLinearConcat": "QLinearConcat",
      "QLinearSoftmax": "QLinearSoftmax",
      "DequantizeLinear": "DequantizeLinear",
      "QuantizeLinear": "QuantizeLinear",
      "QGemm": "QLinearFullyConnected",
      "QLinearGlobalAveragePool" : "QLinearGlobalAveragePool",
      "Yolov10nPost": "Yolov10nPost",
      "Yolov11Post": "Yolov11Post",
      "BroadcastAddMul": "BroadcastAddMul",
      "Logits": "Logits",
      "SpacialDist": "SpacialDist",
      "MeanRatio": "MeanRatio",
      "GlobalResponseNorm": "GlobalResponseNorm",
      "HardSwish": "HardSwish"
  }
  layer_type_map = defaultdict(str)
  for k, v in _layer_map.items():
    layer_type_map[k] = v
  ############
  # property #
  ############
  @property
  def src_graph(self):
    return self.onnx_graph

  ####################
  # Public Functions #
  ####################
  def __init__(self, model_path, all_inference_model, input_shape, input_node=None, output_node=None, dummy=None, debug=None, output_dir=None, onnx_backend="onnxruntime", images=None, data_format=None, dformat=None, dynamic_input=False, summary_csv=None, adjust_file=None):
    from utils.import_helpers import get_cls_and_pkg
    super(OnnxParser, self).__init__()
    self.weight_loaded = True
    from pprint import pprint as pp
    # Build network graph
    self.onnx_graph = OnnxGraph(model_path, all_inference_model, input_node, output_node, onnx_backend, images, data_format, dformat, adjust_file)
    self.input_shape = input_shape
    self.onnx_graph.build(self.input_shape, output_dir)
    self.state_dict = self.onnx_graph.state_dict
    self.shape_dict = self.onnx_graph.shape_dict
    self.pad_value_nodes = set()
    self.output_node = output_node
    self.onnx_model = onnx.load_model(model_path)
    self.input_node = input_node
    self.onnx_backend = onnx_backend
    input_all = [node.name for node in self.onnx_model.graph.input]
    input_initializer = [node.name for node in self.onnx_model.graph.initializer]
    net_feed_input = set(input_all) - set(input_initializer)
    if len(net_feed_input) > 1 and len(net_feed_input) != len(self.input_node):
      raise AssertionError("For multi-input model, provide correct input names")
    if len(self.input_node) == 1 and (self.input_node[0] == 'None' or self.input_node[0] is None):
      self.input_node = list(net_feed_input)
    # self.input_node = input_node if input_node is not ['None'] else [n.name for n in self.onnx_model.graph.input]
    self.dummy = dummy
    self.dummy_layers = set()
    self.debug = debug
    self.input_map = dict(zip(self.input_node, self.input_shape))
    self.summary_csv = summary_csv

  def _get_from_runtime_params(self, gnode, attr_key):
    if attr_key not in self.state_dict:
      for v in gnode.runtime_params:
        if v.endswith(attr_key):
          return v
    return attr_key

  @staticmethod
  def calculate_scale_offset(data_min, data_max, bits, needOffset):
    qmax = 2**(bits - 1) - 1
    qmin = -(2**(bits - 1)) + (1 if bits == 8 else 0)
    
    data_min = min(0.0, data_min)
    data_max = max(0.0, data_max)
    
    if data_min == 0.0 and data_max == 0.0:
      return 1.0, 0
      
    if not needOffset:
      abs_max = max(abs(data_max), abs(data_min))
      scale = (2 * abs_max) / (qmax - qmin - 1)
      return float(scale), 0

    scale = (data_max - data_min) / (qmax - qmin)
    zp_from_min = qmin - data_min / scale
    zp_from_max = qmax - data_max / scale
    zp_err_min = abs(qmin) + abs(data_min / scale)
    zp_err_max = abs(qmax) + abs(data_max / scale)
    zero_point = zp_from_min if zp_err_min < zp_err_max else zp_from_max
    
    nudged_zero_point = max(qmin, min(qmax, int(round(zero_point))))
    
    return float(scale), nudged_zero_point
  
  @staticmethod
  def _convert_axis(IR_node, axis):
    ndim = len(IR_node.attr['_output_shapes'].list.shape[0].dim)
    if axis < -1:
      axis = ndim + axis
    if axis == -1:
      axis = len(IR_node.attr['_output_shapes'].list.shape[0].dim) - 1
    if axis == 0:
      return 0
    elif axis == 1:
      return ndim - 1
    else:
      return axis - 1

  def _replicate_node(self, current_node, outputs):
    for idx, output in enumerate(current_node.layer.output):
      if output in outputs:
        IR_node = self._convert_identity_operation(current_node, new_op="LoopEnd")
        # IR_node.name = output
        if current_node.type in ['Split', 'DistributeFpnProposals', 'GenerateProposals', 'BoxWithNMSLimit', 'BBoxTransform', 'Onnx_BBoxTransform', 'TopK', 'NMSRtmdet', 'NMSPose']:  #, "RoIAlign"] :
          if idx != 0:
            IR_node.input[:] = [current_node.name + '_' + current_node.layer.output[idx]]
          else:
            IR_node.input[:] = [current_node.name]
        else:
          IR_node.input[:] = [current_node.name]
        IR_node.name = output + '_splitOutput'

  def _replicate_node_with_reshape(self, current_node, outputs):
    for idx, output in enumerate(current_node.layer.output):
      if output in outputs:
        IR_node = self._convert_identity_operation(current_node, new_op="Reshape")
        IR_node.name = output
        if current_node.type in ['Split', 'DistributeFpnProposals', 'GenerateProposals', 'BoxWithNMSLimit', 'BBoxTransform', 'Onnx_BBoxTransform', 'TopK', 'NMSRtmdet', 'NMSPose']:
          IR_node.input[:] = [current_node.name + '_' + current_node.layer.output[idx]]
          shape = self._cfirst_to_clast(self.shape_dict[current_node.layer.output[idx]])
        else:
          IR_node.input[:] = [current_node.name]
          shape = self._cfirst_to_clast(self.shape_dict[current_node.name])
        IR_node.name = IR_node.name
        IR_node.attr['_output_shapes'].list.shape.pop()
        self._set_output_shape_from_list(IR_node, shape)

  def gen_IR(self):
    # logger.info('emitter --- %s', (self.src_graph.topological_sort))
    self.dummy_layers_json = dict()
    df = pd.DataFrame(columns=['origin', 'op', 'issue', 'source'])
    if self.debug:
      with open(self.debug, 'r') as fd:
        _gen_dummy = json.load(fd)
      for dummy_layer in _gen_dummy:
        logger.info("cp layer is found from estimator json %s", dummy_layer)

        failed_layer_list = _gen_dummy[dummy_layer]
        for layer in failed_layer_list[:-1]:
          idx_ = layer.find('_ara')
          if idx_ != -1:
            layer = layer[:idx_]
          logger.info("adding cp / dummy layer to the list %s", layer)

          self.dummy_layers.add(layer)
    logger.debug('emitter --- %s', (self.src_graph.topological_sort))
    _flag = False
    for layer in self.src_graph.topological_sort:
      current_node = self.src_graph.get_node(layer)
      logger.debug("parsing layer %s type: %s", current_node.name, current_node.type)
      onnx_node_type = current_node.type
      node_type = OnnxParser.layer_type_map[onnx_node_type]
      replicate = False
      replicate_with_reshape = False
      rename_last_layer = False
      outputs = [mk_node_name(j) for j in self.output_node]

      if hasattr(current_node.layer, 'output') and len(current_node.layer.output) and current_node.name != current_node.layer.output[0]:
        for i in current_node.layer.output:
          if i in outputs:
            if current_node.layer.op_type == 'NonMaxSuppression':
              rename_last_layer = True
            else:
              replicate_with_reshape = True
            break
      elif hasattr(current_node.layer, 'output') and len(current_node.layer.output) and len(current_node.out_edges):
        for i in current_node.layer.output:
          if i in outputs:
            replicate = True
            break
      if onnx_node_type in ["ATen"]:
        onnx_node_type = 'ATen_' + current_node.attrs["operator"].decode("utf-8")
        node_type = OnnxParser.layer_type_map[onnx_node_type]
      if hasattr(self, "rename_" + node_type):
        func = getattr(self, "rename_" + node_type)
        try:
          old_nodes = len(self.IR_graph.node)
          func(current_node)
          if replicate_with_reshape:
            # if current_node.type == "DequantizeLinear":
            #   parent_node_name = current_node.in_edges[0]
            #   parent_node = self.src_graph.get_node(parent_node_name)
            #   self._replicate_node_with_reshape(parent_node, current_node.layer.input[0])
            # else:
            self._replicate_node_with_reshape(current_node, outputs)
          elif replicate:
            self._replicate_node(current_node, outputs)
          elif rename_last_layer:
            self.IR_graph.node[-1].name = current_node.layer.output[0]

        except Exception as e:
          removable_nodes = [i for i in range(old_nodes, len(self.IR_graph.node))]
          logger.exception("Exception occurred while generation IR for op:{%s}", node_type)
          logger.exception("Exception details: {%s}", str(e), exc_info=sys.exc_info())
          for i in removable_nodes:
            del self.IR_graph.node[i]
          if self.dummy and node_type not in ['Input']:
            logger.info("Replacing {%s} {%s} node with DV_ATen op", node_type, current_node.name)
            self.rename_DUMMY(current_node)
            if replicate_with_reshape:
              self._replicate_node_with_reshape(current_node, outputs)
            if replicate:
              self._replicate_node(current_node, outputs)
          row = {}
          row['op'] = current_node.name
          row['issue'] = str(e)
          row['source'] = current_node.layer
          row['origin'] = 'ONNX parser'
          df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
      else:
        try:
          if self.dummy and node_type not in ['Input']:
            logger.info("Replacing unknown {%s} {%s} with DV_ATen op", node_type, current_node.name)
            self.rename_DUMMY(current_node)
            row = {}
            row['op'] = current_node.name
            row['issue'] = "Unsupported op : '" + onnx_node_type + "' , replaced with DV_ATen"
            #row['issue'] = "Replacing with DV_ATen op as node type: '" + onnx_node_type + "' is not supported in ONNX parser"
            row['source'] = current_node.layer
            row['origin'] = 'ONNX parser'
            df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
            if replicate_with_reshape:
              self._replicate_node_with_reshape(current_node, outputs)
            if replicate:
              self._replicate_node(current_node, outputs)
          else:
            self.rename_UNKNOWN(current_node)
        except Exception as e:
          logger.exception("Exception occurred while generation IR for op:{%s}", onnx_node_type)
          logger.exception("Exception details: {%s}", str(e), exc_info=sys.exc_info())
          row = {}
          row['op'] = current_node.name
          row['issue'] = str(e)
          row['source'] = current_node.layer
          row['origin'] = 'ONNX parser'
          df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
    df.to_csv(self.summary_csv, index=False)
    # for idx, node in enumerate(self.onnx_model.graph.input):
    #     self.rename_Input(node)
    # self.gen_Input()

  def _repeatedscalar_to_list(self, data):
    return [int(d) for d in data]

  def _cfirst_to_clast(self, data):
    data = list(data)
    if len(data) == 1 or len(data) == 2:
      return data
    elif len(data) > 2:
      return [int(data[0])] + [int(k) for k in data[2:]] + [int(data[1])]
    else:
      return data

  def _clast_to_cfirst(self, data):
    data = list(data)
    if len(data) > 2:
      return data[0:1] + data[-1:] + data[1:-1]
    elif len(data) in [1, 2]:
      return data

  def _convert_onnx_axis(self, data, axis):  #added for argmax support for 3D tensor axis conversion
    NCDHW_NDHWC = {0: 0, 1: 4, 2: 1, 3: 2, 4: 3}
    NCHW_NHWC = {0: 0, 1: 3, 2: 1, 3: 2}
    NCW_NWC = {0: 0, 1: 2, 2: 1}
    if axis < 0:
      axis += len(data)
    if len(data) in [0, 1, 2]:
      return axis
    elif len(data) == 3:
      return NCW_NWC[axis]
    elif len(data) == 4:
      return NCHW_NHWC[axis]
    else:
      return NCDHW_NDHWC[axis]

  def _set_output_shape(self, source_node, IR_node, preserve_dims=False):
    shape = graph_pb2.TensorShape()
    layer_name = source_node.name
    shape_pytorch = [1, 1, 1, 1]
    if layer_name in self.shape_dict:
      shape_pytorch = self.shape_dict[layer_name]

    if len(shape_pytorch) == 6:
      new_dim = shape.dim.add()
      if shape_pytorch[0] == 1:
        new_dim.size = shape_pytorch[0]
      else:
        new_dim.size = shape_pytorch[0]
      for index in [2, 3, 4, 5, 1]:
        new_dim = shape.dim.add()
        dim = shape_pytorch[index]
        new_dim.size = dim if dim else -1
    elif len(shape_pytorch) == 5:
      new_dim = shape.dim.add()
      if shape_pytorch[0] == 1:
        new_dim.size = shape_pytorch[0]
      else:
        new_dim.size = shape_pytorch[0]
      for index in [2, 3, 4, 1]:
        new_dim = shape.dim.add()
        dim = shape_pytorch[index]
        new_dim.size = dim if dim else -1
    elif len(shape_pytorch) == 4:
      new_dim = shape.dim.add()
      if shape_pytorch[0] == 1:
        new_dim.size = shape_pytorch[0]
      else:
        new_dim.size = shape_pytorch[0]
      for index in [2, 3, 1]:
        new_dim = shape.dim.add()
        dim = shape_pytorch[index]
        new_dim.size = dim if dim else -1
    elif len(shape_pytorch) == 3:
      new_dim = shape.dim.add()
      new_dim.size = shape_pytorch[0]
      for index in [2, 1]:
        new_dim = shape.dim.add()
        dim = shape_pytorch[index]
        new_dim.size = dim if dim else -1
    elif len(shape_pytorch) == 2 and (preserve_dims == False):
      new_dim = shape.dim.add()
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
      new_dim = shape.dim.add()
      new_dim.size = shape_pytorch[0]
      new_dim = shape.dim.add()
      dim = shape_pytorch[1]
      new_dim.size = dim if dim else -1
    else:
      for _s in shape_pytorch:
        new_dim = shape.dim.add()
        new_dim.size = _s
    IR_node.attr["_output_shapes"].list.shape.extend([shape])
  
  def _set_output_shape_multi_out(self, source_node, IR_node, preserve_dims=False):
    for layer_name in source_node.layer.output:
      shape = graph_pb2.TensorShape()
      shape_pytorch = [1, 1, 1, 1]
      if layer_name in self.shape_dict:
        shape_pytorch = self.shape_dict[layer_name]

      if len(shape_pytorch) == 5:
        new_dim = shape.dim.add()
        if shape_pytorch[0] == 1:
          new_dim.size = shape_pytorch[0]
        else:
          new_dim.size = shape_pytorch[0]
        for index in [2, 3, 4, 1]:
          new_dim = shape.dim.add()
          dim = shape_pytorch[index]
          new_dim.size = dim if dim else -1
      elif len(shape_pytorch) == 4:
        new_dim = shape.dim.add()
        if shape_pytorch[0] == 1:
          new_dim.size = shape_pytorch[0]
        else:
          new_dim.size = shape_pytorch[0]
        for index in [2, 3, 1]:
          new_dim = shape.dim.add()
          dim = shape_pytorch[index]
          new_dim.size = dim if dim else -1
      elif len(shape_pytorch) == 3:
        new_dim = shape.dim.add()
        new_dim.size = shape_pytorch[0]
        for index in [2, 1]:
          new_dim = shape.dim.add()
          dim = shape_pytorch[index]
          new_dim.size = dim if dim else -1
      elif len(shape_pytorch) == 2 and (preserve_dims == False):
        new_dim = shape.dim.add()
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
        new_dim = shape.dim.add()
        new_dim.size = shape_pytorch[0]
        new_dim = shape.dim.add()
        dim = shape_pytorch[1]
        new_dim.size = dim if dim else -1
      else:
        for _s in shape_pytorch:
          new_dim = shape.dim.add()
          new_dim.size = _s
      IR_node.attr["_output_shapes"].list.shape.extend([shape])

  ##########
  # Layers #
  ##########
  def rename_UNKNOWN(self, source_node):
    raise NotImplementedError("ONNX parser has unsupported operator [%s] with name [%s]." % (source_node.type, source_node.name))

  def rename_NoOp(self, source_node):
    return

  def rename_DUMMY(self, source_node):
    output_shapes = []
    output_count = 0
    base_name = source_node.name
    IR_node = self._convert_identity_operation(source_node, in_edge_count=len(source_node.in_edges), new_op="Aten")
    op_params = self.__stringify_attrs(source_node.attrs)
    shape_str = ';'.join(','.join(map(str, i)) for i in output_shapes)
    kwargs = {}
    if source_node.type == 'Dropout':
      __base_name = source_node.name
      node_name = __base_name if isinstance(__base_name, str) else __base_name.decode()
      output_shapes.append(self.shape_dict[source_node.name])
      output_count += 1
    else:
      for idx, node in enumerate(source_node.layer.output):
        __base_name = source_node.name if len(source_node.layer.output) < 2 else "{}_{}".format(base_name, node)
        node_name = __base_name if isinstance(__base_name, str) else __base_name.decode()
        output_shapes.append(self.shape_dict[node])
        output_count += 1
    kwargs["out_count"] = output_count
    shape_str = ';'.join(','.join(map(str, i)) for i in output_shapes)
    kwargs["out_shape"] = shape_str
    kwargs["op_params"] = op_params
    kwargs["original_layer_type"] = source_node.type
    assign_IRnode_values(IR_node, kwargs)
    if len(source_node.layer.output) > 1:
      for idx, node in enumerate(source_node.layer.output):
        node_name = node if isinstance(node, str) else node.decode()
        IR_node1 = self._convert_identity_operation(source_node, in_edge_count=len(source_node.in_edges), new_op="Aten")
        # if(len(source_node.layer.output)>0):
        __base_name = "{}_{}".format(base_name, node_name)
        IR_node1.name = __base_name
        IR_node1.attr["ignore_node"].b = True

  def gen_Input(self):
    for node in self.IR_graph.node:
      if node.name in self.src_graph.inp_layers:
        if 'input_node_' + node.name in self.shape_dict:
          node.input.append('input_' + node.name)
        else:
          node.input.append('input')  # handle multi-input here later.
    for i in self.src_graph.inp_layers:
      IR_node = self.IR_graph.node.add()
      IR_node.op = "DataInput"
      if 'input_node_' + i in self.shape_dict:
        IR_node.name = 'input_' + i
        self.input_shape = self.shape_dict['input_node_' + i]
      else:
        IR_node.name = 'input'  # handle multi-input here later.
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

  def rename_Input(self, source_node):
    IR_node = self.IR_graph.node.add()
    IR_node.op = "DataInput"
    IR_node.name = mk_node_name(source_node.name)
    shape = self.input_map[source_node.name]
    shape = self._cfirst_to_clast(shape)
    self._set_output_shape_from_list(IR_node, shape)
    kwargs = {}
    children_nodes = source_node.out_edges
    if len(children_nodes) == 1 and self.src_graph.get_node(children_nodes[0]).type == 'QuantizeLinear':
      child = self.src_graph.get_node(children_nodes[0])
      kwargs['output_scale'] = self.state_dict[child.layer.input[1]]
      kwargs['output_zero_point'] = self.state_dict[child.layer.input[2]]
      IR_node.op = "tflite_DataInput"
    self.convert_inedge(source_node, IR_node, 0, None)
    if len(shape) == 5:
      kwargs['data_format'] = 'NCDHW'
      assign_IRnode_values(IR_node, kwargs)
    elif len(shape) == 4:
      kwargs['data_format'] = 'NCHW'
      assign_IRnode_values(IR_node, kwargs)
    elif len(shape) == 3:
      kwargs['data_format'] = 'NCW'
      assign_IRnode_values(IR_node, kwargs)

  def rename_Pad(self, source_node):
    if source_node.weights_name == '':
      for i in source_node.layer.attribute:
        if i.name == 'pads':
          padding = i.ints                                                                                     
    else:
      padding = copy.deepcopy(self.state_dict[source_node.weights_name + '_1'])    
    if (sum(padding) == 0):
      source_node.real_name = self.src_graph.get_node(source_node.in_edges[0]).real_name
      return
    IR_node = self._convert_identity_operation(source_node, new_op='Pad')
    kwargs = {}
    # padding = copy.deepcopy(self.state_dict[source_node.weights_name + '_1'])
    pad_l = self._cfirst_to_clast(padding[0:4])
    pad_r = self._cfirst_to_clast(padding[4:8])
    padding = pad_l + pad_r
    kwargs['mode'] = 'CONSTANT'
    kwargs['pads'] = padding
    kwargs['constant_values'] = 0
    assign_IRnode_values(IR_node, kwargs)

  def rename_ConvTranspose(self, source_node):
    attr = source_node.attrs
    kwargs = dict()
    # dilation
    if 'dilations' in attr:
      dilations = [int(i) for i in attr['dilations']]
      #if dilations != [1, 1]:
      #    dilations = [1, 1]
      kwargs['dilations'] = [1] + dilations + [1]
    else:
      kwargs['dilations'] = [1] + [1, 1] + [1]
    if len(attr['pads']) == 4:
      kwargs['pads'] = [0] + attr['pads'][0:2] + [0, 0] + attr['pads'][2:] + [0]
    elif len(attr['pads']) == 2:
      kwargs['pads'] = ([0] + attr['pads'][0:2] + [0]) * 2
    elif len(attr['pads']) == 0:
      kwargs['pads'] = [0, 0, 0, 0, 0, 0, 0, 0]
    if 'strides' not in attr:
      kwargs['strides'] = [1] + [1, 1] + [1]
    else:
      kwargs['strides'] = [1] + [int(i) for i in attr['strides']] + [1]
    #default group=1
    #https://github.com/onnx/onnx/blob/main/docs/Operators.md#attributes-17
    kwargs['group'] = attr.get('group', 1)
    weights_name = source_node.weights_name + '_1'
    bias_name = source_node.weights_name + '_2'
    weight = self.state_dict[weights_name]
    dim = weight.ndim - 2
    IR_node = self._convert_identity_operation(source_node, new_op="Conv2DBackpropInput")
    weight = np.transpose(weight, list(range(2, dim + 2)) + [1, 0])
    self.set_weight(source_node.name, 'weights', weight)
    kwargs['kernel_shape'] = list(weight.shape)
    kwargs['num_outputs'] = weight.shape[-2]*kwargs['group']
    # handle bias
    if bias_name in self.state_dict:
      logger.info("Bias is found in state dict %s", bias_name)
      bias = self.state_dict[bias_name]
      self.set_weight(source_node.name, 'bias', bias)
      kwargs['use_bias'] = True
    else:
      kwargs['use_bias'] = False
    assign_IRnode_values(IR_node, kwargs)

  def rename_Onnx_Conv(self, source_node):
    kwargs = dict()
    weights_name = source_node.weights_name + '_1'
    weight = self.state_dict[weights_name]
    self.set_weight(source_node.name, 'weights', weight)
    bias_name = source_node.weights_name + '_2'
    kwargs['no_bias'] = True
    if bias_name in self.state_dict:
      logger.info("Bias is found in state dict %s", bias_name)
      bias = self.state_dict[bias_name]
      self.set_weight(source_node.name, 'bias', bias)
      kwargs['no_bias'] = False
    if 'dilations' in source_node.attrs:
      if len(source_node.attrs['dilations']) == 1:
        dilation_list = source_node.attrs['dilations']
        kwargs['dilations'] = self._repeatedscalar_to_list(dilation_list) + [1]

      else:
        kwargs['dilations'] = self._repeatedscalar_to_list(source_node.attrs['dilations'])
    else:
      kwargs['dilations'] = [1, 1]
    if 'group' in source_node.attrs:
      kwargs['group'] = source_node.attrs['group']
    else:
      kwargs['group'] = 1
    if 'kernel_shape' in source_node.attrs:
      kwargs['kernel_shape'] = self._repeatedscalar_to_list(source_node.attrs['kernel_shape'])
      k_shape = kwargs['kernel_shape']
      kwargs['kernel_shape'] = [weight.shape[0], weight.shape[1]] + k_shape
    else:
      if len(weight.shape) > 4:
        raise NotImplementedError('Only Conv2d is supported')
      kwargs['kernel_shape'] = weight.shape
    if len(kwargs['kernel_shape']) == 2:
      k_shape = kwargs['kernel_shape']
      kwargs['kernel_shape'] = [weight.shape[0], weight.shape[1]] + k_shape
    elif len(kwargs['kernel_shape']) == 1:
      k_shape = kwargs['kernel_shape'] + [1]
      kwargs['kernel_shape'] = [weight.shape[0], weight.shape[1]] + k_shape
    if 'pads' in source_node.attrs:
      kwargs['pads'] = self._repeatedscalar_to_list(source_node.attrs['pads'])
      if len(kwargs['pads']) == 2:
        kwargs['pads'] = kwargs['pads'] * 2
    if 'strides' in source_node.attrs:
      stride_list = source_node.attrs['strides']
      if len(stride_list) == 1:
        stride_list.append(1)
      kwargs['strides'] = self._repeatedscalar_to_list(source_node.attrs['strides'])
    else:
      kwargs['strides'] = [1, 1]
    if 'auto_pad' in source_node.attrs:
      kwargs['auto_pad'] = source_node.attrs['auto_pad']
    IR_node = self._convert_identity_operation(source_node, new_op="Onnx_Conv")
    assign_IRnode_values(IR_node, kwargs)
    return IR_node

  def rename_Conv(self, source_node):
    if source_node.name in self.dummy_layers:
      return self.rename_Onnx_Conv(source_node)
    attr = source_node.attrs
    kwargs = dict()
    weights_name = source_node.weights_name + '_1'
    weight = self.state_dict[weights_name]
    if 'kernel_shape' in source_node.attrs and len(attr['kernel_shape']) > 0:
      k_shape = self._repeatedscalar_to_list(attr['kernel_shape'])
    else:
      k_shape = list(weight.shape[2:])
    kwargs['kernel_shape'] = k_shape + [weight.shape[1]] + [weight.shape[0]]  #list(weight.shape)
    # dilation
    if 'dilations' in attr:
      dilations = self._repeatedscalar_to_list(attr['dilations'])
      kwargs['dilations'] = [1] + dilations + [1]
    else:
      kwargs['dilations'] = [1] + [1, 1] + [1]
    parent = self.src_graph.get_parent(source_node.name, [0])
    input_shape = self.shape_dict[parent.name]
    if 'pads' not in attr or len(attr['pads']) == 0:
      kwargs['pads'] = [0, 0] * (len(k_shape) + 2)
    elif len(attr['pads']) == 4:
      kwargs['pads'] = [0] + self._repeatedscalar_to_list(attr['pads'][0:2]) + [0, 0] + self._repeatedscalar_to_list(attr['pads'][2:]) + [0]
    elif len(attr['pads']) == 2 and len(input_shape) == 3:
      kwargs['pads'] = [0] + self._repeatedscalar_to_list(attr['pads'][0:1]) + [0, 0] + self._repeatedscalar_to_list(attr['pads'][1:]) + [0]
    elif len(attr['pads']) == 2:
      kwargs['pads'] = ([0] + self._repeatedscalar_to_list(attr['pads'][0:2]) + [0]) * 2
    elif len(attr['pads']) == 6:
      kwargs['pads'] = [0] + self._repeatedscalar_to_list(attr['pads'][0:3]) + [0, 0] + self._repeatedscalar_to_list(attr['pads'][3:]) + [0]
    if 'strides' not in attr:
      kwargs['strides'] = [1] + [1, 1] + [1]
    else:
      kwargs['strides'] = [1] + self._repeatedscalar_to_list(attr['strides']) + [1]
    kwargs['group'] = attr.get('group', 1)
    bias_name = source_node.weights_name + '_2'
    dim = len(weight.shape) - 2
    IR_node = self._convert_identity_operation(source_node, new_op="Conv")
    weight = np.transpose(weight, list(range(2, dim + 2)) + [1, 0])
    layout = None
    if dim == 1:
      layout = "NCW"
    elif dim == 2:
      layout = "NCHW"
    elif dim == 3:
      layout = "NCDHW"
    self.set_weight(source_node.name, 'weights', weight)
    parent = self.src_graph.get_parent(source_node.name, [0])
    input_shape = self.shape_dict[parent.name]
    auto_pad = attr.get('auto_pad', "NOTSET".encode("utf-8"))
    kwargs['auto_pad'] = auto_pad
    if auto_pad.decode("utf-8").startswith("SAME"):
      padding = compute_tf_same_padding(self._cfirst_to_clast(input_shape), kwargs['kernel_shape'], kwargs['strides'])  #,data_format=layout)
      kwargs["pads"] = padding
      kwargs["auto_pad"] = "SAME_LOWER".encode("utf-8")
      # elif attr['auto_pad'].decode("utf-8") == "VALID":
      #     dims = len(input_shape)
      #     assign_IRnode_values(IR_node, {'auto_pad' : "VALID", 'pads' : [0, 0] * dims})
    # handle bias
    if bias_name in self.state_dict:
      logger.info("Bias is found in state dict %s", bias_name)
      bias = self.state_dict[bias_name]
      self.set_weight(source_node.name, 'bias', bias)
      kwargs['use_bias'] = True
      kwargs['no_bias'] = False
    else:
      kwargs['use_bias'] = False
      kwargs['no_bias'] = True
    kwargs['data_format'] = layout
    assign_IRnode_values(IR_node, kwargs)

  def rename_GridSample(self, source_node):
    kwargs = dict()
    
    kwargs['align_corners'] = source_node.attrs.get('align_corners', 0)
    kwargs['mode'] = source_node.attrs.get('mode', 'bilinear')
    kwargs['padding_mode'] = source_node.attrs.get('padding_mode', 'zeros')

    IR_node = self._convert_identity_operation(source_node, new_op="Onnx_GridSample")
    assign_IRnode_values(IR_node, kwargs)

  def rename_QuantizeLinear(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="DVCast")
    kwargs = dict()
    kwargs['scale'] = self.state_dict[source_node.layer.input[1]]
    kwargs['offset'] = self.state_dict[source_node.layer.input[2]]
    kwargs['data_fmt'] = 'int8' #str(self.state_dict[source_node.layer.input[2]].dtype)
    kwargs['src_precision'] = 'fp32'
    kwargs['dst_precision'] = kwargs['data_fmt']
    kwargs['is_pqn'] = True
    assign_IRnode_values(IR_node, kwargs)

  def rename_DequantizeLinear(self, source_node):
    # Check if the input is a constant weight, then map to DVTensor -> DVCast, else DVCast
    IR_node = self._convert_identity_operation(source_node, new_op="DVCast")
    kwargs = dict()
    kwargs['scale'] = self.state_dict[source_node.layer.input[1]]
    kwargs['offset'] = self.state_dict[source_node.layer.input[2]]
    kwargs['data_fmt'] = 'int8' #str(self.state_dict[source_node.layer.input[2]].dtype)
    kwargs['src_precision'] = kwargs['data_fmt']
    kwargs['dst_precision'] = 'fp32'
    kwargs['is_pqn'] = True
    assign_IRnode_values(IR_node, kwargs)
    if source_node.layer.input[0] in source_node.initializers:
      IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_0.input):
        IR_node_0.input.pop()
      IR_node_0.name = 'node_' + source_node.layer.input[0]
      IR_node_0.attr['pass_through'].b = True
      weights = np.asarray(self.state_dict[source_node.layer.input[0]], dtype=np.int16) - 128
      self.set_weight(IR_node_0.name, 'weights', weights)
      IR_node.input.insert(0, IR_node_0.name)

  def rename_QLinearConv(self, source_node):
    attr = source_node.attrs
    kwargs = dict()
    kwargs['is_tflite'] = True
    kwargs['auto_pad'] = attr['auto_pad']
    kwargs['dilations'] = [1] + self._repeatedscalar_to_list(attr['dilations']) + [1]
    kwargs['group'] = attr['group']
    kwargs['kernel_shape'] = self._repeatedscalar_to_list(attr['kernel_shape']) + [self.state_dict[source_node.layer.input[3]].shape[1], self.state_dict[source_node.layer.input[3]].shape[0]]
    kwargs['pads'] = self._repeatedscalar_to_list(attr['pads'])
    kwargs['strides'] = [1] + self._repeatedscalar_to_list(attr['strides']) + [1]
    IR_node = self._convert_identity_operation(source_node, new_op="onnxpqn_Conv")
    kwargs['input_scale'] = self.state_dict[source_node.layer.input[1]]
    kwargs['input_zero_point'] = self.state_dict[source_node.layer.input[2]]
    kwargs['output_scale'] = self.state_dict[source_node.layer.input[6]]
    kwargs['output_zero_point'] = self.state_dict[source_node.layer.input[7]]
    kwargs['data_fmt'] = 'int8' # str(self.state_dict[source_node.layer.input[7]].dtype)
    # weights
    self.set_weight(source_node.name, "weights", self.state_dict[source_node.layer.input[3]])  #.transpose(2, 3, 1, 0))
    # bias
    if(len(source_node.layer.input)<9):
      IR_node.attr["use_bias"].b = True
      self.set_weight(source_node.name, "bias", np.zeros_like(self.state_dict[source_node.layer.input[4]], dtype=np.int32))
    else:
      IR_node.attr["use_bias"].b = True
      self.set_weight(source_node.name, "bias", self.state_dict[source_node.layer.input[8]])
    # weight scales
    self.set_weight(source_node.name, "weight_scales", self.state_dict[source_node.layer.input[4]])
    # weight zero points
    self.set_weight(source_node.name, "weight_zero_points", self.state_dict[source_node.layer.input[5]])
    if len(source_node.layer.input) > 9:
      self.set_weight(source_node.name, "bias_scales", self.state_dict[source_node.layer.input[9]])
      self.set_weight(source_node.name, "bias_zero_points", self.state_dict[source_node.layer.input[10]])
    assign_IRnode_values(IR_node, kwargs)

  def rename_QLinearAdd(self, source_node):
    kwargs = dict()
    kwargs['is_tflite'] = True
    kwargs['input_scale'] = [self.state_dict[source_node.layer.input[1]].tolist(), self.state_dict[source_node.layer.input[4]].tolist()]
    kwargs['input_zero_point'] = [self.state_dict[source_node.layer.input[2]].tolist(), self.state_dict[source_node.layer.input[5]].tolist()]  #uint8
    IR_node = self._convert_identity_operation(source_node, new_op="onnxpqn_Add")
    kwargs['output_scale'] = self.state_dict[source_node.layer.input[6]]
    kwargs['output_zero_point'] = self.state_dict[source_node.layer.input[7]]
    kwargs['data_format'] = 'int8' # str(self.state_dict[source_node.layer.input[7]].dtype)
    # One of the input is a constant tensor
    if source_node.layer.input[0] in source_node.initializers:
      const_tensor_inp = self.state_dict[source_node.layer.input[0]]
      kwargs['const_tensor'] = True
      # const_input = const_tensor_inp.reshape([1, const_tensor_inp.size])
      self.set_weight(source_node.name, "weights", const_tensor_inp)
    elif source_node.layer.input[3] in source_node.initializers:
      const_tensor_inp = self.state_dict[source_node.layer.input[3]]
      # if len(weights_list) > 0:
      kwargs['const_tensor'] = True
      input_node = self.src_graph.get_parent(source_node.name, [0])
      # input_shape = tuple(self.get_output_shape(input_node))
      const_input = const_tensor_inp.reshape([1, const_tensor_inp.size])
      # const_input = np.broadcast_to(const_input, input_shape).transpose([0,3,1,2])
      self.set_weight(source_node.name, "weights", const_input)
    assign_IRnode_values(IR_node, kwargs)

  def rename_QLinearMul(self, source_node):
    kwargs = dict()
    logger.debug("Entered rename_QLinearMul")
    kwargs['is_tflite'] = True
    kwargs['input_scale'] = [self.state_dict[source_node.layer.input[1]].tolist(), self.state_dict[source_node.layer.input[4]].tolist()]
    kwargs['input_zero_point'] = [self.state_dict[source_node.layer.input[2]].tolist(), self.state_dict[source_node.layer.input[5]].tolist()]  #uint8
    IR_node = self._convert_identity_operation(source_node, new_op="onnxpqn_Mul")
    kwargs['output_scale'] = self.state_dict[source_node.layer.input[6]]
    kwargs['output_zero_point'] = self.state_dict[source_node.layer.input[7]]
    kwargs['data_format'] = str(self.state_dict[source_node.layer.input[7]].dtype)
    #If one of the input is a constant tensor
    if source_node.layer.input[0] in source_node.initializers:
      const_tensor_inp = self.state_dict[source_node.layer.input[0]]
      kwargs['const_tensor'] = True
      # const_input = const_tensor_inp.reshape([1, const_tensor_inp.size])
      self.set_weight(source_node.name, "weights", const_tensor_inp)
    elif source_node.layer.input[3] in source_node.initializers:
      const_tensor_inp = self.state_dict[source_node.layer.input[3]]
      # if len(weights_list) > 0:
      kwargs['const_tensor'] = True
      input_node = self.src_graph.get_parent(source_node.name, [0])
      # input_shape = tuple(self.get_output_shape(input_node))
      const_input = const_tensor_inp.reshape([1, const_tensor_inp.size])
      # const_input = np.broadcast_to(const_input, input_shape).transpose([0,3,1,2])
      self.set_weight(source_node.name, "weights", const_input)
    assign_IRnode_values(IR_node, kwargs)
    logger.debug("Exited rename_QLinearMul")

  def rename_QLinearGlobalAveragePool(self, source_node):
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
    IR_node = self._convert_identity_operation(source_node, new_op="onnxpqn_Pool")
    kwargs['pooling_type'] = 'AVG'
    kwargs['is_tflite'] = True
    kwargs['input_scale'] = self.state_dict[source_node.layer.input[1]]
    kwargs['input_zero_point'] = self.state_dict[source_node.layer.input[2]]
    kwargs['output_scale'] = self.state_dict[source_node.layer.input[3]]
    kwargs['output_zero_point'] = self.state_dict[source_node.layer.input[4]]
    assign_IRnode_values(IR_node, kwargs)
    IR_node.attr['global_pooling'].b = True

  def rename_QLinearSigmoid(self, source_node):
    kwargs = dict()
    kwargs['is_tflite'] = True
    kwargs['input_scale'] = self.state_dict[source_node.layer.input[1]]
    kwargs['input_zero_point'] = self.state_dict[source_node.layer.input[2]]
    IR_node = self._convert_identity_operation(source_node, new_op="onnxpqn_Sigmoid")
    kwargs['output_scale'] = self.state_dict[source_node.layer.input[3]]
    kwargs['output_zero_point'] = self.state_dict[source_node.layer.input[4]]
    assign_IRnode_values(IR_node, kwargs)

  def rename_QLinearSoftmax(self, source_node):
    attrs = source_node.attrs
    IR_node = self._convert_identity_operation(source_node, new_op='onnxpqn_Softmax')
    kwargs = dict()
    kwargs['is_tflite'] = True
    kwargs['input_scale'] = self.state_dict[source_node.layer.input[1]]
    kwargs['input_zero_point'] = self.state_dict[source_node.layer.input[2]]
    kwargs['output_scale'] = self.state_dict[source_node.layer.input[3]]
    kwargs['output_zero_point'] = self.state_dict[source_node.layer.input[4]]
    kwargs['data_format'] = 'int8' # str(self.state_dict[source_node.layer.input[4]].dtype)
    axis = -1
    if 'axis' in attrs:
      axis = attrs['axis']
    ndim = len(IR_node.attr['_output_shapes'].list.shape[0].dim)
    if ndim == 4 and axis == 2:
      # Replace axis 2 with axis 1. Need to add permute
      axis = 1
      kwargs_perm = dict()
      IR_node_0 = self._convert_identity_operation(source_node, new_op='Transpose')
      IR_node_1 = self._convert_identity_operation(source_node, new_op='Transpose')
      IR_node.name = "sfmx_" + IR_node.name
      IR_node_0.name = "perm_" + IR_node_0.name
      while len(IR_node.input):
        IR_node.input.pop()
      while len(IR_node_1.input):
        IR_node_1.input.pop()
      IR_node.input.insert(0, IR_node_0.name)
      IR_node_1.input.insert(0, IR_node.name)
      kwargs_perm['perm'] = [0,2,1,3]
      kwargs_perm['data_format'] = 'NCHW'
      assign_IRnode_values(IR_node_0, kwargs_perm)
      assign_IRnode_values(IR_node_1, kwargs_perm)
    axis = OnnxParser._convert_axis(IR_node, axis)
    IR_node.attr["axis"].i = axis
    assign_IRnode_values(IR_node, kwargs)

  # def rename_QLinearConcat(self, source_node):
  #   inputs = source_node.layer.input[2::3]
  #   while len(source_node.layer.input) > 0:
  #     source_node.layer.input.pop()
  #   for inp in inputs:
  #     source_node.layer.input.append(inp)
  #   self.rename_Concat(source_node)

  def rename_QLinearConcat(self, source_node):
    attrs = source_node.attrs
    IR_node = self._convert_identity_operation(source_node, new_op="onnxpqn_Concat")
    if 'axis' in attrs:
      axis = OnnxParser._convert_axis(IR_node, attrs['axis'])
      IR_node.attr["axis"].i = axis
    else:
      raise AttributeError('axis not found')
    kwargs = dict()
    kwargs['input_scale'] = [self.state_dict[tensor_name].tolist() for tensor_name in source_node.layer.input[3::3]]
    kwargs['input_zero_point'] = [self.state_dict[tensor_name].tolist() for tensor_name in source_node.layer.input[4::3]]
    kwargs['output_scale'] = self.state_dict[source_node.layer.input[0]]
    kwargs['output_zero_point'] = self.state_dict[source_node.layer.input[1]]
    kwargs['data_format'] = 'int8' # str(self.state_dict[source_node.layer.input[1]].dtype)
    assign_IRnode_values(IR_node, kwargs)

  def rename_QLinearFullyConnected(self, source_node):
    attr = source_node.attrs
    kwargs = dict()
    kwargs['is_tflite'] = True
    IR_node = self._convert_identity_operation(source_node, new_op="onnxpqn_FullyConnected")
    kwargs['input_scale'] = self.state_dict[source_node.layer.input[1]]
    kwargs['input_zero_point'] = self.state_dict[source_node.layer.input[2]]
    # kwargs['output_scale'] = self.state_dict[source_node.layer.input[7]]
    # kwargs['output_zero_point'] = self.state_dict[source_node.layer.input[8]]
    kwargs['data_fmt'] = 'int8' # str(self.state_dict[source_node.layer.input[8]].dtype)
    # weights
    self.set_weight(source_node.name, "weights", self.state_dict[source_node.layer.input[3]].T)  #.transpose(2, 3, 1, 0))
    # bias
    if(len(source_node.layer.input)<9):
      IR_node.attr["use_bias"].b = False
      kwargs['output_scale'] = self.state_dict[source_node.layer.input[6]]
      kwargs['output_zero_point'] = self.state_dict[source_node.layer.input[7]]
    else:
      IR_node.attr["use_bias"].b = True
      self.set_weight(source_node.name, "bias", self.state_dict[source_node.layer.input[6]])
      kwargs['output_scale'] = self.state_dict[source_node.layer.input[7]]
      kwargs['output_zero_point'] = self.state_dict[source_node.layer.input[8]]
    # weight scales
    self.set_weight(source_node.name, "weight_scales", self.state_dict[source_node.layer.input[4]])
    # weight zero points
    self.set_weight(source_node.name, "weight_zero_points", self.state_dict[source_node.layer.input[5]])
    # haven't encounterd the position of the bias scale and offset
    if len(source_node.layer.input) > 9:
      self.set_weight(source_node.name, "bias_scales", self.state_dict[source_node.layer.input[9]])
      self.set_weight(source_node.name, "bias_zero_points", self.state_dict[source_node.layer.input[10]])
    weights_name = source_node.weights_name + '_1'
    bias_name = source_node.weights_name + '_2'
    W = self.state_dict[weights_name]
    transB = attr.get('transB', False)
    transA = attr.get('transA', False)
    if transA:
      raise NotImplementedError("not implemented transA in %s", source_node.name)
    if transB:
      W = W.transpose()
    units = self.src_graph.shape_dict[source_node.name][-1]
    IR_node.attr['units'].i = units
    # if 'flatten' in source_node.attrs:
    #   IR_node.attr['flatten'].b = True if source_node.attrs['flatten'] == 1 else False
    # if 'reshape' in source_node.attrs:
    #   IR_node.attr['reshape'].list.i.extend(source_node.attrs['reshape'])
    # self._set_output_shape(source_node, IR_node, True)
    assign_IRnode_values(IR_node, kwargs)

  def rename_ArgMax(self, source_node):
    axis = int(source_node.attrs.get("axis", 0))
    keepdims = source_node.attrs.get("keepdims", 1)
    select_last_index = source_node.attrs.get("select_last_index", 0)
    parent = self.src_graph.get_parent(source_node.name, [0])
    input_shape = self.shape_dict[parent.name]
    new_shape = self._cfirst_to_clast(input_shape)
    axis_clast = self._convert_onnx_axis(new_shape, axis)
    # if select_last_index == True:
    #   logger.error("Only select_last_index=false is supported")
    if axis < 0:
      axis += len(input_shape)
    kwargs = dict(axis=axis_clast, shape=new_shape, keepdims=keepdims, select_last_index=select_last_index)
    if axis != 1:
      IR_node = self._convert_identity_operation(source_node, new_op="Tf_ArgMax")
    else:
      IR_node = self._convert_identity_operation(source_node, new_op="argmax")
    assign_IRnode_values(IR_node, kwargs)

  def rename_BatchNormalization(self, source_node):
    # TODO
    # output_shape
    if source_node.real_name not in self.dummy_layers:
      self.rename_dvnet_BatchNormalization(source_node)
      return
    IR_node = self._convert_identity_operation(source_node, new_op='Mxnet_Batchnorm')
    IR_node.attr['axis'].i = 1
    IR_node.attr['fix_gamma'].b = False
    IR_node.attr['output_mean_var'].b = False
    attr = source_node.attrs
    IR_node.attr['bias'].b = True
    IR_node.attr['eps'].f = attr['epsilon']
    IR_node.attr['momentum'].f = attr['momentum']
    IR_node.attr['use_global_stats'].b = False
    weights_name = source_node.weights_name + '_1'
    bias_name = source_node.weights_name + '_2'
    mean_name = source_node.weights_name + '_3'
    var_name = source_node.weights_name + '_4'
    if bias_name in self.state_dict:
      beta = self.state_dict[bias_name]
      IR_node.attr['bias'].b = True
    else:
      IR_node.attr['bias'].b = False
    if weights_name in self.state_dict:
      gamma = self.state_dict[weights_name]
      IR_node.attr['scale'].b = True
    else:
      IR_node.attr['scale'].b = False
    mean = self.state_dict[mean_name]
    variance = self.state_dict[var_name]
    if IR_node.attr['scale'].b:
      self.set_weight(source_node.name, "scale", gamma)
    if IR_node.attr['bias'].b:
      self.set_weight(source_node.name, "bias", beta)
    # mean
    self.set_weight(source_node.name, "mean", mean)
    # var
    self.set_weight(source_node.name, "var", variance)

  def rename_dvnet_BatchNormalization(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="BatchNorm")
    attr = source_node.attrs
    # epsilon
    IR_node.attr['epsilon'].f = attr['epsilon']
    weights_name = source_node.weights_name + '_1'
    bias_name = source_node.weights_name + '_2'
    mean_name = source_node.weights_name + '_3'
    var_name = source_node.weights_name + '_4'
    if bias_name in self.state_dict:
      beta = self.state_dict[bias_name]
      IR_node.attr['bias'].b = True
    else:
      IR_node.attr['bias'].b = False
    if weights_name in self.state_dict:
      gamma = self.state_dict[weights_name]
      IR_node.attr['scale'].b = True
    else:
      IR_node.attr['scale'].b = False
    mean = self.state_dict[mean_name]
    variance = self.state_dict[var_name]
    if IR_node.attr['scale'].b:
      self.set_weight(source_node.name, "scale", gamma)
    if IR_node.attr['bias'].b:
      self.set_weight(source_node.name, "bias", beta)
    # mean
    self.set_weight(source_node.name, "mean", mean)
    # var
    self.set_weight(source_node.name, "var", variance)

  def rename_InstanceNormalization(self, source_node):
    if source_node.name not in self.dummy_layers:
      IR_node = self._convert_identity_operation(source_node, new_op='GroupNorm')

      self.weights[source_node.name] = [self.state_dict[source_node.layer.input[1]],\
                                        self.state_dict[source_node.layer.input[2]]]
      kwargs = {}
      kwargs['eps'] = float(source_node.attrs['epsilon'])
      kwargs['groups'] = int(self.state_dict[source_node.layer.input[1]].shape[0])
      assign_IRnode_values(IR_node, kwargs)
    else:
      attr = source_node.attrs
      IR_node = self._convert_identity_operation(source_node, new_op='Mxnet_InstanceNorm')
      weights_name = source_node.weights_name + '_1'
      bias_name = source_node.weights_name + '_2'
      mean_name = source_node.weights_name + '_3'
      var_name = source_node.weights_name + '_4'
      if 'epsilon' in attr:
        kwargs = {}
        kwargs['eps'] = float(source_node.attrs['epsilon'])
        assign_IRnode_values(IR_node, kwargs)
      if bias_name in self.state_dict:
        beta = self.state_dict[bias_name]
        IR_node.attr['bias'].b = True
      else:
        IR_node.attr['bias'].b = False
      if weights_name in self.state_dict:
        gamma = self.state_dict[weights_name]
        IR_node.attr['scale'].b = True
      else:
        IR_node.attr['scale'].b = False
      if mean_name in self.state_dict:
        mean = self.state_dict[mean_name]
        self.set_weight(source_node.name, "mean", mean)
      if var_name in self.state_dict:
        variance = self.state_dict[var_name]
        self.set_weight(source_node.name, "var", variance)
      if IR_node.attr['scale'].b:
        self.set_weight(source_node.name, "scale", gamma)
      if IR_node.attr['bias'].b:
        self.set_weight(source_node.name, "bias", beta)

  def rename_Clip(self, source_node):
    attrs = source_node.attrs
    if 'min' not in attrs:
      if len(source_node.layer.input) > 1 and len(source_node.layer.input[1]):
        attrs['min'] = self.state_dict[source_node.layer.input[1]]
      else:
        attrs['min'] = 0
    if 'max' not in attrs:
      if len(source_node.layer.input) > 2 and len(source_node.layer.input[2]):
        attrs['max'] = self.state_dict[source_node.layer.input[2]]
      else:
        attrs['max'] = -1
    IR_node = self._convert_identity_operation(source_node, new_op='clip')
    assign_IRnode_values(IR_node, {'a_max': attrs['max'], 'a_min': attrs['min']})
    if source_node.layer.input[0] in source_node.initializers:
      IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_0.input):
        IR_node_0.input.pop()
      IR_node_0.name = 'node_' + source_node.layer.input[0]
      shape = self.state_dict[source_node.layer.input[0]].shape
      shape = self._cfirst_to_clast(shape)
      IR_node_0.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_0, shape)
      self.set_weight(IR_node_0.name, 'weights', self.state_dict[source_node.layer.input[0]])
      IR_node.input.insert(0, IR_node_0.name)

  def rename_Constant(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Constant")
    # python if else will treat ndarray of 0 as false
    tensor_value = source_node.attrs['value']
    if type(tensor_value) is not list:
      tensor_value = numpy_helper.to_array(tensor_value)
      if tensor_value.shape == tuple():
        tensor_value = str(tensor_value) if not tensor_value else tensor_value
    assign_IRnode_values(IR_node, {'value': tensor_value})

  def rename_Shape(self, source_node):
    # get the bottom of this node and capture its's output size and
    # expose it as a DummyData
    IR_node = self._convert_identity_operation(source_node, new_op="DummyData")
    parent = self.src_graph.get_parent(source_node.name, [0])
    parent_output_shape = self.shape_dict[parent.name]

  def rename_ScatterND(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Onnx_Scatternd")
    if source_node.layer.input[0] in source_node.initializers:
      IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_0.input):
        IR_node_0.input.pop()
      IR_node_0.name = 'node_' + source_node.layer.input[0]
      shape = self.state_dict[source_node.layer.input[0]].shape
      shape = self._cfirst_to_clast(shape)
      IR_node_0.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_0, shape)
      self.set_weight(IR_node_0.name, 'weights', self.state_dict[source_node.layer.input[0]])
      IR_node.input.insert(0, IR_node_0.name)
    if source_node.layer.input[1] in source_node.initializers:
      IR_node_1 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_1.input):
        IR_node_1.input.pop()
      IR_node_1.name = 'node_' + source_node.layer.input[1]
      shape = self.state_dict[source_node.layer.input[1]].shape
      shape = self._cfirst_to_clast(shape)
      IR_node_1.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_1, shape)
      self.set_weight(IR_node_1.name, 'weights', self.state_dict[source_node.layer.input[1]])
      IR_node.input.insert(1, IR_node_1.name)
    if source_node.layer.input[2] in source_node.initializers:
      IR_node_2 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_2.input):
        IR_node_2.input.pop()
      IR_node_2.name = 'node_' + source_node.layer.input[2]
      shape = self.state_dict[source_node.layer.input[2]].shape
      shape = self._cfirst_to_clast(shape)
      IR_node_2.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_2, shape)
      self.set_weight(IR_node_2.name, 'weights', self.state_dict[source_node.layer.input[2]])
      IR_node.input.insert(2, IR_node_2.name)
    """
        if source_node.layer.input[0] in source_node.initializers:
            self.set_weight(source_node.name,'data',self.state_dict[source_node.layer.input[0]].tolist())
        if source_node.layer.input[1] in source_node.initializers:
            self.set_weight(source_node.name,'indices',self.state_dict[source_node.layer.input[1]].tolist())
        if source_node.layer.input[2] in source_node.initializers:
            self.set_weight(source_node.name,'updates',self.state_dict[source_node.layer.input[2]].tolist())
        """

  def rename_Where(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Onnx_Where")
    if source_node.layer.input[0] in source_node.initializers:
      IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_0.input):
        IR_node_0.input.pop()
      IR_node_0.name = 'node' + source_node.layer.input[0]
      if len(self.state_dict[source_node.layer.input[0]].shape) == 0:
        self.state_dict[source_node.layer.input[0]] = np.array([self.state_dict[source_node.layer.input[0]].tolist()])
      shape = self.state_dict[source_node.layer.input[0]].shape
      shape = self._cfirst_to_clast(shape)
      IR_node_0.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_0, shape)
      self.set_weight(IR_node_0.name, 'weights', self.state_dict[source_node.layer.input[0]])
      IR_node.input.insert(0, IR_node_0.name)
    if source_node.layer.input[1] in source_node.initializers:
      IR_node_1 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_1.input):
        IR_node_1.input.pop()
      IR_node_1.name = 'node' + source_node.layer.input[1]
      if len(self.state_dict[source_node.layer.input[1]].shape) == 0:
        self.state_dict[source_node.layer.input[1]] = np.array([self.state_dict[source_node.layer.input[1]].tolist()])
      shape = self.state_dict[source_node.layer.input[1]].shape
      shape = self._cfirst_to_clast(shape)
      IR_node_1.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_1, shape)
      self.set_weight(IR_node_1.name, 'weights', self.state_dict[source_node.layer.input[1]])
      IR_node.input.insert(1, IR_node_1.name)
    if source_node.layer.input[2] in source_node.initializers:
      IR_node_2 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_2.input):
        IR_node_2.input.pop()
      IR_node_2.name = 'node' + source_node.layer.input[2]
      if len(self.state_dict[source_node.layer.input[2]].shape) == 0:
        self.state_dict[source_node.layer.input[2]] = np.array([self.state_dict[source_node.layer.input[2]].tolist()])
      shape = self.state_dict[source_node.layer.input[2]].shape
      shape = self._cfirst_to_clast(shape)
      IR_node_2.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_2, shape)
      self.set_weight(IR_node_2.name, 'weights', self.state_dict[source_node.layer.input[2]])
      IR_node.input.insert(2, IR_node_2.name)
    """if source_node.layer.input[0] in source_node.initializers:
            self.set_weight(source_node.name,'condition',self.state_dict[source_node.layer.input[0]].tolist())
        if source_node.layer.input[1] in source_node.initializers:
            self.set_weight(source_node.name,'X',self.state_dict[source_node.layer.input[1]].tolist())
        if source_node.layer.input[2] in source_node.initializers:
            self.set_weight(source_node.name,'Y',self.state_dict[source_node.layer.input[2]].tolist())"""

  def rename_Equal(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Onnx_Equal")
    if source_node.layer.input[0] in source_node.initializers:
      IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_0.input):
        IR_node_0.input.pop()
      IR_node_0.name = 'node' + source_node.layer.input[0]
      if len(self.state_dict[source_node.layer.input[0]].shape) == 0:
        self.state_dict[source_node.layer.input[0]] = np.array([self.state_dict[source_node.layer.input[0]].tolist()])
      shape = self.state_dict[source_node.layer.input[0]].shape
      shape = self._cfirst_to_clast(shape)
      IR_node_0.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_0, shape)
      self.set_weight(IR_node_0.name, 'weights', self.state_dict[source_node.layer.input[0]])
      IR_node.input.insert(0, IR_node_0.name)
    if source_node.layer.input[1] in source_node.initializers:
      IR_node_1 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_1.input):
        IR_node_1.input.pop()
      IR_node_1.name = 'node' + source_node.layer.input[1]
      if len(self.state_dict[source_node.layer.input[1]].shape) == 0:
        self.state_dict[source_node.layer.input[1]] = np.array([self.state_dict[source_node.layer.input[1]].tolist()])
      shape = self.state_dict[source_node.layer.input[1]].shape
      shape = self._cfirst_to_clast(shape)
      IR_node_1.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_1, shape)
      self.set_weight(IR_node_1.name, 'weights', self.state_dict[source_node.layer.input[1]])
      IR_node.input.insert(1, IR_node_1.name)
    """if source_node.layer.input[0] in source_node.initializers:
            self.set_weight(source_node.name,'A',self.state_dict[source_node.layer.input[0]].tolist())
        if source_node.layer.input[1] in source_node.initializers:
            self.set_weight(source_node.name,'B',self.state_dict[source_node.layer.input[1]].tolist())"""

  def rename_GreaterOrEqual(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Onnx_Greaterorequal")
    if source_node.layer.input[0] in source_node.initializers:
      IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_0.input):
        IR_node_0.input.pop()
      IR_node_0.name = 'node' + source_node.layer.input[0]
      if len(self.state_dict[source_node.layer.input[0]].shape) == 0:
        self.state_dict[source_node.layer.input[0]] = np.array([self.state_dict[source_node.layer.input[0]].tolist()])
      shape = self.state_dict[source_node.layer.input[0]].shape
      shape = self._cfirst_to_clast(shape)
      IR_node_0.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_0, shape)
      self.set_weight(IR_node_0.name, 'weights', self.state_dict[source_node.layer.input[0]])
      IR_node.input.insert(0, IR_node_0.name)
    if source_node.layer.input[1] in source_node.initializers:
      IR_node_1 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_1.input):
        IR_node_1.input.pop()
      IR_node_1.name = 'node' + source_node.layer.input[1]
      if len(self.state_dict[source_node.layer.input[1]].shape) == 0:
        self.state_dict[source_node.layer.input[1]] = np.array([self.state_dict[source_node.layer.input[1]].tolist()])
      shape = self.state_dict[source_node.layer.input[1]].shape
      shape = self._cfirst_to_clast(shape)
      IR_node_1.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_1, shape)
      self.set_weight(IR_node_1.name, 'weights', self.state_dict[source_node.layer.input[1]])
      IR_node.input.insert(1, IR_node_1.name)
    """if source_node.layer.input[0] in source_node.initializers:
            self.set_weight(source_node.name,'A',self.state_dict[source_node.layer.input[0]].tolist())
        if source_node.layer.input[1] in source_node.initializers:
            self.set_weight(source_node.name,'B',self.state_dict[source_node.layer.input[1]].tolist())"""

  def rename_Resize(self, source_node):
    '''
        enum INTERP_TYPE{
        LINEAR_3D = 0,
        NEAREST_3D = 1,
        NEAREST_4D = 2,
        BILINEAR_FALSE = 3,
        BILINEAR_TRUE  = 4,
        NEAREST_5D  = 5,
        TRILINEAR_FALSE = 6, 
        TRILINEAR_TRUE  = 7
        };
        '''
    input_node = self.get_parent(source_node.name, [0])
    input_node_name = input_node.name
    if input_node.type == "Split":
      input_node_name = source_node.layer.input[0]
    input_shape = self.shape_dict[input_node_name]
    output_shape = self.shape_dict[source_node.name]
    operation = None
    scale = None
    mode = None
    align_corners = None
    if len(input_shape) == 3:
      if source_node.attrs['mode'].decode("utf-8") == 'linear':
        operation = 'Upsample'
        scale = output_shape[-1] / input_shape[-1]
        mode = UPSP_TYPE.LINEAR_3D.value
      elif source_node.attrs['mode'].decode("utf-8") == 'nearest':
        operation = 'Upsample'
        scale = output_shape[-1] / input_shape[-1]
        mode = UPSP_TYPE.NEAREST_3D.value
    elif len(input_shape) == 4:
      if source_node.attrs['mode'].decode("utf-8") == 'nearest':
        operation = 'Upsample'
        scale_h = output_shape[2] / input_shape[2]
        scale_w = output_shape[3] / input_shape[3]
        if scale_h != scale_w:
          kwargs = dict()
          IR_node = self._convert_identity_operation(source_node, new_op='Tf_ResizeNearestNeighbor')
          kwargs = dict(align_corners=False, half_pixel_centers=False)
          assign_IRnode_values(IR_node, kwargs)
          return
        elif not scale_h.is_integer():
          kwargs = dict()
          IR_node = self._convert_identity_operation(source_node, new_op='Tf_ResizeNearestNeighbor')
          kwargs = dict(align_corners=False, half_pixel_centers=False)
          assign_IRnode_values(IR_node, kwargs)
          return
        elif (scale_h == scale_w):
          operation = 'Upsample'
          scale = scale_h
          mode = UPSP_TYPE.NEAREST_4D.value
          kwargs = dict()
          IR_node = self._convert_identity_operation(source_node, new_op="Upsample")
          kwargs['shape'] = self._cfirst_to_clast(input_shape)
          kwargs['scale'] = scale
          kwargs['mode'] = mode
          #kwargs['align_corners'] = align_corners
          kwargs['align_corners'] = "NearestNeighbor"
          kwargs['half_pixel_centers'] = False
          assign_IRnode_values(IR_node, kwargs)
          return
        kwargs = dict()
        IR_node = self._convert_identity_operation(source_node, new_op=operation)
        kwargs['shape'] = self._cfirst_to_clast(input_shape)
        kwargs['align_corners'] = "NearestNeighbor"
        kwargs['half_pixel_centers'] = False
        assign_IRnode_values(IR_node, kwargs)
        return
      elif source_node.attrs['mode'].decode("utf-8") == 'linear':
        operation = 'Upsample'
        scale_h = output_shape[2] / input_shape[2]
        scale_w = output_shape[3] / input_shape[3]
        if (not scale_h.is_integer() or (not scale_w.is_integer())):
          kwargs = dict()
          IR_node = self._convert_identity_operation(source_node, new_op='ONNX_UpSample')
          kwargs = dict(align_corners=False, half_pixel_centers=False)
          assign_IRnode_values(IR_node, kwargs)
          return
        assert (scale_h == scale_w)
        if source_node.attrs['coordinate_transformation_mode'].decode("utf-8") == 'align_corners':
          mode = UPSP_TYPE.BILINEAR_TRUE.value
          scale = [scale_h, ((input_shape[2] - 1) / (output_shape[2] - 1)), ((input_shape[3] - 1) / (output_shape[3] - 1)), output_shape[2], output_shape[3]]
          align_corners = True
        elif source_node.attrs['coordinate_transformation_mode'].decode("utf-8") in ['pytorch_half_pixel', 'half_pixel'] and source_node.name not in self.dummy_layers:
          scale = [output_shape[i] / input_shape[i] for i in range(2, 4)]
          IR_node = self._convert_identity_operation(source_node, new_op='Pytorch_half_pixel_resize')
          return
        else:
          mode = UPSP_TYPE.BILINEAR_FALSE.value
          scale = scale_h
          align_corners = False
      else:
        raise NotImplementedError("Only Nearest and BiLinear Interpolation is supported for 4D Inputs")
    elif len(input_shape) == 5:
      if source_node.attrs['mode'].decode("utf-8") == 'nearest':
        operation = 'Upsample'
        scale_h = output_shape[3] / input_shape[3]
        scale_w = output_shape[4] / input_shape[4]
        scale_d = output_shape[2] / input_shape[2]
        if (scale_h == scale_w == scale_d):
          operation = 'Upsample'
          scale = scale_h
          mode = UPSP_TYPE.NEAREST_5D.value
        else:
          logger.error("Only Uniform scaling on all the sides is supported")
        return
      elif source_node.attrs['mode'].decode("utf-8") == 'linear':
        operation = 'Upsample'
        scale_h = output_shape[3] / input_shape[3]
        scale_w = output_shape[4] / input_shape[4]
        scale_d = output_shape[2] / input_shape[2]
        #assert(scale_h == scale_w == scale_d)
        if source_node.attrs['coordinate_transformation_mode'].decode("utf-8") == 'align_corners':
          mode = UPSP_TYPE.TRILINEAR_TRUE.value
          scale = scale_h
          align_corners = True
        elif source_node.attrs['coordinate_transformation_mode'].decode("utf-8") == 'half_pixel':
          mode = UPSP_TYPE.TRILINEAR_TRUE.value
          scale = scale_h
          align_corners = False
        else:
          mode = UPSP_TYPE.TRILINEAR_FALSE.value
          scale = scale_h
          align_corners = False
      else:
        raise NotImplementedError("Only Nearest and Trilinear Interpolation is supported for 5D Inputs")
    else:
      raise NotImplementedError("Only 3D and 4D and 5D Tensors are supported for Interpolation")
    assert operation is not None
    assert scale is not None
    assert mode is not None
    if mode in [0, 1]:
      input_shape = self._cfirst_to_clast(input_shape)
    elif mode in [2, 3, 4]:
      input_shape = self._cfirst_to_clast(input_shape)
    elif mode in [5, 6, 7]:
      input_shape = self._cfirst_to_clast(input_shape)
    else:
      raise ValueError("Something has gone wrong, Mode is not withing the acceptable values`")
    kwargs = dict()
    kwargs['mode'] = mode
    kwargs['scale'] = scale
    kwargs['align_corners'] = align_corners
    kwargs['shape'] = input_shape
    IR_node = self._convert_identity_operation(source_node, new_op=operation)
    assign_IRnode_values(IR_node, kwargs)
    return

  def rename_Reshape(self, source_node):
    my_input_shape = self.shape_dict[source_node.name]
    if len(my_input_shape) < 6:
      new_shape = self._cfirst_to_clast(my_input_shape)
    else:
      trunc_flag = True if sum(my_input_shape[:-5]) == len(my_input_shape) - 5 else False
      if not trunc_flag:
        logger.error("Current Stack is limited to 5 dims, Please check the dims of %s", source_node.name)
      else:
        logger.warn("%s has more than 5 dims, Truncating to 5 dims", source_node.name)
        trunc_data = my_input_shape[-5:]
        new_shape = [int(trunc_data[0])] + trunc_data[2:] + [int(trunc_data[1])]
    while len(new_shape) > 5:
      new_shape.pop(0)
    # self.shape_dict[source_node.name] = self._clast_to_cfirst(new_shape)
    IR_node = self._convert_identity_operation(source_node, new_op="Reshape")
    IR_node.attr['_output_shapes'].list.shape.pop()
    self._set_output_shape_from_list(IR_node, new_shape)
    if "new_precision" in source_node.attrs:
      IR_node.attr['new_precision'].s = source_node.attrs['new_precision']
    return

  def rename_Gather(self, source_node):
    map_to_embedding = False
    if ((source_node.layer.input[0] in self.shape_dict) and (source_node.layer.input[1] in self.state_dict) and len(self.state_dict[source_node.layer.input[1]].shape) == 0 and self.shape_dict[source_node.layer.input[0]][source_node.attrs['axis']] != 1):
      axis = source_node.attrs['axis']
      data_shape = self.shape_dict[source_node.layer.input[0]]
      index = self.state_dict[source_node.layer.input[1]].min()
      if axis < 0:
        axis += len(data_shape)
      if index < 0:
        index += data_shape[axis]
      IR_node = self._convert_identity_operation(source_node, new_op='Slice')
      kwargs = dict()
      kwargs["starts"] = [0] * len(data_shape)
      kwargs["starts"][axis] = index
      kwargs["starts"] = self._cfirst_to_clast(kwargs["starts"])
      kwargs["starts"] = list(map(int, kwargs["starts"]))
      output_shape = [i.size for i in IR_node.attr['_output_shapes'].list.shape[0].dim]
      output_shape = self._clast_to_cfirst(output_shape)
      output_shape.insert(axis, 1)
      output_shape = self._cfirst_to_clast(output_shape)

      IR_node.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node, output_shape)

      IR_node.name = IR_node.name + '_slice'
      IR_node_ = self._convert_identity_operation(source_node, new_op='Reshape')
      IR_node_.input[:] = [IR_node.name]
      assign_IRnode_values(IR_node, kwargs)
      return
    if source_node.layer.input[1] in self.state_dict.keys():
      indices = self.state_dict[source_node.layer.input[1]].tolist()
      if isinstance(indices, int) and self.shape_dict[source_node.layer.input[0]][source_node.attrs['axis']] == 1:
        IR_node = self._convert_identity_operation(source_node, new_op="Reshape")
        return
    if (source_node.layer.input[0] in self.state_dict) and (source_node.layer.input[1] in self.shape_dict):
      weight = self.state_dict[source_node.layer.input[0]]
      input_shape = self.shape_dict[source_node.layer.input[1]]
      output_shape = self.shape_dict[source_node.layer.name]
      map_to_embedding = (input_shape + list(weight.shape[1:]) == output_shape) and (len(weight.shape) == 2)
    if map_to_embedding:
      IR_node = self._convert_identity_operation(source_node, new_op='Embedding')
      self.set_weight(source_node.name, "weights", weight)
      kwargs = dict()
      kwargs["input_dim"] = weight.shape[0]
      kwargs["output_dim"] = weight.shape[1]
      assign_IRnode_values(IR_node, kwargs)
      return
    #to include the case where the index of Slice might actually be in the prev node's output, such as when the prev node is an argmax node.
    # output_idx_nodes_type = ["ArgMax"]
    # parent_node = self.get_parent(source_node.name, [1])
    # if ((source_node.layer.input[0] in self.shape_dict) and (source_node.layer.input[1] in self.shape_dict)) and (parent_node.layer.op_type in output_idx_nodes_type) :
    #   axis = source_node.attrs['axis']
    #   data_shape = self.shape_dict[source_node.layer.input[0]]
    #   index_list = self.shape_dict[source_node.layer.input[1]]
    #   for index in index_list:
    #     IR_node = self._convert_identity_operation(source_node, new_op='Slice')
    #     kwargs = dict()
    #     kwargs["starts"] = [0] * len(data_shape)
    #     kwargs["starts"][axis] = index
    #     kwargs["starts"] = self._cfirst_to_clast(kwargs["starts"])
    #     kwargs["starts"] = list(map(int, kwargs["starts"]))
    #     output_shape = [i.size for i in IR_node.attr['_output_shapes'].list.shape[0].dim]
    #     output_shape = self._clast_to_cfirst(output_shape)
    #     output_shape.insert(axis, 1)
    #     output_shape = self._cfirst_to_clast(output_shape)

    #     IR_node.attr['_output_shapes'].list.shape.pop()
    #     self._set_output_shape_from_list(IR_node, output_shape)

    #     IR_node.name = IR_node.name + '_slice'
    #     IR_node_ = self._convert_identity_operation(source_node, new_op='Reshape')
    #     IR_node_.input[:] = [IR_node.name]
    #     assign_IRnode_values(IR_node, kwargs)
    #     return

    #Generic Gather
    IR_node = self._convert_identity_operation(source_node, new_op="Onnx_Gather")
    axis = source_node.attrs['axis']
    if axis < 0:
      axis += len(IR_node.attr['_output_shapes'].list.shape[0].dim)
    IR_node.attr['axis'].i = axis
    IR_node.attr['scalar'].b = False
    if source_node.layer.input[0] in source_node.initializers:
      IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_0.input):
        IR_node_0.input.pop()
      IR_node_0.name = 'node' + source_node.layer.input[0]
      if len(self.state_dict[source_node.layer.input[0]].shape) == 0:
        self.state_dict[source_node.layer.input[0]] = np.array([self.state_dict[source_node.layer.input[0]].tolist()])
        IR_node.attr['scalar'].b = True
      shape = self.state_dict[source_node.layer.input[0]].shape
      shape = self._cfirst_to_clast(shape)
      IR_node_0.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_0, shape)
      self.set_weight(IR_node_0.name, 'weights', self.state_dict[source_node.layer.input[0]])
      IR_node.input.insert(0, IR_node_0.name)
    if source_node.layer.input[1] in source_node.initializers:
      IR_node_1 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_1.input):
        IR_node_1.input.pop()
      IR_node_1.name = 'node' + source_node.layer.input[1]
      if len(self.state_dict[source_node.layer.input[1]].shape) == 0:
        self.state_dict[source_node.layer.input[1]] = np.array([self.state_dict[source_node.layer.input[1]].tolist()])
        IR_node.attr['scalar'].b = True
      shape = self.state_dict[source_node.layer.input[1]].shape
      shape = self._cfirst_to_clast(shape)
      # Handle negative indices
      other_input_shape = self.shape_dict[source_node.layer.input[0]]
      num_elements_data_input = other_input_shape[axis]
      indices_array = self.state_dict[source_node.layer.input[1]].copy()
      indices_array[indices_array < 0] += num_elements_data_input
      self.state_dict[source_node.layer.input[1]] = indices_array
      IR_node_1.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_1, shape)
      self.set_weight(IR_node_1.name, 'weights', self.state_dict[source_node.layer.input[1]])
      IR_node.input.insert(1, IR_node_1.name)

  def rename_GatherElements(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Onnx_Gatherelements")
    IR_node.attr['axis'].i = source_node.attrs['axis']
    if source_node.layer.input[0] in source_node.initializers:
      IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_0.input):
        IR_node_0.input.pop()
      IR_node_0.name = 'node' + source_node.layer.input[0]
      if len(self.state_dict[source_node.layer.input[0]].shape) == 0:
        self.state_dict[source_node.layer.input[0]] = np.array([self.state_dict[source_node.layer.input[0]].tolist()])
      shape = self.state_dict[source_node.layer.input[0]].shape
      shape = self._cfirst_to_clast(shape)
      IR_node_0.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_0, shape)
      self.set_weight(IR_node_0.name, 'weights', self.state_dict[source_node.layer.input[0]])
      IR_node.input.insert(0, IR_node_0.name)
    if source_node.layer.input[1] in source_node.initializers:
      IR_node_1 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_1.input):
        IR_node_1.input.pop()
      IR_node_1.name = 'node' + source_node.layer.input[1]
      if len(self.state_dict[source_node.layer.input[1]].shape) == 0:
        self.state_dict[source_node.layer.input[1]] = np.array([self.state_dict[source_node.layer.input[1]].tolist()])
      shape = self.state_dict[source_node.layer.input[1]].shape
      shape = self._cfirst_to_clast(shape)
      IR_node_1.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_1, shape)
      self.set_weight(IR_node_1.name, 'weights', self.state_dict[source_node.layer.input[1]])
      IR_node.input.insert(1, IR_node_1.name)

  def rename_GatherND(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Onnx_Gathernd")
    IR_node.attr['batch_dims'].i = source_node.attrs['batch_dims']
    if source_node.layer.input[0] in source_node.initializers:
      IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_0.input):
        IR_node_0.input.pop()
      IR_node_0.name = 'node' + source_node.layer.input[0]
      if len(self.state_dict[source_node.layer.input[0]].shape) == 0:
        self.state_dict[source_node.layer.input[0]] = np.array([self.state_dict[source_node.layer.input[0]].tolist()])
      shape = self.state_dict[source_node.layer.input[0]].shape
      shape = self._cfirst_to_clast(shape)
      IR_node_0.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_0, shape)
      self.set_weight(IR_node_0.name, 'weights', self.state_dict[source_node.layer.input[0]])
      IR_node.input.insert(0, IR_node_0.name)
    if source_node.layer.input[1] in source_node.initializers:
      IR_node_1 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_1.input):
        IR_node_1.input.pop()
      IR_node_1.name = 'node' + source_node.layer.input[1]
      if len(self.state_dict[source_node.layer.input[1]].shape) == 0:
        self.state_dict[source_node.layer.input[1]] = np.array([self.state_dict[source_node.layer.input[1]].tolist()])
      shape = self.state_dict[source_node.layer.input[1]].shape
      shape = self._cfirst_to_clast(shape)
      IR_node_1.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_1, shape)
      self.set_weight(IR_node_1.name, 'weights', self.state_dict[source_node.layer.input[1]])
      IR_node.input.insert(1, IR_node_1.name)

  def rename_Unsqueeze(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Reshape")
    import copy
    shape_list = copy.deepcopy(self.shape_dict[source_node.name])
    while len(shape_list) > 5:
      shape_list.pop(0)
    IR_node.attr['_output_shapes'].list.shape.pop()
    shape_list = self._cfirst_to_clast(shape_list)
    self._set_output_shape_from_list(IR_node, shape_list)
    assign_IRnode_values(IR_node, {"data_format": "NCH"})

  def rename_Relu(self, source_node):
    if not source_node.name in self.dummy_layers:
      IR_node = self._convert_identity_operation(source_node, new_op="Relu")
    else:
      IR_node = self._convert_identity_operation(source_node, new_op='Mxnet_Activation')
      IR_node.attr['act'].i = 0

  def rename_LeakyRelu(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="LeakyRelu")
    IR_node.attr["alpha"].f = source_node.attrs['alpha']

  def rename_Tanh(self, source_node):
    if source_node.name in self.dummy_layers:
      IR_node = self._convert_identity_operation(source_node, new_op="ONNX_Tanh")
    else:
      IR_node = self._convert_identity_operation(source_node, new_op="Tanh")

  def rename_Sigmoid(self, source_node):
    if not source_node.name in self.dummy_layers:
      IR_node = self._convert_identity_operation(source_node, new_op="Sigmoid")
      if "new_precision" in source_node.attrs:
        IR_node.attr['new_precision'].s = source_node.attrs['new_precision']
    else:
      IR_node = self._convert_identity_operation(source_node, new_op='Mxnet_Activation')
      IR_node.attr['act'].i = 1

  def rename_Mul(self, source_node):
    input_node = self.get_parent(source_node.name, [0])
    if input_node.type == "DequantizeLinear":
      IR_node = self._convert_identity_operation(source_node, new_op='vector_mult')
      if len(source_node.initializers) > 0:
        IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
        while len(IR_node_0.input):
            IR_node_0.input.pop()
        IR_node_0.name = 'dvtensor_' + source_node.name + '_' + source_node.initializers[0]
        inp_shape = tuple(self.shape_dict[self.src_graph.get_parent(source_node.name, [0]).name])
        out_shape = self._cfirst_to_clast(inp_shape)
        IR_node_0.attr['_output_shapes'].list.shape.pop()
        self._set_output_shape_from_list(IR_node_0, out_shape)
        scale = self.state_dict[source_node.initializers[0] + '_1']
        const_tensor = None
        if inp_shape == scale.shape:
          const_tensor = scale
        else:
          const_tensor = np.broadcast_to(scale,inp_shape).copy()
        self.set_weight(IR_node_0.name, "weights", const_tensor)
        IR_node.input.append(IR_node_0.name)
      return
    if len(source_node.initializers) > 0:
      kwargs = dict()
      scale = self.state_dict[source_node.initializers[0] + '_1']
      parent = self.get_parent(source_node.name, [0])
      par_shape = self.shape_dict[parent.name]
      # First Input shape has atleast 2 dimensions and Second input is a constant (can be constant value or array of constant values with same size as first input channels)
      if not("new_precision" in source_node.attrs) and (len(par_shape) > 1 and (scale.size == 1 or (par_shape[1] in scale.shape and scale.size == par_shape[1]))):
        logger.debug("emitting batchnorm for scalar mul %s", source_node.name)
        IR_node = self._convert_identity_operation(source_node, new_op='BatchNorm')
        node_shape = self.shape_dict[source_node.name]  #NCHW
        new_scale = np.array([scale.tolist()] * node_shape[1]) if scale.size == 1 else scale.flatten()
        self.set_weight(source_node.name, 'scale', new_scale.tolist())
        self.set_weight(source_node.name, 'bias', np.zeros(new_scale.shape))
        self.set_weight(source_node.name, 'mean', np.zeros(new_scale.shape))
        self.set_weight(source_node.name, 'var', np.ones(new_scale.shape))
        IR_node.attr['scale'].b = False
        IR_node.attr['epsilon'].f = 0
        IR_node.attr['use_bias'].b = True
        IR_node.attr['bias'].b = True
        out_shape = self.shape_dict[source_node.name]
        if len(out_shape) > 5:
          out_shape.pop(0)
        out_shape_clast = self._cfirst_to_clast(out_shape)
        IR_node.attr['_output_shapes'].list.shape.pop()
        self._set_output_shape_from_list(IR_node, out_shape_clast)
      elif not("new_precision" in source_node.attrs) and (len(scale.shape) == 1 and (len(par_shape) in [5, 4, 3]) and (par_shape[-1] == scale.shape[0])):
        logger.debug("emitting batchnorm for scalar mul %s", source_node.name)
        #converting to nchw
        IR_node_t1 = self._convert_identity_operation(source_node, new_op='Transpose')
        IR_node_mul = self._convert_identity_operation(source_node, new_op='BatchNorm')
        IR_node_t2 = self._convert_identity_operation(source_node, new_op='Transpose')
        transpose_1_name = 'Transpose_1_' + source_node.layer.name
        mul_name = source_node.layer.name + '_original'
        IR_node_t1.name = transpose_1_name
        IR_node_mul.name = mul_name
        IR_node_mul.input[0] = transpose_1_name
        IR_node_t2.input[0] = mul_name
        t1_shape = IR_node_t1.attr['_output_shapes'].list.shape[0].dim
        mul_shape = IR_node_mul.attr['_output_shapes'].list.shape[0].dim
        t2_shape = IR_node_t2.attr['_output_shapes'].list.shape[0].dim
        t1_shape[-2].size, t1_shape[-1].size = t1_shape[-1].size, t1_shape[-2].size
        mul_shape[-2].size, mul_shape[-1].size = mul_shape[-1].size, mul_shape[-2].size
        rank = len(par_shape)
        kwargs["perm"] = [0] + [rank - 1] + [i for i in range(1, rank - 1)]
        format_dict = {3: 'NCH', 4: 'NCHW', 5: 'NCDHW'}
        kwargs['data_format'] = format_dict[rank]
        assign_IRnode_values(IR_node_t1, kwargs)
        kwargs["perm"] = [0] + [i for i in range(2, rank)] + [1]
        kwargs['data_format'] = format_dict[rank]
        assign_IRnode_values(IR_node_t2, kwargs)
        t2_node = netx.onnx_make_dummy_node(source_node.layer.name, 'Transpose')
        new_onnx_node_id = (source_node.layer.name).split('_')[1] + "_1" if '_' in source_node.layer.name else '1'
        new_onnx_node = OnnxGraphNode(t2_node, new_onnx_node_id)
        self.onnx_graph.layer_map[OnnxGraphNode.mk_node_name(t2_node.name)] = new_onnx_node
        self.shape_dict[source_node.layer.name] = [t2_shape[i].size for i in [0] + [rank - 1] + [i for i in range(1, rank - 1)]]
        self.set_weight(mul_name, 'scale', scale.tolist())
        self.set_weight(mul_name, 'bias', np.zeros(scale.shape))
        self.set_weight(mul_name, 'mean', np.zeros(scale.shape))
        self.set_weight(mul_name, 'var', np.ones(scale.shape))
        IR_node_mul.attr['scale'].b = False
        IR_node_mul.attr['epsilon'].f = 0
        IR_node_mul.attr['use_bias'].b = True
        IR_node_mul.attr['bias'].b = True
      else:
        IR_node = self._convert_identity_operation(source_node, new_op='Mul')
        if "new_precision" in source_node.attrs:
          IR_node.attr['new_precision'].s = source_node.attrs['new_precision']
        const_tensor = scale
        inp_shape = tuple(self.shape_dict[self.src_graph.get_parent(source_node.name, [0]).name])
        pos, k = [], len(const_tensor.shape) - 1
        for i in range(len(inp_shape) - 1, -1, -1):
          if k < 0:
            break
          if const_tensor.shape[k] == inp_shape[i] or (const_tensor.shape[k] == 1 or inp_shape[i] == 1):
            pos.insert(0, i)
            k -= 1
        new_shape = [1] * len(inp_shape)
        for i, j in enumerate(pos):
          new_shape[j] = const_tensor.shape[i]
        const_tensor = np.reshape(const_tensor, new_shape)
        if "new_precision" in source_node.attrs:
          if new_shape[1] != inp_shape[1] and new_shape[2] != inp_shape[2]:
            const_tensor = np.broadcast_to(const_tensor, inp_shape)
            const_tensor = const_tensor[:,1,:,:]
            new_shape = const_tensor.shape
        IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
        while len(IR_node_0.input):
          IR_node_0.input.pop()
        IR_node_0.name = 'dvtensor_' + source_node.name + '_' + source_node.initializers[0]
        out_shape = self._cfirst_to_clast(new_shape)
        IR_node_0.attr['_output_shapes'].list.shape.pop()
        self._set_output_shape_from_list(IR_node_0, out_shape)
        self.set_weight(IR_node_0.name, "weights", const_tensor)
        IR_node.input.append(IR_node_0.name)
    else:
      logger.debug("emitting simple mul %s", source_node.name)
      if "new_precision" in source_node.attrs:
        IR_node.attr['new_precision'].s = source_node.attrs['new_precision']
      IR_node = self._convert_identity_operation(source_node, new_op="Mul")
      out_shape = self.shape_dict[source_node.name]
      if len(out_shape) > 5:
        out_shape.pop(0)
      out_shape_clast = self._cfirst_to_clast(out_shape)
      IR_node.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node, out_shape_clast)

  def rename_GlobalMaxPool(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="ReduceMax")
    attr = source_node.attrs
    layer_name = source_node.name
    output = self.shape_dict[layer_name]
    output_shape = self._cfirst_to_clast(output)
    parent = self.src_graph.get_parent(source_node.name, [0])
    parent_shape = self.shape_dict[parent.name]
    kwargs = {}
    kwargs['axes'] = [1]
    kwargs['keepdims'] = True
    assign_IRnode_values(IR_node, kwargs)

  def rename_Maxpool(self, source_node):
    attr = source_node.attrs
    kwargs = dict()
    kwargs['strides'] = [1] + list(attr['strides']) + [1]
    kwargs['ceil_mode'] = attr.get('ceil_mode', 0)
    if 'dilations' not in attr:
      kwargs['dilations'] = [1] + [1, 1] + [1]
    else:
      kwargs['dilations'] = [1] + list(attr['dilations']) + [1]
    if 'pads' in attr:
      pads = self._repeatedscalar_to_list(attr['pads'])
      if len(pads) == 6:
        kwargs['pads'] = [0] + pads[0:3] + [0, 0] + pads[3:] + [0]
      else:
        kwargs['pads'] = [0] + list(attr['pads'])[0:2] + [0, 0] + list(attr['pads'])[2:] + [0]
    elif 'auto_pad' in source_node.attrs:
      input_node = self.src_graph.get_parent(source_node.name, [0])
      input_node_shape = self.shape_dict[input_node.name]
      if attr['auto_pad'].decode("utf-8") == 'SAME_UPPER':
        kwargs['pads'] = compute_tf_same_padding(input_node_shape, list(attr['kernel_shape']), list(attr['strides']))
    else:
      kwargs['pads'] = [0] * 8
    kwargs['kernel_shape'] = [1] + list(attr['kernel_shape']) + [1]
    input_node1 = self.get_parent(source_node.name, [0])
    input_shape = self.shape_dict[input_node1.name]
    #ncdhw format
    #if any of d,h,w is same and rest is no-op
    if kwargs['kernel_shape'].count(1) == len(kwargs['kernel_shape']) - 1:
      IR_node = self._convert_identity_operation(source_node, new_op="ReduceMax")
      axis = kwargs['kernel_shape'][1:-1].index(np.prod(kwargs['kernel_shape'])) + 2
      kwargs['axes'] = [self._convert_onnx_axis(input_shape, axis)]
    else:
      IR_node = self._convert_identity_operation(source_node, new_op="Pool")
      kwargs['pooling_type'] = 'MAX'
    out_shape = self.shape_dict[source_node.name]
    new_shape = self._cfirst_to_clast(out_shape)
    assign_IRnode_values(IR_node, kwargs)

  def rename_ReduceMean(self, source_node):
    input_node = self.src_graph.get_parent(source_node.name, [0])
    input_node_shape = self.shape_dict[input_node.name]
    # axes
    attr = source_node.attrs
    if 'axes' in attr:
      axes = self._repeatedscalar_to_list(source_node.attrs.get('axes'))
    else:
      axes = self.state_dict[source_node.weights_name]
    axes = [self._convert_onnx_axis(input_node_shape, _a) for _a in axes]

    # All values of axes are spatial (H, W)
    def _pool_candidate():
      if np.all(axes == list(range(1, len(input_node_shape) - 1))):
        return True
      return False

    if source_node.name not in self.dummy_layers:
      if not _pool_candidate():
        IR_node = self._convert_identity_operation(source_node, new_op='ReduceMean')
        kwargs = {}
        kwargs['axes'] = axes
        IR_node.attr['keepdims'].b = source_node.attrs.get('keepdims', 1)
        assign_IRnode_values(IR_node, kwargs)
      else:
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
        IR_node.attr['keepdims'].b = source_node.attrs.get('keepdims', 1)
        assign_IRnode_values(IR_node, kwargs)
    else:
      # ReduceMean
      IR_node = self._convert_identity_operation(source_node, new_op='Tf_Mean')
      # keep dims
      if len(source_node.layer.attribute) > 1:
        IR_node.attr['keep_dims'].b = bool(source_node.layer.attribute[1].i)
      else:
        IR_node.attr['keep_dims'].b = 1
      # axes
      IR_node.attr['axis'].list.i.extend(axes)

  def rename_ReduceMax(self, source_node):
    input_node = self.src_graph.get_parent(source_node.name, [0])
    input_node_shape = self.shape_dict[input_node.name]
    axes = source_node.attrs['axes']
    is_redundant = np.all(np.array([input_node_shape[_a] == 1 for _a in axes]))
    if is_redundant:
      self.rename_Reshape(source_node)
      return
    axes = [self._convert_onnx_axis(input_node_shape, _a) for _a in axes]
    IR_node = self._convert_identity_operation(source_node, new_op='ReduceMax')
    kwargs = {}
    kwargs['axes'] = axes
    kwargs['keepdims'] = source_node.attrs.get('keepdims', 1)
    if "new_precision" in source_node.attrs:
      IR_node.attr['new_precision'].s = source_node.attrs['new_precision']
    assign_IRnode_values(IR_node, kwargs)

  def rename_Not(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Onnx_Not")

  def rename_ReduceSum(self, source_node):
    input_node = self.src_graph.get_parent(source_node.name, [0])
    input_node_shape = self.shape_dict[input_node.name]
    #need to handle non constant case
    if hasattr(source_node, 'axes'):
      original_axes = self._repeatedscalar_to_list(source_node.attrs.get('axes'))
    else:
      original_axes = self.state_dict[source_node.weights_name]
    axes = [0] * len(original_axes)
    for i in range(len(axes)):
      if original_axes[i] < 0:
        axes[i] = len(input_node_shape) + original_axes[i]
      else:
        axes[i] = original_axes[i]
    if False:  #np.all(axes == list(range(2, len(input_node_shape)))):
      IR_node = self._convert_identity_operation(source_node, new_op='PoolReduce')
      kwargs = {}
      # strides
      kwargs['strides'] = [1] * len(input_node_shape)
      # window_shape
      kwargs['kernel_shape'] = [1] * len(input_node_shape)
      # pool type
      kwargs['pooling_type'] = 'SUM'
      kwargs['auto_pad'] = "VALID"
      kwargs['pads'] = [0, 0] * len(input_node_shape)
      IR_node.attr['keepdims'].b = bool(source_node.layer.attribute[1].i)
      assign_IRnode_values(IR_node, kwargs)
    else:
      if source_node.name in self.dummy_layers:
        IR_node = self._convert_identity_operation(source_node, new_op='Tf_Sum')
        # keep dims
        IR_node.attr['keep_dims'].b = bool(source_node.layer.attribute[1].i)
        # axes
        axes = [self._convert_onnx_axis(input_node_shape, _a) for _a in axes]
        IR_node.attr['axis'].list.i.extend(axes)
        assign_IRnode_values(IR_node, kwargs)
      else:
        axes = [self._convert_onnx_axis(input_node_shape, _a) for _a in axes]
        IR_node = self._convert_identity_operation(source_node, new_op='ReduceSum')
        kwargs = {}
        kwargs['axes'] = axes
        IR_node.attr['keepdims'].b = bool(source_node.attrs.get('keepdims', 1))
        assign_IRnode_values(IR_node, kwargs)

  def rename_Mish(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Tf_Mish')
    kwargs = dict()
    kwargs["alpha"] = float(10000000000)
    kwargs["beta"] = float(-10000000000)
    kwargs["gamma"] = float(-10000000000)
    kwargs["delta"] = float(10000000000)
    assign_IRnode_values(IR_node, kwargs)

  def rename_onnx_upsample(self, source_node):
    if source_node.attrs['mode'].decode("utf-8") == 'nearest':
      kwargs = dict()
      IR_node = self._convert_identity_operation(source_node, new_op='Tf_ResizeNearestNeighbor')
      kwargs = dict(align_corners=False, half_pixel_centers=False)
      assign_IRnode_values(IR_node, kwargs)
    elif source_node.attrs['mode'].decode("utf-8") == 'linear':
      kwargs = dict()
      IR_node = self._convert_identity_operation(source_node, new_op='ONNX_UpSample')
      kwargs = dict(align_corners=False, half_pixel_centers=False)
      assign_IRnode_values(IR_node, kwargs)

  def rename_caffe2_upsample(self, source_node):
    input_node = self.get_parent(source_node.name, [0])
    input_shape = self.shape_dict[input_node.name]
    kwargs = dict()
    IR_node = self._convert_identity_operation(source_node, new_op='ResizeNearestNeighbor')
    kwargs['shape'] = self._cfirst_to_clast(input_shape)
    kwargs['mode'] = UPSP_TYPE.NEAREST_4D.value
    assign_IRnode_values(IR_node, kwargs)

  def rename_Upsample(self, source_node):
    input_node = self.get_parent(source_node.name, [0])
    input = self.shape_dict[input_node.name]
    input_shape = self._cfirst_to_clast(input)
    layer_name = source_node.name
    output = self.shape_dict[layer_name]
    output_shape = self._cfirst_to_clast(output)
    scale_h = output_shape[1] / input_shape[1]
    scale_w = output_shape[2] / input_shape[2]
    if source_node.attrs['mode'].decode("utf-8") == 'nearest':
      if math.ceil(scale_h) == math.ceil(scale_w):
        if (math.ceil(scale_h) * input_shape[1] - output_shape[1] in [0, 1] and math.ceil(scale_w) * input_shape[2] - output_shape[2] in [0, 1]):
          scale_h = scale_w = float(math.ceil(scale_h))
      if scale_h == scale_w and scale_h.is_integer():
        if source_node.real_name in self.dummy_layers:
          self.rename_onnx_upsample(source_node)
          return
        input_node = self.get_parent(source_node.name, [0])
        input_shape = self.shape_dict[input_node.name]
        kwargs = dict()
        IR_node = self._convert_identity_operation(source_node, new_op='ResizeNearestNeighbor')
        kwargs['shape'] = self._cfirst_to_clast(input_shape)
        kwargs['mode'] = UPSP_TYPE.NEAREST_4D.value
        kwargs['scale'] = scale_h
        assign_IRnode_values(IR_node, kwargs)
      else:
        self.rename_onnx_upsample(source_node)
        return
    elif source_node.attrs['mode'].decode("utf-8") == 'linear':
      if scale_h == scale_w and scale_h.is_integer():
        input_node = self.get_parent(source_node.name, [0])
        input_shape = self.shape_dict[input_node.name]
        kwargs = dict()
        IR_node = self._convert_identity_operation(source_node, new_op='ResizeBilinear')
        kwargs = dict(align_corners=False, half_pixel_centers=False, shape=self._cfirst_to_clast(input_shape))
        assign_IRnode_values(IR_node, kwargs)
      else:
        raise NotImplementedError("no support for linear mode , for non integer or uneven scales")
    else:
      raise NotImplementedError("mode not implemented - Onnx_Upsampling")

  def rename_ResizeNearest(self, source_node):
    input_node = self.get_parent(source_node.name, [0])
    input = self.shape_dict[input_node.name]
    layer_name = source_node.name
    output = self.shape_dict[layer_name]
    kwargs = dict()
    IR_node = self._convert_identity_operation(source_node, new_op='ResizeNearestNeighbor')
    kwargs['mode'] = UPSP_TYPE.NEAREST_4D.value
    input_shape = self._cfirst_to_clast(input)
    kwargs['shape'] = input_shape
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

  def rename_Onnx_Averagepool(self, source_node):
    attr = source_node.attrs
    kwargs = dict()
    kernel_shape = self._repeatedscalar_to_list(attr['kernel_shape'])
    if len(kernel_shape) < 2:
      kwargs['kernel_shape'] = kernel_shape[0] * 2
    else:
      kwargs['kernel_shape'] = kernel_shape
    if 'strides' in attr:
      strides = self._repeatedscalar_to_list(attr['strides'])
      kwargs['strides'] = strides
    else:
      kwargs['strides'] = [1] * len(kwargs['kernel_shape'])
    if 'dilations' not in attr:
      kwargs['dilations'] = [1] + [1, 1] + [1]
    else:
      kwargs['dilations'] = [1] + self._repeatedscalar_to_list(attr['dilations']) + [1]
    if 'pads' in attr:
      pads = self._repeatedscalar_to_list(attr['pads'])
      kwargs['pads'] = pads
    else:
      kwargs['pads'] = [0] * len(kwargs['kernel_shape']) * 2
    if 'ceil_mode' in attr:
      kwargs['ceil_mode'] = attr['ceil_mode']
    else:
      kwargs['ceil_mode'] = 0
    IR_node = self._convert_identity_operation(source_node, new_op="Onnx_Averagepool")
    assign_IRnode_values(IR_node, kwargs)
    return IR_node

  def rename_Avgpool(self, source_node):
    attr = source_node.attrs
    kwargs = dict()
    if 'pads' in attr:
      pads = self._repeatedscalar_to_list(attr['pads'])
      if len(pads) == 4:
        kwargs['pads'] = [0] + pads[0:2] + [0, 0] + pads[2:] + [0]
      elif len(pads) == 6:
        kwargs['pads'] = [0] + pads[0:3] + [0, 0] + pads[3:] + [0]
    else:
      kwargs['pads'] = [0] * 8
    if (source_node.name in self.dummy_layers) or ('count_include_pad' in attr and attr.get('count_include_pad')==0 and any(kwargs['pads'])):
      return self.rename_Onnx_Averagepool(source_node)
    kwargs['strides'] = [1] + self._repeatedscalar_to_list(attr['strides']) + [1]
    if 'dilations' not in attr:
      kwargs['dilations'] = [1] + [1, 1] + [1]
    else:
      kwargs['dilations'] = [1] + self._repeatedscalar_to_list(attr['dilations']) + [1]
    kwargs['kernel_shape'] = [1] + self._repeatedscalar_to_list(attr['kernel_shape']) + [1]
    kwargs['pooling_type'] = 'AVG'
    kwargs['ceil_mode'] = attr.get('ceil_mode', 0)
    IR_node = self._convert_identity_operation(source_node, new_op="Pool")
    #kwargs['pooling_type'] = 'AVG'
    # out_shape = self.shape_dict[source_node.name]
    # new_shape = self._cfirst_to_clast(out_shape)
    # IR_node.attr['_output_shapes'].list.shape.pop()
    # self._set_output_shape_from_list(IR_node, new_shape)
    assign_IRnode_values(IR_node, kwargs)

  def rename_Flatten(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="Reshape")

  def rename_FullyConnected(self, source_node):
    if len(source_node.initializers) != 0:
      # if self.shape_dict[source_node.name][0] > 1:
      #     self.rename_onnx_batched_inrp(source_node)
      #     return

      attr = source_node.attrs
      IR_node = self._convert_identity_operation(source_node, new_op="FullyConnected")

      weights_name = source_node.weights_name + '_1'
      bias_name = source_node.weights_name + '_2'
      # weight = self.state_dict[weights_name]

      W = self.state_dict[weights_name]
      transB = attr.get('transB', False)
      transA = attr.get('transA', False)

      if transA:
        raise NotImplementedError("not implemented transA in %s", source_node.name)
      if transB:
        W = W.transpose()

      output_channels = W.shape[-1]
      # weights
      self.set_weight(source_node.name, 'weights', W)
      # use_bias
      if bias_name in self.state_dict:
        IR_node.attr['use_bias'].b = True
        bias = self.state_dict[bias_name]
        self.set_weight(source_node.name, 'bias', bias)
      else:
        IR_node.attr['use_bias'].b = False

      # units
      IR_node.attr['units'].i = output_channels
      if 'flatten' in source_node.attrs:
        IR_node.attr['flatten'].b = True if source_node.attrs['flatten'] == 1 else False
      if 'reshape' in source_node.attrs:
        IR_node.attr['reshape'].list.i.extend(source_node.attrs['reshape'])
      self._set_output_shape(source_node, IR_node, True)
    elif source_node.attrs.get('alpha') == 1 and source_node.attrs.get('beta') == 0 and len(source_node.in_edges) == 2:
      IR_node = self._convert_identity_operation(source_node, new_op='BatchDot')
      transA = 'True' if source_node.attrs.get('transA') == 1 else 'False'
      transB = 'True' if source_node.attrs.get('transB') == 1 else 'False'
      kwargs = {"transpose_b": transB, "transpose_a": transA}
      assign_IRnode_values(IR_node, kwargs)

  def rename_Dropout(self, source_node):
    source_node.real_name = self.src_graph.get_node(source_node.in_edges[0]).real_name

  def rename_Concat(self, source_node):
    attrs = source_node.attrs
    IR_node = self._convert_identity_operation(source_node, new_op='Concat')
    if 'axis' in attrs:
      axis = OnnxParser._convert_axis(IR_node, attrs['axis'])
      IR_node.attr["axis"].i = axis
    else:
      raise AttributeError('axis not found')
    if len(source_node.initializers) > 0:
      inps = source_node.layer.input
      inits = source_node.initializers
      for i, j in enumerate(inps):
        if j in inits:
          IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
          while len(IR_node_0.input):
            IR_node_0.input.pop()
          IR_node_0.name = 'dvtensor_' + source_node.name + '_' + j
          out_shape = self.state_dict[j].shape
          out_shape = self._cfirst_to_clast(out_shape)
          IR_node_0.attr['_output_shapes'].list.shape.pop()
          self._set_output_shape_from_list(IR_node_0, out_shape)

          self.set_weight(IR_node_0.name, "weights", self.state_dict[j])
          IR_node.input.insert(i, IR_node_0.name)
    if "new_precision" in source_node.attrs:
      IR_node.attr['new_precision'].s = source_node.attrs['new_precision']

  def rename_Add(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Add')

  def rename_Min(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Min')

  def rename_Floor(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Floor')
  
  def rename_Sum(self, source_node):
    if "new_precision" in source_node.attrs:
      self.rename_Sum_16(source_node)
      return
    # If the inputs are DVCast or DequantizeLinear, map to existing VPU add
    input_node = self.get_parent(source_node.name, [0])
    if input_node.type == "DequantizeLinear":
      IR_node = self._convert_identity_operation(source_node, new_op='Mxnet_Add_N')
      return
    if len(source_node.in_edges) == 2 and len(source_node.initializers) == 0:
      IR_node = self._convert_identity_operation(source_node, new_op='Add')
    if len(source_node.initializers) > 0:
      # if np.all(self.state_dict[source_node.initializers[0] + '_1'] == 0):
      #   # Mapping to Broadcast
      #   logger.info("Mapping to Broadcast")
      #   IR_node = self._convert_identity_operation(source_node, new_op='Broadcast')
      #   return
      kwargs = dict()
      scale = self.state_dict[source_node.initializers[0] + '_1']
      parent = self.get_parent(source_node.name, [0])
      if parent is None:
        parent = self.get_parent(source_node.name, [1])
      par_shape = self.shape_dict[parent.name]
      if (len(scale.shape) == 1) and (par_shape[-1] == scale.shape[0]) and (len(par_shape) in [5, 4, 3]):
        logger.debug("emitting batchnorm for scalar add %s", source_node.name)
        #converting to nchw
        IR_node_t1 = self._convert_identity_operation(source_node, new_op='Transpose')
        IR_node_add = self._convert_identity_operation(source_node, new_op='BatchNorm')
        IR_node_t2 = self._convert_identity_operation(source_node, new_op='Transpose')
        transpose_1_name = 'Transpose_1_' + source_node.layer.name
        add_name = source_node.layer.name + '_original'
        IR_node_t1.name = transpose_1_name
        IR_node_add.name = add_name
        IR_node_add.input[0] = transpose_1_name
        IR_node_t2.input[0] = add_name
        t1_shape = IR_node_t1.attr['_output_shapes'].list.shape[0].dim
        add_shape = IR_node_add.attr['_output_shapes'].list.shape[0].dim
        t2_shape = IR_node_t2.attr['_output_shapes'].list.shape[0].dim
        t1_shape[-2].size, t1_shape[-1].size = t1_shape[-1].size, t1_shape[-2].size
        add_shape[-2].size, add_shape[-1].size = add_shape[-1].size, add_shape[-2].size
        rank = len(par_shape)
        kwargs["perm"] = [0] + [rank - 1] + [i for i in range(1, rank - 1)]
        format_dict = {3: 'NCH', 4: 'NCHW', 5: 'NCDHW'}
        kwargs['data_format'] = format_dict[rank]
        assign_IRnode_values(IR_node_t1, kwargs)
        kwargs["perm"] = [0] + [i for i in range(2, rank)] + [1]
        kwargs['data_format'] = format_dict[rank]
        assign_IRnode_values(IR_node_t2, kwargs)
        t2_node = netx.onnx_make_dummy_node(source_node.layer.name, 'Transpose')
        new_onnx_node_id = (source_node.layer.name).split('_')[1] + "_1" if '_' in source_node.layer.name else '1'
        new_onnx_node = OnnxGraphNode(t2_node, new_onnx_node_id)
        self.onnx_graph.layer_map[OnnxGraphNode.mk_node_name(t2_node.name)] = new_onnx_node
        self.shape_dict[source_node.layer.name] = [t2_shape[i].size for i in [0] + [rank - 1] + [i for i in range(1, rank - 1)]]
        # scale = -1*scale
        self.set_weight(add_name, 'scale', np.ones(scale.shape))
        IR_node_add.attr['scale'].b = False
        IR_node_add.attr['epsilon'].f = 0
        self.set_weight(add_name, 'bias', scale.tolist())
        IR_node_add.attr['use_bias'].b = True
        IR_node_add.attr['bias'].b = True
        self.set_weight(add_name, 'mean', np.zeros(scale.shape))
        self.set_weight(add_name, 'var', np.ones(scale.shape))
      elif ((len(scale.shape) == 1) and len(par_shape) > 1 and (par_shape[1] == scale.shape[0])) or (len(par_shape) > 1 and scale.size == 1):  # eltwise mul with scalar value
        logger.debug("emitting batchnorm for scalar add %s", source_node.name)
        IR_node = self._convert_identity_operation(source_node, new_op='BatchNorm')
        a = self.src_graph.get_parent(source_node.name, [0])
        shape = self.shape_dict[a.name]
        val = self.state_dict[source_node.initializers[0]]
        scalar_tensor = -1 * val
        # epsilon
        IR_node.attr['epsilon'].f = 0
        IR_node.attr['bias'].b = False
        IR_node.attr['scale'].b = False
        variance = np.ones(shape[1])
        # mean
        self.set_weight(source_node.name, "mean", scalar_tensor)
        # var
        self.set_weight(source_node.name, "var", variance)
      elif len(par_shape) > 1 and par_shape[1] in scale.shape and scale.size == par_shape[1]:  # eltwise mul with scalar value
        logger.debug("emitting batchnorm for scalar add %s", source_node.name)
        IR_node = self._convert_identity_operation(source_node, new_op='BatchNorm')
        a = self.src_graph.get_parent(source_node.name, [0])
        shape = self.shape_dict[a.name]
        val = scale
        scalar_tensor = (-1 * val.flatten()).tolist()
        # epsilon
        IR_node.attr['epsilon'].f = 0
        IR_node.attr['bias'].b = False
        IR_node.attr['scale'].b = False
        variance = np.ones(shape[1])
        # mean
        self.set_weight(source_node.name, "mean", scalar_tensor)
        # var
        self.set_weight(source_node.name, "var", variance)
      else:
        IR_node = self._convert_identity_operation(source_node, new_op='Add')
        const_tensor = scale
        inp_shape = tuple(self.shape_dict[self.src_graph.get_parent(source_node.name, [0]).name])
        pos, k = [], len(const_tensor.shape) - 1
        for i in range(len(inp_shape) - 1, -1, -1):
          if k < 0:
            break
          if const_tensor.shape[k] == inp_shape[i] or (const_tensor.shape[k] == 1 or inp_shape[i] == 1):
            pos.insert(0, i)
            k -= 1
        new_shape = [1] * len(inp_shape)
        for i, j in enumerate(pos):
          new_shape[j] = const_tensor.shape[i]
        const_tensor = np.reshape(const_tensor, new_shape)
        IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
        while len(IR_node_0.input):
          IR_node_0.input.pop()
        IR_node_0.name = 'dvtensor_' + source_node.name + '_' + source_node.initializers[0]
        out_shape = self._cfirst_to_clast(new_shape)
        IR_node_0.attr['_output_shapes'].list.shape.pop()
        self._set_output_shape_from_list(IR_node_0, out_shape)
        self.set_weight(IR_node_0.name, "weights", const_tensor)
        IR_node.input.append(IR_node_0.name)

  def rename_Sum_16(self, source_node):
    if len(source_node.in_edges) == 2 and len(source_node.initializers) == 0:
      IR_node = self._convert_identity_operation(source_node, new_op='Add')
    elif len(source_node.initializers) > 0:
      IR_node = self._convert_identity_operation(source_node, new_op='Add')
      parent = self.get_parent(source_node.name, [0])
      const_tensor = self.state_dict[source_node.initializers[0] + '_1']
      if parent is None:
        pass
      par_shape = self.shape_dict[parent.name]
      tensor_shape = list(const_tensor.shape)
      # Check if the constant is broadcastable and reshape if needed
      while len(tensor_shape) < len(par_shape):
        tensor_shape.insert(0, 1)
      for dim_inp, dim_tensor in zip(par_shape[::-1], tensor_shape[::-1]):
        if not (dim_inp == dim_tensor or dim_tensor == 1):
          logger.error("Tensor is not broadcastable")
      
      const_tensor = const_tensor.reshape(tensor_shape)
      const_tensor = np.broadcast_to(const_tensor, par_shape)
      tensor_shape = const_tensor.shape
        
      IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_0.input):
        IR_node_0.input.pop()
      IR_node_0.name = 'dvtensor_' + source_node.name + '_' + source_node.initializers[0]
      out_shape = self._cfirst_to_clast(tensor_shape)
      IR_node_0.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_0, out_shape)
      self.set_weight(IR_node_0.name, "weights", const_tensor)
      IR_node.input.append(IR_node_0.name)
    if "new_precision" in source_node.attrs:
      IR_node.attr['new_precision'].s = source_node.attrs['new_precision']
  
  def rename_Sub(self, source_node):
    if "new_precision" in source_node.attrs:
      self.rename_Sub_16(source_node)
      return
    # If the inputs are DVCast or DequantizeLinear, map to existing VPU add
    input_node = self.get_parent(source_node.name, [0])
    input_to_input_node = self.get_parent(input_node.name, [0])
    if input_node.type == "DequantizeLinear" or (input_node.type == "Slice" and input_to_input_node is not None and input_to_input_node.type == "DequantizeLinear"):
      IR_node = self._convert_identity_operation(source_node, new_op='vector_sub')
      if len(source_node.initializers) != 0:
        IR_node_dv = self._convert_identity_operation(source_node, new_op='DVTensor')
        IR_node_cast = self._convert_identity_operation(source_node, new_op='DVCast')
        IR_node_dv.name = "dvtensor_" + IR_node_dv.name
        IR_node_cast.name = "dvcast_" + IR_node_cast.name
        while len(IR_node_dv.input):
          IR_node_dv.input.pop()
        while len(IR_node_cast.input):
          IR_node_cast.input.pop()
        IR_node_cast.input.insert(0, IR_node_dv.name)
        if list(source_node.layer.input).index(source_node.initializers[0]) == 0:
          IR_node.input.insert(0, IR_node_cast.name)
        else:
          IR_node.input.insert(1, IR_node_cast.name)
        fp32_weights = self.state_dict[source_node.initializers[0]]
        scale, offset = OnnxParser.calculate_scale_offset(fp32_weights.min(), fp32_weights.max(), 8, True)
        int_weights = np.round(fp32_weights / scale) + offset
        int_weights = np.clip(int_weights, -128, 127)

        # int_weights = np.round(fp32_weights / 1) + 0
        self.set_weight(IR_node_dv.name, "weights", int_weights)
        out_shape = self._cfirst_to_clast(list(fp32_weights.shape))
        IR_node_dv.attr['_output_shapes'].list.shape.pop()
        IR_node_dv.attr['pass_through'].b = True
        self._set_output_shape_from_list(IR_node_dv, out_shape)
        kwargs_dvcast = dict()
        kwargs_dvcast['scale'] = scale
        kwargs_dvcast['offset'] = offset + 128
        kwargs_dvcast['data_fmt'] = 'int8' #str(self.state_dict[source_node.layer.input[2]].dtype)
        kwargs_dvcast['src_precision'] = 'int8'
        kwargs_dvcast['dst_precision'] = 'fp32'
        assign_IRnode_values(IR_node_cast, kwargs_dvcast)
      return
    if len(source_node.in_edges) == 2:
      if len(source_node.initializers) == 0:
        IR_node = self._convert_identity_operation(source_node, new_op='onnx_sub')
        return
    second_input_constant = True
    if list(source_node.layer.input).index(source_node.initializers[0]) == 0:
      second_input_constant = False
    a = self.src_graph.get_parent(source_node.name, [0])
    shape = self.shape_dict[a.name]
    val = self.state_dict[source_node.initializers[0]]
    scalar_tensor = val

    if (len(val.shape) > 2) and (val.shape[1] == shape[1]) and (np.all((np.array(val.shape[2:]) == 1))) and second_input_constant:
      # If no. of channels are equal # If the shape is NxCx1x1....  # Map to Batchnorm with the same values
      scalar_tensor = val
    elif (len(val.shape) > 1) and (val.shape[0] == shape[1]) and (np.all((np.array(val.shape[1:]) == 1))) and second_input_constant:
      scalar_tensor = np.squeeze(val)
    elif (len(val.shape) > 1) and val.size == shape[1] and second_input_constant:
      scalar_tensor = val
    elif len(val.shape) == 1 and second_input_constant:
      # If the input is constant, it has one value in shape. Reshape and map to batchnorm
      scalar_tensor = np.full((1, shape[1], 1, 1), val[0])
    elif len(val.shape) == 0 and val.size == 1 and second_input_constant:
      scalar_tensor = np.full((1, shape[0], 1, 1), val)
    else:
      # mode: 0 -> x - c
      # mode: 1 -> c - x
      mode = 0 if source_node.layer.input[1] in self.state_dict else 1
      IR_node = self._convert_identity_operation(source_node, new_op='Add')
      const_tensor = self.state_dict[source_node.initializers[0]]
      if mode == 0:
        const_tensor = -1 * const_tensor
      inp_shape = tuple(self.shape_dict[self.src_graph.get_parent(source_node.name, [0]).name])
      # const_tensor = np.reshape(const_tensor, (1,) * (len(inp_shape) - len(const_tensor.shape)) + const_tensor.shape)
      const_tensor = np.broadcast_to(const_tensor, inp_shape)
      IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_0.input):
        IR_node_0.input.pop()
      IR_node_0.name = 'dvtensor_' + source_node.name + '_' + source_node.initializers[0]
      out_shape = self._cfirst_to_clast(const_tensor.shape)
      IR_node_0.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_0, out_shape)
      self.set_weight(IR_node_0.name, "weights", const_tensor)
      IR_node.input.insert(1 - mode, IR_node_0.name)
      if mode == 1:
        IR_node_1 = self._convert_identity_operation(source_node, new_op='BatchNorm')
        IR_node_1.attr['epsilon'].f = 0
        IR_node_1.attr['bias'].b = False
        IR_node_1.attr['scale'].b = False
        IR_node_1.name = IR_node_1.name + '_batchnorm'
        self.set_weight(IR_node_1.name, "mean", np.zeros(()))
        self.set_weight(IR_node_1.name, "var", np.ones(()))
        self.set_weight(IR_node_1.name, "scale", np.ones(()) * -1)
        self.set_weight(IR_node_1.name, "bias", np.zeros(()))
        IR_node.input[1] = IR_node_1.name
      return
    IR_node = self._convert_identity_operation(source_node, new_op='BatchNorm')
    attr = source_node.attrs
    # epsilon
    IR_node.attr['epsilon'].f = 0
    IR_node.attr['bias'].b = True
    IR_node.attr['scale'].b = True
    _idx = 0 if len(shape) - 1 == 0 else 1
    variance = np.ones(shape[_idx])
    scale = np.ones(shape[_idx])
    mean = np.zeros(shape[_idx])
    bias = -1 * scalar_tensor
    # mean
    self.set_weight(source_node.name, "mean", mean)
    # var
    self.set_weight(source_node.name, "var", variance)
    self.set_weight(source_node.name, "scale", scale)
    self.set_weight(source_node.name, "bias", bias)

  def rename_BroadcastAddMul(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Fused_Add_Mul_Broadcast')
    weights0 = self.state_dict[source_node.weights_name + '_1']
    weights1 = self.state_dict[source_node.weights_name + '_2']
    new_weights = np.concatenate((weights0, weights1), axis=0)
    self.set_weight(source_node.name, "weights", new_weights)

  
  def rename_Sub_16(self, source_node):
    if len(source_node.in_edges) == 2 and len(source_node.initializers) == 0:
      IR_node = self._convert_identity_operation(source_node, new_op='Sub')
      if "new_precision" in source_node.attrs:
          IR_node.attr['new_precision'].s = source_node.attrs['new_precision']
    elif len(source_node.initializers) > 0:
      # c-x
      # x-c
      if source_node.layer.input[0] in self.state_dict:
        # c - x
        IR_node = self._convert_identity_operation(source_node, new_op='Sub')
        if "new_precision" in source_node.attrs:
          IR_node.attr['new_precision'].s = source_node.attrs['new_precision']
        parent = self.get_parent(source_node.name, [0])
        const_tensor = self.state_dict[source_node.initializers[0] + '_1']
        if parent is None:
          pass
        par_shape = self.shape_dict[parent.name]
        tensor_shape = list(const_tensor.shape)
        # Check if the constant is broadcastable and reshape if needed
        while len(tensor_shape) < len(par_shape):
          tensor_shape.insert(0, 1)
        for dim_inp, dim_tensor in zip(par_shape[::-1], tensor_shape[::-1]):
          if not (dim_inp == dim_tensor or dim_tensor == 1):
            logger.error("Tensor is not broadcastable")
      
        const_tensor = const_tensor.reshape(tensor_shape)
          
        IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
        while len(IR_node_0.input):
          IR_node_0.input.pop()
        IR_node_0.name = 'dvtensor_' + source_node.name + '_' + source_node.initializers[0]
        out_shape = self._cfirst_to_clast(tensor_shape)
        IR_node_0.attr['_output_shapes'].list.shape.pop()
        self._set_output_shape_from_list(IR_node_0, out_shape)
        self.set_weight(IR_node_0.name, "weights", const_tensor)
        IR_node.input.insert(0, IR_node_0.name)
      elif source_node.layer.input[1] in self.state_dict:
        # x - c
        IR_node = self._convert_identity_operation(source_node, new_op='Sub')
        if "new_precision" in source_node.attrs:
          IR_node.attr['new_precision'].s = source_node.attrs['new_precision']
        parent = self.get_parent(source_node.name, [0])
        const_tensor = self.state_dict[source_node.initializers[0] + '_1']
        if parent is None:
          pass
        par_shape = self.shape_dict[parent.name]
        tensor_shape = list(const_tensor.shape)
        # Check if the constant is broadcastable and reshape if needed
        while len(tensor_shape) < len(par_shape):
          tensor_shape.insert(0, 1)
        for dim_inp, dim_tensor in zip(par_shape[::-1], tensor_shape[::-1]):
          if not (dim_inp == dim_tensor or dim_tensor == 1):
            logger.error("Tensor is not broadcastable")
      
        const_tensor = const_tensor.reshape(tensor_shape)
          
        IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
        while len(IR_node_0.input):
          IR_node_0.input.pop()
        IR_node_0.name = 'dvtensor_' + source_node.name + '_' + source_node.initializers[0]
        out_shape = self._cfirst_to_clast(tensor_shape)
        IR_node_0.attr['_output_shapes'].list.shape.pop()
        self._set_output_shape_from_list(IR_node_0, out_shape)
        self.set_weight(IR_node_0.name, "weights", const_tensor)
        IR_node.input.append(IR_node_0.name)

  def rename_Div(self, source_node):
    input_node = self.get_parent(source_node.name, [0])
    input = self.shape_dict[input_node.name]
    input_shape = self._cfirst_to_clast(input)
    def is_power_of_2(val):
      return None
      if val == 0:
        return None
      if val < 0:
        return is_power_of_2(-val)
      if (val & (val - 1)) == 0:
        return math.log2(val)
      else:
        return None

    if len(source_node.initializers) > 0 and source_node.initializers[0] == source_node.layer.input[1]:
      logger.debug("emitting batchnorm for scalar div %s", source_node.name)
      kwargs = dict()
      scale = self.state_dict[source_node.initializers[0] + '_1']
      input_node = self.get_parent(source_node.name, [0])
      if input_node.type == "DequantizeLinear":
        if isinstance(scale.item(), int) or isinstance(scale.item(), float):
          IR_node = self._convert_identity_operation(source_node, new_op='divideVectorByScalar')
          IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
          IR_node_1 = self._convert_identity_operation(source_node, new_op="DVCast")
          while len(IR_node_0.input):
            IR_node_0.input.pop()
          while len(IR_node_1.input):
            IR_node_1.input.pop()
          IR_node_0.name = 'dvtensor_' + source_node.layer.input[0]
          IR_node_1.name = 'dvcast_' + source_node.layer.input[0]
          shape = [1]
          shape = self._cfirst_to_clast(shape)
          IR_node_0.attr['_output_shapes'].list.shape.pop()
          IR_node_0.attr['pass_through'].b = True
          fp32_weights = np.array(scale)
          scale, offset = OnnxParser.calculate_scale_offset(fp32_weights.min(), fp32_weights.max(), 8, True)
          int_weights = np.round(fp32_weights / scale) + offset
          int_weights = np.clip(int_weights, -128, 127)
          self._set_output_shape_from_list(IR_node_0, shape)
          self.set_weight(IR_node_0.name, 'weights', int_weights)
          IR_node_1.input.append(IR_node_0.name)
          IR_node.input.append(IR_node_1.name)
          kwargs_dvcast = dict()
          kwargs_dvcast['scale'] = scale
          kwargs_dvcast['offset'] = offset + 128
          kwargs_dvcast['data_fmt'] = 'int8' #str(self.state_dict[source_node.layer.input[2]].dtype)
          kwargs_dvcast['src_precision'] = 'int8'
          kwargs_dvcast['dst_precision'] = 'fp32'
          assign_IRnode_values(IR_node_1, kwargs_dvcast)
          return
      if scale.size == 1 or (scale.size == input[1] and len(scale.shape) == 1):
        if isinstance(scale.item(), int) or (isinstance(scale.item(), float) and scale.item().is_integer()):
          shift = is_power_of_2(int(scale.item()))
          if shift is not None:
            logger.debug(f"Converting to dvshift")
            IR_node = self._convert_identity_operation(source_node, new_op='DVShift')
            IR_node.attr['shift'].i = int(shift)
            if "new_precision" in source_node.attrs:
              IR_node.attr['new_precision'].s = source_node.attrs['new_precision']
            return
        IR_node = self._convert_identity_operation(source_node, new_op='BatchNorm')
        if "new_precision" in source_node.attrs:
          IR_node.attr['new_precision'].s = source_node.attrs['new_precision']
        var = np.array(scale) * np.array(scale)
        self.set_weight(IR_node.name, 'var', var)
        self.set_weight(IR_node.name, 'mean', 0.0)
        IR_node.attr['epsilon'].f = float('1e-10')
      elif scale.size == input[1] and len(scale.shape) != 1:
        IR_node = self._convert_identity_operation(source_node, new_op='BatchNorm')
        var = 1.0 / np.array(scale)
        self.set_weight(IR_node.name, 'var', 1.0)
        self.set_weight(IR_node.name, 'mean', 0.0)
        self.set_weight(IR_node.name, 'scale', np.squeeze(var))
        self.set_weight(IR_node.name, 'bias', 0.0)
        IR_node.attr['epsilon'].f = float('1e-10')
        if "new_precision" in source_node.attrs:
          IR_node.attr['new_precision'].s = source_node.attrs['new_precision']
      else:
        IR_node = self._convert_identity_operation(source_node, new_op='Mul')
        IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
        while len(IR_node_0.input):
          IR_node_0.input.pop()
        if "new_precision" in source_node.attrs:
          IR_node.attr['new_precision'].s = source_node.attrs['new_precision']
        IR_node_0.name = 'dvtensor_' + source_node.name + '_' + source_node.initializers[0]
        out_shape = self._cfirst_to_clast(scale.shape)
        IR_node_0.attr['_output_shapes'].list.shape.pop()
        self._set_output_shape_from_list(IR_node_0, out_shape)
        self.set_weight(IR_node_0.name, "weights", 1 / scale)
        if "new_precision" in source_node.attrs:
          IR_node.attr['new_precision'].s = source_node.attrs['new_precision']
        IR_node.input.append(IR_node_0.name)
    elif len(source_node.initializers) > 0 and source_node.initializers[0] == source_node.layer.input[0]:
      raise NotImplementedError("not implemented")
    else:
      input_node1 = self.get_parent(source_node.name, [0])
      input_shape = self.shape_dict[input_node1.name]
      input_node = self.get_parent(source_node.name, [1])
      input_shape1 = self.shape_dict[input_node.name]
      if input_shape1 == input_shape:
        IR_node = self._convert_identity_operation(source_node, new_op='Mxnet_Elemwise_Div')
      else:
        IR_node = self._convert_identity_operation(source_node, new_op='Onnx_div')
        kwargs = {"input_shape": list(input_shape), "input_shape1": list(input_shape1)}
        assign_IRnode_values(IR_node, kwargs)

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
  
  def rename_Slice(self, source_node):
    inputs = source_node.layer.input
    input_node = self.get_parent(source_node.name, [0])
    
    # Retrieve Input Shape and Handle > 5D logic (Batch Removal)
    parent_src = self.src_graph.get_parent(source_node.name, [0])
    raw_input_shape = list(self.shape_dict[parent_src.name])
    original_ndim = len(raw_input_shape)
    
    # Check if we need to reduce dimensions
    remove_batch = original_ndim > 5
    
    if remove_batch:
      # Remove batch dimension (index 0)
      input_shape = raw_input_shape[1:]
      ndim = original_ndim - 1
    else:
      input_shape = raw_input_shape
      ndim = original_ndim

    # Determine execution mode
    is_float = 1 if input_node.type == "DequantizeLinear" else 0
    use_onnx_slice = (is_float == 1)
    
    steps = None
    if len(inputs) == 5:
      steps = self.state_dict[inputs[4]]
      if any(s != 1 for s in steps):
        use_onnx_slice = True

    # ---------------------------------------------------------
    # Mode 1: ONNX_Slice
    # ---------------------------------------------------------
    if use_onnx_slice:
      kwargs = {}
      raw_starts = np.array(self.state_dict[inputs[1]])
      raw_ends = np.array(self.state_dict[inputs[2]])
      
      # Determine axes
      if len(inputs) < 4:
        raw_axes = np.arange(original_ndim)
      else:
        raw_axes = np.array(self.state_dict[inputs[3]])
      
      # Normalize negative axes based on ORIGINAL dimension
      raw_axes = np.where(raw_axes < 0, raw_axes + original_ndim, raw_axes)

      # Determine steps
      if steps is None:
        raw_steps = np.ones(len(raw_starts)) if len(inputs) < 5 else np.array(self.state_dict[inputs[4]])
      else:
        raw_steps = np.array(steps)

      # Filter and Shift for > 5D
      if remove_batch:
        # Keep only axes that are NOT 0
        mask = raw_axes != 0
        
        # Apply mask
        axes = raw_axes[mask] - 1 # Shift remaining axes down by 1
        starts = raw_starts[mask]
        ends = raw_ends[mask]
        final_steps = raw_steps[mask]
      else:
        axes = raw_axes
        starts = raw_starts
        ends = raw_ends
        final_steps = raw_steps

      # Clamp ends to new input shape
      for i, axis in enumerate(axes):
        if axis < len(input_shape) and ends[i] > input_shape[axis]:
          ends[i] = max(0, input_shape[axis])

      kwargs["starts"] = starts
      kwargs["ends"] = ends
      kwargs["axes"] = axes
      kwargs["steps"] = final_steps
      kwargs["is_float"] = is_float

      IR_node = self._convert_identity_operation(source_node, new_op="ONNX_Slice")
      assign_IRnode_values(IR_node, kwargs)

    # ---------------------------------------------------------
    # Mode 2: Slice (Standard)
    # ---------------------------------------------------------
    else:
      # 1. Resolve Raw Axes
      if len(inputs) >= 4:
        raw_axes = self.state_dict[inputs[3]]
      elif len(inputs) >= 3:
        raw_axes = list(range(original_ndim))
      else:
        raw_axes = source_node.attrs.get('axes', list(range(original_ndim)))

      # Normalize negative axes based on ORIGINAL dimension
      norm_axes = [ax + original_ndim if ax < 0 else ax for ax in raw_axes]

      # 2. Resolve Raw Starts/Ends
      def get_slice_param(idx, attr_key):
        if len(inputs) >= 3:
          key = inputs[idx]
          return self.state_dict.get(key, self.shape_dict.get(key))
        return source_node.attrs.get(attr_key)

      raw_starts = get_slice_param(1, 'starts')
      raw_ends = get_slice_param(2, 'ends')

      # 3. Filter and Shift for > 5D
      active_axes = []
      active_starts = []
      active_ends = []

      if remove_batch:
        for i, ax in enumerate(norm_axes):
          if ax != 0:
            active_axes.append(ax - 1) # Shift axis
            active_starts.append(raw_starts[i])
            active_ends.append(raw_ends[i])
      else:
        active_axes = norm_axes
        active_starts = raw_starts
        active_ends = raw_ends

      # 4. Prepare Final Arrays (using reduced ndim)
      final_starts = np.zeros(ndim, dtype=np.int32)
      
      # Prepare reduced output shape for 'ends' initialization
      orig_out_shape = list(self.shape_dict[source_node.name])
      if remove_batch:
        orig_out_shape.pop(0)
      
      # Convert layout (cfirst -> clast) on the reduced shape
      # Assuming _cfirst_to_clast handles the logic based on list length
      out_shape_clast = self._cfirst_to_clast(orig_out_shape)
      final_ends = np.array(out_shape_clast, dtype=np.int32)
      
      # We need a mutable list for axes to map them to target layout
      final_mapped_axes = list(active_axes) 

      # 5. Normalize bounds and map axes
      for i, ax_idx in enumerate(active_axes):
        dim_limit = input_shape[ax_idx]
        
        # Normalize Start
        s = active_starts[i]
        if s < 0: s += dim_limit
        s = max(0, min(s, dim_limit))
        
        # Normalize End
        e = active_ends[i]
        if e < 0: e += dim_limit
        e = max(0, min(e, dim_limit))

        # Convert Axis Layout
        # Pass reduced shape (orig_out_shape) to ensure correct mapping for 5D
        target_axis = self._convert_onnx_axis(orig_out_shape, ax_idx)
        
        final_starts[target_axis] = s
        final_ends[target_axis] = e
        final_mapped_axes[i] = target_axis

      kwargs = {
        "starts": final_starts,
        "ends": final_ends,
        "axes": np.array(final_mapped_axes)
      }

      IR_node = self._convert_identity_operation(source_node, new_op='Slice')
      IR_node.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node, out_shape_clast)
      assign_IRnode_values(IR_node, kwargs)

  def rename_L2Norm_SSD(self, source_node):
    parent = self.src_graph.get_parent(source_node.name, [0])
    IR_node = self._convert_identity_operation(source_node, new_op='L2Norm_SSD')
    parent_shape = self.shape_dict[parent.name]
    #scale = self.state_dict[source_node.layer.input[1]]
    scale = self.state_dict[source_node.layer.input[2]]
    self.set_weight(source_node.name, 'scale', scale)
    IR_node.attr['scale'].b = False
    IR_node.attr['epsilon'].f = 0
    self.set_weight(source_node.name, 'bias', np.zeros(scale.shape))
    IR_node.attr['use_bias'].b = True
    IR_node.attr['bias'].b = True
    self.set_weight(source_node.name, 'mean', np.zeros(scale.shape))
    self.set_weight(source_node.name, 'var', np.ones(scale.shape))

  def rename_L2Norm_AC(self, source_node):
    parent = self.src_graph.get_parent(source_node.name, [0])
    parent_shape = self.shape_dict[parent.name]
    IR_node = self._convert_identity_operation(source_node, new_op='L2Norm_AC')
    kwargs = dict()
    kwargs['_input_shape'] = [1, np.prod(parent_shape)]
    kwargs['_output_shape'] = parent_shape
    kwargs['eps'] = source_node.attrs['_eps']
    kwargs['across_spatial'] = False
    assign_IRnode_values(IR_node, kwargs)

  def rename_L2Norm_scale(self, source_node):

    IR_node = self._convert_identity_operation(source_node, new_op='L2Norm_scale')
    kwargs = dict()
    kwargs['scale_after_div'] = source_node.attrs['scale_after_div']
    kwargs['scale_before_clip'] = source_node.attrs['scale_before_clip']
    kwargs['eps'] = source_node.attrs['eps']
    kwargs['across_spatial'] = source_node.attrs['across_spatial']
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
    IR_node = self._convert_identity_operation(source_node, new_op='L2Norm')
    parent_shape = self.shape_dict[parent.name]
    kwargs = {}
    mode_attr = next((i for i in source_node.layer.attribute if i.name == 'across_spatial'), None)
    kwargs['eps'] = source_node.get_attr('eps', '.000001')
    kwargs['per_channel'] = source_node.get_attr('per_channel', False)
    if mode_attr:
      kwargs['across_spatial'] = bool(mode_attr.i)
    assign_IRnode_values(IR_node, kwargs)

  def rename_LayerNorm(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='LayerNorm')
    dim = len(self.shape_dict[source_node.layer.name])
    shape = self.shape_dict[source_node.layer.name]
    if 'axis' in source_node.attrs:
      axes = source_node.attrs['axis']
    else:
      axes = source_node.attrs['axes']

    if axes < 0:
      axes += dim
    kwargs = {}
    kwargs['axis'] = axes

    if 'epsilon' in source_node.attrs:
      kwargs['eps'] = source_node.attrs['epsilon']
    else:
      kwargs['eps'] = source_node.attrs['eps']

    if 'uses_mean' in source_node.attrs:
      kwargs['uses_mean'] = bool(source_node.attrs['uses_mean'])
    assign_IRnode_values(IR_node, kwargs)
    if len(source_node.layer.input) > 1:
      self.set_weight(IR_node.name, 'scale', self.state_dict[source_node.layer.input[1]])
    else:
      self.set_weight(IR_node.name, 'scale', np.ones(shape[axes]))
    if len(source_node.layer.input) > 2:
      self.set_weight(IR_node.name, 'bias', self.state_dict[source_node.layer.input[2]])
    else:
      self.set_weight(IR_node.name, 'bias', np.zeros(shape[axes]))

  def rename_Softmax(self, source_node):
    attrs = source_node.attrs
    IR_node = self._convert_identity_operation(source_node, new_op='Softmax')
    axis = -1
    if 'axis' in attrs:
      axis = attrs['axis']
    axis = OnnxParser._convert_axis(IR_node, axis)
    IR_node.attr["axis"].i = axis

  def rename_PytorchOnnxSoftmax(self, source_node):
    attrs = source_node.attrs
    IR_node = self._convert_identity_operation(source_node, new_op='Onnx_PytorchOnnxSoftmax')
    dim = 1
    if 'dim' in attrs:
      dim = attrs['dim']
      #axis = attrs['axis']
    IR_node.attr["dim"].i = dim

  def rename_Transpose(self, source_node):

    def permute_is_reshape(input_shape, permute_order):
      total_elements = np.prod(input_shape)
      flat_data = np.arange(total_elements)
      tensor = flat_data.reshape(input_shape)
      permuted_tensor = tensor.transpose(permute_order)
      result_flat = permuted_tensor.flatten()
      return np.array_equal(flat_data, result_flat)
    
    input_node = self.get_parent(source_node.name, [0])
    input_shape = self.shape_dict[input_node.name]
    permute_order = source_node.attrs['perm']
    if permute_is_reshape(input_shape, permute_order):
      self.rename_Reshape(source_node)
      return
    
    IR_node = self._convert_identity_operation(source_node, new_op='Transpose')
    if "new_precision" in source_node.attrs:
      IR_node.attr['new_precision'].s = source_node.attrs['new_precision']
    attr = source_node.attrs
    kwargs = {}
    data = attr['perm']
    if len(data) > 5:
      my_input_shape = self.shape_dict[source_node.name]
      trunc_flag = True if sum(my_input_shape[:-5]) == len(my_input_shape) - 5 else False
      if not trunc_flag:
        logger.error("Current Stack is limited to 5 dims, Please check the dims of %s", source_node.name)
      else:
        kwargs['perm'] = [int(d) - (len(data) - 5) for d in data[-5:]]
    else:
      kwargs['perm'] = self._repeatedscalar_to_list(attr['perm'])
    kwargs['data_format'] = 'NCHW'
    assign_IRnode_values(IR_node, kwargs)

  def rename_Centernet_IR1(self, source_node):
    input_node = self.get_parent(source_node.name, [0])
    input_shape = self.shape_dict[input_node.name]
    output_shape = self.shape_dict[source_node.name]
    scale = output_shape[2] / input_shape[2]
    kwargs = dict()
    IR_node = self._convert_identity_operation(source_node, new_op='ResizeNearestNeighbor')
    kwargs['shape'] = self._cfirst_to_clast(input_shape)
    kwargs['mode'] = 2
    kwargs['scale'] = scale
    assign_IRnode_values(IR_node, kwargs)

  def rename_ReduceSum_To_ELTS2(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='ReduceSum_To_ELTS2')
    kwargs = dict()
    kwargs['mul_scalar'] = float(self.state_dict[source_node.layer.input[2].replace('/', '_') + "_1"])
    assign_IRnode_values(IR_node, kwargs)

  def rename_ReduceSum_To_ELTS3(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='ReduceSum_To_ELTS3')

  def rename_Centernet_IR4(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Slice')
    kwargs = dict()
    output_shape = self.shape_dict[source_node.name]
    transpose_shape = self.shape_dict[self.src_graph.get_parent(source_node.name, [0]).name]
    kwargs['starts'] = [0] + [int((transpose_shape[2 + sha] - output_shape[2 + sha]) / 2) for sha in range(len(output_shape[2:]))] + [0]
    logger.info("starts for slice is in IR4 %s", kwargs['starts'])
    assign_IRnode_values(IR_node, kwargs)
    # TODO sandeep: Get the offsets using subtraction in the matcher itself; concat has the hard-coded shape in it
    
  def rename_Identity(self, source_node):
    if len(source_node.out_edges) == 0:
      self.rename_Reshape(source_node)
      return
    if self.src_graph.get_node(source_node.in_edges[0]).type in ['Split', 'DistributeFpnProposals', 'GenerateProposals', 'BoxWithNMSLimit', 'BBoxTransform', 'Onnx_BBoxTransform', 'TopK', 'NMSRtmdet', 'NMSPose']:  #, "RoIAlign"]:
      source_node.real_name = self.src_graph.get_node(source_node.in_edges[0]).name + '_' + source_node.pytorch_layer.input[0]
    else:
      source_node.real_name = self.src_graph.get_node(source_node.in_edges[0]).real_name

  def rename_Cast(self, source_node):
    to_type = source_node.attrs.get('to')
    if to_type == onnx.TensorProto.INT64:
      to_type = 'INT16'
    elif to_type == onnx.TensorProto.INT8:
      to_type = 'INT8'
    else:
      logger.info("Cast layer - [%s] mapped to Identity", source_node.name)
      source_node.real_name = self.src_graph.get_node(source_node.in_edges[0]).real_name
      return
    IR_node = self._convert_identity_operation(source_node)
    kwargs = {'to': to_type}
    assign_IRnode_values(IR_node, kwargs)

  def rename_Swish(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Swish')


  def rename_HardSwish(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='HardSwish')
  
  def rename_Elu(self, source_node):
    attr = source_node.attrs
    IR_node = self._convert_identity_operation(source_node, new_op='Elu')
    kwargs = {}
    kwargs['alpha'] = attr.get('alpha', 1.0)
    assign_IRnode_values(IR_node, kwargs)

  def rename_Gelu(self, source_node):
    attr = source_node.attrs
    IR_node = self._convert_identity_operation(source_node, new_op='Gelu')
    kwargs = {}
    # approximation : 0 -> sigmoid, 1 -> erf
    kwargs['approximation'] = int(attr.get('approximation', 1.0))
    assign_IRnode_values(IR_node, kwargs)

  def rename_Softplus(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Softplus')

  def rename_DetectionBoxOutput(self, source_node):
    IR_node = self.IR_graph.node.add()
    OnnxParser._copy_and_reop(source_node, IR_node, "DetectionOutput_1")
    self.convert_inedge(source_node, IR_node, 0, None)
    with open(r"./priorbox_output.pickle", "rb") as input_file:
      _out = pickle.load(input_file)
    order = _out['output_order']
    pbox_seq = list()
    for n in order:
      pbox_seq.append(_out[n])
    priorbox_out = np.concatenate(pbox_seq, axis=0)
    self.set_weight(IR_node.name, 'weights', priorbox_out)

  def rename_Advertima_IR(self, source_node):
    input_node = self.get_parent(source_node.name, [0])
    input_shape = self.shape_dict[input_node.name]
    output_shape_node = self.get_parent(source_node.name, [1])
    out_shape = self.shape_dict[output_shape_node.name]
    kwargs = dict()
    IR_node = self._convert_identity_operation(source_node, new_op='ResizeNearestNeighbor')
    IR_node.attr['_output_shapes'].Clear()
    output_shape = list()
    output_shape.append(input_shape[0])
    output_shape.append(input_shape[1])
    output_shape.append(out_shape[2])
    output_shape.append(out_shape[3])
    kwargs['shape'] = self._cfirst_to_clast(input_shape)
    kwargs["_output_shapes"] = [list_to_shape(self._cfirst_to_clast(output_shape))]
    kwargs["mode"] = 2
    kwargs["scale"] = int(out_shape[2] / input_shape[2])
    assign_IRnode_values(IR_node, kwargs)

  def rename_MatMul(self, source_node):
    if len(source_node.initializers) == 0:
      if source_node.name in self.dummy_layers:
        IR_node = self._convert_identity_operation(source_node, new_op='Mxnet_Batch_Dot')
      else:
        IR_node = self._convert_identity_operation(source_node, new_op='BatchDot')
      multiplier = source_node.attrs.get("multiplier", 1.0)
      kwargs = {"transpose_b": 'False', "transpose_a": 'False', "multiplier": multiplier}
      assign_IRnode_values(IR_node, kwargs)
      return
    initializer = self.state_dict[source_node.initializers[0]]
    shape = list(initializer.shape)
    if len(shape) == 2:
      source_node.attrs['flatten'] = 0
      self.rename_FullyConnected(source_node)
      return
    else:
      IR_node = self._convert_identity_operation(source_node, new_op='BatchDot')
      inp_node = self.src_graph.get_parent(source_node.name, [0])
      init_pos = 0 if source_node.layer.input[0] in source_node.initializers else 1
      kwargs = {"transpose_b": 'False', "transpose_a": 'False'}
      assign_IRnode_values(IR_node, kwargs)
      inp_shape = self.shape_dict[inp_node.name]
      init_shape = list(initializer.shape)
      init_shape_new = inp_shape[:-len(init_shape)] + init_shape
      initializer = np.broadcast_to(initializer, init_shape_new)
      IR_node_0 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_0.input):
        IR_node_0.input.pop()
      IR_node_0.name = 'dvtensor_' + source_node.name + '_' + source_node.initializers[0]
      out_shape = self._cfirst_to_clast(shape)
      IR_node_0.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_0, out_shape)
      self.set_weight(IR_node_0.name, "weights", initializer)
      # IR_node.attr['const_tensor'].b = True
      IR_node.input.insert(init_pos, IR_node_0.name)

  def rename_ReduceL2(self, source_node):
    """Currently we only support l2 norm where the input
        is a vector - i.e, only 1 dimension has more that
        one element. The eps value for l2 normalization is
        also different from the default offered by caffe."""

    def _is_vector_like(shape):
      return len(list(filter(lambda x: x > 1, shape))) == 1

    parent = self.src_graph.get_parent(source_node.name, [0])
    parent_shape = self.shape_dict[parent.name]
    axes = [1]
    kwargs = dict()
    kwargs['across_spatial'] = 'False'
    if 'axes' in source_node.attrs:
      axes = self._repeatedscalar_to_list(source_node.attrs['axes'])
    if axes == [1, 2] or axes == [2, 3]:
      self.rename_SpacialDist(source_node)
      return

    if isinstance(axes, list) and axes[0] != 1:
      raise NotImplementedError("L2Norm only supported in channel / spatial dims.")
    IR_node = self._convert_identity_operation(source_node, new_op='L2Norm')
    assign_IRnode_values(IR_node, kwargs)

  def rename_ROIAlign(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='ROIAlign')
    # output_layers = []
    for i in range(0, len(source_node.layer.output)):
      base_name = source_node.name
      # IR_node1 = self._convert_identity_operation(source_node, new_op = 'ROIAlign')
      out1 = base_name  #+ '_' + source_node.pytorch_layer.output[i]
      # output_layers.append(out1)
      # IR_node1.name = out1
      # IR_node1.attr["ignore_node"].b = True
    attr = source_node.attrs
    kwargs = {}
    kwargs['spatial_scale'] = attr.get('spatial_scale', 1.0)
    kwargs['sampling_ratio'] = attr.get('sampling_ratio', -1.0)
    if 'output_height' in attr:
      kwargs['pooled_height'] = attr.get('output_height')
    else:
      kwargs["pooled_height"] = attr.get('pooled_h')
    if 'output_width' in attr:
      kwargs['pooled_width'] = attr.get('output_width')
    else:
      kwargs["pooled_width"] = attr.get('pooled_w')
    kwargs['aligned'] = attr.get('aligned', False)
    assign_IRnode_values(IR_node, kwargs)

  def rename_RoiAlignMax(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='RoiAlignMax')
    attr = source_node.attrs
    kwargs = {}
    rois = self._repeatedscalar_to_list(attr['rois'])
    batch_ind = self._repeatedscalar_to_list(attr['batch_indices'])
    batch_idx = batch_ind
    kwargs['num_rois'] = attr["num_rois"]
    kwargs['sampling_ratio'] = attr.get('sampling_ratio', -1.0)
    kwargs['spatial_scale'] = attr.get('spatial_scale', 1.0)
    if self.onnx_model.HasField('producer_name') and self.onnx_model.producer_name.lower() in ['pytorch']:
      logger.info("Assuming aligned=True for the roi layer in corresponding torch implementation. Inline with Detectron repo...")
      _t = 0.5 / kwargs['spatial_scale']
      rois = [r + _t for r in rois]
    kwargs['rois'] = rois
    kwargs['pooled_height'] = attr.get('output_height')
    kwargs['pooled_width'] = attr.get('output_width')
    kwargs['aligned'] = attr.get('aligned', False)
    kwargs['batch_indices'] = batch_idx
    assign_IRnode_values(IR_node, kwargs)

  def rename_onnx_batched_inrp(self, source_node):
    IR_node_0 = self._convert_identity_operation(source_node, new_op='Identity')
    IR_node_0.name = IR_node_0.name + '_reshape'
    input_node = self.get_parent(source_node.name, [0])
    input_shape = [1] * (4 - len(self.shape_dict[input_node.name])) + self.shape_dict[input_node.name]
    kwargs_2 = {"shape": input_shape}
    assign_IRnode_values(IR_node_0, kwargs_2)
    IR_node_1 = self._convert_identity_operation(source_node, new_op='Identity')
    IR_node_1.name = IR_node_1.name + '_perm'
    IR_node = self._convert_identity_operation(source_node, new_op='onnx_batched_inrp')
    inp_node = self.src_graph.get_parent(source_node.name, [0])
    kwargs = {}
    initializer = self.state_dict[source_node.initializers[0]]
    shape = list(initializer.shape)
    #
    # if len(shape) != 4:
    #     shape = [1]*(4-len(shape)) + shape
    kwargs['kernel_shape'] = shape
    if len(kwargs['kernel_shape']) == 2:
      kwargs['kernel_shape'] = [1, 1] + [kwargs["kernel_shape"][-1]] + [kwargs['kernel_shape'][0]]
    kwargs['strides'] = [1] * 4
    kwargs['dilations'] = [1] * 4
    kwargs['pads'] = [0, 0] * 4
    assign_IRnode_values(IR_node, kwargs)
    shape = self.shape_dict[source_node.name] + [1] * (4 - len(self.shape_dict[source_node.name]))
    shape_list = [shape[i] for i in [3, 1, 2, 0]]
    input_shape_list = [input_shape[i] for i in [3, 1, 2, 0]]
    IR_node.attr['_output_shapes'].list.shape.pop()
    IR_node_1.attr['_output_shapes'].list.shape.pop()
    self._set_output_shape_from_list(IR_node, shape_list, True)
    self._set_output_shape_from_list(IR_node_1, input_shape_list, True)
    initializer = initializer.reshape(kwargs['kernel_shape'])
    initializer.transpose(0, 2, 3, 1)
    self.set_weight(source_node.name, 'weights', initializer)
    IR_node_2 = self._convert_identity_operation(source_node, new_op='Identity')
    IR_node_2.name = IR_node_2.name + '_reshape1'
    kwargs_2 = {"shape": self.shape_dict[source_node.name]}
    assign_IRnode_values(IR_node_2, kwargs_2)

  def rename_Abs(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Onnx_Abs')

  def rename_Focus(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Focus')

  def rename_Rope(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Rope')
    if 'is_constant' in source_node.attrs:
      input1_name = source_node.initializers[0]
      input1 = self.state_dict[input1_name]
      IR_node_1 = self._convert_identity_operation(source_node, new_op="DVTensor")
      while len(IR_node_1.input):
        IR_node_1.input.pop()
      IR_node_1.name = 'dvtensor_' + source_node.name + '_' + input1_name
      out_shape = self._cfirst_to_clast(list(input1.shape))
      IR_node_1.attr['_output_shapes'].list.shape.pop()
      self._set_output_shape_from_list(IR_node_1, out_shape)
      self.set_weight(IR_node_1.name, "weights", input1)
      IR_node.input.insert(1, IR_node_1.name)
  
  ####################
  # Helper Functions #
  ####################
  @staticmethod
  def _copy_and_reop(source_node, IR_node, new_op=None):
    if new_op == None:
      new_op = source_node.type
    IR_node.name = source_node.name
    IR_node.op = new_op

  def convert_inedge(self, source_node, IR_node, start_idx=0, end_idx=None):
    if end_idx == None:
      end_idx = len(source_node.in_edges)
    for idx in range(start_idx, end_idx):
      if self.src_graph.get_node(source_node.in_edges[idx]).type in ['Split', 'DistributeFpnProposals', 'GenerateProposals', 'BoxWithNMSLimit', 'BBoxTransform', 'Onnx_BBoxTransform', 'TopK', 'NMSRtmdet', 'NMSPose']:  #, 'RoiAlign']:
        layer_inputs_without_initializers = [x for x in source_node.pytorch_layer.input if x not in source_node.initializers]
        IR_node.input.append(self.src_graph.get_node(source_node.in_edges[idx]).name + '_' + layer_inputs_without_initializers[idx])
      else:
        IR_node.input.append(self.src_graph.get_node(source_node.in_edges[idx]).real_name)

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

  def _convert_identity_operation(self, source_node, start_idx=0, in_edge_count=None, new_op=None, multi_out = False):
    IR_node = self.IR_graph.node.add()
    OnnxParser._copy_and_reop(source_node, IR_node, new_op)
    self.convert_inedge(source_node, IR_node, start_idx, in_edge_count)
    if multi_out == True:
      self._set_output_shape_multi_out(source_node, IR_node, True)
    else:
      self._set_output_shape(source_node, IR_node, True)
    output_layers = []
    base_name = source_node.name
    if source_node.type == 'Dropout':
      __base_name = source_node.name
      node_name = __base_name if isinstance(__base_name, str) else __base_name.decode()
      output_layers.append(node_name)
    # elif multi_out == True:
    #   output_layers = [i for i in source_node.layer.output]
    else:
      for idx, node in enumerate(source_node.layer.output):
        __base_name = source_node.name if len(source_node.layer.output) < 2 else "{}_{}".format(base_name, node)

        node_name = __base_name if isinstance(__base_name, str) else __base_name.decode()
        if idx == 0:
          node_name = source_node.name
        output_layers.append(node_name)
    assign_IRnode_values(IR_node, {'out_layers': output_layers})
    return IR_node

  def _convert_pooling(self, source_node):
    kwargs = dict()
    attr = source_node.attrs
    kwargs['strides'] = [1] + list(source_node.get_attr('stride')) + [1]
    kwargs['dilations'] = [1] + list(source_node.get_attr('dilation')) + [1]
    if 'pads' in attr:
      pads = self._repeatedscalar_to_list(attr['pads'])
      if len(pads == 6):
        kwargs['pads'] = [0] + pads[0:3] + [0, 0] + pads[3:] + [0]
      else:
        kwargs['pads'] = ([0] + pads + [0]) * 2
    else:
      kwargs['pads'] = [0, 0] * len(list(source_node.get_attr('kernel_size')))
    kwargs['kernel_shape'] = [1] + self._repeatedscalar_to_list(source_node.get_attr('kernel_size')) + [1]
    IR_node = self._convert_identity_operation(source_node, new_op="Pool")
    if source_node.name.startswith('Max'):
      kwargs['pooling_type'] = 'MAX'
    elif source_node.name.startswith('Avg'):
      kwargs['pooling_type'] = 'AVG'
    else:
      raise ValueError('Unknown pooling type')
    assign_IRnode_values(IR_node, kwargs)

  def _set_output_shape_from_list(self, IR_node, shape_list, transpose=False):
    shape = graph_pb2.TensorShape()
    shape_pytorch = shape_list
    # new_dim = shape.dim.add()
    # (batch, C, H, W)  & NHWC
    if len(shape_pytorch) == 4 and not transpose:
      for _s in shape_pytorch:
        new_dim = shape.dim.add()
        new_dim.size = _s
    elif len(shape_pytorch) == 4 and transpose:
      new_dim = shape.dim.add()
      if shape_pytorch[0] == 1:
        new_dim.size = shape_pytorch[0]
      else:
        new_dim.size = shape_pytorch[0]
      for index in [2, 3, 1]:
        new_dim = shape.dim.add()
        dim = shape_pytorch[index]
        new_dim.size = dim if dim else -1
    # elif len(shape_pytorch) == 2 and  (transpose == False ):
    #     new_dim = shape.dim.add()
    #     if shape_pytorch[0] == 1:
    #         new_dim.size = shape_pytorch[0]
    #     else:
    #         new_dim.size = shape_pytorch[0]
    #     for _ in range(2):
    #         new_dim = shape.dim.add()
    #         new_dim.size = 1
    #     new_dim = shape.dim.add()
    #     dim = shape_pytorch[1]
    #     new_dim.size = dim if dim else -1
    elif len(shape_pytorch) == 2 and transpose:
      new_dim = shape.dim.add()
      if shape_pytorch[0] == 1:
        new_dim.size = -1
      else:
        new_dim.size = shape_pytorch[0]
      new_dim = shape.dim.add()
      dim = shape_pytorch[1]
      new_dim.size = dim if dim else -1
    else:
      for _s in shape_pytorch:
        new_dim = shape.dim.add()
        new_dim.size = _s
    IR_node.attr["_output_shapes"].list.shape.extend([shape])

  ## implementing custom ops
  def rename_Neg(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Onnx_Neg')

  def rename_Split(self, source_node):
    axis = source_node.get_attr('axis', 0)
    if axis < 0:
      axis += len(self.shape_dict[source_node.name])
    weights_name = source_node.weights_name + '_1'
    if 'split' in source_node.attrs:
      size_splits = self._repeatedscalar_to_list(source_node.attrs['split'])
    elif weights_name in self.state_dict:  #getting split as initializer
      size_splits = self._repeatedscalar_to_list(self.state_dict[weights_name])
    else:
      size_splits = [self.shape_dict[op][axis] for op in source_node.layer.output]
      #raise AttributeError('split not found')
    IR_node = self._convert_identity_operation(source_node, new_op='Onnx_Split')
    kwargs = {'axis': axis, 'split': len(size_splits), 'split_size': size_splits}
    output_layers = []
    for i in range(0, kwargs['split']):
      base_name = source_node.name
      IR_node1 = self._convert_identity_operation(source_node, new_op='Onnx_Split')
      out1 = base_name + '_' + source_node.pytorch_layer.output[i]
      output_layers.append(out1)
      IR_node1.name = out1
      IR_node1.attr["ignore_node"].b = True
    kwargs['output_layers'] = output_layers
    kwargs['out_layers'] = copy.deepcopy(output_layers)
    assign_IRnode_values(IR_node, kwargs)

  def rename_IBN(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='IBN')
    self.weights[source_node.name] = [self.state_dict[source_node.layer.input[1]],\
                                      self.state_dict[source_node.layer.input[2]]]
    kwargs = {}
    kwargs['in_eps'] = next(i for i in source_node.layer.attribute if i.name == 'in_eps').f
    kwargs['bn_eps'] = next(i for i in source_node.layer.attribute if i.name == 'bn_eps').f
    assign_IRnode_values(IR_node, kwargs)

  """def rename_ReduceSum(self, source_node):
        axes = list(source_node.layer.attribute[0].ints)
        keepdims = int(source_node.layer.attribute[1].i)
        if keepdims==1:
            keepdims=True
        else:
            keepdims=False
        #print("### onnx axes ",axes)
        #print("### onnx keepdims ",keepdims)
        IR_node = self._convert_identity_operation(source_node, new_op='ReduceSum')
        kwargs = {
                'axes':axes,
                'keepdims':keepdims
                }
        assign_IRnode_values(IR_node, kwargs)
    def rename_Sqrt(self, source_node):
        print("SQRT ### ",source_node.layer)
        IR_node = self._convert_identity_operation(source_node, new_op='emit_Mxnet_Sqrt')
    def rename_Pow(self,source_node):
        print("POW ### ",source_node.layer)
        IR_node = self._convert_identity_operation(source_node,new_op='emit_Onnx_Pow')
    def rename_Broadcast(self,source_node):
        IR_node = self._convert_identity_operation(source_node,new_op="emit_Onnx_Broadcast")"""

  def rename_BoxWithNMSLimit(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="BoxWithNMSLimit")
    # for idx, name in enumerate(source_node.layer.output):
    #     if name == IR_node.name:
    #         continue
    #     IR_node_new = self._convert_identity_operation(source_node, new_op = 'Identity')
    #     IR_node_new.name = name
    output_layers = []
    for i in range(0, len(source_node.layer.output)):
      base_name = source_node.name
      IR_node1 = self._convert_identity_operation(source_node, new_op='BoxWithNMSLimit')
      out1 = base_name + '_' + source_node.pytorch_layer.output[i]
      output_layers.append(out1)
      IR_node1.name = out1
      IR_node1.attr["ignore_node"].b = True
    kwargs = {}
    kwargs["out_layers"] = output_layers
    kwargs["score_thres"] = source_node.attrs["score_thresh"]
    kwargs["nms"] = source_node.attrs["nms"]
    kwargs["detection_per_im"] = source_node.attrs["detections_per_im"]
    kwargs["soft_nms_enabled"] = source_node.attrs["soft_nms_enabled"]
    kwargs["soft_nms_method"] = source_node.attrs["soft_nms_method"]
    kwargs["soft_nms_sigma"] = source_node.attrs["soft_nms_sigma"]
    kwargs["soft_nms_min_score_thres"] = source_node.attrs["soft_nms_min_score_thres"]
    kwargs["rotated"] = source_node.attrs["rotated"]
    kwargs["cls_agnostic_bbox_reg"] = source_node.attrs["cls_agnostic_bbox_reg"]
    kwargs["input_boxes_include_bg_cls"] = source_node.attrs["input_boxes_include_bg_cls"]
    kwargs["output_classes_include_bg_cls"] = source_node.attrs["output_classes_include_bg_cls"]
    kwargs["legacy_plus_one"] = source_node.attrs["legacy_plus_one"]
    assign_IRnode_values(IR_node, kwargs)

  def rename_DistributeFpnProposals(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op="DistributeFpn")
    output_layers = []
    for i in range(0, len(source_node.layer.output)):
      base_name = source_node.name
      IR_node1 = self._convert_identity_operation(source_node, new_op='DistributeFpn')
      out1 = base_name + '_' + source_node.pytorch_layer.output[i]
      output_layers.append(out1)
      IR_node1.name = out1
      IR_node1.attr["ignore_node"].b = True
    kwargs = {}
    kwargs["out_layers"] = output_layers
    kwargs['roi_canonical_level'] = source_node.attrs['roi_canonical_level']
    kwargs['roi_canonical_scale'] = source_node.attrs['roi_canonical_scale']
    kwargs['roi_max_level'] = source_node.attrs['roi_max_level']
    kwargs['roi_min_level'] = source_node.attrs['roi_min_level']
    kwargs['legacy_plus_one'] = source_node.attrs['legacy_plus_one']
    assign_IRnode_values(IR_node, kwargs)

  def rename_GenerateProposals(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='GenerateProposals')
    output_layers = []
    for i in range(0, len(source_node.layer.output)):
      base_name = source_node.name
      IR_node1 = self._convert_identity_operation(source_node, new_op='GenerateProposals')
      out1 = base_name + '_' + source_node.pytorch_layer.output[i]
      output_layers.append(out1)
      IR_node1.name = out1
      IR_node1.attr["ignore_node"].b = True
    kwargs = {}
    kwargs["out_layers"] = output_layers
    kwargs['spatial_scale'] = source_node.attrs['spatial_scale']
    kwargs['pre_nms_topN'] = source_node.attrs['pre_nms_topN']
    kwargs['post_nms_topN'] = source_node.attrs['post_nms_topN']
    kwargs['nms_thresh'] = source_node.attrs['nms_thresh']
    kwargs['min_size'] = source_node.attrs['min_size']
    kwargs['angle_bound_on'] = source_node.attrs['angle_bound_on']
    kwargs['angle_bound_lo'] = source_node.attrs['angle_bound_lo']
    kwargs['angle_bound_hi'] = source_node.attrs['angle_bound_hi']
    kwargs['clip_angle_thresh'] = source_node.attrs['clip_angle_thresh']
    kwargs['legacy_plus_one'] = source_node.attrs['legacy_plus_one']
    assign_IRnode_values(IR_node, kwargs)
    self.set_weight(source_node.name, 'weights', self.state_dict[source_node.weights_name])

  def rename_CollectRpnProposals(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='CollectRpnProposals')
    kwargs = {}
    kwargs['rpn_max_level'] = source_node.attrs['rpn_max_level']
    kwargs['rpn_min_level'] = source_node.attrs['rpn_min_level']
    kwargs['rpn_post_nms_topN'] = source_node.attrs['rpn_post_nms_topN']
    assign_IRnode_values(IR_node, kwargs)

  def rename_BatchPermutation(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Onnx_BatchPermutation')

  def rename_ATen_group_norm(self, source_node):
    num_group = source_node.attrs["num_groups"]
    """
        IR_node_0 = self._convert_identity_operation(source_node, new_op='Identity')
        IR_node_0.name = IR_node_0.name + '_reshape'
        parent = self.src_graph.get_parent(source_node.name, [0])
        inp_shape = list(self.shape_dict[parent.name])
        _collapsed_feature_shape = np.prod(inp_shape[1:])
        if not (_collapsed_feature_shape/num_group).is_integer(): 
            logger.warn("_collapsed_feature_shape/num_group is not an integer, check logic and params for node %s", source_node.name)
        reshape_0 = [-1, num_group, _collapsed_feature_shape//num_group] 
        kwargs_0 = {"shape": reshape_0}
        assign_IRnode_values(IR_node_0, kwargs_0)
        IR_node_1 = self._convert_identity_operation(source_node, new_op='Identity')
        IR_node_1.name = IR_node_1.name + '_reshape_1'
        inp_shape[0] = -1
        kwargs_1 = {"shape": inp_shape}
        assign_IRnode_values(IR_node_1, kwargs_1)
        """
    IR_node = self._convert_identity_operation(source_node, new_op='GroupNorm')
    attr = source_node.attrs
    IR_node.attr['eps'].f = attr['eps']
    IR_node.attr['groups'].i = num_group
    IR_node.attr['output_mean_var'].b = False
    weights_name = source_node.weights_name + '_1'
    bias_name = source_node.weights_name + '_2'
    mean_name = source_node.weights_name + '_3'
    var_name = source_node.weights_name + '_4'
    if bias_name in self.state_dict:
      beta = self.state_dict[bias_name]
      IR_node.attr['bias'].b = True
    else:
      IR_node.attr['bias'].b = False
    if weights_name in self.state_dict:
      gamma = self.state_dict[weights_name]
      IR_node.attr['scale'].b = True
    else:
      IR_node.attr['scale'].b = False
    if mean_name in self.state_dict:
      mean = self.state_dict[mean_name]
      self.set_weight(source_node.name, "mean", mean)
    if var_name in self.state_dict:
      variance = self.state_dict[var_name]
      self.set_weight(source_node.name, "var", variance)
    if IR_node.attr['scale'].b:
      self.set_weight(source_node.name, "scale", gamma)
    if IR_node.attr['bias'].b:
      self.set_weight(source_node.name, "bias", beta)

  def rename_BBoxTransform(self, source_node):
    # if source_node.real_name not in self.dummy_layers:
    self.rename_dvnet_BBoxTransform(source_node)
    # return
    # IR_node = self._convert_identity_operation(source_node, new_op = 'Onnx_BBoxTransform')
    # for idx, name in enumerate(source_node.layer.output):
    #     if name == IR_node.name:
    #         continue
    #     IR_node_new = self._convert_identity_operation(source_node, new_op = 'Identity')
    #     IR_node_new.name = name
    # kwargs = {}
    # kwargs["out_layers"] = [name for name in source_node.layer.output]
    # kwargs['apply_scale'] = source_node.attrs['apply_scale']
    # kwargs['angle_bound_on'] = source_node.attrs['angle_bound_on']
    # kwargs['angle_bound_lo'] = source_node.attrs['angle_bound_lo']
    # kwargs['angle_bound_hi'] = source_node.attrs['angle_bound_hi']
    # kwargs['clip_angle_thresh'] = source_node.attrs['clip_angle_thresh']
    # kwargs['legacy_plus_one'] = source_node.attrs['legacy_plus_one']
    # kwargs['rotated'] = source_node.attrs['rotated']
    # kwargs['weights'] = self._repeatedscalar_to_list(source_node.attrs['weights'])
    # assign_IRnode_values(IR_node, kwargs)
  def rename_BBoxTransform_detectron2(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='BBoxTransform_detectron2')
    kwargs = {}
    kwargs['scale_clamp'] = source_node.attrs['scale_clamp']
    kwargs['weights'] = self._repeatedscalar_to_list(source_node.attrs['weights'])
    kwargs['clip_param'] = [float(i) for i in source_node.attrs['clip_param']]
    kwargs['clip_bbox'] = source_node.attrs['clip_bbox']
    assign_IRnode_values(IR_node, kwargs)

  def rename_BBoxTransform_static_detectron2(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='BBoxTransform_static_detectron2')
    kwargs = {}
    kwargs['scale_clamp'] = source_node.attrs['scale_clamp']
    kwargs['weights'] = self._repeatedscalar_to_list(source_node.attrs['weights'])
    assign_IRnode_values(IR_node, kwargs)
    widths = self.state_dict[source_node.layer.input[1]]
    heights = self.state_dict[source_node.layer.input[2]]
    ctr_x = self.state_dict[source_node.layer.input[3]]
    ctr_y = self.state_dict[source_node.layer.input[4]]
    self.set_weight(source_node.name, 'widths', widths)
    self.set_weight(source_node.name, 'heights', heights)
    self.set_weight(source_node.name, 'ctr_x', ctr_x)
    self.set_weight(source_node.name, 'ctr_y', ctr_y)

  def rename_dvnet_BBoxTransform(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='BBoxTransform')
    output_layers = []
    for i in range(0, len(source_node.layer.output)):
      base_name = source_node.name
      IR_node1 = self._convert_identity_operation(source_node, new_op='BBoxTransform')
      out1 = base_name + '_' + source_node.pytorch_layer.output[i]
      output_layers.append(out1)
      IR_node1.name = out1
      IR_node1.attr["ignore_node"].b = True
    kwargs = {}
    kwargs["out_layers"] = output_layers
    kwargs['apply_scale'] = source_node.attrs['apply_scale']
    kwargs['angle_bound_on'] = source_node.attrs['angle_bound_on']
    kwargs['angle_bound_lo'] = source_node.attrs['angle_bound_lo']
    kwargs['angle_bound_hi'] = source_node.attrs['angle_bound_hi']
    kwargs['clip_angle_thresh'] = source_node.attrs['clip_angle_thresh']
    kwargs['legacy_plus_one'] = source_node.attrs['legacy_plus_one']
    kwargs['rotated'] = source_node.attrs['rotated']
    kwargs['weights'] = self._repeatedscalar_to_list(source_node.attrs['weights'])
    assign_IRnode_values(IR_node, kwargs)

  def rename_GroupNorm(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='GroupNorm')
    self.weights[source_node.name] = [self.state_dict[source_node.layer.input[1]],\
                                      self.state_dict[source_node.layer.input[2]]]
    kwargs = {}
    kwargs['eps'] = next(i for i in source_node.layer.attribute if i.name == 'eps').f
    kwargs['groups'] = next(i for i in source_node.layer.attribute if i.name == 'groups').i
    assign_IRnode_values(IR_node, kwargs)

  def rename_Expand(self, source_node):
    shape = self.shape_dict[source_node.name]
    shape_set = {i for i in shape}
    if len(shape_set) == 1 and shape_set.pop() == 1:
      source_node.real_name = self.src_graph.get_node(source_node.in_edges[0]).real_name
      return
    par_shape = self.shape_dict[self.src_graph.get_parent(source_node.name, [0]).name]
    if par_shape == shape:
      source_node.real_name = self.src_graph.get_node(source_node.in_edges[0]).real_name
      return
    IR_node = self._convert_identity_operation(source_node, new_op='Broadcast')
    if len(source_node.in_edges) == 2:
      shape_node = self.src_graph.get_parent(source_node.name, [1])
      shape_IR_node = [i for i in self.IR_graph.node if i.name == shape_node.name][0]
      shape_IR_node.attr["ignore_node"].b = True

  def rename_Tile(self, source_node):
    input_pos = 0 if source_node.layer.input[0] in self.shape_dict else 1
    inp_shape = self.shape_dict[source_node.layer.input[input_pos]]
    shape_attr = self.state_dict[source_node.layer.input[1 - input_pos]].tolist()
    if all([i == 1 or j == 1 for i, j in zip(inp_shape, shape_attr)]):
      IR_node = self._convert_identity_operation(source_node, new_op='Broadcast')
      return
    raise NotImplementedError("ONNX parser has unsupported operator [%s] with name [%s]." % (source_node.type, source_node.name))

  def rename_MaxRoiPool(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='ROIPool')
    kwargs = {}
    kwargs['pooled_shape'] = self._repeatedscalar_to_list(source_node.attrs['pooled_shape'])
    kwargs['spatial_scale'] = source_node.attrs['spatial_scale']
    assign_IRnode_values(IR_node, kwargs)

  def rename_LRN(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Onnx_Lrn')
    # alpha
    IR_node.attr["alpha"].f = float(source_node.attrs.get("alpha", "0.0001"))
    # beta
    IR_node.attr["beta"].f = float(source_node.attrs.get("beta", "0.75"))
    # knorm
    IR_node.attr["bias"].f = float(source_node.attrs.get("bias", "1"))
    # nsize
    IR_node.attr["size"].i = int(source_node.attrs.get("size"))

  def rename_Einsum(self, source_node):
    mode = None
    equation = source_node.attrs['equation'].decode("utf-8")
    parent = self.src_graph.get_parent(source_node.name, [0])
    input_shape = self.shape_dict[parent.name]
    if len(equation) == 12 and equation[3] == ',' and equation[7:9] == '->':
      if equation[0] == equation[4] == equation[9]:
        if (equation[1] == equation[10] and equation[5] == equation[11] and equation[2] == equation[6]):
          mode = 'bdot_f_t'
        if (equation[2] == equation[10] and equation[6] == equation[11] and equation[1] == equation[5]):
          mode = 'bdot_t_f'
    elif len(equation) == 14 and equation[4] == ',' and equation[8:10] == '->':
      if equation[0] == equation[5] == equation[10] and equation[1] == equation[7] and equation[6] == equation[11] and equation[2:4] == equation[12:14]:
        if input_shape[0] == 1:
          mode = 'conv_1x1'
    if mode == 'bdot_f_t':
      IR_node = self._convert_identity_operation(source_node, new_op='BatchDot')
      kwargs = {"transpose_a": 'False', "transpose_b": 'True'}
      assign_IRnode_values(IR_node, kwargs)
    if mode == 'bdot_t_f':
      IR_node = self._convert_identity_operation(source_node, new_op='BatchDot')
      kwargs = {"transpose_a": 'True', "transpose_b": 'False'}
      assign_IRnode_values(IR_node, kwargs)
    if mode == 'conv_1x1':
      weights_name = source_node.weights_name + '_1'
      weight_shape = self.state_dict[weights_name].shape
      if not (len(weight_shape) == 4 and weight_shape[-2:] == (1, 1)):
        self.state_dict[weights_name] = np.array(self.state_dict[weights_name]).reshape([weight_shape[1], weight_shape[2], 1, 1])
      
      return self.rename_Conv(source_node)

  def rename_LpNormalization(self, source_node):
    if source_node.attrs['p'] == 1:
      IR_node = self._convert_identity_operation(source_node, new_op='L1Norm')
      axis = source_node.attrs['axis']
      if axis < 0:
        axis += len(self.shape_dict[source_node.name])
      assign_IRnode_values(IR_node, {'axis': axis})
    else:
      raise NotImplementedError("ONNX parser has unsupported operator [%s] with name [%s]." % (source_node.type, source_node.name))

  def rename_L1Norm(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='L1Norm')
    axis = source_node.attrs['_axes']
    if axis < 0:
      axis += len(self.shape_dict[source_node.name])
    assign_IRnode_values(IR_node, {'axis': axis, 'eps': source_node.attrs['_eps']})

  def rename_DVTensor(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='DVTensor')
    attrs = source_node.attrs
    mode = 0 if attrs.get('value', 0) == 0 else 2
    if attrs.get('value', 0) != 0:
      self.set_weight(source_node.name, "weights", np.full(self.shape_dict[source_node.name], attrs.get('value')))
    assign_IRnode_values(IR_node, {'mode': mode})

  def rename_HardSigmoid(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Hardsigmoid')
    alpha = source_node.attrs.get('alpha', 0.2)
    beta = source_node.attrs.get('beta', 0.5)
    assign_IRnode_values(IR_node, {'alpha': alpha, 'beta': beta})

  def rename_Sin(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Sin')
  
  def rename_Cos(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Cos')

  def rename_NMSPose(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='NMSPose', multi_out = True)
    output_layers = []
    for i in range(0, len(source_node.layer.output)):
      base_name = source_node.name
      IR_node1 = self._convert_identity_operation(source_node, new_op='NMSPose')
      out1 = base_name + '_' + source_node.pytorch_layer.output[i]
      output_layers.append(out1)
      IR_node1.name = out1
      IR_node1.attr["ignore_node"].b = True
    assign_IRnode_values(IR_node, {'out_layers': output_layers,
                                   'center_point_box': source_node.attrs['center_point_box'],
                                   'max_output_boxes_per_class': self.state_dict[source_node.layer.input[4]].item(),
                                   'iou_threshold': self.state_dict[source_node.layer.input[5]].item(),
                                   'score_threshold': self.state_dict[source_node.layer.input[6]].item()})
  
  def rename_NMS(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='NMS')
    assign_IRnode_values(IR_node, {'center_point_box': source_node.attrs['center_point_box']})
    assign_IRnode_values(IR_node, {'max_output_boxes_per_class': self.state_dict[source_node.layer.input[2]].item()})
    assign_IRnode_values(IR_node, {'iou_threshold': self.state_dict[source_node.layer.input[3]].item()})
    assign_IRnode_values(IR_node, {'score_threshold': self.state_dict[source_node.layer.input[4]].item()})
  
  def rename_topk(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='topk', multi_out = True)
    output_layers = []
    for i in range(0, len(source_node.layer.output)):
      base_name = source_node.name
      IR_node1 = self._convert_identity_operation(source_node, new_op='topk')
      out1 = base_name + '_' + source_node.pytorch_layer.output[i]
      output_layers.append(out1)
      IR_node1.name = out1
      IR_node1.attr["ignore_node"].b = True
    assign_IRnode_values(IR_node, {'out_layers': output_layers})
    assign_IRnode_values(IR_node, {'axis': source_node.attrs['axis']})
    assign_IRnode_values(IR_node, {'largest': source_node.get_attr('largest', 1)})
    assign_IRnode_values(IR_node, {'sorted': source_node.get_attr('sorted', 1)})
    assign_IRnode_values(IR_node, {'K': self.state_dict[source_node.layer.input[1]][0].item()})


  def rename_NMSRtmdet(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='NMSRtmdet', multi_out=True)
    output_layers = []
    for i in range(0, len(source_node.layer.output)):
      base_name = source_node.name
      IR_node1 = self._convert_identity_operation(source_node, new_op='NMSRtmdet')
      out1 = base_name + '_' + source_node.pytorch_layer.output[i]
      output_layers.append(out1)
      IR_node1.name = out1
      IR_node1.attr["ignore_node"].b = True
    assign_IRnode_values(IR_node, {'out_layers': output_layers})
    assign_IRnode_values(IR_node, {'center_point_box': source_node.attrs['center_point_box']})
    assign_IRnode_values(IR_node, {'max_output_boxes_per_class': self.state_dict[source_node.layer.input[3]].item()})
    assign_IRnode_values(IR_node, {'iou_threshold': self.state_dict[source_node.layer.input[4]].item()})
    assign_IRnode_values(IR_node, {'score_threshold': self.state_dict[source_node.layer.input[5]].item()})
    assign_IRnode_values(IR_node, {'mul_const_input': source_node.attrs['mul_const_input'][0]})
    assign_IRnode_values(IR_node, {'less_input_value': source_node.attrs['less_input_value'][0]})
    assign_IRnode_values(IR_node, {'where_x_value': source_node.attrs['where_x_value'][0]})
    assign_IRnode_values(IR_node, {'gather_indices_val': source_node.attrs['gather_indices_val'][0]})
    assign_IRnode_values(IR_node, {'gather1_indices_val': source_node.attrs['gather1_indices_val'][0]})
    assign_IRnode_values(IR_node, {'concat_const_input': self._repeatedscalar_to_list(source_node.attrs['concat_const_input'])})
    assign_IRnode_values(IR_node, {'concat_input': self._repeatedscalar_to_list(source_node.attrs['concat_input'])})

  def rename_Yolov10nPost(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Yolov10nPost')
    assign_IRnode_values(IR_node, {'split_value': source_node.attrs['split_value']})
    assign_IRnode_values(IR_node, {'k_value': source_node.attrs['k_value']})
  
  def rename_Yolov11Post(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Yolov11Post')
    assign_IRnode_values(IR_node, {'max_detections': source_node.attrs['max_detections']})
    assign_IRnode_values(IR_node, {'confidence_threshold': source_node.attrs['confidence_threshold']})
    assign_IRnode_values(IR_node, {'max_dim': source_node.attrs['max_dim']})
    assign_IRnode_values(IR_node, {'max_boxes_per_class': source_node.attrs['max_boxes_per_class']})
    assign_IRnode_values(IR_node, {'iou_threshold': source_node.attrs['iou_threshold']})
    assign_IRnode_values(IR_node, {'center_point_box': source_node.attrs['center_point_box']})
  
  def rename_Logits(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='Logits')
    assign_IRnode_values(IR_node, {'min0_value': source_node.attrs['min0_value']})
    assign_IRnode_values(IR_node, {'max0_value': source_node.attrs['max0_value']})
    assign_IRnode_values(IR_node, {'min1_value': source_node.attrs['min1_value']})
    assign_IRnode_values(IR_node, {'min2_value': source_node.attrs['min2_value']})
  
  def rename_SpacialDist(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='SpacialDist')
    axes = [2, 3]
    if 'axes' in source_node.attrs:
      axes = self._repeatedscalar_to_list(source_node.attrs.get('axes'))
    assign_IRnode_values(IR_node, {'axes': axes})
  
  def rename_MeanRatio(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='MeanRatio')
    assign_IRnode_values(IR_node, {'eps': source_node.attrs['eps']})
    assign_IRnode_values(IR_node, {'axis': 3})

  def rename_GlobalResponseNorm(self, source_node):
    IR_node = self._convert_identity_operation(source_node, new_op='GlobalResponseNorm')
    gamma = self.state_dict[source_node.layer.input[2]]
    beta = self.state_dict[source_node.layer.input[3]]
    self.set_weight(source_node.name, "scale", gamma)
    self.set_weight(source_node.name, "bias", beta)
    assign_IRnode_values(IR_node, {'axis': 3})
  