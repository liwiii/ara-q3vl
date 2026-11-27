'''
Copyright (c) 2018-25, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

import copy
import math
import numpy as np
import caffe
from caffe import layers as L
from caffe import params as P
import caffe.proto.caffe_pb2 as caffe_pb2
from mmdnn.conversion.caffe.utils import get_stride_dim, get_pad_dim, get_kernel_dim, set_kernel_dims, set_pad_dim, get_dilation
from mmdnn.conversion.caffe.c2c.common import BasePass
from dv_logging import get_logger
logger = get_logger(__name__, 'dvctoc')


class DynamicDvtensorPatching(BasePass):
  def __init__(self):
    pass
  def do_transformation(self):
    if self.dynamic or self.dynamic_input:
      logger.info("Adding a hanging dvtensor node . ")
      dvtensor_layer = self.G.mk_dvtensor_proto(list([1, 1, 1, 512]), 0, 0, "dvtensor")
      self.G.add_node(dvtensor_layer)

class DVTensorPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    
    convolutions = self.get_nodes_of_type('Convolution')
    # inrps = self.get_nodes_of_type('InnerProduct')
    dvtensor_candidates = convolutions[:]
    # dvtensor_candidates.extend(inrps)
    CM = {'Convolution': 4, 'InnerProduct': 32}
    for layer in dvtensor_candidates:
      # assume that the convolution takes only one input
      predecessor = layer.predecessors()[0]
      layer_top_name = layer.top[0]
      flag = True
      if layer.type == 'Convolution':
        layer_param = layer.convolution_param
        kernels = get_kernel_dim(layer_param)
        assert kernels is not None
        for i in kernels:
          if i != 1:
            flag = False
            break
      shapes = self.caffe_net.blobs[layer_top_name].shape
      pred_shape = self.caffe_net.blobs[layer.bottom[0]].shape
      kernel_c = pred_shape[1]
      if kernel_c % CM[layer.type] != 0:
        extra_c = (CM[layer.type] - kernel_c % CM[layer.type]) % CM[layer.type]
      else:
        continue
      if kernel_c == 2 and shapes[1] == 2 and layer.type == 'Convolution':
        continue
      dvtensor_shape = list(pred_shape)
      dvtensor_shape[1] = extra_c
      dvtensor_layer = self.G.mk_dvtensor_proto(dvtensor_shape, 0, 0, layer.name)
      concat_layer = self.G.mk_concat_proto([layer.bottom[0], dvtensor_layer.top[0]], layer.name)
      if flag and extra_c:
        self.G.add_node(dvtensor_layer)
        self.G.add_node(concat_layer)
        self.G.add_edge(dvtensor_layer, concat_layer)
        self.G.remove_edge(predecessor, layer)
        self.G.add_edge(predecessor, concat_layer)
        self.G.add_edge(concat_layer, layer)
        self.G.add_replace_edge_proto_new(concat_layer, layer, layer.bottom[0], concat_layer.top[0])
        wt_shape = list(self.inp_wts[layer.name][0].shape)
        wt_shape[1] = extra_c
        x = np.zeros(tuple(wt_shape))
        self.inp_wts[layer.name][0] =\
            np.concatenate((self.inp_wts[layer.name][0],x),axis = 1)

class DVTensorCastPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    precision_enum = {'INT8': 0, 'INT16': 1, 'INT32': 2, 'FP32': 3}
    dvtensors = self.get_nodes_of_type('DVTensor')
    for layer in dvtensors:
      successors = layer.successors()
      if len(successors) == 1 and successors[0].type == 'DVCast':
        succ_to_successors = successors[0].successors()
        if len(succ_to_successors) == 1 and succ_to_successors[0].type == 'Onnx_Gather':
          cast_node = successors[0]
          next_node = succ_to_successors[0]
          src_precision = cast_node.dvcast_param.src_precision
          dst_precision = cast_node.dvcast_param.dst_precision
          if (src_precision == precision_enum['INT8'] or src_precision == precision_enum['INT16']) and (dst_precision == precision_enum['FP32']):
            self.G.remove_edge(layer, cast_node)
            self.G.remove_edge(cast_node, next_node)
            self.G.add_edge(layer, next_node)
            self.G.add_replace_edge_proto(layer, next_node, cast_node.top[0])
            self.G.remove_node(cast_node)
            layer.dvtensor_param.is_float = True;

class DVTensorPermutePatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    dvtensors = self.get_nodes_of_type('DVTensor')
    for layer in dvtensors:
      if len(layer.successors()) == 1 and layer.successors()[0].type == 'Permute' and layer.dvtensor_param.tensortype == caffe_pb2.DVTensorParameter.DVTensorType.WEIGHT:
        perm_layer = layer.successors()[0]
        self.inp_wts[layer.name][0] = np.transpose(self.inp_wts[layer.name][0], perm_layer.permute_param.order)
        successors = list(perm_layer.successors())
        for i in successors:
          self.G.add_edge(layer, i)
          self.G.add_replace_edge_proto_new(layer, i, perm_layer.top[0], layer.top[0])
        self.G.remove_node(perm_layer)
        shape_attr = getattr(getattr(layer, 'dvtensor_param'), 'shape')[0].dim
        shape_attr_ = [i for i in shape_attr]
        shape_len = len(shape_attr)
        for i in range(shape_len):
          shape_attr[i] = shape_attr_[perm_layer.permute_param.order[i]]

class DVTensorShiftPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    dvtensors = self.get_nodes_of_type('DVTensor')
    for layer in dvtensors:
      if len(layer.successors()) == 1 and layer.successors()[0].type == 'DVShift':
        shft_layer = layer.successors()[0]
        if len(shft_layer.get_layer().adj_param.precision) and shft_layer.get_layer().adj_param.precision[0] == 1:
          successors = list(shft_layer.successors())
          layer.adj_param.precision.append(1)
          for i in successors:
            self.G.add_edge(layer, i)
            self.G.add_replace_edge_proto_new(layer, i, shft_layer.top[0], layer.top[0])
          self.G.remove_node(shft_layer)
      elif len(layer.successors()) == 1 and layer.successors()[0].type == 'Reshape' and len(layer.successors()[0].successors()) == 1 and layer.successors()[0].successors()[0].type == 'DVShift':
        shft_layer = layer.successors()[0].successors()[0]
        rshp_layer = layer.successors()[0]
        if len(shft_layer.get_layer().adj_param.precision) and shft_layer.get_layer().adj_param.precision[0] == 1:
          successors = list(shft_layer.successors())
          layer.adj_param.precision.append(1)
          for i in successors:
            self.G.add_edge(rshp_layer, i)
            self.G.add_replace_edge_proto_new(rshp_layer, i, shft_layer.top[0], rshp_layer.top[0])
          self.G.remove_node(shft_layer)
        
