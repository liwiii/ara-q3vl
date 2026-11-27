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
from mmdnn.conversion.caffe.utils import get_stride_dim, get_pad_dim, get_kernel_dim, set_pad_dim
from mmdnn.conversion.caffe.c2c.common import BasePass
from dv_logging import get_logger

logger = get_logger(__name__, 'dvctoc')


class ReductionPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    reductions = self.get_nodes_of_type('Reduction')
    for layer in reductions:
      shape = self.caffe_net.blobs[layer.bottom[0]].shape
      rank = len(shape)
      #insert reshapes if reduction is on plane axis
      if len(layer.reduction_param.axis) != 1:
        continue
      else:
        axis = list(layer.reduction_param.axis)[0]
      if axis == 2 and rank == 5:
        ishape = list(shape)
        ishape[3] *= ishape[4]
        del ishape[4]

        predecessor = layer.predecessors()[0]
        ireshape = self.G.mk_reshape_proto(predecessor.name, ishape, layer.name)

        for idx, t in enumerate(predecessor.top):
          if t == layer.bottom[0]:
            ireshape.bottom[0] = t
        layer.bottom[0] = ireshape.top[0]
        self.G.add_node(ireshape)
        self.G.add_edge(predecessor, ireshape)
        self.G.add_edge(ireshape, layer)
        self.G.remove_edge(predecessor, layer)
        oshape = list(shape)
        oshape[2] = 1
        successors = list(layer.successors())
        oreshape = self.G.mk_reshape_proto(layer.name, oshape, layer.name)
        oreshape.bottom[0] = layer.top[0]
        self.G.add_node(oreshape)
        self.G.add_edge(layer, oreshape)
        layer.reduction_param.axis[0] = 2
        for suc in successors:
          for idx, bottom in enumerate(suc.bottom):
            if bottom == layer.top[0]:
              suc.bottom[idx] = oreshape.top[0]
              self.G.add_edge(oreshape, suc)
              self.G.remove_edge(layer, suc)
    """
	  #insert permutes if reduction not on last axis
      if axis != rank - 1:
        order = [i for i in range(rank)]
        order[axis] = order[-1]
        order[-1] = axis
        perm = mk_permute_proto(layer.name, order, 0)
        predecessor = layer.predecessors()[0]        
        for idx,t in enumerate(predecessor.top):
          if t == layer.bottom[0]:
            perm.bottom[0] = t
        layer.bottom[0] = perm.top[0]
        self.G.add_node(perm)
        self.G.add_edge(predecessor,perm)
        self.G.add_edge(perm,layer)
        self.G.remove_edge(predecessor,layer)
        perm1 = mk_permute_proto(layer.name, order, 1)
        successors = list(layer.successors())
        perm1.bottom[0] = layer.top[0]
        self.G.add_node(perm1)
        self.G.add_edge(layer,perm1)
        layer.reduction_param.axis[0] = rank - 1
        for suc in successors:
          for idx,bottom in enumerate(suc.bottom):
            if bottom == layer.top[0]:
              suc.bottom[idx] = perm1.top[0]
              self.G.add_edge(perm1, suc)
              self.G.remove_edge(layer, suc)
    """


class PoolingPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    pools = self.get_nodes_of_type('Pooling')
    for layer in pools:
      predecessor = layer.predecessors()[0]
      layer_param = layer.pooling_param
      shapes = self.caffe_net.blobs[layer.bottom[0]].shape
      if len(shapes) == 3:
        ran = 1
      if len(shapes) == 4:
        ran = 2
      elif len(shapes) == 5:
        ran = 3
      else:
        logger.error("pooling layer with 2-d input")
        continue
      if layer_param.global_pooling or layer_param.pool == P.Pooling.AVE:
        continue
      kernel = get_kernel_dim(layer_param)
      assert kernel is not None
      if len(list(kernel)) == 1:
        val = kernel[0]
        for i in range(ran - 1):
          kernel.append(val)
      if kernel[0] == shapes[2] and kernel[1] == shapes[3]:
        if (len(kernel) > 2) and kernel[2] == shapes[4]:
          continue
        elif len(kernel) == 2:
          continue
      padding_value = 0
      if layer_param.pool == 0:
        padding_value = -math.inf
      elif layer_param.pool == 2:
        padding_value = math.inf
      out_shape = self.caffe_net.blobs[layer.top[0]].shape
      out_shape = list(out_shape)
      strides = get_stride_dim(layer_param)
      pads = get_pad_dim(layer_param)
      if strides is None:
        strides = [1 for i in range(ran)]
      if pads is None:
        pads = [0 for i in range(ran * 2)]
      if kernel[2:] == shapes[2:] and sum(pads) == 0:
        continue
      if len(list(strides)) == 1:
        val = strides[0]
        for i in range(ran - 1):
          strides.append(val)
      if len(list(pads)) == 1:
        val = pads[0]
        for i in range((ran * 2) - 1):
          pads.append(val)
      elif len(list(pads)) == len(list(kernel)):
        pads = list(pads) + list(pads)
      kernels = list(kernel) * 2
      stride = list(strides) * 2
      expected_pad = []
      for idx, kernel in enumerate(kernels):
        if pads[idx] > kernel // 2 and idx < len(pads) / 2:
          expected_pad.append(kernel // 2)
        else:
          expected_pad.append(pads[idx])
      set_pad_dim(layer.pooling_param, expected_pad)
      if pads != expected_pad:
        padding_layer = self.G.mk_pad_proto(list(np.array(pads) - np.array(expected_pad)), 0, padding_value, layer.name)
        # add the padding layer
        self.G.add_node(padding_layer)
        # insert the padding layer before the pooling layer
        self.G.remove_edge(predecessor, layer)
        self.G.add_edge(predecessor, padding_layer)
        self.G.add_edge(padding_layer, layer)
        self.G.add_replace_edge_proto_new(padding_layer, layer, predecessor.name, padding_layer.top[0])
        self.G.add_replace_edge_proto_new(predecessor, padding_layer, "NA", predecessor.top[0])
