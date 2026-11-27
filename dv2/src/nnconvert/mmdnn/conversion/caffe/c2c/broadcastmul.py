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
from mmdnn.conversion.caffe.c2c.common import BasePass
from dv_logging import get_logger
import numpy as np

logger = get_logger(__name__, 'dvctoc')


class BroadcastMulPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    broadcastmuls = self.get_nodes_of_type('BroadcastMul')
    broadcast = None
    for layer in broadcastmuls:
      if not len(layer.bottom) == 2:
        logger.warn("required two bottom to check and apply broadcast")
        return -1
      pred_shape0 = self.caffe_net.blobs[layer.bottom[0]].shape
      pred_shape1 = self.caffe_net.blobs[layer.bottom[1]].shape
      shape0 = [i for i in pred_shape0]
      shape1 = [i for i in pred_shape1]
      if not len(pred_shape0) == len(pred_shape1):
        logger.warn("cannot broadcast inputs of different shape lengths")
        return -1
      layer_ = layer._layer
      layer_name = layer_.name
      if layer_.type == 'BroadcastMul':
        predecessors = layer.predecessors()
        if len(predecessors) < 2:
          logger.error("couldn't find bottoms of broadcast-mul layer")
          return -1
        parent1 = None
        parent2 = None
        for predecessor in predecessors:
          for ind, top in enumerate(predecessor.top):
            if top == layer.bottom[0]:
              parent1 = predecessor
            elif top == layer.bottom[1]:
              parent2 = predecessor
        broadcast_layer_inputs = layer_.bottom
        broadcast_inputs = [str(input) for input in broadcast_layer_inputs]
        broadcast0 = False
        broadcast1 = False
        final_shape = []
        for i in range(len(shape0)):
          if shape0[i] != shape1[i] and shape0[i] == 1:
            broadcast0 = True
            final_shape.append(shape1[i])
          elif shape0[i] != shape1[i] and shape1[i] == 1:
            broadcast1 = True
            final_shape.append(shape0[i])
          else:
            final_shape.append(shape0[i])
        if broadcast0:
          broad_input = broadcast_inputs[0]
          broadcast = self.G.mk_broadcast_proto(broad_input, layer_name, final_shape)
          layer.bottom[0] = broadcast.top[0]
          self.G.add_node(broadcast)
          self.G.remove_edge(parent1, layer)
          self.G.add_edge(parent1, broadcast)
          self.G.add_edge(broadcast, layer)
        if broadcast1:
          broad_input = broadcast_inputs[1]
          broadcast = self.G.mk_broadcast_proto(broad_input, layer_name, final_shape)
          layer.bottom[1] = broadcast.top[0]
          self.G.add_node(broadcast)
          self.G.remove_edge(parent2, layer)
          self.G.add_edge(parent2, broadcast)
          self.G.add_edge(broadcast, layer)
        eltwise_mul_layer_inputs = layer.bottom
        eltwise_mul_inputs = [str(input) for input in eltwise_mul_layer_inputs]
        eltwise_mul = self.G.mk_eltwise_proto(eltwise_mul_inputs, layer.name, 0)
        param = eltwise_mul.eltwise_param
        if len(param.broadcast0) > 0 or len(param.broadcast1) > 0:
          continue
        else:
          for idx, i in enumerate(final_shape):
            param.broadcast0.append(0)
            param.broadcast1.append(0)
        self.G.add_node(eltwise_mul)
        self.G.add_edge(broadcast, eltwise_mul)
        successors = list(layer.successors())
        for successor in successors:
          for ind, bottom in enumerate(successor.bottom):
            if bottom == layer.top[0]:
              successor.bottom[ind] = eltwise_mul.top[0]
              self.G.add_edge(eltwise_mul, successor)
        self.G.remove_node(layer)
