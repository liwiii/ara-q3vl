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
import caffe
from caffe import layers as L
from caffe import params as P
from mmdnn.conversion.caffe.c2c.common import BasePass
from dv_logging import get_logger
logger = get_logger(__name__, 'dvctoc')


class BatchPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    topsort = self.get_topsort()
    parent1 = None
    parent2 = None
    for layer in topsort:
      if layer.type == 'Mxnet_Take':
        predecessors = list(layer.predecessors())
        if len(predecessors) < 2:
          logger.error("couldn't find bottoms of Mxnet_Take layer")
          return -1
        a_top_index = 0
        b_top_index = 0
        for predecessor in predecessors:
          for ind, top in enumerate(predecessor.top):
            if top == layer.bottom[0]:
              parent1 = predecessor
              a_top_index = ind
            elif top == layer.bottom[1]:
              parent2 = predecessor
              b_top_index = ind
        assert parent1 is not None
        assert parent2 is not None
        input_shape = self.caffe_net.blobs[parent1.top[a_top_index]].data.shape
        idx_shape = self.caffe_net.blobs[parent2.top[b_top_index]].data.shape
        output_shape = self.caffe_net.blobs[layer.top[0]].data.shape
        successors = list(layer.successors())
        if layer.mxnet_take_param.axis == 0 and idx_shape[0] > 1:
          order = [1] + list(input_shape)
          reshape = self.G.mk_reshape_proto(parent1.top[a_top_index], order, layer.name)
          layer.bottom[0] = reshape.top[0]
          layer.mxnet_take_param.axis = layer.mxnet_take_param.axis + 1
          self.G.add_node(reshape)
          self.G.remove_edge(parent1, layer)
          self.G.add_edge(parent1, reshape)
          self.G.add_edge(reshape, layer)
          reshape2 = self.G.mk_reshape_proto(layer.top[0], list(output_shape), layer.name)
          self.G.add_node(reshape2)
          self.G.add_edge(layer, reshape2)
          reshape2.bottom[0] = layer.top[0]
          for successor in successors:
            for ind, bottom in enumerate(successor.bottom):
              if bottom == layer.top[0]:
                successor.bottom[ind] = reshape2.top[0]
                self.G.remove_edge(layer, successor)
                self.G.add_edge(reshape2, successor)
