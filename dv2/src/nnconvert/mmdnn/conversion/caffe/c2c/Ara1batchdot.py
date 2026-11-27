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

from mmdnn.conversion.caffe.utils import mk_permute_proto
from mmdnn.conversion.caffe.c2c.common import BasePass

from dv_logging import get_logger

logger = get_logger(__name__, 'dvctoc')

import caffe
from caffe import layers as L
from caffe import params as P
from mmdnn.conversion.caffe.c2c.common import BasePass
from dv_logging import get_logger
logger = get_logger(__name__, 'dvctoc')

class Ara1BatchDotPatching(BasePass):
  def __init__(self):
    pass

  def do_transformation(self):
    topsort = self.get_topsort()
    for layer in topsort:
      if layer.type == 'BatchDot':
        predecessors = list(layer.predecessors())
        if len(predecessors) < 2:
          logger.error("couldn't find bottoms of batch dot layer")
          return -1
        parent1 = None
        parent2 = None
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
        permute_a = False
        permute_b = False
        if layer.batch_dot_param.transpose_a == True:
          permute_a = True
          layer.batch_dot_param.transpose_a = 0
        if layer.batch_dot_param.transpose_b == False:
          permute_b = True
          layer.batch_dot_param.transpose_b = 1
        if permute_a:
          assert parent1 is not None
          ndim = len(self.caffe_net.blobs[parent1.top[a_top_index]].data.shape)
          order = [i for i in range(ndim)]
          tmp = order[-1]
          order[-1] = order[-2]
          order[-2] = tmp
          permute_a_layer = self.G.mk_permute_proto(layer.name, order, parent1.top[a_top_index])
          # a_layer.top[a_top_index] = permute_a_layer.bottom[0]
          layer.bottom[0] = permute_a_layer.top[0]
          self.G.add_node(permute_a_layer)
          self.G.remove_edge(parent1, layer)
          self.G.add_edge(parent1, permute_a_layer)
          self.G.add_edge(permute_a_layer, layer)
        if permute_b:
          assert parent2 is not None
          ndim = len(self.caffe_net.blobs[parent2.top[b_top_index]].data.shape)
          order = [i for i in range(ndim)]
          tmp = order[-1]
          order[-1] = order[-2]
          order[-2] = tmp
          permute_b_layer =self.G.mk_permute_proto(layer.name, order, parent2.top[b_top_index])
          # b_layer.top[b_top_index] = permute_b_layer.bottom[0]
          layer.bottom[1] = permute_b_layer.top[0]
          self.G.add_node(permute_b_layer)
          self.G.remove_edge(parent2, layer)
          self.G.add_edge(parent2, permute_b_layer)
          self.G.add_edge(permute_b_layer, layer)
