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
from mmdnn.conversion.caffe.utils import *
from mmdnn.conversion.caffe.c2c.common import BasePass
from dv_logging import get_logger

logger = get_logger(__name__, 'dvctoc')


class BatchedFCPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    inrps = self.get_nodes_of_type('InnerProduct')
    parent = None
    if self.dynamic is True:
      return
    for layer in inrps:
      predecessors = list(layer.predecessors())
      predecessor = predecessors[0]
      params = layer.inner_product_param
      pred_shape = self.caffe_net.blobs[layer.bottom[0]].shape
      layer_shape = self.caffe_net.blobs[layer.top[0]].shape
      if pred_shape[0] == 1 or len(pred_shape) != 2:
        continue
      pred_shape[0] = -1
      layer_shape[0] = -1
      output_shape = [1] + list(pred_shape) + [1]
      top_index = 0
      for predecessor in predecessors:
        for ind, top in enumerate(predecessor.top):
          if top == layer.bottom[0]:
            parent = predecessor
            top_index = ind
      assert parent is not None
      reshape0 = self.G.mk_reshape_proto(parent.top[0], list(output_shape), layer.name)
      permute0 = self.G.mk_permute_proto(layer.name, [0, 2, 3, 1], reshape0.name)

      self.G.add_node(reshape0)
      self.G.remove_edge(predecessor, layer)
      self.G.add_edge(predecessor, reshape0)
      self.G.add_node(permute0)
      self.G.add_edge(reshape0, permute0)
      conv = self.G.mk_conv_proto(permute0.top[0], layer.name, kernel_size=[1], num_output=params.num_output, bias_term=params.bias_term)
      self.G.add_node(conv)
      self.G.add_edge(permute0, conv)
      permute1 = self.G.mk_permute_proto(layer.name, [0, 3, 1, 2], conv.name)
      reshape1 = self.G.mk_reshape_proto(permute1.top[0], list(layer_shape), layer.name)
      self.inp_wts[layer.name][0] = self.inp_wts[layer.name][0].reshape(list(self.inp_wts[layer.name][0].shape) + [1, 1])
      self.G.add_node(permute1)
      self.G.add_edge(conv, permute1)
      self.G.add_node(reshape1)
      self.G.add_edge(permute1, reshape1)
      successors = list(layer.successors())
      for successor in successors:
        for ind, bottom in enumerate(successor.bottom):
          if bottom == layer.top[0]:
            successor.bottom[ind] = reshape1.top[0]
            self.G.remove_edge(layer, successor)
            self.G.add_edge(reshape1, successor)
      self.G.remove_node(layer)

      # Make sure name of newly added conv layer is same as original conv layer
      self.G.relabel_nodes_locally({f'{conv.type}/{conv.name}': layer.name}, conv.type)
