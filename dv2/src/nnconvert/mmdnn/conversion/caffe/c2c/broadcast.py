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


class BroadcastPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    def is_valid_broadcast_along_height(a, b):
      return (
        len(a) == len(b)
        and a[:-2] == b[:-2]
        and a[-1] == b[-1]
        and (a[-2] == b[-2] or 1 in (a[-2], b[-2]))
      )
    def is_valid_broadcast_along_width(a, b):
      return (
        len(a) == len(b)
        and a[:-1] == b[:-1]
        and (a[-1] == b[-1] or 1 in (a[-1], b[-1]))
      )
    broadcasts = self.get_nodes_of_type('Broadcast')
    for layer in broadcasts:
      parent = list(layer.predecessors())[0]
      parent_shape = list(self.caffe_net.blobs[layer.bottom[0]].shape)
      layer_shape = list(self.caffe_net.blobs[layer.top[0]].shape)
      if is_valid_broadcast_along_height(parent_shape, layer_shape):
        # Add permute before and after broadcast to make broadcast along width
        perm_order = list(range(len(layer_shape)))
        perm_order[-3], perm_order[-2] = perm_order[-2], perm_order[-3]
        permute_b = self.G.mk_permute_proto(layer.name + "0", perm_order, parent.top[0])
        permute_a = self.G.mk_permute_proto(layer.name + "1", perm_order, layer.top[0])
        self.G.add_node(permute_b)
        self.G.add_node(permute_a)
        successors = list(layer.successors())
        for successor in successors:
          self.G.remove_edge(layer, successor)
          self.G.add_edge(permute_a, successor)
          self.G.add_replace_edge_proto(permute_a, successor, layer.top[0])
        self.G.add_edge(layer, permute_a)
        self.G.add_edge(parent, permute_b)
        self.G.add_edge(permute_b, layer)
        self.G.remove_edge(parent, layer)
        self.G.add_replace_edge_proto(permute_b, layer, parent.top[0])
        # Update the param in broadcast
        dims = list(layer.broadcast_param.shape.dim)
        dims[-3], dims[-2] = dims[-2], dims[-3]
        layer.broadcast_param.shape.dim[:] = dims
      elif is_valid_broadcast_along_width(parent_shape, layer_shape):
        # Add permute before and after broadcast to make broadcast along width
        perm_order = list(range(len(layer_shape)))
        perm_order[-3], perm_order[-1] = perm_order[-1], perm_order[-3]
        permute_b = self.G.mk_permute_proto(layer.name + "0", perm_order, parent.top[0])
        permute_a = self.G.mk_permute_proto(layer.name + "1", perm_order, layer.top[0])
        self.G.add_node(permute_b)
        self.G.add_node(permute_a)
        successors = list(layer.successors())
        for successor in successors:
          self.G.remove_edge(layer, successor)
          self.G.add_edge(permute_a, successor)
          self.G.add_replace_edge_proto(permute_a, successor, layer.top[0])
        self.G.add_edge(layer, permute_a)
        self.G.add_edge(parent, permute_b)
        self.G.add_edge(permute_b, layer)
        self.G.remove_edge(parent, layer)
        self.G.add_replace_edge_proto(permute_b, layer, parent.top[0])
        # Update the param in broadcast
        dims = list(layer.broadcast_param.shape.dim)
        dims[-3], dims[-1] = dims[-1], dims[-3]
        layer.broadcast_param.shape.dim[:] = dims

