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


class FlattenPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    topsort = self.get_topsort()
    # do_fork = self.fork_sftmx_argmx
    for layer in topsort:
      if layer.type == 'Flatten':
        new_shape = list(self.caffe_net.blobs[layer.top[0]].data.shape)
        new_shape[0] = -1
        new_layer_proto = self.G.mk_reshape_proto(layer.bottom[0], list(new_shape), layer.name)
        new_layer_proto.bottom[0] = layer.bottom[0]
        new_layer_proto.top[0] = layer.top[0]
        new_layer_proto.name = layer.name
        # Add reshape layer to the graph
        self.G.add_node(new_layer_proto)
        # Add edge from predecessor to the reshape layer
        predecessor = list(layer.predecessors())
        for p in predecessor:
          self.G.add_edge(p, new_layer_proto)
        # Add edges from reshape layer to all the successors
        if len(layer.successors()) > 0:
          successor = list(layer.successors())
          for s in successor:
            self.G.add_edge(new_layer_proto, s)
        # Remove the current layer
        self.G.remove_node(layer)
