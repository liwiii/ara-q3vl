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
logger = get_logger(__name__, 'DVCTOC')


#
# Add layer_type to list to skip DV_CAST being added.
# Layer types will be marked as "needs_generic_cp" as true, but dataprec will be set to INT8.
# Should be enabled for cp implemented DMA operations.
#
class ATenPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    for layer in self.get_topsort():
      if layer.type == 'Aten':
        layer.needs_generic_cp_kernel = True

class powToMul(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    layers = self.get_nodes_of_type('Aten')
    for layer in layers:
      if layer.get_layer().aten_param.original_layer_type != 'Pow':
        continue
      pred = layer.predecessors()[0]
      eltwise_new = self.G.mk_eltwise_proto([layer.bottom[0], layer.bottom[0]], layer.name, operation = 0)
      self.G.add_node(eltwise_new)
      self.G.add_edge(pred, eltwise_new)
      succs = list(layer.successors())
      for s in succs:
        self.G.add_edge(eltwise_new, s)
        self.G.add_replace_edge_proto_new(eltwise_new, s, layer.top[0], eltwise_new.top[0])
      self.G.remove_node(layer)

