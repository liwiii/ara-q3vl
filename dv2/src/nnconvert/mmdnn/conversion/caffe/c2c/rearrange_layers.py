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


class Bnorm_Reshape_Scale(BasePass):

  def __init__(self):
    pass

  def is_valid_pattern(self, scale):
    scale_pred = list(scale.predecessors())
    if len(scale_pred) != 1 or scale_pred[0].type != 'Reshape':
      return False
    scale_pred_2 = list(scale_pred[0].predecessors())
    if len(scale_pred_2) != 1 or scale_pred_2[0].type != 'BatchNorm':
      return False
    return True

  def do_transformation(self):
    scale_layers = self.get_nodes_of_type('Scale')
    for scale in scale_layers:
      if not self.is_valid_pattern(scale):
        continue
      reshape_node = scale.predecessors()[0]
      bnorm_node = reshape_node.predecessors()[0]
      succs = list(scale.successors())
      self.G.remove_edge(bnorm_node, reshape_node)
      self.G.remove_edge(reshape_node, scale)
      self.G.add_edge(bnorm_node, scale)
      self.G.add_edge(scale, reshape_node)
      self.G.add_replace_edge_proto(bnorm_node, scale, reshape_node.top[0])
      self.G.add_replace_edge_proto(scale, reshape_node, bnorm_node.top[0])
      for l in succs:
        self.G.remove_edge(scale, l)
        self.G.add_edge(reshape_node, l)
        self.G.add_replace_edge_proto(reshape_node, l, scale.top[0])
