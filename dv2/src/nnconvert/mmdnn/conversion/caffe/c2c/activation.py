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


class ActivationPatching(BasePass):

  def __init__(self):
    pass

  def move_up_node(self, node):
    layer = copy.deepcopy(node._layer)
    parent = node.predecessors()[0]
    parent_parent = parent.predecessors()[0]
    children = node.successors()
    self.G.remove_node(node)
    for child in children:
      self.G.add_edge(parent, child)
    self.G.add_node(layer)
    self.G.add_edge(layer, parent)
    self.G.remove_edge(parent_parent, parent)
    self.G.add_edge(parent_parent, layer)
    layer.bottom[0] = parent.bottom[0]
    parent.bottom[0] = layer.top[0]
    for child in children:
      for idx in range(len(child.bottom)):
        if child.bottom[idx] == layer.top[0]:
          child.bottom[idx] = parent.top[0]
    # If the activation is output, pass the isOutput flag while swapping
    if node.converter_param.isOutput:
      node.converter_param.isOutput = False
      parent.converter_param.isOutput = True

  def do_patching(self, nodes):
    ignorable_parent = ['Permute', 'Reshape', 'Flatten']
    for node in nodes:
      successors = list(node.successors())
      predecessors = list(node.predecessors())
      while True:
        if len(predecessors) != 1:
          break
        if len(predecessors[0].top) != 1 or predecessors[0].type not in ignorable_parent:
          break
        self.move_up_node(node)
        successors = list(node.successors())
        predecessors = list(node.predecessors())

  def do_transformation(self):
    activations = ['ReLU', 'ReLU6', 'Sigmoid', 'Gelu', 'Swish']
    for act in activations:
      nodes = self.get_nodes_of_type(act)
      self.do_patching(nodes)
