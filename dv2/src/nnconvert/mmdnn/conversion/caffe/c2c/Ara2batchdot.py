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
import numpy as np
from mmdnn.conversion.caffe.c2c.common import BasePass

from dv_logging import get_logger

logger = get_logger(__name__, 'dvctoc')


class Ara2BatchDotPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    bdots = self.get_nodes_of_type('BatchDot')
    for layer in bdots:
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

      if layer.batch_dot_param.transpose_a == False:
        permute_a = True
        layer.batch_dot_param.transpose_a = 1

      if layer.batch_dot_param.transpose_b == True:
        permute_b = True
        layer.batch_dot_param.transpose_b = 0

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
        permute_b_layer = self.G.mk_permute_proto(layer.name, order, parent2.top[b_top_index])
        # b_layer.top[b_top_index] = permute_b_layer.bottom[0]
        layer.bottom[1] = permute_b_layer.top[0]
        self.G.add_node(permute_b_layer)
        self.G.remove_edge(parent2, layer)
        self.G.add_edge(parent2, permute_b_layer)
        self.G.add_edge(permute_b_layer, layer)


class BatchDotMultiplierPatching(BasePass):

  def __init__(self):
    pass

  def replace_node(self, layer):
    predecessors = list(layer.predecessors())
    successors = list(layer.successors())
    if len(predecessors) > 1:
      return
    if len(successors) > 1:
      return
    if len(successors) == 0:
      self.G.remove_node(layer)
      return
    self.G.add_edge(predecessors[0], successors[0])
    self.G.remove_edge(predecessors[0], layer)
    self.G.remove_edge(layer, successors[0])
    self.G.remove_node(layer)
    for idx, b in enumerate(successors[0].bottom):
      if b == layer.top[0]:
        successors[0].bottom[idx] = predecessors[0].top[0]

  def next_scale(self, layer):
    scale_change = 1
    if len(layer.successors()) != 1:
      return None, 1
    tmp = layer.successors()[0]
    removable_bnrm = None
    while tmp.type in ['Reshape', 'Crop', 'Permute', 'PermuteNnp', 'BatchNorm']:
      if tmp.type == 'BatchNorm':
        if np.all(self.inp_wts[tmp.name][0] == 0) and np.all(self.inp_wts[tmp.name][1] == self.inp_wts[tmp.name][1][0]):
          removable_bnrm = tmp
          if np.all(self.inp_wts[tmp.name][1] != 1):
            scale_change = 1 / np.sqrt(self.inp_wts[tmp.name][1][0])

      if len(tmp.successors()) != 1:
        return None, 1
      tmp = list(tmp.successors())[0]
    if tmp.type == 'Scale':
      if removable_bnrm is not None:
        self.replace_node(removable_bnrm)
      return tmp, scale_change
    return None, None

  def do_transformation(self):
    bdots = self.get_nodes_of_type('BatchDot')
    for layer in bdots:
      scale, scale_change = self.next_scale(layer)
      if scale is not None:
        if np.all(self.inp_wts[scale.name][0] == self.inp_wts[scale.name][0][0]) and np.all(self.inp_wts[scale.name][1] == 0):
          layer.batch_dot_param.multiplier = self.inp_wts[scale.name][0][0] * scale_change
          self.replace_node(scale)