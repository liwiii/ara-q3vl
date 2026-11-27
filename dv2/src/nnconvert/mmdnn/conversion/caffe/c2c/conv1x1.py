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

from mmdnn.conversion.caffe.utils import *

from mmdnn.conversion.caffe.c2c.common import BasePass

from dv_logging import get_logger

logger = get_logger(__name__, 'dvctoc')

skip_node = ['BatchNorm', 'Scale', 'ReLU', 'PReLU', 'Sigmoid', 'Mish', 'Swish', 'HardSwish', 'ReLU6']


# move to utils
def get_prime_factors(number):
  number = float(number)
  factors = []
  print("total factors for number:", number, [*(range(3, int(math.sqrt(number)) + 1, 2))], int(math.sqrt(number)))
  while number % 2 == 0:
    factors.append(2)
    number = number / 2

  for i in range(3, int(math.sqrt(number)) + 1, 2):
    while number % i == 0:
      factors.append(i)
      number = number / i
  print("total factors for number:", number, [*(range(3, int(math.sqrt(number)) + 1, 2))], int(math.sqrt(number)), len(factors), factors)
  if len(factors) == 0:
    print("blah")
  return factors


def change_h_w(shape):
  #abhilash's function

  old_h = shape[-2]
  old_w = shape[-1]

  if old_w >= 32:
    return False, []
  #import pdb; pdb.set_trace()
  factors = get_prime_factors(old_h * old_w)
  if len(factors) == 0:
    return False, []
  if len(factors) == 2:
    # perfect prime square
    new_h = 1
    new_w = old_h * old_w
    # kernel has to be added for these cases with h1m for better perf
  else:
    new_h = factors[0]
    new_w = int((old_h * old_w) / factors[0])

  new_shape = list(shape)[:-2] + [new_h, new_w]
  print("changing dims from ", old_w, old_h, " to ", new_w, new_h)
  return True, new_shape


class Conv1x1Patching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    convs = self.get_nodes_of_type('Convolution')
    parent = None
    for layer in convs:
      predecessors = list(layer.predecessors())
      predecessor = predecessors[0]
      params = layer.convolution_param
      strides = np.array(get_stride_dim(params))
      kernels = np.array(get_kernel_dim(params))
      pads = np.array(get_pad_dim(params))

      if not (np.all(strides == 1) and np.all(kernels == 1) and np.all(pads == 0)):
        continue

      pred_shape = self.get_caffe_blob_shape(layer.bottom[0])
      layer_shape = self.get_caffe_blob_shape(layer.top[0])

      change_hw, new_shape = change_h_w(pred_shape)
      print("checking 1x1 patching")
      if not change_hw:
        continue

      top_index = 0

      for predecessor in predecessors:
        for ind, top in enumerate(predecessor.top):
          if top == layer.bottom[0]:
            parent = predecessor
            top_index = ind
      assert parent is not None
      reshape0 = self.G.mk_reshape_proto(parent.top[top_index], list(new_shape), layer.name)
      self.G.add_node(reshape0)
      self.G.remove_edge(predecessor, layer)
      self.G.add_edge(predecessor, reshape0)
      self.G.add_edge(reshape0, layer)

      layer.bottom[0] = reshape0.top[0]

      curr_node = layer
      successors = list(layer.successors())
      if len(successors):
        while len(successors) == 1 and successors[0].type in skip_node:
          curr_node = successors[0]
          successors = list(successors[0].successors())

      else:
        continue

      reshape1 = self.G.mk_reshape_proto(curr_node.top[0], list(layer_shape), layer.name)

      self.G.add_node(reshape1)
      for successor in successors:
        self.G.remove_edge(curr_node, successor)
        self.G.add_edge(reshape1, successor)
        for ind, bottom in enumerate(successor.bottom):
          if bottom == curr_node.top[0]:
            successor.bottom[ind] = reshape1.top[0]
      self.G.add_edge(curr_node, reshape1)
