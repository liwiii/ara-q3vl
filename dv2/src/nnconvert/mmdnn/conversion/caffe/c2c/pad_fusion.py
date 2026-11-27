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

import networkx as nx
from mmdnn.conversion.caffe.c2c.common import BasePass
from mmdnn.conversion.caffe.utils import get_kernel_dim, get_pad_dim, set_pad_dim


class FusePadIntoSuccessor(BasePass):

  def __init__(self):
    pass

  def node_selector(self):
    return ['Padding']

  def conv_k12_explicit_padding_pass(self, successor, layer):
    kernel_size = successor.convolution_param.kernel_size
    conv_pads = successor.convolution_param.pad
    padding_pad = layer.padding_param.pad
    return all([i > 12 for i in kernel_size]) and all([i == 0 for i in conv_pads]) and any([i != 0 for i in padding_pad])

  def do_shift_pads(self, layer_param, pad_params):
    kernel = get_kernel_dim(layer_param)
    pads = get_pad_dim(layer_param)
    pad_layer_pads = get_pad_dim(pad_params)
    kernel = list(kernel) * 2
    if kernel is None:
      return
    for i, val in enumerate(pads):
      if (val >= kernel[i] // 2):
        return
    new_pad_list = []
    for i, val in enumerate(pads):
      assert pad_layer_pads is not None
      new_pad = min(kernel[i] // 2, val + pad_layer_pads[i])
      new_pad_list.append(new_pad)
      pad_layer_pads[i] = max(0, pad_layer_pads[i] - abs(new_pad - val))
    set_pad_dim(layer_param, new_pad_list)

  def do_transformation(self):
    for layer in self.get_nodes_of_type('Padding'):
      successors = list(layer.successors())
      if not len(successors) or len(successors) > 1:
        continue
      successor = successors[0]
      if successor.type in ['Convolution', 'Tflite_Convolution']:
        if self.conv_k12_explicit_padding_pass(successor, layer):
          continue
        self.do_shift_pads(successor.convolution_param, layer.padding_param)
      elif successor.type in ['Pooling','Tflite_Pooling'] and successor.pooling_param.pool == 0:
        self.do_shift_pads(successor.pooling_param, layer.padding_param)
