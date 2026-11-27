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


class DeconvPatching(BasePass):

  def __init__(self):
    pass

  def adjust_kernel_dims(self, layer):
    """if the kernel dimensions are 3x3, we make it 4x4
    and pad the weights with (stride) number of zeroes
    as we currently do not support 3x3 deconvolution.
    """
    layer_param = layer.convolution_param
    kernels = get_kernel_dim(layer_param)
    strides = get_stride_dim(layer_param)
    kernel_d = 1
    assert kernels is not None
    assert strides is not None
    if len(kernels) == 1:
      kernel_w = kernels[0]
      kernel_h = kernels[0]
    elif len(kernels) == 3:
      kernel_w = kernels[-1]
      kernel_h = kernels[-2]
      kernel_d = kernels[-3]
    else:
      kernel_w = kernels[-1]
      kernel_h = kernels[-2]
    if len(strides) == 1:
      stride_w = strides[0]
      stride_h = strides[0]
    elif len(strides) == 3:
      stride_w = strides[-1]
      stride_h = strides[-2]
      stride_d = strides[-3]
    else:
      stride_w = strides[-1]
      stride_h = strides[-2]
    if (kernel_d, kernel_w, kernel_h) == (1, 3, 3):
      new_kernel_w, new_kernel_h = 4, 4
      old_weights = self.caffe_net.params[layer.name][0].data
      weights_shape = old_weights.shape
      if len(self.caffe_net.blobs[layer.top[0]].shape) == 4:
        self.inp_wts[layer.name][0] =\
            np.pad(old_weights,
                  pad_width=((0, 0),
                              (0, 0),
                              (0, new_kernel_h - kernel_h),
                              (0, new_kernel_w - kernel_w)),
                  mode='constant')
        set_kernel_dim(layer.convolution_param, [new_kernel_w, new_kernel_h])
      else:
        self.inp_wts[layer.name][0] =\
            np.pad(old_weights,
                  pad_width=((0, 0),
                              (0, 0),
                              (0, 0),
                              (0, new_kernel_h - kernel_h),
                              (0, new_kernel_w - kernel_w)),
                  mode='constant')
        set_kernel_dim(layer.convolution_param, [1, new_kernel_w, new_kernel_h])

  def do_transformation(self):
    deconvolutions = self.get_nodes_of_type('Deconvolution')
    repedgedeconv = self.get_nodes_of_type('RepEdgeDeconv')
    deconvolutions.extend(repedgedeconv)
    for layer in deconvolutions:
      layer_param = layer.convolution_param
      shapes = self.caffe_net.blobs[layer.top[0]].shape
      pads = get_pad_dim(layer_param)
      kernels = get_kernel_dim(layer_param)
      strides = get_stride_dim(layer_param)
      # if sum(pads) == 0 and np.all(np.array(kernels) == 4):
      #   continue
      # if len(shapes) != 4:
      #   continue
      self.adjust_kernel_dims(layer)
      if len(shapes) == 4:
        ran = 2
      else:
        ran = 3
      if len(list(pads)) == 1:
        val = pads[0]
        for i in range(ran * 2 - 1):
          pads.append(val)
      expected_pad = [0 for i in range(len(pads))]
      # if expected_pad == pads:
      #   continue
      crop_offsets = []
      count = 0
      for idx, expected_pad_val in enumerate(expected_pad):
        crop_offsets.append(pads[idx] - expected_pad_val)
        if pads[idx] == expected_pad_val:
          count += 1
      # NOTE:
      # -----
      # in this case the output of deconvolution needs to be
      # cropped
      offset = []
      if len(shapes) == 4:
        offset.append(crop_offsets[0])
        offset.append(crop_offsets[1])
      else:
        offset.append(crop_offsets[0])
        offset.append(crop_offsets[1])
        offset.append(crop_offsets[2])
      set_pad_dim(layer.convolution_param, expected_pad)
      dummy_layer, crop_layer =\
        self.G.mk_crop_proto(name=layer.name,
                      dummy_shape=list(shapes),
                      crop_offset=offset)
      self.G.add_node(dummy_layer)
      self.G.add_node(crop_layer)
      successors = list(layer.successors())
      if layer.type == 'Deconvolution' and layer_param.group == layer_param.num_output and \
        layer_param.group == self.caffe_net.blobs[layer.bottom[0]].data.shape[1]:
        new_layer = copy.deepcopy(layer)
        new_layer.type = 'RepEdgeDeconv'
        self.G.add_node(new_layer)
        predecessors = list(layer.predecessors())
        for pred in predecessors:
          self.G.remove_edge(pred, layer)
          self.G.add_edge(pred, new_layer)
        for suc in successors:
          self.G.remove_edge(layer, suc)
          self.G.add_edge(new_layer, suc)
        self.G.remove_node(layer)
        layer = copy.deepcopy(new_layer)
      for successor in successors:
        self.G.remove_edge(layer, successor)
        self.G.add_edge(crop_layer, successor)
        self.G.add_replace_edge_proto_new(crop_layer, successor, layer.top[0], crop_layer.top[0])
      self.G.add_edge(layer, crop_layer)
      self.G.add_edge(dummy_layer, crop_layer)
      self.G.add_replace_edge_proto_new(layer, crop_layer, "NA", layer.top[0])
      self.G.add_replace_edge_proto_new(dummy_layer, crop_layer, "NA", dummy_layer.top[0])
