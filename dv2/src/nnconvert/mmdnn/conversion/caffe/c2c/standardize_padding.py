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
from mmdnn.conversion.caffe.utils import get_kernel_dims, get_pad_dims, set_pad_dims


class StandardizePad(BasePass):

  def __init__(self):
    pass

  def node_selector(self):
    return ['Padding']

  def standardize_padding(self, layer_param):
    if len(layer_param.pad) == 0:
      if layer_param.HasField('pad_d'):
        layer_param.ClearField('pad')
        layer_param.pad.append(layer_param.pad_d)
        layer_param.pad.append(layer_param.pad_h)
        layer_param.pad.append(layer_param.pad_w)
        layer_param.pad.append(layer_param.pad_d)
        layer_param.pad.append(layer_param.pad_h)
        layer_param.pad.append(layer_param.pad_w)
        layer_param.ClearField('pad_d')
        layer_param.ClearField('pad_h')
        layer_param.ClearField('pad_w')
      elif layer_param.HasField('pad_h'):
        layer_param.ClearField('pad')
        layer_param.pad.append(layer_param.pad_h)
        layer_param.pad.append(layer_param.pad_w)
        layer_param.pad.append(layer_param.pad_h)
        layer_param.pad.append(layer_param.pad_w)
        layer_param.ClearField('pad_h')
        layer_param.ClearField('pad_w')
      #   layer_param.ClearField('pad_t', layer_param.)
      else:
        if (hasattr(layer_param, 'pad_t') and layer_param.HasField('pad_t')) or \
        (hasattr(layer_param, 'pad_l') and layer_param.HasField('pad_l')) or \
        (hasattr(layer_param, 'pad_b') and layer_param.HasField('pad_b')) or \
        (hasattr(layer_param, 'pad_r') and layer_param.HasField('pad_r')):
          # layer_param.HasField('pad_l') \
          #    or layer_param.HasField('pad_r') or  layer_param.HasField('pad_b'):
          layer_param.pad.append(layer_param.pad_t)
          layer_param.pad.append(layer_param.pad_l)
          layer_param.pad.append(layer_param.pad_b)
          layer_param.pad.append(layer_param.pad_r)
          layer_param.ClearField('pad_t')
          layer_param.ClearField('pad_l')
          layer_param.ClearField('pad_b')
          layer_param.ClearField('pad_r')

  def standardize_pad_padding(self, layer_param):
    if len(layer_param.pad) == 0:
      if layer_param.HasField('pad_t') or layer_param.HasField('pad_l') \
          or layer_param.HasField('pad_r') or  layer_param.HasField('pad_b'):
        layer_param.pad.append(layer_param.pad_t)
        layer_param.pad.append(layer_param.pad_l)
        layer_param.pad.append(layer_param.pad_b)
        layer_param.pad.append(layer_param.pad_r)
        layer_param.ClearField('pad_t')
        layer_param.ClearField('pad_l')
        layer_param.ClearField('pad_b')
        layer_param.ClearField('pad_r')

  def do_transformation(self):
    for layer in self.get_nodes_of_type('Convolution'):
      self.standardize_padding(layer.convolution_param)
    for layer in self.get_nodes_of_type('Padding'):
      self.standardize_pad_padding(layer.padding_param)
    for layer in self.get_nodes_of_type('Pooling'):
      self.standardize_padding(layer.pooling_param)
    for layer in self.get_nodes_of_type('Deconvolution'):
      self.standardize_padding(layer.convolution_param)
    for layer in self.get_nodes_of_type('RepEdgeDeconv'):
      self.standardize_padding(layer.convolution_param)


class StandardizeStrides(BasePass):

  def __init__(self):
    pass

  def node_selector(self):
    return ['Padding']

  def standardize_strides(self, layer_param):
    if layer_param.HasField('stride_h') and len(layer_param.stride):
      logger.error("both stride fields stride and stride_w occur together for layer %s", layer.name)
    if layer_param.HasField('stride_d'):
      layer_param.stride.append(layer_param.stride_d)
      layer_param.ClearField('stride_d')
    if layer_param.HasField('stride_h'):
      layer_param.stride.append(layer_param.stride_h)
      layer_param.ClearField('stride_h')
    if layer_param.HasField('stride_w'):
      layer_param.stride.append(layer_param.stride_w)
      layer_param.ClearField('stride_w')

  def do_transformation(self):
    for layer in self.get_nodes_of_type('Convolution'):
      self.standardize_strides(layer.convolution_param)
    for layer in self.get_nodes_of_type('Deconvolution'):
      self.standardize_strides(layer.convolution_param)
    for layer in self.get_nodes_of_type('Pooling'):
      self.standardize_strides(layer.pooling_param)
    for layer in self.get_nodes_of_type('RepEdgeDeconv'):
      self.standardize_strides(layer.convolution_param)


class StandardizeKernelSize(BasePass):

  def __init__(self):
    pass

  def node_selector(self):
    return ['Padding']

  def standardize_kernel_size(self, layer_param):
    if layer_param.HasField('kernel_h') and len(layer_param.kernel_size):
      logger.error("both kernel fields kernel_size and kernel occur together for layer %s", layer.name)
    if layer_param.HasField('kernel_d'):
      layer_param.kernel_size.append(layer_param.kernel_d)
      layer_param.ClearField('kernel_d')
    if layer_param.HasField('kernel_h'):
      layer_param.kernel_size.append(layer_param.kernel_h)
      layer_param.ClearField('kernel_h')
    if layer_param.HasField('kernel_w'):
      layer_param.kernel_size.append(layer_param.kernel_w)
      layer_param.ClearField('kernel_w')

  def do_transformation(self):
    for layer in self.get_nodes_of_type('Convolution'):
      self.standardize_kernel_size(layer.convolution_param)
    for layer in self.get_nodes_of_type('Deconvolution'):
      self.standardize_kernel_size(layer.convolution_param)
    for layer in self.get_nodes_of_type('Pooling'):
      self.standardize_kernel_size(layer.pooling_param)
    for layer in self.get_nodes_of_type('RepEdgeDeconv'):
      self.standardize_kernel_size(layer.convolution_param)
