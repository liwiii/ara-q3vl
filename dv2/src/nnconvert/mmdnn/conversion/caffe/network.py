'''
Portions of this file are derived from https://github.com/microsoft/MMdnn and are subject to the original MIT License

Copyright (c) Microsoft Corporation. All rights reserved.
SPDX-License-Identifier: MIT

Copyright (c) 2018 - 2025, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Modifications:

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

import numpy as np
DEFAULT_PADDING = 'SAME'


def layer(op):
  '''Decorator for composable network layers.'''

  def layer_decorated(self, *args, **kwargs):
    # Automatically set a name if not provided.
    name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
    # Figure out the layer inputs.
    assert len(args) >= 1
    if len(args) == 1:
      layer_inputs = args[0]
    else:
      layer_inputs = list(args)
    layer_output = op(self, layer_inputs, **kwargs)
    # print('op: %s   shape: %s' % (op, layer_output._keras_shape))
    # print('op: %s   shape: %s' % (op, layer_output.get_shape().as_list()))
    # Add to layer LUT.
    self.layers[name] = layer_output
    self.output = layer_output
    return layer_output

  return layer_decorated


class Network(object):

  def __init__(self, trainable=False):
    self.output = None
    self.layers = {}
    self.trainable = trainable
    self.setup()

  def setup(self):
    raise NotImplementedError('Must be implemented by the subclass')

  def load(self, data_path, session, ignore_missing=False):
    raise NotImplementedError('Must be implemented by the subclass')

  def input(self, shape, name):
    raise NotImplementedError('Must be implemented by the subclass')

  def get_output(self):
    raise NotImplementedError('Must be implemented by the subclass')

  def get_unique_name(self, prefix):
    raise NotImplementedError('Must be implemented by the subclass')

  @layer
  def conv(self, input, k_h, k_w, c_o, s_h, s_w, p_h, p_w, name, group=1, biased=True):
    raise NotImplementedError('Must be implemented by the subclass')

  @layer
  def deconv(self, input, c_o, k_h, k_w, s_h, s_w, p_h, p_w, name):
    raise NotImplementedError('Must be implemented by the subclass')

  @layer
  def relu(self, input, name):
    raise NotImplementedError('Must be implemented by the subclass')

  @layer
  def sigmoid(self, input, name):
    raise NotImplementedError('Must be implemented by the subclass')

  @layer
  def max_pool(self, input, k_h, k_w, s_h, s_w, p_h, p_w, name):
    raise NotImplementedError('Must be implemented by the subclass')

  @layer
  def avg_pool(self, input, k_h, k_w, s_h, s_w, p_h, p_w, name):
    raise NotImplementedError('Must be implemented by the subclass')

  @layer
  def lrn(self, input, local_size, alpha, beta, name, bias=1):
    raise NotImplementedError('Must be implemented by the subclass')

  @layer
  def concat(self, inputs, axis, name):
    raise NotImplementedError('Must be implemented by the subclass')

  @layer
  def add(self, inputs, name):
    raise NotImplementedError('Must be implemented by the subclass')

  @layer
  def fc(self, input, num_out, name):
    raise NotImplementedError('Must be implemented by the subclass')

  @layer
  def softmax(self, input, name):
    raise NotImplementedError('Must be implemented by the subclass')

  @layer
  def batch_normalization(self, input, name, epsilon=0.00001, scale_offset=True):
    raise NotImplementedError('Must be implemented by the subclass')

  @layer
  def dropout(self, input, keep_prob, name):
    raise NotImplementedError('Must be implemented by the subclass')

  @layer
  def crop(self, inputs, offset, name):
    raise NotImplementedError('Must be implemented by the subclass')
