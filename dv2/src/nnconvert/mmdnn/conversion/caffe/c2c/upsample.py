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

from mmdnn.conversion.caffe.c2c.common import BasePass

class UpsampleScale(BasePass):

  def __init__(self):
    pass

  def standardize_upsample_scale(self, layer):
    shape = self.caffe_net.blobs[layer.top[0]].data.shape
    if len(layer.upsample_param.scale) == 1:
      if len(shape) in [4, 5]:
        scale_value = layer.upsample_param.scale[0]
        layer.upsample_param.scale.extend([scale_value] * (len(shape) - 3))
    if len(layer.upsample_param.zero_mode) == 0:
      if len(shape) in [3, 4, 5]:
        layer.upsample_param.zero_mode.extend([0] * (len(shape) - 2))

  def do_transformation(self):
    for layer in self.get_nodes_of_type('Upsample'):
      self.standardize_upsample_scale(layer._layer)
