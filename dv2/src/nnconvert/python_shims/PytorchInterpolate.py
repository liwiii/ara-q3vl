'''
Copyright (c) 2022-25, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

import caffe
import json
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy


class PytorchInterpolate(caffe.Layer):

  def setup(self, bottom, top):
    layer_parameters = json.loads(self.param_str)
    resize_shape = layer_parameters['resize_shape']
    # if len(resize_shape) == 6:
    #     del resize_shape[2]
    #     del resize_shape[2]
    self.resize_shape = resize_shape

  def reshape(self, bottom, top):
    top[0].reshape(*self.resize_shape)

  def forward(self, bottom, top):
    input_data = torch.from_numpy(bottom[0].data)
    print('PytorchInterpolate/forward', input_data.shape, self.resize_shape, flush=True)
    output_data = \
        F.interpolate(input_data, tuple(self.resize_shape[-2:]), mode='bilinear', align_corners=True)
    top[0].data[...] = output_data.numpy()

  def backward(self, top, propagate_down, bottom):
    pass
