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
from mmdnn.conversion.caffe.utils import get_stride_dims, get_pad_dims, get_kernel_dims, set_pad_dims
from mmdnn.conversion.caffe.c2c.common import BasePass
from dv_logging import get_logger
logger = get_logger(__name__, 'dvctoc')


class InnerProductPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    topsort = self.get_topsort()
    for layer in topsort:
      if layer.type != 'InnerProduct':
        continue
      # bottom = layer.bottom[0]
      # if len(self.caffe_net.blobs[bottom].shape) <= 2 :
      #     return
      # if self.caffe_net.blobs[bottom].shape[2] > 1 \
      #    or self.caffe_net.blobs[bottom].shape[3] > 1:
      # import pdb; pdb.set_trace()
      pred_shape = self.caffe_net.blobs[layer.bottom[0]].shape
      if pred_shape[0] > 1:
        #can't bring batch dimension in channels
        continue
      if len(pred_shape) >= 4:
        if set(pred_shape[2:]) == {1}:
          continue
      reshape_layer = L.Reshape(reshape_param={'shape': {'dim': [1, -1, 1, 1]}}).to_proto().layer[0]
      reshape_layer.name = self.G.get_layername(layer.name,'reshape')
      reshape_layer.top[0] = reshape_layer.name
      predecessor = layer.predecessors()[0]
      
      self.G.add_node(reshape_layer)
      self.G.remove_edge(predecessor, layer)
      self.G.add_edge(predecessor, reshape_layer)
      self.G.add_edge(reshape_layer, layer)
      self.G.add_replace_edge_proto_new(reshape_layer, layer, predecessor.top[0], reshape_layer.top[0])
      self.G.add_replace_edge_proto_new(predecessor, reshape_layer, "NA", predecessor.top[0])
