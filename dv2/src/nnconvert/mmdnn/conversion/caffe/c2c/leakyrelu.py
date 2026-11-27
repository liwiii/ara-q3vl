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

from mmdnn.conversion.caffe.utils import get_stride_dim, get_pad_dim, get_kernel_dim, set_pad_dim, mk_padding_proto, mk_crop_proto, mk_pad_proto, mk_prelu_layer

from mmdnn.conversion.caffe.c2c.common import BasePass

from dv_logging import get_logger

logger = get_logger(__name__, 'dvctoc')

class LeakyReluPatching(BasePass):
  def __init__(self):
    pass

  def do_transformation(self):
    relu = self.get_nodes_of_type('ReLU')
    
    for layer in relu:
      if layer.relu_param.negative_slope != 0:
        prelu_layer = mk_prelu_layer(layer.name)
        for idx,top in enumerate(layer.top):
          prelu_layer.top.append(top)

        for idx,bottom in enumerate(layer.bottom):
          prelu_layer.bottom.append(bottom)     

        self.inp_wts[layer.name] = [layer.relu_param.negative_slope]
        self.G.add_node(prelu_layer)

        predecessor = layer.predecessors()[0]
        if len(layer.successors()):
          successors = layer.successors()
          for successor in successors:
            self.G.add_edge(prelu_layer, successor)
            self.G.remove_edge(layer, successor)
        self.G.add_edge(predecessor, prelu_layer)
        self.G.remove_edge(predecessor, layer)
        self.G.remove_node(layer)

      