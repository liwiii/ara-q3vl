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
from mmdnn.conversion.caffe.c2c.common import BasePass
from dv_logging import get_logger
logger = get_logger(__name__, 'dvctoc')


class ForkArgmaxSoftmax(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    topsort = self.get_topsort()
    do_fork = self.fork_sftmx_argmx
    for layer in topsort:
      if layer.type == 'ArgMax':
        if not layer.argmax_param.HasField('axis'):
          layer.argmax_param.axis = 1
        elif layer.argmax_param.axis != 1:
          layer.needs_generic_cp_kernel = True
        argmax_input_shape = self.caffe_net.blobs[layer.bottom[0]].data.shape
        # generalize to 4 dimensions
        argmax_input_shape = list(argmax_input_shape) + [1] * (4 - len(argmax_input_shape))
        #assume that tne network is classification if h == w == 1
        is_classification = (argmax_input_shape[2] == 1) and (argmax_input_shape[3] == 1)
        if is_classification:
          logger.warn("argmax output shape is classification like - not performing any short circuiting or forking -fork_sftmx_argmx flag is ignored")
          return
        softmax = layer.predecessors()[0]
        softmax_children = list(softmax.successors())
        if softmax.type == 'Softmax' and len(softmax_children) == 1:
          softmax_input = softmax.predecessors()[0]
          self.G.remove_edge(softmax, layer)
          self.G.add_edge(softmax_input, layer)
          layer.bottom[:] = softmax_input.top
          if not do_fork:
            self.G.remove_node(softmax)
