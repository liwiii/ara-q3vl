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


class IdentityPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    identities = self.get_nodes_of_type('Identity')
    for layer in identities:
      successors = list(layer.successors())
      if len(successors) > 1:
        logger.info('there are more than 1 tops')
        continue
      if len(successors) == 0:
        predecessors = list(layer.predecessors())
        if len(predecessors) > 0:
          predecessor = predecessors[0]
          # Replace Identity with Reshape
          # Get the top idx of predecessor going into Identity
          idx = 0
          for top, top_name in enumerate(predecessor.top):
            if top_name == layer.bottom[0]:
              idx = top
              break
          new_shape = list(self.caffe_net.blobs[layer.top[idx]].data.shape)
          reshape = self.G.mk_reshape_proto(predecessor.top[idx], new_shape, layer.name)
          self.G.add_node(reshape)
          self.G.remove_edge(predecessor, layer)
          self.G.add_edge(predecessor, reshape)
          self.G.remove_node(layer)
