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
import copy
from mmdnn.conversion.caffe.c2c.common import BasePass


class SwishLut(BasePass):
    def __init__(self):
        pass

    def do_transformation(self):
      swishes = self.get_nodes_of_type('Swish')
      
      for layer in swishes:
        parent = layer.predecessors()[0]
        continue
        if parent.get_layer().type in ['Convolution', 'Eltwise','InnerProduct']:
          continue
        
        lut_swish = self.G.mk_lut_proto(layer.bottom[0],layer.name,'swsh')
        self.G.add_node(lut_swish)
        self.G.add_edge(parent, lut_swish)
        children = list(layer.successors())
        for s in children:
          self.G.add_edge(lut_swish, s)
          self.G.add_replace_edge_proto(lut_swish, s, layer.top[0])
        self.G.remove_node(layer)
