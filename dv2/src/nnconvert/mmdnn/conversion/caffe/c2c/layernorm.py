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

from mmdnn.conversion.caffe.c2c.common import BasePass

from dv_logging import get_logger

logger = get_logger(__name__, 'dvctoc')

class PermuteLayernorm(BasePass):
    def __init__(self):
        pass
    # if multiple swish are successor to one layer, then fuse them into one

    def do_transformation(self):
        def check_combined_permute(list1, list2):
            if len(list1) != len(list2):
                return False
            for i in range(len(list1)):
                if list1[list2[i]] != i:
                    return False
            return True
        
        lnrm_layers = self.get_nodes_of_type('LayerNorm')
        for layer in lnrm_layers:
            predecessors = layer.predecessors()
            successors = layer.successors()
            if len(predecessors) == 1 and len(successors) == 1:
                prev, next = predecessors[0], successors[0]
                if prev.type == "Permute" and next.type == "Permute":
                    prev_order = list(prev.get_layer().permute_param.order)
                    next_order = list(next.get_layer().permute_param.order)
                    if check_combined_permute(prev_order, next_order):
                        # Change the axis of layernorm
                        axes = getattr(getattr(layer, 'layer_norm_param'), 'axis')
                        layer.layer_norm_param.axis = prev_order[axes]
                        # Remove the prev and next permute layers
                        pred = prev.predecessors()[0]
                        self.G.remove_edge(pred, prev)
                        self.G.remove_edge(prev, layer)
                        self.G.add_edge(pred, layer)
                        self.G.add_replace_edge_proto(pred, layer, prev.top[0])

                        self.G.remove_edge(layer, next)
                        for succ in next.successors():
                            self.G.remove_edge(next, succ)
                            self.G.add_edge(layer, succ)
                            self.G.add_replace_edge_proto(layer, succ, next.top[0])
                        
                        self.G.remove_node(prev)
                        self.G.remove_node(next)

class LayerNormAddReshape(BasePass):
  def __init__(self):
    pass

  def do_transformation(self):
    lnorm_layers = self.get_nodes_of_type('LayerNorm')
    for lnrm in lnorm_layers:
      layer_norm_shape = list(self.caffe_net.blobs[lnrm.top[0]].data.shape)
      active_dims = [a for a in layer_norm_shape if a>1]
      if len(active_dims) == 1:
         required_dim = active_dims[0]
         new_reshape_param = [1,1,1,required_dim]
      else:
         continue
      
      lnrm.layer_norm_param.axis = 3
      for idx,pred in enumerate(lnrm.predecessors()):
        reshape_node = self.G.mk_reshape_proto(pred.top[0], new_reshape_param, lnrm.name + f"reshape_{idx}")
        self.G.add_node(reshape_node)
        self.G.add_edge(reshape_node, lnrm)
        self.G.remove_edge(pred, lnrm)
        self.G.add_edge(pred, reshape_node)
        self.G.add_replace_edge_proto_new(reshape_node, lnrm, pred.top[0], reshape_node.top[0])
      

class LayerNormAxis3to1(BasePass):
  def __init__(self):
    pass

  def do_transformation(self):
    lnorm_layers = self.get_nodes_of_type('LayerNorm')
    for layer in lnorm_layers:
      layer_shape = list(self.caffe_net.blobs[layer.top[0]].data.shape)
      axes = getattr(getattr(layer, 'layer_norm_param'), 'axis')

      def ll(aa):
        if len(aa) == 0:
          return None
        return [i for i in aa]
      
      def check_1_successor_1_predecessor(layer):
         if len(layer.successors()) == 1 and len(layer.predecessors()) == 1:
            return True
         return False
      
      if len(layer_shape) in [4] and axes == 3:
        b1 = layer.predecessors()[0]
        if not b1.type == "Reshape":
          continue
        b2 = b1.predecessors()[0]
        if not (b2.type == "PermuteNnp" and ll(b2.permute_nnp_param.order) == [0,1,3,2]):
          continue
        b3 = b2.predecessors()[0]
        if not b3.type == "Reshape":
          continue
        t1 = layer.successors()[0]
        if not t1.type == "Reshape":
          continue
        t2 = t1.successors()[0]
        if not (t2.type == "PermuteNnp" and ll(t2.permute_nnp_param.order) == [0,1,3,2]):
          continue
        t3 = t2.successors()[0]
        if not t3.type == "Reshape":
          continue

        pattern_ok = True
        for x in [layer, b1, b2, b3, t1, t2]:
          if not check_1_successor_1_predecessor(x):
            pattern_ok = False
            break
          
        if pattern_ok == False:
          continue
           
        b4 = b3.predecessors()[0]
        self.G.add_edge(b4, layer)
        self.G.add_replace_edge_proto_new(b4, layer, layer.bottom[0], b4.top[0])

        self.G.remove_edge(b4, b3)
        self.G.remove_edge(b3, b2)
        self.G.remove_edge(b2, b1)
        self.G.remove_edge(b1, layer)
        self.G.remove_edge(layer, t1)
        self.G.remove_edge(t1, t2)
        self.G.remove_edge(t2, t3)

        for t4 in t3.successors():
          self.G.remove_edge(t3, t4)
          self.G.add_edge(layer, t4)
          self.G.add_replace_edge_proto_new(layer, t4, t4.bottom[0], layer.top[0])

        self.G.remove_node(t1)
        self.G.remove_node(t2)
        self.G.remove_node(t3)
        self.G.remove_node(b1)
        self.G.remove_node(b2)
        self.G.remove_node(b3)
        
        layer.layer_norm_param.axis = 1
