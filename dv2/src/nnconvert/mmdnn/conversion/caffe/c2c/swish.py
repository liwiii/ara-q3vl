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

from mmdnn.conversion.caffe.utils import get_kernel_dims, set_pad_dims

from mmdnn.conversion.caffe.c2c.common import BasePass

from dv_logging import get_logger

logger = get_logger(__name__, 'dvctoc')

class MapToAten(BasePass):
  def __init__(self):
    pass

  def do_transformation(self):
    layers = self.get_nodes_of_type('LayerNorm')
    for layer in layers: 
        out_shape = [i for i in self.caffe_net.blobs[layer.top[0]].data.shape]
        aten_node = self.G.mk_aten_proto(layer.bottom[0], layer.name,out_shape)
        self.G.add_node(aten_node)
        self.G.add_edge(layer.predecessors()[0],aten_node)
        for t in list(layer.successors()):
            self.G.add_edge(aten_node,t)
            self.G.add_replace_edge_proto_new(aten_node,t,layer.top[0],aten_node.top[0])
        self.G.remove_node(layer)
        del self.caffe_net.params[layer.name]
    
    layers = self.get_nodes_of_type('GroupNorm')
    for layer in layers: 
        out_shape = [i for i in self.caffe_net.blobs[layer.top[0]].data.shape]
        aten_node = self.G.mk_aten_proto(layer.bottom[0],layer.name,out_shape)
        self.G.add_node(aten_node)
        self.G.add_edge(layer.predecessors()[0],aten_node)
        for t in list(layer.successors()):
            self.G.add_edge(aten_node,t)
            self.G.add_replace_edge_proto_new(aten_node,t,layer.top[0], aten_node.top[0])
        self.G.remove_node(layer)
        del self.caffe_net.params[layer.name]



class MergeSwish(BasePass):
    def __init__(self):
        pass
    # if multiple swish are successor to one layer, then fuse them into one

    def do_transformation(self):
        topsort = self.get_topsort()
        non_swish_layers = [i for i in topsort if i.type != 'Swish']
        for layer in non_swish_layers:
            if len(layer.successors()) < 2:
                continue
            swish_layers = []
            for succ in list(layer.successors()):
                if succ.type == 'Swish':
                    swish_layers.append(succ)
            if len(swish_layers) < 2:
                continue
            swish_0 = swish_layers[0]
            for swish in swish_layers[1:]:
                swish_top = list(swish.successors())
                self.G.remove_node(swish)
                for l in swish_top:
                    self.G.add_edge(swish_0, l)
                    self.G.add_replace_edge_proto_new(swish_0,l,swish.top[0],swish_0.top[0])


class SwishMatcher(BasePass):
    def __init__(self):
        pass
    def is_valid_pattern(self, sigmoid):
        sigmoid_pred = list(sigmoid.predecessors())
        sigmoid_succ = list(sigmoid.successors())
        if len(sigmoid_pred) != 1 or len(sigmoid_succ) != 1:
            return False
        sigmoid_succ_pred_name = [i.name for i in sigmoid_succ[0].predecessors()]
        if len(sigmoid_succ[0].predecessors()) != 2 or sigmoid.name not in sigmoid_succ_pred_name or sigmoid_pred[0].name not in sigmoid_succ_pred_name:
            return False
        if sigmoid_succ[0].type != 'Eltwise' or sigmoid_succ[0].get_layer().eltwise_param.operation != 0:
            return False
        return True

    def do_transformation(self):
        sigmoid_layers = self.get_nodes_of_type('Sigmoid')
        for sigmoid in sigmoid_layers:
            if not self.is_valid_pattern(sigmoid):
                continue
            input_node = sigmoid.predecessors()[0]
            mul_node = sigmoid.successors()[0]
            swish_node = self.G.mk_swish_proto(sigmoid.bottom[0], sigmoid.name)
            self.G.add_node(swish_node)
            self.G.add_edge(input_node, swish_node)
            for l in list(mul_node.successors()):
                    self.G.add_edge(swish_node, l)
                    self.G.add_replace_edge_proto_new(swish_node,l,mul_node.top[0],swish_node.top[0])
            self.G.remove_node(mul_node)
            self.G.remove_node(sigmoid)


            