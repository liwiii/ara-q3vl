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


class AttentionSoftmaxMatcher(BasePass):
    def __init__(self):
        pass
    def is_valid_pattern(self, softmax):
        softmax_pred = list(softmax.predecessors())
        if len(softmax_pred) != 1 and softmax_pred[0].type != "Eltwise":
            return False
        eltw_pred = list(softmax_pred[0].predecessors())
        if len(eltw_pred) != 2:
            return False
        is_input = 1 if (eltw_pred[0].type == "Input") else ( 2 if (eltw_pred[1].type == "Input") else 0)
        if is_input == 0 :
            return False
        input_node = eltw_pred[is_input-1]
        return is_input

    def do_transformation(self):
        softmax_layer = self.get_nodes_of_type('Softmax')
        for softmax in softmax_layer:
            index = self.is_valid_pattern(softmax)
            if not index:
                continue
            input_node = softmax.predecessors()[0].predecessors()[index-1]
            other_index = 0 if (index==2) else 1
            other_node = softmax.predecessors()[0].predecessors()[other_index]
            attentionsoftmax_node = self.G.mk_attentionsoftmax_proto([other_node.top[0], input_node.top[0]], softmax.name)
            self.G.add_node(attentionsoftmax_node)
            self.G.add_edge(input_node, attentionsoftmax_node)
            self.G.add_edge(other_node, attentionsoftmax_node)
            for l in list(softmax.successors()):
                self.G.add_edge(attentionsoftmax_node, l)
                self.G.add_replace_edge_proto_new(attentionsoftmax_node,l,softmax.top[0],attentionsoftmax_node.top[0])
            eltwise_node = softmax.predecessors()[0]
            self.G.remove_node(eltwise_node)
            self.G.remove_node(softmax)
