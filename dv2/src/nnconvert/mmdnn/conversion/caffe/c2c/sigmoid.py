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

from dv_logging import get_logger

logger = get_logger(__name__, 'dvctoc')



class SigmoidConv1(BasePass):
    def __init__(self):
        pass
    def is_valid_pattern(self, sigmoid):
        sigmoid_pred = list(sigmoid.predecessors())
        if len(sigmoid_pred) != 1 or sigmoid_pred[0].type != 'Permute':
            return False
        sigmoid_pred_2 = list(sigmoid_pred[0].predecessors())
        if len(sigmoid_pred_2) != 1 or sigmoid_pred_2[0].type != 'Reshape':
            return False
        sigmoid_pred_3 = list(sigmoid_pred_2[0].predecessors())
        if len(sigmoid_pred_3) != 1 or sigmoid_pred_3[0].type != 'Convolution':
            return False
        return True

    def do_transformation(self):
        sigmoid_layers = self.get_nodes_of_type('Sigmoid')
        for sigmoid in sigmoid_layers:
            if not self.is_valid_pattern(sigmoid):
                continue
            perm_node = sigmoid.predecessors()[0]
            reshape_node = perm_node.predecessors()[0]
            conv_node = reshape_node.predecessors()[0]
            succs = list(sigmoid.successors())
            self.G.remove_edge(conv_node,reshape_node)
            self.G.remove_edge(perm_node,sigmoid)
            self.G.add_edge(conv_node, sigmoid)
            self.G.add_edge(sigmoid, reshape_node)
            self.G.add_replace_edge_proto(conv_node,sigmoid,perm_node.top[0])
            self.G.add_replace_edge_proto(sigmoid,reshape_node,conv_node.top[0])
            for l in succs:
                    self.G.add_edge(perm_node, l)
                    self.G.add_replace_edge_proto(perm_node,l,sigmoid.top[0])


            