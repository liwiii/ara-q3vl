'''
Copyright (c) 2025, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

from .layer import Layer
from caffe import layers as L  # type: ignore


class SoftMax(Layer):
    def __init__(
        self, layer_name, graph, out_group_axis, out_group_size, is_dynamic = False
    ):
        super().__init__(layer_name, graph, out_group_axis, out_group_size)
        self.base_type = "DynamicSoftmax" if is_dynamic else "Softmax"
        self.estimator_group = "DynamicSoftmax" if is_dynamic else "Softmax"
        self.is_dynamic = is_dynamic

    def forward(self, bottom_nodes, top_name, axis):
        if self.out_group_axis < 0:
            self.out_group_axis += len(bottom_nodes[0].shape)
        if self.is_dynamic:
            new_node = L.DynamicSoftmax(
                name=self.layer_name,
            ).to_proto()
        else:
            new_node = L.Softmax(
                name=self.layer_name,
            ).to_proto()
        new_node.layer[0].softmax_param.axis = axis + len(bottom_nodes[0].shape) if axis < 0 else axis
        for (i, top_name) in enumerate(top_name):
            try:
                new_node.layer[0].top[i] = top_name
            except IndexError:
                new_node.layer[0].top.append(top_name)
        for (i, bottom_node) in enumerate(bottom_nodes):
            try:
                new_node.layer[0].bottom[i] = bottom_node.name
            except IndexError:
                new_node.layer[0].bottom.append(bottom_node.name)
        self.set_types(new_node)
        self.graph.add_proto_node(new_node)
        return new_node

    def get_top_shapes(self, inp_shapes):
        new_shapes = inp_shapes
        for new_shape in new_shapes:
            pass
        if self.out_group_axis < 0: 
            self.out_group_axis += len(new_shape)
        if(self.out_group_size == 9999):
            self.out_group_size = new_shapes[0][self.out_group_axis].dim
        assert(new_shapes[0][self.out_group_axis].dim % self.out_group_size == 0)
        new_shapes[1][self.out_group_axis].dim = new_shapes[0][self.out_group_axis].dim // self.out_group_size
        return new_shapes
    
    def get_nickname(self):
        return "softmax"

class AttentionSoftmax(Layer):
    def __init__(
        self, layer_name, graph, out_group_axis, out_group_size, is_dynamic = False
    ):
        super().__init__(layer_name, graph, out_group_axis, out_group_size)
        self.base_type = "DynamicAttentionSoftmax" if is_dynamic else "AttentionSoftmax"
        self.estimator_group = "DynamicAttentionSoftmax" if is_dynamic else "AttentionSoftmax"
        self.is_dynamic = is_dynamic

    def forward(self, bottom_nodes, top_name, axis):
        if self.out_group_axis < 0:
            self.out_group_axis += len(bottom_nodes[0].shape)
        if self.is_dynamic:
            new_node = L.DynamicAttentionSoftmax(
                name=self.layer_name,
            ).to_proto()
        else:
            new_node = L.AttentionSoftmax(
                name=self.layer_name,
            ).to_proto()
        new_node.layer[0].attentionsoftmax_param.axis = axis + len(bottom_nodes[0].shape) if axis < 0 else axis
        for (i, top_name) in enumerate(top_name):
            try:
                new_node.layer[0].top[i] = top_name
            except IndexError:
                new_node.layer[0].top.append(top_name)
        for (i, bottom_node) in enumerate(bottom_nodes):
            try:
                new_node.layer[0].bottom[i] = bottom_node.name
            except IndexError:
                new_node.layer[0].bottom.append(bottom_node.name)
        self.set_types(new_node)
        self.graph.add_proto_node(new_node)
        return new_node

    def get_top_shapes(self, inp_shapes):
        new_shapes = inp_shapes
        for new_shape in new_shapes:
            pass
        if self.out_group_axis < 0: 
            self.out_group_axis += len(new_shape)
        if(self.out_group_size == 9999):
            self.out_group_size = new_shapes[0][self.out_group_axis].dim
        assert(new_shapes[0][self.out_group_axis].dim % self.out_group_size == 0)
        new_shapes[1][self.out_group_axis].dim = new_shapes[0][self.out_group_axis].dim // self.out_group_size
        return new_shapes
    
    def get_nickname(self):
        return "attention_softmax"
# layer {
#   name: "_model_layers_0_self_attn_Softmax"
#   type: "Ara2DynamicAttentionSoftmax"
#   bottom: "_model_layers_0_self_attn_MatMul"
#   bottom: "_model_layers_0_self_attn_MatMul_scale"
#   top: "_model_layers_0_self_attn_Softmax"
#   top: "_model_layers_0_self_attn_Softmax_scale"
#   base_type: "DynamicAttentionSoftmax"
#   estimator_group: "DynamicAttentionSoftmax"
#   attentionsoftmax_param {
#     axis: 3
#   }
# }