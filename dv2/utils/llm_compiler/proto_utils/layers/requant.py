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
import copy
from caffe import layers as L  # type: ignore


class Requant(Layer):
    def __init__(
        self, layer_name, graph, out_group_axis, out_group_size 
    ):
        super().__init__(layer_name, graph, out_group_axis, out_group_size)
        self.base_type = "DynamicRequant"
        self.estimator_group = "DynamicRequant"
        

    def forward(self, bottom_nodes, top_name):
        new_node = L.DynamicRequant(name=top_name[0]).to_proto()
        for (i, top_name) in enumerate(top_name):
            try:
                new_node.layer[0].top[i] = top_name
            except IndexError:
                new_node.layer[0].top.append(top_name)
        bottom_names = [node.name for node in bottom_nodes]

        # put it in order inp1, inp2, inp1_scale, inp2_scale
        bottom_names[1],bottom_names[2] = bottom_names[2],bottom_names[1] 
        for (i, bottom_name) in enumerate(bottom_names):
            try:
                new_node.layer[0].bottom[i] = bottom_name
            except IndexError:
                new_node.layer[0].bottom.append(bottom_name)
        new_node.layer[0].isValueOutput = 0
        self.set_types(new_node)
        self.graph.add_proto_node(new_node)
        return new_node

    def get_top_shapes(self, bottom_shapes):
        # get non scale bottom nodes
        top_shape = bottom_shapes[0]
        top_scale_shape = copy.deepcopy(top_shape)
        if self.out_group_axis < 0:
            self.out_group_axis += len(top_shape)
        assert((top_scale_shape[self.out_group_axis].dim % self.out_group_size) == 0)
        top_scale_shape[self.out_group_axis].dim //= self.out_group_size
        
        return [top_shape, top_scale_shape]
    
    def get_nickname(self):
        return "requant"

# layer {
#   name: "_model_layers_0_self_attn_MatMul"
#   type: "Ara2DynamicBatchDot"
#   bottom: "_model_layers_0_self_attn_Add"
#   bottom: "_model_layers_0_self_attn_2a_Transpose_3"
#   bottom: "_model_layers_0_self_attn_Add_scale"
#   bottom: "_model_layers_0_self_attn_2a_Transpose_3_scale"
#   top: "_model_layers_0_self_attn_MatMul"
#   top: "_model_layers_0_self_attn_MatMul_scale"
#   base_type: "DynamicBatchDot"
#   estimator_group: "DynamicBatchDot"
#   batch_dot_param {
#     transpose_a: 1
#     transpose_b: 0
#     multiplier: 0.08838834916
#   }
# }