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


class Rope(Layer):
    def __init__(
        self, layer_name, graph, rope_table, out_group_axis, out_group_size, broadcast=[0, 1.0, 0.5, 0], is_dynamic=False, output_type=None
    ):
        super().__init__(layer_name, graph, out_group_axis, out_group_size)
        self.is_dynamic = is_dynamic
        self.broadcast = broadcast
        self.base_type = "DynamicEltwise" if is_dynamic else "DynamicEltwise"
        self.estimator_group = "DynamicEltwise" if is_dynamic else "DynamicEltwise"
        self.rope_table = rope_table
        self.output_type = output_type
        

    def forward(self, bottom_nodes, top_name):
        if self.out_group_axis and self.out_group_axis < 0:
            self.out_group_axis += len(bottom_nodes[0].shape)
        if self.is_dynamic:
            new_node = L.DynamicEltwise(name=top_name[0]).to_proto()
        else:
            new_node = L.Eltwise(
                name=top_name[0]
            ).to_proto()

        new_node.layer[0].eltwise_param.operation=4
        new_node.layer[0].eltwise_param.broadcast1.extend(self.broadcast)
        new_node.layer[0].bottom.append(bottom_nodes[0].name)
        new_node.layer[0].bottom.append(self.rope_table[0].name)
        new_node.layer[0].bottom.append(bottom_nodes[1].name)
        for (i, top_name) in enumerate(top_name):
            try:
                new_node.layer[0].top[i] = top_name
            except IndexError:
                new_node.layer[0].top.append(top_name)
        if(self.output_type == 'K'):
            new_node.layer[0].isKeyOutput = 0
        self.set_types(new_node)
        self.graph.add_proto_node(new_node)
        return new_node

    def get_top_shapes(self, inp_shapes):
        return inp_shapes
    
    def get_nickname(self):
        return "rope"


# layer {
#   name: "_model_layers_0_self_attn_Add"
#   type: "Ara2DynamicEltwise"
#   bottom: "_model_layers_0_self_attn_Reshape_1"
#   bottom: "new_permute_0_rope"
#   bottom: "_model_layers_0_self_attn_Reshape_1_scale"
#   top: "_model_layers_0_self_attn_Add"
#   top: "_model_layers_0_self_attn_Add_scale"
#   base_type: "DynamicEltwise"
#   estimator_group: "DynamicEltwise"
#   eltwise_param {
#     operation: ROPE
#     broadcast1: 0.0
#     broadcast1: 1.0
#     broadcast1: 0.5
#     broadcast1: 0.0
#   }
# }