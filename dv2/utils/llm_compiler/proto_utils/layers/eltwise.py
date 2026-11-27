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
from enum import Enum

class Operation(Enum):
    PROD = 0
    SUM = 1

class Eltwise(Layer):
    def __init__(
        self, layer_name, graph, out_group_axis, out_group_size, operation: Operation, is_dynamic = False, broadcast_0 = [0.0, 0.0, 0.0], broadcast_1 = [0.0, 0.0, 0.0]
    ):
        super().__init__(layer_name, graph, out_group_axis, out_group_size)
        self.base_type = "DynamicEltwise" if is_dynamic else "Eltwise"
        self.estimator_group = "DynamicEltwise" if is_dynamic else "Eltwise"
        self.is_dynamic = is_dynamic
        self.broadcast_0 = broadcast_0
        self.broadcast_1 = broadcast_1
        self.operation = operation

    def forward(self, bottom_nodes, top_name):
        bottom_nodes[1], bottom_nodes[2] = bottom_nodes[2], bottom_nodes[1]
        if self.out_group_axis < 0:
            self.out_group_axis += len(bottom_nodes[0].shape)
        if self.is_dynamic:
            new_node = L.DynamicEltwise(
                name=self.layer_name,
            ).to_proto()
        else:
            new_node = L.Eltwise(
                name=self.layer_name,
            ).to_proto()
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
        new_node.layer[0].eltwise_param.operation = self.operation.value
        new_node.layer[0].eltwise_param.broadcast0.extend(self.broadcast_0)
        new_node.layer[0].eltwise_param.broadcast1.extend(self.broadcast_1)

        self.set_types(new_node)
        self.graph.add_proto_node(new_node)
        return new_node

    def get_top_shapes(self, inp_shapes):
        return inp_shapes
    
    def get_nickname(self):
        return "add"
# layer {
#   name: "_model_layers_0_Add_1"
#   type: "Ara2DynamicEltwise"
#   bottom: "_model_layers_0_Add"
#   bottom: "_model_layers_0_down_proj_reshape"
#   bottom: "_model_layers_0_Add_scale"
#   bottom: "_model_layers_0_down_proj_reshape_scale"
#   top: "_model_layers_0_Add_1"
#   top: "_model_layers_0_Add_1_scale"
#   base_type: "DynamicEltwise"
#   estimator_group: "DynamicEltwise"
#   eltwise_param {
#     operation: SUM
#     broadcast0: 0.0
#     broadcast0: 0.0
#     broadcast0: 0.0
#     broadcast1: 0.0
#     broadcast1: 0.0
#     broadcast1: 0.0
#   }
#   isOutput: true
# }