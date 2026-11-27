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


class Gelu(Layer):
    def __init__(
        self, layer_name, graph, out_group_axis, out_group_size, is_dynamic = False
    ):
        super().__init__(layer_name, graph, out_group_axis, out_group_size)
        self.base_type = "DynamicGelu" if is_dynamic else "Gelu"
        self.estimator_group = "DynamicGelu" if is_dynamic else "Gelu"
        self.is_dynamic = is_dynamic # TODO: move this into base model

    def forward(self, bottom_nodes, top_name):
        if self.is_dynamic:
            new_node = L.DynamicGelu(
                name=bottom_nodes[0].name,
            ).to_proto()
        else:
            new_node = L.Gelu(
                name=top_name[0],
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
        if self.out_group_axis < 0:
            self.out_group_axis += len(bottom_nodes[0].shape)
        new_node.layer[0].base_type =  self.base_type
        new_node.layer[0].estimator_group =  "activation"
        new_node.layer[0].type = "Ara2" + self.base_type
        new_node.layer[0].name = new_node.layer[0].top[0]
        self.graph.add_proto_node(new_node)
        return new_node

    def get_top_shapes(self, inp_shapes):
        return inp_shapes
    
    def get_nickname(self):
        return "gelu"


# layer {
#   name: "gelu"
#   type: "Ara2DynamicGelu"
#   bottom: "fc1_linear"
#   bottom: "fc1_linear_scale"
#   top: "gelu"
#   top: "gelu_scale"
#   base_type: "DynamicGelu"
#   estimator_group: "activation"
# }