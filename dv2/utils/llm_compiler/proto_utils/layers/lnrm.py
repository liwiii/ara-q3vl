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


class LayerNorm(Layer):
    def __init__(
        self, layer_name, graph, out_group_axis, out_group_size, uses_mean, output_mean_var = False, eps = 1e-05, is_dynamic=False
    ):
        super().__init__(layer_name, graph, out_group_axis, out_group_size)
        self.uses_mean = uses_mean
        self.output_mean_var = output_mean_var
        self.eps = eps
        self.is_dynamic = is_dynamic
        self.base_type = "DynamicLayerNorm" if is_dynamic else "LayerNorm"
        self.estimator_group = "DynamicLayerNorm" if is_dynamic else "LayerNorm"

    def forward(self, bottom_nodes, top_name, axis = -1):
        if self.out_group_axis < 0:
            self.out_group_axis += len(bottom_nodes[0].shape)
        if axis < 0:
            axis += len(bottom_nodes[0].shape)
        if self.is_dynamic:
            new_node = L.DynamicLayerNorm(
                name=bottom_nodes[0].name,
                axis=axis,
                uses_mean=self.uses_mean,
                output_mean_var=self.output_mean_var,
                eps=self.eps,
            ).to_proto()
        else:
            new_node = L.LayerNorm(
                name=top_name[0],
                axis=axis,
                uses_mean=self.uses_mean,
                output_mean_var=self.output_mean_var,
                eps=self.eps,
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
        return "lnrm"


# layer {
#   name: "_model_layers_0_input_layernorm_Mul_1_ara_LayerNorm_0"
#   type: "Ara2DynamicLayerNorm"
#   bottom: "input_ids_embedded"
#   bottom: "input_ids_embedded_scale"
#   top: "_model_layers_0_input_layernorm_Mul_1_ara_LayerNorm_0"
#   top: "_model_layers_0_input_layernorm_Mul_1_ara_LayerNorm_0_scale"
#   base_type: "DynamicLayerNorm"
#   estimator_group: "DynamicLayerNorm"
#   dynamic_layer_norm_param {
#     axis: 3
#     eps: 1e-06
#     output_mean_var: false
#     uses_mean: false
#   }
# }

# L.DynamicLayerNorm(name=layer_name, bottom=bottom, axis=axis,uses_mean=uses_mean,output_mean_var=output_mean_var, eps=eps)
