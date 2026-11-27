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


class Inrp(Layer):
    def __init__(
        self, layer_name, graph, num_output, bias_term, out_group_axis, out_group_size, is_dynamic=False, attention_scale=0
    ):
        super().__init__(layer_name, graph, out_group_axis, out_group_size)
        self.is_dynamic = is_dynamic
        self.num_output = num_output
        self.bias_term = bias_term
        self.base_type = "DynamicInnerProduct" if is_dynamic else "InnerProduct"
        self.estimator_group = "DynamicInnerProduct" if is_dynamic else "InnerProduct"
        self.attention_scale = attention_scale

    def forward(self, bottom_nodes, top_name, accumulation_dim_name):
        # assert(all(bottom_node.shape[-(len(self.kernel_size )+ 1)].dim_name == accumulation_dim_name for bottom_node in bottom_nodes))
        if self.is_dynamic:
            new_node = L.DynamicInnerProduct(
                name=top_name[0],
                bottom=[bottom_nodes[0].name, bottom_nodes[1].name],
            ).to_proto()
        else:
            new_node = L.InnerProduct(
                name=top_name[0],
                bottom=[bottom_nodes[0].name, bottom_nodes[1].name],
            ).to_proto()
        new_node.layer[0].inner_product_param.num_output=self.num_output
        new_node.layer[0].inner_product_param.bias_term=self.bias_term
        if (self.attention_scale):
            new_node.layer[0].inner_product_param.attentionInrpMultiplier=self.attention_scale 
        # new_node.convolution_param.pad.extend(layer["pad"][0:-1])
        for (i, top_name) in enumerate(top_name):
            try:
                new_node.layer[0].top[i] = top_name
            except IndexError:
                new_node.layer[0].top.append(top_name)
        self.set_types(new_node)
        self.graph.add_proto_node(new_node)
        return new_node

    def get_top_shapes(self, inp_shapes):
        new_shapes = inp_shapes
        channel_dim = 1
        new_shapes[0][channel_dim].dim = self.num_output
        for new_shape in new_shapes:
            pass
        if self.out_group_axis < 0: 
            self.out_group_axis += len(new_shape)

        assert(new_shapes[0][self.out_group_axis].dim % self.out_group_size == 0)
        new_shapes[1][self.out_group_axis].dim = new_shapes[0][self.out_group_axis].dim // self.out_group_size
        return inp_shapes
    
    def get_nickname(self):
        return "inrp"

# layer {
#   name: "_model_layers_0_self_attn_q_proj_MatMul"
#   type: "Ara2DynamicInnerProduct"
#   bottom: "_model_layers_0_input_layernorm_Mul_1_ara_reshape_2"
#   bottom: "_model_layers_0_input_layernorm_Mul_1_ara_reshape_2_scale"
#   top: "_model_layers_0_self_attn_q_proj_MatMul"
#   top: "_model_layers_0_self_attn_q_proj_MatMul_scale"
#   base_type: "DynamicInnerProduct"
#   estimator_group: "DynamicInnerProduct"
#   inner_product_param {
#     num_output: 4096
#     bias_term: false
#   }
# }