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
from caffe import layers as L # type: ignore
from typing import TYPE_CHECKING, List
import copy
if TYPE_CHECKING:
    import graph
class Input(Layer):
    # As of June 2026, for vision inputs, we do not want scale as separate region.
    # scale_as_scale_region should be set to False for vision inputs.
    def __init__(self, layer_name, graph: "graph.DAG", inp_shape: List["graph.NamedDim"], out_group_axis, out_group_size, is_dynamic=False, input_type= None, scale_as_scale_region=True):
        super().__init__(layer_name, graph, out_group_axis, out_group_size)
        self.is_dynamic = is_dynamic
        self.inp_shape = inp_shape
        self.input_type = input_type
        self.scale_as_scale_region = scale_as_scale_region
    def forward(self, bottom, top_name, **kwargs):
        return super().forward(bottom, top_name, **kwargs)
    def get_top_names(self, bottom_names, top_names=None):
        return super().get_top_names(bottom_names, top_names)
    def get_top_shapes(self, inp_shapes):
        return super().get_top_shapes(inp_shapes)
    def __call__(self, layout = "NCHW"):
        # breakpoint()
        if self.out_group_axis and self.out_group_axis < 0:
            self.out_group_axis += len(self.inp_shape)
        node = L.Input(name = self.layer_name, shape=[dict(dim=list(self.inp_shape[i].dim for i in range(len(self.inp_shape))))], ntop=1).to_proto()
        node.layer[0].top[0] = self.layer_name
        node.layer[0].base_type = "Input"
        node.layer[0].estimator_group = "Input"
        if layout is not None: 
            node.layer[0].input_param.layout[:] = [
            layout
        ]  # Assuming NCHW layout, can be changed based on requirements
        if self.input_type == 'K':
            node.layer[0].isKeyInput = 0
        elif self.input_type == 'V':
            node.layer[0].isValueInput = 0
        elif self.input_type == 'R':
            node.layer[0].isRopeInput = True
        elif self.input_type == 'RQ':
            node.layer[0].isRequantInput = 0

        self.graph.add_proto_node(node)
        inp = self.graph.add_nodes_with_shapes([self.layer_name], [self.inp_shape], self.out_group_axis, self.out_group_size)
        if self.is_dynamic:
            new_shape = copy.deepcopy(self.inp_shape)
            if self.out_group_size == 9999:
                new_shape[self.out_group_axis].dim = 1
            else:
                new_shape[self.out_group_axis].dim //= self.out_group_size
            scale_node = L.Input(name = f"{self.layer_name}_scale", shape=[dict(dim=list(new_shape[i].dim for i in range(len(self.inp_shape))))], ntop=1).to_proto()
            scale_node.layer[0].top[0] = f"{self.layer_name}_scale"
            scale_node.layer[0].base_type = "Input"
            scale_node.layer[0].estimator_group = "Input"
            if layout is not None:
                scale_node.layer[0].input_param.layout[:] = [layout]
            if self.input_type == 'K':
                scale_node.layer[0].isKeyInputScale = 0
            elif self.input_type == 'V':
                scale_node.layer[0].isValueInputScale = 0
            elif self.input_type == 'RQ':
                scale_node.layer[0].isRequantInputScale = 0
            elif self.scale_as_scale_region:
                scale_node.layer[0].isInputScale = 0
            self.graph.add_proto_node(scale_node)
            inp_scale = self.graph.add_nodes_with_shapes([f"{self.layer_name}_scale"], [new_shape], self.out_group_axis, self.out_group_size)

            return inp + inp_scale
        return inp
    def get_nickname(self):
        return super().get_nickname()
