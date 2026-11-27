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


from abc import abstractmethod
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import proto_utils.graph

class Layer:
    def __init__(self, layer_name, graph: "proto_utils.graph.DAG", out_group_axis, out_group_size):
        self.layer_name = layer_name
        self.graph = graph
        self.out_group_axis = out_group_axis
        self.out_group_size = out_group_size
    
    @abstractmethod
    def forward(self, bottom_nodes, **kwargs):
        pass
    @abstractmethod
    def get_top_shapes(self, inp_shapes):
        pass
    @abstractmethod
    def get_nickname(self):
        pass

    def get_top_names(self, bottom_names, derive_name_from_layer_name=False):
        if "rope" in self.layer_name:
            top_names = top_names = [f"{bottom_name}_{self.get_nickname()}" for bottom_name in bottom_names]
        else:
            top_names = [self.layer_name, self.layer_name + "_scale"]
        # top_names = [f"{bottom_name}_{self.get_nickname()}" for bottom_name in bottom_names]
        return top_names
    
    def set_types(self, top_node):
        top_node.layer[0].base_type =  self.base_type
        top_node.layer[0].estimator_group =  self.estimator_group
        top_node.layer[0].type = "Ara2" + self.base_type
        top_node.layer[0].name = top_node.layer[0].top[0]

    @abstractmethod
    def __call__(
        self,
        bottom_names_with_shape: "proto_utils.graph.NodeWithShape",
        derive_name_from_layer_name=False,
        **kwargs,
    ) -> "proto_utils.graph.NodeWithShape":
        bottom_names = [node_with_shape.name for node_with_shape in bottom_names_with_shape]
        bottom_shapes = [node_with_shape.shape for node_with_shape in bottom_names_with_shape]
        top_names = self.get_top_names(bottom_names, derive_name_from_layer_name)
        self.forward(bottom_names_with_shape, top_names, **kwargs)
        top_shapes = self.get_top_shapes(bottom_shapes)
        top_nodes_with_name = self.graph.add_nodes_with_shapes(top_names, top_shapes, self.out_group_axis, self.out_group_size)
        self.graph.add_metadata(top_names[0], {
            "outputAxis": self.out_group_axis if self.out_group_axis <=1 else self.out_group_axis + 1, #TODO: bug for DHW!
            "outputGroupSize": self.out_group_size
        })
        print(self.layer_name)
        return top_nodes_with_name



