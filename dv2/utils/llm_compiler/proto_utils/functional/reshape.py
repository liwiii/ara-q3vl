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

from typing import TYPE_CHECKING, List, Self, Tuple
from copy import deepcopy
from caffe import layers as L
if TYPE_CHECKING:
    from proto_utils.graph import NodeWithShape, NamedDim
def _prod(arr):
    ans = 1
    for a in arr:
        ans*=a
    return ans

def locate_name(name, name_dims: List[Tuple[str, int, str]]):
    for (_name, new_dim, new_name) in name_dims:
        if name == _name:
            return (name, new_dim, new_name)
    return None, None, None
if TYPE_CHECKING:
    import graph
from proto_utils.utils_debug import print_shape_int


# To enable printing all shapes for debugging, uncomment following annotation.
# @print_shape_int
def reshape(graph: "graph.DAG", nodes: List['NodeWithShape'], dims_to_reshape_with_new_names_per_node: List[List[Tuple['str', int, 'str']]],out_group_axis_name: str, drop_dim = None):
    ans = []
    if len(dims_to_reshape_with_new_names_per_node) == 1:
        dims_to_reshape_with_new_names_per_node = [dims_to_reshape_with_new_names_per_node[0], dims_to_reshape_with_new_names_per_node[0]]
    for node_with_shape, dims_to_reshape_with_new_names in zip(nodes, dims_to_reshape_with_new_names_per_node):
        new_named_node = deepcopy(node_with_shape)
        curr_prod = 1
        new_prod = 1
        for i, dim in enumerate(new_named_node.shape):
            name, new_dim, new_name = locate_name(dim.dim_name, dims_to_reshape_with_new_names)
            if name:
                curr_prod *= new_named_node.shape[i].dim
                new_named_node.shape[i].dim = new_dim
                new_named_node.shape[i].dim_name = new_name 
                new_prod *= new_dim 
                
        if new_prod!=curr_prod:
            raise "Reshape params incorrect!"
        new_named_node.shape = [namedim for namedim in new_named_node.shape if not(namedim.dim_name == drop_dim and namedim.dim == 1)]       
        new_node = L.Reshape(bottom=node_with_shape.name, reshape_param=dict(shape=dict(dim=[dim.dim for dim in new_named_node.shape]))).to_proto()
        new_node.layer[0].top[0] = node_with_shape.name + "_reshape"
        new_node.layer[0].base_type =  "Reshape"
        new_node.layer[0].estimator_group =  "Reshape"
        new_node.layer[0].type = "Ara2" + "Reshape"
        new_node.layer[0].name = node_with_shape.name + "_reshape"
        graph.add_proto_node(new_node)
        # find out_group_axis
        out_group_axis = None
        for i in range(len(new_named_node.shape)):
            if new_named_node.shape[i].dim_name == out_group_axis_name:
                out_group_axis = i
                break
        ans.extend(graph.add_nodes_with_shapes([new_node.layer[0].name], [new_named_node.shape], out_group_axis = out_group_axis, out_group_size = node_with_shape.out_group_size))
    print(new_node.layer[0].name)
    return ans

def simple_reshape(graph: "graph.DAG", input_nodes: List['NodeWithShape'], output_nodes: List['NamedDim'],out_group_axis_name: str, drop_dim = None):
    dims_to_reshape_with_new_names_per_node = []
    for idx in range(len(input_nodes)):
        node_to_reshape = []
        for dim_idx in range(len(input_nodes[idx].shape)):
            node_to_reshape.append((input_nodes[idx].shape[dim_idx].dim_name, output_nodes[idx][dim_idx].dim, output_nodes[idx][dim_idx].dim_name))
        
        dims_to_reshape_with_new_names_per_node.append(node_to_reshape)
    return reshape(graph, input_nodes, dims_to_reshape_with_new_names_per_node, out_group_axis_name, None)


class ReshapeParam:
    def __init__(self, old_dim_name: str):
        self.old_dim_name: str | None = old_dim_name
        self.new_dim_name: str | None = None
        self.new_size: int = -1
        self.is_group = False
        self.to_drop = False

    def rename(self, new_dim_name):
        self.new_dim_name = new_dim_name
        return self

    def resize(self, new_size):
        self.new_size = new_size
        return self

    def group(self):
        self.is_group = True
        return self

    def drop(self):
        self.to_drop = True
        return self

    def to_tuples(self, group_size):
        assert(self.old_dim_name is not None)
        assert(self.new_size != -1)
        if self.new_dim_name is None:
            self.new_dim_name = self.old_dim_name 

        return ((self.old_dim_name, self.new_size, self.new_dim_name), (self.old_dim_name, self.new_size // (group_size if self.is_group else 1), self.new_dim_name))

    def _new_dim_name(self) -> str:
        if self.new_dim_name is None:
            self.new_dim_name = self.old_dim_name
        assert(self.new_dim_name is not None)
        return self.new_dim_name


class ReshapeMaker:
    def __init__(self, node: List['NodeWithShape'], group_size):
        self.node = node
        self._reshape_params: List[ReshapeParam] = []
        self.group_size = group_size

    def dim(self, dim_name) -> ReshapeParam:
        for named_dim in self.node[0].shape:
            if named_dim.dim_name == dim_name:
                self._reshape_params.append(ReshapeParam(dim_name))
                return self._reshape_params[-1]
        raise ValueError(f"{dim_name} not found in {self.node[0]}")

    def make_dims_to_reshape_with_new_names(self):
        res_node = []
        res_scale_node = []
        for reshape_param in self._reshape_params:
            dim_tuple, scale_dim_tuple = reshape_param.to_tuples(self.group_size)
            res_node.append(dim_tuple)
            res_scale_node.append(scale_dim_tuple)
        return [res_node, res_scale_node]

    def get_group_dim_name(self) -> str:
        for reshape_param in self._reshape_params:
            if reshape_param.is_group:
                return reshape_param._new_dim_name()
        raise ValueError(f"Group Dim not specified in {self._reshape_params}")

    def get_drop_dim_name(self):
        for reshape_param in self._reshape_params:
            if reshape_param.to_drop:
                return reshape_param._new_dim_name()
        return None

    def reset_to(self, new_node, group_size = None):
        self.node = new_node
        self._reshape_params = []
        self.group_size = group_size if group_size else self.group_size


def reshape_with_reshapemaker(graph, input_nodes, reshape_maker: ReshapeMaker):
    return reshape(
        graph,
        input_nodes,
        dims_to_reshape_with_new_names_per_node=reshape_maker.make_dims_to_reshape_with_new_names(),
        out_group_axis_name=reshape_maker.get_group_dim_name(),
        drop_dim=reshape_maker.get_drop_dim_name(),
    )
