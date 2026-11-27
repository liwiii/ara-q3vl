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

from typing import TYPE_CHECKING, List, Tuple
from itertools import repeat
from caffe import layers as L
if TYPE_CHECKING:
    import graph
from proto_utils.utils_debug import print_shape_int

# To enable printing all shapes for debugging, uncomment following annotation.
# @print_shape_int
def permute(graph: "graph.DAG", nodes_with_shapes: List["graph.NodeWithShape"], swap: Tuple[str, str], output_type: List[str] = []): 
    ans = []
    idx = 0
    for node_with_shape, swap in zip(nodes_with_shapes, repeat(swap)):
        permute_order = list(range(len(node_with_shape.shape)))
        idx_before = next(i for i,dim in enumerate(node_with_shape.shape) if dim.dim_name == swap[0])
        idx_after = next(i for i,dim in enumerate(node_with_shape.shape) if dim.dim_name == swap[1])
        permute_order[idx_before], permute_order[idx_after] = permute_order[idx_after], permute_order[idx_before]

        new_node = L.PermuteNnp(bottom=node_with_shape.name, permute_nnp_param=dict(order=permute_order)).to_proto()
        new_node.layer[0].top[0] = node_with_shape.name + "_permute"
        new_node.layer[0].base_type =  "PermuteNnp"
        new_node.layer[0].estimator_group =  "PermuteNnp"
        new_node.layer[0].type = "Ara2" + "PermuteNnp"
        new_node.layer[0].name = new_node.layer[0].top[0]
        if len(output_type) > idx:
            if(output_type[idx] == "K"):
                new_node.layer[0].isKeyOutput = 0
            if(output_type[idx] == "KS"):
                new_node.layer[0].isKeyOutputScale = 0
            if(output_type[idx] == "V"):
                new_node.layer[0].isValueOutput = 0
            if(output_type[idx] == "VS"):
                new_node.layer[0].isValueOutputScale = 0
        graph.add_proto_node(new_node)
        new_shape = list(node_with_shape.shape)
        for i, p_i in enumerate(permute_order):
            new_shape[i] = node_with_shape.shape[p_i]
        new_out_axis_dim = None
        if node_with_shape.out_group_axis:
            out_axis_dim = node_with_shape.shape[node_with_shape.out_group_axis]
            new_out_axis_dim = new_shape.index(out_axis_dim)
        ans.extend(graph.add_nodes_with_shapes([new_node.layer[0].name], [new_shape], out_group_axis=new_out_axis_dim, out_group_size=node_with_shape.out_group_size))
        print(new_node.layer[0].name)
        idx+=1
    return ans

# if __name__ == "__main__":
#     from proto_utils.graph import DAG
#     from proto_utils.layers.input import Input
#     graph = DAG()
#     is_dynamic = True
#     inp_shape = [1,1024,1024,576]
#     inp = Input("input", graph, is_dynamic=is_dynamic, inp_shape=inp_shape, group_axis=1)()
#     perm = permute(graph, inp, [[0,1,3,2], [0,1,3,2]])
#     graph.get_proto()
#     print(perm[0])

# layer {
#   name: "_model_layers_0_self_attn_Transpose_2_scale"
#   type: "Ara2PermuteNnp"
#   bottom: "_model_layers_0_self_attn_Reshape_2_scale"
#   top: "_model_layers_0_self_attn_Transpose_2_scale"
#   base_type: "PermuteNnp"
#   estimator_group: "PermuteNnp"
#   permute_nnp_param {
#     order: 0
#     order: 2
#     order: 1
#     order: 3
#   }
# }
# layer = L.PermuteNnp(bottom=bottom_name, permute_nnp_param=dict(order=order)).to_proto().layer[0]
