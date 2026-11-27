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

# from proto_utils.layers.lnrm import LayerNorm
# from proto_utils.layers.input import Input
# from proto_utils.layers.conv import Conv
# from proto_utils.functional import reshape
# from proto_utils.layers.rope import Rope
# from proto_utils.layers.batchdot import BatchDot
# from proto_utils.layers.softmax import SoftMax
from dataclasses import dataclass
from typing import List
import yaml
from pathlib import Path


# TODO: I don't want deepcopy! figure out how to copy this easily
@dataclass
class NamedDim:
    dim_name: str
    dim: int

    def __hash__(self):
        return hash((self.dim_name, self.dim))
        

class NodeWithShape:
    def __init__(self, name, shape: List[NamedDim], out_group_axis: str, out_group_size: int):
        self.name = name 
        self.shape = shape
        assert(not out_group_axis or out_group_axis >=0)
        self.out_group_axis = out_group_axis
        self.out_group_size = out_group_size
    
    def __hash__(self):
        return hash((self.name, tuple(self.shape)))
    
    def __str__(self):
        return f"{self.name}: {self.shape}, \n out_group_axis = {self.out_group_axis} \n out_group_size = {self.out_group_size}"

class DAG:
    def __init__(self):
        self.proto_nodes = []
        self.nodes_with_shape = set()
        self.metadata = dict()
    def add_proto_node(self, node):
        self.proto_nodes.append(node)
    def add_nodes_with_shapes(self, node_names, node_shapes: List[NamedDim], out_group_axis: str, out_group_size: int):
        ans = []
        for node_name, node_shape in zip(node_names, node_shapes):
            new_node = NodeWithShape(node_name, node_shape, out_group_axis, out_group_size)
            if new_node in self.nodes_with_shape:
                raise "adding node twice!!"
            self.nodes_with_shape.add(new_node)
            ans.append(new_node)
        return ans
    def add_metadata(self, node_name, data):
        self.metadata[node_name] = data
    def get_proto(self, save_to=None):
        if not save_to:
            print(''.join(str(node) for node in self.proto_nodes))
        else:
            with open(save_to, "w+") as f:
                f.write(''.join(str(node) for node in self.proto_nodes))
            print(Path(save_to).absolute())
    
    def get_so_yaml(self, save_to=None):
        yaml_dict = {}
        for node_with_shape in self.nodes_with_shape:
            if not node_with_shape.out_group_axis:
                continue
            yaml_dict[node_with_shape.name] = {
                "outputAxis": node_with_shape.out_group_axis if node_with_shape.out_group_axis <=1 else node_with_shape.out_group_axis + 1,
                "outputGroupSize": node_with_shape.out_group_size
            }
        # update metadata
        # yaml_dict.update(self.metadata)
        # yaml_dict = {"LayerNames":self.metadata}
        if save_to:
            with open(save_to, "w") as file:
                yaml.dump({"LayerNames":yaml_dict}, file)
        else:
            print(yaml_dict)
    

# if __name__ == "__main__":
#     graph = DAG()
#     is_dynamic = True
#     lnorm  = LayerNorm("Layernorm", graph, uses_mean=True, output_mean_var=False, eps = 1e-06, is_dynamic=is_dynamic)
#     conv = Conv("Convolution", graph, 2048, bias_term=True, kernel_size=[2,2,2], stride=[2,2,2], is_dynamic=is_dynamic, out_group_axis=1)
#     rope = Rope('Rope', graph, broadcast=[0, 1.0, 0.5, 0], is_dynamic=is_dynamic)
#     batchdot = BatchDot('Batchdot', graph, is_dynamic=is_dynamic, out_group_axis=-1)
#     softmax = SoftMax('Softmax', graph, is_dynamic=is_dynamic)
#     hidden_dim = 1024
#     inp_shape = [1,1024,1024,hidden_dim]
#     inp = Input("input", graph, is_dynamic=is_dynamic, inp_shape=inp_shape, group_axis=1)()
#     lnorm_out = lnorm(inp,axis=4)
#     proj = conv(lnorm_out, derive_name_from_layer_name=True)
#     proj_reshape = reshape.reshape(graph, proj, [[1, 512, 2048, 512], [1, 512, 32, 512]])
#     proj_embedded = rope(proj_reshape, rope_name = "my_rope")
#     #todo: weird? fix it later
#     batchdot_inp_inp_norm = batchdot(inp + lnorm_out, derive_name_from_layer_name=True)
#     softmax_batchdot = softmax(batchdot_inp_inp_norm, axis = 3)
#     graph.get_proto()
#     print(batchdot_inp_inp_norm[0], batchdot_inp_inp_norm[1])

        