from .layer import Layer
from caffe import layers as L  # type: ignore


class Comm(Layer):
    def __init__(
        self, layer_name, graph, out_group_axis, out_group_size, shard_idx, num_shards, operation, sync_group, num_inputs, is_dynamic = False
    ):
        super().__init__(layer_name, graph, out_group_axis, out_group_size)
        self.base_type = "CommNode"
        self.estimator_group = "CommNode" 
        self.shard_idx = shard_idx
        self.num_shards = num_shards
        self.operation = operation
        self.sync_group = sync_group
        self.num_inputs = num_inputs
        self.is_dynamic = is_dynamic # TODO: move this into base model

    def forward(self, bottom_nodes, top_name):
      new_node = L.CommNode(
        name = top_name[0],
        bottom = [bottom_nodes[i].name for i in range(len(bottom_nodes))]
      ).to_proto()
      new_node.layer[0].comm_node_param.shard_idx = self.shard_idx
      new_node.layer[0].comm_node_param.sync_group = self.sync_group
      new_node.layer[0].comm_node_param.num_shards = self.num_shards
      new_node.layer[0].comm_node_param.operation = self.operation
      new_node.layer[0].comm_node_param.num_inputs = self.num_inputs

      for (i, top_name) in enumerate(top_name):
        try:
            new_node.layer[0].top[i] = top_name
        except IndexError:
            new_node.layer[0].top.append(top_name)
            
      self.set_types(new_node)
      self.graph.add_proto_node(new_node)
      return new_node

    def get_top_shapes(self, inp_shapes):
        # Return shape as [1,1,1,<product of input_dims>*self.num_shards]
        out_shapes = inp_shapes
        result_shapes = []
        for shape in out_shapes:
            # Calculate product of all dimensions in the input shape
            product = 1
            for dim in shape:
                if dim.dim is not None:
                    product *= dim.dim
                    dim.dim = 1
            shape[self.out_group_axis].dim = product*self.num_shards
            
            # Return new shape [1,1,1,product*num_shards]
        # breakpoint()
        
        return out_shapes
    
    def get_nickname(self):
        return "comm_node"


# layer {                                                                                                                                                                                                                                                               [15/28055]
#   name: "Sync_Node"
#   type: "Ara2CommNode"
#   bottom: "data"
#   top: "Sync_Node"
#   base_type: "CommNode"
#   estimator_group: "CommNode"
#   comm_node_param {
#     shard_idx: 2
#     num_shards: 4
#     sync_group: 0
#     operation: SYNC
#     num_inputs: 1
#   }
# }