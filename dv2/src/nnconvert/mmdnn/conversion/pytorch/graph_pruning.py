'''
Copyright (c) 2018-22, Deep Vision, Inc. All rights reserved.
Copyright (c) 2022-25, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

import torch.jit
import torch.onnx.utils
from onnx import TensorProto
from onnx import helper as OH
from onnx import numpy_helper as NH
import onnx
import onnxruntime
from mmdnn.conversion.common.utils import layerLikeObj
from collections import defaultdict, OrderedDict
import io
import numpy as np


class SubgraphEvaluation:

  def __init__(self, layer_args_order_map, layer_map, state_dict, shape_dict):
    self.layer_args_order_map = layer_args_order_map
    self.layer_map = layer_map
    self.state_dict = state_dict
    self.shape_dict = shape_dict

  @staticmethod
  def evaluate_subgraph(nodes, gnode, layer_map, state_dict, shape_dict):
    output_tensor = OH.make_tensor_value_info(gnode._layer.output[0], gnode.shape[1], gnode.shape[0])
    graph = OH.make_graph(nodes, gnode.name, [], [output_tensor])
    model = OH.make_model(graph, producer_name='runtime-graph')
    model.ir_version = 6
    model.opset_import[0].version = 11
    inferred_model = onnx.shape_inference.infer_shapes(model)
    inferred_model = OH.make_model(model.graph, producer_name='runtime_fixed')
    onnx.checker.check_model(inferred_model)
    serialized_model = io.BytesIO()
    onnx.save(inferred_model, serialized_model)
    serialized_model.seek(0)
    serialized_model = serialized_model.read()
    print(nodes)
    session = onnxruntime.InferenceSession(serialized_model)
    binding = session.io_binding()
    binding.bind_output(gnode._layer.output[0])
    session.run_with_iobinding(binding)
    return binding.copy_outputs_to_cpu()[0]

  @staticmethod
  def evaluate_gnode(gnode, layer_map, layer_args_order_map, state_dict, shape_dict):
    print('subgraph evaluation: ', gnode)
    from pprint import pprint as pp
    evaluator = SubgraphEvaluation(layer_args_order_map, layer_map, state_dict, shape_dict)
    evaluated = OrderedDict()
    nodes = evaluator.do_eval(gnode)
    evaluation_result = SubgraphEvaluation.evaluate_subgraph(nodes, gnode, layer_map, state_dict, shape_dict)
    return nodes, evaluation_result

  def do_eval(self, node):
    constant_nodes = []

    def _traverse(curr_node, const_list):
      if curr_node.is_node_deterministic(self.layer_map):
        const_list.append(curr_node)
        for initializer in curr_node.initializers:
          weight = self.state_dict[initializer]
          initializer =\
              self.mk_constant(initializer,
                               weight,
                               weight.dtype)
          initializer =\
              layerLikeObj(dict(type=initializer.op_type,
                                _layer=initializer))
          const_list.append(initializer)
        for in_edge in curr_node.in_edges:
          _traverse(self.layer_map[in_edge], const_list)

    _traverse(node, constant_nodes)
    return list(reversed(list(map(self.evaluate_possible, constant_nodes))))

  def evaluate_possible(self, gnode):
    if gnode.type != 'Shape':
      return gnode._layer
    input_node = gnode.in_edges[0]
    input_layer_shape = self.shape_dict[input_node]
    input_layer_shape = np.asarray(input_layer_shape)
    # input_layer_shape = np.asarray([input_layer_shape[0], input_layer_shape[2], input_layer_shape[1], input_layer_shape[3]])
    self.layer_map[input_node].out_edges = \
        list(filter(lambda it: it != gnode.name, self.layer_map[input_node].out_edges))
    return OH.make_node('Constant',
                        name=gnode._layer.name,
                        inputs=[],
                        outputs=gnode._layer.output,
                        value=OH.make_tensor(name=gnode._layer.name, data_type=gnode.shape[1], dims=gnode.shape[0], vals=input_layer_shape.flatten()))

  def mk_constant(self, name, value, dtype=np.int32):
    return SubgraphEvaluation.make_constant(name, value, dtype)

  @staticmethod
  def make_constant(name, value, dtype=np.int32):
    tensor = NH.from_array(value)
    return OH.make_node('Constant', name=name, inputs=[], outputs=[name], value=tensor)

  def get_arguments(self, gnode, evaluated):
    params = []
    params_idxes = []
    for param in gnode.runtime_params:
      params_idxes.append(self.layer_args_order_map[gnode.name][param])
      if param not in evaluated:
        value = self.state_dict[param].numpy()
        param_node = self.mk_constant(param, value)
        evaluated[param] = param_node
      params.append(param)
    clean_det_args = []
    det_args_idxes = []
    for arg in gnode.in_edges:
      arg_idx_map = self.layer_args_order_map[gnode.name]
      det_args_idxes.append(arg_idx_map[arg])
      res = evaluated.get(arg, None)
      if res is None:
        res = self.do_eval(self.layer_map[arg])
      clean_det_args.append(arg)
    args = params + clean_det_args
    idxes = params_idxes + det_args_idxes
    indexed_args = list(zip(args, idxes))
    indexed_args.sort(key=lambda it: it[1])
    return indexed_args
