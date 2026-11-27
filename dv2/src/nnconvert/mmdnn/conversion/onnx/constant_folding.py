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

import networkx as nx
import onnx
import onnx.helper as OH
import onnx.numpy_helper as NH
import networkx.algorithms.dag as DA
import functools
import numpy as np
from onnx import TensorProto
import onnxruntime
import io
import itertools


class ConstantFoldingImpl:

  class Op:

    def __str__(self):
      return '<{} name={} shape={} is_const={}>'.format(self.op, self.name, self.shape, self.is_const)

    def __repr__(self):
      return str(self)

    def __init__(self, node, shape, G):
      self.name = node.name
      self.op = node.op_type
      self.g_node = node
      self.initializers = []
      self.is_const = None
      self.shape = shape
      G.add_node(self.name, data=self)

    def _initializers_to_const_node(self):
      initz = self.initializers
      self.initializers = []
      for init in initz:
        self.initializers.append(OH.make_node('Constant', name=init.name, inputs=[], outputs=[init.name], value=init))

    def fold(self, G, shape_info, type_info):
      r"""Implement constant folding - as nodes
            are being added to the graph, check if the 
            newest node's inputs are all constants, if they are
            them try evaluating the node using onnxrutime.
            """
      if self.op == 'Shape':
        shape = np.asarray(shape_info[self.g_node.input[0]])
        self.g_node = OH.make_node('Constant',
                                   name=self.name,
                                   inputs=[],
                                   outputs=self.g_node.output,
                                   value=OH.make_tensor(name=self.g_node.name, data_type=type_info[self.g_node.output[0]], dims=shape.shape, vals=shape))
        in_edges = G.in_edges(self.name)
        G.remove_edges_from(list(in_edges))
        self.is_const = True
        return
      elif self.op == 'Constant':
        self.is_const = True
        return
      ancestors = list(map(lambda x: G.nodes[x[0]]['data'], G.in_edges(self.name)))
      if len(ancestors) > 0:
        ancestors_const = list(map(lambda x: x.is_const, ancestors))
        self.is_const = functools.reduce(lambda a, b: a and b, ancestors_const)
      if self.is_const:
        # self._initializers_to_const_node()
        output_type = type_info[self.g_node.output[0]]
        output_shape = shape_info[self.g_node.output[0]]
        output_tensor = OH.make_tensor_value_info(self.g_node.output[0], output_type, output_shape)
        ancestors = list(map(lambda it: it.g_node, ancestors))
        graph = OH.make_graph(ancestors + [self.g_node], self.name, [], [output_tensor], initializer=self.initializers)
        model = OH.make_model(graph, producer_name="onnx-const-roller")
        model.ir_version = 6
        model.opset_import[0].version = 11
        # inferred_model = onnx.shape_inference.infer_shapes(model)
        inferred_model = OH.make_model(model.graph, producer_name="runtime_fixed")
        inferred_model.opset_import[0].version = 11
        serialized_model = io.BytesIO()
        onnx.save(inferred_model, serialized_model)
        serialized_model.seek(0)
        serialized_model = serialized_model.read()
        session = onnxruntime.InferenceSession(serialized_model)
        #binding = session.io_binding()
        #binding.bind_output(self.g_node.output[0])
        #session.run_with_iobinding(binding)
        data = np.asarray([0.0])
        result = session.run(None, {})[0]
        #result = binding.copy_outputs_to_cpu()[0]
        self.g_node = OH.make_node('Constant', name=self.name, inputs=[], outputs=self.g_node.output, value=NH.from_array(result))
        in_edges = G.in_edges(self.name)
        G.remove_edges_from(list(in_edges))
        self.is_const = True
        return

  def __init__(self, model):
    self.source_model = model
    self.G = nx.DiGraph()
    self.tensor_map = {}
    self.q = []

  def compile_model(self, input_nodes, output_nodes, shape_info, type_info, path):
    self.G.remove_nodes_from(list(nx.isolates(self.G)))
    sorted_nodes = list(DA.topological_sort(self.G))
    sorted_nodes = list(map(lambda x: self.G.nodes[x]['data'].g_node, sorted_nodes))
    # copy over the inputs in the next line
    graph = OH.make_graph(sorted_nodes, '{}_const-rolled'.format(self.source_model.graph.name), [], output_nodes)
    buf = io.BytesIO()
    # IMPORTANT: the list of inputs must contain the list of initializers as well
    # for shape inference to work properly, see:
    # https://github.com/onnx/onnx/issues/2903
    # TODO: delete from this all initialzier that we do not need
    graph.initializer.MergeFrom(self.source_model.graph.initializer)
    graph.input.MergeFrom(self.source_model.graph.input)
    model = OH.make_model(graph, producer_name='dv compiler')
    # import pdb; pdb.set_trace()
    model.ir_version = 6
    model.opset_import[0].version = 11
    onnx.checker.check_model(model)
    onnx.save(model, buf)
    buf.seek(0)
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level =\
        onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess_options.optimized_model_filepath = path
    # this should generate an optimized graph ath the
    # location specified by `path`
    onnxruntime.InferenceSession(buf.read(), sess_options)

  def contract_graph(self):
    mergeable_layers = set(['AveragePool', 'Pad', 'Reshape'])

    def check_if_siblings_equal(parent, x, y, level):
      """two sibling nodes are considered 
            equal if their attributes are equal 
            and if their parents are equal
            """
      xn = self.G.nodes[x]['data']
      yn = self.G.nodes[y]['data']
      if xn.op == yn.op and (xn.op == 'Constant' if level > 0 else True) and xn.g_node.attribute == yn.g_node.attribute:
        if level == 0 and xn.op not in mergeable_layers:
          return False
        x_in = list(map(lambda x: x[0], list(self.G.in_edges(x))))
        y_in = list(map(lambda x: x[0], list(self.G.in_edges(y))))
        if parent in x_in:
          x_in.remove(parent)
        if parent in y_in:
          y_in.remove(parent)
        if len(x_in) == len(y_in) == 0:
          return True
        results =\
            list(map(lambda arg: check_if_siblings_equal(None, arg[0], arg[1], level + 1),
                     list(zip(x_in, y_in))))
        return functools.reduce(lambda x, y: x and y, results)
      return False

    def merge_onnx_nodes(x, y):
      x = self.G.nodes[x]['data'].g_node
      y_outputs = list(self.G.nodes[y]['data'].g_node.output)
      for _, consumer in self.G.out_edges(y):
        consumer = self.G.nodes[consumer]
        for idx, ip in enumerate(consumer['data'].g_node.input):
          if ip in y_outputs:
            ip_idx = y_outputs.index(ip)
            consumer['data'].g_node.input[idx] =\
                x.output[ip_idx]

    for node in self.G.nodes:
      out_edges = list(map(lambda it: it[1], list(self.G.out_edges(node))))
      contracted = set()
      for x, y in itertools.combinations(out_edges, 2):
        if x in contracted or y in contracted:
          continue
        if check_if_siblings_equal(node, x, y, 0):
          newG =\
              nx.algorithms.minors.contracted_nodes(self.G,
                                                    x,
                                                    y)
          merge_onnx_nodes(x, y)
          self.G = newG
          contracted.add(y)

  def optimize_graph(self, buf):
    shape_info = {}
    tensor_mapping = {}
    type_info = {}
    initializers = {}
    input_node =\
        set([ip.name for ip in self.source_model.graph.input]) \
        - set([ip.name for ip in self.source_model.graph.initializer])
    input_node =\
        list(filter(lambda it: it.name == list(input_node)[0], self.source_model.graph.input))
    output_node_names =\
        set(map(lambda it: it.name, self.source_model.graph.output))
    for entry in self.source_model.graph.initializer:
      initializers[entry.name] = entry
    for entry in self.source_model.graph.value_info:
      type_info[entry.name] = entry.type.tensor_type.elem_type
      shape_info[entry.name] =\
          list(map(lambda x: x.dim_value,
                   entry.type.tensor_type.shape.dim))
    for _idx, node in enumerate(self.source_model.graph.node):
      if node.name in output_node_names:
        continue
      node = self.Op(node, None, self.G)
      if len(node.g_node.output) > 1:
        print(node.g_node)
      for edge in node.g_node.output:
        tensor_mapping[edge] = node
        node.shape = shape_info.get(edge, None)
      if _idx > 0:
        for edge in node.g_node.input:
          if edge not in tensor_mapping:
            if edge in initializers:
              node.initializers.append(initializers[edge])
              continue
            else:
              continue
          tensor = tensor_mapping[edge]
          self.G.add_edge(tensor.name, node.name)
      node.fold(self.G, shape_info, type_info)
    # for node in input_node :
    #     q.append(i)
    self.contract_graph()
    self.compile_model(input_node, self.source_model.graph.output, type_info, shape_info, buf)


if __name__ == '__main__':
  mp = '/Users/sriduth/code/sw/converter-models/vgg16ssd/vgg16ssd.onnx'
  mp = '/Users/sriduth/code/sw/converter-models/simba_pspnet/simba_pspnet2_20200524_1030/onnx_model.onnx'
  mp = '/auto/work/sandeep/onx/sw/candid/test/networks/dv_cfg0/onnx/resnet18/onnx_model.onnx'
  mp = '/auto/work/aditya/model_repos/osnet_ain_x1_0.onnx'
  mp = '/auto/work/aditya/model_repos/deep-person-reid/reid_1.8.0.onnx'
  mp = '/auto/work/sriduth/customer/mmdetection/converted/ssd512_coco.onnx'
  mp = '/auto/work/aditya/t1/sw/nx_to_onnx.onnx'
  mp = '/auto/work/aditya/test_models/centernet.onnx'
  # mp = '/auto/work/aditya/t1/sw/candid/test/networks/dv_cfg0/onnx/modelzoo/ssd_coco_512/nx_to_onnx.onnx'
  model = onnx.load_model(mp)
  model = onnx.shape_inference.infer_shapes(model)
  run = ConstantFoldingImpl(model)
  run.optimize_graph('./rt_optim.onnx')
# q = []
# for node in model :
#     q.append(node)
# starting nodes = []
# while not q :
#     q_len = len(q)
#     for i in range(0, q_len):
