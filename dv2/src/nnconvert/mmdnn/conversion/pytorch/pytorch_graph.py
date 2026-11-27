'''
Portions of this file are derived from https://github.com/microsoft/MMdnn and are subject to the original MIT License

Copyright (c) Microsoft Corporation. All rights reserved.
SPDX-License-Identifier: MIT

Copyright (c) 2018 - 2025, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Modifications:

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

from mmdnn.conversion.common.DataStructure.graph import GraphNode, Graph
import torch
import torch.jit
import torch.autograd
import torch.serialization
from torch.onnx import ONNX_ARCHIVE_MODEL_PROTO_NAME, ExportTypes, OperatorExportTypes
import torch.onnx.utils
import contextlib
from torch.jit import _unique_state_dict
from collections import defaultdict
import onnx.helper as OH
import onnx.numpy_helper as NH
import onnx
from onnx import TensorProto
import numpy as np
import caffe2.python.onnx.backend as RT
import io
from random import randint
import re
import torch.nn.functional as F
from copy import deepcopy
from utils.dict2object import ObjectLike
from dv_logging import get_logger
logger = get_logger(__name__, 'DVGRPH')
NUMPY_ONNX_TYPE_MAP = {
    np.int8: TensorProto.INT8,
    np.int16: TensorProto.INT16,
    np.int32: TensorProto.INT32,
    np.int64: TensorProto.INT64,
    np.uint8: TensorProto.UINT8,
    np.uint16: TensorProto.UINT16,
    np.uint32: TensorProto.UINT32,
    np.uint64: TensorProto.UINT64,
    np.float: TensorProto.FLOAT,
    np.float64: TensorProto.FLOAT,
    np.dtype('float64'): TensorProto.FLOAT,
    np.dtype('int64'): TensorProto.INT64
}


class LayerLikeObj(ObjectLike):

  def __init__(self, entries):
    super(LayerLikeObj, self).__init__(**entries)


class PytorchGraphNode(GraphNode):

  def __init__(self, layer):

    def _get_attr(attr):
      value = None
      try:
        value = getattr(layer, layer.kindOf(attr))(attr)
      except Exception as e:
        value = []
      return value

    self._layer = layer
    self._name = layer.scopeName()
    self._kind = layer.kind()
    node_id = re.search(r"[\d]+", layer.__str__())
    self.id = node_id.group(0)
    super(PytorchGraphNode, self).__init__(layer)
    self.attrs = defaultdict(list)
    for k in layer.attributeNames():
      self.attrs[k] = _get_attr(k)
#        print('node name is', self._name, layer.__str__())
    self.weights_name = '.'.join(re.findall(r'\[([\w\d.]+)\]', self._name))
    # store weight keys taken as runtime
    self.runtime_params = list()
    self._st = layer.__str__()

  @property
  def name(self):
    name = self._name + self.id
    # Scopes created in a nested scope may have initial characters
    # that are illegal as the initial character of an op name
    # (viz. '-', '\', '/', and '_').
    name = name.replace('-', 'n').replace('\\', 'n').replace('/', 'n').replace('_', 'n').replace('[', 'n').replace(']', 'n')
    return name

  # TODO Unused code
  # @staticmethod
  # def get_name_and_id_for(layer):
  #   scope_name = layer.scopeName()
  #   node_id = re.search(r"[\d]+", layer.__str__())
  #   node_id = node_id.group(0)
  #   if not scope_name:
  #     scope_name = '{}_generated'.format(node_id)
  #   node_name = scope_name + node_id
  #   node_name = node_name.replace('-','n') \
  #                        .replace('\\','n') \
  #                        .replace('/','n') \
  #                        .replace('_','n') \
  #                        .replace('[','n') \
  #                        .replace(']','n')
  #   obj_args = dict(node_id=node_id, node_name=node_name)
  #   return ObjectLike(obj_args)

  @property
  def type(self):
    return self._kind

  @property
  def pytorch_layer(self):
    return self.layer

  def __str__(self):
    return 'PyTorchGraphNode({}, {})'.format(self.name, self.type)

  def __repr__(self):
    return self.__str__()


# See https://github.com/pytorch/pytorch/issues/33463
class _ScopeNameWorkaround(object):

  def __init__(self):
    self.backup = None

  def __enter__(self):

    def _tracing_name(self_, tracing_state):
      if not tracing_state._traced_module_stack:
        return None
      module = tracing_state._traced_module_stack[-1]
      for name, child in module.named_children():
        if child is self_:
          return name
      return None

    def _slow_forward(self_, *input, **kwargs):
      # pylint: disable=I1101
      tracing_state = torch._C._get_tracing_state()
      if not tracing_state or isinstance(self_.forward, torch._C.ScriptMethod):
        return self_.forward(*input, **kwargs)
      if not hasattr(tracing_state, '_traced_module_stack'):
        tracing_state._traced_module_stack = []
      name = _tracing_name(self_, tracing_state)
      if name:
        tracing_state.push_scope('%s[%s]' % (self_._get_name(), name))
      else:
        tracing_state.push_scope(self_._get_name())
      tracing_state._traced_module_stack.append(self_)
      try:
        result = self_.forward(*input, **kwargs)
      finally:
        tracing_state.pop_scope()
        tracing_state._traced_module_stack.pop()
      return result

    self.backup = torch.nn.Module._slow_forward
    setattr(torch.nn.Module, '_slow_forward', _slow_forward)

  def __exit__(self, type, value, tb):
    setattr(torch.nn.Module, '_slow_forward', self.backup)


class PytorchGraph(Graph):

  def __init__(self, model):
    # sanity check.
    super(PytorchGraph, self).__init__(model)
    self.model = model  #LayerLikeObj(dict(state_dict=lambda **kwargs: self.model))
    self.state_dict = _unique_state_dict(self.model)
    self.params_dict = dict()
    self.shape_dict = dict()
    self._onnx_layer_type_map = defaultdict(list)
    self.layer_args_order_map = defaultdict(dict)

  @staticmethod
  def _optimize_graph(graph, aten, export_raw_ir=False):
    # pylint: disable=I1101
    # run dce first to eliminate dead parts of the graph that might have been
    # left behind by things like symbolic_override
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)
    torch._C._jit_pass_peephole(graph)
    torch._C._jit_pass_lint(graph)
    if True:
      graph = torch._C._jit_pass_onnx(graph, aten)
      torch._C._jit_pass_lint(graph)
      torch._C._jit_pass_onnx_peephole(graph)
      torch._C._jit_pass_lint(graph)
    torch._C._jit_pass_dce(graph)
    torch._C._jit_pass_lint(graph)
    graph = torch._C._jit_pass_canonicalize(graph)
    torch._C._jit_pass_lint(graph)
    return graph

  @staticmethod
  def get_node_id(node):
    node_id = re.search(r"[\d]+", node.__str__())
    return node_id.group(0)

  @contextlib.contextmanager
  def set_training(self, model, mode):
    r"""
        A context manager to temporarily set the training mode of 'model'
        to 'mode', resetting it when we exit the with-block.  A no-op if
        mode is None.
        """
    if mode is None:
      yield
      return
    old_mode = model.training
    if old_mode != mode:
      model.train(mode)
    try:
      yield
    finally:
      if old_mode != mode:
        model.train(old_mode)

  def _reshape_get_output_shape(self, gnode):
    input_shape = self.shape_dict[gnode.in_edges[0]]
    my_input_shape = deepcopy(input_shape)
    for _idx, reshape_dim in enumerate(gnode.attrs['shape']):
      if reshape_dim == -1:
        continue
    my_input_shape[_idx] = reshape_dim
    return my_input_shape

  def remove_unrequired_reshapes(self):
    # TODO: implement this in c2c
    logger.debug('removing unrequired reshapes')
    for gnode in self._onnx_layer_type_map['onnx::Reshape']:
      gnode = self.layer_map[gnode]
      previous_input_shape = self.shape_dict[gnode.name]
      calculated_output_shape = self._reshape_get_output_shape(gnode)
      logger.debug('checking reshape %s %s', previous_input_shape, calculated_output_shape)
      if previous_input_shape == calculated_output_shape:
        # both are same - remove this node!
        def _replace_reshape_link(parent_out_edge):
          if parent_out_edge == gnode.name:
            return gnode.out_edges[0]
          else:
            return parent_out_edge

        for parent in gnode.in_edges:
          parent = self.layer_map[parent]
          parent.out_edges = list(map(_replace_reshape_link, parent.out_edges))
        del self.layer_map[gnode.in_edges[1]]
        del self.shape_dict[gnode.in_edges[1]]
        del self.layer_map[gnode.name]
        del self.shape_dict[gnode.name]

  def patch_reshape_dimensions(self):
    logger.debug('fixing output shapes for resize %s', self._onnx_layer_type_map['onnx::Reshape'])
    for gnode in self._onnx_layer_type_map['onnx::Reshape']:
      gnode = self.layer_map[gnode]
      my_input_shape = self._reshape_get_output_shape(gnode)
      self.shape_dict[gnode.name] = my_input_shape
      # old_shape_vector = self.shape_dict[gnode.name]
      # shape_vector = resize_param_node.attrs['value']
      # if len(shape_vector) > 0:
      #     print('fixing outut shape for layer ', gnode.name,
      #           'old: ', old_shape_vector,
      #           'new: ', shape_vector )
      #     self.shape_dict[gnode.name] = shape_vector
  # def add_missing_dims(self):
  #   for gnode, shape in self.shape_dict:
  #     gnode = self.layer_map[gnode]
  #     my_input_tensor = self.layer_map[gnode.in_edges[0]]
  #     my_input_tensor_shape = self.shape_dict[my_input_tensor.name]
  #     my_output_shape = deepcopy(my_input_tensor_shape)
  #     for _idx, output_dim in enumerate(shape):
  #       if output_dim == -1:
  #         continue
  #       my_output_shape

  def try_infer_missing_shapes(self):
    for gnode in self.layer_map.values():
      if gnode.name not in self.shape_dict:
        node_shape = self._try_infer_shape(gnode)
        if node_shape is None:
          logger.warn('could not infer output for %s', gnode.name)
        else:
          logger.info('inferred output shape for %s : %s', gnode.name, node_shape)
          self.shape_dict[gnode.name] = node_shape

  def _try_infer_shape(self, gnode):
    r"""
        Given a node, try getting the shape of the output tensor.
        We use this in the case where `onnx.infer_shape` needs too
        much ceremony to get an output shape.
        """
    logger.debug('trying to infer shape for %s', gnode.name)
    shape = None
    if gnode.type == 'onnx::Constant':
      shape = list(gnode.attrs['value'].numpy().shape)
    if gnode.type == 'onnx::Pad':
      input_to_be_padded = self.layer_map[gnode.in_edges[0]]
      padding_value = self.layer_map[gnode.in_edges[1]]
      pad_input_shape = self.shape_dict[input_to_be_padded.name]
      dummy_input = torch.randn(pad_input_shape)
      pad_output = F.pad(dummy_input, tuple(padding_value.attrs['value'].numpy()))
      shape = list(pad_output.numpy().shape)
    return shape

  def __check_outbound_type(self, nodelist, dtype):
    items = defaultdict(list)
    children_dtype = True
    for node in nodelist:
      node = self.layer_map[node]
      for edge in node.out_edges:
        out_edge = self.layer_map[edge]
        if (out_edge.type == dtype):
          items[node].append(edge)
        children_dtype = children_dtype and out_edge.type == dtype
    return items, children_dtype

  def prune_normalize_layer(self):
    """removes the terminal ReduceL2 layer
        in a pytorch computation graph. 
        TODO: This will be removed
        """
    logger.warn('pruning all terminal normalize layers')
    for resize_layer in self._onnx_layer_type_map['onnx::ReduceL2']:
      current_gnode = self.layer_map[resize_layer]
      resize_layer = current_gnode
      if len(resize_layer.in_edges) > 1:
        break
      main_path = [(0, 'onnx::Clip'), (0, 'onnx::Expand'), (0, 'onnx::Div')]
      ops = [current_gnode]
      for arg_idx, child_ltype in main_path:
        child_gnode = current_gnode.out_edges[arg_idx]
        child_gnode = self.layer_map[child_gnode]
        if child_gnode.type == child_ltype:
          current_gnode = child_gnode
          ops.append(current_gnode)
        else:
          break
      if ops[3].in_edges[0] == resize_layer.in_edges[0]:
        logger.info('subgraph is valid one and pruned')
      parent = self.layer_map[resize_layer.in_edges[0]]
      #            parent.out_edges = list(filter(lambda x: not (x == resize_layer.name), parent.out_edges))
      if len(ops) == 4:
        for gnode in reversed(ops):
          logger.debug('dropping node %s', gnode.name)
          if gnode.type == 'onnx::Expand':
            self.layer_map.pop(gnode.in_edges[1], None)
            self.shape_dict.pop(gnode.in_edges[1], None)
            parent.out_edges = \
                list(filter(lambda it: not(it == gnode.in_edges[1]), parent.out_edges))
          parent.out_edges = \
              list(filter(lambda it: not(it == gnode.name), parent.out_edges))
          del self.layer_map[gnode.name]
          del self.shape_dict[gnode.name]

  def prune_reachable_nodes(self):
    import functools
    flatten = lambda data: functools.reduce(lambda l1, l2: l1 + l2, data.values())

    def _are_inputs_deterministic(node, path=[], **kwargs):
      if type(node) == str:
        node = self.layer_map[node]
      is_deterministic = True
      curr_deterministic_inputs = []
      curr_non_deterministic_inputs = []
      for idx, in_edge in enumerate(node.in_edges):
        in_node = self.layer_map[in_edge]
        if in_node.type in kwargs['deterministic_inputs']:
          is_deterministic = is_deterministic and True
        else:
          possible_transitions = kwargs['possible_transitions']
          exclude_args = kwargs['exclude_args']
          next_item = possible_transitions.get(node.type, None)
          if idx in exclude_args.get(node.type, set()):
            curr_non_deterministic_inputs.append(in_node)
          elif next_item == in_node.type:
            print('visiting', idx, node, in_node)
            is_next_node_det, _ = _are_inputs_deterministic(in_node, path, **kwargs)
            is_deterministic = is_deterministic and is_next_node_det
          else:
            is_deterministic = False
        if is_deterministic:
          curr_deterministic_inputs.append(in_node)
        else:
          curr_non_deterministic_inputs.append(in_node)
      path.append(dict(node=node, deterministic_inputs=curr_deterministic_inputs, non_deterministic_inputs=curr_non_deterministic_inputs))
      return is_deterministic, path

    def __do_eval(it, evaluated):
      if type(it) == PytorchGraphNode:
        it = evaluated['_get_item_for'](it)
      node = it['node']
      evaluators = {
          'onnx::Shape': __evaluate_shape,
          'onnx::Constant': __evaluate_constant,
          'onnx::Slice': __evaluate_slice,
          'onnx::Concat': __evaluate_concat,
          'onnx::Unsqueeze': __evaluate_unsqueeze,
          'onnx::Gather': __evaluate_gather
      }
      evaluated[node.name] = \
          evaluators[node.type](node, it.get('deterministic_inputs', []), evaluated)
      return evaluated[node.name]

    def __mk_constant(name, value, dtype=np.int32):
      onnx_tensor_dtype = NUMPY_ONNX_TYPE_MAP[dtype]
      return OH.make_node('Constant',
                          name=name,
                          inputs=[],
                          outputs=[name],
                          value=OH.make_tensor(name='{}_tensor'.format(name), data_type=onnx_tensor_dtype, dims=value.shape, vals=value.flatten().astype(dtype)))

    def __get_arguments(gnode, det_args, evaluated):
      params = []
      params_idxes = []
      for param in gnode.runtime_params:
        params_idxes.append(self.layer_args_order_map[gnode.name][param])
        if param not in evaluated:
          value = self.state_dict[param].numpy()
          param_node = __mk_constant(param, value)
          evaluated[param] = param_node
        params.append(param)
      clean_det_args = []
      det_args_idxes = []
      for arg in det_args:
        arg_idx_map = self.layer_args_order_map[gnode.name]
        det_args_idxes.append(arg_idx_map[arg.name])
        res = evaluated.get(arg.name, None)
        if res is None:
          res = __do_eval(arg, evaluated)
        clean_det_args.append(arg.name)
      args = params + clean_det_args
      idxes = params_idxes + det_args_idxes
      indexed_args = list(zip(args, idxes))
      indexed_args.sort(key=lambda it: it[1])
      return indexed_args

    def __evaluate_constant(gnode, det_args, evaluated):
      values = np.asarray(gnode.attrs['value'])
      dtype = values.dtype
      return __mk_constant(gnode.name, values, dtype=dtype)

    def __evaluate_gather(gnode, det_args, evaluated):
      arguments = __get_arguments(gnode, det_args, evaluated)
      in_tensors = list(map(lambda it: it[0], arguments))
      attrs = dict(gnode.attrs)
      return OH.make_node('Gather', inputs=in_tensors, outputs=[gnode.name], **attrs)

    def __evaluate_unsqueeze(gnode, det_args, evaluated):
      arguments = __get_arguments(gnode, det_args, evaluated)
      in_tensors = list(map(lambda it: it[0], arguments))
      attrs = dict(gnode.attrs)
      return OH.make_node('Unsqueeze', inputs=in_tensors, outputs=[gnode.name], **attrs)

    def __evaluate_shape(gnode, det_args, evaluated):
      input_layer_shape = self.shape_dict[gnode.in_edges[0]]
      input_layer_shape = np.asarray(input_layer_shape)
      return OH.make_node('Constant',
                          name=gnode.name,
                          inputs=[],
                          outputs=[gnode.name],
                          value=OH.make_tensor(name='{}::shape::evaluated_tensor'.format(gnode.name),
                                               data_type=TensorProto.INT32,
                                               dims=input_layer_shape.shape,
                                               vals=input_layer_shape.flatten()))

    def __evaluate_slice(gnode, det_args, evaluated):
      arguments = __get_arguments(gnode, det_args, evaluated)
      in_tensors = list(map(lambda it: it[0], arguments))
      node = OH.make_node('Slice', name=gnode.name, inputs=in_tensors, outputs=[gnode.name], **gnode.attrs)
      return node

    def __evaluate_concat(gnode, det_args, evaluated):
      input_tensors = []
      params = gnode.runtime_params
      arguments = __get_arguments(gnode, det_args, evaluated)
      arguments = list(map(lambda it: it[0], arguments))
      attrs = gnode.attrs
      node = OH.make_node('Concat', name=gnode.name, inputs=arguments, outputs=[gnode.name], axis=attrs['axis'])
      evaluated[gnode.name] = node
      return node

    def _evaluate_path(path):
      evaluated = dict()

      def __get_path_node_for(gnode):
        for item in path:
          if item['node'] == gnode:
            return item
        return {'node': gnode, 'deterministic_inputs': []}

      evaluated['_get_item_for'] = __get_path_node_for
      # build onnx nodes
      output_node = None
      for item in path:
        if output_node is None:
          output_node = item['node']
        __do_eval(item, evaluated)
      evaluated.pop('_get_item_for')
      nodes = list(filter(lambda n: type(n) is not TensorProto, evaluated.values()))
      graph = OH.make_graph(nodes, output_node.name, [], [])
      model = OH.make_model(graph, producer_name="runtime-graph")
      model.ir_version = 6
      model.opset_import[0].version = 11
      inferred_model = onnx.shape_inference.infer_shapes(model)
      inferred_model_layer_shapes = inferred_model.graph.value_info
      for layer in inferred_model_layer_shapes:
        if layer.name == output_node.name:
          dims = layer.type.tensor_type.shape.dim
          dims = tuple(map(lambda it: it.dim_value, dims))
          output_tensor = OH.make_tensor_value_info(output_node.name, TensorProto.INT32, dims)
          graph = OH.make_graph(nodes, output_node.name, [], [output_tensor])
          model = OH.make_model(graph, producer_name="runtime-graph")
          model.ir_version = 6
          model.opset_import[0].version = 11
      inferred_model = OH.make_model(inferred_model.graph, producer_name='runtime_fixed')
      onnx.checker.check_model(inferred_model)
      serialized_model = io.BytesIO()
      onnx.save(inferred_model, serialized_model)
      serialized_model.seek(0)
      serialized_model = serialized_model.read()
      prepared_model = RT.prepare(model)
      outputs = prepared_model.run([])
      return outputs[0], evaluated

    def _squash_till_unsqueeze(shape_gather_map, gather_unsqueeze_map, unsqueeze_concat_map, concat_reshape_map):
      pruned = []
      for shape, gathers in shape_gather_map.items():
        for gather, unsqueezes in gather_unsqueeze_map.items():
          if gather.name in gathers:
            for unsqueeze, concats in unsqueeze_concat_map.items():
              if unsqueeze.name in unsqueezes:
                for concat, reshape in concat_reshape_map.items():
                  if concat.name in concats:
                    for rs in reshape:
                      is_deterministic, path = \
                          _are_inputs_deterministic(
                              rs,
                              [],
                              exclude_args={
                                  'onnx::Reshape': {0}
                              },
                              deterministic_inputs={'onnx::Shape',
                                                    'onnx::Constant'},
                              possible_transitions={
                                  'onnx::Reshape': 'onnx::Concat',
                                  'onnx::Concat': 'onnx::Unsqueeze',
                                  'onnx::Unsqueeze': 'onnx::Gather',
                                  'onnx::Gather': 'onnx::Shape'
                              })
                      if is_deterministic:
                        path_filtered = \
                            reversed(list(filter(lambda it: len(it['non_deterministic_inputs']) == 0, path)))
                        output, heap = _evaluate_path(path_filtered)
                        pruned.append((rs, path, output, heap))
      return pruned

    def _squash_till_resize(shape_slice_map, slice_concat_map, concat_resize_map):
      """identify a subsection of the graph of a known pattern.
            1 - generate a set of nodes for which input can be inferred AOT
            2 - evaluate the sub graph
            3 - store result as a data node, 
            4 - prune subgraph from main graph
            """
      pruned = []
      for concat, resizes in concat_resize_map.items():
        for _slice, concats in slice_concat_map.items():
          if concat.name in concats:
            for shape, slices in shape_slice_map.items():
              if _slice.name in slices:
                for rs in resizes:
                  is_deterministic, path = \
                      _are_inputs_deterministic(rs,
                                                [],
                                                exclude_args={
                                                    'onnx::Resize': {0}
                                                },
                                                deterministic_inputs={'onnx::Shape', 'onnx::Constant'},
                                                possible_transitions={
                                                    'onnx::Resize': 'onnx::Concat',
                                                    'onnx::Concat': 'onnx::Slice',
                                                    'onnx::Slice': 'onnx::Shape'
                                                })
                  if is_deterministic:
                    path_filtered = \
                        reversed(
                            list(
                                filter(lambda it: len(it['non_deterministic_inputs']) == 0,
                                       path)
                            )
                        )
                    output, heap = _evaluate_path(path_filtered)
                    pruned.append((rs, path, output, heap))
      return pruned

    def _squash_l2norm(reducel2_clip_map, clip_expand_map, expand_div_map):
      """ current ONNX operator export capability of pytorch
            exports the L2 normalization operation as a set of multiple
            nodes in the output ONNX graph.
            Since we support L2 normalization as a whole internally, 
            we find and remove the subgraph that represents the l2 normalization
            operation and replace it with a new node with the required attrs.
            """
      pruned = []
      for reducel2, clips in reducel2_clip_map.items():
        for clip, expands in clip_expand_map.items():
          if clip.name in clips:
            for expand, divs in expand_div_map.items():
              if expand.name in expands:
                for div in divs:
                  is_deterministic, path = \
                      _are_inputs_deterministic(
                          div,
                          [],
                          exclude_args={},
                          deterministic_inputs={'onnx::Shape',
                                                'onnx::Constant'},
                          possible_transitions={
                              'onnx::Div': 'onnx::Expand',
                              'onnx::Expand': 'onnx::Clip',
                              'onnx::Clip': 'onnx::ReduceL2'
                          })
                  reduce_parent = self.layer_map[reducel2.in_edges[0]]
                  l2out_node = list(filter(lambda it: it['node'].type == 'onnx::Div', path))[0]
                  l2out_node = l2out_node['node']
                  intermediate_node = \
                      LayerLikeObj(dict(name='{}_replaced'.format(reducel2.name),
                                        type='dv::L2Normalize',
                                        in_edges=[reduce_parent.name],
                                        out_edges=l2out_node.out_edges
                      ))
                  self.shape_dict[intermediate_node.name] = self.shape_dict[l2out_node.name]
                  self.layer_map[intermediate_node.name] = intermediate_node
                  reduce_parent.out_edges = [intermediate_node.name]
                  for item in path:
                    del self.layer_map[item['node'].name]

    """for the shape operator, check what are the input layers and fold we reach a resize operator.
        when we reach the resize operators check if we can calculate the inputs for it inplace
        """

    def _check_outbound_type(items, onnx_type):
      try:
        if type(items) is not list:
          items = flatten(items)
      except Exception as e:
        items = []
      return self.__check_outbound_type(items, onnx_type)

    logger.debug('layers used in model: %s', self._onnx_layer_type_map.keys())
    ltmap = self._onnx_layer_type_map
    slices, all_slices = _check_outbound_type(ltmap['onnx::Shape'], 'onnx::Slice')
    concats, all_concats = _check_outbound_type(slices, 'onnx::Concat')
    resizes, all_resizes = _check_outbound_type(concats, 'onnx::Resize')
    concat_pruned_graphs = _squash_till_resize(slices, concats, resizes)
    gathers, all_gather = _check_outbound_type(ltmap['onnx::Shape'], 'onnx::Gather')
    unsqueezes, all_unsqueezes = _check_outbound_type(gathers, 'onnx::Unsqueeze')
    concats, all_concats = _check_outbound_type(unsqueezes, 'onnx::Concat')
    reshapes, all_reshapes = _check_outbound_type(concats, 'onnx::Reshape')
    unsqueeze_pruned_graphs = _squash_till_unsqueeze(gathers, unsqueezes, concats, reshapes)
    pruned_graphs = concat_pruned_graphs + unsqueeze_pruned_graphs
    for resize_node, _, pruned_graph_value, intermediate_nodes in pruned_graphs:
      pruned_node_scope_name = 'pruned'
      intermediate_node_names = intermediate_nodes.keys()

      def _remove_pruned_graph_edge(it):
        connection = self.layer_map[it]
        return (connection.name not in intermediate_node_names)
      self.layer_map[resize_node].in_edges = \
          list(filter(_remove_pruned_graph_edge, self.layer_map[resize_node].in_edges))
      for node_name in intermediate_nodes.keys():
        node = self.layer_map.pop(node_name, None)
        if node is None:
          continue
        else:
          pruned_node_scope_name += '_{}'.format(node.id)
          logger.info('pruning node: %s type: %s ', node_name, node.type)
        for in_edge in node.in_edges:
          if in_edge in self.layer_map:
            logger.info('removing reference to node: %s from: %s', node_name, in_edge)
            user = self.layer_map[in_edge]
            user.out_edges = list(filter(lambda x: x != node.name, user.out_edges))
      pruned_node_scope_name += '[{}]'.format(randint(7000, 9000))
      pruned_graph_node = dict(scopeName=lambda: pruned_node_scope_name,
                               kind=lambda: 'onnx::Constant',
                               __str__=lambda: pruned_node_scope_name,
                               attributeNames=list,
                               _source=resize_node)
      pruned_graph_node = LayerLikeObj(pruned_graph_node)
      pruned_graph_node = PytorchGraphNode(pruned_graph_node)
      pruned_graph_node.attrs = defaultdict(list)
      pruned_graph_node.attrs['value'] = pruned_graph_value
      #self.layer_name_map[pruned_graph_node.name] = pruned_graph_node.name
      #self.layer_map[pruned_graph_node.name] = pruned_graph_node
      #self.shape_dict[pruned_graph_node.name] = list(pruned_graph_value.shape)
      logger.debug('%s and all reachable ancestors reduced to %s value: %s', resize_node, pruned_graph_node.name, pruned_graph_value)
      self.layer_map[resize_node].attrs['shape'] = pruned_graph_value
      #self._make_connection(pruned_graph_node.name, resize_node)
    clips, all_clip = _check_outbound_type(ltmap['onnx::ReduceL2'], 'onnx::Clip')
    expands, all_expands = _check_outbound_type(clips, 'onnx::Expand')
    divs, all_divs = _check_outbound_type(expands, 'onnx::Div')
    _squash_l2norm(clips, expands, divs)
    return

  def _inspect_arguments(self, gnodes):
    arguments = defaultdict(dict)
    for gnode in gnodes:
      gnode = self.layer_map[gnode]
      for _idx, in_edge in enumerate(gnode.in_edges):
        edge = self.layer_map[in_edge]
        if edge.type == 'onnx::Constant':
          arguments[gnode.name][_idx] = edge.attrs['value']
        else:
          arguments[gnode.name][_idx] = \
              '{} = {}'.format(edge.__str__(), self.shape_dict[edge.name])
    return arguments

  def _follow_outputs(self, gnodes):
    outputs = {}
    for gnode in gnodes:
      gnode = self.layer_map[gnode]
      out_edges = gnode.out_edges
      if len(out_edges) == 0:
        return gnode
      else:
        outputs[gnode.name] = self._follow_outputs(out_edges)
    return outputs

  def collapse_pad_node_args_to_pad(self):
    """for onnx padding layers, the constant node
        that represents tha padding values is removed
        and set as a prop on the padding OP node itself.
        If there are dangling DummyData nodes in the
        final caffe output because of pad node's shape
        node, the final model will be invalid.
        """
    for pad_node in self._onnx_layer_type_map['onnx::Pad']:
      if pad_node not in self.layer_map:
        continue
      pad_node = self.layer_map[pad_node]
      pad_shape_node = self.layer_map[pad_node.in_edges[1]]
      self.layer_map.pop(pad_node.in_edges[1])
      pad_node.attrs['padding_attrs'] = pad_shape_node.attrs

  def collapse_clip_to_relu6(self):
    """the threshold arguments for clip operator
        are given as seperate constant nodes to the clip OP
        node, remove them and set it as a prop on the
        OP node
        """
    for clip_node in self._onnx_layer_type_map['onnx::Clip']:
      if clip_node not in self.layer_map:
        continue
      clip_node = self.layer_map[clip_node]
      if len(clip_node.in_edges) is not 3:
        continue
      min_val = self.layer_map[clip_node.in_edges[1]]
      max_val = self.layer_map[clip_node.in_edges[2]]
      if min_val.attrs['value'] == '0' and max_val.attrs['value'] == '6':
        clip_node.attrs['min'] = 0
        clip_node.attrs['max'] = 6
      clip_node.in_edges = [clip_node.in_edges[0]]
      self.layer_map.pop(min_val.name, None)
      self.layer_map.pop(max_val.name, None)

  def build(self, shape):
    """
        build graph for pytorch 0.4.0
        """
    from torch.onnx.symbolic_helper import _set_opset_version
    _set_opset_version(11)
    dummy_input = torch.autograd.Variable(torch.randn(shape), requires_grad=False)
    with _ScopeNameWorkaround():
      trace, _params_dict, _out = \
          torch.onnx.utils._model_to_graph(self.model,
                                           (dummy_input, ),
                                           _retain_param_name=True)
    logger.debug('params after trace %s', _params_dict.keys())
    logger.debug('state dict keys %s', self.state_dict.keys())
    self.state_dict = _params_dict
    with open('./trace_graph.txt', 'w') as f:
      f.write(trace.__str__())
    with open('./model_graph.txt', 'w') as f:
      f.write(self.model.__str__())
    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
    nodes = trace.nodes()
    np.load = np_load_old
    const_users = open('./constant.txt', 'w')
    nodes = list(nodes)
    for node in nodes:
      print('node is', node)
      op = list(node.outputs())[0]
      node_id = PytorchGraph.get_node_id(node)
      node_scope = node.scopeName()
      node_name = node_scope + node_id
      node_name = node_name.replace('-','n') \
                           .replace('\\','n') \
                           .replace('/','n') \
                           .replace('_','n') \
                           .replace('[','n') \
                           .replace(']','n')
      curr_node = PytorchGraphNode(node)
      self._onnx_layer_type_map[curr_node.type].append(node_name)
      try:
        op_type = op.type()
        output_shape = None
        if op_type is not None:
          output_shape = op_type.sizes()
      except Exception as e:
        output_shape = []
        import traceback
        traceback.print_exc()
      self.layer_map[curr_node.name] = curr_node
      self.layer_name_map[curr_node.name] = curr_node.name
      if output_shape is not None:
        self.shape_dict[curr_node.name] = output_shape
      node_inputs = list(node.inputs())
      for _input_idx, node_input in enumerate(node_inputs):
        # input_node_name_details = PytorchGraphNode.get_name_and_id_for(node_input.node())
        # input_node_id = input_node_name_details.node_id
        # input_node_name = input_node_name_details.node_name
        # is_param = node_input.node().kind() == 'prim::Param'
        # if is_param:
        #     curr_node.runtime_params.append(node_input.debugName())
        input_node_id = PytorchGraph.get_node_id(node_input.node())
        input_node_scope_name = node_input.node().scopeName()
        if node_input.node().kind() == 'onnx::Constant':
          const_users.write('{} \t {}\n'.format(node.__str__().rstrip(), node_input.node().__str__().rstrip()))
        if input_node_id and node_input.node().kind() == 'prim::Param':
          # print('got param node: ', node_input.debugName())
          curr_node.runtime_params.append(node_input.debugName())
          self.layer_args_order_map[curr_node.name][node_input.debugName()] = _input_idx
        elif input_node_id:
          node_input_name = \
              node_input.node().scopeName() + PytorchGraph.get_node_id(node_input.node())
          node_input_name = node_input_name \
              .replace('-','n') \
              .replace('\\','n') \
              .replace('/','n') \
              .replace('_','n') \
              .replace('[','n') \
              .replace(']','n')
          self.layer_args_order_map[curr_node.name][node_input_name] = _input_idx
          self._make_connection(node_input_name, node_name)
    try:
      node_variety = defaultdict(list)
      self.collapse_clip_to_relu6()
      self.prune_reachable_nodes()
      self.try_infer_missing_shapes()
      self.patch_reshape_dimensions()
      #self.remove_unrequired_reshapes()
      self.collapse_pad_node_args_to_pad()
      #            self.prune_normalize_layer()
      for key, value in self.layer_map.items():
        node_variety[value.type].append(value.name)
    except Exception as e:
      import traceback
      traceback.print_exc()
    const_users.close()
    logger.info("DONE BUILDING")
    super(PytorchGraph, self).build()
