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
from tensorflow.core.framework.node_def_pb2 import NodeDef
from tensorflow.core.framework import attr_value_pb2
import mmdnn.conversion.tensorflow.tf_graph_utils as netx


class TensorflowGraphNode(GraphNode):

  def __init__(self, layer):
    super(TensorflowGraphNode, self).__init__(layer)
    self.in_nodes = list()
    self.out_nodes = list()
    self._scope = str()

  @property
  def scope(self):
    return self._scope

  @scope.setter
  def scope(self, scope):
    self._scope = scope

  @property
  def name(self):
    return self.layer.name

  @property
  def type(self):
    return self.layer.op

  @property
  def tf_layer(self):
    return self.layer

  def get_attr(self, name, default_value=None):
    if name in self.layer.attr:
      attr = self.layer.attr[name]
      field = attr.WhichOneof('value')
      val = getattr(attr, field) if field else default_value
      if isinstance(val, attr_value_pb2.AttrValue.ListValue):
        return list(val.ListFields()[0][1])
      else:
        return val.decode('utf-8') if isinstance(val, bytes) else val
    else:
      return default_value


class TensorflowGraph(Graph):
  multi_tensor_type = ["Slice", "Split", "Unpack"]

  def __init__(self, model, preout=[], postin=[]):
    pass
    # sanity check.
    super(TensorflowGraph, self).__init__(model)
    self.model = model
    self.preout = preout
    self.postin = postin

  def build(self):
    # get networkx DiGraph object
    G = netx.nx_graph_object(self.model, "tf")
    while netx.ch_node:
      # pattern matchers and subgraph replacemnet utility
      netx.ch_node = False
      G = netx.tf_subGraph_match_replace_L2Norm(G, self.model)
      G = netx.tf_subGraph_match_replace_prelu(G, self.model)
      G = netx.tf_subGraph_match_replace_lrelu(G, self.model)
      G = netx.tf_subGraph_match_replace_atrous_conv2d(G, self.model)
      G = netx.tf_subGraph_match_replace_SplitSoftmax(G, self.model)
      G = netx.tf_subGraph_match_replace_Relu6(G, self.model)
      G = netx.tf_subGraph_match_replace_Relu6_1(G, self.model)
      G = netx.tf_subGraph_match_replace_BatchnormTG(G, self.model)
      G = netx.tf_subGraph_match_replace_BatchnormScaledTG(G, self.model)
      G = netx.tf_subGraph_match_replace_Batchnorm(G, self.model)
      G = netx.tf_subGraph_match_replace_BatchnormScaled(G, self.model)
      G = netx.tf_subGraph_match_replace_BiFPN(G, self.model)
      G = netx.tf_subGraph_match_replace_BiFPNinterim(G, self.model)
      G = netx.tf_subgraph_match_replace_Mish(G, self.model)
      G = netx.tf_subgraph_match_replace_Yolov4_IR1(G, self.model)
      G = netx.tf_subgraph_match_replace_UpsampleFromConcat(G, self.model)
      G = netx.tf_subgraph_match_replace_Swish(G, self.model)
      #G = netx.tf_subGraph_match_replace_Relu6Scaled(G, self.model)
      G = netx.tf_subGraph_match_replace_HardSwish(G, self.model)
      G = netx.tf_subGraph_match_replace_AddV2(G, self.model)
      G = netx.tf_subGraph_match_replace_ConstMul(G, self.model)
      G = netx.tf_subGraph_match_replace_Upsample3D(G, self.model)
      G = netx.tf_subGraph_match_replace_Upsample_HW(G, self.model)
      G = netx.tf_subGraph_match_replace_Deconv3D(G, self.model)
      # dump the transformed graphdef for debugging -- figure out debug mode .
      #netx.dump_graphdef(G)
    nx_sorted = set(netx.topological_sort(G))
    for n_name in self.model.node:
      n_name = n_name.name
      if n_name not in nx_sorted:
        continue
      G_temp = G.nodes[n_name]
      layer = G_temp.get('data', None)
      self.layer_map[layer.name] = TensorflowGraphNode(layer)
      self.layer_name_map[layer.name] = layer.name
      for pred in layer.input:
        if pred not in self.layer_map:
          if not pred.split(':')[0] in self.layer_map:  #test
            new_node = NodeDef()
            new_node.name = pred
            new_node.op = "NoOp"
            self.layer_map[pred] = TensorflowGraphNode(new_node)
            self.layer_name_map[pred] = pred
        self.tf_make_connection(pred, layer.name)
    super(TensorflowGraph, self).build()

  def tf_make_connection(self, src, dst):
    if ':' not in src and self.get_node(src).type in self.multi_tensor_type:
      src += ':0'
    self._make_connection(src, dst)
    src_node = self.get_node(src.split(':')[0])
    dst_node = self.get_node(dst.split(':')[0])
    if not src_node in self.layer_map[dst.split(':')[0]].in_nodes:
      self.layer_map[dst.split(':')[0]].in_nodes.append(src_node)
    if not dst_node in self.layer_map[src.split(':')[0]].out_nodes:
      self.layer_map[src.split(':')[0]].out_nodes.append(dst_node)
