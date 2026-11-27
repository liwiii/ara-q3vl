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
from networkx.algorithms import isomorphism
import sys
import os
import itertools
import re
from tensorflow.python.framework import tensor_util
import tensorflow as tf
import copy
sys.path.append(os.getcwd() + '/' + os.path.dirname(__file__) + '/../../caffe/python')
sys.path.append(os.getcwd() + '/' + os.path.dirname(__file__) + '/../')
from dv_logging import get_logger
logger = get_logger(__name__, 'DVTFUTIL')
constant_tensor = ['Const', 'Constant', 'Identity', 'Identity Constant']
# Add similar builds for other networks.
ch_node = True


# Takes in a model class from TF and creates an equivalent networkx Graph Object.
# rerturns a DiGraph Networkx Object.
def tf_make_Graph(model):
  G = nx.DiGraph()
  edge_list = list()
  for i, layer in enumerate(model.node):
    G.add_node(layer.name, data=layer, type=layer.op)
    for idx, pred in enumerate(layer.input):
      # if str(layer.input[idx].split(':')[0]) == layer.name :
      edge_list.append(tuple((layer.input[idx].split(':')[0], layer.name)))
  G.add_edges_from(edge_list)
  return G


# Ig this Works for all networks ??
def real_variable_name(real_name):
  return real_name.replace('/', '_').replace('-', '_').replace('[', '_').replace(']', '_')


# Can use something like this
def nx_graph_object(model, model_type="tf"):
  G = None
  if model_type == "tf":
    G = tf_make_Graph(model)
  if G == None:
    print("....Yet to implement for %s", model_type)
  return G


def topological_sort(G):
  return list(nx.lexicographical_topological_sort(G))


# returns a DiGraph given a path to a dot file
def read_dot(path):
  try:
    G = nx.DiGraph(nx.drawing.nx_pydot.read_dot(path))
  except Exception as e:
    print("Error : ", s)
    sys.exit()
  return G


# Display Graph as MathPlotlib diag.
# Use only to visulalize VERY SMALL Networks.
def display_graph(G):
  import matplotlib.pyplot as plt
  pos = nx.spring_layout(G)
  nx.draw_networkx_nodes(G, pos, node_size=500)
  nx.draw_networkx_labels(G, pos)
  nx.draw_networkx_edges(G, pos, arrows=True)
  plt.show()


def tf_create_dummy_node(name, _type="NoOp"):
  from tensorflow.core.framework.node_def_pb2 import NodeDef
  new_node = NodeDef()
  new_node.name = name
  new_node.op = _type
  return new_node


def get_predecessor(G, node):
  return list(G.predecessors(node))


def get_successors(G, node):
  return list(G.successors(node))


def remove_subgraph(G, remove_list):
  G.remove_nodes_from(remove_list)
  return G


def reverse_dict(dct):
  return dict((v, k) for k, v in dct.items())


def replace_node_with(nodes, node_name, new_node):
  """clears the data node with .name == node_name in nodes
    and copies all attributes to it from new_node
    """
  for idx, node in enumerate(nodes):
    if node.name == node_name:
      nodes[idx].Clear()
      nodes[idx].MergeFrom(new_node)


def write_dot_file(G):
  from networkx.drawing.nx_pydot import write_dot
  write_dot(G, "grid.dot")


def remove_ancestors(G, node_list=[]):
  remove_list = set()
  for node in node_list:
    try:
      G_t = G.nodes[node]
      G_t['data'].input[:] = []
    except:
      print("node not present in the graph. node error trace :", node)
    succ = nx.algorithms.dag.ancestors(G, source=node)
    remove_list.update(succ)
  rlist = list(remove_list)
  rlist.append(node_list[0])
  subG = nx.DiGraph(create_subG_copy(G, rlist))
  print("SUbG : ", len(rlist))
  G = remove_subgraph(G, rlist)
  return G, subG


def remove_descendants(G, node_list=[]):
  remove_list = set()
  for node in node_list:
    try:
      G_t = G.nodes[node]
    except:
      print("node not present in the graph. node error trace :", node)
    succ = nx.algorithms.dag.descendants(G, source=node)
    remove_list.update(succ)
  rlist = list(remove_list)
  rlist.extend(node_list)
  subG = create_subG_copy(G, rlist)
  G = remove_subgraph(G, rlist)
  return G, subG


def create_subG_copy(G, node_list=[]):
  return G.subgraph(node_list).copy()


def dump_graphdef(G):
  from tensorflow.python.platform import gfile
  gdef = tf.GraphDef()
  top_sort = topological_sort(G)
  for node in top_sort:
    data = G.nodes[node]['data']
    gdef.node.append(data)
  with gfile.GFile("nnconvert_netx.pb", "wb") as f:
    f.write(gdef.SerializeToString())


class BaseMatcher:

  def __init__(self):
    pass

  def subgraph_to_match(self):
    pass

  def input_nodes(self):
    pass

  def output_node(self):
    pass

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove):
    return matched_subgraph

  def node_comparator_by_type(self, n1, n2):
    n1t = n1.get('type', True)
    n2t = n2.get('type', False)
    # Const* vs Constant
    if not n1 or not n2:
      return False
    x = re.match(n2t + "*", n1t)
    if x == None:
      return False
    return True

  def replace_subgraph_instances(self, source_graph, model):
    matches = isomorphism.DiGraphMatcher(source_graph, self.subgraph_to_match(), node_match=self.node_comparator_by_type)

    def _get_input_for_nodes(node_names):
      """for the given nodes, get the unique
            set of inputs to these nodes
            """
      names = set()
      for name in node_names:
        names.update(source_graph.nodes[name]['data'].input)
      return names

    matched_nodes = set()
    for match in list(matches.subgraph_isomorphisms_iter()):
      match = reverse_dict(match)
      iso_nodes = set(list(match.values()))
      if matched_nodes.intersection(iso_nodes) == iso_nodes:
        continue
      matched_nodes.update(iso_nodes)
      nodes_to_remove = list(match.values())
      inputs_to_collapsed_node =\
              list(_get_input_for_nodes(map(match.get, self.input_nodes())))
      name_of_node = match[self.output_node()]
      output_edges = [i for i in source_graph.out_edges if i[0] == name_of_node]
      inputs_to_collapsed_node = [i for i in inputs_to_collapsed_node if i not in nodes_to_remove]
      replacement_node = self.mk_replacement_node(source_graph, match, model, inputs_to_collapsed_node, nodes_to_remove)
      # replacement_node == None ==> some extra condition in mk_replacement_node is not satisfied.
      # * don't replace. !!
      if replacement_node == None:
        continue
      # copy over the shape data
      # TODO: refactor into a function that copies essential
      # attributes
      replacement_node.attr['shape'].MergeFrom(source_graph.nodes[name_of_node]['data'].attr['shape'])
      replacement_node.attr['_output_shapes'].CopyFrom(source_graph.nodes[name_of_node]['data'].attr['_output_shapes'])
      replace_node_with(model.node, replacement_node.name, replacement_node)
      # only delete the old nodes from the graphs once the new entry is
      # made as the new graph node _might_ use information from the nodes
      # that are going to be deleted from the graph.
      source_graph = remove_subgraph(source_graph, nodes_to_remove)
      source_graph.add_node(replacement_node.name, data=replacement_node, type=replacement_node.op)
      new_edges =\
              [(input_node, name_of_node) for input_node in inputs_to_collapsed_node]
      source_graph.add_edges_from(new_edges)
      source_graph.add_edges_from(output_edges)
      logger.info("Node successfully replaced. Replaced node: %s, Type: %s", replacement_node.name, replacement_node.op)
      global ch_node
      ch_node = True
    return source_graph


class PReLUMatcher(BaseMatcher):

  def __init__(self):
    pass

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    neg1 = tf_create_dummy_node("tf/Neg1", "Neg")
    relu1 = tf_create_dummy_node("tf/Relu1", "Relu")
    relu2 = tf_create_dummy_node("tf/Relu2", "Relu")
    mul = tf_create_dummy_node("tf/Mul", "Mul")
    neg2 = tf_create_dummy_node("tf/Neg2", "Neg")
    addv2 = tf_create_dummy_node("tf/AddV2", "AddV2")
    edge_list.append((neg1.name, relu1.name))
    edge_list.append((relu1.name, mul.name))
    edge_list.append((neg2.name, mul.name))
    edge_list.append((relu2.name, addv2.name))
    edge_list.append((mul.name, addv2.name))
    for node in [neg1, relu1, relu2, mul, neg2, addv2]:
      G.add_node(node.name, data=node, type=node.op)
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['tf/Neg1']

  def output_node(self):
    return "tf/AddV2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    node_def = tf_create_dummy_node(matched_subgraph[self.output_node()], 'PRelu')
    node_def.input[:] = inputs_to_collapsed
    return node_def


class AtrousConv2DMatcher(BaseMatcher):

  def __init__(self):
    pass

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    self.spaceToBatch = tf_create_dummy_node("tf/SpaceToBatch", "SpaceToBatchND")
    self.conv2D = tf_create_dummy_node("tf/Conv2D", "DepthwiseConv2dNative")
    self.batchToSpace = tf_create_dummy_node("tf/BatchToSpace", "BatchToSpaceND")
    edge_list.append(("tf/SpaceToBatch", "tf/Conv2D"))
    edge_list.append(("tf/Conv2D", "tf/BatchToSpace"))
    G.add_node(self.spaceToBatch.name, data=self.spaceToBatch, type="SpaceToBatchND")
    G.add_node(self.conv2D.name, data=self.conv2D, type="DepthwiseConv2dNative")
    G.add_node(self.batchToSpace.name, data=self.batchToSpace, type="BatchToSpaceND")
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return [self.spaceToBatch.name]

  def output_node(self):
    return self.batchToSpace.name

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    blocking_node = source_graph.nodes[matched_subgraph[self.spaceToBatch.name]]['data'].input[1]
    blocking_nodedef = source_graph.nodes[blocking_node]
    paddings_node = source_graph.nodes[matched_subgraph[self.spaceToBatch.name]]['data'].input[2]
    padding_nodedef = source_graph.nodes[paddings_node]
    croppings_node = source_graph.nodes[matched_subgraph[self.batchToSpace.name]]['data'].input[2]
    cropping_nodedef = source_graph.nodes[croppings_node]
    actual_out_node = source_graph.nodes[matched_subgraph[self.batchToSpace.name]]['data']
    output_shape = actual_out_node.attr['_output_shapes'].list.shape[0]
    # extra checks to ensure correct matchings.
    node_def = copy.deepcopy(source_graph.nodes[matched_subgraph[self.conv2D.name]]['data'])
    logger.info("conv2d layer def before %s", node_def)
    # only take the original data input; keep the weights as it is
    logger.debug("inputs to be collapsed %s", inputs_to_collapsed)
    logger.debug("changing input0 from %s to %s", node_def.input[0], inputs_to_collapsed[0])
    found = False
    for in1 in inputs_to_collapsed:
      if 'paddings' in in1 or 'block_shape' in in1:
        continue
      else:
        node_def.input[0] = in1
        found = True
        break
    if not found:
      logger.error("input mapping is not valid")
      raise RuntimeError
    #node_def.input[0] = inputs_to_collapsed[0]
    ## hacking to get dilation 2; still lot depends on padding and crop of BTS and STB layer params
    try:
      dilation = tensor_util.MakeNdarray(blocking_nodedef['data'].attr['value'].tensor)
    except:
      dilation = [2, 2]
      logger.error("Unable to get padding value form STB. Using default value of 2")
    node_def.attr['dilations'].list.i[1] = dilation[0]
    node_def.attr['dilations'].list.i[2] = dilation[1]
    node_def.attr['padding'].s = b"SAME"
    node_def.name = matched_subgraph[self.output_node()]
    logger.info("conv2d layer def after %s", node_def)
    return node_def


class LeakyReluMatcher(BaseMatcher):

  def __init__(self):
    pass

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    mul1 = tf_create_dummy_node("tf/Mul", "Mul")
    max1 = tf_create_dummy_node("tf/Max", "Maximum")
    edge_list.append(("tf/Mul", "tf/Max"))
    G.add_node(mul1.name, data=mul1, type="Mul")
    G.add_node(max1.name, data=max1, type="Maximum")
    node = tf_create_dummy_node("tf/Const", "Const")
    G.add_node(node.name, data=node, type='Const')
    edge_list.append(("tf/Const", "tf/Mul"))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ["tf/Mul"]

  def output_node(self):
    return "tf/Max"

  # def is_matched_subgraph_valid(self, source_graph, match):
  #     inputs_to_max = source_graph.nodes[match['tf/Max']]['data'].input
  #     inputs_to_mul = source_graph.nodes[match['tf/Mul']]['data'].input
  #     # ensure that both of these share the same input parent
  #     return len(set(inputs_to_max).intersection(set(inputs_to_mul))) > 0
  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    # extra checks to ensure correct matchings.
    inputs_to_max = source_graph.nodes[matched_subgraph['tf/Max']]['data'].input
    inputs_to_mul = source_graph.nodes[matched_subgraph['tf/Mul']]['data'].input
    if len(set(inputs_to_max).intersection(set(inputs_to_mul))) == 0:
      return None
    node_def = tf_create_dummy_node(matched_subgraph[self.output_node()], 'LeakyRelu')
    node_def.input[:] = inputs_to_collapsed
    attr_alpha = source_graph.nodes[matched_subgraph['tf/Const']]['data']
    attr_alpha = attr_alpha.attr['value'].tensor.float_val[0]
    node_def.attr['alpha'].tensor.float_val.append(attr_alpha)
    return node_def


class Upsample3DMatcher(BaseMatcher):
  #Matches to ResizeNearestNeighbor. scale_d, scale_h, scale_w are always integers.
  def __init__(self):
    pass

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    node = tf_create_dummy_node("tf/ExpandDims_1", "ExpandDims")
    G.add_node(node.name, data=node, type="ExpandDims")
    node = tf_create_dummy_node("tf/Tile_1", "Tile")
    G.add_node(node.name, data=node, type="Tile")
    node = tf_create_dummy_node("tf/Shape_1", "Shape")
    G.add_node(node.name, data=node, type="Shape")
    node = tf_create_dummy_node("tf/Mul_1", "Mul")
    G.add_node(node.name, data=node, type="Mul")
    node = tf_create_dummy_node("tf/Reshape_1", "Reshape")
    G.add_node(node.name, data=node, type="Reshape")
    node = tf_create_dummy_node("tf/ExpandDims_2", "ExpandDims")
    G.add_node(node.name, data=node, type="ExpandDims")
    node = tf_create_dummy_node("tf/Tile_2", "Tile")
    G.add_node(node.name, data=node, type="Tile")
    node = tf_create_dummy_node("tf/Shape_2", "Shape")
    G.add_node(node.name, data=node, type="Shape")
    node = tf_create_dummy_node("tf/Mul_2", "Mul")
    G.add_node(node.name, data=node, type="Mul")
    node = tf_create_dummy_node("tf/Reshape_2", "Reshape")
    G.add_node(node.name, data=node, type="Reshape")
    node = tf_create_dummy_node("tf/ExpandDims_3", "ExpandDims")
    G.add_node(node.name, data=node, type="ExpandDims")
    node = tf_create_dummy_node("tf/Tile_3", "Tile")
    G.add_node(node.name, data=node, type="Tile")
    node = tf_create_dummy_node("tf/Shape_3", "Shape")
    G.add_node(node.name, data=node, type="Shape")
    node = tf_create_dummy_node("tf/Mul_3", "Mul")
    G.add_node(node.name, data=node, type="Mul")
    node = tf_create_dummy_node("tf/Reshape_3", "Reshape")
    G.add_node(node.name, data=node, type="Reshape")
    edge_list.append(("tf/ExpandDims_1", "tf/Tile_1"))
    edge_list.append(("tf/Tile_1", "tf/Reshape_1"))
    edge_list.append(("tf/Shape_1", "tf/Mul_1"))
    edge_list.append(("tf/Mul_1", "tf/Reshape_1"))
    edge_list.append(("tf/Reshape_1", "tf/ExpandDims_2"))
    edge_list.append(("tf/Reshape_1", "tf/Shape_2"))
    edge_list.append(("tf/ExpandDims_2", "tf/Tile_2"))
    edge_list.append(("tf/Tile_2", "tf/Reshape_2"))
    edge_list.append(("tf/Shape_2", "tf/Mul_2"))
    edge_list.append(("tf/Mul_2", "tf/Reshape_2"))
    edge_list.append(("tf/Reshape_2", "tf/ExpandDims_3"))
    edge_list.append(("tf/Reshape_2", "tf/Shape_3"))
    edge_list.append(("tf/ExpandDims_3", "tf/Tile_3"))
    edge_list.append(("tf/Tile_3", "tf/Reshape_3"))
    edge_list.append(("tf/Shape_3", "tf/Mul_3"))
    edge_list.append(("tf/Mul_3", "tf/Reshape_3"))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ["tf/Shape_1"]

  def output_node(self):
    return "tf/Reshape_3"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    output_node_name = matched_subgraph['tf/Reshape_3']
    input_node_name = matched_subgraph['tf/Shape_1']
    node_def = tf_create_dummy_node(output_node_name, 'ResizeNearestNeighbor')
    output_node = source_graph.nodes[output_node_name]['data']
    node_def.attr['T'].CopyFrom(output_node.attr['T'])
    node_def.input.MergeFrom(source_graph.nodes[input_node_name]['data'].input)
    return node_def


class UpsampleHWMatcher(BaseMatcher):
  #Matches to ResizeNearestNeighbor. scale_d, scale_h, scale_w are always integers.
  def __init__(self):
    pass

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    node = tf_create_dummy_node("tf/ExpandDims_1", "ExpandDims")
    G.add_node(node.name, data=node, type="ExpandDims")
    node = tf_create_dummy_node("tf/Identity_1", "Identity")
    G.add_node(node.name, data=node, type="Identity")
    node = tf_create_dummy_node("tf/Shape_1", "Shape")
    G.add_node(node.name, data=node, type="Shape")
    node = tf_create_dummy_node("tf/Identity_2", "Identity")
    G.add_node(node.name, data=node, type="Identity")
    node = tf_create_dummy_node("tf/Reshape_1", "Reshape")
    G.add_node(node.name, data=node, type="Reshape")
    node = tf_create_dummy_node("tf/ExpandDims_2", "ExpandDims")
    G.add_node(node.name, data=node, type="ExpandDims")
    node = tf_create_dummy_node("tf/Tile_2", "Tile")
    G.add_node(node.name, data=node, type="Tile")
    node = tf_create_dummy_node("tf/Shape_2", "Shape")
    G.add_node(node.name, data=node, type="Shape")
    node = tf_create_dummy_node("tf/Mul_2", "Mul")
    G.add_node(node.name, data=node, type="Mul")
    node = tf_create_dummy_node("tf/Reshape_2", "Reshape")
    G.add_node(node.name, data=node, type="Reshape")
    node = tf_create_dummy_node("tf/ExpandDims_3", "ExpandDims")
    G.add_node(node.name, data=node, type="ExpandDims")
    node = tf_create_dummy_node("tf/Tile_3", "Tile")
    G.add_node(node.name, data=node, type="Tile")
    node = tf_create_dummy_node("tf/Shape_3", "Shape")
    G.add_node(node.name, data=node, type="Shape")
    node = tf_create_dummy_node("tf/Mul_3", "Mul")
    G.add_node(node.name, data=node, type="Mul")
    node = tf_create_dummy_node("tf/Reshape_3", "Reshape")
    G.add_node(node.name, data=node, type="Reshape")
    edge_list.append(("tf/ExpandDims_1", "tf/Identity_1"))
    edge_list.append(("tf/Identity_1", "tf/Reshape_1"))
    edge_list.append(("tf/Shape_1", "tf/Identity_2"))
    edge_list.append(("tf/Identity_2", "tf/Reshape_1"))
    edge_list.append(("tf/Reshape_1", "tf/ExpandDims_2"))
    edge_list.append(("tf/Reshape_1", "tf/Shape_2"))
    edge_list.append(("tf/ExpandDims_2", "tf/Tile_2"))
    edge_list.append(("tf/Tile_2", "tf/Reshape_2"))
    edge_list.append(("tf/Shape_2", "tf/Mul_2"))
    edge_list.append(("tf/Mul_2", "tf/Reshape_2"))
    edge_list.append(("tf/Reshape_2", "tf/ExpandDims_3"))
    edge_list.append(("tf/Reshape_2", "tf/Shape_3"))
    edge_list.append(("tf/ExpandDims_3", "tf/Tile_3"))
    edge_list.append(("tf/Tile_3", "tf/Reshape_3"))
    edge_list.append(("tf/Shape_3", "tf/Mul_3"))
    edge_list.append(("tf/Mul_3", "tf/Reshape_3"))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ["tf/Shape_1"]

  def output_node(self):
    return "tf/Reshape_3"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    output_node_name = matched_subgraph['tf/Reshape_3']
    input_node_name = matched_subgraph['tf/Shape_1']
    node_def = tf_create_dummy_node(output_node_name, 'ResizeNearestNeighbor')
    output_node = source_graph.nodes[output_node_name]['data']
    node_def.attr['T'].CopyFrom(output_node.attr['T'])
    node_def.input.MergeFrom(source_graph.nodes[input_node_name]['data'].input)
    return node_def


class Deconv3DMatcher(BaseMatcher):

  def __init__(self):
    pass

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    node = tf_create_dummy_node("tf/Shape", "Shape")
    G.add_node(node.name, data=node, type="Shape")
    node = tf_create_dummy_node("tf/StridedSlice_1", "StridedSlice")
    G.add_node(node.name, data=node, type="StridedSlice")
    node = tf_create_dummy_node("tf/StridedSlice_2", "StridedSlice")
    G.add_node(node.name, data=node, type="StridedSlice")
    node = tf_create_dummy_node("tf/StridedSlice_3", "StridedSlice")
    G.add_node(node.name, data=node, type="StridedSlice")
    node = tf_create_dummy_node("tf/StridedSlice_4", "StridedSlice")
    G.add_node(node.name, data=node, type="StridedSlice")
    node = tf_create_dummy_node("tf/Identity", "Identity")
    G.add_node(node.name, data=node, type="Identity")
    node = tf_create_dummy_node("tf/Mul_1", "Mul")
    G.add_node(node.name, data=node, type="Mul")
    node = tf_create_dummy_node("tf/Mul_2", "Mul")
    G.add_node(node.name, data=node, type="Mul")
    node = tf_create_dummy_node("tf/Pack", "Pack")
    G.add_node(node.name, data=node, type="Pack")
    node = tf_create_dummy_node("tf/Conv3DBackpropInputV2", "Conv3DBackpropInputV2")
    G.add_node(node.name, data=node, type="Conv3DBackpropInputV2")
    edge_list.append(("tf/Shape", "tf/StridedSlice_1"))
    edge_list.append(("tf/Shape", "tf/StridedSlice_2"))
    edge_list.append(("tf/Shape", "tf/StridedSlice_3"))
    edge_list.append(("tf/Shape", "tf/StridedSlice_4"))
    edge_list.append(("tf/StridedSlice_1", "tf/Pack"))
    edge_list.append(("tf/StridedSlice_2", "tf/Identity"))
    edge_list.append(("tf/StridedSlice_3", "tf/Mul_1"))
    edge_list.append(("tf/StridedSlice_4", "tf/Mul_2"))
    edge_list.append(("tf/Identity", "tf/Pack"))
    edge_list.append(("tf/Mul_1", "tf/Pack"))
    edge_list.append(("tf/Mul_2", "tf/Pack"))
    edge_list.append(("tf/Pack", "tf/Conv3DBackpropInputV2"))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ["tf/Shape"]

  def output_node(self):
    return "tf/Conv3DBackpropInputV2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    output_node_name = matched_subgraph[self.output_node()]
    output_node = source_graph.nodes[output_node_name]['data']
    output_node_shape = [_size.size for _size in output_node.attr['_output_shapes'].list.shape[0].dim]
    subgraph_input_name = matched_subgraph[self.input_nodes()[0]]
    subgraph_input_node = source_graph.nodes[subgraph_input_name]['data']
    input_name = subgraph_input_node.input[0]
    input_node = source_graph.nodes[input_name]['data']
    input_node_shape = [_size.size for _size in input_node.attr['_output_shapes'].list.shape[0].dim]
    scale_d = output_node_shape[1] / input_node_shape[1]
    scale_h = output_node_shape[2] / input_node_shape[2]
    scale_w = output_node_shape[3] / input_node_shape[3]
    filter_id = [_inp for _inp in output_node.input if 'Read' in _inp]
    filter_node = source_graph.nodes[filter_id[0]]['data']
    kernel_shape = [_size.size for _size in filter_node.attr['value'].tensor.tensor_shape.dim][:3]
    if (kernel_shape == [1, 1, 1] and int(tensor_util.MakeNdarray(filter_node.attr['value'].tensor)[0][0][0][0][0]) == 1 and scale_d.is_integer() and scale_h.is_integer() and
        scale_w.is_integer()):
      node_def = tf_create_dummy_node(output_node_name, 'ResizeNearestNeighbor')
      node_def.attr['T'].CopyFrom(output_node.attr['T'])
      node_def.input.MergeFrom(source_graph.nodes[matched_subgraph[self.input_nodes()[0]]]['data'].input)
      node_def.attr['zero_mode'].list.i.extend([2, 2, 2])
    else:
      node_def = tf_create_dummy_node(output_node_name, 'Conv3DBackpropInputV2')
      node_def.Clear()
      node_def.MergeFrom(output_node)
      node_def.input.remove(matched_subgraph['tf/Pack'])
      node_def.op = 'Conv2DBackpropInput'
      if subgraph_input_node.input[0] == node_def.input[1]:
        #moving weight node to pos 1
        node_def.input.insert(0, node_def.input.pop())
    return node_def


class L2NormMatcher(BaseMatcher):

  def __init__(self):
    pass

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    node = tf_create_dummy_node("tf/Square", "Square")
    G.add_node(node.name, data=node, type="Square")
    node = tf_create_dummy_node("tf/Sum", "Sum")
    G.add_node(node.name, data=node, type="Sum")
    node = tf_create_dummy_node("tf/Max", "Maximum")
    G.add_node(node.name, data=node, type="Maximum")
    node = tf_create_dummy_node("tf/Rsqrt", "Rsqrt")
    G.add_node(node.name, data=node, type="Rsqrt")
    node = tf_create_dummy_node("tf/Mul")
    G.add_node(node.name, data=node, type="Mul")
    node = tf_create_dummy_node("tf/Const", "Const")
    G.add_node(node.name, data=node, type="Const")
    node = tf_create_dummy_node("tf/Const1", "Const")
    G.add_node(node.name, data=node, type='Const')
    edge_list.append(("tf/Square", "tf/Sum"))
    # const node that stores the reduction indices parameter
    edge_list.append(("tf/Const1", "tf/Sum"))
    # const node fed to act as a min value clamp
    edge_list.append(("tf/Const", "tf/Max"))
    edge_list.append(("tf/Sum", "tf/Max"))
    edge_list.append(("tf/Max", "tf/Rsqrt"))
    edge_list.append(("tf/Rsqrt", "tf/Mul"))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ["tf/Square"]

  def output_node(self):
    return "tf/Mul"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    norm_eps = source_graph.nodes[matched_subgraph['tf/Const']]['data']
    reduction_axis = source_graph.nodes[matched_subgraph['tf/Const1']]['data']
    # extract the values
    norm_eps = norm_eps.attr['value'].tensor.float_val[0]
    reduction_axis = reduction_axis.attr['value'].tensor.int_val[0]
    # if the reduction axis is - 1 or
    # if the number of input_dims - 1 is the
    # treduction axis, then we are doing a spatial normalize
    spatial_normalize = True
    if reduction_axis == -1:
      spatial_normalize = False
    else:
      l2norm_input = source_graph.nodes[matched_subgraph[self.input_nodes()[0]]]['data'].input[0]
      input_shape = source_graph.nodes[l2norm_input]['data'].attr['_output_shapes']
      input_dims = len(input_shape.list.shape[0].dim)
      if input_dims - 1 == reduction_axis:
        spatial_normalize = False
    node_def = tf_create_dummy_node(matched_subgraph[self.output_node()], 'L2Norm')
    node_def.input[:] = inputs_to_collapsed
    node_def.attr['eps'].f = norm_eps
    node_def.attr['reduction_axis'].i = reduction_axis
    node_def.attr['across_spatial'].b = spatial_normalize
    return node_def


class SplitSoftmaxMatcher(BaseMatcher):

  def __init__(self):
    pass

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    node = tf_create_dummy_node("tf_Max", "Max")
    G.add_node(node.name, data=node, type="Max")
    node = tf_create_dummy_node("tf_Sub", "Sub")
    G.add_node(node.name, data=node, type="Sub")
    node = tf_create_dummy_node("tf_Exp", "Exp")
    G.add_node(node.name, data=node, type="Exp")
    node = tf_create_dummy_node("tf_Sum", "Sum")
    G.add_node(node.name, data=node, type="Sum")
    node = tf_create_dummy_node("tf_RealDiv", "RealDiv")
    G.add_node(node.name, data=node, type="RealDiv")
    edge_list.append(tuple(("tf_Max", "tf_Sub")))
    edge_list.append(tuple(("tf_Sub", "tf_Exp")))
    edge_list.append(tuple(("tf_Exp", "tf_Sum")))
    edge_list.append(tuple(("tf_Exp", "tf_RealDiv")))
    edge_list.append(tuple(("tf_Sum", "tf_RealDiv")))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ["tf_Max"]

  def output_node(self):
    return "tf_RealDiv"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    output_node_name = matched_subgraph[self.output_node()]
    node_def = tf_create_dummy_node(output_node_name, 'Softmax')
    node_def.input[:] = inputs_to_collapsed
    return node_def


class Relu6Matcher_1(BaseMatcher):

  def __init__(self):
    pass

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    relu = tf_create_dummy_node("tf/Relu", "Relu")
    minimum = tf_create_dummy_node("tf/Min", "Minimum")
    const1 = tf_create_dummy_node("tf/Const1", "Const")
    edge_list.append((relu.name, minimum.name))
    edge_list.append((const1.name, minimum.name))
    for node in [relu, minimum, const1]:
      G.add_node(node.name, data=node, type=node.op)
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['tf/Relu']

  def output_node(self):
    return "tf/Min"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    # extract min input value.
    min_input = source_graph.nodes[matched_subgraph['tf/Const1']]['data']
    min_input = min_input.attr['value'].tensor.float_val[0]
    if not (min_input == 6):
      logger.error("Min and Max node inputs not consistent with Relu6.")
      return None
    node_def = tf_create_dummy_node(matched_subgraph[self.output_node()], 'Relu6')
    node_def.input[:] = inputs_to_collapsed
    return node_def


class Relu6Matcher(BaseMatcher):

  def __init__(self):
    pass

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    relu = tf_create_dummy_node("tf/Relu", "Relu")
    minimum = tf_create_dummy_node("tf/Min", "Minimum")
    maximum = tf_create_dummy_node("tf/Max", "Maximum")
    const1 = tf_create_dummy_node("tf/Const1", "Const")
    const2 = tf_create_dummy_node("tf/Const2", "Const")
    edge_list.append((relu.name, minimum.name))
    edge_list.append((minimum.name, maximum.name))
    edge_list.append((const1.name, minimum.name))
    edge_list.append((const2.name, maximum.name))
    for node in [relu, minimum, maximum, const1, const2]:
      G.add_node(node.name, data=node, type=node.op)
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['tf/Relu']

  def output_node(self):
    return "tf/Max"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    # extract min and max input values.
    min_input = source_graph.nodes[matched_subgraph['tf/Const1']]['data']
    min_input1 = min_input.attr['value'].tensor.float_val
    if len(min_input1):
      min_input = min_input1[0]
    else:
      min_input = tensor_util.MakeNdarray(min_input.attr['value'].tensor)
    max_input = source_graph.nodes[matched_subgraph['tf/Const2']]['data']
    max_input1 = max_input.attr['value'].tensor.float_val
    if len(max_input1):
      max_input = max_input1[0]
    else:
      max_input = tensor_util.MakeNdarray(max_input.attr['value'].tensor)
    if not (min_input == 6 and max_input == 0):
      logger.error("Min and Max node inputs not consistent with Relu6.")
      return None
    node_def = tf_create_dummy_node(matched_subgraph[self.output_node()], 'Relu6')
    node_def.input[:] = inputs_to_collapsed
    return node_def


class BatchnormMatcher(BaseMatcher):

  def __init__(self):
    pass

  def node_comparator_by_type(self, n1, n2):
    n1t = n1.get('type', True)
    n2t = n2.get('type', False)
    if n2t == 'Constant_':
      # Identity Constant vs Constant ---
      for const in constant_tensor:
        if const in n1t:
          return True
      return False
    # Const* vs Constant
    p = re.compile('.*' + n2t + '.*', re.IGNORECASE)
    x = p.match(n1t)
    return not x == None

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    add1 = tf_create_dummy_node("tf/Add1", "Add")
    rsqrt = tf_create_dummy_node("tf/Rsqrt", "Rsqrt")
    mul1 = tf_create_dummy_node("tf/Mul1", "Mul")
    mul2 = tf_create_dummy_node("tf/Mul2", "Mul")
    sub = tf_create_dummy_node("tf/Sub", "Sub")
    add2 = tf_create_dummy_node("tf/Add2", "Add")
    add1_const1 = tf_create_dummy_node("tf/Const1", "Constant_")
    add1_const2 = tf_create_dummy_node("tf/Const2", "Constant_")
    mul2_const = tf_create_dummy_node("tf/Const3", "Constant_")
    sub_const = tf_create_dummy_node("tf/Const4", "Constant_")
    edge_list.append((add1.name, rsqrt.name))
    edge_list.append((rsqrt.name, mul1.name))
    edge_list.append((rsqrt.name, mul2.name))
    edge_list.append((mul1.name, add2.name))
    edge_list.append((mul2.name, sub.name))
    edge_list.append((sub.name, add2.name))
    edge_list.append((add1_const1.name, add1.name))
    edge_list.append((add1_const2.name, add1.name))
    edge_list.append((sub_const.name, sub.name))
    edge_list.append((mul2_const.name, mul2.name))
    for node in [add1, rsqrt, mul1, mul2, sub, add2, add1_const1, add1_const2, mul2_const, sub_const]:
      G.add_node(node.name, data=node, type=node.op)
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['tf/Mul1']

  def output_node(self):
    return "tf/Add2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    node_def = tf_create_dummy_node(matched_subgraph[self.output_node()], 'BatchNorm')
    eps = source_graph.nodes[matched_subgraph['tf/Const1']]['data']
    if len(eps.attr['value'].tensor.float_val) is not 0:
      eps = eps.attr['value'].tensor.float_val[0]
      var = source_graph.nodes[matched_subgraph['tf/Const2']]['data'].input[0]
      node_def.attr['variance'].MergeFrom(source_graph.nodes[var]['data'].attr['value'])
    else:
      eps = source_graph.nodes[matched_subgraph['tf/Const2']]['data']
      eps = eps.attr['value'].tensor.float_val[0]
      var = source_graph.nodes[matched_subgraph['tf/Const1']]['data'].input[0]
      node_def.attr['variance'].MergeFrom(source_graph.nodes[var]['data'].attr['value'])
    node_def.attr['epsilon'].tensor.float_val.append(eps)
    mean = source_graph.nodes[matched_subgraph['tf/Const3']]['data'].input[0]
    node_def.attr['mean'].MergeFrom(source_graph.nodes[mean]['data'].attr['value'])
    bias = source_graph.nodes[matched_subgraph['tf/Const4']]['data'].input[0]
    node_def.attr['bias'].MergeFrom(source_graph.nodes[bias]['data'].attr['value'])
    node_def.input[:] = inputs_to_collapsed
    return node_def


class BiFPNMatcher(BaseMatcher):

  def __init__(self):
    pass

  def node_comparator_by_type(self, n1, n2):
    n1t = n1.get('type', True)
    n2t = n2.get('type', False)
    if n2t == 'Constant_':
      # Identity Constant vs Constant ---
      for const in constant_tensor:
        if const in n1t:
          return True
      return False
    # Const* vs Constant
    p = re.compile('.*' + n2t + '.*', re.IGNORECASE)
    x = p.match(n1t)
    return not x == None

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    mul1 = tf_create_dummy_node("tf/Mul1", "Mul")
    mul2 = tf_create_dummy_node("tf/Mul2", "Mul")
    mul3 = tf_create_dummy_node("tf/Mul3", "Mul")
    identity1_1 = tf_create_dummy_node("tf/Ident1_1", "Identity")
    identity1_2 = tf_create_dummy_node("tf/Ident1_2", "Identity")
    identity2_1 = tf_create_dummy_node("tf/Ident2_1", "Identity")
    identity2_2 = tf_create_dummy_node("tf/Ident2_2", "Identity")
    identity3_1 = tf_create_dummy_node("tf/Ident3_1", "Identity")
    identity3_2 = tf_create_dummy_node("tf/Ident3_2", "Identity")
    const1 = tf_create_dummy_node("tf/Const1", "Const")
    const2 = tf_create_dummy_node("tf/Const2", "Const")
    const3 = tf_create_dummy_node("tf/Const3", "Const")
    addn_1 = tf_create_dummy_node("tf/Addn1", "AddN")
    addv2_1 = tf_create_dummy_node("tf/Addv2_1", "AddV2")
    addv2_2 = tf_create_dummy_node("tf/Addv2_2", "AddV2")
    addv2_3 = tf_create_dummy_node("tf/Addv2_3", "AddV2")
    realdiv1 = tf_create_dummy_node("tf/Realdiv1", "RealDiv")
    realdiv2 = tf_create_dummy_node("tf/Realdiv2", "RealDiv")
    realdiv3 = tf_create_dummy_node("tf/Realdiv3", "RealDiv")
    addn_2 = tf_create_dummy_node("tf/Addn2", "AddN")
    edge_list.append((const1.name, identity1_1.name))
    edge_list.append((const2.name, identity2_1.name))
    edge_list.append((const1.name, identity1_2.name))
    edge_list.append((const2.name, identity2_2.name))
    edge_list.append((const3.name, identity3_1.name))
    edge_list.append((const3.name, identity3_2.name))
    edge_list.append((identity1_1.name, mul1.name))
    edge_list.append((identity2_1.name, mul2.name))
    edge_list.append((identity3_1.name, mul3.name))
    edge_list.append((identity1_2.name, addn_1.name))
    edge_list.append((identity2_2.name, addn_1.name))
    edge_list.append((identity3_2.name, addn_1.name))
    edge_list.append((addn_1.name, addv2_1.name))
    edge_list.append((addn_1.name, addv2_2.name))
    edge_list.append((addn_1.name, addv2_3.name))
    edge_list.append((addv2_1.name, realdiv1.name))
    edge_list.append((addv2_2.name, realdiv2.name))
    edge_list.append((addv2_3.name, realdiv3.name))
    edge_list.append((mul1.name, realdiv1.name))
    edge_list.append((mul2.name, realdiv2.name))
    edge_list.append((mul3.name, realdiv3.name))
    edge_list.append((realdiv1.name, addn_2.name))
    edge_list.append((realdiv2.name, addn_2.name))
    edge_list.append((realdiv3.name, addn_2.name))
    for node in [
        mul1, mul2, mul3, identity1_1, identity1_2, identity2_1, identity2_2, identity3_1, identity3_2, const1, const2, const3, addn_1, addv2_1, addv2_2, addv2_3, realdiv1,
        realdiv2, realdiv3, addn_2
    ]:
      G.add_node(node.name, data=node, type=node.op)
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['tf/Const1', 'tf/Const2', 'tf/Const3', 'tf/Mul1', 'tf/Mul2', 'tf/Mul3']

  def output_node(self):
    return "tf/Addn2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    node_def = tf_create_dummy_node(matched_subgraph[self.output_node()], 'BiFPN')
    input_1 = source_graph.nodes[matched_subgraph['tf/Mul1']]['data'].input[0]
    input_2 = source_graph.nodes[matched_subgraph['tf/Mul1']]['data'].input[1]
    if source_graph.nodes[input_1]['type'] == 'Identity':
      node_name = input_2
    else:
      node_name = input_1
    node_def.attr[node_name].MergeFrom(source_graph.nodes[matched_subgraph['tf/Const1']]['data'].attr['value'])
    input_1 = source_graph.nodes[matched_subgraph['tf/Mul2']]['data'].input[0]
    input_2 = source_graph.nodes[matched_subgraph['tf/Mul2']]['data'].input[1]
    if source_graph.nodes[input_1]['type'] == 'Identity':
      node_name = input_2
    else:
      node_name = input_1
    node_def.attr[node_name].MergeFrom(source_graph.nodes[matched_subgraph['tf/Const2']]['data'].attr['value'])
    input_1 = source_graph.nodes[matched_subgraph['tf/Mul3']]['data'].input[0]
    input_2 = source_graph.nodes[matched_subgraph['tf/Mul3']]['data'].input[1]
    if source_graph.nodes[input_1]['type'] == 'Identity':
      node_name = input_2
    else:
      node_name = input_1
    node_def.attr[node_name].MergeFrom(source_graph.nodes[matched_subgraph['tf/Const3']]['data'].attr['value'])
    node_def.input[:] = inputs_to_collapsed
    return node_def


class MishMatcher(BaseMatcher):

  def __init__(self):
    pass

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    mul1 = tf_create_dummy_node("tf/Mul1", "Mul")
    less1 = tf_create_dummy_node("tf/Less1", "Less")
    greater1 = tf_create_dummy_node("tf/Greater1", "Greater")
    exp1 = tf_create_dummy_node("tf/Exp1", "Exp")
    zeroslike1 = tf_create_dummy_node("tf/ZerosLike1", "ZerosLike")
    logical_and1 = tf_create_dummy_node("tf/LogicalAnd1", "LogicalAnd")
    add1 = tf_create_dummy_node("tf/Add1", "Add")
    log1 = tf_create_dummy_node("tf/Log1", "Log")
    select1 = tf_create_dummy_node("tf/Select1", "Select")
    less2 = tf_create_dummy_node("tf/Less2", "Less")
    exp2 = tf_create_dummy_node("tf/Exp2", "Exp")
    select2 = tf_create_dummy_node("tf/Select2", "Select")
    tanh1 = tf_create_dummy_node("tf/Tanh1", "Tanh")
    select3 = tf_create_dummy_node("tf/Select3", "Select")
    greater2 = tf_create_dummy_node("tf/Greater2", "Greater")
    less1_const1 = tf_create_dummy_node("tf/Const1", "Const")
    less1_const2 = tf_create_dummy_node("tf/Const2", "Const")
    add1_const3 = tf_create_dummy_node("tf/Const3", "Const")
    greater1_const4 = tf_create_dummy_node("tf/Const4", "Const")
    greater2_const1 = tf_create_dummy_node("tf/Const5", "Const")
    edge_list.append((less1.name, logical_and1.name))
    edge_list.append((greater1.name, logical_and1.name))
    edge_list.append((exp1.name, add1.name))
    edge_list.append((add1.name, log1.name))
    edge_list.append((zeroslike1.name, select1.name))
    edge_list.append((log1.name, select1.name))
    edge_list.append((logical_and1.name, select1.name))
    edge_list.append((select1.name, select2.name))
    edge_list.append((less2.name, select2.name))
    edge_list.append((exp2.name, select2.name))
    edge_list.append((select2.name, select3.name))
    edge_list.append((greater2.name, select3.name))
    edge_list.append((select3.name, tanh1.name))
    edge_list.append((tanh1.name, mul1.name))
    edge_list.append((add1_const3.name, add1.name))
    edge_list.append((less1_const1.name, less1.name))
    edge_list.append((less1_const2.name, less2.name))
    edge_list.append((greater1_const4.name, greater1.name))
    edge_list.append((greater2_const1.name, greater2.name))
    for node in [
        mul1, less1, greater1, exp1, zeroslike1, logical_and1, add1, log1, select1, less2, exp2, select2, tanh1, less1_const1, less1_const2, add1_const3, greater1_const4, select3,
        greater2_const1, greater2
    ]:
      G.add_node(node.name, data=node, type=node.op)
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['tf/Mul1']

  def output_node(self):
    return "tf/Mul1"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    node_def = tf_create_dummy_node(matched_subgraph[self.output_node()], 'Mish')
    node_def.attr['alpha'].tensor.float_val.append(source_graph.nodes[matched_subgraph['tf/Const1']]['data'].attr['value'].tensor.float_val[0])
    node_def.attr['gamma'].tensor.float_val.append(source_graph.nodes[matched_subgraph['tf/Const2']]['data'].attr['value'].tensor.float_val[0])
    node_def.attr['beta'].tensor.float_val.append(source_graph.nodes[matched_subgraph['tf/Const4']]['data'].attr['value'].tensor.float_val[0])
    node_def.attr['delta'].tensor.float_val.append(source_graph.nodes[matched_subgraph['tf/Const5']]['data'].attr['value'].tensor.float_val[0])
    node_def.input[:] = inputs_to_collapsed
    return node_def


class Yolov4_IR1Matcher(BaseMatcher):

  def __init__(self):
    pass

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    shape1 = tf_create_dummy_node("tf/Shape1", "Shape")
    stridedSlice1 = tf_create_dummy_node("tf/StridedSlice1", "StridedSlice")
    stridedSlice2 = tf_create_dummy_node("tf/StridedSlice2", "StridedSlice")
    mul1 = tf_create_dummy_node("tf/Mul1", "Mul")
    mul2 = tf_create_dummy_node("tf/Mul2", "Mul")
    pack1 = tf_create_dummy_node("tf/Pack1", "Pack")
    resize1 = tf_create_dummy_node("tf/Resizenn", "ResizeNearestNeighbor")
    stridedSlice1_const1 = tf_create_dummy_node("tf/Const1", "Const")
    stridedSlice1_const2 = tf_create_dummy_node("tf/Const2", "Const")
    stridedSlice1_const3 = tf_create_dummy_node("tf/Const3", "Const")
    stridedSlice2_const1 = tf_create_dummy_node("tf/Const4", "Const")
    stridedSlice2_const2 = tf_create_dummy_node("tf/Const5", "Const")
    stridedSlice2_const3 = tf_create_dummy_node("tf/Const6", "Const")
    mul1_const1 = tf_create_dummy_node("tf/Const7", "Const")
    mul2_const1 = tf_create_dummy_node("tf/Const8", "Const")
    edge_list.append((shape1.name, stridedSlice1.name))
    edge_list.append((shape1.name, stridedSlice2.name))
    edge_list.append((stridedSlice1.name, mul1.name))
    edge_list.append((stridedSlice2.name, mul2.name))
    edge_list.append((mul1.name, pack1.name))
    edge_list.append((mul2.name, pack1.name))
    edge_list.append((mul1_const1.name, mul1.name))
    edge_list.append((mul2_const1.name, mul2.name))
    edge_list.append((pack1.name, resize1.name))
    edge_list.append((stridedSlice1_const1.name, stridedSlice1.name))
    edge_list.append((stridedSlice1_const2.name, stridedSlice1.name))
    edge_list.append((stridedSlice1_const3.name, stridedSlice1.name))
    edge_list.append((stridedSlice2_const1.name, stridedSlice2.name))
    edge_list.append((stridedSlice2_const2.name, stridedSlice2.name))
    edge_list.append((stridedSlice2_const3.name, stridedSlice2.name))
    for node in [
        shape1, mul1, mul2, stridedSlice1, stridedSlice2, pack1, stridedSlice1_const1, stridedSlice1_const2, stridedSlice1_const3, stridedSlice2_const1, stridedSlice2_const2,
        stridedSlice2_const3, mul1_const1, mul2_const1, resize1
    ]:
      G.add_node(node.name, data=node, type=node.op)
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['tf/Shape1']

  def output_node(self):
    return "tf/Resizenn"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    node_def = tf_create_dummy_node(matched_subgraph[self.output_node()], 'Yolov4_IR1')
    node_def.input[:] = inputs_to_collapsed
    return node_def


class BiFPNinterimMatcher(BaseMatcher):

  def __init__(self):
    pass

  def node_comparator_by_type(self, n1, n2):
    n1t = n1.get('type', True)
    n2t = n2.get('type', False)
    if n2t == 'Constant_':
      # Identity Constant vs Constant ---
      for const in constant_tensor:
        if const in n1t:
          return True
      return False
    # Const* vs Constant
    p = re.compile('.*' + n2t + '.*', re.IGNORECASE)
    x = p.match(n1t)
    return not x == None

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    mul1 = tf_create_dummy_node("tf/Mul1", "Mul")
    mul2 = tf_create_dummy_node("tf/Mul2", "Mul")
    identity1_1 = tf_create_dummy_node("tf/Ident1_1", "Identity")
    identity1_2 = tf_create_dummy_node("tf/Ident1_2", "Identity")
    identity2_1 = tf_create_dummy_node("tf/Ident2_1", "Identity")
    identity2_2 = tf_create_dummy_node("tf/Ident2_2", "Identity")
    const1 = tf_create_dummy_node("tf/Const1", "Const")
    const2 = tf_create_dummy_node("tf/Const2", "Const")
    addn_1 = tf_create_dummy_node("tf/Addn1", "AddN")
    addv2_1 = tf_create_dummy_node("tf/Addv2_1", "AddV2")
    addv2_2 = tf_create_dummy_node("tf/Addv2_2", "AddV2")
    realdiv1 = tf_create_dummy_node("tf/Realdiv1", "RealDiv")
    realdiv2 = tf_create_dummy_node("tf/Realdiv2", "RealDiv")
    addn_2 = tf_create_dummy_node("tf/Addn2", "AddN")
    edge_list.append((const1.name, identity1_1.name))
    edge_list.append((const2.name, identity2_1.name))
    edge_list.append((const1.name, identity1_2.name))
    edge_list.append((const2.name, identity2_2.name))
    edge_list.append((identity1_1.name, mul1.name))
    edge_list.append((identity2_1.name, mul2.name))
    edge_list.append((identity1_2.name, addn_1.name))
    edge_list.append((identity2_2.name, addn_1.name))
    edge_list.append((addn_1.name, addv2_1.name))
    edge_list.append((addn_1.name, addv2_2.name))
    edge_list.append((addv2_1.name, realdiv1.name))
    edge_list.append((addv2_2.name, realdiv2.name))
    edge_list.append((mul1.name, realdiv1.name))
    edge_list.append((mul2.name, realdiv2.name))
    edge_list.append((realdiv1.name, addn_2.name))
    edge_list.append((realdiv2.name, addn_2.name))
    for node in [mul1, mul2, identity1_1, identity1_2, identity2_1, identity2_2, const1, const2, addn_1, addv2_1, addv2_2, realdiv1, realdiv2, addn_2]:
      G.add_node(node.name, data=node, type=node.op)
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['tf/Const1', 'tf/Const2', 'tf/Mul1', 'tf/Mul2']

  def output_node(self):
    return "tf/Addn2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    node_def = tf_create_dummy_node(matched_subgraph[self.output_node()], 'BiFPNinterim')
    input_1 = source_graph.nodes[matched_subgraph['tf/Mul1']]['data'].input[0]
    input_2 = source_graph.nodes[matched_subgraph['tf/Mul1']]['data'].input[1]
    if source_graph.nodes[input_1]['type'] == 'Identity':
      node_name = input_2
    else:
      node_name = input_1
    node_def.attr[node_name].MergeFrom(source_graph.nodes[matched_subgraph['tf/Const1']]['data'].attr['value'])
    input_1 = source_graph.nodes[matched_subgraph['tf/Mul2']]['data'].input[0]
    input_2 = source_graph.nodes[matched_subgraph['tf/Mul2']]['data'].input[1]
    if source_graph.nodes[input_1]['type'] == 'Identity':
      node_name = input_2
    else:
      node_name = input_1
    node_def.attr[node_name].MergeFrom(source_graph.nodes[matched_subgraph['tf/Const2']]['data'].attr['value'])
    node_def.input[:] = inputs_to_collapsed
    return node_def


class BatchnormScaledMatcher(BaseMatcher):

  def __init__(self):
    pass

  def node_comparator_by_type(self, n1, n2):
    n1t = n1.get('type', True)
    n2t = n2.get('type', False)
    if n2t == 'Constant_':
      # Identity Constant vs Constant ---
      for const in constant_tensor:
        if const == n1t:
          return True
      return False
    # Const* vs Constant
    p = re.compile('.*' + n2t + '.*', re.IGNORECASE)
    x = p.match(n1t)
    return not x == None

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    add1 = tf_create_dummy_node("tf/Add1", "Add")
    rsqrt = tf_create_dummy_node("tf/Rsqrt", "Rsqrt")
    mul1 = tf_create_dummy_node("tf/Mul1", "Mul")
    mul2 = tf_create_dummy_node("tf/Mul2", "Mul")
    mul3 = tf_create_dummy_node("tf/Mul3", "Mul")
    sub = tf_create_dummy_node("tf/Sub", "Sub")
    add2 = tf_create_dummy_node("tf/Add2", "Add")
    add1_const1 = tf_create_dummy_node("tf/Const1", "Constant_")
    add1_const2 = tf_create_dummy_node("tf/Const2", "Constant_")
    mul2_const = tf_create_dummy_node("tf/Const3", "Constant_")
    sub_const = tf_create_dummy_node("tf/Const4", "Constant_")
    mul1_const = tf_create_dummy_node("tf/Const5", "Constant_")
    edge_list.append((add1.name, rsqrt.name))
    edge_list.append((rsqrt.name, mul1.name))
    edge_list.append((mul1.name, mul2.name))
    edge_list.append((mul1.name, mul3.name))
    edge_list.append((mul2.name, sub.name))
    edge_list.append((mul3.name, add2.name))
    edge_list.append((sub.name, add2.name))
    edge_list.append((add1_const1.name, add1.name))
    edge_list.append((add1_const2.name, add1.name))
    edge_list.append((mul2_const.name, mul2.name))
    edge_list.append((sub_const.name, sub.name))
    edge_list.append((mul1_const.name, mul1.name))
    for node in [add1, rsqrt, mul1, mul2, sub, add2, mul3, add1_const1, add1_const2, mul2_const, sub_const, mul1_const]:
      G.add_node(node.name, data=node, type=node.op)
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['tf/Mul3']

  def output_node(self):
    return "tf/Add2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    eps = source_graph.nodes[matched_subgraph['tf/Const2']]['data']
    if len(eps.attr['value'].tensor.float_val) is not 0:
      eps = eps.attr['value'].tensor.float_val[0]
      node_def = tf_create_dummy_node(matched_subgraph[self.output_node()], 'BatchNorm')
      node_def.attr['epsilon'].tensor.float_val.append(eps)
      var = source_graph.nodes[matched_subgraph['tf/Const1']]['data'].input[0]
      node_def.attr['variance'].MergeFrom(source_graph.nodes[var]['data'].attr['value'])
      scale = source_graph.nodes[matched_subgraph['tf/Const5']]['data'].input[0]
      node_def.attr['scale'].MergeFrom(source_graph.nodes[scale]['data'].attr['value'])
      mean = source_graph.nodes[matched_subgraph['tf/Const3']]['data'].input[0]
      node_def.attr['mean'].MergeFrom(source_graph.nodes[mean]['data'].attr['value'])
      bias = source_graph.nodes[matched_subgraph['tf/Const4']]['data'].input[0]
      node_def.attr['bias'].MergeFrom(source_graph.nodes[bias]['data'].attr['value'])
    else:
      eps = source_graph.nodes[matched_subgraph['tf/Const1']]['data']
      eps = eps.attr['value'].tensor.float_val[0]
      node_def = tf_create_dummy_node(matched_subgraph[self.output_node()], 'BatchNorm')
      node_def.attr['epsilon'].tensor.float_val.append(eps)
      var = source_graph.nodes[matched_subgraph['tf/Const2']]['data'].input[0]
      node_def.attr['variance'].MergeFrom(source_graph.nodes[var]['data'].attr['value'])
      scale = source_graph.nodes[matched_subgraph['tf/Const5']]['data'].input[0]
      node_def.attr['scale'].MergeFrom(source_graph.nodes[scale]['data'].attr['value'])
      mean = source_graph.nodes[matched_subgraph['tf/Const3']]['data'].input[0]
      node_def.attr['mean'].MergeFrom(source_graph.nodes[mean]['data'].attr['value'])
      bias = source_graph.nodes[matched_subgraph['tf/Const4']]['data'].input[0]
      node_def.attr['bias'].MergeFrom(source_graph.nodes[bias]['data'].attr['value'])
    node_def.input[:] = inputs_to_collapsed
    return node_def


class BatchnormScaledTG(BaseMatcher):

  def __init__(self):
    pass

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    add1 = tf_create_dummy_node("tf/Add1", "Add")
    rsqrt = tf_create_dummy_node("tf/Rsqrt", "Rsqrt")
    mul1 = tf_create_dummy_node("tf/Mul1", "Mul")
    mul2 = tf_create_dummy_node("tf/Mul2", "Mul")
    mul3 = tf_create_dummy_node("tf/Mul3", "Mul")
    sub = tf_create_dummy_node("tf/Sub", "Sub")
    add2 = tf_create_dummy_node("tf/Add2", "Add")
    add1_const1 = tf_create_dummy_node("tf/Const1", "Const")
    add1_const2 = tf_create_dummy_node("tf/Const2", "Const")
    mul2_const = tf_create_dummy_node("tf/Const3", "Const")
    sub_const = tf_create_dummy_node("tf/Const4", "Const")
    mul1_const = tf_create_dummy_node("tf/Const5", "Const")
    edge_list.append((add1.name, rsqrt.name))
    edge_list.append((rsqrt.name, mul1.name))
    edge_list.append((mul1.name, mul2.name))
    edge_list.append((mul1.name, mul3.name))
    edge_list.append((mul2.name, sub.name))
    edge_list.append((mul3.name, add2.name))
    edge_list.append((sub.name, add2.name))
    edge_list.append((add1_const1.name, add1.name))
    edge_list.append((add1_const2.name, add1.name))
    edge_list.append((mul2_const.name, mul2.name))
    edge_list.append((sub_const.name, sub.name))
    edge_list.append((mul1_const.name, mul1.name))
    for node in [add1, rsqrt, mul1, mul2, sub, add2, mul3, add1_const1, add1_const2, mul2_const, sub_const, mul1_const]:  #, mul2_const, sub_const, mul1_const, mul3_const]:
      G.add_node(node.name, data=node, type=node.op)
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['tf/Mul3']

  def output_node(self):
    return "tf/Add2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    node_def = tf_create_dummy_node(matched_subgraph[self.output_node()], 'BatchNorm')
    eps = source_graph.nodes[matched_subgraph['tf/Const2']]['data']
    if len(eps.attr['value'].tensor.float_val) is not 0:
      eps = eps.attr['value'].tensor.float_val[0]
      node_def.attr['epsilon'].tensor.float_val.append(eps)
      var = source_graph.nodes[matched_subgraph['tf/Const1']]['data'].name
      node_def.attr['variance'].MergeFrom(source_graph.nodes[var]['data'].attr['value'])
      scale = source_graph.nodes[matched_subgraph['tf/Const5']]['data'].name
      node_def.attr['scale'].MergeFrom(source_graph.nodes[scale]['data'].attr['value'])
      mean = source_graph.nodes[matched_subgraph['tf/Const3']]['data'].name
      node_def.attr['mean'].MergeFrom(source_graph.nodes[mean]['data'].attr['value'])
      bias = source_graph.nodes[matched_subgraph['tf/Const4']]['data'].name
      node_def.attr['bias'].MergeFrom(source_graph.nodes[bias]['data'].attr['value'])
    else:
      eps = source_graph.nodes[matched_subgraph['tf/Const1']]['data']
      eps = eps.attr['value'].tensor.float_val[0]
      node_def.attr['epsilon'].tensor.float_val.append(eps)
      var = source_graph.nodes[matched_subgraph['tf/Const2']]['data'].name
      node_def.attr['variance'].MergeFrom(source_graph.nodes[var]['data'].attr['value'])
      scale = source_graph.nodes[matched_subgraph['tf/Const5']]['data'].name
      node_def.attr['scale'].MergeFrom(source_graph.nodes[scale]['data'].attr['value'])
      mean = source_graph.nodes[matched_subgraph['tf/Const3']]['data'].name
      node_def.attr['mean'].MergeFrom(source_graph.nodes[mean]['data'].attr['value'])
      bias = source_graph.nodes[matched_subgraph['tf/Const4']]['data'].name
      node_def.attr['bias'].MergeFrom(source_graph.nodes[bias]['data'].attr['value'])
    node_def.input[:] = inputs_to_collapsed
    return node_def


class BatchnormMatcherTG(BaseMatcher):

  def __init__(self):
    pass

  def node_comparator_by_type(self, n1, n2):
    n1t = n1.get('type', True)
    n2t = n2.get('type', False)
    if n2t == 'Constant_':
      # Identity Constant vs Constant ---
      for const in constant_tensor:
        if const in n1t:
          return True
      return False
    # Const* vs Constant
    p = re.compile('.*' + n2t + '.*', re.IGNORECASE)
    x = p.match(n1t)
    return not x == None

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    add1 = tf_create_dummy_node("tf/Add1", "Add")
    rsqrt = tf_create_dummy_node("tf/Rsqrt", "Rsqrt")
    mul1 = tf_create_dummy_node("tf/Mul1", "Mul")
    mul2 = tf_create_dummy_node("tf/Mul2", "Mul")
    sub = tf_create_dummy_node("tf/Sub", "Sub")
    add2 = tf_create_dummy_node("tf/Add2", "Add")
    add1_const1 = tf_create_dummy_node("tf/Const1", "Const")
    add1_const2 = tf_create_dummy_node("tf/Const2", "Const")
    mul2_const = tf_create_dummy_node("tf/Const3", "Const")
    sub_const = tf_create_dummy_node("tf/Const4", "Const")
    edge_list.append((add1.name, rsqrt.name))
    edge_list.append((rsqrt.name, mul1.name))
    edge_list.append((rsqrt.name, mul2.name))
    edge_list.append((mul1.name, add2.name))
    edge_list.append((mul2.name, sub.name))
    edge_list.append((sub.name, add2.name))
    edge_list.append((add1_const1.name, add1.name))
    edge_list.append((add1_const2.name, add1.name))
    edge_list.append((sub_const.name, sub.name))
    edge_list.append((mul2_const.name, mul2.name))
    for node in [add1, rsqrt, mul1, mul2, sub, add2, add1_const1, add1_const2, mul2_const, sub_const]:
      G.add_node(node.name, data=node, type=node.op)
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['tf/Mul1']

  def output_node(self):
    return "tf/Add2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    node_def = tf_create_dummy_node(matched_subgraph[self.output_node()], 'BatchNorm')
    eps = source_graph.nodes[matched_subgraph['tf/Const1']]['data']
    if len(eps.attr['value'].tensor.float_val) is not 0:
      eps = eps.attr['value'].tensor.float_val[0]
      node_def.attr['epsilon'].tensor.float_val.append(eps)
      var = source_graph.nodes[matched_subgraph['tf/Const2']]['data'].name
      node_def.attr['variance'].MergeFrom(source_graph.nodes[var]['data'].attr['value'])
      mean = source_graph.nodes[matched_subgraph['tf/Const3']]['data'].name
      node_def.attr['mean'].MergeFrom(source_graph.nodes[mean]['data'].attr['value'])
      bias = source_graph.nodes[matched_subgraph['tf/Const4']]['data'].name
      node_def.attr['bias'].MergeFrom(source_graph.nodes[bias]['data'].attr['value'])
    else:
      eps = source_graph.nodes[matched_subgraph['tf/Const2']]['data']
      eps = eps.attr['value'].tensor.float_val[0]
      node_def.attr['epsilon'].tensor.float_val.append(eps)
      var = source_graph.nodes[matched_subgraph['tf/Const1']]['data'].name
      node_def.attr['variance'].MergeFrom(source_graph.nodes[var]['data'].attr['value'])
      mean = source_graph.nodes[matched_subgraph['tf/Const3']]['data'].name
      node_def.attr['mean'].MergeFrom(source_graph.nodes[mean]['data'].attr['value'])
      bias = source_graph.nodes[matched_subgraph['tf/Const4']]['data'].name
      node_def.attr['bias'].MergeFrom(source_graph.nodes[bias]['data'].attr['value'])
    node_def.input[:] = inputs_to_collapsed
    return node_def


class UpsampleFromConcatMatcher(BaseMatcher):

  def __init__(self):
    pass

  def subgraph_to_match(self):
    logger.debug("trying to match upsample")
    G = nx.DiGraph()
    edge_list = list()
    pack1 = tf_create_dummy_node("tf_Pack1", "Pack")
    pack2 = tf_create_dummy_node("tf_Pack2", "Pack")
    reshape = tf_create_dummy_node("tf_reshape", "Reshape")
    for node in [pack1, pack2, reshape]:
      G.add_node(node.name, data=node, type=node.op)
    edge_list.append(tuple((pack1.name, pack2.name)))
    edge_list.append(tuple((pack2.name, reshape.name)))
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ["tf_Pack1"]

  def output_node(self):
    return "tf_reshape"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    logger.debug("trying to match upsample1")
    output_node_name = matched_subgraph[self.output_node()]
    node_def = tf_create_dummy_node(output_node_name, 'ResizeNearestNeighbor')
    node_def.input[:] = inputs_to_collapsed
    return node_def


class Relu6ScaledMatcher(BaseMatcher):

  def __init__(self):
    pass

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    add = tf_create_dummy_node("tf/Add", "Add")
    relu6 = tf_create_dummy_node("tf/Relu6", "Relu6")
    mul = tf_create_dummy_node("tf/Mul1", "Mul")
    add_const = tf_create_dummy_node("tf/Const1", "Const")
    mul_const = tf_create_dummy_node("tf/Const2", "Const")
    edge_list.append((add.name, relu6.name))
    edge_list.append((relu6.name, mul.name))
    edge_list.append((add_const.name, add.name))
    edge_list.append((mul_const.name, mul.name))
    for node in [add, relu6, mul, add_const, mul_const]:
      G.add_node(node.name, data=node, type=node.op)
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['tf/Add']

  def output_node(self):
    return "tf/Mul1"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    node_def = tf_create_dummy_node(matched_subgraph[self.output_node()], 'Relu6')
    node_def.input[:] = inputs_to_collapsed
    return node_def


class HardSwishMatcher(BaseMatcher):

  def __init__(self):
    pass

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    add = tf_create_dummy_node("tf/Add", "Add")
    relu6 = tf_create_dummy_node("tf/Relu6", "Relu6")
    mul1 = tf_create_dummy_node("tf/Mul", "Mul")
    mul2 = tf_create_dummy_node("tf/Mul2", "Mul")
    add_const = tf_create_dummy_node("tf/Const1", "Const")
    mul_const = tf_create_dummy_node("tf/Const2", "Const")
    edge_list.append((add.name, relu6.name))
    edge_list.append((relu6.name, mul1.name))
    edge_list.append((mul1.name, mul2.name))
    edge_list.append((add_const.name, add.name))
    edge_list.append((mul_const.name, mul2.name))
    for node in [add, relu6, mul1, mul2, add_const, mul_const]:
      G.add_node(node.name, data=node, type=node.op)
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ['tf/Add']

  def output_node(self):
    return "tf/Mul2"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    node_def = tf_create_dummy_node(matched_subgraph[self.output_node()], 'HardSwish')
    node_def.input[:] = inputs_to_collapsed
    return node_def


class AddV2Matcher(BaseMatcher):

  def __init__(self):
    pass

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    add1 = tf_create_dummy_node("tf/Add1", "AddV2")
    const1 = tf_create_dummy_node("tf/biases1", "Const")
    const2 = tf_create_dummy_node("tf/constant1", "Const")
    edge_list.append((const1.name, add1.name))
    edge_list.append((const2.name, add1.name))
    for node in [add1, const1, const2]:
      G.add_node(node.name, data=node, type=node.op)
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return []

  def output_node(self):
    return "tf/Add1"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    node_def = tf_create_dummy_node(matched_subgraph[self.output_node()], "Const")
    bias_node = source_graph.nodes[matched_subgraph["tf/biases1"]]
    const_node = source_graph.nodes[matched_subgraph["tf/constant1"]]
    val1 = tf.make_ndarray(bias_node["data"].attr["value"].tensor)
    val2 = tf.make_ndarray(const_node["data"].attr["value"].tensor)
    value = tf.make_tensor_proto(val1 + val2)
    node_def.attr["value"].tensor.CopyFrom(value)
    node_def.input[:] = inputs_to_collapsed
    return node_def


class SwishMatcher(BaseMatcher):

  def __init__(self):
    pass

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    sig = tf_create_dummy_node("tf/Sigmoid", "Sigmoid")
    mul = tf_create_dummy_node("tf/Mul", "Mul")
    edge_list.append(("tf/Sigmoid", "tf/Mul"))
    G.add_node(sig.name, data=sig, type="Sigmoid")
    G.add_node(mul.name, data=sig, type="Mul")
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return ["tf/Sigmoid"]

  def output_node(self):
    return "tf/Mul"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    # extra checks to ensure correct matchings.
    inputs_to_sig = source_graph.nodes[matched_subgraph['tf/Sigmoid']]['data'].input
    inputs_to_mul = source_graph.nodes[matched_subgraph['tf/Mul']]['data'].input
    if len(set(inputs_to_sig).intersection(set(inputs_to_mul))) == 0:
      return None
    node_def = tf_create_dummy_node(matched_subgraph[self.output_node()], 'Swish')
    node_def.input[:] = inputs_to_collapsed
    return node_def


class ConstMulMatcher(BaseMatcher):

  def __init__(self):
    pass

  def subgraph_to_match(self):
    G = nx.DiGraph()
    edge_list = list()
    mul = tf_create_dummy_node("tf/Mul", "Mul")
    const1 = tf_create_dummy_node("tf/const1", "Const")
    const2 = tf_create_dummy_node("tf/const2", "Const")
    edge_list.append((const1.name, mul.name))
    edge_list.append((const2.name, mul.name))
    for node in [mul, const1, const2]:
      G.add_node(node.name, data=node, type=node.op)
    G.add_edges_from(edge_list)
    return G

  def input_nodes(self):
    return []

  def output_node(self):
    return "tf/Mul"

  def mk_replacement_node(self, source_graph, matched_subgraph, model, inputs_to_collapsed, nodes_to_remove=None):
    node_def = tf_create_dummy_node(matched_subgraph[self.output_node()], "Const")
    const1_node = source_graph.nodes[matched_subgraph["tf/const1"]]
    const2_node = source_graph.nodes[matched_subgraph["tf/const2"]]
    val1 = tf.make_ndarray(const1_node["data"].attr["value"].tensor)
    val2 = tf.make_ndarray(const2_node["data"].attr["value"].tensor)
    value = tf.make_tensor_proto(val1 * val2)
    node_def.attr["value"].tensor.CopyFrom(value)
    node_def.input[:] = inputs_to_collapsed
    return node_def


def tf_subGraph_match_replace_Upsample3D(G, model):
  return Upsample3DMatcher().replace_subgraph_instances(G, model)


def tf_subGraph_match_replace_Upsample_HW(G, model):
  return UpsampleHWMatcher().replace_subgraph_instances(G, model)


def tf_subGraph_match_replace_Deconv3D(G, model):
  return Deconv3DMatcher().replace_subgraph_instances(G, model)


def tf_subGraph_match_replace_L2Norm(G, model):
  return L2NormMatcher().replace_subgraph_instances(G, model)


def tf_subGraph_match_replace_prelu(G, model):
  return PReLUMatcher().replace_subgraph_instances(G, model)


def tf_subGraph_match_replace_lrelu(G, model):
  return LeakyReluMatcher().replace_subgraph_instances(G, model)


def tf_subGraph_match_replace_atrous_conv2d(G, model):
  return AtrousConv2DMatcher().replace_subgraph_instances(G, model)


def tf_subGraph_match_replace_SplitSoftmax(G, model):
  return SplitSoftmaxMatcher().replace_subgraph_instances(G, model)


def tf_subGraph_match_replace_Relu6(G, model):
  return Relu6Matcher().replace_subgraph_instances(G, model)


def tf_subGraph_match_replace_Relu6_1(G, model):
  return Relu6Matcher_1().replace_subgraph_instances(G, model)


def tf_subGraph_match_replace_Batchnorm(G, model):
  return BatchnormMatcher().replace_subgraph_instances(G, model)


def tf_subGraph_match_replace_BatchnormScaled(G, model):
  return BatchnormScaledMatcher().replace_subgraph_instances(G, model)


def tf_subGraph_match_replace_BatchnormScaledTG(G, model):
  return BatchnormScaledTG().replace_subgraph_instances(G, model)


def tf_subGraph_match_replace_BatchnormTG(G, model):
  return BatchnormMatcherTG().replace_subgraph_instances(G, model)


def tf_subGraph_match_replace_BiFPNinterim(G, model):
  return BiFPNinterimMatcher().replace_subgraph_instances(G, model)


def tf_subGraph_match_replace_BiFPN(G, model):
  return BiFPNMatcher().replace_subgraph_instances(G, model)


def tf_subgraph_match_replace_Mish(G, model):
  return MishMatcher().replace_subgraph_instances(G, model)


def tf_subgraph_match_replace_Yolov4_IR1(G, model):
  return Yolov4_IR1Matcher().replace_subgraph_instances(G, model)


def tf_subgraph_match_replace_UpsampleFromConcat(G, model):
  return UpsampleFromConcatMatcher().replace_subgraph_instances(G, model)


def tf_subGraph_match_replace_Relu6Scaled(G, model):
  return Relu6ScaledMatcher().replace_subgraph_instances(G, model)


def tf_subGraph_match_replace_HardSwish(G, model):
  return HardSwishMatcher().replace_subgraph_instances(G, model)


def tf_subgraph_match_replace_Swish(G, model):
  return SwishMatcher().replace_subgraph_instances(G, model)


def tf_subGraph_match_replace_AddV2(G, model):
  return AddV2Matcher().replace_subgraph_instances(G, model)


def tf_subGraph_match_replace_ConstMul(G, model):
  return ConstMulMatcher().replace_subgraph_instances(G, model)
