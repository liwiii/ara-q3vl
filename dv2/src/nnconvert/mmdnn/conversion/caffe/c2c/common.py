'''
Copyright (c) 2018-25, Kinara, Inc. All rights reserved.
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
from functools import wraps
from caffe.proto import caffe_pb2
import os
import caffe
import google.protobuf as pb
import copy
import random

from caffe import layers as L
from caffe import params as P

from dv_logging import get_logger
from mmdnn.conversion.caffe.c2c import common

logger = get_logger(__name__, 'DVCTOC')

ARASUFFIX = "_ara"

class BasePass:

  def checkpoint_graph(self):
    layer_pb = caffe_pb2.NetParameter()
    for node in self.G.get_topsort():
      layer_pb.layer.append(node.get_layer())

    if not os.path.exists('./checkpoints'):
      os.makedirs('./checkpoints')
    filename = './checkpoints/shape_{}.prototxt'.format(self.idx)
    self.G.to_dotfile('./checkpoints/shape_{}.dot'.format(self.idx))
    with open(filename, 'w') as f:
      f.write(pb.text_format.MessageToString(layer_pb))
      f.flush()
      os.fsync(f.fileno())
    self.idx = self.idx + 1
    return caffe.Net(filename, caffe.TEST)

  def set_graph(self, G):
    self.G = G
    self.idx = -1

  def set_caffe_net(self, caffe_net):
    self.caffe_net = caffe_net

  def get_topsort(self):
    return self.G.get_topsort()

  def get_nodes_of_type(self, node_type):
    return self.G.get_nodes_of_type(node_type)

  def get_caffe_blob_shape(self, bottom_name):
    caffe_net = self.checkpoint_graph()
    return caffe_net.blobs[bottom_name].shape

  def get_caffe_param_shape(self, layer_name, idx):
    caffe_net = self.checkpoint_graph()
    return caffe_net.params[layer_name][idx].shape

  def swap_nodes(self, x, y):
    """ in a run of nodes 
      x -> M -> y -> N, swap M and y to get
      x -> y -> M -> N
      """

    M = x.successors()[0]
    self.G.remove_edge(x, M)
    self.G.remove_edge(M, y)
    self.G.add_edge(x, y)
    self.G.add_edge(y, M)
    self.G.add_replace_edge_proto(y, M, x.name)
    self.G.add_replace_edge_proto(x, y, M.name)
    for N in y.successors():
      self.G.add_edge(M, N)
      self.G.add_replace_edge_proto(M, N, "Junk")

  def set_pp_yaml(self, pp_yaml):
    self.pp_yaml = pp_yaml

  def set_cp_only_list(self, cp_only_list):
    self.cp_only_list = cp_only_list

  def set_caffe_cp_only_list(self, caffe_cp_only_list):
    self.caffe_cp_only_list = caffe_cp_only_list

  def set_skip_list(self, skip_list):
    self.skip_list = skip_list

  def set_fork_sftmx_argmx(self, fork_sftmx_argmx):
    self.fork_sftmx_argmx = fork_sftmx_argmx

  def set_cp_precision(self, cp_precision):
    self.cp_precision = cp_precision

  def set_inp_wts(self, inp_wts):
    self.inp_wts = inp_wts

  def set_dynamic(self, dynamic):
    self.dynamic = dynamic

  def set_dry_run(self, dry_run):
    self.dry_run = dry_run

  def set_fuse_act(self, fuse_act):
    self.fuse_act = fuse_act

  def set_dynamic_input(self, dynamic_input):
    self.dynamic_input = dynamic_input

  def set_no_final_perm_opt(self, no_final_perm_opt):
    self.no_final_perm_opt = no_final_perm_opt

  def set_platform_target(self, platform_target):
    self.platform_target = platform_target
  
  def set_inode_names(self,inode_names):
    self.inode_names = inode_names

  def set_onode_names(self,onode_names):
    self.onode_names = onode_names

  def set_input_permuted(self, input_permuted):
    self.input_permuted = input_permuted
  
  def set_output_permuted(self, output_permuted):
    self.output_permuted = output_permuted

  def set_tf2onnx_happened(self, tf2onnx_happened):
    self.tf2onnx_happened = tf2onnx_happened
  
  def set_adjust_file(self, adjust_file):
    self.adjust_file = adjust_file

  def get_output_nodes(self):
    all_nodes = self.G.get_topsort()
    output_nodes = []
    for node in all_nodes:
      if len(node.successors()) == 0:
        output_nodes.append(node)
    return output_nodes
  
  def set_mutation_observer(self, mutation_observer):
    self.mutation_observer = mutation_observer


class DVGraphNode:

  def __init__(self, G, layer):
    self.G = G
    self._layer = layer

  def _sort_edges_by_weight(self, edges):
    data = []
    for (u, v) in edges:
      edge_data = self.G.get_edge_data(u, v)
      idx = edge_data['idx']
      data.append((u, v, idx))
    return sorted(data, key=lambda it: it[2])

  def successors(self):
    successors = list(sorted(self.G.successors(str(self))))
    return list(map(lambda x: self.G.nodes[x]['node_obj'], successors))

  def predecessors(self):
    predecessors = list(sorted(self.G.predecessors(str(self))))
    predecessors = map(lambda it: (it, str(self)), predecessors)
    predecessors = self._sort_edges_by_weight(predecessors)
    predecessors = map(lambda it: it[0], predecessors)
    return list(map(lambda x: self.G.nodes[x]['node_obj'], predecessors))

  def get_layer(self):
    return self._layer

  def __getattr__(self, attr):
    return getattr(self._layer, attr)

  def __str__(self):
    return '{}/{}'.format(self.type, self.name)


class DVDiGraph(nx.DiGraph):
  """custom digraph class that we use to make
    things easier for us to use
    """

  # pylint: disable=E0101
  def __init__(self, *args, **kwargs):
    self.mutation_observers = []
    return super().__init__(*args, **kwargs)

  def add_mutation_observer(self, observer):
    self.mutation_observers.append(observer)

  def _on_mutation(self, *args):
    for observer in self.mutation_observers:
      observer(*args)

  def add_node(self, node):
    __name = node.name
    if ARASUFFIX in __name:
      __name = __name.split(ARASUFFIX)[0]
    node = DVGraphNode(self, node)
    self._on_mutation('add_node', node)
    return super().add_node(str(node), node_obj=node, label=__name)

  def remove_node(self, node):
    if type(node) != DVGraphNode:
      node = DVGraphNode(self, node)
    self._on_mutation('remove_node', node)
    return super().remove_node(str(node))

  def add_edge(self, n1, n2, idx=0):
    if type(n1) != DVGraphNode:
      n1 = DVGraphNode(self, n1)
    if type(n2) != DVGraphNode:
      n2 = DVGraphNode(self, n2)
    n2_pred = list(n2.predecessors())
    self._on_mutation('add_edge', n1, n2, len(n2_pred) + 1)
    # n2._layer.bottom.extend([n1._layer.name])
    # import pdb;pdb.set_trace()
    # print(n2._layer.bottom)
    # n2._layer.bottom.extend([n1._layer.name])
    # n2._layer.bottom.append(n1._layer.name)
    return super().add_edge(str(n1), str(n2), idx=len(n2_pred) + 1)

  """
  Changes name of any node in the graph to the new name provided in mapping.

  Args:
    mapping: {old_name:new_name}
    layertype: type of node to be relabelled
  """

  def relabel_nodes_locally(self, mapping, layertype):
    nodes = [n for n in list(self.nodes) if n in mapping]

    top_mapping = dict()
    for old in nodes:
      try:
        new = mapping[old]
        if new == old:
          continue
        new_node = copy.deepcopy(self.nodes[old]['node_obj'])
        new_node.name = new
        top_list = list(new_node.top)
        len_top_list = len(top_list)
        if len_top_list == 1:
          new_node.top[:] = [new]
          top_mapping[top_list[0]] = new
        else:
          for idx in range(len_top_list):
            curr_top = top_list[idx]
            top_list.remove(curr_top)
            if idx != 0:
              new_top = f"{new}_{str(idx)}"
            else:
              new_top = new
            top_list.insert(idx, new_top)
            top_mapping[curr_top] = new_top

          new_node.top[:] = top_list
        self.add_node(new_node)
      except KeyError as ke:
        logger.debug("Key Error in relabelling: ", str(ke))
        continue
      for successor in self.nodes[old]['node_obj'].successors():
        bottom_list = successor.bottom
        if len(bottom_list) == 1:
          successor.bottom[:] = [new]
        else:

          new_bottom_list = list(bottom_list)
          bottom_list_len = len(new_bottom_list)
          for idx in range(bottom_list_len):
            if new_bottom_list[idx] in top_mapping.keys():
              curr_bottom = new_bottom_list[idx]
              new_bottom_list.remove(curr_bottom)
              new_bottom_list.insert(idx, top_mapping[curr_bottom])
          successor.bottom[:] = new_bottom_list

      new_edges = []
      for (from_node, to_node, data) in self.edges(data=True):
        if from_node == old:
          new_edges.append((layertype + '/' + new, to_node, data))
        elif to_node == old:
          new_edges.append((from_node, layertype + '/' + new, data))
      self.remove_node(self.nodes[old]['node_obj'])
      self.add_edges_from(new_edges)

  def remove_edge(self, n1, n2):
    if type(n1) != DVGraphNode:
      n1 = DVGraphNode(self, n1)
    if type(n2) != DVGraphNode:
      n2 = DVGraphNode(self, n2)
    self._on_mutation('remove_edge', n1, n2)
    return super().remove_edge(str(n1), str(n2))

  def add_edges_from(self, collection):
    for (n1, n2, idx) in collection:
      if isinstance(n1, str):
        n1 = self.nodes[n1]['node_obj']
      if isinstance(n2, str):
        n2 = self.nodes[n2]['node_obj']
      self.add_edge(n1, n2, idx)

  def get_topsort(self):
    nodes = []
    topsort = list(nx.lexicographical_topological_sort(self, key=lambda x: self.nodes[x]['label']))
    for node in topsort:
      node_obj = super().nodes[node]['node_obj']
      nodes.append(node_obj)
    sorted_nodes = []
    for node in nodes:
      if node.type == 'Input':
        sorted_nodes.insert(0, node)
      else:
        sorted_nodes.append(node)
    return sorted_nodes

  def get_shortest_path(self, inp_node, dst_node):
    dist = -1
    try:
      dist = nx.shortest_path_length(self, inp_node, dst_node)
    except nx.exception.NetworkXNoPath:
      pass
    except Exception as e:
      logger.warn("[E] fetching path lengths error %s -> %s , %s", str(e), inp_node, dst_node)
    return dist

  def set_caffe_net(self, caffe_net):
    self.caffe_net = caffe_net

  def get_nodes_of_type(self, node_type, ignore_precision=True):
    nodes = []
    for node_obj in self.get_topsort():
      if node_obj.type == node_type:
        nodes.append(node_obj)
    return nodes

  def get_nodes_of_base_type(self, node_type):
    nodes = []
    for node_obj in self.get_topsort():
      if node_obj.base_type == node_type:
        nodes.append(node_obj)
    return nodes

  def from_proto(self):
    pass

  def to_dotfile(self, path):
    import pydot
    from networkx.drawing.nx_pydot import write_dot
    write_dot(self, path)

  def add_replace_edge_proto(self, layer1, layer2, replace_layer_name):
    present = False
    for idx, layer_name in enumerate(layer2.bottom):
      if layer_name == replace_layer_name:
        layer2.bottom[idx] = layer1.top[0]
        present = True
    if not present:
      layer2.bottom.extend([layer1.top[0]])

  def add_replace_edge_proto_new(self, layer1, layer2, replace_layer_name, layer_bottom):
    present = False
    for idx, layer_name in enumerate(layer2.bottom):
      if layer_name == replace_layer_name:
        layer2.bottom[idx] = layer_bottom
        present = True
    if not present:
      layer2.bottom.extend([layer1.top[0]])

  def get_layername(self, name, layertype):
    idx = 0
    node_list = []

    for i in list(self.nodes):
      curr_list = i.split('/')
      if len(curr_list) == 2:
        node_list.append(curr_list[1])
      else:
        node_list.append('/'.join(curr_list[1:]))
    if '_ara' in name:
      pos = name.find('_ara')
      name = name[0:pos]
    while True:
      curr_name = f"{name}_ara_{layertype}_{str(idx)}"
      if curr_name not in node_list:
        return curr_name
      idx += 1

  # Unused
  def mk_padding_proto(self, l, r, t, b, method, pad_value, layer_name):
    layer = L.Padding(padding_param=dict(pad=[t, l, b, r], pad_method=method, pad_value=pad_value)).to_proto().layer[0]
    layer.name = self.get_layername(layer_name, 'pad')
    layer.top[0] = layer.name
    return layer

  def mk_crop_proto(self, name, dummy_shape=[], crop_offset=[]):
    crop_layer = L.Crop(crop_param=dict(offset=crop_offset)).to_proto().layer[0]
    crop_layer.name = self.get_layername(name, 'crop')
    crop_layer.top[0] = crop_layer.name
    dummy_layer = L.DummyData(dummy_data_param=dict(shape=dict(dim=dummy_shape))).to_proto().layer[0]
    dummy_layer.name = self.get_layername(name, 'dummy')
    dummy_layer.top[0] = dummy_layer.name
    return dummy_layer, crop_layer

  def mk_dvshift_proto(self, bottom, prefix):
    dvshiftname = f"{prefix}custom_ara_dvshift"
    layer = L.DVShift(bottom=bottom, dvshift_param=dict(shift=0)).to_proto().layer[0]
    layer.name = self.get_layername(dvshiftname, 'dvshift')
    layer.top[0] = layer.name
    return layer

  def mk_permute_proto(self, name, order, bottom=None):
    if bottom == None:
      bottom = name
    layer = L.Permute(bottom=bottom, permute_param=dict(order=order)).to_proto().layer[0]
    layer.name = self.get_layername(name, 'permute')
    layer.top[0] = layer.name
    return layer

  def mk_permute_nnp_proto(self, layer_name, bottom_name, order, isKeyOutput=None, isValueOutput=None):
    layer = L.PermuteNnp(bottom=bottom_name, permute_nnp_param=dict(order=order)).to_proto().layer[0]
    layer.name = self.get_layername(layer_name, 'permute_nnp')
    layer.top[0] = layer.name
    if isKeyOutput is not None:
      layer.isKeyOutput = isKeyOutput
    if isValueOutput is not None:
      layer.isValueOutput = isValueOutput
    return layer

  def mk_aten_proto(self, bottom, name, shape_list):
    # currently works for only 1 output
    layer = L.Aten(bottom=bottom, aten_param=dict(shape_param=[dict(dim=shape_list)], op_params={})).to_proto().layer[0]
    layer.name = self.get_layername(name, 'aten')
    layer.top[0] = layer.name
    return layer

  def mk_broadcast_proto(self, input, name, final_shape):
    layer = L.Broadcast(bottom=input, broadcast_param=dict(shape=dict(dim=final_shape))).to_proto().layer[0]
    layer.name = self.get_layername(name, 'broadcast')
    layer.top[0] = layer.name
    return layer

  def mk_lut_proto(self, bottom, name, type):
    layer = L.Lut(bottom=bottom, lut_param=dict(type=type)).to_proto().layer[0]
    layer.name = self.get_layername(name, 'lut')
    layer.top[0] = layer.name
    return layer

  def mk_dvcast_proto(self, bottom, name, src_precision, dst_precision):
    dvcastname = "custom_ara_dvcast"
    layer = L.DVCast(bottom=bottom, dataPrecision=dst_precision, dvcast_param=dict(src_precision=src_precision, dst_precision=dst_precision)).to_proto().layer[0]
    layer.name = self.get_layername(dvcastname, 'dvcast')
    layer.top[0] = layer.name
    return layer

  def mk_reshape_proto(self, bottom, shape, name=''):
    if not name:
      name = bottom
    layer = L.Reshape(bottom=bottom, reshape_param=dict(shape=dict(dim=shape))).to_proto().layer[0]
    layer.name = self.get_layername(name, 'reshape')
    layer.top[0] = layer.name
    return layer

  def mk_dvtensor_proto(self, shape, tensortype, value, layer_name):
    dvtensorname = "custom_ara_dvtensor"
    layer = L.DVTensor(dvtensor_param=dict(tensortype=tensortype, value=value, shape=dict(dim=shape))).to_proto().layer[0]
    layer.name = self.get_layername(dvtensorname, 'dvtensor')
    layer.top[0] = layer.name
    return layer

  def mk_unpack_proto(self, shape, layer_name, inp_prc=8, out_prc=8, group_size=64, group_axis = 1):
    layer = L.UnpackNNP(unpack_nnp_param=dict(inp_prc=inp_prc, out_prc=out_prc, group_size=group_size, group_axis=group_axis, shape=dict(dim=shape))).to_proto().layer[0]
    layer.name = self.get_layername(layer_name, 'unpack')
    layer.top[0] = layer.name
    return layer

  def mk_concat_proto(self, inputs, layer_name, new_axis=1):
    layer = L.Concat(bottom=inputs, concat_param=dict(axis=new_axis)).to_proto().layer[0]
    layer.name = self.get_layername(layer_name, 'concat')
    layer.top[0] = layer.name
    return layer

  def mk_eltwise_proto(self, inputs, layer_name, operation, broadcast1=None):
    if(broadcast1 is not None):
      layer = L.Eltwise(bottom=inputs, operation=operation, broadcast1=broadcast1).to_proto().layer[0]
    else:
      layer = L.Eltwise(bottom=inputs, operation=operation).to_proto().layer[0]
    layer.name = self.get_layername(layer_name, 'eltwise')
    layer.top[0] = layer.name
    return layer

  def mk_conv_proto(self, input, layer_name, kernel_size, num_output, stride=None, pad=None, group=None, bias_term=False, dilation=None):
    layer = None
    if stride is None:
      layer = L.Convolution(bottom=input, kernel_size=kernel_size, num_output=num_output, bias_term=bias_term).to_proto().layer[0]
    else:
      layer = L.Convolution(bottom=input, kernel_size=kernel_size, stride=stride, num_output=num_output, pad=pad, group=group, bias_term=bias_term, dilation=dilation).to_proto().layer[0]
    layer.name = self.get_layername(layer_name, 'conv')
    layer.top[0] = layer.name
    return layer

  def mk_bnrm_proto(self, input, layer_name, use_global_stats=True, eps=1e-5):
    layer = None
    layer = L.BatchNorm(bottom=input, use_global_stats=use_global_stats, eps=eps).to_proto().layer[0]
    layer.name = self.get_layername(layer_name, 'batchnorm')
    layer.top[0] = layer.name
    return layer

  def mk_bdot_proto(self, input, layer_name, transpose_a=1, transpose_b=0, multiplier=1.0):
    layer = None
    layer = L.BatchDot(bottom=input, transpose_a=transpose_a, transpose_b=transpose_b, multiplier=multiplier).to_proto().layer[0]
    layer.name = self.get_layername(layer_name, 'batchdot')
    layer.top[0] = layer.name
    return layer

  def mk_scale_proto(self, input, layer_name, bias_term=False):
    layer = None
    layer = L.Scale(bottom=input, bias_term=bias_term).to_proto().layer[0]
    layer.name = self.get_layername(layer_name, 'scale')
    layer.top[0] = layer.name
    return layer

  def mk_softmax_proto(self, input, layer_name, axis=1):
    layer = None
    layer = L.Softmax(bottom=input, axis=axis).to_proto().layer[0]
    layer.name = self.get_layername(layer_name, 'softmax')
    layer.top[0] = layer.name
    return layer

  def mk_slice_proto(self, input, layer_name, dim_along_axis, axis=1, slice_point=[]):
    layer = None
    layer = L.Slice(bottom=input, axis=axis, slice_point=slice_point).to_proto().layer[0]
    layer.name = self.get_layername(layer_name, 'slice')
    slice_points = [0] + slice_point + [dim_along_axis]
    intervals = [slice_points[i + 1] - slice_points[i] for i in range(len(slice_points) - 1)]
    num_output_blobs = len(intervals)
    layer.top.pop()
    for i in range(num_output_blobs):
      layer.top.append(layer.name + '_' + str(i))
    return layer

  def mk_inrp_proto(self, inputs, num_output, bias_term, layer_name=None):
    layer = L.InnerProduct(bottom=inputs, num_output=num_output, bias_term=bias_term).to_proto().layer[0]
    if layer_name:
      # layer.name = layer_name  #+ "_inrp_c2c"
      layer.name = self.get_layername(layer_name, 'inrp')
      layer.top[0] = layer.name  #+ "_inrp_c2c"
    return layer

  def mk_swish_proto(self, inputs, layer_name=None):
    layer = L.Swish(bottom=inputs,).to_proto().layer[0]
    if layer_name:
      # layer.name = layer_name + "_swish_c2c"
      layer.name = self.get_layername(layer_name, 'swish')
      # layer.top[0] = layer.name + "_swish_top_c2c"
      layer.top[0] = layer.name
    return layer

  # Unused
  def mk_ibn_proto(self, layer_name='IBN'):
    layer = L.IBN(ibn_param=dict(in_eps=1e-3, batch_norm_param=dict(eps=1e-5))).to_proto().layer[0]
    layer.name = self.get_layername(layer_name, 'ibn')
    layer.top[0] = layer.name
    return layer

  def mk_pad_proto(self, pads, method, pad_value, layer_name):
    layer = L.Padding(padding_param=dict(pad=pads, pad_method=method, pad_value=pad_value)).to_proto().layer[0]
    layer.name = self.get_layername(layer_name, 'pad')
    layer.top[0] = layer.name
    return layer

  def mk_loop_begin_proto(self, name, loopend_layername='None'):
    layer = L.LoopBegin(loop_begin_param=dict(loopend_layername=loopend_layername)).to_proto().layer[0]
    layer.name = self.get_layername(name, 'loop_begin')
    layer.ClearField('top')
    return layer

  def mk_loop_end_proto(self, name, loopbegin_layername='None'):
    layer = L.LoopEnd(loop_end_param=dict(loopbegin_layername=loopbegin_layername)).to_proto().layer[0]
    layer.name = self.get_layername(name, 'loop_end')
    if loopbegin_layername in ['None']:
      layer.top[0] = layer.name
    else:
      layer.ClearField('top')
    return layer

  def mk_attentionsoftmax_proto(self, inputs, layername='None', axis=3):
    layer = None
    layer = L.AttentionSoftmax(bottom = inputs, attentionsoftmax_param=dict(axis=axis)).to_proto().layer[0]
    layer.name =  layername
    layer.top[0] = layer.name
    return layer
