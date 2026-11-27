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

from mmdnn.conversion.caffe.c2c.common import BasePass
from dv_logging import get_logger

logger = get_logger(__name__, 'dvctoc')


# class Axis2to1(BasePass):
#   def __init__(self):
#     pass

#   def do_transformation(self):
#     layers = self.get_nodes_of_type('Softmax')
#     for layer in layers:
#       if layer.softmax_param.axis != 2:
#         continue
#       layer.softmax_param.axis = 1
#       bottom_shape = [i for i in self.caffe_net.blobs[layer.bottom[0]].data.shape]
#       top_shape = [i for i in self.caffe_net.blobs[layer.top[0]].data.shape]
#       bottom_shape_new = [bottom_shape[0]*bottom_shape[1]]+bottom_shape[2:]
#       reshape_0 = self.G.mk_reshape_proto(layer.bottom[0], bottom_shape_new, layer.name)
#       bottom_layer = layer.predecessors()[0]
#       self.G.add_node(reshape_0)
#       self.G.add_edge(bottom_layer, reshape_0)
#       self.G.add_edge(reshape_0, layer)
#       self.G.remove_edge(bottom_layer, layer)
#       self.G.add_replace_edge_proto_new(reshape_0, layer, layer.bottom[0], reshape_0.top[0])
#       reshape_1 = self.G.mk_reshape_proto(layer.top[0], top_shape, layer.name)
#       succs = layer.successors()
#       self.G.add_node(reshape_1)
#       self.G.add_edge(layer, reshape_1)
#       for succ in succs:
#         self.G.add_edge(reshape_1, succ)
#         self.G.remove_edge(layer, succ)
#         self.G.add_replace_edge_proto_new(reshape_1, succ, layer.top[0], reshape_1.top[0])

class Axis2to1(BasePass):
  def __init__(self):
    pass

  def do_transformation(self):
    layers = self.get_nodes_of_type('Softmax')
    for layer in layers:
      if layer.softmax_param.axis != 2:
        continue
      layer.softmax_param.axis = 1
      succs = list(layer.successors())
      size = len(self.caffe_net.blobs[layer.top[0]].data.shape)
      perm1 = self.G.mk_permute_proto(layer.name, [0,2,1]+[i for i in range(3,size)], bottom=layer.bottom[0])
      self.G.add_node(perm1)
      bottom_layer = layer.predecessors()[0]
      self.G.add_edge(bottom_layer, perm1)
      self.G.add_edge(perm1, layer)
      self.G.remove_edge(bottom_layer, layer)
      self.G.add_replace_edge_proto_new(perm1, layer, layer.bottom[0], perm1.top[0])
      perm2 = self.G.mk_permute_proto(layer.name, [0,2,1]+[i for i in range(3,size)], bottom=layer.top[0])      
      self.G.add_node(perm2)
      self.G.add_edge(layer, perm2)
      for succ in succs:
        self.G.add_edge(perm2, succ)
        self.G.remove_edge(layer, succ)
        self.G.add_replace_edge_proto_new(perm2, succ, layer.top[0], perm2.top[0])


class SoftmaxPermuteAxis2to1_1(BasePass):
  def __init__(self):
    pass

  '''
  If Softmax axis 2 is followed by permute 0,2,1. Then move softmax down of permute
  '''

  def do_transformation(self):
    layers = self.get_nodes_of_type('Softmax')
    for layer in layers:
      if layer.softmax_param.axis != 2:
        continue
      if len(layer.successors()) != 1:
          continue
      permute = layer.successors()[0]        
      if permute.type not in ['Permute','PermuteNnp']:
        continue
      if permute.type == 'Permute' and permute.permute_param.order != [0,2,1]:
        continue
      layer.softmax_param.axis = 1
      pars = list(layer.predecessors())
      succs = list(permute.successors())
      self.G.remove_edge(pars[0], layer)
      self.G.remove_edge(layer, permute)
      self.G.add_edge(permute, layer)
      self.G.add_edge(pars[0], permute)
      self.G.add_replace_edge_proto_new(pars[0], permute, layer.top[0], layer.bottom[0])
      self.G.add_replace_edge_proto_new(permute, layer, layer.bottom[0], permute.top[0])
      for succ in succs:
        self.G.add_edge(layer, succ)
        self.G.remove_edge(permute, succ)
        self.G.add_replace_edge_proto_new(layer, succ, permute.top[0], layer.top[0])


class SoftmaxPermuteAxis2to1_2(BasePass):
  def __init__(self):
    pass
  '''
  If Softmax axis 2 is followed by reshape permutennp and reshape. Then move softmax down of 2nd reshape
  '''
  def do_transformation(self):
    layers = self.get_nodes_of_type('Softmax')
    for layer in layers:
      if layer.softmax_param.axis != 2:
        continue
      if len(layer.successors()) != 1:
          continue
      reshape1 = layer.successors()[0]        
      if reshape1.type not in ['Reshape']:
        continue
      if len(reshape1.successors()) != 1:
          continue
      permute = reshape1.successors()[0]  
      if permute.type not in ['Permute','PermuteNnp']:
        continue
      if permute.type == 'Permute' and permute.permute_param.order != [0,1,3,2]:
        continue
      if len(permute.successors()) != 1:
          continue
      reshape2 = permute.successors()[0]        
      if reshape2.type not in ['Reshape']:
        continue
      reshape1_inp_shape = list(self.caffe_net.blobs[reshape1.bottom[0]].data.shape)
      reshape1_out_shape = list(self.caffe_net.blobs[reshape1.top[0]].data.shape)
      if reshape1_out_shape != [reshape1_inp_shape[0], 1, reshape1_inp_shape[1], reshape1_inp_shape[2]]:
        continue
      reshape2_inp_shape = list(self.caffe_net.blobs[reshape2.bottom[0]].data.shape)
      reshape2_out_shape = list(self.caffe_net.blobs[reshape2.top[0]].data.shape)
      if reshape2_inp_shape != [reshape2_out_shape[0], 1, reshape2_out_shape[1], reshape2_out_shape[2]]:
        continue
      layer.softmax_param.axis = 1
      pars = list(layer.predecessors())
      succs = list(reshape2.successors())
      self.G.remove_edge(pars[0], layer)
      self.G.remove_edge(layer, reshape1)
      self.G.add_edge(pars[0], reshape1)
      self.G.add_edge(reshape2, layer)
      self.G.add_replace_edge_proto_new(pars[0], reshape1, layer.top[0], layer.bottom[0])
      self.G.add_replace_edge_proto_new(reshape2, layer, layer.bottom[0], reshape2.top[0])
      for succ in succs:
        self.G.add_edge(layer, succ)
        self.G.remove_edge(reshape2, succ)
        self.G.add_replace_edge_proto_new(layer, succ, reshape2.top[0], layer.top[0])

class PermuteSoftmaxPermute(BasePass):
  '''
  Permute([0,3,1,2]) -> Softmax(axis = 3) -> Permute([0,3,2,1])
  |||
  Permute([0,2,1,3]) -> Softmax(axis=1)
  '''
  def __init__(self):
    pass

  def is_feasible(self, permute_layer, sfmx_layer, permute2_layer):
    if permute_layer.permute_param.order != [0,3,1,2]:
      return False
    if permute2_layer.permute_param.order != [0,3,2,1]:
      return False
    if sfmx_layer.softmax_param.axis != 3:
      return False
    
    return True

  def do_transformation(self):
    layers = self.get_nodes_of_type('Permute')
    removed_layers = set()
    for layer in layers:
      if layer.name in removed_layers:
        continue
      permute_layer = layer
      successors = permute_layer.successors()
      if len(successors) != 1:
        continue
      sfmx_layer = successors[0]
      if sfmx_layer.type != "Softmax":
        continue
      successors = sfmx_layer.successors()
      if len(successors) != 1:
        continue
      permute2_layer = successors[0]
      if permute2_layer.type != "Permute":
        continue
      if not self.is_feasible(permute_layer, sfmx_layer, permute2_layer):
        continue
      successors = permute2_layer.successors()
      next_layers = successors
      #Change 1st permute order to [0,2,1,3]
      permute_layer.permute_param.ClearField('order')
      permute_layer.permute_param.order.extend([0,2,1,3])
      sfmx_layer.softmax_param.axis = 1
      # Remove 2nd permute
      self.G.remove_edge(sfmx_layer, permute2_layer)
      for n_layer in next_layers:
        self.G.remove_edge(permute2_layer, n_layer)
        self.G.add_edge(sfmx_layer, n_layer)
        self.G.add_replace_edge_proto(sfmx_layer, n_layer, permute2_layer.top[0])
      removed_layers.add(permute2_layer.name)
      self.G.remove_node(permute2_layer)

