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

import copy
import math
import numpy as np
import caffe
from caffe import layers as L
from caffe import params as P
from mmdnn.conversion.caffe.c2c.common import BasePass
from dv_logging import get_logger

logger = get_logger(__name__, 'dvctoc')


class EltwisePatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    eltwises = self.get_nodes_of_type('Eltwise')
    for layer in eltwises:
      assert len(layer.bottom) == 2
      pred_shape0 = self.caffe_net.blobs[layer.bottom[0]].shape
      pred_shape1 = self.caffe_net.blobs[layer.bottom[1]].shape
      if layer.eltwise_param.operation == 1 and np.prod(pred_shape1) < np.prod(pred_shape0):
        bottom1 = layer.bottom[1]
        layer.bottom[1] = layer.bottom[0]
        layer.bottom[0] = bottom1
        factors = list(layer.eltwise_param.factor)
        flip_factors = factors[int(len(factors)/2):] + factors[:int(len(factors)/2)] 
        layer.eltwise_param.ClearField('factor')
        layer.eltwise_param.factor.extend(flip_factors)

      pred_shape0 = self.caffe_net.blobs[layer.bottom[0]].shape
      pred_shape1 = self.caffe_net.blobs[layer.bottom[1]].shape
      assert len(pred_shape0) == len(pred_shape1)
      new_shape = self.caffe_net.blobs[layer.top[0]].shape
      param = layer.eltwise_param
      if len(param.broadcast0) > 0 or len(param.broadcast1) > 0:
        continue
      else:
        for idx, i in enumerate(pred_shape0):
          if i == 1 and new_shape[idx] != 1:
            param.broadcast0.append(1)
          else:
            param.broadcast0.append(0)
          if pred_shape1[idx] == 1 and new_shape[idx] != 1:
            param.broadcast1.append(1)
          else:
            param.broadcast1.append(0)


class EltwiseBroadcastPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    eltwises = self.get_nodes_of_type('Eltwise')
    for eltwise in eltwises:
      if eltwise.name in self.cp_only_list:
        broadcast0_v = eltwise.eltwise_param.broadcast0
        broadcast1_v = eltwise.eltwise_param.broadcast1
        out_shape = self.caffe_net.blobs[eltwise.top[0]].data.shape
        predecessors = list(eltwise.predecessors())
        for i, brodcast_v in enumerate([broadcast0_v, broadcast1_v]):
          if 1 in brodcast_v:
            for idx, p in enumerate(predecessors):
              if eltwise.bottom[i] in p.top:
                break
            pred = predecessors[idx]
            broadcast_new = self.G.mk_broadcast_proto(eltwise.bottom[i], eltwise.name, [j for j in out_shape])
            if len(pred.get_layer().adj_param.precision) and pred.get_layer().adj_param.precision[0] == 1:
              broadcast_new.adj_param.precision.append(1)
            self.G.add_node(broadcast_new)
            self.G.add_edge(pred, broadcast_new)
            self.G.add_edge(broadcast_new, eltwise)
            self.G.remove_edge(pred, eltwise)
            self.G.add_replace_edge_proto_new(broadcast_new, eltwise, eltwise.bottom[i], broadcast_new.top[0])
        for i in range(len(broadcast0_v)):
          broadcast0_v[i] = 0
          broadcast1_v[i] = 0
        self.cp_only_list.remove(eltwise.name)


class ZeroAdd(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    # TODO: Zippin
    # return
    eltwises = self.get_nodes_of_type('Eltwise')
    for eltwise in eltwises:
      predecessors = list(eltwise.predecessors())
      successors = list(eltwise.successors())
      dvtensor_idx = -1
      other_idx = 0
      if len(predecessors) == 2:
        if predecessors[0].type == 'DVTensor':
          dvtensor_idx = 0
          other_idx = 1
        elif predecessors[1].type == 'DVTensor':
          dvtensor_idx = 1
          other_idx = 0

      if dvtensor_idx == -1:
        continue

      #check possible cases of 0 addition
      def _redundant_addition():
        return (predecessors[dvtensor_idx].get_layer().dvtensor_param.tensortype == 2 and
        self.caffe_net.params[predecessors[dvtensor_idx].get_layer().name][0].data.max() == 0 and
        self.caffe_net.params[predecessors[dvtensor_idx].get_layer().name][0].data.min() == 0) or \
        (predecessors[dvtensor_idx].get_layer().dvtensor_param.tensortype == 0 and
        predecessors[dvtensor_idx].get_layer().dvtensor_param.value == 0)

      if _redundant_addition():
        self.G.remove_edge(predecessors[dvtensor_idx], eltwise)
        self.G.remove_edge(predecessors[other_idx], eltwise)
        for successor in successors:
          self.G.add_edge(predecessors[other_idx], successor)
          self.G.remove_edge(eltwise, successor)
          self.G.add_replace_edge_proto_new(predecessors[other_idx], successor, eltwise.top[0], predecessors[other_idx].top[0])
        self.G.remove_node(eltwise)
        self.G.remove_node(predecessors[dvtensor_idx])

class Eltwise_bc3bc0_to_bc0bc3(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    eltwises = self.get_nodes_of_type('Eltwise')
    for eltwise in eltwises:
      broadcast0 = eltwise.eltwise_param.broadcast0
      broadcast1 = eltwise.eltwise_param.broadcast1
      operation = eltwise.get_layer().eltwise_param.operation #0 is sum, 1 is product
      
      if (broadcast0 == [0,0,0,1,1] and broadcast1 == [0,0,0,0,0]) or (broadcast0 == [0,0,1,1] and broadcast1 == [0,0,0,0]) and operation in [0,1]:
        tmp = eltwise.bottom[0]
        eltwise.bottom[0] = eltwise.bottom[1]
        eltwise.bottom[1] = tmp
        eltwise.eltwise_param.ClearField('broadcast0')
        eltwise.eltwise_param.ClearField('broadcast1')
        eltwise.eltwise_param.broadcast0.extend(broadcast1)
        eltwise.eltwise_param.broadcast1.extend(broadcast0)
        

class BroadcastEltwisePatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    eltwises = self.get_nodes_of_type('Eltwise')
    for eltwise in eltwises:
      if eltwise.name in self.cp_only_list:
        continue
      predecessors = list(eltwise.predecessors())
      if len(predecessors) != 2:
        continue
      broadcast1 = predecessors[0] if predecessors[0].type == "Broadcast" else None
      broadcast2 = predecessors[1] if predecessors[1].type == "Broadcast" else None

      if broadcast1 is None and broadcast2 is None:
        continue

      if broadcast1 is not None:
        b1_op_shape = self.caffe_net.blobs[broadcast1.top[0]].data.shape
        b1_pred = broadcast1.predecessors()[0]
        b1_inp_shape = self.caffe_net.blobs[b1_pred.top[0]].data.shape
        broadcast0_v = eltwise.eltwise_param.broadcast0 if (list(eltwise.bottom).index(predecessors[0].top[0]) == 0) else eltwise.eltwise_param.broadcast1
        for ind, (i, o) in enumerate(zip(b1_inp_shape, b1_op_shape)):
          broadcast0_v[ind] = 1 if (o > i) else 0
        self.G.remove_edge(b1_pred, broadcast1)
        self.G.add_edge(b1_pred, eltwise)
        self.G.add_replace_edge_proto(b1_pred, eltwise, broadcast1.top[0])
        self.G.remove_node(broadcast1)

      if broadcast2 is not None:
        b2_op_shape = self.caffe_net.blobs[broadcast2.top[0]].data.shape
        b2_pred = broadcast2.predecessors()[0]
        b2_inp_shape = self.caffe_net.blobs[b2_pred.top[0]].data.shape
        broadcast1_v = eltwise.eltwise_param.broadcast0 if (list(eltwise.bottom).index(predecessors[1].top[0]) == 0) else eltwise.eltwise_param.broadcast1
        for ind, (i, o) in enumerate(zip(b2_inp_shape, b2_op_shape)):
          broadcast1_v[ind] = 1 if (o > i) else 0
        self.G.remove_edge(b2_pred, broadcast2)
        self.G.add_edge(b2_pred, eltwise)
        self.G.add_replace_edge_proto(b2_pred, eltwise, broadcast2.top[0])
        self.G.remove_node(broadcast2)


class EltwiseConstantPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    eltwises = self.get_nodes_of_type('Eltwise')
    for eltwise in eltwises:
      predecessors = list(eltwise.predecessors())
      n_const_inp = 0
      for pred in predecessors:
        if pred.type == "DVTensor":
          n_const_inp += 1
      if n_const_inp != 2:
        continue
      operation = eltwise.get_layer().eltwise_param.operation
      const1 = self.inp_wts[predecessors[0].name][0]
      const2 = self.inp_wts[predecessors[1].name][0]
      new_const = None
      if operation == 0:
        new_const = const1 * const2
      elif operation == 1:
        new_const = const1 + const2
      if new_const is None:
        continue
      new_shape = self.caffe_net.blobs[eltwise.get_layer().top[0]].data.shape
      new_const = np.reshape(new_const, new_shape)
      successors = list(eltwise.successors())
      dvtensor_node = self.G.mk_dvtensor_proto(list(new_shape), 2, 0, eltwise.name)
      self.G.add_node(dvtensor_node)
      self.inp_wts[dvtensor_node.name] = [new_const]
      for successor in successors:
        self.G.add_edge(dvtensor_node, successor)
        self.G.remove_edge(eltwise, successor)
        self.G.add_replace_edge_proto(dvtensor_node, successor, eltwise.top[0])
      for pred in predecessors:
        self.G.remove_edge(pred, eltwise)
        self.G.remove_node(pred)
        self.inp_wts.pop(pred.name)
        del self.caffe_net.params[pred.name]
      self.G.remove_node(eltwise)


class EltwiseStridePatching(BasePass):

  def __init__(self):
    self.pass_through = ['ReLU', 'Sigmoid', 'BatchNorm', 'Scale']
    self.conv_list = set()

  def is_conv1x1_s2(self, node):
    if node.type not in ['Convolution','Ara1Convolution','Ara2Convolution']:
      return False
    conv_param = node.convolution_param
    if set(conv_param.stride) == {2} and set(conv_param.kernel_size) == {1} and set(conv_param.pad) == {0}:
      return True
    return False

  def all_successors_conv1x1_s2(self, node):
    successors = list(node.successors())
    if len(successors) == 0:
      return False
    for successor in successors:
      if successor.type in self.pass_through:
        new_successors = list(successor.successors())
        if len(new_successors) == 0:
          return False
        for new_suc in new_successors:
          if not self.is_conv1x1_s2(new_suc):
            return False
          else:
            self.conv_list.add(new_suc)
            
      elif not self.is_conv1x1_s2(successor):
        return False
      else:
        self.conv_list.add(successor)
    return True        

  def do_transformation(self):
    eltwises = self.get_nodes_of_type('Eltwise')
    for eltwise in eltwises:
      self.conv_list.clear()
      if self.all_successors_conv1x1_s2(eltwise):
        eltwise.eltwise_param.stride.append(2)
        for conv in list(self.conv_list):
          conv.convolution_param.ClearField('stride')
          conv.convolution_param.stride.append(1)


class AntiBroadcastPatching(BasePass):

  def __init__(self):
    self.pass_through = ['ReLU', 'Sigmoid', 'BatchNorm', 'Scale']
    self.conv_types = ['Convolution','Ara1Convolution','Ara2Convolution']
    self.conv_list = set()

  def is_conv1x1_s2(self, node):
    if node.type not in self.conv_types:
      return False
    conv_param = node.convolution_param
    if set(conv_param.stride) == {2} and set(conv_param.kernel_size) == {1} and set(conv_param.pad) == {0}:
      return True
    return False

  def all_successors_conv1x1_s2(self, node):
    successors = list(node.successors())
    if len(successors) == 0:
      return False
    for successor in successors:
      if successor.type in self.pass_through:
        new_successors = list(successor.successors())
        if len(new_successors) == 0:
          return False
        for new_suc in new_successors:
          if not self.is_conv1x1_s2(new_suc):
            return False
          else:
            self.conv_list.add(new_suc)
            
      elif not self.is_conv1x1_s2(successor):
        return False
      else:
        self.conv_list.add(successor)
    return True     

  def predecessor_conv3x3_s1(self, node):
    predecessors = list(node.predecessors())
    pred_index_pair = dict()
    for pred in predecessors:
      tmp = pred
      while tmp.type in self.pass_through or tmp.type in self.conv_types:
        if len(tmp.predecessors()) > 1:
          break
        tmp = list(tmp.predecessors())[0]

        if tmp.type in self.conv_types:
          conv_param = tmp.convolution_param
          if set(conv_param.stride) == {1} and set(conv_param.kernel_size) == {3} and set(conv_param.pad) == {1}:
            for idx, b in enumerate(node.bottom):
              if b == pred.top[0]:
                pred_index_pair[idx] = tmp
    
    return pred_index_pair
          

  def do_transformation(self):
    eltwises = self.get_nodes_of_type('Eltwise')
    for eltwise in eltwises:
      self.conv_list.clear()
      if self.all_successors_conv1x1_s2(eltwise):
        pred_index_pair = self.predecessor_conv3x3_s1(eltwise)
        for idx in pred_index_pair:
          pred_index_pair[idx].convolution_param.ClearField('stride')
          pred_index_pair[idx].convolution_param.stride.append(2)
          eltwise.eltwise_param.factor.extend([2]*len(eltwise.bottom))
          eltwise.eltwise_param.factor[idx] = 1
          for conv in list(self.conv_list):
            conv.convolution_param.ClearField('stride')
            conv.convolution_param.stride.append(1)

class Eltwise_bc2bc0_to_bc0bc2(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    eltwises = self.get_nodes_of_type('Eltwise')
    for eltwise in eltwises:
      broadcast0 = eltwise.eltwise_param.broadcast0
      broadcast1 = eltwise.eltwise_param.broadcast1
      operation = eltwise.get_layer().eltwise_param.operation #0 is sum, 1 is product
      
      if (broadcast0 == [0,0,0,1,0] and broadcast1 == [0,0,0,0,0]) or (broadcast0 == [0,0,1,0] and broadcast1 == [0,0,0,0]) and operation in [0,1]:
        tmp = eltwise.bottom[0]
        eltwise.bottom[0] = eltwise.bottom[1]
        eltwise.bottom[1] = tmp
        eltwise.eltwise_param.ClearField('broadcast0')
        eltwise.eltwise_param.ClearField('broadcast1')
        eltwise.eltwise_param.broadcast0.extend(broadcast1)
        eltwise.eltwise_param.broadcast1.extend(broadcast0)
