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
import caffe
from caffe import layers as L
from caffe import params as P
from mmdnn.conversion.caffe.c2c.common import BasePass
from mmdnn.conversion.caffe.utils import get_axis
from dv_logging import get_logger

logger = get_logger(__name__, 'dvctoc')
sfmx_type_list = ['Softmax', 'Tf_Softmax', 'Mxnet_Softmax']
sigm_type_list = ['Sigmoid', 'Tf_Sigmoid', 'Mxnet_Sigmoid']
conf_type_list = sfmx_type_list + sigm_type_list


class BatchPermutationPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    concats = self.get_nodes_of_type('Concat')
    for concat in concats:
      predecessors = list(concat.predecessors())
      successors = list(concat.successors())
      if (len(successors)) == 1 and successors[0].type == 'Onnx_Batchpermutation':
        for predecessor in predecessors:
          self.G.remove_edge(predecessor, concat)
          self.G.add_edge(predecessor, successors[0])
          for top in predecessor.top:
            if top in concat.bottom:
              successors[0].bottom.append(top)
        self.G.remove_edge(concat, successors[0])
        self.G.remove_node(concat)
        successors[0].bottom.remove(concat.top[0])
        successors[0].onnx_batchpermutation_param.axis = concat.concat_param.axis


class ConcatPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    concats = self.get_nodes_of_type('Concat') + self.get_nodes_of_type('Tflite_Concat') + self.get_nodes_of_type('Pqn_Concat') 
    for concat in concats:
      predecessors = list(concat.predecessors())
      successors = list(concat.successors())
      patched = True
      for pred in predecessors:
        if pred.type != 'DVShift':
          patched = False
      if patched:
        continue
      # Both the inputs to concat can be from same node which can be seen in concat._layer.bottom
      if len(predecessors) == 1 and len(concat.get_layer().bottom) == 1:
        for suc in successors:
          self.G.remove_edge(concat, suc)
          self.G.add_edge(predecessors[0], suc)
          for top in predecessors[0].top:
            if top in concat.bottom:
              index = list(suc.bottom).index(concat.top[0])
              suc.bottom[index] = top
        self.G.remove_edge(predecessors[0], concat)
        self.G.remove_node(concat)
        continue
      pred_types = list()
      for predecessor in predecessors:
        pred_types.append(predecessor.type)
      if len(set(pred_types)) == 1 and pred_types[0] in conf_type_list:
        logger.info("found conf + concat pattern")
        copy_conf = copy.deepcopy(predecessors[0]._layer)
        copy_conf.name += '_c2c_softmax_flip'
        copy_conf.top[0] = copy_conf.name
        copy_conf.bottom[0] = concat.top[0]
        i = 0
        pred_successors_count = 0
        for predecessor in predecessors:
          pred_successors_count = max(pred_successors_count, len(predecessor.successors()))
        if pred_successors_count > 1:
          continue
        for predecessor in predecessors:
          pred_predecessor = list(predecessor.predecessors())
          self.G.remove_node(predecessor)
          self.G.add_edge(pred_predecessor[0], concat)
          concat.bottom[i] = pred_predecessor[0].top[0]
          i += 1
        successors = list(concat.successors())
        self.G.add_node(copy_conf)
        self.G.add_edge(concat, copy_conf)
        for successor in successors:
          self.G.remove_edge(concat, successor)
          self.G.add_edge(copy_conf, successor)
          for ind in range(len(successor.bottom)):
            if successor.bottom[ind] == concat.top[0]:
              successor.bottom[ind] = copy_conf.top[0]
      if len(set(pred_types)) == 1 and pred_types[0] == 'Convolution':
        successors = list(concat.successors())
        if len(successors) == 1 and successors[0].type == 'ReLU':
          relu_successors = list(successors[0].successors())
          relu_name = successors[0].top[0]
          count = 0
          for predecessor in predecessors:
            consumer_copy = copy.deepcopy(successors[0]._layer)
            consumer_copy.name += '_' + str(count)
            consumer_copy.bottom[0] = predecessor.top[0]
            consumer_copy.top[0] = consumer_copy.name
            concat.bottom[count] = consumer_copy.top[0]
            self.G.add_node(consumer_copy)
            self.G.remove_edge(predecessor, concat)
            self.G.add_edge(predecessor, consumer_copy)
            self.G.add_edge(consumer_copy, concat)
            count += 1
          self.G.remove_node(successors[0])
          for successor in relu_successors:
            self.G.add_edge(concat, successor)
            for ind in range(len(successor.bottom)):
              if successor.bottom[ind] == relu_name:
                successor.bottom[ind] = concat.top[0]
      predecessors = list(concat.predecessors())
      idx = 0
      for predecessor in predecessors:
        if predecessor.get_layer().type == 'DVShift':
          continue
        for i in concat.bottom:
          pj = 0
          for j in predecessor.top:
            if i == j:
              bottom = i
              dvshift_layer = self.G.mk_dvshift_proto(bottom, concat.name)
              if len(concat.get_layer().adj_param.precision) and concat.get_layer().adj_param.precision[0] == 1:
                dvshift_layer.adj_param.precision.append(1)
              self.G.add_node(dvshift_layer)
              self.G.add_edge(predecessor, dvshift_layer)
              self.G.add_edge(dvshift_layer, concat)
              self.G.add_replace_edge_proto_new(dvshift_layer, concat, bottom,dvshift_layer.top[0])
            pj += 1
        self.G.remove_edge(predecessor, concat)
        idx += 1


class ConcatAxis0To1Patching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    concats = self.get_nodes_of_type('Concat') + self.get_nodes_of_type('Tflite_Concat') + self.get_nodes_of_type('Pqn_Concat')
    for concat in concats:
      if get_axis(concat.concat_param) == 0:
        predecessors = list(concat.predecessors())
        patched = False
        for pred in predecessors:
          if pred.type == 'LSTM' or pred.type == 'DVShift':
            patched = True
        if patched:
          continue
        successors = list(concat.successors())
        rank = len(self.caffe_net.blobs[concat.bottom[0]].data.shape)
        if rank <= 1:
          for p in predecessors:
            for t in p.top:
              if t in concat.bottom:
                p_shape_new = list(self.caffe_net.blobs[t].data.shape)
                p_shape_new.insert(0, 1)
                reshape = self.G.mk_reshape_proto(t, p_shape_new, concat.name)
                self.G.add_node(reshape)
                self.G.add_edge(p, reshape)
                self.G.add_edge(reshape, concat)
                self.G.add_replace_edge_proto_new(reshape, concat, t, reshape.top[0])
            if all([i not in concat.bottom for i in p.top]):
              self.G.remove_edge(p, concat)
          concat.concat_param.axis = 1
          concat_shape = list(self.caffe_net.blobs[concat.top[0]].data.shape)
          reshape = self.G.mk_reshape_proto(concat.name, concat_shape, concat.name)
          self.G.add_node(reshape)
          self.G.add_edge(concat, reshape)
          for s in successors:
            self.G.add_edge(reshape, s)
            self.G.remove_edge(concat, s)
            self.G.add_replace_edge_proto_new(reshape, s, concat.top[0], reshape.top[0])
        else:
          order = [1, 0] + [i for i in range(2, rank)]
          for p in predecessors:
            for t in p.top:
              if t in concat.bottom:
                perm = self.G.mk_permute_proto(concat.name, order, t)
                self.G.add_node(perm)
                self.G.add_edge(p, perm)
                self.G.add_edge(perm, concat)
                self.G.add_replace_edge_proto_new(perm, concat, t, perm.top[0])
            if all([i not in concat.bottom for i in p.top]):
              self.G.remove_edge(p, concat)
          concat.concat_param.axis = 1
          perm = self.G.mk_permute_proto(concat.name, order)
          self.G.add_node(perm)
          self.G.add_edge(concat, perm)
          for s in successors:
            self.G.add_edge(perm, s)
            self.G.remove_edge(concat, s)
            self.G.add_replace_edge_proto_new(perm, s, concat.top[0], perm.top[0])
