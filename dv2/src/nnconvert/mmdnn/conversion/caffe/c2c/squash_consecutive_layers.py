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
from mmdnn.conversion.caffe.c2c.common import BasePass
from caffe import layers as L
from caffe import params as P
from dv_logging import get_logger
import numpy as np
logger = get_logger(__name__, 'DVCTOC')


class SquashConsecutivePads(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    pads = self.get_nodes_of_type('Padding')
    removed_pads = []
    for layer in pads:
      if layer.name in removed_pads:
        continue
      successors = list(layer.successors())
      if not len(successors) or len(successors) > 1:
        continue
      successor = successors[0]
      if successor.type != 'Padding':
        continue
      # layer.padding_param.pad_l += \
      #     successor.padding_param.pad_l
      # layer.padding_param.pad_r += \
      #     successor.padding_param.pad_r
      # layer.padding_param.pad_t += \
      #     successor.padding_param.pad_t
      # layer.padding_param.pad_b += \
      #     successor.padding_param.pad_b
      for idx, pad_val in enumerate(layer.padding_param.pad):
        layer.padding_param.pad[idx] += successor.padding_param.pad[idx]
      for sucessor_sucessor in list(successor.successors()):
        self.G.remove_edge(successor, sucessor_sucessor)
        self.G.add_edge(layer, sucessor_sucessor)
        self.G.add_replace_edge_proto_new(layer, sucessor_sucessor, successor.name, layer.top[0])
      self.G.remove_node(successor)
      removed_pads.append(successor.name)


class SquashConsecutiveCrops(BasePass):

  def __init__(self):
    pass

  def _can_merge_crop(self, crop, child_crop, crop_shape, child_crop_shape):
    crop_shape =\
        list(map(lambda x: x.dim, crop_shape.dummy_data_param.shape))[0]
    child_crop_shape =\
        list(map(lambda x: x.dim, child_crop_shape.dummy_data_param.shape))[0]
    if crop.crop_param.axis == child_crop.crop_param.axis == 2:
      crop_offset = crop.crop_param.offset
      child_crop_offset = child_crop.crop_param.offset
      if len(crop_offset) == 1:
        crop_offset.append(crop_offset[0])
      if len(child_crop_offset) == 1:
        child_crop_offset.append(child_crop_offset[0])
      offsets = list(zip(crop.crop_param.offset, child_crop.crop_param.offset))
      offsets = list(map(sum, offsets))
      crop_shape = list(zip(crop_shape, child_crop_shape))
      crop_shape = list(map(min, crop_shape))
      return True, offsets, crop_shape
    else:
      return False, None, None

  def do_transformation(self):
    crops = self.get_nodes_of_type('Crop')
    for crop in crops:
      successors = list(crop.successors())
      predecessors = list(crop.predecessors())
      # TODO: replace this with a pattern match for nodes that
      # have Crop -> Crop
      if not (len(successors) == 1 and successors[0].type == 'Crop'):
        continue
      crop_input_layer = predecessors[0]
      crop_shape_layer = predecessors[1]
      crop_consumer = successors[0]
      # crop_consumer_shape = crop_consumer.predecessors()[0]
      if crop_consumer.predecessors()[0] == crop:
        crop_consumer_shape = crop_consumer.predecessors()[1]
      else:
        crop_consumer_shape = crop_consumer.predecessors()[0]
      crop_consumer_successor_all = list(crop_consumer.successors())
      if len(crop_consumer_successor_all) > 0:
        crop_consumer_successor = crop_consumer_successor_all[0]
      else:
        crop_consumer_successor = None
      can_merge, offsets, crop_shape =\
          self._can_merge_crop(crop, crop_consumer,
                               crop_shape_layer, crop_consumer_shape)
      if can_merge:
        merged_crop_shape, merged_crop_layer = self.G.mk_crop_proto(crop.name, crop_shape, offsets)
        self.G.add_node(merged_crop_shape)
        self.G.add_node(merged_crop_layer)
        self.G.remove_edge(crop_input_layer, crop)
        self.G.remove_edge(crop_shape_layer, crop)
        self.G.remove_edge(crop, crop_consumer)
        self.G.remove_edge(crop_consumer_shape, crop_consumer)
        self.G.add_edge(crop_input_layer, merged_crop_layer)
        self.G.add_edge(merged_crop_shape, merged_crop_layer)
        if crop_consumer_successor is not None:
          for crop_consumer_successor in crop_consumer_successor_all:
            self.G.remove_edge(crop_consumer, crop_consumer_successor)
            self.G.add_edge(merged_crop_layer, crop_consumer_successor)
            self.G.add_replace_edge_proto_new(merged_crop_layer, crop_consumer_successor, crop_consumer.name, merged_crop_layer.top[0])
        self.G.add_replace_edge_proto_new(crop_input_layer, merged_crop_layer, "NA", crop_input_layer.top[0])
        self.G.add_replace_edge_proto_new(merged_crop_shape, merged_crop_layer, "NA", merged_crop_shape.top[0])
        self.G.remove_node(crop)
        self.G.remove_node(crop_shape_layer)
        self.G.remove_node(crop_consumer)
        self.G.remove_node(crop_consumer_shape)
        crops.remove(crop_consumer)


class SquashConsecutiveReshapes(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    reshapes = self.get_nodes_of_type('Reshape')
    topsort_nodes = self.get_topsort()
    for layer in reshapes:
      successors = list(layer.successors())
      if len(successors) > 1:
        logger.debug('there are more than 1 tops')
        continue
      if len(successors) == 0 and layer.name.endswith("_splitOutput"):
        predecessors = list(layer.predecessors())
        if len(predecessors) > 0:
          for predecessor in predecessors:
            predecessor.converter_param.isOutput = True
        self.G.remove_node(layer)
        continue
      if len(successors) == 0:
        continue
      successor = successors[0]
      predecessors = list(layer.predecessors())
      if successor.type == 'Reshape':
        for predecessor in predecessors:
          # If the layer is an output layer, fork a branch from its predecessor
          if layer.converter_param.isOutput:
            self.G.remove_edge(layer, successor)
            self.G.add_edge(predecessor, successor)
            self.G.add_replace_edge_proto_new(predecessor, successor, layer.top[0], layer.bottom[0])
          else:
            self.G.add_edge(predecessor, successor)
            self.G.remove_node(layer)
            self.G.add_replace_edge_proto_new(predecessor, successor, layer.top[0], layer.bottom[0])

class SquashConsecutiveBatchNorms(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    batchnorms = self.get_nodes_of_type("BatchNorm")
    successive_batchnorms = {}
    for layer in batchnorms:
      successors = list(layer.successors())
      if len(successors) != 1:
        continue
      successor = successors[0]
      if successor.type == "Scale":
        scale_successors = list(successor.successors())
        if len(scale_successors) != 1:
          continue
        node = scale_successors[0]
        if node.type == "BatchNorm":
          node_successors = list(node.successors())
          if len(node_successors) != 1:
            continue
          if node_successors[0].type == "Scale":
            successive_batchnorms[layer] = node

    batchnorm_layers = list(successive_batchnorms.keys())

    for cur_layer in batchnorm_layers:
      if cur_layer not in successive_batchnorms.keys():
        continue
      next_layer = successive_batchnorms[cur_layer]
      cur_layer_succ = cur_layer.successors()[0]
      next_layer_succ = next_layer.successors()[0]
      cur_layer_wts = self.inp_wts[cur_layer.name]
      next_layer_wts = self.inp_wts[next_layer.name]
      cur_layer_succ_wts = self.inp_wts[cur_layer_succ.name]
      next_layer_succ_wts = self.inp_wts[next_layer_succ.name]
      # General Case
      # ((((((X - M) / sqrt(V+e)) * S + B) - M1) / sqrt(V1+e1)) * S1 + B1)
      # (((X * VS - MVS + B) - M1) * V1) * S1 + B1
      # Y * V1S1 - M1V1S1 + B1
      # (X * V S V1 S1) - M V S V1 S1 + B V1 S1 - M1 V1 S1 + B1
      M = cur_layer_wts[0].copy()
      V = pow(cur_layer_wts[1].copy() + cur_layer.batch_norm_param.eps, -0.5)
      S = cur_layer_succ_wts[0].copy()
      B = cur_layer_succ_wts[1].copy()
      M1 = next_layer_wts[0].copy()
      V1 = pow(next_layer_wts[1].copy() + next_layer.batch_norm_param.eps, -0.5)
      S1 = next_layer_succ_wts[0].copy()
      B1 = next_layer_succ_wts[1].copy()

      cur_layer_wts[0] = np.zeros(cur_layer_wts[0].shape, cur_layer_wts[0].dtype)
      cur_layer_wts[1] = np.ones(cur_layer_wts[1].shape, cur_layer_wts[1].dtype)
      cur_layer_succ_wts[0] = (V * S * V1 * S1)
      cur_layer_succ_wts[1] = (-1 * M * V * S * V1 * S1 + B * V1 * S1 - M1 * V1 * S1 + B1)
      # Remove the next_layer and next_layer_succ
      succ = list(next_layer_succ.successors())
      for s in succ:
        self.G.add_edge(cur_layer_succ, s)
        self.G.remove_edge(next_layer_succ, s)
        self.G.add_replace_edge_proto(cur_layer_succ, s, next_layer_succ.top[0])
      self.G.remove_node(next_layer_succ)
      self.G.remove_node(next_layer)
      if next_layer in batchnorm_layers:
        del successive_batchnorms[next_layer]

      # if np.all(cur_layer_wts[0] == 0) and np.all(cur_layer_wts[1] == 1) and np.all(next_layer_succ_wts[0] == 1) and np.all(
      #     next_layer_succ_wts[1] == 0):
      #   # Mul and Add/Sub Pattern internally. Fuse to single batchnorm
      #   # ((X - 0) * 1) * S
      #   cur_layer_succ_wts[1] = (next_layer_wts[0] * -1).copy()
      #   # Remove the next_layer and next_layer_succ
      #   succ = list(next_layer_succ.successors())
      #   for s in succ:
      #     self.G.add_edge(cur_layer_succ, s)
      #     self.G.remove_edge(next_layer_succ, s)
      #     self.G.add_replace_edge_proto(cur_layer_succ, s, next_layer_succ.top[0])
      #   logger.info("Removing " + next_layer_succ.name)
      #   self.G.remove_node(next_layer_succ)
      #   logger.info("Removing " + next_layer.name)
      #   self.G.remove_node(next_layer)
      #   if next_layer in batchnorm_layers:
      #     del successive_batchnorms[next_layer]
      # else:
      #   del successive_batchnorms[cur_layer]
