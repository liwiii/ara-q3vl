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
from dv_logging import get_logger

logger = get_logger(__name__, 'dvctoc')


class ReshapePatching(BasePass):

  global_pass = 0

  def __init__(self):
    ReshapePatching.global_pass += 1

  def do_transformation(self):
    reshapes = self.get_nodes_of_type('Reshape')
    for layer in reshapes:
      children = list(layer.successors())
      parents = list(layer.predecessors())
      if len(children) < 2:
        continue
      # logger.info("Removing " + layer.top[0])
      # import pdb
      # pdb.set_trace()
      self.G.remove_node(layer)
      for i in range(len(children)):
        reshape = self.G.mk_reshape_proto(parents[0].top[0], list(layer.reshape_param.shape.dim), layer.name)
        self.G.add_node(reshape)
        self.G.add_edge(parents[0], reshape)
        self.G.add_edge(reshape, children[i])
        # logger.info("Adding edge from " + parents[0].top[0] + " to " + reshape.top[0])
        # logger.info("Adding edge from " + reshape.top[0] + " to " + children[i].top[0])
        for ind in range(len(children[i].bottom)):
          if children[i].bottom[ind] == layer.top[0]:
            children[i].bottom[ind] = reshape.top[0]

class MoveReshapeBatchDim(BasePass):

  BINARY_OPS = {'BatchDot', 'Eltwise'}
  BATCH_SENSITIVE_OPS = {'Permute', 'PermuteNnp'}

  def __init__(self):
    pass

  def _is_reshape_eligible(self, reshape_node):
    """Check if a reshape node has shape (N,1,H,W) with N!=1, making it a swap candidate."""
    shape = list(self.caffe_net.blobs[reshape_node.top[0]].data.shape)
    return (
        len(shape) == 4
        and shape[0] != 1
        and shape[1] == 1
    )

  def _is_safe_to_transform(self, reshape_node):
    """
    Returns True if swapping dim[0] and dim[1] on this reshape is safe.
    Checks all downstream consumers recursively for problematic ops.
    """
    for node in reshape_node.successors():
        op_type = node.type

        if op_type in self.BATCH_SENSITIVE_OPS:
            return False
        
        if op_type in self.BINARY_OPS:
            for other_input in node.predecessors():
                if other_input == reshape_node:
                    continue
                if other_input.type != 'Reshape' or not self._is_reshape_eligible(other_input):
                    return False

    return True
  
  def do_transformation(self):
    reshapes = self.get_nodes_of_type('Reshape')
    for reshape in reshapes:
      if self._is_reshape_eligible(reshape) and self._is_safe_to_transform(reshape) :
        reshape_shape = list(self.caffe_net.blobs[reshape.top[0]].data.shape)
        reshape_shape[0], reshape_shape[1] = reshape_shape[1], reshape_shape[0]
        dims = reshape.reshape_param.shape.dim
        dims.clear()
        dims.extend(reshape_shape)

class FuseReshape(BasePass):

  def __init__(self):
    pass

  """
  following func is a pure hack in order to support networks 
  having reshape as the last layer which shouldn't be removed
  since it preserves the output name and shape property.
  e.g. consider incb-36, yolov7-tiny-det.
  if argmax and dout are there, we can't do bcz of 
  structured output emission.
  """

  def _is_valid_case_for_removal(self, layer):
    successors = list(layer.successors())
    predecessors = list(layer.predecessors())
    if len(successors) > 0:
      return True
    elif len(successors) == 0:
      if len(predecessors) == 1:
        layertype = predecessors[0].type
        if (layertype == "ArgMax" and layer.argmax_param.out_max_val == True) or layertype == "DetectionOutput":
          return True
    return False

  def do_transformation(self):
    reshapes = self.get_nodes_of_type('Reshape')
    idx = 0
    while idx < len(reshapes):
      try:
        layer = reshapes[idx]
        #Reshape layer has only one input
        parent = layer.predecessors()[0]
        children = list(layer.successors())
        remove_node = True
        increment = True
        if not self._is_valid_case_for_removal(layer):
          idx += 1
          continue
        for child in children:
          if child.type == 'Reshape':
            self.G.remove_edge(layer, child)
            self.G.add_edge(parent, child)
            # self.G.add_replace_edge_proto(parent, child, layer.name)
            child.bottom[:] = layer.bottom
            increment = False
          else:
            remove_node = False
        if layer.converter_param.isOutput:
          remove_node = False
        if remove_node:
          self.G.remove_edge(parent, layer)
          self.G.remove_node(layer)
          increment = True
        if increment:
          idx += 1
      except Exception as e:
        idx += 1

class DynamicReshape(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    reshapes = self.get_nodes_of_type('Reshape')
    dynamic_layers = ['TopK', 'NMSRtmdet', 'NMSPose']
    for node in reshapes:
      parent_node = node.predecessors()[0]
      if parent_node.type in dynamic_layers:
        for idx, blob_name in enumerate(parent_node.top):
          if blob_name == node.bottom[0]:
            parent_shape = self.caffe_net.blobs[parent_node.top[idx]].data.shape
            node.reshape_param.ClearField('shape')
            node.reshape_param.shape.dim.extend(parent_shape)
            break
