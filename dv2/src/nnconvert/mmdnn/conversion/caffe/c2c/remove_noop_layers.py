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
from caffe import params as P


class RemoveNoopAvgPool(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    pools = self.get_nodes_of_type('Pooling')
    for layer in pools:
      if not (layer.pooling_param.pool == P.Pooling.AVE):
        continue
      pooling_param = layer.pooling_param
      kernel = pooling_param.kernel_size
      stride = pooling_param.stride
      pad = pooling_param.pad
      # flag = False
      # if len(kernel) == 0:
      #     flag = True
      # for i in kernel:
      #     if i != 1:
      #         flag = True
      #         break
      if len(kernel) > 0 and len(pad) > 0 and kernel == [1] * len(kernel) and pad == [0] * len(pad) and stride == [1] * len(stride):
        # this pool layer does nothing - remove it
        successors = list(layer.successors())
        # pooling layer has only one input
        predecessor = layer.predecessors()[0]
        for successor in successors:
          self.G.add_edge(predecessor, successor)
          self.G.remove_edge(layer, successor)
          self.G.add_replace_edge_proto_new(predecessor, successor, layer.top[0], predecessor.top[0])
        self.G.remove_node(layer)


class RemoveNoopSoftmax(BasePass):
    def __init__(self):
        pass

    def do_transformation(self):
        softmaxes = self.get_nodes_of_type('Softmax')

        for layer in softmaxes:

            if len(layer.successors())==1 and layer.successors()[0].get_layer().type == 'ArgMax':
                successors = layer.successors()
                predecessor = layer.predecessors()[0]
                for successor in successors:
                    self.G.add_edge(predecessor, successor)
                    self.G.remove_edge(layer, successor)
                    self.G.add_replace_edge_proto(predecessor,successor,layer.top[0])

                self.G.remove_node(layer)

class RemoveNoopPool(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    pools = self.get_nodes_of_type('Pooling')
    for layer in pools:
      if not (layer.pooling_param.pool in [P.Pooling.AVE, P.Pooling.MAX, P.Pooling.SUMPOOL]):
        continue
      pooling_param = layer.pooling_param
      kernel = pooling_param.kernel_size
      pad = pooling_param.pad
      stride = pooling_param.stride
      # flag = False
      # if kernel == [1]*len(kernel) and pad == [0]*len(pad):
      #     flag = True
      #     break
      if len(kernel) > 0 and len(pad) > 0 and kernel == [1] * len(kernel) and pad == [0] * len(pad) and stride == [1] * len(stride):
        # print("Pool Layer removed: ", layer.name)
        # this pool layer does nothing - remove it
        successors = list(layer.successors())
        # pooling layer has only one input
        predecessor = layer.predecessors()[0]
        for successor in successors:
          self.G.add_edge(predecessor, successor)
          self.G.remove_edge(layer, successor)
          self.G.add_replace_edge_proto_new(predecessor, successor, layer.top[0], predecessor.top[0])
        self.G.remove_node(layer)


class RemoveNoopPad(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    pads = self.get_nodes_of_type('Padding')
    for layer in pads:
      padding_param = layer.padding_param
      total = 0
      if len(padding_param.pad) > 0:
        pad_l = padding_param.pad[1]
        pad_r = padding_param.pad[3]
        pad_t = padding_param.pad[0]
        pad_b = padding_param.pad[2]
        total = sum(padding_param.pad)
      else:
        pad_l = padding_param.pad_l
        pad_r = padding_param.pad_r
        pad_t = padding_param.pad_t
        pad_b = padding_param.pad_b
        total = pad_l + pad_r + pad_t + pad_b
      if not (total == 0):
        continue
      predecessor = layer.predecessors()[0]
      for successor in list(layer.successors()):
        self.G.add_edge(predecessor, successor)
        self.G.remove_edge(layer, successor)
        self.G.add_replace_edge_proto_new(predecessor, successor, layer.name, predecessor.top[0])
      self.G.remove_node(layer)


class RemoveNoopCrop(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    crops = self.get_nodes_of_type('Crop')
    for crop in crops:
      successors = list(crop.successors())
      predecessors = list(crop.predecessors())
      crop_input_layer = predecessors[0]
      crop_shape_layer = predecessors[1]
      if crop_shape_layer.type != 'DummyData' and crop_input_layer.type == 'DummyData':
        temp = crop_shape_layer
        crop_shape_layer = crop_input_layer
        crop_input_layer = temp
      crop_input_shape = self.caffe_net.blobs[crop_input_layer.top[0]].shape
      crop_shape = crop_shape_layer.dummy_data_param.shape[0].dim
      count = 0
      for offset in crop.crop_param.offset:
        if offset == 0:
          count += 1
      flag = False
      for idx in range(len(crop_shape)):
        if crop_shape[idx] != crop_input_shape[idx]:
          flag = True
          break
      if flag or (count != len(crop.crop_param.offset)):
        continue
      for successor in successors:
        self.G.add_edge(crop_input_layer, successor)
        self.G.add_replace_edge_proto_new(crop_input_layer, successor, crop.top[0], crop_input_layer.top[0])
      self.G.remove_node(crop)
      self.G.remove_node(crop_shape_layer)


class RemoveNoopReshape(BasePass):

  def __init__(self):
    pass

  """
  following func is a pure hack in order to support networks 
  having reshape as the last layer which shouldn't be removed
  since it preserves the output name and shape property.
  e.g. consider incb-36, yolov7-tiny-det. 
  if softmax and normalize are there, we can't do bcz precision
  which percolates to the next layer is fp32 and int32
  """

  def _is_valid_case_for_removal(self, layer):
    successors = list(layer.successors())
    predecessors = list(layer.predecessors())
    if len(successors) > 0:
      return True
    elif len(successors) == 0:
      if len(predecessors) == 1:
        layertype = predecessors[0].type
        if(layertype == "ArgMax" and layer.argmax_param.out_max_val == True) or layertype == "DetectionOutput":
          return True
    return False

  def do_transformation(self):
    reshapes = self.get_nodes_of_type('Reshape')
    for layer in reshapes:
      successors = list(layer.successors())
      predecessors = list(layer.predecessors())
      if not self._is_valid_case_for_removal(layer):
        continue
      predecessor = predecessors[0]
      predecessor_idx = 0
      for idx, predecessor_top in enumerate(predecessor.top):
        if predecessor_top == layer.bottom[0]:
          predecessor_idx = idx
          break
      reshape_input_shape = self.caffe_net.blobs[predecessor.top[predecessor_idx]].data.shape
      reshape_output_shape = self.caffe_net.blobs[layer.top[0]].data.shape
      # If the layer is an output node, pass isOutput flag to previous layer
      if layer.converter_param.isOutput:
        predecessor.converter_param.isOutput = True
        layer.converter_param.isOutput = False
      if len(successors) == 0:
        self.G.remove_edge(predecessor, layer)
        self.G.remove_node(layer)
        continue
      if reshape_input_shape == reshape_output_shape:
        successors = list(layer.successors())
        for successor in successors:
          self.G.add_edge(predecessor, successor)
          self.G.remove_edge(layer, successor)
          self.G.add_replace_edge_proto_new(predecessor, successor, layer.name, predecessor.top[predecessor_idx])
        self.G.remove_node(layer)
