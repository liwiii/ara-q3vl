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
import random
import numpy as np
from dv_logging import get_logger

logger = get_logger(__name__, 'DVCTOC')


class HtoWforReduction(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    reduction_layers = self.get_nodes_of_type('Reduction')
    for layer in reduction_layers:
      if not getattr(getattr(layer, 'reduction_param'), 'keep_dims'):  # ss_rw need to revisit
        continue
      layer_shape = list(self.caffe_net.blobs[layer.top[0]].data.shape)
      axes = getattr(getattr(layer, 'reduction_param'), 'axis')
      hw_axes = None
      if len(layer_shape) in [1, 2, 3, 4]:
        hw_axes = [2, 3]
      elif len(layer_shape) in [5]:
        hw_axes = [3, 4]
      assert hw_axes is not None
      if any([i in axes for i in hw_axes]):
        continue
      # todo for rank 4 : n c h w -> n 1 c hw
      if len(layer_shape) == 3:
        bottom_layer = layer.predecessors()[0]
        bottom_shape = self.caffe_net.blobs[layer.bottom[0]].data.shape
        bottom_shape_new = list(bottom_shape)
        bottom_shape_new.insert(1, 1)
        for i, j in enumerate(axes):
          if j == 1:
            axes[i] = 2
        reshape_0 = self.G.mk_reshape_proto(layer.bottom[0], bottom_shape_new, layer.name)
        self.G.add_node(reshape_0)
        self.G.add_edge(bottom_layer, reshape_0)
        self.G.add_edge(reshape_0, layer)
        self.G.remove_edge(bottom_layer, layer)
        self.G.add_replace_edge_proto_new(reshape_0, layer, bottom_layer.top[0], reshape_0.top[0])

        top_layers = list(layer.successors())
        reshape_1 = self.G.mk_reshape_proto(layer.top[0], layer_shape, layer.name)
        self.G.add_node(reshape_1)
        self.G.add_edge(layer, reshape_1)
        for t in top_layers:
          self.G.add_edge(reshape_1, t)
          self.G.remove_edge(layer, t)
          self.G.add_replace_edge_proto_new(reshape_1, t, layer.top[0], reshape_1.top[0])


class HtoWforL1Norm(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    l1norm_layers = self.get_nodes_of_type('L1Norm')
    for layer in l1norm_layers:
      layer_shape = list(self.caffe_net.blobs[layer.top[0]].data.shape)
      axes = getattr(getattr(layer, 'l1norm_param'), 'axis')
      hw_axes = None
      if len(layer_shape) in [1, 2, 3, 4]:
        hw_axes = [2, 3]
      elif len(layer_shape) in [5]:
        hw_axes = [3, 4]
      if axes in hw_axes:
        continue

      bottom_layer = layer.predecessors()[0]
      bottom_shape = self.caffe_net.blobs[layer.bottom[0]].data.shape
      bottom_shape_new = list(bottom_shape)
      if len(layer_shape) == 3:
        bottom_shape_new.insert(2, 1)
      elif len(layer_shape) == 4:
        if layer_shape[3] != 1:
          continue
        bottom_shape_new[3] = layer_shape[2]
        bottom_shape_new[2] = 1
      elif len(layer_shape) == 5:
        if layer_shape[4] != 1:
          continue
        bottom_shape_new[4] = layer_shape[3]
        bottom_shape_new[3] = 1
      else:
        continue

      reshape_0 = self.G.mk_reshape_proto(layer.bottom[0], bottom_shape_new, layer.name)
      self.G.add_node(reshape_0)
      self.G.add_edge(bottom_layer, reshape_0)
      self.G.add_edge(reshape_0, layer)
      self.G.remove_edge(bottom_layer, layer)
      self.G.add_replace_edge_proto_new(reshape_0, layer, bottom_layer.top[0], reshape_0.top[0])

      top_layers = list(layer.successors())
      reshape_1 = self.G.mk_reshape_proto(layer.top[0], layer_shape, layer.name)
      self.G.add_node(reshape_1)
      self.G.add_edge(layer, reshape_1)
      for t in top_layers:
        self.G.add_edge(reshape_1, t)
        self.G.remove_edge(layer, t)
        self.G.add_replace_edge_proto_new(reshape_1, t, layer.top[0], reshape_1.top[0])


class HtoWforScaleNorm(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    l2norm_layers = self.get_nodes_of_type('Normalize')
    for layer in l2norm_layers:
      layer_shape = list(self.caffe_net.blobs[layer.top[0]].data.shape)
      spatial_dim = getattr(getattr(layer, 'norm_param'), 'across_spatial')
      if not spatial_dim:
        continue
      bottom_layer = layer.predecessors()[0]
      bottom_shape = self.caffe_net.blobs[bottom_layer.top[0]].data.shape
      bottom_shape_new = list(bottom_shape)
      if len(layer_shape) == 3:
        bottom_shape_new.insert(2, 1)
      elif len(layer_shape) == 4:
        if layer_shape[3] != 1:
          continue
        bottom_shape_new[3] = layer_shape[2]
        bottom_shape_new[2] = 1
      elif len(layer_shape) == 5:
        if layer_shape[4] != 1:
          continue
        bottom_shape_new[4] = layer_shape[3]
        bottom_shape_new[3] = 1
      else:
        continue
      reshape_0 = self.G.mk_reshape_proto(bottom_layer.name, bottom_shape_new, bottom_layer.name)
      self.G.add_node(reshape_0)
      self.G.add_edge(bottom_layer, reshape_0)
      self.G.add_edge(reshape_0, layer)
      self.G.remove_edge(bottom_layer, layer)
      self.G.add_replace_edge_proto(reshape_0, layer, bottom_layer.top[0])
      top_layers = layer.successors()
      reshape_1 = self.G.mk_reshape_proto(layer.name, layer_shape, layer.name)
      self.G.add_node(reshape_1)
      self.G.add_edge(layer, reshape_1)
      for t in top_layers:
        self.G.add_edge(reshape_1, t)
        self.G.remove_edge(layer, t)
        self.G.add_replace_edge_proto(reshape_1, t, layer.top[0])


class HtoWforLayerNorm(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    l1norm_layers = self.get_nodes_of_type('LayerNorm')
    for layer in l1norm_layers:
      layer_shape = list(self.caffe_net.blobs[layer.top[0]].data.shape)
      axes = getattr(getattr(layer, 'layer_norm_param'), 'axis')
      if len(layer_shape) in [1, 2, 3, 4]:
        hw_axes = [2, 3]
      elif len(layer_shape) in [5]:
        hw_axes = [3, 4]
      # if axes in hw_axes:
      #     continue

      bottom_layer = layer.predecessors()[0]
      bottom_shape = self.caffe_net.blobs[bottom_layer.top[0]].data.shape
      bottom_shape_new = list(bottom_shape)
      if len(layer_shape) == 3:
        bottom_shape_new.insert(2, 1)
        if layer.layer_norm_param.axis == 2:
          layer.layer_norm_param.axis = 3
      elif len(layer_shape) == 4:
        if layer_shape[3] != 1:
          continue
        bottom_shape_new[3] = layer_shape[2]
        layer.layer_norm_param.axis = 3
        bottom_shape_new[2] = 1
      elif len(layer_shape) == 5:
        if layer_shape[4] != 1:
          continue
        bottom_shape_new[4] = layer_shape[3]
        bottom_shape_new[3] = 1
        layer.layer_norm_param.axis = 4
      else:
        continue

      reshape_0 = self.G.mk_reshape_proto(bottom_layer.name, bottom_shape_new, layer.name)
      self.G.add_node(reshape_0)
      self.G.add_edge(bottom_layer, reshape_0)
      self.G.add_edge(reshape_0, layer)
      self.G.remove_edge(bottom_layer, layer)
      self.G.add_replace_edge_proto_new(reshape_0, layer, bottom_layer.top[0], reshape_0.top[0])

      top_layers = list(layer.successors())
      reshape_1 = self.G.mk_reshape_proto(layer.name, layer_shape, layer.name)
      self.G.add_node(reshape_1)
      self.G.add_edge(layer, reshape_1)
      for t in top_layers:
        self.G.add_edge(reshape_1, t)
        self.G.remove_edge(layer, t)
        self.G.add_replace_edge_proto_new(reshape_1, t, layer.top[0], reshape_1.top[0])


class HtoWforGroupNorm(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    l1norm_layers = self.get_nodes_of_type('GroupNorm')
    for layer in l1norm_layers:
      layer_shape = list(self.caffe_net.blobs[layer.top[0]].data.shape)
      # axes = getattr(getattr(layer,'group_norm_param'),'axis')
      # removing unused code
      # if len(layer_shape) in [1, 2, 3, 4]:
      #   hw_axes = [2, 3]
      # elif len(layer_shape) in [5]:
      #   hw_axes = [3, 4]

      bottom_layer = layer.predecessors()[0]
      bottom_shape = self.caffe_net.blobs[bottom_layer.top[0]].data.shape
      bottom_shape_new = list(bottom_shape)
      if len(layer_shape) == 3:
        bottom_shape_new.insert(2, 1)
      elif len(layer_shape) >= 4:
        bottom_shape_new = [1, layer_shape[1], 1] + [int(np.prod(layer_shape[2:]))]
      else:
        continue

      reshape_0 = self.G.mk_reshape_proto(bottom_layer.top[0], bottom_shape_new, layer.name)
      self.G.add_node(reshape_0)
      self.G.add_edge(bottom_layer, reshape_0)
      self.G.add_edge(reshape_0, layer)
      self.G.remove_edge(bottom_layer, layer)
      self.G.add_replace_edge_proto(reshape_0, layer, bottom_layer.top[0])

      top_layers = layer.successors()
      reshape_1 = self.G.mk_reshape_proto(layer.top[0], layer_shape, layer.name)
      self.G.add_node(reshape_1)
      self.G.add_edge(layer, reshape_1)
      for t in top_layers:
        self.G.add_edge(reshape_1, t)
        self.G.remove_edge(layer, t)
        self.G.add_replace_edge_proto(reshape_1, t, layer.top[0])


class HtoWforSoftmax(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    softmax_layers = self.get_nodes_of_type('Softmax')
    for layer in softmax_layers:
      layer_shape = list(self.caffe_net.blobs[layer.top[0]].data.shape)
      axes = getattr(getattr(layer, 'softmax_param'), 'axis')
      hw_axes = None
      if len(layer_shape) in [1, 2, 3, 4]:
        hw_axes = [2, 3]
      elif len(layer_shape) in [5]:
        hw_axes = [3, 4]
      if axes in hw_axes:
        continue

      bottom_layer = layer.predecessors()[0]
      bottom_shape = self.caffe_net.blobs[layer.bottom[0]].data.shape
      bottom_shape_new = list(bottom_shape)
      if len(layer_shape) == 3:
        bottom_shape_new.insert(2, 1)
      elif len(layer_shape) == 4:
        if layer_shape[3] != 1:
          continue
        bottom_shape_new[3] = layer_shape[2]
        bottom_shape_new[2] = 1
      elif len(layer_shape) == 5:
        if layer_shape[4] != 1:
          continue
        bottom_shape_new[4] = layer_shape[3]
        bottom_shape_new[3] = 1
      else:
        continue

      reshape_0 = self.G.mk_reshape_proto(layer.bottom[0], bottom_shape_new, layer.name)
      self.G.add_node(reshape_0)
      self.G.add_edge(bottom_layer, reshape_0)
      self.G.add_edge(reshape_0, layer)
      self.G.remove_edge(bottom_layer, layer)
      self.G.add_replace_edge_proto_new(reshape_0, layer, bottom_layer.top[0], reshape_0.top[0])

      top_layers = list(layer.successors())
      reshape_1 = self.G.mk_reshape_proto(layer.top[0], layer_shape, layer.name)
      self.G.add_node(reshape_1)
      self.G.add_edge(layer, reshape_1)
      for t in top_layers:
        self.G.add_edge(reshape_1, t)
        self.G.remove_edge(layer, t)
        self.G.add_replace_edge_proto_new(reshape_1, t, layer.top[0], reshape_1.top[0])


class HtoWforBatchNorm(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    batch_norm_layers = self.get_nodes_of_type('BatchNorm')
    for layer in batch_norm_layers:
      layer_shape = list(self.caffe_net.blobs[layer.top[0]].data.shape)

      bottom_layer = layer.predecessors()[0]
      bottom_shape = self.caffe_net.blobs[bottom_layer.top[0]].data.shape
      bottom_shape_new = list(bottom_shape)

      if bottom_layer.type in ['Convolution', 'InnerProduct']:
        continue
      if len(layer_shape) == 3:
        bottom_shape_new.insert(2, 1)
      elif len(layer_shape) == 4:
        if layer_shape[3] != 1:
          continue
        bottom_shape_new[3] = layer_shape[2]
        bottom_shape_new[2] = 1
      elif len(layer_shape) == 5:
        if layer_shape[4] != 1:
          continue
        bottom_shape_new[4] = layer_shape[3]
        bottom_shape_new[3] = 1
      else:
        continue
      if list(bottom_shape) == bottom_shape_new:
        continue

      idx = 0
      if bottom_layer.type == "Slice":
        bottom_layer_top_list = [n for n in bottom_layer.top]
        idx = bottom_layer_top_list.index(layer.get_layer().bottom[0])
      reshape_0 = self.G.mk_reshape_proto(bottom_layer.name, bottom_shape_new, layer.name)
      self.G.add_node(reshape_0)
      self.G.add_edge(bottom_layer, reshape_0)
      self.G.add_edge(reshape_0, layer)
      self.G.remove_edge(bottom_layer, layer)
      reshape_0.bottom[0] = bottom_layer.top[idx]
      self.G.add_replace_edge_proto_new(reshape_0, layer, bottom_layer.top[idx], reshape_0.top[0])

      top_layers = list(layer.successors())
      reshape_1 = self.G.mk_reshape_proto(layer.name, layer_shape, layer.name)
      self.G.add_node(reshape_1)
      self.G.add_edge(layer, reshape_1)
      for t in top_layers:
        self.G.add_edge(reshape_1, t)
        self.G.remove_edge(layer, t)
        self.G.add_replace_edge_proto_new(reshape_1, t, layer.top[0], reshape_1.top[0])


class HtoWforLayernNorm(BasePass):
  # For LayerNorm with axis = 2 and rank 3. Move height to width
  def __init__(self):
    pass

  def do_transformation(self):
    l1norm_layers = self.get_nodes_of_type('LayerNorm')
    shape_dict = dict()
    for layer in l1norm_layers:
      layer_shape = list(self.caffe_net.blobs[layer.top[0]].data.shape)
      axes = getattr(getattr(layer, 'layer_norm_param'), 'axis')
      if len(layer_shape) not in [3] or axes != 2:
        continue

      bottom_layer = layer.predecessors()[0]
      bottom_shape = shape_dict[layer.bottom[0]] if layer.bottom[0] in shape_dict else self.caffe_net.blobs[layer.bottom[0]].data.shape
      bottom_shape_new = list(bottom_shape)
      bottom_shape_new.insert(2, 1)

      reshape_0 = self.G.mk_reshape_proto(layer.bottom[0], bottom_shape_new, layer.name)
      self.G.add_node(reshape_0)
      self.G.add_edge(bottom_layer, reshape_0)
      self.G.add_edge(reshape_0, layer)
      self.G.remove_edge(bottom_layer, layer)
      self.G.add_replace_edge_proto_new(reshape_0, layer, bottom_layer.top[0], reshape_0.top[0])
      shape_dict[reshape_0.top[0]] = bottom_shape_new

      layer.layer_norm_param.axis = 3

      top_layers = list(layer.successors())
      reshape_1 = self.G.mk_reshape_proto(layer.top[0], layer_shape, layer.name)
      self.G.add_node(reshape_1)
      self.G.add_edge(layer, reshape_1)
      for t in top_layers:
        self.G.add_edge(reshape_1, t)
        self.G.remove_edge(layer, t)
        self.G.add_replace_edge_proto_new(reshape_1, t, layer.top[0], reshape_1.top[0])
      shape_dict[reshape_1.top[0]] = layer_shape
