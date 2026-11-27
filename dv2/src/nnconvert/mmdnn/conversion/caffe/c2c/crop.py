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


def need_multi_descriptor(ishape, oshape):
  crop_dims = 0
  for idx in range(len(ishape)):
    if ishape[idx] != oshape[idx]:
      crop_dims += 1
  if crop_dims > 2:
    return True
  return False


class CropPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    crops = self.get_nodes_of_type("Crop")
    for crop in crops:
      ishape = list(self.caffe_net.blobs[crop.bottom[0]].data.shape)
      oshape = list(self.caffe_net.blobs[crop.top[0]].data.shape)
      crop_param = crop.crop_param
      if len(ishape) != 5:
        continue
      if not need_multi_descriptor(ishape, oshape):
        continue
      offset = list(crop_param.offset)
      axis = 0
      if crop_param.axis != 0:
        offset = [0] * crop_param.axis + offset
      offset = offset[-3:]
      reshape1_dims = [np.prod(ishape[0:3])] + list(ishape[3:])
      reshape_layer = self.G.mk_reshape_proto(crop.top[0], reshape1_dims, crop.name)
      self.G.add_node(reshape_layer)
      dummy_layer1, crop_layer1 =\
        self.G.mk_crop_proto(name=crop.name,
                      dummy_shape=list([reshape1_dims[0]] + oshape[-2:]),
                      crop_offset=[0]*(len(reshape1_dims)-2) + offset[-2:])
      self.G.add_node(dummy_layer1)
      self.G.add_node(crop_layer1)
      reshape2_dims = [np.prod(ishape[0:2])] + [ishape[2]] + [np.prod(oshape[3:])]
      reshape_layer2 = self.G.mk_reshape_proto(crop_layer1.name, reshape2_dims, crop.name)
      self.G.add_node(reshape_layer2)
      crop2_dims = reshape2_dims
      crop2_dims[1] = oshape[2]
      dummy_layer2, crop_layer2 =\
        self.G.mk_crop_proto(name=crop.name,
                      dummy_shape=crop2_dims,
                      crop_offset=[0] + [offset[2]] + [0])
      self.G.add_node(dummy_layer2)
      self.G.add_node(crop_layer2)
      reshape_layer3 = self.G.mk_reshape_proto(crop_layer2.name, list(oshape), crop.name)
      self.G.add_node(reshape_layer3)
      crop_layer1.crop_param.axis = 0
      crop_layer2.crop_param.axis = 0
      predecessors = list(crop.predecessors())
      for predecessor in predecessors:
        if predecessor.type == 'DummyData':
          self.G.remove_node(predecessor)
      successors = list(crop.successors())
      predecessors = list(crop.predecessors())
      reshape_layer.bottom[0] = predecessors[0].top[0]
      reshape_layer2.bottom[0] = crop_layer1.top[0]
      reshape_layer3.bottom[0] = crop_layer2.top[0]
      crop_layer1.bottom.append(reshape_layer.top[0])
      crop_layer1.bottom.append(dummy_layer1.top[0])
      crop_layer2.bottom.append(reshape_layer2.top[0])
      crop_layer2.bottom.append(dummy_layer2.top[0])
      self.G.add_edge(predecessors[0], reshape_layer)
      self.G.add_edge(reshape_layer, crop_layer1)
      self.G.add_edge(crop_layer1, reshape_layer2)
      self.G.add_edge(reshape_layer2, crop_layer2)
      self.G.add_edge(crop_layer2, reshape_layer3)
      self.G.add_edge(dummy_layer1, crop_layer1)
      self.G.add_edge(dummy_layer2, crop_layer2)
      for succ in successors:
        self.G.add_edge(reshape_layer3, succ)
        for idx, bottom in enumerate(succ.bottom):
          if succ.bottom[idx] == crop.top[0]:
            succ.bottom[idx] = reshape_layer3.top[0]
      self.G.remove_node(crop)
