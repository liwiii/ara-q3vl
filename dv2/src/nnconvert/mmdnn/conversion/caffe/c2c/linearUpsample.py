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


class LinearUpsample(BasePass):

  def __init__(self):
    pass

  def standardize_upsample_scale(self, layer):
    shape = self.caffe_net.blobs[layer.top[0]].data.shape
    if len(layer.linear_upsample_param.scale) == 1:
      if len(shape) in [4, 5]:
        scale_value = layer.linear_upsample_param.scale[0]
        layer.linear_upsample_param.scale.extend([scale_value] * (len(shape) - 3))
    if len(layer.linear_upsample_param.zero_mode) == 0:
      if len(shape) in [3, 4, 5]:
        layer.linear_upsample_param.zero_mode.extend([0] * (len(shape) - 2))

  def do_transformation(self):
    for layer in self.get_nodes_of_type('LinearUpsample'):
      if layer.linear_upsample_param.upsample_mode == 4:
        layer.linear_upsample_param.permute = 1
        above = layer.predecessors()[0]
        successors = list(layer.successors())
        successors = layer.successors()
        pre_permute = self.G.mk_permute_proto(name=layer.name, order=[0, 2, 3, 1], bottom=above.name)  # NCHW -> NHWC
        self.G.add_node(pre_permute)

        post_permute = self.G.mk_permute_proto(name=layer.name, order=[0, 3, 1, 2])  # NHWC -> NCHW
        self.G.add_node(post_permute)
        self.G.add_edge(above, pre_permute)
        self.G.add_edge(pre_permute, layer)
        self.G.add_edge(layer, post_permute)
        for s in successors:
          self.G.add_edge(post_permute, s)
        self.G.remove_edge(above, layer)
        for s in successors:
          self.G.remove_edge(layer, s)
        pre_permute.bottom[0] = layer._layer.bottom[0]
        layer._layer.bottom[0] = pre_permute.top[0]
        post_permute.bottom[0] = layer._layer.top[0]
        for s in successors:
          for idx, i in enumerate(s._layer.bottom):
            if i == layer._layer.top[0]:
              s._layer.bottom[idx] = post_permute.top[0]
      self.standardize_upsample_scale(layer._layer)
