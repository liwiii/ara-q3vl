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
import copy
import math
import numpy as np
import caffe
from caffe import layers as L
from caffe import params as P
from mmdnn.conversion.caffe.utils import get_stride_dim, get_pad_dim, get_kernel_dim, set_kernel_dims, set_kernel_dim, set_pad_dim, set_stride_dim, get_dilation, get_group_dim, get_num_output, get_bias_term
from mmdnn.conversion.caffe.c2c.common import BasePass
from dv_logging import get_logger

logger = get_logger(__name__, 'DVCTOC')


class ConvolutionPatching(BasePass):

  def __init__(self):
    pass

  def _do_pathching_depthwise(self, layer):
    pass

  def fused_top(self, layer):
    top = set()
    for l in self.G.get_topsort():
      if l.name == layer.name:
        continue
      top_list = l.top
      if any(e in top for e in top_list):
        return True
      top.update(top_list)
    return False

  def eval(self, input_node_shape, pads, weights):
    dim = len(weights.shape)
    for i in range(dim - 2):
      if input_node_shape[i + 2] + pads[i] > weights.shape[i + 2]:
        return False
    return True

  def conv_k12_explicit_padding(self, layer, kernels, pads):
    if all([i > 12 for i in kernels]) and any([i != 0 for i in pads]):
      pad_layer = self.G.mk_pad_proto([i for i in pads], 0, 0, layer.name)
      set_pad_dim(layer.convolution_param, [0] * len(pads))
      predecessors = list(layer.predecessors())
      self.G.add_node(pad_layer)
      self.G.add_edge(pad_layer, layer)
      for p in predecessors:
        for t in p.top:
          if t in layer.bottom:
            self.G.remove_edge(p, layer)
            self.G.add_edge(p, pad_layer)
            self.G.add_replace_edge_proto_new(pad_layer, layer, t, pad_layer.top[0])
            self.G.add_replace_edge_proto_new(p, pad_layer, "NA", t)
      return True
    return False

  def conv_asymmetric(self, layer):
    '''
      # Pointnet: Appending zeros to the inp_wts (filters) in H direction so that 1xk=>kxk
      # Format: DCHW
      # When zeros are appended to the filter in H direction, padding also need to be changed accordingly
      # Consider following formula for the output shape
      # output_h = ceil(((input_h) - (filter_h_after_dilation)) / (stride_h) + 1)
      # filter_h_after_dilation = filter_h + (filter_h - 1)*(dilation_h - 1)
      # Note that after appending 0s filter_h = filter_w
      # add_padding is calculated such that the values of output_h before and after appending the zeros are equal
      '''

    layer_param = layer.convolution_param
    dilation = get_dilation(layer_param)
    ishape = list(self.inp_wts[layer.name][0].shape)
    if len(ishape) >= 4 and not (ishape[-2] == ishape[-1]) and (ishape[-2] == 1 or ishape[-1] == 1):
      if ishape[-1] == 1:
        unity = -1
        non_unity = -2
      else:
        unity = -2
        non_unity = -1
      #If the filter shape is ___x1xW
      paddim = get_pad_dim(layer_param)
      assert paddim is not None
      pad_dim = [dim for dim in paddim]
      zero_arr_shape_arr = ishape[:]
      zero_arr_shape_arr[unity] = zero_arr_shape_arr[non_unity] - 1
      first_half_shape = zero_arr_shape_arr[:]
      second_half_shape = zero_arr_shape_arr[:]
      first_half_shape[unity] = first_half_shape[unity] // 2
      second_half_shape[unity] = zero_arr_shape_arr[unity] - first_half_shape[unity]
      first_half = np.zeros(shape=first_half_shape)
      second_half = np.zeros(shape=second_half_shape)
      self.inp_wts[layer.name][0] = np.concatenate((first_half, self.inp_wts[layer.name][0]), axis=unity)
      self.inp_wts[layer.name][0] = np.concatenate((self.inp_wts[layer.name][0], second_half), axis=unity)
      set_kernel_dims(layer_param, ishape[non_unity], ishape[non_unity])
      assert dilation is not None
      add_padding = dilation[unity] * (ishape[non_unity] - 1)  # d_h* (k_w - 1)
      extra_padding = [0] * len(pad_dim)
      extra_padding[unity] = add_padding
      if add_padding > 1:
        extra_padding[unity] = int(math.ceil(add_padding / 2))
        extra_padding[unity - len(pad_dim) // 2] = int(math.floor(add_padding / 2))
      set_pad_dim(layer_param, np.add(pad_dim, extra_padding))
    pads = get_pad_dim(layer_param)
    kernels = get_kernel_dim(layer_param)
    return pads, kernels, layer

  def conv1d_asymmetric(self, layer):
    layer_top_name = layer.top[0]
    layer_param = layer.convolution_param
    shapes = self.caffe_net.blobs[layer_top_name].shape
    shapes = [s for s in shapes]
    input_shapes = self.caffe_net.blobs[layer.bottom[0]].shape
    input_shapes = [s for s in input_shapes]
    strides = get_stride_dim(layer_param)
    kernels = get_kernel_dim(layer_param)
    pads = get_pad_dim(layer_param)
    bottom_layer = layer.predecessors()[0]
    bottom_shape = self.caffe_net.blobs[bottom_layer.top[0]].data.shape
    bottom_shape_new = list(bottom_shape)
    bottom_shape_new.append(1)
    reshape_0 = self.G.mk_reshape_proto(bottom_layer.name, bottom_shape_new, layer.name)
    self.G.add_node(reshape_0)
    self.G.add_edge(bottom_layer, reshape_0)
    self.G.add_edge(reshape_0, layer)
    self.G.remove_edge(bottom_layer, layer)
    self.G.add_replace_edge_proto_new(reshape_0, layer, bottom_layer.top[0], reshape_0.top[0])
    self.inp_wts[layer.name][0] = np.reshape(self.inp_wts[layer.name][0], self.inp_wts[layer.name][0].shape + (1,))
    assert pads is not None
    assert strides is not None
    assert kernels is not None
    set_pad_dim(layer_param, pads[:1] + [0] + pads[1:2] + [0])
    set_stride_dim(layer_param, strides[:] + [1])
    set_kernel_dim(layer_param, kernels[:] + [1])
    pads, kernels, layer = self.conv_asymmetric(layer)
    top_layers = list(layer.successors())
    reshape_1 = self.G.mk_reshape_proto(layer.name, list(shapes), layer.name)
    self.G.add_node(reshape_1)
    self.G.add_edge(layer, reshape_1)
    for t in top_layers:
      self.G.add_edge(reshape_1, t)
      self.G.remove_edge(layer, t)
      self.G.add_replace_edge_proto_new(reshape_1, t, layer.top[0], reshape_1.top[0])

  """
  Replaces Conv node in original graph with newly added node

  Args:
    old_conv_layer: Proto of conv node in original graph
    new_conv_layer: Proto of conv node that replaces old node
  """

  def replace_conv(self, old_conv_layer, new_conv_layer):
    pds = old_conv_layer.predecessors()[0]
    sus = old_conv_layer.successors()[0]

    self.G.add_node(new_conv_layer)
    self.G.add_edge(pds, new_conv_layer)
    self.G.remove_edge(pds, old_conv_layer)

    self.G.add_edge(new_conv_layer, sus)
    self.G.add_replace_edge_proto(new_conv_layer, sus, old_conv_layer.name)
    self.G.remove_node(old_conv_layer)
    self.G.relabel_nodes_locally({f'{new_conv_layer.type}/{new_conv_layer.name}': old_conv_layer.name}, new_conv_layer.type)

  def do_transformation(self):
    convolutions = self.get_nodes_of_type('Convolution')
    for layer in convolutions:
      # assume that the convolution takes only one input
      predecessor = layer.predecessors()[0]
      layer_top_name = layer.top[0]
      layer_param = layer.convolution_param
      shapes = self.caffe_net.blobs[layer_top_name].shape
      shapes = [s for s in shapes]
      input_shapes = self.caffe_net.blobs[layer.bottom[0]].shape
      input_shapes = [s for s in input_shapes]
      if len(shapes) == 4:
        ran = 2
      else:
        ran = 3
      strides = get_stride_dim(layer_param)
      kernels = get_kernel_dim(layer_param)
      pads = get_pad_dim(layer_param)
      dilation = get_dilation(layer_param)
      num_group = get_group_dim(layer_param)
      num_output = get_num_output(layer_param)
      bias_term = get_bias_term(layer_param)
      dil = 1
      if len(input_shapes) == 3:
        self.conv1d_asymmetric(layer)
        continue
      pads, kernels, layer = self.conv_asymmetric(layer)
      assert pads is not None
      if kernels == [4, 4] and all(p == 0 for p in pads) and strides == [4, 4]:
        kernels = [6, 6]
        pads = [2, 2, 2, 2]
        assert strides is not None
        assert dilation is not None
        strides = [n for n in strides]
        dilation = [n for n in dilation]
        conv_layer = self.G.mk_conv_proto(layer.bottom[0], layer.name, kernels, num_output, stride=strides, pad=pads, group=num_group, bias_term=bias_term, dilation=dilation)
        self.inp_wts[layer.name][0] = np.pad(self.inp_wts[layer.name][0], pad_width=((0, 0), (0, 0), (2, 0), (2, 0)), mode='constant', constant_values=0.0)
        self.replace_conv(layer, conv_layer)
        continue
      if kernels == [2, 2] and all(p == 0 for p in pads) and strides == [2, 2]:
        kernels = [3, 3]
        pads = [1, 1, 1, 1]
        assert strides is not None
        assert dilation is not None
        strides = [n for n in strides]
        dilation = [n for n in dilation]

        conv_layer = self.G.mk_conv_proto(layer.bottom[0], layer.name, kernels, num_output, stride=strides, pad=pads, group=num_group, bias_term=bias_term, dilation=dilation)
        self.inp_wts[layer.name][0] = np.pad(self.inp_wts[layer.name][0], pad_width=((0, 0), (0, 0), (1, 0), (1, 0)), mode='constant', constant_values=0.0)
        self.replace_conv(layer, conv_layer)
        continue
      if kernels == [2, 2] and strides == [1, 1]:
        kernels = [3, 3]
        if len(pads) == 4:
          pads = [pads[0] + 1, pads[1] + 1, pads[2], pads[3]]
        elif len(pads) == 2:
          pads = [pads[0] + 1, pads[1] + 1, pads[0] + 1, pads[1] + 1]
        elif not pads:
          pads = [0, 0, 1, 1]
        assert strides is not None
        assert dilation is not None
        strides = [n for n in strides]
        dilation = [n for n in dilation]
        conv_layer = self.G.mk_conv_proto(layer.bottom[0], layer.name, kernels, num_output, stride=strides, pad=pads, group=num_group, bias_term=bias_term, dilation=dilation)
        self.inp_wts[layer.name][0] = np.pad(self.inp_wts[layer.name][0], pad_width=((0, 0), (0, 0), (1, 0), (1, 0)), mode='constant', constant_values=0.0)
        self.replace_conv(layer, conv_layer)
        continue

      if self.conv_k12_explicit_padding(layer, kernels, pads):
        continue
      if dilation is None:
        dilation = [1 for i in range(ran)]
      else:
        dil = dilation[-1]
      if strides is None:
        strides = [1 for i in range(ran)]
      if pads is None:
        pads = [0 for i in range(ran * 2)]
      if len(list(kernels)) == 1:
        val = kernels[0]
        for i in range(ran - 1):
          kernels.append(val)
      if len(list(strides)) == 1:
        val = strides[0]
        for i in range(ran - 1):
          strides.append(val)
      if len(list(pads)) == 1:
        val = pads[0]
        for i in range(ran * 2 - 1):
          pads.append(val)
      elif len(list(pads)) == 3 or len(list(pads)) == 2:
        pad_copy = copy.deepcopy(pads)
        pads.extend(pad_copy)
      if len(list(dilation)) == 1:
        val = dilation[0]
        for i in range(ran - 1):
          dilation.append(val)
      # caffeemitter depthwise logic
      dilation = [d for d in dilation]
      strides = [s for s in strides]
      pads = [p for p in pads]
      # if num_group == input_shapes[1]:
      #   logger.debug("Depthwise Convolution ...")

      if num_group == input_shapes[1]:
        logger.debug("Depthwise Convolution ...")
        # self._do_pathching_depthwise(layer)
        assert kernels is not None
        if num_group == kernels[0]:
          num_group = kernels[0]
        if kernels == [1, 1] and (not np.any(pads)):
          logger.info("mapping cndw1x1 to 3x3 " + layer.name)
          kernels = [3, 3]
          pads = [1 for i in range(ran * 2)]
          conv_layer = self.G.mk_conv_proto(predecessor.top[0], layer.name, kernels, num_output, stride=strides, pad=pads, group=num_group, bias_term=bias_term, dilation=dilation)

          dim = len(strides)
          self.inp_wts[layer.name][0] = np.pad(self.inp_wts[layer.name][0], pad_width=((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
          self.replace_conv(layer, conv_layer)
          continue
        elif kernels == [2, 2]:
          logger.info("mapping cndw2x2 to 3x3")
          kernels = [3, 3]
          pads = get_pad_dim(layer_param)
          assert pads is not None
          pads = [int(i) for i in pads]
          pads[0:2] = [1, 1]
          conv_layer = self.G.mk_conv_proto(predecessor.top[0], layer.name, kernels, num_output, stride=strides, pad=pads, group=num_group, bias_term=bias_term, dilation=dilation)
          dim = len(strides)
          self.inp_wts[layer.name][0] = np.pad(self.inp_wts[layer.name][0], pad_width=((0, 0), (0, 0), (1, 0), (1, 0)), mode='constant', constant_values=0)
          self.replace_conv(layer, conv_layer)
          continue
          self.G.add_node(conv_layer)
          self.G.add_edge(layer.predecessors()[0], conv_layer)
          if (layer.successors()):
            self.G.add_edge(conv_layer, layer.successors()[0])
          continue

        elif all(x > 1 for x in dilation[0:-1]):
          # dilation = [1]*len(dilation)
          kernels = [kernels[i] + (kernels[i] - 1) * (dilation[i] - 1) for i in range(len(kernels))]
          dim = len(strides)
          logger.info("converting depthwise dilated conv to non-dilated conv with weights inflation")
          conv_layer = self.G.mk_conv_proto(layer.bottom[0], layer.name, kernels, num_output, stride=strides, pad=pads, group=num_group, bias_term=bias_term, dilation=[1] * len(dilation))
          self.G.add_node(conv_layer)
          for idx, p in enumerate(layer.predecessors()):
            self.G.add_edge(layer.predecessors()[idx], conv_layer)
          layer_successors = list(layer.successors())
          for idx, succ in enumerate(layer_successors):
            self.G.add_edge(conv_layer, layer_successors[idx])
          # self.inp_wts[layer.name][0] = np.transpose(self.inp_wts[layer.name][0], [dim + 1, dim] + list(range(0, dim)))
          # insert (d-1) zeros in between each filter element
          weights = self.inp_wts[layer.name][0]
          weights_shape = list(weights.shape)
          assert dim == 2
          # TODO for 3D dilations
          for d in range(dim):
            weights_shape[d + 2] = weights_shape[d + 2] + (weights_shape[d + 2] - 1) * (dilation[d] - 1)
          weights_new = np.zeros(tuple(weights_shape))
          for h in range(weights.shape[-2]):
            for w in range(weights.shape[-1]):
              weights_new[:, :, h * dilation[-2], w * dilation[-1]] = weights[:, :, h, w]
          self.inp_wts[layer.name][0] = weights_new
          self.replace_conv(layer, conv_layer)
          continue
      # caffeemitter Conv logic
      ## INRP LOGIC FROM CAFFE
      elif all(p == 0 for p in pads) and input_shapes[2:] == kernels and num_group != input_shapes[1]:
        if self.fused_top(layer):
          continue
        inrp_layer = self.G.mk_inrp_proto(layer.bottom[0], num_output, bias_term, layer.name)
        reshape_layer = self.G.mk_reshape_proto(inrp_layer.top[0], shapes, layer.name)
        predecessor = layer.predecessors()[0]
        self.G.add_node(inrp_layer)
        self.G.add_node(reshape_layer)
        self.G.add_edge(predecessor, inrp_layer)
        self.G.add_edge(inrp_layer, reshape_layer)
        self.G.remove_edge(predecessor, layer)
        for succ in list(layer.successors()):
          self.G.add_edge(reshape_layer, succ)
          self.G.add_replace_edge_proto_new(reshape_layer, succ, layer.top[0], reshape_layer.top[0])
        self.G.remove_node(layer)
        self.G.relabel_nodes_locally({f'{inrp_layer.type}/{inrp_layer.name}': layer.name}, inrp_layer.type)
        continue
      elif len(set(shapes[2:])) == 1 and shapes[2] == 1 and self.eval(input_shapes, pads, self.inp_wts[layer.name][0]) and num_group != input_shapes[1]:
        if self.fused_top(layer):
          continue
        inrp_layer = self.G.mk_inrp_proto(layer.bottom[0], num_output, bias_term, layer.name)
        reshape_layer = self.G.mk_reshape_proto(layer.top[0], shapes, layer.name)
        self.G.add_node(inrp_layer)
        self.G.add_node(reshape_layer)
        self.G.remove_edge(predecessor, layer)
        self.G.add_edge(predecessor, inrp_layer)
        self.G.add_edge(inrp_layer, reshape_layer)
        self.G.add_replace_edge_proto_new(inrp_layer, reshape_layer, layer.top[0], inrp_layer.top[0])
        dim = len(strides)

        if len(self.inp_wts[layer.name][0].shape) == 4:
          ws = self.inp_wts[layer.name][0].shape
          self.inp_wts[layer.name][0] = self.inp_wts[layer.name][0][:, :, pads[0]:pads[0] + min(input_shapes[2], ws[2]), pads[1]:pads[1] + min(input_shapes[3], ws[3])]
        elif len(self.inp_wts[layer.name][0].shape) == 5:
          self.inp_wts[layer.name][0] = self.inp_wts[layer.name][0][:, :, pads[0]:pads[0] + input_shapes[3], pads[1]:pads[1] + input_shapes[2], pads[2]:pads[2] + input_shapes[4]]
        for succ in list(layer.successors()):
          self.G.add_edge(reshape_layer, succ)
          self.G.add_replace_edge_proto_new(reshape_layer, succ, layer.top[0], reshape_layer.top[0])
        self.G.remove_node(layer)
        self.G.relabel_nodes_locally({f'{inrp_layer.type}/{inrp_layer.name}': layer.name}, inrp_layer.type)
        continue
      # ----------------------
      flag = True
      assert kernels is not None
      for i in kernels:
        if i != 1:
          flag = False
          break
      if flag and dil == 6:
        continue
      kernels = list(kernels) * 2
      dilation = list(dilation) * 2
      strides = list(strides) * 2
      expected_pad = []
      for idx, kernel in enumerate(kernels):
        expected_pad.append((kernel + (kernel - 1) * (dilation[idx] - 1)) // 2)
      pad_vals = []
      count = []
      need_padding = False
      for idx in range(len(expected_pad)):
        if expected_pad[idx] < pads[idx]:
          need_padding = True
      if not need_padding:
        continue
      for idx, e_pad in enumerate(expected_pad):
        pad_vals.append(max(0, pads[idx] - e_pad))
      new_pads = list(np.array(pads) - np.array(pad_vals))
      pad_layer = self.G.mk_pad_proto(pad_vals, 0, 0, layer.name)
      set_pad_dim(layer.convolution_param, new_pads)
      self.G.add_node(pad_layer)
      self.G.add_edge(pad_layer, layer)
      self.G.remove_edge(predecessor, layer)
      self.G.add_edge(predecessor, pad_layer)
      p_tops = list(predecessor.top)
      for t in predecessor.top:
        if t in layer.bottom:
          self.G.add_replace_edge_proto_new(pad_layer, layer, t, pad_layer.top[0])
          self.G.add_replace_edge_proto_new(predecessor, pad_layer, "NA", t)


class FuseConvolution(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    # TODO: Fix the bug in converter/estimator post r6.1 for the conv fusion to work
    # Disabling for now
    if True:
      return
    convolutions = self.get_nodes_of_type('Convolution')
    conv_dict = {}
    for conv1 in convolutions:
      for conv2 in convolutions:
        if conv1.name != conv2.name and len(conv1.successors()) == 1 and conv2.name in [n.name for n in list(conv1.successors())]:
          logger.info("Successor convolution node found")
          conv_dict[conv1] = conv2
    for conv in conv_dict:
      p_conv_param = conv.convolution_param
      s_conv_param = conv_dict[conv].convolution_param
      p_strides, p_kernels, p_pads, p_dilations = get_stride_dim(p_conv_param), get_kernel_dim(p_conv_param), get_pad_dim(p_conv_param), get_pad_dim(p_conv_param)
      s_strides, s_kernels, s_pads, s_dilations = get_stride_dim(s_conv_param), get_kernel_dim(s_conv_param), get_pad_dim(s_conv_param), get_pad_dim(s_conv_param)
      p_kernels = np.array([i for i in p_kernels])
      s_kernels = np.array([i for i in s_kernels])
      if np.array_equal(p_kernels, s_kernels) and np.all(p_kernels == 1):
        # Fusing the convolution layers
        p_wts, p_bias = self.inp_wts[conv.name][0], self.inp_wts[conv.name][1]
        s_wts, s_bias = self.inp_wts[conv_dict[conv].name][0], self.inp_wts[conv_dict[conv].name][1]
        p_wts_sh = p_wts.shape
        s_wts_sh = s_wts.shape
        if p_wts_sh[-1] * s_wts_sh[-2] > p_wts_sh[-2] * (p_wts_sh[-1] + s_wts_sh[-2]):
          continue
        new_wts = np.squeeze(s_wts) @ np.squeeze(p_wts)  # (p_wts@(s_wts.T)).T
        new_bias = (np.expand_dims(p_bias, 1).T @ (s_wts.T)).T.flatten() + s_bias
        self.inp_wts[conv.name][0] = new_wts
        self.inp_wts[conv.name][1] = new_bias
        del self.inp_wts[conv_dict[conv].name]
        p_conv_param.num_output = s_conv_param.num_output
        successors = list(conv_dict[conv].successors())
        # self.G.remove_edge(conv, conv_dict[conv])
        for idx, sc in enumerate(successors):
          self.G.add_edge(conv, sc)
          self.G.remove_edge(conv_dict[conv], sc)
          self.G.add_replace_edge_proto_new(conv, sc, conv_dict[conv].top[0], conv.top[0])
        self.G.remove_node(conv_dict[conv])
        del self.caffe_net.params[conv_dict[conv].name]


class SliceConvolution1(BasePass):

# This identifies the pattern convolution + swish + slice (on the outtermost dim) and splits into two convolutions on the out channels
  def __init__(self):
    pass

  def do_transformation(self):
    #TODO: check if slice is on the outter most dimension
    slice_layers = self.get_nodes_of_type('Slice')
    for slice_l in slice_layers:

      conv_node_names = []
      if not len(slice_l.predecessors()) == 1:
        continue
      reshape1_l = slice_l.predecessors()[0]
      if reshape1_l.type != 'Reshape':
        continue
      if not len(reshape1_l.predecessors()) == 1:
        continue
      
      permute_l = reshape1_l.predecessors()[0]
      if permute_l.type != 'PermuteNnp':
        continue
      # if not permute_l.type == 'Permute':
      #   continue
      if not len(permute_l.predecessors()) == 1:
        continue

      reshape2_l = permute_l.predecessors()[0]
      if reshape2_l.type != 'Reshape':
        continue

      conv_l = reshape2_l.predecessors()[0]
      if conv_l.type != 'Convolution':
        continue
      

      # swish_l = slice_l.predecessors()[0]
      # if swish_l.type != 'Swish':
      #   continue
      # if not len(swish_l.predecessors()) == 1:
      #   continue
      # conv_l = swish_l.predecessors()[0]
      # if conv_l.type != 'Convolution':
      #   continue

      conv_bottom_l = conv_l.predecessors()[0]
      
      # if not slice_l.slice_param.axis == 1:
      #   continue
      
      if not len(slice_l.slice_param.slice_point) == 2:
        continue

      dim_along_axis = self.caffe_net.blobs[slice_l.bottom[0]].data.shape[slice_l.slice_param.axis]
      conv_out_channels = conv_l.convolution_param.num_output

      if not conv_out_channels == dim_along_axis:
        continue

      num_output_0 = slice_l.slice_param.slice_point[0]
      num_output_1 =  num_output_0
      num_output_2 = num_output_0

      def ll(aa):
        if len(aa) == 0:
          return None
        return [i for i in aa]
      

      conv_0_l = self.G.mk_conv_proto(conv_l.bottom[0], conv_l.name + '_0', ll(conv_l.convolution_param.kernel_size), num_output_0, ll(conv_l.convolution_param.stride), ll(conv_l.convolution_param.pad), conv_l.convolution_param.group, conv_l.convolution_param.bias_term, ll(conv_l.convolution_param.dilation))
      self.G.add_node(conv_0_l)
      conv_node_names.append(f"{conv_0_l.name}")
      self.G.add_edge(conv_bottom_l, conv_0_l)

      # swish_l_0 = self.G.mk_swish_proto(conv_0_l.top[0], swish_l.name + '_0')
      # self.G.add_node(swish_l_0)
      # self.G.add_edge(conv_0_l, swish_l_0)

      conv_1_l = self.G.mk_conv_proto(conv_l.bottom[0], conv_l.name + '_1', ll(conv_l.convolution_param.kernel_size), num_output_1, ll(conv_l.convolution_param.stride), ll(conv_l.convolution_param.pad), conv_l.convolution_param.group, conv_l.convolution_param.bias_term, ll(conv_l.convolution_param.dilation))
      self.G.add_node(conv_1_l)
      conv_node_names.append(f"{conv_1_l.name}")
      self.G.add_edge(conv_bottom_l, conv_1_l)

      conv_2_l = self.G.mk_conv_proto(conv_l.bottom[0], conv_l.name + '_2', ll(conv_l.convolution_param.kernel_size), num_output_2, ll(conv_l.convolution_param.stride), ll(conv_l.convolution_param.pad), conv_l.convolution_param.group, conv_l.convolution_param.bias_term, ll(conv_l.convolution_param.dilation))
      self.G.add_node(conv_2_l)
      conv_node_names.append(f"{conv_2_l.name}")
      self.G.add_edge(conv_bottom_l, conv_2_l)

      # swish_l_1 = self.G.mk_swish_proto(conv_1_l.top[0], swish_l.name + '_1')
      # self.G.add_node(swish_l_1)
      # self.G.add_edge(conv_1_l, swish_l_1)

      for top_l in slice_l.successors():
        top_l_new = top_l.successors()[0].successors()[0].successors()[0]
        for tops_bottom_blob in top_l.bottom:
          if tops_bottom_blob == slice_l.top[0]:
            self.G.add_edge(conv_0_l, top_l_new)
            self.G.add_replace_edge_proto_new(conv_0_l, top_l_new, top_l.successors()[0].successors()[0].top[0],conv_0_l.top[0])
          elif tops_bottom_blob == slice_l.top[1]:
            self.G.add_edge(conv_1_l, top_l_new)
            self.G.add_replace_edge_proto_new(conv_1_l, top_l_new, top_l.successors()[0].successors()[0].top[0],conv_1_l.top[0])
          else: 
            self.G.add_edge(conv_2_l, top_l_new)
            self.G.add_replace_edge_proto_new(conv_2_l, top_l_new, top_l.successors()[0].successors()[0].top[0],conv_2_l.top[0])
        self.G.remove_edge(slice_l, top_l)
        self.G.remove_node(top_l.successors()[0].successors()[0])
        self.G.remove_node(top_l.successors()[0])
        self.G.remove_node(top_l)

      conv_wts = np.copy(self.inp_wts[conv_l.name][0])
      # conv_bias = np.copy(self.inp_wts[conv_l.name][1])
      # conv_bias = np.zeros(4000)
      # self.inp_wts[conv_0_l.name] = [np.copy(conv_wts[:num_output_0]), np.copy(conv_bias[:num_output_0])]
      # self.inp_wts[conv_1_l.name] = [np.copy(conv_wts[num_output_0:2*num_output_0]), np.copy(conv_bias[num_output_0:2*num_output_0])]
      # self.inp_wts[conv_2_l.name] = [np.copy(conv_wts[:num_output_0]), np.copy(conv_bias[:num_output_0])]
      
      self.inp_wts[conv_0_l.name] = [np.copy(conv_wts[:num_output_0])]
      self.inp_wts[conv_1_l.name] = [np.copy(conv_wts[num_output_0:2*num_output_0])]
      self.inp_wts[conv_2_l.name] = [np.copy(conv_wts[-num_output_0:])]

      self.inp_wts.pop(conv_l.name)
      self.G.remove_edge(conv_bottom_l, conv_l)

      self.G.remove_node(reshape1_l)
      self.G.remove_node(permute_l)
      self.G.remove_node(reshape2_l)
      self.G.remove_node(conv_l)
      self.G.remove_node(slice_l)
      self.mutation_observer["family"][conv_l.name] = conv_node_names

class SliceConvolution2(BasePass):

# This identifies the pattern convolution + swish + slice (on the outtermost dim) and splits into two convolutions on the out channels
  def __init__(self):
    pass

  def do_transformation(self):
    #TODO: check if slice is on the outter most dimension
    slice_layers = self.get_nodes_of_type('Slice')
    for slice_l in slice_layers:
      conv_node_names = []
      if not len(slice_l.predecessors()) == 1:
        continue
      swish_l = slice_l.predecessors()[0]
      if swish_l.type != 'Swish':
        continue
      if not len(swish_l.predecessors()) == 1:
        continue
      conv_l = swish_l.predecessors()[0]
      if conv_l.type != 'Convolution':
        continue

      conv_bottom_l = conv_l.predecessors()[0]
      
      if not slice_l.slice_param.axis == 1:
        continue
      
      if not len(slice_l.slice_param.slice_point) == 1:
        continue
      
      num_output_0 = slice_l.slice_param.slice_point[0]
      num_output_1 = conv_l.convolution_param.num_output - num_output_0

      def ll(aa):
        if len(aa) == 0:
          return None
        return [i for i in aa]

      conv_0_l = self.G.mk_conv_proto(conv_l.bottom[0], conv_l.name + '_0', ll(conv_l.convolution_param.kernel_size), num_output_0, ll(conv_l.convolution_param.stride), ll(conv_l.convolution_param.pad), conv_l.convolution_param.group, conv_l.convolution_param.bias_term, ll(conv_l.convolution_param.dilation))
      self.G.add_node(conv_0_l)
      conv_node_names.append(f"{conv_0_l.name}")
      self.G.add_edge(conv_bottom_l, conv_0_l)

      swish_l_0 = self.G.mk_swish_proto(conv_0_l.top[0], swish_l.name + '_0')
      self.G.add_node(swish_l_0)
      self.G.add_edge(conv_0_l, swish_l_0)

      conv_1_l = self.G.mk_conv_proto(conv_l.bottom[0], conv_l.name + '_1', ll(conv_l.convolution_param.kernel_size), num_output_1, ll(conv_l.convolution_param.stride), ll(conv_l.convolution_param.pad), conv_l.convolution_param.group, conv_l.convolution_param.bias_term, ll(conv_l.convolution_param.dilation))
      conv_node_names.append(f"{conv_1_l.name}")
      self.G.add_node(conv_1_l)
      self.G.add_edge(conv_bottom_l, conv_1_l)

      swish_l_1 = self.G.mk_swish_proto(conv_1_l.top[0], swish_l.name + '_1')
      self.G.add_node(swish_l_1)
      self.G.add_edge(conv_1_l, swish_l_1)

      for top_l in slice_l.successors():
        for tops_bottom_blob in top_l.bottom:
          if tops_bottom_blob == slice_l.top[0]:
            self.G.add_edge(swish_l_0, top_l)
            self.G.add_replace_edge_proto_new(swish_l_0, top_l, tops_bottom_blob, swish_l_0.top[0])
          elif tops_bottom_blob == slice_l.top[1]:
            self.G.add_edge(swish_l_1, top_l)
            self.G.add_replace_edge_proto_new(swish_l_1, top_l, tops_bottom_blob, swish_l_1.top[0])
        self.G.remove_edge(slice_l, top_l)
      
      conv_wts = np.copy(self.inp_wts[conv_l.name][0])
      conv_bias = np.copy(self.inp_wts[conv_l.name][1])
      self.inp_wts[conv_0_l.name] = [np.copy(conv_wts[:num_output_0]), np.copy(conv_bias[:num_output_0])]
      self.inp_wts[conv_1_l.name] = [np.copy(conv_wts[num_output_0:]), np.copy(conv_bias[num_output_0:])]
      
      self.inp_wts.pop(conv_l.name)
      self.G.remove_edge(conv_bottom_l, conv_l)
      
      self.G.remove_node(conv_l)
      self.G.remove_node(swish_l)
      self.G.remove_node(slice_l)
      self.mutation_observer["family"][conv_l.name] = conv_node_names

class MoveReshapefromConvBNorm(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    all_reshapes = self.get_nodes_of_type('Reshape')
    reshapes = []
    for i in all_reshapes:
      p = list(i.predecessors())
      s = list(i.successors())
      if len(p) == 1 and p[0].type == 'Convolution' and len(s) == 1 and s[0].type == 'BatchNorm':
        reshapes.append(i)
    for i in reshapes:
      conv = i.predecessors()[0]
      bnorm = i.successors()[0]
      self.G.add_edge(conv, bnorm)
      self.G.remove_edge(conv, i)
      self.G.remove_edge(i, bnorm)
      self.G.add_replace_edge_proto_new(conv, bnorm, i.top[0], conv.top[0])
      par = bnorm
      if len(bnorm.successors()) == 1 and bnorm.successors()[0].type == 'Scale':
        par = bnorm.successors()[0]
      child = list(par.successors())
      self.G.add_replace_edge_proto_new(par, i, conv.top[0], par.top[0])
      self.G.add_edge(par, i)
      for c in child:
        self.G.add_edge(i, c)
        self.G.remove_edge(par, c)
        self.G.add_replace_edge_proto_new(i, c, par.top[0], i.top[0])


class MoveReshapePermfromConvBnorm(BasePass):
  """
  This pattern is applied to the case when there are reshape, perms between convs and bnorms. This will shift bnorm above so that the reshape perm pattern can be simplified
  """

  def __init__(self):
    pass

  def do_transformation(self):
    all_convs = self.get_nodes_of_type('Convolution')
    intermediate_nodes = ['Reshape', 'Permute', 'PermuteNnp']
    replacement_node = ['BatchNorm']
    for conv in all_convs:
      intermediate_list = []
      fin_node = conv
      conv_s = conv.successors()
      while(len(conv_s) == 1):
        if conv_s[0].type in intermediate_nodes:
          intermediate_list.append(conv_s[0])
          conv_s = conv_s[0].successors()
        else:
          break
      if len(conv_s) and conv_s[0].type in replacement_node and len(intermediate_list):
        scale_node = conv_s[0].successors()
        if len(scale_node) != 1 or scale_node[0].type != 'Scale':
          continue
        scale_s = scale_node[0].successors()
        self.G.add_edge(conv, conv_s[0])
        self.G.remove_edge(conv, intermediate_list[0])
        self.G.remove_edge(intermediate_list[-1], conv_s[0])
        self.G.add_edge(scale_node[0], intermediate_list[0])
        self.G.add_replace_edge_proto_new(conv, conv_s[0], intermediate_list[-1].top[0], conv.top[0]) #updating the bottom of bnorm
        self.G.add_replace_edge_proto_new(scale_node[0], intermediate_list[0], conv.top[0], scale_node[0].top[0]) #updating the bottom of the first reshape/perm after the original conv
        for node in scale_s:
          self.G.remove_edge(scale_node[0], node)
          self.G.add_edge(intermediate_list[-1], node)
        self.G.add_replace_edge_proto_new(intermediate_list[-1], node, scale_node[0].top[0], intermediate_list[-1].top[0]) #updating the bottom of the node after the scale node.
      else:
        continue



class GroupConvPatching(BasePass):
  """ this patching is used to split a grouped convolution into group number of individual convolutions.

  Args:
      BasePass
  """
  def __init__(self):
    pass

  def do_transformation(self):
    # this pattern splits a grouped conv into group number of individual convolutions.
    conv_nodes = self.get_nodes_of_type('Convolution')

    def ll(aa):
      if len(aa) == 0:
        return None
      return [i for i in aa]

    for conv in conv_nodes:
      group_size = conv.convolution_param.group
      in_channels = self.caffe_net.blobs[conv.bottom[0]].data.shape[1]
      if group_size == 1 or (group_size == in_channels): #regular convolution / depthwise convolution
        continue
      else:
        interval_size = in_channels // group_size
        slice_point = [interval_size*(i+1) for i in range(group_size-1)]
        conv_pred = conv.predecessors()[0]
        conv_wts = np.copy(self.inp_wts[conv.name][0])
        num_per_group = conv.convolution_param.num_output // group_size
        if conv.convolution_param.bias_term :
          conv_bias = np.copy(self.inp_wts[conv.name][1])
        if conv.successors():
          conv_suc = conv.successors()[0]
        concat = self.G.mk_concat_proto(conv.top[0], conv.name, new_axis = 1) 
        concat.bottom.pop()
        self.G.add_node(concat)
        if conv.successors():
          self.G.remove_edge(conv,conv_suc)
          self.G.add_edge(concat, conv_suc)
          self.G.add_replace_edge_proto_new(concat, conv_suc, conv.top[0], concat.top[0])
        else:
          concat.top[:] = conv.top
        axis = 1
        dim_along_axis = self.caffe_net.blobs[conv.top[0]].data.shape[axis]
        slice_node = self.G.mk_slice_proto(conv.bottom[0], conv.name, dim_along_axis, axis=axis, slice_point=slice_point)
        self.G.add_node(slice_node)
        self.G.remove_edge(conv_pred, conv)
        self.G.add_edge(conv_pred, slice_node)
        for i in range(group_size):
          suffix = '_' + str(i)
          # prefix = str(i) + '_'
          conv_new = self.G.mk_conv_proto(slice_node.top[i], conv.name + suffix, ll(conv.convolution_param.kernel_size),num_per_group, ll(conv.convolution_param.stride), ll(conv.convolution_param.pad), 1, conv.convolution_param.bias_term, ll(conv.convolution_param.dilation))
          self.G.add_node(conv_new)
          self.G.add_edge(slice_node, conv_new)
          self.G.add_edge(conv_new, concat)
          if conv_new.convolution_param.bias_term:
            self.inp_wts[conv_new.name] = [np.copy(conv_wts[i*num_per_group:(i+1)*num_per_group]), np.copy(conv_bias[i*num_per_group:(i+1)*num_per_group])]
          else:
            self.inp_wts[conv_new.name] = [np.copy(conv_wts[i*num_per_group:(i+1)*num_per_group])]
          concat.bottom.append(conv_new.top[0])

        self.inp_wts.pop(conv.name) 
        self.G.remove_node(conv)
            
            
class DepthwiseConvPatching(BasePass):
  """
  this pattern is for the case when in_channels == number of groups != out_channels, to split into adjacent cndws.

  Args:
      BasePass 

  """
  def __init__(self):
    pass

  def do_transformation(self):
    conv_nodes = self.get_nodes_of_type('Convolution')
    flag =1

    def ll(aa):
      if len(aa) == 0:
        return None
      return [i for i in aa]

    def split_by_k(arr,k):
      return [arr[i::k] for i in range(k)]
  
    for idx,conv in enumerate(conv_nodes):
      # if conv.name == '_supercombo_vision__en_final_conv_reparam_conv_Conv':
      group_size = conv.convolution_param.group
      out_channels = conv.convolution_param.num_output
       #flag decides how we would check the num of in_channels for a particular conv.
      if flag:
        in_channels = self.caffe_net.blobs[conv.bottom[0]].data.shape[1] 
      else:
        in_channels = conv_nodes[idx-1].convolution_param.num_output

      if conv.successors()[0].type == 'Convolution':
          flag =0
      else: flag =1


      if group_size == 1 and in_channels == 1:  # this would fall in the case of a normal convolution.
        continue
      if group_size == in_channels and (in_channels != out_channels): 


        num_cndw = out_channels // in_channels
        conv_pred = conv.predecessors()[0]
        conv_wts = np.copy(self.inp_wts[conv.name][0])
        split_conv_wts = split_by_k(conv_wts, num_cndw)
        concat = self.G.mk_concat_proto(conv.top[0], conv.name, new_axis = 1)
        concat.bottom.pop() 

        shape_conv = self.caffe_net.blobs[conv.top[0]].data.shape
        shape_1 = list((1,num_cndw, in_channels) + shape_conv[2:])
      
        reshape_1 = self.G.mk_reshape_proto(concat.top[0], shape_1, conv.name+ "_reshape_1" )
        order = [0,2,1,3,4]#NCDHW -> NDCHW
        permute = self.G.mk_permute_proto(conv.name + "_permute", order, reshape_1.top[0])
        shape_2 = list(shape_conv)
        reshape_2 = self.G.mk_reshape_proto(permute.top[0], shape_2, conv.name + "_reshape_2")

        self.G.add_node(concat)
        self.G.add_node(reshape_1)
        self.G.add_edge(concat, reshape_1)
        self.G.add_replace_edge_proto_new(concat, reshape_1, concat.top[0], concat.top[0])
        self.G.add_node(permute)
        self.G.add_edge(reshape_1, permute)
        self.G.add_node(reshape_2)
        self.G.add_edge(permute, reshape_2)
        # self.G.add_replace_edge_proto_new()

        if conv.convolution_param.bias_term:
          conv_bias = np.copy(self.inp_wts[conv.name][1])
          split_conv_biases = split_by_k(conv_bias,num_cndw)
        if conv.successors():
          for conv_suc in conv.successors():
          # conv_suc = conv.successors()[0]
            self.G.remove_edge(conv, conv_suc)
            self.G.add_edge(reshape_2, conv_suc)
            self.G.add_replace_edge_proto_new(reshape_2, conv_suc, conv.top[0],reshape_2.top[0]) #for altering conv's successors bottom
        else:
          reshape_2.top[:] = conv.top
        self.G.remove_edge(conv_pred, conv)
        for i in range(num_cndw):
          suffix = '_' + str(i)
          cndw_new = self.G.mk_conv_proto(conv.bottom[0], conv.name + suffix, ll(conv.convolution_param.kernel_size), in_channels, ll(conv.convolution_param.stride), ll(conv.convolution_param.pad), group_size, conv.convolution_param.bias_term, ll(conv.convolution_param.dilation))
          self.G.add_node(cndw_new)
          self.G.add_edge(conv_pred, cndw_new)
          self.G.add_edge(cndw_new, concat)
          if cndw_new.convolution_param.bias_term:
            self.inp_wts[cndw_new.name] = [split_conv_wts[i], split_conv_biases[i]]
          else:
            self.inp_wts[cndw_new.name] = split_conv_wts[i]
          concat.bottom.append(cndw_new.top[0])
        
        
        self.inp_wts.pop(conv.name)
        self.G.remove_node(conv)


class SliceConvolutionCrop(BasePass):

  """
  This pass looks for the pattern of conv followed by multiple crops that are trying to split the conv across 
  it's out channels . This pattern is replaced by mutiple individual convs so that we can avoid the crop operation. 
  """
  def __init__(self):
    pass
  
  def is_slice_on_channel_dim(self, axis, node_order, num_splits):
    """
    Back-tracks the slice/crop axis through the reshape/permute chain to check
    if it originates from the conv output channel dimension (axis=1 in NCHW).
    Returns True if the split axis maps back to channel dim (axis 1).
    """
    current_axis = axis

    for node in reversed(node_order):
        if node.type == 'Reshape':
            input_shape = None
            output_shape = list(node.reshape_param.shape.dim)

            # Try to get the input shape from the node's bottom blob
            if hasattr(node, 'input_shape') and len(node.input_shape) > 0:
                input_shape = list(node.input_shape[0].dim)
            elif hasattr(self, 'blob_shapes') and node.bottom[0] in self.blob_shapes:
                input_shape = list(self.blob_shapes[node.bottom[0]])

            if input_shape is None:
                # Cannot back-track through reshape without shape info, conservatively return False
                return False

            # Compute cumulative sizes to map output axis -> input axis
            if current_axis < 0 or current_axis >= len(output_shape):
                return False

            # Find the corresponding input axis by matching cumulative products
            out_cumulative = 1
            for i in range(current_axis + 1):
                out_cumulative *= output_shape[i]

            in_cumulative = 1
            mapped_axis = None
            for i, dim in enumerate(input_shape):
                in_cumulative *= dim
                if in_cumulative == out_cumulative:
                    mapped_axis = i
                    break

            if mapped_axis is None:
                return False
            current_axis = mapped_axis

        elif node.type in ['Permute', 'PermuteNnp']:
            permute_order = list(node.permute_param.order)
            if current_axis < 0 or current_axis >= len(permute_order):
                return False
            try:
                # current_axis = permute_order.index(current_axis)
                current_axis = permute_order[current_axis]
            except ValueError:
                return False
        else:
            return False

    # After back-tracking, check if we've landed on axis 1 (channel dim in NCHW)
    return current_axis == 1
  
  def find_reshape_perm(self, axis=0,node_order=None, num_splits=0):
    if node_order is None:
      return []
    rev_node_order = node_order[::-1] #reverse the list
    type_param = []
    current_axis = axis

    for node in rev_node_order:
        if node.type == 'Reshape':
          shape_from_node = list(node.reshape_param.shape.dim)

          if axis < 0 or axis >= len(shape_from_node):
            raise IndexError(f"Tracked axis {current_axis} out of range for reshape param {shape_from_node}")

          shape_from_node[current_axis] //= num_splits
          type_param.append(('Reshape',shape_from_node))

        elif node.type in ['Permute', 'PermuteNnp']:
          permute_order = list(node.permute_param.order)
          
          if current_axis < 0 or current_axis >= len(permute_order):
                raise IndexError(f"Tracked axis {current_axis} out of range for Permute order {permute_order}")
          
          type_param.append((node.type,permute_order))
          current_axis = permute_order[current_axis]

          # if len(prev_shape) < len(permute_order):
          #   if permute_order == [2,4,0,1,3]:
          #     type_param.append(())
        else:
          raise NotImplementedError(f"Backward tracking for node type '{node.type}' is not implemented.")
    return type_param[::-1]

  def ll(self, aa):
    if len(aa) == 0:
      return None
    return [i for i in aa]

  def do_transformation(self):
    conv_layers = self.get_nodes_of_type('Convolution')
    intermediate_layers_list = ["Reshape", "Permute", "PermuteNnp"]
    # replacement_layers_list = ["Crop","Slice"]
    replacement_layers_list = ["Slice"]


    for conv_l in conv_layers:
      conv_s = conv_l.successors()
      num_output = conv_l.convolution_param.num_output
      prev_node = conv_l
      conv_wts = np.copy(self.inp_wts[conv_l.name][0])
      conv_bias = None
      do_transformation = True
      replication_list = []
      if conv_l.convolution_param.bias_term:
        conv_bias = np.copy(self.inp_wts[conv_l.name][1]) 
      while(len(conv_s) == 1):
        if conv_s[0].type in intermediate_layers_list:
          replication_list.append(conv_s[0])
          prev_node = conv_s
          conv_s = conv_s[0].successors()
        else:
          break
      #this is a hardcoding that might have to be changed to generalize this pass (in the case of unequal slice/crop)
      slice_node = None
      if len(conv_s) == 1 and conv_s[0].type == "Slice":
        slice_node = conv_s[0]
        branch_layers = slice_node.successors()
      else:
        branch_layers = conv_s

      if len(conv_s) == 0:
        continue
      num_output_split = int(num_output / len(branch_layers))
      conv_bottom = conv_l.predecessors()[0]

      for idx, layer in enumerate(branch_layers):
        branch_id = f"branch_{idx}"
        if slice_node is None and layer.type not in replacement_layers_list:
          do_transformation = False
          break
        else:
          conv_0_l = self.G.mk_conv_proto(conv_l.bottom[0], conv_l.name + f'_{idx}', self.ll(conv_l.convolution_param.kernel_size), num_output_split, self.ll(conv_l.convolution_param.stride), self.ll(conv_l.convolution_param.pad), conv_l.convolution_param.group, conv_l.convolution_param.bias_term, self.ll(conv_l.convolution_param.dilation))
          axis = 0
          offset = idx
          if layer.type == "Crop":
            axis = layer.crop_param.axis
            offset = layer.crop_param.offset[0]
          elif slice_node is not None:
            axis = slice_node.slice_param.axis
            branch_bottom = layer.bottom[0]
            offset = list(slice_node.top).index(branch_bottom)
          chain_params = self.find_reshape_perm( axis=axis, node_order=replication_list, num_splits=len(branch_layers))

          if not self.is_slice_on_channel_dim(axis, replication_list, len(branch_layers)):
            do_transformation = False
            break

          layer_pred = layer.predecessors()
          self.G.add_node(conv_0_l)
          self.G.add_edge(conv_bottom, conv_0_l)

          current_tail_node = conv_0_l
          for count,op in enumerate(chain_params):
            op_type = op[0]
            op_param = op[1]
            original_node_name = replication_list[count].name
            new_node_name = f"{original_node_name}_{branch_id}"
            if op_type == 'Reshape':
              new_node = self.G.mk_reshape_proto(current_tail_node.top[0], self.ll(op_param), new_node_name)
            elif op_type == 'Permute':
              new_node = self.G.mk_permute_proto(name=new_node_name, order=self.ll(op_param), bottom=current_tail_node.top[0])
            elif op_type == 'PermuteNnp':
              new_node = self.G.mk_permute_nnp_proto(layer_name=new_node_name,bottom_name=current_tail_node.top[0], order=self.ll(op_param))
            self.G.add_node(new_node)
            self.G.add_edge(current_tail_node, new_node)
            current_tail_node = new_node

          successor_nodes = [layer] if slice_node is not None else layer.successors()
          layer = layer.predecessors()[0] if slice_node is not None else layer

          for replace_succ in successor_nodes:
            self.G.add_edge(current_tail_node, replace_succ)
            for replace_succ_bottom in replace_succ.bottom:
              if replace_succ_bottom in layer.top:
                self.G.add_replace_edge_proto_new(current_tail_node, replace_succ,replace_succ_bottom,current_tail_node.top[0])
            self.G.remove_edge(layer, replace_succ)
            
          for pred in layer_pred:
            if pred.type == 'DummyData':
              self.G.remove_node(pred)

          if slice_node is None:
            self.G.remove_node(layer)
          if conv_bias is None:
            self.inp_wts[conv_0_l.name] = [np.copy(conv_wts[offset*num_output_split:(offset+1)*num_output_split])]
          else:
            self.inp_wts[conv_0_l.name] = [np.copy(conv_wts[offset*num_output_split:(offset+1)*num_output_split]), np.copy(conv_bias[offset*num_output_split:(offset+1)*num_output_split])]
      
      #removes all nodes from conv_bottom to prev_node[0] (excluding conv_bottom)

      if do_transformation:
        for node in replication_list:
          self.G.remove_node(node)
        if slice_node is not None:
          self.G.remove_node(slice_node)
        # self.G.remove_nodes_list(conv_l, None)
        self.inp_wts.pop(conv_l.name)
        self.G.remove_edge(conv_bottom, conv_l)
        self.G.remove_node(conv_l)

from mmdnn.conversion.caffe.c2c.common import BasePass
import numpy as np

class SplitConvAfterConcat(BasePass):
  """
  Matches the pattern: Concat -> Convolution -> ReLU
  It splits the Convolution across the concatenated input channels into multiple 
  smaller convolutions (one for each input to the Concat). It then sums their 
  outputs using a tree of 2-input Eltwise Add layers, allowing us to safely 
  remove the original Concat and original Convolution.
  """
  def __init__(self):
    pass

  def do_transformation(self):
    concat_nodes = self.get_nodes_of_type('Concat')

    def ll(aa):
      if len(aa) == 0:
        return None
      return [i for i in aa]

    for concat_node in concat_nodes:
      # 1. Verify Concat is along the channel axis (axis == 1)
      axis = concat_node.concat_param.axis if hasattr(concat_node.concat_param, 'axis') else 1
      if axis != 1:
        continue

      # 2. Verify Concat is followed strictly by one Convolution layer
      successors = list(concat_node.successors())
      if len(successors) != 1 or successors[0].type != 'Convolution':
        continue

      conv_node = successors[0]

      # Grouped convolutions require much more complex channel mapping, skip them
      if conv_node.convolution_param.group != 1:
        continue

      # 3. Verify Convolution is followed by a ReLU (or similar activation)
      conv_successors = list(conv_node.successors())
      if not any((succ.type == 'ReLU' and succ.relu_param.negative_slope == 0) for succ in conv_successors):
         continue

      conv_param = conv_node.convolution_param
      conv_wts = np.copy(self.inp_wts[conv_node.name][0])
      has_bias = conv_param.bias_term
      if has_bias:
        conv_bias = np.copy(self.inp_wts[conv_node.name][1])

      concat_bottom_blobs = list(concat_node.bottom)
      num_inputs = len(concat_bottom_blobs)
      if num_inputs < 2:
        continue

      new_conv_nodes = []
      start_channel = 0

      # -------------------------------------------------------------
      # Step A: Split the Convolution across the input channels
      # -------------------------------------------------------------
      for i, bottom_blob_name in enumerate(concat_bottom_blobs):
        # Find the predecessor node that generated this specific input blob
        pred_node = None
        for p in concat_node.predecessors():
          if bottom_blob_name in p.top:
            pred_node = p
            break

        if pred_node is None:
          continue

        # Get channel size for this specific branch
        in_channels = self.caffe_net.blobs[bottom_blob_name].data.shape[1]
        end_channel = start_channel + in_channels

        # Slice the weights along the input channel dimension (axis=1)
        split_wts = conv_wts[:, start_channel:end_channel, :, :]
        start_channel = end_channel

        conv_name_i = conv_node.name + f'_split_{i}'

        # To keep the math exactly equivalent, ONLY add the original bias to the 
        # FIRST convolution. All subsequent convolutions should not add a bias.
        bias_term_i = has_bias if i == 0 else False

        # Make the smaller convolution node
        conv_new = self.G.mk_conv_proto(
          bottom_blob_name,
          conv_name_i,
          ll(conv_param.kernel_size),
          conv_param.num_output,
          ll(conv_param.stride),
          ll(conv_param.pad),
          1, # group
          bias_term_i,
          ll(conv_param.dilation)
        )

        # Apply structural topology updates
        self.G.add_node(conv_new)
        self.G.add_edge(pred_node, conv_new)

        # Save partitioned weights and parameters
        if bias_term_i:
          self.inp_wts[conv_new.name] = [np.copy(split_wts), np.copy(conv_bias)]
        else:
          self.inp_wts[conv_new.name] = [np.copy(split_wts)]

        new_conv_nodes.append(conv_new)

      if len(new_conv_nodes) != num_inputs:
        continue # Abort if we failed to match all branches

      # -------------------------------------------------------------
      # Step B: Build the Add (Eltwise) tree for 2 inputs at a time
      # -------------------------------------------------------------
      prev_add_node = None
      for i in range(1, num_inputs):
        add_name = conv_node.name + f'_add_{i}'

        if i == 1:
          # First addition takes the first two convolutions
          bottom_blobs = [new_conv_nodes[0].top[0], new_conv_nodes[1].top[0]]
          add_node = self.G.mk_eltwise_proto(bottom_blobs, add_name, operation=1) # operation=0 as requested
          
          self.G.add_node(add_node)
          self.G.add_edge(new_conv_nodes[0], add_node)
          self.G.add_edge(new_conv_nodes[1], add_node)
        else:
          # Subsequent additions take the previous Add output + the next convolution
          bottom_blobs = [prev_add_node.top[0], new_conv_nodes[i].top[0]]
          add_node = self.G.mk_eltwise_proto(bottom_blobs, add_name, operation=1)
          
          self.G.add_node(add_node)
          self.G.add_edge(prev_add_node, add_node)
          self.G.add_edge(new_conv_nodes[i], add_node)

        prev_add_node = add_node

      # -------------------------------------------------------------
      # Step C: Rewire the Successor (ReLU) and clean up
      # -------------------------------------------------------------
      # Connect the final Add node to whatever followed the original Conv (e.g., ReLU)
      for succ in conv_successors:
        self.G.add_edge(prev_add_node, succ)
        # Update the Protobuf data-flow mappings
        self.G.add_replace_edge_proto_new(prev_add_node, succ, conv_node.top[0], prev_add_node.top[0])

      # Unhook edges from original Concat and Conv
      for p in list(concat_node.predecessors()):
        self.G.remove_edge(p, concat_node)

      self.G.remove_edge(concat_node, conv_node)

      for succ in list(conv_node.successors()):
        self.G.remove_edge(conv_node, succ)

      # Remove original nodes and pop old weights
      self.G.remove_node(concat_node)
      self.G.remove_node(conv_node)
      self.inp_wts.pop(conv_node.name)

class ConvolutionKernel32to1(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    convolutions = self.get_nodes_of_type('Convolution')

    for conv in convolutions:
      conv_input_shape = list(self.caffe_net.blobs[conv.bottom[0]].data.shape)
      conv_kernel_size = conv.convolution_param.kernel_size
      conv_stride = conv.convolution_param.stride
      k_size = -1
      if len(conv_kernel_size) == len(conv_stride) == 2:
        if conv_kernel_size[0] == conv_kernel_size[1] == conv_stride[0] == conv_stride[1]:
          k_size = conv_kernel_size[0]
      
      #for clip_vit_B32
      if len(conv_input_shape) == 4 and k_size != -1 and conv_input_shape[-2]==conv_input_shape[-1] and conv_input_shape[-2]%k_size == 0 and k_size > 3:
        factor = conv_input_shape[-2]//k_size 
        channels = conv_input_shape[1]
        reshape1 = self.G.mk_reshape_proto(conv.bottom[0], [channels, factor, k_size, factor*k_size], conv.name)
        self.G.add_node(reshape1)
        permutennp1 = self.G.mk_permute_nnp_proto(conv.name, reshape1.top[0], [0,2,1,3])

        self.G.add_node(permutennp1)
        reshape2 = self.G.mk_reshape_proto(permutennp1.name, [1, channels*k_size, factor*factor, k_size], conv.name)
        self.G.add_node(reshape2)
        permutennp2 = self.G.mk_permute_nnp_proto(conv.name, reshape2.top[0], [0,1,3,2])
        self.G.add_node(permutennp2)
        reshape3 = self.G.mk_reshape_proto(permutennp2.name, [1, channels*k_size*k_size, 1, factor*factor], conv.name)
        self.G.add_node(reshape3)
        reshape4 = self.G.mk_reshape_proto(conv.name, [1, self.inp_wts[conv.name][0].shape[0], factor, factor], conv.name)
        self.G.add_node(reshape4)

        conv_pred = conv.predecessors()[0]
        self.G.add_edge(conv_pred, reshape1)
        self.G.add_replace_edge_proto_new(conv_pred, reshape1, reshape1.bottom[0], conv_pred.top[0])

        self.G.add_edge(reshape1, permutennp1)
        self.G.add_replace_edge_proto_new(reshape1, permutennp1, permutennp1.bottom[0], reshape1.top[0])

        self.G.add_edge(permutennp1, reshape2)
        self.G.add_replace_edge_proto_new(permutennp1, reshape2, reshape2.bottom[0], permutennp1.top[0])

        self.G.add_edge(reshape2, permutennp2)
        self.G.add_replace_edge_proto_new(reshape2, permutennp2, permutennp2.bottom[0], reshape2.top[0])

        self.G.add_edge(permutennp2, reshape3)
        self.G.add_replace_edge_proto_new(permutennp2, reshape3, reshape3.bottom[0], permutennp2.top[0])
        
        self.G.add_edge(reshape3, conv)
        self.G.add_replace_edge_proto_new(reshape3, conv, conv.bottom[0], reshape3.top[0])
        
        successors = list(conv.successors())
        for successor in successors:
          for ind, bottom in enumerate(successor.bottom):
            if bottom == conv.top[0]:
              successor.bottom[ind] = reshape4.top[0]
              self.G.remove_edge(conv, successor)
              self.G.add_edge(reshape4, successor)

        self.G.add_edge(conv, reshape4)
        self.G.remove_edge(conv_pred, conv)

        conv.convolution_param.kernel_size[0] = 1
        conv.convolution_param.kernel_size[1] = 1
        conv.convolution_param.stride[0] = 1
        conv.convolution_param.stride[1] = 1

        self.inp_wts[conv.name][0] = np.reshape(self.inp_wts[conv.name][0], (self.inp_wts[conv.name][0].shape[0], self.inp_wts[conv.name][0].shape[1] * self.inp_wts[conv.name][0].shape[2] * self.inp_wts[conv.name][0].shape[3], 1, 1))