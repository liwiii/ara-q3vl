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
import numpy as np

# class BDotPattern1PatternMatching(BasePass):
#   '''
#   (8, 1, 2304, 2304)->(8, 2304, 2304)
#   (1, 8, 2304, 40)->(8, 2304, 40)
#   (8, 2304, 40)
#   (1, 8, 2304, 40)
#   '''
#   def __init__(self):
#     pass

#   def do_transformation(self):
#     layers = self.get_nodes_of_type('BatchDot')
#     for batchdot in layers:
#       # check for valid pattern
#       if not len(batchdot.predecessors()) == 2:
#         continue
#       reshape1, reshape2 = batchdot.predecessors()
#       if reshape1.type != 'Reshape' or reshape2.type != 'Reshape':
#         continue
#       if len(reshape1.predecessors()) != 1 or len(reshape1.predecessors()) != 1:
#         continue
#       perm1, perm2 = reshape1.predecessors()[0], reshape2.predecessors()[0]
#       if perm1.type != 'PermuteNnp' or perm2.type != 'PermuteNnp':
#         continue
#       if len(perm1.successors()) != 1 or len(perm2.successors()) != 1:
#         continue
#       if not len(batchdot.successors()) == 1:
#         continue
#       reshape3 = batchdot.successors()[0]
#       if reshape3.type != 'Reshape':
#         continue
#       if not len(reshape3.successors()) == 1:
#         continue
#       perm3 = reshape3.successors()[0]
#       if perm3.type != 'PermuteNnp':
#         continue
#       perm1_shape = self.caffe_net.blobs[reshape1.bottom[0]].data.shape
#       perm2_shape = self.caffe_net.blobs[reshape2.bottom[0]].data.shape
#       reshape1_shape = self.caffe_net.blobs[reshape1.top[0]].data.shape
#       reshape2_shape = self.caffe_net.blobs[reshape2.top[0]].data.shape
#       if perm1_shape[-2:] != reshape1_shape[-2:]:
#         continue
#       if perm2_shape[-2:] != reshape2_shape[-2:]:
#         continue
#       batchdot_shape = self.caffe_net.blobs[batchdot.top[0]].data.shape
#       reshape3_shape = self.caffe_net.blobs[reshape3.top[0]].data.shape
#       if batchdot_shape[-2:] != reshape3_shape[-2:]:
#         continue
#       # pattern is valid
#       # removing all permutennp
#       perm1_par = perm1.predecessors()[0]
#       self.G.add_edge(perm1_par, reshape1)
#       self.G.add_replace_edge_proto_new(perm1_par, reshape1, reshape1.bottom[0], perm1_par.top[0])
#       self.G.remove_node(perm1)
#       perm2_par = perm2.predecessors()[0]
#       self.G.add_edge(perm2_par, reshape2)
#       self.G.add_replace_edge_proto_new(perm2_par, reshape2, reshape2.bottom[0], perm2_par.top[0])
#       self.G.remove_node(perm2)
#       succs = list(perm3.successors())
#       for succ in succs:
#         self.G.add_edge(reshape3, succ)
#         self.G.add_replace_edge_proto_new(reshape3, succ, succ.bottom[0], reshape3.top[0])
#       self.G.remove_node(perm3)
#       reshape1.reshape_param.ClearField('shape')
#       reshape1.reshape_param.shape.dim.extend([i for i in reshape1_shape[:-2]] + [reshape1_shape[-1],reshape1_shape[-2]])
#       reshape2.reshape_param.ClearField('shape')
#       reshape2.reshape_param.shape.dim.extend([i for i in reshape2_shape[:-2]] + [reshape2_shape[-1],reshape2_shape[-2]])
#       reshape3.reshape_param.ClearField('shape')
#       reshape3.reshape_param.shape.dim.extend([i for i in reshape3_shape[:-2]] + [reshape3_shape[-1],reshape3_shape[-2]])
#       batchdot.bottom[0], batchdot.bottom[1] = batchdot.bottom[1], batchdot.bottom[0]
#       batchdot.batch_dot_param.transpose_a, batchdot.batch_dot_param.transpose_b = batchdot.batch_dot_param.transpose_b, batchdot.batch_dot_param.transpose_a


class BDotPermPattern1(BasePass):
  def __init__(self):
    pass

  '''
  If pattern is batchdot->reshape->permutennp. Then remove permute if possible and handle batchdot inputs.
  '''

  def do_transformation(self):
    layers = self.get_nodes_of_type('BatchDot')
    for batchdot in layers:
      if not len(batchdot.successors()) == 1:
        continue
      reshape3 = batchdot.successors()[0]
      if reshape3.type != 'Reshape':
        continue
      if not len(reshape3.successors()) == 1:
        continue
      perm3 = reshape3.successors()[0]
      if perm3.type != 'Permute':
        continue
      batchdot_shape = self.caffe_net.blobs[batchdot.top[0]].data.shape
      reshape3_shape = self.caffe_net.blobs[reshape3.top[0]].data.shape
      if batchdot_shape[-2:] != reshape3_shape[-2:]:
        continue
      if perm3.permute_param.order != [i for i in range(len(perm3.permute_param.order) - 2)] + [len(perm3.permute_param.order) - 1, len(perm3.permute_param.order) - 2]:
        continue
      succs = list(perm3.successors())
      for succ in succs:
        self.G.add_edge(reshape3, succ)
        self.G.add_replace_edge_proto_new(reshape3, succ, succ.bottom[0], reshape3.top[0])
      self.G.remove_node(perm3)
      reshape3.reshape_param.ClearField('shape')
      reshape3.reshape_param.shape.dim.extend([i for i in reshape3_shape[:-2]] + [reshape3_shape[-1], reshape3_shape[-2]])
      batchdot.bottom[0], batchdot.bottom[1] = batchdot.bottom[1], batchdot.bottom[0]
      batchdot.batch_dot_param.transpose_a, batchdot.batch_dot_param.transpose_b = batchdot.batch_dot_param.transpose_b, batchdot.batch_dot_param.transpose_a
      batchdot.batch_dot_param.transpose_a, batchdot.batch_dot_param.transpose_b = 1 - batchdot.batch_dot_param.transpose_a, 1 - batchdot.batch_dot_param.transpose_b


class BDotPermPattern2(BasePass):

  def __init__(self):
    pass

  '''
  If pattern is batchdot->permutennp. Then remove permute if possible and handle batchdot inputs.
  '''

  def do_transformation(self):
    layers = self.get_nodes_of_type('BatchDot')
    for batchdot in layers:
      if not len(batchdot.successors()) == 1:
        continue
      perm3 = batchdot.successors()[0]
      if perm3.type not in ['Permute','PermuteNnp']:
        continue
      if hasattr(perm3,'permute_nnp_param'):
        if perm3.permute_nnp_param.order != [i for i in range(len(perm3.permute_nnp_param.order) - 2)] + [len(perm3.permute_nnp_param.order) - 1, len(perm3.permute_nnp_param.order) - 2]:
          continue
      else:
        if perm3.permute_param.order != [i for i in range(len(perm3.permute_param.order) - 2)] + [len(perm3.permute_param.order) - 1, len(perm3.permute_param.order) - 2]:
          continue
      succs = list(perm3.successors())
      for succ in succs:
        self.G.add_edge(batchdot, succ)
        for idx, b in enumerate(succ.bottom):
          if b == perm3.top[0]:
            succ.bottom[idx] = batchdot.top[0]
      self.G.remove_node(perm3)
      batchdot.bottom[0], batchdot.bottom[1] = batchdot.bottom[1], batchdot.bottom[0]
      batchdot.batch_dot_param.transpose_a, batchdot.batch_dot_param.transpose_b = batchdot.batch_dot_param.transpose_b, batchdot.batch_dot_param.transpose_a
      batchdot.batch_dot_param.transpose_a, batchdot.batch_dot_param.transpose_b = 1 - batchdot.batch_dot_param.transpose_a, 1 - batchdot.batch_dot_param.transpose_b


class BDotPermPattern3(BasePass):
  def __init__(self):
    pass

  '''
  If pattern is permutennp->reshape->batchdot. Then remove permute if possible and handle batchdot inputs.
  '''

  def do_transformation(self):
    layers = self.get_nodes_of_type('Permute')
    for perm in layers:
      # check for valid pattern
      if perm.permute_param.order != [i for i in range(len(perm.permute_param.order) - 2)] + [len(perm.permute_param.order) - 1, len(perm.permute_param.order) - 2]:
        continue
      if not len(perm.successors()) == 1:
        continue
      reshape3 = perm.successors()[0]
      if reshape3.type != 'Reshape':
        continue
      if not len(reshape3.successors()) == 1:
        continue
      batchdot = reshape3.successors()[0]
      if batchdot.type != 'BatchDot':
        continue
      perm_shape = self.caffe_net.blobs[reshape3.bottom[0]].data.shape
      reshape3_shape = self.caffe_net.blobs[reshape3.top[0]].data.shape
      if perm_shape[-2:] != reshape3_shape[-2:]:
        continue
      # pattern is valid
      if reshape3.top[0] == batchdot.bottom[0]:
        if batchdot.batch_dot_param.transpose_a == 0:
          batchdot.batch_dot_param.transpose_a = 1 - batchdot.batch_dot_param.transpose_a
        else:
          continue
      else:
        if batchdot.batch_dot_param.transpose_b == 1:
          batchdot.batch_dot_param.transpose_b = 1 - batchdot.batch_dot_param.transpose_b
        else:
          continue
      # removing all permutennp
      perm_par = perm.predecessors()[0]
      self.G.add_edge(perm_par, reshape3)
      self.G.add_replace_edge_proto_new(perm_par, reshape3, reshape3.bottom[0], perm_par.top[0])
      self.G.remove_node(perm)
      reshape3.reshape_param.ClearField('shape')
      reshape3.reshape_param.shape.dim.extend([i for i in reshape3_shape[:-2]] + [reshape3_shape[-1], reshape3_shape[-2]])


class BDotPermPattern4(BasePass):

  def __init__(self):
    pass

  '''
  If pattern is permutennp->batchdot. Then remove permute if possible and handle batchdot inputs.
  '''
  def do_transformation(self):
    layers = self.get_nodes_of_type('Permute')
    for perm in layers:
      # check for valid pattern
      if perm.permute_param.order != [i for i in range(len(perm.permute_param.order) - 2)] + [len(perm.permute_param.order) - 1, len(perm.permute_param.order) - 2]:
        continue
      if not len(perm.successors()) == 1:
        continue
      batchdot = perm.successors()[0]
      if batchdot.type != 'BatchDot':
        continue
      # pattern is valid
      if perm.top[0] == batchdot.bottom[0]:
        if batchdot.batch_dot_param.transpose_a == 0:
          batchdot.batch_dot_param.transpose_a = 1 - batchdot.batch_dot_param.transpose_a
        else:
          continue
      else:
        if batchdot.batch_dot_param.transpose_b == 1:
          batchdot.batch_dot_param.transpose_b = 1 - batchdot.batch_dot_param.transpose_b
        else:
          continue
      # removing all permutennp
      perm_par = perm.predecessors()[0]
      self.G.add_edge(perm_par, batchdot)
      self.G.add_replace_edge_proto_new(perm_par, batchdot, perm.top[0], perm_par.top[0])
      self.G.remove_node(perm)


class GeGLuSplit_1(BasePass):
  def __init__(self):
    pass

  def do_transformation(self):
    layers = self.get_nodes_of_type('Convolution')
    for conv in layers:
      # check for valid pattern
      if not len(conv.successors()) == 1:
        continue
      bnorm = conv.successors()[0]
      if bnorm.type != 'BatchNorm':
        continue
      if not len(bnorm.successors()) == 1:
        continue
      scale = bnorm.successors()[0]
      if scale.type != 'Scale':
        continue
      if not len(scale.successors()) == 1:
        continue
      reshape1 = scale.successors()[0]
      if reshape1.type != 'Reshape':
        continue
      if not len(reshape1.successors()) == 1:
        continue
      perm1 = reshape1.successors()[0]
      if perm1.type != 'Permute':
        continue
      if not len(perm1.successors()) == 2:
        continue
      reshape21, reshape22 = perm1.successors()
      if reshape21.type != 'Reshape':
        continue
      if not len(reshape21.successors()) == 1:
        continue
      crop1 = reshape21.successors()[0]
      if crop1.type != 'Crop':
        continue
      if not len(crop1.predecessors()) == 2:
        continue
      crop1_par = crop1.predecessors()
      if crop1_par[0].name == reshape21.name:
        data1 = crop1_par[1]
      else:
        data1 = crop1_par[0]
      if data1.type != 'DummyData':
        continue
      if not len(crop1.successors()) == 1:
        continue
      crop1_child = crop1.successors()[0]
      if reshape22.type != 'Reshape':
        continue
      if not len(reshape22.successors()) == 1:
        continue
      crop2 = reshape22.successors()[0]
      if crop2.type != 'Crop':
        continue
      if not len(crop2.predecessors()) == 2:
        continue
      crop2_par = crop2.predecessors()
      if crop2_par[0].name == reshape22.name:
        data2 = crop2_par[1]
      else:
        data2 = crop2_par[0]
      if data2.type != 'DummyData':
        continue
      if not len(crop2.successors()) == 1:
        continue
      crop2_child = crop2.successors()[0]
      if crop1_child.type not in ['Eltwise', 'Gelu'] or crop2_child.type not in ['Eltwise', 'Gelu'] and crop1_child.type == crop2_child.type:
        continue
      if crop1_child.type == 'Eltwise':
        eltwise, gelu = crop1_child, crop2_child
      else:
        gelu, eltwise = crop1_child, crop2_child
      if not len(gelu.successors()) == 1:
        continue
      if gelu.successors()[0].name != eltwise.name:
        continue
      if not len(eltwise.successors()) == 1:
        continue
      reshape3 = eltwise.successors()[0]
      if reshape3.type != 'Reshape':
        continue
      if not len(reshape3.successors()) == 1:
        continue
      perm2 = reshape3.successors()[0]
      if perm2.type != 'Permute':
        continue
      if not len(perm2.successors()) == 1:
        continue
      reshape4 = perm2.successors()[0]
      if reshape4.type != 'Reshape':
        continue

      if perm1.permute_param.order != [0, 1, 3, 2]:
        continue
      if perm2.permute_param.order != [0, 1, 3, 2]:
        continue
      if reshape21.reshape_param.shape.dim != reshape22.reshape_param.shape.dim:
        continue
      if eltwise.eltwise_param.operation != 0:
        continue
      if eltwise.eltwise_param.broadcast0 != [0.0, 0.0, 0.0]:
        continue
      if eltwise.eltwise_param.broadcast1 != [0.0, 0.0, 0.0]:
        continue

      def ll(aa):
        if len(aa) == 0:
          return None
        return [i for i in aa]

      conv2 = self.G.mk_conv_proto(conv.bottom[0], conv.name, ll(conv.convolution_param.kernel_size), conv.convolution_param.num_output // 2, ll(conv.convolution_param.stride), ll(conv.convolution_param.pad), conv.convolution_param.group, conv.convolution_param.bias_term, ll(conv.convolution_param.dilation))
      conv.convolution_param.num_output = conv.convolution_param.num_output // 2
      self.G.add_node(conv2)
      self.G.add_edge(conv.predecessors()[0], conv2)
      bnorm2 = self.G.mk_bnrm_proto(conv2.top[0], bnorm.name, bnorm.batch_norm_param.use_global_stats, bnorm.batch_norm_param.eps)
      self.G.add_node(bnorm2)
      self.G.add_edge(conv2, bnorm2)
      scale2 = self.G.mk_scale_proto(bnorm2.top[0], scale.name, scale.scale_param.bias_term)
      self.G.add_node(scale2)
      self.G.add_edge(bnorm2, scale2)
      self.G.add_edge(scale2, gelu)
      self.G.add_replace_edge_proto_new(scale2, gelu, gelu.bottom[0], scale2.top[0])
      self.G.add_edge(scale, eltwise)
      if gelu.top[0] == eltwise.bottom[0]:
        crop1_top = eltwise.bottom[1]
      else:
        crop1_top = eltwise.bottom[0]
      self.G.add_replace_edge_proto_new(scale, eltwise, crop1_top, scale.top[0])
      succs = list(reshape4.successors())
      for succ in succs:
        self.G.add_edge(eltwise, succ)
        self.G.add_replace_edge_proto_new(eltwise, succ, reshape4.top[0], eltwise.top[0])
      self.G.remove_node(reshape1)
      self.G.remove_node(perm1)
      self.G.remove_node(reshape21)
      self.G.remove_node(data1)
      self.G.remove_node(crop1)
      self.G.remove_node(reshape22)
      self.G.remove_node(crop2)
      self.G.remove_node(data2)
      self.G.remove_node(reshape3)
      self.G.remove_node(perm2)
      self.G.remove_node(reshape4)
      num_output = conv.convolution_param.num_output
      conv_wt_1 = np.copy(self.inp_wts[conv.name][0])
      self.inp_wts[conv.name] = [np.copy(conv_wt_1[:num_output])]
      self.inp_wts[conv2.name] = [np.copy(conv_wt_1[num_output:])]
      bnorm_wt_1 = np.copy(self.inp_wts[bnorm.name][0])
      bnorm_wt_2 = np.copy(self.inp_wts[bnorm.name][1])
      bnorm_wt_3 = np.copy(self.inp_wts[bnorm.name][2])
      self.inp_wts[bnorm.name] = [np.copy(bnorm_wt_1[:num_output]), np.copy(bnorm_wt_2[:num_output]), np.copy(bnorm_wt_3)]
      self.inp_wts[bnorm2.name] = [np.copy(bnorm_wt_1[num_output:]), np.copy(bnorm_wt_2[num_output:]), np.copy(bnorm_wt_3)]
      scale_wt_1 = np.copy(self.inp_wts[scale.name][0])
      scale_wt_2 = np.copy(self.inp_wts[scale.name][1])
      self.inp_wts[scale.name] = [np.copy(scale_wt_1[:num_output]), np.copy(scale_wt_2[:num_output])]
      self.inp_wts[scale2.name] = [np.copy(scale_wt_1[num_output:]), np.copy(scale_wt_2[num_output:])]


class GeGLuSplit_2(BasePass):
  def __init__(self):
    pass

  def do_transformation(self):
    layers = self.get_nodes_of_type('Convolution')
    for conv in layers:
      # check for valid pattern
      if not len(conv.successors()) == 1:
        continue
      bnorm = conv.successors()[0]
      if bnorm.type != 'BatchNorm':
        continue
      if not len(bnorm.successors()) == 1:
        continue
      scale = bnorm.successors()[0]
      if scale.type != 'Scale':
        continue
      if not len(scale.successors()) == 1:
        continue
      reshape1 = scale.successors()[0]
      if reshape1.type != 'Reshape':
        continue
      if not len(reshape1.successors()) == 1:
        continue
      perm1 = reshape1.successors()[0]
      if perm1.type != 'Permute':
        continue
      if not len(perm1.successors()) == 1:
        continue
      reshape2 = perm1.successors()[0]
      if reshape2.type != 'Reshape':
        continue
      if not len(reshape2.successors()) == 2:
        continue
      crop1, crop2 = reshape2.successors()
      if crop1.type != 'Crop':
        continue
      if not len(crop1.predecessors()) == 2:
        continue
      crop1_par = crop1.predecessors()
      if crop1_par[0].name == reshape2.name:
        data1 = crop1_par[1]
      else:
        data1 = crop1_par[0]
      if data1.type != 'DummyData':
        continue
      if not len(crop1.successors()) == 1:
        continue
      crop1_child = crop1.successors()[0]
      if crop2.type != 'Crop':
        continue
      if not len(crop2.predecessors()) == 2:
        continue
      crop2_par = crop2.predecessors()
      if crop2_par[0].name == reshape2.name:
        data2 = crop2_par[1]
      else:
        data2 = crop2_par[0]
      if data2.type != 'DummyData':
        continue
      if not len(crop2.successors()) == 1:
        continue
      crop2_child = crop2.successors()[0]
      if crop1_child.type not in ['Eltwise', 'Gelu'] or crop2_child.type not in ['Eltwise', 'Gelu'] and crop1_child.type == crop2_child.type:
        continue
      if crop1_child.type == 'Eltwise':
        eltwise, gelu = crop1_child, crop2_child
      else:
        gelu, eltwise = crop1_child, crop2_child
      if not len(gelu.successors()) == 1:
        continue
      if gelu.successors()[0].name != eltwise.name:
        continue
      if not len(eltwise.successors()) == 1:
        continue
      reshape3 = eltwise.successors()[0]
      if reshape3.type != 'Reshape':
        continue
      if not len(reshape3.successors()) == 1:
        continue
      perm2 = reshape3.successors()[0]
      if perm2.type != 'Permute':
        continue
      if not len(perm2.successors()) == 1:
        continue
      reshape4 = perm2.successors()[0]
      if reshape4.type != 'Reshape':
        continue

      if perm1.permute_param.order != [0, 1, 3, 2]:
        continue
      if perm2.permute_param.order != [0, 1, 3, 2]:
        continue
      if eltwise.eltwise_param.operation != 0:
        continue
      if eltwise.eltwise_param.broadcast0 != [0.0, 0.0, 0.0]:
        continue
      if eltwise.eltwise_param.broadcast1 != [0.0, 0.0, 0.0]:
        continue

      def ll(aa):
        if len(aa) == 0:
          return None
        return [i for i in aa]

      conv2 = self.G.mk_conv_proto(conv.bottom[0], conv.name, ll(conv.convolution_param.kernel_size), conv.convolution_param.num_output // 2, ll(conv.convolution_param.stride), ll(conv.convolution_param.pad), conv.convolution_param.group, conv.convolution_param.bias_term, ll(conv.convolution_param.dilation))
      conv.convolution_param.num_output = conv.convolution_param.num_output // 2
      self.G.add_node(conv2)
      self.G.add_edge(conv.predecessors()[0], conv2)
      bnorm2 = self.G.mk_bnrm_proto(conv2.top[0], bnorm.name, bnorm.batch_norm_param.use_global_stats, bnorm.batch_norm_param.eps)
      self.G.add_node(bnorm2)
      self.G.add_edge(conv2, bnorm2)
      scale2 = self.G.mk_scale_proto(bnorm2.top[0], scale.name, scale.scale_param.bias_term)
      self.G.add_node(scale2)
      self.G.add_edge(bnorm2, scale2)
      self.G.add_edge(scale2, gelu)
      self.G.add_replace_edge_proto_new(scale2, gelu, gelu.bottom[0], scale2.top[0])
      self.G.add_edge(scale, eltwise)
      if gelu.top[0] == eltwise.bottom[0]:
        crop1_top = eltwise.bottom[1]
      else:
        crop1_top = eltwise.bottom[0]
      self.G.add_replace_edge_proto_new(scale, eltwise, crop1_top, scale.top[0])
      succs = list(reshape4.successors())
      for succ in succs:
        self.G.add_edge(eltwise, succ)
        self.G.add_replace_edge_proto_new(eltwise, succ, reshape4.top[0], eltwise.top[0])
      self.G.remove_node(reshape1)
      self.G.remove_node(perm1)
      self.G.remove_node(reshape2)
      self.G.remove_node(data1)
      self.G.remove_node(crop1)
      self.G.remove_node(crop2)
      self.G.remove_node(data2)
      self.G.remove_node(reshape3)
      self.G.remove_node(perm2)
      self.G.remove_node(reshape4)
      num_output = conv.convolution_param.num_output
      conv_wt_1 = np.copy(self.inp_wts[conv.name][0])
      self.inp_wts[conv.name] = [np.copy(conv_wt_1[:num_output])]
      self.inp_wts[conv2.name] = [np.copy(conv_wt_1[num_output:])]
      bnorm_wt_1 = np.copy(self.inp_wts[bnorm.name][0])
      bnorm_wt_2 = np.copy(self.inp_wts[bnorm.name][1])
      bnorm_wt_3 = np.copy(self.inp_wts[bnorm.name][2])
      self.inp_wts[bnorm.name] = [np.copy(bnorm_wt_1[:num_output]), np.copy(bnorm_wt_2[:num_output]), np.copy(bnorm_wt_3)]
      self.inp_wts[bnorm2.name] = [np.copy(bnorm_wt_1[num_output:]), np.copy(bnorm_wt_2[num_output:]), np.copy(bnorm_wt_3)]
      scale_wt_1 = np.copy(self.inp_wts[scale.name][0])
      scale_wt_2 = np.copy(self.inp_wts[scale.name][1])
      self.inp_wts[scale.name] = [np.copy(scale_wt_1[:num_output]), np.copy(scale_wt_2[:num_output])]
      self.inp_wts[scale2.name] = [np.copy(scale_wt_1[num_output:]), np.copy(scale_wt_2[num_output:])]


class GeGLuSplit_3(BasePass):
  def __init__(self):
    pass

  def do_transformation(self):
    layers = self.get_nodes_of_type('Convolution')
    for conv in layers:
      # check for valid pattern
      if not len(conv.successors()) == 1:
        continue
      reshape1 =  conv.successors()[0]
      if reshape1.type != 'Reshape':
        continue
      if not len(reshape1.successors()) == 1:
        continue
      perm1 = reshape1.successors()[0]
      if perm1.type != 'Permute':
        continue
      if not len(perm1.successors()) == 1:
        continue
      reshape2 = perm1.successors()[0]
      if reshape2.type != 'Reshape':
        continue
      if not len(reshape2.successors()) == 2:
        continue
      crop1, crop2 = reshape2.successors()
      if crop1.type != 'Crop':
        continue
      if not len(crop1.predecessors()) == 2:
        continue
      crop1_par = crop1.predecessors()
      if crop1_par[0].name == reshape2.name:
        data1 = crop1_par[1]
      else:
        data1 = crop1_par[0]
      if data1.type != 'DummyData':
        continue
      if not len(crop1.successors()) == 1:
        continue
      crop1_child = crop1.successors()[0]
      if crop2.type != 'Crop':
        continue
      if not len(crop2.predecessors()) == 2:
        continue
      crop2_par = crop2.predecessors()
      if crop2_par[0].name == reshape2.name:
        data2 = crop2_par[1]
      else:
        data2 = crop2_par[0]
      if data2.type != 'DummyData':
        continue
      if not len(crop2.successors()) == 1:
        continue
      crop2_child = crop2.successors()[0]
      if crop1_child.type not in ['Eltwise', 'Gelu'] or crop2_child.type not in ['Eltwise', 'Gelu'] and crop1_child.type == crop2_child.type:
        continue
      if crop1_child.type == 'Eltwise':
        eltwise, gelu = crop1_child, crop2_child
      else:
        gelu, eltwise = crop1_child, crop2_child
      if not len(gelu.successors()) == 1:
        continue
      if gelu.successors()[0].name != eltwise.name:
        continue
      if not len(eltwise.successors()) == 1:
        continue
      reshape3 = eltwise.successors()[0]
      if reshape3.type != 'Reshape':
        continue
      if not len(reshape3.successors()) == 1:
        continue
      perm2 = reshape3.successors()[0]
      if perm2.type != 'Permute':
        continue
      if not len(perm2.successors()) == 1:
        continue
      reshape4 = perm2.successors()[0]
      if reshape4.type != 'Reshape':
        continue

      if perm1.permute_param.order != [0, 1, 3, 2]:
        continue
      if perm2.permute_param.order != [0, 1, 3, 2]:
        continue
      if eltwise.eltwise_param.operation != 0:
        continue
      if eltwise.eltwise_param.broadcast0 != [0.0, 0.0, 0.0]:
        continue
      if eltwise.eltwise_param.broadcast1 != [0.0, 0.0, 0.0]:
        continue

      def ll(aa):
        if len(aa) == 0:
          return None
        return [i for i in aa]

      conv2 = self.G.mk_conv_proto(conv.bottom[0], conv.name, ll(conv.convolution_param.kernel_size), conv.convolution_param.num_output // 2, ll(conv.convolution_param.stride), ll(conv.convolution_param.pad), conv.convolution_param.group, conv.convolution_param.bias_term, ll(conv.convolution_param.dilation))
      conv.convolution_param.num_output = conv.convolution_param.num_output // 2
      self.G.add_node(conv2)
      self.G.add_edge(conv.predecessors()[0], conv2)
      self.G.add_edge(conv2, gelu)
      self.G.add_replace_edge_proto_new(conv2, gelu, gelu.bottom[0], conv2.top[0])
      self.G.add_edge(conv, eltwise)
      if gelu.top[0] == eltwise.bottom[0]:
        crop1_top = eltwise.bottom[1]
      else:
        crop1_top = eltwise.bottom[0]
      self.G.add_replace_edge_proto_new(conv, eltwise, crop1_top, conv.top[0])
      succs = list(reshape4.successors())
      for succ in succs:
        self.G.add_edge(eltwise, succ)
        self.G.add_replace_edge_proto_new(eltwise, succ, reshape4.top[0], eltwise.top[0])
      self.G.remove_node(reshape1)
      self.G.remove_node(perm1)
      self.G.remove_node(reshape2)
      self.G.remove_node(data1)
      self.G.remove_node(crop1)
      self.G.remove_node(crop2)
      self.G.remove_node(data2)
      self.G.remove_node(reshape3)
      self.G.remove_node(perm2)
      self.G.remove_node(reshape4)
      num_output = conv.convolution_param.num_output
      conv_wt_1 = np.copy(self.inp_wts[conv.name][0])
      bias_true = len(self.inp_wts[conv.name]) > 1
      if bias_true:
        conv_wt_2 = np.copy(self.inp_wts[conv.name][1])
      self.inp_wts[conv.name] = [np.copy(conv_wt_1[:num_output])]
      self.inp_wts[conv2.name] = [np.copy(conv_wt_1[num_output:])]
      if bias_true:
        self.inp_wts[conv.name].append(np.copy(conv_wt_2[:num_output]))
        self.inp_wts[conv2.name].append(np.copy(conv_wt_2[num_output:]))


class MultiHeadAttentionSplit(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    SIZE = -1
    layers = self.get_nodes_of_type('Softmax')
    for softmax0 in layers:
      if not len(softmax0.successors()) == 1:
        continue
      bdot2 = softmax0.successors()[0]
      if bdot2.type != 'BatchDot':
        continue
      if not len(softmax0.predecessors()) == 1:
        continue
      bdot1 = softmax0.predecessors()[0]
      if bdot1.type != 'BatchDot':
        continue
      if not len(bdot1.predecessors()) == 2:
        continue
      reshape12, reshape22 = bdot1.predecessors()
      if reshape12.type != 'Reshape' or reshape22.type != 'Reshape':
        continue
      if reshape12.top[0] != bdot1.bottom[0]:
        reshape12, reshape22 = reshape22, reshape12
      if not len(reshape12.predecessors()) == 1:
        continue
      conv1 = reshape12.predecessors()[0]
      if conv1.type != 'Convolution':
        continue
      if not len(conv1.predecessors()) == 1:
        continue
      reshape11 = conv1.predecessors()[0]
      if reshape11.type != 'Reshape':
        continue
      if not len(reshape22.predecessors()) == 1:
        continue
      conv2 = reshape22.predecessors()[0]
      if conv2.type != 'Convolution':
        continue
      if not len(conv2.predecessors()) == 1:
        continue
      reshape21 = conv2.predecessors()[0]
      if reshape21.type != 'Reshape':
        continue
      tmp1, tmp2 = bdot2.predecessors()
      if tmp1.top[0] != bdot2.bottom[0]:
        tmp1, tmp2 = tmp2, tmp1
      if tmp1.type != 'Reshape':
        continue
      reshape33 = tmp1
      if not len(reshape33.predecessors()) == 1:
        continue
      perm31 = reshape33.predecessors()[0]
      if perm31.type != 'Permute':
        continue
      if not len(perm31.predecessors()) == 1:
        continue
      reshape32 = perm31.predecessors()[0]
      if reshape32.type != 'Reshape':
        continue
      if not len(reshape32.predecessors()) == 1:
        continue
      conv3 = reshape32.predecessors()[0]
      if conv3.type != 'Convolution':
        continue
      if not len(conv3.predecessors()) == 1:
        continue
      reshape31 = conv3.predecessors()[0]
      if reshape31.type != 'Reshape':
        continue
      if not len(bdot2.successors()) == 1:
        continue
      reshape_out = bdot2.successors()[0]
      if reshape_out.type != 'Reshape':
        continue
      # checks
      if not (conv1.convolution_param.num_output == conv2.convolution_param.num_output == conv3.convolution_param.num_output):
        continue
      if conv1.convolution_param.num_output % 8 != 0:
        continue
      if not (reshape11.name == reshape21.name == reshape31.name):
        continue
      in_size = 1
      for i in reshape11.reshape_param.shape.dim:
        in_size *= i
      if in_size < SIZE:
        continue
      # modifying the network
      # self.G.add_edge(reshape11, conv2)
      # self.G.add_replace_edge_proto_new(reshape11, conv2, reshape21.top[0], reshape11.top[0])
      # self.G.add_edge(reshape11, conv3)
      # self.G.add_replace_edge_proto_new(reshape11, conv3, reshape31.top[0], reshape11.top[0])
      # self.G.remove_node(reshape21)
      # self.G.remove_node(reshape31)
      concat = self.G.mk_concat_proto(bdot2.top[0], bdot2.name, new_axis=0)
      concat.bottom.pop()
      self.G.add_node(concat)
      self.G.remove_edge(bdot2, reshape_out)
      self.G.add_edge(concat, reshape_out)
      self.G.add_replace_edge_proto_new(concat, reshape_out, bdot2.top[0], concat.top[0])
      num_output = conv3.convolution_param.num_output // 8
      conv1.convolution_param.num_output = conv2.convolution_param.num_output = conv3.convolution_param.num_output = num_output
      batch = reshape12.reshape_param.shape.dim[0] // 8
      reshape12.reshape_param.shape.dim[0] = reshape22.reshape_param.shape.dim[0] = reshape32.reshape_param.shape.dim[1] = reshape33.reshape_param.shape.dim[0] = batch

      def ll(aa):
        if len(aa) == 0:
          return None
        return [i for i in aa]

      for i in range(8):
        conv_new = []
        prefix = 'set' + str(i) + '_'
        conv_new.append(self.G.mk_conv_proto(conv1.bottom[0], prefix + conv1.name, ll(conv1.convolution_param.kernel_size), num_output, ll(conv1.convolution_param.stride), ll(conv1.convolution_param.pad), conv1.convolution_param.group, conv1.convolution_param.bias_term, ll(conv1.convolution_param.dilation)))
        self.G.add_node(conv_new[-1])
        self.G.add_edge(reshape11, conv_new[-1])
        conv_new.append(self.G.mk_conv_proto(conv2.bottom[0], prefix + conv2.name, ll(conv2.convolution_param.kernel_size), num_output, ll(conv2.convolution_param.stride), ll(conv2.convolution_param.pad), conv2.convolution_param.group, conv2.convolution_param.bias_term, ll(conv2.convolution_param.dilation)))
        self.G.add_node(conv_new[-1])
        self.G.add_edge(reshape11, conv_new[-1])
        conv_new.append(self.G.mk_conv_proto(conv3.bottom[0], prefix + conv3.name, ll(conv3.convolution_param.kernel_size), num_output, ll(conv3.convolution_param.stride), ll(conv3.convolution_param.pad), conv3.convolution_param.group, conv3.convolution_param.bias_term, ll(conv3.convolution_param.dilation)))
        self.G.add_node(conv_new[-1])
        self.G.add_edge(reshape11, conv_new[-1])
        cnv_wt_1 = np.copy(self.inp_wts[conv1.name][0])
        self.inp_wts[conv_new[0].name] = [np.copy(cnv_wt_1[(i) * num_output:(i + 1) * num_output])]
        cnv_wt_2 = np.copy(self.inp_wts[conv2.name][0])
        self.inp_wts[conv_new[1].name] = [np.copy(cnv_wt_2[(i) * num_output:(i + 1) * num_output])]
        cnv_wt_3 = np.copy(self.inp_wts[conv3.name][0])
        self.inp_wts[conv_new[2].name] = [np.copy(cnv_wt_3[(i) * num_output:(i + 1) * num_output])]
        reshape_new1 = self.G.mk_reshape_proto(conv_new[0].top[0], ll(reshape12.reshape_param.shape.dim), prefix + reshape12.name)
        reshape_new2 = self.G.mk_reshape_proto(conv_new[1].top[0], ll(reshape22.reshape_param.shape.dim), prefix + reshape22.name)
        reshape_new3 = self.G.mk_reshape_proto(conv_new[2].top[0], ll(reshape32.reshape_param.shape.dim), prefix + reshape32.name)
        self.G.add_node(reshape_new1)
        self.G.add_node(reshape_new2)
        self.G.add_node(reshape_new3)
        self.G.add_edge(conv_new[0], reshape_new1)
        self.G.add_edge(conv_new[1], reshape_new2)
        self.G.add_edge(conv_new[2], reshape_new3)
        permute_new = self.G.mk_permute_proto(prefix + perm31.name, ll(perm31.permute_param.order), bottom=reshape_new3.top[0])
        self.G.add_node(permute_new)
        self.G.add_edge(reshape_new3, permute_new)
        reshape_new4 = self.G.mk_reshape_proto(permute_new.top[0], ll(reshape33.reshape_param.shape.dim), prefix + reshape33.name)
        self.G.add_node(reshape_new4)
        self.G.add_edge(permute_new, reshape_new4)
        bdot1_new = self.G.mk_bdot_proto([reshape_new1.top[0], reshape_new2.top[0]], prefix + bdot1.name, bdot1.batch_dot_param.transpose_a, bdot1.batch_dot_param.transpose_b, bdot1.batch_dot_param.multiplier)
        self.G.add_node(bdot1_new)
        self.G.add_edge(reshape_new1, bdot1_new)
        self.G.add_edge(reshape_new2, bdot1_new)
        softmax_new = self.G.mk_softmax_proto(bdot1_new.top[0], prefix + softmax0.name, softmax0.softmax_param.axis)
        self.G.add_node(softmax_new)
        self.G.add_edge(bdot1_new, softmax_new)
        bdot2_new = self.G.mk_bdot_proto([reshape_new4.top[0], softmax_new.top[0]], prefix + bdot2.name, bdot2.batch_dot_param.transpose_a, bdot2.batch_dot_param.transpose_b, bdot2.batch_dot_param.multiplier)
        self.G.add_node(bdot2_new)
        self.G.add_edge(reshape_new4, bdot2_new)
        self.G.add_edge(softmax_new, bdot2_new)
        dvshift0 = self.G.mk_dvshift_proto(bdot2_new.top[0], prefix)
        self.G.add_node(dvshift0)
        self.G.add_edge(bdot2_new, dvshift0)
        self.G.add_edge(dvshift0, concat)
        concat.bottom.append(dvshift0.top[0])
      self.inp_wts.pop(conv1.name)
      self.inp_wts.pop(conv2.name)
      self.inp_wts.pop(conv3.name)
      self.G.remove_node(conv1)
      self.G.remove_node(conv2)
      self.G.remove_node(conv3)
      self.G.remove_node(reshape12)
      self.G.remove_node(reshape22)
      self.G.remove_node(reshape32)
      self.G.remove_node(perm31)
      self.G.remove_node(reshape33)
      self.G.remove_node(bdot1)
      self.G.remove_node(bdot2)
      self.G.remove_node(softmax0)


class MultiHeadAttentionSplit_LLM(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    layers = self.get_nodes_of_type('AttentionSoftmax')
    for attn_sfmx in layers:
      input_shape = self.caffe_net.blobs[attn_sfmx.bottom[0]].data.shape
      input_size = 1
      for dim in input_shape:
        input_size *= dim
      if input_size >= pow(2,25): # We need to split the pattern into eight if the size of input to attention softmax is above 32 MB so that it is stored in L2 Cache
        splits_num = 8
      elif input_size >= pow(2,24): # We need to split the pattern into four if the size of input to attention softmax is above 16 MB MB so that it is stored in L2 Cache
        splits_num = 4
      elif input_size >= pow(2,23): # We need to split the pattern into two if the size of input to attention softmax is above 8 MB so that it is stored in L2 Cache
        splits_num = 2
      else:
        continue
      if not len(attn_sfmx.successors()) == 1:
        continue
      bdot2 = attn_sfmx.successors()[0]
      if bdot2.type != 'BatchDot':
        continue
      if not len(attn_sfmx.predecessors()) == 2:
        continue
      bdot1 = attn_sfmx.predecessors()[0] if attn_sfmx.predecessors()[0].type == 'BatchDot' else attn_sfmx.predecessors()[1]
      if bdot1.type != 'BatchDot':
        continue
      if not len(bdot1.predecessors()) == 2:
        continue
      eltwise1, eltwise2 = bdot1.predecessors()
      if eltwise1.type != 'Eltwise' or eltwise2.type!= 'Eltwise':
        continue
      if not len(eltwise1.predecessors()) == 2 or not len(eltwise2.predecessors()) == 2:
        continue
      if eltwise1.top[0] != bdot1.bottom[0]:
        eltwise1, eltwise2 = eltwise2, eltwise1
      #K Cache
      if len(eltwise1.successors()) != 2:
        continue
      reshape_k = eltwise1.successors()[0] if eltwise1.successors()[0].type == 'Reshape' else eltwise1.successors()[1]
      if reshape_k.type != 'Reshape':
        continue
      if len(reshape_k.successors()) != 1:
        continue
      #K Cache permute
      permute_k = reshape_k.successors()[0]
      if permute_k.type != 'Permute':
        continue
      #Adding K Cache Concat
      concat_k = self.G.mk_concat_proto(eltwise1.top[0], eltwise1.name, new_axis=0)
      concat_k.bottom.pop()
      self.G.add_node(concat_k)
      self.G.remove_edge(eltwise1, reshape_k)
      self.G.add_edge(concat_k, reshape_k)
      self.G.add_replace_edge_proto_new(concat_k, reshape_k, eltwise1.top[0], concat_k.top[0])
      reshape1 = eltwise1.predecessors()[0] if eltwise1.predecessors()[0].type == 'Reshape' else eltwise1.predecessors()[1]
      reshape2 = eltwise2.predecessors()[0] if eltwise2.predecessors()[0].type == 'Reshape' else eltwise2.predecessors()[1]
      if reshape1.type != 'Reshape' or reshape2.type != 'Reshape':
        continue
      if not len(reshape1.predecessors()) == 1:
        continue
      scale1 = reshape1.predecessors()[0]
      if scale1.type != 'Scale':
        continue
      if not len(scale1.predecessors()) == 1:
        continue
      batchnorm1 = scale1.predecessors()[0]
      if not len(batchnorm1.predecessors()) == 1:
        continue
      conv1 = batchnorm1.predecessors()[0]
      if conv1.type != 'Convolution':
        continue
      if not len(conv1.predecessors()) == 1:
        continue
      conv1_parent = conv1.predecessors()[0]
      if not len(reshape2.predecessors()) == 1:
        continue
      scale2 = reshape2.predecessors()[0]
      if scale2.type != 'Scale':
        continue
      if not len(scale2.predecessors()) == 1:
        continue
      batchnorm2 = scale2.predecessors()[0]
      if not len(batchnorm2.predecessors()) == 1:
        continue
      conv2 = batchnorm2.predecessors()[0]
      if conv2.type != 'Convolution':
        continue
      if not len(conv2.predecessors()) == 1:
        continue
      conv2_parent = conv2.predecessors()[0]
      tmp1, tmp2 = bdot2.predecessors()
      if tmp1.top[0] != bdot2.bottom[0]:
        tmp1, tmp2 = tmp2, tmp1
      if tmp1.type != 'PermuteNnp':
        continue
      permutennp = tmp1
      if not len(permutennp.predecessors()) == 1:
        continue
      reshape3 = permutennp.predecessors()[0]
      if reshape3.type != 'Reshape':
        continue
      if not len(reshape3.predecessors()) == 1:
        continue
      scale3 = reshape3.predecessors()[0]
      if scale3.type != 'Scale':
        continue
      if not len(scale3.predecessors()) == 1:
        continue
      #V Cache
      if not len(scale3.successors()) == 2:
        continue
      reshape_v = scale3.successors()[0] if scale3.successors()[0].name != reshape3.name else scale3.successors()[1]
      if reshape_v.type != 'Reshape':
        continue
      if len(reshape_v.successors()) != 1:
        continue
      #V Cache permute
      permute_v = reshape_v.successors()[0]
      if permute_v.type != 'Permute':
        continue
      #Adding V Cache Concat
      concat_v = self.G.mk_concat_proto(scale3.top[0], scale3.name, new_axis=0)
      concat_v.bottom.pop()
      self.G.add_node(concat_v)
      self.G.remove_edge(scale3, reshape_v)
      self.G.add_edge(concat_v, reshape_v)
      self.G.add_replace_edge_proto_new(concat_v, reshape_v, scale3.top[0], concat_v.top[0])
      batchnorm3 = scale3.predecessors()[0]
      if not len(batchnorm3.predecessors()) == 1:
        continue
      conv3 = batchnorm3.predecessors()[0]
      if conv3.type != 'Convolution':
        continue
      if not len(conv3.predecessors()) == 1:
        continue
      conv3_parent = conv3.predecessors()[0]
      if not len(bdot2.successors()) == 1:
        continue
      reshape_out = bdot2.successors()[0]
      if reshape_out.type != 'Reshape':
        continue
      # checks
      if not (conv1.convolution_param.num_output == conv2.convolution_param.num_output == conv3.convolution_param.num_output):
        continue
      if conv1.convolution_param.num_output % 8 != 0:
        continue
      if not (conv1_parent.name == conv2_parent.name == conv3_parent.name):
        continue
      concat = self.G.mk_concat_proto(bdot2.top[0], bdot2.name, new_axis=0)
      concat.bottom.pop()
      self.G.add_node(concat)
      self.G.remove_edge(bdot2, reshape_out)
      self.G.add_edge(concat, reshape_out)
      self.G.add_replace_edge_proto_new(concat, reshape_out, bdot2.top[0], concat.top[0])
      num_output = conv3.convolution_param.num_output // splits_num
      conv1.convolution_param.num_output = conv2.convolution_param.num_output = conv3.convolution_param.num_output = num_output
      batch = reshape1.reshape_param.shape.dim[1] // splits_num
      reshape1.reshape_param.shape.dim[1] = reshape2.reshape_param.shape.dim[1] = reshape3.reshape_param.shape.dim[1] = batch
      def ll(aa):
        return [i for i in aa]

      for i in range(splits_num):
        conv_new = []
        prefix = 'set' + str(i) + '_'
        #Add Convs. TODO: Confirm if attributes are correctly passed
        conv_new.append(self.G.mk_conv_proto(conv1.bottom[0], prefix + conv1.name, ll(conv1.convolution_param.kernel_size), num_output, bias_term = conv1.convolution_param.bias_term))
        self.G.add_node(conv_new[-1])
        self.G.add_edge(conv1_parent, conv_new[-1])
        conv_new.append(self.G.mk_conv_proto(conv2.bottom[0], prefix + conv2.name, ll(conv2.convolution_param.kernel_size), num_output, bias_term = conv2.convolution_param.bias_term))
        self.G.add_node(conv_new[-1])
        self.G.add_edge(conv1_parent, conv_new[-1])
        conv_new.append(self.G.mk_conv_proto(conv3.bottom[0], prefix + conv3.name, ll(conv3.convolution_param.kernel_size), num_output, bias_term = conv3.convolution_param.bias_term))
        self.G.add_node(conv_new[-1])
        self.G.add_edge(conv1_parent, conv_new[-1])
        cnv_wt_1 = np.copy(self.inp_wts[conv1.name][0])
        self.inp_wts[conv_new[0].name] = [np.copy(cnv_wt_1[(i) * num_output:(i + 1) * num_output])]
        cnv_wt_2 = np.copy(self.inp_wts[conv2.name][0])
        self.inp_wts[conv_new[1].name] = [np.copy(cnv_wt_2[(i) * num_output:(i + 1) * num_output])]
        cnv_wt_3 = np.copy(self.inp_wts[conv3.name][0])
        self.inp_wts[conv_new[2].name] = [np.copy(cnv_wt_3[(i) * num_output:(i + 1) * num_output])]
        #Add Batchnorms
        batchnorm1_new = self.G.mk_bnrm_proto(conv_new[0].top[0], prefix + batchnorm1.name, batchnorm1.batch_norm_param.use_global_stats, batchnorm1.batch_norm_param.eps)
        batchnorm2_new = self.G.mk_bnrm_proto(conv_new[1].top[0], prefix + batchnorm2.name, batchnorm2.batch_norm_param.use_global_stats, batchnorm2.batch_norm_param.eps)
        batchnorm3_new = self.G.mk_bnrm_proto(conv_new[2].top[0], prefix + batchnorm3.name, batchnorm3.batch_norm_param.use_global_stats, batchnorm3.batch_norm_param.eps)
        self.G.add_node(batchnorm1_new)
        self.G.add_node(batchnorm2_new)
        self.G.add_node(batchnorm3_new)
        self.G.add_edge(conv_new[0], batchnorm1_new)
        self.G.add_edge(conv_new[1], batchnorm2_new)
        self.G.add_edge(conv_new[2], batchnorm3_new)
        batchnorm_wts_1 = self.inp_wts[batchnorm1.name]
        self.inp_wts[batchnorm1_new.name] = [np.copy(batchnorm_wts_1[0][(i) * num_output:(i + 1) * num_output]), np.copy(batchnorm_wts_1[1][(i) * num_output:(i + 1) * num_output]), np.copy(batchnorm_wts_1[2])]
        batchnorm_wts_2 = self.inp_wts[batchnorm2.name]
        self.inp_wts[batchnorm2_new.name] = [np.copy(batchnorm_wts_2[0][(i) * num_output:(i + 1) * num_output]), np.copy(batchnorm_wts_2[1][(i) * num_output:(i + 1) * num_output]), np.copy(batchnorm_wts_2[2])]
        batchnorm_wts_3 = self.inp_wts[batchnorm3.name]
        self.inp_wts[batchnorm3_new.name] = [np.copy(batchnorm_wts_3[0][(i) * num_output:(i + 1) * num_output]), np.copy(batchnorm_wts_3[1][(i) * num_output:(i + 1) * num_output]), np.copy(batchnorm_wts_3[2])]  
        #Add Scales
        scale1_new = self.G.mk_scale_proto(batchnorm1_new.top[0], prefix + scale1.name, scale1.scale_param.bias_term)
        scale2_new = self.G.mk_scale_proto(batchnorm2_new.top[0], prefix + scale2.name, scale2.scale_param.bias_term)
        scale3_new = self.G.mk_scale_proto(batchnorm3_new.top[0], prefix + scale3.name, scale3.scale_param.bias_term)
        self.G.add_node(scale1_new)
        self.G.add_node(scale2_new)
        self.G.add_node(scale3_new)
        self.G.add_edge(batchnorm1_new, scale1_new)
        self.G.add_edge(batchnorm2_new, scale2_new)
        self.G.add_edge(batchnorm3_new, scale3_new)
        scale_wt_1 = self.inp_wts[scale1.name]
        self.inp_wts[scale1_new.name] = [np.copy(scale_wt_1[0][(i) * num_output:(i + 1) * num_output]), np.copy(scale_wt_1[1][(i) * num_output:(i + 1) * num_output])]
        scale_wt_2 = self.inp_wts[scale2.name]
        self.inp_wts[scale2_new.name] = [np.copy(scale_wt_2[0][(i) * num_output:(i + 1) * num_output]), np.copy(scale_wt_2[1][(i) * num_output:(i + 1) * num_output])]
        scale_wt_3 = self.inp_wts[scale3.name]
        self.inp_wts[scale3_new.name] = [np.copy(scale_wt_3[0][(i) * num_output:(i + 1) * num_output]), np.copy(scale_wt_3[1][(i) * num_output:(i + 1) * num_output])]
        #Add Reshapes
        reshape_new1 = self.G.mk_reshape_proto(scale1_new.top[0], ll(reshape1.reshape_param.shape.dim), prefix + reshape1.name)
        reshape_new2 = self.G.mk_reshape_proto(scale2_new.top[0], ll(reshape2.reshape_param.shape.dim), prefix + reshape2.name)
        reshape_new3 = self.G.mk_reshape_proto(scale3_new.top[0], ll(reshape3.reshape_param.shape.dim), prefix + reshape3.name)
        self.G.add_node(reshape_new1)
        self.G.add_node(reshape_new2)
        self.G.add_node(reshape_new3)
        self.G.add_edge(scale1_new, reshape_new1)
        self.G.add_edge(scale2_new, reshape_new2)
        self.G.add_edge(scale3_new, reshape_new3)
        #Add eltwises
        eltwise1_new = self.G.mk_eltwise_proto([reshape_new1.top[0], eltwise1.bottom[1]], prefix + eltwise1.name, operation = eltwise1.eltwise_param.operation, broadcast1=ll(eltwise1.eltwise_param.broadcast1))
        eltwise2_new = self.G.mk_eltwise_proto([reshape_new2.top[0], eltwise2.bottom[1]], prefix + eltwise2.name, operation = eltwise2.eltwise_param.operation, broadcast1=ll(eltwise2.eltwise_param.broadcast1))
        self.G.add_node(eltwise1_new)
        self.G.add_node(eltwise2_new)
        self.G.add_edge(reshape_new1, eltwise1_new)
        self.G.add_edge(reshape_new2, eltwise2_new)
        #Add Permutennp
        permutennp_new = self.G.mk_permute_nnp_proto(prefix + permutennp.name, reshape_new3.top[0], ll(permutennp.permute_nnp_param.order))
        self.G.add_node(permutennp_new)
        self.G.add_edge(reshape_new3, permutennp_new)
        #Add Bdot1
        bdot1_new = self.G.mk_bdot_proto([eltwise1_new.top[0], eltwise2_new.top[0]], prefix + bdot1.name, bdot1.batch_dot_param.transpose_a, bdot1.batch_dot_param.transpose_b, bdot1.batch_dot_param.multiplier)
        self.G.add_node(bdot1_new)
        self.G.add_edge(eltwise1_new, bdot1_new)
        self.G.add_edge(eltwise2_new, bdot1_new)
        #Add Softmax
        attn_sfmx_new = self.G.mk_attentionsoftmax_proto([bdot1_new.top[0], attn_sfmx.bottom[1]], prefix + attn_sfmx.name, attn_sfmx.attentionsoftmax_param.axis)
        self.G.add_node(attn_sfmx_new)
        self.G.add_edge(bdot1_new, attn_sfmx_new)
        #Add Bdot2
        bdot2_new = self.G.mk_bdot_proto([permutennp_new.top[0], attn_sfmx_new.top[0]], prefix + bdot2.name, bdot2.batch_dot_param.transpose_a, bdot2.batch_dot_param.transpose_b, bdot2.batch_dot_param.multiplier)
        self.G.add_node(bdot2_new)
        self.G.add_edge(permutennp_new, bdot2_new)
        self.G.add_edge(attn_sfmx_new, bdot2_new)
        # Add edges to concat_k for k_cache
        dvshift_k = self.G.mk_dvshift_proto(eltwise1_new.top[0], prefix + 'k_cache_')
        self.G.add_node(dvshift_k)
        self.G.add_edge(eltwise1_new, dvshift_k)
        self.G.add_edge(dvshift_k, concat_k)
        concat_k.bottom.append(dvshift_k.top[0])
        #Add edges to concat_v for v_cache
        dvshift_v = self.G.mk_dvshift_proto(scale3_new.top[0], prefix + 'v_cache_')
        self.G.add_node(dvshift_v)
        self.G.add_edge(scale3_new, dvshift_v)
        self.G.add_edge(dvshift_v, concat_v)
        concat_v.bottom.append(dvshift_v.top[0])
        #Add edges to concat for bdot2 output
        dvshift0 = self.G.mk_dvshift_proto(bdot2_new.top[0], prefix)
        self.G.add_node(dvshift0)
        self.G.add_edge(bdot2_new, dvshift0)
        self.G.add_edge(dvshift0, concat)
        concat.bottom.append(dvshift0.top[0])
      self.inp_wts.pop(conv1.name)
      self.inp_wts.pop(conv2.name)
      self.inp_wts.pop(conv3.name)
      self.inp_wts.pop(batchnorm1.name)
      self.inp_wts.pop(batchnorm2.name)
      self.inp_wts.pop(batchnorm3.name)
      self.inp_wts.pop(scale1.name)
      self.inp_wts.pop(scale2.name)
      self.inp_wts.pop(scale3.name)
      self.G.remove_node(conv1)
      self.G.remove_node(conv2)
      self.G.remove_node(conv3)
      self.G.remove_node(batchnorm1)
      self.G.remove_node(batchnorm2)
      self.G.remove_node(batchnorm3)
      self.G.remove_node(scale1)
      self.G.remove_node(scale2)
      self.G.remove_node(scale3)
      self.G.remove_node(reshape1)
      self.G.remove_node(reshape2)
      self.G.remove_node(reshape3)
      self.G.remove_node(eltwise1)
      self.G.remove_node(eltwise2)
      self.G.remove_node(permutennp)
      self.G.remove_node(bdot1)
      self.G.remove_node(bdot2)
      self.G.remove_node(attn_sfmx)

class UnetPattern2_2(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    SIZE = -1
    layers = self.get_nodes_of_type('Softmax')
    for softmax0 in layers:
      if not len(softmax0.successors()) == 1:
        continue
      bdot2 = softmax0.successors()[0]
      if bdot2.type != 'BatchDot':
        continue
      if not len(softmax0.predecessors()) == 1:
        continue
      bdot1 = softmax0.predecessors()[0]
      if bdot1.type != 'BatchDot':
        continue
      if not len(bdot1.predecessors()) == 2:
        continue
      reshape12, reshape22 = bdot1.predecessors()
      if reshape12.type != 'Reshape' or reshape22.type != 'Reshape':
        continue
      if reshape22.top[0] != bdot1.bottom[0]:
        reshape12, reshape22 = reshape22, reshape12
      if not len(reshape12.predecessors()) == 1:
        continue
      conv1 = reshape12.predecessors()[0]
      if conv1.type != 'Convolution':
        continue
      if not len(conv1.predecessors()) == 1:
        continue
      reshape11 = conv1.predecessors()[0]
      if reshape11.type != 'Reshape':
        continue
      if not len(reshape22.predecessors()) == 1:
        continue
      conv2 = reshape22.predecessors()[0]
      if conv2.type != 'Convolution':
        continue
      if not len(conv2.predecessors()) == 1:
        continue
      reshape21 = conv2.predecessors()[0]
      if reshape21.type != 'Reshape':
        continue
      tmp1, tmp2 = bdot2.predecessors()
      if tmp1.top[0] != bdot2.bottom[0]:
        tmp1, tmp2 = tmp2, tmp1
      if tmp1.type != 'Reshape':
        continue
      reshape33 = tmp1
      if not len(reshape33.predecessors()) == 1:
        continue
      perm31 = reshape33.predecessors()[0]
      if perm31.type != 'Permute':
        continue
      if not len(perm31.predecessors()) == 1:
        continue
      reshape32 = perm31.predecessors()[0]
      if reshape32.type != 'Reshape':
        continue
      if not len(reshape32.predecessors()) == 1:
        continue
      conv3 = reshape32.predecessors()[0]
      if conv3.type != 'Convolution':
        continue
      if not len(conv3.predecessors()) == 1:
        continue
      reshape31 = conv3.predecessors()[0]
      if reshape31.type != 'Reshape':
        continue
      if not len(bdot2.successors()) == 1:
        continue
      dvshift_out = bdot2.successors()[0]
      if dvshift_out.type != 'DVShift':
        continue
      # checks
      if not (self.caffe_net.blobs[conv1.bottom[0]].data.shape[3] == self.caffe_net.blobs[conv2.bottom[0]].data.shape[3] == self.caffe_net.blobs[conv3.bottom[0]].data.shape[3]):
        continue
      if self.caffe_net.blobs[conv1.bottom[0]].data.shape[3] * self.caffe_net.blobs[conv2.bottom[0]].data.shape[3] <= 5308416:
        continue
      if self.caffe_net.blobs[conv1.bottom[0]].data.shape[3] % 4 != 0:
        continue
      if not (reshape11.name == reshape21.name == reshape31.name):
        continue
      in_size = 1
      for i in reshape11.reshape_param.shape.dim:
        in_size *= i
      if in_size < SIZE:
        continue
      split_axis_shape = self.caffe_net.blobs[reshape12.top[0]].data.shape[2]
      interval_size = split_axis_shape // 4
      slice_point = [i * interval_size for i in range(1, 4)]
      axis = 2
      dim_along_axis = self.caffe_net.blobs[reshape12.top[0]].data.shape[axis]
      slice = self.G.mk_slice_proto(reshape12.top[0], reshape12.name, dim_along_axis, axis=axis, slice_point=slice_point)
      self.G.add_node(slice)
      self.G.remove_edge(reshape12, bdot1)
      self.G.add_edge(reshape12, slice)
      concat = self.G.mk_concat_proto(bdot2.top[0], bdot2.name, new_axis=2)
      concat.bottom.pop()
      self.G.add_node(concat)
      self.G.remove_edge(bdot2, dvshift_out)
      self.G.add_edge(concat, dvshift_out)
      self.G.add_replace_edge_proto_new(concat, dvshift_out, bdot2.top[0], concat.top[0])

      for i in range(4):
        suffix = '_' + str(i)
        prefix = str(i) + '_'
        bdot1_new = self.G.mk_bdot_proto([reshape22.top[0], slice.top[i]], bdot1.name + suffix, bdot1.batch_dot_param.transpose_a, bdot1.batch_dot_param.transpose_b, bdot1.batch_dot_param.multiplier)
        self.G.add_node(bdot1_new)
        self.G.add_edge(slice, bdot1_new)
        self.G.add_edge(reshape22, bdot1_new)
        softmax_new = self.G.mk_softmax_proto(bdot1_new.top[0], softmax0.name + suffix, softmax0.softmax_param.axis)
        self.G.add_node(softmax_new)
        self.G.add_edge(bdot1_new, softmax_new)
        bdot2_new = self.G.mk_bdot_proto([reshape33.top[0], softmax_new.top[0]], bdot2.name + suffix, bdot2.batch_dot_param.transpose_a, bdot2.batch_dot_param.transpose_b, bdot2.batch_dot_param.multiplier)
        self.G.add_node(bdot2_new)
        self.G.add_edge(reshape33, bdot2_new)
        self.G.add_edge(softmax_new, bdot2_new)
        dvshift0 = self.G.mk_dvshift_proto(bdot2_new.top[0], prefix)
        self.G.add_node(dvshift0)
        self.G.add_edge(bdot2_new, dvshift0)
        self.G.add_edge(dvshift0, concat)
        concat.bottom.append(dvshift0.top[0])
      self.G.remove_node(bdot1)
      self.G.remove_node(bdot2)
      self.G.remove_node(softmax0)

class ClipVitBdotPerm(BasePass):
  def __init__(self):
    pass

  def is_hw_permute(self, perm):
    n = len(perm)

    if n < 2:
        return False

    expected = list(range(n))
    expected[-2], expected[-1] = expected[-1], expected[-2]

    return perm == expected

  def ll(self, aa):
    if len(aa) == 0:
      return None
    return [i for i in aa]

  def do_transformation(self):
    bdot_layers = self.get_nodes_of_type('BatchDot')
    for layer in bdot_layers:
      if not len(layer.successors()) == 1:
        continue
      layer_pred = layer.predecessors()
      if not len(layer_pred) == 2:
        continue
      found = -1
      permute_node = None
      for idx, pred in enumerate(layer_pred):
        if pred.type == 'PermuteNnp':
          found = idx
          permute_node = pred
          break
      if found == -1:
        continue
      if layer_pred[found^1].type != 'Softmax':
        continue
      if permute_node is not None:
        reshape_node = permute_node.predecessors()
        if len(reshape_node) != 1 or reshape_node[0].type != 'Reshape':
          continue
        reshape_shape = list(self.caffe_net.blobs[reshape_node[0].top[0]].data.shape)
        if len(reshape_shape) == 3:
          dims = reshape_node[0].reshape_param.shape.dim
          reshape_shape = [1] + reshape_shape
          dims.clear()
          dims.extend(reshape_shape)
          perm_order = permute_node.permute_nnp_param.order
          if not self.is_hw_permute(perm_order):
            continue
          new_perm_order = [0,1,3,2] #hardcoding this for now
          perm_order.clear()
          perm_order.extend(new_perm_order)
          softmax_node = layer_pred[found^1]
          softmax_shape = list(self.caffe_net.blobs[softmax_node.top[0]].data.shape)
          post_softmax_reshape_shape = [1] + softmax_shape
          reshape_node = self.G.mk_reshape_proto(softmax_node.top[0], self.ll(post_softmax_reshape_shape), softmax_node.name + '_post_reshape')
          self.G.add_node(reshape_node)
          self.G.add_edge(softmax_node, reshape_node)
          self.G.add_edge(reshape_node, layer)
          self.G.remove_edge(softmax_node,layer)
          self.G.add_replace_edge_proto_new(reshape_node, layer, softmax_node.top[0], reshape_node.top[0])

         

        





        


        









