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
import numpy as np
from mmdnn.conversion.caffe.c2c.common import BasePass
from caffe import layers as L
from caffe import params as P
from dv_logging import get_logger
import copy
logger = get_logger(__name__, 'dvctoc')


class RemoveRedundantLayers(BasePass):
  '''
    # If all the successors or of Reshape or all are Permutes, and if the characteristics of the successors are same,
    # i.e., same shape for the reshape layer or same permute order for the permute layers, then the redundant layers can be removed
    # Scenario : (same applicable to permutes)
    # ->  Layer1 ->  Reshape1 ->  Layer2                    Layer1  ->  Reshape ->  Layer2
    #            ->  Reshape2 ->  Layer3                                        ->  Layer3
    #            ->  Reshape3 ->  Layer4          ===                           ->  Layer4
    #            :                                                               :
    #            :                                                               :
    # Even though the reshape is a NoOp, it may help in further optimizations like permute
  '''

  def __init__(self):
    pass

  def do_transformation(self):
    transform = True
    skip_nodes = []
    while transform:
      nodes = self.get_topsort()
      reshape_succ = dict()
      permute_succ = dict()
      for node in nodes:
        if node.type in ["Slice"]:
          continue
        successors = list(node.successors())
        if len(successors) > 1:
          s_type = list(set([s.type for s in successors]))
          net_out = False
          for s in successors:
            if len(s.successors()) == 0:
              net_out = True
          if len(s_type) == 1 and s_type[0] == "Reshape" and not net_out:
            # Check if all the successors are of Reshape
            # To avoid nesting, donot append the node if it is a successor of any node which is already in the dictionary
            found = False
            for k, v in reshape_succ.items():
              if node.name in v:
                found = True
                break
            if not found and node.name not in skip_nodes:
              net_out = False
              reshape_succ[node] = [s.name for s in successors]
          elif len(s_type) == 1 and s_type[0] == "Permute" and not net_out:
            # Check if all the successors are of Permute
            found = False
            for k, v in permute_succ.items():
              if node.name in v:
                found = True
                break
            if not found and node.name not in skip_nodes:
              permute_succ[node] = [s.name for s in successors]
      if len(reshape_succ) == 0 and len(permute_succ) == 0:
        transform = False
        continue
      i = 0
      for node in reshape_succ:
        successor = node.successors()[0]
        shape = list(successor.reshape_param.shape.dim)
        reshape_node = self.G.mk_reshape_proto(node.top[0], shape, node.name)
        # Check if shape attributes are same for all the successors
        same_shape = True
        for s in node.successors()[1:]:
          if not np.array_equal(shape, list(s.reshape_param.shape.dim)):
            same_shape = False
            break
        if not same_shape:
          skip_nodes.append(node.name)
          continue
        same_top = True
        for s in node.successors()[1:]:
          if s.bottom[0] != node.successors()[0].bottom[0]:
            same_top = False
            break
        if not same_top:
          skip_nodes.append(node.name)
          continue
        successors = list(node.successors())
        self.G.add_node(reshape_node)
        self.G.add_edge(node, reshape_node)
        for s in successors:
          self.G.remove_edge(node, s)
          for gs in list(s.successors()):
            self.G.add_edge(reshape_node, gs)
            self.G.remove_edge(s, gs)
            self.G.add_replace_edge_proto_new(reshape_node, gs, s.top[0], reshape_node.top[0])
          self.G.remove_node(s)
        i += 1
      i = 0
      for node in permute_succ:
        successor = node.successors()[0]
        order = successor.permute_param.order
        order = [n for n in order]
        permute_node = self.G.mk_permute_proto(node.top[0], order)
        same_order = True
        for s in node.successors()[1:]:
          if not np.array_equal(order, [n for n in s.permute_param.order]):
            same_order = False
            break
        if not same_order:
          skip_nodes.append(node.name)
          continue
        same_top = True
        for s in node.successors()[1:]:
          if s.bottom[0] != node.successors()[0].bottom[0]:
            same_top = False
            break
        if not same_top:
          skip_nodes.append(node.name)
          continue
        successors = list(node.successors())
        self.G.add_node(permute_node)
        self.G.add_edge(node, permute_node)
        for s in successors:
          self.G.remove_edge(node, s)
          for gs in list(s.successors()):
            self.G.add_edge(permute_node, gs)
            self.G.remove_edge(s, gs)
            self.G.add_replace_edge_proto_new(permute_node, gs, s.top[0], permute_node.top[0])
          self.G.remove_node(s)
        i += 1


class RemoveRedundantPermutes(BasePass):
  '''
    # If there are permute+reshape layers on 2 branches merging to concat, the permute orders of the permute are same and also if the inputs
    # coming into the permute layers differ in shape at only one axis, those 2 permute+reshape layers can be removed and
    # a single permute+reshape layer can be added after concat
    # Scenario:
    # Layer1  ->  Permute1  ->  Reshape1  ->| ->  Concat  ->  Layer3        Layer1  ->| ->  Concat  ->  Permute ->  Reshape ->  Layer3
    # Layer2  ->  Permute2  ->  Reshape2  ->|                         ===   Layer2  ->|
  '''

  def __init__(self):
    pass

  def do_transformation(self):
    concats = self.get_nodes_of_type('Concat')
    allowed = ['DVShift']
    i = 0
    for layer in concats:
      if len(layer.predecessors()) != 2:
        continue
      parent1 = layer.predecessors()[0]
      parent2 = layer.predecessors()[1]
      pt_branch1, pt_branch2 = False, False
      while not (pt_branch1 and pt_branch2):
        if not pt_branch1:
          if parent1.type == 'Reshape':
            if parent1.predecessors()[0].type == 'Permute' and len(parent1.predecessors()[0].successors()) == 1 and len(parent1.successors()) == 1:
              logger.info("Permute+Reshape pattern found before concat for parent1")
              pt_branch1 = True
            else:
              break
          elif parent1.type in allowed:
            parent1 = parent1.predecessors()[0]
          else:
            break
        if not pt_branch2:
          if parent2.type == 'Reshape':
            if parent2.predecessors()[0].type == 'Permute' and len(parent2.predecessors()[0].successors()) == 1 and len(parent2.successors()) == 1:
              logger.info("Permute+Reshape pattern found before concat for parent2")
              pt_branch2 = True
            else:
              break
          elif parent2.type in allowed:
            parent2 = parent2.predecessors()[0]
          else:
            break
      if pt_branch1 and pt_branch2:
        perm1_layer = parent1.predecessors()[0]
        rshp1_layer = parent1
        perm2_layer = parent2.predecessors()[0]
        rshp2_layer = parent2
        rshp1, rshp2 = list(rshp1_layer.reshape_param.shape.dim), list(rshp2_layer.reshape_param.shape.dim)
        perm1, perm2 = list(perm1_layer.permute_param.order), list(perm2_layer.permute_param.order)
        if np.array_equal(perm1, perm2):
          new_shp = []
          for i, j in zip(rshp1, rshp2):
            # pylint: disable=W0106
            new_shp.append(i) if i == j else new_shp.append(i + j)
          # Get new axis for concat
          new_axis = -1
          perm1_inp_shape = list(self.caffe_net.blobs[perm1_layer.predecessors()[0].top[0]].data.shape)
          perm2_inp_shape = list(self.caffe_net.blobs[perm2_layer.predecessors()[0].top[0]].data.shape)
          n_changes = 0
          for idx, (s1, s2) in enumerate(zip(perm1_inp_shape, perm2_inp_shape)):
            if s1 != s2:
              new_axis = idx
              n_changes += 1
          # Ignore if the shape is different at multiple axes. Cannot remove permute+reshape
          if n_changes > 1:
            logger.info("Ignoring...")
            continue
          new_concat = self.G.mk_concat_proto([layer.bottom[0], layer.bottom[1]], layer.name, new_axis)
          # new_concat.bottom[:] = layer.bottom
          # Remove perm1_layer, perm2_layer, rshp1_layer, rshp2_layer
          self.G.add_replace_edge_proto_new(perm1_layer.predecessors()[0], rshp1_layer.successors()[0], rshp1_layer.top[0], perm1_layer.predecessors()[0].top[0])
          self.G.add_edge(perm1_layer.predecessors()[0], rshp1_layer.successors()[0])
          self.G.remove_edge(perm1_layer.predecessors()[0], perm1_layer)
          self.G.remove_edge(perm1_layer, rshp1_layer)
          self.G.remove_edge(rshp1_layer, rshp1_layer.successors()[0])
          self.G.remove_node(perm1_layer)
          self.G.remove_node(rshp1_layer)
          self.G.add_replace_edge_proto_new(perm2_layer.predecessors()[0], rshp2_layer.successors()[0], rshp2_layer.top[0], perm2_layer.predecessors()[0].top[0])
          self.G.add_edge(perm2_layer.predecessors()[0], rshp2_layer.successors()[0])
          self.G.remove_edge(perm2_layer.predecessors()[0], perm2_layer)
          self.G.remove_edge(perm2_layer, rshp2_layer)
          self.G.remove_edge(rshp2_layer, rshp2_layer.successors()[0])
          self.G.remove_node(perm2_layer)
          self.G.remove_node(rshp2_layer)
          # Add new permute+reshape layers to concat
          successors = list(layer.successors())
          new_perm = self.G.mk_permute_proto(layer.name, perm1)
          new_rshp = self.G.mk_reshape_proto(new_perm.top[0], new_shp, layer.name)
          self.G.add_node(new_perm)
          self.G.add_node(new_rshp)
          self.G.add_edge(layer, new_perm)
          self.G.add_edge(new_perm, new_rshp)
          for s in successors:
            self.G.add_edge(new_rshp, s)
            self.G.remove_edge(layer, s)
            self.G.add_replace_edge_proto_new(new_rshp, s, layer.top[0], new_rshp.top[0])
          # Replace old concat layer with new_concat
          self.G.add_node(new_concat)
          for p in list(layer.predecessors()):
            self.G.add_edge(p, new_concat)
            self.G.remove_edge(p, layer)
          for s in list(layer.successors()):
            self.G.add_edge(new_concat, s)
            self.G.remove_edge(layer, s)
            self.G.add_replace_edge_proto_new(new_concat, s, layer.top[0], new_concat.top[0])
          self.G.remove_node(layer)
          self.G.relabel_nodes_locally({f'{layer.type}/{new_concat.name}': layer.name}, layer.type)


class RemoveReshapePermutes(BasePass):
  '''
  # No Generalization. Specific to Zippin model
  '''

  def __init__(self):
    pass

  def do_transformation(self):
    reshapes = self.get_nodes_of_type('Reshape')
    idx = 0

    def is_feasible(order1, order2, shp, rshp):
      arr = np.array(list(range(np.prod(shp))))
      brr = np.reshape(arr, shp)
      # permute, Reshape & Permute
      brr = np.transpose(brr, order1)
      brr = np.reshape(brr, rshp)
      brr = np.transpose(brr, order2)
      brr = brr.flatten()
      return True if np.array_equal(arr, brr) else False

    for node in reshapes:
      # Only applicable for Permute-1->Reshape->Permute-2. Check there are no other children for Permute-1, no other parents for Reshape,
      # no ther children for Reshape
      # Also, none of the predecessor, node, successor should have isOutput flag set to true
      if len(node.predecessors()) == 1 and node.predecessors()[0].type == 'Permute' and \
         len(node.successors()) == 1 and node.successors()[0].type == 'Permute' and \
         len(node.predecessors()[0].successors()) == 1:
        perm1, perm2 = node.predecessors()[0], node.successors()[0]
        order1, order2 = list(perm1.permute_param.order), list(perm2.permute_param.order)
        shape = list(node.reshape_param.shape.dim)
        perm1_in_shape = self.caffe_net.blobs[perm1.bottom[0]].shape
        if is_feasible(order1, order2, perm1_in_shape, shape):
          remove_perm1 = True
          parent_node = perm1.predecessors()[0]
          child_nodes = list(perm2.successors())
          if perm1.converter_param.isOutput:
            # Fork out a new branch from the parent of 1st permute and pass the isOutput flag
            remove_perm1 = False
          n_shape = [shape[a] for a in order2]
          rshp_node = self.G.mk_reshape_proto(parent_node.top[0], n_shape, node.name)
          self.G.add_node(rshp_node)
          self.G.add_edge(parent_node, rshp_node)
          rshp_node.bottom[:] = perm1.bottom
          if remove_perm1:
            self.G.remove_edge(parent_node, perm1)
            self.G.remove_node(perm1)
          self.G.remove_node(node)
          for c in child_nodes:
            self.G.add_edge(rshp_node, c)
            self.G.remove_edge(perm2, c)
            self.G.add_replace_edge_proto_new(rshp_node, c, perm2.top[0], rshp_node.top[0])
          self.G.remove_node(perm2)
      idx += 1


class PermuteReshapePermute(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    all_perms = self.get_nodes_of_type('Permute')
    permutes, skip = [], []
    for p in all_perms:
      if (len(p.successors()) == 1 and p.successors()[0].get_layer().type == 'Reshape' and len(p.successors()[0].successors()) == 1 and p.successors()[0].successors()[0].get_layer().type == 'Permute'):
        if p not in skip:
          permutes.append(p)
          skip.append(p.successors()[0].successors()[0])

    def check(shape1, shape2):
      result = [-1] * len(shape2)
      if len(shape1) > len(shape2):
        return [False, result]
      i, j = 0, -1
      for i in range(len(shape1)):
        num1 = shape1[i]
        if num1 != 1:
          done = 1
          while True:
            j += 1
            if j == len(shape2):
              return [False, result]
            num2 = shape2[j]
            if num2 == 1:
              continue
            q = (num1 // done) // num2
            r = (num1 // done) % num2
            if r != 0 or q == 0:
              return [False, result]
            result[j] = i
            if q == 1:
              break
            else:
              done = done * num2
      return [True, result]

    for perm1 in permutes:
      reshape = perm1.successors()[0]
      perm2 = perm1.successors()[0].successors()[0]
      shape_1 = self.caffe_net.blobs[perm1.bottom[0]].data.shape
      shape_2 = self.caffe_net.blobs[perm1.top[0]].data.shape
      shape_3 = self.caffe_net.blobs[reshape.top[0]].data.shape
      shape_4 = self.caffe_net.blobs[perm2.top[0]].data.shape
      if check(shape_2, shape_3)[0]:
        shape_3to2 = check(shape_2, shape_3)[1]
        perm2_order = perm2.permute_param.order
        shape_3to4_pos = {i: j for j, i in enumerate(perm2_order)}
        perm1_order = perm1.permute_param.order
        shape_1_chunks = [[] for i in shape_1]
        for i, j in enumerate(shape_3):
          if j == 1:
            continue
          pos_2 = shape_3to2[i]
          pos_1 = perm1_order[pos_2]
          pos_4 = shape_3to4_pos[i]
          shape_1_chunks[pos_1].append([shape_3[i], pos_4])
        new_reshape_shape, new_perm_order_rev = [], []
        for i in shape_1_chunks:
          for j in i:
            new_reshape_shape.append(j[0])
            new_perm_order_rev.append(j[1])
        perm_pos_left = [i for i in range(len(shape_4)) if i not in new_perm_order_rev]
        new_reshape_shape = [1] * (len(shape_4) - len(new_reshape_shape)) + new_reshape_shape
        new_perm_order_rev = perm_pos_left + new_perm_order_rev
        new_perm_order_dict = {j: i for i, j in enumerate(new_perm_order_rev)}
        new_perm_order = [new_perm_order_dict[i] for i in range(len(shape_4))]
        reshape.reshape_param.ClearField('shape')
        reshape.reshape_param.shape.dim.extend(new_reshape_shape)
        perm2.permute_param.ClearField('order')
        perm2.permute_param.order.extend(new_perm_order)
        perm1_par = perm1.predecessors()[0]
        self.G.remove_node(perm1)
        self.G.add_edge(perm1_par, reshape)
        self.G.add_replace_edge_proto_new(perm1_par, reshape, perm1.top[0], perm1_par.top[0])
      elif check(shape_3, shape_2)[0]:
        shape_2to3 = check(shape_3, shape_2)[1]
        perm2_order = perm2.permute_param.order
        perm1_order = perm1.permute_param.order
        shape_3to4_pos = {i: j for j, i in enumerate(perm2_order)}
        shape_4_chunks = [[] for i in shape_4]
        for i, j in enumerate(shape_2):
          if j == 1:
            continue
          pos_3 = shape_2to3[i]
          pos_4 = shape_3to4_pos[pos_3]
          pos_1 = perm1_order[i]
          shape_4_chunks[pos_4].append([shape_2[i], pos_1])
        new_perm_order = []
        for i in shape_4_chunks:
          for j in i:
            new_perm_order.append(j[1])
        perm_pos_left = [i for i in range(len(shape_1)) if i not in new_perm_order]
        new_perm_order = perm_pos_left + new_perm_order
        reshape.reshape_param.ClearField('shape')
        reshape.reshape_param.shape.dim.extend(shape_4)
        perm1.permute_param.ClearField('order')
        perm1.permute_param.order.extend(new_perm_order)
        perm2_child = list(perm2.successors())
        self.G.remove_node(perm2)
        for s in perm2_child:
          for t in s.bottom:
            if t in perm2.top:
              self.G.add_edge(reshape, s)
              self.G.add_replace_edge_proto_new(reshape, s, t, reshape.top[0])

class yolov8_concat_slice_reshape(BasePass):
  def __init__(self):
    pass

  def check_pattern(self, concat):
    if not all([pred.type == "Reshape" for pred in concat.predecessors()]):
      return False
    successors = concat.successors()
    if len(successors) != 1 or successors[0].type != "Reshape":
      return False
    rshp_succ = successors[0]
    successors = rshp_succ.successors()
    if not (successors and all(succ.type == "Crop" for succ in successors)):
      return False
    return True

  def do_transformation(self):
    all_concats = self.get_nodes_of_type('Concat')
    concat_node_list = []
    for concat in all_concats:
      if not self.check_pattern(concat):
        continue
      successors = concat.successors()
      rshp_succ = successors[0]
      rshp_succ_shape = self.caffe_net.blobs[rshp_succ.top[0]].shape
      succ_to_rshp_succ = rshp_succ.successors()

      for o_idx, prev_rshp in enumerate(concat.predecessors()):
        orig_shape = list(prev_rshp.reshape_param.shape.dim)
        prev_rshp.reshape_param.ClearField('shape')
        rshp_succ_shape_list = list(rshp_succ_shape)
        rshp_succ_shape_list[-1] = orig_shape[-1]
        prev_rshp.reshape_param.shape.dim.extend(rshp_succ_shape_list)
      
      self.G.remove_edge(concat, rshp_succ)
      for idx, succ_crop in enumerate(succ_to_rshp_succ):
        self.G.remove_edge(rshp_succ, succ_crop)
        self.G.add_edge(concat, succ_crop)
        self.G.add_replace_edge_proto(concat, succ_crop, rshp_succ.top[0])
      self.G.remove_node(rshp_succ)

      succ_crop = concat.successors()
      pred_rshp = concat.predecessors()
      for c_idx, succ in enumerate(succ_crop):
        new_crop_layers = []
        for r_idx, pred in enumerate(pred_rshp):
          orig_dummy_layer = [btm for btm in succ.predecessors() if btm.type == "DummyData"][0]
          orig_dummy_shape = list(self.caffe_net.blobs[orig_dummy_layer.top[0]].shape)
          orig_dummy_shape[-1] = list(pred.reshape_param.shape.dim)[-1]
          offset = list(succ.crop_param.offset)
          dummy, new_succ = self.G.mk_crop_proto(succ.name + f"_{c_idx}_{r_idx}", orig_dummy_shape, offset)
          self.G.add_node(dummy)
          self.G.add_node(new_succ)
          self.G.add_edge(dummy, new_succ)
          self.G.add_edge(pred, new_succ)
          new_succ.bottom.append(pred.top[0])
          new_succ.bottom.append(dummy.top[0])
          new_succ.crop_param.axis = 0
          new_crop_layers.append(new_succ)
        new_concat = self.G.mk_concat_proto([layer.top[0] for layer in new_crop_layers], concat.name + f"_{c_idx}", new_axis=3)
        # self.mutation_observer["family"][concat.name].append(f"{new_concat.name}")
        concat_node_list.append(f"{new_concat.name}")
        self.G.add_node(new_concat)
        for layer in new_crop_layers:
          self.G.add_edge(layer, new_concat)
        nxt_layers = succ.successors()
        for nxt_layer in nxt_layers:
          self.G.remove_edge(succ, nxt_layer)
          self.G.add_edge(new_concat, nxt_layer)
          self.G.add_replace_edge_proto(new_concat, nxt_layer, succ.top[0])
        orig_dummy_layer = [btm for btm in succ.predecessors() if btm.type == "DummyData"][0]
        self.G.remove_node(orig_dummy_layer)
        self.G.remove_node(succ)
      for r_idx, pred in enumerate(pred_rshp):
        self.G.remove_edge(pred, concat)
      self.G.remove_node(concat)
      self.mutation_observer["family"][concat.name] = concat_node_list
      # print("Done")


class yolov8_reshape_concat_split(BasePass):

  def __init__(self):
    pass

  def pattern_ok(self, concat_mid):
    if len(concat_mid.predecessors()) == 3:
      p1, p2, p3 = concat_mid.predecessors()
      if not (p1.type == p2.type == p3.type == 'Reshape'):
        return False, []
      chk = True
      conc = []
      for x in [p1,p2,p3]:
        if len(x.predecessors())!=1:
          chk = False
          break
        if x.predecessors()[0].type != 'Concat':
          chk = False
          break
        conc.append(x.predecessors()[0])
      if not chk:
        return False, []
    else:
      return False, []
    if len(concat_mid.successors()) == 1:
      if concat_mid.successors()[0].type !=  'Slice':
        return False, []
      s1 = concat_mid.successors()[0]
      return True, [p1,p2,p3,conc[0], conc[1], conc[2], s1]
    else: 
      return False, []
        

  def do_transformation(self):
    all_concats = self.get_nodes_of_type('Concat')
    for concat_mid in all_concats:
      ok, arr = self.pattern_ok(concat_mid)
      if not ok:
        continue
      rshp1, rshp2, rshp3, conc1, conc2, conc3, slice = arr
      conc1_in1_s = [i for i in self.caffe_net.blobs[conc1.bottom[0]].shape]
      conc1_in2_s = [i for i in self.caffe_net.blobs[conc1.bottom[1]].shape]
      conc2_in1_s = [i for i in self.caffe_net.blobs[conc2.bottom[0]].shape]
      conc2_in2_s = [i for i in self.caffe_net.blobs[conc2.bottom[1]].shape]
      conc3_in1_s = [i for i in self.caffe_net.blobs[conc3.bottom[0]].shape]
      conc3_in2_s = [i for i in self.caffe_net.blobs[conc3.bottom[1]].shape]
      rshp1_in_s = [i for i in self.caffe_net.blobs[rshp1.bottom[0]].shape]
      rshp2_in_s = [i for i in self.caffe_net.blobs[rshp2.bottom[0]].shape]
      rshp3_in_s = [i for i in self.caffe_net.blobs[rshp3.bottom[0]].shape]
      rshp1_out_s = [i for i in self.caffe_net.blobs[rshp1.top[0]].shape]
      rshp2_out_s = [i for i in self.caffe_net.blobs[rshp2.top[0]].shape]
      rshp3_out_s = [i for i in self.caffe_net.blobs[rshp3.top[0]].shape]
      slice_in_s = [i for i in self.caffe_net.blobs[slice.bottom[0]].shape]
      if not (conc1.concat_param.axis == 1 
            and conc2.concat_param.axis == 1
            and conc3.concat_param.axis == 1
            and conc1_in1_s[1] == conc2_in1_s[1] == conc3_in1_s[1]
            and conc1_in2_s[1] == conc2_in2_s[1] == conc3_in2_s[1]
            and rshp1_in_s[:2] == rshp1_out_s[:2]
            and rshp2_in_s[:2] == rshp2_out_s[:2]
            and rshp3_in_s[:2] == rshp3_out_s[:2]
            and concat_mid.concat_param.axis == 2
            and slice.slice_param.axis == 1
            and slice.slice_param.slice_point[0] == conc1_in1_s[1]):
        continue
      node_r11 = self.G.mk_reshape_proto(conc1.bottom[0], conc1_in1_s[:2] + [conc1_in1_s[-2]*conc1_in1_s[-1]], conc1.bottom[0])
      node_r12 = self.G.mk_reshape_proto(conc1.bottom[1], conc1_in2_s[:2] + [conc1_in2_s[-2]*conc1_in2_s[-1]], conc1.bottom[1])
      node_r21 = self.G.mk_reshape_proto(conc2.bottom[0], conc2_in1_s[:2] + [conc2_in1_s[-2]*conc2_in1_s[-1]], conc2.bottom[0])
      node_r22 = self.G.mk_reshape_proto(conc2.bottom[1], conc2_in2_s[:2] + [conc2_in2_s[-2]*conc2_in2_s[-1]], conc2.bottom[1])
      node_r31 = self.G.mk_reshape_proto(conc3.bottom[0], conc3_in1_s[:2] + [conc3_in1_s[-2]*conc3_in1_s[-1]], conc3.bottom[0])
      node_r32 = self.G.mk_reshape_proto(conc3.bottom[1], conc3_in2_s[:2] + [conc3_in2_s[-2]*conc3_in2_s[-1]], conc3.bottom[1])
      node_c1 = self.G.mk_concat_proto([node_r11.top[0], node_r21.top[0], node_r31.top[0]], concat_mid.name+"_1", 2)
      node_c2 = self.G.mk_concat_proto([node_r12.top[0], node_r22.top[0], node_r32.top[0]], concat_mid.name+"_2", 2)
      self.mutation_observer["family"][concat_mid.name] = [f"{node_c1.name}, {node_c2.name}"]
      self.G.add_node(node_r11)
      self.G.add_node(node_r12)
      self.G.add_node(node_r21)
      self.G.add_node(node_r22)
      self.G.add_node(node_r31)
      self.G.add_node(node_r32)
      self.G.add_node(node_c1)
      self.G.add_node(node_c2)
      self.G.add_edge(conc1.predecessors()[0], node_r11)
      self.G.add_edge(conc1.predecessors()[1], node_r12)
      self.G.add_edge(conc2.predecessors()[0], node_r21)
      self.G.add_edge(conc2.predecessors()[1], node_r22)
      self.G.add_edge(conc3.predecessors()[0], node_r31)
      self.G.add_edge(conc3.predecessors()[1], node_r32)
      self.G.add_edge(node_r11, node_c1)
      self.G.add_edge(node_r21, node_c1)
      self.G.add_edge(node_r31, node_c1)
      self.G.add_edge(node_r12, node_c2)
      self.G.add_edge(node_r22, node_c2)
      self.G.add_edge(node_r32, node_c2)
      for s in slice.successors():
        if slice.top[0] in s.bottom:
          self.G.add_edge(node_c1, s)
          self.G.add_replace_edge_proto_new(node_c1, s, slice.top[0], node_c1.top[0])
        if slice.top[1] in s.bottom:
          self.G.add_edge(node_c2, s)
          self.G.add_replace_edge_proto_new(node_c2, s, slice.top[1], node_c2.top[0])
      self.G.remove_node(rshp1)
      self.G.remove_node(rshp2)
      self.G.remove_node(rshp3)
      self.G.remove_node(conc1)
      self.G.remove_node(conc2)
      self.G.remove_node(conc3)
      self.G.remove_node(slice)
      self.G.remove_node(concat_mid)
      # import pdb; pdb.set_trace()

      
