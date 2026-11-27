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
from networkx.algorithms.traversal.depth_first_search import dfs_tree
logger = get_logger(__name__, 'dvctoc')
CP_OPS = ['Onnx_Batchpermutation', 'DistributeFpn', 'BoxWithNMSLimit', 'BBoxTransform', 'CollectRpnProposals']


def forked(node):
  #hack dummydata + crop
  if node.type == "Crop" and len(node.predecessors()) == 2:
    for pred in list(node.predecessors()):
      if pred.type == "DummyData":
        return False
  if len(node.successors()) > 1 or len(node.predecessors()) > 1:
    return True
  else:
    return False


def next_critical_node(node):
  parent_node = node
  while (not forked(node)) and (node.type not in CP_OPS) and len(node.successors()):
    parent_node = node
    node = node.successors()[0]
  children = list(node.successors())
  if node.type in CP_OPS:
    tail = parent_node
  elif len(node.predecessors()) > 1:
    tail = parent_node
    children = [node]
  else:
    tail = node
  return tail, children


class DynamicInputPatching(BasePass):

  def __init__(self):
    pass

  def patch_a_branch(self, node):
    #put loop begin before this node
    #iterate till a fork/cp layer is found
    #put loop end before the fork
    #if fork recurse, with both the branches
    #if cp recurse after cp node
    pred = None
    if node.type == "LoopBegin" or node.type == "LoopEnd":
      return
    if node.type in CP_OPS:
      for successor in list(node.successors()):
        self.patch_a_branch(successor)
      return
    predecessors = list(node.predecessors())
    for pred in predecessors:
      if pred.type == "LoopBegin":
        return
    lb_layer = self.G.mk_loop_begin_proto(name=node.name)
    self.G.add_node(lb_layer)
    self.G.add_edge(lb_layer, node)
    for b in node.bottom:
      for p in predecessors:
        for t in p.top:
          if t == b:
            pred = p
            break
      self.G.add_edge(pred, lb_layer)
      self.G.remove_edge(pred, node)
      assert pred is not None
      for idx, top in enumerate(pred.top):
        if top in node.bottom:
          lb_layer.bottom.append(top)
          top_name = lb_layer.name + '_' + pred.name + '_' + str(idx)
          lb_layer.top.append(top_name)
          for j, bottom in enumerate(node.bottom):
            if bottom == top:
              node.bottom[j] = top_name
    # all_tops = list(lb_layer.top)
    tail_node, children = next_critical_node(node)
    if tail_node.type == "LoopEnd":
      return
    le_layer = self.G.mk_loop_end_proto(name=tail_node.name)
    le_layer.bottom.append(tail_node.top[0])
    self.G.add_node(le_layer)
    if len(tail_node.successors()):
      for suc in list(tail_node.successors()):
        for idx, bottom in enumerate(suc.bottom):
          if bottom == tail_node.top[0]:
            suc.bottom[idx] = le_layer.top[0]
        self.G.add_edge(le_layer, suc)
        self.G.remove_edge(tail_node, suc)
    self.G.add_edge(tail_node, le_layer)
    for child in children:
      self.patch_a_branch(child)

  def search_loopend(self, dfsorder):
    for node in dfsorder:
      try:
        nview = self.G.nodes[node]['node_obj']
        if nview.type in ["LoopEnd"]:
          return nview
      except Exception as e:
        logger.warn("[E] Corresponding loopend not found, ", exc_info=e)
    return -1

  def do_transformation(self):
    if self.dynamic_input is False:
      return
    input_nodes = self.get_nodes_of_type('Input')
    successors = []
    for node in input_nodes:
      successors.extend(node.successors())
    for node in successors:
      self.patch_a_branch(node)
    loopbegin = self.get_nodes_of_type('LoopBegin')
    for node in loopbegin:
      nodename = "LoopBegin/{}".format(node.name)
      dfsorder = list(dfs_tree(self.G, source=nodename))
      loopend = self.search_loopend(dfsorder)
      le_nodename = "LoopEnd/{}".format(loopend.name)
      if loopend == -1:
        logger.warn("[W] Corresponding loopend not found, %s ", node.name)
        continue
      logger.info("Corresponding loopend node [%s] for loopBegin node [%s]", loopend.name, node.name)
      lb_layerobj = self.G.nodes[nodename]['node_obj']._layer
      lb_layerobj.loop_begin_param.loopend_layername = node.name
      le_layerobj = self.G.nodes[le_nodename]['node_obj']._layer
      le_layerobj.loop_end_param.loopbegin_layername = node.name
