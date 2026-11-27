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
always_fuse = {'Scale', 'Dropout', 'ReLU', 'ReLU6', 'PReLU', 'Sigmoid', 'HardSwish', 'Swish'}
fusing_layers = {'InnerProduct': {'BatchNorm'}.union(always_fuse), 'Deconvolution': always_fuse, 'Convolution': {'BatchNorm'}.union(always_fuse), 'BatchNorm': always_fuse}
for f in always_fuse:
  fusing_layers[f] = always_fuse


def do_layers_fuse(ordered_layer_list):
  ordered_layer_list = list(map(lambda l: l.type, ordered_layer_list))
  next_item = ordered_layer_list.pop(0)
  if next_item not in fusing_layers:
    return False
  else:
    allowed_next = fusing_layers[next_item]
    for it in ordered_layer_list:
      if it in allowed_next:
        allowed_next = fusing_layers[it]
      else:
        return False
    return True


def get_run_of_3(start_node):
  children = list(start_node.successors())
  if len(children) == 1:
    grandchildren = list(children[0].successors())
    if len(grandchildren) == 1:
      return (start_node, children[0], grandchildren[0])
  return None


class ReorderLayersForFusion(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    # force the entire topsort to be generated
    topsort_nodes = self.get_topsort()
    for layer in topsort_nodes:
      if layer.type not in fusing_layers:
        continue
      lineage = get_run_of_3(layer)
      if lineage is None:
        # there is no linear sequence of node a -> b -> c
        # such that we can check of a -> c fuse
        continue
      # in l1 -> x -> l2, check if l1 and l2 can fuse
      if do_layers_fuse([lineage[0], lineage[2]]) and lineage[1].type in ['Crop', 'Reshape']:
        self.G.remove_edge(lineage[0], lineage[1])
        self.G.remove_edge(lineage[1], lineage[2])
        succ = list(lineage[2].successors())
        # bottom_list = []
        for successor in succ:
          self.G.remove_edge(lineage[2], successor)
          self.G.add_edge(lineage[1], successor)
          self.G.add_replace_edge_proto_new(lineage[1], successor, lineage[2].top[0], lineage[1].top[0])
        self.G.add_edge(lineage[0], lineage[2])
        self.G.add_edge(lineage[2], lineage[1])
        lineage[2].bottom[:] = lineage[0].top
        # lineage[1].bottom[:] = lineage[2].top
        for i in range(len(lineage[1].bottom)):
          if lineage[1].bottom[i] == lineage[0].top[0]:
            lineage[1].bottom[i] = lineage[2].top[0]
