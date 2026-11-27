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
from mmdnn.conversion.caffe.c2c.common import BasePass
'''
Only one input for permute layer but multiple outputs may be possible
CASES:
          LAYER1          LAYER1
            |               |
            v               v
          PERMUTE1        PERMUTE1
            |               |     \
            v               v      v
          PERMUTE2        LAYER3   PERMUTE2
            |               |
            v               v
          LAYER2          LAYER2

Handling only inline Permutes. Conditions to check:
1. All Successors needs to be only permutes
2. If All the Successor Permutes then they need to be of same order
'''


class FusePermute(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    permutes = self.get_nodes_of_type('Permute')
    all_permutes = self.get_nodes_of_type('Permute')
    #Pre process the permutes list to avoid duplicates
    #This is done so that if there are multiple permute layers in line, all permute layers except first are removed
    '''
      Function to check whether all the successor nodes are permutes
    '''

    def successors_are_permutes(node):
      for successor in node.successors():
        if successor.get_layer().type != 'Permute':
          return False
      return True

    '''
      Function to check whether all the successor permute nodes have same order
    '''

    def successors_have_same_order(node):
      if len(node.successors()) < 2:
        return True

      successors = node.successors()
      order_shape = successors[0].permute_param.order
      first_successor_order = [order_shape[i] for i in range(len(order_shape))]

      for successor in successors:
        order_shape = successor.permute_param.order
        successor_order = [order_shape[i] for i in range(len(order_shape))]
        if first_successor_order != successor_order:
          return False
      return True

    '''
      Function to remove successor inline permutes recursively(Depth wise)
    '''

    def remove_successor_duplicates(node):
      i = 0
      while i < len(node.successors()) and node.successors()[i] in permutes:
        permutes.remove(node.successors()[i])
        remove_successor_duplicates(node.successors()[i])
        i += 1

    for layer in all_permutes:
      cur_layer = layer
      if cur_layer in permutes and successors_are_permutes(cur_layer):
        if successors_have_same_order(cur_layer):
          remove_successor_duplicates(cur_layer)

    idx = 0
    while idx < len(permutes):
      layer = permutes[idx]
      order_shape = layer.permute_param.order
      cur_order = [order_shape[i] for i in range(len(order_shape))]
      increment = True

      # Check: The Patching will be done only if all the successors are Permutes and of same order
      if not successors_are_permutes(layer) or not successors_have_same_order(layer):
        idx = idx + 1 if increment else idx
        continue

      #Parse through each of the successor of permute layer
      for successor in list(layer.successors()):
        if successor.type == 'Permute':
          #Get the successor of the current layer
          cur_layer = successor
          next_layers = list(cur_layer.successors())
          cur_layer_shape = cur_layer.permute_param.order
          #Rearrange the original order based on successors permute order
          new_order = [cur_order[i] for i in cur_layer_shape]
          cur_order = copy.deepcopy(new_order)
          #Point the predecessor of the current layer to successors
          for l in next_layers:
            self.G.remove_edge(cur_layer, l)
            self.G.add_edge(layer, l)
            self.G.add_replace_edge_proto_new(layer, l, cur_layer.name, layer.top[0])
          self.G.remove_edge(layer, cur_layer)
          self.G.remove_node(cur_layer)
          increment = False
      #If the final order is same as original order, remove the original permute layer aswell
      is_required = False
      for i in range(len(cur_order)):
        if i != cur_order[i]:
          is_required = True
          break
      if not is_required:
        prev_layer = layer.predecessors()[0]
        next_layers = list(layer.successors())
        for l in next_layers:
          self.G.remove_edge(layer, l)
          self.G.add_edge(prev_layer, l)
          self.G.add_replace_edge_proto_new(prev_layer, l, layer.name, prev_layer.top[0])
        self.G.remove_edge(prev_layer, layer)
        self.G.remove_node(layer)
        increment = True
      #Assign the final order to the first permute layer
      else:
        for i in range(len(order_shape)):
          order_shape[i] = cur_order[i]
      idx = idx + 1 if increment else idx


class FusePermuteNnp(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    permutes = self.get_nodes_of_type('PermuteNnp')
    all_permutes = self.get_nodes_of_type('PermuteNnp')
    #Pre process the permutes list to avoid duplicates
    #This is done so that if there are multiple permute layers in line, all permute layers except first are removed
    '''
      Function to check whether all the successor nodes are permutes
    '''

    def successors_are_permutes(node):
      for successor in node.successors():
        if successor.get_layer().type != 'PermuteNnp':
          return False
      return True

    '''
      Function to check whether all the successor permute nodes have same order
    '''

    def successors_have_same_order(node):
      if len(node.successors()) < 2:
        return True

      successors = node.successors()
      order_shape = successors[0].permute_param.order
      first_successor_order = [order_shape[i] for i in range(len(order_shape))]

      for successor in successors:
        order_shape = successor.permute_param.order
        successor_order = [order_shape[i] for i in range(len(order_shape))]
        if first_successor_order != successor_order:
          return False
      return True

    '''
      Function to remove successor inline permutes recursively(Depth wise)
    '''

    def remove_successor_duplicates(node):
      i = 0
      while i < len(node.successors()) and node.successors()[i] in permutes:
        permutes.remove(node.successors()[i])
        remove_successor_duplicates(node.successors()[i])
        i += 1

    for layer in all_permutes:
      cur_layer = layer
      if cur_layer in permutes and successors_are_permutes(cur_layer):
        if successors_have_same_order(cur_layer):
          remove_successor_duplicates(cur_layer)

    idx = 0
    while idx < len(permutes):
      layer = permutes[idx]
      order_shape = layer.permute_param.order
      cur_order = [order_shape[i] for i in range(len(order_shape))]
      increment = True

      # Check: The Patching will be done only if all the successors are Permutes and of same order
      if not successors_are_permutes(layer) or not successors_have_same_order(layer):
        idx = idx + 1 if increment else idx
        continue

      #Parse through each of the successor of permute layer
      for successor in list(layer.successors()):
        if successor.type == 'PermuteNnp':
          #Get the successor of the current layer
          cur_layer = successor
          next_layers = list(cur_layer.successors())
          cur_layer_shape = cur_layer.permute_param.order
          #Rearrange the original order based on successors permute order
          new_order = [cur_order[i] for i in cur_layer_shape]
          cur_order = copy.deepcopy(new_order)
          #Point the predecessor of the current layer to successors
          for l in next_layers:
            self.G.remove_edge(cur_layer, l)
            self.G.add_edge(layer, l)
            self.G.add_replace_edge_proto_new(layer, l, cur_layer.name, layer.top[0])
          self.G.remove_edge(layer, cur_layer)
          self.G.remove_node(cur_layer)
          increment = False
      #If the final order is same as original order, remove the original permute layer aswell
      is_required = False
      for i in range(len(cur_order)):
        if i != cur_order[i]:
          is_required = True
          break
      if not is_required:
        prev_layer = layer.predecessors()[0]
        next_layers = list(layer.successors())
        for l in next_layers:
          self.G.remove_edge(layer, l)
          self.G.add_edge(prev_layer, l)
          self.G.add_replace_edge_proto_new(prev_layer, l, layer.name, prev_layer.top[0])
        self.G.remove_edge(prev_layer, layer)
        self.G.remove_node(layer)
        increment = True
      #Assign the final order to the first permute layer
      else:
        for i in range(len(order_shape)):
          order_shape[i] = cur_order[i]
      idx = idx + 1 if increment else idx


class PermuteNnp(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    permutes = self.get_nodes_of_type('Permute')
    for permute in permutes:
      perm_order = list(permute.get_layer().permute_param.order)
      if perm_order == [0, 1, 3, 2]:
        bottom_layer = permute.predecessors()[0]
        permute_0 = self.G.mk_permute_nnp_proto(permute.name, permute.bottom[0], perm_order, permute.isKeyOutput, permute.isValueOutput)
        successors = list(permute.successors())
        self.G.remove_node(permute)
        self.G.add_node(permute_0)
        self.G.add_edge(bottom_layer, permute_0)
        # self.G.remove_edge(bottom_layer, permute)
        for s in successors:
          self.G.add_edge(permute_0, s)
          # self.G.remove_edge(permute, s)
          self.G.add_replace_edge_proto_new(permute_0, s, permute.top[0], permute_0.top[0])


class RemoveRedundantPermute1(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    permutes = self.get_nodes_of_type('Permute')
    for permute in permutes:
      perm_order = list(permute.get_layer().permute_param.order)
      if perm_order == [i for i in range(len(perm_order))]:
        bottom_layer = permute.predecessors()[0]
        # permute_0 = mk_permute_nnp_proto(permute.bottom[0],perm_order,0)
        # self.G.add_node(permute_0)
        # self.G.add_edge(bottom_layer,permute_0)
        # self.G.remove_edge(bottom_layer,permute)

        successors = list(permute.successors())
        for s in successors:
          self.G.add_edge(bottom_layer, s)
          self.G.add_replace_edge_proto_new(bottom_layer, s, permute.top[0], bottom_layer.top[0])
        self.G.remove_node(permute)


class Ara2OptimizedPermute(BasePass):
  def __init__(self):
    pass

  def do_transformation(self):
    permutes = self.get_nodes_of_type('Permute')
    for layer in permutes:
      cur_layer = layer
      perm_order = cur_layer.permute_param.order
      layer_top_name = cur_layer.top[0]
      input_shape = self.caffe_net.blobs[cur_layer.bottom[0]].shape
      output_shape = self.caffe_net.blobs[layer_top_name].shape
      
      print("perm order is", perm_order , "shape length", len(input_shape))
      print("input_shape", [int(val) for val in input_shape])
      print("output_shape", [int(val) for val in output_shape])
      if len(input_shape) == 3 and (perm_order == [2, 0, 1] or perm_order == [0, 2, 1]):
        # case for permute of NCW to NWC
        print("considering padding to layer", input_shape[1], input_shape[2])
        if input_shape[2] % 32 != 0:
          shape1 = (input_shape[2] + 31)//32 * 32
          padding_size = shape1 - input_shape[2]
          print("changing dims for layer", input_shape[2], shape1, padding_size)

          pad_layer = mk_pad_proto([0, padding_size, 0, 0], 0, 0, cur_layer.name)
          predecessor = cur_layer.predecessors()[0]
          self.G.add_node(pad_layer)
          self.G.remove_edge(predecessor, cur_layer)
          self.G.add_edge(pad_layer, cur_layer)
          self.G.add_edge(predecessor, pad_layer)
          self.G.add_replace_edge_proto(predecessor, cur_layer, predecessor.top[0])


          #import pdb; pdb.set_trace()
          print("adding padding layer", pad_layer.name, pad_layer.padding_param.pad)

          crop_layer, dummy_layer = mk_crop_proto(cur_layer.name, output_shape, [0, 0, 0, 0])

          sucessors = cur_layer.sucessors()

          self.G.add_node(crop_layer)
          self.G.add_node(dummy_layer)
          

          self.G.add_edge(cur_layer, crop_layer)
          self.G.add_edge(dummy_layer, crop_layer)

          if sucessors is not None :
            for successor in sucessors:
              self.G.remove_edge(cur_layer, successor)
              self.G.add_edge(crop_layer, successor)
              self.G.add_replace_edge_proto(crop_layer, successor, cur_layer.top[0])

          self.G.add_replace_edge_proto(cur_layer, crop_layer, "NA")
          self.G.add_replace_edge_proto(dummy_layer, crop_layer, "NA")
          


      if len(input_shape) == 4 and perm_order == [0, 3, 2, 1]:
        # case for permute of NCHW to NWHC
        print("considering padding to layer")
        if input_shape[3] % 32 != 0:
          shape1 = (input_shape[3] + 31)//32 * 32
          padding_size = shape1 - input_shape[3]
          print("changing dims for layer", input_shape[2], shape1, padding_size)
          pad_layer = mk_pad_proto([0, padding_size, 0, 0], 0, 0, cur_layer.name)
          #import pdb; pdb.set_trace()
          print("adding padding layer", pad_layer.name, pad_layer.padding_param.pad)
          crop_layer, dummy_layer = mk_crop_proto(cur_layer.name, output_shape, [0, 0, 0, 0])
        
class PermuteLarge(BasePass):

  def __init__(self):
    pass
    # example case
    # permute layer -> reduce_layer
    # permute input shape = 1,100,100,300, order 0,2,1,3
    # reduce axis = 3
    # move reduce above permute so that permute's input shape gets reduced
  def do_transformation(self):
    permutes = self.get_nodes_of_type('Permute')
    permutes = [i for i in permutes if i.permute_param.order == [0, 2, 1, 3]]
    for permute in permutes:
      if len(permute.successors()) and permute.successors()[0].reduction_param.axis == [3]:
        bottom_layer = permute.predecessors()[0]
        reduce_layer = permute.successors()[0]
        successors = list(reduce_layer.successors())
        self.G.remove_edge(bottom_layer, permute)
        self.G.remove_edge(permute, reduce_layer)
        self.G.add_edge(bottom_layer, reduce_layer)
        self.G.add_edge(reduce_layer, permute)
        self.G.add_replace_edge_proto_new(bottom_layer, reduce_layer, permute.top[0], bottom_layer.top[0])
        self.G.add_replace_edge_proto_new(reduce_layer, permute, bottom_layer.top[0], reduce_layer.top[0])
        for s in successors:
          self.G.add_edge(permute, s)
          self.G.remove_edge(reduce_layer, s)
          self.G.add_replace_edge_proto_new(permute, s, reduce_layer.top[0], permute.top[0])


class PermuteForHW(BasePass):
  # If possible do permutation swapping H and W
  def __init__(self):
    pass

  def do_transformation(self):
    if self.dynamic or self.dynamic_input:
      return
    permutes = self.get_nodes_of_type('Permute')
    for permute in permutes:
      if self.no_final_perm_opt and len(permute.successors()) == 0:
        continue
      perm_order = tuple(permute.get_layer().permute_param.order)
      cases = {
          0: (0, 2, 3, 1),
          1: (0, 3, 1, 2),
          2: (0, 2, 1),
          3: (0, 2, 3, 4, 1),
          4: (0, 4, 1, 2, 3),
          5: (0, 1, 3, 4, 2),
      }
      if perm_order in [cases[i] for i in cases]:
        case = [cases[i] for i in cases].index(perm_order)
        bottom_layer = permute.predecessors()[0]
        bottom_shape = self.caffe_net.blobs[permute.bottom[0]].data.shape
        bottom_shape_new = list(bottom_shape)
        top_shape = self.caffe_net.blobs[permute.top[0]].data.shape
        if case == 0:
          bottom_shape_new = [bottom_shape_new[0], 1, bottom_shape_new[1], bottom_shape_new[2] * bottom_shape_new[3]]
        if case == 1:
          bottom_shape_new = [bottom_shape_new[0], 1, bottom_shape_new[1] * bottom_shape_new[2], bottom_shape_new[3]]
        if case == 2:
          bottom_shape_new = [bottom_shape_new[0], 1, bottom_shape_new[1], bottom_shape_new[2]]
        if case == 3:
          bottom_shape_new = [bottom_shape_new[0], 1, 1, bottom_shape_new[1], bottom_shape_new[2] * bottom_shape_new[3] * bottom_shape_new[4]]
        if case == 4:
          bottom_shape_new = [bottom_shape_new[0], 1, 1, bottom_shape_new[1] * bottom_shape_new[2] * bottom_shape_new[3], bottom_shape_new[4]]
        if case == 5:
          bottom_shape_new = [bottom_shape_new[0], bottom_shape_new[1], bottom_shape_new[2], bottom_shape_new[3] * bottom_shape_new[4]]
        reshape_0 = self.G.mk_reshape_proto(permute.bottom[0], bottom_shape_new, permute.name)

        self.G.add_node(reshape_0)
        self.G.add_edge(bottom_layer, reshape_0)
        self.G.add_edge(reshape_0, permute)
        self.G.remove_edge(bottom_layer, permute)
        self.G.add_replace_edge_proto_new(reshape_0, permute, permute.bottom[0], reshape_0.top[0])
        new_perm_order = None
        if case in [0, 1, 2, 5]:
          new_perm_order = [0, 1, 3, 2]
        if case in [3, 4]:
          new_perm_order = [0, 1, 2, 4, 3]
        permute.permute_param.ClearField('order')
        permute.permute_param.order.extend(new_perm_order)
        top_layers = list(permute.successors())
        reshape_1 = self.G.mk_reshape_proto(permute.name, list(top_shape), permute.name)
        self.G.add_node(reshape_1)
        self.G.add_edge(permute, reshape_1)
        for t in top_layers:
          self.G.add_edge(reshape_1, t)
          self.G.remove_edge(permute, t)
          self.G.add_replace_edge_proto_new(reshape_1, t, permute.top[0], reshape_1.top[0])


class PermuteToReshape(BasePass):
  # if a permute is permuting 2 adjacent axes, one of which is of length 1
  # then replace permute whith reshape
  def __init__(self):
    pass

  def is_valid(self, order):
    indexes = []
    for i, j in enumerate(order):
      if i != j:
        indexes.append(i)
    if len(indexes) == 0:
      return True, indexes
    elif len(indexes) == 2 and abs(indexes[0] - indexes[1]) == 1:
      return True, indexes
    return False, indexes

  def map_to_reshape(self, permute, indexes):
    input_shape = self.caffe_net.blobs[permute.bottom[0]].data.shape
    top_shape = self.caffe_net.blobs[permute.top[0]].data.shape
    if len(indexes) > 1 and input_shape[indexes[0]] != 1 and input_shape[indexes[1]] != 1:
      return
    reshape_0 = self.G.mk_reshape_proto(permute.bottom[0], list(top_shape), permute.name)
    self.G.add_node(reshape_0)
    self.G.add_edge(permute.predecessors()[0], reshape_0)
    for t in list(permute.successors()):
      self.G.add_edge(reshape_0, t)
      self.G.add_replace_edge_proto_new(reshape_0, t, permute.top[0], reshape_0.top[0])
    self.G.remove_node(permute)

  def do_transformation(self):
    permutes_nnp = self.get_nodes_of_type('PermuteNnp')
    permutes = self.get_nodes_of_type('Permute')
    for permute in permutes:
      if self.no_final_perm_opt and len(permute.successors()) == 0:
        continue
      check, indexes = self.is_valid(permute.permute_param.order)
      if not check:
        continue
      self.map_to_reshape(permute, indexes)
    for permute in permutes_nnp:
      if self.no_final_perm_opt and len(permute.successors()) == 0:
        continue
      check, indexes = self.is_valid(permute.permute_nnp_param.order)
      if not check:
        continue
      self.map_to_reshape(permute, indexes)

class SimplifyReshapePermuteSequence(BasePass):
  """
  Identifies linear chains of Reshape, Permute, and PermuteNnp operations,
  simplifies them using the simplify_sequence logic, and updates the graph.
  """
  def __init__(self):
    super(SimplifyReshapePermuteSequence, self).__init__()

  def do_transformation(self):
    def simplify_sequence(input_shape, sequence):
      """
      Simplifies a sequence of Reshape ('R') and Permute ('P') operations.
      Returns a simplified sequence with 0 or 1 Permutes, or None if impossible.
      Format of sequence:[('R', (new_shape)), ('P', (axes)), ...]
      """
      axes = []
      stride = 1
      for s in reversed(input_shape):
        if s > 1:
          axes.append([(s, stride)])
        else:
          axes.append([])
        stride *= s
      axes = axes[::-1]
      
      for op in sequence:
        if op[0] == 'P':
          axes = [axes[i] for i in op[1]]
        elif op[0] == 'R':
          new_shape = op[1]
          flat_blocks = []
          for axis in axes:
            flat_blocks.extend(axis)
            
          # Compact contiguous blocks
          compacted = []
          for s, w in flat_blocks:
            if not compacted:
              compacted.append((s, w))
            else:
              prev_s, prev_w = compacted[-1]
              if prev_w == s * w:
                compacted[-1] = (prev_s * s, w)
              else:
                compacted.append((s, w))
                
          new_axes = []
          for t in new_shape:
            if t == 1:
              new_axes.append([])
              continue
            current_axis = []
            current_s = 1
            while current_s < t:
              if not compacted: return None
              s, w = compacted.pop(0)
              
              if current_s * s <= t:
                current_axis.append((s, w))
                current_s *= s
              else:
                if t % current_s != 0: return None
                needed = t // current_s
                if s % needed != 0: return None
                current_axis.append((needed, w * (s // needed)))
                compacted.insert(0, (s // needed, w))
                current_s *= needed
            new_axes.append(current_axis)
          if compacted: return None
          axes = new_axes

      final_blocks = []
      for axis in axes:
        final_blocks.extend(axis)
        
      curr_shape = input_shape
      for op in sequence:
        if op[0] == 'R': curr_shape = tuple(op[1])
        elif op[0] == 'P': curr_shape = tuple(curr_shape[i] for i in op[1])

      if not final_blocks:
        return [('R', curr_shape)]

      sorted_blocks = sorted(final_blocks, key=lambda x: x[1], reverse=True)
      sorted_sizes = [b[0] for b in sorted_blocks]
      P = [sorted_blocks.index(b) for b in final_blocks]
      
      simplified = []
      if tuple(sorted_sizes) != input_shape:
        simplified.append(('R', tuple(sorted_sizes)))
      if P != list(range(len(P))): 
        simplified.append(('P', tuple(P)))
        
      if not simplified or simplified[-1] != ('R', curr_shape):
        if simplified and simplified[-1][0] == 'R':
          simplified.pop()
        simplified.append(('R', curr_shape))
        
      return simplified

    # We need to process nodes carefully. Once a sequence is replaced, 
    # the graph topology changes. We use a visited set to avoid reprocessing.
    visited = set()
    
    # Target operations for our sequence chains
    target_ops = ['Reshape', 'Permute', 'PermuteNnp']
    
    # Get all nodes that are potentially part of a chain
    candidates = [n for n in self.G.get_topsort() if n.type in target_ops]
    
    for node in candidates:
      	#skip for reshape->permutennp->reshape->permutennp->reshape->conv32x32 pattern in clip_vit_B32
      if node.name.startswith('_conv1_Conv_ara'):
        continue
      if node in visited:
        continue
      
      # Check if this node is the START of a sequence.
      preds = list(node.predecessors())
      if len(preds) == 1:
        prev = preds[0]
        if prev.type in target_ops and len(list(prev.successors())) == 1:
          # This node is in the middle of a chain, skip it. 
          continue

      # Start tracing the chain forward from 'node'
      chain = []
      curr = node
      valid_chain = True
      
      while curr and curr.type in target_ops:
        chain.append(curr)
        visited.add(curr)
        
        succs = list(curr.successors())
        if len(succs) == 1:
          next_node = succs[0]
          if next_node.type in target_ops:
            curr = next_node
          else:
            curr = None # End of chain
        else:
          curr = None # End of chain or branching
      
      if len(chain) < 2:
        continue

      # --- Preparation for Simplification ---
      first_node = chain[0]
      if len(first_node.predecessors()) == 0:
        continue 
      
      predecessor_node = first_node.predecessors()[0]
      if first_node.bottom[0] not in self.caffe_net.blobs:
        continue
        
      input_shape = tuple(map(int, self.caffe_net.blobs[first_node.bottom[0]].data.shape))
      curr_shape = input_shape
      
      # 2. Build Sequence Definition
      sequence_def = []
      permute_count = 0
      for n in chain:
        if n.type == 'Reshape':
          if n.top[0] not in self.caffe_net.blobs:
            valid_chain = False
            break
          curr_shape = tuple(map(int, self.caffe_net.blobs[n.top[0]].data.shape))
          sequence_def.append(('R', curr_shape))
          
        elif n.type == 'Permute':
          order = tuple(n.permute_param.order)
          curr_shape = tuple(curr_shape[i] for i in order)
          sequence_def.append(('P', order))
          permute_count += 1
          
        elif n.type == 'PermuteNnp':
          # PermuteNnp swaps across height and width (usually the last two dimensions)
          order = tuple(n.permute_nnp_param.order)
          curr_shape = tuple(curr_shape[i] for i in order)
          sequence_def.append(('P', order))
          permute_count += 1

      if not valid_chain:
        continue
      if permute_count < 2:
        continue

      # 3. Run Simplification Logic
      simplified_ops = simplify_sequence(input_shape, sequence_def)
      
      if simplified_ops is None:
        continue
        
      if len(simplified_ops) >= len(chain):
        pass

      # --- Apply Transformation (Graph Surgery) ---
      
      # Disconnect the Chain
      last_node = chain[-1]
      successors = list(last_node.successors())
      
      self.G.remove_edge(predecessor_node, first_node)
      for s in successors:
        self.G.remove_edge(last_node, s)
        
      for n in chain:
        self.G.remove_node(n)
        
      # Create New Nodes
      current_top_name = predecessor_node.top[0]
      previous_layer_node = predecessor_node
      
      for i, (op_type, params) in enumerate(simplified_ops):
        new_layer_name = "{}_optimized_{}".format(first_node.name, i)
        
        if op_type == 'R':
          new_shape = list(params)
          new_node = self.G.mk_reshape_proto(previous_layer_node.top[0], new_shape, new_layer_name)
          
        elif op_type == 'P':
          new_order = list(params)
          
          # Check if the generated permutation is strictly a Height/Width swap 
          # so we can use PermuteNnp instead of standard Permute.
          is_hw_swap = False
          if len(new_order) >= 2:
            expected_nnp_order = list(range(len(new_order)))
            expected_nnp_order[-2], expected_nnp_order[-1] = expected_nnp_order[-1], expected_nnp_order[-2]
            if new_order == expected_nnp_order:
              is_hw_swap = True
              
          if is_hw_swap:
            # Note: adjust args below if mk_permute_nnp_proto does not take `new_order` as a parameter
            new_node = self.G.mk_permute_nnp_proto(new_layer_name, previous_layer_node.top[0], new_order)
          else:
            new_node = self.G.mk_permute_proto(new_layer_name,new_order, previous_layer_node.top[0])
        
        self.G.add_node(new_node)
        self.G.add_edge(previous_layer_node, new_node)
        
        previous_layer_node = new_node
        current_top_name = new_node.top[0]

      # Reconnect to the original successors
      final_new_node = previous_layer_node
      
      for s in successors:
        self.G.add_edge(final_new_node, s)
        self.G.add_replace_edge_proto_new(final_new_node, s, last_node.top[0], final_new_node.top[0])

class PermuteForSpacialDist(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    spdt = self.get_nodes_of_type('SpacialDist')
    spdt = [i for i in spdt if i.spacial_dist_param.axes == [1, 2]]
    for spdt_node in spdt:
      perm_order = [0, 3, 1, 2]
      last_node = spdt_node
      bottom_layer = spdt_node.predecessors()[0]
      permute_0 = self.G.mk_permute_proto(bottom_layer.name, perm_order, spdt_node.name)
      self.G.add_node(permute_0)
      self.G.add_edge(bottom_layer, permute_0)
      self.G.add_replace_edge_proto_new(bottom_layer, permute_0, permute_0.bottom[0], bottom_layer.top[0])
      self.G.add_edge(permute_0, spdt_node)
      self.G.add_replace_edge_proto_new(permute_0, spdt_node, spdt_node.bottom[0], permute_0.top[0])
      self.G.remove_edge(bottom_layer, spdt_node)
      spdt_child = spdt_node.successors()
      if len(spdt_child) and spdt_child[0].type == 'MeanRatio':
        mnrt_child = spdt_child[0].successors()
        if len(mnrt_child) and mnrt_child[0].type == 'GlobalResponseNorm':
          grn = mnrt_child[0]
          last_node = grn
      
      if last_node == grn and bottom_layer.name in [i.name for i in grn.predecessors()]:
        idx = 0
        if len(grn.predecessors()) == 2 and grn.predecessors()[1] == bottom_layer.name:
          idx = 1
        permute_1 = self.G.mk_permute_proto(bottom_layer.name, perm_order, grn.name)
        self.G.add_node(permute_1)
        self.G.add_edge(bottom_layer, permute_1)
        self.G.add_replace_edge_proto_new(bottom_layer, permute_1, permute_1.bottom[0], bottom_layer.top[0])
        self.G.add_edge(permute_1, grn)
        self.G.add_replace_edge_proto_new(permute_1, grn, grn.bottom[idx], permute_1.top[0])
        self.G.remove_edge(bottom_layer, grn)
        
      perm_order = [0, 2, 3, 1]
      permute_2 = self.G.mk_permute_proto(last_node.name, perm_order)
      self.G.add_node(permute_2)
      for s in list(last_node.successors()):
        self.G.add_edge(permute_2, s)
        self.G.add_replace_edge_proto_new(permute_2, s, s.bottom[0], permute_2.top[0])
        self.G.remove_edge(last_node, s)
      self.G.add_edge(last_node, permute_2)
      self.G.add_replace_edge_proto_new(last_node, permute_2, permute_2.bottom[0], last_node.top[0])
      spdt_node.spacial_dist_param.axes[0], spdt_node.spacial_dist_param.axes[1] = 2, 3
      grn.global_response_norm_param.axis = 1
      spdt_child[0].mean_ratio_param.axis = 1
  