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

from dv_logging import get_logger
from mmdnn.conversion.caffe.c2c.common import BasePass

logger = get_logger(__name__, 'DVCTOC')

class RemoveRedundantDVShift(BasePass):
  '''
  Removed redundant DVShift layers. 
  1. If the DVShift layer is followed (eventually) by a DVCast layer or FP32 layer, 
     assign the parent precision to src_precision of DVCast and remove DVShift.
     Additionally, update the precision of any intermediate pass_through layers to match.
  2. If the DVShift layer is at the end of the network, remove it
  '''
  def __init__(self):
    pass

  def do_transformation(self):
    layers = self.get_nodes_of_type('DVShift')
    pass_through_layers = ["Reshape", "Permute", "PermuteNnp"]
    precision_enum = {'INT8': 0, 'INT16': 1, 'INT32': 2, 'FP32': 3}
    removed_layers = []
    
    # RULE 1: DVShift followed by DVCast or FP32
    for layer in layers:
      if layer.name in removed_layers:
        continue
      
      prev_layers = list(layer.predecessors())
      if not prev_layers:
        continue
      parent = prev_layers[0]
      
      successors = list(layer.successors())
      if len(successors) != 1:
        continue
      
      immediate_successor = successors[0]
      target_successor = immediate_successor
      
      # Traverse forward to look through any pass_through_layers
      is_linear_path = True
      intermediate_pt_layers = []  # <--- Track layers between DVShift and Target
      
      while target_successor.type in pass_through_layers:
        intermediate_pt_layers.append(target_successor)
        next_successors = list(target_successor.successors())
        if len(next_successors) != 1:
          is_linear_path = False
          break
        target_successor = next_successors[0]
        
      if not is_linear_path:
        continue

      is_fp32 = len(target_successor.get_layer().adj_param.precision) and target_successor.get_layer().adj_param.precision[0] == 3
      is_dvcast = target_successor.type == "DVCast"

      if is_fp32 or is_dvcast:
        dtype_par = parent
        while dtype_par.type in pass_through_layers:
          dtype_par = list(dtype_par.predecessors())[0]
        
        # Check if original parent source is INT16
        if len(dtype_par.get_layer().adj_param.precision) and dtype_par.get_layer().adj_param.precision[0] == 1:
          
          # If target is DVCast, we update its expectation to INT16
          if is_dvcast:
            target_successor.dvcast_param.src_precision = precision_enum['INT16']
            
            # ---> NEW: Update precision metadata for intermediate pass-through layers
            for pt_node in intermediate_pt_layers:
              pt_layer = pt_node.get_layer()
              if len(pt_layer.adj_param.precision):
                pt_layer.adj_param.precision[0] = precision_enum['INT16']
              else:
                pt_layer.adj_param.precision.append(precision_enum['INT16'])
        
        # GRAPH REWIRING: Remove DVShift and bridge Parent to the Immediate Successor
        removed_layers.append(layer.name)
        self.G.remove_edge(layer, immediate_successor)
        self.G.remove_edge(parent, layer)
        self.G.remove_node(layer)
        self.G.add_edge(parent, immediate_successor)
        self.G.add_replace_edge_proto(parent, immediate_successor, layer.name)
    
    # RULE 2: DVShift at the end of the network
    for layer in layers:
      if layer.name in removed_layers:
        continue
      successors = list(layer.successors())
      parent = list(layer.predecessors())[0]
      while len(successors) == 1 and successors[0].type in pass_through_layers:
        successors = list(successors[0].successors())
      
      if len(successors) == 0:
        immediate_successors = list(layer.successors())
        if len(immediate_successors) == 0:
          self.G.remove_edge(parent, layer)
          self.G.remove_node(layer)
          removed_layers.append(layer.name)
        elif len(immediate_successors) == 1:
          self.G.remove_edge(layer, immediate_successors[0])
          self.G.remove_edge(parent, layer)
          self.G.remove_node(layer)
          self.G.add_edge(parent, immediate_successors[0])
          self.G.add_replace_edge_proto(parent, immediate_successors[0], layer.name)
          removed_layers.append(layer.name)
class RemoveDvshiftReshapePerm(BasePass):
  '''
  When there is a reshape-permute sequence between a DVShift (8B to 16B) and a DVShift(16B to 8B), then remove these DVShifts
  When there is a reshape-permute sequence between a DVShift (16B to 8B) and a DVShift(8B to 16B), then remove these DVShifts
  '''
  def __init__(self):
    pass

  def do_transformation(self):
    dvshift_layers = self.get_nodes_of_type('DVShift')
    intermediate_layers_list = ["Reshape", "Permute", "PermuteNnp", "Crop"]
    removed_layers = []
    for layer in dvshift_layers:
      if layer.name not in removed_layers:
        dvshift_s = layer.successors()
        while(len(dvshift_s) == 1):
          if dvshift_s[0].type in intermediate_layers_list:
            dvshift_s = dvshift_s[0].successors()
          else:
            break
        if dvshift_s[0].type == 'DVShift' and dvshift_s[0].adj_param != layer.adj_param:
          if len(layer.predecessors()) == 1 and len(dvshift_s[0].successors()) == 1:
            p1 = layer.predecessors()[0]
            p2 = dvshift_s[0].predecessors()[0]
            s1 = layer.successors()[0]
            s2 = dvshift_s[0].successors()[0] 
            removed_layers.append(layer.name)
            removed_layers.append(dvshift_s[0].name)
            self.G.remove_edge(p1,layer)
            self.G.remove_edge(layer,s1)
            self.G.add_edge(p1,s1)
            self.G.add_replace_edge_proto(p1, s1, layer.name)
            self.G.remove_node(layer)

            self.G.remove_edge(p2,dvshift_s[0])
            self.G.remove_edge(dvshift_s[0],s2)
            self.G.add_edge(p2,s2)
            self.G.add_replace_edge_proto(p2, s2, dvshift_s[0].name)
            self.G.remove_node(dvshift_s[0])

class RemoveDVShiftReshape(BasePass):
  '''
  DVShift(8B to 16B) + Reshape + DVShift(16B to 8B) -> Remove DVShifts
  DVShift(16B to 8B) + Reshape + DVShift(8B to 16B) -> Remove DVShifts
  '''
  def __init__(self):
    pass

  def do_transformation(self):
    layers = self.get_nodes_of_type('Reshape')
    precision_enum = {'INT8': 0, 'INT16': 1, 'INT32': 2, 'FP32': 3}
    removed_layers = []
    for layer in layers:
      if layer.name in removed_layers:
        continue
      prev_layers = list(layer.predecessors())
      parent = prev_layers[0]
      succesors = list(layer.successors())
      if len(succesors) != 1:
        continue
      successor = succesors[0]
      if not(parent.type == "DVShift" and successor.type == "DVShift"):
        continue

      # TODO: Handle case where there are multiple successors to DVShift after the Reshape
      p_to_parent = list(parent.predecessors())[0] if len(parent.predecessors()) else None
      c_to_successor = list(successor.successors())[0] if len(successor.successors()) else None
      if p_to_parent is not None and c_to_successor is not None:
        self.G.remove_edge(p_to_parent, parent)
        for parent_s in parent.successors():
          if parent_s.type == "DVShift" and parent_s.adj_param != parent.adj_param: #in this case we can remove the successor dvshift as well, but we should add an additional check to verify that the last dvshift before parent has the same adj param as the parent_s dvshift. 
            parent_ss = parent_s.successors()[0]  # hardcoding this for now
            self.G.remove_edge(parent, parent_s)
            self.G.remove_edge(parent_s, parent_ss)
            self.G.remove_node(parent_s)
            self.G.add_edge(p_to_parent, parent_ss)
            self.G.add_replace_edge_proto(p_to_parent, parent_ss, parent_s.name)
          else:
            self.G.remove_edge(parent, parent_s)
            self.G.add_edge(p_to_parent, parent_s)
            self.G.add_replace_edge_proto(p_to_parent, parent_s, parent.name)
        self.G.remove_node(parent)

        self.G.remove_edge(layer, successor)
        self.G.remove_edge(successor, c_to_successor)
        self.G.remove_node(successor)
        self.G.add_edge(layer, c_to_successor)
        self.G.add_replace_edge_proto(layer, c_to_successor, successor.name)

class DVShiftPatching(BasePass):
  def __init__(self):
    pass

  def do_transformation(self):
    layers = self.get_nodes_of_type('DVShift')
    removed_layers = []
    for layer in layers:
      if layer.name in removed_layers:
        continue
      prev_layers = list(layer.predecessors())
      parent = prev_layers[0]
      succesors = list(layer.successors())
      if len(succesors) != 1:
        continue
      successor = succesors[0]
      if successor.type != 'DVShift':
        continue
      removed_layers.append(layer.name)
      removed_layers.append(successor.name)
      succesors = list(successor.successors())
      self.G.remove_edge(layer, successor)
      self.G.remove_edge(parent, layer)
      self.G.remove_node(layer)
      for s in succesors:
        self.G.remove_edge(successor, s)
        self.G.add_edge(parent, s)
        self.G.add_replace_edge_proto(parent, s, successor.name)
      self.G.remove_node(successor)

class DVShiftSplit(BasePass):
  def __init__(self):
    super().__init__()

  def do_transformation(self):
    # Create a static list of layers to iterate over to avoid issues 
    # when adding/removing nodes during iteration
    layers = list(self.get_nodes_of_type('DVShift'))
    
    for layer in layers:
      prev_layers = list(layer.predecessors())
      if not prev_layers:
        continue
        
      parent = prev_layers[0]
      successors = list(layer.successors())
      
      # If it only has 1 output, no splitting is needed
      if len(successors) <= 1:
        continue

      # Separate branches into those that are DVShifts and those that aren't
      dv_successors = [s for s in successors if s.type == "DVShift"]
      non_dv_successors = [s for s in successors if s.type != "DVShift"]

      # If there are NO DVShift branches, do nothing (1 shared DVShift for all)
      if not dv_successors:
        continue

      # 1. Create dedicated DVShifts for the DVShift branches
      for idx, s in enumerate(dv_successors):
        new_name = f"{layer.name}_split_{idx}_"
        
        # Create the new node
        shft_layer = self.G.mk_dvshift_proto(parent.top[0], new_name)
        
        # Copy parameters
        shft_layer.dvshift_param.shift = layer.dvshift_param.shift
        for pr in layer.adj_param.precision:
          shft_layer.adj_param.precision.append(pr)
        
        self.G.add_node(shft_layer)
        
        # Disconnect this specific successor from the original layer
        self.G.remove_edge(layer, s)
        
        # Connect parent -> new layer -> successor
        self.G.add_edge(parent, shft_layer)
        self.G.add_edge(shft_layer, s)
        
        # Update underlying protobuf/tensor inputs
        self.G.add_replace_edge_proto(shft_layer, s, layer.name)

      # 2. Handle the original layer
      # The original layer is already connected to parent and `non_dv_successors`.
      # If `non_dv_successors` is empty, the original layer is now a dead end, so we remove it.
      if not non_dv_successors:
        self.G.remove_edge(parent, layer)
        self.G.remove_node(layer)

class InsertDVShift(BasePass):
  def __init__(self):
    pass

  def do_transformation(self):
    if self.adjust_file is None or "Layers" not in self.adjust_file:
      return
    for node in self.get_topsort():
      layer = node.get_layer()
      if layer.name in self.adjust_file["Layers"]:
        if "isCaffeName" in self.adjust_file["Layers"][layer.name] and self.adjust_file["Layers"][layer.name]["isCaffeName"]:
          predecessors = list(node.predecessors())
          node.adj_param.precision.append(1)
          layer.fuse = False
          for idx, p in enumerate(predecessors):
            shift_layer = self.G.mk_dvshift_proto(node.bottom[idx], f"{layer.name}_pre_dvshift_{idx}_")
            self.G.add_node(shift_layer)
            self.G.add_edge(p,shift_layer)
            self.G.add_edge(shift_layer, node)
            self.G.add_replace_edge_proto_new(shift_layer, layer, p.top[idx], shift_layer.top[idx])
            shift_layer.adj_param.precision.append(1)
            self.G.remove_edge(p, node)
          successors = list(node.successors())
          for idx,s in enumerate(successors):
            shift_layer_post = self.G.mk_dvshift_proto(node.top[0], f"{layer.name}_post_dvshift_{idx}_")
            self.G.add_node(shift_layer_post)
            self.G.add_edge(node, shift_layer_post)
            self.G.add_edge(shift_layer_post, s)
            self.G.add_replace_edge_proto(shift_layer_post, s, layer.name)
            self.G.remove_edge(node,s)
            