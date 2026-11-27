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
#
# Add layer_type to list to skip DV_CAST being added.
# Layer types will be marked as "needs_generic_cp" as true, but dataprec will be set to INT8.
# Should be enabled for cp implemented DMA operations.
#
# Removed Onnx_Slice from skip_list because in videopose3d this was preventing dvcast being added around onnx_slice, which made output data type fp32 and succeeding eltwise got int8 and fp32 as input.
skip_list = ['Mxnet_Flip', 'Mxnet_Slice_Like', 'Mxnet_Repeat', 'Mxnet_Embedding', 'Scatternd', 'Aten', 'DuplicateN','CPMultiout']
skip_list_16bit = ['Onnx_Batchpermutation', 'GenerateProposals', 'DistributeFpn', 'CollectRpnProposals', 'BoxWithNMSLimit', 'BBoxTransform']
skip_first_input = ['BBoxTransform_detectron2']
dlp_only = ['Eltwise']
fuse_list = ['ReLU','PReLU','Swish','ReLU6','Sigmoid','HardSwish','Mish','Gelu']


class DVCastPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    cp_only_list = self.cp_only_list
    skip_fuse_list = self.skip_list
    caffe_cp_only_list = self.caffe_cp_only_list
    cp_precision = self.cp_precision
    layer_16_bit = [("CollectRpnProposals","DVShift"), \
                    ("Softmax","LoopEnd")]
    bottom_type = None
    bottom = None
    dvcast_layer = None
    for node in self.get_topsort():
      precision_enum = {'INT8': 0, 'INT16': 1, 'INT32': 2, 'FP32': 3}
      layer = node.get_layer()
      current_layer_name = layer.name
      current_layer_type = layer.type
      if self.fuse_act is False:
        if (layer.type in fuse_list and current_layer_name in skip_fuse_list): #or (layer.type in fuse_list and self.dry_run == 2) or (layer.type in fuse_list and (node.predecessors()[0].get_layer().type not in ["Convolution","BatchNorm"])):
          # cp_only_list.remove(current_layer_name)
          layer.fuse = False
        elif (layer.type in fuse_list and current_layer_name in cp_only_list and self.dry_run == 2):
          layer.fuse = False

      if self.dry_run:
        continue
      if current_layer_type == 'Reshape' and len(node.successors()) == 0:
        continue
      if layer.type in skip_list_16bit:
        logger.info("cp layer in skip_list_16bit. ignoring dvcast addition. %s", layer.name)
        continue
      if current_layer_name in cp_only_list and current_layer_name not in caffe_cp_only_list:
        logger.debug("cp layer: %s", current_layer_name)
        layer.dataPrecision = precision_enum[cp_precision[current_layer_name]]
        layer.needs_generic_cp_kernel = True
      elif current_layer_name in caffe_cp_only_list:
        logger.debug("caffe cp layer: %s", current_layer_name)
        layer.dataPrecision = precision_enum[cp_precision[current_layer_name]]
      else:
        logger.debug("non cp layer: %s", current_layer_name)
      predecessors = list(node.predecessors())
      _cnt = 0
      for idx, predecessor in enumerate(predecessors):
        """important : If output network of c2c is 
          fed again to c2c multiple dvcasts shouldn't 
          be added.
          """
        
        if layer.type in skip_list:
          logger.info("cp layer in skip_list. ignoring dvcast addition. %s", layer.name)
          layer.dataPrecision = precision_enum["INT8"]
          cp_precision[layer.name] = "INT8"
          continue
        for i in layer.bottom:
          for j in predecessor.top:
            if i == j:
              bottom_type = predecessor.get_layer().type
              bottom = predecessor.name
              name_to_be_replaced = j
        if bottom_type in skip_list_16bit:
          continue
        if current_layer_type in skip_first_input and idx == 0 and len(predecessors) != 1:
          logger.info("cp layer in skip_list. ignoring dvcast addition. %s", layer.name)
          continue
        if (bottom_type, current_layer_type) in layer_16_bit:
          if bottom in caffe_cp_only_list:
            dvcast_layer = self.G.mk_dvcast_proto(bottom, node.name,precision_enum[cp_precision[bottom]], precision_enum['INT16'])
          elif bottom_type == "Softmax":
            if len(node.successors()) != 0 and node.successors()[0].get_layer().type == "BoxWithNMSLimit":
              dvcast_layer = self.G.mk_dvcast_proto(bottom, node.name, precision_enum['INT32'], precision_enum['INT16'])
          elif bottom_type in list(list(zip(*layer_16_bit))[0]):  #["CollectRpnProposals", "Softmax"]:
            dvcast_layer = self.G.mk_dvcast_proto(bottom, node.name, precision_enum['INT16'], precision_enum['INT8'])
          self.G.add_node(dvcast_layer)
          self.G.remove_edge(predecessor, node)
          self.G.add_edge(predecessor, dvcast_layer)
          self.G.add_edge(dvcast_layer, node)
          assert dvcast_layer is not None
          self.G.add_replace_edge_proto_new(dvcast_layer, node, bottom, dvcast_layer.top[0])
          _cnt = _cnt + 1
        elif (current_layer_name in cp_only_list and current_layer_name not in caffe_cp_only_list) and (bottom not in cp_only_list and
                                                                                                        bottom not in caffe_cp_only_list):
          dvcast_layer = self.G.mk_dvcast_proto(bottom, node.name, precision_enum['INT8'], precision_enum[cp_precision[current_layer_name]])
          self.G.add_node(dvcast_layer)
          self.G.remove_edge(predecessor, node)
          self.G.add_edge(predecessor, dvcast_layer)
          self.G.add_edge(dvcast_layer, node)
          self.G.add_replace_edge_proto_new(dvcast_layer, node, bottom, dvcast_layer.top[0])
          _cnt = _cnt + 1
        elif (bottom in cp_only_list or bottom in caffe_cp_only_list) and cp_precision[bottom] != "INT8" and current_layer_name not in cp_only_list:
          for i in layer.bottom:
            for j in predecessor.top:
              if i == j:
                bottom_blob = j
          dvcast_layer = self.G.mk_dvcast_proto(bottom_blob, node.name, precision_enum[cp_precision[bottom]], precision_enum['INT8'])
          self.G.add_node(dvcast_layer)
          self.G.remove_edge(predecessor, node)
          self.G.add_edge(predecessor, dvcast_layer)
          self.G.add_edge(dvcast_layer, node)
          self.G.add_replace_edge_proto_new(dvcast_layer, node, name_to_be_replaced, dvcast_layer.top[0])
          _cnt = _cnt + 1
        elif bottom in caffe_cp_only_list and current_layer_name in cp_only_list and cp_precision[current_layer_name] != cp_precision[bottom]:
          dvcast_layer = self.G.mk_dvcast_proto(bottom, node.name, precision_enum[cp_precision[bottom]], precision_enum[cp_precision[current_layer_name]])
          self.G.add_node(dvcast_layer)
          self.G.remove_edge(predecessor, node)
          self.G.add_edge(predecessor, dvcast_layer)
          self.G.add_edge(dvcast_layer, node)
          self.G.add_replace_edge_proto_new(dvcast_layer, node, bottom, dvcast_layer.top[0])
          _cnt = _cnt + 1


class RemoveRedundantCast(BasePass):
  '''
  If a node is input to multiple dvcast successors, move dvcast up
  '''

  def __init__(self):
    pass

  def do_transformation(self):
    precision_enum = {'INT8': 0, 'INT16': 1, 'INT32': 2, 'FP32': 3}
    pass_through_layers = {'Reshape'}
    removed_nodes = set()
    for node in self.get_topsort():
      # If the node is already removed, continue
      if node.name in removed_nodes:
        continue
      if node.type in ["Slice"]:
        continue
      # Check if all the successors are dvcast
      successors = node.successors()
      non_dvcast_layer = False
      for s in successors:
        while(s.type in pass_through_layers and len(s.successors())):
          s = s.successors()[0]
        if s.type != "DVCast":
          non_dvcast_layer = True
          break
        src_precision = s.dvcast_param.src_precision
        dst_precision = s.dvcast_param.dst_precision
      if non_dvcast_layer or len(successors) < 2:
        continue

      # Remove the dvcast layers from each branch
      for s in successors:
        while(s.type in pass_through_layers and len(s.successors())):
          s = s.successors()[0]
        pred = s.predecessors()[0]
        succ = s.successors()
        self.G.remove_edge(pred, s)
        for n in succ:
          self.G.remove_edge(s, n)
          self.G.add_edge(pred, n)
          self.G.add_replace_edge_proto(pred, n, s.top[0])
        self.G.remove_node(s)
        removed_nodes.add(s.name)

      # Add dvcast layer after the node
      dvcast_node = self.G.mk_dvcast_proto(node.top[0], node.name, src_precision, dst_precision)
      self.G.add_node(dvcast_node)
      self.G.add_edge(node, dvcast_node)

      for s in node.successors():
        if s.name != dvcast_node.name:
          self.G.remove_edge(node, s)
          self.G.add_edge(dvcast_node, s)
          self.G.add_replace_edge_proto(dvcast_node, s, node.top[0])
      
class RemoveRedundantCastII(BasePass):
  '''
  DVCast -> NoOp -> DVCast === NoOp
  '''

  def __init__(self):
    pass

  def do_transformation(self):
    pass_through_layers = {'Reshape'}
    layers = self.get_nodes_of_type('DVCast')
    removed_nodes = set()

    for layer in layers:
      if layer.name in removed_nodes:
        continue
      successors = layer.successors()
      predecessor = layer.predecessors()[0]
      layer_idx = 0
      for idx, name in enumerate(predecessor.top):
        for b_idx, b_name in enumerate(layer.bottom):
          if name == b_name:
            layer_idx = idx
            break
      if len(successors) != 1:
        continue
      successor = successors[0]
      if successor.type not in pass_through_layers:
        continue
      successors_to_successor = successor.successors()
      all_dvcasts = all(successor_to_successor.type == "DVCast" for successor_to_successor in successors_to_successor)
      if not all_dvcasts:
        continue
      for successor_to_successor in successors_to_successor:
        # next_layer = successor_to_successor.successors()[0]
        self.G.remove_edge(successor, successor_to_successor)
        for next_layer in successor_to_successor.successors():
          self.G.remove_edge(successor_to_successor, next_layer)
          self.G.add_edge(successor, next_layer)
          self.G.add_replace_edge_proto(successor, next_layer, successor_to_successor.top[0])
        self.G.remove_node(successor_to_successor)
        removed_nodes.add(successor_to_successor.name)
      self.G.remove_edge(predecessor, layer)
      self.G.remove_edge(layer, successor)
      self.G.add_edge(predecessor, successor)
      self.G.add_replace_edge_proto_new(predecessor, successor, layer.top[0], predecessor.top[layer_idx])
      self.G.remove_node(layer)
      removed_nodes.add(layer.name)

class RemoveRedundantCastGather(BasePass):
  '''
  Optimizes Onnx_Gather operations by removing DVCast layers surrounding them.
  
  Scenario: 
    Source -> DVCast -> [PassThrough Layers] -> Onnx_Gather -> [PassThrough Layers] -> DVCast -> Destination
  
  Transformation:
    Source -> [PassThrough Layers] -> Onnx_Gather -> [PassThrough Layers] -> Destination
    
  PassThrough Layers allowed: Reshape, Permute
  '''

  def __init__(self):
    pass

  def get_producer_node(self, node, input_blob_name):
    """Helper to find the specific predecessor node that produces a specific blob."""
    for pred in node.predecessors():
      if input_blob_name in pred.top:
        return pred
    return None

  def do_transformation(self):
    pass_through_types = ['Reshape', 'Permute']
    gather_nodes = self.get_nodes_of_type('Onnx_Gather')
    removed_nodes = set()

    for node in gather_nodes:
      if node.name in removed_nodes:
        continue

      # ==========================================
      # 1. Trace Output Side (Downstream)
      # ==========================================
      # Chain: Gather -> [PassThrough] -> DVCast
      
      output_dvcast = None
      current_node_out = node
      last_pass_through_node_out = node # The node immediately before the Output DVCast

      # We walk down until we hit a DVCast or a non-pass-through layer
      while True:
        successors = current_node_out.successors()
        # If branching or dead end, we can't safely assume simple chain optimization
        if len(successors) != 1:
          output_dvcast = None
          break
        
        succ = successors[0]
        
        if succ.type == 'DVCast':
          output_dvcast = succ
          break
        elif succ.type in pass_through_types:
          last_pass_through_node_out = succ
          current_node_out = succ
        else:
          # Hit a computation layer that isn't DVCast
          output_dvcast = None
          break

      if output_dvcast is None:
        continue

      # ==========================================
      # 2. Trace Input Side (Upstream) - DATA Only
      # ==========================================
      # Chain: DVCast -> [PassThrough] -> Gather
      # Gather Input[0] is Data, Input[1] is Indices. We strictly trace Input[0].

      input_dvcast = None
      current_node_in = node
      first_pass_through_node_in = node # The node immediately after the Input DVCast
      
      # The blob we are currently tracing back
      current_blob_to_trace = node.bottom[0] 

      while True:
        pred = self.get_producer_node(current_node_in, current_blob_to_trace)
        
        if pred is None: 
          input_dvcast = None 
          break

        if pred.type == 'DVCast':
          input_dvcast = pred
          # The node that DVCast feeds into (could be Gather or a Reshape)
          first_pass_through_node_in = current_node_in 
          break
        elif pred.type in pass_through_types:
          current_node_in = pred
          current_blob_to_trace = pred.bottom[0] # Move up the chain
        else:
          # Hit a computation layer that isn't DVCast
          input_dvcast = None
          break
      
      if input_dvcast is None:
        continue

      # Safety Check: Ensure Input DVCast doesn't feed other branches 
      # (If it does, we cannot simply remove it without affecting other paths)
      if len(input_dvcast.successors()) != 1:
        continue
      
      # ==========================================
      # 3. Apply Transformation
      # ==========================================
      
      # --- A. Bypass Input DVCast ---
      # Current: Source -> InputDVCast -> FirstPassThrough (or Gather)
      # New:     Source -> FirstPassThrough (or Gather)
      
      source_node = input_dvcast.predecessors()[0]
      
      self.G.remove_edge(source_node, input_dvcast)
      self.G.remove_edge(input_dvcast, first_pass_through_node_in)
      
      self.G.add_edge(source_node, first_pass_through_node_in)
      
      # Replace the input blob of the next node with the source's output blob
      # We use replace_edge_proto_new to be explicit about blob names
      self.G.add_replace_edge_proto_new(source_node, first_pass_through_node_in, input_dvcast.top[0], source_node.top[0])

      # --- B. Bypass Output DVCast ---
      # Current: LastPassThrough (or Gather) -> OutputDVCast -> Destinations
      # New:     LastPassThrough (or Gather) -> Destinations
      
      destinations = output_dvcast.successors()
      
      self.G.remove_edge(last_pass_through_node_out, output_dvcast)
      
      for dest in destinations:
        self.G.remove_edge(output_dvcast, dest)
        self.G.add_edge(last_pass_through_node_out, dest)
        
        # Update destination to look for LastPassThrough's output instead of DVCast's output
        self.G.add_replace_edge_proto(last_pass_through_node_out, dest, output_dvcast.top[0])

      # --- C. Cleanup ---
      self.G.remove_node(input_dvcast)
      self.G.remove_node(output_dvcast)
      removed_nodes.add(input_dvcast.name)
      removed_nodes.add(output_dvcast.name)

      # --- D. Update the dtype ---
      node.onnx_gather_param.dtype = 0


class RemoveRedundantShiftGather(BasePass):
  '''
  Optimizes Onnx_Gather operations by checking for the pattern:
  [DVShift] -> [Passthrough] -> Gather -> [Passthrough] -> [DVCast]
  and removing the DVShift layer if the pattern is present.
  
  Scenario: 
    Source -> DVShift -> [PassThrough Layers] -> Onnx_Gather -> [PassThrough Layers] -> DVCast -> Destination
  
  Transformation:
    Source -> [PassThrough Layers] -> Onnx_Gather -> [PassThrough Layers] -> DVCast -> Destination
    
  PassThrough Layers allowed: Reshape, Permute, PermuteNnp
  Note: Gather can have multiple downstream branches, but EVERY branch must end in a DVCast.
  '''

  def __init__(self):
    pass

  def get_producer_node(self, node, input_blob_name):
    """Helper to find the specific predecessor node that produces a specific blob."""
    for pred in node.predecessors():
      if input_blob_name in pred.top:
        return pred
    return None

  def do_transformation(self):
    pass_through_types = ['Reshape', 'Permute', 'PermuteNnp']
    gather_nodes = self.get_nodes_of_type('Onnx_Gather')
    removed_nodes = set()

    for node in gather_nodes:
      if node.name in removed_nodes:
        continue

      # We will collect all pass-through layers (upstream and downstream)
      # to update their precision if required.
      collected_pass_throughs = set()

      # ==========================================
      # 1. Trace Output Side (Downstream)
      # ==========================================
      # Chain: Gather -> [PassThrough] -> DVCast
      # We use BFS to trace all paths. Every path must terminate in DVCast.
      
      output_dvcasts = set()
      is_valid_pattern = True
      
      # Queue for BFS traversal
      queue = [node]

      while queue:
        current_node_out = queue.pop(0)
        successors = current_node_out.successors()
        
        # If a branch reaches a dead-end before hitting a DVCast, invalid pattern
        if not successors:
          is_valid_pattern = False
          break
        
        for succ in successors:
          if succ.type == 'DVCast':
            output_dvcasts.add(succ)
          elif succ.type in pass_through_types:
            if succ not in collected_pass_throughs:
              collected_pass_throughs.add(succ)
              queue.append(succ)
          else:
            # Hit a computation layer that isn't DVCast
            is_valid_pattern = False
            break
            
        if not is_valid_pattern:
          break

      # If the pattern is invalid or we didn't find any DVCasts, skip optimization
      if not is_valid_pattern or len(output_dvcasts) == 0:
        continue

      # ==========================================
      # 2. Trace Input Side (Upstream) - DATA Only
      # ==========================================
      # Chain: DVShift -> [PassThrough] -> Gather
      # Gather Input[0] is Data, Input[1] is Indices. We strictly trace Input[0].

      input_dvshift = None
      current_node_in = node
      first_pass_through_node_in = node # The node immediately after the Input DVShift
      
      # The blob we are currently tracing back
      current_blob_to_trace = node.bottom[0] 

      while True:
        pred = self.get_producer_node(current_node_in, current_blob_to_trace)
        
        if pred is None: 
          input_dvshift = None 
          break

        if pred.type == 'DVShift':
          input_dvshift = pred
          # The node that DVShift feeds into (could be Gather or a Reshape)
          first_pass_through_node_in = current_node_in 
          break
        elif pred.type in pass_through_types:
          collected_pass_throughs.add(pred)
          current_node_in = pred
          current_blob_to_trace = pred.bottom[0] # Move up the chain
        else:
          # Hit a computation layer that isn't DVShift
          input_dvshift = None
          break
      
      if input_dvshift is None:
        continue

      # Safety Check: Ensure Input DVShift doesn't feed other branches 
      # (If it does, we cannot simply remove it without affecting other paths)
      if len(input_dvshift.successors()) != 1:
        continue
      
      # ==========================================
      # 3. Apply Transformation
      # ==========================================
      
      # --- A. Bypass Input DVShift ---
      # Current: Source -> InputDVShift -> FirstPassThrough (or Gather)
      # New:     Source -> FirstPassThrough (or Gather)
      
      source_node = input_dvshift.predecessors()[0]
      
      # Remove links to/from DVShift
      self.G.remove_edge(source_node, input_dvshift)
      self.G.remove_edge(input_dvshift, first_pass_through_node_in)
      
      # Create new direct link bypassing DVShift
      self.G.add_edge(source_node, first_pass_through_node_in)
      
      # Replace the input blob of the next node with the source's output blob
      self.G.add_replace_edge_proto_new(source_node, first_pass_through_node_in, input_dvshift.top[0], source_node.top[0])

      # --- B. Update Gather dtype and DVCast src_precision ---
      if len(input_dvshift.adj_param.precision) and input_dvshift.adj_param.precision[0] == 1:
        node.onnx_gather_param.dtype = 0
        new_src_precision = 0
      else:
        node.onnx_gather_param.dtype = 1
        new_src_precision = 1
        
        # If precision is NOT 1, force adj_param.precision to 1 
        # on all connected pass through layers in the chain
        for pt_node in collected_pass_throughs:
          del pt_node.adj_param.precision[:]  # Clear existing array if populated
          pt_node.adj_param.precision.append(1)

      # Apply updated precision to ALL collected downstream DVCast nodes
      for output_dvcast in output_dvcasts:
        output_dvcast.dvcast_param.src_precision = new_src_precision

      # --- C. Cleanup ---
      # Per instructions, we ONLY remove DVShift. DVCast is retained.
      self.G.remove_node(input_dvshift)
      removed_nodes.add(input_dvshift.name)