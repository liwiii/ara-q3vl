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
import copy


class RemoveRedundantPermuteAttention(BasePass):
  def __init__(self):
    pass
  '''
  Unet network had multiple sections where a portion had permute at start and at end. Then the permuted outputs were combined (with other section's output) usng eltwise. This pattern matcher removes mulptile permutes and handles rest of network accordingly
  '''

  def do_transformation(self):
    layers = self.get_nodes_of_type('LayerNorm')
    skip = set()
    kk = 0
    for lnrm1 in layers:
      if lnrm1.name in skip:
        continue
      # pattern check
      if len(lnrm1.successors()) != 3:
        continue
      lnrm1_ch = []
      for i in lnrm1.successors():
        if i.type != 'Reshape':
          continue
        if len(i.successors()) != 1:
          continue
        j = i.successors()[0]        
        if j.type != 'Permute':
          continue
        if len(j.successors()) != 1:
          continue
        k = j.successors()[0]
        if k.type != 'Convolution':
          continue
        lnrm1_ch.append([i,j,k])
      if len(lnrm1.predecessors()) != 1:
        continue
      reshape1 = lnrm1.predecessors()[0]
      if reshape1.type != 'Reshape':
        continue
      if len(reshape1.predecessors()) != 1:
        continue
      perm1 = reshape1.predecessors()[0]
      if perm1.type != 'Permute':
        continue
      if len(perm1.predecessors()) != 1:
        continue
      perm1_par = perm1.predecessors()[0]      
      if len(reshape1.successors()) != 2:
        continue
      eltwise1, tmp = reshape1.successors()
      if eltwise1.name == lnrm1.name:
        eltwise1 = tmp
      if eltwise1.type != 'Eltwise':
        continue     
      if len(eltwise1.predecessors()) != 2:
        continue
      tmp, perm2 = eltwise1.predecessors()
      if perm2.name == reshape1.name:
        perm2 = tmp
      if perm2.type != 'Permute':
        continue      
      
      if len(eltwise1.successors()) != 2:
        continue
      eltwise2, lnrm2 = eltwise1.successors()
      if eltwise2.type != 'Eltwise':
        eltwise2, lnrm2 = lnrm2, eltwise2
      if eltwise2.type != 'Eltwise':
        continue   
      if lnrm2.type != 'LayerNorm':
        continue
      if len(lnrm2.successors()) != 1:
        continue
      lnrm2_rshp = lnrm2.successors()[0]
      if lnrm2_rshp.type != 'Reshape':
        continue  
      if len(lnrm2_rshp.successors()) != 1:
        continue
      lnrm2_perm = lnrm2_rshp.successors()[0]
      if lnrm2_perm.type != 'Permute':
        continue   
      if len(lnrm2_perm.successors()) != 1:
        continue
      lnrm2_conv = lnrm2_perm.successors()[0]
      if lnrm2_conv.type != 'Convolution':
        continue  
      tmp, perm3 = eltwise2.predecessors()
      if perm3.name == eltwise1.name:
        perm3 = tmp
      if perm3.type != 'Permute':
        continue  
      
      if len(eltwise2.successors()) != 2:
        continue
      eltwise3, lnrm3 = eltwise2.successors()
      if eltwise3.type != 'Eltwise':
        eltwise3, lnrm3 = lnrm3, eltwise3
      if eltwise3.type != 'Eltwise':
        continue   
      if lnrm3.type != 'LayerNorm':
        continue
      if len(lnrm3.successors()) != 1:
        continue
      lnrm3_rshp = lnrm3.successors()[0]
      if lnrm3_rshp.type != 'Reshape':
        continue  
      if len(lnrm3_rshp.successors()) != 1:
        continue
      lnrm3_perm = lnrm3_rshp.successors()[0]
      if lnrm3_perm.type != 'Permute':
        continue   
      if len(lnrm3_perm.successors()) != 1:
        continue
      lnrm3_conv = lnrm3_perm.successors()[0]
      if lnrm3_conv.type != 'Convolution':
        continue      
      tmp, perm4 = eltwise3.predecessors()
      if perm4.name == eltwise2.name:
        perm4 = tmp
      if perm4.type != 'Permute':
        continue 
      
      if len(eltwise3.successors()) != 1:
        continue
      reshape2 = eltwise3.successors()[0]
      if reshape2.type != 'Reshape':
        continue 
      if len(reshape2.successors()) != 1:
        continue
      perm5 = reshape2.successors()[0]
      if perm5.type != 'Permute':
        continue  

      # parameter check
      if perm1.permute_param.order != [0, 2, 3, 1]:
        continue
      jj = [i[1] for i in lnrm1_ch] + [lnrm2_perm, lnrm3_perm, perm5]
      for j in jj:
        if j.permute_param.order != [0, 3, 1, 2]:
          continue
      jj = [perm4, perm3, perm2]
      for j in jj:
        if j.permute_param.order != [0, 2, 1]:
          continue
      bottom_shape = [i for i in self.caffe_net.blobs[reshape1.bottom[0]].data.shape]
      top_shape = [i for i in self.caffe_net.blobs[reshape1.top[0]].data.shape]
      if top_shape != [bottom_shape[0], bottom_shape[1]*bottom_shape[2], bottom_shape[3]]:
        continue
      bottom_shape = [i for i in self.caffe_net.blobs[reshape2.bottom[0]].data.shape]
      top_shape = [i for i in self.caffe_net.blobs[reshape2.top[0]].data.shape]
      if bottom_shape != [top_shape[0], top_shape[1]*top_shape[2], top_shape[3]]:
        continue
      jj = [i[0] for i in lnrm1_ch]+[lnrm2_rshp, lnrm3_rshp]
      for j in jj:
        bottom_shape = [i for i in self.caffe_net.blobs[j.bottom[0]].data.shape]
        top_shape = [i for i in self.caffe_net.blobs[j.top[0]].data.shape]
        if top_shape != [bottom_shape[0], 1, bottom_shape[1], bottom_shape[2]]:
          continue

      # patter matched
      skip.add(lnrm2)
      skip.add(lnrm3)
      kk += 1
      self.G.add_edge(perm1_par, reshape1)
      self.G.add_replace_edge_proto_new(perm1_par, reshape1, reshape1.bottom[0], perm1.bottom[0])
      self.G.remove_node(perm1)
      dims = [i for i in reshape1.reshape_param.shape.dim]
      reshape1.reshape_param.shape.dim[1], reshape1.reshape_param.shape.dim[2] = dims[2], dims[1]
      self.G.add_edge(lnrm1_ch[0][0], lnrm1_ch[0][2])
      self.G.add_edge(lnrm1_ch[0][0], lnrm1_ch[1][2])
      self.G.add_edge(lnrm1_ch[0][0], lnrm1_ch[2][2])
      self.G.add_replace_edge_proto_new(lnrm1_ch[0][0], lnrm1_ch[0][2], lnrm1_ch[0][1].top[0], lnrm1_ch[0][0].top[0])
      self.G.add_replace_edge_proto_new(lnrm1_ch[0][0], lnrm1_ch[1][2], lnrm1_ch[1][1].top[0], lnrm1_ch[0][0].top[0])
      self.G.add_replace_edge_proto_new(lnrm1_ch[0][0], lnrm1_ch[2][2], lnrm1_ch[2][1].top[0], lnrm1_ch[0][0].top[0])
      dims = [i for i in lnrm1_ch[0][0].reshape_param.shape.dim]
      lnrm1_ch[0][0].reshape_param.shape.dim[1], lnrm1_ch[0][0].reshape_param.shape.dim[2], lnrm1_ch[0][0].reshape_param.shape.dim[3] = dims[3], dims[1], dims[2]
      self.G.remove_node(lnrm1_ch[0][1])
      self.G.remove_node(lnrm1_ch[1][1])
      self.G.remove_node(lnrm1_ch[2][1])
      self.G.remove_node(lnrm1_ch[1][0])
      self.G.remove_node(lnrm1_ch[2][0])
      lnrm1.layer_norm_param.axis = 1

      self.G.add_edge(perm2.predecessors()[0],eltwise1)
      self.G.add_replace_edge_proto_new(perm2.predecessors()[0], eltwise1, perm2.top[0], perm2.bottom[0])
      lnrm2.layer_norm_param.axis = 1
      dims = [i for i in lnrm2_rshp.reshape_param.shape.dim]
      lnrm2_rshp.reshape_param.shape.dim[1], lnrm2_rshp.reshape_param.shape.dim[2], lnrm2_rshp.reshape_param.shape.dim[3] = dims[3], dims[1], dims[2]
      self.G.add_edge(lnrm2_rshp, lnrm2_conv)
      self.G.add_replace_edge_proto_new(lnrm2_rshp, lnrm2_conv, lnrm2_perm.top[0], lnrm2_rshp.top[0])
      self.G.remove_node(perm2)
      self.G.remove_node(lnrm2_perm)

      self.G.add_edge(perm3.predecessors()[0],eltwise2)
      self.G.add_replace_edge_proto_new(perm3.predecessors()[0], eltwise2, perm3.top[0], perm3.bottom[0])
      lnrm3.layer_norm_param.axis = 1
      dims = [i for i in lnrm3_rshp.reshape_param.shape.dim]
      lnrm3_rshp.reshape_param.shape.dim[1], lnrm3_rshp.reshape_param.shape.dim[2], lnrm3_rshp.reshape_param.shape.dim[3] = dims[3], dims[1], dims[2]
      self.G.add_edge(lnrm3_rshp, lnrm3_conv)
      self.G.add_replace_edge_proto_new(lnrm3_rshp, lnrm3_conv, lnrm3_perm.top[0], lnrm3_rshp.top[0])
      self.G.remove_node(perm3)
      self.G.remove_node(lnrm3_perm)

      self.G.add_edge(perm4.predecessors()[0],eltwise3)
      self.G.add_replace_edge_proto_new(perm4.predecessors()[0], eltwise3, perm4.top[0], perm4.bottom[0])
      dims = [i for i in reshape2.reshape_param.shape.dim]
      reshape2.reshape_param.shape.dim[1], reshape2.reshape_param.shape.dim[2], reshape2.reshape_param.shape.dim[3] = dims[3], dims[1], dims[2]
      succs = perm5.successors()
      for succ in succs:
        self.G.add_edge(reshape2,succ)
        self.G.add_replace_edge_proto_new(reshape2, succ, perm5.top[0], reshape2.top[0])
      self.G.remove_node(perm4)
      self.G.remove_node(perm5)
    
      

      



      
    