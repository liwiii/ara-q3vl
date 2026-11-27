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
import numpy as np
import caffe
from caffe import layers as L
from caffe import params as P
from mmdnn.conversion.caffe.utils import get_stride_dims, get_pad_dims, get_kernel_dims, set_kernel_dims, set_pad_dims
from mmdnn.conversion.caffe.c2c.common import BasePass
from dv_logging import get_logger
logger = get_logger(__name__, 'dvctoc')


class InputPatching(BasePass):

  def __init__(self):
    pass

  def patch_input(self, layer):
    if not self.pp_yaml:
      return
    cfg = self.pp_yaml
    if 'preprocessing_params' in cfg and cfg['preprocessing_params']['bgr2rgb']:
      layer.transform_param.bgr2rgb = True
    if 'preprocess' in cfg and cfg['preprocess']:
      interpolate_map = {"NN": 0, "LINEAR": 1, "CUBIC": 2, "AREA": 3, "LANCZOS4": 4}
      layer.transform_param.scale.extend([cfg['preprocessing_params']['scale']['b'], cfg['preprocessing_params']['scale']['g'], cfg['preprocessing_params']['scale']['r']])
      layer.transform_param.interpolate = interpolate_map[cfg['preprocessing_params']['interpolation']]
      layer.transform_param.mean_value.extend([cfg['preprocessing_params']['mean']['b'], cfg['preprocessing_params']['mean']['g'], cfg['preprocessing_params']['mean']['r']])
      layer.transform_param.center_crop = cfg['preprocessing_params']['center_crop']

  def do_transformation(self):
    if self.pp_yaml:
      input_nodes = self.get_nodes_of_type('Input')
      for input_node in input_nodes:
        self.patch_input(input_node)


class InputOutputOrderPatching(BasePass):
  
  def __init__(self):
    pass

  def do_transformation(self):
    input_nodes = self.get_nodes_of_type('Input')
    output_nodes = self.get_output_nodes() 
    inode_order = self.inode_names
    onode_order = self.onode_names


    if self.onode_names and all(x == "None" for x in self.onode_names):
      return
      # onode_order = [node.name for node in output_nodes]

    prop_back_list = ['Reshape','ReLU','Swish','Softmax']
    for node in input_nodes:
      index = inode_order.index(node.name)
      node.get_layer().input_index = index
    
    output_nodes_alloc = []
    for node in output_nodes:
      while(node.type in prop_back_list):
        node = node.predecessors()[0]
      if node not in output_nodes_alloc:
        output_nodes_alloc.append(node)
    
    for node in output_nodes_alloc:
      if node.name in onode_order:
        index = onode_order.index(node.name)
      else:
        index = -1
      node.get_layer().output_index = index