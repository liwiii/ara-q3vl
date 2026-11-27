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
from mmdnn.conversion.caffe.c2c.common import BasePass
from dv_logging import get_logger
logger = get_logger(__name__, 'dvctoc')


class InputPermuteFusion(BasePass):
  '''
  If input layer is followed by Permute layer, then transpose the input layer and remove the Permute layer.
  '''

  def __init__(self):
    pass


  def do_transformation(self):
    if not self.tf2onnx_happened:
      return
    input_nodes = self.get_nodes_of_type('Input')
    for input_node in input_nodes:
      if len(input_node.successors()) == 1 and input_node.successors()[0].type == 'Permute':
        permute = input_node.successors()[0]
        order = permute.permute_param.order
        shape = input_node.get_layer().input_param.shape[0].dim
        shape_new = [0]*len(shape)
        for i,j in enumerate(order):
          shape_new[i] = shape[j]
        while len(shape): shape.pop()
        for i,j in enumerate(shape_new):
          shape.insert(i,j)
        successors = list(permute.successors())
        for succ in successors:
          self.G.add_edge(input_node, succ)
          self.G.add_replace_edge_proto_new(input_node, succ, permute.top[0], permute.bottom[0])
        self.G.remove_node(permute)
        self.input_permuted[input_node.get_layer().name] =  tuple(order)
        logger.warning("Tf2onnx added permute layer after Input layer. Input layer named: "+input_node.get_layer().name+" is transposed and succeeding permute layer is removed")



class OutputPermuteRemoval(BasePass):
  '''
  If output layer is a Permute layer, remove it and note down the layer top name.
  '''

  def __init__(self):
    pass


  def do_transformation(self):
    if not self.tf2onnx_happened:
      return
    permute_layers = self.get_nodes_of_type('Permute')
    for layer in permute_layers:
      if len(layer.successors()):
        continue
      if len(layer.predecessors()) != 1:
        continue
      order = layer.permute_param.order
      self.output_permuted[layer.get_layer().bottom[0]] =  tuple(order)
      predecessor = layer.predecessors()[0]
      name = layer.name
      self.G.remove_node(layer)
      self.inp_wts[name] = self.inp_wts.pop(predecessor.name)
      predecessor.get_layer().__setattr__('name',name)


class HandleLastPermuteReshape(BasePass):
  '''
  In most tf2onnx cases, onnx_parser will add reshape layer succeeding permute layer. Remove the transform and handle this case.
  '''

  def __init__(self):
    pass


  def do_transformation(self):
    if not self.tf2onnx_happened:
      return
    permute_layers = self.get_nodes_of_type('Permute')
    for permute in permute_layers:
      successors = permute_layers[0].successors()
      if len(successors) != 1 or successors[0].get_layer().type != 'Reshape' or len(successors[0].successors()) != 0:
        continue
      reshape = successors[0]
      perm_shape = self.caffe_net.blobs[permute.top[0]].data.shape
      reshape_shape = self.caffe_net.blobs[reshape.top[0]].data.shape
      if perm_shape != reshape_shape:
        continue
      order = permute.permute_param.order
      reshape_param_shape = reshape.get_layer().reshape_param.shape.dim
      new_shape = [0]*len(reshape_param_shape)
      for i,j in enumerate(order):
        new_shape[j] = reshape_param_shape[i]
      while len(reshape_param_shape): reshape_param_shape.pop()
      for i,j in enumerate(new_shape):
        reshape_param_shape.insert(i,j)
      self.output_permuted[reshape.top[0]] =  tuple(order)
      predecessors = list(permute.predecessors())
      for par in predecessors:
        self.G.add_edge(par,reshape)
        self.G.add_replace_edge_proto_new(par, reshape, reshape.bottom[0],permute.bottom[0])
      self.G.remove_node(permute)

