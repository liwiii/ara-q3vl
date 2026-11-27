'''
Portions of this file are derived from https://github.com/microsoft/MMdnn and are subject to the original MIT License

Copyright (c) Microsoft Corporation. All rights reserved.
SPDX-License-Identifier: MIT

Copyright (c) 2018 - 2025, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Modifications:

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

import numpy as np
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from dv_logging import get_logger
logger = get_logger(__name__, 'DVCOPR')


class Parser(object):

  def __init__(self):
    self.IR_graph = GraphDef()
    self.weight_loaded = False
    # name --> (weight_name --> ndarray)
    self.weights = dict()

  def run(self, dest_path):
    self.gen_IR()
    self.save_to_json(dest_path + ".json")
    self.save_to_proto(dest_path + ".pb")
    self.save_weights(dest_path + ".npy")

  @property
  def src_graph(self):
    raise NotImplementedError

  def get_son(self, name, path, set_flag=False):
    return self.src_graph.get_son(name, path, set_flag)

  def get_parent(self, name, path, set_flag=False):
    return self.src_graph.get_parent(name, path, set_flag)

  def set_weight(self, layer_name, weight_name, data):
    if not layer_name in self.weights:
      self.weights[layer_name] = dict()
    layer = self.weights[layer_name]
    layer[weight_name] = data

  def save_to_json(self, filename):
    import google.protobuf.json_format as json_format
    json_str = json_format.MessageToJson(self.IR_graph, preserving_proto_field_name=True)
    with open(filename, "w") as of:
      of.write(json_str)
    logger.info("IR network structure is saved as [{}]".format(filename))
    return json_str

  def save_to_proto(self, filename):
    #import pdb; pdb.set_trace()
    proto_str = self.IR_graph.SerializeToString()
    with open(filename, 'wb') as of:
      of.write(proto_str)
    logger.info("IR network structure is saved as [{}]".format(filename))
    return proto_str

  def save_weights(self, filename):
    if self.weight_loaded:
      with open(filename, 'wb') as of:
        np.save(of, self.weights)
      logger.info("IR weights are saved as [{}]".format(filename))
    else:
      logger.warn("weights are not loaded")

  def convert_inedge(self, source_node, IR_node, start_idx=0, end_idx=None):
    if end_idx == None: end_idx = len(source_node.in_edges)
    for idx in range(start_idx, end_idx):
      IR_node.input.append(self.src_graph.get_node(source_node.in_edges[idx]).real_name)

  def convert_inedge_tflite(self, source_node, IR_node, start_idx = 0, end_idx = None):
    if end_idx == None: end_idx = len(source_node.in_edges)
    for idx in range(start_idx, end_idx):
      if self.src_graph.get_node(source_node.in_edges[idx]).type in ['SPLIT']:
        IR_node.input.append(self.src_graph.get_node(source_node.in_edges[idx]).real_name.lstrip('_') + '_output' + str(source_node.in_edges[idx]).split(':')[1])
      elif self.src_graph.get_node(source_node.in_edges[idx]).type in ['SPLIT_V']:
        in_name = self.src_graph.get_node(source_node.in_edges[idx]).real_name.lstrip('_') + '_out'
        id = source_node.layer['actual_inputs'][idx]
        split_outs = [s for s in self.src_graph.model['ops'] if id in s['outputs']][0]['outputs']
        inp_idx = split_outs.index(id)
        in_name = in_name + str(inp_idx)
        IR_node.input.append(in_name)
      else:
        IR_node.input.append(self.src_graph.get_node(source_node.in_edges[idx]).real_name.lstrip('_'))

  @staticmethod
  def channel_first_conv_kernel_to_IR(tensor):
    dim = tensor.ndim
    tensor = np.transpose(tensor, list(range(2, dim)) + [1, 0])
    return tensor

  @staticmethod
  def channel_first_shape_to_IR(shape):
    return [shape[0]] + list(shape[2:]) + [shape[1]]

  @staticmethod
  def channel_first_axis_to_IR(index):
    if index == 0:
      return 0
    elif index == 1:
      return -1
    else:
      return index - 1
