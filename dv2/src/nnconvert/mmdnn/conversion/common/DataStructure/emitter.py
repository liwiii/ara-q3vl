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

from six import string_types as _string_types
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from dv_logging import get_logger
logger = get_logger(__name__, 'DVCOEM')


class Emitter(object):

  def __init__(self):
    self.body_code = str()
    self.weights_dict = dict()
    self.used_layers = set()
    self.weight_loaded = False
    self.layers_codes = dict()

  def run(self, dstNetworkPath, dstWeightPath=None, phase='test'):
    self.save_code(dstNetworkPath, phase)

  # share functions
  def add_body(self, indent, codes):
    if isinstance(codes, _string_types):
      codes = [codes]
    for code in codes:
      temp = self.body_code
      temp += ("    " * indent) + code + '\n'
      try:
        exec(compile(temp, 'file', 'exec'))
        self.body_code = temp
      except Exception as e:
        raise NotImplementedError("error in generated code " + str(e))

  def _load_weights(self, file_name=None):
    import numpy as np
    self.weight_loaded = True
    # logger.info(file_name)
    try:
      self.weights_dict = np.load(file_name, allow_pickle=True).item()
    except:
      self.weights_dict = np.load(file_name, encoding='bytes', allow_pickle=True).item()

  def parent_variable_name(self, IR_node, path_or_name=[0]):
    if isinstance(path_or_name, _string_types):
      path = [IR_node.in_edges.index(path_or_name)]
    elif isinstance(path_or_name, list):
      path = path_or_name
    else:
      raise ValueError
    return self.IR_graph.get_parent_variable_name(IR_node.name, path)

  def _build(self):
    self.IR_graph.build()

  def gen_code(self, phase):
    raise NotImplementedError("do not use base emitter class.")

  def save_code(self, filepath, phase):
    code = self.gen_code(phase)
    with open(filepath, 'w') as fout:
      fout.write(code)
    logger.info("target network code snippet is saved as [{}]".format(filepath))

  @staticmethod
  def save_weights(weights, filename):
    import numpy as np
    with open(filename, 'wb') as of:
      np.save(of, weights)
    logger.info("target weights are saved as [{}]".format(filename))

  @staticmethod
  def _image_in_transpose_str(dim):
    dims = [dim]
    dims.extend(range(dim))
    return ','.join('%s' % id for id in dims)

  @staticmethod
  def _image_out_transpose_str(dim):
    dims = list(range(1, dim + 1))
    dims.append(0)
    return ','.join('%s' % id for id in dims)

  @staticmethod
  def _conv_kernel_transpose_str(dim):
    dims = [dim + 1, dim]
    dims.extend(range(dim))
    return ','.join('%s' % id for id in dims)
