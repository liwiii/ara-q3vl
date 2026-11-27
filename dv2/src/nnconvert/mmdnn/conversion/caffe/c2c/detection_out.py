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
logger = get_logger(__name__, 'dvctoc')


class DetectionOutPatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    nodes = self.get_nodes_of_type('DetectionOutput')
    # What happens if few of the layers are executed on CP instead of DLP
    sfmx_type_list = ['Softmax', 'Tf_Softmax', 'Mxnet_Softmax']
    sigm_type_list = ['Sigmoid', 'Tf_Sigmoid', 'Mxnet_Sigmoid']
    # priorbox_type_list = ['PriorBox']
    conf_type_list = sfmx_type_list + sigm_type_list
    conv_type_list = ['Convolution', 'InnerProduct', 'Tf_Conv2D', 'Tf_Matmul', 'Mxnet_Convolution', 'Mxnet_Fullyconnected']
    softmax_layer = None
    conf_layer_type = None
    conf_features = list()
    for layer in nodes:
      if layer.detection_output_param.is_ssd_cmp_added:
        continue
      dout_predecessors = list(layer.predecessors())
      for dout_predecessor in dout_predecessors:
        node = dout_predecessor
        while node and node.type not in conf_type_list:
          # found = False
          predecessors = list(node.predecessors())
          if len(predecessors) == 0:
            node = None
          else:
            node = predecessors[0]
        if node:
          softmax_layer = node
          break
      if softmax_layer:
        if softmax_layer.type in sfmx_type_list:
          conf_layer_type = 1
        elif softmax_layer.type in sigm_type_list:
          conf_layer_type = 0
        else:
          logger.error("Softmax/Sigmoid not found : %s", softmax_layer.name)
          return -1
        softmax_predecessors = list(softmax_layer.predecessors())
        if len(softmax_predecessors) == 1:
          concat_layer = softmax_predecessors[0]
          while concat_layer:
            concat_layer = softmax_predecessors[0]
            if concat_layer.type == 'Concat':
              break
            softmax_predecessors = list(concat_layer.predecessors())
        else:
          logger.error("Concat is not found at bottom of softmax")
          return -1
        if concat_layer.type != 'Concat':
          logger.error("Concat not found at bottom of softmax found layer: %s", concat_layer.name)
          return -1
          # raise
        concat_predecessors = list(concat_layer.predecessors())
        for concat_predecessor in concat_predecessors:
          node = concat_predecessor
          while node and node.type not in conv_type_list:
            found = False
            predecessors = list(node.predecessors())
            if len(predecessors) == 0:
              node = None
            else:
              node = predecessors[0]
          if node:
            conf_features.append(node)
          else:
            logger.error("No conf feature found in path %s", concat_predecessor.name)
      else:
        logger.error("Unable to find the confidence path for the detection_out layer")
      for conf in conf_features:
        cmp_layer = L.DVSSDCompare(dvssd_compare_param=dict(num_classes=layer.detection_output_param.num_classes,
                                                            threshold=math.log(layer.detection_output_param.confidence_threshold /
                                                                               (1 - layer.detection_output_param.confidence_threshold)),
                                                            bg_id=layer.detection_output_param.background_label_id,
                                                            conf_type=conf_layer_type)).to_proto().layer[0]
        cmp_layer.name = conf.name + '_ssd_cmp'
        cmp_layer.top[0] = conf.name + '_ssd_cmp'
        cmp_layer.bottom.append(conf.top[0])
        self.G.add_node(cmp_layer)
        self.G.add_edge(conf, cmp_layer)
