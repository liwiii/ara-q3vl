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

import caffe
from caffe import layers as L
from caffe import params as P
from mmdnn.conversion.caffe.c2c.common import BasePass
import os.path as p
import copy
import csv
from dv_logging import get_logger
logger = get_logger(__name__, 'dvctoc')


class NamePatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    '''
      Reduce the length of the names
    '''
    nodes = self.get_topsort()
    name_lists = []
    name_dict = dict()
    for node in nodes:
      name_lists.append(node.name)
      for top in node.top:
        name_lists.append(top)
    # common_prefix = p.commonprefix(name_lists)
    name_set = set(name_lists)
    for idx,name in enumerate(sorted(name_set)):
      if len(name) > 100:
        name_dict[name] = 'node_' + str(idx)
    headers = ['OriginalName', 'ConvertedName']
    # Copy the dict to a csv
    with open('./layer_map_c2c.csv', 'w') as csvfile:
      for header in headers:
        csvfile.write(header + ',')
      csvfile.write('\n')
      for k, v in name_dict.items():
        csvfile.write(k + ',' + v + '\n')
    for node in nodes:
      node.name = name_dict.get(node.name, node.name)
      node.get_layer().name = name_dict.get(node.get_layer().name, node.get_layer().name)
      for idx, bottom in enumerate(node.bottom):
        node.bottom[idx] = name_dict.get(bottom, bottom)
      for idx, top in enumerate(node.top):
        node.top[idx] = name_dict.get(top, top)
    orig_blob_keys = [key for key in self.caffe_net.blobs.keys()]
    orig_param_keys = [key for key in self.caffe_net.params.keys()]
    orig_wt_keys = [key for key in self.inp_wts.keys()]
    for key in orig_blob_keys:
      self.caffe_net.blobs[name_dict.get(key, key)] = self.caffe_net.blobs.pop(key)
    for key in orig_param_keys:
      self.caffe_net.params[name_dict.get(key, key)] = self.caffe_net.params.pop(key)
    for key in orig_wt_keys:
      self.inp_wts[name_dict.get(key, key)] = self.inp_wts.pop(key)
