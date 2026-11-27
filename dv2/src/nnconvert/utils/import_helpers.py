'''
Copyright (c) 2022-25, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

from importlib import import_module
from utils.dict2object import ObjectLike
import json
import os
import numpy as np


def get_cls_and_pkg(absolute_class_name):
  input_parts = absolute_class_name.split('.')
  cls = input_parts.pop()
  pkg = '.'.join(input_parts)
  pkg = import_module(pkg)
  cls = getattr(pkg, cls)
  return ObjectLike(cls=cls, pkg=pkg)


def get_variable_name(layer_name):
  return layer_name.replace('/', '_').replace('-', '_').replace('[', '_').replace(']', '_')


def dump_json(data, out_file):
  with open(out_file, 'w') as f:
    json.dump(data, f, indent=4)


def dump_np_data(data, out_file_dir, filename):
  if not os.path.exists(out_file_dir):
    os.makedirs(out_file_dir)
  filename = filename + '.bin'
  data.astype('float32').tofile(os.path.join(out_file_dir, filename))
