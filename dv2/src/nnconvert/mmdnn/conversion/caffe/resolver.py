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

import sys
SHARED_CAFFE_RESOLVER = None


class CaffeResolver(object):

  def __init__(self):
    self.import_caffe()

  def import_caffe(self):
    self.caffe = None
    # try:
    import caffe
    self.caffe = caffe
    # except ImportError:
    #     # # Fall back to the protobuf implementation
    #     # from mmdnn.conversion.caffe import caffe_pb2
    #     # self.caffepb = caffe_pb2
    #     # show_fallback_warning()
    if self.caffe:
      self.caffepb = self.caffe.proto.caffe_pb2
    self.NetParameter = self.caffepb.NetParameter

  def has_pycaffe(self):
    return self.caffe is not None


def get_caffe_resolver():
  global SHARED_CAFFE_RESOLVER
  if SHARED_CAFFE_RESOLVER is None:
    SHARED_CAFFE_RESOLVER = CaffeResolver()
  return SHARED_CAFFE_RESOLVER


def has_pycaffe():
  return get_caffe_resolver().has_pycaffe()


def show_fallback_warning():
  msg = '''
------------------------------------------------------------
    WARNING: PyCaffe not found!
    Falling back to a pure protocol buffer implementation.
    * Conversions will be drastically slower.
    * This backend is UNTESTED!
------------------------------------------------------------
'''
  sys.stderr.write(msg)
