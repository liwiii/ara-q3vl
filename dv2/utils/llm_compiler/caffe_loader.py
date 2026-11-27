'''
Copyright (c) 2025, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

import os
import sys

class CaffeLoader:
    def __init__(self, build_path):
        self.build_path = build_path
        self._import_caffe()

    def _import_caffe(self):
        caffe_python_path = os.path.join(self.build_path, 'dv2/python/')
        if caffe_python_path not in sys.path:
            sys.path.append(caffe_python_path)
        caffe_proto_path = os.path.join(self.build_path, 'dv2/python/caffe/proto/')
        if caffe_proto_path not in sys.path:
            sys.path.append(caffe_proto_path)
        try:
            import caffe
            from caffe.proto import caffe_pb2
            self.caffe = caffe
            self.caffe_pb2 = caffe_pb2
            from proto_utils.layers import Conv, Rope, Input, BatchDot, AttentionSoftmax, SoftMax, Inrp, LayerNorm, Eltwise, Swish, Requant, Comm
            from proto_utils.functional import (
                permute,
                reshape,
                simple_reshape,
                ReshapeMaker,
                reshape_with_reshapemaker,
            )
            from proto_utils.graph import DAG, NamedDim
            from proto_utils.layers.eltwise import Operation
            from proto_utils import Network
            # Assigning imported classes to instance variables
            self.Conv = Conv
            self.Rope = Rope
            self.Input = Input
            self.BatchDot = BatchDot
            self.SoftMax = SoftMax
            self.AttentionSoftmax = AttentionSoftmax
            self.Inrp = Inrp
            self.LayerNorm = LayerNorm
            self.Eltwise = Eltwise
            self.Swish = Swish
            self.Comm = Comm
            self.Requant = Requant
            self.permute = permute
            self.reshape = reshape
            self.simple_reshape = simple_reshape
            self.ReshapeMaker = ReshapeMaker
            self.reshape_with_reshapemaker = reshape_with_reshapemaker
            self.NamedDim = NamedDim
            self.DAG = DAG
            self.Network = Network
            self.EltwiseOperation = Operation
        except ImportError as e:
            raise ImportError(f"Could not import Caffe {caffe_python_path}: {e}")

    def get_layers(self):
      if self.caffe is None:
          raise RuntimeError("Caffe is not loaded. Call _import_caffe() first.")

    def get_caffe(self):
        return self.caffe
    def get_caffe_pb2(self):
        return self.caffe_pb2
    def get_proto_utils(self):
        return {
            'Conv': self.Conv,
            'Rope': self.Rope,
            'Input': self.Input,
            'BatchDot': self.BatchDot,
            'AttentionSoftmax': self.AttentionSoftmax,
            'SoftMax': self.SoftMax,
            'Inrp': self.Inrp,
            'LayerNorm': self.LayerNorm,
            'Eltwise': self.Eltwise,
            'Swish': self.Swish,
            'Requant': self.Requant,
            'Comm': self.Comm,
            'permute': self.permute,
            'reshape': self.reshape,
            'simple_reshape': self.simple_reshape,
            'NamedDim': self.NamedDim,
            'DAG': self.DAG,
            'Network': self.Network,
            'EltwiseOperation': self.EltwiseOperation,
            'ReshapeMaker': self.ReshapeMaker,
            'reshape_with_reshapemaker': self.reshape_with_reshapemaker
        }
