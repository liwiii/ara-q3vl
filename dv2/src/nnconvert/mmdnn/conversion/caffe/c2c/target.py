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

class TargetType(BasePass):
    def __init__(self):
        pass

    def do_transformation(self):
        all_nodes = self.G.get_topsort()
        for node in all_nodes:
            if node.type.startswith("Tflite_"):
                node.get_layer().base_type = node.get_layer().type[len('Tflite_'):]
            elif node.type.startswith("Pqn_"):
                node.get_layer().base_type = node.get_layer().type[len('Pqn_'):]
            else:
                node.get_layer().base_type = node.get_layer().type
            if node.type not in ['Input',"LSTM","Aten"] and not (node.type.startswith("Onnx_") or node.type.startswith("Fused_") or node.type.startswith("Mxnet_") or node.type.startswith("Tf_") or node.type.startswith("divideVectorByScalar") or node.type.startswith("vector_") or node.type.startswith("Tflite_") or node.type.startswith("Pqn_")):
                node.get_layer().type = self.platform_target + node.get_layer().type
