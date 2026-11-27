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
import numpy as np

estimator_group = {
    #"Ara1Swish":      "activation",
    "Ara2Swish":      "activation",
    "Ara1Hardsigmoid":  "activation",
    "Ara2Gelu":      "activation",
    "Ara2Sigmoid":       "activation",
    "Ara2Tanh":       "activation",
    "Ara2Hardsigmoid":       "activation",
}

class ShapePatching(BasePass):
    def __init__(self):
        pass

    def do_transformation(self):
        return
        all_nodes = self.G.get_topsort()
        for node in all_nodes:
            if node.base_type == "Input":
                node.get_layer().input_param.shape[0].dim[0] = 8
            elif node.base_type == "DummyData":
                node.get_layer().dummy_data_param.shape[0].dim[0] = 8
            elif node.base_type == "Reshape":
                node.get_layer().reshape_param.shape.dim[0] = 8
            elif node.base_type == "DVTensor":
                if node.get_layer().dvtensor_param.tensortype == 2:
                    tensor = self.inp_wts[node.name]
                    b_tensor = np.tile(tensor, (8, 1, 1))
                    self.inp_wts[node.name] = b_tensor
                node.get_layer().dvtensor_param.shape[0].dim[0] = 8

