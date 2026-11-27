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

estimator_group = {
    #"Ara1Swish":      "activation",
    "Ara2Swish":      "activation",
    "Ara1Hardsigmoid":  "activation",
    "Ara2Gelu":      "activation",
    "Ara2Sigmoid":       "activation",
    "Ara2TanH":       "activation",
    "Ara2Hardsigmoid":       "activation",
    "Tflite_Sigmoid":        "activation",
    "Pqn_Sigmoid":        "activation",
    "Ara2Logits":         "activation",
    "Ara2LeakyRelu":     "activation",
    "Ara2ELU":     "activation",
    "Ara2HardSwish":"activation"
}

class GroupPatching(BasePass):
    def __init__(self):
        pass

    def do_transformation(self):
        all_nodes = self.G.get_topsort()
        for node in all_nodes:
            if node.type not in estimator_group or (hasattr(node,"adj_param") and len(node.adj_param.precision) > 0 and node.adj_param.precision[0] == 1 and node.type == 'Ara2Gelu'):
                node.get_layer().estimator_group = node.base_type
            else:
                node.get_layer().estimator_group =  estimator_group[node.type]
