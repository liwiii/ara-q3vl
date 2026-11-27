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

from proto_utils.layers.layer import Layer
from .graph import DAG, NodeWithShape
from typing import List
from abc import abstractmethod
class Network:
    def __init__(self, network_name: str, graph: DAG):
        self.graph = graph
        self.network_name = network_name
    
    @abstractmethod
    def forward(self, bottom_nodes: List[NodeWithShape], **kwargs) -> List[NodeWithShape]:
        pass

    def __call__(self, bottom_nodes: List[NodeWithShape], **kwargs):
        return self.forward(bottom_nodes, **kwargs)
    
    def update_name_if_layer(self, layer):
        if isinstance(layer, Layer):
            layer.layer_name = self.network_name + "_" + layer.layer_name
        if isinstance(layer, List):
            for item in layer:
                self.update_name_if_layer(item)
        if isinstance(layer, dict):
            for item in layer.values():
                self.update_name_if_layer(item)
        if isinstance(layer, Network):
            layer.network_name = self.network_name + "_" + layer.network_name
            layer.update_layer_names()
    
    def update_layer_names(self):
        for key, value in self.__dict__.items():
            self.update_name_if_layer(value)


