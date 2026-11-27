'''
Copyright (c) 2018-22, Deep Vision, Inc. All rights reserved.
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


import os
import copy
from mmdnn.conversion.common.DataStructure.graph import GraphNode, Graph
from mmdnn.conversion.tflite.tflite_graph_utils import TFlitePatternMatcher

class OpCodeMapper(object):
  """Maps an opcode index to an op name."""

  def __init__(self, data):
    self.code_to_name = {}
    for idx, d in enumerate(data["operator_codes"]):
      self.code_to_name[idx] = d["builtin_code"]

  def __call__(self, x):
    if x not in self.code_to_name:
      s = "NoOp"
    else:
      s = self.code_to_name[x]
    return s

class TFLiteGraphNode(GraphNode):


    def __init__(self, layer):
        super(TFLiteGraphNode, self).__init__(layer)
        self.attr = layer


    @property
    def name(self):
        return self.layer["name"]


    @property
    def type(self):
        return self.layer['type']



    @property
    def tflite_layer(self):
        return self.layer


    def get_attr(self, name, default_value=None):
        if name in self.attr:
            return self.attr[name]
        return None

class TFLiteGraph(Graph):

    def __init__(self, model):
        super(TFLiteGraph, self).__init__(model)


    def build(self,model_json):

        self.input_layers = list()
        input_dict = dict() 
        patternMatcher = TFlitePatternMatcher(model_json)

        for input in model_json['inputs']:
            input.update({"name":"loc_" + str(input['index'])})
            input.update({"type":"INPUT"})
            op_scales = copy.deepcopy(input['quantization_parameters']['scales'])
            op_zero_points = copy.deepcopy(input['quantization_parameters']['zero_points'])
            input['output_scales'] = op_scales
            input['output_zero_points'] = op_zero_points
            input_dict.update({input['name']: input})
            self.layer_map[input["name"]] = TFLiteGraphNode(input)
            self.layer_name_map[input["name"]] = input["name"]  

        patternMatcher.add_inputs(input_dict)

        # Dict needed for split since all the outputs have different names
        output_map = {}
        
        for layer in model_json['ops']:

            layer.update({"name":"loc_" + str(layer['outputs'][0])})
            layer.update({"type":layer["op_name"]})
            # Handle split layer
            layer['actual_inputs'] = copy.deepcopy(layer['inputs'])
            if layer['type'] == "SPLIT_V":
                for op in layer['outputs']:
                    output_map[op] = layer['outputs'][0]
            # Add quantization params to layer
            inp_scales = []
            inp_zero_points = []
            op_scales = []
            op_zero_points = []
            for inp_ind in layer['inputs']:
                if 'scales' in model_json['tensor_details'][inp_ind]['quantization_parameters']:
                    if len(model_json['tensor_details'][inp_ind]['quantization_parameters']['scales']) > 0:
                        inp_scales.append(copy.deepcopy(model_json['tensor_details'][inp_ind]['quantization_parameters']['scales']))
                        inp_zero_points.append(copy.deepcopy(model_json['tensor_details'][inp_ind]['quantization_parameters']['zero_points']))
            
            for op_ind in layer['outputs']:
                op_scales.append(copy.deepcopy(model_json['tensor_details'][op_ind]['quantization_parameters']['scales']))
                op_zero_points.append(copy.deepcopy(model_json['tensor_details'][op_ind]['quantization_parameters']['zero_points']))

            layer['input_scales'] = inp_scales
            layer['input_zero_points'] = inp_zero_points
            layer['output_scales'] = op_scales
            layer['output_zero_points'] = op_zero_points
            input_dict.update({layer['name']: layer})
            patternMatcher.add_node(layer=layer, output_map=output_map)

        patternMatcher.matchKnownPatterns()
        network = patternMatcher.getTopologicalSort()

        self.layer_map.clear()
        self.layer_name_map.clear()
        
        for node in network:
            data_dict = patternMatcher.getNodeAttribute(node)
            if(len(data_dict) == 0):
                continue
            nodeAttributes = patternMatcher.getNodeAttribute(node)["data"]
            self.layer_map[nodeAttributes['name']] = TFLiteGraphNode(nodeAttributes)
            self.layer_name_map[nodeAttributes['name']] = nodeAttributes['name']
            if not nodeAttributes['type'] == "INPUT":
                for input in nodeAttributes["inputs"]:
                    self._make_connection("loc_"+str(input), nodeAttributes["name"])

        super(TFLiteGraph, self).build()