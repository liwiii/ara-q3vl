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

import networkx as nx
from networkx.algorithms import isomorphism
from networkx.drawing.nx_pydot import write_dot

from dv_logging import get_logger

logger = get_logger(__name__, "DVTFLPM")

class TFlitePatternMatcher:

    def __init__(self, model) -> None:
        self.graph = nx.DiGraph()
        self.model = model

    def add_inputs(self, input_dict):

        all_inputs = input_dict.keys()
        for input in all_inputs:
            layer = input_dict[input]
            layer_name = layer['name']
            op_type = layer["type"]

            if op_type == "INPUT":
                self.graph.add_node(layer_name, data=layer, op_type=op_type)
    
    def add_node(self, layer, output_map=dict()):
        layer_name = layer['name']
        op_type = layer["type"]
        inputs = layer["inputs"]

        # Update the layer
        for i in range(len(layer["inputs"])):
            if layer["inputs"][i] in output_map:
                layer["actual_inputs"][i] = layer["inputs"][i]
                layer["inputs"][i] = output_map[layer["inputs"][i]]

        self.graph.add_node(layer_name, data=layer, op_type=op_type)

        for input in inputs:
            if input in output_map:
                self.graph.add_edge('loc_'+str(output_map[input]), layer_name)
            else:
                self.graph.add_edge('loc_'+str(input), layer_name)

    def getNodeAttribute(self, node : str):
        return self.graph.nodes[node]

    def match_node_by_type(self, n1, n2):
        op1 = n1.get("op_type")
        op2 = n2.get("op_type")
        if op1 == "any" or op2 == "any":
            return True
        return op1 == op2

    def matchKnownPatterns(self):
        '''
            Just add your pattern matcher function here.
            matchKnownPatterns will automatically pick it up and 
            use it as a functor
        '''

        def swishPatternMatcher():

            def swish_match_node_by_op(n1, n2):
                op1 = n1.get("op_type")
                op2 = n2.get("op_type")
                if op1 == "any" or op2 == "any":
                    return True
                return op1 == op2

            swishPattern = nx.DiGraph()
            swishPattern.add_node("mul", op_type="MUL")
            swishPattern.add_node("logistic", op_type="LOGISTIC")
            swishPattern.add_node("input", op_type="any")

            swishPattern.add_edge("input", "logistic")
            swishPattern.add_edge("input", "mul")
            swishPattern.add_edge("logistic", "mul")

            matches = isomorphism.DiGraphMatcher(self.graph, swishPattern, node_match=swish_match_node_by_op)
            for match in list(matches.subgraph_isomorphisms_iter()):
                current_topological_sort = list(nx.topological_sort(self.graph))
                match_reversed = {y: x for x, y in match.items()}
                match = match_reversed
                logistic_loc_position = match["logistic"]
                mul_loc_position = match["mul"]

                logistic_node_data = self.getNodeAttribute(logistic_loc_position)["data"]
                mul_node_data = self.getNodeAttribute(mul_loc_position)["data"]
                
                swish_input_node = "loc_" + str(logistic_node_data["inputs"][0])
                swish_output_nodes = []
                nodes_after_mul = current_topological_sort[current_topological_sort.index(mul_loc_position)+1:]

                for node in nodes_after_mul:
                    node_data = self.getNodeAttribute(node)["data"]
                    node_pos = int(str(mul_loc_position).split("_")[1])
                    if node_pos in node_data["inputs"]:
                        swish_output_nodes.append(node)

                # outputs_list = [int(str(x).split("_")[1]) for x in swish_output_nodes]
                # if len(nodes_after_mul) == 0:
                #     outputs_list = [int(str(mul_loc_position).split("_")[1])]
                outputs_list = self.getNodeAttribute(mul_loc_position)["data"]["outputs"]
                
                param_dict = {
                    'index' : int(mul_node_data["index"]) + 1, 
                    'op_name' : "SWISH",
                    'inputs'  : [int(swish_input_node.split("_")[1])] ,
                    'outputs' : outputs_list,
                    'attributes' : {},
                    'name': mul_loc_position,
                    'type': "SWISH",
                    'input_scales': [logistic_node_data["input_scales"][0],
                                    logistic_node_data["output_scales"][0]],
                    'input_zero_points' : [logistic_node_data["input_zero_points"][0],
                                           logistic_node_data["output_zero_points"][0]],
                    'output_scales' : [mul_node_data["output_scales"][0]],
                    "output_zero_points" : [mul_node_data["output_zero_points"][0]]
                }

                self.graph.remove_node(logistic_node_data['name'])
                self.graph.remove_node(mul_node_data['name'])
                self.add_node(param_dict)
                for output_node in swish_output_nodes:
                    self.graph.add_edge(param_dict['name'], output_node)
                
                logger.info("replaced {} and {} with swish".format(logistic_loc_position, mul_loc_position))

        
        def layerNormHPatternMatcher():
            def lrnmh_match_node_by_op(n1, n2):
                op1 = n1.get("op_type")
                op2 = n2.get("op_type")
                if op1 == "any" or op2 == "any":
                    return True
                return op1 == op2
            
            layerNormHPattern = nx.DiGraph()
            logger.info("start layernorm_h pattern")
            layerNormHPattern.add_node("mean_1", op_type="MEAN")
            layerNormHPattern.add_node("SquaredDifference", op_type="SQUARED_DIFFERENCE")
            layerNormHPattern.add_node("mean_2", op_type="MEAN")
            layerNormHPattern.add_node("add_1", op_type="ADD")
            layerNormHPattern.add_node("rsqrt", op_type="RSQRT")
            layerNormHPattern.add_node("mul_1", op_type="MUL")
            layerNormHPattern.add_node("mul_2", op_type="MUL")
            layerNormHPattern.add_node("mul_3", op_type="MUL")
            layerNormHPattern.add_node("sub_1", op_type="SUB")
            layerNormHPattern.add_node("add_2", op_type="ADD")

            layerNormHPattern.add_edge("mean_1","SquaredDifference")
            layerNormHPattern.add_edge("SquaredDifference","mean_2")
            layerNormHPattern.add_edge("mean_2","add_1")
            layerNormHPattern.add_edge("add_1","rsqrt")
            layerNormHPattern.add_edge("rsqrt","mul_1")
            layerNormHPattern.add_edge("mul_1","mul_2")
            layerNormHPattern.add_edge("mean_1","mul_3")
            layerNormHPattern.add_edge("mul_1","mul_3")
            layerNormHPattern.add_edge("mul_3","sub_1")
            layerNormHPattern.add_edge("sub_1","add_2")
            layerNormHPattern.add_edge("mul_2","add_2")

            matches = isomorphism.DiGraphMatcher(self.graph, layerNormHPattern, node_match=lrnmh_match_node_by_op)
            write_dot(layerNormHPattern ,'layernorm.dot')
            for match in list(matches.subgraph_isomorphisms_iter()):
                current_topological_sort = list(nx.topological_sort(self.graph))
                match_reversed = {y: x for x, y in match.items()}
                match = match_reversed
                
                mean_1_position = match["mean_1"]
                squared_difference_position = match["SquaredDifference"]
                mean_2_position = match["mean_2"]
                add_1_position = match["add_1"]
                rsqrt_position = match["rsqrt"]
                mul_1_position = match["mul_1"]
                mul_2_position = match["mul_2"]
                mul_3_position = match["mul_3"]
                sub_1_position = match["sub_1"]
                add_2_position = match["add_2"]

                mean_1_node_data = self.getNodeAttribute(mean_1_position)["data"]
                squared_difference_data = self.getNodeAttribute(squared_difference_position)["data"]
                mean_2_data = self.getNodeAttribute(mean_2_position)["data"]
                
                add_1_data = self.getNodeAttribute(add_1_position)["data"]
                rsqrt_data = self.getNodeAttribute(rsqrt_position)["data"]
                mul_1_data = self.getNodeAttribute(mul_1_position)["data"]
                mul_2_data = self.getNodeAttribute(mul_2_position)["data"]
                mul_3_data = self.getNodeAttribute(mul_3_position)["data"]
                sub_1_data = self.getNodeAttribute(sub_1_position)["data"]
                add_2_data = self.getNodeAttribute(add_2_position)["data"]
                add_1_constant_index = add_1_data['inputs'][1]
                mul_1_constant_index = mul_1_data['inputs'][1]
                sub_1_constant_index = sub_1_data['inputs'][0]

                nodes_after_add2 = current_topological_sort[current_topological_sort.index(add_2_position)+1:]
                layerNorm_h_output_nodes = []
                
                for node in nodes_after_add2:
                    node_data = self.getNodeAttribute(node)["data"]
                    node_pos = int(str(add_2_position).split('_')[1])
                    if node_pos in node_data["inputs"]:
                        layerNorm_h_output_nodes.append(node)

                # outputs_list = [int(str(x).split("_")[1]) for x in layerNorm_h_output_nodes]
                # if len(nodes_after_add2) == 0:
                #     outputs_list = [int(str(add_2_position).split("_")[1])]

                outputs_list = self.getNodeAttribute(add_2_position)["data"]["outputs"]
                
                mean1_axis_loc = mean_1_node_data["inputs"][1]
                mean2_axis_loc = mean_2_data['inputs'][1]

                mean1_axis = self.model["tensor_details"][mean1_axis_loc]['value']
                mean2_axis = self.model["tensor_details"][mean2_axis_loc]['value']

                
                #import pdb;pdb.set_trace()
                param_dict = {
                    'name': str(add_2_position),
                    'type': "LAYERNORM_H",
                    'inputs'  : [mean_1_node_data["inputs"][0]] ,
                    'outputs' : outputs_list,
                    'input_scales': [mean_1_node_data["input_scales"][0],
                                    mean_1_node_data["output_scales"][0]],
                    'input_zero_points' : [mean_1_node_data["input_zero_points"][0],
                                           mean_1_node_data["output_zero_points"][0]],
                    'mean_1_axis' : mean1_axis,
                    'mean_1_output_scales' : mean_1_node_data['output_scales'],
                    'mean_1_output_zero_points' : mean_1_node_data['output_zero_points'],
                    'squared_difference_output_zero_points' : squared_difference_data['output_zero_points'],
                    'squared_difference_output_scales' : squared_difference_data['output_scales'],
                    'mean_2_axis' : mean2_axis,
                    'mean_2_output_scales' : mean_2_data['output_scales'],
                    'mean_2_output_zero_points' : mean_2_data['output_zero_points'],
                    'add_1_output_scales'  :  add_1_data['output_scales'],
                    'add_1_output_zero_points'  :  add_1_data['output_zero_points'],
                    'add_1_constant' : self.model['tensor_details'][add_1_constant_index]['value'],
                    'add_1_constant_scale': add_1_data['input_scales'][1],
                    'add_1_constant_zero_points': add_1_data['input_zero_points'][1],
                    'rsqrt_output_scales'  :  rsqrt_data['output_scales'],
                    'rsqrt_output_zero_points'  :  rsqrt_data['output_zero_points'],
                    'mul_1_output_scales'  :  mul_1_data['output_scales'],
                    'mul_1_output_zero_points'  :  mul_1_data['output_zero_points'],
                    'mul_1_constant' : self.model['tensor_details'][mul_1_constant_index]['value'],
                    'mul_1_constant_scale': mul_1_data['input_scales'][1],
                    'mul_1_constant_zero_points': mul_1_data['input_zero_points'][1],
                    'mul_2_output_scales'  :  mul_2_data['output_scales'],
                    'mul_2_output_zero_points'  :  mul_2_data['output_zero_points'],
                    'mul_3_output_scales'  :  mul_3_data['output_scales'],
                    'mul_3_output_zero_points'  :  mul_3_data['output_zero_points'],
                    'sub_1_output_scales'  :  sub_1_data['output_scales'],
                    'sub_1_output_zero_points'  :  sub_1_data['output_zero_points'],
                    'sub_1_constant' : self.model['tensor_details'][sub_1_constant_index]['value'],
                    'sub_1_constant_scale': sub_1_data['input_scales'][0],
                    'sub_1_constant_zero_points': sub_1_data['input_zero_points'][0],
                    'add_2_output_scales'  :  add_2_data['output_scales'],
                    'add_2_output_zero_points'  :  add_2_data['output_zero_points'],
                    'output_scales' : [add_2_data["output_scales"][0]],
                    "output_zero_points" : [add_2_data["output_zero_points"][0]]
                }

                self.graph.remove_node(mean_1_node_data['name'])
                self.graph.remove_node(squared_difference_data['name'])
                self.graph.remove_node(mean_2_data['name'])
                self.graph.remove_node(add_1_data['name'])
                self.graph.remove_node(rsqrt_data['name'])
                self.graph.remove_node(mul_1_data['name'])
                self.graph.remove_node(mul_2_data['name'])
                self.graph.remove_node(mul_3_data['name'])
                self.graph.remove_node(sub_1_data['name'])
                self.graph.remove_node(add_2_data['name'])

                self.add_node(param_dict)
                for output_node in layerNorm_h_output_nodes:
                    self.graph.add_edge(param_dict['name'], output_node)
                logger.info("done with layernorm_h")
        
        def reshapePatternMatcher():
            def rshp_match_node_by_op(n1, n2):
                op1 = n1.get("op_type")
                op2 = n2.get("op_type")
                if op1 == "any" or op2 == "any":
                    return True
                return op1 == op2
            
            rshpPattern = nx.DiGraph()
            logger.info("start reshape pattern")
            rshpPattern.add_node("reshape_1", op_type="RESHAPE")
            rshpPattern.add_node("shape_1", op_type="SHAPE")
            rshpPattern.add_node("gather_1", op_type="GATHER")
            rshpPattern.add_node("concatenation_1", op_type="CONCATENATION")
            rshpPattern.add_node("gather_2", op_type="GATHER")
            rshpPattern.add_node("reduce_prod_1", op_type="REDUCE_PROD")
            rshpPattern.add_node("reduce_prod_2", op_type="REDUCE_PROD")
            rshpPattern.add_node("pack_1", op_type="PACK")

            rshpPattern.add_edge("reshape_1","shape_1")
            rshpPattern.add_edge("shape_1","gather_1")
            rshpPattern.add_edge("gather_1","concatenation_1")
            rshpPattern.add_edge("shape_1","gather_2")
            rshpPattern.add_edge("gather_1","reduce_prod_1")
            rshpPattern.add_edge("gather_2","reduce_prod_2")
            rshpPattern.add_edge("reduce_prod_1","pack_1")
            rshpPattern.add_edge("reduce_prod_2","pack_1")

            matches = isomorphism.DiGraphMatcher(self.graph, rshpPattern, node_match=rshp_match_node_by_op)
            write_dot(rshpPattern ,'reshape_1.dot')

            for match in list(matches.subgraph_isomorphisms_iter()):
                logger.info("Match found")
                current_topological_sort = list(nx.topological_sort(self.graph))
                match_reversed = {y: x for x, y in match.items()}
                match = match_reversed
                reshape1_position = match["reshape_1"]
                shape1_position = match["shape_1"]
                gather1_position = match["gather_1"]
                concat1_position = match["concatenation_1"]
                gather2_position = match["gather_2"]
                reduceprod1_position = match["reduce_prod_1"]
                reduceprod2_position = match["reduce_prod_2"]
                pack1_position = match["pack_1"]

                reshape1_data = self.getNodeAttribute(reshape1_position)["data"]
                shape1_data = self.getNodeAttribute(shape1_position)["data"]
                gather1_data = self.getNodeAttribute(gather1_position)["data"]
                concat1_data = self.getNodeAttribute(concat1_position)["data"]
                gather2_data = self.getNodeAttribute(gather2_position)["data"]
                reduceprod1_data = self.getNodeAttribute(reduceprod1_position)["data"]
                reduceprod2_data = self.getNodeAttribute(reduceprod2_position)["data"]
                pack1_data = self.getNodeAttribute(pack1_position)["data"]

                for pos in [concat1_position, pack1_position]:
                    nodes_after_cur = current_topological_sort[current_topological_sort.index(pos)+1:]
                    rshp_output_nodes = []

                    for node in nodes_after_cur:
                        if "data" in self.getNodeAttribute(node):
                            node_data = self.getNodeAttribute(node)["data"]
                            node_pos = int(str(pos).split('_')[1])
                            if "inputs" in node_data:
                                if node_pos in node_data["inputs"]:
                                    rshp_output_nodes.append(node)
                    for node in rshp_output_nodes:
                        self.getNodeAttribute(node)["data"]["inputs"].remove(int(str(pos).split("_")[1]))
                
                self.graph.remove_node(reshape1_data['name'])
                self.graph.remove_node(shape1_data['name'])
                self.graph.remove_node(gather1_data['name'])
                self.graph.remove_node(concat1_data['name'])
                self.graph.remove_node(gather2_data['name'])
                self.graph.remove_node(reduceprod1_data['name'])
                self.graph.remove_node(reduceprod2_data['name'])
                self.graph.remove_node(pack1_data['name'])
            
            logger.info("done with reshape")
        
        def reshape1PatternMatcher():
            def rshp_match_node_by_op(n1, n2):
                op1 = n1.get("op_type")
                op2 = n2.get("op_type")
                if op1 == "any" or op2 == "any":
                    return True
                return op1 == op2
            
            rshpPattern = nx.DiGraph()
            logger.info("start reshape-1 pattern")
            # rshpPattern.add_node("reshape_1", op_type="RESHAPE")
            rshpPattern.add_node("shape_1", op_type="SHAPE")
            rshpPattern.add_node("gather_1", op_type="GATHER")
            rshpPattern.add_node("concatenation_1", op_type="CONCATENATION")
            rshpPattern.add_node("gather_2", op_type="GATHER")
            rshpPattern.add_node("reduce_prod_1", op_type="REDUCE_PROD")
            rshpPattern.add_node("reduce_prod_2", op_type="REDUCE_PROD")
            rshpPattern.add_node("pack_1", op_type="PACK")

            # rshpPattern.add_edge("reshape_1","shape_1")
            rshpPattern.add_edge("shape_1","gather_1")
            rshpPattern.add_edge("gather_1","concatenation_1")
            rshpPattern.add_edge("shape_1","gather_2")
            rshpPattern.add_edge("gather_1","reduce_prod_1")
            rshpPattern.add_edge("gather_2","reduce_prod_2")
            rshpPattern.add_edge("reduce_prod_1","pack_1")
            rshpPattern.add_edge("reduce_prod_2","pack_1")

            matches = isomorphism.DiGraphMatcher(self.graph, rshpPattern, node_match=rshp_match_node_by_op)
            write_dot(rshpPattern ,'reshape_1.dot')

            for match in list(matches.subgraph_isomorphisms_iter()):
                logger.info("Match found")
                current_topological_sort = list(nx.topological_sort(self.graph))
                match_reversed = {y: x for x, y in match.items()}
                match = match_reversed
                # reshape1_position = match["reshape_1"]
                shape1_position = match["shape_1"]
                gather1_position = match["gather_1"]
                concat1_position = match["concatenation_1"]
                gather2_position = match["gather_2"]
                reduceprod1_position = match["reduce_prod_1"]
                reduceprod2_position = match["reduce_prod_2"]
                pack1_position = match["pack_1"]

                # reshape1_data = self.getNodeAttribute(reshape1_position)["data"]
                shape1_data = self.getNodeAttribute(shape1_position)["data"]
                gather1_data = self.getNodeAttribute(gather1_position)["data"]
                concat1_data = self.getNodeAttribute(concat1_position)["data"]
                gather2_data = self.getNodeAttribute(gather2_position)["data"]
                reduceprod1_data = self.getNodeAttribute(reduceprod1_position)["data"]
                reduceprod2_data = self.getNodeAttribute(reduceprod2_position)["data"]
                pack1_data = self.getNodeAttribute(pack1_position)["data"]

                for pos in [concat1_position, pack1_position]:
                    nodes_after_cur = current_topological_sort[current_topological_sort.index(pos)+1:]
                    rshp_output_nodes = []

                    for node in nodes_after_cur:
                        if "data" in self.getNodeAttribute(node):
                            node_data = self.getNodeAttribute(node)["data"]
                            node_pos = int(str(pos).split('_')[1])
                            if "inputs" in node_data:
                                if node_pos in node_data["inputs"]:
                                    rshp_output_nodes.append(node)
                    for node in rshp_output_nodes:
                        self.getNodeAttribute(node)["data"]["inputs"].remove(int(str(pos).split("_")[1]))
                
                # self.graph.remove_node(reshape1_data['name'])
                self.graph.remove_node(shape1_data['name'])
                self.graph.remove_node(gather1_data['name'])
                self.graph.remove_node(concat1_data['name'])
                self.graph.remove_node(gather2_data['name'])
                self.graph.remove_node(reduceprod1_data['name'])
                self.graph.remove_node(reduceprod2_data['name'])
                self.graph.remove_node(pack1_data['name'])
            
            logger.info("done with reshape-1")
        
        def FCPatternMatcher():
            def fc_match_node_by_op(n1, n2):
                op1 = n1.get("op_type")
                op2 = n2.get("op_type")
                if op1 == "any" or op2 == "any":
                    return True
                return op1 == op2
            
            fcPattern = nx.DiGraph()
            logger.info("start fc pattern")
            fcPattern.add_node("fc", op_type="FULLY_CONNECTED")
            fcPattern.add_node("rshp", op_type="RESHAPE")
            fcPattern.add_node("add", op_type="ADD")

            fcPattern.add_edge("fc", "rshp")
            fcPattern.add_edge("rshp", "add")

            matches = isomorphism.DiGraphMatcher(self.graph, fcPattern, node_match=fc_match_node_by_op)
            write_dot(fcPattern ,'fc_1.dot')
            return

            for match in list(matches.subgraph_isomorphisms_iter()):
                current_topological_sort = list(nx.topological_sort(self.graph))
                match_reversed = {y: x for x, y in match.items()}
                match = match_reversed
                fc_position = match["fc"]
                rshp_position = match["rshp"]
                add_position = match["add"]

                fc_data = self.getNodeAttribute(fc_position)["data"]
                rshp_data = self.getNodeAttribute(rshp_position)["data"]
                add_data = self.getNodeAttribute(add_position)["data"]

                nodes_after_add = current_topological_sort[current_topological_sort.index(add_position)+1:]
                add_output_nodes = []
                
                for node in nodes_after_add:
                    if "data" not in self.getNodeAttribute(node):
                        continue
                    node_data = self.getNodeAttribute(node)["data"]
                    node_pos = int(str(add_position).split('_')[1])
                    if "inputs" not in node_data:
                        continue
                    if node_pos in node_data["inputs"]:
                        add_output_nodes.append(node)
                
                bias_appended = False
                
                for input in add_data["inputs"]:
                    if bias_appended:
                        break
                    if 'value' in self.model["tensor_details"][input]:
                        if len(fc_data["inputs"]) == 2:
                            fc_data["inputs"].append(input)
                            bias_appended = True
                        elif len(fc_data["inputs"]) == 3 and fc_data["inputs"][-1] == -1:
                            fc_data["inputs"][-1] = input
                            bias_appended = True
                
                if not bias_appended:
                    continue

                # Bias added to FC. Attach add outputs to reshape and remove add node
                for node in add_output_nodes:
                    self.getNodeAttribute(node)["data"]["inputs"].remove(int(str(add_position).split("_")[1]))
                    self.getNodeAttribute(node)["data"]["inputs"].append(int(str(rshp_position).split("_")[1]))
                    self.graph.add_edge(rshp_position, node)
                
                self.graph.remove_node(add_data['name'])
            
            logger.info("done with fc")
  
        for k, v in locals().items():
            if callable(v) and v.__module__ == __name__:
                v()

    def getTopologicalSort(self) -> list:
        return list(nx.topological_sort(self.graph))

    def getGraph(self):
        return self.graph

    def saveGraph(self):
        write_dot(self.graph, "graph.dot")