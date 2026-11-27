'''
# Copyright (c) 2018-22, Deep Vision, Inc. All rights reserved.
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

import numpy as np
import argparse
import onnx

import networkx as nx
from networkx.algorithms import isomorphism
from networkx.drawing.nx_pydot import write_dot
from six import text_type as _text_type
from itertools import product 
import sys

def dotMatcher(onnxModel, startNode, endNode, dotFile):
    print("Converting ONNX -> NX ")
    G = onnx_to_nx(onnxModel)
    __checks(G, startNode, endNode)
    print("Basic Checks passed ")
    nodes_between_set = set()
    for start, end in product(startNode, endNode): 
        paths_between_set = nx.all_simple_paths(G,source=start,target=end)
        __nodes_between_set = set([node for path in paths_between_set for node in path])
        if len(__nodes_between_set) == 0:
            __nodes_between_set.add(start)
        
        nodes_between_set.update(__nodes_between_set)
    SG = G.subgraph(nodes_between_set)
    if SG.number_of_nodes() > 500: 
        print("Extraction may take time, Large graph detected...")
    if args.relabelNodes:
        _type_map = dict()
        relabel = dict()
        for idx, node in enumerate(SG):
            if SG.nodes[node]["type"] not in _type_map:
                _type_map[SG.nodes[node]["type"]] = 0
            _type_map[SG.nodes[node]["type"]] += 1
            new_node_name = "{}_{}".format(SG.nodes[node]["type"], _type_map[SG.nodes[node]["type"]])
            relabel[node] = new_node_name
    print("Extracted Subgraph!")
    save_subgraph(SG, dotFile)

    if args.testPatterns:
        SG = nx.DiGraph(nx.drawing.nx_pydot.read_dot(dotFile))
        if SG.has_node("\\n"):
            SG.remove_node("\\n")
        matches = isomorphism.DiGraphMatcher(G,SG,node_match=node_comparator_by_type)
        print("Number of patterns in the graph: ", len(list(matches.subgraph_isomorphisms_iter())))



#utility
def node_comparator_by_type(n1, n2) :
        n1t = n1.get('type', True)
        n2t = n2.get('type', False)
        # Const* vs Constant
        if not n1 or not n2:
            return False
        return n1t == n2t

def __checks(G, startNode, endNode):
    # valid inputnode
    for node in startNode: 
        if not G.has_node(node): 
            sys.exit("Error: invalid startNode ")
    # valid output
    for node in endNode: 
        if not G.has_node(node): 
            sys.exit("Error: invalid endNode ")
    

def _output_tensor_to_name_map(onnx_model):
    _output_tensor_map = dict() 
    for nodes in onnx_model.graph.node:
        for _idx, tensors in enumerate(nodes.output):
            _output_tensor_map[tensors] = nodes.name
    return _output_tensor_map

def add_node(G, node):
    G.add_node(node.name, type=node.op_type)
    
def add_initializer_node(G, node):
    G.add_node(node.name, type="tensor")
    
def add_input_node(G, onnx_model):
    for idx, node in enumerate(onnx_model.graph.input):
        G.add_node(node.name, type="input")

def add_output_node(G, onnx_model):
    _out_nodes = list()
    for idx, node in enumerate(onnx_model.graph.output):
        _out_nodes.append(node.name)
        G.add_node(node.name, data=node, type='output')
    return _out_nodes

def onnx_to_nx(onnx_model):
    onnx_model = onnx.load(onnx_model)
    G = nx.DiGraph()
    edge_list = list()
    tensor_to_name_map = _output_tensor_to_name_map(onnx_model)
    add_input_node(G, onnx_model)
    out_nodes = add_output_node(G, onnx_model)
    for _idx, node in enumerate(onnx_model.graph.initializer):
        add_initializer_node(G, node)
    for _idx, node in enumerate(onnx_model.graph.node):
        add_node(G, node)
        for _inp in node.input:
            if _inp in tensor_to_name_map:
                _inp = tensor_to_name_map[_inp]
            edge_list.append(tuple((_inp, node.name)))
    add_output_node(G, onnx_model)
    
    G.add_edges_from(edge_list)
    return G

def save_subgraph(G, dotFile='./subgraph.dot'):
    write_dot(G, dotFile)
    print("Subgraph Saved ", dotFile)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('-inet', "--inputONNX",
                           help='Input onnx model Path', required=True)
    required_args.add_argument('-dfile', "--dotFile",
                           help='Output dotFile Path', required=True)
    required_args.add_argument('-start', "--startNode", nargs='+', type=_text_type, 
                           help='start nodes', required=False)  
    required_args.add_argument('-end', "--endNode", nargs='+', type=_text_type, 
                           help='end nodes', required=False)  
    optional_args = parser.add_argument_group('optional arguments')
    optional_args.add_argument('-relabelNodes', "--relabelNodes", 
                                action='store_true', help='relabel nodes to have simpler node names in subgraph')
    optional_args.add_argument('-test', "--testPatterns", 
                                action='store_true', help='loads the same pattern and print number of matches in original graph')
     
    args = parser.parse_args()
    startNode = [node for node in args.startNode]
    endNode = [node for node in args.endNode]

    dotMatcher(args.inputONNX, startNode, endNode, args.dotFile)