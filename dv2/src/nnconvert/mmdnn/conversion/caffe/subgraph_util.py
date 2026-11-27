'''
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
import argparse
import google.protobuf as pb
import google.protobuf.text_format
import sys as _sys
import os
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../../../python/caffe/proto')
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../../../python')
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../')
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../../../analyzer/python')
import caffe
from caffe import layers as L
import caffe.proto.caffe_pb2 as caffe_pb2
from caffe import params as P
from caffe import to_proto
from dv_logging import get_logger
logger = get_logger(__name__, 'DVCTOC')

def _main(_args=None):
    #print(COPYRIGHT_MSG)
    parser = argparse.ArgumentParser()  
    required_args = parser.add_argument_group('required arguments')
    required_args.add_argument('-inet', "--inputPrototxt",
                            help='Input Caffe Prototxt Path', required=True)
    required_args.add_argument('-onet', "--outputPrototxt",
                            help='Output Caffe Prototxt Path', required=True)

    required_args.add_argument('-ishape', "--inputShape",nargs='+',
                            help='Input Shape', required=True)  
    required_args.add_argument('-inputnn', "--inputnodename",nargs='+',
                            help='Input Node Name', required=True)  
    required_args.add_argument('-outputnn', "--outputnodename",nargs='+',
                            help='Output Node Name', required=True)  
    
    args = parser.parse_args() if _args == None else parser.parse_args(_args)

    #if inputShape, inode and onode are list of string then coming from stand alone arguments, if list of list then coming from caffe2caffe
    
    if isinstance(args.inputShape[0], list) and isinstance(args.inputnodename[0], list) and isinstance(args.outputnodename[0], list):
        args.inputShape = args.inputShape[0]
        args.inputnodename = args.inputnodename[0]
        args.outputnodename = args.outputnodename[0]
    
    if(len(args.inputShape)!=len(args.inputnodename)):
        raise Exception("Error: number of input node names and input shapes is not equal")
    
    # Load the prototxt file
    with open(args.inputPrototxt, 'rb') as f:
        net = caffe_pb2.NetParameter()
        pb.text_format.Merge(f.read(), net)

    # Create a NetworkX graph object
    graph = nx.DiGraph()

    # Add nodes to the graph for each layer in the network
    for layer in net.layer:
        graph.add_node(layer.name, type=layer.type, data=layer)
    for inn in args.inputnodename:
        if(inn not in graph.nodes):
            raise Exception("Input node name not present in DAG")
    for onn in args.outputnodename:
        if(onn not in graph.nodes):
            raise Exception("Output node name not present in DAG")
    list_of_dim=[]
    for i in range(len(args.inputnodename)):
        list_of_dim.append([(list(map(int, j.split(',')))) for j in(args.inputShape)[i].split(' ')])

    # Add edges to the graph to represent connuth/error?error=RequestTimeoutections between layers
    unique_top_layers = set()
    for layer in net.layer:
        for top in layer.top:
            if top in unique_top_layers:
                raise Exception("Unique top layer found, not extracting subgraph")
            unique_top_layers.add(top)
        for bottom in layer.bottom:
            graph.add_edge(bottom,layer.name)

    finalset=set()
    for _i,i in enumerate(args.inputnodename):
        for _j,j in enumerate(args.outputnodename):
            paths_between_generator = nx.all_simple_paths(graph,source=i,target=j)
            nodes_between_set = {node for path in paths_between_generator for node in path}
            finalset=finalset.union(nodes_between_set)
            

    SG = graph.subgraph(finalset)
    layer_pb = caffe_pb2.NetParameter()
    topsort = list(nx.lexicographical_topological_sort(SG))
    count=0
    for name in topsort:
        if(nx.get_node_attributes(SG, "data")[name].name in args.inputnodename and (nx.get_node_attributes(SG, "data")[name].type)!='Input'):
            for _i,nesteddims in enumerate(list_of_dim[count]):
                _data=0
                inp_layer=(L.Input(shape=dict(dim=nesteddims)).to_proto().layer[0])
                inp_layer.top[0]='input'+str(count)
                inp_layer.name='input'+str(count)
                layer_pb.layer.append(inp_layer)
                _data=nx.get_node_attributes(SG, "data")[name]
                if(_i==0):
                    _data.ClearField("bottom")
                    _data.bottom.append('input'+str(_i))
                else:
                    _data.bottom.append('input'+str(_i))
            layer_pb.layer.append(nx.get_node_attributes(SG, "data")[name])
            count+=1
        else:
            layer_pb.layer.append(nx.get_node_attributes(SG, "data")[name])
    
    try:
        with open(args.outputPrototxt, 'w') as f:
                f.write(pb.text_format.MessageToString(layer_pb))
                f.flush()
    except:
        logger.error("Something went wrong writing to the file")
        return -1
    try:
        net = caffe.Net(args.outputPrototxt, caffe.TEST)
    except Exception as e:
        logger.error("Error with saved prototxt")
        return -1
    
    return 0
    
if __name__ == "__main__":
    ret = _main()
    if ret != 0:
        exit(int(ret))
