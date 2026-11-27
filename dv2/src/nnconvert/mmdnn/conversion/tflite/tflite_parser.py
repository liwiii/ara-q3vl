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


import numpy as np
from mmdnn.conversion.tflite.tflite_graph import TFLiteGraph
from tensorflow.python.framework import tensor_util
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from tensorflow.core.framework import attr_value_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.DataStructure.parser import Parser
from mmdnn.conversion.common.utils import *
import tensorflow as tf
import copy
import subprocess
import shutil
import constants
import os
import json
from dv_logging import get_logger

logger = get_logger(__name__, 'DVPRSR')

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('true', '1'):
        return True
    elif v.lower() in ('false', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class TFLiteParser(Parser):

    channels_last = ['NDHWC', 'NHWC', 'NWC']
    channels_first = ['NCDHW', 'NCHW', 'NCW']
    numpy_dtype = {
        'int8': np.int8,
        'uint8': np.uint8,
        'float': np.float32,
        'double': np.double
    }

    @property
    def src_graph(self):
        return self.tflite_graph
    
    def get_weights_list(self,source_node):
        i = 0
        weights = []
        for input in source_node.get_attr('inputs'):
            if input < 0:
                continue
            if i == 0:
                i = i+1 
                continue
            else:
                if 'value' in self.src_graph.model['tensor_details'][input]:
                    data = np.array(copy.deepcopy(self.src_graph.model['tensor_details'][input]['value']))
                    shape = self.src_graph.model['tensor_details'][input]['shape']
                    data = data.reshape(shape)
                else:
                    continue
                weights.append(data)
                i = i+1
        return weights
    
    @staticmethod
    def _convert_dict(dict_obj):
        for k,v in dict_obj.items():
            if isinstance(v, np.ndarray):
                dict_obj[k] = v.tolist()
            elif isinstance(v, type):
                dict_obj[k] = v.__name__
            elif isinstance(v, dict):
                TFLiteParser._convert_dict(v)

    @staticmethod
    def _load_model(network, output_dir):
    #TODO 
    #CALL FLATC COMPILER WITH SCHEMA HERE. DUMP JSON IN TMP AND LOAD AND DELETE 
    #SEE IF COMPILING IS BETTER INSTEAD OF USING BINARY. 
        cmd = [constants.FLATC_BINARY, "-t", "--strict-json", "--defaults-json", "-o", output_dir, os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.fbs"), "--", network]
        subprocess.run(cmd)
        json_file = os.path.join(output_dir, os.path.splitext(os.path.basename(network))[0] + '.json')
        with open(json_file, 'r') as f:
            tmp_json = json.load(f)
        #shutil.rmtree('/tmp/nnconvert')
        md = tf.lite.Interpreter(network, experimental_preserve_all_tensors=True)
        # Run the model once to get the output shapes of all the tensors
        md.allocate_tensors()
        input_details = md.get_input_details()
        input_shape = input_details[0]['shape']
        input_data = np.random.randint(-128, high=127, size=input_shape, dtype=np.int8)
        md.set_tensor(input_details[0]['index'], input_data)
        md.invoke()
        mdl_dict = dict()
        mdl_dict['tensor_details'] = md.get_tensor_details()
        mdl_dict['ops'] = md._get_ops_details()
        mdl_dict['inputs'] = md.get_input_details()
        mdl_dict['outputs'] = md.get_output_details()
        const_tensors = []
        for tensor in mdl_dict['tensor_details']:
            tensor_idx = tensor['index']
            const_tensor_found = True
            for op in mdl_dict['ops']:
                if tensor_idx in op['outputs']:
                    const_tensor_found = False
                    break
            if const_tensor_found:
                const_tensors.append(tensor_idx)
        # Append the value of the tensors if there to the tensor_details
        for ind in range(len(mdl_dict['tensor_details'])):
            try:
                if mdl_dict['tensor_details'][ind]['index'] in const_tensors:
                    mdl_dict['tensor_details'][ind]['value'] = md.get_tensor(mdl_dict['tensor_details'][ind]['index'])
            except Exception as e:
                continue
        for k,v in mdl_dict.items():
            for item in v:
                TFLiteParser._convert_dict(item)
        # Update ops with builtin_options of tmp_json
        for ind, op in enumerate(mdl_dict['ops']):
            op['attributes'] = tmp_json['subgraphs'][0]['operators'][ind].get('builtin_options', {})
        model_json = copy.deepcopy(mdl_dict)
        return model_json

    def __init__(self, weights, input_shape, in_node_name, dst_node_name, hints, dynamic_input, output_dir, debug=False):

        super(TFLiteParser, self).__init__()
        self.data_shape  = dict()
        
        assert isinstance(input_shape, list)
        assert isinstance(in_node_name, list)
        if len(input_shape) < len(in_node_name):
            raise ValueError("InputShape for each node must be present")
        elif len(input_shape) > len(in_node_name):
            raise ValueError("More input shapes provided than number of input node names")
        for i in range(len(input_shape)):
            self.data_shape[in_node_name[i]] = tuple([1] + list(map(int, input_shape[i])))
        self.model = TFLiteParser._load_model(weights, output_dir)
        self.weight_loaded = True
        self.debug = debug


        self.data_format = 'NHWC' #DV specific hardcoding 
        self.tflite_graph = TFLiteGraph(self.model)
        self.tflite_graph.build(self.model)

    def gen_IR(self):
        self.dummy_layers = set()
        if self.debug:
            import json
            with open(self.debug, 'r') as fd:
                _gen_dummy = json.load(fd)
            for dummy_layer in _gen_dummy:
                logger.info("cp layer is found from estimator json %s", dummy_layer)
                failed_layer_list = _gen_dummy[dummy_layer]
                for layer in failed_layer_list:
                    logger.info("adding dummy layer ot the list %s", layer)
                    self.dummy_layers.add(layer)

        self.IR_layer_map = dict()
        # self.rename_inputs()
        for layer in self.tflite_graph.topological_sort:
            current_node = self.tflite_graph.get_node(layer)
            node_type = current_node.type
            if node_type == 'NoOp':
                continue
            if hasattr(self, "rename_" + node_type):
                func = getattr(self, "rename_" + node_type)
                func(current_node)

            else:
                self.rename_UNKNOWN(current_node)

    
    @staticmethod
    def _copy_shape(IR_node, output_list):
        if not output_list == None:
            for dim in output_list:
                new_dim = IR_node.attr["shape"].shape.dim.add()
                if dim == None:
                    new_dim.size = -1
                else:
                    new_dim.size = dim
        else:
            IR_node.attr["shape"].shape.unknown_rank = True

    def _copy_and_reop(self, source_node, IR_node, new_op = None):
        new_op = source_node.type if new_op == None else new_op
        if source_node.name.startswith('_'):
            source_node.real_name = source_node.name[1:]
        IR_node.name = source_node.real_name
        IR_node.op = new_op
        assign_IRnode_values(IR_node, {'data_format' : 'NHWC'})
        self.IR_layer_map[IR_node.name] = IR_node

    @staticmethod
    def tensor_shape_to_list(shapes):
        if isinstance(shapes, attr_value_pb2.AttrValue):
            return [dim.size for dim in shapes.shape.dim]

        else:
            ret = []
            for shape in shapes:
                this_one = [dim.size for dim in shape.dim]
                ret.append(this_one)
            return ret
    
    def get_lnrm_array(self, np_array, axis):
        if np_array.ndim == 4:
            np_array = np_array
        elif np_array.ndim == 1:
            size_ = np_array.size
            shape_ = []
            for i in range(4):
                if axis == i:
                    shape_.append(size_)
                else:
                    shape_.append(1)
            np_array = np_array.reshape(shape_)
        return np_array


    
    def get_output_shape(self, source_node):
        if source_node.type == "INPUT":
            return source_node.get_attr('shape')
        return self.src_graph.model['tensor_details'][source_node.get_attr('outputs')[0]]['shape']
    
    def get_output_shape_signature(self, source_node):
        if source_node.type == "INPUT":
            return source_node.get_attr('shape')
        return self.src_graph.model['tensor_details'][source_node.get_attr('outputs')[0]]['shape_signature']
    
    def get_tflite_format(self, shape):
        tflite_formats = ["None", "None", "NC", "NWC", "NHWC"]
        return tflite_formats[len(shape)]

    def set_output_shape(self, source_node, IR_node):
        output_list = self.src_graph.model['tensor_details'][int(source_node.get_attr('outputs')[0])]['shape']
        shape = graph_pb2.TensorShape()
        for dim in output_list:
            new_dim = shape.dim.add()
            if dim == None:
                new_dim.size = -1
            else:
                new_dim.size = dim
        IR_node.attr["_output_shapes"].list.shape.extend([shape])

    def _convert_identity_operation(self, source_node, start_idx = 0, end_idx = None, new_op=None):
        IR_node = self.IR_graph.node.add()

        # name, op
        self._copy_and_reop(source_node, IR_node, new_op)

        # input edge
        self.convert_inedge_tflite(source_node, IR_node, start_idx, end_idx)

        # output shape
        self.set_output_shape(source_node, IR_node)

        return IR_node

    def _convert_padding(self, source_node, IR_node, kernel_size):
        # TODO: Fused conv and pool with padding is different from defused operators
        input_shape = shape_to_list(self.IR_layer_map[IR_node.input[0]].attr["_output_shapes"].list.shape[0])
        strides = [source_node.get_attr('attributes')['stride_h'],source_node.get_attr('attributes')['stride_w']]
        
        
        if source_node.get_attr('attributes')['padding'] == 'VALID':
            dims = len(input_shape)
            assign_IRnode_values(IR_node, {'auto_pad' : "VALID", 'pads' : [0, 0] * dims})
        
        elif source_node.get_attr('attributes')['padding'] == 'SAME':
            padding = compute_tf_same_padding(
                input_shape,
                kernel_size,
                strides)
            logger.debug("padding for layer is %s", str(padding))
            assign_IRnode_values(IR_node, {'auto_pad' : "SAME_LOWER", 'pads' : padding})

        else:
            assert False
            
    def rename_inputs(self):
        for input in self.tflite_graph.model['inputs']:
            IR_node = self.IR_graph.node.add()
            IR_node.name = "loc_" + str(input['index'])
            IR_node.op = "tflite_DataInput"
            kwargs = {}
            kwargs['data_format'] = 'NHWC'
            kwargs['output_scale'] = input['quantization_parameters']['scales'][0]
            kwargs['output_zero_point'] = input['quantization_parameters']['zero_points'][0]
            assign_IRnode_values(IR_node, kwargs)
            output_list = input['shape']
            shape = graph_pb2.TensorShape()
            for dim in output_list:
                new_dim = shape.dim.add()
                if dim == None:
                    new_dim.size = -1
                else:
                    new_dim.size = dim
            IR_node.attr["_output_shapes"].list.shape.extend([shape])
            self._copy_shape(IR_node,output_list)   
            self.IR_layer_map[IR_node.name] = IR_node  
            

    def rename_UNKNOWN(self, source_node):
        raise NotImplementedError("{} op not implemented".format(source_node.type))

    def rename_INPUT(self, source_node):
        IR_node = self.IR_graph.node.add()
        IR_node.name = source_node.name
        IR_node.op = "tflite_DataInput"
        kwargs = {}
        kwargs['data_format'] = 'NHWC'
        kwargs['output_scale'] = source_node.get_attr('output_scales')[0]
        kwargs['output_zero_point'] = source_node.get_attr('output_zero_points')[0]
        assign_IRnode_values(IR_node, kwargs)
        output_list = source_node.layer['shape']
        shape = graph_pb2.TensorShape()
        for dim in output_list:
            new_dim = shape.dim.add()
            if dim == None:
                new_dim.size = -1
            else:
                new_dim.size = dim
        IR_node.attr["_output_shapes"].list.shape.extend([shape])
        self._copy_shape(IR_node,output_list) 
        self.IR_layer_map[IR_node.name] = IR_node

    def rename_CONV_2D(self, source_node):
        weights_list = self.get_weights_list(source_node)
        IR_node = self._convert_identity_operation(source_node, new_op='tflite_Conv')
        kwargs = {}
        kwargs['is_tflite'] = True
        kwargs['strides'] = [1] + [source_node.get_attr('attributes')['stride_h']] + [source_node.get_attr('attributes')['stride_w']] + [1]
        kwargs['dilations']= [1] + [source_node.get_attr('attributes')['dilation_h_factor']] + [source_node.get_attr('attributes')['dilation_w_factor']] + [1]
        kwargs['kernel_shape'] = list(weights_list[0].shape)[1:-1] + [weights_list[0].shape[-1]] + [weights_list[0].shape[0]]
        kwargs['input_scale'] = source_node.get_attr('input_scales')[0]
        kwargs['output_scale'] = source_node.get_attr('output_scales')[0][0]
        kwargs['input_zero_point'] = source_node.get_attr('input_zero_points')[0]
        kwargs['output_zero_point'] = source_node.get_attr('output_zero_points')[0][0]
        kwargs['fused_activation_type'] = str(source_node.get_attr('attributes')['fused_activation_function']).lower()
        self._convert_padding(source_node, IR_node, kwargs['kernel_shape'])
        assign_IRnode_values(IR_node, kwargs)
        # weights
        #tflite weights are in d,h,w,c format. Transpose it to h,w,c,d for IR to consume.
        self.set_weight(source_node.name, "weights", weights_list[0].transpose(1,2,3,0))
        # bias
        IR_node.attr["use_bias"].b = True
        self.set_weight(source_node.name, "bias", weights_list[1])
        # weight scales
        self.set_weight(source_node.name, "weight_scales", source_node.get_attr('input_scales')[1])
        # weight zero points
        self.set_weight(source_node.name, "weight_zero_points", source_node.get_attr('input_zero_points')[1])
    
    def rename_DEPTHWISE_CONV_2D(self, source_node):
        weights_list = self.get_weights_list(source_node)
        IR_node = self._convert_identity_operation(source_node, new_op='tflite_DepthwiseConv')
        input_node = self.src_graph.get_parent(source_node.name, [0])
        input_node_shape = self.get_output_shape(input_node)
        output_node_shape = self.get_output_shape(source_node)
        input_channels = input_node_shape[-1]
        output_channels = output_node_shape[-1]
        depth_multiplier = int(output_channels / input_channels)
        weights = weights_list[0]
        weights = weights.reshape(list(weights.shape[1:-1])+[input_channels, depth_multiplier])
        kwargs = {}
        kwargs['is_tflite'] = True
        kwargs['strides'] = [1] + [source_node.get_attr('attributes')['stride_h']] + [source_node.get_attr('attributes')['stride_w']] + [1]
        kwargs['dilations']= [1] + [source_node.get_attr('attributes')['dilation_h_factor']] + [source_node.get_attr('attributes')['dilation_w_factor']] + [1]
        kwargs['kernel_shape'] = list(weights.shape)
        kwargs['input_scale'] = source_node.get_attr('input_scales')[0]
        kwargs['output_scale'] = source_node.get_attr('output_scales')[0][0]
        kwargs['input_zero_point'] = source_node.get_attr('input_zero_points')[0]
        kwargs['output_zero_point'] = source_node.get_attr('output_zero_points')[0][0]
        kwargs['fused_activation_type'] = str(source_node.get_attr('attributes')['fused_activation_function']).lower()
        self._convert_padding(source_node, IR_node, kwargs['kernel_shape'])
        assign_IRnode_values(IR_node, kwargs)
        # weights
        #tflite weights are in d,h,w,c format. Transpose it to h,w,c,d for IR to consume.
        self.set_weight(source_node.name, "weights", weights)
        # bias
        IR_node.attr["use_bias"].b = True
        self.set_weight(source_node.name, "bias", weights_list[1])
        # weight scales
        self.set_weight(source_node.name, "weight_scales", source_node.get_attr('input_scales')[1])
        # weight zero points
        self.set_weight(source_node.name, "weight_zero_points", source_node.get_attr('input_zero_points')[1])
    
    def rename_ADD(self, source_node):
        weights_list = self.get_weights_list(source_node)
        IR_node = self._convert_identity_operation(source_node, new_op = "tflite_Add")
        kwargs = {}
        kwargs['is_tflite'] = True
        kwargs['output_scale'] = source_node.get_attr('output_scales')[0][0]
        kwargs['output_zero_point'] = source_node.get_attr('output_zero_points')[0][0]
        kwargs['input_scale'] = [sc[0] for sc in source_node.get_attr('input_scales')]
        kwargs['input_zero_point'] = [sc[0] for sc in source_node.get_attr('input_zero_points')]
        kwargs['fused_activation_type'] = str(source_node.get_attr('attributes')['fused_activation_function']).lower()
        # One of the input is a constant tensor
        if len(weights_list) > 0:
            kwargs['const_tensor'] = True
            input_node = self.src_graph.get_parent(source_node.name, [0])
            input_shape = tuple(self.get_output_shape(input_node))
            const_input = weights_list[0].reshape([1,weights_list[0].size,1,1])
            # const_input = np.broadcast_to(const_input, input_shape).transpose([0,3,1,2])
            self.set_weight(source_node.name, "weights", const_input)
        assign_IRnode_values(IR_node, kwargs)

    def rename_MUL(self, source_node):
        weights_list = self.get_weights_list(source_node)
        IR_node = self._convert_identity_operation(source_node, new_op = "tflite_Mul")
        kwargs = {}
        kwargs['is_tflite'] = True
        kwargs['output_scale'] = source_node.get_attr('output_scales')[0][0]
        kwargs['output_zero_point'] = source_node.get_attr('output_zero_points')[0][0]
        kwargs['input_scale'] = [sc[0] for sc in source_node.get_attr('input_scales')]
        kwargs['input_zero_point'] = [sc[0] for sc in source_node.get_attr('input_zero_points')]
        kwargs['fused_activation_type'] = str(source_node.get_attr('attributes')['fused_activation_function']).lower()
        if len(weights_list) == 1 and weights_list[0].size == 1:
            kwargs['const_input'] = int(weights_list[0])
            op_shape = tuple(self.get_output_shape(source_node))
            self.set_weight(source_node.name, "weights", np.full(op_shape, int(weights_list[0])).transpose([0,3,1,2]))
        kwargs['const_tensor'] = True if 'const_input' in kwargs else False
        assign_IRnode_values(IR_node, kwargs)

    def rename_LOGISTIC(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = "tflite_Sigmoid")
        kwargs = {}
        kwargs['is_tflite'] = True
        kwargs['output_scale'] = source_node.get_attr('output_scales')[0][0]
        kwargs['output_zero_point'] = source_node.get_attr('output_zero_points')[0][0]
        kwargs['input_scale'] = [sc[0] for sc in source_node.get_attr('input_scales')]
        kwargs['input_zero_point'] = [sc[0] for sc in source_node.get_attr('input_zero_points')]
        assign_IRnode_values(IR_node, kwargs)

    def rename_SWISH(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = "tflite_Swish")
        kwargs = {}
        kwargs['is_tflite'] = True
        kwargs['output_scale'] = source_node.get_attr('output_scales')[0][0]
        kwargs['output_zero_point'] = source_node.get_attr('output_zero_points')[0][0]
        kwargs['input_scale'] = [sc[0] for sc in source_node.get_attr('input_scales')]
        kwargs['input_zero_point'] = [sc[0] for sc in source_node.get_attr('input_zero_points')]
        assign_IRnode_values(IR_node, kwargs)

    def rename_AVERAGE_POOL_2D(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='tflite_Pool')
        kwargs = {}
        kwargs['is_tflite'] = True
        kwargs['strides'] = [1,source_node.get_attr('attributes').get('stride_h'), source_node.get_attr('attributes').get('stride_w'),1]
        kwargs['kernel_shape'] = [1,source_node.get_attr('attributes').get('filter_height'), source_node.get_attr('attributes').get('filter_width'),1]
        kwargs['pooling_type'] = 'AVG'
        self._convert_padding(source_node, IR_node, kwargs['kernel_shape'][1:-1])
        kwargs['output_scale'] = source_node.get_attr('output_scales')[0][0]
        kwargs['output_zero_point'] = source_node.get_attr('output_zero_points')[0][0]
        kwargs['input_scale'] = [sc[0] for sc in source_node.get_attr('input_scales')]
        kwargs['input_zero_point'] = [sc[0] for sc in source_node.get_attr('input_zero_points')]
        assign_IRnode_values(IR_node, kwargs)
    
    def rename_MAX_POOL_2D(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op='tflite_Pool')
        kwargs = {}
        kwargs['is_tflite'] = True
        kwargs['strides'] = [1,source_node.get_attr('attributes').get('stride_h'), source_node.get_attr('attributes').get('stride_w'),1]
        kwargs['kernel_shape'] = [1,source_node.get_attr('attributes').get('filter_height'), source_node.get_attr('attributes').get('filter_width'),1]
        kwargs['pooling_type'] = 'MAX'
        self._convert_padding(source_node, IR_node, kwargs['kernel_shape'][1:-1])
        kwargs['output_scale'] = source_node.get_attr('output_scales')[0][0]
        kwargs['output_zero_point'] = source_node.get_attr('output_zero_points')[0][0]
        kwargs['input_scale'] = [sc[0] for sc in source_node.get_attr('input_scales')]
        kwargs['input_zero_point'] = [sc[0] for sc in source_node.get_attr('input_zero_points')]
        assign_IRnode_values(IR_node, kwargs)
    
    def rename_PAD(self, source_node):
        weights_list = self.get_weights_list(source_node)
        IR_node = self._convert_identity_operation(source_node, new_op = 'tflite_Pad')
        kwargs = {}
        kwargs['is_tflite'] = True
        kwargs['mode'] = 'constant'

        # paddings
        padding = weights_list[0].tolist()
        kwargs['output_zero_point'] = source_node.get_attr('output_zero_points')[0][0]
        shapes = np.array(padding)
        kwargs['pads'] = convert_tf_pad_to_onnx(shapes)

        assign_IRnode_values(IR_node, kwargs)
    
    def rename_MEAN(self, source_node):
        weights_list = self.get_weights_list(source_node)
        input_node = self.src_graph.get_parent(source_node.name, [0])
        input_node_shape = self.get_output_shape(input_node)
        # import pdb;pdb.set_trace()
        # axes = np.array(self.get_parent(source_node.name, [1]).layer['value'])
        axes = weights_list[0].tolist()
        if np.all(axes == list(range(1,len(input_node_shape)-1))):
            IR_node = self._convert_identity_operation(source_node, new_op='tflite_PoolReduce')
            kwargs = {}
            kwargs['is_tflite'] = True
            kwargs['strides'] = [1]*len(input_node_shape)
            kwargs['kernel_shape'] = [1]*len(input_node_shape)
            kwargs['pooling_type'] = 'AVG'
            kwargs['auto_pad'] = "VALID"
            kwargs['pads'] = [0, 0] * len(input_node_shape)
            kwargs['output_scale'] = source_node.get_attr('output_scales')[0][0]
            kwargs['output_zero_point'] = source_node.get_attr('output_zero_points')[0][0]
            kwargs['input_scale'] = [sc[0] for sc in source_node.get_attr('input_scales')]
            kwargs['input_zero_point'] = [sc[0] for sc in source_node.get_attr('input_zero_points')]
            IR_node.attr['keepdims'].b = source_node.layer['attributes']['keep_dims']
            assign_IRnode_values(IR_node, kwargs)
        else:
            IR_node = self._convert_identity_operation(source_node, new_op='Tf_Mean')
            IR_node.attr['keep_dims'].b = source_node.layer['attributes']['keep_dims']
            IR_node.attr['axis'].list.i.extend(axes)

    def rename_FULLY_CONNECTED(self, source_node):
        weights_list = self.get_weights_list(source_node)
        IR_node = self._convert_identity_operation(source_node, new_op='tflite_FullyConnected')
        units = self.get_output_shape(source_node)[-1]
        kwargs = {}
        kwargs['is_tflite'] = True
        kwargs['input_scale'] = source_node.get_attr('input_scales')[0]
        kwargs['output_scale'] = source_node.get_attr('output_scales')[0][0]
        kwargs['input_zero_point'] = source_node.get_attr('input_zero_points')[0]
        kwargs['output_zero_point'] = source_node.get_attr('output_zero_points')[0][0]
        kwargs['fused_activation_type'] = str(source_node.get_attr('attributes')['fused_activation_function']).lower()
        assign_IRnode_values(IR_node, kwargs)
        # weights
        #tflite weights are in d,h,w,c format. Transpose it to h,w,c,d for IR to consume.
        self.set_weight(source_node.name, "weights", weights_list[0].T)
        IR_node.attr['units'].i = units
        IR_node.attr["use_bias"].b = True
        # bias
        if len(weights_list) == 2:
            self.set_weight(source_node.name, "bias", weights_list[1])
        else:
            self.set_weight(source_node.name, "bias", [0]*units)
        # weight scales
        self.set_weight(source_node.name, "weight_scales", source_node.get_attr('input_scales')[1])
        # weight zero points
        self.set_weight(source_node.name, "weight_zero_points", source_node.get_attr('input_zero_points')[1])


    def rename_SOFTMAX(self, source_node):
        len_ = len(self.get_output_shape(source_node))
        axis = source_node.layer['attributes'].get('axis', len_)
        IR_node = self._convert_identity_operation(source_node, new_op="tflite_Softmax")
        kwargs = {}
        kwargs['is_tflite'] = True
        kwargs['input_scale'] = source_node.get_attr('input_scales')[0]
        kwargs['output_scale'] = source_node.get_attr('output_scales')[0][0]
        kwargs['input_zero_point'] = source_node.get_attr('input_zero_points')[0]
        kwargs['output_zero_point'] = source_node.get_attr('output_zero_points')[0][0]
        assign_IRnode_values(IR_node, kwargs)
        IR_node.attr['axis'].i = axis-1
    
    def rename_RESHAPE(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op="Reshape")
        kwargs = {}
        new_shape = self.get_output_shape(source_node)
        data_format = self.get_tflite_format(new_shape)
        kwargs['shape'] = new_shape
        kwargs['data_format'] = data_format
        assign_IRnode_values(IR_node, kwargs)
    
    def rename_TRANSPOSE(self, source_node):
        weights_list = self.get_weights_list(source_node)
        IR_node = self._convert_identity_operation(source_node, new_op="Transpose")
        kwargs = {}
        new_shape = self.get_output_shape(source_node)
        data_format = self.get_tflite_format(new_shape)
        kwargs['data_format'] = data_format
        kwargs['perm'] = weights_list[0]
        assign_IRnode_values(IR_node, kwargs)
    
    def rename_SPLIT_V(self, source_node):
        weights_list = self.get_weights_list(source_node)
        axis = int(weights_list[1])
        size_splits = weights_list[0].tolist()
        num_splits = source_node.get_attr('attributes').get('num_splits')
        if num_splits == 1:
            source_node.real_name = self.src_graph.get_parent(source_node.name, [1]).real_name
        else:
            IR_node = self._convert_identity_operation(source_node, start_idx=0, new_op = 'Tf_SplitV')
            kwargs = {
                'axis' : axis,
                'split' : num_splits,
                'split_size': size_splits
            }
            assign_IRnode_values(IR_node, kwargs)
            for i in range(0,kwargs['split']):
                base_name = source_node.name
                IR_node1 = self._convert_identity_operation(source_node, start_idx=0, new_op = 'Tf_SplitV')
                out1 = base_name + '_out' + str(i)
                IR_node1.name = out1
                IR_node1.attr["ignore_node"].b = True

    def rename_LAYERNORM_H(self,source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = "tflite_LayerNorm_H")
        kwargs = {}
        kwargs['is_tflite'] = True
        kwargs['output_scales'] = source_node.get_attr('output_scales')[0][0]
        kwargs['output_zero_points'] = source_node.get_attr('output_zero_points')[0][0]
        kwargs['input_scales'] =  source_node.get_attr('input_scales')[0]
        kwargs['input_zero_points'] = source_node.get_attr('input_zero_points')[0]
        kwargs['mean_1_axis'] = source_node.get_attr('mean_1_axis')[0]
        kwargs['mean_1_output_scales'] = source_node.get_attr('mean_1_output_scales')[0][0]
        kwargs['mean_1_output_zero_points'] = source_node.get_attr('mean_1_output_zero_points')[0][0]
        kwargs['squared_difference_output_zero_points'] = source_node.get_attr('squared_difference_output_zero_points')[0][0]
        kwargs['squared_difference_output_scales'] = source_node.get_attr('squared_difference_output_scales')[0][0]
        kwargs['mean_2_axis'] = source_node.get_attr('mean_2_axis')[0]
        kwargs['mean_2_output_scales'] = source_node.get_attr('mean_2_output_scales')[0][0]
        kwargs['mean_2_output_zero_points'] = source_node.get_attr('mean_2_output_zero_points')[0][0]
        kwargs['add_1_output_scales'] = source_node.get_attr('add_1_output_scales')[0][0]
        kwargs['add_1_output_zero_points'] =  source_node.get_attr('add_1_output_zero_points')[0][0]
        kwargs['add_1_constant'] = source_node.get_attr('add_1_constant').item()
        kwargs['add_1_constant_scale'] = source_node.get_attr('add_1_constant_scale')[0]
        kwargs['add_1_constant_zero_points'] = source_node.get_attr('add_1_constant_zero_points')[0]
        kwargs['rsqrt_output_scales'] =  source_node.get_attr('rsqrt_output_scales')[0][0]
        kwargs['rsqrt_output_zero_points'] = source_node.get_attr('rsqrt_output_zero_points')[0][0]
        kwargs['mul_1_output_scales'] = source_node.get_attr('mul_1_output_scales')[0][0]
        kwargs['mul_1_output_zero_points'] = source_node.get_attr('mul_1_output_zero_points')[0][0]
        kwargs['mul_1_constant_scale'] = source_node.get_attr('mul_1_constant_scale')[0]
        kwargs['mul_1_constant_zero_points'] = source_node.get_attr('mul_1_constant_zero_points')[0]
        kwargs['mul_2_output_scales'] = source_node.get_attr('mul_2_output_scales')[0][0]
        kwargs['mul_2_output_zero_points'] = source_node.get_attr('mul_2_output_zero_points')[0][0]
        kwargs['mul_3_output_scales'] = source_node.get_attr('mul_3_output_scales')[0][0]
        kwargs['mul_3_output_zero_points'] = source_node.get_attr('mul_3_output_zero_points')[0][0]
        kwargs['sub_1_output_scales'] = source_node.get_attr('sub_1_output_scales')[0][0]
        kwargs['sub_1_output_zero_points'] = source_node.get_attr('sub_1_output_zero_points')[0][0]
        kwargs['sub_1_constant_scale'] = source_node.get_attr('sub_1_constant_scale')[0]
        kwargs['sub_1_constant_zero_points'] = source_node.get_attr('sub_1_constant_zero_points')[0]
        kwargs['add_2_output_scales'] = source_node.get_attr('add_2_output_scales')[0][0]
        kwargs['add_2_output_zero_points'] = source_node.get_attr('add_2_output_zero_points')[0][0]

        assign_IRnode_values(IR_node, kwargs)
        #changing weights for nhwc in TFlite to nchw in caffe
        weights = self.get_lnrm_array(np.array(source_node.get_attr('mul_1_constant')), 1)
        bias = self.get_lnrm_array(np.array(source_node.get_attr('sub_1_constant')), 1)

        self.set_weight(source_node.name, "weights", weights.transpose(0,3,1,2))
        self.set_weight(source_node.name, "bias", bias.transpose(0,3,1,2))
    
    def rename_BATCH_MATMUL(self, source_node):
        IR_node = self._convert_identity_operation(source_node, new_op = "tflite_BatchDot")
        input_node1 = self.src_graph.get_parent(source_node.name, [0])
        input_node2 = self.src_graph.get_parent(source_node.name, [1])
        kwargs = {}
        kwargs['is_tflite'] = True
        kwargs['input_scale'] = [sc[0] for sc in source_node.get_attr('input_scales')]
        kwargs['output_scale'] = source_node.get_attr('output_scales')[0][0]
        kwargs['input_zero_point'] = [sc[0] for sc in source_node.get_attr('input_zero_points')]
        kwargs['output_zero_point'] = source_node.get_attr('output_zero_points')[0][0]
        kwargs['const_tensor'] = False
        kwargs['transpose_a'] = source_node.get_attr('attributes').get('adj_x')
        kwargs['transpose_b'] = source_node.get_attr('attributes').get('adj_y')
        kwargs['data_format'] = "NHWC"
        assign_IRnode_values(IR_node, kwargs)
    
    def rename_CONCATENATION(self, source_node):
        n = len(source_node.in_edges)
        IR_node = self._convert_identity_operation(source_node, start_idx=0, end_idx=n, new_op='Concat')
        axis = source_node.get_attr('attributes').get('axis')
        if axis < 0:
            axis += len(self.get_output_shape(source_node))
        IR_node.attr["axis"].i = axis

    def rename_STRIDED_SLICE(self, source_node):
        pass

    def rename_SHAPE(self, source_node):
        pass

    def rename_PACK(self, source_node):
        pass
