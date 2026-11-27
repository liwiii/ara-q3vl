'''
Portions of this file are derived from https://github.com/microsoft/MMdnn and are subject to the original MIT License

Copyright (c) Microsoft Corporation. All rights reserved.
SPDX-License-Identifier: MIT

Copyright (c) 2018 - 2025, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Modifications:

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

from __future__ import division
import os
import sys
import math
import numpy as np
# import mxnet as mx
import copy
from packaging import version
import caffe
from caffe import layers as L
from caffe import params as P
from mmdnn.conversion.common.IR.IR_graph import IRGraph, IRGraphNode
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.IR.graph_pb2 import NodeDef, GraphDef, DataType
from mmdnn.conversion.common.DataStructure.emitter import Emitter
from mmdnn.conversion.common.utils import *
from mmdnn.conversion.common.types import UPSP_TYPE, fusedActivation
import warnings
import pandas as pd
from dv_logging import get_logger
from copy import deepcopy
from mmdnn.conversion._script.dump_code import dump_code
import re

NDHWC_NCDHW = {0: 0, 1: 2, 2: 3, 3: 4, 4: 1}
NHWC_NCHW = {0: 0, 1: 2, 2: 3, 3: 1, -1: 1, -2: 3, -3: 2, -4: 0}
NWC_NCW = {0: 0, 1: 2, 2: 1}
logger = get_logger(__name__, 'DVEMIT')


def eval(input_node_shape, kpads, weights):
  dim = len(weights.shape)
  for i in range(dim - 2):
    if input_node_shape[i + 1] + kpads[i] > weights.shape[i]:
      return False
  return True


def clast2first(shape):
  if len(shape) > 1:
    return [shape[0], shape[-1]] + shape[1:-1]
  else:
    return shape


def sanitize_name(name):
    # Match a leading number followed by underscore
    match = re.match(r'^(\d+)_([A-Za-z_].*)$', name)
    if match:
        number, rest = match.groups()
        return f"{rest}_{number}"
    return name # if name is "380_DequantizeLinear, then new name will be DequantizeLinear_380"

def load_csv(CSV_FILE_PATH):
    try:
        name_mapping = pd.read_csv(CSV_FILE_PATH, sep=',')
    # removing this FileNotFoundError, because for frameworks other than onnx, we don't dump layer_name_map.csv
    except FileNotFoundError:
        logger.info("layer_name_map.csv doesn't exist")
        return None
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: '{CSV_FILE_PATH}'")
    except pd.errors.ParserError:
        raise ValueError(f"CSV file is malformed/unparseable: '{CSV_FILE_PATH}'")
    name_mapping.columns = ['index','Src_Name', 'Intermediate_name', 'ARANet_name', 'SrcType', 'ARANetType']
    name_mapping["ARANetType"] = name_mapping["ARANetType"].astype("object")
    return name_mapping

class CaffeEmitter(Emitter):

    def __init__(self, model, dummy, summary_csv, aten_debug, output_dir, plt_target):
        from six import string_types as _string_types
        super(CaffeEmitter, self).__init__()
        if isinstance(model, _string_types):
            network_path = model
        else:
            network_path = model[0]
            self._load_weights(model[1])
        self.IR_graph = IRGraph(network_path)
        self.dummy = dummy
        self.summary_csv = summary_csv
        self.aten_debug = aten_debug
        self.output_dir = os.getcwd()
        self.plt_target = plt_target
        self.csv_file_path = os.path.join(os.path.dirname(network_path), "stats/layer_name_map.csv")
        self.name_map_df = load_csv(self.csv_file_path)
        super(CaffeEmitter, self)._build()

    @property
    def header_code(self):
        return """from __future__ import print_function
import numpy as np
import sys, argparse
import caffe
import traceback
from caffe import layers as L
from caffe import params as P
from caffe import to_proto
from six import text_type as _text_type
from dv_logging import get_logger
logger = get_logger(__name__, 'DVPROTO')
__weights_dict = dict()
def load_weights(weight_file):
    if weight_file == None:
        return
    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes', allow_pickle=True).item()
    return weights_dict
def KitModel(weight_file = None):
    n = caffe.NetSpec()
"""

    @property
    def end_code(self):
        return """    return n
def make_net(prototxt):
    n = KitModel()
    with open(prototxt, 'w') as fpb:
        proto = n.to_proto()
        print(proto, file=fpb)
def gen_weight(weight_file, model, prototxt):
    global __weights_dict
    __weights_dict = load_weights(weight_file)
    try:
        net = caffe.Net(prototxt, caffe.TEST)
    except Exception as e:
        logger.error("Exception occured while loading the model %s",str(e))
        return False

    for key in __weights_dict:
        # RNN has specific order of blob placement
        # i2h_w -> i2h_b -> h2h_w -> h2h_b
        if isinstance(__weights_dict[key], list):
            ind = 0
            for blob in __weights_dict[key]:
                net.params[key][ind].data.flat = blob
                ind = ind + 1
        else:

            if 'weights' in __weights_dict[key]:
                net.params[key][0].data.flat = __weights_dict[key]['weights']
            elif 'mean' in __weights_dict[key]:
                net.params[key][0].data.flat = __weights_dict[key]['mean']
                net.params[key][1].data.flat = __weights_dict[key]['var']
                if 'scale' in __weights_dict[key]:
                    net.params[key][2].data.flat = __weights_dict[key]['scale']
                if 'bias' in __weights_dict[key]:
                    net.params[key][3].data.flat = __weights_dict[key]['bias']
                    __weights_dict[key].pop('bias')
            elif 'scale' in __weights_dict[key]:
                net.params[key][0].data.flat = __weights_dict[key]['scale']
            if 'bias' in __weights_dict[key]:
                net.params[key][1].data.flat = __weights_dict[key]['bias']
                if 'weight_scales' in __weights_dict[key]:
                    net.params[key][2].data.flat = __weights_dict[key]['weight_scales']
                    net.params[key][3].data.flat = __weights_dict[key]['weight_zero_points']
            else:
                if 'weight_scales' in __weights_dict[key]:
                    net.params[key][1].data.flat = __weights_dict[key]['weight_scales']
                    net.params[key][2].data.flat = __weights_dict[key]['weight_zero_points']
            if 'gamma' in __weights_dict[key]: # used for prelu, not sure if other layers use this too
                net.params[key][0].data.flat = __weights_dict[key]['gamma']

            if 'widths' in __weights_dict[key]:
                net.params[key][0].data.flat = __weights_dict[key]['widths']
            if 'heights' in __weights_dict[key]:
                net.params[key][1].data.flat = __weights_dict[key]['heights']
            if 'ctr_x' in __weights_dict[key]:
                net.params[key][2].data.flat = __weights_dict[key]['ctr_x']
            if 'ctr_y' in __weights_dict[key]:
                net.params[key][3].data.flat = __weights_dict[key]['ctr_y']
    total = 0
    for i in net.params:
        for j in net.params[i]:
            total += j.data.size*32
    if total > 2*(1024*1024*1024):
        net.save_hdf5(model)
    else:
        net.save(model)
    return True
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Generate caffe model and prototxt')
    parser.add_argument('--weight_file', '-w', type=_text_type, default='IR weight file')
    parser.add_argument('--prototxt', '-p', type=_text_type, default='caffe_converted.prototxt')
    parser.add_argument('--model', '-m', type=_text_type, default='caffe_converted.caffemodel')
    args = parser.parse_args()
    # For some reason argparser gives us unicode, so we need to conver to str first
    try:
        make_net(str(args.prototxt))
        if(gen_weight(str(args.weight_file), str(args.model), str(args.prototxt))):
          sys.exit(0)
        sys.exit(1)
    except Exception as e:
        logger.error("Exception while running generated code %s",str(e))
        sys.exit(1)
"""

    def _convert_axis_cfirst(self, IR_node, axis):
        input_node_name = self.parent_variable_name(IR_node)
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        input_node_shape = shape_to_list(input_node.get_attr("_output_shapes")[0])
        if axis == len(input_node_shape) - 1:
            axis = 1
        elif axis > 0 and len(input_node_shape) > 2:
            axis = axis + 1
        return axis

    def gen_code(self, phase='test'):
        self.phase = phase
        self.add_body(0, self.header_code)
        train_layers = ["LinearRegressionOutput", "Cast", "cast"]
        df = pd.DataFrame(columns=['origin', 'op', 'issue', 'source'])
        for i in self.IR_graph.layer_map:
            if self.IR_graph.layer_map[i].type in train_layers:
                msg = "Ignoring train layer: " + self.IR_graph.layer_map[i].type
                logger.warn(msg)
                layer_name = self.IR_graph.layer_map[i].name
                outs = self.IR_graph.layer_map[i].out_edges
                ins = self.IR_graph.layer_map[i].in_edges
                if len(ins) != 1:
                    raise ValueError("Input to training layers must be one")
                for j in outs:
                    for k, l in enumerate(self.IR_graph.layer_map[j].in_edges):
                        if l == layer_name:
                            self.IR_graph.layer_map[j].in_edges[k] = ins[0]
                self.IR_graph.layer_map[ins[0]].out_edges = list(filter((layer_name).__ne__, self.IR_graph.layer_map[ins[0]].out_edges))
                self.IR_graph.layer_map[ins[0]].out_edges.extend(outs)
        currWeightDict = {}
        idx = -1
        prev_body_code = ""
        map_to_aten = False
        index_list = []
        while idx + 1 < len(self.IR_graph.topological_sort):
            idx = idx + 1
            index_list.append(idx)
            len_index_list = len(index_list)
            if len_index_list > 2 and index_list[len_index_list - 1] == index_list[len_index_list - 3]:
                raise Exception("All flows for converting to DVNET have failed")
            layer = self.IR_graph.topological_sort[idx]
            current_node = self.IR_graph.get_node(layer)
            logger.debug("emitting layer type %s layer_name: %s", current_node.type, current_node.name)
            node_type = current_node.type
            if current_node.get_attr('ignore_node') or node_type in ["onnxpqn_QuantizeLinear", "onnxpqn_DequantizeLinear"]:
                continue
            # if current_node.name == 'Transpose__753':
            #   logger.info("breaking the emitter at node: %s", current_node.name)
            #    break
            if node_type == "Relu" and len(current_node.out_edges) == 1 and self.IR_graph.get_node(current_node.out_edges[0]).type == "clip":
                clip_node = self.IR_graph.get_node(current_node.out_edges[0])
                if clip_node.get_attr('a_min') == 0.0 and clip_node.get_attr('a_max') == 6.0:
                    node_type = "Relu6"
                else:
                    raise NotImplementedError("clip can only have min 0 and max 6 (relu6)")
            if node_type == "clip":
                if len(current_node.in_edges) == 1 and self.IR_graph.get_node(current_node.in_edges[0]).type == "Relu":
                    continue
            if hasattr(self, "emit_" + node_type) and not map_to_aten:
                func = getattr(self, "emit_" + node_type)
                try:
                    func(current_node)
                except NotImplementedError as ne:
                    if self.dummy:
                        self.emit_ATen_emitter(current_node)
                        row = {}
                        row['op'] = current_node.name
                        row['issue'] = str(ne)
                        row['source'] = current_node.layer
                        row['origin'] = 'Caffe emitter'
                        df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
                    else:
                        raise NotImplementedError()
                except Exception as e:
                    if self.dummy:
                        self.emit_ATen_emitter(current_node)
                    logger.exception("error at node name: %s node type: %s error msg: %s node info: %s", current_node.name, node_type, e, current_node.layer)
                    row = {}
                    row['op'] = current_node.name
                    row['issue'] = str(e)
                    row['source'] = current_node.layer
                    row['origin'] = 'Caffe emitter'
                    df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
            elif node_type not in train_layers:
                logger.warn("CaffeEmitter does not support [%s] yet. node: %s. layer: %s", node_type, current_node.layer, layer)
                if self.dummy:
                    map_to_aten = False
                    self.emit_ATen_emitter(current_node)
                    row = {}
                    row['op'] = current_node.name
                    row['issue'] = "Unsupported op : '" + current_node.type + "' , replaced with DV_ATen"
                    # row['issue'] = "Replacing with DV_ATen op as node type: '" + current_node.type + "' is not supported in CaffeEmitter"
                    row['source'] = current_node.layer
                    row['origin'] = 'Caffe emitter'
                    df = pd.concat([df, pd.DataFrame(row, index=[0])], ignore_index=True)
                else:
                    self.emit_UNKNOWN(current_node)
            else:
                logger.warn("CaffeEmitter ignoring [%s]", node_type)
            if self.aten_debug:
                temp_body_code = prev_body_code
                curr_body_code = self.body_code
                self.add_body(0, "")
                self.add_body(0, self.end_code)

                if current_node.name in currWeightDict:
                    # currWeightDict[current_node.name] = self.weights_dict[current_node.name]
                    currWeightDict.pop(current_node.name)
                if not self.save_code_for_aten_debug(f"{self.output_dir}/debug/aten_test_{idx}", currWeightDict):
                    idx = idx - 1
                    map_to_aten = True
                    self.body_code = temp_body_code
                else:
                    self.body_code = curr_body_code
                    prev_body_code = curr_body_code
        df.to_csv(self.summary_csv, mode='a', index=False, header=False)
        if self.name_map_df is not None:
            self.name_map_df.to_csv(self.csv_file_path, index=False)
        self.add_body(0, "")
        self.add_body(0, self.end_code)

        return self.body_code

    def save_code_for_aten_debug(self, filepath, currWeightDict):
        pyfile = filepath + ".py"
        npyfile = filepath + ".npy"
        if not os.path.exists(os.path.join(self.output_dir, "debug")):
            os.makedirs(os.path.join(self.output_dir, "debug"))
        if not os.path.exists(os.path.join(self.output_dir, "debug/params")):
            os.makedirs(os.path.join(self.output_dir, "debug/params"))
        with open(os.path.join(self.output_dir, pyfile), 'w+') as fout:
            fout.write(self.body_code)
        self.save_weights({}, npyfile)

        if dump_code('caffe', filepath, npyfile, os.path.join(self.output_dir, "debug/params/aten_param"), None) == 0:
            os.remove(pyfile)
            os.remove(npyfile)
            return True
        else:
            os.rename(pyfile, filepath + "_fail.py")
            os.rename(npyfile, filepath + "_fail.npy")
        return False
        # logger.info("target network code snippet is saved as [{}]".format(filepath))

    def run(self, dstNetworkPath, dstWeightPath=None, phase='test'):
        super(CaffeEmitter, self).run(dstNetworkPath, dstWeightPath, phase)
        if self.weight_loaded:
            self.save_weights(self.weights_dict, dstWeightPath)

    @staticmethod
    def _convert_list_to_string(list):
        return [str(i) for i in list]

    @staticmethod
    def _shapeToStr(shapes):
        return [dim.size if dim.size > 0 else 1 for dim in shapes.dim]

    # def _get_stride(self,IR_node, field):
    #     if len(IR_node.get_attr('strides')) == 4:
    #         stride_d = 1
    #         stride_h = IR_node.get_attr('strides')[1]
    #         stride_w = IR_node.get_attr('strides')[2]
    #     elif len(IR_node.get_attr('strides')) == 5:
    #         stride_d = IR_node.get_attr('strides')[1]
    #         stride_h = IR_node.get_attr('strides')[2]
    #         stride_w = IR_node.get_attr('strides')[3]
    #     elif len(IR_node.get_attr('strides')) == 2:
    #         stride_d = 1
    #         stride_h = IR_node.get_attr('strides')[0]
    #         stride_w = IR_node.get_attr('strides')[1]
    #     return stride_d, stride_h, stride_w
    def _get_kernel_field(self, IR_node, field):
        kernel_d = 1
        kernel_h = 1
        kernel_w = 1
        kernel_c = 1
        kernel_o = 1
        default_values = [1] * 5
        if IR_node.get_attr(field) is None:
            return default_values
        if len(IR_node.get_attr(field)) == 5:
            return IR_node.get_attr(field)
        else:
            return [1] * (5 - len(IR_node.get_attr(field))) + IR_node.get_attr(field)

    def _get_symmetric_padding(self, IR_node):
        strides = IR_node.get_attr('strides')
        # check if have pad layer
        IR_parent_node = self.IR_graph.get_parent(IR_node.name, [0])
        pads = IR_node.get_attr('pads')
        # Pad_h < kernel_h (vgg19 caffe2caffe)
        sym_pad = []
        half_length = len(pads) // 2
        if IR_node.type == "Conv2DBackpropInput":
            for i in range(1, half_length - 1):
                if pads[i] > 0:
                    sym_pad.append(pads[i])
                else:
                    sym_pad.append(0)
        else:
            for i in range(1, half_length - 1):
                if pads[i]:
                    sym_pad.append(pads[i] + (0 if pads[i] == pads[i + half_length] else strides[i]))
                else:
                    sym_pad.append(0)
        return sym_pad

    @staticmethod
    def convert_nhwc_nchw(shape):
        return [shape[0]] + [shape[-1]] + shape[1:-1]

    def check_if_need_transpose(self, IR_node):
        parent = self.IR_graph.get_parent(IR_node.name, [0])
        assert parent is not None
        while parent.type == 'Flatten' or parent.type == 'Dropout' or parent.type == 'Reshape':
            if parent.type == 'Reshape' and parent.get_attr('data_format') in ['NHWC', 'NWC', 'NC']:
                return
            parent = self.IR_graph.get_parent(parent.name, [0])
            assert parent is not None
        dim = len(parent.layer.attr['_output_shapes'].list.shape[0].dim)
        if dim > 2:
            original_dims = self.weights_dict[IR_node.name]['weights'].shape
            dims = [i.size for i in parent.layer.attr['_output_shapes'].list.shape[0].dim[1:]] + [-1]
            self.weights_dict[IR_node.name]['weights'] = np.reshape(self.weights_dict[IR_node.name]['weights'], dims)
            self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], [dim - 2] + list(range(0, dim - 2)) + [dim - 1])
            self.weights_dict[IR_node.name]['weights'] = np.reshape(self.weights_dict[IR_node.name]['weights'], original_dims)

    def emit_DetectionOutput_1(self, IR_node):
        perm_tf1 = IR_node.variable_name + '_perm_loc'
        perm_order_str = '{' + "'order': {}".format([0, 2, 1]) + '}'
        logger.debug("transpose layer emit %s, %s, %s", IR_node.variable_name, self.parent_variable_name(IR_node), perm_order_str)
        self.add_body(1, "n.{:<15} = L.DetectionOutput(n.{}, n.{}, detection_output_param=dict(num_classes={}, share_location={}, background_label_id={},nms_param=dict(nms_threshold={},top_k={}), code_type={}, keep_top_k={}, variance_encoded_in_target={}, confidence_threshold={}), ntop=1)".format(IR_node.variable_name, IR_node.in_edges[0], IR_node.in_edges[1], IR_node.get_attr('classes') or '81', IR_node.get_attr('share_loc') or 'True', IR_node.get_attr('bg_lid') or '0', IR_node.get_attr('nms_threshold') or '0.45', IR_node.get_attr('top_k') or '100', IR_node.get_attr('code_type') or '5', IR_node.get_attr('keep_top_k') or '100', IR_node.get_attr('var_encoded') or 'False', IR_node.get_attr('conf_threshold') or '0.02'))
        self.weights_dict[IR_node.variable_name]['weights'] = self.weights_dict[IR_node.name]['weights']

    def emit_DetectionOutput(self, IR_node):
        perm_tf1 = IR_node.variable_name + '_perm_loc'
        perm_order_str = '{' + "'order': {}".format([0, 2, 1]) + '}'
        logger.debug("transpose layer emit %s, %s, %s", IR_node.variable_name, self.parent_variable_name(IR_node), perm_order_str)
        self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(perm_tf1, IR_node.in_edges[0], perm_order_str))
        perm_tf2 = IR_node.variable_name + '_perm_conf'
        perm_order_str = '{' + "'order': {}".format([0, 2, 1]) + '}'
        logger.debug("transpose layer emit %s, %s, %s", IR_node.variable_name, self.parent_variable_name(IR_node), perm_order_str)
        self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(perm_tf2, IR_node.in_edges[1], perm_order_str))
        self.add_body(1, "n.{:<15} = L.DetectionOutput({}, detection_output_param=dict(num_classes={}, share_location={}, background_label_id={},nms_param=dict(nms_threshold={},top_k={}), code_type={}, keep_top_k={}, variance_encoded_in_target={}, confidence_threshold={}), ntop=1)".format(IR_node.variable_name, 'n.' + perm_tf1 + ', ' + 'n.' + perm_tf2, IR_node.get_attr('classes'), IR_node.get_attr('share_loc'), IR_node.get_attr('bg_lid'), IR_node.get_attr('nms_threshold'), IR_node.get_attr('top_k'), IR_node.get_attr('code_type'), IR_node.get_attr('keep_top_k'), IR_node.get_attr('var_encoded'), IR_node.get_attr('conf_threshold')))
        self.weights_dict[IR_node.name]['weights'] = self.weights_dict[IR_node.name]['weights']

    def emit_Mxnet_Convolution(self, IR_node):
        if not (IR_node.get_attr('data_format') == "NCHW" or IR_node.get_attr('data_format') == "NHWC" or IR_node.get_attr('data_format') == None):
            raise NotImplementedError("CaffeEmitter: Convolution data-format neither of NCHW,NHWC,none(default)")
        dim = IR_node.get_attr('dim')
        self.add_body(1, "n.{:<15} = L.Mxnet_Convolution(n.{}, kernel=dict(dims={}, shape={}), stride=dict(dims={}, shape={}), dilate=dict(dims={}, shape={}), pad=dict(dims={}, shape={}), num_filter={}, num_group={}, no_bias={})".format(IR_node.variable_name, self.parent_variable_name(IR_node), dim, [IR_node.get_attr('kernel_shape')[0], IR_node.get_attr('kernel_shape')[1]], dim, [IR_node.get_attr('strides')[1], IR_node.get_attr('strides')[2]], dim, [IR_node.get_attr('dilations')[1], IR_node.get_attr('dilations')[2]], dim, [IR_node.get_attr('pads')[1], IR_node.get_attr('pads')[2]], IR_node.get_attr('num_filter'), IR_node.get_attr('num_group'), IR_node.get_attr('no_bias')))
        self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)

    def emit_Tf_Conv2D(self, IR_node):
        # strides
        if len(IR_node.get_attr('strides')) > 2:
            stride_h = IR_node.get_attr('strides')[1]
            stride_w = IR_node.get_attr('strides')[2]
        else:
            stride_h = IR_node.get_attr('strides')[0]
            stride_w = IR_node.get_attr('strides')[1]
        # dilation_rate
        dilation_h = 1
        dilation_w = 1
        if IR_node.get_attr('dilations') is not None:
            dilation_h = IR_node.get_attr('dilations')[1]
            dilation_w = IR_node.get_attr('dilations')[2]
        # padding
        padding = IR_node.get_attr('padding')
        if padding == 'SAME':
            padding_type = 0
        else:
            padding_type = 1
        # kernel_shape
        kernel_h = IR_node.get_attr('kernel_shape')[0]
        kernel_w = IR_node.get_attr('kernel_shape')[1]
        kernel_c = IR_node.get_attr('kernel_shape')[2]
        kernel_d = IR_node.get_attr('kernel_shape')[3]
        self.add_body(1, "n.{:<15} = L.Tf_Conv2D(n.{}, strides={}, dilations={}, padding_type={}, filter=dict(shape=dict(dims=4, shape={})))".format(IR_node.variable_name, self.parent_variable_name(IR_node), [stride_h, stride_w], [dilation_h, dilation_w], padding_type, [kernel_h, kernel_w, kernel_c, kernel_d]))
        self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], [3, 2, 0, 1])
        self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)

    def emit_Onnx_Conv(self, IR_node):
        dilations = IR_node.get_attr('dilations')
        group = IR_node.get_attr('group')
        kernel_shape = IR_node.get_attr('kernel_shape')
        pads = IR_node.get_attr('pads')
        strides = IR_node.get_attr('strides')
        if IR_node.get_attr('auto_pad') is not None:
            auto_pad = IR_node.get_attr('auto_pad')
        input_node_name = self.parent_variable_name(IR_node)
        self.add_body(1, "n.{:<15} = L.Onnx_Conv(n.{}, dilations={}, group={}, kernel_shape={}, pads={}, strides={}, no_bias={}, ntop=1)".format(IR_node.variable_name, input_node_name, dilations, group, kernel_shape, pads, strides, IR_node.get_attr('no_bias')))
        self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)

    def emit_Conv(self, IR_node):
        # implement asymmetric paddings by applying symmetric padding then cropping
        # sym_pads = self._get_symmetric_padding(IR_node)
        # output_flag = False
        dilations = IR_node.get_attr('dilations')[1:-1]
        kernels = IR_node.get_attr('kernel_shape')[:-2]
        kernel_c = IR_node.get_attr('kernel_shape')[-2]
        pads = IR_node.get_attr('pads')[1:-1]
        spatial_pad = pads[:len(pads) // 2 - 1] + pads[len(pads) // 2 + 1:]
        spatial_kernels = kernels * 2
        spatial_dilations = dilations * 2
        strides = IR_node.get_attr('strides')[1:-1]
        # kpads = sym_pads #pads[1:len(pads)/2]
        np_spatial_kernels = np.array(spatial_kernels)
        np_spatial_dilations = np.array(spatial_dilations)
        np_new_kernels = np_spatial_kernels + (np_spatial_kernels - 1) * (np_spatial_dilations - 1)
        kpads = spatial_pad
        flag = False
        num_output = IR_node.get_attr('kernel_shape')[-1]
        if IR_node.type == "DepthwiseConv":
            num_group = kernel_c
            num_output = IR_node.get_attr("_output_shapes")[0].dim[-1].size
        else:
            num_group = IR_node.get_attr("group", 1)
        layouts = [None, 'NCHW', 'NHWC', 'NDHWC', 'NCDHW', 'NCH', 'NHC', 'NCW', 'NWC']
        if IR_node.get_attr('data_format') not in layouts:
            raise NotImplementedError("CaffeEmitter: Convolution data-format neither of NCHW,NHWC,NDHWC,NCH,NHC,NCW,NWC,none(default)")
        """ if not (np.all((np.array(kpads) <= np.array(spatial_kernels)//2) == True) or \
            np.all((np.array(kpads) >= np.array(spatial_kernels)//2) == True)) :
            raise NotImplementedError("Pad in Height and Pad in Width both should be either ==, <= or >= kernel_size//2 for Conv") """
        input_node_name = self.parent_variable_name(IR_node)
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        input_node_shape = shape_to_list(input_node.get_attr("_output_shapes")[0])
        output_node_shape = shape_to_list(IR_node.get_attr("_output_shapes")[0])
        globalconv = False
        check_for_crop = True
        input_node_name = self.parent_variable_name(IR_node)
        self.add_body(
        1,
        "n.{:<15} = L.Convolution(n.{}, kernel_size={}, stride = {}, num_output={}, pad={}, group={}, \
        bias_term={},dilation={}, ntop=1)".format(
            IR_node.variable_name,
            input_node_name,
            kernels,
            strides,
            num_output,
            # [pad_f, pad_t, pad_l], #replace with [pad_f, pad_t, pad_l, pad_g, pad_b, pad_r], after analyzer changes
            spatial_pad,
            num_group,
            IR_node.get_attr('use_bias', False),
            dilations))
        # hwcd -> hwcd -> cd hw
        # dchw -> 1,0 / 0,1 -> cd hw
        dim = len(IR_node.get_attr('strides')) - 2
        if self.weight_loaded:
            if IR_node.type == "DepthwiseConv":
                self.weights_dict[IR_node.name]['weights'] = np.swapaxes(self.weights_dict[IR_node.name]['weights'], -1, -2)
            self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], [dim + 1, dim] + list(range(0, dim)))
            self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)
        # MatMul -> Conv is a 1x1 kernel; no need of crop for this case
        # check_if_need_crop function has an assumption of 4D tensor
        if len(input_node_shape) == 4 and check_for_crop:
            self.check_if_need_crop(IR_node)

    def emit_Onnx_GridSample(self, IR_node):
        sampleMode = {
            "linear": 0,
            "bilinear": 1
        }

        paddingMode = {
            "zeros": 0
        }
        align_corners = IR_node.get_attr('align_corners')
        mode = sampleMode[IR_node.get_attr('mode')]
        padding_mode = paddingMode[IR_node.get_attr('padding_mode')]
        input0 = self.IR_graph.get_parent(IR_node.name, [0])
        input1 = self.IR_graph.get_parent(IR_node.name, [1])
        input0_name = input0.real_variable_name
        input1_name = input1.real_variable_name

        input_node_name = self.parent_variable_name(IR_node)
        self.add_body(1, "n.{:<15} = L.GridSample(n.{}, n.{}, onnx_gridsample_param=dict(align_corners={}, mode={}, padding_mode={}), ntop=1)".format(IR_node.variable_name, input0_name, input1_name, align_corners, mode, padding_mode))

    def emit_Onnx_Scatternd(self, IR_node):
        bottom_array = []
        in_edges_index = 0
        if IR_node.name in self.weights_dict.keys() and 'data' in self.weights_dict[IR_node.name].keys():
            bottom_array.append('data')
        else:
            bottom_array.append(IR_node.in_edges[in_edges_index])
            in_edges_index += 1
        if IR_node.name in self.weights_dict.keys() and 'indices' in self.weights_dict[IR_node.name].keys():
            bottom_array.append('indices')
        else:
            bottom_array.append(IR_node.in_edges[in_edges_index])
            in_edges_index += 1
        if IR_node.name in self.weights_dict.keys() and 'updates' in self.weights_dict[IR_node.name].keys():
            bottom_array.append('updates')
        else:
            bottom_array.append(IR_node.in_edges[in_edges_index])
            in_edges_index += 1
        self.add_body(1, "n.{:<15} = L.Scatternd(bottom={},ntop=1)".format(IR_node.variable_name, bottom_array))

    def emit_Onnx_Where(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Onnx_Where(n.{},n.{},n.{},ntop=1)".format(IR_node.variable_name, IR_node.in_edges[0], IR_node.in_edges[1], IR_node.in_edges[2]))

    def emit_Onnx_Equal(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Onnx_Equal(n.{},n.{},ntop=1)".format(IR_node.variable_name, IR_node.in_edges[0], IR_node.in_edges[1]))

    def emit_Onnx_Greaterorequal(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Onnx_Greaterorequal(n.{},n.{},ntop=1)".format(IR_node.variable_name, IR_node.in_edges[0], IR_node.in_edges[1]))

    def emit_Onnx_Gather(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Onnx_Gather(n.{},n.{},axis={},scalar={},dtype=3,indices_dtype=3,ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), IR_node.in_edges[1], IR_node.get_attr('axis'), IR_node.get_attr('scalar')))

    def emit_Onnx_Gatherelements(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Onnx_Gatherelements(n.{},n.{},axis={},ntop=1)".format(IR_node.variable_name, IR_node.in_edges[0], IR_node.in_edges[1], IR_node.get_attr('axis')))

    def emit_Onnx_Gathernd(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Onnx_Gathernd(n.{},n.{},batch_dims={},ntop=1)".format(IR_node.variable_name, IR_node.in_edges[0], IR_node.in_edges[1], IR_node.get_attr('batch_dims')))

    def is_valid_node(self, node_type):
        return hasattr(self, "emit_" + node_type)

    def compute_output_shape(self, IR_node, kernels):
        parent_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert parent_node is not None
        parent_shape = parent_node.get_attr('_output_shapes')[0]
        parent_shape = shape_to_list(parent_shape)
        total_dim = len(parent_shape)
        if len(parent_shape) >= 3:
            strides = IR_node.get_attr('strides')[1:-1]
            pads = IR_node.get_attr('pads')
            oshape = []
            if IR_node.type == 'Pool':
                for i in range(len(strides)):
                    oshape.append((parent_shape[1 + i] + pads[1 + i] + pads[1 + i + total_dim] - kernels[i] + strides[i] - 1) // strides[i] + 1)
            elif IR_node.type == 'Conv2DBackpropInput':
                for i in range(len(strides)):
                    oshape.append((parent_shape[1 + i] - 1) * strides[i] - pads[1 + i] - pads[1 + i + total_dim] + kernels[i])
            elif IR_node.type == 'Conv':
                dilate = IR_node.get_attr('dilations')[1:-1]
                for i in range(len(strides)):
                    oshape.append((parent_shape[1 + i] + pads[1 + i] + pads[1 + i + total_dim] - kernels[i] - (kernels[i] - 1) * (dilate[i] - 1)) // strides[i] + 1)
            else:
                for i in range(len(strides)):
                    oshape.append((parent_shape[1 + i] + pads[1 + i] + pads[1 + i + total_dim] - kernels[i]) // strides[i] + 1)
            return oshape
        else:
            assert False

    def check_if_need_crop_deconv(self, IR_node):
        shape = IR_node.get_attr('_output_shapes')[0]
        shape = shape_to_list(shape)
        ir_o = shape[1:-1]
        if np.any(np.array(ir_o) < 0):
            return
        k_s = IR_node.get_attr('kernel_shape')[0:-2]
        caffe_o = self.compute_output_shape(IR_node, k_s)
        # if asymmetric padding, set offset to 1
        pads = IR_node.get_attr('pads')
        if np.any(np.array(caffe_o) > np.array(ir_o)):
            crop_layer_variable_name = IR_node.real_variable_name + "_crop"
            self.add_body(1, "n.{:<15} = L.Crop(n.{}, L.DummyData(shape=[dict(dim=[1, {}, {}])], \
                ntop=1), ntop=1, offset={})".format(crop_layer_variable_name, IR_node.real_name.replace('/', '_'), shape[-1], ','.join([str(i) for i in ir_o]), [0] * (len(shape) - 2)))
            # Change the layer name
            IR_node.real_name = IR_node.real_name.replace('/', '_') + "_crop"

    def check_if_need_crop(self, IR_node):
        shape = IR_node.get_attr('_output_shapes')[0]
        shape = shape_to_list(shape)
        ir_o = shape[1:-1]
        if np.any(np.array(ir_o) < 0):
            return
        kernel_shape = IR_node.get_attr('kernel_shape')
        k_s = kernel_shape[0:-2]
        if IR_node.type == 'Conv' or IR_node.type == 'DepthwiseConv':
            k_s = kernel_shape[0:-2]
        elif IR_node.type == 'Pool':
            k_s = kernel_shape[1:-1]
        caffe_o = self.compute_output_shape(IR_node, k_s)
        # if asymmetric padding, set offset to 1
        pads = IR_node.get_attr('pads')
        offset = []
        for i in range(len(pads) // 2):
            offset.append(0 if pads[i] == pads[i + len(pads) // 2] else 1)
        if np.any(np.array(caffe_o) > np.array(ir_o)):
            crop_layer_variable_name = IR_node.real_name.replace('/', '_').replace('-', '_') + "_crop"
            self.add_body(1, "n.{:<15} = L.Crop(n.{}, L.DummyData(shape=[dict(dim=[1, {}, {}])], \
                ntop=1), ntop=1, offset={})".format(crop_layer_variable_name, IR_node.real_name.replace('/', '_').replace('-', '_'), shape[-1], ','.join([str(i) for i in ir_o]), offset[1:-1]))
            # Change the layer name
            IR_node.real_name = IR_node.real_name.replace('/', '_').replace('-', '_') + "_crop"

    def emit_Mxnet_Pooling(self, IR_node):
        pool_type = IR_node.get_attr("pooling_type")
        if pool_type == 'avg':
            pool_type = 0
        elif pool_type == 'lp':
            pool_type = 1
        elif pool_type == 'max':
            pool_type = 2
        else:
            pool_type = 3
        pooling_convention = IR_node.get_attr("pooling_convention")
        if pooling_convention == 'full':
            pooling_convention = 0
        elif pooling_convention == 'same':
            pooling_convention = 1
        else:
            pooling_convention = 2
        strides = IR_node.get_attr('strides')[1:-1]
        kernels = IR_node.get_attr('kernel_shape')[1:-1]
        pads = IR_node.get_attr('pads')[1:len(kernels) + 1]
        self.add_body(1, "n.{:<15} = L.Mxnet_Pooling(n.{},kernel = dict(dims={}, shape={}), pool_type = {}, global_pool= {}, pooling_convention = {}, \
            stride = dict(dims={}, shape={}), pad = dict(dims={}, shape={}), p_value= {}, count_include_pad = {})".format(
        IR_node.variable_name,
        self.parent_variable_name(IR_node),
        len(kernels),
        kernels,
        pool_type,
        IR_node.get_attr('global_pooling'),
        pooling_convention,
        len(strides),
        strides,
        len(pads),
        pads,
        IR_node.get_attr('p_value'),
        IR_node.get_attr('count_include_pad'),
    ))

    def emit_Onnx_Averagepool(self, IR_node):
        strides = IR_node.get_attr('strides')
        padding = IR_node.get_attr('pads')
        kernel_shape = IR_node.get_attr('kernel_shape')
        ceil_mode = IR_node.get_attr('ceil_mode', 0)
        count_include_pad = 0
        self.add_body(1, "n.{:<15} = L.Onnx_Averagepool(n.{},ceil_mode = {} , count_include_pad = {}, kernel_shape={} ,pads = {}, \
            strides={})".format(IR_node.variable_name, self.parent_variable_name(IR_node), ceil_mode, count_include_pad, kernel_shape, padding, strides))

    def emit_Pool(self, IR_node):
        pooling_type = IR_node.get_attr('pooling_type')
        if pooling_type == 'MAX':
            pooling_type = P.Pooling.MAX
        elif pooling_type == 'AVG':
            pooling_type = P.Pooling.AVE
        elif pooling_type == 'STOCHASTIC':
            pooling_type = P.Pooling.STOCHASTIC
        elif pooling_type == 'SUM':
            pooling_type = 3
        else:
            raise ValueError()
        parent_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert parent_node is not None
        input_shape = parent_node.get_attr("_output_shapes")[0]
        input_shape = shape_to_list(input_shape)
        strides = IR_node.get_attr('strides')[1:-1]
        if len(strides) == 0:
            strides = [1]*(len(input_shape) - 2)
        kernels = IR_node.get_attr('kernel_shape')[1:-1]
        if len(strides) != len(input_shape) - 2:
            strides = strides * (len(input_shape) - 2 - len(strides))
        if len(kernels) != len(input_shape) - 2:
            kernels = kernels * (len(input_shape) - 2 - len(kernels))
        parent_node = self.IR_graph.get_parent(IR_node.name, [0])
        ceil_mode = IR_node.get_attr('ceil_mode', 0)
        # xprint("ceil_mode: ",ceil_mode)
        # Assume n, h, w, c format
        # input width shape being same as kernel shape implies global pooling
        # set strides to 1
        kernels_half = list(np.array(kernels) // 2)
        if pooling_type == 0:
            padding_value = -sys.float_info.max
        elif pooling_type == 1:
            padding_value = 0
        elif pooling_type == 2:
            padding_value = sys.float_info.max
        if input_shape[1:-1] == kernels and np.all(np.array(IR_node.get_attr('pads')) == 0):
            IR_node.layer.attr['global_pooling'].b = True
        if IR_node.layer.attr['global_pooling'].b:
            if len(input_shape) == 3 and input_shape[0] > 1:
                ishape = clast2first(input_shape)
                dim_str = "'dim': {}".format([1, ishape[0] * ishape[1], 1, ishape[2]])
                dim_str = " reshape_param={'shape': { " + dim_str + '} }'
                reshape_layer_variable_name = IR_node.real_variable_name + "_reshape1"
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape_layer_variable_name, self.parent_variable_name(IR_node), dim_str))
                self.add_body(1, "n.{:<15} = L.Pooling(n.{}, round_mode = {}, pool={}, global_pooling=True, ntop=1)".format(IR_node.variable_name + "_ara_pool", reshape_layer_variable_name, ceil_mode == 0, pooling_type))
                output_shape = shape_to_list(IR_node.get_attr('_output_shapes')[0])
                oshape = clast2first(output_shape)
                dim_str = "'dim': {}".format(oshape)
                dim_str = " reshape_param={'shape': { " + dim_str + '} }'
                # reshape_layer_variable_name = IR_node.real_variable_name + "_reshape2"
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(IR_node.variable_name, IR_node.variable_name + "_ara_pool", dim_str))
                # IR_node.real_name = reshape_layer_variable_name
            else:
                self.add_body(1, "n.{:<15} = L.Pooling(n.{}, round_mode={}, pool={}, global_pooling=True, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), ceil_mode == 0, pooling_type))
        else:
            # pads = self._get_symmetric_padding(IR_node)
            pads = IR_node.get_attr('pads')[1:-1]
            pads = pads[:len(pads) // 2 - 1] + pads[len(pads) // 2 + 1:]
            spatial_kernels = kernels * 2
            self.add_body(1, "n.{:<15} = L.Pooling(n.{}, round_mode={}, pool={}, kernel_size={}, pad={}, stride={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), ceil_mode == 0, pooling_type, kernels, pads, strides))

    def emit_PoolReduce(self, IR_node):
        pooling_type = IR_node.get_attr('pooling_type')
        if pooling_type == 'MAX':
            pooling_type = P.Pooling.MAX
        elif pooling_type == 'AVG':
            pooling_type = P.Pooling.AVE
        elif pooling_type == 'STOCHASTIC':
            pooling_type = P.Pooling.STOCHASTIC
        elif pooling_type == 'SUM':
            pooling_type = P.Pooling.SUMPOOL
        else:
            raise ValueError()
        pool_layer = IR_node.variable_name
        if not IR_node.layer.attr['keepdims'].b:
            pool_layer = IR_node.variable_name + 'ara_pool'
        self.add_body(1, "n.{:<15} = L.Pooling(n.{}, pool={}, global_pooling=True, ntop=1)".format(pool_layer, self.parent_variable_name(IR_node), pooling_type))
        shape = IR_node.get_attr("_output_shapes")[0]
        shape = shape_to_list(shape)
        shape[0] = -1
        if not IR_node.layer.attr['keepdims'].b:
            if len(shape) > 2:
                channels = shape.pop(len(shape) - 1)
                shape.insert(1, channels)
            dim_str = "'dim': {}".format(shape)
            dim_str = " reshape_param={'shape': { " + dim_str + '} }'
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(IR_node.variable_name, pool_layer, dim_str))

    def emit_ResizeBilinear(self, IR_node):
        output_shape = IR_node.get_attr("_output_shapes")[0]
        input_shape = IR_node.get_attr('shape')
        output_shape = shape_to_list(output_shape)
        if len(input_shape) != len(output_shape):
            logger.error("output and input shape ranks different for upsample")
            raise ValueError()
        if len(input_shape) != 4:
            logger.error("expected 4 dimensional tensor for upsample")
            raise ValueError()
        if input_shape[-1] != output_shape[-1]:
            logger.error("currently only spatial dimension can be upsampled")
            raise NotImplementedError()
        if IR_node.get_attr('half_pixel_centers'):
            logger.error("currently only half_pixel_centers false is supported")
            raise NotImplementedError()
        if IR_node.get_attr('align_corners'):
            if input_shape[1] == input_shape[2] and input_shape[1] == 1:
                logger.info("encountered input spatial dimensions as 1")
                scale_h = (output_shape[1]) / (input_shape[1])
                scale_w = (output_shape[2]) / (input_shape[2])
                # nhwc -> nchw
                nchw_outshape = [output_shape[0]] + [output_shape[-1]] + output_shape[1:-1]
                dim_str = "'dim': {}".format(nchw_outshape)
                dim_str = "{'shape': { " + dim_str + '} }'
                self.add_body(1, "n.{:<15} = L.Broadcast(n.{},broadcast_param = {}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), dim_str))
            elif input_shape[1] == 1 or input_shape[2] == 1:
                logger.error("currently dvmc doesn't support uneven scales")
                raise NotImplementedError()
            else:
                scale_h = (output_shape[1] - 1) / (input_shape[1] - 1)
                scale_w = (output_shape[2] - 1) / (input_shape[2] - 1)
                if scale_h != scale_w:
                    logger.error("currently dvmc doesn't support uneven scales")
                    raise NotImplementedError()
                if not scale_h.is_integer():
                    logger.error("currently dvmc doesn't support non integer (outshape-1)/(inshape-1)")
                    raise NotImplementedError()
                scale_h = int(scale_h)
                scale_w = int(scale_w)
                self.add_body(1, "n.{:<15} = L.RepEdgeDeconv(n.{}, convolution_param=dict(kernel_h={}, kernel_w={}, stride={}, num_output={}, pad_h={}, pad_w={}, bias_term={},group={}), ntop=1)".format(IR_node.variable_name + "_ara_redeconv", self.parent_variable_name(IR_node), scale_h * 2, scale_w * 2, scale_h, input_shape[-1], 0, 0, IR_node.get_attr('use_bias', False), input_shape[-1]))
                crop_layer_variable_name = IR_node.real_variable_name + "_crop"
                self.add_body(1, "n.{:<15} = L.Crop(n.{}, L.DummyData(shape=[dict(dim=[1, {}, {}, {}])], \
                            ntop=1), ntop=1, offset={})".format(IR_node.variable_name, IR_node.variable_name + "_ara_redeconv", output_shape[3], output_shape[1], output_shape[2], [scale_h - 1, scale_w - 1]))
                # IR_node.real_name = crop_layer_variable_name
                half_filter_row = [i / scale_w for i in range(1, scale_w)]
                filter_row = np.array([half_filter_row + [1] + list(reversed(half_filter_row)) + [0]])
                filter = filter_row.T * filter_row
                if self.weight_loaded:
                    self.weights_dict[IR_node.variable_name + "_ara_redeconv"] = {'weights': filter}
        else:
            scale_h = (output_shape[1]) / (input_shape[1])
            scale_w = (output_shape[2]) / (input_shape[2])
            if scale_h != scale_w:
                logger.error("currently dvmc doesn't support uneven scales")
                raise NotImplementedError()
            if not scale_h.is_integer():
                logger.error("currently dvmc doesn't support non integer (outshape)/(inshape)")
                raise NotImplementedError()
            scale_h = int(scale_h)
            scale_w = int(scale_w)
            self.add_body(1, "n.{:<15} = L.RepEdgeDeconv(n.{}, convolution_param=dict(kernel_h={}, kernel_w={}, stride={}, num_output={}, pad_h={}, pad_w={},rep_edge_r=1,rep_edge_b=1, bias_term={},group={}), ntop=1)".format(IR_node.variable_name + "_ara_redeconv", self.parent_variable_name(IR_node), scale_h * 2, scale_w * 2, scale_h, input_shape[-1], 0, 0, IR_node.get_attr('use_bias', False), input_shape[-1]))
            # crop_layer_variable_name = IR_node.real_variable_name + "_crop"
            self.add_body(1, "n.{:<15} = L.Crop(n.{}, L.DummyData(shape=[dict(dim=[1, {}, {}, {}])], \
                        ntop=1), ntop=1, offset={})".format(IR_node.variable_name, IR_node.variable_name + "_ara_redeconv", output_shape[3], output_shape[1], output_shape[2], [scale_h - 1, scale_w - 1]))
            # IR_node.real_name = crop_layer_variable_name
            half_filter_row = [i / scale_w for i in range(1, scale_w)]
            filter_row = np.array([half_filter_row + [1] + list(reversed(half_filter_row)) + [0]])
            filter = filter_row.T * filter_row
            if self.weight_loaded:
                self.weights_dict[IR_node.variable_name + "_ara_redeconv"] = {'weights': filter}

    def emit_Tf_Cropandresize(self, IR_node):
        if IR_node.get_attr('method') == 'nearest':
            self.add_body(1, "n.{:<15} = L.Tf_Cropandresize(n.{},box_ind = {},crop_size = dict(dims = {}, shape ={}),method={},extrapolation_value={},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), IR_node.get_attr('box_ind'), 1, [IR_node.get_attr('shape')[1], IR_node.get_attr('shape')[2]], 1, IR_node.get_attr('e_val')))
        else:
            self.add_body(1, "n.{:<15} = L.Tf_Cropandresize(n.{},box_ind = {},crop_size = dict(dims = {}, shape ={}),method={},extrapolation_value={},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), IR_node.get_attr('box_ind'), 1, [IR_node.get_attr('shape')[1], IR_node.get_attr('shape')[2]], 0, IR_node.get_attr('e_val')))

    def emit_Tf_ResizeNearestNeighbor(self, IR_node):
        output_shape = IR_node.get_attr("_output_shapes")[0]
        output_shape = shape_to_list(output_shape)
        self.add_body(1, "n.{:<15} = L.Tf_Resizenearestneighbor(n.{},size=dict(dims={}, shape={}),align_corners = {}, half_pixel_centers = {})".format(IR_node.variable_name, self.parent_variable_name(IR_node), 1, [output_shape[1], output_shape[2]], IR_node.get_attr("align_corners"), IR_node.get_attr("half_pixel_centers")))

    def emit_ONNX_UpSample(self, IR_node):
        output_shape = IR_node.get_attr("_output_shapes")[0]
        output_shape = shape_to_list(output_shape)
        parent_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert parent_node is not None
        input_shape = parent_node.get_attr("_output_shapes")[0]
        input_shape = shape_to_list(input_shape)
        scale_h = output_shape[1] / input_shape[1]
        scale_w = output_shape[2] / input_shape[2]
        self.add_body(1, "n.{:<15} = L.Onnx_Upsample(n.{},scales=dict(shape=dict(dims=2, shape={})),f_scale_h={},f_scale_w={})".format(
        IR_node.variable_name,
        self.parent_variable_name(IR_node),
        [1, 1],
        scale_h,
        scale_w,
    ))

    def emit_ONNX_Slice(self, IR_node):
        # If all the starts are 0 and ends are greater than the size in that dimension, no need to compare axes
        starts = IR_node.layer.attr["starts"]
        ends = IR_node.layer.attr["ends"]
        axes = IR_node.layer.attr["axes"]
        steps = IR_node.layer.attr["steps"]
        is_float = IR_node.layer.attr["is_float"]
        self.add_body(1, "n.{:<15} = L.Onnx_Slice(n.{},starts=dict(shape=dict(shape={},dims={})),ends=dict(shape=dict(shape={},dims={})),axes=dict(shape=dict(shape={},dims={})),steps=dict(shape=dict(shape={},dims={})), isfloat={}, adj_param=dict(precision=[3]))".format(IR_node.variable_name, self.parent_variable_name(IR_node), starts.list.i, len(starts.list.i), ends.list.i, len(ends.list.i), axes.list.i, len(axes.list.i), steps.list.i, len(steps.list.i), is_float.i))

    def emit_ResizeNearestNeighbor(self, IR_node):
        output_shape = IR_node.get_attr("_output_shapes")[0]
        output_shape = shape_to_list(output_shape)
        input_shape = IR_node.get_attr('shape')
        # a custom mode we implemented, originally not present in any network
        zero_mode = []
        if 'zero_mode' in IR_node.layer.attr:
            zero_mode = IR_node.get_attr('zero_mode')
        elif len(output_shape) in [3, 4, 5]:
            zero_mode = [0] * (len(output_shape) - 2)
        if len(input_shape) != len(output_shape):
            logger.error("output and input shape ranks different for upsample")
            raise ValueError()
        if len(input_shape) in [4, 5]:
            if input_shape[-1] != output_shape[-1]:
                logger.error("currently only spatial dimension can be upsampled")
                raise NotImplementedError()
            output_shape_spatial = output_shape[1:-1]
            input_shape_spatial = input_shape[1:-1]
            scales = [oo / ii for oo, ii in zip(output_shape_spatial, input_shape_spatial)]
            croppable = False
            if all([(math.ceil(ss) * ii - oo) in [0, 1] for ss, oo, ii in zip(scales, output_shape_spatial, input_shape_spatial)]):
                croppable = True
                if all([i.is_integer() for i in scales]):
                    croppable = False
                scales = [float(math.ceil(ss)) for ss in scales]
            if scales[-2] != scales[-1]:
                logger.error("currently dvnc doesn't support uneven scales across H and W")
                raise NotImplementedError()
            elif len(scales) == 3 and (scales[0] not in [1, scales[-1]]):
                logger.error("currently for 3D networks dvnc only supports cases (scale_d = scale_h = scale_w) or (scale_d = 1, scale_h = scale_w)")
                raise NotImplementedError()
            if (isinstance(scales[-1], int) or not scales[-1].is_integer()) and not croppable:
                logger.error("currently dvnc doesn't support non integer,non croppable scales")
                raise NotImplementedError()
            base_name = IR_node.variable_name
            if croppable:
                base_name = base_name + "ara_upsample"
            self.add_body(1, "n.{:<15} = L.Upsample(n.{}, scale={}, zero_mode={}, ntop=1)".format(base_name, self.parent_variable_name(IR_node), list(map(int, scales)), zero_mode))
            if croppable:
                # crop_layer_variable_name = IR_node.real_variable_name + "_crop"
                offset = [0] * (len(output_shape) - 2)
                self.add_body(1, "n.{:<15} = L.Crop(n.{}, L.DummyData(shape=[dict(dim=[1, {}, {}])], \
                            ntop=1), ntop=1, offset={})".format(IR_node.variable_name, base_name, output_shape[-1], ','.join(str(i) for i in output_shape[1:-1]), offset))
                # IR_node.real_name = crop_layer_variable_name
        else:
            logger.error("expected 4 or 5 dimensional tensor for upsample")
            raise ValueError()

    def emit_UNKNOWN(self, IR_node):
        logger.warn("unknown node %s with type %s", IR_node.variable_name, IR_node.type)

    def emit_Conv2DBackpropInput(self, IR_node):
        dcnv_layer_name = IR_node.variable_name
        if (IR_node.get_attr('adj') is not None and sum(IR_node.get_attr('adj')) > 0) or (IR_node.get_attr('pads') is not None and sum(IR_node.get_attr('pads')) > 0) or IR_node.get_attr('target_shape') is not None:
            dcnv_layer_name = IR_node.variable_name + "_dcnv"
        self.add_body(1, "n.{:<15} = L.Deconvolution(n.{}, convolution_param=dict(kernel_size = {}, stride={}, num_output={},dilation={}, pad = {}, bias_term={}, group={}), ntop=1)".format(dcnv_layer_name, self.parent_variable_name(IR_node), IR_node.get_attr('kernel_shape')[:-2], IR_node.get_attr('strides')[1:-1], IR_node.get_attr('num_outputs'), IR_node.get_attr('dilations')[1:-1], [0] * 2 * len(IR_node.get_attr('strides')[1:-1]), IR_node.get_attr('use_bias', False), IR_node.get_attr('group')))
        if (IR_node.get_attr('adj') is not None and sum(IR_node.get_attr('adj')) > 0):
            if len(IR_node.get_attr('adj')) == 1:
                padding = [0, IR_node.get_attr('adj')[0]]  # Removing N,C padding [0,0,0,0]
            elif len(IR_node.get_attr('adj')) == 2:
                padding = [0, 0, IR_node.get_attr('adj')[0], IR_node.get_attr('adj')[1]]
            elif len(IR_node.get_attr('adj')) == 3:
                padding = [0, 0, 0, IR_node.get_attr('adj')[0], IR_node.get_attr('adj')[1], IR_node.get_attr('adj')[2]]
            else:
                raise NotImplementedError("unsupported length for adj parameter")
            self.add_body(1, "n.{:<15} = L.Padding(n.{}, padding_param=dict(pad ={}, pad_method={}, pad_value={}), ntop=1)".format(IR_node.variable_name, dcnv_layer_name, padding, 0, 0))
            # IR_node.real_name = padding_layer_name
        if (IR_node.get_attr('pads') is not None and sum(IR_node.get_attr('pads')) > 0):
            # crop_layer_variable_name = IR_node.real_variable_name + "_crop"
            new_shapes = [i.size for i in IR_node.get_attr('_output_shapes')[0].dim[1:-1]]
            new_shapes = [1, IR_node.get_attr('_output_shapes')[0].dim[-1].size] + new_shapes
            offsets = clast2first(IR_node.get_attr('pads')[:len(IR_node.get_attr('pads')) // 2])
            self.add_body(1, "n.{:<15} = L.Crop(n.{}, L.DummyData(shape=[dict(dim={})], \
                        ntop=1), ntop=1, offset={},axis=1)".format(IR_node.variable_name, dcnv_layer_name, new_shapes, offsets[1:]))
            # IR_node.real_name = crop_layer_variable_name
        if IR_node.get_attr('target_shape') is not None:
            input_node = self.IR_graph.get_parent(IR_node.name, [0])
            assert input_node is not None
            input_node_shape = shape_to_list(input_node.get_attr('_output_shapes')[0])
            oshape = shape_to_list(IR_node.get_attr('_output_shapes')[0])
            ispatial = input_node_shape[1:-1]
            ospatial = oshape[1:-1]
            strides = IR_node.get_attr('strides')[1:-1]
            kernels = IR_node.get_attr('kernel_shape')[:-2]
            deconv_shape = (np.array(ispatial) - 1) * strides + kernels
            diff = deconv_shape - ospatial
            offsets = [int(i) for i in list(np.ceil(diff / 2))]
            # crop_layer_variable_name = IR_node.real_variable_name + "_crop"
            new_shapes = [i.size for i in IR_node.get_attr('_output_shapes')[0].dim[1:-1]]
            new_shapes = [1, IR_node.get_attr('_output_shapes')[0].dim[-1].size] + new_shapes
            self.add_body(1, "n.{:<15} = L.Crop(n.{}, L.DummyData(shape=[dict(dim={})], \
                        ntop=1), ntop=1, offset={})".format(IR_node.variable_name, dcnv_layer_name, new_shapes, offsets))
            # IR_node.real_name = crop_layer_variable_name
        dim = len(IR_node.get_attr('strides')) - 2
        if self.weight_loaded:
            if dcnv_layer_name != IR_node.variable_name:
                self.weights_dict[dcnv_layer_name] = dict()
            self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], [dim + 1, dim] + list(range(0, dim)))
            self.weights_dict[dcnv_layer_name] = self.weights_dict.pop(IR_node.name)

    def emit_DataInput(self, IR_node):
        shape = self._shapeToStr(IR_node.get_attr('_output_shapes')[0])
        # if IR_node.get_attr('data_format') != 'TNC':
        shape = clast2first(shape)
        self.add_body(1, "n.{:<15} = L.Input(shape=[dict(dim={})], ntop=1)".format(IR_node.variable_name, shape))

    def emit_Dropout(self, IR_node):
        in_place = False
        self.add_body(1, "n.{:<15} = L.Dropout(n.{}, dropout_ratio={} , in_place={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), 1 - IR_node.get_attr('keep_prob'), in_place))

    def emit_FullyConnected(self, IR_node):
        parent_node = self.IR_graph.get_parent(IR_node.name, [0])
        # TODO FIXME for volumetric data
        assert parent_node is not None
        input_shape = parent_node.get_attr("_output_shapes")[0]
        input_shape = clast2first(shape_to_list(input_shape))
        # prod = 1
        axis = 1
        if len(input_shape) == 1:
            axis = 0
        flatten = IR_node.get_attr('flatten', True)
        inrp_layer_name = ""
        if not flatten and len(input_shape) > 2:
            current_input = self.parent_variable_name(IR_node)
            effective_ndim = len(input_shape)
            
            # Pre-process: If 3D, reshape to 4D to satisfy Caffe Convolution requirements
            if len(input_shape) == 3:
                logger.info("Input shape length is 3")
                # Explicitly writing dimensions (Batch, Length, 1, Channels)
                # Assuming input_shape contains [Batch, Length, Channels]
                dim_b = input_shape[0]
                dim_l = input_shape[1]
                dim_c = input_shape[2]
                
                reshape_in_dims = [1, dim_b, dim_l, dim_c]
                reshape_in_param = "{{'shape': {{'dim': {}}} }}".format(reshape_in_dims)
                
                reshaped_name = IR_node.variable_name + "_reshape_4d"
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, reshape_param={}, ntop=1)".format(reshaped_name, current_input, reshape_in_param))
                
                current_input = reshaped_name
                effective_ndim = 4 # Treat as 4D for the permute logic

            # 1. Permute: Move last dimension (Features/Channels) to index 1 (Caffe Channels)
            # Order becomes [0, last, 1, 2, ... last-1]
            permute_order = [0, effective_ndim - 1] + list(range(1, effective_ndim - 1))
            permute_param = "{{'order': {}}}".format(permute_order)
            self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(IR_node.variable_name + "_perm", current_input, permute_param))

            # 2. Convolution: 1x1 Conv
            self.add_body(1, "n.{:<15} = L.Convolution(n.{}, num_output={}, bias_term={}, kernel_size=1, stride=1, pad=0, dilation=1, group=1, ntop=1)".format(IR_node.variable_name + "_conv", IR_node.variable_name + "_perm", IR_node.layer.attr["units"].i, IR_node.get_attr('use_bias', False)))
            if self.name_map_df is not None:
                mask = self.name_map_df['Intermediate_name'] == IR_node.variable_name

                if mask.any():
                    self.name_map_df.loc[mask, 'Intermediate_name'] = IR_node.variable_name + "_conv"
                    self.name_map_df.loc[mask, 'ARANet_name'] = IR_node.variable_name + "_conv"
            # 3. Inverse Permute: Move Channel dimension (1) back to last
            # Order becomes [0, 2, 3, ... last, 1]
            inv_permute_order = [0] + list(range(2, effective_ndim)) + [1]
            inv_permute_param = "{{'order': {}}}".format(inv_permute_order)
            
            # Determine output name for the permute layer
            # If 3D, this is an intermediate layer. If >3D, this is the final layer.
            permute_out_name = IR_node.variable_name + "_inv_perm" if len(input_shape) == 3 else IR_node.variable_name
            self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(permute_out_name, IR_node.variable_name + "_conv", inv_permute_param))

            # Post-process: If originally 3D, reshape back to (Batch, Length, New_Channels)
            if len(input_shape) == 3:
                dim_out_c = IR_node.layer.attr["units"].i
                reshape_out_dims = [dim_b, dim_l, dim_out_c]
                reshape_out_param = "{{'shape': {{'dim': {}}} }}".format(reshape_out_dims)
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, reshape_param={}, ntop=1)".format(IR_node.variable_name, permute_out_name, reshape_out_param))
        else:
            inrp_layer_name = IR_node.variable_name
            if IR_node.get_attr('reshape', False):
                inrp_layer_name = inrp_layer_name + "_ara_inrp"
            self.add_body(1, "n.{:<15} = L.InnerProduct(n.{}, num_output={}, bias_term={}, axis={}, ntop=1)".format(inrp_layer_name, self.parent_variable_name(IR_node), IR_node.layer.attr["units"].i, IR_node.get_attr('use_bias', False), axis))
            if IR_node.get_attr('reshape', False):
                output_shape = IR_node.get_attr('reshape')
                dim_str = "'dim': {}".format(output_shape)
                dim_str = "{'shape': { " + dim_str + '} }'
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, reshape_param={}) ".format(IR_node.variable_name, inrp_layer_name, dim_str))
                # IR_node.real_name = IR_node.variable_name + "_reshape2"
        if self.weight_loaded:
            # self.check_if_need_transpose(IR_node)
            if inrp_layer_name != IR_node.variable_name:
                node_name = IR_node.name + "_conv"
                self.weights_dict[node_name] = dict()
                self.weights_dict[node_name]['weights'] = np.expand_dims(np.transpose(self.weights_dict[IR_node.name]['weights'], (1, 0)), axis=(-2, -1))
                if 'bias' in self.weights_dict[IR_node.name]:
                    self.weights_dict[node_name]['bias'] = self.weights_dict[IR_node.name]['bias']
                self.weights_dict[IR_node.variable_name + "_conv"] = self.weights_dict.pop(node_name)
                self.weights_dict.pop(IR_node.name)
            else:
                self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], (1, 0))
                self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)

    def emit_Mxnet_Upsampling(self, IR_node):
        output_shape = IR_node.get_attr("_output_shapes")[0]
        output_shape = shape_to_list(output_shape)
        scale = IR_node.get_attr('scale')
        upsample_mode = IR_node.get_attr("upsample_mode")
        zero_mode = []
        if 'zero_mode' in IR_node.layer.attr:
            zero_mode = IR_node.get_attr('zero_mode')
        elif len(output_shape) in [3, 4, 5]:
            zero_mode = [0] * (len(output_shape) - 2)
        if (IR_node.get_attr('sample_type') == 'bilinear'):
            self.add_body(1, "n.{:<15} = L.Mxnet_Upsampling(n.{},scale={},num_filter = {},sample_type = 1, num_args= {})".format(IR_node.variable_name, self.parent_variable_name(IR_node), scale, IR_node.get_attr('num_filter'), IR_node.get_attr('num_args')))
        elif len(IR_node.in_edges) == 1:
            self.add_body(1, "n.{:<15} = L.Upsample(n.{}, scale={}, zero_mode={},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), [int(scale)] * (len(output_shape) - 2), zero_mode))
        else:
            bottom_array = []
            for i in range(0, len(IR_node.in_edges)):
                bottom_array.append(IR_node.variable_name + '_' + IR_node.in_edges[i] + '_' + str(i))
                self.add_body(1, "n.{:<15} = L.Upsample(n.{}, scale={}, zero_mode={},ntop=1)".format(
            IR_node.variable_name + '_' + IR_node.in_edges[i] + '_' + str(i),
            IR_node.in_edges[i],
            [int(scale)] * (len(output_shape) - 2),
            zero_mode,
        ))
            bottom_array_str = ', '.join(['n.' + i for i in bottom_array])

            if (IR_node.get_attr('multi_input_mode') == 'concat'):
                self.add_body(1, "n.{:<15} = L.Concat({}, axis={})".format(IR_node.variable_name, bottom_array_str, 1))
            else:
                self.add_body(1, "n.{:<15} = L.Eltwise({}, operation=1, ntop=1)".format(
            IR_node.variable_name,
            bottom_array_str,
        ))

    def emit_Mxnet_L2Normalization(self, IR_node):
        mode_map = {1: 'P.Mxnet_L2Normalizationmodetype_instance', 3: 'P.Mxnet_L2Normalizationmodetype_spatial', 2: 'P.Mxnet_L2Normalizationmodetype_channel'}
        mode = mode_map[IR_node.get_attr('mode')]
        self.add_body(1, "n.{:<15} = L.Mxnet_L2Normalization(n.{}, mxnet_l2normalization_param=dict(eps={}, mode={}), ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), IR_node.get_attr('eps'), IR_node.get_attr('mode')))

    def emit_L2Norm_AC(self, IR_node):
        ishape = IR_node.get_attr('_input_shape')
        dim_str = "'dim': {}".format(ishape)
        dim_str = " reshape_param={'shape': { " + dim_str + '} }'
        iname = IR_node.real_variable_name + "_reshape1"
        self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(iname, self.parent_variable_name(IR_node), dim_str))
        self.add_body(1, "n.{:<15} = L.Normalize(n.{},across_spatial={},eps={},ntop=1)".format(IR_node.variable_name + '_l2norm', iname, IR_node.get_attr('across_spatial', 'False'), IR_node.get_attr('eps', '9.9e-06')))
        oshape = IR_node.get_attr('_output_shape')
        dim_str = "'dim': {}".format(oshape)
        dim_str = " reshape_param={'shape': { " + dim_str + '} }'
        self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(IR_node.variable_name, IR_node.variable_name + '_l2norm', dim_str))

    def emit_L2Norm_SSD(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Normalize(n.{},across_spatial={},eps={},ntop=1)".format(IR_node.variable_name + '_l2norm', self.parent_variable_name(IR_node), IR_node.get_attr('across_spatial', 'False'), IR_node.get_attr('eps', '1e-10')))
        bnorm_layer_name = IR_node.variable_name + "_bnorm"
        self.add_body(1, "n.{:<15} = L.BatchNorm(n.{}, eps={}, use_global_stats={}, ntop=1)".format(bnorm_layer_name, IR_node.variable_name + '_l2norm', IR_node.get_attr('epsilon'), self.phase == 'test'))
        scale_layer_var_name = IR_node.variable_name
        shape = IR_node.get_attr("_output_shapes")[0]
        shape = shape_to_list(shape)
        axis = 1
        if len(shape) == 1:
            axis = 0
        self.add_body(1, "n.{:<15} = L.Scale(n.{}, bias_term={}, in_place=False, axis={}, ntop=1)".format(scale_layer_var_name, bnorm_layer_name, True, axis))
        self.weights_dict[bnorm_layer_name] = copy.deepcopy(self.weights_dict[IR_node.name])
        if self.weight_loaded:
            if 'scale' not in self.weights_dict[IR_node.name]:
                self.weights_dict[scale_layer_var_name]['scale'] = 1
            self.weights_dict[bnorm_layer_name]['scale'] = 1

            if 'bias' in self.weights_dict[IR_node.name]:
                self.weights_dict[bnorm_layer_name].pop('bias', None)
            self.weights_dict[scale_layer_var_name] = self.weights_dict.pop(IR_node.name)
            self.weights_dict[scale_layer_var_name].pop('mean', None)
            self.weights_dict[scale_layer_var_name].pop('var', None)

    def emit_L2Norm_scale(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Normalize(n.{},across_spatial={},eps={},scale_before_clip={},scale_after_div={},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), IR_node.get_attr('across_spatial', 'True'), IR_node.get_attr('eps', '1e-10'), IR_node.get_attr('scale_before_clip', '1'), IR_node.get_attr('scale_after_div', '1')))

    def emit_L2Norm(self, IR_node):
        if IR_node.get_attr('per_channel'):
            shape = IR_node.get_attr("_output_shapes")[0]
            ishape = clast2first(shape_to_list(shape))
            dim_str = "'dim': {}".format(ishape[1:])
            dim_str = " reshape_param={'shape': { " + dim_str + '} }'
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {}) ".format(IR_node.variable_name + "_reshape1", self.parent_variable_name(IR_node), dim_str))
            self.add_body(1, "n.{:<15} = L.Normalize(n.{},across_spatial={},eps={},ntop=1)".format(IR_node.variable_name, IR_node.variable_name + "_reshape1", IR_node.get_attr('across_spatial', 'True'), IR_node.get_attr('eps', '1e-10')))
            dim_str = "'dim': {}".format(ishape)
            dim_str = " reshape_param={'shape': { " + dim_str + '} }'
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {}) ".format(IR_node.variable_name + "_reshape2", IR_node.variable_name, dim_str))
            IR_node.real_name = IR_node.variable_name + "_reshape2"
        else:
            self.add_body(1, "n.{:<15} = L.Normalize(n.{},across_spatial={},eps={},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), IR_node.get_attr('across_spatial', 'True'), IR_node.get_attr('eps', '1e-10')))

    def emit_Mxnet_Batchnorm(self, IR_node):
        if IR_node.get_attr('output_mean_var'):
            outputs = 'n.{}'.format(IR_node.variable_name) + ',n.{}'.format(IR_node.variable_name + '_mean') + ',n.{}'.format(IR_node.variable_name + '_var')
            ntop = 3
        else:
            outputs = 'n.{}'.format(IR_node.variable_name)
            ntop = 1
        self.add_body(1, "{:<15} = L.Mxnet_Batchnorm(n.{}, eps={}, use_global_stats={}, axis={}, fix_gamma={}, output_mean_var={},ntop={})".format(outputs, self.parent_variable_name(IR_node), IR_node.get_attr('eps'), IR_node.get_attr('use_global_stats'), IR_node.get_attr('axis'), IR_node.get_attr('fix_gamma'), IR_node.get_attr('output_mean_var'), ntop))
        self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)

    def emit_LayerNorm(self, IR_node):
        if 'output_mean_var' not in IR_node.layer.attr:
            IR_node.layer.attr['output_mean_var'].b = False
        if IR_node.get_attr('output_mean_var'):
            outputs = 'n.{}'.format(IR_node.variable_name) + ',n.{}'.format(IR_node.variable_name + '_mean') + ',n.{}'.format(IR_node.variable_name + '_var')
            ntop = 3
        else:
            outputs = 'n.{}'.format(IR_node.variable_name)
            ntop = 1
        if 'uses_mean' not in IR_node.layer.attr:
            IR_node.layer.attr['uses_mean'].b = True
        self.add_body(1, "{:<17} = L.LayerNorm(n.{}, eps= {}, axis = {}, output_mean_var= {}, uses_mean={}, ntop ={})".format(outputs, self.parent_variable_name(IR_node), IR_node.get_attr('eps'), IR_node.get_attr('axis'), IR_node.get_attr('output_mean_var'), IR_node.get_attr('uses_mean'), ntop))
        self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)
    
    def emit_topk(self, IR_node):
        self.add_body(1, "n.{}, n.{} = L.TopKRtmdet(n.{}, axis={}, largest={}, sorted={}, k={}, ntop=2)".format(IR_node.get_attr("out_layers")[0].decode(), IR_node.get_attr("out_layers")[1].decode(), self.parent_variable_name(IR_node, [0]), IR_node.get_attr("axis"), IR_node.get_attr("largest", 1), IR_node.get_attr("sorted", 1), IR_node.get_attr("K")))

    def emit_GroupNorm(self, IR_node):
        outputs = 'n.{}'.format(IR_node.variable_name)
        ntop = 1
        self.add_body(
        1,
        "{:<15} = L.GroupNorm(n.{}, eps= {}, groups = {}, output_mean_var= {}, ntop ={})".format(
            outputs,
            self.parent_variable_name(IR_node),
            IR_node.get_attr('eps'),
            IR_node.get_attr('groups'),
            False,  #IR_node.get_attr('output_mean_var'),
            ntop))
        self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)

    def emit_BatchNorm(self, IR_node):
        Precision = {
            'INT8': 0,
            'INT16': 1,
            'INT32': 2,
            'FP32': 3,
            'ZERO': 999,
        }
        prec = Precision[IR_node.get_attr("new_precision", 'ZERO')]
        # logger.info("emitting batchnorm layer %s %s", IR_node.name, IR_node.variable_name)
        bnorm_input = self.parent_variable_name(IR_node)
        if prec == Precision['INT16']:
            pre_dvshift_0 = IR_node.variable_name + "_pre_dvshift_0"
            self.add_body(1, "n.{:<15} = L.DVShift(n.{}, dvshift_param=dict(shift={}), ntop=1, adj_param=dict(precision=[{}]))".format(pre_dvshift_0, bnorm_input, 0, prec))
            bnorm_input = pre_dvshift_0
        bnorm_layer_name = IR_node.variable_name + "_bnorm"
        self.add_body(1, "n.{:<15} = L.BatchNorm(n.{}, eps={}, use_global_stats={}, ntop=1, adj_param=dict(precision=[{}]))".format(bnorm_layer_name, bnorm_input, IR_node.get_attr('epsilon'), self.phase == 'test', prec))
        scale_layer_var_name = IR_node.variable_name
        shape = IR_node.get_attr("_output_shapes")[0]
        shape = shape_to_list(shape)
        axis = 1
        if len(shape) == 1:
            axis = 0
        self.add_body(
        1,
        "n.{:<15} = L.Scale(n.{}, bias_term={}, in_place=False, axis={}, ntop=1)".format(
            scale_layer_var_name,
            bnorm_layer_name,
            True,  #IR_node.get_attr('use_bias', False)
            axis))
        self.weights_dict[bnorm_layer_name] = copy.deepcopy(self.weights_dict[IR_node.name])
        if self.weight_loaded:
            if 'scale' not in self.weights_dict[IR_node.name]:
                self.weights_dict[IR_node.name]['scale'] = 1
            self.weights_dict[bnorm_layer_name]['scale'] = 1

            if 'bias' in self.weights_dict[IR_node.name]:
                self.weights_dict[bnorm_layer_name].pop('bias', None)
            self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)
            self.weights_dict[IR_node.variable_name].pop('mean', None)
            self.weights_dict[IR_node.variable_name].pop('var', None)
        if prec == Precision['INT16']:
            post_dvshift = IR_node.variable_name + "_post_dvshift"
            self.add_body(1, "n.{:<15} = L.DVShift(n.{}, dvshift_param=dict(shift={}), ntop=1)".format(post_dvshift, scale_layer_var_name, 0))
            IR_node.real_name = post_dvshift

    def emit_Scale(self, IR_node):
        shape = IR_node.get_attr("_output_shapes")[0]
        shape = shape_to_list(shape)
        axis = 1
        if len(shape) == 1:
            axis = 0
        self.add_body(1, "n.{:<15} = L.Scale(n.{}, bias_term={}, in_place=False, axis={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), IR_node.get_attr('use_bias', False), axis))
        if self.weight_loaded:
            self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)

    def emit_Constant(self, IR_node):
        if IR_node.get_attr('value'):
            value = IR_node.get_attr('value')
        else:
            value = self.weights_dict[IR_node.name]['value'][0]
        # In order to use Eltwise prod for division
        son_node = self.IR_graph.get_son(IR_node.name, [0])
        if son_node is not None and son_node.layer.op == 'Div':
            value = 1.0 / value
        assert son_node is not None
        shape = son_node.get_attr("_output_shapes")[0]
        shape = shape_to_list(shape)
        if len(shape) == 4:
            shape[1], shape[2], shape[3] = shape[3], shape[1], shape[2]
            shape[0] = 1
        if len(shape) == 0:
            shape = [1]
        shape = list(map(lambda x: str(x), shape))
        self.add_body(1, "n.{:<15} = L.DummyData(shape=[dict(dim=[{}])], data_filler=dict(type='constant', value={}), ntop=1)".format(IR_node.variable_name, ', '.join(shape), value))

    def emit_Onnx_Not(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Onnx_Not(n.{},ntop=1)".format(IR_node.variable_name, IR_node.in_edges[0]))

    def emit_Onnx_Lrn(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Onnx_Lrn(n.{}, size={}, alpha={}, beta={}, bias={})".format(IR_node.variable_name, self.parent_variable_name(IR_node), IR_node.get_attr('size'), IR_node.get_attr('alpha'), IR_node.get_attr('beta'), IR_node.get_attr('bias')))

    def emit_Tf_LRN(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Lrn(n.{}, depth_radius={}, alpha={}, beta={}, bias={})".format(IR_node.variable_name, self.parent_variable_name(IR_node), IR_node.get_attr('depth_radius'), IR_node.get_attr('alpha'), IR_node.get_attr('beta'), IR_node.get_attr('bias')))

    def emit_Mxnet_LRN(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Lrn(n.{}, nsize={}, alpha={}, beta={}, knorm={})".format(IR_node.variable_name, self.parent_variable_name(IR_node), IR_node.get_attr('size'), IR_node.get_attr('alpha'), IR_node.get_attr('beta'), IR_node.get_attr('k')))

    def emit_Mxnet_Activation(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Activation(n.{}, act_type={})".format(IR_node.variable_name, self.parent_variable_name(IR_node), IR_node.get_attr('act')))

    def emit_BiFPN(self, IR_node):
        coef1 = IR_node.get_attr('w1')
        coef2 = IR_node.get_attr('w2')
        coef3 = IR_node.get_attr('w3')
        eps = 0.000099
        if len(IR_node.in_edges) == 2:
            sumc = (coef1 + coef2 + eps)
            coef1 = coef1 / sumc
            coef2 = coef2 / sumc
        else:
            sumc = (coef1 + coef2 + coef3 + eps)
            coef1 = coef1 / sumc
            coef2 = coef2 / sumc
            coef3 = coef3 / sumc
        if len(IR_node.in_edges) == 3:
            dummy_name = IR_node.name.replace('/', '_') + "_dummy"
            parent1 = self.IR_graph.get_parent(IR_node.name, [0])
            assert parent1 is not None
            parent2 = self.IR_graph.get_parent(IR_node.name, [1])
            assert parent2 is not None
            input_nodes = [parent1.real_variable_name.replace('/', '_'), parent2.real_variable_name.replace('/', '_')]
            input_layers = ', '.join(('n.' + node_name) for node_name in input_nodes)
            self.add_body(1, "n.{:<15} = L.Eltwise({}, operation=1, coeff={}, ntop=1)".format(dummy_name, input_layers, [coef1, coef2]))
            parent3 = self.IR_graph.get_parent(IR_node.name, [2])
            assert parent3 is not None
            input_nodes = [dummy_name, parent3.real_variable_name.replace('/', '_')]
            input_layers = ''
            input_layers = ', '.join(('n.' + node_name) for node_name in input_nodes)
            self.add_body(1, "n.{:<15} = L.Eltwise({}, operation=1, coeff={}, ntop=1)".format(IR_node.variable_name, input_layers, [1, coef3]))
        else:
            parent1 = self.IR_graph.get_parent(IR_node.name, [0])
            assert parent1 is not None
            parent2 = self.IR_graph.get_parent(IR_node.name, [1])
            assert parent2 is not None
            input_nodes = [parent1.real_variable_name.replace('/', '_'), parent2.real_variable_name.replace('/', '_')]
            input_layers = ', '.join(('n.' + node_name) for node_name in input_nodes)
            self.add_body(1, "n.{:<15} = L.Eltwise({}, operation=1, coeff={}, ntop=1)".format(IR_node.variable_name, input_layers, [coef1, coef2]))

    def emit_BroadCastMul(self, IR_node):
        parents = []
        for num in range(0, len(IR_node.in_edges)):
            parent = self.IR_graph.get_parent(IR_node.name, [num])
            assert parent is not None
            parents.append(parent.real_variable_name)
        input_layers = ', '.join(parents)
        if len(input_layers) < 2:
            logger.error("number of inputs is less than 2 for broadcast mul node %s", IR_node.name)
            raise ValueError()
        if len(IR_node.in_edges) > 2:
            raise NotImplementedError("Not supported more than two inputs to broadcast mul")
        in_edges = [self.IR_graph.get_node(IR_node.in_edges[0]).real_name, self.IR_graph.get_node(IR_node.in_edges[1]).real_name]
        in_edges_shapes = [self.IR_graph.get_node(IR_node.in_edges[0]).get_attr('_output_shapes'), self.IR_graph.get_node(IR_node.in_edges[1]).get_attr('_output_shapes')]
        in_edges_shapes = [[j.size for j in i[0].dim] for i in in_edges_shapes]
        input_layers = parents
        input_node0 = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node0 is not None
        shape0 = clast2first(shape_to_list(input_node0.get_attr("_output_shapes")[0]))
        input_node1 = self.IR_graph.get_parent(IR_node.name, [1])
        assert input_node1 is not None
        shape1 = clast2first(shape_to_list(input_node1.get_attr("_output_shapes")[0]))
        broadcast0 = False
        broadcast1 = False
        final_shape = []
        for i in range(len(shape0)):
            if shape0[i] != shape1[i] and shape0[i] == 1:
                broadcast0 = True
                final_shape.append(shape1[i])
            elif shape0[i] != shape1[i] and shape1[i] == 1:
                broadcast1 = True
                final_shape.append(shape0[i])
            else:
                final_shape.append(shape0[i])
        if broadcast0:
            dim_str = "'dim': {}".format(final_shape)
            dim_str = "{'shape': { " + dim_str + '} }'
            broadcast0_name = IR_node.variable_name + '_broadcast0'
            self.add_body(1, "n.{:<15} = L.Broadcast(n.{},broadcast_param = {}, ntop=1)".format(broadcast0_name, self.parent_variable_name(IR_node, [0]), dim_str))
            input_layers[0] = broadcast0_name
        if broadcast1:
            dim_str = "'dim': {}".format(final_shape)
            dim_str = "{'shape': { " + dim_str + '} }'
            broadcast1_name = IR_node.variable_name + '_broadcast1'
            self.add_body(1, "n.{:<15} = L.Broadcast(n.{},broadcast_param = {}, ntop=1)".format(broadcast1_name, self.parent_variable_name(IR_node, [1]), dim_str))
            input_layers[1] = broadcast1_name
        new_name = IR_node.variable_name + "_bc"
        self.add_body(1, "n.{:<15} = L.Eltwise(n.{},n.{}, operation=0, ntop=1)".format(new_name, input_layers[0], input_layers[1]))
        IR_node.real_name = new_name

    def emit_Add(self, IR_node):
        Precision = {
            'INT8': 0,
            'INT16': 1,
            'INT32': 2,
            'FP32': 3,
            'ZERO': 999,
        }
        prec = Precision[IR_node.get_attr("new_precision", 'ZERO')]
        if len(IR_node.in_edges) > 2:
            raise NotImplementedError("Not supported more than two inputs to elementwise add: {}".format(IR_node.name))
        input_layers = []
        for num in range(len(IR_node.in_edges)):
            parent = self.IR_graph.get_parent(IR_node.name, [num])
            assert parent is not None
            input_layers.append(parent.real_variable_name)
        if len(input_layers) < 2:
            logger.error("number of inputs is less than 2 for Eltwise node %s", IR_node.name)
            raise ValueError("number of inputs is less than 2 for Eltwise node: {}".format(IR_node.name))
        in_edges = [self.IR_graph.get_node(edge).real_name for edge in IR_node.in_edges]
        in_edges_shapes = [self.IR_graph.get_node(edge).get_attr('_output_shapes') for edge in IR_node.in_edges]
        in_edges_shapes = [[j.size for j in i[0].dim] for i in in_edges_shapes]
        in_edges_shapes = [clast2first(i) for i in in_edges_shapes]
        # If both parents are same, add padding to one of the parent
        if in_edges[0] == in_edges[1]:
            padding_layer_name = in_edges[1] + "_padding"
            self.add_body(1, "n.{:<15} = L.Padding(n.{}, padding_param=dict(pad_l={},pad_r={},pad_t={}, pad_b={},pad_method={},pad_value={}), ntop=1)".format(padding_layer_name, in_edges[1], 0, 0, 0, 0, 0, 0))
            self.add_body(1, "n.{:<15} = L.Eltwise(bottom={}, operation=1, ntop=1)".format(IR_node.variable_name, [in_edges[0], padding_layer_name]))
        else:
            max_len = len(max(in_edges_shapes, key=len))
            for i, shape in enumerate(in_edges_shapes):
                if len(shape) != max_len:
                    new_shape = [1] * (max_len - len(shape)) + shape
                    dim_str = generate_reshape_param(new_shape)
                    reshape1 = IR_node.real_name + "_reshape1"
                    self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape1, self.parent_variable_name(IR_node, [i]), dim_str))
                    input_layers[i] = reshape1

            if prec == Precision['INT16']:
                pre_dvshift_0 = IR_node.variable_name + "_pre_dvshift_0"
                pre_dvshift_1 = IR_node.variable_name + "_pre_dvshift_1"
                post_dvshift = IR_node.variable_name + "_post_dvshift"
                self.add_body(1, "n.{:<15} = L.DVShift(n.{}, dvshift_param=dict(shift={}), ntop=1, adj_param=dict(precision=[{}]))".format(pre_dvshift_0, input_layers[0], 0, prec))
                self.add_body(1, "n.{:<15} = L.DVShift(n.{}, dvshift_param=dict(shift={}), ntop=1, adj_param=dict(precision=[{}]))".format(pre_dvshift_1, input_layers[1], 0, prec))
                self.add_body(1, "n.{:<15} = L.Eltwise(n.{},n.{}, operation=1, ntop=1, adj_param=dict(precision=[{}]))".format(IR_node.variable_name, pre_dvshift_0, pre_dvshift_1, prec))
                self.add_body(1, "n.{:<15} = L.DVShift(n.{}, dvshift_param=dict(shift={}), ntop=1)".format(post_dvshift, IR_node.variable_name, 0))
                IR_node.real_name = post_dvshift
            else:
                self.add_body(1, "n.{:<15} = L.Eltwise(n.{},n.{}, operation=1, ntop=1, adj_param=dict(precision=[{}]))".format(IR_node.variable_name, input_layers[0], input_layers[1], prec))

    def emit_Flatten(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Flatten(n.{})".format(
        IR_node.variable_name,
        self.parent_variable_name(IR_node),
    ))

    def emit_Min(self, IR_node):
        if len(IR_node.in_edges) > 2:
            raise NotImplementedError("Not supported more than two inputs to elementwise min: {}".format(IR_node.name))

        input_layers = []
        for num in range(len(IR_node.in_edges)):
            parent = self.IR_graph.get_parent(IR_node.name, [num])
            assert parent is not None
            input_layers.append(parent.real_variable_name)
        # Case: One of the input is a constant tensor
        if (IR_node.get_attr('const_tensor')):
            # input_layers = [self.IR_graph.get_parent(IR_node.name, [num]).real_variable_name for num in range(len(IR_node.in_edges))]
            input_shape = clast2first([i.size for i in IR_node.get_attr('_output_shapes')[0].dim])
            dummy_layer_name = IR_node.variable_name + "_const_tensor"
            for key in self.weights_dict[IR_node.name]:
                self.weights_dict[IR_node.name][key] = np.broadcast_to(self.weights_dict[IR_node.name][key], input_shape)
            self.weights_dict[dummy_layer_name] = self.weights_dict.pop(IR_node.name)
            self.add_body(1, "n.{:<15} = L.DVTensor(dvtensor_param=dict(tensortype={}, value= {},shape=dict(dim={})), ntop=1)".format(dummy_layer_name, 2, 0, '[1]' if str(input_shape) == '[]' else str(input_shape)))
            self.add_body(1, "n.{:<15} = L.Eltwise(n.{},n.{}, operation=3, ntop=1)".format(IR_node.variable_name, input_layers[0], dummy_layer_name))
            return

        if len(input_layers) < 2:
            logger.error("number of inputs is less than 2 for Eltwise node %s", IR_node.name)
            raise ValueError("number of inputs is less than 2 for Eltwise node: {}".format(IR_node.name))

        in_edges = [self.IR_graph.get_node(IR_node.in_edges[0]).real_name, self.IR_graph.get_node(IR_node.in_edges[1]).real_name]
        in_edges_shapes = [self.IR_graph.get_node(IR_node.in_edges[0]).get_attr('_output_shapes'), self.IR_graph.get_node(IR_node.in_edges[1]).get_attr('_output_shapes')]
        in_edges_shapes = [[j.size for j in i[0].dim] for i in in_edges_shapes]
        in_edges_shapes = [clast2first(i) for i in in_edges_shapes]

        if len(in_edges_shapes[0]) < len(in_edges_shapes[1]):
            new_shape = [1] * (len(in_edges_shapes[1]) - len(in_edges_shapes[0])) + in_edges_shapes[0]
            dim_str = "'dim': {}".format(new_shape)
            dim_str = " reshape_param={'shape': { " + dim_str + '} }'
            reshape1 = IR_node.real_name + "_reshape1"
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape1, self.parent_variable_name(IR_node, [0]), dim_str))
            input_layers[0] = reshape1
        if len(in_edges_shapes[0]) > len(in_edges_shapes[1]):
            new_shape = [1] * (len(in_edges_shapes[0]) - len(in_edges_shapes[1])) + in_edges_shapes[1]
            dim_str = "'dim': {}".format(new_shape)
            dim_str = " reshape_param={'shape': { " + dim_str + '} }'
            reshape1 = IR_node.real_name + "_reshape1"
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape1, self.parent_variable_name(IR_node, [1]), dim_str))
            input_layers[1] = reshape1
        self.add_body(1, "n.{:<15} = L.Eltwise(n.{},n.{}, operation=3, ntop=1)".format(IR_node.variable_name, input_layers[0], input_layers[1]))

    def emit_Floor(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Floor(n.{}, floor_param=dict(shift=0), ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_argmax(self, IR_node):
        axis = IR_node.get_attr('axis')
        select_last_index = 0
        if IR_node.get_attr('select_last_index') is not None:
            select_last_index = IR_node.get_attr('select_last_index')
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        input_node_shape = shape_to_list(input_node.get_attr('_output_shapes')[0])
        if len(input_node_shape) == 4:
            axis = NHWC_NCHW[IR_node.layer.attr['axis'].i]
        elif len(input_node_shape) == 3:
            if axis == 1:
                axis = 2
            elif axis == 2:
                axis = 1
        if axis != 1:
            logger.error("argmax: axis != 1 is not supported by compiler")
        argmax_layer = IR_node.variable_name
        if not IR_node.get_attr('keepdims', True):
            argmax_layer = argmax_layer + "_ara_argmax"
        self.add_body(1, "n.{:<15} = L.ArgMax(n.{},axis={},select_last_index={})".format(argmax_layer, self.parent_variable_name(IR_node), axis, select_last_index))
        if not IR_node.get_attr('keepdims', True):
            shape = IR_node.get_attr("_output_shapes")[0]
            shape = shape_to_list(shape)
            shape = clast2first(shape)
            dim_str = "'dim': {}".format(shape)
            dim_str = "{'shape': { " + dim_str + '} }'
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, reshape_param={}) ".format(IR_node.variable_name, argmax_layer, dim_str))
            # IR_node.real_name = IR_node.real_name + '_reshape'

    def emit_Tf_ArgMax(self, IR_node):
        axis = IR_node.get_attr('axis')
        select_last_index = 0
        if IR_node.get_attr('select_last_index') is not None:
            select_last_index = IR_node.get_attr('select_last_index')
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        input_node_shape = shape_to_list(input_node.get_attr('_output_shapes')[0])
        if len(input_node_shape) == 4:
            axis = NHWC_NCHW[IR_node.layer.attr['axis'].i]
        elif len(input_node_shape) == 3:
            if axis == 1:
                axis = 2
            elif axis == 2:
                axis = 1
        argmax_layer = IR_node.variable_name
        if not IR_node.get_attr('keepdims', True):
            argmax_layer = argmax_layer + "_ara_argmax"
        self.add_body(1, "n.{:<15} = L.Tf_Argmax(n.{},dimension=dict(shape=dict(dims={},shape={})),select_last_index={},ntop=1)".format(argmax_layer, self.parent_variable_name(IR_node), 1, [IR_node.get_attr('axis')], select_last_index))
        if not IR_node.get_attr('keepdims', True):
            shape = IR_node.get_attr("_output_shapes")[0]
            shape = shape_to_list(shape)
            shape = clast2first(shape)
            dim_str = "'dim': {}".format(shape)
            dim_str = "{'shape': { " + dim_str + '} }'
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, reshape_param={}) ".format(IR_node.variable_name, argmax_layer, dim_str))

    def emit_Squeeze(self, IR_node):
        shape = IR_node.get_attr("_output_shapes")[0]
        shape = shape_to_list(shape)
        if not shape:
            shape = IR_node.get_attr("shape")
        if shape:
            logger.info("shape after squeeze is %s", shape)
            caffe_shape = deepcopy(shape)
            if len(caffe_shape) > 1:
                channel = caffe_shape.pop(1)
                caffe_shape.append(channel)
            logger.info("caffe shape for squeeze is %s", caffe_shape)
            dim_str = "'dim': {}".format(caffe_shape)
            dim_str = " reshape_param={'shape': { " + dim_str + '} }'
            logger.debug("reshape shape is %s", str(shape))
            input_node = self.IR_graph.get_parent(IR_node.name, [0])
            assert input_node is not None
            input_shape = shape_to_list(input_node.get_attr("_output_shapes")[0])
            if IR_node.get_attr('data_format') == 'NHWC' or IR_node.get_attr('data_format') == 'NWC' or IR_node.get_attr('data_format') == 'NC':
                permute1 = False
                permute2 = False
                if len(input_shape) == 4:
                    permute1 = [0, 2, 3, 1]
                elif len(input_shape) == 3:
                    permute1 = [0, 2, 1]
                if len(shape) == 4:
                    permute2 = [0, 3, 1, 2]
                elif len(shape) == 3:
                    permute2 = [0, 2, 1]
                perm_tf = self.parent_variable_name(IR_node)
                if permute1:
                    perm_tf = IR_node.variable_name + '_perm_tf'
                    perm_order_str = '{' + "'order': {}".format(permute1) + '}'
                    logger.debug("transpose layer emit %s, %s, %s", IR_node.variable_name, self.parent_variable_name(IR_node), perm_order_str)
                    self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(perm_tf, self.parent_variable_name(IR_node), perm_order_str))
                dim_str = "'dim': {}".format(shape)
                dim_str = " reshape_param={'shape': { " + dim_str + '} }'
                reshape_layer_name = IR_node.variable_name
                if permute2:
                    reshape_layer_name = reshape_layer_name + "_ara_reshape"
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape_layer_name, perm_tf, dim_str))
                if permute2:
                    # perm_caffe = IR_node.variable_name + '_perm_caffe'
                    perm_order_str = '{' + "'order': {}".format(permute2) + '}'
                    logger.debug("transpose layer emitt %s, %s, %s", IR_node.variable_name, self.parent_variable_name(IR_node), perm_order_str)
                    self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(IR_node.variable_name, reshape_layer_name, perm_order_str))
                    # IR_node.real_name = perm_caffe
            elif shape:
                d_format = IR_node.get_attr('data_format')
                if d_format is None or d_format != 'NCW':
                    if len(shape) > 2:
                        channels = shape.pop(len(shape) - 1)
                        shape.insert(1, channels)
                dim_str = "'dim': {}".format(shape)
                dim_str = " reshape_param={'shape': { " + dim_str + '} }'
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(IR_node.variable_name, self.parent_variable_name(IR_node), dim_str))
            else:
                parent = self.IR_graph.get_parent(IR_node.name, [0])
                assert parent is not None
                IR_node.real_name = parent.real_name

    def emit_DVTensor(self, IR_node):
        mode = 0
        if IR_node.name in self.weights_dict:
            weight = self.weights_dict[IR_node.name]
            un = np.unique(weight, return_inverse=False)
            mode = 0 if (un.size == 1 and un[0] == 0) else 2
        out_shape = clast2first(shape_to_list(IR_node.get_attr('_output_shapes')[0]))
        pass_through = IR_node.get_attr('pass_through', False)
        self.add_body(1, "n.{:<15} = L.DVTensor(dvtensor_param=dict(tensortype={}, value= {}, shape=dict(dim={}), pass_through={}), ntop=1)".format(IR_node.variable_name, mode, 0, str(out_shape), pass_through))
        if mode == 2:
            self.weights_dict[IR_node.name] = self.weights_dict.pop(IR_node.name)
    
    def emit_DVCast(self, IR_node):
        Precision = {
            'INT8': 0,
            'INT16': 1,
            'INT32': 2,
            'FP32': 3,
            'UINT8': 8,
            'ZERO': 999,
        }
        src_precision = Precision['FP32']
        dst_precision = Precision['FP32']
        scale = IR_node.get_attr("scale")
        offset = IR_node.get_attr("offset")
        is_pqn = IR_node.get_attr("is_pqn", False)
        if IR_node.get_attr("src_precision").upper() == 'UINT8':
            src_precision = Precision['INT8']
            sign = 1
            offset -= 128
        elif IR_node.get_attr("src_precision").upper() == 'INT8':
            src_precision = Precision['INT8']
            sign = 1
        if IR_node.get_attr("dst_precision").upper() == 'UINT8':
            dst_precision = Precision['INT8']
            sign = 1
            offset -= 128
        elif IR_node.get_attr("dst_precision").upper() == 'INT8':
            dst_precision = Precision['INT8']
            sign = 1
        scale = IR_node.get_attr("scale")
        offset = IR_node.get_attr("offset") - 128
        self.add_body(1, "n.{:<15} = L.DVCast(n.{}, dvcast_param=dict(src_precision={}, dst_precision={}, scale={}, offset={}, sign={}, is_pqn={}), dataPrecision={},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), src_precision, dst_precision, scale, offset, sign, is_pqn, dst_precision))
    
    def emit_DVShift(self, IR_node):
        shift = IR_node.get_attr('shift')
        self.add_body(1, "n.{:<15} = L.DVShift(n.{}, dvshift_param=dict(shift={}), ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), shift))

    def emit_Concat(self, IR_node):
        Precision = {
            'INT8': 0,
            'INT16': 1,
            'INT32': 2,
            'FP32': 3,
            'ZERO': 999,
        }
        prec = Precision[IR_node.get_attr("new_precision", 'ZERO')]
        output_shape = [int(i.size) for i in IR_node.get_attr('_output_shapes')[0].dim]
        # NCHW to NHWC
        # axis_array = (0, 2, 3, 1)#(2, 3, 1, 0)
        # axis_array = (0, 2, 3, 1)#(2, 3, 1, 0)
        axis = IR_node.get_attr('axis')
        if len(output_shape) > 2:
            axis = self._convert_axis_cfirst(IR_node, IR_node.get_attr('axis'))

        if (IR_node.get_attr('const_tensor')):
            dummy_layer_name = IR_node.variable_name + "_const_tensor"
            input_layers = [self.IR_graph.get_node(i).real_variable_name for i in IR_node.in_edges]
            for i, j in enumerate(self.weights_dict[IR_node.name]):
                self.add_body(1, "n.{:<15} = L.DVTensor(dvtensor_param=dict(tensortype={}, value= {},shape=dict(dim={})), ntop=1)".format(dummy_layer_name + '_' + str(i), 2, 0, str(list(j.shape))))
                self.weights_dict[dummy_layer_name + '_' + str(i)] = {'weights': j}
                input_layers.insert(IR_node.get_attr('dvtensor_pos')[i], dummy_layer_name + '_' + str(i))
            input_layers = ', '.join(['n.' + i for i in input_layers])
            self.weights_dict.pop(IR_node.name)
            self.add_body(1, "n.{:<15} = L.Concat({}, axis={})".format(IR_node.variable_name, input_layers, axis))
            return
        input_layers = [self.IR_graph.get_node(edge).real_variable_name for edge in IR_node.in_edges]
        for in_node in IR_node.in_edges:
            if not self.is_valid_node(self.IR_graph.get_node(in_node).type):
                parent = self.IR_graph.get_parent(IR_node.name, [0])
                assert parent is not None
                IR_node.real_name = parent.real_name
                logger.warn("some of the inputs are not well defined - not adding entry in prototxt for node %s. in_node details: %s real_name: %s", IR_node.name, in_node, IR_node.real_name)
                return
        if prec == Precision['INT16']:
            for idx, layer in enumerate(input_layers):
                pre_dvshift_idx = IR_node.variable_name + f"_pre_dvshift_{idx}"
                self.add_body(1, "n.{:<15} = L.DVShift(n.{}, dvshift_param=dict(shift={}), ntop=1, adj_param=dict(precision=[{}]))".format(pre_dvshift_idx, input_layers[idx], 0, prec))
                input_layers[idx] = pre_dvshift_idx
        input_layers = ', '.join('n.' + layer for layer in input_layers)
            
        self.add_body(1, "n.{:<15} = L.Concat({}, axis={}, adj_param=dict(precision=[{}]))".format(IR_node.variable_name, input_layers, axis, prec))
        if prec == Precision['INT16']:
            post_dvshift = IR_node.variable_name + "_post_dvshift"
            self.add_body(1, "n.{:<15} = L.DVShift(n.{}, dvshift_param=dict(shift={}), ntop=1)".format(post_dvshift, IR_node.variable_name, 0))
            IR_node.real_name = post_dvshift

    def emit_Sigmoid(self, IR_node):
        Precision = {
            'INT8': 0,
            'INT16': 1,
            'INT32': 2,
            'FP32': 3,
            'ZERO': 999,
        }
        prec = Precision[IR_node.get_attr("new_precision", 'ZERO')]
        self.add_body(1, "n.{:<15} = L.Sigmoid(n.{}, ntop=1, adj_param=dict(precision=[{}]))".format(IR_node.variable_name, self.parent_variable_name(IR_node), prec))
        if prec == Precision['INT16']:
            post_dvshift = IR_node.variable_name + "_post_dvshift"
            self.add_body(1, "n.{:<15} = L.DVShift(n.{}, dvshift_param=dict(shift={}), ntop=1)".format(post_dvshift, IR_node.variable_name, 0))
            IR_node.real_name = post_dvshift

    def emit_Relu(self, IR_node):
        in_place = False
        self.add_body(1, "n.{:<15} = L.ReLU(n.{}, in_place={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), in_place))

    def emit_HardSigmoid(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Lut(n.{}, type={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), '"hsig"'))

    def emit_Mxnet_LeakyRelu(self, IR_node):
        act_type = IR_node.get_attr('act_type')
        # elu and leaky
        if act_type == 0 or act_type == 2:
            self.add_body(1, "n.{:<15} = L.Mxnet_Leakyrelu(n.{}, slope={}, act_type={})".format(IR_node.variable_name, self.parent_variable_name(IR_node), IR_node.get_attr('slope'), act_type))
        # rrelu
        if act_type == 4:
            self.add_body(1, "n.{:<15} = L.Mxnet_Leakyrelu(n.{}, lower_bound={}, upper_bound={}, act_type={})".format(IR_node.variable_name, self.parent_variable_name(IR_node), IR_node.get_attr('lower_bound'), IR_node.get_attr('upper_bound'), act_type))
        # gelu and selu
        if act_type == 1 or act_type == 5:
            self.add_body(1, "n.{:<15} = L.Mxnet_Leakyrelu(n.{}, act_type={})".format(IR_node.variable_name, self.parent_variable_name(IR_node), act_type))
        # prelu
        if act_type == 3:
            self.add_body(1, "n.{:<15} = L.Mxnet_Leakyrelu(n.{}, act_type={})".format(IR_node.variable_name, self.parent_variable_name(IR_node), act_type))
            self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)

    def emit_LeakyRelu(self, IR_node):
        in_place = False
        if self.plt_target == "Ara2":
            self.add_body(1, "n.{:<15} = L.LeakyRelu(n.{}, leaky_relu_param=dict(alpha={}), ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), IR_node.IR_layer.attr['alpha'].f))
            return
        self.add_body(1, "n.{:<15} = L.ReLU(n.{}, in_place={}, negative_slope={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), in_place, IR_node.IR_layer.attr['alpha'].f))

    def emit_PRelu(self, IR_node):
        in_place = False
        self.add_body(1, "n.{:<15} = L.PReLU(n.{}, in_place={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), in_place))

    def emit_Tanh(self, IR_node):
        self.add_body(1, "n.{:<15} = L.TanH(n.{}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_ONNX_Tanh(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Onnx_Tanh(n.{}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Softmax(self, IR_node):
        axis = IR_node.layer.attr['axis'].i
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        input_node_shape = shape_to_list(input_node.get_attr('_output_shapes')[0])
        if len(input_node_shape) == 4:
            axis = NHWC_NCHW[axis]
        elif len(input_node_shape) == 3:
            if axis == 1:
                axis = 2
            elif axis == 2:
                axis = 1
        self.add_body(1, "n.{:<15} = L.Softmax(n.{},axis={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), axis))

    def emit_Tf_Softmax(self, IR_node):
        axis = IR_node.layer.attr['axis'].i
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        input_node_shape = shape_to_list(input_node.get_attr('_output_shapes')[0])
        if len(input_node_shape) == 4:
            axis = NHWC_NCHW[axis]
        self.add_body(1, "n.{:<15} = L.Tf_Softmax(n.{}, ntop=1)".format(
        IR_node.variable_name,
        self.parent_variable_name(IR_node),
    ))

    def emit_Onnx_PytorchOnnxSoftmax(self, IR_node):
        dim = IR_node.layer.attr['dim'].i
        self.add_body(1, "n.{:<15} = L.Softmax(n.{}, axis={}, ntop=1)".format(
        IR_node.variable_name,
        self.parent_variable_name(IR_node),
        dim,
    ))

    def emit_Mxnet_SoftmaxOutput(self, IR_node):
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        input_node_shape = shape_to_list(input_node.get_attr('_output_shapes')[0])
        multi_output = IR_node.get_attr('multi_output')
        preserve_shape = IR_node.get_attr('preserve_shape')
        if multi_output:
            multi_output = 1
        elif not preserve_shape and not multi_output:
            if len(input_node_shape) == 4 and version.parse(mx.__version__) >= version.parse('1.9.0'):
                multi_output = len(input_node_shape) - 1
                preserve_shape = 1
        elif preserve_shape:
            multi_output = len(input_node_shape) - 1
        self.add_body(1, "n.{:<15} = L.Mxnet_Softmaxoutput(n.{},preserve_shape={},multi_output={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), preserve_shape, multi_output))

    def emit_Mxnet_Softmaxactivation(self, IR_node):
        mode_enum = {'channel': 0, 'instance': 1}
        self.add_body(1, "n.{:<15} = L.Mxnet_Softmaxactivation(n.{},mode={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), mode_enum[IR_node.get_attr('mode')]))

    def emit_Mxnet_InstanceNorm(self, IR_node):
        eps = IR_node.get_attr('eps')
        self.add_body(1, "n.{:<15} = L.Mxnet_Instancenorm(n.{},eps={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), eps))

    def emit_Pad(self, IR_node):
        pads = IR_node.get_attr('pads')

        # TODO; get padding type and value; raise exception if stack doesn't support
        if pads[3] > 0 or pads[7] > 0:
            logger.warn("channel padding is not yet supported by DVNet")
            # a -> b | a -> x1[Pad_spatial] -> x[DummyPermute] -> y[Pad] -> z[Permute] -> b
            self.add_body(
          1,
          "n.{:<15} = L.Padding(n.{}, padding_param=dict(pad_l={},pad_r={},pad_t={}, pad_b={},pad_method={},pad_value={}), ntop=1)".format(
              IR_node.variable_name + "_ara_pad",
              self.parent_variable_name(IR_node),
              pads[2],
              pads[6],
              pads[1],
              pads[5],
              0,  #pad method is CONSTANT enum
              0))  #pad value
            logger.debug("output shape is " + str(IR_node.get_attr('_output_shapes')))
            perm_order = [0] * len(shape_to_list(IR_node.get_attr('_output_shapes')[0]))
            if len(perm_order) == 4:  # NCHW
                perm_order = [0, 3, 2, 1]
            elif len(perm_order) == 3:  # NCW
                perm_order = [0, 2, 1]
            perm_order_str = '{' + "'order': {}".format(perm_order) + '}'
            # Channel -- > Width permute
            perm1 = IR_node.variable_name + "_perm1"
            self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(perm1, IR_node.variable_name + "_ara_pad", perm_order_str))
            # Pad on width
            pad_channel = IR_node.variable_name + "_ch_pad"
            self.add_body(
          1,
          "n.{:<15} = L.Padding(n.{}, padding_param=dict(pad_l={},pad_r={},pad_t={}, pad_b={},pad_method={},pad_value={}), ntop=1)".format(
              pad_channel,
              perm1,
              pads[3],
              pads[7],
              0,
              0,
              0,  #pad method is CONSTANT enum
              0))  #pad value
            # Permute back
            perm2 = IR_node.variable_name
            self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(perm2, pad_channel, perm_order_str))
            # IR_node.real_name = perm2
        else:
            self.add_body(
          1,
          "n.{:<15} = L.Padding(n.{}, padding_param=dict(pad_l={},pad_r={},pad_t={}, pad_b={},pad_method={},pad_value={}), ntop=1)".format(
              IR_node.variable_name,
              self.parent_variable_name(IR_node),
              pads[2],
              pads[6],
              pads[1],
              pads[5],
              0,  #pad method is CONSTANT enum
              0))  #pad value

    def emit_Mxnet_Pad(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Pad(n.{}, mode={}, pad_width=dict(dims={},shape={}), constant_value={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), IR_node.get_attr('mode'), IR_node.get_attr('dim'), IR_node.get_attr('pads'), IR_node.get_attr('constant_values')))

    def reduction(self, IR_node, op, axes):
        Precision = {
            'INT8': 0,
            'INT16': 1,
            'INT32': 2,
            'FP32': 3,
            'ZERO': 999,
        }
        prec = Precision[IR_node.get_attr("new_precision", 'ZERO')]
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        input_node_shape = shape_to_list(input_node.get_attr('_output_shapes')[0])
        caffe_axes = [self._convert_axis_cfirst(IR_node, axis) for axis in axes]
        reduce_name = IR_node.variable_name
        if IR_node.get_attr('keepdims') == False:
            reduce_name = IR_node.variable_name + "_ara_reduce"
        reduction_input = self.parent_variable_name(IR_node)
        if prec == Precision['INT16']:
            pre_dvshift_0 = IR_node.variable_name + "_pre_dvshift_0"
            self.add_body(1, "n.{:<15} = L.DVShift(n.{}, dvshift_param=dict(shift={}), ntop=1, adj_param=dict(precision=[{}]))".format(pre_dvshift_0, self.parent_variable_name(IR_node), 0, prec))
            reduction_input = pre_dvshift_0
        self.add_body(1, "n.{:<15} = L.Reduction(n.{}, operation={}, axis={}, keep_dims={}, adj_param=dict(precision=[{}]), ntop=1)".format(reduce_name, reduction_input, op, caffe_axes, True, prec))
        if IR_node.get_attr('keepdims') == False:
            # TODO: remove this hack when kernels are added
            shape = IR_node.get_attr("_output_shapes")[0]
            shape = clast2first(shape_to_list(shape))
            dim_str = "'dim': {}".format(shape)
            dim_str = "{'shape': { " + dim_str + '} }'
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, reshape_param={}, ntop=1) ".format(IR_node.variable_name, reduce_name, dim_str))
            reduce_name = IR_node.variable_name
        if prec == Precision['INT16']:
            post_dvshift = IR_node.variable_name + "_post_dvshift"
            self.add_body(1, "n.{:<15} = L.DVShift(n.{}, dvshift_param=dict(shift={}), ntop=1)".format(post_dvshift, reduce_name, 0))
            IR_node.real_name = post_dvshift

    def emit_Mxnet_Mean(self, IR_node):
        exclude = IR_node.get_attr('exclude')
        keepdims = IR_node.get_attr('keepdims')
        old_axes = IR_node.get_attr('axis')
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        input_node_shape = shape_to_list(input_node.get_attr('_output_shapes')[0])
        axes = old_axes
        if len(input_node_shape) == 5:
            axes = [NDHWC_NCDHW[i] for i in axes]
        elif len(input_node_shape) == 4:
            axes = [NHWC_NCHW[i] for i in axes]
        elif len(input_node_shape) == 3:
            axes = [NWC_NCW[i] for i in axes]
        dimensions = len(axes)
        self.add_body(
        1,
        "n.{:<15} = L.Mxnet_Mean(n.{}, axis=dict(dims={}, shape={}),exclude={},keepdims={})".format(
            IR_node.variable_name,
            self.parent_variable_name(IR_node),
            len(axes),
            axes,
            0,  # TODO : fix this later in analyzer
            keepdims))

    def emit_ReduceMean(self, IR_node):
        # Convert NHWC (IR) to NCHW (Caffe): [0,1,2,3]->[0,3,1,2]
        old_axes = IR_node.get_attr('axes')
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        input_node_shape = shape_to_list(input_node.get_attr('_output_shapes')[0])
        axes = old_axes
        axes = [self._convert_axis_cfirst(IR_node, axis) for axis in axes]
        reshapable = True
        for axis in axes:
            if input_node_shape[axis] != 1:
                reshapable = False
        if reshapable:
            if not IR_node.get_attr('keepdims'):
                logger.info("redundant mean layer converted to reshape: %s", IR_node.name)
                shape = IR_node.get_attr("_output_shapes")[0]
                shape = shape_to_list(shape)
                shape = [shape[0]] + [shape[-1]] + shape[1:-1]
                dim_str = "'dim': {}".format(shape)
                dim_str = "{'shape': { " + dim_str + '} }'
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, reshape_param={}) ".format(IR_node.variable_name, self.parent_variable_name(IR_node), dim_str))
            else:
                logger.info("ignoring redundant mean layer: %s", IR_node.name)
                parent = self.IR_graph.get_parent(IR_node.name, [0])
                assert parent is not None
                IR_node.real_name = parent.real_name
        else:
            self.reduction(IR_node, 4, old_axes)

    def emit_ReduceMax(self, IR_node):
        # Convert NHWC (IR) to NCHW (Caffe): [0,1,2,3]->[0,3,1,2]
        Precision = {
            'INT8': 0,
            'INT16': 1,
            'INT32': 2,
            'FP32': 3,
            'ZERO': 999,
        }
        prec = Precision[IR_node.get_attr("new_precision", 'ZERO')]
        old_axes = IR_node.get_attr('axes')
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        input_node_shape = shape_to_list(input_node.get_attr('_output_shapes')[0])
        axes = old_axes
        axes = [self._convert_axis_cfirst(IR_node, _a) for _a in axes]
        reshapable = False
        for axis in axes:
            if input_node_shape[axis] != 1:
                reshapable = False
        if reshapable:
            if not IR_node.get_attr('keepdims'):
                logger.info("redundant max layer converted to reshape: %s", IR_node.name)
                shape = IR_node.get_attr("_output_shapes")[0]
                shape = shape_to_list(shape)
                shape = [shape[0]] + [shape[-1]] + shape[1:-1]
                dim_str = "'dim': {}".format(shape)
                dim_str = "{'shape': { " + dim_str + '} }'
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, reshape_param={}, adj_param=dict(precision=[{}])) ".format(IR_node.variable_name, self.parent_variable_name(IR_node), dim_str, prec))
            else:
                logger.info("ignoring redundant max layer: %s", IR_node.name)
                parent = self.IR_graph.get_parent(IR_node.name, [0])
                assert parent is not None
                IR_node.real_name = parent.real_name
        else:
            self.reduction(IR_node, 5, old_axes)

    def emit_Tf_Mean(self, IR_node):
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        input_node_shape = shape_to_list(input_node.get_attr('_output_shapes')[0])
        axes = IR_node.get_attr('axis')
        if len(input_node_shape) == 4:
            axes = [NHWC_NCHW[i] for i in axes]
        elif len(input_node_shape) == 3:
            axes = [NWC_NCW[i] for i in axes]
        self.add_body(1, "n.{:<15} = L.Tf_Mean(n.{}, tf_mean_param=dict(reduction_indices=dict(shape=dict(shape={},dims={})),keep_dims={}) ,ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), axes, len(axes), IR_node.get_attr('keep_dims')))

    def emit_ReduceSum(self, IR_node):
        # Convert NHWC (IR) to NCHW (Caffe): [0,1,2,3]->[0,3,1,2]
        old_axes = IR_node.get_attr('axes')
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        input_node_shape = shape_to_list(input_node.get_attr('_output_shapes')[0])
        axes = old_axes
        axes = [self._convert_axis_cfirst(IR_node, _a) for _a in axes]
        self.reduction(IR_node, 1, old_axes)

    def emit_Relu6(self, IR_node):
        in_place = False
        top_name = IR_node.variable_name
        if len(IR_node.out_edges) == 1 and self.IR_graph.get_node(IR_node.out_edges[0]).type == "clip":
            top_name = self.IR_graph.get_node(IR_node.out_edges[0]).variable_name
        self.add_body(1, "n.{:<15} = L.ReLU6(n.{}, in_place={}, ntop=1)".format(top_name, self.parent_variable_name(IR_node), in_place))
        # self.emit_Relu(IR_node)
    def emit_Tf_Relu6(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Relu6(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0])))

    def emit_Tf_SoftPlus(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Softplus(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0])))

    def emit_Tf_SoftSign(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Softsign(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0])))

    def emit_Tf_DepthwiseConv2dNative(self, IR_node):
        # stride
        if len(IR_node.get_attr('strides')) > 2:
            stride_h = IR_node.get_attr('strides')[1]
            stride_w = IR_node.get_attr('strides')[2]
        else:
            stride_h = IR_node.get_attr('strides')[0]
            stride_w = IR_node.get_attr('strides')[1]
        # dilation_rate
        dilation_h = 1
        dilation_w = 1
        if IR_node.get_attr('dilations') is not None:
            dilation_h = IR_node.get_attr('dilations')[1]
            dilation_w = IR_node.get_attr('dilations')[2]
        # padding
        padding = IR_node.get_attr('padding')
        padding_type = 1
        if padding == 'SAME':
            padding_type = 0
        # kernel_shape
        kernel_h = IR_node.get_attr('kernel_shape')[0]
        kernel_w = IR_node.get_attr('kernel_shape')[1]
        kernel_c = IR_node.get_attr('kernel_shape')[2]
        kernel_d = IR_node.get_attr('kernel_shape')[3]
        self.add_body(1, "n.{:<15} = L.Tf_Depthwiseconv2Dnative(n.{}, strides={}, dilations={}, padding_type={}, filter=dict(shape=dict(dims=4, shape={})))".format(IR_node.variable_name, self.parent_variable_name(IR_node), [stride_h, stride_w], [dilation_h, dilation_w], padding_type, [kernel_h, kernel_w, kernel_c, kernel_d]))
        self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], [3, 2, 0, 1])
        self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)

    def emit_DepthwiseConv(self, IR_node):
        self.emit_Conv(IR_node)

    def emit_Const(self, IR_node):
        pass

    def emit_Shape(self, IR_node):
        pass

    def emit_Identity(self, IR_node):
        pass

    # def emit_Cast(self, IR_node):
    #   pass

    def emit_Mxnet_Transpose(self, IR_node):
        axes = IR_node.get_attr('perm')
        dimensions = len(axes)
        self.add_body(1, "n.{:<15} = L.Mxnet_Transpose(n.{}, axes=dict(dims={}, shape={}))".format(IR_node.variable_name, self.parent_variable_name(IR_node), dimensions, axes))

    def emit_Transpose(self, IR_node):
        Precision = {
            'INT8': 0,
            'INT16': 1,
            'INT32': 2,
            'FP32': 3,
            'ZERO': 999,
        }
        prec = Precision[IR_node.get_attr("new_precision", 'ZERO')]
        perm_order = IR_node.get_attr("perm")
        NCHW_dict = {'N': 0, 'C': 1, 'H': 2, 'W': 3}
        NCW_dict = {'N': 0, 'C': 1, 'W': 2}

        def _apply_permute(ls, order):
            return [ls[i] for i in order]

        if IR_node.get_attr('data_format') == 'NHWC':
            if len(perm_order) == 4:
                permuted = _apply_permute('NHWC', perm_order)
                permuted = _apply_permute(permuted, [0, 3, 1, 2])
                perm_order = [NCHW_dict[i] for i in permuted]
            elif len(perm_order) == 3:
                permuted = _apply_permute('NWC', perm_order)
                permuted = _apply_permute(permuted, [0, 2, 1])
                perm_order = [NCW_dict[i] for i in permuted]
        need_layer = False
        for p in range(len(perm_order)):
            if p != perm_order[p]:
                need_layer = True
                break
        if need_layer:
            perm_order_str = '{' + "'order': {}".format(perm_order) + '}'
            logger.debug("transpose layer emit %s, %s, %s", IR_node.variable_name, self.parent_variable_name(IR_node), perm_order_str)
            permute_input = self.parent_variable_name(IR_node)
            if prec == Precision['INT16']:
                pre_dvshift_0 = IR_node.variable_name + "_pre_dvshift_0"
                self.add_body(1, "n.{:<15} = L.DVShift(n.{}, dvshift_param=dict(shift={}), ntop=1, adj_param=dict(precision=[{}]))".format(pre_dvshift_0, self.parent_variable_name(IR_node), 0, prec))
                permute_input = pre_dvshift_0
            self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1, adj_param=dict(precision=[{}]))".format(IR_node.variable_name, permute_input, perm_order_str, prec))
            if prec == Precision['INT16']:
                post_dvshift = IR_node.variable_name + "_post_dvshift"
                self.add_body(1, "n.{:<15} = L.DVShift(n.{}, dvshift_param=dict(shift={}), ntop=1)".format(post_dvshift, IR_node.variable_name, 0))
                IR_node.real_name = post_dvshift
        elif self.IR_graph.get_parent(IR_node.name, [0]) is None:
            logger.info("Ignoring the node from weights")
        else:
            parent = self.IR_graph.get_parent(IR_node.name, [0])
            assert parent is not None
            IR_node.real_name = parent.real_name

    def emit_Reshape(self, IR_node):
        Precision = {
            'INT8': 0,
            'INT16': 1,
            'INT32': 2,
            'FP32': 3,
            'ZERO': 999,
        }
        prec = Precision[IR_node.get_attr("new_precision", 'ZERO')]
        shape = IR_node.get_attr("_output_shapes")[0]
        shape = shape_to_list(shape)
        # special case of matmul connversion to conv
        sonnode = self.IR_graph.get_son(IR_node.name, [0])
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        if len(shape) == 2 and sonnode and sonnode.layer.op == 'Conv':
            logger.info("ignoring reshape before Matmul")
            IR_node.real_name = input_node.real_name
            return
        logger.debug("reshape shape is %s", str(shape))
        input_shape = shape_to_list(input_node.get_attr("_output_shapes")[0])
        input_layers = self.get_input_layers_ignoring_qdq(IR_node)
        if IR_node.get_attr('data_format') == 'NHWC' or IR_node.get_attr('data_format') == 'NWC' or IR_node.get_attr('data_format') == 'NC':
            permute1 = False
            permute2 = False
            if len(input_shape) == 4:
                permute1 = [0, 2, 3, 1]
            elif len(input_shape) == 3:
                permute1 = [0, 2, 1]
            if len(shape) == 4:
                permute2 = [0, 3, 1, 2]
            elif len(shape) == 3:
                permute2 = [0, 2, 1]
            perm_tf = input_layers[0]
            if permute1:
                perm_tf = IR_node.variable_name + '_perm_tf'
                perm_order_str = '{' + "'order': {}".format(permute1) + '}'
                logger.debug("transpose layer emit %s, %s, %s", IR_node.variable_name, input_layers[0], perm_order_str)
                self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(perm_tf, input_layers[0], perm_order_str))
            dim_str = "'dim': {}".format(shape)
            dim_str = " reshape_param={'shape': { " + dim_str + '} }'
            reshape_layer_name = IR_node.variable_name
            if permute2:
                reshape_layer_name = reshape_layer_name + "_ara_reshape"
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape_layer_name, perm_tf, dim_str))
            if permute2:
                # perm_caffe = IR_node.variable_name + '_perm_caffe'
                perm_order_str = '{' + "'order': {}".format(permute2) + '}'
                logger.debug("transpose layer emitt %s, %s, %s", IR_node.variable_name, input_layers[0], perm_order_str)
                self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(IR_node.variable_name, reshape_layer_name, perm_order_str))
                # IR_node.real_name = perm_caffe
        elif shape:
            d_format = IR_node.get_attr('data_format')
            if d_format is None or d_format != 'NCW':
                if len(shape) > 2:
                    channels = shape.pop(len(shape) - 1)
                    shape.insert(1, channels)
            dim_str = "'dim': {}".format(shape)
            dim_str = " reshape_param={'shape': { " + dim_str + '} }'
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {}, adj_param=dict(precision=[{}]))".format(sanitize_name(IR_node.variable_name), input_layers[0], dim_str, prec))
        else:
            IR_node.real_name = input_node.real_name

    def emit_Split(self, IR_node):
        output_layers = ', '.join(('n.' + self.IR_graph.get_node(edge).real_variable_name) for edge in IR_node.out_edges)
        axis = IR_node.layer.attr['axis'].i
        caffe_axis = self._convert_axis_cfirst(IR_node, axis)
        num_splits = IR_node.layer.attr['num_split'].i
        output_layers = []
        reshapes = []
        for i in range(num_splits):
            output_layers.append('n.' + IR_node.variable_name + str(i))
            reshapes.append((IR_node.variable_name + str(i), clast2first(shape_to_list(IR_node.get_attr('_output_shapes')[i]))))
        output_layers_list = output_layers
        output_layers = ','.join(output_layers)
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        input_node_shape = shape_to_list(input_node.get_attr('_output_shapes')[0])
        per_split_size = input_node_shape[axis] / num_splits
        assert per_split_size.is_integer()
        per_split_size = int(per_split_size)
        slice_point_list = list(range(per_split_size, per_split_size * num_splits, per_split_size))
        if len(slice_point_list) == 0:
            IR_node.real_name = input_node.real_name
            return
        self.add_body(1, "{} = L.Slice(n.{}, slice_param=dict(axis={}, slice_point={}), ntop={})".format(output_layers, self.parent_variable_name(IR_node), caffe_axis, slice_point_list, num_splits))
        if IR_node.get_attr('squeeze_axis', False):
            for i in range(num_splits):
                dim_str = "'dim': {}".format(reshapes[i][1])
                dim_str = " reshape_param={'shape': { " + dim_str + '} }'
                reshape_layer_variable_name = reshapes[i][0]
                self.add_body(1, "n.{:<15} = L.Reshape({}, {})".format(reshape_layer_variable_name, output_layers_list[i], dim_str))

    def emit_Tf_SplitV(self, IR_node):
        output_layers = ', '.join(('n.' + self.IR_graph.get_node(edge).real_variable_name) for edge in IR_node.out_edges)
        caffe_axis = NHWC_NCHW[IR_node.layer.attr['axis'].i]
        axis = IR_node.layer.attr['axis'].i
        num_splits = IR_node.layer.attr['split'].i
        split_size = IR_node.layer.attr['split_size'].list.i
        output_layers = []
        slice_point_list = []
        for i in range(num_splits):
            output_layers.append('n.' + IR_node.variable_name + '_out' + str(i))
            if i == 0:
                slice_point_list.append(split_size[0])
                # output_layers.append('n.' + IR_node.variable_name)
            else:
                slice_point_list.append(split_size[i] + slice_point_list[i - 1])
                # output_layers.append('n.' + IR_node.variable_name + '_' + str(i))
        slice_point_list.pop()
        output_layers = ','.join(output_layers)
        self.add_body(1, "{} = L.Slice(n.{}, slice_param=dict(axis={}, slice_point={}), ntop={})".format(output_layers, self.parent_variable_name(IR_node), caffe_axis, slice_point_list, num_splits))

    def emit_Slice(self, IR_node):
        starts = IR_node.get_attr('starts')
        tensor1 = self.parent_variable_name(IR_node, [0])
        shrink_axis_mask = IR_node.get_attr('shrink_axis_mask')
        strides = IR_node.get_attr('strides')
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        input_shape = shape_to_list(input_node.get_attr("_output_shapes")[0])
        offset = starts
        shape = shape_to_list(IR_node.get_attr('_output_shapes')[0])
        strides = [1] * len(starts) if not strides else strides
        if shrink_axis_mask and sum(shrink_axis_mask) > 0:
            old_shape = deepcopy(shape)
            for i in range(len(shrink_axis_mask)):
                if shrink_axis_mask[i] == 1:
                    old_shape.insert(i, 1)
            permute1 = False
            permute2 = False
            if len(input_shape) == 4:
                permute1 = [0, 2, 3, 1]
            elif len(input_shape) == 3:
                permute1 = [0, 2, 1]
            if len(shape) == 4:
                permute2 = [0, 3, 1, 2]
            elif len(shape) == 3:
                permute2 = [0, 2, 1]
            perm_tf = self.parent_variable_name(IR_node)
            out_name = IR_node.variable_name
            if permute1:
                perm_tf = IR_node.variable_name + '_perm_tf'
                perm_order_str = '{' + "'order': {}".format(permute1) + '}'
                logger.debug("transpose layer emit %s, %s, %s", IR_node.variable_name, self.parent_variable_name(IR_node), perm_order_str)
                self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(perm_tf, tensor1, perm_order_str))
                out_name = perm_tf
            if input_shape != old_shape:
                new_dim_str = "{}".format(str(old_shape)[1:-1])
                self.add_body(1, "n.{:<15} = L.Crop(n.{}, L.DummyData(shape=[dict(dim=[{}])], \
                        ntop=1), ntop=1, offset={},axis=0)".format(IR_node.variable_name + "_ara_crop", perm_tf, new_dim_str, offset))
                out_name = IR_node.variable_name + "_ara_crop"
            dim_str = "'dim': {}".format(shape)
            dim_str = " reshape_param={'shape': { " + dim_str + '} }'
            reshape_layer_variable_name = IR_node.variable_name
            if permute2:
                reshape_layer_variable_name = IR_node.variable_name + "_ara_reshape"
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape_layer_variable_name, out_name, dim_str))
            if permute2:
                # perm_caffe = IR_node.variable_name + '_perm_caffe'
                perm_order_str = '{' + "'order': {}".format(permute2) + '}'
                logger.debug("transpose layer emitt %s, %s, %s", IR_node.variable_name, self.parent_variable_name(IR_node), perm_order_str)
                self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(IR_node.variable_name, reshape_layer_variable_name, perm_order_str))
                # IR_node.real_name = perm_caffe
        else:
            # if IR_node.get_attr('data_format') != 'TNC':
            shape = clast2first(shape)
            offset = clast2first(offset)
            strides = clast2first(strides)
            input_shape = clast2first(input_shape)
            while len(shape) > 5:
                shape.pop(0)
            while len(offset) > 5:
                offset.pop(0)
            dim_str = "{}".format(shape)
            if sum(strides) > len(strides):
                stride_idx = -1
                for idx, i in enumerate(strides):
                    if stride_idx == -1 and i != 1:
                        stride_idx = idx
                        continue
                    if stride_idx != -1 and i != 1:
                        logger.error("stride in slice is only supported in one dimension")
                        raise NotImplementedError()
                col = (input_shape[stride_idx] / strides[stride_idx])
                if not col.is_integer():
                    logger.error("stride must perfectly divide input dim")
                    raise NotImplementedError()
                if col != shape[stride_idx]:
                    logger.error("end must span entire axis")
                    raise NotImplementedError()
                strided_shape = input_shape[:stride_idx] + [int(col), strides[stride_idx]] + input_shape[stride_idx + 1:]
                re_str = "'dim': {}".format(strided_shape)
                re_str = " reshape_param={'shape': { " + re_str + '} }'
                reshape1 = IR_node.variable_name + '_reshape1'
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape1, tensor1, re_str))
                dim_str = "{}".format(shape[:stride_idx] + [shape[stride_idx]] + [1] + shape[stride_idx + 1:])
                offset.insert(stride_idx, 0)
                self.add_body(1, "n.{:<15} = L.Crop(n.{}, L.DummyData(shape=[dict(dim={})], \
                            ntop=1), ntop=1, offset={},axis=0)".format(IR_node.variable_name + "_ara_crop", reshape1, dim_str, offset))
                # reshape2 = IR_node.variable_name + '_reshape2'
                oshape = "'dim': {}".format(shape)
                oshape = " reshape_param={'shape': { " + oshape + '} }'
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(IR_node.variable_name, IR_node.variable_name + "_ara_crop", oshape))
                # IR_node.real_name = reshape2
            else:
                self.add_body(1, "n.{:<15} = L.Crop(n.{}, L.DummyData(shape=[dict(dim={})], \
                            ntop=1), ntop=1, offset={},axis=0)".format(IR_node.variable_name, tensor1, dim_str, offset))

    def emit_IBN(self, IR_node):
        self.add_body(1, "n.{:<15} = L.IBN(n.{}, ibn_param=dict(in_eps={}, batch_norm_param=dict(eps={}, use_global_stats=True)), ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), IR_node.get_attr('in_eps'), IR_node.get_attr('bn_eps')))

    def emit_Pack(self, IR_node):
        axis = IR_node.layer.attr['axis'].i
        input_node_0 = self.IR_graph.get_parent(IR_node.name, [0])
        input_node_1 = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node_0 is not None
        assert input_node_1 is not None
        input_node_0_shape = shape_to_list(input_node_0.get_attr('_output_shapes')[0])
        input_node_1_shape = shape_to_list(input_node_1.get_attr('_output_shapes')[0])
        if len(input_node_0_shape) == 4:
            axis = NHWC_NCHW[axis]
        input_node_0_shape.insert(axis, 1)
        input_node_1_shape.insert(axis, 1)
        dim_str = "'dim': {}".format(input_node_0_shape)
        dim_str = " reshape_param={'shape': { " + dim_str + '} }'
        self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(IR_node.variable_name + '_reshape1', input_node_0.real_variable_name, dim_str))
        dim_str = "'dim': {}".format(input_node_1_shape)
        dim_str = " reshape_param={'shape': { " + dim_str + '} }'
        self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(IR_node.variable_name + '_reshape2', input_node_1.real_variable_name, dim_str))
        self.add_body(1, "n.{:<15} = L.Concat(bottom={}, axis={},ntop=1)".format(IR_node.variable_name, ', '.join(['n.' + IR_node.variable_name + '_reshape1', 'n.' + IR_node.variable_name + '_reshape2']), axis))

    def emit_Abs(self, IR_node):
        self.add_body(1, "n.{:<15} = L.AbsVal(n.{}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Mxnet_Elemwise_Mul(self, IR_node):
        input_layers = ', '.join(('n.' + self.IR_graph.get_node(edge).real_variable_name) for edge in IR_node.in_edges)
        self.add_body(1, "n.{:<15} = L.Eltwise({},operation=0)".format(IR_node.variable_name, input_layers))

    def emit_Mxnet_Minus_Scalar(self, IR_node):
        input_nodes = []
        for edge in IR_node.in_edges:
            if self.IR_graph.get_node(edge).type == 'Split':
                input_nodes.append(self.IR_graph.get_node(IR_node.name).real_variable_name)
            else:
                input_nodes.append(self.IR_graph.get_node(edge).real_variable_name)
        shape = IR_node.get_attr("_output_shapes")[0]
        shape = shape_to_list(shape)
        self.add_body(1, "n.{:<15} = L.BatchNorm(n.{}, eps={}, use_global_stats={}, ntop=1)".format(IR_node.variable_name, input_nodes[0], 0.0, self.phase == 'test'))
        self.weights_dict[IR_node.variable_name] = dict()
        self.weights_dict[IR_node.variable_name]['var'] = np.array([1] * shape[1], dtype=np.float32)
        self.weights_dict[IR_node.variable_name]['scale'] = np.array([1] * shape[1], dtype=np.float32)
        value = IR_node.get_attr('value')
        self.weights_dict[IR_node.variable_name]['mean'] = np.full(shape, value, dtype=np.float32)

    def emit_Sub(self, IR_node):
        Precision = {
            'INT8': 0,
            'INT16': 1,
            'INT32': 2,
            'FP32': 3,
            'ZERO': 999,
        }
        prec = Precision[IR_node.get_attr("new_precision", 'ZERO')]
        slice_output = False
        input_nodes = []
        for num in range(len(IR_node.in_edges)):
            parent = self.IR_graph.get_parent(IR_node.name, [num])
            assert parent is not None
            input_nodes.append(parent.real_variable_name)
        input_layers = ', '.join(('n.' + node_name) for node_name in input_nodes)
        if len(input_layers) == 0:
            raise ValueError("Error at Eltwise . Number of inputs is less than 2 for node " + str(IR_node.name))
        if len(IR_node.in_edges) == 1:
            self.add_body(1, "n.{:<15} = L.Eltwise({}, L.DummyData(shape=[dict(dim=[1])], data_filler=dict(type='constant', value=0), ntop=1), ntop=1)".format(IR_node.variable_name, input_layers))
            return
        if prec == Precision['INT16']:
            pre_dvshift_0 = IR_node.variable_name + "_pre_dvshift_0"
            pre_dvshift_1 = IR_node.variable_name + "_pre_dvshift_1"
            post_dvshift = IR_node.variable_name + "_post_dvshift"
            self.add_body(1, "n.{:<15} = L.DVShift(n.{}, dvshift_param=dict(shift={}), ntop=1, adj_param=dict(precision=[{}]))".format(pre_dvshift_0, input_nodes[0], 0, prec))
            self.add_body(1, "n.{:<15} = L.DVShift(n.{}, dvshift_param=dict(shift={}), ntop=1, adj_param=dict(precision=[{}]))".format(pre_dvshift_1, input_nodes[1], 0, prec))
            self.add_body(1, "n.{:<15} = L.Eltwise(n.{},n.{}, operation=5, ntop=1, adj_param=dict(precision=[{}]))".format(IR_node.variable_name, pre_dvshift_0, pre_dvshift_1, prec))
            self.add_body(1, "n.{:<15} = L.DVShift(n.{}, dvshift_param=dict(shift={}), ntop=1)".format(post_dvshift, IR_node.variable_name, 0))
            IR_node.real_name = post_dvshift
        else:
            self.add_body(1, "n.{:<15} = L.Eltwise({}, coeff = [1, -1], operation=5, ntop=1)".format(IR_node.variable_name, input_layers))

    def emit_Div(self, IR_node):
        if len(IR_node.in_edges) == 2:
            # layer1 = 'n.' + self.IR_graph.get_node(IR_node.in_edges[0]).real_variable_name
            input_layers = ', '.join(('n.' + self.IR_graph.get_node(edge).real_variable_name) for edge in IR_node.in_edges)
            # value = 1 / float(self.IR_graph.get_node(IR_node.in_edges[1]).value)
            # no point in having this, since the const node is already been added as value.
            if self.IR_graph.get_node(IR_node.in_edges[1]).layer.attr['value'].f != 0:
                self.IR_graph.get_node(IR_node.in_edges[1]).layer.attr['value'].f = 1 / self.IR_graph.get_node(IR_node.in_edges[1]).layer.attr['value'].f
            elif self.IR_graph.get_node(IR_node.in_edges[1]).layer.attr['y'].f != 0:
                self.IR_graph.get_node(IR_node.in_edges[1]).layer.attr['value'].f = 1 / self.IR_graph.get_node(IR_node.in_edges[1]).layer.attr['y'].f
            input_layers = ', '.join(('n.' + self.IR_graph.get_node(edge).real_variable_name) for edge in IR_node.in_edges)
            logger.info("ignoring div operator since divisor value is 1 %s", self.IR_graph.get_node(IR_node.in_edges[1]).layer.attr['value'].f)
            if self.IR_graph.get_node(IR_node.in_edges[1]).layer.attr['value'].f == 1.0:
                logger.info("ignoring div operator since divisor value is 1 %s", self.IR_graph.get_node(IR_node.in_edges[1]).layer.attr['value'].f)
                parent = self.IR_graph.get_parent(IR_node.name, [0])
                assert parent is not None
                IR_node.real_name = parent.real_name
                return
            # self.add_body(1, "n.{:<15} = L.Eltwise({}, L.DummyData(shape=[dict(dim=[1])], data_filler=dict(type='constant', value={}), ntop=1), operation=0, ntop=1)".format(
            self.add_body(1, "n.{:<15} = L.Eltwise({}, operation=0, ntop=1)".format(IR_node.variable_name, input_layers))
        elif len(IR_node.in_edges) == 1:
            layer1 = 'n.' + self.IR_graph.get_node(IR_node.in_edges[0]).real_variable_name
            self.add_body(1, "n.{:<15} = L.Eltwise({}, L.DummyData(shape=[dict(dim=[1])], data_filler=dict(type='constant', value=1), ntop=1), operation=0, ntop=1)".format(IR_node.variable_name, layer1))
        else:
            assert False

    # Added for different shaped tensors
    def emit_Onnx_div(self, IR_node):
        inp0 = IR_node.get_attr("input_shape")
        inp1 = IR_node.get_attr("input_shape1")
        old_out_shape = [i.size for i in IR_node.get_attr("_output_shapes")[0].dim]
        out_shape = clast2first(old_out_shape)
        # out_shape = [out_shape[0],out_shape[-1],out_shape[1],out_shape[2]]
        # print("### Inp ",inp0)
        # print("### Inp2 ",inp1)
        # print("### ",out_shape)
        dim_str = ""
        brod_input = ""
        other_input = ""
        if (inp0 == out_shape):
            if (len(inp0)==len(inp1)):
                dim_str = "'dim': {}".format(out_shape)
                brod_input = self.parent_variable_name(IR_node, [1])
                other_input = self.parent_variable_name(IR_node, [0])
                dim_str = "{'shape': { " + dim_str + '} }'
                broadcast0_name = IR_node.variable_name + '_broadcast0'
                self.add_body(1, "n.{:<15} = L.Broadcast(n.{},broadcast_param = {}, ntop=1)".format(broadcast0_name, brod_input, dim_str))
                self.add_body(1, "n.{:<15} = L.Mxnet_Elemwise_Div(n.{},n.{})".format(IR_node.variable_name, other_input, broadcast0_name))
            else:
                dim_str = "'dim': {}".format(out_shape)
                brod_input = self.parent_variable_name(IR_node, [1])
                other_input = self.parent_variable_name(IR_node, [0])
                dim_str = "{'shape': { " + dim_str + '} }'
                rshp_str = "'dim': {}".format([1]*(len(inp0)-1)+inp1[:])
                rshp_str = " reshape_param={'shape': { " + rshp_str + '} }'
                reshape_name = IR_node.real_name + "_reshape"
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape_name, brod_input, rshp_str))
                broadcast0_name = IR_node.variable_name + '_broadcast0'
                self.add_body(1, "n.{:<15} = L.Broadcast(n.{},broadcast_param = {}, ntop=1)".format(broadcast0_name, reshape_name, dim_str))
                self.add_body(1, "n.{:<15} = L.Mxnet_Elemwise_Div(n.{},n.{})".format(IR_node.variable_name, other_input, broadcast0_name))
        else:
            if (len(inp0)==len(inp1)):
                dim_str = "'dim': {}".format(out_shape)
                brod_input = self.parent_variable_name(IR_node, [0])
                other_input = self.parent_variable_name(IR_node, [1])
                dim_str = "{'shape': { " + dim_str + '} }'
                broadcast0_name = IR_node.variable_name + '_broadcast0'
                self.add_body(1, "n.{:<15} = L.Broadcast(n.{},broadcast_param = {}, ntop=1)".format(broadcast0_name, brod_input, dim_str))
                self.add_body(1, "n.{:<15} = L.Mxnet_Elemwise_Div(n.{},n.{})".format(IR_node.variable_name, broadcast0_name, other_input))
            else:
                dim_str = "'dim': {}".format(out_shape)
                brod_input = self.parent_variable_name(IR_node, [0])
                other_input = self.parent_variable_name(IR_node, [1])
                dim_str = "{'shape': { " + dim_str + '} }'
                rshp_str = "'dim': {}".format([1]*(len(inp1)-1)+inp0[:])
                rshp_str = " reshape_param={'shape': { " + rshp_str + '} }'
                reshape_name = IR_node.real_name + "_reshape"
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape_name, brod_input, rshp_str))
                broadcast0_name = IR_node.variable_name + '_broadcast0'
                self.add_body(1, "n.{:<15} = L.Broadcast(n.{},broadcast_param = {}, ntop=1)".format(broadcast0_name, reshape_name, dim_str))
                self.add_body(1, "n.{:<15} = L.Mxnet_Elemwise_Div(n.{},n.{})".format(IR_node.variable_name, broadcast0_name, other_input))
        # self.add_body(1, "n.{:<15} = L.Onnx_Div(n.{},n.{})".format(
        #    IR_node.variable_name,
        #     self.parent_variable_name(IR_node,[0]),
        #     self.parent_variable_name(IR_node,[1])
        #     ))
    def emit_Onnx_Neg(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Onnx_Neg(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    # def emit_Onnx_Pow(self,IR_node):
    #    print("### POW node ",IR_node)
    #    power = IR_node.get_attr('Y')
    #    print(power)
    def emit_PythonLayer(self, IR_node):
        import json
        layer_class = IR_node.get_attr('class_name')
        layer_args = IR_node.get_attr('layer_args')
        #        self.add_body(1, "from {} import {}".format(layer_class, layer_class))
        self.add_body(1, "n.{:<15} = L.Python(n.{}, name='{}', ntop=1, python_param={})".format(IR_node.variable_name, self.parent_variable_name(IR_node), IR_node.variable_name, json.dumps({'module': layer_class, 'layer': layer_class, 'param_str': layer_args}, indent=2)))

    def emit_Mul(self, IR_node):
        Precision = {
            'INT8': 0,
            'INT16': 1,
            'INT32': 2,
            'FP32': 3,
            'ZERO': 999,
        }
        prec = Precision[IR_node.get_attr("new_precision", 'ZERO')]
        if len(IR_node.in_edges) == 2:
            input_layers = [self.IR_graph.get_node(edge).real_variable_name for edge in IR_node.in_edges]
            in_edges = IR_node.in_edges
            in_edges_shapes = [self.IR_graph.get_node(IR_node.in_edges[0]).get_attr('_output_shapes'), self.IR_graph.get_node(IR_node.in_edges[1]).get_attr('_output_shapes')]
            in_edges_shapes = [[j.size for j in i[0].dim] for i in in_edges_shapes]
            in_edges_shapes = [clast2first(i) for i in in_edges_shapes]
            assert len(in_edges_shapes) == 2
            if len(in_edges_shapes[0]) < len(in_edges_shapes[1]):
                new_shape = [1] * (len(in_edges_shapes[1]) - len(in_edges_shapes[0])) + in_edges_shapes[0]
                dim_str = "'dim': {}".format(new_shape)
                dim_str = " reshape_param={'shape': { " + dim_str + '} }'
                reshape1 = IR_node.real_name + "_reshape1"
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape1, self.parent_variable_name(IR_node, [0]), dim_str))
                input_layers[0] = reshape1
            if len(in_edges_shapes[0]) > len(in_edges_shapes[1]):
                new_shape = [1] * (len(in_edges_shapes[0]) - len(in_edges_shapes[1])) + in_edges_shapes[1]
                dim_str = "'dim': {}".format(new_shape)
                dim_str = " reshape_param={'shape': { " + dim_str + '} }'
                reshape1 = IR_node.real_name + "_reshape1"
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape1, self.parent_variable_name(IR_node, [1]), dim_str))
                input_layers[1] = reshape1
            if prec == Precision['INT16']:
                pre_dvshift_0 = IR_node.variable_name + "_pre_dvshift_0"
                pre_dvshift_1 = IR_node.variable_name + "_pre_dvshift_1"
                post_dvshift = IR_node.variable_name + "_post_dvshift"
                self.add_body(1, "n.{:<15} = L.DVShift(n.{}, dvshift_param=dict(shift={}), ntop=1, adj_param=dict(precision=[{}]))".format(pre_dvshift_0, input_layers[0], 0, prec))
                self.add_body(1, "n.{:<15} = L.DVShift(n.{}, dvshift_param=dict(shift={}), ntop=1, adj_param=dict(precision=[{}]))".format(pre_dvshift_1, input_layers[1], 0, prec))
                self.add_body(1, "n.{:<15} = L.Eltwise(n.{},n.{}, operation=0, ntop=1, adj_param=dict(precision=[{}]))".format(IR_node.variable_name, pre_dvshift_0, pre_dvshift_1, prec))
                self.add_body(1, "n.{:<15} = L.DVShift(n.{}, dvshift_param=dict(shift={}), ntop=1)".format(post_dvshift, IR_node.variable_name, 0))
                IR_node.real_name = post_dvshift
            else:
                input_layers = ','.join(['n.' + i for i in input_layers])
                self.add_body(1, "n.{:<15} = L.Eltwise({}, operation=0, ntop=1, adj_param=dict(precision=[{}]))".format(IR_node.variable_name, input_layers, Precision[IR_node.get_attr("new_precision", 'ZERO')]))
        elif len(IR_node.in_edges) == 1:
            if (IR_node.get_attr('const_tensor')):
                input_layers = []
                for num in range(len(IR_node.in_edges)):
                    parent = self.IR_graph.get_parent(IR_node.name, [num])
                    assert parent is not None
                    input_layers.append(parent.real_variable_name)
                input_shape = [i.size for i in IR_node.get_attr('_output_shapes')[0].dim]
                dummy_layer_name = IR_node.variable_name + "_const_tensor"
                self.weights_dict[dummy_layer_name] = self.weights_dict.pop(IR_node.name)
                input_shape = [i for i in np.array(self.weights_dict[dummy_layer_name]["weights"]).shape]
                self.add_body(1, "n.{:<15} = L.DVTensor(dvtensor_param=dict(tensortype={}, value= {},shape=dict(dim={})), ntop=1)".format(dummy_layer_name, 2, 0, '[1]' if str(input_shape) == '[]' else str(input_shape)))
                self.add_body(1, "n.{:<15} = L.Eltwise(n.{},n.{}, operation=0, ntop=1, adj_param=dict(precision=[{}]))".format(IR_node.variable_name, input_layers[0], dummy_layer_name, Precision[IR_node.get_attr("new_precision", 'ZERO')]))
            else:
                self.emit_Scale(IR_node)
        else:
            assert False

    def emit_Mxnet_Abs(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Abs(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Mxnet_Sum(self, IR_node):
        if len(IR_node.get_attr("axis")) != 1:
            logger.error("Currently FP32 sum op supports single axis")
            raise NotImplementedError()
        self.add_body(1, "n.{:<15} = L.Mxnet_Sum(n.{}, axis = dict(dims={}, shape={}),keepdims = {}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), 1, [IR_node.get_attr("axis")[0]], int(IR_node.get_attr("keepdims"))))

    def emit_Mxnet_Log(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Log(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Mxnet_Multi_All_Finite(self, IR_node):
        input_layers = ', '.join(('n.' + self.IR_graph.get_node(edge).real_variable_name) for edge in IR_node.in_edges)
        self.add_body(1, "n.{:<15} = L.Mxnet_Multi_All_Finite({}, num_arrays={})".format(IR_node.variable_name, input_layers, IR_node.get_attr("num_arrays")))

    def emit_Mxnet_Reciprocal(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Reciprocal(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Mxnet_Square(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Square(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Mxnet_Erf(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Erf(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Mxnet_Gammaln(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Gammaln(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Mxnet_Rcbrt(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Rcbrt(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Mxnet_Sqrt(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Sqrt(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Mxnet_Rsqrt(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Rsqrt(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Mxnet_All_Finite(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_All_Finite(n.{}, init_output={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), IR_node.get_attr('init_output')))

    def emit_Mxnet_Expm1(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Expm1(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Mxnet_Log2(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Log2(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Mxnet_Cbrt(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Cbrt(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Mxnet_Gamma(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Gamma(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Mxnet_Logical_Not(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Logical_Not(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Mxnet_Negative(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Negative(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Mxnet_Sign(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Sign(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Mxnet_Exp(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Exp(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Mxnet_Log10(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Log10(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Tf_Mul(self, IR_node):
        input_layers = ', '.join(('n.' + self.IR_graph.get_node(edge).real_variable_name) for edge in IR_node.in_edges)
        self.add_body(1, "n.{:<15} = L.Tf_Mul({})".format(IR_node.variable_name, input_layers))

    def emit_Tf_Acos(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Acos(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Tf_Atanh(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Atanh(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Tf_Erf(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Erf(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Tf_Floor(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Floor(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Tf_Argmin(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Argmin(n.{},dimension=dict(shape=dict(dims={},shape={})),ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), 1, [IR_node.get_attr('axis')]))

    def emit_Tf_AvgPool(self, IR_node):
        pad_enum = {'SAME': 0, 'VALID': 1}
        self.add_body(1, "n.{:<15} = L.Tf_Avgpool(n.{},ksize={},strides={},padding='{}',data_format='NCHW',padding_type={},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), self.convert_nhwc_nchw(IR_node.get_attr('ksize')), self.convert_nhwc_nchw(IR_node.get_attr('strides')), IR_node.get_attr('padding'), pad_enum[IR_node.get_attr('padding')]))

    def emit_UpSampling2D(self, IR_node):
        scales = IR_node.get_attr('scales')
        scale = tuple(scales)[0]
        shape = IR_node.get_attr('_output_shapes')[0]
        shape = shape_to_list(shape)
        self.add_body(1, "n.{:<15} = L.Deconvolution(n.{}, convolution_param=dict(kernel_size={}, stride={}, pad={}, num_output={}, group={}, bias_term={}), param=[dict(lr_mult=0)], ntop=1)".format(IR_node.variable_name, IR_node.in_edges[0], 2 * scale - scale % 2, scale, int(math.ceil((scale - 1) / 2)), shape[-1], shape[-1], False))

    # def emit_Square(self, IR_node):
    #   input_layers = ', '.join(('n.' + self.IR_graph.get_node(edge).real_variable_name) for edge in IR_node.in_edges)
    #   self.add_body(1, "n.{:<15} = L.Square({}, ntop=1)".format(
    #     IR_node.variable_name,
    #     input_layers))

    def emit_Elu(self, IR_node):
        in_place = False
        self.add_body(1, "n.{:<15} = L.ELU(n.{},alpha={}, in_place={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), IR_node.IR_layer.attr['alpha'].f, in_place))

    def emit_clip(self, IR_node):
        in_place = False
        if IR_node.get_attr('a_max') == -1:
            self.add_body(1, "n.{:<15} = L.ReLU(n.{}, in_place={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), in_place))
        elif IR_node.get_attr('a_min') == 0.0 and IR_node.get_attr('a_max') <= 6.0:
            self.add_body(1, "n.{:<15} = L.ReLU6(n.{}, in_place={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), in_place))
        else:
            raise ValueError("Unsupported min and max values for clip")

    def emit_Mxnet_Elemwise_Sub(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Elemwise_Sub(n.{},n.{})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), self.parent_variable_name(IR_node, [1])))

    def emit_Mxnet_Elemwise_Div(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Elemwise_Div(n.{},n.{})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), self.parent_variable_name(IR_node, [1])))

    def emit_Mxnet_Add_N(self, IR_node):
        input_layers = ', '.join(('n.' + self.IR_graph.get_node(edge).real_variable_name) for edge in IR_node.in_edges)
        self.add_body(1, "n.{:<15} = L.Mxnet_Add_N({}, adj_param=dict(precision=[3]))".format(IR_node.variable_name, input_layers))

    def emit_Tf_Acosh(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Acosh(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Tf_Asin(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Asin(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Tf_Asinh(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Asinh(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    # def emit_Tf_bessel_i0(self, IR_node):
    #   self.add_body(1,"n.{:<15} = L.Tf_Besseli0(n.{},ntop=1)".format(
    #     IR_node.variable_name,
    #     self.parent_variable_name(IR_node)
    #     ))
    # def emit_Tf_cumsum(self, IR_node):
    #   self.add_body(1,"n.{:<15} = L.Tf_Cumsum(n.{},n.axis={},ntop=1)".format(
    #     IR_node.variable_name,
    #     self.parent_variable_name(IR_node),
    #     IR_node.get_attr('axis')
    #     ))

    def emit_Tf_divnonan(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Divnonan(n.{},n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), self.parent_variable_name(IR_node, [1])))

    def emit_Tf_less(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Less(n.{},n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), self.parent_variable_name(IR_node, [1])))

    def emit_Tf_floordiv(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Floordiv(n.{},n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), self.parent_variable_name(IR_node, [1])))

    def emit_Tf_Atan2(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Atan2(n.{}, n.{})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), self.parent_variable_name(IR_node, [1])))

    def emit_Tf_Equal(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Equal(n.{}, n.{})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), self.parent_variable_name(IR_node, [1])))

    def emit_Tf_Greaterequal(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Greaterequal(n.{}, n.{})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), self.parent_variable_name(IR_node, [1])))

    def emit_Tf_Fill(self, IR_node):
        temp = IR_node.get_attr('dims')
        dims = [temp[0], temp[3], temp[1], temp[2]]
        value = IR_node.get_attr('value')
        self.add_body(1, "n.{:<15} = L.Tf_Fill(dims={},value={})".format(IR_node.variable_name, dims, value))

    def emit_Tf_SpaceToBatchND(self, IR_node):
        block_shape = IR_node.get_attr('block_shape')
        padding = IR_node.get_attr('padding')
        self.add_body(1, "n.{:<15} = L.Tf_Spacetobatchnd(n.{},block_shape=dict(shape=dict(dims=2, shape={})),paddings=dict(shape=dict(dims=4, shape={})))".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), block_shape, padding))

    def emit_Tf_Floormod(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Floormod(n.{},n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), self.parent_variable_name(IR_node, [1])))

    def emit_Tf_Expm1(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Expm1(n.{})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0])))

    def emit_Tf_Abs(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Abs(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Tf_Atan(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Atan(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Tf_Cosh(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Cosh(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Tf_Cos(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Cos(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Tf_Exp(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Exp(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Tf_Add_N(self, IR_node):
        input_layers = ', '.join(('n.' + self.IR_graph.get_node(edge).real_variable_name) for edge in IR_node.in_edges)
        self.add_body(1, "n.{:<15} = L.Tf_Addn({})".format(IR_node.variable_name, input_layers))

    def emit_Tf_Greater(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Greater(n.{},n.{})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), self.parent_variable_name(IR_node, [1])))

    def emit_Tf_Minimum(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Minimum(n.{},n.{})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), self.parent_variable_name(IR_node, [1])))

    def emit_Tf_Maximum(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Maximum(n.{},n.{})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), self.parent_variable_name(IR_node, [1])))

    def emit_Tf_Sigmoid(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Sigmoid(n.{})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0])))

    def emit_Tf_swish_f32(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_swish_f32(n.{})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0])))

    def emit_Tf_Lessequal(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Lessequal(n.{},n.{})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), self.parent_variable_name(IR_node, [1])))

    def emit_Tf_Maxpool(self, IR_node):
        padding = IR_node.get_attr('padding')
        if padding == 'SAME':
            padding = 0
        else:
            padding = 1
        self.add_body(1, "n.{:<15} = L.Tf_Maxpool(n.{}, ksize={}, strides={}, padding_type={},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), IR_node.get_attr('ksize'), IR_node.get_attr('strides'), padding))

    def emit_Tf_Erfc(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Tf_Erfc(n.{})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0])))

    def emit_Tf_Matmul(self, IR_node):
        input_layers = ', '.join(('n.' + self.IR_graph.get_node(edge).real_variable_name) for edge in IR_node.in_edges)
        for in_node in IR_node.in_edges:
            if not self.is_valid_node(self.IR_graph.get_node(in_node).type):
                parent = self.IR_graph.get_parent(IR_node.name, [0])
                assert parent is not None
                IR_node.real_name = parent.real_name
                logger.warn("some of the inputs are not well defined - not adding entry in prototxt for node %s. in_node details: %s real_name: %s", IR_node.name, in_node, IR_node.real_name)
                return
        self.add_body(1, "n.{:<15} = L.Tf_Matmul({},transpose_a={}, transpose_b={},ntop=1)".format(IR_node.variable_name, input_layers, IR_node.get_attr('transpose_a'), IR_node.get_attr('transpose_b')))

    def emit_Tf_Mish(self, IR_node):
        in_place = False
        self.add_body(1, "n.{:<15} = L.Mish(n.{}, in_place={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), in_place))

    def emit_ReduceSum_To_ELTS2(self, IR_node):
        elt_layer_name = IR_node.variable_name + '_elt_concat'
        input_layers = ', '.join(('n.' + self.parent_variable_name(IR_node, [num])) for num in range(0, len(IR_node.in_edges)))
        logger.info("IR3 input layers: %s", input_layers)
        self.parent_variable_name(IR_node, [0])
        if len(input_layers) < 2:
            logger.error("number of inputs is less than 2 for Eltwise node %s", IR_node.name)
            raise ValueError()
        if len(IR_node.in_edges) > 2:
            raise NotImplementedError("Not supported more than two inputs to elementwise add")
        in_edges = [self.IR_graph.get_node(IR_node.in_edges[0]).real_name, self.IR_graph.get_node(IR_node.in_edges[1]).real_name]
        self.add_body(1, "n.{:<15} = L.Eltwise({}, operation=1, ntop=1)".format(
        elt_layer_name,
        input_layers,
    ))
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        input_node_shape = shape_to_list(input_node.get_attr('_output_shapes')[0])
        bn_layer_name = IR_node.variable_name + '_scalar_mul_bn'
        # scalar mul layer using BatchNorm
        self.add_body(1, "n.{:<15} = L.BatchNorm(n.{}, eps={}, use_global_stats={}, ntop=1)".format(bn_layer_name, elt_layer_name, 0, self.phase == 'test'))
        scale_layer_name = IR_node.variable_name + "_scalar_mul_scale"
        shape = IR_node.get_attr("_output_shapes")[0]
        shape = shape_to_list(shape)
        axis = 1
        if len(shape) == 1:
            axis = 0
        self.add_body(
        1,
        "n.{:<15} = L.Scale(n.{}, bias_term={}, in_place=False, axis={}, ntop=1)".format(
            scale_layer_name,
            bn_layer_name,
            True,  #IR_node.get_attr('use_bias', False)
            axis))
        num_channels = IR_node.get_attr("_output_shapes")[0].dim[3].size
        if self.weight_loaded:
            self.weights_dict[scale_layer_name] = dict()
            self.weights_dict[bn_layer_name] = dict()
            self.weights_dict[scale_layer_name]['scale'] = np.ones(num_channels) * IR_node.layer.attr['mul_scalar'].f
            self.weights_dict[scale_layer_name]['bias'] = np.zeros(num_channels)
            self.weights_dict[bn_layer_name]['scale'] = 1
            self.weights_dict[bn_layer_name]['var'] = np.ones(num_channels)
            self.weights_dict[bn_layer_name]['mean'] = np.zeros(num_channels)
        IR_node.real_name = scale_layer_name

    def emit_ReduceSum_To_ELTS3(self, IR_node):
        elt_layer_name_1 = IR_node.name + '_elt1'
        elt_layer_name_2 = IR_node.name + '_elt2'
        # input_layers = ', '.join(('n.' + self.IR_graph.get_parent(IR_node.name, [num]).real_variable_name) for num in range(0, 2))
        # input_layers = ', '.join(('n.' + self.parent_variable_name(IR_node, [num])) for num in range(0, len(IR_node.in_edges)))
        input_layers = 'n.' + self.parent_variable_name(IR_node, [0]) + ', n.' + self.parent_variable_name(IR_node, [1])
        logger.info("IR2 input layers: %s", input_layers)
        if len(input_layers) < 2:
            logger.error("number of inputs is less than 2 for Eltwise node %s", IR_node.name)
            raise ValueError()
        in_edges = [self.IR_graph.get_node(IR_node.in_edges[0]).name, self.IR_graph.get_node(IR_node.in_edges[1]).name]
        self.add_body(1, "n.{:<15} = L.Eltwise({}, operation=1, ntop=1)".format(
        elt_layer_name_1,
        input_layers,
    ))
        # input_layer = ', '.join(('n.' + self.IR_graph.get_parent(IR_node.name, [num]).real_variable_name) for num in range(0, len(IR_node.in_edges)))
        input_layer = 'n.' + elt_layer_name_1 + ', n.' + self.parent_variable_name(IR_node, [2])
        if len(input_layer) < 2:
            logger.error("number of inputs is less than 2 for Eltwise node %s", IR_node.name)
            raise ValueError()
        in_edges = [self.IR_graph.get_node(IR_node.in_edges[0]).real_name, self.IR_graph.get_node(IR_node.in_edges[1]).real_name]
        self.add_body(1, "n.{:<15} = L.Eltwise({}, operation=1, ntop=1)".format(
        elt_layer_name_2,
        input_layer,
    ))
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        input_node_shape = shape_to_list(input_node.get_attr('_output_shapes')[0])
        IR_node.real_name = elt_layer_name_2

    def emit_BatchDot(self, IR_node):
        multiplier = IR_node.get_attr("multiplier", 1)
        if (IR_node.get_attr('const_tensor')):
            dummy_layer_name = IR_node.variable_name + "_const_tensor"
            self.weights_dict[dummy_layer_name] = self.weights_dict.pop(IR_node.name)
            tensor_shape = [i for i in np.array(self.weights_dict[dummy_layer_name]["weights"]).shape]
            self.add_body(1, "n.{:<15} = L.DVTensor(dvtensor_param=dict(tensortype={}, value= {},shape=dict(dim={})), ntop=1)".format(dummy_layer_name, 2, 0, '[1]' if str(tensor_shape) == '[]' else str(tensor_shape)))
            input_layers = [self.IR_graph.get_node(IR_node.in_edges[0]).real_variable_name]
            input_layers.insert(IR_node.get_attr('dvtensor_pos'), dummy_layer_name)
            input_layers = ', '.join(['n.' + i for i in input_layers])
            self.add_body(1, "n.{:<15} = L.BatchDot({},transpose_a={}, transpose_b={}, multiplier={}, ntop=1)".format(IR_node.variable_name, input_layers, IR_node.get_attr('transpose_a'), IR_node.get_attr('transpose_b'), multiplier))
            return
        input_layers = ', '.join(('n.' + self.IR_graph.get_node(edge).real_variable_name) for edge in IR_node.in_edges)
        self.add_body(1, "n.{:<15} = L.BatchDot({},transpose_a={}, transpose_b={}, multiplier={}, ntop=1)".format(IR_node.variable_name, input_layers, IR_node.get_attr('transpose_a'), IR_node.get_attr('transpose_b'), multiplier))
        # Pointnet(Tensorflow BatchMatMulV2)
        if IR_node.get_attr('data_format') == "NHWC":
            perm_order_str = '{' + "'order': {}".format([0, 2, 1]) + '}'
            perm1 = IR_node.real_name + "_perm1"
            self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(perm1, IR_node.variable_name, perm_order_str))
            IR_node.real_name = perm1

    def emit_Mxnet_Batch_Dot(self, IR_node):
        input_layers = ', '.join(('n.' + self.IR_graph.get_node(edge).real_variable_name) for edge in IR_node.in_edges)
        self.add_body(1, "n.{:<15} = L.Mxnet_Batch_Dot({},transpose_a={}, transpose_b={},ntop=1)".format(IR_node.variable_name, input_layers, IR_node.get_attr('transpose_a'), IR_node.get_attr('transpose_b')))

    def emit_RNN(self, IR_node):
        if IR_node.get_attr('mode') == 'lstm':
            bid = IR_node.get_attr('bidirectional')
            state_size = IR_node.get_attr('state_size')
            batch_size = IR_node.get_attr('batch_size')
            T_ = IR_node.get_attr('T_')
            logger.info("emitting LSTM")
            c0_name = IR_node.variable_name + '_c0'
            h0_name = IR_node.variable_name + '_h0'
            cont_name = IR_node.variable_name + '_cont_dummy'
            state_shape = [1 + int(bid), batch_size, state_size]
            cont_shape = [T_, batch_size]
            self.add_body(1, "n.{:<15} = L.DummyData(shape=[dict(dim={})], data_filler=dict(type='constant', value={}), ntop=1)".format(cont_name, cont_shape, 1))
            self.add_body(1, "n.{:<15} = L.Input(shape=[dict(dim={})], ntop=1)".format(c0_name, state_shape))
            self.add_body(1, "n.{:<15} = L.Input(shape=[dict(dim={})], ntop=1)".format(h0_name, state_shape))
            output_layers = ', '.join(['n.' + IR_node.variable_name, 'n.' + IR_node.variable_name + '_hT', 'n.' + IR_node.variable_name + '_cT'])
            input_layers = ', '.join(['n.' + self.parent_variable_name(IR_node), 'n.' + cont_name, 'n.' + h0_name, 'n.' + c0_name])
            self.add_body(1, "{} = L.LSTM({}, recurrent_param=dict(expose_hidden={}, bidirectional={}, num_output={}), ntop=3)".format(output_layers, input_layers, True, bid, state_size))
        logger.info("Emitting layer in caffe %s", IR_node.layer)

    def emit_Mxnet_RNN(self, IR_node):

        def _variable_name(layer):
            return layer.replace('/', '_').replace('-', '_').replace('[', '_').replace(']', '_').replace('.', '_')

        if IR_node.get_attr('mode') == 'lstm':
            bid = IR_node.get_attr('bidirectional')
            L_ = IR_node.get_attr('num_layers')
            state_size = IR_node.get_attr('state_size')
            batch_size = IR_node.get_attr('batch_size')
            T_ = IR_node.get_attr('T_')
            logger.info("emitting LSTM")
            base_name = _variable_name(IR_node.get_attr('base_name'))
            # c0_name = base_name + '_c0'
            # h0_name = base_name + '_h0'
            cont_name = base_name + '_cont_dummy'
            per_layer_B = 1 + int(bid)
            state_shape = [L_ * per_layer_B, batch_size, state_size]
            rnn_prefix = IR_node.get_attr('rnn_prefix')
            cont_shape = [T_, batch_size]
            self.add_body(1, "n.{:<15} = L.DummyData(shape=[dict(dim={})], data_filler=dict(type='constant', value={}), ntop=1)".format(cont_name, cont_shape, 1))
            if IR_node.layer.input[3] not in self.IR_graph.layer_map:
                c0_name = IR_node.layer.input[3]
                self.add_body(1, "n.{:<15} = L.Input(shape=[dict(dim={})], ntop=1)".format(c0_name, state_shape))
            else:
                c0_name = self.IR_graph.layer_map[IR_node.layer.input[3]].real_name
            if IR_node.layer.input[2] not in self.IR_graph.layer_map:
                h0_name = IR_node.layer.input[2]
                self.add_body(1, "n.{:<15} = L.Input(shape=[dict(dim={})], ntop=1)".format(h0_name, state_shape))
            else:
                h0_name = self.IR_graph.layer_map[IR_node.layer.input[2]].real_name
            prev_layer_name = self.parent_variable_name(IR_node)
            h_slice_layer_name = base_name + 'h_split'
            c_slice_layer_name = base_name + 'c_split'
            c_slice_outs = []
            h_slice_outs = []
            if L_ > 1:
                for ll in range(L_):
                    h_slice_outs.append('n.' + h_slice_layer_name + "_" + str(ll))
                    c_slice_outs.append('n.' + c_slice_layer_name + "_" + str(ll))
                self.add_body(1, "{} = L.Slice(n.{}, slice_param=dict(axis={}, slice_point={}), ntop={})".format(', '.join(h_slice_outs), h0_name, 0, list(range(per_layer_B, L_ * per_layer_B, per_layer_B)), L_))
                self.add_body(1, "{} = L.Slice(n.{}, slice_param=dict(axis={}, slice_point={}), ntop={})".format(', '.join(c_slice_outs), c0_name, 0, list(range(per_layer_B, L_ * per_layer_B, per_layer_B)), L_))
            else:
                h_slice_outs.append('n.' + h0_name)
                c_slice_outs.append('n.' + c0_name)
            for ll in range(L_):
                layer_name = base_name + "_l" + str(ll)
                if ll == L_ - 1:
                    out_name = base_name + "_output"
                else:
                    out_name = layer_name
                if L_ == 1:
                    out1_name = 'n.' + base_name + "_state_output"
                    out2_name = 'n.' + base_name + "_statecell_output"
                else:
                    out1_name = 'n.' + layer_name + '_hT'
                    out2_name = 'n.' + layer_name + '_cT'
                logger.info("layer_name : %d, %s", ll, layer_name)
                output_layers = ', '.join(['n.' + out_name, out1_name, out2_name])
                input_layers = ', '.join(['n.' + prev_layer_name, 'n.' + cont_name, h_slice_outs[ll], c_slice_outs[ll]])
                self.add_body(1, "{} = L.LSTM({}, recurrent_param=dict(expose_hidden={}, bidirectional={}, num_output={}), ntop=3)".format(output_layers, input_layers, True, bid, state_size))
                prev_layer_name = layer_name
                l_i2h_weight = rnn_prefix + "l" + str(ll) + "_i2h_weight"
                l_i2h_bias = rnn_prefix + "l" + str(ll) + "_i2h_bias"
                l_h2h_weight = rnn_prefix + "l" + str(ll) + "_h2h_weight"
                l_h2h_bias = rnn_prefix + "l" + str(ll) + "_h2h_bias"
                r_i2h_weight = rnn_prefix + "r" + str(ll) + "_i2h_weight"
                r_i2h_bias = rnn_prefix + "r" + str(ll) + "_i2h_bias"
                r_h2h_weight = rnn_prefix + "r" + str(ll) + "_h2h_weight"
                r_h2h_bias = rnn_prefix + "r" + str(ll) + "_h2h_bias"
                if bid:
                    weights_order_list = [l_i2h_weight, l_i2h_bias, l_h2h_weight, l_h2h_bias, r_i2h_weight, r_i2h_bias, r_h2h_weight, r_h2h_bias]
                else:
                    weights_order_list = [l_i2h_weight, l_i2h_bias, l_h2h_weight, l_h2h_bias]
                final_weights = list()
                logger.debug(self.weights_dict.keys())
                if ll == L_ - 1:
                    layer_name1 = base_name + "_output"
                else:
                    layer_name1 = base_name + "_l" + str(ll)
                for wei in range(int(len(weights_order_list) / 2)):
                    weight_name = weights_order_list[2 * wei]
                    bias_name = weights_order_list[2 * wei + 1]
                    logger.debug(self.weights_dict[layer_name1].keys())
                    final_weights.append(self.weights_dict[layer_name1][weight_name].transpose((1, 0)))
                    final_weights.append(self.weights_dict[layer_name1][bias_name])
                self.weights_dict.pop(layer_name1)
                self.weights_dict[layer_name1] = final_weights
            c_concat_ins = []
            h_concat_ins = []
            if L_ > 1:
                for ll in range(L_):
                    h_concat_ins.append('n.' + base_name + "_l" + str(ll) + "_hT")
                    c_concat_ins.append('n.' + base_name + "_l" + str(ll) + "_cT")
                self.add_body(1, "n.{:<15} = L.Concat({}, axis={},ntop=1)".format(base_name + '_state_output', ', '.join(h_concat_ins), 0))
                self.add_body(1, "n.{:<15} = L.Concat({}, axis={},ntop=1)".format(base_name + '_statecell_output', ', '.join(c_concat_ins), 0))
            logger.info("Emitting layer in caffe %s", IR_node.name)
        else:
            logger.error("RNN mode: %s not supported . Layer: %s", IR_node.get_attr('mode'), IR_node.name)
            raise NotImplementedError()
        # IR_node.real_name = layer_name1
    def emit_Max(self, IR_node):
        axes = IR_node.get_attr('axis')
        if len(axes) > 1:
            # if len(axes) == 3 or len(axes) == 2:
            # TODO
            logger.error("only single axis currently supported for max op")
            raise NotImplementedError()
            # for idx, i in enumerate(axes):
            #     axes[idx] = self._convert_axis_cfirst(IR_node,axis=i)
        axis = axes[0]
        input_node_name = self.parent_variable_name(IR_node)
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        ishape = shape_to_list(input_node.get_attr("_output_shapes")[0])
        indim = len(ishape)
        parent = self.parent_variable_name(IR_node)
        permute = False
        if axis != indim - 1:
            permute = True
        pshape = ishape
        permute_order = []
        if permute:
            permute_order = [i for i in range(indim)]
            permute_order[axis] = indim - 1
            permute_order[indim - 1] = axis
            pshape = []
            for i in permute_order:
                pshape.append(ishape[i])
            perm_order_str = '{' + "'order': {}".format(permute_order) + '}'
            perm1 = IR_node.real_name + "_perm1"
            self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(perm1, parent, perm_order_str))
            parent = perm1
        r1shape = [1] + [np.prod(pshape[:-1])] + [1, pshape[-1]]
        r2shape = pshape[:-1] + [1]
        dim_str = "'dim': {}".format(r1shape)
        dim_str = " reshape_param={'shape': { " + dim_str + '} }'
        reshape1 = IR_node.real_name + "_reshape1"
        self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape1, parent, dim_str))
        parent = reshape1
        self.add_body(1, "n.{:<15} = L.Pooling(n.{}, pool={}, global_pooling=True, ntop=1)".format(IR_node.variable_name, parent, 0))
        parent = IR_node.real_name
        dim_str = "'dim': {}".format(r2shape)
        dim_str = " reshape_param={'shape': { " + dim_str + '} }'
        reshape2 = IR_node.real_name + "_reshape2"
        self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape2, parent, dim_str))
        parent = reshape2
        p2shape = r2shape
        if permute:
            perm_order_str = '{' + "'order': {}".format(permute_order) + '}'
            perm1 = IR_node.real_name + "_perm2"
            self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(perm1, parent, perm_order_str))
            parent = perm1
            p2shape = []
            for i in permute_order:
                p2shape.append(r2shape[i])
        final_shape = clast2first(shape_to_list(IR_node.get_attr("_output_shapes")[0]))
        if p2shape != final_shape:
            dim_str = "'dim': {}".format(final_shape)
            dim_str = " reshape_param={'shape': { " + dim_str + '} }'
            reshape3 = IR_node.real_name + "_reshape3"
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape3, parent, dim_str))
            parent = reshape3
        IR_node.real_name = parent

    def emit_Mxnet_Flip(self, IR_node):
        input_layers = ', '.join(('n.' + self.IR_graph.get_node(edge).real_variable_name) for edge in IR_node.in_edges)
        axis = "'dim': {}".format(IR_node.get_attr('axis'))
        axis = "{'axis': { " + axis + '} }'
        self.add_body(1, "n.{:<15} = L.Mxnet_Flip({},axis=dict(dims={}, shape={}), ntop=1)".format(
        IR_node.variable_name,
        input_layers,
        len(IR_node.get_attr('axis')),
        IR_node.get_attr('axis'),
    ))

    def emit_Fused_Add_Mul_Broadcast(self, IR_node):
        logger.info("Emitting Fused_Add_Mul_Broadcast")
        self.add_body(1, "n.{:<15} = L.Fused_Add_Mul_Broadcast(n.{}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0])))
    
    def emit_onnx_sub(self, IR_node):
        logger.info("emitting sub layer.  %s %s", IR_node.name, IR_node.variable_name)
        inp_node_1 = IR_node.variable_name + '_temp'
        shape = IR_node.get_attr("_output_shapes")[0]
        shape = shape_to_list(shape)
        self.add_body(1, "n.{:<15} = L.BatchNorm(n.{}, eps={}, use_global_stats={}, ntop=1)".format(inp_node_1, self.parent_variable_name(IR_node, [1]), 0.0, self.phase == 'test'))
        scale_layer_var_name = inp_node_1 + "_scale"
        axis = 1
        if len(shape) == 1:
            axis = 0
        self.add_body(1, "n.{:<15} = L.Scale(n.{}, bias_term={}, in_place=False, axis={}, ntop=1)".format(scale_layer_var_name, inp_node_1, True, axis))
        self.weights_dict[inp_node_1] = dict()
        self.weights_dict[scale_layer_var_name] = dict()
        self.weights_dict[inp_node_1]['var'] = np.array([1] * shape[-1], dtype=np.float32)
        self.weights_dict[inp_node_1]['scale'] = np.array([1] * shape[-1], dtype=np.float32)
        self.weights_dict[scale_layer_var_name]['scale'] = np.array([-1] * shape[-1], dtype=np.float32)
        self.weights_dict[scale_layer_var_name]['bias'] = np.array([0] * shape[-1], dtype=np.float32)
        self.weights_dict[inp_node_1]['mean'] = np.zeros(shape, dtype=np.float32)  #np.array([0]*shape,dtype=np.float32)
        self.add_body(1, "n.{:<15} = L.Eltwise(n.{}, n.{}, operation=1, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), scale_layer_var_name))

    def emit_vector_sub(self, IR_node):
        input_layers = [self.IR_graph.get_node(edge).real_variable_name for edge in IR_node.in_edges]
        input_layers = ','.join(['n.' + i for i in input_layers])
        self.add_body(1, "n.{:<15} = L.vector_sub({}, adj_param=dict(precision=[3]), ntop=1)".format(IR_node.variable_name, input_layers))
    
    def emit_vector_mult(self, IR_node):
        input_layers = [self.IR_graph.get_node(edge).real_variable_name for edge in IR_node.in_edges]
        input_layers = ','.join(['n.' + i for i in input_layers])
        self.add_body(1, "n.{:<15} = L.vector_mult({}, adj_param=dict(precision=[3]), ntop=1)".format(IR_node.variable_name, input_layers))
    
    def emit_divideVectorByScalar(self, IR_node):
        input_layers = [self.IR_graph.get_node(edge).real_variable_name for edge in IR_node.in_edges]
        input_layers = ','.join(['n.' + i for i in input_layers])
        self.add_body(1, "n.{:<15} = L.divideVectorByScalar({}, adj_param=dict(precision=[3]), ntop=1)".format(IR_node.variable_name, input_layers))

    def emit_onnx_matmul(self, IR_node):
        logger.info("Mapping matmul to conv 1x1")
        reshape_layer_name = IR_node.name + '_reshape'
        IR_node_reshape = self.IR_graph.get_node(reshape_layer_name)
        out_shape = IR_node_reshape.get_attr("shape")  #[-1, IR_node.get_attr('kernel_shape')[-1]]
        dim_str = "'dim': {}".format(out_shape)
        dim_str = " reshape_param={'shape': { " + dim_str + '} }'
        self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape_layer_name, self.parent_variable_name(IR_node), dim_str))
        order1 = order2 = [0, 3, 2, 1]
        if len(out_shape) > 1 and out_shape[1] == 1:
            order1 = [0, 3, 1, 2]
            order2 = [0, 2, 3, 1]
        dummy_layer_name = IR_node.real_variable_name + '_perm'
        perm_order_str = '{' + "'order': {}".format(order1) + '}'
        self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(dummy_layer_name, reshape_layer_name, perm_order_str))

        pads = IR_node.get_attr('pads')[1:-1]
        spatial_pad = pads[:len(pads) // 2 - 1] + pads[len(pads) // 2 + 1:]
        self.add_body(
        1,
        "n.{:<15} = L.Convolution(n.{}, kernel_size={}, stride = {}, num_output={}, pad={}, group={}, \
        bias_term={},dilation={}, ntop=1)".format(
            IR_node.variable_name,
            dummy_layer_name,
            IR_node.get_attr('kernel_shape')[:-2],
            IR_node.get_attr('strides')[1:-1],
            IR_node.get_attr('kernel_shape')[-1],
            # [pad_f, pad_t, pad_l], #replace with [pad_f, pad_t, pad_l, pad_g, pad_b, pad_r], after analyzer changes
            spatial_pad,
            IR_node.get_attr("group", 1),
            IR_node.get_attr('use_bias', False),
            IR_node.get_attr('dilations')[1:-1]))

        dim = len(IR_node.get_attr('strides')) - 2
        self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], [dim + 1, dim] + list(range(0, dim)))

        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        input_node_shape = shape_to_list(input_node.get_attr("_output_shapes")[0])
        curr_name = IR_node.variable_name
        if len(input_node_shape) == 4:
            self.check_if_need_crop(IR_node)
            curr_name = IR_node.real_name
        reshape_layer_name = IR_node.variable_name + '_reshape1'

        IR_node_reshape = self.IR_graph.get_node(reshape_layer_name)
        out_shape = IR_node_reshape.get_attr("shape")
        perm_order_str = '{' + "'order': {}".format(order2) + '}'
        if len(out_shape) in [2, 3, 4, 5]:
            dummy_layer_name = IR_node.real_variable_name + '_perm_2'
            self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(dummy_layer_name, curr_name, perm_order_str))
            IR_node.real_name = dummy_layer_name
        dim_str = "'dim': {}".format(out_shape)
        dim_str = " reshape_param={'shape': { " + dim_str + '} }'
        self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape_layer_name, IR_node.real_name, dim_str))
        IR_node.real_name = reshape_layer_name

    def emit_onnx_batched_inrp(self, IR_node):
        logger.info("Mapping batched inrp to conv 1x1")
        reshape_layer_name = IR_node.name + '_reshape1'
        IR_node_reshape = self.IR_graph.get_node(reshape_layer_name)
        out_shape = IR_node_reshape.get_attr("shape")  #[-1, IR_node.get_attr('kernel_shape')[-1]]
        dim_str = "'dim': {}".format(out_shape)
        dim_str = " reshape_param={'shape': { " + dim_str + '} }'
        self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape_layer_name, self.parent_variable_name(IR_node), dim_str))
        dummy_layer_name = IR_node.name + '_perm'
        perm_order_str = '{' + "'order': {}".format([0, 3, 1, 2]) + '}'
        self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(dummy_layer_name, reshape_layer_name, perm_order_str))
        IR_node.in_edges = [dummy_layer_name]
        kernels = IR_node.get_attr('kernel_shape')[:-2]
        pads = IR_node.get_attr('pads')[1:-1]
        spatial_pad = pads[:len(pads) // 2 - 1] + pads[len(pads) // 2 + 1:]
        conv_layer = IR_node.variable_name + "_conv"
        self.add_body(
        1,
        "n.{:<15} = L.Convolution(n.{}, kernel_size={}, stride = {}, num_output={}, pad={}, group={}, \
            bias_term={},dilation={}, ntop=1)".format(
            conv_layer,
            self.parent_variable_name(IR_node),
            kernels,
            IR_node.get_attr('strides')[1:-1],
            IR_node.get_attr('kernel_shape')[-1],
            # [pad_f, pad_t, pad_l], #replace with [pad_f, pad_t, pad_l, pad_g, pad_b, pad_r], after analyzer changes
            spatial_pad,
            1,
            IR_node.get_attr('use_bias', False),
            IR_node.get_attr('dilations')[1:-1]))
        # dim = len(IR_node.get_attr('strides')) - 2
        self.weights_dict[conv_layer] = dict()
        self.weights_dict[conv_layer]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], [3, 2, 0, 1])
        self.weights_dict[IR_node.name].pop()

        dummy_layer_name = IR_node.real_variable_name + '_perm_2'
        self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(dummy_layer_name, conv_layer, perm_order_str))
        reshape_layer_name = IR_node.name + "_reshape1"
        IR_node_reshape = self.IR_graph.get_node(reshape_layer_name)
        out_shape = IR_node_reshape.get_attr("shape")  #[-1, IR_node.get_attr('kernel_shape')[-1]]
        dim_str = "'dim': {}".format(out_shape)
        dim_str = " reshape_param={'shape': { " + dim_str + '} }'
        self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(IR_node.variable_name, dummy_layer_name, dim_str))

    def emit_Cast(self, IR_node):
        Precision = {
        'INT8': 0,
        'INT16': 1,
        'INT32': 2,
        'FP32': 3,
        'ZERO': 999,
    }
        self.add_body(1, "n.{:<15} = L.DVCast(n.{}, src_precision={}, dst_precision={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), 3, Precision[IR_node.get_attr('to', 'INT8')]))

    def emit_Mxnet_Take(self, IR_node):
        mode = {'clip': 0, 'raise': 1, 'wrap': 2}
        in_edges = [self.IR_graph.get_node(IR_node.in_edges[0]).real_name, self.IR_graph.get_node(IR_node.in_edges[1]).real_name]
        self.add_body(1, "n.{:<15} = L.Mxnet_Take(n.{},n.{}, mode={}, axis={}, ntop=1)".format(IR_node.variable_name, in_edges[0], in_edges[1], mode[IR_node.get_attr('mode')], self._convert_axis_cfirst(IR_node, IR_node.get_attr('axis'))))

    def emit_Mxnet_slice_like(self, IR_node):
        axis = IR_node.get_attr('axis')
        n_dim = len(axis)
        self.add_body(1, "n.{:<15} = L.Mxnet_Slice_Like(n.{},n.{},axes=dict(dims={}, shape={}))".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), self.parent_variable_name(IR_node, [1]), n_dim, axis))

    def emit_Repeat(self, IR_node):
        axis = IR_node.get_attr('axis')
        repeats = IR_node.get_attr('repeats')
        self.add_body(1, "n.{:<15} = L.Repeat(n.{},axis={},repeats={})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), axis, repeats))

    def emit_Mxnet_Repeat(self, IR_node):
        axis = IR_node.get_attr('axis')
        repeats = IR_node.get_attr('repeats')
        self.add_body(1, "n.{:<15} = L.Mxnet_Repeat(n.{},axis={},repeats={})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), axis, repeats))

    def emit_ROIAlign(self, IR_node):
        in_edges = []
        parent_nodes = []
        for num in range(0, len(IR_node.in_edges)):
            parent = self.IR_graph.get_parent(IR_node.name, [num])
            assert parent is not None
            parent_nodes.append(parent)
            in_edges.append(parent.real_variable_name)
        # # rois = IR_node.get_attr(('batch_indices')
        parent_shape = [shape_to_list(node.get_attr("_output_shapes")[0]) for node in parent_nodes]
        if len(parent_shape[1]) == 4:
            in_edges.reverse()
        dummy_layer_name = IR_node.real_variable_name + '_perm'
        perm_order_str = '{' + "'order': {}".format([0, 2, 3, 1]) + '}'
        self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(dummy_layer_name, in_edges[0], perm_order_str))
        out_list = IR_node.get_attr("out_layers")
        if out_list is None:
            out_list = [IR_node.variable_name]
        out_layers = [n.decode("utf-8") for n in out_list]
        dummy_layer_name_1 = IR_node.real_variable_name + '_dummy'
        self.add_body(
        1,
        "n.{:<15} = L.ROIAlign(n.{}, n.{}, \
                        spatial_scale={}, sampling_ratio={}, \
                        pooled_height={}, pooled_width={},  \
                        aligned={}, \
                        ntop=1)".format(
            dummy_layer_name_1,
            dummy_layer_name,
            in_edges[1],
            # rois,b_idx,
            # IR_node.get_attr('num_rois'),
            IR_node.get_attr('spatial_scale', 1.0),
            IR_node.get_attr('sampling_ratio', -1.0),
            IR_node.get_attr('pooled_height'),
            IR_node.get_attr('pooled_width'),
            IR_node.get_attr('aligned', True),
        ))
        perm_order_str = '{' + "'order': {}".format([0, 3, 1, 2]) + '}'
        self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(out_layers[0], dummy_layer_name_1, perm_order_str))

    def emit_onnxpqn_Conv(self, IR_node):
        is_tflite = IR_node.get_attr('is_tflite', False)
        # implement asymmetric paddings by applying symmetric padding then cropping
        # sym_pads = self._get_symmetric_padding(IR_node)
        # output_flag = False
        dilations = IR_node.get_attr('dilations')[1:-1]
        kernels = IR_node.get_attr('kernel_shape')[:-2]
        kernel_c = IR_node.get_attr('kernel_shape')[-2]
        pads = IR_node.get_attr('pads')
        spatial_pad = pads[:len(pads) // 2 - 1] + pads[len(pads) // 2 + 1:]
        spatial_kernels = kernels * 2
        spatial_dilations = dilations * 2
        strides = IR_node.get_attr('strides')[1:-1]
        # kpads = sym_pads #pads[1:len(pads)/2]
        np_spatial_kernels = np.array(spatial_kernels)
        np_spatial_dilations = np.array(spatial_dilations)
        # np_new_kernels = np_spatial_kernels + (np_spatial_kernels - 1) * (np_spatial_dilations - 1)
        kpads = spatial_pad
        flag = False
        num_output = IR_node.get_attr('kernel_shape')[-1]
        if IR_node.type == "tflite_DepthwiseConv":
            num_group = kernel_c
            num_output = IR_node.get_attr("_output_shapes")[0].dim[-1].size
        else:
            num_group = IR_node.get_attr("group", 1)
        layouts = [None, 'NCHW', 'NHWC', 'NDHWC', 'NCDHW', 'NCH', 'NHC', 'NCW', 'NWC']
        if IR_node.get_attr('data_format') not in layouts:
            raise NotImplementedError("CaffeEmitter: Convolution data-format neither of NCHW,NHWC,NDHWC,NCH,NHC,NCW,NWC,none(default)")
            """ if not (np.all((np.array(kpads) <= np.array(spatial_kernels)//2) == True) or \
                    np.all((np.array(kpads) >= np.array(spatial_kernels)//2) == True)) :
                    raise NotImplementedError("Pad in Height and Pad in Width both should be either ==, <= or >= kernel_size//2 for Conv") """
        input_node_name = self.parent_variable_name(IR_node)
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        input_node_shape = shape_to_list(input_node.get_attr("_output_shapes")[0])
        output_node_shape = shape_to_list(IR_node.get_attr("_output_shapes")[0])
        if len(dilations)==0:
            dilations = [1]*(len(input_node_shape)-2)
        globalconv = False
        check_for_crop = True
        output_scale = float(IR_node.get_attr('output_scale', 1))
        output_zero_point = IR_node.get_attr('output_zero_point', 0) - 128
        input_scale = IR_node.get_attr('input_scale', [1] * len(IR_node.in_edges))
        input_zero_point = IR_node.get_attr('input_zero_point', [0] * len(IR_node.in_edges)) - 128
        input_node_name = self.parent_variable_name(IR_node)
        # data_fmt = IR_node.get_attr('data_fmt', 'int8')
        data_fmt = 'int8'
        activation = 'none'
        activation = IR_node.get_attr('fused_activation_type')
        if activation is None:
            activation = 'none'
        # if output_zero_point == -128:
        #     activation = 'relu'
        assert activation in ['relu', 'relu6', 'none']
        conv_params = "kernel_size={}, stride = {}, num_output={}, pad={}, group={}, \
            bias_term={},dilation={}, is_pqn={}, pqn_qt=dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={}, data_format='{}'), fused_activation={}".format(
            kernels,
            strides,
            num_output,
            # [pad_f, pad_t, pad_l], #replace with [pad_f, pad_t, pad_l, pad_g, pad_b, pad_r], after analyzer changes
            spatial_pad,
            num_group,
            IR_node.get_attr('use_bias', False),
            dilations,
            is_tflite,
            output_scale,
            output_zero_point,
            input_scale,
            input_zero_point,
            data_fmt,
            fusedActivation[activation].value
        )
        if is_tflite:
            self.add_body(1, "n.{:<15} = L.Pqn_Convolution(n.{}, convolution_param=dict({}), ntop=1)".format(IR_node.variable_name, input_node_name, conv_params))
        else:
            self.add_body(1, "n.{:<15} = L.Convolution(n.{}, {}, ntop=1)".format(IR_node.variable_name, input_node_name, conv_params))

        dim = len(IR_node.get_attr('strides')) - 2
        if self.weight_loaded:
            if IR_node.type == "DepthwiseConv":
                self.weights_dict[IR_node.name]['weights'] = np.swapaxes(self.weights_dict[IR_node.name]['weights'], -1, -2)

        key_name = IR_node.variable_name if IR_node.variable_name in self.weights_dict else IR_node.name
        if is_tflite:
            if 'weight_scales' not in self.weights_dict[key_name]:
                self.weights_dict[key_name]['weight_scales'] = np.array([1] * num_output)
            if 'weight_zero_points' not in self.weights_dict[key_name]:
                self.weights_dict[key_name]['weight_zero_points'] = np.array([0] * num_output)

    def emit_onnxpqn_Add(self, IR_node):
        if len(IR_node.in_edges) > 2:
            raise NotImplementedError("Not supported more than two inputs to elementwise add")
        input_layers = self.get_input_layers_ignoring_qdq(IR_node)
        # Get the scales and offsets of inputs and output
        output_scale = float(IR_node.get_attr('output_scale', 1))
        output_zero_point = IR_node.get_attr('output_zero_point', 0) - 128
        input_scale = IR_node.get_attr('input_scale', [1] * len(IR_node.in_edges))
        input_zero_point = IR_node.get_attr('input_zero_point', [0] * len(IR_node.in_edges))
        input_zero_point = [x - 128 for x in input_zero_point]
        is_tflite = IR_node.get_attr('is_tflite', False)
        # data_format = IR_node.get_attr('data_format', 'int8')
        data_format = 'int8'
        activation = 'none'
        activation = IR_node.get_attr('fused_activation_type')
        if activation is None:
            activation = 'none'
        assert activation in ['relu', 'relu6', 'none']

        pqn_qt = "dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={}, data_format='{}')".format(output_scale, output_zero_point, input_scale, input_zero_point, data_format)

        # Case: One of the input is a constant tensor
        if (IR_node.get_attr('const_tensor')):
            input_layers = self.get_input_layers_ignoring_qdq(IR_node)
            input_shape = clast2first([i.size for i in IR_node.get_attr('_output_shapes')[0].dim])
            # input_shape = [i.size for i in IR_node.get_attr('_output_shapes')[0].dim]
            dummy_layer_name = IR_node.variable_name + "_const_tensor"
            for key in self.weights_dict[IR_node.name]:
                self.weights_dict[IR_node.name][key] = np.broadcast_to(self.weights_dict[IR_node.name][key], input_shape)
            self.weights_dict[dummy_layer_name] = self.weights_dict.pop(IR_node.name)
            self.add_body(1, "n.{:<15} = L.DVTensor(dvtensor_param=dict(tensortype={}, value= {},shape=dict(dim={})), ntop=1)".format(dummy_layer_name, 2, 0, '[1]' if str(input_shape) == '[]' else str(input_shape)))
            elt_params = "operation = 1"
            if is_tflite:
                self.add_body(1, "n.{:<15} = L.Tflite_Eltwise(n.{},n.{}, eltwise_param=dict({}), ntop=1)".format(IR_node.variable_name, input_layers[0], dummy_layer_name, elt_params))
            else:
                self.add_body(1, "n.{:<15} = L.Eltwise(n.{},n.{}, {}, ntop=1)".format(IR_node.variable_name, input_layers[0], dummy_layer_name, elt_params))
            return

        if len(input_layers) < 2:
            logger.error("number of inputs is less than 2 for Eltwise node %s", IR_node.name)
            raise ValueError()
        in_edges = self.get_input_layers_ignoring_qdq(IR_node)
        in_edges_shapes = [self.IR_graph.get_node(input_layers[0]).get_attr('_output_shapes'), self.IR_graph.get_node(input_layers[1]).get_attr('_output_shapes')]
        in_edges_shapes = [[j.size for j in i[0].dim] for i in in_edges_shapes]
        in_edges_shapes = [clast2first(i) for i in in_edges_shapes]
        if in_edges[0] == in_edges[1]:
            padding_layer_name = in_edges[1] + "_padding"
            self.add_body(
                1,
                "n.{:<15} = L.Padding(n.{}, padding_param=dict(pad_l={},pad_r={},pad_t={}, pad_b={},pad_method={},pad_value={}), ntop=1)".format(
                    padding_layer_name,
                    in_edges[1],
                    0,
                    0,
                    0,
                    0,
                    0,  #pad method is CONSTANT enum
                    0))
            self.add_body(1, "n.{:<15} = L.Eltwise(bottom={}, operation=1, ntop=1)".format(IR_node.variable_name, [in_edges[0], padding_layer_name]))
        else:
            if len(in_edges_shapes[0]) < len(in_edges_shapes[1]):
                new_shape = [1] * (len(in_edges_shapes[1]) - len(in_edges_shapes[0])) + in_edges_shapes[0]
                dim_str = "'dim': {}".format(new_shape)
                dim_str = " reshape_param={'shape': { " + dim_str + '} }'
                reshape1 = IR_node.real_name + "_reshape1"
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape1, self.parent_variable_name(IR_node, [0]), dim_str))
                input_layers[0] = reshape1
            if len(in_edges_shapes[0]) > len(in_edges_shapes[1]):
                new_shape = [1] * (len(in_edges_shapes[0]) - len(in_edges_shapes[1])) + in_edges_shapes[1]
                dim_str = "'dim': {}".format(new_shape)
                dim_str = " reshape_param={'shape': { " + dim_str + '} }'
                reshape1 = IR_node.real_name + "_reshape1"
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape1, self.parent_variable_name(IR_node, [1]), dim_str))
                input_layers[1] = reshape1
            if is_tflite:
                self.add_body(1, "n.{:<15} = L.Pqn_Eltwise(bottom={}, eltwise_param=dict(operation=1, pqn_qt={}), ntop=1)".format(IR_node.variable_name, [input_layers[0], input_layers[1]], pqn_qt))
            else:
                self.add_body(1, "n.{:<15} = L.Eltwise(n.{},n.{}, operation=1, ntop=1)".format(IR_node.variable_name, input_layers[0], input_layers[1]))

    def emit_onnxpqn_Mul(self, IR_node):
        logger.debug("Entered emit_onnxpqn_Mul")
        if len(IR_node.in_edges) > 2:
            raise NotImplementedError("Not supported more than two inputs to elementwise mul")

        input_layers = self.get_input_layers_ignoring_qdq(IR_node)
        # Get the scales and offsets of inputs and output
        output_scale = float(IR_node.get_attr('output_scale', 1))
        output_zero_point = IR_node.get_attr('output_zero_point', 0) - 128
        input_scale = IR_node.get_attr('input_scale', [1] * len(IR_node.in_edges))
        input_zero_point = IR_node.get_attr('input_zero_point', [0] * len(IR_node.in_edges))
        input_zero_point = [x - 128 for x in input_zero_point]
        is_tflite = IR_node.get_attr('is_tflite', False)
        # data_format = IR_node.get_attr('data_format', 'int8')
        data_format = 'int8'
        activation = 'none'
        activation = IR_node.get_attr('fused_activation_type')
        if activation is None:
            activation = 'none'
        assert activation in ['relu', 'relu6', 'none']

        pqn_qt = "dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={}, data_format='{}')".format(output_scale, output_zero_point, input_scale, input_zero_point, data_format)

         # Case: One of the input is a constant tensor
        if (IR_node.get_attr('const_tensor')):
            input_layers = [self.IR_graph.get_parent(IR_node.name, [num]).real_variable_name for num in range(len(IR_node.in_edges))]
            input_shape = clast2first([i.size for i in IR_node.get_attr('_output_shapes')[0].dim])
            # input_shape = [i.size for i in IR_node.get_attr('_output_shapes')[0].dim]
            dummy_layer_name = IR_node.variable_name + "_const_tensor"
            for key in self.weights_dict[IR_node.name]:
                self.weights_dict[IR_node.name][key] = np.broadcast_to(self.weights_dict[IR_node.name][key], input_shape)
            self.weights_dict[dummy_layer_name] = self.weights_dict.pop(IR_node.name)
            self.add_body(1, "n.{:<15} = L.DVTensor(dvtensor_param=dict(tensortype={}, value= {},shape=dict(dim={})), ntop=1)".format(dummy_layer_name, 2, 0, '[1]' if str(input_shape) == '[]' else str(input_shape)))
            elt_params = "operation = 0"
            if is_tflite:
                self.add_body(1, "n.{:<15} = L.Tflite_Eltwise(n.{},n.{}, eltwise_param=dict({}), ntop=1)".format(IR_node.variable_name, input_layers[0], dummy_layer_name, elt_params))
            else:
                self.add_body(1, "n.{:<15} = L.Eltwise(n.{},n.{}, {}, ntop=1)".format(IR_node.variable_name, input_layers[0], dummy_layer_name, elt_params))
            return

        if len(input_layers) < 2:
            logger.error("number of inputs is less than 2 for Eltwise node %s", IR_node.name)
            raise ValueError()
        in_edges = [self.IR_graph.get_node(IR_node.in_edges[0]).real_name, self.IR_graph.get_node(IR_node.in_edges[1]).real_name]
        in_edges_shapes = [self.IR_graph.get_node(IR_node.in_edges[0]).get_attr('_output_shapes'), self.IR_graph.get_node(IR_node.in_edges[1]).get_attr('_output_shapes')]
        in_edges_shapes = [[j.size for j in i[0].dim] for i in in_edges_shapes]
        in_edges_shapes = [clast2first(i) for i in in_edges_shapes]
        if in_edges[0] == in_edges[1]:
            padding_layer_name = in_edges[1] + "_padding"
            self.add_body(
                1,
                "n.{:<15} = L.Padding(n.{}, padding_param=dict(pad_l={},pad_r={},pad_t={}, pad_b={},pad_method={},pad_value={}), ntop=1)".format(
                    padding_layer_name,
                    input_layers[1],
                    0,
                    0,
                    0,
                    0,
                    0,  #pad method is CONSTANT enum
                    0))
            self.add_body(1, "n.{:<15} = L.Eltwise(bottom={}, operation=1, ntop=1)".format(IR_node.variable_name, [input_layers[0], padding_layer_name]))
        else:
            if len(in_edges_shapes[0]) < len(in_edges_shapes[1]):
                new_shape = [1] * (len(in_edges_shapes[1]) - len(in_edges_shapes[0])) + in_edges_shapes[0]
                dim_str = "'dim': {}".format(new_shape)
                dim_str = " reshape_param={'shape': { " + dim_str + '} }'
                reshape1 = IR_node.real_name + "_reshape1"
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape1, self.parent_variable_name(IR_node, [0]), dim_str))
                input_layers[0] = reshape1
            if len(in_edges_shapes[0]) > len(in_edges_shapes[1]):
                new_shape = [1] * (len(in_edges_shapes[0]) - len(in_edges_shapes[1])) + in_edges_shapes[1]
                dim_str = "'dim': {}".format(new_shape)
                dim_str = " reshape_param={'shape': { " + dim_str + '} }'
                reshape1 = IR_node.real_name + "_reshape1"
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape1, self.parent_variable_name(IR_node, [1]), dim_str))
                input_layers[1] = reshape1
            if is_tflite:
                self.add_body(1, "n.{:<15} = L.Pqn_Eltwise(bottom={}, eltwise_param=dict(operation=0, pqn_qt={}), ntop=1)".format(IR_node.variable_name, [input_layers[0], input_layers[1]], pqn_qt))
            else:
                self.add_body(1, "n.{:<15} = L.Eltwise(n.{},n.{}, operation=1, ntop=1)".format(IR_node.variable_name, input_layers[0], input_layers[1]))
        logger.debug("Exited emit_onnxpqn_Mul")
    
    def emit_onnxpqn_Sigmoid(self, IR_node):
        input_layers = self.get_input_layers_ignoring_qdq(IR_node)
        output_scale = float(IR_node.get_attr('output_scale', 1))
        output_zero_point = IR_node.get_attr('output_zero_point', 0) - 128
        input_scale = IR_node.get_attr('input_scale', [1] * len(IR_node.in_edges))
        input_zero_point = IR_node.get_attr('input_zero_point', [0] * len(IR_node.in_edges)) - 128
        # input_zero_point = [x - 128 for x in input_zero_point]
        data_format = IR_node.get_attr('data_foramt', 'int8')
        is_tflite = IR_node.get_attr('is_tflite', False)
        sigmoid_params = "pqn_qt=dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={}, data_format='{}'), is_pqn = {}".format(output_scale, output_zero_point, input_scale, input_zero_point, data_format, is_tflite)
        self.add_body(1, "n.{:<15} = L.Pqn_Sigmoid(bottom={}, sigmoid_param=dict({}), ntop=1)".format(IR_node.variable_name, [input_layers[0]], sigmoid_params))
        logger.debug("emit_onnxPqn_Sigmoid Successful")

    def emit_onnxpqn_Concat(self, IR_node):
        output_shape = [int(i.size) for i in IR_node.get_attr('_output_shapes')[0].dim]
        axis = IR_node.get_attr('axis')
        output_scale = float(IR_node.get_attr('output_scale', 1))
        output_zero_point = IR_node.get_attr('output_zero_point', 0) - 128
        input_scale = IR_node.get_attr('input_scale', [1] * len(IR_node.in_edges))
        input_zero_point = IR_node.get_attr('input_zero_point', [0] * len(IR_node.in_edges))
        input_zero_point = [x - 128 for x in input_zero_point]
        data_format = IR_node.get_attr('data_foramt', 'int8')
        if len(output_shape) > 2:
            axis = self._convert_axis_cfirst(IR_node, IR_node.get_attr('axis'))
        concat_params = "axis={}, pqn_qt=dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={}, data_format='{}')".format(axis, output_scale, output_zero_point, input_scale, input_zero_point, data_format)
        input_layers = self.get_input_layers_ignoring_qdq(IR_node)
        for in_node in IR_node.in_edges:
            if not self.is_valid_node(self.IR_graph.get_node(in_node).type):
                parent = self.IR_graph.get_parent(IR_node.name, [0])
                assert parent is not None
                IR_node.real_name = parent.real_name
                logger.warn("some of the inputs are not well defined - not adding entry in prototxt for node %s. in_node details: %s real_name: %s", IR_node.name, in_node, IR_node.real_name)
                return
        self.add_body(1, "n.{:<15} = L.Pqn_Concat(bottom={}, concat_param=dict({}))".format(IR_node.variable_name, input_layers, concat_params))
        logger.debug("emit_onnxpqn_Concat Successful")

    def emit_onnxpqn_Softmax(self, IR_node):
        input_layers = [self.IR_graph.get_parent(IR_node.name, [num]).real_variable_name for num in range(len(IR_node.in_edges))]
        axis = IR_node.layer.attr['axis'].i
        output_scale = float(IR_node.get_attr('output_scale', 1))
        output_zero_point = IR_node.get_attr('output_zero_point', 0) - 128
        input_scale = IR_node.get_attr('input_scale', [1] * len(IR_node.in_edges))
        input_zero_point = IR_node.get_attr('input_zero_point', [0] * len(IR_node.in_edges)) - 128
        # input_zero_point = [x - 128 for x in input_zero_point]
        data_format = IR_node.get_attr('data_foramt', 'int8')
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        input_node_shape = shape_to_list(input_node.get_attr('_output_shapes')[0])
        if len(input_node_shape) == 4:
            axis = NHWC_NCHW[axis]
        elif len(input_node_shape) == 3:
            if axis == 1:
                axis = 2
            elif axis == 2:
                axis = 1
        softmax_params = "output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={}, data_format='{}'".format(output_scale, output_zero_point, input_scale, input_zero_point, data_format)
        self.add_body(1, "n.{:<15} = L.Pqn_Softmax(bottom={}, softmax_param=dict(pqn_qt=dict({}), axis={}), ntop=1)".format(IR_node.variable_name, [input_layers[0]], softmax_params, axis))
        # else:
        #     self.add_body(1, "n.{:<15} = L.Softmax(n.{}, axis={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), axis))

    def emit_onnxpqn_FullyConnected(self, IR_node):
        parent_node = self.IR_graph.get_parent(IR_node.name, [0])
        # TODO FIXME for volumetric data
        input_shape = parent_node.get_attr("_output_shapes")[0]
        input_shape = clast2first(shape_to_list(input_shape))
        output_scale = float(IR_node.get_attr('output_scale', 1))
        output_zero_point = IR_node.get_attr('output_zero_point', 0) - 128
        input_scale = IR_node.get_attr('input_scale', [1] * len(IR_node.in_edges))
        input_zero_point = IR_node.get_attr('input_zero_point', [0] * len(IR_node.in_edges)) - 128
        # input_zero_point = [x - 128 for x in input_zero_point]
        is_tflite = IR_node.get_attr('is_tflite', False)
        prod = 1
        flatten = IR_node.get_attr('flatten', True)
        # data_format = IR_node.get_attr('data_format', 'int8')
        data_format = 'int8'
        if not flatten and len(input_shape) > 2:
            shape = [np.prod(input_shape[:-1]), input_shape[-1]]
            dim_str = "'dim': {}".format(shape)
            dim_str = "{'shape': { " + dim_str + '} }'
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, reshape_param={}) ".format(IR_node.variable_name + "_reshape1", self.parent_variable_name(IR_node), dim_str))
            inrp_param = "num_output={}, bias_term={}, is_pqn={}, pqn_qt=dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={},data_format='{}')".format(IR_node.layer.attr["units"].i, IR_node.get_attr('use_bias', False), is_tflite, output_scale, output_zero_point, input_scale, input_zero_point,data_format)
            self.add_body(1, "n.{:<15} = L.Pqn_InnerProduct(n.{}, inner_product_param=dict({}), ntop=1)".format(IR_node.variable_name, IR_node.variable_name + "_reshape1", inrp_param))

            output_shape = IR_node.get_attr("_output_shapes")[0]
            output_shape = clast2first(shape_to_list(output_shape))
            dim_str = "'dim': {}".format(output_shape)
            dim_str = "{'shape': { " + dim_str + '} }'
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, reshape_param={}) ".format(IR_node.variable_name + "_reshape2", IR_node.variable_name, dim_str))
            IR_node.real_name = IR_node.variable_name + "_reshape2"
        else:
            inrp_param = "num_output={}, bias_term={}, is_pqn={}, pqn_qt=dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={},data_format='{}')".format(IR_node.layer.attr["units"].i, IR_node.get_attr('use_bias', False), is_tflite, output_scale, output_zero_point, input_scale, input_zero_point,data_format)
            self.add_body(1, "n.{:<15} = L.Pqn_InnerProduct(n.{}, inner_product_param=dict({}), ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), inrp_param))
            if IR_node.get_attr('reshape', False):
                output_shape = IR_node.get_attr('reshape')
                dim_str = "'dim': {}".format(output_shape)
                dim_str = "{'shape': { " + dim_str + '} }'
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, reshape_param={}) ".format(IR_node.variable_name + "_reshape2", IR_node.variable_name, dim_str))
                IR_node.real_name = IR_node.variable_name + "_reshape2"
        if self.weight_loaded:
            # self.check_if_need_transpose(IR_node)
            self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], (1, 0))
            self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)

    def emit_onnxpqn_Pool(self, IR_node):
        data_format = 'int8'
        pooling_type = IR_node.get_attr('pooling_type')
        output_scale = float(IR_node.get_attr('output_scale', 1))
        output_zero_point = IR_node.get_attr('output_zero_point', 0) - 128
        input_scale = IR_node.get_attr('input_scale', [1] * len(IR_node.in_edges))
        input_zero_point = IR_node.get_attr('input_zero_point', [0] * len(IR_node.in_edges)) - 128
        is_tflite = IR_node.get_attr('is_tflite', False)

        if pooling_type == 'MAX':
            pooling_type = P.Pooling.MAX
        elif pooling_type == 'AVG':
            pooling_type = P.Pooling.AVE
        elif pooling_type == 'STOCHASTIC':
            pooling_type = P.Pooling.STOCHASTIC
        elif pooling_type == 'SUM':
            pooling_type = 3
        else:
            raise ValueError()

        parent_node = self.IR_graph.get_parent(IR_node.name, [0])
        input_shape = parent_node.get_attr("_output_shapes")[0]
        input_shape = shape_to_list(input_shape)

        strides = IR_node.get_attr('strides')[1:-1]
        kernels = IR_node.get_attr('kernel_shape')[1:-1]

        if len(strides) != len(input_shape) - 2:
            strides = strides * (len(input_shape) - 2 - len(strides))

        if len(kernels) != len(input_shape) - 2:
            kernels = kernels * (len(input_shape) - 2 - len(kernels))

        parent_node = self.IR_graph.get_parent(IR_node.name, [0])

        ceil_mode = IR_node.get_attr('ceil_mode', 0)
        # xprint("ceil_mode: ",ceil_mode)

        # Assume n, h, w, c format
        # input width shape being same as kernel shape implies global pooling
        # set strides to 1

        kernels_half = list(np.array(kernels) // 2)

        if pooling_type == 0:
            padding_value = -sys.float_info.max
        elif pooling_type == 1:
            padding_value = 0
        elif pooling_type == 2:
            padding_value = sys.float_info.max

        if input_shape[1:-1] == kernels and np.all(np.array(IR_node.get_attr('pads')) == 0):
            IR_node.layer.attr['global_pooling'].b = True

        if IR_node.layer.attr['global_pooling'].b:
            pool_param = "pool={}, global_pooling=True, pqn_qt=dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={}, data_format='{}')".format(pooling_type, output_scale, output_zero_point, input_scale, input_zero_point,data_format)
            if len(input_shape) == 3 and input_shape[0] > 1:
                ishape = clast2first(input_shape)
                dim_str = "'dim': {}".format([1, ishape[0] * ishape[1], 1, ishape[2]])
                dim_str = " reshape_param={'shape': { " + dim_str + '} }'
                reshape_layer_variable_name = IR_node.real_variable_name + "_reshape1"

                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape_layer_variable_name, self.parent_variable_name(IR_node), dim_str))
                self.add_body(1, "n.{:<15} = L.Pqn_Pooling(n.{}, pooling_param=dict({}), ntop=1)".format(IR_node.variable_name, reshape_layer_variable_name, pool_param))
                output_shape = shape_to_list(IR_node.get_attr('_output_shapes')[0])
                oshape = clast2first(output_shape)
                dim_str = "'dim': {}".format(oshape)
                dim_str = " reshape_param={'shape': { " + dim_str + '} }'
                reshape_layer_variable_name = IR_node.real_variable_name + "_reshape2"

                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape_layer_variable_name, IR_node.variable_name, dim_str))
                IR_node.real_name = reshape_layer_variable_name
            else:
                self.add_body(1, "n.{:<15} = L.Pqn_Pooling(n.{}, pooling_param=dict({}), ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), pool_param))
        else:
            pads = IR_node.get_attr('pads')[1:-1]
            pads = pads[:len(pads) // 2 - 1] + pads[len(pads) // 2 + 1:]
            spatial_kernels = kernels * 2
            pool_param = "pool={}, kernel_size={}, pad={}, stride={}, pqn_qt=dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={},data_format='{}')".format(pooling_type, kernels, pads, strides, output_scale, output_zero_point, input_scale, input_zero_point,data_format)
            self.add_body(1, "n.{:<15} = L.Pqn_Pooling(n.{}, pooling_param=dict({}), ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), pool_param))

    def emit_RoiAlignMax(self, IR_node):
        in_edges = []
        parent_nodes = []
        for num in range(0, len(IR_node.in_edges)):
            parent = self.IR_graph.get_parent(IR_node.name, [num])
            assert parent is not None
            parent_nodes.append(parent)
            in_edges.append(parent.real_variable_name)
        parent_shape = [shape_to_list(node.get_attr("_output_shapes")[0]) for node in parent_nodes]
        if len(parent_shape[1]) == 4:
            in_edges.reverse()
        dummy_layer_name = IR_node.real_variable_name + '_perm'
        perm_order_str = '{' + "'order': {}".format([0, 2, 3, 1]) + '}'
        self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(dummy_layer_name, in_edges[0], perm_order_str))
        self.add_body(
            1,
            "n.{:<15} = L.ROIAlignMax(n.{},n.{},\
                            spatial_scale={}, sampling_ratio={}, \
                            pooled_height={}, pooled_width={},  \
                            aligned={}, \
                            ntop=1)".format(
                IR_node.variable_name + "_ara_roiam",
                dummy_layer_name,
                in_edges[1],
                # rois,b_idx,
                # IR_node.get_attr('num_rois'),
                IR_node.get_attr('spatial_scale', 1.0),
                IR_node.get_attr('sampling_ratio', -1.0),
                IR_node.get_attr('pooled_height'),
                IR_node.get_attr('pooled_width'),
                IR_node.get_attr('aligned', True),
            ))
        output_shape = clast2first(shape_to_list(IR_node.get_attr('_output_shapes')[0]))
        # perm_order_str = '{' + "'order': {}".format([0,3,2,1]) + '}'
        # dummy_layer_name = IR_node.real_variable_name + '_perm_2'
        # self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(
        #       dummy_layer_name,
        #       IR_node.variable_name,
        #       perm_order_str
        #       ))
        dim_str = "'dim': {}".format(output_shape)
        dim_str = " reshape_param={'shape': { " + dim_str + '} }'
        dummy_layer_name = IR_node.real_variable_name + '_reshape'
        self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(IR_node.variable_name, IR_node.variable_name + "_ara_roiam", dim_str))
        # IR_node.real_name = dummy_layer_name

    def emit_HardSwish(self, IR_node):
        self.add_body(1, "n.{:<15} = L.HardSwish(n.{})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0])))

    def emit_Swish(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Swish(n.{})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0])))
    
    def emit_Elu(self, IR_node):
        alpha = IR_node.get_attr('alpha')
        self.add_body(1, "n.{:<15} = L.ELU(n.{}, elu_param=dict(alpha={}))".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), alpha))

    def emit_Gelu(self, IR_node):
        approximation = IR_node.get_attr('approximation')
        self.add_body(1, "n.{:<15} = L.Gelu(n.{}, gelu_param=dict(approximation={}))".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]),approximation))

    def emit_Softplus(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Softplus(n.{})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0])))

    def emit_Focus(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Focus(n.{})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0])))

    def emit_Rope(self, IR_node):
        input_layers = [self.IR_graph.get_node(edge).real_variable_name for edge in IR_node.in_edges]
        input_layers = ','.join(['n.' + i for i in input_layers])
        self.add_body(1, "n.{:<15} = L.Eltwise({}, broadcast1=[0, 1, 0, 0.5], operation=4, ntop=1)".format(IR_node.variable_name, input_layers))

    def emit_Onnx_Abs(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Onnx_Abs(n.{},ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Onnx_Split(self, IR_node):
        axis = IR_node.layer.attr['axis'].i
        num_splits = IR_node.layer.attr['split'].i
        split_size = IR_node.layer.attr['split_size'].list.i
        slice_point_list = []
        output_layers = IR_node.layer.attr['output_layers'].list.s
        output_layer = []
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        if input_node.type == "DVCast":
            st_idx = 0
            for i in range(num_splits):
                out_layer = str(output_layers[i].decode("utf-8"))
                starts = [st_idx]
                ends = [st_idx + split_size[i]]
                st_idx += split_size[i]
                axes = [axis]
                steps = [1]
                self.add_body(1, "n.{:<15} = L.Onnx_Slice(n.{},starts=dict(shape=dict(shape={},dims={})),ends=dict(shape=dict(shape={},dims={})),axes=dict(shape=dict(shape={},dims={})),steps=dict(shape=dict(shape={},dims={})), isfloat=1, adj_param=dict(precision=[3]))".format(out_layer, self.parent_variable_name(IR_node), starts, len(starts), ends, len(ends), axes, len(axes), steps, len(steps)))
            return
        for i in range(num_splits):
            output_layer.append('n.' + str(output_layers[i].decode("utf-8")))
            if i == 0:
                slice_point_list.append(split_size[0])
            else:
                slice_point_list.append(split_size[i] + slice_point_list[i - 1])
        slice_point_list.pop()
        output_layer = ','.join(output_layer)
        self.add_body(1, "{} = L.Slice(n.{}, slice_param=dict(axis={}, slice_point={}), ntop={})".format(output_layer, self.parent_variable_name(IR_node), axis, slice_point_list, num_splits))

    def emit_Dummy(self, IR_node):
        input_node_name = self.parent_variable_name(IR_node)
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert input_node is not None
        input_shape = clast2first(shape_to_list(input_node.get_attr("_output_shapes")[0]))
        output_shape = clast2first(shape_to_list(IR_node.get_attr('_output_shapes')[0]))
        if np.prod(input_shape) == np.prod(output_shape):
            # put reshape
            dim_str = "'dim': {}".format(output_shape)
            dim_str = " reshape_param={'shape': { " + dim_str + '} }'
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(IR_node.variable_name, self.parent_variable_name(IR_node), dim_str))
            return
        last_layer = self.parent_variable_name(IR_node)
        rshape = input_shape
        if len(input_shape) != len(output_shape):
            if len(input_shape) > len(output_shape):
                rshape = input_shape[:len(output_shape) - 1] + [np.prod(input_shape[len(output_shape) - 1:])]
            elif len(input_shape) < len(output_shape):
                rshape = input_shape + [1] * (len(output_shape) - len(input_shape))
            dim_str = "'dim': {}".format(rshape)
            dim_str = " reshape_param={'shape': { " + dim_str + '} }'
            last_layer = IR_node.variable_name + '_reshape'
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(last_layer, self.parent_variable_name(IR_node), dim_str))
        if np.any(np.greater(output_shape, rshape)):
            if not np.any(np.greater(rshape, output_shape)):
                pad_name = IR_node.variable_name
            else:
                pad_name = IR_node.variable_name + "_padding"
            pad_value = [output_shape[i] - rshape[i] if output_shape[i] > rshape[i] else 0 for i in range(len(rshape))]
            pad_n = pad_value[0]
            pad_c = pad_value[1]
            pad_value = pad_value[2:] + [0] * len(pad_value[2:]) + [pad_c] + [0] + [pad_n] + [0]
            self.add_body(1, "n.{:<15} = L.Padding(n.{}, padding_param=dict(pad={}, pad_method={},pad_value={}), ntop=1)".format(pad_name, last_layer, pad_value, 0, 0))
            IR_node.real_name = pad_name
            last_layer = pad_name
        if np.any(np.greater(rshape, output_shape)):
            new_shape = "{}".format(str(output_shape))
            self.add_body(1, "n.{:<15} = L.Crop(n.{}, L.DummyData(shape=[dict(dim={})], \
                    ntop=1), ntop=1, offset={},axis=0)".format(IR_node.variable_name, last_layer, new_shape, [0] * len(output_shape)))
            IR_node.real_name = IR_node.variable_name

    def emit_Mxnet_Sequencemask(self, IR_node):
        input_names = []
        for num in range(0, len(IR_node.in_edges)):
            parent = self.IR_graph.get_parent(IR_node.name, [num])
            assert parent is not None
            input_names.append('n.' + parent.real_variable_name)
        input_layers = ', '.join(input_names)
        self.add_body(1, "n.{:<15} = L.Mxnet_Sequencemask({},use_sequence_length={},axis={},value={})".format(IR_node.variable_name, input_layers, IR_node.get_attr('use_sequence_length'), IR_node.get_attr('axis'), IR_node.get_attr('value')))

    def emit_Sequencemask(self, IR_node):
        input_names = []
        for num in range(0, len(IR_node.in_edges)):
            parent = self.IR_graph.get_parent(IR_node.name, [num])
            assert parent is not None
            input_names.append('n.' + parent.real_variable_name)
        input_layers = ', '.join(input_names)
        self.add_body(1, "n.{:<15} = L.Sequencemask({},use_sequence_length={},axis={},value={})".format(IR_node.variable_name, input_layers, IR_node.get_attr('use_sequence_length'), IR_node.get_attr('axis'), IR_node.get_attr('value')))

    def emit_MaskSoftmax(self, IR_node):
        input_names = []
        for num in range(0, len(IR_node.in_edges)):
            parent = self.IR_graph.get_parent(IR_node.name, [num])
            assert parent is not None
            input_names.append('n.' + parent.real_variable_name)
        input_layers = ', '.join(input_names)
        self.add_body(1, "n.{:<15} = L.MaskSoftmax({},use_sequence_length={},axis={},value={})".format(IR_node.variable_name, input_layers, IR_node.get_attr('use_sequence_length'), IR_node.get_attr('axis'), IR_node.get_attr('value')))

    def emit_Embedding(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Embedding(n.{},input_dim={},output_dim={})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), IR_node.get_attr('input_dim'), IR_node.get_attr('output_dim')))
        self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)

    def emit_Mxnet_Embedding(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Mxnet_Embedding(n.{},input_dim={},output_dim={},sparse_grad={})".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), IR_node.get_attr('input_dim'), IR_node.get_attr('output_dim'), IR_node.get_attr('sparse_grad', "False")))
        self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)

    def emit_DistributeFpn(self, IR_node):
        out_list = IR_node.get_attr("out_layers")
        out_layers = [n.decode("utf-8") for n in out_list]
        out_layers = ', '.join(['n.' + i for i in out_layers])
        self.add_body(1, "{} = L.DistributeFpn(n.{},name='{}',roi_canonical_level={},roi_canonical_scale={},roi_max_level={},roi_min_level={},legacy_plus_one={},ntop=5)".format(out_layers, self.parent_variable_name(IR_node, [0]), IR_node.variable_name, IR_node.get_attr('roi_canonical_level'), IR_node.get_attr('roi_canonical_scale'), IR_node.get_attr('roi_max_level'), IR_node.get_attr('roi_min_level'), IR_node.get_attr('legacy_plus_one')))

    def emit_BoxWithNMSLimit(self, IR_node):
        parent0 = self.IR_graph.get_parent(IR_node.name, [0])
        assert parent0 is not None
        parent1 = self.IR_graph.get_parent(IR_node.name, [1])
        assert parent1 is not None
        parent2 = self.IR_graph.get_parent(IR_node.name, [2])
        assert parent2 is not None
        input_nodes = [parent0.real_variable_name.replace('/', '_'), parent1.real_variable_name.replace('/', '_'), parent2.real_variable_name.replace('/', '_')]
        out_list = IR_node.get_attr("out_layers")
        out_layers = [n.decode("utf-8") for n in out_list]
        out_layers[0] = IR_node.name
        # following is a hack bcz onnx model doesn't produce all outputs
        out_layers = ', '.join(['n.' + i for i in out_layers[:3]])
        input_layers = ', '.join(('n.' + node_name) for node_name in input_nodes)
        self.add_body(1, "{}= L.BoxWithNMSLimit({},name='{}',score_thres={},nms={},detection_per_im={},soft_nms_enabled={},soft_nms_method='{}',soft_nms_sigma={},soft_nms_min_score_thres={},rotated={},cls_agnostic_bbox_reg={},input_boxes_include_bg_cls={},output_classes_include_bg_cls={},legacy_plus_one={},ntop=3)".format(out_layers, input_layers, IR_node.variable_name, IR_node.get_attr('score_thres'), IR_node.get_attr('nms'), IR_node.get_attr('detection_per_im'), IR_node.get_attr('soft_nms_enabled'), IR_node.get_attr('soft_nms_method'), IR_node.get_attr('soft_nms_sigma'), IR_node.get_attr('soft_nms_min_score_thres'), IR_node.get_attr('rotated'), IR_node.get_attr('cls_agnostic_bbox_reg'), IR_node.get_attr('input_boxes_include_bg_cls'), IR_node.get_attr('output_classes_include_bg_cls'), IR_node.get_attr('legacy_plus_one')))

    def emit_GenerateProposals(self, IR_node):
        perm_order_str = '{' + "'order': {}".format([0, 2, 3, 1]) + '}'
        parent0 = self.IR_graph.get_parent(IR_node.name, [0])
        assert parent0 is not None
        parent1 = self.IR_graph.get_parent(IR_node.name, [1])
        assert parent1 is not None
        self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(IR_node.variable_name + '_perm_scores', parent0.real_variable_name, perm_order_str))
        self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(IR_node.variable_name + '_perm_boxes', parent1.real_variable_name, perm_order_str))
        in_list = []
        for num in range(0, len(IR_node.in_edges)):
            parent = self.IR_graph.get_parent(IR_node.name, [num])
            assert parent is not None
            in_list.append(parent.real_variable_name)
        in_list[0] = IR_node.variable_name + '_perm_scores'
        in_list[1] = IR_node.variable_name + '_perm_boxes'
        out_list = IR_node.get_attr("out_layers")
        input_layers = ', '.join(['n.' + i for i in in_list])
        out_layers = [n.decode("utf-8") for n in out_list]
        out_layers = ', '.join(['n.' + i for i in out_layers])
        self.add_body(
        1,
        "{} = L.GenerateProposals({},name='{}',spatial_scale={},pre_nms_topN={},post_nms_topN={},nms_thresh={},min_size={},angle_bound_on={},angle_bound_lo={},angle_bound_hi={},clip_angle_thresh={},legacy_plus_one={},ntop=2)".format(
            # self.parent_variable_name(IR_node,[0]),
            out_layers,
            input_layers,
            IR_node.variable_name,
            IR_node.get_attr('spatial_scale'),
            IR_node.get_attr('pre_nms_topN'),
            IR_node.get_attr('post_nms_topN'),
            IR_node.get_attr('nms_thresh'),
            IR_node.get_attr('min_size'),
            IR_node.get_attr('angle_bound_on'),
            IR_node.get_attr('angle_bound_lo'),
            IR_node.get_attr('angle_bound_hi'),
            IR_node.get_attr('clip_angle_thresh'),
            IR_node.get_attr('legacy_plus_one')))
        self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)

    def emit_Upsample(self, IR_node):
        logger.warn("Interpolation of any kind does not support AntiAliasing post processing yet. Outputs will be different if antialiasing is applied")
        output_shape = IR_node.get_attr("_output_shapes")[0]
        input_shape = IR_node.get_attr('shape')
        # Input shape is n,d,h,w,c
        output_shape = shape_to_list(output_shape)
        if len(input_shape) != len(output_shape):
            logger.error("output and input shape ranks different for upsample")
            raise ValueError()
        if len(input_shape) != 4 and len(input_shape) != 3 and len(input_shape) != 5:
            logger.error("expected 3 or 4 or 5 dimensional tensor for Interpolation, recieved %d", len(input_shape))
            raise ValueError()
        mode = IR_node.get_attr('mode')
        align_mode = IR_node.get_attr('align_corners')
        if align_mode is False:
            assert len(output_shape) == 4 or len(output_shape) == 5
            assert mode == 3 or mode == 6 or mode == 7
            if mode == 3:
                scale_h = (output_shape[1]) / (input_shape[1])
                scale_w = (output_shape[2]) / (input_shape[2])
                self.emit_ONNX_UpSample(IR_node)
                return
            elif mode == 6:
                scale = IR_node.get_attr('scale')
                self.add_body(1, "n.{:<15} = L.Upsample(n.{}, scale = {}, zero_mode = {}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), [scale], [1]))
            elif mode == 7:
                scale_d = (output_shape[1]) / (input_shape[1])
                scale_h = (output_shape[2]) / (input_shape[2])
                scale_w = (output_shape[3]) / (input_shape[3])
                if scale_h != scale_w:  # or scale_d != scale_w:
                    logger.error("currently dvmc doesn't support uneven scales")
                    raise NotImplementedError()
                if not scale_h.is_integer():
                    logger.error("currently dvmc doesn't support non integer (outshape)/(inshape)")
                    raise NotImplementedError()
                scale_h = int(scale_h)
                scale_w = int(scale_w)
                scale_d = int(scale_d)
                if scale_d == scale_w:
                    self.add_body(1, "n.{:<15} = L.RepEdgeDeconv(n.{}, convolution_param=dict(kernel_h={}, kernel_w={}, kernel_d={},stride={}, num_output={}, pad_h={}, pad_w={}, pad_d={}, rep_edge_r=1,rep_edge_b=1,rep_edge_l=1,rep_edge_t=1,rep_edge_f=1,rep_edge_ba=1,bias_term={},group={}), ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), scale_h * 2, scale_w * 2, scale_d * 2, [scale_h, scale_d, scale_w], input_shape[-1], 1, 1, 1, IR_node.get_attr('use_bias', False), input_shape[-1]))
                else:
                    self.add_body(1, f"n.{'{:<15}'.format(IR_node.variable_name)} = L.RepEdgeDeconv(n.{self.parent_variable_name(IR_node)}, convolution_param=dict(kernel_h={scale_h * 2}, kernel_w={scale_w * 2}, kernel_d={scale_d * 2 if scale_d > 1 else 1},stride={[scale_d, scale_h, scale_w]}, num_output={input_shape[-1]}, pad_h=1, pad_w=1, pad_d={0 if scale_d == 1 else 1}, rep_edge_r=1,rep_edge_b=1,rep_edge_l=1,rep_edge_t=1,rep_edge_f={0 if scale_d == 1 else 1},rep_edge_ba={0 if scale_d == 1 else 1},bias_term={IR_node.get_attr('use_bias', False)},group={input_shape[-1]}), ntop=1)")
                # crop_layer_variable_name = IR_node.real_variable_name + "_crop"
                # self.add_body(1,
                #         "n.{:<15} = L.Crop(n.{}, L.DummyData(shape=[dict(dim=[1, {}, {}, {}, {}])], \
                #             ntop=1), ntop=1, offset={})".format(
                #             crop_layer_variable_name,
                #             IR_node.real_variable_name,
                #             output_shape[4],
                #             output_shape[1],
                #             output_shape[2],
                #             output_shape[3],
                #             [int(scale_d/2),
                #              int(scale_h/2),
                #             int(scale_w/2)]))
                # IR_node.real_name = crop_layer_variable_name
                filter_row = [i / (scale_w * 2) for i in range(1, scale_w * 2, 2)]
                filter_row.extend(list(reversed(filter_row)))
                filter_row = np.array(filter_row)
                w = len(filter_row)
                filter_2d = np.zeros([w, w])
                for i in range(w):
                    filter_2d[i] = filter_row * filter_row[i]
                wts = filter_2d
                if scale_d == scale_w:
                    filter_3d = np.zeros([w, w, w])
                    for i in range(w):
                        filter_3d[i] = filter_2d * filter_row[i]
                    wts = filter_3d
                if self.weight_loaded:
                    self.weights_dict[IR_node.variable_name] = {'weights': wts}
        elif align_mode == "NearestNeighbor":
            assert len(output_shape) in [3, 4, 5]
            assert mode in [0, 2, 5]
            zero_mode = []
            if mode == 2:
                output_shape = IR_node.get_attr("_output_shapes")[0]
                output_shape = shape_to_list(output_shape)
                input_shape = IR_node.get_attr('shape')
                # a custom mode we implemented, originally not present in any network
                if 'zero_mode' in IR_node.layer.attr:
                    zero_mode = IR_node.get_attr('zero_mode')
                elif len(output_shape) in [3, 4, 5]:
                    zero_mode = [0] * (len(output_shape) - 2)
                if len(input_shape) != len(output_shape):
                    logger.error("output and input shape ranks different for upsample")
                    raise ValueError()
                if len(input_shape) in [4, 5]:
                    if input_shape[-1] != output_shape[-1]:
                        logger.error("currently only spatial dimension can be upsampled")
                        raise NotImplementedError()
                    output_shape_spatial = output_shape[1:-1]
                    input_shape_spatial = input_shape[1:-1]
                    scales = [oo / ii for oo, ii in zip(output_shape_spatial, input_shape_spatial)]
                    if scales[-2] != scales[-1]:
                        logger.error("currently dvnc doesn't support uneven scales across H and W")
                        raise NotImplementedError()
                    elif len(scales) == 3 and (scales[0] not in [1, scales[-1]]):
                        logger.error("currently for 3D networks dvnc only supports cases (scale_d = scale_h = scale_w) or (scale_d = 1, scale_h = scale_w)")
                        raise NotImplementedError()
                    croppable = False
                    if all([oo == (math.ceil(ss) * ii - 1) for ss, oo, ii in zip(scales, output_shape_spatial, input_shape_spatial)]):
                        croppable = True
                        scales = [float(math.ceil(ss)) for ss in scales]
                    if (isinstance(scales[-1], int) or not scales[-1].is_integer()) and not croppable:
                        logger.error("currently dvnc doesn't support non integer,non croppable scales")
                        raise NotImplementedError()
                    self.add_body(1, "n.{:<15} = L.Upsample(n.{}, scale={}, zero_mode={}, ntop=1)".format(
              IR_node.variable_name,
              self.parent_variable_name(IR_node),
              list(map(int, scales)),
              zero_mode,
          ))
                    if croppable:
                        crop_layer_variable_name = IR_node.real_variable_name + "_crop"
                        offset = [0] * (len(output_shape) - 2)
                        self.add_body(1, "n.{:<15} = L.Crop(n.{}, L.DummyData(shape=[dict(dim=[1, {}, {}])], \
                                    ntop=1), ntop=1, offset={})".format(crop_layer_variable_name, IR_node.real_variable_name, output_shape[-1], ','.join(str(i) for i in output_shape[1:-1]), offset))
                        IR_node.real_name = crop_layer_variable_name
                    return
                else:
                    logger.error("expected 4 or 5 dimensional tensor for upsample")
                    raise ValueError()
            elif mode == 5:
                output_shape = IR_node.get_attr("_output_shapes")[0]
                output_shape = shape_to_list(output_shape)
                input_shape = IR_node.get_attr('shape')
                output_shape_spatial = output_shape[1:-1]
                input_shape_spatial = input_shape[1:-1]
                zero_mode = IR_node.get_attr("zero_mode")
                scales = [oo / ii for oo, ii in zip(output_shape_spatial, input_shape_spatial)]
                self.add_body(1, "n.{:<15} = L.Upsample(n.{}, scale = {}, zero_mode = {}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), list(map(int, scales)), zero_mode))
                return
        elif align_mode is True:
            assert mode in [4, 7]
            scale = IR_node.get_attr('scale')
            if mode == 4:
                if ((input_shape[1] == input_shape[2]) or (input_shape[2] == input_shape[3])) and input_shape[2] == 1:
                    logger.info("encountered input spatial dimensions as 1")
                    if (input_shape[1] == input_shape[2]):
                        # self.emit_ResizeBilinear(IR_node=IR_node)
                        nchw_outshape = [output_shape[0], output_shape[3], output_shape[1], output_shape[2]]
                        dim_str = "'dim': {}".format(nchw_outshape)
                        dim_str = "{'shape': { " + dim_str + '} }'
                        self.add_body(1, "n.{:<15} = L.Broadcast(n.{},broadcast_param = {}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), dim_str))
                        return
                    elif (input_shape[2] == input_shape[3]):
                        nchw_outshape = [output_shape[0], output_shape[1], output_shape[2], output_shape[3]]
                        dim_str = "'dim': {}".format(nchw_outshape)
                        dim_str = "{'shape': { " + dim_str + '} }'
                        self.add_body(1, "n.{:<15} = L.Broadcast(n.{},broadcast_param = {}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), dim_str))
                        return
                else:
                    scale_h = output_shape[1] / input_shape[1]
                    scale_w = output_shape[2] / output_shape[2]
                    if (not scale_h.is_integer() or not scale_w.is_integer()):
                        scale_h = (output_shape[1] - 1) / (input_shape[1] - 1)
                        scale_w = (output_shape[2] - 1) / (input_shape[2] - 1)
                        if scale_h != scale_w:
                            logger.error("currently dvmc doesn't support uneven scales")
                            raise NotImplementedError()
                        if not scale_h.is_integer():
                            logger.error("currently dvmc doesn't support non integer (outshape-1)/(inshape-1)")
                            raise NotImplementedError()
                        scale_h = int(scale_h)
                        scale_w = int(scale_w)
                        self.add_body(1, "n.{:<15} = L.RepEdgeDeconv(n.{}, convolution_param=dict(kernel_h={}, kernel_w={}, stride={}, num_output={}, pad_h={}, pad_w={}, bias_term={},group={}), ntop=1)".format(IR_node.variable_name + "_ara_redeconv", self.parent_variable_name(IR_node), scale_h * 2, scale_w * 2, scale_h, input_shape[-1], 0, 0, IR_node.get_attr('use_bias', False), input_shape[-1]))
                        crop_layer_variable_name = IR_node.real_variable_name + "_crop"
                        self.add_body(1, "n.{:<15} = L.Crop(n.{}, L.DummyData(shape=[dict(dim=[1, {}, {}, {}])], \
                                    ntop=1), ntop=1, offset={})".format(IR_node.variable_name, IR_node.variable_name + "_ara_redeconv", output_shape[3], output_shape[1], output_shape[2], [scale_h - 1, scale_w - 1]))
                        # IR_node.real_name = crop_layer_variable_name
                        half_filter_row = [i / scale_w for i in range(1, scale_w)]
                        filter_row = np.array([half_filter_row + [1] + list(reversed(half_filter_row)) + [0]])
                        filter = filter_row.T * filter_row
                        if self.weight_loaded:
                            self.weights_dict[IR_node.variable_name + "_ara_redeconv"] = {'weights': filter}
                    else:
                        scale_rows = (input_shape[1] - 1) / (output_shape[1] - 1)
                        scale_cols = (input_shape[2] - 1) / (output_shape[2] - 1)
                        if scale is None:
                            scale = [int(output_shape[1] / input_shape[1]), scale_rows, scale_cols]
                        self.add_body(1, "n.{:<15} = L.LinearUpsample(n.{}, scale = {}, upsample_mode = {}, zero_mode = {}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), scale, mode, [0]))
            elif mode == 7:
                scale_depth = (max(1, input_shape[1] - 1)) / (output_shape[1] - 1)
                scale_rows = (max(1, input_shape[2] - 1)) / (output_shape[2] - 1)
                scale_cols = (max(1, input_shape[3] - 1)) / (output_shape[3] - 1)
                self.add_body(1, "n.{:<15} = L.LinearUpsample(n.{}, scale = {}, upsample_mode = {}, zero_mode = {}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), [scale, scale_depth, scale_rows, scale_cols], mode, [0]))
            else:
                logger.error("Invalid Mode for given Align_corners = True")
                raise ValueError()
        else:
            logger.error("Invalid Align Corners Encountered. Got {}, shoud be one of False, NearestNeighbor, True".format(align_mode))
            raise ValueError()

    def emit_CollectRpnProposals(self, IR_node):
        in_list = []
        for num in range(0, len(IR_node.in_edges)):
            parent = self.IR_graph.get_parent(IR_node.name, [num])
            assert parent is not None
            in_list.append('n.' + parent.real_variable_name)
        input_layers = ', '.join(in_list)
        self.add_body(1, "n.{:<15} = L.CollectRpnProposals({},rpn_max_level={},rpn_min_level={},rpn_post_nms_topN={})".format(IR_node.variable_name, input_layers, IR_node.get_attr('rpn_max_level'), IR_node.get_attr('rpn_min_level'), IR_node.get_attr('rpn_post_nms_topN')))

    def emit_Onnx_BatchPermutation(self, IR_node):
        in_list = []
        for num in range(0, len(IR_node.in_edges)):
            parent = self.IR_graph.get_parent(IR_node.name, [num])
            assert parent is not None
            in_list.append('n.' + parent.real_variable_name)
        input_layers = ', '.join(in_list)
        self.add_body(1, "n.{:<15} = L.Onnx_Batchpermutation({})".format(IR_node.variable_name, input_layers))

    def emit_Onnx_GroupNorm(self, IR_node):
        reshape_layer_name = IR_node.name + '_reshape'
        IR_node_reshape = self.IR_graph.get_node(reshape_layer_name)
        out_shape = IR_node_reshape.get_attr("shape")
        dim_str = "'dim': {}".format(out_shape)
        dim_str = " reshape_param={'shape': { " + dim_str + '} }'
        self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape_layer_name, self.parent_variable_name(IR_node), dim_str))
        eps = IR_node.get_attr('eps')
        self.add_body(1, "n.{:<15} = L.Mxnet_Instancenorm(n.{},eps={}, ntop=1)".format(IR_node.variable_name, reshape_layer_name, eps))
        reshape_layer_name = IR_node.name + '_reshape_1'
        IR_node_reshape = self.IR_graph.get_node(reshape_layer_name)
        out_shape = IR_node_reshape.get_attr("shape")
        dim_str = "'dim': {}".format(out_shape)
        dim_str = " reshape_param={'shape': { " + dim_str + '} }'
        self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(IR_node_reshape.name, IR_node.variable_name, dim_str))
        IR_node.real_name = reshape_layer_name

    def emit_BBoxTransform(self, IR_node):
        in_list = []
        for num in range(0, len(IR_node.in_edges)):
            parent = self.IR_graph.get_parent(IR_node.name, [num])
            assert parent is not None
            in_list.append('n.' + parent.real_variable_name)
        input_layers = ', '.join(in_list)
        out_list = IR_node.get_attr("out_layers")
        out_layers = [n.decode("utf-8") for n in out_list]
        out_layers = ', '.join(['n.' + i for i in out_layers])
        self.add_body(
        1,
        "{} = L.BBoxTransform({},name='{}',apply_scale={},angle_bound_on={},angle_bound_lo={},angle_bound_hi={},clip_angle_thresh={},legacy_plus_one={},rotated={},weights={},ntop=2)".format(
            # self.parent_variable_name(IR_node,[0]),
            out_layers,
            input_layers,
            IR_node.variable_name,
            IR_node.get_attr('apply_scale'),
            IR_node.get_attr('angle_bound_on'),
            IR_node.get_attr('angle_bound_lo'),
            IR_node.get_attr('angle_bound_hi'),
            IR_node.get_attr('clip_angle_thresh'),
            IR_node.get_attr('legacy_plus_one'),
            IR_node.get_attr('rotated'),
            IR_node.get_attr('weights')))

    def emit_BBoxTransform_detectron2(self, IR_node):
        input_layers = ', '.join(('n.' + self.IR_graph.get_node(edge).real_variable_name) for edge in IR_node.in_edges)
        self.add_body(1, "n.{} = L.BBoxTransform_detectron2({},scale_clamp={},weights={},static_boxes=False,clip_bbox={},clips={},ntop=1)".format(IR_node.variable_name, input_layers, IR_node.get_attr('scale_clamp'), IR_node.get_attr('weights'), IR_node.get_attr('clip_bbox'), IR_node.get_attr('clip_param')))

    def emit_BBoxTransform_static_detectron2(self, IR_node):
        input_layers = ', '.join(('n.' + self.IR_graph.get_node(edge).real_variable_name) for edge in IR_node.in_edges)
        self.add_body(1, "n.{} = L.BBoxTransform_detectron2({},scale_clamp={},weights={},static_boxes=True,ntop=1)".format(IR_node.variable_name, input_layers, IR_node.get_attr('scale_clamp'), IR_node.get_attr('weights')))

    def emit_Onnx_BBoxTransform(self, IR_node):
        in_list = []
        for num in range(0, len(IR_node.in_edges)):
            parent = self.IR_graph.get_parent(IR_node.name, [num])
            assert parent is not None
            in_list.append('n.' + parent.real_variable_name)
        input_layers = ', '.join(in_list)
        out_list = IR_node.get_attr("out_layers")
        out_layers = [n.decode("utf-8") for n in out_list]
        out_layers = ', '.join(['n.' + i for i in out_layers])
        self.add_body(
        1,
        "{} = L.Onnx_Bboxtransform({},name='{}',apply_scale={},angle_bound_on={},angle_bound_lo={},angle_bound_hi={},clip_angle_thresh={},legacy_plus_one={},rotated={},weights={},ntop=2)".format(
            # self.parent_variable_name(IR_node,[0]),
            out_layers,
            input_layers,
            IR_node.variable_name,
            IR_node.get_attr('apply_scale'),
            IR_node.get_attr('angle_bound_on'),
            IR_node.get_attr('angle_bound_lo'),
            IR_node.get_attr('angle_bound_hi'),
            IR_node.get_attr('clip_angle_thresh'),
            IR_node.get_attr('legacy_plus_one'),
            IR_node.get_attr('rotated'),
            IR_node.get_attr('weights')))

    def emit_LoopEnd(self, IR_node):
        self.add_body(1, "n.{:<15} = L.Identity(n.{}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Pytorch_half_pixel_resize(self, IR_node):
        output_shape = IR_node.get_attr("_output_shapes")[0]
        output_shape = shape_to_list(output_shape)
        parent_node = self.IR_graph.get_parent(IR_node.name, [0])
        assert parent_node is not None
        input_shape = parent_node.get_attr("_output_shapes")[0]
        input_shape = shape_to_list(input_shape)
        scale_h = int(output_shape[1] / input_shape[1])
        scale_w = int(output_shape[2] / input_shape[2])
        self.add_body(1, "n.{:<15} = L.RepEdgeDeconv(n.{}, convolution_param=dict(kernel_h={}, kernel_w={}, stride={}, num_output={}, pad_h={}, pad_w={}, rep_edge_r=1,rep_edge_b=1,rep_edge_l=1,rep_edge_t=1,bias_term={},group={}), ntop=1)".format(IR_node.variable_name + "_ara_redeconv", self.parent_variable_name(IR_node), scale_h * 2, scale_w * 2, scale_h, input_shape[-1], 0, 0, IR_node.get_attr('use_bias', False), input_shape[-1]))
        # crop_layer_variable_name = IR_node.real_variable_name + "_crop"
        self.add_body(1, "n.{:<15} = L.Crop(n.{}, L.DummyData(shape=[dict(dim=[1, {}, {}, {}])], \
                    ntop=1), ntop=1, offset={})".format(IR_node.variable_name, IR_node.variable_name + "_ara_redeconv", output_shape[3], output_shape[1], output_shape[2], [int(scale_h / 2), int(scale_w / 2)]))
        assert scale_h == scale_w
        if scale_w % 2 == 0:
            half_filter_row = [(2 * i - 1) / (2 * scale_w) for i in range(1, scale_w + 1)]
            filter_row = np.array([half_filter_row + list(reversed(half_filter_row))])
            filter = filter_row.T * filter_row
        else:
            half_filter_row = [i / scale_w for i in range(1, scale_w)]
            filter_row = np.array([half_filter_row + [1] + list(reversed(half_filter_row)) + [0]])
            filter = filter_row.T * filter_row
        if self.weight_loaded:
            self.weights_dict[IR_node.variable_name + "_ara_redeconv"] = {'weights': filter}
        # IR_node.real_name = crop_layer_variable_name

    def emit_Broadcast(self, IR_node):
        shape = clast2first(shape_to_list(IR_node.get_attr("_output_shapes")[0]))
        dim_str = "'dim': {}".format(shape)
        dim_str = "{'shape': { " + dim_str + '} }'
        self.add_body(1, "n.{:<15} = L.Broadcast(n.{},broadcast_param = {}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), dim_str))

    def emit_ROIPool(self, IR_node):
        self.add_body(1, "n.{:<15} = L.ROIPool(n.{}, n.{}, pooled_shape={}, spatial_scale={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), self.parent_variable_name(IR_node, [1]), IR_node.get_attr('pooled_shape'), IR_node.get_attr('spatial_scale')))

    def emit_ATen_emitter(self, IR_node):
        in_list = []
        for num in range(0, len(IR_node.in_edges)):
            parent = self.IR_graph.get_parent(IR_node.name, [num])
            assert parent is not None
            in_list.append('n.' + parent.real_variable_name)
        input_layers = ', '.join(in_list)
        out_layers = None
        out_count = 1
        self.weights_dict.pop(IR_node.name, None)

        if IR_node.get_attr("out_layers"):
            out_list = IR_node.get_attr("out_layers")
            out_layers = [n.decode("utf-8") for n in out_list]
            out_count = len(out_layers)
            out_layers = ', '.join(['n.' + i for i in out_layers])
        else:
            out_layers = 'n.' + IR_node.real_variable_name
        shape_list = []
        if IR_node.get_attr("out_shape"):
            shape_str = IR_node.get_attr("out_shape")
            shape_str_list = shape_str.split(';')
            for i in shape_str_list:
                x = i.split(',')
                z = []
                for y in x:
                    z.append(int(y))
                shape_list.append(z)
        elif IR_node.get_attr('_output_shapes'):
            for i in range(out_count):
                for shape in IR_node.get_attr('_output_shapes'):
                    shape_list.append(clast2first(shape_to_list(shape)))
        shape_list = shape_list[:out_count]
        aten_params = "shape_param=["
        for x in shape_list:
            aten_params += "dict(dim=[{}]),".format(','.join(str(item) for item in x))
        aten_params = aten_params[:-1]
        aten_params += "]"
        param = "aten_param=dict({},op_params='{}',original_layer_type='{}', value={})".format(aten_params, "None", IR_node.type, 1)
        if len(IR_node.in_edges):
            s = "{} = L.Aten({}, name='{}', {}, ntop={})".format(out_layers, input_layers, IR_node.variable_name, param, out_count)
        else:
            s = "{} = L.Aten(name='{}', {}, ntop={})".format(out_layers, IR_node.variable_name, param, out_count)
        if IR_node.variable_name in self.weights_dict:
            self.weights_dict.pop(IR_node.variable_name)
        self.add_body(1, s)

    def emit_Aten(self, IR_node):
        in_list = []
        for num in range(0, len(IR_node.in_edges)):
            parent = self.IR_graph.get_parent(IR_node.name, [num])
            assert parent is not None
            in_list.append('n.' + parent.real_variable_name)
        input_layers = ', '.join(in_list)
        out_list = IR_node.get_attr("out_layers")
        out_count = IR_node.get_attr("out_count")
        out_layers = [n.decode("utf-8") for n in out_list]
        if (len(out_layers) > 1):
            out_layers = ', '.join(['n.' + self.IR_graph.get_node(i).variable_name for i in out_layers])
        else:
            out_layers = "n.{}".format(IR_node.variable_name)
        shape_str = IR_node.get_attr("out_shape")
        shape_str_list = shape_str.split(';')
        shape_list = []
        for i in shape_str_list:
            x = i.split(',')
            z = []
            for y in x:
                z.append(int(y))
            shape_list.append(z)
        aten_params = "shape_param=["
        for x in shape_list:
            aten_params += "dict(dim=[{}]),".format(','.join(str(item) for item in x))
        aten_params = aten_params[:-1]
        aten_params += "]"
        op_params = IR_node.get_attr("op_params")
        if isinstance(op_params, bytes):
            op_params = op_params.decode("utf-8")
        param = "aten_param=dict({},op_params='{}',original_layer_type='{}',value={})".format(aten_params, op_params, IR_node.get_attr('original_layer_type', 'Not_provided'), IR_node.get_attr('value', 1))
        caffe_params_ = ""
        if len(input_layers):
            caffe_params_ += "{}, ".format(input_layers)
        caffe_params_ += "name='{}', {}, ntop={}".format(IR_node.variable_name, param, out_count)
        s = "{} = L.Aten({})".format(out_layers, caffe_params_)
        self.add_body(1, s)

    def emit_L1Norm(self, IR_node):
        self.add_body(1, "n.{:<15} = L.L1Norm(n.{}, axis={}, eps={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), IR_node.get_attr('axis'), IR_node.get_attr('eps', 1e-5)))

    def emit_Hardsigmoid(self, IR_node):
        alpha = IR_node.get_attr("alpha")
        beta = IR_node.get_attr("beta")
        self.add_body(1, "n.{:<15} = L.Hardsigmoid(n.{}, alpha={}, beta={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), alpha, beta))

    def emit_Sin(self, IR_node):
        self.add_body(1, "n.{} = L.Sin(n.{}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))

    def emit_Cos(self, IR_node):
        self.add_body(1, "n.{} = L.Cos(n.{}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node)))
    
    def emit_NMSPose(self, IR_node):
        Precision = {
            'INT8': 0,
            'INT16': 1,
            'INT32': 2,
            'FP32': 3,
            'ZERO': 999,
        }
        prec = Precision['FP32']
        self.add_body(1, "n.{}, n.{} = L.NMSPose(n.{}, n.{}, n.{}, n.{}, center_point_box={}, max_output_boxes_per_class={}, iou_threshold={}, score_threshold={}, ntop=2, adj_param=dict(precision=[{}]))".format(IR_node.get_attr("out_layers")[0].decode(), IR_node.get_attr("out_layers")[1].decode(), self.parent_variable_name(IR_node, [0]), self.parent_variable_name(IR_node, [1]), self.parent_variable_name(IR_node, [2]), self.parent_variable_name(IR_node, [3]), IR_node.get_attr("center_point_box"), IR_node.get_attr("max_output_boxes_per_class"), IR_node.get_attr("iou_threshold"), IR_node.get_attr("score_threshold"), prec))
    
    def emit_NMS(self, IR_node):
        self.add_body(1, "n.{} = L.TVM_NMS(n.{}, n.{}, center_point_box={}, max_output_boxes_per_class={}, iou_threshold={}, score_threshold={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), self.parent_variable_name(IR_node, [1]), IR_node.get_attr("center_point_box"), IR_node.get_attr("max_output_boxes_per_class"), IR_node.get_attr("iou_threshold"), IR_node.get_attr("score_threshold")))

    def emit_NMSRtmdet(self, IR_node):
        self.add_body(1, "n.{}, n.{} = L.NMSRtmdet(n.{}, n.{}, n.{}, center_point_box={}, max_output_boxes_per_class={}, iou_threshold={}, score_threshold={}, mul_const_input={}, less_input_value={}, where_x_value={}, gather_indices_val={}, gather1_indices_val={}, concat_const_input={}, concat_input={}, ntop=2)".format(IR_node.get_attr("out_layers")[0].decode(), IR_node.get_attr("out_layers")[1].decode(), self.parent_variable_name(IR_node, [0]), self.parent_variable_name(IR_node, [1]), self.parent_variable_name(IR_node, [2]), IR_node.get_attr("center_point_box"), IR_node.get_attr("max_output_boxes_per_class"), IR_node.get_attr("iou_threshold"), IR_node.get_attr("score_threshold"), IR_node.get_attr("mul_const_input"),  IR_node.get_attr("less_input_value"),  IR_node.get_attr("where_x_value"),  IR_node.get_attr("gather_indices_val"),  IR_node.get_attr("gather1_indices_val"), IR_node.get_attr("concat_const_input"), IR_node.get_attr("concat_input")))

    def emit_Yolov10nPost(self, IR_node):
        self.add_body(1, "n.{} = L.Yolov10nPost(n.{}, n.{}, split_value={}, k_value={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), self.parent_variable_name(IR_node, [1]), IR_node.get_attr("split_value"), IR_node.get_attr("k_value")))

    def emit_Logits(self, IR_node):
        self.add_body(1, "n.{} = L.Logits(n.{}, min0_value={}, max0_value={}, min1_value={}, min2_value={}, ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node, [0]), IR_node.get_attr("min0_value"), IR_node.get_attr("max0_value"), IR_node.get_attr("min1_value"), IR_node.get_attr("min2_value")))

    def emit_Yolov11Post(self, IR_node):
        input_layers = []
        for num in range(len(IR_node.in_edges)):
            parent = self.IR_graph.get_parent(IR_node.name, [num])
            assert parent is not None
            input_layers.append(f"n.{parent.real_variable_name}")
        input_layers_str = ",".join(input_layers)
        max_boxes_per_class = IR_node.get_attr('max_boxes_per_class')
        iou_threshold = IR_node.get_attr('iou_threshold')
        max_dim = int(IR_node.get_attr('max_dim'))
        max_detections = IR_node.get_attr('max_detections')
        confidence_threshold = IR_node.get_attr('confidence_threshold')
        center_point_box = IR_node.get_attr('center_point_box')
        self.add_body(1, "n.{} = L.Yolov11Post({}, max_boxes_per_class={}, iou_threshold={}, max_dim={}, max_detections={}, confidence_threshold={}, center_point_box={} , ntop=1)".format(IR_node.variable_name, input_layers_str, max_boxes_per_class, iou_threshold, max_dim, max_detections, confidence_threshold, center_point_box))

    def emit_GlobalResponseNorm(self, IR_node):
        input_layers = [self.IR_graph.get_node(edge).real_variable_name for edge in IR_node.in_edges]
        input_layers = ','.join(['n.' + i for i in input_layers])
        axis = IR_node.get_attr('axis')
        self.add_body(1, "n.{:<15} = L.GlobalResponseNorm({}, global_response_norm_param=dict(axis={}), ntop=1)".format(IR_node.variable_name, input_layers, axis))

    def emit_SpacialDist(self, IR_node):
        input_layers = [self.IR_graph.get_node(edge).real_variable_name for edge in IR_node.in_edges]
        input_layers = ','.join(['n.' + i for i in input_layers])
        self.add_body(1, "n.{:<15} = L.SpacialDist({}, axes={}, ntop=1)".format(IR_node.variable_name, input_layers, IR_node.get_attr('axes')))

    def emit_MeanRatio(self, IR_node):
        input_layers = [self.IR_graph.get_node(edge).real_variable_name for edge in IR_node.in_edges]
        input_layers = ','.join(['n.' + i for i in input_layers])
        axis = IR_node.get_attr('axis')
        self.add_body(1, "n.{:<15} = L.MeanRatio({}, eps={}, mean_ratio_param=dict(axis={}), ntop=1)".format(IR_node.variable_name, input_layers, IR_node.get_attr('eps'), axis))

    def emit_tflite_DataInput(self, IR_node):
        shape = self._shapeToStr(IR_node.get_attr('_output_shapes')[0])
        # if IR_node.get_attr('data_format') != 'TNC':

        shape = clast2first(shape)
        output_scale = float(IR_node.get_attr('output_scale', 1))
        output_zero_point = IR_node.get_attr('output_zero_point', 0)
        self.add_body(1, "n.{:<15} = L.Input(shape=[dict(dim={})], tflite_qt=dict(output_scale={}, output_zeropoint={}), ntop=1)".format(IR_node.variable_name, shape, output_scale, output_zero_point))

    def emit_tflite_Conv(self, IR_node):
        # implement asymmetric paddings by applying symmetric padding then cropping
        # sym_pads = self._get_symmetric_padding(IR_node)
        # output_flag = False
        dilations = IR_node.get_attr('dilations')[1:-1]
        kernels = IR_node.get_attr('kernel_shape')[:-2]
        kernel_c = IR_node.get_attr('kernel_shape')[-2]
        pads = IR_node.get_attr('pads')[1:-1]
        spatial_pad = pads[:len(pads) // 2 - 1] + pads[len(pads) // 2 + 1:]
        spatial_kernels = kernels * 2
        spatial_dilations = dilations * 2
        strides = IR_node.get_attr('strides')[1:-1]
        is_tflite = IR_node.get_attr('is_tflite', False)
        # kpads = sym_pads #pads[1:len(pads)/2]
        np_spatial_kernels = np.array(spatial_kernels)
        np_spatial_dilations = np.array(spatial_dilations)
        np_new_kernels = np_spatial_kernels + (np_spatial_kernels - 1) * (np_spatial_dilations - 1)
        kpads = spatial_pad

        flag = False
        num_output = IR_node.get_attr('kernel_shape')[-1]
        if IR_node.type == "tflite_DepthwiseConv":
            num_group = kernel_c
            num_output = IR_node.get_attr("_output_shapes")[0].dim[-1].size

        else:
            num_group = IR_node.get_attr("group", 1)

        layouts = [None, 'NCHW', 'NHWC', 'NDHWC', 'NCDHW', 'NCH', 'NHC', 'NCW', 'NWC']
        if IR_node.get_attr('data_format') not in layouts:
            raise NotImplementedError("CaffeEmitter: Convolution data-format neither of NCHW,NHWC,NDHWC,NCH,NHC,NCW,NWC,none(default)")

        input_node_name = self.parent_variable_name(IR_node)
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        input_node_shape = shape_to_list(input_node.get_attr("_output_shapes")[0])
        output_scale = float(IR_node.get_attr('output_scale', 1))
        output_zero_point = IR_node.get_attr('output_zero_point', 0)
        input_scale = IR_node.get_attr('input_scale', [1] * len(IR_node.in_edges))
        input_zero_point = IR_node.get_attr('input_zero_point', [0] * len(IR_node.in_edges))
        activation = 'none'
        activation = IR_node.get_attr('fused_activation_type')
        if activation is None:
            activation = 'none'
        assert activation in ['relu', 'relu6', 'none']
        check_for_crop = True
        input_node_name = self.parent_variable_name(IR_node)
        conv_params = "kernel_size={}, stride = {}, num_output={}, pad={}, group={}, \
    bias_term={},dilation={}, is_pqn={}, tflite_qt=dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={}), fused_activation={}".format(
        kernels,
        strides,
        num_output,
        # [pad_f, pad_t, pad_l], #replace with [pad_f, pad_t, pad_l, pad_g, pad_b, pad_r], after analyzer changes
        spatial_pad,
        num_group,
        IR_node.get_attr('use_bias', False),
        dilations,
        is_tflite,
        output_scale,
        output_zero_point,
        input_scale,
        input_zero_point,
        fusedActivation[activation].value)
        self.add_body(1, "n.{:<15} = L.Tflite_Convolution(n.{}, convolution_param=dict({}), ntop=1)".format(IR_node.variable_name, input_node_name, conv_params))
        dim = len(IR_node.get_attr('strides')) - 2
        if self.weight_loaded:
            if IR_node.type == "tflite_DepthwiseConv":
                self.weights_dict[IR_node.name]['weights'] = np.swapaxes(self.weights_dict[IR_node.name]['weights'], -1, -2)
            self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], [dim + 1, dim] + list(range(0, dim)))
            self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)
        key_name = IR_node.variable_name if IR_node.variable_name in self.weights_dict else IR_node.name
        if 'weight_scales' not in self.weights_dict[key_name]:
            self.weights_dict[key_name]['weight_scales'] = np.array([1] * num_output)
        if 'weight_zero_points' not in self.weights_dict[key_name]:
            self.weights_dict[key_name]['weight_zero_points'] = np.array([0] * num_output)

    def emit_tflite_DepthwiseConv(self, IR_node):
        self.emit_tflite_Conv(IR_node)

    def emit_tflite_Add(self, IR_node):
        if len(IR_node.in_edges) > 2:
            raise NotImplementedError("Not supported more than two inputs to elementwise add")

        input_layers = [self.IR_graph.get_parent(IR_node.name, [num]).real_variable_name for num in range(len(IR_node.in_edges))]
        # Get the scales and offsets of inputs and output
        output_scale = float(IR_node.get_attr('output_scale', 1))
        output_zero_point = IR_node.get_attr('output_zero_point', 0)
        input_scale = IR_node.get_attr('input_scale', [1] * len(IR_node.in_edges))
        input_zero_point = IR_node.get_attr('input_zero_point', [0] * len(IR_node.in_edges))
        is_tflite = IR_node.get_attr('is_tflite', False)
        activation = 'none'
        activation = IR_node.get_attr('fused_activation_type')
        if activation is None:
            activation = 'none'
        assert activation in ['relu', 'relu6', 'none']

        elt_params = "operation=1, tflite_qt=dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={}), fused_activation={}".format(output_scale, output_zero_point, input_scale, input_zero_point, fusedActivation[activation].value)

        # Case: One of the input is a constant tensor
        if (IR_node.get_attr('const_tensor')):
            input_layers = [self.IR_graph.get_parent(IR_node.name, [num]).real_variable_name for num in range(len(IR_node.in_edges))]
            input_shape = clast2first([i.size for i in IR_node.get_attr('_output_shapes')[0].dim])
            dummy_layer_name = IR_node.variable_name + "_const_tensor"
            for key in self.weights_dict[IR_node.name]:
                self.weights_dict[IR_node.name][key] = np.broadcast_to(self.weights_dict[IR_node.name][key], input_shape)
            self.weights_dict[dummy_layer_name] = self.weights_dict.pop(IR_node.name)
            self.add_body(1, "n.{:<15} = L.DVTensor(dvtensor_param=dict(tensortype={}, value= {},shape=dict(dim={})), ntop=1)".format(dummy_layer_name, 2, 0, '[1]' if str(input_shape) == '[]' else str(input_shape)))
            self.add_body(1, "n.{:<15} = L.Tflite_Eltwise(n.{},n.{}, eltwise_param=dict({}), ntop=1)".format(IR_node.variable_name, input_layers[0], dummy_layer_name, elt_params))
            return
        if len(input_layers) < 2:
            logger.error("number of inputs is less than 2 for Eltwise node %s", IR_node.name)
            raise ValueError()
        in_edges = [self.IR_graph.get_node(IR_node.in_edges[0]).real_name, self.IR_graph.get_node(IR_node.in_edges[1]).real_name]
        in_edges_shapes = [self.IR_graph.get_node(IR_node.in_edges[0]).get_attr('_output_shapes'), self.IR_graph.get_node(IR_node.in_edges[1]).get_attr('_output_shapes')]
        in_edges_shapes = [[j.size for j in i[0].dim] for i in in_edges_shapes]
        in_edges_shapes = [clast2first(i) for i in in_edges_shapes]
        if in_edges[0] == in_edges[1]:
            pass
        else:
            if len(in_edges_shapes[0]) < len(in_edges_shapes[1]):
                new_shape = [1] * (len(in_edges_shapes[1]) - len(in_edges_shapes[0])) + in_edges_shapes[0]
                dim_str = "'dim': {}".format(new_shape)
                dim_str = " reshape_param={'shape': { " + dim_str + '} }'
                reshape1 = IR_node.real_name + "_reshape1"
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape1, self.parent_variable_name(IR_node, [0]), dim_str))
                input_layers[0] = reshape1
            if len(in_edges_shapes[0]) > len(in_edges_shapes[1]):
                new_shape = [1] * (len(in_edges_shapes[0]) - len(in_edges_shapes[1])) + in_edges_shapes[1]
                dim_str = "'dim': {}".format(new_shape)
                dim_str = " reshape_param={'shape': { " + dim_str + '} }'
                reshape1 = IR_node.real_name + "_reshape1"
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape1, self.parent_variable_name(IR_node, [1]), dim_str))
                input_layers[1] = reshape1
            self.add_body(1, "n.{:<15} = L.Tflite_Eltwise(bottom={}, eltwise_param=dict({}), ntop=1)".format(IR_node.variable_name, [input_layers[0], input_layers[1]], elt_params))

    def emit_tflite_Pool(self, IR_node):
        pooling_type = IR_node.get_attr('pooling_type')
        output_scale = float(IR_node.get_attr('output_scale', 1))
        output_zero_point = IR_node.get_attr('output_zero_point', 0)
        input_scale = IR_node.get_attr('input_scale', [1] * len(IR_node.in_edges))
        input_zero_point = IR_node.get_attr('input_zero_point', [0] * len(IR_node.in_edges))
        is_tflite = IR_node.get_attr('is_tflite', False)

        if pooling_type == 'MAX':
            pooling_type = P.Pooling.MAX
        elif pooling_type == 'AVG':
            pooling_type = P.Pooling.AVE
        elif pooling_type == 'STOCHASTIC':
            pooling_type = P.Pooling.STOCHASTIC
        elif pooling_type == 'SUM':
            pooling_type = 3
        else:
            raise ValueError()

        parent_node = self.IR_graph.get_parent(IR_node.name, [0])
        input_shape = parent_node.get_attr("_output_shapes")[0]
        input_shape = shape_to_list(input_shape)

        strides = IR_node.get_attr('strides')[1:-1]
        kernels = IR_node.get_attr('kernel_shape')[1:-1]

        if len(strides) != len(input_shape) - 2:
            strides = strides * (len(input_shape) - 2 - len(strides))

        if len(kernels) != len(input_shape) - 2:
            kernels = kernels * (len(input_shape) - 2 - len(kernels))

        parent_node = self.IR_graph.get_parent(IR_node.name, [0])

        ceil_mode = IR_node.get_attr('ceil_mode', 0)
        # xprint("ceil_mode: ",ceil_mode)

        # Assume n, h, w, c format
        # input width shape being same as kernel shape implies global pooling
        # set strides to 1

        kernels_half = list(np.array(kernels) // 2)

        if pooling_type == 0:
            padding_value = -sys.float_info.max
        elif pooling_type == 1:
            padding_value = 0
        elif pooling_type == 2:
            padding_value = sys.float_info.max

        if input_shape[1:-1] == kernels and np.all(np.array(IR_node.get_attr('pads')) == 0):
            IR_node.layer.attr['global_pooling'].b = True

        if IR_node.layer.attr['global_pooling'].b:
            pool_param = "pool={}, global_pooling=True, tflite_qt=dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={})".format(pooling_type, output_scale, output_zero_point, input_scale, input_zero_point)
            if len(input_shape) == 3 and input_shape[0] > 1:
                ishape = clast2first(input_shape)
                dim_str = "'dim': {}".format([1, ishape[0] * ishape[1], 1, ishape[2]])
                dim_str = " reshape_param={'shape': { " + dim_str + '} }'
                reshape_layer_variable_name = IR_node.real_variable_name + "_reshape1"

                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape_layer_variable_name, self.parent_variable_name(IR_node), dim_str))
                self.add_body(1, "n.{:<15} = L.Tflite_Pooling(n.{}, pooling_param=dict({}), ntop=1)".format(IR_node.variable_name, reshape_layer_variable_name, pool_param))
                output_shape = shape_to_list(IR_node.get_attr('_output_shapes')[0])
                oshape = clast2first(output_shape)
                dim_str = "'dim': {}".format(oshape)
                dim_str = " reshape_param={'shape': { " + dim_str + '} }'
                reshape_layer_variable_name = IR_node.real_variable_name + "_reshape2"

                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, {})".format(reshape_layer_variable_name, IR_node.variable_name, dim_str))
                IR_node.real_name = reshape_layer_variable_name
            else:
                self.add_body(1, "n.{:<15} = L.Tflite_Pooling(n.{}, pooling_param=dict({}), ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), pool_param))
        else:
            pads = IR_node.get_attr('pads')[1:-1]
            pads = pads[:len(pads) // 2 - 1] + pads[len(pads) // 2 + 1:]
            spatial_kernels = kernels * 2
            pool_param = "pool={}, kernel_size={}, pad={}, stride={}, tflite_qt=dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={})".format(pooling_type, kernels, pads, strides, output_scale, output_zero_point, input_scale, input_zero_point)
            self.add_body(1, "n.{:<15} = L.Tflite_Pooling(n.{}, pooling_param=dict({}), ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), pool_param))

    def emit_tflite_FullyConnected(self, IR_node):
        parent_node = self.IR_graph.get_parent(IR_node.name, [0])
        # TODO FIXME for volumetric data
        input_shape = parent_node.get_attr("_output_shapes")[0]
        input_shape = clast2first(shape_to_list(input_shape))
        output_scale = float(IR_node.get_attr('output_scale', 1))
        output_zero_point = IR_node.get_attr('output_zero_point', 0)
        input_scale = IR_node.get_attr('input_scale', [1] * len(IR_node.in_edges))
        input_zero_point = IR_node.get_attr('input_zero_point', [0] * len(IR_node.in_edges))
        is_tflite = IR_node.get_attr('is_tflite', False)
        prod = 1
        flatten = IR_node.get_attr('flatten', True)

        if not flatten and len(input_shape) > 2:
            shape = [np.prod(input_shape[:-1]), input_shape[-1]]
            dim_str = "'dim': {}".format(shape)
            dim_str = "{'shape': { " + dim_str + '} }'
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, reshape_param={}) ".format(IR_node.variable_name + "_reshape1", self.parent_variable_name(IR_node), dim_str))
            inrp_param = "num_output={}, bias_term={}, is_pqn={}, tflite_qt=dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={})".format(IR_node.layer.attr["units"].i, IR_node.get_attr('use_bias', False), is_tflite, output_scale, output_zero_point, input_scale, input_zero_point)
            self.add_body(1, "n.{:<15} = L.Tflite_InnerProduct(n.{}, inner_product_param=dict({}), ntop=1)".format(IR_node.variable_name, IR_node.variable_name + "_reshape1", inrp_param))

            output_shape = IR_node.get_attr("_output_shapes")[0]
            output_shape = clast2first(shape_to_list(output_shape))
            dim_str = "'dim': {}".format(output_shape)
            dim_str = "{'shape': { " + dim_str + '} }'
            self.add_body(1, "n.{:<15} = L.Reshape(n.{}, reshape_param={}) ".format(IR_node.variable_name + "_reshape2", IR_node.variable_name, dim_str))
            IR_node.real_name = IR_node.variable_name + "_reshape2"
        else:
            inrp_param = "num_output={}, bias_term={}, is_pqn={}, tflite_qt=dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={})".format(IR_node.layer.attr["units"].i, IR_node.get_attr('use_bias', False), is_tflite, output_scale, output_zero_point, input_scale, input_zero_point)
            self.add_body(1, "n.{:<15} = L.Tflite_InnerProduct(n.{}, inner_product_param=dict({}), ntop=1)".format(IR_node.variable_name, self.parent_variable_name(IR_node), inrp_param))
            if IR_node.get_attr('reshape', False):
                output_shape = IR_node.get_attr('reshape')
                dim_str = "'dim': {}".format(output_shape)
                dim_str = "{'shape': { " + dim_str + '} }'
                self.add_body(1, "n.{:<15} = L.Reshape(n.{}, reshape_param={}) ".format(IR_node.variable_name + "_reshape2", IR_node.variable_name, dim_str))
                IR_node.real_name = IR_node.variable_name + "_reshape2"
        if self.weight_loaded:
            # self.check_if_need_transpose(IR_node)
            self.weights_dict[IR_node.name]['weights'] = np.transpose(self.weights_dict[IR_node.name]['weights'], (1, 0))
            self.weights_dict[IR_node.variable_name] = self.weights_dict.pop(IR_node.name)

    def emit_tflite_Pad(self, IR_node):
        pads = IR_node.get_attr('pads')
        output_zero_point = IR_node.get_attr('output_zero_point', 0)
        if pads[3] > 0 or pads[7] > 0:
            logger.warn("channel padding is not yet supported by DVNet")
            # a -> b | a -> x1[Pad_spatial] -> x[DummyPermute] -> y[Pad] -> z[Permute] -> b
            self.add_body(
          1,
          "n.{:<15} = L.Padding(n.{}, padding_param=dict(pad_l={},pad_r={},pad_t={}, pad_b={},pad_method={},pad_value={}), tflite_qt=dict(output_zeropoint={}), ntop=1)".format(
              IR_node.variable_name,
              self.parent_variable_name(IR_node),
              pads[2],
              pads[6],
              pads[1],
              pads[5],
              0,  #pad method is CONSTANT enum
              output_zero_point,
              output_zero_point))  #pad value
            logger.debug("output shape is " + str(IR_node.get_attr('_output_shapes')))
            perm_order = [0] * len(shape_to_list(IR_node.get_attr('_output_shapes')[0]))
            if len(perm_order) == 4:  # NCHW
                perm_order = [0, 3, 2, 1]
            elif len(perm_order) == 3:  # NCW
                perm_order = [0, 2, 1]
            perm_order_str = '{' + "'order': {}".format(perm_order) + '}'
            # Channel -- > Width permute
            perm1 = IR_node.variable_name + "_perm1"
            self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(perm1, IR_node.variable_name, perm_order_str))
            # Pad on width
            pad_channel = IR_node.variable_name + "_ch_pad"

        # TODO; get padding type and value; raise exception if stack doesn't support
        if pads[3] > 0 or pads[7] > 0:
            logger.warn("channel padding is not yet supported by DVNet")
            # a -> b | a -> x1[Pad_spatial] -> x[DummyPermute] -> y[Pad] -> z[Permute] -> b
            self.add_body(
          1,
          "n.{:<15} = L.Padding(n.{}, padding_param=dict(pad_l={},pad_r={},pad_t={}, pad_b={},pad_method={},pad_value={}),tflite_qt=dict(output_zeropoint={}), ntop=1)".format(
              IR_node.variable_name,
              self.parent_variable_name(IR_node),
              pads[2],
              pads[6],
              pads[1],
              pads[5],
              0,  #pad method is CONSTANT enum
              output_zero_point,
              output_zero_point))  #pad value
            logger.debug("output shape is " + str(IR_node.get_attr('_output_shapes')))
            perm_order = [0] * len(shape_to_list(IR_node.get_attr('_output_shapes')[0]))
            if len(perm_order) == 4:  # NCHW
                perm_order = [0, 3, 2, 1]
            elif len(perm_order) == 3:  # NCW
                perm_order = [0, 2, 1]
            perm_order_str = '{' + "'order': {}".format(perm_order) + '}'
            # Channel -- > Width permute
            perm1 = IR_node.variable_name + "_perm1"
            self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(perm1, IR_node.variable_name, perm_order_str))
            # Pad on width
            pad_channel = IR_node.variable_name + "_ch_pad"
            self.add_body(
          1,
          "n.{:<15} = L.Padding(n.{}, padding_param=dict(pad_l={},pad_r={},pad_t={}, pad_b={},pad_method={},pad_value={}),tflite_qt=dict(output_zeropoint={}), ntop=1)".format(
              pad_channel,
              perm1,
              pads[3],
              pads[7],
              0,
              0,
              0,  #pad method is CONSTANT enum
              output_zero_point,
              output_zero_point))  #pad value
            # Permute back
            perm2 = IR_node.variable_name + "_perm2"
            self.add_body(1, "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(perm2, pad_channel, perm_order_str))
            IR_node.real_name = perm2

        else:
            self.add_body(
          1,
          "n.{:<15} = L.Padding(n.{}, padding_param=dict(pad_l={},pad_r={},pad_t={}, pad_b={},pad_method={},pad_value={}),tflite_qt=dict(output_zeropoint={}), ntop=1)".format(
              IR_node.variable_name,
              self.parent_variable_name(IR_node),
              pads[2],
              pads[6],
              pads[1],
              pads[5],
              0,  #pad method is CONSTANT enum
              output_zero_point,
              output_zero_point))  #pad value

    def emit_tflite_PoolReduce(self, IR_node):
        pooling_type = IR_node.get_attr("pooling_type")
        output_scale = float(IR_node.get_attr("output_scale", 1))
        output_zero_point = IR_node.get_attr("output_zero_point", 0)
        input_scale = IR_node.get_attr("input_scale", [1] * len(IR_node.in_edges))
        input_zero_point = IR_node.get_attr(
            "input_zero_point", [0] * len(IR_node.in_edges)
        )
        is_tflite = IR_node.get_attr("is_tflite", False)

        if pooling_type == "MAX":
            pooling_type = P.Pooling.MAX
        elif pooling_type == "AVG":
            pooling_type = P.Pooling.AVE
        elif pooling_type == "STOCHASTIC":
            pooling_type = P.Pooling.STOCHASTIC
        elif pooling_type == "SUM":
            pooling_type = P.Pooling.SUMPOOL
        else:
            raise ValueError()

        reshape_layer = IR_node.variable_name + "_reshape"

        pool_param = "pool={}, global_pooling=True, tflite_qt=dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={})".format(
            pooling_type, output_scale, output_zero_point, input_scale, input_zero_point
        )

        self.add_body(
            1,
            "n.{:<15} = L.Tflite_Pooling(n.{}, pooling_param=dict({}), ntop=1)".format(
                IR_node.variable_name, self.parent_variable_name(IR_node), pool_param
            ),
        )

        shape = IR_node.get_attr("_output_shapes")[0]
        shape = shape_to_list(shape)
        shape[0] = -1

        if not IR_node.layer.attr["keepdims"].b:
            if len(shape) > 2:
                channels = shape.pop(len(shape) - 1)
                shape.insert(1, channels)
            dim_str = "'dim': {}".format(shape)
            dim_str = " reshape_param={'shape': { " + dim_str + "} }"
            self.add_body(
                1,
                "n.{:<15} = L.Reshape(n.{}, {})".format(
                    reshape_layer, IR_node.variable_name, dim_str
                ),
            )
            IR_node.real_name = reshape_layer

    def emit_tflite_Mul(self, IR_node):
        is_tflite = IR_node.get_attr("is_tflite", False)
        elt_params = "operation=0"

        output_scale = float(IR_node.get_attr("output_scale", 1))
        output_zero_point = IR_node.get_attr("output_zero_point", 0)
        input_scale = IR_node.get_attr("input_scale", [1] * len(IR_node.in_edges))
        input_zero_point = IR_node.get_attr(
            "input_zero_point", [0] * len(IR_node.in_edges)
        )
        const_input = IR_node.get_attr("const_input", 0)

        activation = "none"
        activation = IR_node.get_attr("fused_activation_type")
        if activation is None:
            activation = "none"
        assert activation in ["relu", "relu6", "none"]

        elt_params = "operation=0, const_input={}, tflite_qt=dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={}), fused_activation = {}".format(
            const_input,
            output_scale,
            output_zero_point,
            input_scale,
            input_zero_point,
            fusedActivation[activation].value,
        )

        if len(IR_node.in_edges) == 2:
            input_layers = [
                self.IR_graph.get_node(edge).real_variable_name
                for edge in IR_node.in_edges
            ]

            in_edges = IR_node.in_edges
            in_edges_shapes = [
                self.IR_graph.get_node(IR_node.in_edges[0]).get_attr("_output_shapes"),
                self.IR_graph.get_node(IR_node.in_edges[1]).get_attr("_output_shapes"),
            ]
            in_edges_shapes = [[j.size for j in i[0].dim] for i in in_edges_shapes]
            in_edges_shapes = [clast2first(i) for i in in_edges_shapes]

            assert len(in_edges_shapes) == 2
            if len(in_edges_shapes[0]) < len(in_edges_shapes[1]):
                new_shape = [1] * (
                    len(in_edges_shapes[1]) - len(in_edges_shapes[0])
                ) + in_edges_shapes[0]
                dim_str = "'dim': {}".format(new_shape)
                dim_str = " reshape_param={'shape': { " + dim_str + "} }"
                reshape1 = IR_node.real_name + "_reshape1"
                self.add_body(
                    1,
                    "n.{:<15} = L.Reshape(n.{}, {})".format(
                        reshape1, self.parent_variable_name(IR_node, [0]), dim_str
                    ),
                )
                input_layers[0] = reshape1
            if len(in_edges_shapes[0]) > len(in_edges_shapes[1]):
                new_shape = [1] * (
                    len(in_edges_shapes[0]) - len(in_edges_shapes[1])
                ) + in_edges_shapes[1]
                dim_str = "'dim': {}".format(new_shape)
                dim_str = " reshape_param={'shape': { " + dim_str + "} }"
                reshape1 = IR_node.real_name + "_reshape1"
                self.add_body(
                    1,
                    "n.{:<15} = L.Reshape(n.{}, {})".format(
                        reshape1, self.parent_variable_name(IR_node, [1]), dim_str
                    ),
                )
                input_layers[1] = reshape1
            if is_tflite:
                self.add_body(
                    1,
                    "n.{:<15} = L.Tflite_Eltwise(n.{},n.{}, eltwise_param=dict({}), ntop=1)".format(
                        IR_node.variable_name,
                        input_layers[0],
                        input_layers[1],
                        elt_params,
                    ),
                )
            else:
                input_layers = ",".join(["n." + i for i in input_layers])
                self.add_body(
                    1,
                    "n.{:<15} = L.Eltwise({}, operation=0, ntop=1)".format(
                        IR_node.variable_name, input_layers
                    ),
                )

        elif len(IR_node.in_edges) == 1:
            if IR_node.get_attr("const_tensor"):
                input_layers = [
                    self.IR_graph.get_parent(IR_node.name, [num]).real_variable_name
                    for num in range(len(IR_node.in_edges))
                ]
                input_shape = [
                    i.size for i in IR_node.get_attr("_output_shapes")[0].dim
                ]
                dummy_layer_name = IR_node.variable_name + "_const_tensor"
                self.weights_dict[dummy_layer_name] = self.weights_dict.pop(
                    IR_node.name
                )
                input_shape = [
                    i
                    for i in np.array(
                        self.weights_dict[dummy_layer_name]["weights"]
                    ).shape
                ]
                self.add_body(
                    1,
                    "n.{:<15} = L.DVTensor(dvtensor_param=dict(tensortype={}, value= {},shape=dict(dim={})), ntop=1)".format(
                        dummy_layer_name,
                        2,
                        0,
                        "[1]" if str(input_shape) == "[]" else str(input_shape),
                    ),
                )
                if is_tflite:
                    self.add_body(
                        1,
                        "n.{:<15} = L.Tflite_Eltwise(n.{},n.{}, eltwise_param=dict({}), ntop=1)".format(
                            IR_node.variable_name,
                            input_layers[0],
                            dummy_layer_name,
                            elt_params,
                        ),
                    )
                else:
                    self.add_body(
                        1,
                        "n.{:<15} = L.Eltwise(n.{},n.{}, operation=0, ntop=1)".format(
                            IR_node.variable_name, input_layers[0], dummy_layer_name
                        ),
                    )
            else:
                self.emit_Scale(IR_node)
        else:
            assert False

    def emit_tflite_Softmax(self, IR_node):
        axis = IR_node.layer.attr["axis"].i
        input_node = self.IR_graph.get_parent(IR_node.name, [0])
        input_node_shape = shape_to_list(input_node.get_attr("_output_shapes")[0])

        if len(input_node_shape) == 4:
            axis = NHWC_NCHW[axis]
        elif len(input_node_shape) == 3:
            if axis == 1:
                axis = 2
            elif axis == 2:
                axis = 1
        # Get the scales and offsets of inputs and output
        output_scale = float(IR_node.get_attr("output_scale", 1))
        output_zero_point = IR_node.get_attr("output_zero_point", 0)
        input_scale = IR_node.get_attr("input_scale", [1] * len(IR_node.in_edges))
        input_zero_point = IR_node.get_attr(
            "input_zero_point", [0] * len(IR_node.in_edges)
        )
        is_tflite = IR_node.get_attr("is_tflite", False)

        sftmax_param = "axis={},tflite_qt=dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={})".format(
            axis, output_scale, output_zero_point, input_scale, input_zero_point
        )

        if is_tflite:
            self.add_body(
                1,
                "n.{:<15} = L.Tflite_Softmax(n.{},softmax_param=dict({}), ntop=1)".format(
                    IR_node.variable_name,
                    self.parent_variable_name(IR_node),
                    sftmax_param,
                ),
            )
        else:
            self.add_body(
                1,
                "n.{:<15} = L.Softmax(n.{},{}, ntop=1)".format(
                    IR_node.variable_name,
                    self.parent_variable_name(IR_node),
                    sftmax_param,
                ),
            )

    def emit_tflite_Sigmoid(self, IR_node):
        is_tflite = IR_node.get_attr("is_tflite", False)

        if is_tflite:
            output_scale = float(IR_node.get_attr("output_scale", 1))
            output_zero_point = IR_node.get_attr("output_zero_point", 0)
            input_scale = IR_node.get_attr("input_scale", [1] * len(IR_node.in_edges))
            input_zero_point = IR_node.get_attr(
                "input_zero_point", [0] * len(IR_node.in_edges)
            )
            tflite_param = "tflite_qt=dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={})".format(
                output_scale, output_zero_point, input_scale, input_zero_point
            )

            self.add_body(
                1,
                "n.{:<15} = L.Tflite_Sigmoid(n.{}, sigmoid_param=dict({}), ntop=1)".format(
                    IR_node.variable_name,
                    self.parent_variable_name(IR_node),
                    tflite_param,
                ),
            )
            return
        self.add_body(
            1,
            "n.{:<15} = L.Sigmoid(n.{}, ntop=1)".format(
                IR_node.variable_name, self.parent_variable_name(IR_node)
            ),
        )

    def emit_tflite_LayerNorm_H(self, IR_node):
        output_shape = IR_node.get_attr("_output_shapes")[0]
        output_shape = shape_to_list(output_shape)
        parent_node = self.IR_graph.get_parent(IR_node.name, [0])
        tflite_axis = IR_node.get_attr("mean_1_axis")
        caffe_axis = NHWC_NCHW[tflite_axis]

        p_name = self.parent_variable_name(IR_node)

        if caffe_axis == 1:
            p_name = IR_node.variable_name + "_b_perm1"
            self.add_body(
                1,
                "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(
                    p_name,
                    self.parent_variable_name(IR_node),
                    "{" + "'order': {}".format([0, 2, 1, 3]) + "}",
                ),
            )

        self.add_body(
            1,
            "n.{:<15} = L.Tflite_LayerNorm_h(n.{},tflite_qt=dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={}), axis = {}, mean_1_output_scales = {}, mean_1_zero_points = {}, squared_difference_output_zero_points = {}, squared_difference_output_scales = {}, mean_2_axis = {}, mean_2_output_scales = {}, mean_2_output_zero_points = {}, add_1_output_scales = {}, add_1_output_zero_points = {}, add_1_constant ={},add_1_constant_scale ={},add_1_constant_zero_points ={},  rsqrt_output_scales = {},  rsqrt_output_zero_points = {}, mul_1_output_scales = {}, mul_1_output_zero_points = {}, mul_1_constant_scale = {}, mul_1_constant_zero_points = {}, mul_2_output_scales = {}, mul_2_output_zero_points = {}, mul_3_output_scales = {}, mul_3_output_zero_points = {}, sub_1_output_scales = {}, sub_1_output_zero_points = {}, sub_1_constant_scale = {}, sub_1_constant_zero_points = {},add_2_output_scales = {}, add_2_output_zero_points = {}, ntop=1)".format(
                IR_node.variable_name,
                p_name,
                IR_node.get_attr("output_scales"),
                IR_node.get_attr("output_zero_points"),
                IR_node.get_attr("input_scales"),
                IR_node.get_attr("input_zero_points"),
                2,
                IR_node.get_attr("mean_1_output_scales"),
                IR_node.get_attr("mean_1_output_zero_points"),
                IR_node.get_attr("squared_difference_output_zero_points"),
                IR_node.get_attr("squared_difference_output_scales"),
                2,
                IR_node.get_attr("mean_2_output_scales"),
                IR_node.get_attr("mean_2_output_zero_points"),
                IR_node.get_attr("add_1_output_scales"),
                IR_node.get_attr("add_1_output_zero_points"),
                IR_node.get_attr("add_1_constant"),
                IR_node.get_attr("add_1_constant_scale"),
                IR_node.get_attr("add_1_constant_zero_points"),
                IR_node.get_attr("rsqrt_output_scales"),
                IR_node.get_attr("rsqrt_output_zero_points"),
                IR_node.get_attr("mul_1_output_scales"),
                IR_node.get_attr("mul_1_output_zero_points"),
                IR_node.get_attr("mul_1_constant_scale"),
                IR_node.get_attr("mul_1_constant_zero_points"),
                IR_node.get_attr("mul_2_output_scales"),
                IR_node.get_attr("mul_2_output_zero_points"),
                IR_node.get_attr("mul_3_output_scales"),
                IR_node.get_attr("mul_3_output_zero_points"),
                IR_node.get_attr("sub_1_output_scales"),
                IR_node.get_attr("sub_1_output_zero_points"),
                IR_node.get_attr("sub_1_constant_scale"),
                IR_node.get_attr("sub_1_constant_zero_points"),
                IR_node.get_attr("add_2_output_scales"),
                IR_node.get_attr("add_2_output_zero_points"),
            ),
        )
        if caffe_axis == 1:
            p_name = IR_node.variable_name + "_b_perm2"
            self.add_body(
                1,
                "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(
                    p_name,
                    IR_node.variable_name,
                    "{" + "'order': {}".format([0, 2, 1, 3]) + "}",
                ),
            )
            IR_node.real_name = p_name

    def emit_tflite_BatchDot(self, IR_node):
        if IR_node.get_attr("const_tensor"):
            dummy_layer_name = IR_node.variable_name + "_const_tensor"
            self.weights_dict[dummy_layer_name] = self.weights_dict.pop(IR_node.name)
            tensor_shape = [
                i
                for i in np.array(self.weights_dict[dummy_layer_name]["weights"]).shape
            ]
            self.add_body(
                1,
                "n.{:<15} = L.DVTensor(dvtensor_param=dict(tensortype={}, value= {},shape=dict(dim={})), ntop=1)".format(
                    dummy_layer_name,
                    2,
                    0,
                    "[1]" if str(tensor_shape) == "[]" else str(tensor_shape),
                ),
            )
            input_layers = [
                self.IR_graph.get_node(IR_node.in_edges[0]).real_variable_name
            ]
            input_layers.insert(IR_node.get_attr("dvtensor_pos"), dummy_layer_name)
            input_layers = ", ".join(["n." + i for i in input_layers])
            self.add_body(
                1,
                "n.{:<15} = L.BatchDot({},transpose_a={}, transpose_b={},ntop=1)".format(
                    IR_node.variable_name,
                    input_layers,
                    IR_node.get_attr("transpose_a"),
                    IR_node.get_attr("transpose_b"),
                ),
            )
            return

        output_scale = float(IR_node.get_attr("output_scale", 1))
        output_zero_point = IR_node.get_attr("output_zero_point", 0)
        input_scale = IR_node.get_attr("input_scale", [1] * len(IR_node.in_edges))
        input_zero_point = IR_node.get_attr(
            "input_zero_point", [0] * len(IR_node.in_edges)
        )
        is_tflite = IR_node.get_attr("is_tflite", False)

        input_layers = ", ".join(
            ("n." + self.IR_graph.get_node(edge).real_variable_name)
            for edge in IR_node.in_edges
        )
        input1_node = self.IR_graph.get_parent(IR_node.name, [0])
        input2_node = self.IR_graph.get_parent(IR_node.name, [1])
        input1_shape = clast2first(
            shape_to_list(input1_node.get_attr("_output_shapes")[0])
        )
        input2_shape = clast2first(
            shape_to_list(input2_node.get_attr("_output_shapes")[0])
        )
        output_shape = shape_to_list(IR_node.get_attr("_output_shapes")[0])

        if is_tflite:
            perm1, perm2 = (
                IR_node.real_name + "_b_perm1",
                IR_node.real_name + "_b_perm2",
            )
            perm3 = IR_node.real_name + "_perm1"
            rshp1, rshp2 = perm1, perm2
            rshp3 = IR_node.variable_name
            transpose_a, transpose_b = True, False
            if not IR_node.get_attr("transpose_a"):
                self.add_body(
                    1,
                    "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(
                        perm1,
                        "{}".format(
                            self.IR_graph.get_node(
                                IR_node.in_edges[0]
                            ).real_variable_name
                        ),
                        "{" + "'order': {}".format([0, 2, 1, 3]) + "}",
                    ),
                )
                input1_shape = [
                    input1_shape[0],
                    input1_shape[2],
                    input1_shape[1],
                    input1_shape[3],
                ]
            else:
                self.add_body(
                    1,
                    "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(
                        perm1,
                        "{}".format(
                            self.IR_graph.get_node(
                                IR_node.in_edges[0]
                            ).real_variable_name
                        ),
                        "{" + "'order': {}".format([0, 2, 3, 1]) + "}",
                    ),
                )
                input1_shape = [
                    input1_shape[0],
                    input1_shape[2],
                    input1_shape[3],
                    input1_shape[1],
                ]
            if IR_node.get_attr("transpose_b"):
                self.add_body(
                    1,
                    "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(
                        perm2,
                        "{}".format(
                            self.IR_graph.get_node(
                                IR_node.in_edges[1]
                            ).real_variable_name
                        ),
                        "{" + "'order': {}".format([0, 2, 1, 3]) + "}",
                    ),
                )
                input2_shape = [
                    input2_shape[0],
                    input2_shape[2],
                    input2_shape[1],
                    input2_shape[3],
                ]
            else:
                self.add_body(
                    1,
                    "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(
                        perm2,
                        "{}".format(
                            self.IR_graph.get_node(
                                IR_node.in_edges[1]
                            ).real_variable_name
                        ),
                        "{" + "'order': {}".format([0, 2, 3, 1]) + "}",
                    ),
                )
                input2_shape = [
                    input2_shape[0],
                    input2_shape[2],
                    input2_shape[3],
                    input2_shape[1],
                ]

            if input1_shape[0] > 1:
                # Supports only if N > 1. Reshape accordingly
                new_shape = [
                    1,
                    input1_shape[0] * input1_shape[1],
                    input1_shape[2],
                    input1_shape[3],
                ]
                dim_str = "'dim': {}".format(new_shape)
                dim_str = " reshape_param={'shape': { " + dim_str + "} }"
                rshp1 = perm1 + "_reshape1"
                self.add_body(
                    1, "n.{:<15} = L.Reshape(n.{}, {})".format(rshp1, perm1, dim_str)
                )

                new_shape = [
                    1,
                    input2_shape[0] * input2_shape[1],
                    input2_shape[2],
                    input2_shape[3],
                ]
                dim_str = "'dim': {}".format(new_shape)
                dim_str = " reshape_param={'shape': { " + dim_str + "} }"
                rshp2 = perm2 + "_reshape1"
                self.add_body(
                    1, "n.{:<15} = L.Reshape(n.{}, {})".format(rshp2, perm2, dim_str)
                )

            bdot_param = "transpose_a={}, transpose_b={}, is_pqn={}, tflite_qt=dict(output_scale={}, output_zeropoint={}, input_scale={}, input_zeropoint={})".format(
                transpose_a,
                transpose_b,
                is_tflite,
                output_scale,
                output_zero_point,
                input_scale,
                input_zero_point,
            )
            self.add_body(
                1,
                "n.{:<15} = L.Tflite_BatchDot({},batch_dot_param=dict({}), ntop=1)".format(
                    IR_node.variable_name, "n.{}, n.{}".format(rshp1, rshp2), bdot_param
                ),
            )
            if output_shape[0] > 1:
                # Support only N > 1
                new_shape = output_shape
                rshp3 = perm3 + "_reshape1"
                new_shape = output_shape
                dim_str = "'dim': {}".format(new_shape)
                dim_str = " reshape_param={'shape': { " + dim_str + "} }"
                self.add_body(
                    1,
                    "n.{:<15} = L.Reshape(n.{}, {})".format(
                        rshp3, IR_node.variable_name, dim_str
                    ),
                )

            self.add_body(
                1,
                "n.{:<15} = L.Permute(n.{}, permute_param={}, ntop=1)".format(
                    perm3, rshp3, "{" + "'order': {}".format([0, 3, 1, 2]) + "}"
                ),
            )
            IR_node.real_name = perm3
            return
        else:
            self.add_body(
                1,
                "n.{:<15} = L.BatchDot({},transpose_a={}, transpose_b={},ntop=1)".format(
                    IR_node.variable_name,
                    input_layers,
                    IR_node.get_attr("transpose_a"),
                    IR_node.get_attr("transpose_b"),
                ),
            )
    
    def get_input_layers_ignoring_qdq(self, IR_node):
        input_layers = [self.IR_graph.get_parent(IR_node.name, [num]).real_variable_name for num in range(len(IR_node.in_edges))]
        # for i, input_layer in enumerate(input_layers):
        #     # while self.IR_graph.get_node(input_layer).get_attr('ignore_node'):
        #     while self.IR_graph.get_node(input_layer).type in ["onnxpqn_QuantizeLinear", "onnxpqn_DequantizeLinear"]:
        #         input_layer = self.IR_graph.get_parent(input_layer, [0]).real_variable_name
        #     input_layers[i] = input_layer
        return input_layers
