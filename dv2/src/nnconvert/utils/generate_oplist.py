'''
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

import subprocess
import tensorflow as tf
import sys as _sys
import os
import json
import argparse
import collections
_sys.path.append(os.path.join(os.getcwd(), os.path.dirname(__file__), '..'))
from dv_logging import get_logger
logger = get_logger(__name__, 'DVOPUTIL')


class helper():

  def __init__(self: type, args):
    self.fp = str(args.input_network)
    self.output_file = str(args.output_file)

  def onnx_helper(self: type) -> None:
    import onnx
    import onnxruntime as ort
    from google.protobuf.json_format import MessageToDict
    logger.debug("loading onnx model from '%s'", self.fp)
    meta = {}
    onnx_model = onnx.load_model(self.fp)
    graph = onnx_model.graph.node
    meta['onnx_version'] = onnx.__version__
    meta['onnxruntime_version'] = ort.__version__
    if onnx_model.HasField('producer_name'):
      meta['producer'] = onnx_model.producer_name
    if onnx_model.HasField('producer_version'):
      meta['producer_version'] = onnx_model.producer_version
    meta['opset'] = (onnx_model.opset_import[0].version)
    for layer in graph:
      node = MessageToDict(layer, preserving_proto_field_name=True)
      if layer.op_type not in meta.keys():
        meta[layer.op_type] = [node['attribute']] if 'attribute' in node else []
      else:
        if len(layer.attribute):
          meta[layer.op_type].append(node['attribute'])
    with open(self.output_file, 'w', encoding='utf8') as f:
      json.dump(meta, f, indent=4)
    logger.info("saved meta info at: {}".format(os.path.abspath(self.output_file)))

  def mxnet_helper(self: type) -> None:
    import mxnet
    logger.debug("loading mxnet model from '%s'", self.fp)
    meta = dict()
    meta['mxnet_version'] = mxnet.__version__
    with open(self.fp) as f:
      mxnet_model = json.load(f)
      if 'attrs' in mxnet_model and 'mxnet_version' in mxnet_model["attrs"]:
        meta['mxnet_model_version'] = mxnet_model["attrs"]["mxnet_version"]
      for layer in mxnet_model["nodes"]:
        if layer["op"] == 'null':
          continue
        if layer["op"] not in meta:
          meta[layer["op"]] = [layer['attrs']] if 'attrs' in layer else []
        else:
          if 'attrs' in layer:
            meta[layer["op"]].append(layer['attrs'])
    with open(self.output_file, 'w', encoding='utf8') as f:
      json.dump(meta, f, indent=4)
    logger.info("saved meta info at: {}".format(os.path.abspath(self.output_file)))

  def tensorflow_helper(self: type) -> None:
    from google.protobuf.json_format import MessageToDict
    logger.debug("loading tensorflow model feom '%s'", self.fp)
    meta = dict()
    meta['tf_version'] = tf.__version__
    with tf.io.gfile.GFile(self.fp, 'rb') as f:
      graph_def = tf.compat.v1.GraphDef()
      graph_def.ParseFromString(f.read())
      for layer in graph_def.node:
        layer = MessageToDict(layer, preserving_proto_field_name=True)
        if layer["op"] in ["Const", "Constant"]:
          continue
        if layer["op"] not in meta:
          meta[layer["op"]] = [layer['attr']] if 'attr' in layer else []
        else:
          if 'attr' in layer:
            meta[layer["op"]].append(layer['attr'])
    with open(self.output_file, 'w', encoding='utf8') as f:
      json.dump(meta, f, indent=4)
    logger.info("saved meta info at: {}".format(os.path.abspath(self.output_file)))

  def caffe_helper(self: type) -> None:
    _sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../python/caffe/proto')
    _sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../analyzer/python')
    _sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../python')
    import caffe
    import re
    from caffe.proto import caffe_pb2
    import google.protobuf as pb
    from google.protobuf.json_format import MessageToDict
    meta = {}
    layer_pb2 = caffe_pb2.NetParameter()
    logger.debug("loading onnx model feom '%s'", self.fp)
    with open(self.fp, 'r') as f:
      pb.text_format.Merge(f.read(), layer_pb2)
    for idx, layer in enumerate(layer_pb2.layer):
      layer = MessageToDict(layer, preserving_proto_field_name=True)
      r = re.compile(r"\w*_param")  # capture all _param for layer
      possible_matches = [key for key in list(layer.keys()) if r.match(key)]
      li = [layer[li] for li in possible_matches]
      if layer["type"] not in meta:
        meta[layer["type"]] = li
      else:
        if possible_matches:
          meta[layer["type"]].append(li)
    with open(self.output_file, 'w', encoding='utf8') as f:
      json.dump(meta, f, indent=4)
    logger.info("saved meta info at: {}".format(os.path.abspath(self.output_file)))


def _main(_args=None):
  parser = argparse.ArgumentParser()
  requiredNamed = parser.add_argument_group('required arguments')
  requiredNamed.add_argument('-i', "--input_network", help='Input model path', required=True)
  requiredNamed.add_argument('-sf', '--src_framework', help='src framework', choices=['tensorflow', 'mxnet', 'onnx', 'caffe', 'tflite'], required=True)
  requiredNamed.add_argument('-o', '--output_file', help='output path for json file', required=True)
  args = parser.parse_args() if _args == None else parser.parse_args(_args)
  H = helper(args)
  if args.src_framework == 'onnx':
    H.onnx_helper()
  elif args.src_framework == 'tensorflow':
    H.tensorflow_helper()
  elif args.src_framework == 'mxnet':
    H.mxnet_helper()
  elif args.src_framework == 'caffe':
    H.caffe_helper()
  elif args.src_framework == 'tflite':
    logger.warn("currently tflite oplist generator is not supported")


if __name__ == '__main__':
  try:
    _main()
  except Exception as e:
    logger.warn("couldn't generate oplist for framework")
