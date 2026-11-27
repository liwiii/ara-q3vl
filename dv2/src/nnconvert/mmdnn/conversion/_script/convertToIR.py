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

import sys as _sys
import os
import google.protobuf.text_format as text_format
from six import text_type as _text_type
import json
import torch
import torchvision
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../../../')
from utils.import_helpers import get_cls_and_pkg
from dv_logging import get_logger
import importlib
import argparse
logger = get_logger(__name__)


def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('true', '1'):
    return True
  elif v.lower() in ('false', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def _convert(args):
  if args.inputShape != None:
    inputshape = []
    for x in args.inputShape:
      shape = x.split(',')
      inputshape.append([int(x) for x in shape])
  else:
    inputshape = [None]
  if args.srcFramework == 'caffe':
    from mmdnn.conversion.caffe.transformer import CaffeTransformer
    transformer = CaffeTransformer(args.network, args.weights, "tensorflow", inputshape[0], phase=args.caffePhase)
    graph = transformer.transform_graph()
    data = transformer.transform_data()
    from mmdnn.conversion.caffe.writer import JsonFormatter, ModelSaver, PyWriter
    JsonFormatter(graph).dump(args.dstPath + ".json")
    logger.info("IR network structure is saved as [{}.json]".format(args.dstPath))
    prototxt = graph.as_graph_def().SerializeToString()
    with open(args.dstPath + ".pb", 'wb') as of:
      of.write(prototxt)
    logger.info("IR network structure is saved as [{}.pb]".format(args.dstPath))
    import numpy as np
    with open(args.dstPath + ".npy", 'wb') as of:
      np.save(of, data)
    logger.info("IR weights are saved as [{}.npy]".format(args.dstPath))
    return 0
  elif args.srcFramework == 'caffe2':
    raise NotImplementedError("Caffe2 is not supported yet.")
  elif args.srcFramework == 'keras':
    if args.network != None:
      model = (args.network, args.weights)
    else:
      model = args.weights
    from mmdnn.conversion.keras.keras2_parser import Keras2Parser
    parser = Keras2Parser(model)
  elif args.srcFramework == 'tensorflow' or args.srcFramework == 'tf':
    assert args.network or args.weights
    if not args.network:
      if args.dstNodeName is None:
        raise ValueError("Need to provide the output node of Tensorflow model.")
      if args.inNodeName is None:
        raise ValueError("Need to provide the input node of Tensorflow model.")
      if inputshape is None:
        raise ValueError("Need to provide the input node shape of Tensorflow model.")
      assert len(args.inNodeName) == len(inputshape)
      from mmdnn.conversion.tensorflow.tensorflow_frozenparser import TensorflowParser2
      for i, sl in enumerate(inputshape):
        inputshape[i] = [sl[0]] + sl[2:] + [sl[1]]
      parser = TensorflowParser2(args.weights, inputshape, args.inNodeName, args.dstNodeName, args.dynamic_input, args.config, args.hints, args.dummy, args.output_dir)
    else:
      from mmdnn.conversion.tensorflow.tensorflow_parser import TensorflowParser
      if args.inNodeName and inputshape[0]:
        parser = TensorflowParser(args.network, args.weights, args.dstNodeName, inputshape[0], args.inNodeName, args.dynamic_input)
      else:
        parser = TensorflowParser(args.network, args.weights, args.dstNodeName)
  elif args.srcFramework == 'mxnet':
    assert args.inputShape != None
    if args.weights == None:
      raise ValueError("Need to provide the params file of MXNet model")
      # model = (args.network, inputshape, args.inNodeName)
    else:
      # import re
      # if re.search('.', args.weights):
      #     args.weights = args.weights[:-7]
      # prefix, epoch = args.weights.rsplit('-', 1)
      model_args = (args.network, args.weights, 0, inputshape, args.inNodeName, args.hints, args.dummy, args.output_dir, args.dynamic_input)
    from mmdnn.conversion.mxnet.mxnet_parser import MXNetParser
    parser = MXNetParser(model_args)
  elif args.srcFramework == 'tflite':
    assert args.inputShape != None
    updated_inputshape = []
    for inpshape in inputshape:
      assert inpshape is not None
      updated_inputshape.append(inpshape[1:])
    from mmdnn.conversion.tflite.tflite_parser import TFLiteParser  
    parser = TFLiteParser(weights=args.weights, input_shape=updated_inputshape, in_node_name=args.inNodeName, dst_node_name=args.dstNodeName, hints=args.hints, dynamic_input=args.dynamic_input, output_dir=args.output_dir)
  elif args.srcFramework == 'cntk':
    from mmdnn.conversion.cntk.cntk_parser import CntkParser
    model = args.network or args.weights
    parser = CntkParser(model)
  elif args.srcFramework == 'pytorch':
    assert inputshape != None
    model_class = get_cls_and_pkg(args.network).cls
    model = model_class()
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    assert inputshape[0] is not None
    dummy_input = torch.randn(inputshape[0], device='cpu')
    torch.onnx.export(model, dummy_input, args.output_dir + '/pytorch_onnx.onnx', opset_version=13)
    args.network = args.output_dir + '/pytorch_onnx.onnx'
    src_validator = 'onnx_validator'
    module = importlib.import_module('validate_source.' + src_validator)
    cl = getattr(module, src_validator)
    run_model = getattr(cl, 'run_model_torch')
    x = run_model(args)
    if x != 0:
      logger.error("input network validation failed")
      logger.error("FAILED")
      _sys.exit(x)
    from mmdnn.conversion.onnx.onnx_parser import OnnxParser
    parser = OnnxParser(args.network, args.weights, inputshape, args.inNodeName, args.dstNodeName, args.dummy, args.hints, args.output_dir, args.onnx_backend, args.images,
                        args.data_format, args.dformat, args.dynamic_input, args.summary_csv, args.adjust_file)
  elif args.srcFramework == 'torchscript':
    assert inputshape != None
    model = torch.jit.load(args.network, map_location='cpu')
    assert inputshape[0] is not None
    dummy_input = torch.randn(inputshape[0], device='cpu')
    torch.onnx.export(model, dummy_input, args.output_dir + '/torchscript_onnx.onnx', export_params=True, opset_version=13)
    args.network = args.output_dir + '/torchscript_onnx.onnx'
    src_validator = 'onnx_validator'
    module = importlib.import_module('validate_source.' + src_validator)
    cl = getattr(module, src_validator)
    run_model = getattr(cl, 'run_model_torch')
    x = run_model(args)
    if x != 0:
      logger.error("input network validation failed")
      logger.error("FAILED")
      _sys.exit(x)
    from mmdnn.conversion.onnx.onnx_parser import OnnxParser
    parser = OnnxParser(args.network, args.weights, inputshape, args.inNodeName, args.dstNodeName, args.dummy, args.hints, args.output_dir, args.onnx_backend, args.images,
                        args.data_format, args.dformat, args.dynamic_input, args.summary_csv, args.adjust_file)
  elif args.srcFramework == 'torch' or args.srcFramework == 'torch7':
    from mmdnn.conversion.torch.torch_parser import TorchParser
    model = args.network or args.weights
    assert model != None
    parser = TorchParser(model, inputshape[0])
  elif args.srcFramework == 'onnx':
    from mmdnn.conversion.onnx.onnx_parser import OnnxParser
    parser = OnnxParser(args.network, args.weights, inputshape, args.inNodeName, args.dstNodeName, args.dummy, args.hints, args.output_dir, args.onnx_backend, args.images,
                        args.data_format, args.dformat, args.dynamic_input, args.summary_csv, args.adjust_file)
  elif args.srcFramework == 'darknet':
    from mmdnn.conversion.darknet.darknet_parser import DarknetParser
    parser = DarknetParser(args.network, args.weights, args.darknetStart)
  elif args.srcFramework == 'coreml':
    from mmdnn.conversion.coreml.coreml_parser import CoremlParser
    parser = CoremlParser(args.network)
  else:
    raise ValueError("Unknown framework [{}].".format(args.srcFramework))
  parser.run(args.dstPath)
  return 0


def _get_parser():
  parser = argparse.ArgumentParser(description='Convert other model file formats to IR format.')
  parser.add_argument('--srcFramework',
                      '-f',
                      type=_text_type,
                      choices=["caffe", "cntk", "mxnet", "keras", "tensorflow", 'tf', 'torch', 'torch7', 'onnx', 'darknet', 'coreml', 'pytorch', 'tflite', 'torchscript'],
                      help="Source toolkit name of the model to be converted.")
  parser.add_argument('--weights',
                      '-w',
                      '-iw',
                      type=_text_type,
                      default=None,
                      help='Path to the model weights file of the external tool (e.g caffe weights proto binary, keras h5 binary')
  parser.add_argument('--network', '-n', '-in', type=_text_type, default=None, help='Path to the model network file of the external tool (e.g caffe prototxt, keras json')
  parser.add_argument('--dstPath', '-d', '-o', type=_text_type, required=True, help='Path to save the IR model.')
  parser.add_argument('--inNodeName', '-inode', nargs='+', type=_text_type, default=None, help="[Tensorflow/MXNet] Input nodes' name of the graph.")
  parser.add_argument('--dstNodeName', '-node', nargs='+', type=_text_type, default=None, help="[Tensorflow] Output nodes' name of the graph.")
  parser.add_argument('--inputShape', nargs='+', type=_text_type, default=None, help='[Tensorflow/MXNet/Torch7] Input shape of model (channel, height, width)')
  # Caffe
  parser.add_argument('--caffePhase', type=_text_type, default='TRAIN', help='[Caffe] Convert the specific phase of caffe model.')
  # Darknet
  parser.add_argument('--darknetStart', type=_text_type, choices=["0", "1"], help='[Darknet] Parse the darkent model weight file from the start.')
  parser.add_argument('--config', type=_text_type, default=None, help='pipeline config for tf ssd models')
  parser.add_argument('--hints', type=_text_type, default=None, help='DV - hints flag [ path to json emitted by estimator ]')
  parser.add_argument('--dummy', type=str2bool, default=False, required=False, help='Set to replace unknown ops by dummy layers.')
  parser.add_argument('--output_dir', type=_text_type, help='Ouptut dir to save the output model files and intermediate files')
  parser.add_argument('--onnx_backend', type=_text_type, default="onnxruntime", choices=["onnxruntime"], help='backend for inferring from ONNX model')
  parser.add_argument('--images', type=_text_type, default=None, help='image directory for output comparison')
  parser.add_argument('--data_format', type=_text_type, help='data layout of input')
  parser.add_argument('--dformat', type=_text_type, help='data format of input nodes')
  parser.add_argument('--dynamic_input', type=str2bool, default=False, required=False, help="Network level dynamic inputs")
  parser.add_argument('--summary_csv', type=_text_type, required=False, help='file path to dump warnings/errors')
  parser.add_argument('--adjust_file', type=_text_type, default=None, required=False, help='adjust file')
  parser.add_argument('--platform_target', help='platform_target', type=str, required=False, default='Ara1')
  return parser


def _main():
  parser = _get_parser()
  args = parser.parse_args()
  logger.info("%s", args)
  ret = _convert(args)
  _sys.exit(int(ret))  # cast to int or else the exit code is always 1


if __name__ == '__main__':
  _main()
