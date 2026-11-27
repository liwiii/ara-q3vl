#!/bin/env python3
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
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../../../python/caffe/proto')
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../../../analyzer/python')
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../../../python')
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../')
import argparse
import mmdnn.conversion._script.convertToIR as convertToIR
import mmdnn.conversion._script.IRToCode as IRToCode
import mmdnn.conversion._script.IRToModel as IRToModel
from six import text_type as _text_type
# import mxnet as mx
import uuid
import dv_logging
logger = dv_logging.get_logger(__name__, 'CNVRT')


def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('true', '1'):
    return True
  elif v.lower() in ('false', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def _get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--uuid', '-u', type=_text_type, default='123456789', help='UUID for temporary files')
  parser.add_argument('--srcFramework',
                      '-sf',
                      type=_text_type,
                      choices=["caffe", "cntk", "mxnet", "keras", "tensorflow", 'tf', 'pytorch', 'tflite', 'onnx', 'torchscript'],
                      help="Source toolkit name of the model to be converted.")
  parser.add_argument('--inputWeight',
                      '-iw',
                      type=_text_type,
                      default=None,
                      help='Path to the model weights file of the external tool (e.g caffe weights proto binary, keras h5 binary')
  parser.add_argument('--inputNetwork', '-in', type=_text_type, default=None, help='Path to the model network file of the external tool (e.g caffe prototxt, keras json')
  parser.add_argument('--dstFramework',
                      '-df',
                      type=_text_type,
                      choices=['caffe', 'cntk', 'mxnet', 'keras', 'tensorflow', 'coreml', 'pytorch', 'onnx'],
                      required=True,
                      help='Format of model at srcModelPath (default is to auto-detect).')
  parser.add_argument('--outputModel', '-om', type=_text_type, required=True, help='Path to save the destination model')
  parser.add_argument('--set_np_shape', type=str2bool, default=False, help='Turn on/off numpy shape semantics')
  parser.add_argument('--dump_tag', type=_text_type, default=None, help='Tensorflow model dump type', choices=['SERVING', 'TRAINING'])
  parser.add_argument('--dummy', type=str2bool, default=False, required=False, help='Set to replace unknown ops by dummy layers.')
  parser.add_argument('--summary_csv', type=_text_type, required=False, help='file path for dumping warnings/errors')
  parser.add_argument('--debug', type = str2bool, default=False, help="Turn on/off Aten debug mode")
  parser.add_argument('--adjust_file', type=_text_type, default=None, help="adjust file")
  parser.add_argument('--platform_target', help='platform_target', type=str, required=False, default='Ara1')
  # parser.add_argument('--output_dir', type = _text_type, required=False, help="Path to save the debug files model")
  return parser


def _extract_ir_args(args, unknown_args, temp_filename):
  unknown_args.extend(['--srcFramework', args.srcFramework])
  if args.inputWeight is not None:
    unknown_args.extend(['--weights', args.inputWeight])
  if args.inputNetwork is not None:
    unknown_args.extend(['--network', args.inputNetwork])
  if args.dummy is not None:
    unknown_args.extend(['--dummy', str(args.dummy)])
  if args.summary_csv is not None:
    unknown_args.extend(['--summary_csv', str(args.summary_csv)])
  unknown_args.extend(['--dstPath', temp_filename])
  if args.adjust_file is not None:
    unknown_args.extend(['--adjust_file', str(args.adjust_file)])
  ir_parser = convertToIR._get_parser()
  return ir_parser.parse_known_args(unknown_args)


def _extract_code_args(args, unknown_args, temp_filename, network_filename):
  unknown_args.extend(['--dstFramework', args.dstFramework])
  unknown_args.extend(['--IRModelPath', temp_filename + '.pb'])
  unknown_args.extend(['--IRWeightPath', temp_filename + '.npy'])
  unknown_args.extend(['--dstModelPath', network_filename + '.py'])
  unknown_args.extend(['--dstWeightPath', temp_filename + '.npy'])
  unknown_args.extend(['--debug',str(args.debug)])
  # unknown_args.extend(['--output_dir',str(args.output_dir)])
  if args.dummy is not None:
    unknown_args.extend(['--dummy', str(args.dummy)])
  if args.summary_csv is not None:
    unknown_args.extend(['--summary_csv', str(args.summary_csv)])
  if args.platform_target:
    unknown_args.extend(['--platform_target', str(args.platform_target)])
  code_parser = IRToCode._get_parser()
  return code_parser.parse_known_args(unknown_args)


def _extract_model_args(args, unknown_args, temp_filename):
  unknown_args.extend(['--framework', args.dstFramework])
  unknown_args.extend(['--inputNetwork', temp_filename + '.pb'])
  unknown_args.extend(['--inputWeight', temp_filename + '.npy'])
  unknown_args.extend(['--output', args.outputModel])
  model_parser = IRToModel._get_parser()
  return model_parser.parse_known_args(unknown_args)


def remove_temp_files(temp_filename, verbose=False):
  exts = ['.json', '.pb', '.npy', '.py']
  # exts = ['.pb', '.npy', '.py']
  for ext in exts:
    temp_file = temp_filename + ext
    if os.path.isfile(temp_file):
      os.remove(temp_file)
      if verbose:
        print('temporary file [{}] has been removed.'.format(temp_file))


def get_network_filename(framework, temp_filename, output_model_filename):
  if framework in ['pytorch']:
    return os.path.join(os.path.dirname(output_model_filename), os.path.basename(output_model_filename).split('.')[0])
  return temp_filename


def _main(_args=None):

  def _trc(frm, ev, arg):
    if ev == 'call':
      logger.info(frm, arg)

  # sys.settrace(_trc)
  parser = _get_parser()
  args, unknown_args = parser.parse_known_args() if _args == None else parser.parse_known_args(_args)
  logger.info('invoking convert.py with args: %s', args)
  temp_filename = args.uuid
  ir_args, unknown_args = _extract_ir_args(args, unknown_args, temp_filename)
  # if args.set_np_shape == True:
  #   mx.set_np_shape(True)
  ret = convertToIR._convert(ir_args)
  if int(ret) != 0:
    return int(ret)
  if args.dstFramework != 'coreml':
    network_filename = get_network_filename(args.dstFramework, temp_filename, args.outputModel)
    code_args, unknown_args = _extract_code_args(args, unknown_args, temp_filename, network_filename)
    ret = IRToCode._convert(code_args)
    if int(ret) != 0:
      return int(ret)
    from mmdnn.conversion._script.dump_code import dump_code
    ret = dump_code(args.dstFramework, network_filename + '.py', temp_filename + '.npy', args.outputModel, args.dump_tag)
    #remove_temp_files(temp_filename)
    return ret
  model_args, unknown_args = _extract_model_args(args, unknown_args, temp_filename)
  ret = IRToModel._convert(model_args)
  #remove_temp_files(temp_filename)
  return int(ret)


if __name__ == '__main__':
  ret = _main()
  if ret != 0:
    _sys.exit(int(ret))
