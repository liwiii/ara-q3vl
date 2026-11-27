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

from six import text_type as _text_type
import sys
import importlib.util
import os.path
from dv_logging import get_logger
logger = get_logger(__name__, 'DVDMPCD')


def dump_code(framework, network_filepath, weight_filepath, dump_filepath, dump_tag):
  if network_filepath.endswith('.py'):
    network_filepath = network_filepath[:-3]
  sys.path.insert(0, os.path.dirname(os.path.abspath(network_filepath)))
  try:
    # MainModel = imp.load_source('MainModel', network_filepath + '.py')
    spec = importlib.util.spec_from_file_location('MainModel', network_filepath + '.py')
    MainModel = importlib.util.module_from_spec(spec)
    sys.modules['MainModel'] = MainModel
    spec.loader.exec_module(MainModel)
  except Exception as e:
    logger.error("Exception loading generated code: %s", str(e))
    return -1
  if framework == 'caffe':
    from mmdnn.conversion.caffe.saver import save_model
  elif framework == 'cntk':
    from mmdnn.conversion.cntk.saver import save_model
  elif framework == 'keras':
    from mmdnn.conversion.keras.saver import save_model
  elif framework == 'mxnet':
    from mmdnn.conversion.mxnet.saver import save_model
  elif framework == 'pytorch':
    from mmdnn.conversion.pytorch.saver import save_model
  elif framework == 'tensorflow':
    from mmdnn.conversion.tensorflow.saver import save_model
    return save_model(MainModel, network_filepath, weight_filepath, dump_filepath, dump_tag)
    
  elif framework == 'onnx':
    from mmdnn.conversion.onnx.saver import save_model
  else:
    raise NotImplementedError("{} saver is not finished yet.".format(framework))
  return save_model(MainModel, network_filepath, weight_filepath, dump_filepath)


def _get_parser():
  import argparse
  parser = argparse.ArgumentParser(description='Dump the model code into target model.')
  parser.add_argument('-f',
                      '--framework',
                      type=_text_type,
                      choices=["caffe", "cntk", "mxnet", "keras", "tensorflow", 'pytorch', 'onnx'],
                      required=True,
                      help='Format of model at srcModelPath (default is to auto-detect).')
  parser.add_argument('-in', '--inputNetwork', type=_text_type, required=True, help='Path to the model network architecture file.')
  parser.add_argument('-iw', '--inputWeight', type=_text_type, required=True, help='Path to the model network weight file.')
  parser.add_argument('-o', '-om', '--outputModel', type=_text_type, required=True, help='Path to save the target model')
  parser.add_argument('--dump_tag', type=_text_type, default=None, help='Tensorflow model dump type', choices=['SERVING', 'TRAINING'])
  return parser


def _main():
  parser = _get_parser()
  args = parser.parse_args()
  ret = dump_code(args.framework, args.inputNetwork, args.inputWeight, args.outputModel, args.dump_tag)
  sys.exit(int(ret))


if __name__ == '__main__':
  _main()
