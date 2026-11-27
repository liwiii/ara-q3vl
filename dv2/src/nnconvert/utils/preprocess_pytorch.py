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

import onnx
import torch
import onnxsim
import os
from onnx import helper
import dv_logging
import validate_source.pytorch_validator as pytorch_validator
from utils.import_helpers import get_cls_and_pkg
logger = dv_logging.get_logger(__name__, 'DVPTPR')
DFormatDictTorch = {
    "uint8": torch.uint8,
    #  "uint16": torch.uint16,
    #  "uint32": torch.uint32,
    #  "uint64": torch.uint64,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float": torch.float32,
    "double": torch.double,
}


def preprocess(args):
  cl = getattr(pytorch_validator, "pytorch_validator")
  run_model = getattr(cl, 'run_model')
  x = run_model(args)
  if x != 0:
    logger.error("input network validation failed")
    return x
  if args.input_shape != None:
    inputshape = []
    for x in args.input_shape:
      shape = x.split(',')
      inputshape.append([int(x) for x in shape])
  else:
    inputshape = [None]
  model_class = get_cls_and_pkg(args.input_network).cls
  model = model_class()
  model.load_state_dict(torch.load(args.input_weight, map_location='cpu'))
  dummy_input = []
  for i, input in enumerate(inputshape):
    dummy_input.append(torch.randn(input, device='cpu', dtype=DFormatDictTorch[args.dformat[i]]))
  torch.onnx.export(model, tuple(dummy_input), args.output_dir + '/pytorch_onnx.onnx', opset_version=13)
  return 0
