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

import torch
import dv_logging
import validate_source.torchscript_validator as torchscript_validator
from utils.preprocess_pytorch import DFormatDictTorch
logger = dv_logging.get_logger(__name__, 'DVTSPR')


def preprocess(args):
  cl = getattr(torchscript_validator, "torchscript_validator")
  run_model = getattr(cl, 'run_model')
  x = run_model(args)
  if x != 0:
    logger.error("input network validation failed")
    return x
  assert args.input_shape != None
  inputshape = []
  for x in args.input_shape:
    shape = x.split(',')
    inputshape.append([int(x) for x in shape])
  model = torch.jit.load(args.input_network, map_location='cpu')
  dummy_input = torch.randn(inputshape[0], device='cpu', dtype=DFormatDictTorch[args.dformat[0]])
  torch.onnx.export(model, dummy_input, args.output_dir + '/torchscript_onnx.onnx', export_params=True, opset_version=13)
  return 0
