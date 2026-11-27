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

import sys as _sys
from importlib import import_module
from dv_logging import get_logger
import torch as T
import os
from utils.import_helpers import get_cls_and_pkg
logger = get_logger(__name__)


class torchscript_validator:

  @staticmethod
  def run_model(args):
    try:
      logger.info('looking for class %s, sys.path is %s', args.input_network, _sys.path)
      model = T.jit.load(args.input_network, map_location='cpu')
      logger.info('loaded torchscript model successfully')
      return 0
    except Exception as e:
      logger.error('could not load pytorch model at path, encountered exception, sys.path is %s', _sys.path, exc_info=e)
    return -1
