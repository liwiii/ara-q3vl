'''
Copyright (c) 2020,2023 Kinara, Inc. All rights reserved.
Copyright 2026 NXP
 
NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

import logging
import os
IS_CONFIGURED = False


def setup_logging(log_file=None, log_level="INFO"):
  global IS_CONFIGURED
  if IS_CONFIGURED:
    return
  IS_CONFIGURED = True
  logger = logging.getLogger('dv')
  if log_file is not None:
    handler = logging.FileHandler(filename=log_file)
  else:
    handler = logging.StreamHandler()
  mode = os.environ.get('DV_LOGGING', '0')
  if mode == '1':
    formatter = logging.Formatter(fmt='%(asctime)s,%(msecs)d %(levelname)-8s [%(pathname)s %(funcName)s:%(lineno)d] %(message)s')
  else:
    formatter = logging.Formatter('[%(levelname).1s:%(asctime)s] %(dv_component)s %(message)s', '%y%m%d:%H%M%S')
  handler.setFormatter(formatter)
  logger.addHandler(handler)
  if log_level and type(logging.getLevelName(log_level)):
    logger.setLevel(logging.getLevelName(log_level))
  else:
    logger.setLevel(logging.INFO)
  if mode == '1':
    try:
      for log_level in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
        logging.addLevelName(log_level, logging.getLevelName(log_level).lower())
    except Exception:
      pass


def get_logger(name, component_code='DVCONV'):
  """
   Wrapper function that sets the dv_component as per what is passed.
   Each call to `get_logger` can specify the component
   """
  if not IS_CONFIGURED:
    setup_logging()
  logger = logging.getLogger('dv.{}'.format(name))
  return logging.LoggerAdapter(logger, {'dv_component': component_code})
