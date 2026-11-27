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

import tensorflow.lite as lite
import numpy as np
from dv_logging import get_logger
logger = get_logger(__name__, 'DVIPVAL')


class tflite_validator:

  @staticmethod
  def run_model(args):
    ip = lite.Interpreter(args.input_weight)
    data_dict = dict()
    for i in range(len(ip.get_input_details())):
      loc_name = 'loc_' + str(ip.get_input_details()[i]['index'])
      if args.dtype == "int8":
        data_dict[loc_name] = np.random.randint(-128, high=127, size=tuple(ip.get_input_details()[i]['shape']), dtype=np.int8)
      else:
        data_dict[loc_name] = np.random.random_sample(tuple(ip.get_input_details()[i]['shape'])).astype(np.float32)
    ip.allocate_tensors()
    output_details = ip.get_output_details()
    try:
      for index, name in enumerate(data_dict):
        inp_id = ip.get_input_details()[index]["index"]
        img = data_dict[name]
        ip.set_tensor(inp_id, img)
        ip.invoke()
      return 0
    except Exception as e:
      logger.error("exception occurred when validating mxnet", exc_info=e)
      return -1
