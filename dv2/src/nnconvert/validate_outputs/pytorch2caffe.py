'''
Copyright (c) 2018-22, Deep Vision, Inc. All rights reserved.
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
import os
import numpy as np
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../analyzer/python')
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../python')
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../python/caffe/proto')
import collections
import caffe
from dv_logging import get_logger
import torch
import yaml
from utils.preprocessor import Preprocessor
from pathlib import Path
from utils.import_helpers import dump_np_data
from utils.import_helpers import get_cls_and_pkg
from utils.import_helpers import get_variable_name
from utils.import_helpers import dump_json
from utils.io_utils import *
logger = get_logger(__name__, 'DVOPVL')


def _do_compare(args):
  output_model = os.path.join(args.output_dir, 'params', args.output_model_name)
  caffe_net = caffe.Net(output_model + '.prototxt', caffe.TEST, weights=output_model + '.caffemodel')
  model_class = get_cls_and_pkg(args.input_network).cls
  torch_net = model_class()
  torch_net.load_state_dict(torch.load(args.input_weight, map_location='cpu'))
  # switch to inference mode
  torch_net.eval()
  input_shape = []
  if args.input_shape != None:
    for x in args.input_shape:
      shape = x.split(',')
      input_shape.append([int(x) for x in shape])
  input_image_dict = generate_input(args)
  for image in input_image_dict:
    torch_input = []
    for i in range(len(args.input_shape)):
      input_node_name = 'input_{}'.format(i)
      input_data = torch.autograd.Variable(torch.from_numpy(input_image_dict[image][input_node_name]), requires_grad=False).float()
      torch_input.append(input_data)
      # since we support only 1 input node currently, we keep over
      # writing the previous input blob data
      caffe_net.blobs['input'].data[...] = input_image_dict[image][input_node_name]
    torch_out = torch_net(*torch_input)
    caffe_out = caffe_net.forward()
    caffe_out_tensor = list(caffe_out.values())[0]
    torch_out_tensor = None
    if type(torch_out) != list:
      torch_out_tensor = torch_out.detach().numpy()
    else:
      torch_out_tensor = np.asarray(torch_out)
    logger.info('caffe_shape: %s, torch_shape: %s', caffe_out_tensor.shape.__str__(), torch_out_tensor.shape.__str__())
    torch_out_file = os.path.join(args.output_dir, 'outputs', image, 'fp32', 'source')
    caffe_out_file = os.path.join(args.output_dir, 'outputs', image, 'fp32', 'dvnet')
    l = list(caffe_out.keys())[0]
    dump_np_data(torch_out_tensor, torch_out_file, l)
    dump_np_data(caffe_out_tensor, caffe_out_file, l)
    logger.info("Source net output saved to : %s, layer : %s", torch_out_file, l)
    logger.info("DVNet net output saved to : %s, layer : %s", caffe_out_file, l)
    if caffe_out_tensor.shape != torch_out_tensor.shape:
      logger.warn(('caffe output tensor shape and pytorch output tensor shape differ - not checking'
                   ' caffe out shape: %s, pytorch out shape: %s'), caffe_out_tensor.shape.__str__(), torch_out_tensor.shape.__str__())
      return 0
    diff = np.subtract(caffe_out_tensor, torch_out_tensor)
    logger.info('image: %s, mse: %s, diff_max: %s', image, np.square(diff).mean(), diff.max())
  return 0


class pytorch2caffe:

  @staticmethod
  def compare_outputs(args, mse_logs):
    return _do_compare(args)
