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

import numpy as np
import argparse
import collections
import yaml
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from utils.preprocessor import Preprocessor
import dv_logging

logger = dv_logging.get_logger(__name__, 'DVGI')
CLAST = ['NC', 'NWC', 'NHC', 'NHWC', 'NDHWC', 'DHWC', 'HWC', 'HC', 'WC', 'C']
DFormatDict = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "float16": np.float16,
    "float": np.float32,
    "float64": np.float64,
    "float32": np.float32,
    "double": np.double,
}


def load_yaml(yaml_path):
  with open(yaml_path, 'r') as stream:
    cfg = yaml.safe_load(stream)
  return cfg


def absoluteFilePaths(directory):
  all_files = []
  for dirpath, _, filenames in os.walk(directory):
    for f in filenames:
      all_files.append(os.path.abspath(os.path.join(dirpath, f)))
  return all_files


def get_input_shape(args_shape):
  input_shape = []
  if args_shape != None:
    for x in args_shape:
      shape = x.split(',')
      input_shape.append([int(x) for x in shape])
  return input_shape


def clast2first(data):
  if len(data.shape) == 5:
    x = data.transpose(0, 4, 1, 2, 3)
  elif len(data.shape) == 4:
    x = data.transpose(0, 3, 1, 2)
  elif len(data.shape) == 3:
    x = data.transpose(0, 2, 1)
  else:
    x = data
  return x


def convert_shape_clast(shape, data_format):
  new_shape = shape
  for i in range(len(new_shape)):
    if data_format[i] in CLAST:
      new_shape[i] = [new_shape[i][0]] + new_shape[i][2:] + [new_shape[i][1]]
  return new_shape


def generate_input_yaml(args):
  input_image_dict = {}
  input_shape = get_input_shape(args.input_shape)
  try:
    cfg = load_yaml(args.pp_yaml)
    if 'preprocess' in cfg and cfg['preprocess']:
      assert len(args.input_node_name) == 1
    if 'preprocess' in cfg and cfg['preprocess']:
      if args.images:
        images = absoluteFilePaths(args.images) if len(args.input_node_name) == 1 else args.images
      else:
        images = absoluteFilePaths(cfg['network']['images']['verify']) if len(args.input_node_name) == 1 else \
         cfg['network']['images']['verify']
    else:
      if args.images:
        images = args.images
      else:
        images = cfg['network']['images']['verify']
    if args.data_format:
      if 'preprocess' in cfg and cfg['preprocess']:
        for i in args.data_format:
          assert i in ['CHW', 'HWC']
    assert len(images) > 0
    if 'preprocess' in cfg and cfg['preprocess'] and 'preprocessing_params' in cfg:
      yshape = cfg['preprocessing_params']['input_shape']
      assert [yshape['c'], yshape['h'], yshape['w']] == input_shape[0][1:]
      assert len(input_shape[0][1:]) == 3
      assert len(args.input_node_name) == 1
      preprocessor = Preprocessor(cfg['preprocessing_params']['bgr2rgb'], cfg['preprocessing_params']['center_crop'], cfg['preprocessing_params']['resize_with_aspectratio'], cfg['preprocessing_params']['interpolation'], {'c': input_shape[0][1], 'h': input_shape[0][2], 'w': input_shape[0][3]}, cfg['preprocessing_params']['mean'], cfg['preprocessing_params']['scale'])
      for image in images:
        pp_image = preprocessor.read_image(image)
        if pp_image is None:
          raise ValueError("opencv error while reading image " + str(image))
        pp_image = preprocessor.preprocess(pp_image)
        if args.data_format[0] in CLAST:
          input_image_dict[image.split('/')[-1]] = {args.input_node_name[0]: pp_image.reshape([1] + list(pp_image.shape))}
        else:
          input_image_dict[image.split('/')[-1]] = {args.input_node_name[0]: clast2first(pp_image.reshape([1] + list(pp_image.shape)))}
    else:
      return generate_input_raw(args, images)
  except Exception as e:
    logger.error("Exception while generating inputs through yaml")
    logger.error("%s", str(e), exc_info=sys.exc_info())
    logger.info("Falling back to default inputs")
    return generate_input_default(args)
  return input_image_dict


def generate_input_raw(args, images):
  input_image_dict = {}
  input_shape = get_input_shape(args.input_shape)
  shape = input_shape
  if args.data_format:
    shape = convert_shape_clast(input_shape, args.data_format)
  try:
    images_dirs = os.listdir(images)
    for image in images_dirs:
      data_dict = {}
      layers = os.listdir(os.path.join(images, image))
      layer_names = []
      for layer in layers:
        layer_name = os.path.splitext(layer)[0]
        layer_names.append(layer_name)
      for i in range(len(args.input_node_name)):
        input_name = args.input_node_name[i]
        if input_name in layer_names:
          layer_path = os.path.join(images, image, input_name + ".bin")
          if args.dformat[i] not in DFormatDict.keys():
            raise ValueError("Specified dformat not in DformatDict. Supported values for dformat are: " + str(DFormatDict.keys()) + ", and dformat provided is: "+ args.dformat[i])
          pp_image = np.fromfile(layer_path, dtype=DFormatDict[args.dformat[i]])
          data_dict[input_name] = pp_image.reshape(shape[i])
        else:
          raise FileNotFoundError("image not found: " + os.path.join(images, image, input_name) + ".bin")
      input_image_dict[image] = data_dict
  except Exception as e:
    logger.error("Exception while generating raw inputs")
    logger.error("%s", str(e), exc_info=sys.exc_info())
    logger.info("Falling back to default inputs")
    return generate_input_default(args)
  return input_image_dict


def generate_input_default(args):
  data_dict = {}
  input_image_dict = {}
  input_shape = get_input_shape(args.input_shape)
  shape = input_shape
  dtype_ = getattr(args, 'dtype', 'float')
  if args.data_format:
    shape = convert_shape_clast(input_shape, args.data_format)
  for i in range(len(args.input_node_name)):
    if dtype_ == 'int8':
      data_dict[args.input_node_name[i]] = np.random.randint(-128, high=127, size=tuple(shape[i]), dtype=np.int8)
    elif dtype_ == 'uint8':
      data_dict[args.input_node_name[i]] = np.random.randint(0, high=255, size=tuple(shape[i]), dtype=np.uint8)
    else:
      data_dict[args.input_node_name[i]] = np.random.random_sample(tuple(shape[i])).astype(DFormatDict[args.dformat[i]])
  input_image_dict['default'] = data_dict
  return input_image_dict


def generate_input(args):
  if args.pp_yaml:
    x = generate_input_yaml(args)
  elif args.images:
    x = generate_input_raw(args, args.images)
  else:
    logger.info("validating via default inputs")
    x = generate_input_default(args)
  return x


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('-p', "--pp_yaml", help='Preprocessing yaml path', required=False)
  parser.add_argument('-d', '--data_format', help='data format for each node', required=True, nargs='+', default=['CHW'])
  parser.add_argument('-names', '--input_node_name', help='input node names', required=False, nargs='+', default=['data'])
  parser.add_argument('-shapes', '--input_shape', help='input node shapes', required=True, nargs='+', default=[])
  parser.add_argument('-images', '--images', help='path of images', required=False)
  parser.add_argument('-dformat', '--dformat', help='data format of input nodes', required=False)
  cmd_args = parser.parse_args()
  args = dict()
  args = collections.namedtuple('args', 'pp_yaml data_format input_node_name input_shape images dformat')
  # args.pp_yaml = '/auto/work/himanshu/gitlab/detectionout/candid/test/networks/dv_cfg0/mxnet/layer/run.yaml'
  # args.data_format = ['CHW']
  # args.input_node_name = ['data']
  # args.input_shape = ['8,64']
  # args.images = None
  x = generate_input(cmd_args)
