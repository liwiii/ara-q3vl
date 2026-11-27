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
import time

os.environ['GLOG_minloglevel'] = '1'
os.environ['MXNET_SUBGRAPH_BACKEND'] = 'NONE'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import subprocess
import argparse
from six import text_type as _text_type
import numpy as np
import warnings
import collections
import importlib
import logging
import uuid
import dv_logging
import utils.preprocess_tf
import utils.preprocess_pytorch
import utils.preprocess_torchscript
# import utils.preprocess_mxnet
import utils.onnx_fallback
import utils.encodings_json_to_yaml
from utils.generate_oplist import _main as oplist_main
from mmdnn.conversion._script.convert import _main as convert_main
from mmdnn.conversion.caffe.caffe2caffe import _main as caffe2caffe_main
from utils.io_utils import DFormatDict
import shutil
import json
from copy import deepcopy
import csv
import tabulate
import textwrap
import traceback
import random
import torch
# import mxnet
import tensorflow
import glob

_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/')
logger = dv_logging.get_logger(__name__, 'DVCONV')
CLAST = ['NHWC', 'DHWC', 'HWC', 'HC', 'WC', 'C']


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
  parser.add_argument('-src_framework', type=_text_type, choices=["caffe", "onnx", "cntk", "mxnet", "keras", "tensorflow", 'tf', 'pytorch', 'tflite', 'torchscript'], required=True, help="Source toolkit name of the model to be converted.")
  parser.add_argument('-input_weight', type=_text_type, default=None, required=True, help='Path to the model weights file of the external tool (e.g caffe weights proto binary, keras h5 binary/ , ckpt if -od is true')
  parser.add_argument('-input_network', type=_text_type, default=None, help='Path to the model network file of the external tool (e.g caffe prototxt, keras json')
  parser.add_argument('-dst_framework', type=_text_type, choices=['caffe', 'cntk', 'mxnet', 'keras', 'tensorflow', 'coreml', 'pytorch', 'onnx'], required=True, help='Format of model at srcModelPath (default is to auto-detect).')
  parser.add_argument('-output_model_name', type=_text_type, required=True, help='Output model name to save. .prototxt and .caffemodel will be saved in output_dir')
  parser.add_argument('-output_dir', type=_text_type, required=True, help='Output dir to save the output model files and intermediate files')
  parser.add_argument('-set_np_shape', type=str2bool, default=False, help='Turn on/off numpy shape semantics')
  parser.add_argument('-dump_tag', type=_text_type, default=None, help='Tensorflow model dump type', choices=['SERVING', 'TRAINING'])
  parser.add_argument('-input_node_name', nargs='+', type=_text_type, required=False, help="[Tensorflow/MXNet] Input nodes' name of the graph.")
  parser.add_argument('-state_name', nargs='+', type=_text_type, required=False, default=[], help="[MXNet] State names of the lstm model.")
  parser.add_argument('-output_node_name', nargs='+', type=_text_type, required=False, default=None, help="[Tensorflow] Output nodes' name of the graph.")
  parser.add_argument('-input_shape', nargs='+', type=_text_type, required=True, help='[Tensorflow/MXNet/Torch7] Input shape of model (channel, height, width)')
  parser.add_argument('-dformat', nargs='+', type=_text_type, required=False, help='Data format of every input node')
  parser.add_argument('-state_shape', nargs='+', type=_text_type, required=False, default=[], help='State shape of lstm model')
  parser.add_argument('-label_name', nargs='+', type=_text_type, required=False, help='[MXNet] Label names of model(if any) separated by space ')
  parser.add_argument('-label_shape', nargs='+', type=_text_type, required=False, help='[MXNet] Label shapes(c,h,w) of model in same order as label names separated by space')
  parser.add_argument('-log_level', type=_text_type, required=False, help='Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
  parser.add_argument('-log_file', type=_text_type, required=False, help='path to log file')
  parser.add_argument('-transform_graph', type=str2bool, default=False, help='Turn on/off GraphTransform tool in TF')
  parser.add_argument('-config', type=_text_type, required=False, help='pipeline config if -od is true')
  parser.add_argument('-dry_run', type=int, required=False, default=0, help='dryrun to validate')
  parser.add_argument('-manifest_json', type=_text_type, required=False, default=None, help='manifest json from fwgen')
  parser.add_argument('-hints', type=_text_type, nargs='+', required=False, help='DV - hints flag [ path to json emitted by estimator ]')
  parser.add_argument('-fork_sftmx_argmx', type=str2bool, default=False, help='in sfmx->argmx, make sfmx a seperate output node')
  parser.add_argument('-dummy', type=str2bool, default=True, required=False, help='Set to replace unknown ops by dummy layers.')
  parser.add_argument('-pp_yaml', type=_text_type, required=False, help='preprocessing argument(yaml) file')
  parser.add_argument('-dump_debug', type=str2bool, default=False, help='dump debug data')
  parser.add_argument('-images', type=_text_type, required=False, help='raw image directory with valid data bin files for forward pass on Src and Converted model. Currently only supported for src models in [ONNX, ]')
  parser.add_argument('-data_format', type=_text_type, nargs='+', choices=['HWC', 'HC', 'WC', 'C', 'CHW', 'CH', 'CW', 'CDHW', 'TNC', 'NTC', 'DHWC', 'NHWC', 'NCHW', 'NC', 'NCW', 'NCH', 'N', 'NCDHW'], required=False, help='data layout of input')
  parser.add_argument('-name', type=_text_type, required=False, help='name of the model')
  parser.add_argument('-ignore_mse', type=str2bool, default=False, help='ignore error metrics')
  parser.add_argument('-onnx_backend', type=_text_type, default="onnxruntime", choices=["onnxruntime"], help='Backend for running ONNX models. Choices = ["onnxruntime"]')
  parser.add_argument('-dynamic', type=str2bool, default=False, help='dynamic shape network like densepose')
  parser.add_argument('-platform_target', type=_text_type, default="Ara1", choices=["Ara1", "Ara2", "TFLite", 'Python'], help='Platform Target')
  parser.add_argument('-dynamic_input', required=False, type=str2bool, default=False, help="Network level dynamic input")
  parser.add_argument('-summary_csv', type=_text_type, required=False, help='file path for dumping warnings/errors')
  parser.add_argument('-iname', nargs='+', type=_text_type, required=False, help="mapping of names from old special characters(names containing .,:,/,:: or numeric etc.) to valid names")
  parser.add_argument('-debug', type=str2bool, default=False, help='Turn on/off Aten debug mode')
  parser.add_argument('-c2c_config', type=_text_type, required=False, help="Path to the Config File for c2c Passes")
  parser.add_argument('-no_final_perm_opt', type=str2bool, default=False, help="don't run PermuteToReshape c2c transform")
  parser.add_argument('-force_fallback', type=str2bool, default=False, help='Force tf/mxnet model onnx fallback flow')
  parser.add_argument('-dtype', type=_text_type, required=False, help="input data type")
  parser.add_argument('-random_seed', type=_text_type, required=False, help="seed to initialize pseudo-random number generators in converter")
  parser.add_argument('-tf2onnx_happened', type=_text_type, default=False, required=False, help="did tf model got converted to onnx model ")
  parser.add_argument('-retain_ext_data', type=str2bool, default=False, required=False, help="retain externally saved weights data")
  parser.add_argument('-adjust_file', type=_text_type, default=None, help="Adjust file path")
  parser.add_argument('-encodings_json_file', type=_text_type, default=None, required=False, help="onnx encodings json file path")

  return parser

def remove_all_but_selected_files(dirpath, preserve_files_list):
  if not os.path.isdir(dirpath):
    logger.error('Invalid dirpath: %s', dirpath)
    raise
  try:
    files = glob.glob(os.path.join(dirpath, "*"))
    for f in files:
      if (os.path.isfile(f) or os.path.islink(f)) and os.path.basename(f) not in preserve_files_list:
          os.unlink(f)
      elif os.path.isdir(f):
          shutil.rmtree(f)
  except Exception as e:
    logger.error('Failed to remove specified files from %s: %s', dirpath, e)

def remove_temp_files(temp_filename):
  exts = ['.json', '.pb', '.npy', '.py', '-0000.params', '.onnx']
  
  for ext in exts:
    temp_file = temp_filename + ext
    if os.path.isfile(temp_file):
      os.remove(temp_file)
      logger.debug('temporary file [{}] has been removed.'.format(temp_file))
  
  if args.src_framework in ['onnx'] and not args.retain_ext_data:
    remove_all_but_selected_files(os.path.join(args.output_dir, 'onnx_assets'), ['__model.onnx'])

def exit_gracefully(args, status):
  if args.src_framework in ['onnx']:
    print_error_summary(args.summary_csv)
    if not args.retain_ext_data:
      remove_all_but_selected_files(os.path.join(args.output_dir, 'onnx_assets'), ['__model.onnx'])
  logger.error("FAILED")
  _sys.exit(status)


def cc_convert_cmd(args):
  CMD = [str(_sys.executable), os.path.dirname(os.path.abspath(__file__)) + '/mmdnn/conversion/caffe/caffe2caffe.py']
  CMD.append("-inet")
  CMD.append(args.input_network)
  CMD.append("-iwt")
  CMD.append(args.input_weight)
  CMD.append("-onet")
  CMD.append(args.output_model + '.prototxt')
  CMD.append("-owt")
  CMD.append(args.output_model + '.caffemodel')
  CMD.append("-output_dir")
  CMD.append(str(args.output_dir))
  CMD.append("-src_framework")
  CMD.append(str(args.src_framework))
  CMD.append("-output_model_name")
  CMD.append(str(args.output_model_name))
  if args.manifest_json is not None:
    CMD.append("-manifest_json")
    CMD.append(args.manifest_json)
  if args.fork_sftmx_argmx:
    CMD.append("-fork_sftmx_argmx")
    CMD.append('true')
  if args.dummy_json:
    CMD.append("-dummy_json")
    CMD.append(args.dummy_json)
  if args.pp_yaml:
    CMD.append("-pp_yaml")
    CMD.append(args.pp_yaml)
  if args.name:
    CMD.append("-name")
    CMD.append(args.name)
  if args.data_format:
    data_format = ''
    for i in args.data_format:
      data_format += i
      data_format += ' '
    CMD.append("-data_format")
    CMD.append(data_format)
  if args.dformat:
    dformat = ''
    for i in args.dformat:
      dformat += i
      dformat += ' '
    CMD.append("-dformat")
    CMD.append(dformat)
  if args.input_node_name:
    inode = ' '.join(args.input_node_name)
    CMD.append("-inode")
    CMD.append(inode)
  if args.output_node_name:
    onode = ' '.join(args.output_node_name)
    CMD.append("-onode")
    CMD.append(onode)
  if args.input_shape:
    CMD.append("-input_shape")
    CMD.append(args.input_shape)
  if args.dynamic_input:
    CMD.append("-dynamic_input")
    CMD.append(str(args.dynamic_input))
  if args.images:
    CMD.append("-images")
    CMD.append(args.images)
  if args.dynamic:
    CMD.append("-dynamic")
    CMD.append(str(args.dynamic))
  if args.c2c_config:
    CMD.append("-c2c_config")
    CMD.append(args.c2c_config)
  if args.platform_target:
    CMD.append("-platform_target")
    CMD.append(str(args.platform_target))
  if args.dry_run:
    CMD.append("-dry_run")
    CMD.append(str(args.dry_run))
  if args.no_final_perm_opt:
    CMD.append("-no_final_perm_opt")
    CMD.append(str(args.no_final_perm_opt))
  if args.dtype:
    CMD.append("-dtype")
    CMD.append(args.dtype)
  if args.summary_csv:
    CMD.append("-summary_csv")
    CMD.append(args.summary_csv)
  if args.tf2onnx_happened:
    CMD.append("-tf2onnx_happened")
    CMD.append(str(args.tf2onnx_happened))
  if args.adjust_file:
    CMD.append("-adjust_file")
    CMD.append(str(args.adjust_file))
  CMD.append("-ignore_mse")
  CMD.append(str(args.ignore_mse))
  return CMD


def nnconv_cmd(args):
  CMD = [str(_sys.executable), os.path.dirname(os.path.abspath(__file__)) + '/mmdnn/conversion/_script/convert.py']
  CMD.append("-u")
  CMD.append(args.tmp)
  CMD.append("-sf")
  CMD.append(args.src_framework)
  CMD.append("-df")
  CMD.append(args.dst_framework)
  if args.input_network is not None:
    CMD.append("-in")
    CMD.append(args.input_network)
  CMD.append("-iw")
  CMD.append(args.input_weight)
  CMD.append("-om")
  CMD.append(os.path.join(args.output_dir, 'params', args.output_model_name))
  CMD.append("--inputShape")
  CMD.extend(args.input_shape)
  CMD.append("--set_np_shape")
  CMD.append(str(args.set_np_shape))
  CMD.append("--transform_graph")
  CMD.append(str(args.transform_graph))
  if args.input_node_name:
    CMD.append("--inNodeName")
    CMD.extend(args.input_node_name)
  if args.output_node_name:
    CMD.append("--dstNodeName")
    CMD.extend(args.output_node_name)
  if args.config:
    CMD.append("--config")
    CMD.append(args.config)
  if args.hints and len(args.hints) and os.path.exists(args.hints[-1]):
    if not args.dry_run:
      CMD.append("--hints")
      CMD.append(str(args.hints[-1]))
  CMD.append("--dummy")
  CMD.append(str(args.dummy))
  CMD.append("--adjust_file")
  CMD.append(str(args.adjust_file))
  CMD.append("--output_dir")
  CMD.append(str(args.output_dir))
  if args.src_framework in ["pytorch", "onnx", "torchscript"] and args.onnx_backend:
    CMD.append("--onnx_backend")
    CMD.append(args.onnx_backend)
  if args.images and args.src_framework in ["pytorch", "onnx", "torchscript"]:
    CMD.append("--images")
    CMD.append(args.images)
  CMD.append("--data_format")
  CMD.append(' '.join(i for i in args.data_format))
  CMD.append("--dformat")
  CMD.append(' '.join(i for i in args.dformat))
  if args.dynamic_input:
    CMD.append("--dynamic_input")
    CMD.append(str(args.dynamic_input))
  CMD.append("--summary_csv")
  CMD.append(str(args.summary_csv))
  CMD.append("--debug")
  CMD.append(str(args.debug))
  if args.c2c_config:
    CMD.append("-c2c_config")
    CMD.append(args.c2c_config)
  if args.dtype:
    CMD.append("-dtype")
    CMD.append(args.dtype)
  if args.platform_target:
    CMD.append("--platform_target")
    CMD.append(args.platform_target)
  cmds = ""
  for c in CMD:
    cmds = cmds + " " + c
  logger.info("converter cmd: %s", cmds)
  return CMD


SPECIAL_CHARS = [".", "/", ":", "[", "]", ",", "{", "}", "-", "\\", ";"]


def special(name):
  for char in SPECIAL_CHARS:
    if char in name:
      return True
  if name.isdigit():
    return True
  return False


def validate_args(args):
  logger.info("validating arguments ...")
  if args.data_format is None:
    args.data_format = []
    dim_len = len(args.input_shape[0].split(','))
    if dim_len == 4:
      if args.src_framework in ['tensorflow', 'tf']:
        args.data_format.append('DHWC')
      else:
        args.data_format.append('CDHW')
    else:
      if args.src_framework in ['tensorflow', 'tf']:
        args.data_format.append('HWC')
      else:
        args.data_format.append('CHW')
  if args.src_framework in ['onnx'] and args.iname is not None and len(args.iname):
    for text in args.iname:
      imap = text.split("@")
      if special(imap[1]):
        logger.error("iname map must contain valid symbols")
        return False
  if args.dformat is None:
    args.dformat = ['float'] * len(args.input_shape)
  if len(args.data_format) != 1 and len(args.data_format) != len(args.input_node_name):
    logger.error("data_format must be None or match input node count")
    return False
  if len(args.dformat) != 1 and len(args.dformat) != len(args.input_shape):
    logger.error("dformat must be None or match input shape count")
    return False
  for i in args.dformat:
    if i not in DFormatDict:
      logger.error("dformat " + i + " is not valid. Valid : " + ", ".join([i for i in DFormatDict.keys()]))
  if len(args.data_format) == 1:
    for i in range(len(args.input_shape[1:])):
      args.data_format.append(args.data_format[0])
  if args.images:
    if not os.path.exists(args.images):
      logger.error("directory doesn't exist: %s", str(args.images))
      return False
  if args.pp_yaml:
    if not os.path.exists(args.pp_yaml):
      logger.error("file not found: %s", str(args.pp_yaml))
      return False
  if args.src_framework == 'tensorflow' or args.src_framework == 'tf':
    args.src_framework = 'tensorflow'
    for i in args.data_format:
      if i not in CLAST:
        logger.error("data format must be in %s for tensorflow", CLAST)
        return False
    if args.config:
      if not os.path.exists(args.config):
        logger.error("file not found: %s", str(args.config))
        return False
      if args.input_network:
        logger.warning("ignoring input network for tensorflow converter")
        args.input_network = None
      if args.output_node_name:
        logger.warning("ignoring output node name for tensorflow converter")
        args.output_node_name = None
      if args.input_node_name:
        logger.warning("ignoring input node name for tensorflow converter")
        args.input_node_name = None
    else:
      if args.input_network:
        logger.warning("ignoring input network for tensorflow converter")
        args.input_network = None
      if not args.output_node_name:
        logger.error("need output node name for tensorflow converter")
        return False
      if not args.input_node_name:
        logger.error("need input node name for tensorflow converter")
        return False
  elif args.src_framework == 'onnx':
    logger.info('Onnx Framework. Beta mode.')
    if args.output_node_name == None:
      args.output_node_name = ['None']
    if args.input_node_name == None:
      args.input_node_name = ['None']
    if args.encodings_json_file:
      if not os.path.exists(args.encodings_json_file):
        logger.error("file not found: %s", str(args.encodings_json_file))
        return False
  else:
    if args.src_framework == 'mxnet':
      logger.error("Mxnet support has been deprecated")
      exit_gracefully(args, -1)
      # if not args.input_node_name:
      #   args.input_node_name = ['data']
    if args.src_framework != 'caffe' and args.output_node_name:
      logger.warning("ignoring output nodes for %s converter", str(args.src_framework))
      args.output_node_name = ['None']
  if not os.path.exists(str(args.input_weight)) and not args.config:
    logger.error("file not found: %s", str(args.input_weight))
    return False
  args.summary_csv = os.path.join(args.output_dir, 'stats', 'summary.csv')
  logger.info("dumping warnings/errors to %s", args.summary_csv)
  if args.input_network:
    if not (args.src_framework == 'pytorch' or args.src_framework == 'tflite' or args.src_framework == 'torchscript'):
      if not os.path.exists(str(args.input_network)):
        logger.error("file not found: %s", str(args.input_network))
        return False
    else:
      logger.debug("input network is pytorch, model class %s must be present on the PYTHONPATH", args.input_network)
  if len(args.state_name) != len(args.state_shape):
    logger.error("state_shape count must match state_name count")
    return False
  for i in args.state_shape:
    i = i.split(',')
    try:
      i = list(map(int, i))
    except Exception as e:
      logger.error("can't convert state shape to integers")
      return False
  for i in args.input_shape:
    i = i.split(',')
    try:
      i = list(map(int, i))
    except Exception as e:
      logger.error("can't convert input shape to integers")
      return False
  if args.images and args.src_framework not in ["pytorch", "onnx", "torchscript"]:
    logger.warn('raw_images argument only supported for src frameworks in ["pytorch", "onnx", "torchscript"]')
  logger.info("arguments validated")
  return True


def generate_oplist(args):
  CMD = [str(_sys.executable), os.path.dirname(os.path.abspath(__file__)) + '/utils/generate_oplist.py']
  CMD.append("-i")
  if args.input_network is not None:
    CMD.append(args.input_network)
  else:
    CMD.append(args.input_weight)
  CMD.append("-sf")
  CMD.append(args.src_framework)
  CMD.append("-o")
  CMD.append(os.path.join(str(args.output_dir), 'stats', 'op_list.json'))
  cmds = ""
  for c in CMD:
    cmds = cmds + " " + c
  logger.info("running op helper cmd: %s", cmds)
  return CMD


def print_error_summary(csv_path):
  if csv_path is not None and os.path.exists(csv_path):
    with open(csv_path, "r") as csv_file:
      csv_reader = csv.DictReader(csv_file, delimiter=',')
      if not len(list(csv_reader)):
        return
    with open(csv_path, "r") as csv_file:
      csv_reader = csv.DictReader(csv_file, delimiter=',')
      table = []
      for row in csv_reader:
        source_contents = row['source'].split('\n')
        row_source = "\n".join(["\n".join(textwrap.wrap(elem, 70)) for elem in source_contents if 'doc_string' not in elem])
        issue_contents = row['issue'].split('\n')
        row_issue = "\n".join(["\n".join(textwrap.wrap(elem, 70)) for elem in issue_contents])
        table.append([row['origin'], row['op'], row_issue, row_source])

      print("\nConverter Error Summary:")
      print(tabulate.tabulate(table, headers=["Origin", "Op", "Issue", "Source"], tablefmt="grid", maxcolwidths=[30, 30, None, None]))
      print("\n")


def run_validator_oplist_nnconv(args):
  if args.src_framework == 'onnx':
    onnx_assets_path = os.path.join(args.output_dir, 'onnx_assets')
    if os.path.isdir(onnx_assets_path):
      try:
        shutil.rmtree(onnx_assets_path)
        logger.info(f"Removed existing directory: {onnx_assets_path}")
      except Exception as e:
        logger.error(f"Error removing directory {onnx_assets_path}: {e}")
        raise
    try:
      os.makedirs(onnx_assets_path)
      logger.info(f"Created directory: {onnx_assets_path}")
    except Exception as e:
      logger.error(f"Error creating directory {onnx_assets_path}: {e}")
      raise
    args.tmp_onnx = os.path.join(args.output_dir, 'onnx_assets', 'dv_netspec_defn_' + uuid.uuid4().hex)
  if not args.config:
    src_validator = args.src_framework + '_validator'
    module = importlib.import_module('validate_source.' + src_validator)
    cl = getattr(module, src_validator)
    run_model = getattr(cl, 'run_model')
    x = run_model(args)
    if x != 0:
      logger.error("input network validation failed")
      exit_gracefully(args, x)
  logger.info("generating an op level report for the framework.")
  util_oplist_cmd = generate_oplist(args)
  # subprocess.run(util_oplist_cmd)
  oplist_main(util_oplist_cmd[2:])
  logger.debug("op report generated.")
  logger.info("beginning model conversion %s", args)
  cmd = nnconv_cmd(args)
  # x = subprocess.run(cmd).returncode
  x = convert_main(cmd[2:])
  return x


def model_generated(args):
  output_model = os.path.join(args.output_dir, 'params', args.output_model_name)
  weights = output_model + '.caffemodel'
  network = output_model + '.prototxt'
  if not os.path.exists(weights) or not os.path.exists(network):
    return False
  return True


def gen_random_seed(random_seed, output_dir):
  seed = 0
  random_seed_file = os.path.join(output_dir, 'SEED_CONVERTER')
  if random_seed is None or len(random_seed.strip()) == 0:
    if os.path.exists(random_seed_file):
      with open(random_seed_file, 'r') as f:
        seed = int(f.read())
    else:
      t = int(time.time() * 1000.0)
      seed = (((t & 0xff000000) >> 24) + ((t & 0x00ff0000) >> 8) + ((t & 0x0000ff00) << 8) + ((t & 0x000000ff) << 24))
      with open(random_seed_file, 'w') as f:
        f.write(str(seed))
  else:
    seed = int(random_seed)
    with open(random_seed_file, 'w') as f:
      f.write(str(seed))
  return seed

def set_random_seed(seed_val):
  random.seed(seed_val)
  np.random.seed(seed_val)
  torch.manual_seed(seed_val)
  # mxnet.random.seed(seed_val)
  tensorflow.random.set_seed(seed_val)

def _main(args, unknown_args):
  if not validate_args(args):
    exit_gracefully(args, -1)
  if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
  seed = gen_random_seed(args.random_seed, args.output_dir)
  set_random_seed(seed)
  for i in range(len(args.data_format)):
    if args.data_format[i] in ['CHW', 'CDHW', 'HWC', 'WC', 'C', 'CH', 'CW', 'DHWC']:
      args.input_shape[i] = str('1,' + args.input_shape[i])
  if not os.path.exists(os.path.join(args.output_dir, 'params')):
    os.makedirs(os.path.join(args.output_dir, 'params'))
  if not os.path.exists(os.path.join(args.output_dir, 'stats')):
    os.makedirs(os.path.join(args.output_dir, 'stats'))

  if args.src_framework in ["onnx"] and args.encodings_json_file:
    en_generate_precision_adjust_yaml_result = 0
    output_adjust_precision_yaml_path = args.output_dir + '/stats/adjust_precision.yaml'
    en_generate_precision_adjust_yaml_result = utils.encodings_json_to_yaml.generate_precision_adjust_yaml(args, output_adjust_precision_yaml_path)
    args.adjust_file = output_adjust_precision_yaml_path
    if en_generate_precision_adjust_yaml_result!=0:
      exit_gracefully(args, en_generate_precision_adjust_yaml_result)
    logger.info("successfully generated adjust_precision.yaml")


  output_model = os.path.join(args.output_dir, 'params', args.output_model_name)
  if args.src_framework == 'tf':
    args.src_framework = 'tensorflow'
  util_oplist_cmd = generate_oplist(args)
  if args.src_framework == "caffe" and args.dst_framework == "caffe":
    logger.info("generating an op level report for the model.")
    # subprocess.run(util_oplist_cmd)
    oplist_main(util_oplist_cmd[2:])
    subprocess.run(util_oplist_cmd)
    cc_args_type = collections.namedtuple('CCArgs', 'input_network input_weight output_model manifest_json dummy_json fork_sftmx_argmx pp_yaml name data_format input_node_name output_node_name input_shape output_dir images dynamic dformat dynamic_input platform_target summary_csv dry_run c2c_config dtype no_final_perm_opt src_framework tf2onnx_happened output_model_name ignore_mse adjust_file')
    dummy_json = os.path.join(args.output_dir, "dummy_layers.json")
    cc_args = cc_args_type(input_network=args.input_network, input_weight=args.input_weight, output_model=output_model, manifest_json=args.hints, dummy_json=dummy_json, fork_sftmx_argmx=args.fork_sftmx_argmx, pp_yaml=args.pp_yaml, name=args.name, data_format=args.data_format, input_node_name=args.input_node_name, output_node_name=args.output_node_name, input_shape=args.input_shape, output_dir=args.output_dir, images=args.images, dynamic=args.dynamic, dformat=args.dformat, dynamic_input=args.dynamic_input, platform_target=args.platform_target, summary_csv=args.summary_csv, dry_run=args.dry_run, c2c_config=args.c2c_config, no_final_perm_opt=args.no_final_perm_opt, dtype=args.dtype, src_framework=args.src_framework, tf2onnx_happened=args.tf2onnx_happened, output_model_name=args.output_model_name, ignore_mse=args.ignore_mse, adjust_file=args.adjust_file)
    cmd = cc_convert_cmd(cc_args)
    # proc = subprocess.run(cmd).returncode
    try:
      proc = caffe2caffe_main(cmd[2:])
    except Exception as e:
      c2c_logs_file = args.output_dir + '/stats/c2c_logs.log'
      with open(c2c_logs_file, 'w') as f:
        f.writelines(traceback.format_exc())
      logger.error("exception occured during model normalization: %s", str(e))
      proc = -1
    if proc != 0:
      logger.error("Caffe to Caffe conversion failed for input_file %s. ret_code: %s" % (args.input_network, proc))
      exit_gracefully(args, proc)
    else:
      logger.info("successful conversion of Caffe2Caffe")
      logger.info("PASSED")
      return 0
  args.tmp = os.path.join(args.output_dir, 'dv_netspec_defn_' + uuid.uuid4().hex)
  logger.info("running input network validator")
  args_copy = deepcopy(args)
  if args.src_framework == 'tensorflow' and args.config:
    logger.info('preprocessing graph')
    obj = utils.preprocess_tf.preprocess(args, args.tmp)
    args.input_weight = obj['pruned_graph']
    args.tflite_graph = obj['tflite_graph']
    args.input_node_name = obj['inode']
    args.output_node_name = obj['onode']
    x = utils.preprocess_tf.preprocess_names(args)
    if x != 0:
      logger.error("Preprocessing tensorflow network failed")
      exit_gracefully(args, x)
  if args.src_framework == 'pytorch':
    logger.info('preprocessing pytorch graph')
    x = utils.preprocess_pytorch.preprocess(args)
    if x != 0:
      logger.info('preprocessing of pytorch graph failed')
      exit_gracefully(args, x)
    args.input_network = args.output_dir + '/pytorch_onnx.onnx'
    args.input_weight = args.output_dir + '/pytorch_onnx.onnx'
    args.src_framework = 'onnx'
    args.input_node_name = ['None']
  if args.src_framework == 'torchscript':
    logger.info('preprocessing torchscript graph')
    x = utils.preprocess_torchscript.preprocess(args)
    if x != 0:
      logger.info('preprocessing of torchscript graph failed')
      exit_gracefully(args, x)
    args.input_network = args.output_dir + '/torchscript_onnx.onnx'
    args.input_weight = args.output_dir + '/torchscript_onnx.onnx'
    args.src_framework = 'onnx'
  if args.src_framework == 'mxnet':
    x = utils.preprocess_mxnet.preprocess(args)
    if x != 0:
      logger.error("Preprocessing mxnet network failed")
      exit_gracefully(args, x)
  if not args.force_fallback:
    x = run_validator_oplist_nnconv(args)
  else:
    x = 1
  y = -1
  if x != 0:
    if args.src_framework in ['mxnet', 'tensorflow']:
      y = utils.onnx_fallback.run(args_copy)
      if y == 0:
        x = run_validator_oplist_nnconv(args_copy)
        if x == 0:
          args = args_copy
  if x != 0:
    logger.error("model conversion failed")
    exit_gracefully(args, x)
  logger.info("validating network outputs")
  dummy_layers = os.path.join(args.output_dir, 'dummy_layers.json')
  if args.dummy and os.path.exists(dummy_layers):
    with open(dummy_layers, 'r') as fd:
      dummy_layer = json.load(fd)
      if dummy_layer:
        logger.info("Some ops replaced with dummy layers. Skipping MSE comparison.")
        logger.info("List of dummy layers : %s", dummy_layers)
        logger.info("PASSED")
        remove_temp_files(args.tmp)
        return 0
  if not args.config:
    if args.src_framework == 'pytorch' or args.src_framework == 'torchscript':
      args.src_framework = 'onnx'
      args.input_network = args.tmp + '.onnx'
    out_validator = args.src_framework + '2' + args.dst_framework
  else:
    out_validator = 'tflite' + '2' + args.dst_framework
  module = importlib.import_module('validate_outputs.' + out_validator)
  cl = getattr(module, out_validator)
  compare_outputs = getattr(cl, 'compare_outputs')
  mse_logs = []
  validator_rv = 0
  if model_generated(args):
    validator_rv = compare_outputs(args, mse_logs)
    if validator_rv != 0 and args.src_framework not in ['onnx']:
      exit_gracefully(args, validator_rv)
  else:
    logger.error("model couldn't be generated successfully")
    exit_gracefully(args, -1)
  logger.info("beginning model normalization")
  cc_args_type = collections.namedtuple('CCArgs', 'input_network input_weight output_model manifest_json dummy_json fork_sftmx_argmx pp_yaml name data_format input_node_name output_node_name input_shape output_dir images dynamic dformat dynamic_input platform_target summary_csv dry_run c2c_config dtype no_final_perm_opt src_framework tf2onnx_happened output_model_name ignore_mse adjust_file')
  
  cc_input_nw_file = '{}.prototxt'.format(output_model)
  cc_input_wt_file = '{}.caffemodel'.format(output_model)
  dummy_json = os.path.join(args.output_dir, "dummy_layers.json")
  cc_args = cc_args_type(input_network=cc_input_nw_file, input_weight=cc_input_wt_file, output_model=output_model, manifest_json=args.hints, dummy_json=dummy_json, fork_sftmx_argmx=args.fork_sftmx_argmx, pp_yaml=args.pp_yaml, name=args.name, data_format=args.data_format, input_node_name=args.input_node_name, output_node_name=args.output_node_name, input_shape=args.input_shape, output_dir=args.output_dir, images=args.images, dynamic=args.dynamic, dformat=args.dformat, dynamic_input=args.dynamic_input, platform_target=args.platform_target, summary_csv=args.summary_csv, dry_run=args.dry_run, c2c_config=args.c2c_config, no_final_perm_opt=args.no_final_perm_opt, dtype=args.dtype, src_framework=args.src_framework, tf2onnx_happened=args.tf2onnx_happened, output_model_name=args.output_model_name, ignore_mse=args.ignore_mse, adjust_file=args.adjust_file)
  cc_cmd = cc_convert_cmd(cc_args)
  # cc_result = subprocess.run(cc_cmd).returncode
  cc_result = 0
  en_json_to_yaml_result = 0
  try:
    cc_result = caffe2caffe_main(cc_cmd[2:])
  except Exception as e:
    c2c_logs_file = args.output_dir + '/stats/c2c_logs.log'
    with open(c2c_logs_file, 'w') as f:
      f.writelines(traceback.format_exc())
    logger.error("exception occured during model normalization: %s", str(e))
    cc_result = -1
  if args.src_framework in ['onnx', 'tflite']:
    logger.info("onnx2caffe MSE info:")
    for mse_log in mse_logs:
      logger.info(mse_log)
  if validator_rv != 0:
    logger.error("mse check failed")
  if cc_result != 0:
    logger.error('model normalization failed')
  if args.src_framework in ["onnx"] and args.encodings_json_file:
    logger.info("converting encodings json to scale_adjust_adjustment yaml")
    layer_name_map_csv_path = args.output_dir + '/stats/layer_name_map.csv'
    output_scale_offet_yaml_path = args.output_dir + '/stats/output_scale_offset_adjustment.yaml'
    en_json_to_yaml_result = utils.encodings_json_to_yaml.convert_json_to_yaml(args, cc_input_nw_file, layer_name_map_csv_path, output_scale_offet_yaml_path)
  if validator_rv != 0:
    exit_gracefully(args, validator_rv)
  if cc_result != 0:
    exit_gracefully(args, cc_result)
  if en_json_to_yaml_result!=0:
    exit_gracefully(args, en_json_to_yaml_result)
  logger.info("PASSED")
  remove_temp_files(args.tmp)
  return 0


if __name__ == '__main__':
  parser = _get_parser()
  args, unknown_args = parser.parse_known_args()
  from dv_logging import setup_logging
  setup_logging(log_file=args.log_file, log_level=args.log_level)
  logger.info('starting converter with args %s', args)
  _main(args, unknown_args)
