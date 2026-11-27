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

from __future__ import division
import os
import sys
import numpy as np
import json
import yaml
from six import text_type, binary_type, integer_types
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from types import SimpleNamespace
import google.protobuf as pb
from caffe.proto import caffe_pb2
from mmdnn.conversion.caffe.c2c import common

from dv_logging import get_logger

logger = get_logger(__name__, 'DVCTOC')

__all__ = [
    "assign_IRnode_values", "convert_onnx_pad_to_tf", 'convert_tf_pad_to_onnx', 'compute_tf_same_padding', 'is_valid_padding', 'download_file', 'compute_tf_deconv_same_padding',
    'shape_to_list', 'list_to_shape', 'compute_tf_explicit_padding', 'AdjustFileConfigurator', 'generate_reshape_param'
]


def layerLikeObj(kv):
  return SimpleNamespace(**kv)


def assign_attr_value(attr, val):
  from mmdnn.conversion.common.IR.graph_pb2 import TensorShape
  '''Assign value to AttrValue proto according to data type.'''
  if isinstance(val, bool):
    attr.b = val
  elif isinstance(val, integer_types):
    attr.i = val
  elif isinstance(val, float):
    attr.f = val
  elif isinstance(val, binary_type) or isinstance(val, text_type):
    if hasattr(val, 'encode'):
      val = val.encode()
    attr.s = val
  elif isinstance(val, TensorShape):
    attr.shape.MergeFromString(val.SerializeToString())
  elif isinstance(val, list):
    if not val: return
    if isinstance(val[0], integer_types):
      attr.list.i.extend(val)
    elif isinstance(val[0], TensorShape):
      attr.list.shape.extend(val)
    elif isinstance(val[0], float):
      attr.list.f.extend(val)
    elif isinstance(val[0], text_type):
      if hasattr(val[0], 'encode'):
        for i in range(len(val)):
          val[i] = val[i].encode()
      attr.list.s.extend(val)
    else:
      raise NotImplementedError('AttrValue cannot be of list[{}].'.format(val[0]))
  elif isinstance(val, np.ndarray):
    assign_attr_value(attr, val.tolist())
  else:
    raise NotImplementedError('AttrValue cannot be of %s' % type(val))

def generate_reshape_param(shape):
  dim_str = f"'dim': {str(shape)}"
  dim_str = f" reshape_param={'shape': { {dim_str} } }"
  return dim_str


def assign_IRnode_values(IR_node, val_dict):
  if 'out_layers' in val_dict:
    IR_node.attr['out_layers'].list.s[:] = []
  for name, val in val_dict.items():
    assign_attr_value(IR_node.attr[name], val)


# For padding
def convert_tf_pad_to_onnx(pads):
  pads = np.reshape(pads, -1).tolist()
  dims = len(pads)
  assert dims % 2 == 0
  ret = []
  for idx in range(0, dims, 2):
    ret.append(pads[idx])
  for idx in range(1, dims, 2):
    ret.append(pads[idx])
  return ret


def convert_onnx_pad_to_tf(pads):
  return np.transpose(np.array(pads).reshape([2, -1])).reshape(-1, 2).tolist()


def is_valid_padding(pads):
  return sum(np.reshape(pads, -1)) == 0


def shape_to_list(shape):
  return [dim.size for dim in shape.dim]


def list_to_shape(shape):
  ret = graph_pb2.TensorShape()
  for dim in shape:
    new_dim = ret.dim.add()
    new_dim.size = dim
  return ret


def compute_tf_deconv_same_padding(input_shape, output_shape, kernel_shape, strides, data_format='NHWC'):
  """ Convert [SAME] padding in tensorflow, keras to onnx pads,
        i.e. [x1_begin, x2_begin...x1_end, x2_end,...] """
  # print (input_shape)
  # print (kernel_shape)
  # print (strides)
  if data_format.startswith('NC'):
    # Not tested
    input_shape = input_shape[2:]
    output_shape = output_shape[2:]
    remove_dim = len(strides) - len(input_shape)
    if remove_dim > 0:
      strides = strides[remove_dim::]
  else:
    input_shape = input_shape[1:-1]
    output_shape = output_shape[1:-1]
    remove_dim = len(input_shape) - len(strides) + 1
    if remove_dim < 0:
      strides = strides[1:remove_dim]
  # print (input_shape)
  # print (kernel_shape)
  # print (strides)
  import math
  up_list = [0]
  down_list = [0]
  #import pdb; pdb.set_trace()
  for idx in range(0, len(input_shape)):
    # kernel_shape[idx] = (kernel_shape[idx] - 1) * dilation_rate + 1
    # output_shape = input_shape[idx] * strides[idx]
    #output_shape = input_shape[idx] * strides[idx] + 1 - strides[idx]
    # this_padding = int(math.ceil((output_shape - kernel_shape[idx])/strides[idx])) + 1 - input_shape[idx]
    this_padding = (input_shape[idx] - 1) * strides[idx] + kernel_shape[idx] - output_shape[idx]
    #this_padding = max(0, this_padding)
    up_list.append(this_padding // 2)
    down_list.append(this_padding - this_padding // 2)
  # print ([0] + up_list + [0] + down_list if data_format.startswith('NC') else up_list + [0] + down_list + [0])
  # print ('-----------------------------------------------------')
  return [0] + up_list + [0] + down_list if data_format.startswith('NC') else up_list + [0] + down_list + [0]


def compute_tf_same_padding(input_shape, kernel_shape, strides, data_format='NHWC'):
  """ Convert [SAME] padding in tensorflow, keras to onnx pads,
        i.e. [x1_begin, x2_begin...x1_end, x2_end,...] """
  # print (input_shape)
  # print (kernel_shape)
  # print (strides)
  if data_format.startswith('NC'):
    # Not tested
    input_shape = input_shape[2:]
    remove_dim = len(strides) - len(input_shape)
    if remove_dim > 0:
      strides = strides[remove_dim::]
  else:
    input_shape = input_shape[1:-1]
    remove_dim = len(input_shape) - len(strides) + 1
    if remove_dim < 0:
      strides = strides[1:remove_dim]
  # print (input_shape)
  # print (kernel_shape)
  # print (strides)
  up_list = [0]
  down_list = [0]
  for idx in range(0, len(input_shape)):
    # kernel_shape[idx] = (kernel_shape[idx] - 1) * dilation_rate + 1
    output_shape = (input_shape[idx] + strides[idx] - 1) // strides[idx]
    this_padding = (output_shape - 1) * strides[idx] + kernel_shape[idx] - input_shape[idx]
    this_padding = max(0, this_padding)
    up_list.append(this_padding // 2)
    down_list.append(this_padding - this_padding // 2)
  # print ([0] + up_list + [0] + down_list if data_format.startswith('NC') else up_list + [0] + down_list + [0])
  # print ('-----------------------------------------------------')
  return [0] + up_list + [0] + down_list if data_format.startswith('NC') else up_list + [0] + down_list + [0]


def compute_tf_explicit_padding(pads, data_format='NHWC'):
  padding = []
  if data_format.startswith('NC'):
    pads = pads[:2] + pads[4:] + pads[2:4]
  elif data_format.startswith('NDC'):
    pads = pads[:4] + pads[6:] + pads[4:6]
  pad_begin = []
  pad_end = []
  for idx in range(0, len(pads), 2):
    pad_begin.append(pads[idx])
    pad_end.append(pads[idx + 1])
  padding = padding + pad_begin + pad_end
  return padding


# network library
def sizeof_fmt(num, suffix='B'):
  for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
    if abs(num) < 1024.0:
      return "%3.1f %s%s" % (num, unit, suffix)
    num /= 1024.0
  return "%.1f %s%s" % (num, 'Yi', suffix)


def _progress_check(count, block_size, total_size):
  read_size = count * block_size
  read_size_str = sizeof_fmt(read_size)
  if total_size > 0:
    percent = int(count * block_size * 100 / total_size)
    percent = min(percent, 100)
    sys.stdout.write("\rprogress: {} downloaded, {}%.".format(read_size_str, percent))
    if read_size >= total_size:
      sys.stdout.write("\n")
  else:
    sys.stdout.write("\rprogress: {} downloaded.".format(read_size_str))
  sys.stdout.flush()


def _single_thread_download(url, file_name):
  from six.moves import urllib
  result, _ = urllib.request.urlretrieve(url, file_name, _progress_check)
  return result


def _downloader(start, end, url, filename):
  import requests
  headers = {'Range': 'bytes=%d-%d' % (start, end)}
  r = requests.get(url, headers=headers, stream=True)
  with open(filename, "r+b") as fp:
    fp.seek(start)
    var = fp.tell()
    fp.write(r.content)


def _multi_thread_download(url, file_name, file_size, thread_count):
  import threading
  fp = open(file_name, "wb")
  fp.truncate(file_size)
  fp.close()
  part = file_size // thread_count
  for i in range(thread_count):
    start = part * i
    if i == thread_count - 1:
      end = file_size
    else:
      end = start + part
    t = threading.Thread(target=_downloader, kwargs={'start': start, 'end': end, 'url': url, 'filename': file_name})
    t.setDaemon(True)
    t.start()
  main_thread = threading.current_thread()
  for t in threading.enumerate():
    if t is main_thread:
      continue
    t.join()
  return file_name


def download_file(url, directory='./', local_fname=None, force_write=False, auto_unzip=False, compre_type=''):
  """Download the data from source url, unless it's already here.
    Args:
        filename: string, name of the file in the directory.
        work_directory: string, path to working directory.
        source_url: url to download from if file doesn't exist.
    Returns:
        Path to resulting file.
    """
  if not os.path.isdir(directory):
    os.mkdir(directory)
  if not local_fname:
    k = url.rfind('/')
    local_fname = url[k + 1:]
  local_fname = os.path.join(directory, local_fname)
  if os.path.exists(local_fname) and not force_write:
    print("File [{}] existed!".format(local_fname))
    return local_fname
  else:
    print("Downloading file [{}] from [{}]".format(local_fname, url))
    try:
      import wget
      ret = wget.download(url, local_fname)
      print("")
    except:
      ret = _single_thread_download(url, local_fname)
  if auto_unzip:
    if ret.endswith(".tar.gz") or ret.endswith(".tgz"):
      try:
        import tarfile
        tar = tarfile.open(ret)
        tar.extractall(directory)
        tar.close()
      except:
        print("Unzip file [{}] failed.".format(ret))
    elif ret.endswith('.zip'):
      try:
        import zipfile
        zip_ref = zipfile.ZipFile(ret, 'r')
        zip_ref.extractall(directory)
        zip_ref.close()
      except:
        print("Unzip file [{}] failed.".format(ret))
  return ret


"""
    r = requests.head(url)
    try:
        file_size = int(r.headers['content-length'])
        return _multi_thread_download(url, local_fname, file_size, 5)
    except:
        # not support multi-threads download
        return _single_thread_download(url, local_fname)
    return result
"""

def caffe_to_graph(caffe_prototxt, log=True):
  """construct a networkx graph from a caffe
  prototxt
  """
  G = common.DVDiGraph()
  # load caffe proto
  layer_pb = caffe_pb2.NetParameter()
  with open(caffe_prototxt, 'r') as f:
    pb.text_format.Merge(f.read(), layer_pb)
  edge_list = list()
  blob = dict()
  if len(layer_pb.input) > 0:
    for inp in layer_pb.input:
      a = caffe_pb2.LayerParameter()
      a.top.append(inp)
      a.type = 'Input'
      if len(layer_pb.input_shape) > 0:
        a.input_param.shape.extend(layer_pb.input_shape)
      else:
        t = caffe_pb2.BlobShape()
        t.dim.extend(layer_pb.input_dim)
        a.input_param.shape.append(t)
      a.name = a.top[0]
      blob[inp] = a
      G.add_node(a)
  for idx, _layer in enumerate(layer_pb.layer):
    # data = _layer
    # op = _layer.type
    top = _layer.top
    bottom = _layer.bottom
    # add G.node with metadata
    G.add_node(_layer)
    for _idx, b in enumerate(bottom):
      src_layer = blob[b]
      if log:
        logger.info('add edge from %s -> %s', src_layer.name, _layer.name)
      edge_list.append((src_layer, _layer, _idx))
    for t in top:
      blob[t] = _layer
  G.add_edges_from(edge_list)
  return G

class AdjustFileConfigurator:
  def __init__(self, adjust_file_path):
    self.allowed_formats = [".json", "yaml", ".yml"]
    self.allowed_precisions = ["int16", "fp32"]
    self.adjust_file_path = adjust_file_path
    self.cfg = self.get_config()
    if self.cfg is not None:
      logger.info("Adjust file load is successful")
  
  def get_config(self):
    if not self.adjust_file_path:
      return None
    try:
      with open(self.adjust_file_path, 'r') as f:
        if self.adjust_file_path.endswith('.json'):
          return json.load(f)
        elif self.adjust_file_path.endswith(('.yaml', '.yml')):
          return yaml.safe_load(f)
        else:
          logger.warning("Unsupported file format. Please provide a JSON or YAML file.")
          return None
    except FileNotFoundError as e:
      logger.info(f"Failed to load the file: {str(e)}")
      return None
    except (json.JSONDecodeError, yaml.YAMLError) as e:
      logger.warning(f"Failed to load the file: {str(e)}")
      return None
  
  def get_layers_with_oprec(self, prec):
    result = []
    if prec not in self.allowed_precisions:
      logger.warning("Given precision not in allowed precisions. Continuing with INT8")
      return result
    for layer in self.cfg["Layers"]:
      if layer["Precision"] not in self.allowed_precisions:
        logger.warning("Allowed precisions")
        logger.warning(f"Precision for the layer {layer}: {layer['Precision']} is not in allowed precisions. Continuing with INT8 for {layer}")
      if layer["Precision"] == prec:
        result.append(layer)
    return result