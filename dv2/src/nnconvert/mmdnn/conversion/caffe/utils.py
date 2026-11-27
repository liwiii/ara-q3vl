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

import re
from types import SimpleNamespace
import caffe
from caffe import layers as L
from caffe import params as P
from math import nan
import uuid
import numpy as np
from dv_logging import get_logger
from collections import namedtuple
import random

logger = get_logger(__name__, 'DVCCUTIL')


def get_lower_case(text):
  '''
    Convert PascalCase name to words concatenated by '_'.
    'PascalCase' -> 'pascal_case'
    '''
  name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', text)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def get_upper_case(text):
  '''
    'pascal_case' -> 'PascalCase'
    '''
  return ''.join([item.title() for item in text.split('_')])


def get_real_name(text):
  text = text.strip().split(':')
  return ''.join(text[:-1])


def _hw_prop_setter(layer_param, h_prop, w_prop, a_prop, h_val, w_val):
  # resp =\
  #     _hw_prop_getter(layer_param, h_prop, w_prop, a_prop)
  # if resp != (h_val, w_val):
  #     setattr(layer_param, w_prop, w_val)
  #     setattr(layer_param, h_prop, h_val)
  #     array_container = getattr(layer_param, a_prop)
  #     layer_param.ClearField(a_prop)
  # else:
  #     arr = getattr(layer_param, a_prop)
  #     if type(arr) == int:
  #         setattr(layer_param, a_prop, h_val)
  #     else:
  #         if len(arr) == 0:
  #             arr = h_val
  #         elif len(arr) == 1:
  #             arr[0] = h_val
  #         else:
  #             arr[0] = h_val
  #             arr[1] = w_val
  if getattr(layer_param, h_prop) > 0 and getattr(layer_param, w_prop) > 0:
    setattr(layer_param, w_prop, w_val)
    setattr(layer_param, h_prop, h_val)
    layer_param.ClearField(a_prop)
  else:
    arr = getattr(layer_param, a_prop)
    if type(arr) == int:
      setattr(layer_param, a_prop, h_val)
    else:
      if len(arr) == 0:
        arr = h_val
      elif len(arr) == 1:
        arr[-1] = h_val
      elif len(arr) == 2:
        arr[0] = h_val
        arr[1] = w_val
      elif len(arr) == 3:
        arr[1] = h_val
        arr[2] = w_val
      elif len(arr) == 4:
        arr[0] = h_val
        arr[2] = h_val
        arr[1] = w_val
        arr[3] = w_val
      elif len(arr) == 6:
        arr[1] = h_val
        arr[4] = h_val
        arr[2] = w_val
        arr[5] = w_val


        # arr[-2] = h_val
        # arr[-1] = w_val
def set_stride_dims(layer_param, stride_h, stride_w):
  _hw_prop_setter(layer_param, 'stride_h', 'stride_w', 'stride', stride_h, stride_w)


def set_pad_dims(layer_param, pad_h, pad_w):
  _hw_prop_setter(layer_param, 'pad_h', 'pad_w', 'pad', pad_h, pad_w)


def set_kernel_dims(layer_param, kernel_h, kernel_w):
  _hw_prop_setter(layer_param, 'kernel_h', 'kernel_w', 'kernel_size', kernel_h, kernel_w)


# IMPORTANT: the python protobuf WILL contain
# all the attributes, before getting or setting
# a value, check where the value of the attrobute
# actually comes from - does it come from a repeated
# field or specific fields?
def _hw_prop_getter(layer_param, h_prop, w_prop, a_prop):
  if layer_param.HasField(w_prop):
    w_val = getattr(layer_param, w_prop)
    h_val = getattr(layer_param, h_prop)
    # the default value of int fields
    # will be 0 even if they are not
    # there in the prototxt
    if w_val > 0 or h_val > 0:
      return w_val, h_val
  arr = getattr(layer_param, a_prop)
  if type(arr) == int:
    return arr, arr
  if len(arr) == 0:
    return None
  elif len(arr) == 1:
    return arr[-1], arr[-1]
  else:
    return arr[-2], arr[-1]


def get_stride_dims(layer_param, default=(1, 1)):
  strides = _hw_prop_getter(layer_param, 'stride_h', 'stride_w', 'stride')
  if strides is None:
    strides = (1, 1)
  return SimpleNamespace(stride_w=strides[0], stride_h=strides[1], t=strides)


def get_pad_dims(layer_param, default=(0, 0)):
  pads = _hw_prop_getter(layer_param, 'pad_h', 'pad_w', 'pad')
  if pads is None:
    pads = default
  return SimpleNamespace(pad_w=pads[0], pad_h=pads[1], t=pads)


def get_kernel_dims(layer_param):
  kernel = _hw_prop_getter(layer_param, 'kernel_h', 'kernel_w', 'kernel_size')
  assert kernel is not None
  return SimpleNamespace(kernel_w=kernel[0], kernel_h=kernel[1], t=kernel)

def layerwise_mse(__input_layer_pb, net1, net2):
  # TODO: ADI: enable layerwise mse check in c2c
  
  for idx, _layer in enumerate(__input_layer_pb.layer):
    layer = _layer.name
    if layer not in mutation_observer["meta"]: 
      logger.error("Err key not found in mutation %s", layer)
      continue
    
    if isinstance(mutation_observer["meta"][layer]["shuffled"], list) or mutation_observer["meta"][layer]["shuffled"]!= False:
      logger.info("[M] Shuffled TC %s", layer)
      continue 
    logger.info("[D] Test for %s", layer)
    if '_ara' in layer:
      logger.info("[Z] skipping %s, node introduced by DV found")
      continue
    if layer not in net2.blobs:
      logger.info("[A] node missing. Possible: fused / optimized out, layer %s",layer )
      continue
    d1 = net1.blobs[layer].data[...]
    d2 = net2.blobs[layer].data[...]

    if d1.shape != d2.shape: 
      logger.error("Err Shape mismatch, possible reshapes removed %s, %s, %s", layer, d1.shape, d2.shape)
      continue
    diff = abs(d1 - d2)
    mse = np.square(diff).mean()
    eamax = diff.max()
    epmax = eamax/np.abs(d2).mean()
    if eamax > 0.0001 and epmax > 0.0001: 
      logger.info("Err high mse b/w layers, possible fusion case %s",layer)
    else: 
      logger.info("[S] mse comparison good: %s", layer)
      
def compare_outputs(net1, net2, images=None, dformat=None, input_permuted={}, output_permuted={}, dtype='float'):
  import sys
  sys.path.insert(0, sys.path[0] + '/../../../')
  import utils.io_utils as io_utils
  sys.path.pop(0)

  def _fetch_lname_from_top_name(net, top_name):
    layer2tops = net.top_names
    for layer in layer2tops:
      if top_name in layer2tops[layer]:
        return layer

  assert len(net1.inputs) == len(net2.inputs)
  input_names = net1.inputs
  input_shapes = []
  dformat_ = []
  for idx, input in enumerate(input_names):
    shape = net1.blobs[input].data.shape
    input_shapes.append(','.join([str(i) for i in shape]))
    if (dformat != None and idx < len(dformat) and dformat[idx] != None and dformat[idx] != ''):
      dformat_.append(dformat[idx])
    else:
      dformat_.append("float")
  Args = namedtuple('Args', ['input_shape', 'data_format', 'input_node_name', 'dformat', 'dtype'])
  args = Args(input_shapes, None, input_names, dformat_, dtype)
  if images:
    try:
      feed_dict = io_utils.generate_input_raw(args, images)
    except Exception as e:
      feed_dict = io_utils.generate_input_default(args)
  else:
    feed_dict = io_utils.generate_input_default(args)
  feed_dict = list(feed_dict.values())[0]
  for input in net1.inputs:
    net1_shape = net1.blobs[input].data.shape
    net2_shape = net2.blobs[input].data.shape
    if input in input_permuted:
      net1_shape = tuple([net1_shape[i] for i in input_permuted[input]])
    assert net1_shape == net2_shape
    net1.blobs[input].data[...] = feed_dict[input]
    net2.blobs[input].data[...] = feed_dict[input] if input not in input_permuted else feed_dict[input].transpose(*input_permuted[input])
  net1.forward()
  net2.forward()
  #TODO In case of SSD, out SSCP layers are coming as outputs
  #Very dirty hack , fix should be coming from estimator
  intermediate_outputs = []
  for i in net1.outputs:
    layer = net1.layer_dict[_fetch_lname_from_top_name(net1, i)]
    if not ((layer.type == 'Reshape' or layer.type == 'Identity') and i.endswith('_splitOutput')):
      intermediate_outputs.append(i)
  if len(intermediate_outputs) != len(net2.outputs):
    logger.warn("num of outputs differ b/w src and dst networks. please check src: %d, dst: %d. src_list: %s, dst_list: %s", len(intermediate_outputs), len(net2.outputs), intermediate_outputs, net2.outputs)
  counter = 0

  for out1 in intermediate_outputs:
    out1_type = net1.layer_dict[_fetch_lname_from_top_name(net1, out1)].type
    out1_data = net1.blobs[out1].data
    for out2 in net2.outputs:
      out2_type = net2.layer_dict[_fetch_lname_from_top_name(net2, out2)].type
      out2_data = net2.blobs[out2].data
      if (out1_data.flatten().shape == out2_data.flatten().shape) or (out1_type == out2_type and out1_type == 'ArgMax'):
        if (np.isnan(np.sum(out1_data.flatten())) or np.isnan(np.sum(out2_data.flatten()))) and (np.any(np.isnan(out1_data.flatten()) != np.isnan(out2_data.flatten()))):  #Hyperbolic Ops
          logger.info("Error: np.NaN for a non np.NaN value present in data")
          diff = np.ones(out1_data.flatten().shape)
          EMAX = 1
          E_PAMX = 1
        else:
          if out2 in output_permuted:
            out2_data = out2_data.transpose(*output_permuted[out2])
          net1_data = np.nan_to_num(out1_data)
          net2_data = np.nan_to_num(out2_data)
          diff = net1_data.flatten() - net2_data.flatten()
          try:
            EMAX = abs(diff).max()
            E_PAMX = abs(diff).max() / np.abs(net1_data).mean()
          except Exception as e:
            logger.warn(str(e))
        if 0 not in diff.shape:
          logger.info("layer:: %s(original_net) vs %s(dv_net), EMAX = %s, E_PAMX = %s", out1, out2, EMAX, E_PAMX)
        if (0 in diff.shape) or (abs(diff).max() < 1e-4 or abs(diff).max() / np.abs(out1_data).mean() < 1e-4):
          counter = counter + 1  #unusually fail is not good. Shape 0 in detection layer

  if counter >= len(intermediate_outputs):
    logger.info("c2c conversion ok!")
    return 0
  else:
    logger.error("c2c conversion failed")
    return -1


def standardize_strides(layer_param, shape):
  if layer_param.HasField('stride_h') and len(layer_param.stride):
    logger.error("both stride fields stride and stride_w occur together for layer %s", layer.name)
  if layer_param.HasField('stride_h'):
    layer_param.stride.append(layer_param.stride_h)
    layer_param.ClearField('stride_h')
  if layer_param.HasField('stride_w'):
    layer_param.stride.append(layer_param.stride_w)
    layer_param.ClearField('stride_w')
  if layer_param.HasField('stride_d'):
    layer_param.stride.append(layer_param.stride_d)
    layer_param.ClearField('stride_d')


def standardize_kernel_size(layer_param, shape):
  if layer_param.HasField('kernel_h') and len(layer_param.kernel_size):
    logger.error("both kernel fields kernel_size and kernel occur together for layer %s", layer.name)
  if layer_param.HasField('kernel_h'):
    layer_param.kernel_size.append(layer_param.kernel_h)
    layer_param.ClearField('kernel_h')
  if layer_param.HasField('kernel_w'):
    layer_param.kernel_size.append(layer_param.kernel_w)
    layer_param.ClearField('kernel_w')
  if layer_param.HasField('kernel_d'):
    layer_param.kernel_size.append(layer_param.kernel_d)
    layer_param.ClearField('kernel_d')


def standardize_padding(layer_param, shape):
  if layer_param.HasField('pad_w'):
    layer_param.pad.append(layer_param.pad_h)
    layer_param.pad.append(layer_param.pad_w)
    layer_param.pad.append(layer_param.pad_h)
    layer_param.pad.append(layer_param.pad_w)
    if hasattr(layer_param, 'pad_l') and layer_param.HasField('pad_l'):
      logger.error("both pad fields pad_l and pad_w occur together for layer %s", layer.name)
  if hasattr(layer_param, 'pad_l') and layer_param.HasField('pad_l'):
    if len(layer_param.pad) > 0:
      logger.error("pad field present with pad_l")
      raise ValueError()
    layer_param.pad.append(layer_param.pad_t)
    layer_param.pad.append(layer_param.pad_l)
    layer_param.pad.append(layer_param.pad_b)
    layer_param.pad.append(layer_param.pad_r)
    layer_param.ClearField('pad_t')
    layer_param.ClearField('pad_l')
    layer_param.ClearField('pad_b')
    layer_param.ClearField('pad_r')
  layer_param.ClearField('pad_w')
  layer_param.ClearField('pad_h')
  if len(layer_param.pad) == 1:
    layer_param.pad.extend(list(layer_param.pad) * ((len(shape) - 2) * 2 - 1))
  elif len(layer_param.pad) != (len(shape) - 2) * 2:
    layer_param.pad.extend(list(layer_param.pad))


def standardize_fields(layer_param, shape):
  standardize_padding(layer_param, shape)
  standardize_strides(layer_param, shape)
  standardize_kernel_size(layer_param, shape)


def _hw_prop_getter_test(layer_param, a_prop):
  arr = getattr(layer_param, a_prop)
  if len(arr) == 0:
    if a_prop == 'pad':
      return [0]
    elif a_prop == 'stride':
      return [1]
    elif a_prop == 'dilation':
      return [1]
    return None
  if type(arr) == int:
    return [arr]
  return arr


def get_group_dim(layer_param):
  return getattr(layer_param, "group")


def get_kernel_dim(layer_param):
  return _hw_prop_getter_test(layer_param, 'kernel_size')


def get_pad_dim(layer_param):
  return _hw_prop_getter_test(layer_param, 'pad')


def get_stride_dim(layer_param):
  return _hw_prop_getter_test(layer_param, 'stride')


def get_dilation(layer_param):
  return _hw_prop_getter_test(layer_param, 'dilation')


def get_num_output(layer_param):
  return getattr(layer_param, "num_output")


def get_bias_term(layer_param):
  return getattr(layer_param, 'bias_term')


def get_axis(layer_param):
  return getattr(layer_param, 'axis')


def set_pad_dim(layer_param, pad_list):
  layer_param.ClearField('pad')
  layer_param.pad.extend(pad_list)


def set_kernel_dim(layer_param, kernels):
  layer_param.ClearField('kernel_size')
  layer_param.kernel_size.extend(kernels)


def set_stride_dim(layer_param, stride):
  layer_param.ClearField('stride')
  layer_param.stride.extend(stride)
