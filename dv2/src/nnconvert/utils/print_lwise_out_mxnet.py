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

import sys
import os
sys.path.append(os.getcwd() + '/' + os.path.dirname(__file__) + '/../../caffe/python')
sys.path.append(os.getcwd() + '/' + os.path.dirname(__file__) + '/../')
import mxnet as mx
import mxnet.gluon as gluon
from collections import namedtuple
import caffe
import numpy as np
import argparse
from mxnet import image
from mxnet.gluon.utils import download
from mxnet import nd


def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('true', '1'):
    return True
  elif v.lower() in ('false', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def transform(data):
  data = data.transpose((2, 0, 1)).expand_dims(axis=0)
  rgb_mean = nd.array([0.485, 0.456, 0.406]).reshape((1, 3, 1, 1))
  rgb_std = nd.array([0.229, 0.224, 0.225]).reshape((1, 3, 1, 1))
  return (data.astype('float32') / 255 - rgb_mean) / rgb_std


parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument('-p', "--prototxt", help='Input Caffe Prototxt Path', required=True)
requiredNamed.add_argument('-cm', "--caffemodel", help='Input Caffe Model Path', required=True)
requiredNamed.add_argument('-m', "--mxInputPrefix", help='Input prefix Mxnet Model Path', required=True)
requiredNamed.add_argument('-o', '--outputFile', help='Output Stats CSV File', required=True)
requiredNamed.add_argument('-ch', '--channels', help='channels', required=True)
requiredNamed.add_argument('-hi', '--height', help='height', required=True)
requiredNamed.add_argument('-wi', '--width', help='width', required=True)
requiredNamed.add_argument('-label', '--label', help='label name', required=True)
requiredNamed.add_argument('-purpose', '--purpose', help='classification[0]/detection[1]/segmentation[2]', required=True)
requiredNamed.add_argument('-if', '--inputFile', help='input image file', required=True)
requiredNamed.add_argument('-lf', '--labelFile', help='label file', required=True)
requiredNamed.add_argument('-set_np_shape', '--set_np_shape', type=str2bool, help='enable/disable numpy shape semantics', required=True)
args = parser.parse_args()
protoFile = args.prototxt
caffemodel = args.caffemodel
outFile = args.outputFile
channels = int(args.channels)
height = int(args.height)
width = int(args.width)
label = args.label
mxInputPrefix = args.mxInputPrefix
purpose = int(args.purpose)
imageFile = args.inputFile
labelFile = args.labelFile
if args.set_np_shape == True:
  mx.set_np_shape(True)
with open(labelFile, 'r') as f:
  text_labels = [' '.join(l.split()[1:]) for l in f]
data = image.imread(imageFile)
data = image.resize_short(data, height)
data, _ = image.center_crop(data, (width, height))
data = transform(data)
caffe_net = caffe.Net(protoFile, caffemodel, caffe.TEST)
caffe_net.blobs['data'].data[...] = data.asnumpy()
caffe_out = caffe_net.forward()
sym, arg_params, aux_params = mx.model.load_checkpoint(mxInputPrefix, 0)
Batch = namedtuple('Batch', ['data'])
all_layers = sym.get_internals()
import json
import csv
stats = [[
    'layerName', 'type', 'caffe_mean', 'mx_mean', 'caffe_std', 'mx_std', 'max_diff', 'min_diff', 'max_pdiff', 'min_pdiff', 'caffe_diff_max', 'mx_diff_max', 'caffe_diff_min',
    'mx_diff_min', 'caffe_pdiff_max', 'mx_pdiff_max', 'caffe_pdiff_min', 'mx_pdiff_min', 'diff_mean', 'diff_std'
]]
all_layers_length = len(all_layers)
layer_wise = input("Print every layer stat[yes(1)/no(0)]:")
if layer_wise == '1':
  csvFile = open(outFile, 'w')
  writer = csv.writer(csvFile)
  writer.writerow(stats[0])
  for i in range(0, all_layers_length):
    layer_dict = json.loads(all_layers[i].tojson())
    if layer_dict['nodes'][len(layer_dict['nodes']) - 1]['op'] != "null":
      print(layer_dict['nodes'][len(layer_dict['nodes']) - 1]['op'])
      caffe_layer_out = None
      if layer_dict['nodes'][len(layer_dict['nodes']) - 1]['op'] != 'Activation':  #and layer_dict['nodes'][len(layer_dict['nodes'])-1]['attrs']['act_type'] != 'relu':
        if layer_dict['nodes'][len(layer_dict['nodes']) - 1]['op'] != 'elemwise_add':
          if layer_dict['nodes'][len(layer_dict['nodes']) - 1]['op'] == 'Deconvolution':
            caffe_layer_out = (caffe_net.blobs[layer_dict['nodes'][len(layer_dict['nodes']) - 1]['name'] + '_padding'].data)
          else:
            caffe_layer_out = (caffe_net.blobs[layer_dict['nodes'][len(layer_dict['nodes']) - 1]['name']].data)
        else:
          caffe_layer_out = (caffe_net.blobs[layer_dict['nodes'][len(layer_dict['nodes']) - 1]['name'][1:]].data)
          #below is a hack
      if layer_dict['nodes'][len(layer_dict['nodes']) - 1]['name'] == 'pooling0':
        caffe_layer_out = (caffe_net.blobs[layer_dict['nodes'][len(layer_dict['nodes']) - 1]['name'] + '_crop'].data)
      sym3 = all_layers[i]
      print(all_layers[i])
      mod3 = mx.mod.Module(symbol=sym3, label_names=(label,), context=mx.cpu())
      mod3.bind(for_training=False, data_shapes=[('data', (1, channels, height, width))])
      mod3.set_params(arg_params, aux_params)
      mx_data = Batch([mx.nd.array(data)])
      mod3.forward(mx_data)
      mx_layer_out = mod3.get_outputs()[0].asnumpy()
      assert caffe_layer_out is not None
      diff = caffe_layer_out - mx_layer_out
      div = diff / mx_layer_out
      div[np.isnan(div)] = 0
      shape = caffe_layer_out.shape
      index_diff_max = np.unravel_index(np.argmax(diff, axis=None), shape)
      index_diff_min = np.unravel_index(np.argmin(diff, axis=None), shape)
      index_div_max = np.unravel_index(np.argmax(div, axis=None), shape)
      index_div_min = np.unravel_index(np.argmin(div, axis=None), shape)
      stats_i = [
          layer_dict['nodes'][len(layer_dict['nodes']) - 1]['name'], layer_dict['nodes'][len(layer_dict['nodes']) - 1]['op'],
          np.mean(caffe_layer_out),
          np.mean(mx_layer_out),
          np.std(caffe_layer_out),
          np.std(mx_layer_out),
          diff.max(),
          diff.min(),
          div.max(),
          div.min(), caffe_layer_out[index_diff_max], mx_layer_out[index_diff_max], caffe_layer_out[index_diff_min], mx_layer_out[index_diff_min], caffe_layer_out[index_div_max],
          mx_layer_out[index_div_max], caffe_layer_out[index_div_min], mx_layer_out[index_div_min],
          abs(diff).mean(),
          abs(diff).std()
      ]
      print(stats_i)
      writer.writerow(stats_i)
      csvFile.flush()
else:
  sym3 = all_layers[len(all_layers) - 1]
  mod3 = mx.mod.Module(symbol=sym3, label_names=(label,), context=mx.cpu())
  mod3.bind(for_training=False, data_shapes=[('data', (1, channels, height, width))])
  mod3.set_params(arg_params, aux_params)
  mx_data = Batch([mx.nd.array(data)])
  mod3.forward(mx_data)
  mx_final_out = mod3.get_outputs()[0].asnumpy()
  caffe_final_out = caffe_out[next(iter(caffe_out))]  # prints the output of caffe network complete run
  diff = abs(mx_final_out - caffe_final_out)
  print("mx_final_output:", mx_final_out)
  print("caffe_final_output:", caffe_final_out)
  print(diff.mean())
  prob = mx_final_out
  if purpose == 0:
    idx = prob.argsort()[0][-5:]
    for i in reversed(idx):
      print('MXNET:With confidence = %.5f, it contains %s' % (mx_final_out[0][i], text_labels[i]))
    print()
    prob = caffe_final_out
    idx = prob.argsort()[0][-5:]
    for i in reversed(idx):
      #i = int(i.asscalar())
      print('CAFFE:With confidence = %.5f, it contains %s' % (caffe_final_out[0][i], text_labels[i]))
  elif purpose == 2:
    mx_argmax = mx_final_out[0].argmax(axis=0)
    caffe_argmax = caffe_final_out[0].argmax(axis=0)
    print("mx segmentation:", mx_argmax)
    print("caffe segmentation:", caffe_argmax)
    print(abs(mx_argmax - caffe_argmax).max())
  else:
    print("purpose not supported")
