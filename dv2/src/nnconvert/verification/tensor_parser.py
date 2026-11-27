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

import glob as glob
import tensorflow as tf
import cv2
from numpy import array
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import sys, argparse
import numpy as np
import os

''' Returns the prototxt file list
	Args:
		files_path: Prototxt directory path
	Returns:
		List of file names
'''


def get_data(files_path):
  prototxt_file_list = []
  files_regex = files_path
  if files_path[-1] == '/':
    files_regex = files_path + '*.prototxt'
  else:
    files_regex = files_path + '/*.prototxt'
  for name in glob.glob(files_regex):
    prototxt_file_list.append(name[name.rfind('/') + 1:])
    #print name
  return prototxt_file_list


''' Matches the padding between Tensorflow and Caffe
	Args:
		tf_pad_type: string representing the Tensorflow padding type
		caffe_pad_h: Caffe pad height
		caffe_pad_w: Caffe pad width
'''


def pad_check(tf_pad_type, caffe_pad_h, caffe_pad_w):
  if tf_pad_type.s == "VALID":
    assert caffe_pad_h == 0 and caffe_pad_w == 0


''' Matching a convolutional layer parameters
	Args:
		node: Tensorflow graph node
		net: Caffe net object
		prev_node_shape: Previous Tensorflow node shape
		cur_node_shape: Current Tensorflow node shape
		filter_shape: Convolutional filter shape
'''


def Conv2D(node, net, prev_node_shape, cur_node_shape, filter_shape):
  assert node != None and net != None
  conv_node = node.name.replace("/", "_")
  # print prev_node_shape
  # print cur_node_shape
  tf_pad_type = node.attr['padding']
  data_format = node.attr['data_format'].s
  #print data_format
  if str(data_format) == "NHWC":
    tf_stride = node.attr['strides'].list.i[1]
  else:
    tf_stride = node.attr['strides'].list.i[2]
  #print "Stride list"
  #print stride.list.i[1]
  dilation = node.attr['dilations']
  #print conv_node
  #print stride
  #print dilation
  assert (dilation.list[i] == 1 for i in range(4))
  for i in range(len(net.layer)):
    if net.layer[i].name == conv_node:
      #print net.layer[i]
      caffe_stride = net.layer[i].convolution_param.stride[0]
      caffe_kernel_h = net.layer[i].convolution_param.kernel_h
      caffe_kernel_w = net.layer[i].convolution_param.kernel_w
      caffe_pad_h = net.layer[i].convolution_param.pad_h
      caffe_pad_w = net.layer[i].convolution_param.pad_w
      #print caffe_kernel_h
      #print filter_shape
      pad_check(tf_pad_type, caffe_pad_h, caffe_pad_w)
      # Match Stride
      assert caffe_stride == tf_stride
      # Match convolution kernel size
      assert filter_shape[0] == caffe_kernel_h and filter_shape[1] == caffe_kernel_w
      #assert caffe_kernel_size
      # Match the number of output channels
      assert cur_node_shape[3] == net.layer[i].convolution_param.num_output
    elif net.layer[i].type == 'DummyData':
      #print net.layer[i].dummy_data_param.shape[0].dim[0]
      caffe_output_h = net.layer[i].dummy_data_param.shape[0].dim[2]
      caffe_output_w = net.layer[i].dummy_data_param.shape[0].dim[3]
      # print "DummyData: " + str(caffe_output_h) + " " + str(caffe_output_w)
      # print "Current node: " + str(cur_node_shape)
      #print data_format
      #print str(data_format.s)=="NHWC"
      # Matching output dimensions from this convolutional layer
      if str(data_format) == "NHWC":
        assert caffe_output_h == cur_node_shape[1] and caffe_output_w == cur_node_shape[2]
      else:
        assert caffe_output_h == cur_node_shape[2] and caffe_output_w == cur_node_shape[3]


''' Matching a convolutional layer parameters
	Args:
		node: Tensorflow graph node
		net: Caffe net object
		prev_node_shape: Previous Tensorflow node shape
		cur_node_shape: Current Tensorflow node shape
		filter_shape: Convolutional filter shape
'''


def DepthwiseConv2dNative(node, net, prev_node_shape, cur_node_shape, filter_shape):
  assert node != None and net != None
  conv_node = node.name.replace("/", "_")
  # print prev_node_shape
  # print cur_node_shape
  tf_pad_type = node.attr['padding']
  data_format = node.attr['data_format'].s
  #print data_format
  if data_format == "NHWC":
    tf_stride = node.attr['strides'].list.i[1]
  else:
    tf_stride = node.attr['strides'].list.i[2]
  #print "Stride list"
  #print stride.list.i[1]
  dilation = node.attr['dilations']
  #print conv_node
  #print stride
  #print dilation
  assert (dilation.list[i] == 1 for i in range(4))
  for i in range(len(net.layer)):
    if net.layer[i].name == conv_node:
      # print net.layer[i]
      caffe_stride = net.layer[i].convolution_param.stride[0]
      caffe_kernel_h = net.layer[i].convolution_param.kernel_h
      caffe_kernel_w = net.layer[i].convolution_param.kernel_w
      caffe_pad_h = net.layer[i].convolution_param.pad_h
      caffe_pad_w = net.layer[i].convolution_param.pad_w
      #print caffe_kernel_h
      #print filter_shape
      pad_check(tf_pad_type, caffe_pad_h, caffe_pad_w)
      # Match Stride
      assert caffe_stride == tf_stride
      # Match convolution kernel size
      assert filter_shape[0] == caffe_kernel_h and filter_shape[1] == caffe_kernel_w
      #assert caffe_kernel_size
      # Match the number of output channels
      assert cur_node_shape[3] == net.layer[i].convolution_param.num_output
      # Need to take dilations and group convolutions into account
    elif net.layer[i].type == 'DummyData':
      caffe_output_h = net.layer[i].dummy_data_param.shape[0].dim[2]
      caffe_output_w = net.layer[i].dummy_data_param.shape[0].dim[3]
      # print "DummyData: " + str(caffe_output_h) + " " + str(caffe_output_w)
      # print "Current node: " + str(cur_node_shape)
      #print data_format
      #print str(data_format.s)=="NHWC"
      # print cur_node_shape[2]
      # print cur_node_shape[3]
      # Matching output dimensions from this convolutional layer
      #print "Current node shape: "+str(cur_node_shape
      if str(data_format) == "NHWC":
        assert caffe_output_h == cur_node_shape[1] and caffe_output_w == cur_node_shape[2]
      else:
        assert caffe_output_h == cur_node_shape[2] and caffe_output_w == cur_node_shape[3]


''' Matching a Max pool layer parameters
	Args:
		node: Tensorflow graph node
		net: Caffe net object
'''


def MaxPool(node, net):
  assert node != None and net != None
  tf_pad_type = node.attr['padding']
  tf_ksize = node.attr['ksize'].list.i[2]
  tf_stride = node.attr['strides'].list.i[1]
  for i in range(len(net.layer)):
    caffe_stride = net.layer[i].pooling_param.stride
    caffe_kernel_size = net.layer[i].pooling_param.kernel_size
    caffe_pad_h = net.layer[i].convolution_param.pad_h
    caffe_pad_w = net.layer[i].convolution_param.pad_w
    pad_check(tf_pad_type, caffe_pad_h, caffe_pad_w)
    # Match Stride
    assert caffe_stride == tf_stride
    # Match convolution kernel size
    assert tf_ksize == caffe_kernel_size
    #assert caffe_kernel_size


''' Load the Tensorflow model
	Args:
		frozen_graph_path: Path to Tensorflow frozen graph
		files_path: Path to generated caffe prototxt files
'''


def load_graph(frozen_graph_path, files_path):
  restored_graph_def = None
  with tf.gfile.GFile(frozen_graph_path, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())
  input_node = None
  # Get the normalized absolute file path
  files_path = os.path.abspath(files_path)
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(restored_graph_def, name="")
    with tf.Session() as sess:
      '''
			for op in graph.get_operations():
				print(op.name)
			'''
      all_ops = [node for node in restored_graph_def.node]
      #files_dir = '/home/deepvision/caffe_tut/open_converter/mobilenet_v1/prototxt/'
      prototxt_files = get_data(files_path)
      # print prototxt_files
      # print str(len(prototxt_files)) + ' : files'
      prev_node_shape = (1, 224, 224, 3)
      cnt = 0
      input_layer_shape = None
      for node in restored_graph_def.node:
        conv_node = str(node.name.replace("/", "_")) + '.prototxt'
        if conv_node in prototxt_files:
          cnt += 1
      assert cnt == len(prototxt_files)
      ind = 0
      for node in restored_graph_def.node:
        ind += 1
        conv_node = str(node.name.replace("/", "_")) + '.prototxt'
        if 'value' in node.attr and len(tf.make_ndarray(node.attr['value'].tensor).shape) == 4:
          input_layer_shape = tf.make_ndarray(node.attr['value'].tensor).shape
          # print node.name + ' : ' + str(input_layer_shape)
        if node.op not in ['Const', 'Identity'] and conv_node in prototxt_files:
          #continue
          # print conv_node
          layer_type = node.op
          prototxt_path = files_path + '/' + conv_node
          net = caffe_pb2.NetParameter()
          with open(prototxt_path) as f:
            content = f.read()
            txtf.Merge(content, net)
          cur_node_shape = graph.get_tensor_by_name(node.name + ':0').shape
          if layer_type == 'Conv2D':
            #weights_node_shape = graph.get_tensor_by_name(node.name[0:node.name.rfind('/')]+'/weights/read:0').shape
            #print weights_node_shap
            #print 'Next layer: '+all_ops[ind+1]
            Conv2D(node, net, prev_node_shape, cur_node_shape, input_layer_shape)
          elif layer_type == 'DepthwiseConv2dNative':
            DepthwiseConv2dNative(node, net, prev_node_shape, cur_node_shape, input_layer_shape)
          elif layer_type == 'MaxPool':
            MaxPool(node, net)
          #print prev_node_shape
          #print graph.get_tensor_by_name(node.name+':0').shape
          prev_node_shape = cur_node_shape
      # print cnt


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='To compare the layer parameters of each layer')
  parser.add_argument('--frozen_graph_path', type=str, help='specify tensorflow frozen pb path')
  parser.add_argument('--files_path', type=str, help='specify path to the directory containing the generated prototxt files corresponding to each Tensorflow layer')
  #parser.add_argument('--output_dir', type=str, help='specify output directory for the prototxt files')
  args = parser.parse_args()
  load_graph(str(args.frozen_graph_path), str(args.files_path))
