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

from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf
import tensorflow as tf
import sys, argparse
import caffe
import collections
import numpy as np
import csv

''' Compare the parameter values of the corresponding layers in Tensorflow and Caffe
	Args:
		caffe_layer_val: Numpy array representing the Caffe layer blob
		tf_layer_val: Numpy array representing the Tensorflow layer tensor value
		writer: CSV writer object
		layer: Current layer name
'''


def compare_values(caffe_layer_val, tf_layer_val, writer, layer):
  if not np.array_equal(tf_layer_val, caffe_layer_val):
    #print "Std of caffe layer: "+ str(np.std(caffe_layer_val))
    #print "Std of Tensorflow layer: "+str(np.std(tf_layer_val))
    #print "Mean of caffe layer: "+ str(np.mean(caffe_layer_val))
    #print "Mean of Tensorflow layer: "+ str(np.mean(tf_layer_val))
    error_layer_val = tf_layer_val - caffe_layer_val
    writer.writerow({
        'Layer name': layer,
        'tf_mean': np.mean(tf_layer_val),
        'caffe_mean': np.mean(caffe_layer_val),
        'tf_std': np.std(tf_layer_val),
        'caffe_std': np.std(caffe_layer_val),
        'tf_min': np.min(tf_layer_val),
        'caffe_min': np.min(caffe_layer_val),
        'tf_max': np.max(tf_layer_val),
        'caffe_max': np.max(caffe_layer_val),
        'tf_shape': tf_layer_val.shape,
        'caffe_layer_shape': caffe_layer_val.shape,
        'error_mean': np.mean(error_layer_val),
        'error_std': np.std(error_layer_val),
        'error_min': np.min(error_layer_val),
        'error_max': np.max(error_layer_val)
    })
    assert np.isclose(np.std(caffe_layer_val), np.std(tf_layer_val), rtol=10e-7) and np.isclose(np.mean(caffe_layer_val), np.mean(tf_layer_val), rtol=10e-7)


''' Matches the appropriate Tensorflow and Caffe layer values
	Args: 
		net: Caffe net object
		tf_layer_weights: Dict representing the Tensorflow layer weights (Picks up all inputs for each Tensorflow operation)
		cur_node: Current Tensorflow node object
		writer: CSV Writer object
'''


def compare(net, tf_layer_weights, cur_node, writer):
  # print "In compare:"
  # print cur_node.op
  conv_name = cur_node.name.replace("/", "_")
  if cur_node.op == 'FusedBatchNorm':
    if len(tf_layer_weights) == 4:
      batch_norm = net.params[conv_name]
      scale = net.params[conv_name + '_scale']
      #			print "Caffe batch_norm moving mean std: "+str(np.std(batch_norm[0].data))
      #			print "Caffe batch_norm moving variance std: "+str(np.std(batch_norm[1].data))
      #			print "Caffe scale gamma: "+str(np.std(scale[0].data))
      #			print "Caffe scale beta: "+str(np.std(scale[1].data))
      #			print "Tensorflow moving mean std: "+str(np.std(tf_layer_weights.values()[2]))
      #			print "Tensorflow moving variance std: "+str(np.std(tf_layer_weights.values()[3]))
      #			print "Tensorflow gamma: "+str(np.std(tf_layer_weights.values()[0]))
      #			print "Tensorflow beta: "+str(np.std(tf_layer_weights.values()[1]))
      for i in range(len(tf_layer_weights)):
        tf_layer_val = tf_layer_weights.values()[i]
        caffe_layer_val = net.params[conv_name][0].data
        if i > 1:
          #print str(tf_layer_val.shape)+' : '+str(scale[i/2].data.shape)+str(np.array_equal(tf_layer_val,scale[i/2].data))
          caffe_layer_val = batch_norm[i % 2].data
        else:
          #print str(tf_layer_val.shape)+' : '+str(batch_norm[i/2].data.shape)+str(np.array_equal(tf_layer_val, batch_norm[i/2].data))
          caffe_layer_val = scale[i % 2].data
#				print tf_layer_weights.keys()[i]
#				print "Caffe layer shape: "+str(caffe_layer_val.shape)
#				print "Tensorflow layer shape: "+str(tf_layer_val.shape)
#				print np.array_equal(tf_layer_val, caffe_layer_val)
        compare_values(caffe_layer_val, tf_layer_val, writer, cur_node.op)
  elif cur_node.op == 'Conv2D':
    for i in range(len(tf_layer_weights)):
      caffe_layer_val = net.params[conv_name][i].data
      tf_layer_val = tf_layer_weights.values()[i]
      if len(tf_layer_val.shape) == 4:
        tf_layer_val = tf_layer_val.transpose(3, 2, 0, 1)
      # print "Caffe layer shape: " + str(caffe_layer_val.shape)
      # print "Tensorflow layer shape: " + str(tf_layer_val.shape)
      #			print np.array_equal(tf_layer_val,caffe_layer_val)
      compare_values(caffe_layer_val, tf_layer_val, writer, cur_node.op)
  elif cur_node.op == 'DepthwiseConv2dNative':
    for i in range(len(tf_layer_weights)):
      caffe_layer_val = net.params[conv_name][i].data
      tf_layer_val = tf_layer_weights.values()[i]
      if len(tf_layer_val.shape) == 4:
        tf_layer_val = tf_layer_val.transpose(3, 2, 0, 1)
#			print "Caffe layer shape: "+str(caffe_layer_val.shape)
#			print "Tensorflow layer shape: "+str(tf_layer_val.shape)
#			print np.array_equal(tf_layer_val,caffe_layer_val)
      compare_values(caffe_layer_val, tf_layer_val, writer, cur_node.op)
  #elif cur_node.op=='MaxPool':


#		for i in range(len(tf_layer_weights)):
#			caffe_layer_val = net.params[conv_name][i].data
#			tf_layer_val= tf_layer_weights.values()[i]
#			if len(tf_layer_val)==4:
#				tf_layer_val = tf_layer_val.transpose(3,2,0,1)#
##			print "Caffe layer shape: "+str(caffe_layer_val.shape)
##			print "Tensorflow layer shape: "+str(tf_layer_val.shape)
##			print np.array_equal(tf_layer_val,caffe_layer_val)
#			compare_values(caffe_layer_val, tf_layer_val)
  # print "Exit"
''' Picks up and matches the appropriate Caffe layers for the corresponding Tensorflow layers
	Args:
		frozen_graph_path: Path to Tensorflow frozen graph
		prototxt_path: Path to the MMdnn prototxt
		model_path: Path to MMdnn caffemodel
'''


def compare_param_values(frozen_graph_path, prototxt_path, model_path):
  fieldnames = [
      'Layer name', 'tf_mean', 'caffe_mean', 'tf_std', 'caffe_std', 'tf_min', 'caffe_min', 'tf_max', 'caffe_max', 'tf_shape', 'caffe_layer_shape', 'error_mean', 'error_std',
      'error_min', 'error_max'
  ]
  csv_file = open('parameter_value_comparison.csv', 'w')
  writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
  writer.writeheader()
  net = caffe.Net(prototxt_path, model_path, caffe.TEST)
  #print net.params.keys()
  restored_graph_def = None
  with tf.gfile.GFile(frozen_graph_path, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())
  input_node = None
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(restored_graph_def, name="")
    with tf.Session() as sess:
      tf_layer_weights = collections.OrderedDict()
      for node in restored_graph_def.node:
        #print(node.name)
        if 'value' in node.attr:
          array = tf.make_ndarray(node.attr['value'].tensor)
          tf_layer_weights[node.name] = array
          #print array.shape
          #print array
        if node.op not in ['Const', 'Identity']:
          #print layer_weights
          if len(tf_layer_weights) > 0:
            compare(net, tf_layer_weights, node, writer)
            tf_layer_weights.clear()
  csv_file.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='To compare the layer parameter values and outputs from each layer')
  parser.add_argument('--frozen_graph_path', type=str, help='specify tensorflow frozen pb path')
  parser.add_argument('--prototxt_path', type=str, help='specify path to the converted prototxt file obtained using MMdnn')
  parser.add_argument('--model_path', type=str, help='specify path to the converted caffemodel obtained using MMdnn')
  #parser.add_argument('--output_dir', type=str, help='specify output directory for the prototxt files')
  args = parser.parse_args()
  compare_param_values(str(args.frozen_graph_path), str(args.prototxt_path), str(args.model_path))
#compare_param_values(frozen_graph_path)
