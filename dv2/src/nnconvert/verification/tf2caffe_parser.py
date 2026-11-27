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

''' Slices the generated MMdnn prototxt to produce the Tensorflow - Caffe layer mapping
	Args:
		frozen_graph_path: Path to the Tensorflow frozen graph
		prototxt_path: Path to the MMdnn prototxt
'''

def generate_prototxt(frozen_graph_path, prototxt_path):
  input_node_names = []
  restored_graph_def = None
  x = tf.placeholder(tf.float32)
  with tf.gfile.GFile(frozen_graph_path, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())
  net = caffe_pb2.NetParameter()
  caffe_layers = []
  caffe_layers_map = dict()
  rev_caffe_layers_map = dict()
  n_dummy = 0
  n_crop = 0
  n_scale = 0
  with open(prototxt_path) as f:
    content = f.read()
    txtf.Merge(content, net)
    #print dir(net)
    for i in range(0, len(net.layer)):
      #print net.layer[i]
      if net.layer[i].type == 'DummyData':
        n_dummy += 1
      if net.layer[i].type == 'Crop':
        n_crop += 1
      if net.layer[i].type == 'Scale':
        n_scale += 1
      caffe_layers.append(net.layer[i])
      caffe_layers_map[str(net.layer[i].name)] = i
      rev_caffe_layers_map[i] = str(net.layer[i].name)
  #print caffe_layers_map
  tensorflow_layers = []
  for node in restored_graph_def.node:
    converted_name = node.name.replace("/", "_")
    #converted_name.replace("/","_")
    #print converted_name
    if converted_name in caffe_layers_map:
      tensorflow_layers.append(node.name)
  # print len(tensorflow_layers)
  # print len(caffe_layers_map)
  # print n_dummy
  # print n_crop
  # print n_scale
  n_mapped = 0
  tf2caffe_map = dict()
  output_path = 'prototxt'
  try:
    import os
    os.makedirs(os.path.abspath(output_path))
  except:
    #print "Couldn't make directory: "+output_dir
    print("Reverting to defaults")
    #output_path='/prototxt'
  for ind in range(len(tensorflow_layers)):
    copy_list = []
    layer_prototxt = caffe_pb2.NetParameter()
    conv_tf_l = tensorflow_layers[ind].replace("/", "_")
    low = caffe_layers_map[conv_tf_l]
    high = len(caffe_layers_map)
    tf2caffe_map[conv_tf_l] = []
    if ind == len(tensorflow_layers) - 1:
      n_mapped += len(caffe_layers_map) - caffe_layers_map[conv_tf_l]
    else:
      conv_tf_next_l = tensorflow_layers[ind + 1].replace("/", "_")
      n_mapped += caffe_layers_map[conv_tf_next_l] - caffe_layers_map[conv_tf_l]
      high = caffe_layers_map[conv_tf_next_l]
    for j in range(low, high):
      tf2caffe_map[conv_tf_l].append(rev_caffe_layers_map[j])
      copy_list.append(caffe_layers[j])
    layer_prototxt.layer.extend(copy_list)
    layer_prototxt_writer = open(output_path + '/' + conv_tf_l + '.prototxt', 'w')
    layer_prototxt_writer.write(str(layer_prototxt))
    layer_prototxt_writer.close()
  # print n_mapped
  #print tf2caffe_map


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Generate prototxt files for each tensorflow layer')
  parser.add_argument('--frozen_graph_path', type=str, help='specify tensorflow frozen pb path')
  parser.add_argument('--prototxt_path', type=str, help='specify path to the converted prototxt file obtained using MMdnn')
  #parser.add_argument('--output_dir', type=str, help='specify output directory for the prototxt files')
  args = parser.parse_args()
  generate_prototxt(str(args.frozen_graph_path), str(args.prototxt_path))
