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
import glob as glob
import tensorflow as tf
import sys, argparse
import cv2
from numpy import array
import collections
import numpy as np
import caffe
import os
import csv

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
  return prototxt_file_list


''' Load the caffe model
	Args:
		prototxt_path: Path to the MMdnn prototxt
		model_path: Path to the MMdnn caffemodel
		img: OpenCV image
		img_dim: Dimensions of the image: (H,W,C) format
	Returns:
		Net caffe object
'''


def load_caffe_graph(prototxt_path, model_path, img, img_dim):
  net = caffe.Net(prototxt_path, model_path, caffe.TEST)
  input_layer = list(net.blobs.keys())[0]
  reshape_dim = tuple([1]) + tuple(img_dim)
  new_img = array(img).reshape(reshape_dim)
  net.blobs[input_layer].data[...] = new_img.transpose(0, 3, 1, 2)
  net.forward()
  return net


''' Load the Tensorflow model
	Args:
		frozen_graph_path: Path to Tensorflow frozen graph
		img: OpenCV image
		img_dim: Dimensions of the image: (H,W,C) format
	Returns:
		A dict with tensorflow layer values
'''


def load_tf_graph(frozen_graph_path, img, img_dim):
  #net = caffe.Net(prototxt_path, model_path, caffe.TEST)
  reshape_dim = tuple([1]) + tuple(img_dim)
  img = array(img).reshape(reshape_dim)
  #import pdb; pdb.set_trace()
  #print net.params.keys()
  restored_graph_def = None
  with tf.gfile.GFile(frozen_graph_path, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())
  input_node = None
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(restored_graph_def, name="")
    '''
		for node in restored_graph_def.node:
			print node.name
		'''
    with tf.Session() as sess:
      inp = graph.get_tensor_by_name('input:0')
      #output = graph.get_tensor_by_name('MobilenetV1/Predictions/Softmax:0')
      #prob=sess.run(output, feed_dict={inp: img})
      layers = []
      for node in restored_graph_def.node:
        #print(node.name)
        if 'value' in node.attr:
          array1 = tf.make_ndarray(node.attr['value'].tensor)
          #print node.name+' : '+str(array1.shape)
        if node.op not in ['Const', 'Identity']:
          new_tensor = graph.get_tensor_by_name(node.name + ':0')
          layers.append(new_tensor)
      output_hidden_layers = sess.run(layers, feed_dict={inp: img})
      tf_output = collections.OrderedDict()
      for ind in range(len(output_hidden_layers)):
        cur_layer = layers[ind].name
        converted_layer = cur_layer.replace('/', '_')
        converted_layer = converted_layer[:converted_layer.find(':0')]
        tf_output[converted_layer] = output_hidden_layers[ind]
      return tf_output


''' Adds statistics to a CSV file
	Args:
		writer: CSV writer object
		tf_layer_val: Numpy array of tensor value
		caffe_layer_val: Numpy array of caffe blob value
		caffe_layer: Current caffe layer name
		mismatch_layer: List of mismatched layers
		prev_layers: List of incoming layers to caffe_layer
'''


def print_stats(writer, tf_layer_val, caffe_layer_val, caffe_layer, mismatch_layers, prev_layers):
  error_layer_val = tf_layer_val - caffe_layer_val
  writer.writerow({
      'Layer name': caffe_layer,
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
      'error_max': np.max(error_layer_val),
      'input_layers': str(prev_layers)
  })
  if np.std(error_layer_val) >= 1e-4:
    mismatch_layers.append(caffe_layer)


'''
def check_batchnorm_relu(caffe_layer, tf_layer_output, tf_output):
	if 'BatchNorm' in caffe_layer:
		tf_layer_prefix = caffe_layer[0:caffe_layer.index('BatchNorm')]
		print "BatchNorm layer found"
		if (tf_layer_prefix+'Relu6') in tf_output:
			print "Taking Relu6 output"
			tf_layer_output = tf_output[tf_layer_prefix+'Relu6'].transpose(0,3,1,2)
		if (tf_layer_prefix+'Relu') in tf_output:
			print "Taking Relu output"
			tf_layer_output = tf_output[tf_layer_prefix+'Relu'].transpose(0,3,1,2)
	return tf_layer_output
def check_relu(caffe_layer, tf_layer_output , tf_output):
	if 'Conv' in caffe_layer:
		if 'Conv2D' in caffe_layer:
			tf_layer_prefix = caffe_layer[0:caffe_layer.index('Conv2D')]
			print "ReLU layer found"
			#import pdb;pdb.set_trace()
			if (tf_layer_prefix+'Relu6') in tf_output:
				print "Taking Relu6 output"
				tf_layer_output = tf_output[tf_layer_prefix+'Relu6'].transpose(0,3,1,2)
			if (tf_layer_prefix+'Relu') in tf_output:
				print "Taking Relu output"
				tf_layer_output = tf_output[tf_layer_prefix+'Relu'].transpose(0,3,1,2)
	return tf_layer_output
'''
''' Checks if a prefix is a substring of the layer_dict
	Args:
		layer_prefix: a string representing the layer prefix
		layer_dict: Dict with layers and their values
	Returns:
		A boolean value indicating if the prefix is a substring of a key in the dict
'''


def check_substring(layer_prefix, layer_dict):
  is_substring = False
  for layer in layer_dict:
    if layer_prefix in layer:
      is_substring = True
      break
  return is_substring


''' Checks and skips to a Relu/Relu6 layer output corresponding to the current Tensorflow layer
	Args: 
		caffe_layer: Current caffe layer
		tf_layer_output: Numpy array representing the Tensor value
		tf_output: A dict with all Tensorflow layer values
		caffe_output: A dict with all Caffe blob values
	Returns:
		Numpy array value 
'''


def check_comp_relu(caffe_layer, tf_layer_output, tf_output, caffe_output):
  #import pdb;pdb.set_trace()
  if '_' not in caffe_layer:
    return tf_layer_output
  #tf_layer_batchnorm  = caffe_layer[0: caffe_layer.rindex('_')]+'_BatchNorm_FusedBatchNorm'
  tf_layer_batchnorm = caffe_layer[0:caffe_layer.rindex('_')] + '_BatchNorm'
  if check_substring(tf_layer_batchnorm, tf_output) and check_substring(tf_layer_batchnorm, caffe_output):
    return tf_layer_output
  index_list = [i for i, ltr in enumerate(caffe_layer) if ltr == '_']
  index_list.append(len(caffe_layer))
  index_list = index_list[::-1]
  for ind in index_list:
    tf_layer_prefix = caffe_layer[0:ind]
    if (tf_layer_prefix + '_Relu6') in tf_output:
      # print "Taking Relu6 output"
      tf_layer_output = tf_output[tf_layer_prefix + '_Relu6'].transpose(0, 3, 1, 2)
      break
    if (tf_layer_prefix + '_Relu') in tf_output:
      # print "Taking Relu output"
      tf_layer_output = tf_output[tf_layer_prefix + '_Relu'].transpose(0, 3, 1, 2)
      break
  return tf_layer_output


''' Function that generates a layerwise output comparison in a CSV file
	Args:
		tf_output: A dict with all Tensorflow layer values
		net: Caffe net object
		output_tag: Name/Tag for the CSV file
'''


def compare_values(tf_output, net, output_tag):
  caffe_output_layers = net.blobs.keys()
  cnt = 0
  mismatch_layers = []
  #fieldnames= ['Layer name','tf_mean','caffe_mean','tf_std','caffe_std','tf_min','caffe_min','tf_max','caffe_max', 'tf_shape','caffe_layer_shape']
  fieldnames = [
      'Layer name', 'tf_mean', 'caffe_mean', 'tf_std', 'caffe_std', 'tf_min', 'caffe_min', 'tf_max', 'caffe_max', 'tf_shape', 'caffe_layer_shape',
      'error_mean', 'error_std', 'error_min', 'error_max', 'input_layers'
  ]
  csv_file = open(output_tag + '_final_output_comparison.csv', 'w')
  writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
  #output_comparison_content = list()
  writer.writeheader()
  for i in range(len(caffe_output_layers)):
    caffe_layer = caffe_output_layers[i]
    if 'DummyData' in caffe_layer:
      #print caffe_layer
      #print net.blobs[caffe_layer].data[...].shape
      continue
    if 'crop' in caffe_layer:
      #print caffe_layer
      #print net.blobs[caffe_layer].data[...].shape
      continue
    if caffe_layer in tf_output:
      cnt += 1
      tf_layer_output = tf_output[caffe_layer]
      caffe_layer_output = net.blobs[caffe_layer].data[...]
      # print caffe_layer
      #print caffe_layer_output.shape
      #print tf_layer_output.shape
      if len(tf_layer_output.shape) == 4:
        #print tf_layer_output.shape
        tf_layer_output = tf_layer_output.transpose(0, 3, 1, 2)
        #print tf_layer_output.shape
        #print caffe_layer_output.shape
        if tf_layer_output.shape == caffe_layer_output.shape:
          tf_layer_output = check_comp_relu(caffe_layer, tf_layer_output, tf_output, caffe_output_layers)
          print_stats(writer, tf_layer_output, caffe_layer_output, caffe_layer, mismatch_layers, net.bottom_names[caffe_layer])
        else:
          if len(tf_layer_output.shape) == len(caffe_layer_output.shape):
            caffe_layer_output = net.blobs[caffe_layer + '_crop'].data[...]
            tf_layer_output = check_comp_relu(caffe_layer, tf_layer_output, tf_output, caffe_output_layers)
            #print np.mean(np.abs(tf_layer_output - caffe_layer_output))
            #import pdb; pdb.set_trace()
            # print "Padding mismatch. Choosing crop layer"
            assert caffe_layer_output.shape == tf_layer_output.shape
          else:
            caffe_layer_output = net.blobs[caffe_layer + '_reshape'].data[...]
            tf_layer_output = check_comp_relu(caffe_layer, tf_layer_output, tf_output, caffe_output_layers)
          print_stats(writer, tf_layer_output, caffe_layer_output, caffe_layer, mismatch_layers, net.bottom_names[caffe_layer])
      else:
        tf_layer_output = check_comp_relu(caffe_layer, tf_layer_output, tf_output, caffe_output_layers)
        if tf_layer_output.shape != caffe_layer_output.shape:
          # print "Shapes don't match after " + caffe_layer + ' TF shape: ' + str(tf_layer_output.shape) + ' caffe shape: ' + str(
              # caffe_layer_output.shape)
          # print "Squeezing outputs now "
          caffe_layer_output = np.squeeze(caffe_layer_output)
          tf_layer_output = np.squeeze(tf_layer_output)
          print_stats(writer, tf_layer_output, caffe_layer_output, caffe_layer, mismatch_layers, net.bottom_names[caffe_layer])
        else:
          #print np.mean(np.abs(tf_layer_output - caffe_layer_output))
          print_stats(writer, tf_layer_output, caffe_layer_output, caffe_layer, mismatch_layers, net.bottom_names[caffe_layer])
  #print output_comparison_content
  csv_file.close()
  # print cnt
  # print "Mismatch layers are:"
  # for l in mismatch_layers:
  #   print l


def children(op):
  return set(op for out in op.outputs for op in out.consumers())


def parents(op):
  return set(input.op for input in op.inputs)


''' Skips to the correct layer (Basically skips over layers in Caffe where the output is fed back to the same layer Eg: BatchNorm - Scale)
	Args:
		index: Position in the list of blobs
		tf_output: A dict with all Tensorflow layer values
		net: Caffe net object
		cur_layer: Current layer name
	Returns:
		Tensorflow layer name
'''


def get_tf_layer(index, tf_output, net, cur_layer):
  #print "In func: "+net._layer_names[index]
  tf_name = cur_layer
  #import pdb;pdb.set_trace()
  index += 1
  while index < len(net.layers):
    caffe_layer_name = net._layer_names[index]
    #print "Next layer: "+caffe_layer_name
    if len(net.bottom_names[caffe_layer_name]) != 1 or len(net.top_names[caffe_layer_name]) != 1:
      break
    if net.bottom_names[caffe_layer_name][0] == net.top_names[caffe_layer_name][0]:
      tf_name = net._layer_names[index]
      index += 1
    else:
      break
  return tf_name


''' Function to compute layer wise comparison after copying the previous layer value from Tensorflow to Caffe (Layer isolation and comparison. Still needs to be debugged). Generates a CSV
	Args:
		img: OpenCV image
		frozen_graph_path: Tensorflow frozen graph path
		files_list: List of prototxt files
		net: Caffe net object
		output_tag: Name/Tag for the CSV file
'''


def compute_layer_diffs(img, frozen_graph_path, files_list, tf_output, net, output_tag):
  img = array(img).reshape(1, 224, 224, 3)
  #print net.params.keys()
  restored_graph_def = None
  mismatch_layers = []
  with tf.gfile.GFile(frozen_graph_path, "rb") as f:
    restored_graph_def = tf.GraphDef()
    restored_graph_def.ParseFromString(f.read())
  input_node = None
  fieldnames = [
      'Layer name', 'tf_mean', 'caffe_mean', 'tf_std', 'caffe_std', 'tf_min', 'caffe_min', 'tf_max', 'caffe_max', 'tf_shape', 'caffe_layer_shape',
      'error_mean', 'error_std', 'error_min', 'error_max'
  ]
  csv_file = open(output_tag + '_layer_comparison.csv', 'w')
  writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
  #output_comparison_content = list()
  writer.writeheader()
  with tf.Graph().as_default() as graph:
    tf.import_graph_def(restored_graph_def, name="")
    '''
		for node in restored_graph_def.node:
			print node.name
		'''
    with tf.Session() as sess:
      '''
			for node in restored_graph_def.node:
				print node.name+' : '+node.op
			'''
      tf.compat.v1.graph_util.remove_training_nodes(restored_graph_def)
      graph = tf.get_default_graph()
      ops = graph.get_operations()
      tf_op_children = collections.OrderedDict()
      tf_op_parents = collections.OrderedDict()
      for op in ops:
        #print op.name
        tf_op_children[op] = children(op)
        #print d[op]
        tf_op_parents[op] = parents(op)
        #print p[op]
        #d = collections.OrderedDict({op: children(op) for op in ops})
      n_cnt = 0
      n_match = 0
      #print files_list
      tensors_list = []
      rev_mapping = dict()
      for key in tf_op_children:
        if key.type not in ['Const', 'Identity']:
          #print str(key.name)+' : '+str(d[key])
          rev_mapping[str(key.name).replace("/", "_")] = str(key.name)
          conv_key = str(key.name).replace("/", "_") + '.prototxt'
          inp_list = [str(op.name) for op in tf_op_children[key]]
          tensors_list.append(str(key.name).replace("/", "_"))
          if conv_key in files_list:
            n_match += 1
          n_cnt += 1
  #print(n_cnt)
      assert n_match == len(files_list)
      caffe_cnt = 0
      blobs_list = net.blobs.keys()
      #print blobs_list[0]
      #print tensors_list[0]
      #net.blobs[blobs_list[0]].data[...] = tf_output[tensors_list[0]]
      #print net.blobs['MobilenetV1_MobilenetV1_Conv2d_0_Conv2D_crop'].data.shape
      #print tf_output['MobilenetV1_MobilenetV1_Conv2d_0_Conv2D'].shape
      #print net.blobs['MobilenetV1_MobilenetV1_Conv2d_0_Conv2D'].data
      prev_layer = 'input'
      #net.blobs['MobilenetV1_MobilenetV1_Conv2d_0_Conv2D'].data[...]= tf_output['MobilenetV1_MobilenetV1_Conv2d_0_Conv2D'].data
      #print d['']
      #rint net.blobs[prev_layer].data.shape
      #print tf_output[prev_layer].shape
      caffe_layer_val = net.blobs[prev_layer].data
      tf_layer_val = tf_output['input'].transpose(0, 3, 1, 2)
      for i in range(1, len(net.layers)):
        caffe_layer_name = net._layer_names[i]
        #caffe_op_parents[caffe_layer_name] = [net._blob_names[bi] for bi in list(net._bottom_ids(i))]
        #caffe_op_children[caffe_layer_name] = [net._blob_names[bi] for bi in list(net._top_ids(i))]
        if caffe_layer_name not in tf_output:
          #print caffe_layer_name+" not in TF"
          continue
        #print caffe_layer_name
        if caffe_layer_name in net.blobs:
          # print 'Caffe: ' + str(net.blobs[caffe_layer_name].data.shape)
          tf_layer_val = tf_output[caffe_layer_name]
          end_layer = None
          if len(net.blobs[caffe_layer_name].data.shape) == 4:
            tf_layer_val = tf_output[caffe_layer_name].transpose(0, 3, 1, 2)
            if net.blobs[caffe_layer_name].data.shape != tf_output[caffe_layer_name].transpose(0, 3, 1, 2).shape:
              end_layer = caffe_layer_name + '_crop'
          if not end_layer:
            end_layer = net.top_names[caffe_layer_name][0]
          prev_layer = net.bottom_names[caffe_layer_name][0]
          #import pdb;pdb.set_trace()
          for p_l in net.bottom_names[caffe_layer_name]:
            #print "Parent layer: "+p_l
            #print "Caffe: "+str(net.blobs[p_l].data.shape)
            tf_pl = p_l
            if 'crop' in p_l:
              tf_pl = p_l[:p_l.rindex('_crop')]
              net.blobs[p_l].data[...] = tf_output[tf_pl].transpose(0, 3, 1, 2)
              continue
            if p_l == 'Placeholder':
              tf_pl = 'input'
            #print "Tensorflow: "+str(tf_output[tf_pl].shape)
            if len(net.blobs[p_l].data.shape) == 4:
              #print "Old std: "+np.std()
              new_l = get_tf_layer(net.bottom_names.keys().index(p_l), tf_output, net, p_l)
              #print "New layer: "+new_l
              net.blobs[p_l].data[...] = tf_output[new_l].transpose(0, 3, 1, 2)
            else:
              #net.blobs[p_l].data[...].reshape(tf_output[tf_pl].shape)
              net.blobs[p_l].data[...] = tf_output[tf_pl].reshape(net.blobs[p_l].data.shape)
          # print "Previous layer: " + str(prev_layer)
          # print "End layer: " + str(end_layer)
          end_layer_index = net.blobs.keys().index(end_layer)
          if end_layer_index + 1 < len(net.blobs.keys()):
            end_layer = net.blobs.keys()[end_layer_index + 1]
          output = net.forward(start=prev_layer, end=end_layer)
          caffe_layer_val = net.blobs[caffe_layer_name].data
          if tf_layer_val.shape != caffe_layer_val.shape and len(tf_layer_val.shape) == 4:
            # print "Padding mismatch. Choosing crop layer"
            caffe_layer_val = net.blobs[caffe_layer_name + '_crop'].data[...]
            assert caffe_layer_val.shape == tf_layer_val.shape
          #import pdb; pdb.set_trace()
          #caffe_layer_mod_val = get_tf_layer(net.bottom_names.keys().index(caffe_layer_val), tf_output,)
          if 'BatchNorm' in caffe_layer_name:
            tf_layer_prefix = caffe_layer_name[0:caffe_layer_name.index('BatchNorm')]
            # print "BatchNorm layer found"
            if (tf_layer_prefix + 'Relu6') in tf_output:
              # print "Taking Relu6 output"
              tf_layer_val = tf_output[tf_layer_prefix + 'Relu6'].transpose(0, 3, 1, 2)
            if (tf_layer_prefix + 'Relu') in tf_output:
              # print "Taking Relu output"
              tf_layer_val = tf_output[tf_layer_prefix + 'Relu'].transpose(0, 3, 1, 2)
          if tf_layer_val.shape != caffe_layer_val.shape:
            # print "Shapes don't match after " + caffe_layer_name + ' TF shape: ' + str(tf_layer_val.shape) + ' caffe shape: ' + str(
                # caffe_layer_val.shape)
            caffe_layer_val = np.squeeze(caffe_layer_val)
            tf_layer_val = np.squeeze(tf_layer_val)
          print_stats(writer, tf_layer_val, caffe_layer_val, caffe_layer_name, mismatch_layers, net.bottom_names[caffe_layer_name])
          #print "Comparison of "+caffe_layer_name+' - '+str(np.std(caffe_layer_val))+ ' : '+str(np.std(tf_layer_val))
  csv_file.close()
  '''
	for key in tf_op_children:
		if key.type not in ['Const','Identity']:
			conv_name = key.name.replace("/","_")
			if conv_name in net.blobs.keys():
				print "Tensorflow layer: "+key.name
				print "TF child nodes: "+str(tf_op_children[key])
				print "TF parent nodes: "+str(tf_op_parents[key])
				caffe_index = net.blobs.keys().index(conv_name)
				print "Caffe layer: "+conv_name
				print "Caffe child nodes: "+str(caffe_op_children[conv_name])
				print "Caffe parent nodes: "+str(caffe_op_parents[conv_name])
	'''
  '''
		for key in d:
			if key.type not in ['Const', 'Identity']:
				#print key.name+' : '+' '.join([str(op.name) for op in d[key]])
				#print tf_output[key.name.replace("/","_")].shape
			#print net.blobs.keys()
			conv_name = key.name.replace("/","_")
			if conv_name in net.blobs.keys():
				#print net.blobs[conv_name].data[...]
				cur_layer = conv_name
				#print cur_layer
				tf_cur_layer_output= None
				if len(net.blobs[cur_layer].data.shape)==4:
					if net.blobs[cur_layer].data.shape != tf_output[cur_layer].transpose(0,3,1,2).shape:
						cur_layer = cur_layer+'_crop'
					print cur_layer
					#net.forward(start=prev_layer, end=cur_layer)
					print str(np.std(net.blobs[cur_layer].data[...])) + ' : '+str(np.std(tf_output[conv_name].transpose(0,3,1,2)))
					tf_cur_layer_output = tf_output[conv_name].transpose(0,3,1,2)
				else:
					print cur_layer
					print str(np.std(net.blobs[cur_layer].data[...])) + ' : '+str(np.std(tf_output[conv_name]))
					tf_cur_layer_output = tf_output[conv_name]
				prev_layer=cur_layer
				net.blobs[cur_layer].data[...]= tf_cur_layer_output
				caffe_cnt+=1
				#break
			else:
				print 'Not found: '+conv_name
				n
				#print net.blobs[str(key.name).replace("/","_")][0].data.shape
	'''


''' Generate 2 CSV files with layer statistics of all layers present in Tensorflow and Caffe
	Args:
		tf_output: A dict with all Tensorflow layer values
		net: Caffe net object
		output_tag: Name/Tag for the CSV files generated
'''


def debug(tf_output, net, output_tag):
  tf_fieldnames = ['Layer name', 'tf_mean', 'tf_std', 'tf_min', 'tf_max', 'tf_shape']
  csv_file = open(output_tag + '_all_tf_layers.csv', 'w')
  writer = csv.DictWriter(csv_file, fieldnames=tf_fieldnames)
  writer.writeheader()
  for layer in tf_output:
    tf_layer_val = tf_output[layer]
    writer.writerow({
        'Layer name': layer,
        'tf_mean': np.mean(tf_layer_val),
        'tf_std': np.std(tf_layer_val),
        'tf_min': np.min(tf_layer_val),
        'tf_max': np.max(tf_layer_val),
        'tf_shape': tf_layer_val.shape
    })
  csv_file.close()
  csv_file = open(output_tag + '_all_caffe_layers.csv', 'w')
  caffe_fieldnames = ['Layer name', 'caffe_mean', 'caffe_std', 'caffe_min', 'caffe_max', 'caffe_shape']
  writer = csv.DictWriter(csv_file, fieldnames=caffe_fieldnames)
  writer.writeheader()
  for i in range(len(net.layers)):
    layer = net._layer_names[i]
    if layer in net.blobs:
      caffe_layer_val = net.blobs[layer].data
      #print np.mean(caffe_layer_val)
      #print caffe_layer_val.shape
      writer.writerow({
          'Layer name': layer,
          'caffe_mean': np.mean(caffe_layer_val),
          'caffe_std': np.std(caffe_layer_val),
          'caffe_min': np.min(caffe_layer_val),
          'caffe_max': np.max(caffe_layer_val),
          'caffe_shape': caffe_layer_val.shape
      })
  csv_file.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='To compare the layer parameter values and outputs from each layer')
  parser.add_argument('--frozen_graph_path', required=True, type=str, help='specify tensorflow frozen pb path')
  parser.add_argument('--img_path', required=True, type=str, help='specify path to test image')
  parser.add_argument('--prototxt_path', required=True, type=str, help='specify path to the converted prototxt file obtained using MMdnn')
  parser.add_argument('--model_path', required=True, type=str, help='specify path to the converted caffemodel obtained using MMdnn')
  parser.add_argument('--files_path',
                      required=True,
                      type=str,
                      help='specify path to directory containing the prototxt files corresponding to each tensorflow layer')
  parser.add_argument('--output_tag', required=True, type=str, help='specify the name of the model')
  parser.add_argument('--input_shape', required=True, type=str, help='specify H,W,C of the model input')
  args = parser.parse_args()
  img_dim = list(map(int, args.input_shape.split(',')))
  nh = (img_dim[0])
  nw = (img_dim[1])
  img = cv2.imread(str(args.img_path))
  img = cv2.resize(img, (nh, nw))
  tf_output = load_tf_graph(str(args.frozen_graph_path), img, img_dim)
  net = load_caffe_graph(str(args.prototxt_path), str(args.model_path), img, img_dim)
  compare_values(tf_output, net, str(args.output_tag))
  files_path = os.path.abspath(str(args.files_path))
  files_list = get_data(files_path)
#debug(tf_output,net,str(args.output_tag))
#compute_layer_diffs(img, str(args.frozen_graph_path),files_list, tf_output,net, str(args.output_tag))
