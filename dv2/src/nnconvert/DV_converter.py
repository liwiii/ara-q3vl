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

import argparse
from subprocess import Popen, PIPE
import os, sys
'''
Input Framework Tensorflow
Output Framework Caffe
'''
srcFW = 'tensorflow'
dstFW = 'caffe'
args = None
caffe_python_infer_file = 'skipnet_segmentation_converter_without_preprocessing_without_postprocessing.py'


def parse_args():
  """Parses command line arguments."""
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument("--frozen_graph", type=str, required=True, help="Tensorflow Frozen Graph file")
  parser.add_argument("--inputShape", type=str, required=True, help="InputShape in H,W,C. Don't need batch")
  parser.add_argument("--inNode", type=str, required=True, help="Input nodes. 'comma' seperated")
  parser.add_argument("--outNode", type=str, required=True, help="Output Nodes. 'comma' seperated")
  parser.add_argument("--output_folder", type=str, default='.', help="Output directory to save model files")
  parser.add_argument("--output_name", type=str, required=True, help="Name or tag for the folder")
  parser.add_argument("--run_inference", type=bool, default=False, help="Name or tag for the folder")
  return parser.parse_args()


def convert():
  assert args is not None
  input_shape = args.inputShape.split(',')
  nh, nw = input_shape[0], input_shape[1]
  in_nodes_list = args.inNode.split(',')
  out_nodes_list = args.outNode.split(',')
  in_nodes = ' '.join(in_nodes_list)
  out_nodes = ' '.join(out_nodes_list)
  out_file = os.path.join(args.output_folder, args.output_name)
  model2ir_command = ['mmtoir', '-f', srcFW, '--inNodeName', in_nodes, '--dstNodeName', out_nodes, '--inputShape', args.inputShape, '-o', out_file, '-w', args.frozen_graph]
  print("Executing command " + str(' '.join(model2ir_command)))
  model2ir_process = Popen(model2ir_command, stdout=PIPE, stderr=PIPE)
  stdout, stderr = model2ir_process.communicate()
  print("Command: {}\nStdout: {}\nStderr: {}\n".format(model2ir_command, stdout, stderr))
  if model2ir_process.returncode != 0:
    print("error while executing command: " + str(' '.join(model2ir_command)))
    sys.exit(-1)
  tf_ir_meta_frozen = out_file + '.pb'
  ir_json = out_file + '.json'
  ir_npy = out_file + '.npy'
  caffe_code_file = out_file + '_caffe_code.py'
  caffe_weights_file = out_file + '_caffe_weights.npy'
  ir2code_command = ['mmtocode', '-f', dstFW, '-n', tf_ir_meta_frozen, '-w', ir_npy, '-o', caffe_code_file, '-ow', caffe_weights_file]
  print("Executing command " + str(' '.join(ir2code_command)))
  ir2code_process = Popen(ir2code_command, stdout=PIPE, stderr=PIPE)
  stdout, stderr = ir2code_process.communicate()
  print("Command: {}\nStdout: {}\nStderr: {}\n".format(ir2code_command, stdout, stderr))
  if ir2code_process.returncode != 0:
    print("error while executing command: " + str(' '.join(ir2code_command)))
    sys.exit(-1)
  caffe_model_name = out_file + 'caffe_model'
  code2model_command = ['mmtomodel', '-f', dstFW, '-in', caffe_code_file, '-iw', caffe_weights_file, '-o', caffe_model_name]
  print("Executing command " + str(' '.join(code2model_command)))
  code2model_process = Popen(code2model_command, stdout=PIPE, stderr=PIPE)
  stdout, stderr = code2model_process.communicate()
  print("Command: {}\nStdout: {}\nStderr: {}\n".format(code2model_command, stdout, stderr))
  if code2model_process.returncode != 0:
    print("error while executing command: " + str(' '.join(code2model_command)))
    sys.exit(-1)
  caffe_prototxt_file = caffe_model_name + '.prototxt'
  caffe_model_file = caffe_model_name + '.caffemodel'
  input_layer = in_nodes_list[0].replace('/', '_').replace('-', '_').replace('[', '_').replace(']', '_')
  output_layer = out_nodes_list[0].replace('/', '_').replace('-', '_').replace('[', '_').replace(']', '_')
  if not args.run_inference:
    print("Not running inference")
    sys.exit(0)
  caffe_infer_command = [
      'python', caffe_python_infer_file, '--caffe_prototxt', caffe_prototxt_file, '--caffe_model', caffe_model_file, '--nh', nh, '--nw', nw, '--input_layer', input_layer,
      '--output_layer', output_layer
  ]
  print("Executing command " + str(' '.join(caffe_infer_command)))
  caffe_infer_process = Popen(caffe_infer_command, stdout=PIPE, stderr=PIPE)
  stdout, stderr = caffe_infer_process.communicate()
  print("Command: {}\nStdout: {}\nStderr: {}\n".format(caffe_infer_command, stdout, stderr))
  if caffe_infer_process.returncode != 0:
    print("error while executing command: " + str(' '.join(caffe_infer_command)))
    sys.exit(-1)


if __name__ == '__main__':
  args = parse_args()
  convert()
