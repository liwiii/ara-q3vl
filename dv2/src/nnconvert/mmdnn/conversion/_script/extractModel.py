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

from six import text_type as _text_type
from dv_logging import get_logger
logger = get_logger(__name__)


def generate_label(predict, label_file, offset):
  import os
  if not os.path.exists(label_file):
    return predict
  with open(label_file, 'r') as f:
    labels = [l.rstrip() for l in f]
  ret = []
  for i, j in predict:
    ret.append((labels[i - offset], i, j))
  return ret


def extract_model(args):
  if args.framework == 'caffe':
    from mmdnn.conversion.examples.caffe.extractor import caffe_extractor
    extractor = caffe_extractor()
  elif args.framework == 'keras':
    from mmdnn.conversion.examples.keras.extractor import keras_extractor
    extractor = keras_extractor()
  elif args.framework == 'tensorflow' or args.framework == 'tf':
    from mmdnn.conversion.examples.tensorflow.extractor import tensorflow_extractor
    extractor = tensorflow_extractor()
  elif args.framework == 'mxnet':
    from mmdnn.conversion.examples.mxnet.extractor import mxnet_extractor
    extractor = mxnet_extractor()
  elif args.framework == 'cntk':
    from mmdnn.conversion.examples.cntk.extractor import cntk_extractor
    extractor = cntk_extractor()
  elif args.framework == 'pytorch':
    from mmdnn.conversion.examples.pytorch.extractor import pytorch_extractor
    extractor = pytorch_extractor()
  elif args.framework == 'darknet':
    from mmdnn.conversion.examples.darknet.extractor import darknet_extractor
    extractor = darknet_extractor()
  elif args.framework == 'coreml':
    from mmdnn.conversion.examples.coreml.extractor import coreml_extractor
    extractor = coreml_extractor()
  else:
    raise ValueError("Unknown framework [{}].".format(args.framework))
  files = extractor.download(args.network, args.path)
  if files and args.image:
    predict = extractor.inference(args.network, files, args.path, args.image)
    if type(predict) == list:
      logger.debug("%s", predict)
    else:
      assert predict is not None
      if predict.ndim == 1:
        if predict.shape[0] == 1001:
          offset = 1
        else:
          offset = 0
        top_indices = predict.argsort()[-5:][::-1]
        predict = [(i, predict[i]) for i in top_indices]
        predict = generate_label(predict, args.label, offset)
        for line in predict:
          logger.debug("%s", line)
      else:
        logger.debug("%s", predict.shape)
        logger.debug("%s", predict)


def _main():
  import argparse
  parser = argparse.ArgumentParser(description='Extract pre-trained models for frameworks.')
  parser.add_argument('--framework',
                      '-f',
                      type=_text_type,
                      required=True,
                      choices=["caffe", "cntk", "mxnet", "keras", "tensorflow", 'tf', 'pytorch', 'darknet', 'coreml'],
                      help="Framework name")
  parser.add_argument('--network', '-n', type=_text_type, default=None, help='Path to the model network file of the external tool (e.g caffe prototxt, keras json')
  parser.add_argument('-i', '--image', type=_text_type, help='Test Image Path')
  parser.add_argument('--path', '-p', '-o', type=_text_type, default='./', help='Path to save the pre-trained model files (e.g keras h5)')
  parser.add_argument('-l', '--label', type=_text_type, default='mmdnn/conversion/examples/data/imagenet_1000.txt', help='Path of label.')
  args = parser.parse_args()
  extract_model(args)


if __name__ == '__main__':
  _main()
