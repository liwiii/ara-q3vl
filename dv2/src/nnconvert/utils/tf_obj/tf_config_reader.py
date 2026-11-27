# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
# 
# Copyright (c) 2022-25, Kinara, Inc. All rights reserved.
# Copyright 2025-2026 NXP
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import tensorflow.compat.v1 as tf
import sys as _sys
import os
from google.protobuf import text_format
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/')
from object_detection.protos import pipeline_pb2


def get_configs_from_pipeline_file(pipeline_config_path, config_override=None):
  '''
  read .config and convert it to proto_buffer_object
  '''
  pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
  with tf.gfile.GFile(pipeline_config_path, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)
  if config_override:
    text_format.Merge(config_override, pipeline_config)
  #print(pipeline_config)
  return pipeline_config


def create_configs_from_pipeline_proto(pipeline_config):
  '''
  Returns the configurations as dictionary
  '''
  configs = {}
  configs["model"] = pipeline_config.model
  configs["train_config"] = pipeline_config.train_config
  configs["train_input_config"] = pipeline_config.train_input_reader
  configs["eval_config"] = pipeline_config.eval_config
  configs["eval_input_configs"] = pipeline_config.eval_input_reader
  # Keeps eval_input_config only for backwards compatibility. All clients should
  # read eval_input_configs instead.
  if configs["eval_input_configs"]:
    configs["eval_input_config"] = configs["eval_input_configs"][0]
  if pipeline_config.HasField("graph_rewriter"):
    configs["graph_rewriter_config"] = pipeline_config.graph_rewriter
  return configs


if __name__ == '__main__':
  import sys
  configs = get_configs_from_pipeline_file(sys.argv[1])
  config_as_dict = create_configs_from_pipeline_proto(configs)
