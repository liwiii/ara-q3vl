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

import mxnet as mx
import mxnet.ndarray as nd
import pandas as pd
import json
import re
import dv_logging
import os
logger = dv_logging.get_logger(__name__, 'DVPTPR')


def preprocess(args):
  # Special characters
  spl_char_string = r'[-.:\/\\]'
  regex = re.compile(spl_char_string)
  new_json = os.path.join(args.output_dir, "_model-symbol.json")
  new_params = os.path.join(args.output_dir, "_model-0000.params")
  csv_file = os.path.join(args.output_dir, "_model-name-map.csv")
  map_dict = {}
  try:
    # Modify the input_network
    with open(args.input_network, "r") as f:
      net = json.load(f)
      for node in net['nodes']:
        if regex.search(node['name']) is not None:
          orig_name = node['name']
          for ch in spl_char_string:
            node['name'] = node['name'].replace(ch, '_')
          conv_name = node['name']
          map_dict[orig_name] = conv_name
      if len(map_dict) > 0:
        logger.warning("Please avoid the names with the special characters {}".format(spl_char_string))
        logger.info("Names of the following layers and arguments will be replaced according to the following dictionary")
        logger.info(map_dict)
      else:
        return 0
      with open(new_json, "w") as f:
        json.dump(net, f)
    # Modify the input weights file
    orig_params = nd.load(args.input_weight)
    for node in map_dict:
      if 'arg:' + node in orig_params:
        orig_params['arg:' + map_dict[node]] = orig_params.pop('arg:' + node)
      elif 'aux:' + node in orig_params:
        orig_params['aux:' + map_dict[node]] = orig_params.pop('aux:' + node)
    nd.save(new_params, orig_params)
    # Modify the input and output node names
    for i in range(len(args.input_node_name)):
      if args.input_node_name[i] in map_dict:
        args.input_node_name[i] = map_dict[args.input_node_name[i]]
    if args.output_node_name is not None:
      for i in range(len(args.output_node_name)):
        if args.output_node_name[i] in map_dict:
          args.output_node_name[i] = map_dict[args.output_node_name[i]]
    # Export the dictionary to csv file
    df = pd.DataFrame(map_dict.items(), columns=["Original", "Renamed"])
    sr = pd.Series(["Argument" if 'arg:' + node in orig_params else "Layer" for node in map_dict], name="Type")
    final_df = pd.concat([df, sr], axis=1)
    final_df.to_csv(csv_file)
    logger.info("Rename map exported to: {}".format(csv_file))
    # Update the input network path and input weight path
    args.input_network = new_json
    args.input_weight = new_params
  except Exception as e:
    logger.error("Preprocessing failed with the below error:\n{}".format(str(e)))
    return -1
  return 0
