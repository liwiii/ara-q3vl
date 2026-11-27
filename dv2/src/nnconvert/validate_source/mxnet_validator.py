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
import mxnet.symbol as sym
import numpy as np
import warnings
import mxnet.ndarray as nd
import json
from dv_logging import get_logger
from utils.io_utils import generate_input_default
logger = get_logger(__name__, 'DVIPVAL')
logger.info("mxnet %s", mx.__version__)


def looks_like_weight(name):
  weight_like = ('_weight', '_bias', '_beta', '_gamma', '_moving_var', '_moving_mean', '_running_var', '_running_mean')
  return name.endswith(weight_like)


def find_states(sym_json, shape_dict):
  """
    states hold the name and shape
    of each rnn state in the model
    which needs to be passed 
    explicitly in arg_params.
    Imp:Batch-size is determined from
    network's batch-size.
    """
  symbol = sym.load(sym_json)
  all_layers = symbol.get_internals()
  states = dict()
  souts = []
  with open(sym_json, 'r') as f:
    x = json.load(f)
  for i in x['nodes']:
    if i['op'] == 'RNN':
      inputs = i['inputs']
      if len(inputs) > 2:
        h0 = x['nodes'][inputs[2][0]]
        if h0['op'] == 'null':
          state_out = i['name'] + '_state_output'
          arg_shape, oshape, aux_shape = all_layers[state_out].infer_shape(**shape_dict)
          #assumption : state output has single blob
          assert oshape is not None
          states[h0['name']] = oshape[0]
          souts.append(state_out)
      if len(inputs) > 3:
        c0 = x['nodes'][inputs[3][0]]
        if c0['op'] == 'null':
          state_out = i['name'] + '_statecell_output'
          arg_shape, oshape, aux_shape = all_layers[state_out].infer_shape(**shape_dict)
          assert oshape is not None
          states[c0['name']] = oshape[0]
          souts.append(state_out)
  return states, souts


class mxnet_validator:

  @staticmethod
  def run_model(args):
    input_shape = []
    if args.input_shape != None:
      for x in args.input_shape:
        shape = x.split(',')
        input_shape.append([int(x) for x in shape])
    state_shape = []
    if args.state_shape != None:
      for x in args.state_shape:
        shape = x.split(',')
        state_shape.append([int(x) for x in shape])
    label_name = None
    label_shape = None
    if args.label_shape:
      if not args.label_name:
        logger.error("please provide both label name and shape")
        return -1
    if args.label_name:
      if not args.label_shape:
        logger.error("please provide both label name and shape")
        return -1
    input_label_shape = []
    if args.label_shape:
      for x in args.label_shape:
        shape = x.split(',')
        input_label_shape.append([int(x) for x in shape])
    if args.label_name:
      if len(args.label_name) != len(args.label_shape):
        logger.error("number of label names and shapes didn't match")
        return -1
      label_name = tuple(args.label_name)
      label_shape = []
      for i in range(len(args.label_name)):
        label_shape.append((args.label_name[i], tuple(input_label_shape[i])))
    data_shapes = []
    data_dict = generate_input_default(args)['default']
    data_list = []
    for i in range(len(args.input_node_name)):
      data_shapes.append((
          args.input_node_name[i],
          tuple(input_shape[i]),
      ))
      data_list.append(mx.nd.array(data_dict[args.input_node_name[i]]))
    if args.set_np_shape == True:
      mx.set_np_shape(True)
    try:
      warnings.filterwarnings("ignore")
      with open(args.input_network, 'r') as f:
        sym_json = json.load(f)
      temp_params = args.tmp + '-0000.params'
      temp_json = args.tmp + '-symbol.json'
      nodes = set()
      for idx, i in enumerate(sym_json['nodes']):
        if i['name'] not in nodes:
          nodes.add(i['name'])
        else:
          logger.warn("duplicate node %s found, renaming it to %s", i['name'], i['name'] + str(idx))
          i['name'] = i['name'] + str(idx)
          nodes.add(i['name'])
        if 'attrs' in i:
          if '__dtype__' in i['attrs']:
            if i['attrs']['__dtype__'] != '0':
              logger.warning("precision typecasted from %s to %s in symbol json for layer %s", i['attrs']['__dtype__'], '0', i['name'])
              i['attrs']['__dtype__'] = '0'
      with open(temp_json, 'w') as f:
        json.dump(sym_json, f)
      s = mx.sym.load(temp_json)
      s.save(temp_json)
      args.input_network = temp_json
      wts = nd.load(args.input_weight)
      for wt in wts:
        wts[wt] = nd.cast(wts[wt], dtype='float32')
      if len(wts) == 0 and isinstance(wts, list):
        wts = {}
        wts[str('arg:' + args.tmp.split('/')[-1])] = nd.array(np.array([0]))
      nd.save(temp_params, wts)
      args.input_weight = temp_params
      symbol = sym.load(args.input_network)
      save_dict = nd.load(args.input_weight)
      arg_params = {}
      aux_params = {}
      all_nodes = {}
      if save_dict and isinstance(save_dict, dict):
        for k, v in save_dict.items():
          tp, name = k.split(':', 1)
          if tp == 'arg':
            arg_params[name] = v
          elif tp == 'aux':
            aux_params[name] = v
          else:
            all_nodes[name] = v
      for i in all_nodes.keys():
        if looks_like_weight(i):
          continue
        elif not args.label_name:
          logger.warn("unrecognized node: '%s', not present in -label_name", i)
        elif i in args.label_name:
          continue
        else:
          logger.warn("unrecognized node: '%s', please check if it is a label node", i)
      states = dict()
      for idx, i in enumerate(args.state_name):
        states[i] = state_shape[idx]
      shape_dict = dict()
      for i in data_shapes:
        shape_dict[i[0]] = i[1]
      snames, souts = find_states(args.input_network, shape_dict)
      if set(snames) != set(args.state_name):
        logger.warn("state inputs found : %s ,state inputs provided : %s", set(snames), set(args.state_name))
      for i in states:
        arg_params[i] = mx.nd.array(np.random.random_sample(states[i]))
      inputs = [mx.sym.var(i) for i in args.input_node_name]
      model = mx.gluon.nn.SymbolBlock(outputs=symbol, inputs=inputs)
      try:
        model.load_parameters(args.input_weight, ctx=mx.cpu(), allow_missing=False, ignore_extra=False)
      except Exception as e:
        logger.warn("error loading parameters: %s", str(e))
        logger.info("loading with allow_missing=True, ignore_extra=True")
        model.load_parameters(args.input_weight, ctx=mx.cpu(), allow_missing=True, ignore_extra=True)
      model.collect_params().initialize(ctx=mx.cpu())
      output = model(*data_list)
      one_output = output[0].asnumpy()
      model.save_params(temp_params)
      return 0
    except Exception as e:
      logger.error("exception occurred when validating mxnet", exc_info=e)
      return -1
