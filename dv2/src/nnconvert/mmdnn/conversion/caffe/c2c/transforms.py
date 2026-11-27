'''
Copyright (c) 2018-25, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

import networkx as nx
from functools import wraps
from itertools import chain
from collections import defaultdict
import google.protobuf as pb
import os
import caffe
import copy
from caffe.proto import caffe_pb2
import importlib
import pandas as pd
import yaml
from pathlib import Path
from dv_logging import get_logger
from mmdnn.conversion.caffe.c2c import common
from mmdnn.conversion.common.utils import caffe_to_graph
import json

logger = get_logger(__name__, 'DVCTOC')

IMPORT_BASE_PATH = 'mmdnn.conversion.caffe.c2c.'
C2C_TRANSFORM_NAME = "C2C Transform"
C2C_REMOVED_NODES = "Removed Nodes"
C2C_ADDED_NODES = "New Nodes"
C2C_FILE_NAME = "File Name"


class C2CLogger():

  def __init__(self):
    self.file_name = "pass_0"

  def get_file_name(self):
    return self.file_name

  def set_file_name(self, file_name):
    self.file_name = file_name


optimization_passes = {}
ignore_prefix_passes = []
C2C_TRANSFORM_NAME = "C2C Transform"
C2C_REMOVED_NODES = "Removed Nodes"
C2C_ADDED_NODES = "New Nodes"
C2C_FILE_NAME = "File Name"


class C2CLogger():

  def __init__(self):
    self._file_name = "pass_0"

  def get_file_name(self):
    return self._file_name

  def set_file_name(self, file_name):
    self._file_name = file_name


mutation_observer = {}


def list_check_equal(l1: list, l2: list):
  '''
  if list a == list b -> return False else True
  '''
  return (list(set(l1) - set(l2)) != [] and list(set(l2) - set(l1)) != [])


def initialize_observer(dv_graph, original_node_names):
  global mutation_observer
  mutation_observer["family"] = {}
  mutation_observer["meta"] = {}

  origial_name_set = set()

  for node in list(dv_graph.nodes):
    result = node.split('/')
    curr_node_name = ""
    if len(result) == 2:
      curr_node_name = result[1]
    else:
      curr_node_name = '/'.join(result[1:])
    index = curr_node_name.find('_ara')
    if index != -1:
      curr_node_name = curr_node_name[:index]
    origial_name_set.add(curr_node_name)

    original_node_types[curr_node_name] = result[0]
    node_name_mapping[curr_node_name] = curr_node_name

    mutation_observer["family"][curr_node_name] = [curr_node_name]
    mutation_observer["meta"][curr_node_name] = {"input": [[]], "output": [[]], "shuffled": False, "isPresent": True}

    __input = [f'{n.type}/{n.name}' for n in dv_graph.nodes[node]['node_obj'].predecessors()]
    __output = [f'{n.type}/{n.name}' for n in dv_graph.nodes[node]['node_obj'].successors()]
    __labelinput = [dv_graph.nodes[node]['label'] for node in __input]
    __labeloutput = [dv_graph.nodes[node]['label'] for node in __output]

    mutation_observer["meta"][curr_node_name]["input"] = [__labelinput]
    mutation_observer["meta"][curr_node_name]["output"] = [__labeloutput]

    mutation_observer["meta"][curr_node_name]["shuffled"] = False
  for name in origial_name_set:
    original_node_names.append(name)
  original_node_names = sorted(original_node_names)


def update_observer(dv_graph, original_node_names):
  global mutation_observer

  # OBSERVER
  updated_node_list = [node.split('/')[1] for node in dv_graph.nodes]
  nodes_queried = set()
  try:
    for _node in dv_graph.nodes:
      node = _node.split('/')[1]
      nodes_queried.add(node)
      if node in original_node_names:
        matches = [n.split('/')[1] for n in dv_graph.nodes if n.split('/')[1].startswith(node + "_ara")]
        matches.insert(0, node)
        mutation_observer["family"][node] = matches
      if node not in mutation_observer["meta"]:
        logger.debug("[D] new node added. push to meta %s", node)
        mutation_observer["meta"][node] = {"input": [], "output": [], "shuffled": False}
      __input = [f'{n.type}/{n.name}' for n in dv_graph.nodes[_node]['node_obj'].predecessors()]
      __output = [f'{n.type}/{n.name}' for n in dv_graph.nodes[_node]['node_obj'].successors()]
      __labelinput = [dv_graph.nodes[x]['label'] for x in __input]
      __labeloutput = [dv_graph.nodes[x]['label'] for x in __output]

      shuffled = False
      if len(mutation_observer["meta"][node]["input"]) >= 1:
        shuffled = list_check_equal(__labelinput, mutation_observer["meta"][node]["input"][-1])

      if shuffled:
        mutation_observer["meta"][node]["shuffled"] = True

      if len(mutation_observer["meta"][node]["input"]) < 1 or shuffled:
        if len(mutation_observer["meta"][node]["input"]) < 1 or list_check_equal(__labelinput, mutation_observer["meta"][node]["input"][-1]):
          mutation_observer["meta"][node]["input"].append(__labelinput)
        if len(mutation_observer["meta"][node]["output"]) < 1 or list_check_equal(__labeloutput, mutation_observer["meta"][node]["output"][-1]):
          mutation_observer["meta"][node]["output"].append(__labeloutput)
  except Exception as e:
    logger.debug("Err In OBSERVER %s", str(e))

    # OBSERVER END


def recursive_import(patching_pass):
  if isinstance(patching_pass, str):
    mod, patch = patching_pass.split(', ')
    if patch.startswith(tuple(ignore_prefix_passes)):
      return None
    module = importlib.import_module(IMPORT_BASE_PATH + mod)
    return getattr(module, patch)
  elif isinstance(patching_pass, list):
    resulting_pass = []
    for i in range(len(patching_pass)):
      passes = recursive_import(patching_pass[i])
      # if passes is not None:
      resulting_pass.append(passes)
    return resulting_pass
  elif isinstance(patching_pass, dict):
    resulting_passes = {}
    for key in patching_pass.keys():
      val = recursive_import(patching_pass[key])
      # if val is not None:
      resulting_passes[key] = val
    return resulting_passes
  elif patching_pass == None:
    return None
  else:
    raise Exception("Config Not proper")


__idx = 0


class MutationObserver(list):

  def __init__(self):
    self.passes = []

  def on_mutation(self, *event):
    if event[0] == 'add_node' or event[0] == 'remove_node':
      layer_info = event[1].get_layer()
      pass_to_run = []
      if optimization_passes['LayerSpecificPatchings'] is not None:
        pass_to_run = optimization_passes['LayerSpecificPatchings'].get(layer_info.type, None)
      if pass_to_run is not None:
        self.passes.extend(pass_to_run)

  def clear(self):
    self.passes = []

  def get_passes(self):
    return self.passes


def checkpoint_graph(dv_graph, name, output_dir):
  layer_pb = caffe_pb2.NetParameter()
  output_file = os.path.join(output_dir, 'params', 'checkpoints')
  for node in dv_graph.get_topsort():
    layer_pb.layer.append(node.get_layer())
  if not os.path.exists(output_file):
    os.makedirs(output_file)
  filename = os.path.join(output_dir, 'params', f'checkpoints', f'{name}.prototxt')
  dv_graph.to_dotfile(os.path.join(output_dir, 'params', 'checkpoints', f'{name}.dot'))
  with open(filename, 'w') as f:
    f.write(pb.text_format.MessageToString(layer_pb))
    f.flush()
    os.fsync(f.fileno())
  return caffe.Net(filename, caffe.TEST)


"""
After every transformation checks if relabelling of any node is required to preserve layername mapping

Args:
  dv_graph: graph undergoing transformations
  original_node_types: map of node names and its type
  original_node_names: list containing names of node in original graph
  node_name_mapping: map of node name and its latest name(changes if node is relabelled)
  inp_wts: weights
"""

def relabel_nodes_if_needed(dv_graph, original_node_types, original_node_names, node_name_mapping, inp_wts):
  new_nodes = sorted(list(dv_graph.nodes))
  original_node_name_set = set(node for node in original_node_names if original_node_types[node] not in ['Input', 'Mxnet_Take'])
  subgraph_nodes = defaultdict(list)
  subgraph_has_ara_node = defaultdict(bool)

  # for each new node, pattern match it against original
  # node to extract corresponding subgraph
  for new_node in new_nodes:
    layer_type = new_node.split("/")[0]
    curr_node_name = "/".join(new_node.split("/")[1:])
    # if current node name corresponds to non replaced node, and is not output node
    # then its a candidate for renaming
    if curr_node_name in original_node_name_set:
      if dv_graph.out_degree(new_node) != 0:
        subgraph_nodes[curr_node_name].append(new_node)
    
    # if current node name corresponds to a replaced node, then rename it as long as its output node
    elif ("_ara" in curr_node_name) and (curr_node_name.split("_ara")[0] in original_node_name_set):
      subgraph_has_ara_node[curr_node_name.split("_ara")[0]] = True
      try:
        new_layer_name = dv_graph.get_layername(curr_node_name, layer_type)
        original_name = curr_node_name.split("_ara")[0]
        if dv_graph.out_degree(new_node) == 0:
          logger.debug(f"replacing node ara {new_node} with {layer_type}/{new_layer_name}")
          dv_graph.relabel_nodes_locally({curr_node_name: original_name}, layer_type)
          if new_node in inp_wts:
            inp_wts[f"{layer_type}/{original_name}"] = inp_wts.pop(new_node)
      except Exception as e:
          logger.debug(f"error replacing node {e}")

  for original_node in subgraph_nodes:
    subgraph = subgraph_nodes[original_node]
    # if there are replaced instances of original node, rename the candidates
    if subgraph_has_ara_node[original_node] or len(subgraph) > 1:
      for new_node in subgraph:
        try:
          layer_type = new_node.split("/")[0]
          curr_node_name = "/".join(new_node.split("/")[1:])
          new_layer_name = dv_graph.get_layername(curr_node_name, layer_type)
          logger.debug(f"replacing node {new_node} with {layer_type}/{new_layer_name}")
          dv_graph.relabel_nodes_locally({curr_node_name: new_layer_name}, layer_type)
          node_name_mapping[curr_node_name] = new_layer_name
          if new_node in inp_wts:
            inp_wts[f"{layer_type}/{new_layer_name}"] = inp_wts.pop(new_node)
        except Exception as e:
          logger.info(f"error replacing node {e}")


  return dv_graph, node_name_mapping, inp_wts


def custom_copy(dict1, dict2):
  dict2.clear()
  for i in dict1:
    dict2[i] = dict1[i]


__idx = -1

prev_inp_wts = {}
pass_done = False
last_successful_idx = -1
df = pd.DataFrame(columns=['origin', 'op', 'issue', 'source'])
original_node_types = dict()
node_name_mapping = dict()


def run_transformations(caffe_input_net, dv_graph, pp_yaml, cp_only_list, caffe_cp_only_list, skip_list, fork_sftmx_argmx, cp_precision, inp_wts, dynamic, dynamic_input, platform_target="Ara1", summary_csv="None", dry_run=0, c2c_config=None, no_final_perm_opt=False, input_permuted={}, output_permuted={}, output_dir='.', tf2onnx_happened=False, inode_names=None, onode_names=None, adjust_file=None):
  global optimization_passes, ignore_prefix_passes
  global mutation_observer
  config_path = os.path.join(os.path.dirname(__file__), "TransformConfig.yaml")
  if c2c_config and os.path.exists(c2c_config):
    config_path = c2c_config
  limit = {}
  pass_config = yaml.safe_load(Path(config_path).read_text())
  ignore_prefix_passes = pass_config['Ignore']['Ara1']
  optimization_passes = recursive_import(pass_config['Patching'])
  limit = pass_config['Limit']
  fuse_act = pass_config['Fuse_act'][f'dryrun_{dry_run}'][0]
  logger.info(f"fuse_act is {fuse_act}")
  observer = MutationObserver()
  dv_graph.add_mutation_observer(observer.on_mutation)
  layer_specific_passes = []
  common_passes = []
  standardize_patching = []
  pattern_matchers = []
  name_patching = []
  target_patching = []

  if isinstance(optimization_passes['LayerSpecificPatchings'], dict):
    if isinstance([i for i in optimization_passes['LayerSpecificPatchings'].values()][0], list):
      layer_specific_passes = list(chain.from_iterable(optimization_passes['LayerSpecificPatchings'].values()))
  if isinstance(optimization_passes['FinalPatchings'], list):
    common_passes = optimization_passes['FinalPatchings']
  if isinstance(optimization_passes['Standardize'], list):
    standardize_patching = optimization_passes['Standardize']
  if isinstance(optimization_passes['PatternMatchers'], list):
    pattern_matchers = optimization_passes['PatternMatchers']
  if isinstance(optimization_passes['NamePatchings'], list):
    name_patching = optimization_passes['NamePatchings']
  if isinstance(optimization_passes['TargetPatchings'], list):
    target_patching = optimization_passes['TargetPatchings']
  LAYER_NAMES = ["loc_285_perm_caffe", "loc_286"]
  df_c2c = {C2C_TRANSFORM_NAME: [], C2C_FILE_NAME: [], C2C_ADDED_NODES: [], C2C_REMOVED_NODES: []}
  df_layer = pd.DataFrame(columns=[C2C_TRANSFORM_NAME, C2C_FILE_NAME] + LAYER_NAMES)
  c2c_logger = C2CLogger()

  def log_layer(prev_file_name, cur_file_name, transform_name):
    if len(LAYER_NAMES) == 0:
      return
    original_net = caffe_pb2.NetParameter()
    updated_net = caffe_pb2.NetParameter()
    prev_file_path = "./checkpoints/{}.prototxt".format(prev_file_name)
    cur_file_path = "./checkpoints/{}.prototxt".format(cur_file_name)

    with open(prev_file_path, "r") as f:
      s = f.read()
      pb.text_format.Merge(s, original_net)
    with open(cur_file_path, "r") as f:
      s = f.read()
      pb.text_format.Merge(s, updated_net)
    prev_layer_names = [layer.name for layer in original_net.layer]
    cur_layer_names = [layer.name for layer in updated_net.layer]
    new_row = {C2C_TRANSFORM_NAME: transform_name, C2C_FILE_NAME: cur_file_name}
    remove_layers = []

    for name in LAYER_NAMES:
      if name not in prev_layer_names and name not in cur_layer_names:
        continue
      if name in prev_layer_names and name not in cur_layer_names:
        new_row[name] = "Removed or name changed"
        remove_layers.append(name)
        continue
      prev_layer_idx = prev_layer_names.index(name)
      cur_layer_idx = cur_layer_names.index(name)
      prev_layer = original_net.layer[prev_layer_idx]
      cur_layer = updated_net.layer[cur_layer_idx]
      if prev_layer == cur_layer:
        new_row[name] = "No Change"
      else:
        new_row[name] = "Params updated. '{}' to '{}'".format(str(prev_layer), str(cur_layer))
    for rlayer in remove_layers:
      LAYER_NAMES.remove(rlayer)
    df_layer.loc[len(df_layer.index)] = new_row

  def log_transform(output_dir, prev_file_name, cur_file_name, transform_name):
    original_net = caffe_pb2.NetParameter()
    updated_net = caffe_pb2.NetParameter()
    output_path = os.path.join(output_dir, 'params', 'checkpoints')
    prev_file_path = os.path.join(output_path, "{}.prototxt".format(prev_file_name))
    cur_file_path = os.path.join(output_path, "{}.prototxt".format(cur_file_name))

    with open(prev_file_path, "r") as f:
      s = f.read()
      pb.text_format.Merge(s, original_net)
    with open(cur_file_path, "r") as f:
      s = f.read()
      pb.text_format.Merge(s, updated_net)

    original_layer_dict = {layer.name: layer for layer in original_net.layer}
    updated_layer_dict = {layer.name: layer for layer in updated_net.layer}

    original_layers = set(list(original_layer_dict.keys()))
    updated_layers = set(list(updated_layer_dict.keys()))
    if original_layers != updated_layers:
      removed_layers = original_layers - updated_layers
      new_layers = updated_layers - original_layers
      df_c2c[C2C_TRANSFORM_NAME].append(transform_name)
      df_c2c[C2C_FILE_NAME].append(cur_file_name)
      removed_layers_string = ','.join(['{}({})'.format(layer, original_layer_dict[layer].type) for layer in removed_layers])
      new_layers_string = ','.join(['{}({})'.format(layer, updated_layer_dict[layer].type) for layer in new_layers])
      df_c2c[C2C_REMOVED_NODES].append(removed_layers_string)
      df_c2c[C2C_ADDED_NODES].append(new_layers_string)

  original_node_names = []

  checkpoint_graph(dv_graph, 'original_input', output_dir)
  initialize_observer(dv_graph, original_node_names)


  def _run_passes(caffe_input_net, passes_to_run, limit, layer_param_update=True, output_dir='.'):
    global __idx
    global pass_done
    global prev_inp_wts
    global last_successful_idx
    global df
    global original_node_types
    global node_name_mapping
    nonlocal dv_graph
    nonlocal inp_wts
    idx = 0
    while len(passes_to_run) > 0:
      if idx > limit:
        return
      n = len(passes_to_run)
      for _ in range(n):
        observer.clear()
        try:
          if passes_to_run[0] == None:
            passes_to_run.pop(0)
            continue
          optim_pass = passes_to_run[0]()
          passes_to_run.pop(0)
          optim_pass.set_graph(dv_graph)
          optim_pass.set_caffe_net(caffe_input_net)
          optim_pass.set_pp_yaml(pp_yaml)
          optim_pass.set_cp_only_list(cp_only_list)
          optim_pass.set_caffe_cp_only_list(caffe_cp_only_list)
          optim_pass.set_skip_list(skip_list)
          optim_pass.set_dry_run(dry_run)
          optim_pass.set_fork_sftmx_argmx(fork_sftmx_argmx)
          optim_pass.set_cp_precision(cp_precision)
          optim_pass.set_inp_wts(inp_wts)
          optim_pass.set_dynamic(dynamic)
          optim_pass.set_fuse_act(fuse_act)
          optim_pass.set_platform_target(platform_target)
          optim_pass.set_inode_names(inode_names)
          optim_pass.set_onode_names(onode_names)
          optim_pass.set_dynamic_input(dynamic_input)
          optim_pass.set_input_permuted(input_permuted)
          optim_pass.set_output_permuted(output_permuted)
          optim_pass.set_tf2onnx_happened(tf2onnx_happened)
          optim_pass.set_no_final_perm_opt(no_final_perm_opt)
          optim_pass.set_adjust_file(adjust_file)
          optim_pass.set_mutation_observer(mutation_observer)
          prev_file_name = c2c_logger.get_file_name()
          __idx += 1
          logger.info("[D] Pass= %s : %s", __idx, str(optim_pass.__class__.__name__))
          optim_pass.do_transformation()
          passes_to_run.extend(observer.passes)
          dv_graph, node_name_mapping, inp_wts = relabel_nodes_if_needed(dv_graph, original_node_types, original_node_names, node_name_mapping, inp_wts)
          caffe_input_net_updated = checkpoint_graph(dv_graph, 'pass_{}'.format(__idx), output_dir)
          c2c_logger.set_file_name('pass_{}'.format(__idx))
          cur_file_name = c2c_logger.get_file_name()
          log_transform(output_dir, prev_file_name, cur_file_name, optim_pass.__class__.__name__)
          custom_copy(inp_wts, prev_inp_wts)
          last_successful_idx = __idx
          pass_done = True
          update_observer(dv_graph, original_node_names)

        except Exception as e:
          logger.info(f"Something went wrong in pass {e}")
          if pass_done:
            layer_pb = caffe_pb2.NetParameter()
            for node in dv_graph.get_topsort():
              layer_pb.layer.append(node.get_layer())

            if not os.path.exists('./debug/c2c'):
              os.makedirs('./debug/c2c')
            filename = './debug/c2c/fail_{}.prototxt'.format(str(__idx))
            dv_graph.to_dotfile('./debug/c2c/fail_{}.dot'.format(str(__idx)))
            with open(filename, 'w') as f:
              f.write(pb.text_format.MessageToString(layer_pb))
              f.flush()
              os.fsync(f.fileno())
          if pass_done:
            inp_wts = copy.deepcopy(prev_inp_wts)
            filename = os.path.join(output_dir, 'params', 'checkpoints', f'pass_{last_successful_idx}.prototxt')
            caffe_input_net = caffe.Net(filename, caffe.TEST)
            for i in caffe_input_net.params:
              for idx, j in enumerate(caffe_input_net.params[i]):
                try:
                  j.data.flat = inp_wts[i][idx]
                except Exception as e:
                  logger.error("weights copy problem", exc_info=e)

            dv_graph = caffe_to_graph(filename)

            row = {}
            row['op'] = type(optim_pass).__name__
            row['issue'] = str(e)
            row['source'] = __idx
            row['origin'] = 'DVNET_Optimization'
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
          continue
        if caffe_input_net is None:
          logger.error("Invalid proto.")
          exit(1)

        for param_name in caffe_input_net.params:
          if param_name in caffe_input_net_updated.params:
            caffe_input_net_updated.params[param_name][:] =\
                caffe_input_net.params[param_name]
        caffe_input_net = caffe_input_net_updated
      idx += 1
      unique_set = set()
      unique = []
      for i in passes_to_run:
        if i not in unique_set:
          unique.append(i)
          unique_set.add(i)
      passes_to_run = list(unique)

  _run_passes(caffe_input_net, pattern_matchers, int(limit['pattern_matchers'][0]), output_dir=output_dir)
  _run_passes(caffe_input_net, standardize_patching, int(limit['standardize_patching'][0]), output_dir=output_dir)
  _run_passes(caffe_input_net, layer_specific_passes, int(limit['layer_specific_passes'][0]), output_dir=output_dir)
  _run_passes(caffe_input_net, common_passes, int(limit['common_passes'][0]), output_dir=output_dir)
  if not dry_run:
    _run_passes(caffe_input_net, name_patching, int(limit['name_patching'][0]), output_dir=output_dir)
  _run_passes(caffe_input_net, target_patching, int(limit['target_patching'][0]), layer_param_update=False, output_dir=output_dir)
  output_file = os.path.join(output_dir, 'stats', '.mo.json')
  mo = json.dumps(mutation_observer)
  with open(output_file, 'w') as f:
    f.write(mo)
  df = pd.DataFrame.from_dict(df_c2c)
  df.to_csv("transforms_c2c.csv")
  return dv_graph, mutation_observer, inp_wts
