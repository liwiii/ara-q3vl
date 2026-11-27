'''
Copyright (c) 2025, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

import yaml
import os
import numpy as np
from copy import deepcopy
from typing import List
from padding_util import next_multiple_of, align, pad_to_shape
from utils import quantize, get_scale_tensor_float, create_config, create_run_py, create_sdk_config
from constants import *
import json
from pathlib import Path
#TODO: in long term, MULTI_ROPE seems to be better default.
#      After it is validated to be stable in stage_r2.1,
#      update the default.
DEFAULT_ROPE_TYPE="IMAGE"

class Model:
    def __init__(self, config, hf_config, proto_utils, onnx_model, caffe, output_dir, sdk = False):
        self.config = config
        self.hf_config = hf_config
        self.proto_utils = proto_utils
        self.onnx_model = onnx_model
        self.caffe = caffe
        self.output_dir = output_dir
        self.sdk = sdk
        self.get_weights_config()
        self.get_model_info()
        self.caffe_model_layer_idx_map = {}
        self.rope_onnx_initializer_names = {'sin': SIN_CACHE, 'cos':COS_CACHE}

        # Used for Qwen3/3VL
        self.use_qk_norm = False

    def get_model_info(self):  
        self.architecture = self.hf_config.get('architectures')[0]
        self.attention_bias = self.hf_config.get('attention_bias', False)
        self.bos_token_id = self.hf_config.get('bos_token_id') or self.hf_config.get('text_config', {}).get('bos_token_id')
        self.eos_token_id = self.hf_config.get('eos_token_id') or self.hf_config.get('text_config', {}).get('eos_token_id')
        #TODO: #Llama3.1 and Llama3.2 have multiple end of sequence tokens. We pick the last one, but at infer time it need not be the last one
        if (isinstance(self.eos_token_id, list)): 
            self.eos_token_id = self.eos_token_id[-1]
        self.hidden_act = self.hf_config.get('hidden_act') or self.hf_config.get('text_config', {}).get('hidden_act')
        self.hidden_size = self.hf_config.get('hidden_size') or self.hf_config.get('text_config', {}).get('hidden_size')
        self.intermediate_size = self.hf_config.get('intermediate_size') or self.hf_config.get('text_config', {}).get('intermediate_size')
        self.max_position_embeddings = self.config.get('prmpt_context_size', 4096)
        self.mlp_bias = self.hf_config.get('mlp_bias', False)
        self.model_type = self.hf_config.get('model_type')
        self.num_attention_heads = self.hf_config.get('num_attention_heads') or self.hf_config.get('text_config', {}).get('num_attention_heads')
        self.head_dim = self.hf_config.get('head_dim') or self.hf_config.get('text_config', {}).get('head_dim') or (self.hidden_size // self.num_attention_heads)        
        self.num_hidden_layers = self.hf_config.get('num_hidden_layers') or self.hf_config.get('text_config', {}).get('num_hidden_layers')
        self.num_key_value_heads = self.hf_config.get('num_key_value_heads') or self.hf_config.get('text_config', {}).get('num_key_value_heads')
        self.rms_norm_eps = self.hf_config.get('rms_norm_eps') or self.hf_config.get('text_config', {}).get('rms_norm_eps')
        # Construction of LMHEAD node checks that vocab size aligns with 64.  
        self.vocab_size = next_multiple_of(self.hf_config.get('vocab_size') or self.hf_config.get('text_config', {}).get('vocab_size'), 64)
        self.prmpt_prc_sizes = self.config.get('prmpt_prc_sizes', [128])
        self.prmpt_context_size = self.config.get('prmpt_context_size', 4096)
        self.context_size = self.prmpt_context_size
        self.tkn_gen_sizes = self.config.get('tkn_gen_sizes', [self.context_size])
        self.specd_size = self.config.get('specd_size', 1)        
        self.num_mcs_shards = self.config.get('num_mcs_shards', 1)
        self.run_suffix = self.config.get('run_suffix', 'output')
        self.token_size = self.prmpt_prc_sizes[0]
        self.license_key = self.config.get('license_key', None)
        # builder has communicated rope type to model through kinara_run_config
        if self.config.get('is_vlm', None):
            self.vl_rope_type = str(self.config.get('rope', {'vl_rope_type': DEFAULT_ROPE_TYPE})['vl_rope_type'])
            if self.vl_rope_type == "MULTI_ROPE":
                self.num_video_ropes = self.config.get('num_video_ropes')  
                self.img_tkn_id = self.config.get('img_tkn_id')
                self.video_tkn_id = self.config.get('video_tkn_id')
        else:
            self.vl_rope_type = None 
        print("vl rope type is", self.vl_rope_type)   
        self.set_attention_bias()
        self.print_model_info()
        self.get_activations_config()

    def get_activations_config(self):
        self.qxk_out_scale_32_layers  = self.config.get('qxk_out_scale_32_layers', [])
        self.attention_scale_layers  = self.config.get('attention_scale_layers', [])
        self.use_8bit_prompt = self.config.get('use_8bit_prompt', False)



    def dump_model_gen_yaml(self, filepath):
        # args used by llm_model_constructor
        yaml_dict = {}
        yaml_dict["hidden_size"] = self.hidden_size
        yaml_dict["vocab_size"] = self.vocab_size
        yaml_dict["kv_hidden_size"] = self.num_key_value_heads*self.head_dim
        yaml_dict["num_attention_heads"] = self.num_attention_heads
        yaml_dict["in_precision"] = 8
        yaml_dict["ou_precision"] = 32
        yaml_dict["bos_token_id"] = self.bos_token_id
        yaml_dict["eos_token_id"] = self.eos_token_id
        yaml_dict["pad_token_id"] = self.eos_token_id
        yaml_dict["context_size"] = self.context_size
        yaml_dict["specd_size"] = self.specd_size
        yaml_dict["num_mcs_shards"] = self.num_mcs_shards
        yaml_dict["model_type"] = self.model_type

        if self.vl_rope_type == "MULTI_ROPE":
            yaml_dict['num_video_ropes'] =  self.num_video_ropes
            yaml_dict['img_tkn_id'] = self.img_tkn_id
            yaml_dict['video_tkn_id'] = self.video_tkn_id
        with open(filepath, "w") as file:
                yaml.dump(yaml_dict, file) 
        
    def set_attention_bias(self):
        if not self.attention_bias:
            initializers = self.onnx_model.graph.initializer
            for initializer in initializers:
                if (QKV_BIAS in initializer.name):
                    self.attention_bias = True


    def print_model_info(self):
        print(f"Architecture: {self.architecture}")
        print(f"Attention Bias: {self.attention_bias}")
        print(f"BOS Token ID: {self.bos_token_id}")
        print(f"EOS Token ID: {self.eos_token_id}")
        print(f"Hidden Activation: {self.hidden_act}")
        print(f"Hidden Size: {self.hidden_size}")
        print(f"Intermediate Size: {self.intermediate_size}")
        print(f"Max Position Embeddings: {self.max_position_embeddings}")
        print(f"MLP Bias: {self.mlp_bias}")
        print(f"Model Type: {self.model_type}")
        print(f"Number of Attention Heads: {self.num_attention_heads}")
        print(f"Number of Hidden Layers: {self.num_hidden_layers}")
        print(f"Number of Key Value Heads: {self.num_key_value_heads}")
        print(f"RMS Norm Epsilon: {self.rms_norm_eps}")
        print(f"Vocab Size: {self.vocab_size}")
        print(f"Prompt Processing Sizes: {self.prmpt_prc_sizes}")
        print(f"Prompt Context Size: {self.prmpt_context_size}")
        print(f"Token Generation Sizes: {self.tkn_gen_sizes}")
        print(f"Specd size: {self.specd_size}")
        print(f"Num shards for Multichip sharding: {self.num_mcs_shards}")
        print(f"Run Suffix: {self.run_suffix}")
    
 
    def dump_embeddings_table(self):
      embedding_data = None
      initializers = self.onnx_model.graph.initializer
      for initializer in initializers:
          if initializer.name == EMBEDDINGS:
              print(f"Embedding Table: {initializer.name}")
              print(f"  Shape: {initializer.dims}")
              print(f"  Data Type: {initializer.data_type}")
              embedding_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
          if initializer.name == LMHEAD_WEIGHT:
              print(f"Lmhead weight (for fallback when tie_word_embeddings and embed_tokens is missing): {initializer.name}")
              print(f"  Shape: {initializer.dims}")
              print(f"  Data Type: {initializer.data_type}")
              initializer = align(initializer, 64)
              print(f"Post padding:  Shape: {initializer.dims}")
              # Just like make_lmhead_caffemodel
              raw_weights = np.frombuffer(initializer.raw_data, dtype=np.float32)
              raw_weights_transposed_flattened = np.transpose(raw_weights.reshape(initializer.dims)).flatten()
              lm_head_data = raw_weights_transposed_flattened


      # tie_word_embeddings: True means that embedding_data and lm_head_data are same. Example.
      # Llama3o2-3b-Instruct-w4g64: old ORT-GenAI does embed_tokens --copy--> lm_head : into onnx
      # Qwen3VL: This SDK model.py does lm_head --copy--> embed_tokens : into caffemodel

      if self.config.get('tie_word_embeddings', False) and embedding_data is None:
        print(f"Copying {LMHEAD_WEIGHT} --> {EMBEDDINGS} for tie_word_embeddings.")
        embedding_data = lm_head_data

      embedding_table, embedding_table_scale = quantize(embedding_data)
      return (embedding_data, embedding_table, embedding_table_scale)


                
    def dump_rope_table(self):
        initializers = self.onnx_model.graph.initializer
        sin_found = False
        cos_found = False
        for initializer in initializers:
            if initializer.name == self.rope_onnx_initializer_names['sin']:
                print(f"Sin Table: {initializer.name}")
                print(f"  Shape: {initializer.dims}")
                sin_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
                print(f"sin_data from onnx {sin_data.shape}")
                # Alignment for Qwen3VL-LLM needed
                sin_data =  pad_to_shape(sin_data, initializer.dims, (next_multiple_of(initializer.dims[0], self.max_position_embeddings), next_multiple_of(initializer.dims[1], self.head_dim // 2)))
                print(f"sin_data post alignment {sin_data.shape}")
                sin_found = True 
            if initializer.name == self.rope_onnx_initializer_names['cos']:
                print(f"Cos Table: {initializer.name}")
                print(f"  Shape: {initializer.dims}")
                cos_data = np.frombuffer(initializer.raw_data, dtype=np.float32)
                print(f"cos_data from onnx {cos_data.shape}")
                # Alignment for Qwen3VL-LLM needed
                cos_data =  pad_to_shape(cos_data, initializer.dims, (next_multiple_of(initializer.dims[0], self.max_position_embeddings), next_multiple_of(initializer.dims[1], self.head_dim // 2)))
                print(f"cos_data post alignment {cos_data.shape}")
                cos_found = True
        if sin_found and cos_found:
            #              C                  H                  W
            # cos -> [num_rope_tables, max_position_embeddings, head_dim]
            # sin -> [num_rope_tables, max_position_embeddings, head_dim]
            # cos_first_table, sin_first_table, cos_second_table, sin_second_table ...
            '''
              \ num_rope_table
                \    head_dim <-- axis of concat
                  \ __________
            max   | ----------
            pos   |
            embed |
                  |

                  cos

              \ num_rope_table
                \    head_dim <-- axis of concat
                  \ __________
            max   |
            pos   |
            embed |
                  |

            sin

            \ num_rope_table
               \
                \    head_dim*2 <-- axis of concat
                  \ _______________________
            max   | cos        sin
            pos   | cos        sin
            embed |
                  |

                  cos + sin
            '''
            sin_data = sin_data.reshape((-1, self.max_position_embeddings, self.head_dim // 2))
            cos_data = cos_data.reshape((-1, self.max_position_embeddings, self.head_dim // 2))
            sin_data_replicated = np.concatenate([sin_data, sin_data], axis=-1)
            cos_data_replicated = np.concatenate([cos_data, cos_data], axis=-1)
            rope_table = np.concatenate([cos_data_replicated, sin_data_replicated], axis=-1)
            rope_table_quantized = np.clip(rope_table*128, -128, 127).astype(np.int8)
            return (rope_table, rope_table_quantized)
        else:
            raise RuntimeError("RoPE tables not found in the ONNX model. Please check the model.")

    def dump_assets(self):
        embedding_data, embedding_table, embedding_table_scale = self.dump_embeddings_table()
        rope_table, rope_table_quantized = self.dump_rope_table()
        assets_dir =  os.path.join(self.output_dir,"assets")
        os.makedirs(assets_dir, exist_ok=True)
        embedding_table_scale.tofile(os.path.join(assets_dir, "embedding_table_scale.bin"))
        embedding_table.tofile(os.path.join(assets_dir, "embedding_table.bin"))
        embedding_data.tofile(os.path.join(assets_dir, "embedding_table_float.bin"))
        rope_table_quantized.tofile(os.path.join(assets_dir, "rope_table.bin"))
        rope_table.tofile(os.path.join(assets_dir, "rope_table_float.bin"))
        self.dump_model_gen_yaml(os.path.join(assets_dir, "model_gen.yaml"))
        with open(os.path.join(assets_dir,"weight_config.json"), "w", encoding="utf-8") as f:
            json.dump(self.weight_keys, f)

        


    def get_weights_config(self):
        self.weight_dict = {}
        def scale_offset_values(name, config, out_dict):
            if name in config:
                out_dict[name]=config[name]
        scale_offset_values(DOWN_PROJ, self.config["weight_config"], self.weight_dict)
        scale_offset_values(GATE_PROJ, self.config["weight_config"], self.weight_dict)
        scale_offset_values(UP_PROJ, self.config["weight_config"], self.weight_dict)
        scale_offset_values(O_PROJ, self.config["weight_config"], self.weight_dict)
        scale_offset_values(K_PROJ, self.config["weight_config"], self.weight_dict)
        scale_offset_values(Q_PROJ, self.config["weight_config"], self.weight_dict)
        scale_offset_values(V_PROJ, self.config["weight_config"], self.weight_dict)
        scale_offset_values(LMHEAD, self.config["weight_config"], self.weight_dict) 
        weight_dict_2 = {}
        self.weight_config = {}
        for layer in self.weight_dict.keys():
            layer_dict = self.weight_dict[layer]
            if(len(layer_dict)):
                for precision in layer_dict.keys():
                    indices = layer_dict[precision]
                    if(len(indices)):
                        for idx in indices:
                            _key = layer + '_' + precision + '_'
                            if(int(idx) not in weight_dict_2.keys()):
                                weight_dict_2[int(idx)] = ''
                            weight_dict_2[int(idx)] = weight_dict_2[int(idx)] + _key
                            if int(idx) not in self.weight_config.keys():
                                self.weight_config[int(idx)] = {}
                            self.weight_config[int(idx)][layer] = precision
        self.weight_keys = {}
        for key, value in weight_dict_2.items():
          if value not in self.weight_keys.keys():
            self.weight_keys[value] = []
          self.weight_keys[value].append(key)
        print("Weight Keys: ")
        print(self.weight_keys)
    
    def make_named_shape(self, dim_names: List[str], dim_vals: List[int]):
        assert(len(dim_names) == len(dim_vals))
        NamedDim = self.proto_utils['NamedDim']
        out_list = []
        for idx in range(len(dim_names)):
            out_list.append(NamedDim(dim_names[idx], dim_vals[idx]))
        return out_list

    def make_reshape(self, graph, input_nodes, output_node, out_group_axis_name: str, out_group_size):
        scale_node = []
        NamedDim = self.proto_utils['NamedDim']
        for dim in output_node:
            if dim.dim_name == out_group_axis_name:
                new_dim = dim.dim // out_group_size
                scale_node.append(NamedDim(dim.dim_name, new_dim))
            else:
                scale_node.append(dim)
        output_nodes = [output_node, scale_node]
        return self.proto_utils['simple_reshape'](graph, input_nodes, output_nodes, out_group_axis_name=out_group_axis_name)

    def make_lmhead(self):
        lmhead_out = os.path.join(self.output_dir, f"lmhead/split_{self.num_hidden_layers}")
        os.makedirs(lmhead_out, exist_ok=True)
        proto_out = os.path.join(lmhead_out, "lmhead.prototxt")
        out_caffemodel = os.path.join(lmhead_out, "lmhead.caffemodel") 
        out_so = os.path.join(lmhead_out, "lmhead_so.yaml")
        img_out = os.path.join(lmhead_out, "imgs") 
        graph = self.proto_utils['DAG']()
        self.make_lmhead_graph(graph)
        self.get_proto(graph, save_to=proto_out)
        self.make_lmhead_caffemodel(proto_out, out_caffemodel)
        self.get_so_yaml(graph,out_so, self.num_hidden_layers)
        self.make_inputs(proto_out, img_out)
        if self.sdk:
            config_out = os.path.join(lmhead_out, "dvrun_config.yaml")
            create_sdk_config(self.caffe, proto_out, out_caffemodel, img_out, out_so, config_out, self.license_key)
        else:
            config_out = os.path.join(lmhead_out, "config_ara2.yaml")
            create_config(self.caffe, proto_out, out_caffemodel, img_out, out_so, config_out )
            create_run_py(lmhead_out)


    def make_lmhead_graph(self, graph): 
        
        hidden_state_named_shape = self.make_named_shape([DIM0, DIM1, TOKEN, HIDDEN], [1,1,self.specd_size,self.hidden_size]) 
        input = self.proto_utils['Input'](INPUT, graph,hidden_state_named_shape,3,64, True)
        rms_norm = self.proto_utils['LayerNorm'](INPUT_RMS, graph, 3, 64, False, False, self.rms_norm_eps,True)
        lm_head = self.proto_utils['Inrp'](LMHEAD, graph, self.vocab_size, False,1,64,True)
        
        rms_out = rms_norm(input(), derive_name_from_layer_name=True)
        if self.specd_size == 1:
            # DIM0, DIM1, TOKEN, HIDDEN -> DIM0, HIDDEN, DIM2, TOKEN
            rms_out_reshape_dims = self.make_named_shape([DIM0, HIDDEN, DIM2, TOKEN], [1,self.hidden_size, 1, 1])
        else:
            # DIM0, DIM1, TOKEN, HIDDEN -> TOKEN, HIDDEN, DIM0, DIM2, 
            rms_out_reshape_dims = self.make_named_shape([TOKEN, HIDDEN, DIM0, DIM2], [self.specd_size, self.hidden_size, 1, 1])            
        reshape = self.make_reshape(graph,rms_out, rms_out_reshape_dims, HIDDEN, 64)
        out = lm_head(reshape, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        
        ### Get caffemodel
    
    def make_lmhead_caffemodel(self, proto_out, out_caffemodel):
        initializers = self.onnx_model.graph.initializer
        lm_head_data = None
        for initializer in initializers:
            if initializer.name == LMHEAD_WEIGHT:
                print(f"LM Head: {initializer.name}")
                print(f"  Shape: {initializer.dims}")
                print(f"  Data Type: {initializer.data_type}")
                
                # This is needed because sometimes, the initializer dimensions are
                # not a multiple of 64. e.g. https://kinara.atlassian.net/browse/CP-877
                # We are choosing to do on the fly alignment here, instead of emitting
                # an aligned .onnx model from previous compiler phase so that the .onnx
                # model stays in sync with original input model's dimensions.
                initializer = align(initializer, 64)
                print(f"Post padding:  Shape: {initializer.dims}")

                raw_weights = np.frombuffer(initializer.raw_data, dtype=np.float32)
                raw_weights_transposed_flattened = np.transpose(raw_weights.reshape(initializer.dims)).flatten()
                lm_head_data = raw_weights_transposed_flattened
                print(f" New shape: {np.transpose(raw_weights.reshape(initializer.dims)).shape}")

            if  LMHEAD_RMS in initializer.name:
                print(f"RMS Norm: {initializer.name}")
                print(f"  Shape: {initializer.dims}")
                print(f"  Data Type: {initializer.data_type}")
                raw_weights = np.frombuffer(initializer.raw_data, dtype=np.float32)
                raw_weights_transposed_flattened = np.transpose(raw_weights.reshape(initializer.dims)).flatten()
                rms_norm_data = raw_weights_transposed_flattened
                print(f" New shape: {np.transpose(raw_weights.reshape(initializer.dims)).shape}") 

        if lm_head_data is None:
            raise RuntimeError("LM Head not found in the ONNX model. Please check the model.")
            
        source_net = self.caffe.Net(proto_out, self.caffe.TEST)
        for param_name in source_net.params.keys():
            if(param_name == LMHEAD):
                param = source_net.params[param_name][0]
                param_data = lm_head_data.reshape(param.data.shape)
                param.data[...] = param_data
                print(f"LM Head parameters set for {LMHEAD}")
            if (param_name == INPUT_RMS):
                param = source_net.params[param_name][0]
                param_data = rms_norm_data.reshape(param.data.shape)
                param.data[...] = param_data
                print(f"RMS Norm parameters set for {INPUT_RMS}")
        source_net.save_hdf5(out_caffemodel)
        print(f"LM Head Caffe model saved to {out_caffemodel}")

    def make_attention_prompt(self, graph, layer_idx = 0):

        # It will handle the attention layer conversion
        kv_hidden_size = self.num_key_value_heads*self.head_dim
        
        ######################### Initialization #########################
        #Named shapes
       
        input_shape = self.make_named_shape([DIM0, DIM1, TOKEN, HIDDEN],[1,1,self.token_size,self.hidden_size])
        kv_input_shape = self.make_named_shape([DIM0, DIM1, TOKEN_CACHE, HIDDEN],[1,1,self.prmpt_context_size,kv_hidden_size])
        rope_input_shape = self.make_named_shape([DIM0, DIM1, TOKEN, SIN_COS],[1,1,self.token_size, self.head_dim * 2])
        valid_till_shape = self.make_named_shape([DIM0, DIM1, DIM2, DIM3],[1,1,1,1]) 

    
        #inputs
        input_ids = self.proto_utils['Input'](INPUT, graph,input_shape,-1,64, True)
        past_keys = self.proto_utils['Input'](PAST_KEYS, graph, kv_input_shape,-1,64, True, input_type='K')
        past_values = self.proto_utils['Input'](PAST_VALUES, graph, kv_input_shape,2,64, True, input_type='V')
        position_ids = self.proto_utils['Input'](POSTION_EMBEDS, graph, rope_input_shape, 1, 1, False, input_type='R')
        valid_till = self.proto_utils['Input'](VALID_TILL, graph, valid_till_shape, 1, 1, False)
        #layernorms
        input_rms_norm = self.proto_utils['LayerNorm'](INPUT_RMS, graph, -1, 64, False, False, self.rms_norm_eps,True)

        # Q/K normalization layers (for Qwen3)
        # out_group_axis=-1: after reshape to channels=1, HEAD_DIM will be at the last axis
        if self.use_qk_norm:
            q_norm = self.proto_utils['LayerNorm'](Q_NORM, graph, -1, 64, False, False, self.rms_norm_eps, True)
            k_norm = self.proto_utils['LayerNorm'](K_NORM, graph, -1, 64, False, False, self.rms_norm_eps, True)

        # Post-attention normalization (for Gemma possibly)


        # Projections
        attention_scale = 0
        if layer_idx in self.attention_scale_layers and not self.use_qk_norm:
            attention_scale = 1/(self.head_dim**0.25)
        k_proj = self.proto_utils['Conv'](K_PROJ, graph, kv_hidden_size, self.attention_bias, [1, 1], [1,1], 1, 64,HIDDEN, True, attention_scale = attention_scale )
        q_proj = self.proto_utils['Conv'](Q_PROJ, graph, self.num_attention_heads*self.head_dim, self.attention_bias, [1, 1], [1,1], 1, 64,HIDDEN, True, attention_scale = attention_scale )
        v_proj = self.proto_utils['Conv'](V_PROJ, graph, kv_hidden_size, self.attention_bias, [1, 1], [1,1], -1, 64,HIDDEN, True )
        o_proj = self.proto_utils['Conv'](O_PROJ, graph, self.hidden_size, False, [1, 1], [1,1], 1, 64,HIDDEN, True )

        # Rope
        permuted_position_ids = self.proto_utils['permute'](graph,position_ids(), (TOKEN, SIN_COS))
        k_rope = self.proto_utils['Rope'](K_ROPE, graph, permuted_position_ids, 2, 64, is_dynamic=True)
        q_rope = self.proto_utils['Rope'](Q_ROPE, graph, permuted_position_ids, 2, 64, is_dynamic=True)

        # BatchDots
        q_kt_mult = 1/(self.head_dim**0.5)
        if layer_idx in self.attention_scale_layers and not self.use_qk_norm:
            q_kt_mult = 1
        
        
        QxKt = self.proto_utils['BatchDot'](Q_KT, graph, -1, 1024, q_kt_mult,True)
        PxV = self.proto_utils['BatchDot'](P_V, graph, -1, 64, 1,True)

        # SoftMax
        Attn = self.proto_utils['AttentionSoftmax'](SFMX, graph, -1, 64, True)

        ############### Attention Layer Construction ################
        input_ids_out = input_ids()
        rms_norm_out = input_rms_norm(input_ids_out, derive_name_from_layer_name=True)
        rms_norm_out_permuted = self.proto_utils['permute'](graph, rms_norm_out, (TOKEN, HIDDEN))
        # DIM0, DIM1, HIDDEN, TOKEN -> DIM0, HIDDEN, DIM1, TOKEN
        rms_out_reshape_dims = self.make_named_shape([DIM0, HIDDEN, DIM1, TOKEN], [1,self.hidden_size, 1, self.token_size])
        proj_input = self.make_reshape(graph, rms_norm_out_permuted, rms_out_reshape_dims, HIDDEN, 64)
        k_proj_input = deepcopy(proj_input)
        q_proj_input = deepcopy(proj_input)
        v_proj_input = deepcopy(proj_input)
        ############## K Proj ################
        k_proj_out = k_proj(k_proj_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        # DIM0, HIDDEN, DIM1, TOKEN -> DIM0, NUM_HEADS, HEAD_DIM, TOKEN
        k_out_reshape_dims = self.make_named_shape([DIM0, NUM_HEADS, HEAD_DIM, TOKEN], [1,self.num_key_value_heads, self.head_dim, self.token_size])

        ## error
        k_proj_out_mha = self.make_reshape(graph, k_proj_out, k_out_reshape_dims, HEAD_DIM, 64)

        # Apply K normalization if enabled (Qwen3)
        # OCheck fix: reshape to channels=1 before norm, reshape back after
        # Current shape: [DIM0=1, NUM_HEADS, HEAD_DIM, TOKEN] (C=NUM_HEADS, fails OCheck)
        # Fix: permute HEAD_DIM<->TOKEN, reshape to [1,1,NUM_HEADS*TOKEN,HEAD_DIM], norm (C=1), reshape back, permute back
        if self.use_qk_norm:
            # Step 1: Permute HEAD_DIM <-> TOKEN: [DIM0, NUM_HEADS, HEAD_DIM, TOKEN] -> [DIM0, NUM_HEADS, TOKEN, HEAD_DIM]
            k_proj_permuted = self.proto_utils['permute'](graph, k_proj_out_mha, (HEAD_DIM, TOKEN))
            # Step 2: Reshape to channels=1: [1, NUM_HEADS, TOKEN, HEAD_DIM] -> [1, 1, NUM_HEADS*TOKEN, HEAD_DIM]
            k_prenorm_dims = self.make_named_shape(
                [DIM0, DIM1, DIM2, HEAD_DIM],
                [1, 1, self.num_key_value_heads * self.token_size, self.head_dim])
            k_proj_prenorm = self.make_reshape(graph, k_proj_permuted, k_prenorm_dims, HEAD_DIM, 64)
            # Step 3: Apply K normalization (channels=1, axis=-1 normalizes HEAD_DIM)
            k_proj_normed_flat = k_norm(k_proj_prenorm, derive_name_from_layer_name=True)
            # Step 4: Reshape back: [1, 1, NUM_HEADS*TOKEN, HEAD_DIM] -> [1, NUM_HEADS, TOKEN, HEAD_DIM]
            k_postnorm_dims = self.make_named_shape(
                [DIM0, NUM_HEADS, TOKEN, HEAD_DIM],
                [1, self.num_key_value_heads, self.token_size, self.head_dim])
            k_proj_postnorm = self.make_reshape(graph, k_proj_normed_flat, k_postnorm_dims, HEAD_DIM, 64)
            # Step 5: Permute back: [DIM0, NUM_HEADS, TOKEN, HEAD_DIM] -> [DIM0, NUM_HEADS, HEAD_DIM, TOKEN]
            k_proj_out_mha = self.proto_utils['permute'](graph, k_proj_postnorm, (TOKEN, HEAD_DIM))


        k_rope_out = k_rope(k_proj_out_mha)
        # DIM0, NUM_HEADS, HEAD_DIM, TOKEN -> DIM0, DIM1, HIDDEN, TOKEN
        k_rope_out_reshape_dims = self.make_named_shape([DIM0, DIM1, HIDDEN, TOKEN], [1,1,kv_hidden_size,self.token_size ])
        k_rope_out_reshaped = self.make_reshape(graph, k_rope_out, k_rope_out_reshape_dims, HIDDEN, 64)
        
        present_keys = self.proto_utils['permute'](graph, k_rope_out_reshaped, (TOKEN, HIDDEN), ['K','KS'])

        ############## V Proj ################
        v_proj_out = v_proj(v_proj_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        # DIM0, HIDDEN, DIM1, TOKEN -> DIM0, DIM1, HIDDEN, TOKEN
        v_out_reshape_dims = self.make_named_shape([DIM0, DIM1, HIDDEN, TOKEN], [1, 1, kv_hidden_size, self.token_size])
        v_proj_out_reshaped = self.make_reshape(graph, v_proj_out, v_out_reshape_dims, TOKEN, 64)
        present_values = self.proto_utils['permute'](graph, v_proj_out_reshaped, (TOKEN, HIDDEN), ['V','VS'])
        ############## Q Proj ################
        q_proj_out = q_proj(q_proj_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)

        # DIM0, HIDDEN, DIM1, TOKEN -> DIM0, NUM_HEADS, HEAD_DIM, TOKEN
        q_out_reshape_dims = self.make_named_shape([DIM0, NUM_HEADS, HEAD_DIM, TOKEN], [1,self.num_attention_heads, self.head_dim, self.token_size])
        q_proj_out_mha = self.make_reshape(graph, q_proj_out, q_out_reshape_dims, HEAD_DIM, 64)

        if self.use_qk_norm:
            # Step 1: Permute HEAD_DIM <-> TOKEN
            q_proj_permuted = self.proto_utils['permute'](graph, q_proj_out_mha, (HEAD_DIM, TOKEN))
            # Step 2: Reshape to channels=1
            q_prenorm_dims = self.make_named_shape(
                [DIM0, DIM1, DIM2, HEAD_DIM],
                [1, 1, self.num_attention_heads * self.token_size, self.head_dim])
            q_proj_prenorm = self.make_reshape(graph, q_proj_permuted, q_prenorm_dims, HEAD_DIM, 64)
            # Step 3: Apply Q normalization
            q_proj_normed_flat = q_norm(q_proj_prenorm, derive_name_from_layer_name=True)
            # Step 4: Reshape back
            q_postnorm_dims = self.make_named_shape(
                [DIM0, NUM_HEADS, TOKEN, HEAD_DIM],
                [1, self.num_attention_heads, self.token_size, self.head_dim])
            q_proj_postnorm = self.make_reshape(graph, q_proj_normed_flat, q_postnorm_dims, HEAD_DIM, 64)
            # Step 5: Permute back
            q_proj_out_mha = self.proto_utils['permute'](graph, q_proj_postnorm, (TOKEN, HEAD_DIM))

        q_rope_out = q_rope(q_proj_out_mha)

        ############ QxKt ################
        past_keys_permuted = self.proto_utils['permute'](graph, past_keys(), (TOKEN_CACHE, HIDDEN))
        # DIM0, DIM1, HIDDEN, TOKEN_CACHE -> DIM0, NUM_HEADS, HEAD_DIM, TOKEN_CACHE
        pk_out_reshape_dims = self.make_named_shape([DIM0, NUM_HEADS, HEAD_DIM, TOKEN_CACHE], [1,self.num_key_value_heads, self.head_dim, self.prmpt_context_size])
        past_keys_reshaped = self.make_reshape(graph, past_keys_permuted, pk_out_reshape_dims, HEAD_DIM, 64)
        QxKt_out = QxKt(q_rope_out + past_keys_reshaped, derive_name_from_layer_name=True)
        ############ Softmax ################
        attn_out = Attn(QxKt_out + valid_till(), axis = -1)
        attn_out_permuted = self.proto_utils['permute'](graph, attn_out, (TOKEN, TOKEN_CACHE))

        ################ PxV ################
        # DIM0, DIM1, TOKEN_CACHE, HIDDEN -> DIM0, TOKEN_CACHE, NUM_HEADS, HEAD_DIM
        pv_out_reshape_dims = self.make_named_shape([DIM0, TOKEN_CACHE, NUM_HEADS, HEAD_DIM], [1, self.prmpt_context_size, self.num_key_value_heads, self.head_dim])
        past_values_reshaped = self.make_reshape(graph, past_values(), pv_out_reshape_dims, TOKEN_CACHE, 64)

        past_values_permuted = self.proto_utils['permute'](graph, past_values_reshaped, (TOKEN_CACHE, NUM_HEADS))
        PxV_out = PxV(attn_out_permuted + past_values_permuted, derive_name_from_layer_name=True)
        PxV_out_permuted = self.proto_utils['permute'](graph, PxV_out, (TOKEN, HEAD_DIM))
        # DIM0, NUM_HEADS, HEAD_DIM, TOKEN -> DIM0, HIDDEN, DIM1, TOKEN

        # Use num_attention_heads * head_dim (not hidden_size) because PxV output has
        # all Q attention heads, and head_dim may differ from hidden_size/num_attention_heads
        # (e.g. Qwen3-0.6B: hidden_size=1024, but num_attention_heads*head_dim=16*128=2048)
        q_hidden_size = self.num_attention_heads * self.head_dim
        pxv_out_reshape_dims = self.make_named_shape([DIM0, HIDDEN, DIM1, TOKEN], [1, q_hidden_size, 1, self.token_size])

        PxV_out_reshaped = self.make_reshape(graph, PxV_out_permuted, pxv_out_reshape_dims, HIDDEN, 64)
        ############# O_proj ###############
        o_proj_out =  o_proj(PxV_out_reshaped, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        # DIM0, HIDDEN, DIM1, TOKEN -> DIM0, DIM1, HIDDEN, TOKEN
        o_out_reshape_dims = self.make_named_shape([DIM0, DIM1, HIDDEN, TOKEN], [1, 1, self.hidden_size, self.token_size])
        o_proj_out_reshaped = self.make_reshape(graph, o_proj_out, o_out_reshape_dims, HIDDEN, 64)

        o_proj_out_permuted = self.proto_utils['permute'](graph, o_proj_out_reshaped, (HIDDEN, TOKEN))

        return o_proj_out_permuted + input_ids_out
    
    def make_attention_mlp_connection_prompt(self, graph, input_nodes):
        attention_residual = self.proto_utils['Eltwise'](ATTENTION_RESIDUAL, graph, -1, 64,  self.proto_utils['EltwiseOperation'].SUM, True)
        residual_out = attention_residual(input_nodes)
        return residual_out
    
    def make_mlp_prompt(self, graph, input_nodes):
        # It will handle the MLP layer conversion
        residual_nodes = deepcopy(input_nodes)

        # Post Attention Layer Norm
        post_attention_rms_norm = self.proto_utils['LayerNorm'](POST_ATTENTION_RMS, graph, -1, 64, False, False, self.rms_norm_eps,True)

        # Projections
        gate_proj = self.proto_utils['Conv'](GATE_PROJ, graph, self.intermediate_size, self.mlp_bias, [1, 1], [1,1], 1, 64,HIDDEN, True )
        up_proj = self.proto_utils['Conv'](UP_PROJ, graph, self.intermediate_size, self.mlp_bias, [1, 1], [1,1], 1, 64,HIDDEN, True )
        down_proj = self.proto_utils['Conv'](DOWN_PROJ, graph, self.hidden_size, self.mlp_bias, [1, 1], [1,1], 1, 64,HIDDEN, True )

        #Eltwise ops
        swish_mul = self.proto_utils['Eltwise'](UP_GATE_MUL, graph, -1, 64,  self.proto_utils['EltwiseOperation'].PROD, True)
        final_residue = self.proto_utils['Eltwise'](MLP_RESIDUAL, graph, -1, 64,  self.proto_utils['EltwiseOperation'].SUM, True)

        #swish 
        swish = self.proto_utils['Swish'](GATE_PROJ_ACT,graph, 1,64,True)

        #### Up block #####
        layernorm_out = post_attention_rms_norm(input_nodes , derive_name_from_layer_name=True)
        layernorm_out_perm = self.proto_utils['permute'](graph, layernorm_out, (HIDDEN, TOKEN))
        # DIM0, DIM1, HIDDEN, TOKEN -> DIM0, HIDDEN, DIM1, TOKEN
        rms_out_reshape_dims = self.make_named_shape([DIM0, HIDDEN, DIM1, TOKEN], [1, self.hidden_size, 1,self.token_size])
        up_input = self.make_reshape(graph, layernorm_out_perm, rms_out_reshape_dims, HIDDEN, 64)
        gate_input = deepcopy(up_input)
        ########## UP PROJ ############
        up_output = up_proj(up_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        # DIM0, HIDDEN, DIM1, TOKEN -> DIM0, DIM1, HIDDEN, TOKEN
        up_out_reshape_dims = self.make_named_shape([DIM0,  DIM1, HIDDEN, TOKEN], [1, 1, self.intermediate_size, self.token_size])
        up_output_reshaped = self.make_reshape(graph, up_output, up_out_reshape_dims, HIDDEN, 64)
        up_output_permuted = self.proto_utils['permute'](graph, up_output_reshaped, (HIDDEN, TOKEN))
        
        ######## GATE PROJ ######
        gate_output = gate_proj(gate_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        swish_output = swish(gate_output, derive_name_from_layer_name=True)
        # DIM0, HIDDEN, DIM1, TOKEN -> DIM0, DIM1, HIDDEN, TOKEN
        swish_out_reshape_dims = self.make_named_shape([DIM0,  DIM1, HIDDEN, TOKEN], [1, 1, self.intermediate_size, self.token_size])
        swish_output_reshaped = self.make_reshape(graph, swish_output, swish_out_reshape_dims, HIDDEN, 64)
        swish_output_permuted = self.proto_utils['permute'](graph, swish_output_reshaped, (HIDDEN, TOKEN))

        ######## DOWN BLOCK ##############
        swish_mul_output =  swish_mul(swish_output_permuted + up_output_permuted)
        swish_mul_output_permuted = self.proto_utils['permute'](graph, swish_mul_output, (HIDDEN, TOKEN))
        # DIM0, DIM1, HIDDEN, TOKEN -> DIM0, HIDDEN, DIM1, TOKEN
        swish_mul_out_reshape_dims = self.make_named_shape([DIM0, HIDDEN, DIM1, TOKEN], [1, self.intermediate_size,  1, self.token_size])
        down_input = self.make_reshape(graph, swish_mul_output_permuted, swish_mul_out_reshape_dims, HIDDEN, 64)

        down_proj_output = down_proj(down_input, derive_name_from_layer_name=True, accumulation_dim_name="HIDDEN")

        # DIM0, HIDDEN, DIM1, TOKEN -> DIM0, DIM1, HIDDEN, TOKEN
        down_out_reshape_dims = self.make_named_shape([DIM0, DIM1, HIDDEN, TOKEN], [1, 1, self.hidden_size ,self.token_size])
        down_proj_reshaped = self.make_reshape(graph, down_proj_output, down_out_reshape_dims, HIDDEN, 64)
        down_proj_permuted = self.proto_utils['permute'](graph, down_proj_reshaped, ('HIDDEN', 'TOKEN'))
        final_out = final_residue(residual_nodes + down_proj_permuted)
        return final_out

    def make_decoder_prompt_graph(self, layer_idx):
        graph = self.proto_utils['DAG']()
        attention_outs = self.make_attention_prompt(graph, layer_idx = layer_idx)
        mlp_ins = self.make_attention_mlp_connection_prompt(graph,attention_outs)
        final_out = self.make_mlp_prompt(graph,mlp_ins)
        return graph

    def make_decoder_prompt(self, layer_idx):
        graph = self.make_decoder_prompt_graph(layer_idx)

        decoder_out = os.path.join(self.output_dir,'prmpt_prc_128',f'split_{layer_idx}')
        caffemodel_dir = os.path.join(self.output_dir,'caffemodels')
        os.makedirs(decoder_out, exist_ok=True)
        os.makedirs(caffemodel_dir, exist_ok=True)  
        proto_out = os.path.join(decoder_out,f"split_{layer_idx}.prototxt")
        out_caffemodel = os.path.join(caffemodel_dir,f'split_{layer_idx}.caffemodel')
        out_so = os.path.join(decoder_out,f"split_{layer_idx}_so.yaml")
        img_out = os.path.join(decoder_out,f"imgs") 
        self.get_proto(graph,save_to=proto_out)
        self.make_decoder_caffemodel(proto_out, layer_idx, out_caffemodel)
        self.get_so_yaml(graph,out_so, layer_idx, self.use_8bit_prompt)
        self.make_inputs(proto_out, img_out)
        if self.sdk:
            config_out = os.path.join(decoder_out, "dvrun_config.yaml")
            create_sdk_config(self.caffe, proto_out, out_caffemodel, img_out, out_so, config_out, self.license_key)
        else:
            config_out = os.path.join(decoder_out, "config_ara2.yaml")
            create_config(self.caffe, proto_out, out_caffemodel, img_out, out_so, config_out )
            create_run_py(decoder_out)
    
    def make_attention_token(self, graph, token_cache_size, layer_idx = 0):

        # It will handle the attention layer conversion
        kv_hidden_size = self.num_key_value_heads*self.head_dim
        ######################### Initialization #########################
        #Named shapes
        input_shape = self.make_named_shape([DIM0, DIM1, TOKEN, HIDDEN],[1,1,self.specd_size,self.hidden_size])
        kv_input_shape = self.make_named_shape([DIM0, DIM1, TOKEN_CACHE, HIDDEN],[1,1,token_cache_size,kv_hidden_size])
        rope_input_shape = self.make_named_shape([DIM0, DIM1, TOKEN, SIN_COS],[self.specd_size,1,1, self.head_dim * 2])
        valid_till_shape = self.make_named_shape([DIM0, DIM1, DIM2, DIM3],[1,1,1,1]) 
        requant_input_shape = self.make_named_shape([DIM0, TOKEN, DIM1, HIDDEN],[1,64,1,kv_hidden_size])

    
        #inputs
        input_ids = self.proto_utils['Input'](INPUT, graph,input_shape,-1,64, True)
        past_keys = self.proto_utils['Input'](PAST_KEYS, graph, kv_input_shape,-1,64, True, input_type='K')
        past_values = self.proto_utils['Input'](PAST_VALUES, graph, kv_input_shape,2,64, True, input_type='V')
        requant_in = self.proto_utils['Input'](PAST_VALUES_REQUANT, graph, requant_input_shape,1,64, True, input_type='RQ')

        position_ids = self.proto_utils['Input'](POSTION_EMBEDS, graph, rope_input_shape, 1, 1, False, input_type='R')
        valid_till = self.proto_utils['Input'](VALID_TILL, graph, valid_till_shape, 1, 1, False)
        #layernorms
        input_rms_norm = self.proto_utils['LayerNorm'](INPUT_RMS, graph, -1, 64, False, False, self.rms_norm_eps,True)

        # Q/K normalization layers (for Qwen3, Gemma3)
        if self.use_qk_norm:
            q_norm = self.proto_utils['LayerNorm'](Q_NORM, graph, -1, 64, False, False, self.rms_norm_eps, True)
            k_norm = self.proto_utils['LayerNorm'](K_NORM, graph, -1, 64, False, False, self.rms_norm_eps, True)
        # Projections
        # When Q/K norms are enabled (Qwen3, Gemma3), skip split attention_scale on
        # Q/K projections — the norms already handle normalization, so we keep the
        # full 1/sqrt(head_dim) scaling at QxKt instead of splitting across Q and K.
        attention_scale = 0
        if layer_idx in self.attention_scale_layers and not self.use_qk_norm:
            attention_scale = 1/(self.head_dim**0.25) 
        k_proj = self.proto_utils['Inrp'](K_PROJ, graph, kv_hidden_size, self.attention_bias,1,64,True, attention_scale = attention_scale)
        q_proj = self.proto_utils['Inrp'](Q_PROJ, graph, self.num_attention_heads * self.head_dim, self.attention_bias,1,64,True, attention_scale = attention_scale)
        v_proj = self.proto_utils['Inrp'](V_PROJ, graph, kv_hidden_size, self.attention_bias,1,1,True)
        o_proj = self.proto_utils['Inrp'](O_PROJ, graph, self.hidden_size, False,1,64,True)

        # Rope
        k_position_ids_input = position_ids()
        q_positon_ids_input = deepcopy(k_position_ids_input)
        k_rope = self.proto_utils['Rope'](K_ROPE, graph, k_position_ids_input, -1, 64, is_dynamic=True, output_type = 'K', broadcast=[0, 1.0, 0, 0.5])
        q_rope = self.proto_utils['Rope'](Q_ROPE, graph, q_positon_ids_input, -1, 64, is_dynamic=True, broadcast=[0, 1.0, 0, 0.5])

        #Requant
        requant = self.proto_utils['Requant'](REQUANT, graph, 1, 64)
        # BatchDots
        bdot_out_grp = 1024
        if token_cache_size%1024 != 0:
            bdot_out_grp = 9999

        q_kt_mult = 1/(self.head_dim**0.5)
        if layer_idx in self.attention_scale_layers and not self.use_qk_norm:
            q_kt_mult = 1
        

        QxKt = self.proto_utils['BatchDot'](Q_KT, graph, -1, bdot_out_grp, q_kt_mult,True)
        PxV = self.proto_utils['BatchDot'](P_V, graph, -1, 64, 1, True)

        # SoftMax
        Attn = self.proto_utils['AttentionSoftmax'](SFMX, graph, -1, 64, True)

        ############### Attention Layer Construction ################
        input_ids_out = input_ids()
        rms_norm_out = input_rms_norm(input_ids_out, derive_name_from_layer_name=True)
        # DIM0, DIM1, TOKEN, HIDDEN -> TOKEN, HIDDEN, DIM0, DIM1
        rms_out_reshape_dims = self.make_named_shape([TOKEN, HIDDEN, DIM0, DIM1], [self.specd_size,self.hidden_size, 1, 1])
        proj_input = self.make_reshape(graph, rms_norm_out, rms_out_reshape_dims, HIDDEN, 64)
        k_proj_input = deepcopy(proj_input)
        q_proj_input = deepcopy(proj_input)
        v_proj_input = deepcopy(proj_input)
        ############## K Proj ################
        # breakpoint()
        k_proj_out = k_proj(k_proj_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        # TOKEN, HIDDEN, DIM0, DIM1 -> TOKEN, NUM_HEADS, DIM0, HEAD_DIM
        k_out_reshape_dims = self.make_named_shape([TOKEN, NUM_HEADS, DIM0, HEAD_DIM], [self.specd_size,self.num_key_value_heads, 1, self.head_dim])
        k_proj_out_mha = self.make_reshape(graph, k_proj_out, k_out_reshape_dims, HEAD_DIM, 64)

        # Apply K normalization if enabled (Qwen3)
        # OCheck fix: permute to swap NUM_HEADS(C) with DIM0(H) to get C=1, then permute back
        # Current shape: [TOKEN, NUM_HEADS, DIM0, HEAD_DIM] (C=NUM_HEADS, fails OCheck)
        # Fix: permute NUM_HEADS<->DIM0: [TOKEN, DIM0=1, NUM_HEADS, HEAD_DIM] (C=1)
        # Note: reshape would be logically equivalent here since DIM0=1, but the Ara2
        # hardware does not have a matching kernel for that reshape configuration,
        # so we must use permute which has proper hardware support.
        if self.use_qk_norm:
            # Step 1: Permute NUM_HEADS <-> DIM0: [TOKEN, NUM_HEADS, DIM0, HEAD_DIM] -> [TOKEN, DIM0, NUM_HEADS, HEAD_DIM] (C=1)
            k_proj_prenorm = self.proto_utils['permute'](graph, k_proj_out_mha, (NUM_HEADS, DIM0))
            # Step 2: Apply K normalization (channels=1, axis=-1 normalizes HEAD_DIM)
            k_proj_normed_flat = k_norm(k_proj_prenorm, derive_name_from_layer_name=True)
            # Step 3: Permute back: [TOKEN, DIM0, NUM_HEADS, HEAD_DIM] -> [TOKEN, NUM_HEADS, DIM0, HEAD_DIM]
            k_proj_out_mha = self.proto_utils['permute'](graph, k_proj_normed_flat, (DIM0, NUM_HEADS))
        k_rope_out = k_rope(k_proj_out_mha)
        ############## V Proj ################
        v_proj_out = v_proj(v_proj_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        # TOKEN, HIDDEN, DIM0, DIM1 -> DIM0, TOKEN, DIM1, HIDDEN
        v_out_reshape_dims = self.make_named_shape([DIM0, TOKEN, DIM1, HIDDEN], [1, self.specd_size, 1, kv_hidden_size])
        v_proj_out_reshaped = self.make_reshape(graph, v_proj_out, v_out_reshape_dims, TOKEN, 1)
        ############## Requant ###############
        vt = valid_till()
        vt_copy = deepcopy(vt)
        past_values_last = requant_in()
        requant_out = requant(past_values_last + v_proj_out_reshaped + vt, derive_name_from_layer_name=True)
        ############## Q Proj ################
        q_proj_out = q_proj(q_proj_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)

        # TOKEN, HIDDEN, DIM0, DIM1 -> TOKEN, NUM_HEADS, DIM0, HEAD_DIM
        q_out_reshape_dims = self.make_named_shape([TOKEN, NUM_HEADS, DIM0, HEAD_DIM], [self.specd_size,self.num_attention_heads, 1, self.head_dim])
        q_proj_out_mha = self.make_reshape(graph, q_proj_out, q_out_reshape_dims, HEAD_DIM, 64)
        # Apply Q normalization if enabled (Qwen3)
        # OCheck fix: same permute pattern as K norm above
        if self.use_qk_norm:
            # Step 1: Permute NUM_HEADS <-> DIM0: [TOKEN, NUM_HEADS, DIM0, HEAD_DIM] -> [TOKEN, DIM0, NUM_HEADS, HEAD_DIM] (C=1)
            q_proj_prenorm = self.proto_utils['permute'](graph, q_proj_out_mha, (NUM_HEADS, DIM0))
            # Step 2: Apply Q normalization (channels=1, axis=-1 normalizes HEAD_DIM)
            q_proj_normed_flat = q_norm(q_proj_prenorm, derive_name_from_layer_name=True)
            # Step 3: Permute back: [TOKEN, DIM0, NUM_HEADS, HEAD_DIM] -> [TOKEN, NUM_HEADS, DIM0, HEAD_DIM]
            q_proj_out_mha = self.proto_utils['permute'](graph, q_proj_normed_flat, (DIM0, NUM_HEADS))
        q_rope_out = q_rope(q_proj_out_mha)


        # TOKEN, NUM_HEADS, DIM0, HEAD_DIM -> DIM0, NUM_HEADS,  HEAD_DIM, TOKEN
        if self.specd_size == 1:
            q_rope_out_reshape_dims = self.make_named_shape([DIM0, NUM_HEADS,  HEAD_DIM, TOKEN], [1,self.num_attention_heads, self.head_dim, 1])
            q_rope_out_reshaped = self.make_reshape(graph, q_rope_out, q_rope_out_reshape_dims, HEAD_DIM, 64)
        else:
            # Reshape, permute, reshape
            # TOKEN, NUM_HEADS, DIM0, HEAD_DIM --> 1, 1, TOKEN, NUM_HEADS*HEAD_DIM
            # (_model_layers_0_self_attn_MatMul_ara_permute_nnp_0)
            q_rope_out_initial_reshape_dims = self.make_named_shape([DIM0, DIM1, TOKEN, HIDDEN], [1, 1, self.specd_size, self.num_attention_heads*self.head_dim])
            q_rope_out_initial_reshaped = self.make_reshape(graph, q_rope_out, q_rope_out_initial_reshape_dims, HIDDEN, 64)

            # 1, 1, TOKEN, NUM_HEADS*HEAD_DIM --> 1, 1, NUM_HEADS*HEAD_DIM, TOKEN (_model_layers_0_premute_000)
            # q_rope_out_initial_reshaped_permuted = self.proto_utils['permute'](graph, q_rope_out_initial_reshaped, (1, 1, self.num_attention_heads * self.head_dim, self.specd_size))
            q_rope_out_initial_reshaped_permuted = self.proto_utils['permute'](graph, q_rope_out_initial_reshaped, (TOKEN, HIDDEN))

            # 1, 1, NUM_HEADS*HEAD_DIM, TOKEN --> 1, NUM_HEADS, HEAD_DIM, TOKEN (_model_layers_0_reshape_001)
            q_rope_out_reshape_dims = self.make_named_shape([DIM0, NUM_HEADS, HEAD_DIM, TOKEN], [1, self.num_attention_heads, self.head_dim, self.specd_size])
            # HEAD_DIM is output group axis.
            # We pass 'HEAD_DIM' from dims. not HIDDEN from input.            
            q_rope_out_reshaped = self.make_reshape(graph, q_rope_out_initial_reshaped_permuted, q_rope_out_reshape_dims, HEAD_DIM, 64)

        ############ QxKt ################
        past_keys_permuted = self.proto_utils['permute'](graph, past_keys(), (TOKEN_CACHE, HIDDEN))
        # DIM0, DIM1, HIDDEN, TOKEN_CACHE -> DIM0, NUM_HEADS, HEAD_DIM, TOKEN_CACHE
        pk_out_reshape_dims = self.make_named_shape([DIM0, NUM_HEADS, HEAD_DIM, TOKEN_CACHE], [1,self.num_key_value_heads, self.head_dim, token_cache_size])
        past_keys_reshaped = self.make_reshape(graph, past_keys_permuted, pk_out_reshape_dims, HEAD_DIM, 64)
        QxKt_out = QxKt(q_rope_out_reshaped + past_keys_reshaped, derive_name_from_layer_name=True)

        ############ Softmax ################  
        attn_out = Attn(QxKt_out + vt_copy, axis = -1)
        if self.specd_size == 1:
            # DIM0, NUM_HEADS, TOKEN, TOKEN_CACHE -> DIM0, NUM_HEADS, TOKEN_CACHE, TOKEN
            attn_out_reshape_dims = self.make_named_shape([DIM0, NUM_HEADS, TOKEN_CACHE, TOKEN], [1,self.num_attention_heads, token_cache_size, 1])
            attn_out_reshaped = self.make_reshape(graph, attn_out, attn_out_reshape_dims, TOKEN_CACHE, 64)
        else:
            # DIM0, NUM_HEADS, TOKEN, TOKEN_CACHE -> DIM0, TOKEN, NUM_HEADS, TOKEN_CACHE (_model_layers_0_premute_001)
            attn_out_permuted = self.proto_utils['permute'](graph, attn_out, (TOKEN, NUM_HEADS))

            # DIM0, TOKEN, NUM_HEADS, TOKEN_CACHE --> TOKEN, NUM_HEADS, TOKEN_CACHE, DIM0 (_model_layers_0_self_attn_rshp_Softmax)
            attn_out_reshape_dims = self.make_named_shape([TOKEN, NUM_HEADS, TOKEN_CACHE, DIM0], [self.specd_size, self.num_attention_heads, token_cache_size, 1])
            attn_out_reshaped = self.make_reshape(graph, attn_out_permuted, attn_out_reshape_dims, TOKEN_CACHE, 64)

        ################ PxV ################
        # DIM0, DIM1, TOKEN_CACHE, HIDDEN -> DIM1, TOKEN_CACHE, NUM_HEADS, HEAD_DIM        
        # We are choosing DIM1 instead of DIM0 because later in the code 'attn_out_reshaped' and 'past_values_permuted' will be concatenated which will both have DIM0 in a way that PxV_out will contain DIM0 twice. Then in specd flow pxv_out_reshape_dims will have DIM0 twice which will throw error in make_reshape.
        pv_out_reshape_dims = self.make_named_shape([DIM1, TOKEN_CACHE, NUM_HEADS, HEAD_DIM], [1, token_cache_size, self.num_key_value_heads, self.head_dim])
        past_values_reshaped = self.make_reshape(graph, past_values(), pv_out_reshape_dims, TOKEN_CACHE, 64)

        past_values_permuted = self.proto_utils['permute'](graph, past_values_reshaped, (TOKEN_CACHE, NUM_HEADS))
        # 'attn_out_reshaped' contains DIM0 and if 'past_values_permuted' also has DIM0, it will fail.
        # For time being we workaround by having DIM1 in past_values_permuted transitively.        
        PxV_out = PxV(attn_out_reshaped + past_values_permuted, derive_name_from_layer_name=True)

        # DIM0, NUM_HEADS, TOKEN, HIDDEN -> DIM0, HIDDEN, DIM1, TOKEN
        # Use num_attention_heads * head_dim (not hidden_size) because PxV output has
        # all Q attention heads (e.g. Qwen3-0.6B: hidden_size=1024, num_heads*head_dim=2048)
        q_hidden_size = self.num_attention_heads * self.head_dim
        if self.specd_size == 1:
            pxv_out_reshape_dims = self.make_named_shape([DIM0, HIDDEN, DIM1, TOKEN], [1, q_hidden_size, 1, 1])
        else:
          #_model_layers_0_post_attention_layernorm_Mul_1_ara_reshape_0
            pxv_out_reshape_dims = self.make_named_shape([TOKEN, HIDDEN, DIM0, DIM1], [self.specd_size, q_hidden_size, 1, 1])

        print(f"make_attention_token PxV_out[0].shape {PxV_out[0].shape}")
        print(f"make_attention_token PxV_out[1].shape {PxV_out[1].shape}")
        print(f"make_attention_token pxv_out_reshape_dims {pxv_out_reshape_dims}")
        PxV_out_reshaped = self.make_reshape(graph, PxV_out, pxv_out_reshape_dims, HIDDEN, 64)
        ############# O_proj ###############
        o_proj_out =  o_proj(PxV_out_reshaped, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        # DIM0, HIDDEN, DIM1, TOKEN -> DIM0, DIM1, TOKEN, HIDDEN
        o_out_reshape_dims = self.make_named_shape([DIM0, DIM1, TOKEN, HIDDEN], [1, 1, self.specd_size, self.hidden_size])
        o_proj_out_reshaped = self.make_reshape(graph, o_proj_out, o_out_reshape_dims, HIDDEN, 64)
        # Apply post-attention norm if enabled (Gemma3)

        return o_proj_out_reshaped + input_ids_out
    
    def make_attention_mlp_connection_token(self, graph, input_nodes):
        attention_residual = self.proto_utils['Eltwise'](ATTENTION_RESIDUAL, graph, -1, 64,  self.proto_utils['EltwiseOperation'].SUM, True)
        residual_out = attention_residual(input_nodes)
        return residual_out
    
    def make_mlp_token(self, graph, input_nodes):
        # It will handle the MLP layer conversion
        residual_nodes = deepcopy(input_nodes)

        # Post Attention Layer Norm
        post_attention_rms_norm = self.proto_utils['LayerNorm'](POST_ATTENTION_RMS, graph, -1, 64, False, False, self.rms_norm_eps,True)

        # Projections
        gate_proj = self.proto_utils['Inrp'](GATE_PROJ, graph, self.intermediate_size, self.mlp_bias,1,64,True)
        up_proj = self.proto_utils['Inrp'](UP_PROJ, graph, self.intermediate_size, self.mlp_bias,1,64,True)
        down_proj = self.proto_utils['Inrp'](DOWN_PROJ, graph, self.hidden_size, self.mlp_bias,1,64,True)

        #Eltwise ops
        swish_mul = self.proto_utils['Eltwise'](UP_GATE_MUL, graph, -1, 64,  self.proto_utils['EltwiseOperation'].PROD, True)
        final_residue = self.proto_utils['Eltwise'](MLP_RESIDUAL, graph, -1, 64,  self.proto_utils['EltwiseOperation'].SUM, True)

        #swish 
        swish = self.proto_utils['Swish'](GATE_PROJ_ACT,graph, 1,64,True)

        #### Up block #####
        layernorm_out = post_attention_rms_norm(input_nodes , derive_name_from_layer_name=True)
        if self.specd_size == 1:
            # DIM0, DIM1, TOKEN, HIDDEN -> DIM0, HIDDEN, DIM1, TOKEN
            rms_out_reshape_dims = self.make_named_shape([DIM0, HIDDEN, DIM1, TOKEN], [1, self.hidden_size, 1, 1])
        else:
            # DIM0, DIM1, TOKEN, HIDDEN -> TOKEN, HIDDEN, DIM0, DIM1
            rms_out_reshape_dims = self.make_named_shape([TOKEN, HIDDEN, DIM0, DIM1], [self.specd_size, self.hidden_size, 1, 1])
        up_input = self.make_reshape(graph, layernorm_out, rms_out_reshape_dims, HIDDEN, 64)
        gate_input = deepcopy(up_input)
        ########## UP PROJ ############
        up_output = up_proj(up_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        # DIM0, HIDDEN, DIM1, TOKEN -> TOKEN, DIM0, DIM1, HIDDEN
        up_out_reshape_dims = self.make_named_shape([TOKEN, DIM0, DIM1, HIDDEN], [self.specd_size, 1, 1, self.intermediate_size])
        up_output_reshaped = self.make_reshape(graph, up_output, up_out_reshape_dims, HIDDEN, 64)
        
        ######## GATE PROJ ######
        gate_output = gate_proj(gate_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        swish_output = swish(gate_output, derive_name_from_layer_name=True)
        if self.specd_size == 1:
            # DIM0, HIDDEN, DIM1, TOKEN -> DIM0, DIM1, TOKEN, HIDDEN
            swish_out_reshape_dims = self.make_named_shape([DIM0, DIM1, TOKEN, HIDDEN], [1, 1, 1, self.intermediate_size])
        else:
            # DIM0, HIDDEN, DIM1, TOKEN -> TOKEN, DIM0, DIM1, HIDDEN
            swish_out_reshape_dims = self.make_named_shape([TOKEN, DIM0, DIM1, HIDDEN], [self.specd_size, 1, 1, self.intermediate_size])
        swish_output_reshaped = self.make_reshape(graph, swish_output, swish_out_reshape_dims, HIDDEN, 64)

        ######## DOWN BLOCK ##############
        swish_mul_output =  swish_mul(swish_output_reshaped + up_output_reshaped)
        # DIM0, DIM1, TOKEN, HIDDEN -> TOKEN, HIDDEN, DIM0, DIM1
        swish_mul_out_reshape_dims = self.make_named_shape([TOKEN, HIDDEN, DIM0, DIM1], [self.specd_size, self.intermediate_size,  1, 1])
        down_input = self.make_reshape(graph, swish_mul_output, swish_mul_out_reshape_dims, HIDDEN, 64)
        
        down_proj_output = down_proj(down_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)

        # DIM0, HIDDEN, DIM1, TOKEN -> DIM0, DIM1, TOKEN, HIDDEN
        down_out_reshape_dims = self.make_named_shape([DIM0, DIM1, TOKEN, HIDDEN], [1, 1, self.specd_size, self.hidden_size])
        down_proj_reshaped = self.make_reshape(graph, down_proj_output, down_out_reshape_dims, HIDDEN, 64)

        final_out = final_residue(residual_nodes + down_proj_reshaped)
        return final_out

    def make_decoder_token_graph(self, layer_idx, ctx):
        graph = self.proto_utils['DAG']()
        attention_outs = self.make_attention_token(graph, ctx, layer_idx=layer_idx)
        mlp_ins = self.make_attention_mlp_connection_token(graph,attention_outs)
        final_out = self.make_mlp_token(graph,mlp_ins)
        return graph
    
    def make_decoder_token(self, layer_idx, ctx):
        graph = self.make_decoder_token_graph(layer_idx, ctx)
        decoder_out = os.path.join(self.output_dir,'tkn_gen_'+ str(ctx),f'split_{layer_idx}')
        os.makedirs(decoder_out, exist_ok=True) 
        proto_out = os.path.join(decoder_out,f"split_{layer_idx}.prototxt")
        out_caffemodel = os.path.join(decoder_out,f'split_{layer_idx}.caffemodel')
        out_so = os.path.join(decoder_out,f"split_{layer_idx}_so.yaml")
        img_out = os.path.join(decoder_out,f"imgs") 
        self.get_proto(graph, save_to=proto_out)
        # self.make_decoder_caffemodel(proto_out, layer_idx, out_caffemodel)
        self.get_so_yaml(graph,out_so, layer_idx)
        self.make_inputs(proto_out, img_out)
        config_out = os.path.join(decoder_out, "config_ara2.yaml")
        if self.sdk:
            config_out = os.path.join(decoder_out, "dvrun_config.yaml")
            create_sdk_config(self.caffe, proto_out, self.caffe_model_layer_idx_map[layer_idx], img_out, out_so, config_out, self.license_key)
        else:
            create_config(self.caffe, proto_out, self.caffe_model_layer_idx_map[layer_idx], img_out, out_so, config_out )
            create_run_py(decoder_out) 

    # TODO: Use make_inputs from utils.py
    def make_inputs(self, proto, img_out_dir):
        img_dir = os.path.join(img_out_dir, 'chksum0')
        os.makedirs(img_dir, exist_ok=True)
        net = self.caffe.Net(proto, self.caffe.TEST)
        input_dict = {}
        for k in net.inputs:
            if 'valid_till' in k:
                input_dict[k] = np.array([self.token_size-1], dtype= np.float32)
                continue
                
            if 'scale' not in k:
                input_dict[k]= (np.random.rand(*(net.blobs[k].data.shape)).astype(np.float32)*2) -1
                continue
        ### SCALE SHAPES
        for k in net.inputs:
            if INPUT+'_scale' in k:
                input_dict[k] = get_scale_tensor_float(input_dict[INPUT], input_dict[INPUT].shape, net.blobs[k].data.shape, 64, 3) #NCHW
            if PAST_KEYS + '_scale' in k:
                input_dict[k] =  get_scale_tensor_float(input_dict[PAST_KEYS], input_dict[PAST_KEYS].shape, net.blobs[k].data.shape, 64, 3) #NCHW
            if PAST_VALUES + '_scale' in k:
                input_dict[k] =  get_scale_tensor_float(input_dict[PAST_VALUES], input_dict[PAST_VALUES].shape, net.blobs[k].data.shape, 64, 2) #NCHW
            if PAST_VALUES_REQUANT + '_scale' in k:
                input_dict[k] =  get_scale_tensor_float(input_dict[PAST_VALUES_REQUANT], input_dict[PAST_VALUES_REQUANT].shape, net.blobs[k].data.shape, 64, 2) #NCHW 

            if (self.architecture == 'Qwen2_5_VLForConditionalGeneration') and ('vision' in k and 'scale' in k):
                print(f'Qwen2_5_VLForConditionalGeneration: vision scale {k}: make random of shape {net.blobs[k].data.shape}')
                input_dict[k] = np.random.rand(*(net.blobs[k].data.shape))  # Vision scale is different from LLM input scale

            elif k.endswith('_scale'):
                # Qwen3VL vision input scales and all unhandled scales - treat them as normal input scale
                # We are yet to see it passing.
                # This does not work for Qwen2.5VL: cannot reshape array of size 677376 into shape (64,336)
                accompanying_input = k.removesuffix('_scale')
                input_dict[k] =  get_scale_tensor_float(input_dict[accompanying_input], input_dict[accompanying_input].shape, net.blobs[k].data.shape, 64, 3) #NCHW

        for k in input_dict.keys():
            input_dict[k].astype(np.float32).tofile(os.path.join(img_dir, k+'.bin')) 
    
    def make_decoder_caffemodel(self, proto, layer_idx, out_caffemodel):
    # --------- Load from ONNX (decoder_initializers) ----------
        weights = {}
        decoder_initializers = [
            initializer
            for initializer in self.onnx_model.graph.initializer
            if f".{layer_idx}." in initializer.name
        ]
                # Check if model has separate Q/K/V weights (Qwen3) or fused QKV (LLaMA, Qwen2)
        has_separate_qkv = any(
            Q_PROJ_WEIGHT in initializer.name
            for initializer in decoder_initializers
        )
        if has_separate_qkv:
            # Separate Q, K, V weights (Qwen3, etc.)
            required = [Q_PROJ_WEIGHT, K_PROJ_WEIGHT, V_PROJ_WEIGHT, O_PROJ_WEIGHT, UP_PROJ_WEIGHT, GATE_PROJ_WEIGHT, DOWN_PROJ_WEIGHT, INPUT_LAYERNORM_WEIGHT, POST_ATTENTION_LAYERNORM_WEIGHT]
            if self.attention_bias:
                required.extend([Q_PROJ_BIAS, K_PROJ_BIAS, V_PROJ_BIAS])
        else:
            # Fused QKV weight (LLaMA, Qwen2, etc.)

            required = [QKV_WEIGHT, O_PROJ_WEIGHT, UP_PROJ_WEIGHT, GATE_PROJ_WEIGHT, DOWN_PROJ_WEIGHT, INPUT_LAYERNORM_WEIGHT, POST_ATTENTION_LAYERNORM_WEIGHT]
            if (self.attention_bias):
                required.append(QKV_BIAS)
        # Also require Q/K norm weights if qk_norm is enabled
        if self.use_qk_norm:
            required.append(Q_NORM_WEIGHT)
            required.append(K_NORM_WEIGHT)
        for initializer in decoder_initializers:
            print("name of initializer:",initializer.name, initializer.dims)
            for tag in required:
                if tag in initializer.name:
                    print(f"{tag} : {initializer.name} weights")
                    print(f"  Shape: {initializer.dims}")
                    print(f"  Data Type: {initializer.data_type}")
                    raw_weights = np.frombuffer(initializer.raw_data, dtype=np.float32)
                    raw_weights_transposed_flattened = np.transpose(raw_weights.reshape(initializer.dims)).flatten()
                    weights[tag] = raw_weights_transposed_flattened
                    print(f" New shape: {np.transpose(raw_weights.reshape(initializer.dims)).shape}")

        # --------- Validate required weights ---------
        for r in required:
            assert r in weights and weights[r] is not None, f"Missing weight: {r}"


        # --------- Slice QKV Weights ----------
        if not has_separate_qkv:
            qkv_proj_weight = weights[QKV_WEIGHT]
            q_size = self.hidden_size*(self.num_attention_heads * self.head_dim)
            kv_size = self.hidden_size*self.num_key_value_heads * self.head_dim

            weights[Q_PROJ_WEIGHT] = qkv_proj_weight[:q_size]
            weights[K_PROJ_WEIGHT] = qkv_proj_weight[q_size:q_size+kv_size]
            weights[V_PROJ_WEIGHT] = qkv_proj_weight[q_size+kv_size:]

            # ---------- Slice QKV Bias ----------
            if self.attention_bias:
                qkv_proj_bias = weights[QKV_BIAS].flatten()
                q_size = self.num_attention_heads * self.head_dim
                kv_size = self.num_key_value_heads * self.head_dim

                weights[Q_PROJ_BIAS] = qkv_proj_bias[:q_size]
                weights[K_PROJ_BIAS] = qkv_proj_bias[q_size:q_size+kv_size]
                weights[V_PROJ_BIAS] = qkv_proj_bias[q_size+kv_size:] 

        # --------- Assign to Caffe Params ----------
        caffe_param_map = {
            Q_PROJ:Q_PROJ_WEIGHT,
            K_PROJ:K_PROJ_WEIGHT,
            V_PROJ:V_PROJ_WEIGHT,
            O_PROJ:O_PROJ_WEIGHT,
            UP_PROJ: UP_PROJ_WEIGHT,
            GATE_PROJ: GATE_PROJ_WEIGHT,
            DOWN_PROJ: DOWN_PROJ_WEIGHT,
            INPUT_RMS: INPUT_LAYERNORM_WEIGHT,
            POST_ATTENTION_RMS: POST_ATTENTION_LAYERNORM_WEIGHT
        }
        # Add Q/K norm weights to caffe param map if enabled
        if self.use_qk_norm:
            caffe_param_map[Q_NORM] = Q_NORM_WEIGHT
            caffe_param_map[K_NORM] = K_NORM_WEIGHT

        caffe_param_bias_map = {}
        if self.attention_bias:
            caffe_param_bias_map[Q_PROJ] = Q_PROJ_BIAS
            caffe_param_bias_map[K_PROJ] = K_PROJ_BIAS
            caffe_param_bias_map[V_PROJ] = V_PROJ_BIAS

        source_net = self.caffe.Net(proto, self.caffe.TEST)
        for param_name, weight_key in caffe_param_map.items():
            print(f"trying to copy weights for {param_name} and {weight_key}")
            if param_name in source_net.params and weight_key in weights:
                param = source_net.params[param_name][0]
                param_data = weights[weight_key].reshape(param.data.shape)
                param.data[...] = param_data
                
                print(f"Copying weights from {weight_key} to {param_name} with shape {param.data.shape}")

        for param_name, weight_key in caffe_param_bias_map.items():
            if param_name in source_net.params and weight_key in weights:
                param = source_net.params[param_name][1]
                param_data = weights[weight_key].reshape(param.data.shape)
                param.data[...] = param_data
                print(f"Copying bias from {weight_key} to {param_name}")

        source_net.save_hdf5(out_caffemodel)
        self.caffe_model_layer_idx_map[layer_idx] = out_caffemodel
        print(f"decoder Caffe model saved to {out_caffemodel}")

    def get_so_yaml(self,graph, save_to=None, layer_idx = 0, use_8bit_prompt = False):
        yaml_dict = {}
        weight_config = self.weight_config[layer_idx]
        for node_with_shape in graph.nodes_with_shape:
            if not node_with_shape.out_group_axis:
                continue
            yaml_dict[node_with_shape.name] = {
                "outputAxis": node_with_shape.out_group_axis if node_with_shape.out_group_axis <=1 else node_with_shape.out_group_axis + 1,
                "outputGroupSize": node_with_shape.out_group_size,
            }

            small_precisions = [INT2, INT3, INT4]
            if node_with_shape.name in weight_config.keys():
                if use_8bit_prompt:
                    yaml_dict[node_with_shape.name]['weightPrecision'] = [{node_with_shape.name: INT8}]
                else:
                    yaml_dict[node_with_shape.name]['weightPrecision'] = [{node_with_shape.name: weight_config[node_with_shape.name]}]
                    if weight_config[node_with_shape.name] in small_precisions:
                        yaml_dict[node_with_shape.name]['groupSize'] = 64  
                
            if node_with_shape.name in LMHEAD or node_with_shape.name in "lmhead_comm":
                yaml_dict[node_with_shape.name]['outputDataPrecision'] = [{node_with_shape.name: INT32}] 
            
            if node_with_shape.name in [Q_PROJ, K_PROJ, V_PROJ, Q_KT, O_PROJ, UP_PROJ, GATE_PROJ, DOWN_PROJ, GATE_PROJ_ACT, SFMX]:
                yaml_dict[node_with_shape.name]['validTill_applied'] = True
            if node_with_shape.name == Q_KT and layer_idx in self.qxk_out_scale_32_layers:
                yaml_dict[node_with_shape.name]['outputDataPrecision'] = [{node_with_shape.name: INT32}] 



        if save_to:
            with open(save_to, "w") as file:
                yaml.dump({"validTill":self.token_size-1,"killAxis": 4,"LayerNames":yaml_dict}, file)
        else:
            print(yaml_dict)
    
    def get_proto(self, graph, save_to=None):
        if not save_to:
            print(''.join(str(node) for node in graph.proto_nodes))
        else:
            with open(save_to, "w+") as f:
                f.write('platform_target: "Ara2"\n')
                f.write('is_converted: true\n')
                f.write(f'context_size: {self.prmpt_context_size}\n')
                f.write(''.join(str(node) for node in graph.proto_nodes))
    
    def convert_model(self):
        self.make_lmhead()
        for idx in range(self.num_hidden_layers):
            self.make_decoder_prompt(idx)
            for token_size in self.tkn_gen_sizes:
                self.make_decoder_token(idx, int(token_size))
            print(f"####################### DONE WITH DECODER {idx} ###########################")
    

class Qwen2Model(Model):

    def __init__(self, config, hf_config, proto_utils, onnx_model, caffe, output_dir, sdk):
        super().__init__(config, hf_config, proto_utils, onnx_model, caffe, output_dir, sdk)
    
    def get_activations_config(self):
        super().get_activations_config()
        if self.qxk_out_scale_32_layers == []:
            self.qxk_out_scale_32_layers = [i for i in range(self.num_hidden_layers)]
        if self.attention_scale_layers == []:
            self.attention_scale_layers = [0, self.num_hidden_layers-1]

class Qwen2_5VLModel(Qwen2Model):

    def __init__(self, config, hf_config, proto_utils, onnx_model, caffe, output_dir:dict, sdk):
        super().__init__(config, hf_config, proto_utils, onnx_model, caffe, output_dir['llm'], sdk)
        # From here onwards, self.output_dir is LLM-specific.
        #  Model.output_dir should also be made a dict.

        import sys
        sys.path.append(str(Path(__file__).absolute().parent / 'qwen2_5_vl'))
        from qwen2_5_vl.model import Qwen2_5_VL_VisionModel
        self.vision_model = Qwen2_5_VL_VisionModel(
            config=self.config, hf_config_json=self.hf_config, proto_utils=self.proto_utils, onnx_model=self.onnx_model, graph=self.proto_utils["DAG"](), caffe=self.caffe, output_dir=output_dir['vision'], sdk=self.sdk
        )

    def dump_assets(self):
        self.vision_model.dump_assets()  # Vision
        super().dump_assets()  # LLM


    def convert_model(self):
        self.vision_model.convert_model() # Vision
        super().convert_model()  # LLM
