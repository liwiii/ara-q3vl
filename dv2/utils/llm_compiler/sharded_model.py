import yaml
import os
import numpy as np
from copy import deepcopy
from typing import List
from padding_util import next_multiple_of, align
from utils import quantize, get_scale_tensor_float, create_config, create_run_py, create_sdk_config
from constants import *
import json
from model import Model

"""
ShardedModel Class

This class extends the base Model class to provide sharded (distributed) implementation
for building and compiling LLM models. It handles the sharding of model parameters 
across multiple devices/nodes and manages the communication between shards.

Key methods:
- __init__: Initialize the sharded model with configuration and number of shards
- make_decoder_caffemodel: Creates Caffe model files for each shard with proper sharding
- make_attention_prompt: Builds attention layer for prompt processing with sharding
- make_mlp_prompt: Builds MLP layer for prompt processing with sharding
- make_attention_mlp_connection_prompt: Handles attention-MLP connection for prompt processing
- make_attention_token: Builds attention layer for token generation with sharding
- make_mlp_token: Builds MLP layer for token generation with sharding
- make_attention_mlp_connection_token: Handles attention-MLP connection for token generation
- make_decoder_prompt: Generates all decoder components for prompt processing across shards
- make_decoder_token: Generates all decoder components for token generation across shards
- make_lmhead: Creates LM head components for each shard
- make_lmhead_graph: Builds LM head network graph with proper sharding
- make_lmhead_caffemodel: Creates LM head Caffe model with shard-specific weights

The class implements sharding logic in projection weights and biases, handling 
different sharding schemes for attention and MLP components. Communication nodes are 
inserted at key points to enable proper cross-shard operations during inference.
"""
class ShardedModel(Model):
    def __init__(self, config, hf_config, proto_utils, onnx_model, caffe, output_dir, sdk=False, num_shards=1):
        super().__init__(config, hf_config, proto_utils, onnx_model, caffe, output_dir, sdk)
        self.num_shards = num_shards
        if not self.check_mcs_possible():
            raise ValueError(f"number of shards {self.num_shards} not valid for the model config")


    def check_mcs_possible(self):
        #check heads
        if not ((self.num_shards == 2) or (self.num_shards == 4)):
            print(f"num_shards accepted values only 2 or 4, but given {self.num_shards}")
            return False
        if self.num_attention_heads % self.num_shards != 0:
            print(f"num_attention_heads {self.num_attention_heads} should be divisible by num_shards {self.num_shards}")
            return False
        if self.num_key_value_heads % self.num_shards != 0:
            print(f"num_key_value_heads {self.num_key_value_heads} should be divisible by num_shards {self.num_shards}")
            return False
        #check sizes
        if self.hidden_size % (64 * self.num_shards) != 0:
            print(f"Issue with hidden size {self.hidden_size} not being divisible by 64 * num_shards {self.num_shards * 64}")
            return False
        if self.num_attention_heads * self.head_dim % (64 * self.num_shards)!= 0:
            print(f"num_attention_heads {self.num_attention_heads} * head_dim {self.head_dim} not divisible by 64 * num_shards {self.num_shards * 64}")
            return False
        if self.num_key_value_heads * self.head_dim % (64 * self.num_shards) != 0:
            print(f"num_key_value_heads {self.num_key_value_heads} * head_dim {self.head_dim} not divisible by 64 * num_shards {self.num_shards * 64}")
            return False
        if self.vocab_size % (64 * self.num_shards) != 0:
            print(f"vocab_size {self.vocab_size} not divisible by 64 * self.num_shards {self.num_shards * 64}")
            return False
        if self.intermediate_size % (64 * self.num_shards) != 0:
            print(f"intermediate_size {self.intermediate_size} not divisible by 64 * self.num_shards {self.num_shards * 64}")
            return False 
        return True

    def make_decoder_caffemodel(self, proto_out, layer_idx, out_caffemodel, shard_idx=0):
        # Override for sharded implementation - following exact logic from model.py but with sharding
        # Load from ONNX (decoder_initializers) - similar to original but with sharding
        weights = {}
        decoder_initializers = [
            initializer
            for initializer in self.onnx_model.graph.initializer
            if f".{layer_idx}." in initializer.name
        ]
        required = [QKV_WEIGHT, O_PROJ_WEIGHT, UP_PROJ_WEIGHT, GATE_PROJ_WEIGHT, DOWN_PROJ_WEIGHT, INPUT_LAYERNORM_WEIGHT, POST_ATTENTION_LAYERNORM_WEIGHT]
        if (self.attention_bias):
            required.append(QKV_BIAS)
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

        # Validate required weights
        for r in required:
            assert r in weights and weights[r] is not None, f"Missing weight: {r}"


        # Slice QKV Weights - this is where we need to add shard-specific slicing
        qkv_proj_weight = weights[QKV_WEIGHT]
        q_size = self.hidden_size*(self.num_attention_heads * self.head_dim)
        kv_size = self.hidden_size*self.num_key_value_heads * self.head_dim

        # Slice the QKV weights for projection layers
        weights[Q_PROJ_WEIGHT] = qkv_proj_weight[:q_size]
        weights[K_PROJ_WEIGHT] = qkv_proj_weight[q_size:q_size+kv_size]
        weights[V_PROJ_WEIGHT] = qkv_proj_weight[q_size+kv_size:]

        # Slice QKV Bias - this is where we need to add shard-specific slicing for bias too
        if self.attention_bias:
            qkv_proj_bias = weights[QKV_BIAS].flatten()
            q_size = self.num_attention_heads * self.head_dim
            kv_size = self.num_key_value_heads * self.head_dim

            weights[Q_PROJ_BIAS] = qkv_proj_bias[:q_size]
            weights[K_PROJ_BIAS] = qkv_proj_bias[q_size:q_size+kv_size]
            weights[V_PROJ_BIAS] = qkv_proj_bias[q_size+kv_size:] 

        # --------- Apply Sharding to Projection Weights ----------
        # For each projection layer, slice the weights based on shard_idx along the correct dimension
        if self.num_shards > 1:
            # Define projection layers and their corresponding output channel calculation
            proj_layers = [
                Q_PROJ_WEIGHT,
                K_PROJ_WEIGHT, 
                V_PROJ_WEIGHT,
                O_PROJ_WEIGHT,
                UP_PROJ_WEIGHT,
                GATE_PROJ_WEIGHT,
                DOWN_PROJ_WEIGHT
            ]
            
            # Slice each projection layer
            for weight_key in proj_layers:
                if weight_key in weights:
                    proj_weights = weights[weight_key]
                    if len(proj_weights) > 0:
                        # Calculate weight_size
                        weight_size = len(weights[weight_key])

                        
                        # Only shard if we have enough channels to split
                        if weight_size > 0 and weight_size >= self.num_shards:
                            shard_size = weight_size // self.num_shards
                            start = shard_idx * shard_size
                            end = (shard_idx + 1) * shard_size
                            weights[weight_key] = proj_weights[start:end]
            
            # Handle bias terms for attention projections
            if self.attention_bias:
                bias_layers = [
                    Q_PROJ_BIAS,
                    K_PROJ_BIAS,
                    V_PROJ_BIAS
                ]
                
                for bias_key in bias_layers:
                    if bias_key in weights:
                        bias_weights = weights[bias_key]
                        if len(bias_weights) > 0:
                            # Calculate total bias channels 
                            total_channels = len(bias_weights)
                            if total_channels > 0 and total_channels >= self.num_shards:
                                shard_size = total_channels // self.num_shards
                                start = shard_idx * shard_size
                                end = (shard_idx + 1) * shard_size
                                weights[bias_key] = bias_weights[start:end]

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

        caffe_param_bias_map = {}
        if self.attention_bias:
            caffe_param_bias_map[Q_PROJ] = Q_PROJ_BIAS
            caffe_param_bias_map[K_PROJ] = K_PROJ_BIAS
            caffe_param_bias_map[V_PROJ] = V_PROJ_BIAS

        source_net = self.caffe.Net(proto_out, self.caffe.TEST)
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
        
        # Update the model mapping with shard-specific hierarchy
        # Store in 2D dictionary: layer_idx -> shard_idx -> caffemodel_path
        if layer_idx not in self.caffe_model_layer_idx_map:
            self.caffe_model_layer_idx_map[layer_idx] = {}
        if shard_idx not in self.caffe_model_layer_idx_map[layer_idx]:
            self.caffe_model_layer_idx_map[layer_idx][shard_idx] = {}
        self.caffe_model_layer_idx_map[layer_idx][shard_idx] = out_caffemodel
        
        print(f"decoder Caffe model saved to {out_caffemodel}")
    
    def make_attention_prompt(self, graph, layer_idx=0, shard_idx=0):
        # Override for sharded implementation
        # It will handle the attention layer conversion
        hidden_size_sharded = self.hidden_size//self.num_shards
        kv_hidden_size = self.num_key_value_heads * self.head_dim
        # Dividing by num_shards for sharding
        kv_hidden_size_sharded = kv_hidden_size // self.num_shards
        q_hidden_size = self.num_attention_heads * self.head_dim
        q_hidden_size_sharded = q_hidden_size // self.num_shards
        # Dividing num_heads by num_shards for each shard
        num_key_value_heads_sharded = self.num_key_value_heads // self.num_shards
        num_attention_heads_sharded = self.num_attention_heads // self.num_shards
        
        ######################### Initialization #########################
        #Named shapes
        input_shape = self.make_named_shape([DIM0, DIM1, TOKEN, HIDDEN],[1,1,self.token_size,self.hidden_size])
        # kv_input_shape adjusted for sharding - the hidden dimension is divided by num_shards
        kv_input_shape = self.make_named_shape([DIM0, DIM1, TOKEN_CACHE, HIDDEN],[1,1,self.prmpt_context_size,kv_hidden_size_sharded])
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

        # Projections - sharded
        attention_scale = 0
        if layer_idx in self.attention_scale_layers:
            attention_scale = 1/(self.head_dim**0.25) 
        k_proj = self.proto_utils['Conv'](K_PROJ, graph, kv_hidden_size_sharded, self.attention_bias, [1, 1], [1,1], 1, 64,HIDDEN, True, attention_scale = attention_scale )
        q_proj = self.proto_utils['Conv'](Q_PROJ, graph, q_hidden_size_sharded, self.attention_bias, [1, 1], [1,1], 1, 64,HIDDEN, True, attention_scale = attention_scale )
        v_proj = self.proto_utils['Conv'](V_PROJ, graph, kv_hidden_size_sharded, self.attention_bias, [1, 1], [1,1], -1, 64,HIDDEN, True )
        o_proj = self.proto_utils['Conv'](O_PROJ, graph, hidden_size_sharded, False, [1, 1], [1,1], 1, 64,HIDDEN, True )

        # Rope
        permuted_position_ids = self.proto_utils['permute'](graph,position_ids(), (TOKEN, SIN_COS))
        k_rope = self.proto_utils['Rope'](K_ROPE, graph, permuted_position_ids, 2, 64, is_dynamic=True)
        q_rope = self.proto_utils['Rope'](Q_ROPE, graph, permuted_position_ids, 2, 64, is_dynamic=True)

        # BatchDots
        q_kt_mult = 1/(self.head_dim**0.5)
        if layer_idx in self.attention_scale_layers:
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
        # Note: NUM_HEADS should also be sharded here
        k_out_reshape_dims = self.make_named_shape([DIM0, NUM_HEADS, HEAD_DIM, TOKEN], [1,num_key_value_heads_sharded, self.head_dim, self.token_size])
        k_proj_out_mha = self.make_reshape(graph, k_proj_out, k_out_reshape_dims, HEAD_DIM, 64)

        k_rope_out = k_rope(k_proj_out_mha)
        # DIM0, NUM_HEADS, HEAD_DIM, TOKEN -> DIM0, DIM1, HIDDEN, TOKEN
        k_rope_out_reshape_dims = self.make_named_shape([DIM0, DIM1, HIDDEN, TOKEN], [1,1,kv_hidden_size_sharded,self.token_size ])
        k_rope_out_reshaped = self.make_reshape(graph, k_rope_out, k_rope_out_reshape_dims, HIDDEN, 64)
        
        present_keys = self.proto_utils['permute'](graph, k_rope_out_reshaped, (TOKEN, HIDDEN), ['K','KS'])

        ############## V Proj ################
        v_proj_out = v_proj(v_proj_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        # DIM0, HIDDEN, DIM1, TOKEN -> DIM0, DIM1, HIDDEN, TOKEN
        v_out_reshape_dims = self.make_named_shape([DIM0, DIM1, HIDDEN, TOKEN], [1, 1, kv_hidden_size_sharded, self.token_size])
        v_proj_out_reshaped = self.make_reshape(graph, v_proj_out, v_out_reshape_dims, TOKEN, 64)
        present_values = self.proto_utils['permute'](graph, v_proj_out_reshaped, (TOKEN, HIDDEN), ['V','VS'])
        ############## Q Proj ################
        q_proj_out = q_proj(q_proj_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)

        # DIM0, HIDDEN, DIM1, TOKEN -> DIM0, NUM_HEADS, HEAD_DIM, TOKEN
        # Note: NUM_HEADS should also be sharded here
        q_out_reshape_dims = self.make_named_shape([DIM0, NUM_HEADS, HEAD_DIM, TOKEN], [1,num_attention_heads_sharded, self.head_dim, self.token_size])
        q_proj_out_mha = self.make_reshape(graph, q_proj_out, q_out_reshape_dims, HEAD_DIM, 64)

        q_rope_out = q_rope(q_proj_out_mha)

        # QxKt
        past_keys_permuted = self.proto_utils['permute'](graph, past_keys(), (TOKEN_CACHE, HIDDEN))
        # DIM0, DIM1, HIDDEN, TOKEN_CACHE -> DIM0, NUM_HEADS, HEAD_DIM, TOKEN_CACHE
        # Note: NUM_HEADS should also be sharded here
        pk_out_reshape_dims = self.make_named_shape([DIM0, NUM_HEADS, HEAD_DIM, TOKEN_CACHE], [1,num_key_value_heads_sharded, self.head_dim, self.prmpt_context_size])
        past_keys_reshaped = self.make_reshape(graph, past_keys_permuted, pk_out_reshape_dims, HEAD_DIM, 64)
        QxKt_out = QxKt(q_rope_out + past_keys_reshaped, derive_name_from_layer_name=True)
        
        # Softmax
        attn_out = Attn(QxKt_out + valid_till(), axis = -1)
        attn_out_permuted = self.proto_utils['permute'](graph, attn_out, (TOKEN, TOKEN_CACHE))

        # PxV
        # DIM0, DIM1, TOKEN_CACHE, HIDDEN -> DIM0, TOKEN_CACHE, NUM_HEADS, HEAD_DIM
        # Note: NUM_HEADS should be sharded here 
        pv_out_reshape_dims = self.make_named_shape([DIM0, TOKEN_CACHE, NUM_HEADS, HEAD_DIM], [1, self.prmpt_context_size, num_key_value_heads_sharded, self.head_dim])
        past_values_reshaped = self.make_reshape(graph, past_values(), pv_out_reshape_dims, TOKEN_CACHE, 64)

        past_values_permuted = self.proto_utils['permute'](graph, past_values_reshaped, (TOKEN_CACHE, NUM_HEADS))
        PxV_out = PxV(attn_out_permuted + past_values_permuted, derive_name_from_layer_name=True)
        
        # Add permutation NNP to swap last two dimensions before comm node, for PxV batchdot
        PxV_out_permuted = self.proto_utils['permute'](graph, PxV_out, (TOKEN, HEAD_DIM))

        # Add comm node after PxV batchdot - operation=0, sync_group=0
        comm_node_after_PxV = self.proto_utils['Comm'](
            "PxV_comm",
            graph,
             out_group_axis=3,  # Grouping along the token_cache dimension (the last but one dimension)
            out_group_size=64,  # One element per group
            shard_idx=shard_idx,
            num_shards=self.num_shards,
            operation=0,
            sync_group=0,
            num_inputs=2
        )
        
        # Apply comm node after PxV
        comm_out_PxV = comm_node_after_PxV(PxV_out_permuted, derive_name_from_layer_name=True)
        # breakpoint() 
        # Continue with original computation flow
        # The dimensions have been properly adjusted for sharding through the reshapes and comm nodes
        # Now follow through to o_proj with the correct shapes
        
        # Continue with o_proj (this part is same as original)
        # Need to compute the pxv_out_reshape_dims correctly for the final reshaping
        # This should match what the original code expected
        pxv_out_reshape_dims = self.make_named_shape([DIM0, HIDDEN, DIM1, TOKEN], [1, self.hidden_size, 1, self.token_size])
        
        # Re-reshape as needed for the o_proj operation following comm
        comm_out_reshaped = self.make_reshape(graph, comm_out_PxV, pxv_out_reshape_dims, HIDDEN, 64)  # This is the same dimension from original
        
        ### Continue with o_proj (this part is same as original) ###
        o_proj_out =  o_proj(comm_out_reshaped, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        
        # Add comm node after o_proj - operation=0, sync_group=0
        comm_node_after_o_proj = self.proto_utils['Comm'](
            "o_proj_comm",
            graph,
             out_group_axis=3,  # Grouping along the token dimension (the last but one)
            out_group_size=64,  # One element per group
            shard_idx=shard_idx,
            num_shards=self.num_shards,
            operation=0,
            sync_group=1,
            num_inputs=2
        )
        
        # Apply comm node after o_proj 
        comm_out_o_proj = comm_node_after_o_proj(o_proj_out, derive_name_from_layer_name=True)
        
        # DIM0, HIDDEN, DIM1, TOKEN -> DIM0, DIM1, HIDDEN, TOKEN
        o_out_reshape_dims = self.make_named_shape([DIM0, DIM1, HIDDEN, TOKEN], [1, 1, self.hidden_size, self.token_size])
        o_proj_out_reshaped = self.make_reshape(graph, comm_out_o_proj, o_out_reshape_dims, HIDDEN, 64)

        o_proj_out_permuted = self.proto_utils['permute'](graph, o_proj_out_reshaped, (HIDDEN, TOKEN))

        return o_proj_out_permuted + input_ids_out
    
    def make_mlp_prompt(self, graph, input_nodes, shard_idx=0):
        # Override for sharded implementation
        # It will handle the MLP layer conversion
        residual_nodes = deepcopy(input_nodes)

        # Post Attention Layer Norm
        post_attention_rms_norm = self.proto_utils['LayerNorm'](POST_ATTENTION_RMS, graph, -1, 64, False, False, self.rms_norm_eps,True)

        # Projections - sharded
        gate_proj = self.proto_utils['Conv'](GATE_PROJ, graph, self.intermediate_size // self.num_shards, self.mlp_bias, [1, 1], [1,1], 1, 64,HIDDEN, True )
        up_proj = self.proto_utils['Conv'](UP_PROJ, graph, self.intermediate_size // self.num_shards, self.mlp_bias, [1, 1], [1,1], 1, 64,HIDDEN, True )
        down_proj = self.proto_utils['Conv'](DOWN_PROJ, graph, self.hidden_size//self.num_shards, False, [1, 1], [1,1], 1, 64,HIDDEN, True )

        #Eltwise ops
        swish_mul = self.proto_utils['Eltwise'](UP_GATE_MUL, graph, -1, 64,  self.proto_utils['EltwiseOperation'].PROD, True)
        final_residue = self.proto_utils['Eltwise'](MLP_RESIDUAL, graph, -1, 64,  self.proto_utils['EltwiseOperation'].SUM, True)

        #swish 
        swish = self.proto_utils['Swish'](GATE_PROJ_ACT,graph, 1,64,True)

        # Get the layernorm output first
        layernorm_out = post_attention_rms_norm(input_nodes)
        layernorm_out_perm = self.proto_utils['permute'](graph, layernorm_out, (HIDDEN, TOKEN))
        # DIM0, DIM1, HIDDEN, TOKEN -> DIM0, HIDDEN, DIM1, TOKEN
        rms_out_reshape_dims = self.make_named_shape([DIM0, HIDDEN, DIM1, TOKEN], [1, self.hidden_size, 1,self.token_size])
        up_input = self.make_reshape(graph, layernorm_out_perm, rms_out_reshape_dims, HIDDEN, 64)

        # Gate and up projections
        gate_input = deepcopy(up_input)
        gate_output = gate_proj(gate_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        up_output = up_proj(up_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)

        # Swish activation for gate output (not up output)
        swish_output = swish(gate_output, derive_name_from_layer_name=True)
        
        # Reshape and permute up output
        # DIM0, HIDDEN, DIM1, TOKEN -> DIM0, DIM1, HIDDEN, TOKEN
        up_out_reshape_dims = self.make_named_shape([DIM0,  DIM1, HIDDEN, TOKEN], [1, 1, self.intermediate_size//self.num_shards, self.token_size])
        up_output_reshaped = self.make_reshape(graph, up_output, up_out_reshape_dims, HIDDEN, 64)
        up_output_permuted = self.proto_utils['permute'](graph, up_output_reshaped, (HIDDEN, TOKEN))
        
        # Reshape and permute swish output  
        # DIM0, HIDDEN, DIM1, TOKEN -> DIM0, DIM1, HIDDEN, TOKEN
        swish_out_reshape_dims = self.make_named_shape([DIM0,  DIM1, HIDDEN, TOKEN], [1, 1, self.intermediate_size//self.num_shards, self.token_size])
        swish_output_reshaped = self.make_reshape(graph, swish_output, swish_out_reshape_dims, HIDDEN, 64)
        swish_output_permuted = self.proto_utils['permute'](graph, swish_output_reshaped, (HIDDEN, TOKEN))

        # Element-wise multiplication (gate * swish(up))
        swish_mul_output = swish_mul(swish_output_permuted + up_output_permuted, derive_name_from_layer_name=True)
        
        # Re-permute and reshape for down_proj
        swish_mul_output_permuted = self.proto_utils['permute'](graph, swish_mul_output, (HIDDEN, TOKEN))

        
        # Add comm node after swish_mul eltwise - operation=0, sync_group=0
        comm_node_after_swish_mul = self.proto_utils['Comm'](
            "swish_mul_comm",
            graph,
             out_group_axis=3,  # Grouping along the token dimension
            out_group_size=64,  # One element per group
            shard_idx=shard_idx,
            num_shards=self.num_shards,
            operation=0,
            sync_group=2,
            num_inputs=2
        )
        
        # Apply comm node after swish_mul
        comm_out_swish_mul = comm_node_after_swish_mul(swish_mul_output_permuted, derive_name_from_layer_name=True)

        # DIM0, DIM1, HIDDEN, TOKEN -> DIM0, HIDDEN, DIM1, TOKEN
        swish_mul_out_reshape_dims = self.make_named_shape([DIM0, HIDDEN, DIM1, TOKEN], [1, self.intermediate_size,  1, self.token_size])
        down_input = self.make_reshape(graph, comm_out_swish_mul, swish_mul_out_reshape_dims, HIDDEN, 64)
        # Down projection
        down_proj_output = down_proj(down_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        
        # Add comm node after down_proj - operation=0, sync_group=0
        comm_node_after_down_proj = self.proto_utils['Comm'](
            "down_proj_comm",
            graph,
             out_group_axis=3,  # Grouping along the token dimension
            out_group_size=64,  # One element per group
            shard_idx=shard_idx,
            num_shards=self.num_shards,
            operation=0,
            sync_group=3,
            num_inputs=2
        )
        
        # Apply comm node after down_proj
        comm_out_down_proj = comm_node_after_down_proj(down_proj_output, derive_name_from_layer_name=True)
        
        # Final reshape and permutation for residual connection
        # DIM0, HIDDEN, DIM1, TOKEN -> DIM0, DIM1, HIDDEN, TOKEN
        # breakpoint()
        down_out_reshape_dims = self.make_named_shape([DIM0, DIM1, HIDDEN, TOKEN], [1, 1, self.hidden_size ,self.token_size])
        down_proj_reshaped = self.make_reshape(graph, comm_out_down_proj, down_out_reshape_dims, HIDDEN, 64)
        down_proj_permuted = self.proto_utils['permute'](graph, down_proj_reshaped, ('HIDDEN', 'TOKEN'))
        
        # Apply the final residual connection (skip connection) 
        # This part should include the original input (residual)
        residual_out = final_residue(down_proj_permuted + residual_nodes, derive_name_from_layer_name=True)
        
        return residual_out
    
    def make_attention_mlp_connection_prompt(self, graph, input_nodes, shard_idx=0):
        attention_residual = self.proto_utils['Eltwise'](ATTENTION_RESIDUAL, graph, -1, 64,  self.proto_utils['EltwiseOperation'].SUM, True)
        residual_out = attention_residual(input_nodes)
        return residual_out

    #TODO: Handle for Batch 4 
    def make_attention_token(self, graph, token_cache_size, layer_idx=0, shard_idx=0):
        # Override for sharded implementation
        # It will handle the attention layer conversion for token processing
        hidden_size_sharded = self.hidden_size//self.num_shards
        kv_hidden_size = self.num_key_value_heads*self.head_dim
        kv_hidden_size_sharded = kv_hidden_size//self.num_shards
        
        ######################### Initialization #########################
        #Named shapes
        input_shape = self.make_named_shape([DIM0, DIM1, TOKEN, HIDDEN],[1,1,self.specd_size,self.hidden_size])
        kv_input_shape = self.make_named_shape([DIM0, DIM1, TOKEN_CACHE, HIDDEN],[1,1,token_cache_size,kv_hidden_size_sharded])
        rope_input_shape = self.make_named_shape([DIM0, DIM1, TOKEN, SIN_COS],[self.specd_size,1,1, self.head_dim * 2])
        valid_till_shape = self.make_named_shape([DIM0, DIM1, DIM2, DIM3],[1,1,1,1]) 
        requant_input_shape = self.make_named_shape([DIM0, TOKEN, DIM1, HIDDEN],[1,64,1,kv_hidden_size_sharded])

    
        #inputs
        input_ids = self.proto_utils['Input'](INPUT, graph,input_shape,-1,64, True)
        past_keys = self.proto_utils['Input'](PAST_KEYS, graph, kv_input_shape,-1,64, True, input_type='K')
        past_values = self.proto_utils['Input'](PAST_VALUES, graph, kv_input_shape,2,64, True, input_type='V')
        requant_in = self.proto_utils['Input'](PAST_VALUES_REQUANT, graph, requant_input_shape,1,64, True, input_type='RQ')

        position_ids = self.proto_utils['Input'](POSTION_EMBEDS, graph, rope_input_shape, 1, 1, False, input_type='R')
        valid_till = self.proto_utils['Input'](VALID_TILL, graph, valid_till_shape, 1, 1, False)
        #layernorms
        input_rms_norm = self.proto_utils['LayerNorm'](INPUT_RMS, graph, -1, 64, False, False, self.rms_norm_eps,True)

        # Projections - sharded
        attention_scale = 0
        if layer_idx in self.attention_scale_layers:
            attention_scale = 1/(self.head_dim**0.25) 
        k_proj = self.proto_utils['Inrp'](K_PROJ, graph, kv_hidden_size_sharded, self.attention_bias,1,64,True, attention_scale = attention_scale)
        q_proj = self.proto_utils['Inrp'](Q_PROJ, graph, hidden_size_sharded, self.attention_bias,1,64,True, attention_scale = attention_scale)
        v_proj = self.proto_utils['Inrp'](V_PROJ, graph, kv_hidden_size_sharded, self.attention_bias,1,1,True)
        o_proj = self.proto_utils['Inrp'](O_PROJ, graph, hidden_size_sharded, False,1,64,True)

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
        if layer_idx in self.attention_scale_layers:
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
        k_proj_out = k_proj(k_proj_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        # TOKEN, HIDDEN, DIM0, DIM1 -> TOKEN, NUM_HEADS, DIM0, HEAD_DIM
        # For sharded version, we divide num_heads by num_shards
        k_out_reshape_dims = self.make_named_shape([TOKEN, NUM_HEADS, DIM0, HEAD_DIM], [self.specd_size,self.num_key_value_heads//self.num_shards, 1, self.head_dim])
        k_proj_out_mha = self.make_reshape(graph, k_proj_out, k_out_reshape_dims, HEAD_DIM, 64)
        k_rope_out = k_rope(k_proj_out_mha)
        
        # Note: for token processing the dimensions are handled differently - no comm node here for K proj
        ############## V Proj ################
        v_proj_out = v_proj(v_proj_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        # TOKEN, HIDDEN, DIM0, DIM1 -> DIM0, TOKEN, DIM1, HIDDEN
        v_out_reshape_dims = self.make_named_shape([DIM0, TOKEN, DIM1, HIDDEN], [1, self.specd_size, 1, kv_hidden_size_sharded])
        v_proj_out_reshaped = self.make_reshape(graph, v_proj_out, v_out_reshape_dims, TOKEN, 1)
        ############## Requant ###############
        vt = valid_till()
        vt_copy = deepcopy(vt)
        past_values_last = requant_in()
        requant_out = requant(past_values_last + v_proj_out_reshaped + vt, derive_name_from_layer_name=True)
        
        ############## Q Proj ################
        q_proj_out = q_proj(q_proj_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)

        # TOKEN, HIDDEN, DIM0, DIM1 -> TOKEN, NUM_HEADS, DIM0, HEAD_DIM
        # For sharded version, we divide num_heads by num_shards
        q_out_reshape_dims = self.make_named_shape([TOKEN, NUM_HEADS, DIM0, HEAD_DIM], [self.specd_size,self.num_attention_heads//self.num_shards, 1, self.head_dim])
        q_proj_out_mha = self.make_reshape(graph, q_proj_out, q_out_reshape_dims, HEAD_DIM, 64)

        q_rope_out = q_rope(q_proj_out_mha)
        
        # TOKEN, NUM_HEADS, DIM0, HEAD_DIM -> DIM0, NUM_HEADS,  HEAD_DIM, TOKEN
        if self.specd_size == 1:
            q_rope_out_reshape_dims = self.make_named_shape([DIM0, NUM_HEADS,  HEAD_DIM, TOKEN], [1,self.num_attention_heads//self.num_shards, self.head_dim, 1])
            q_rope_out_reshaped = self.make_reshape(graph, q_rope_out, q_rope_out_reshape_dims, HEAD_DIM, 64)
        else:
            # Reshape, permute, reshape
            # TOKEN, NUM_HEADS, DIM0, HEAD_DIM --> 1, 1, TOKEN, NUM_HEADS*HEAD_DIM
            q_rope_out_initial_reshape_dims = self.make_named_shape([DIM0, DIM1, TOKEN, HIDDEN], [1, 1, self.specd_size, (self.num_attention_heads//self.num_shards)*self.head_dim])
            q_rope_out_initial_reshaped = self.make_reshape(graph, q_rope_out, q_rope_out_initial_reshape_dims, HIDDEN, 64)

            # 1, 1, TOKEN, NUM_HEADS*HEAD_DIM --> 1, 1, NUM_HEADS*HEAD_DIM, TOKEN 
            q_rope_out_initial_reshaped_permuted = self.proto_utils['permute'](graph, q_rope_out_initial_reshaped, (TOKEN, HIDDEN))

            # 1, 1, NUM_HEADS*HEAD_DIM, TOKEN --> 1, NUM_HEADS, HEAD_DIM, TOKEN 
            q_rope_out_reshape_dims = self.make_named_shape([DIM0, NUM_HEADS, HEAD_DIM, TOKEN], [1, self.num_attention_heads//self.num_shards, self.head_dim, self.specd_size])
            # HEAD_DIM is output group axis.
            # We pass 'HEAD_DIM' from dims. not HIDDEN from input.            
            q_rope_out_reshaped = self.make_reshape(graph, q_rope_out_initial_reshaped_permuted, q_rope_out_reshape_dims, HEAD_DIM, 64)

        ############ QxKt ################
        past_keys_permuted = self.proto_utils['permute'](graph, past_keys(), (TOKEN_CACHE, HIDDEN))
        # DIM0, DIM1, HIDDEN, TOKEN_CACHE -> DIM0, NUM_HEADS, HEAD_DIM, TOKEN_CACHE
        pk_out_reshape_dims = self.make_named_shape([DIM0, NUM_HEADS, HEAD_DIM, TOKEN_CACHE], [1,self.num_key_value_heads//self.num_shards, self.head_dim, token_cache_size])
        past_keys_reshaped = self.make_reshape(graph, past_keys_permuted, pk_out_reshape_dims, HEAD_DIM, 64)
        QxKt_out = QxKt(q_rope_out_reshaped + past_keys_reshaped, derive_name_from_layer_name=True)
        
        ############ Softmax ################  
        attn_out = Attn(QxKt_out + vt_copy, axis = -1)

        if self.specd_size == 1:
            # DIM0, NUM_HEADS, TOKEN, TOKEN_CACHE -> DIM0, NUM_HEADS, TOKEN_CACHE, TOKEN
            attn_out_reshape_dims = self.make_named_shape([DIM0, NUM_HEADS, TOKEN_CACHE, TOKEN], [1,self.num_attention_heads//self.num_shards, token_cache_size, 1])
            attn_out_reshaped = self.make_reshape(graph, attn_out, attn_out_reshape_dims, TOKEN_CACHE, 64)
        else:
            # DIM0, NUM_HEADS, TOKEN, TOKEN_CACHE -> DIM0, TOKEN, NUM_HEADS, TOKEN_CACHE 
            attn_out_permuted = self.proto_utils['permute'](graph, attn_out, (TOKEN, NUM_HEADS))

            # DIM0, TOKEN, NUM_HEADS, TOKEN_CACHE --> TOKEN, NUM_HEADS, TOKEN_CACHE, DIM0 
            attn_out_reshape_dims = self.make_named_shape([TOKEN, NUM_HEADS, TOKEN_CACHE, DIM0], [self.specd_size, self.num_attention_heads//self.num_shards, token_cache_size, 1])
            attn_out_reshaped = self.make_reshape(graph, attn_out_permuted, attn_out_reshape_dims, TOKEN_CACHE, 64)

        ################ PxV ################
        # DIM0, DIM1, TOKEN_CACHE, HIDDEN -> DIM1, TOKEN_CACHE, NUM_HEADS, HEAD_DIM        
        pv_out_reshape_dims = self.make_named_shape([DIM1, TOKEN_CACHE, NUM_HEADS, HEAD_DIM], [1, token_cache_size, self.num_key_value_heads//self.num_shards, self.head_dim])
        past_values_reshaped = self.make_reshape(graph, past_values(), pv_out_reshape_dims, TOKEN_CACHE, 64)

        past_values_permuted = self.proto_utils['permute'](graph, past_values_reshaped, (TOKEN_CACHE, NUM_HEADS))
        PxV_out = PxV(attn_out_reshaped + past_values_permuted, derive_name_from_layer_name=True)
        
        # Add comm node after PxV batchdot in token processing - operation=0, sync_group=0
        comm_node_after_PxV = self.proto_utils['Comm'](
            "PxV_comm_token",
            graph,
             out_group_axis=3,  # Grouping along the token dimension
            out_group_size=64,  # One element per group
            shard_idx=shard_idx,
            num_shards=self.num_shards,
            operation=0,
            sync_group=0,
            num_inputs=2
        )
        
        # Apply comm node after PxV
        comm_out_PxV = comm_node_after_PxV(PxV_out, derive_name_from_layer_name=True)
        
        # We need to ensure the same dimensional flow as original
        # The comm node should be applied before downstream reshapes
        # DIM0, NUM_HEADS, TOKEN, HIDDEN -> DIM0, HIDDEN, DIM1, TOKEN 
        if self.specd_size == 1:
            pxv_out_reshape_dims = self.make_named_shape([DIM0, HIDDEN, DIM1, TOKEN], [1, self.hidden_size, 1, 1])
        else:
            pxv_out_reshape_dims = self.make_named_shape([TOKEN, HIDDEN, DIM0, DIM1], [self.specd_size, self.hidden_size, 1, 1])

        PxV_out_reshaped = self.make_reshape(graph, comm_out_PxV, pxv_out_reshape_dims, HIDDEN, 64)
        ############# O_proj ###############
        o_proj_out =  o_proj(PxV_out_reshaped, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        
        # Add comm node after o_proj in token processing - operation=0, sync_group=0
        comm_node_after_o_proj = self.proto_utils['Comm'](
            "o_proj_comm_token",
            graph,
             out_group_axis=3,  # Grouping along the token dimension
            out_group_size=64,  # One element per group
            shard_idx=shard_idx,
            num_shards=self.num_shards,
            operation=0,
            sync_group=1,
            num_inputs=2
        )
        
        # Apply comm node after o_proj
        comm_out_o_proj = comm_node_after_o_proj(o_proj_out, derive_name_from_layer_name=True)
        
        # DIM0, HIDDEN, DIM1, TOKEN -> DIM0, DIM1, TOKEN, HIDDEN
        o_out_reshape_dims = self.make_named_shape([DIM0, DIM1, TOKEN, HIDDEN], [1, 1, self.specd_size, self.hidden_size])
        o_proj_out_reshaped = self.make_reshape(graph, comm_out_o_proj, o_out_reshape_dims, HIDDEN, 64)

        return o_proj_out_reshaped + input_ids_out
    
    def make_mlp_token(self, graph, input_nodes, shard_idx=0):
        # Override for sharded implementation
        # It will handle the MLP layer conversion for token processing
        residual_nodes = deepcopy(input_nodes)

        # Post Attention Layer Norm
        post_attention_rms_norm = self.proto_utils['LayerNorm'](POST_ATTENTION_RMS, graph, -1, 64, False, False, self.rms_norm_eps,True)

        # Projections - sharded
        gate_proj = self.proto_utils['Inrp'](GATE_PROJ, graph, self.intermediate_size // self.num_shards, self.mlp_bias,1,64,True)
        up_proj = self.proto_utils['Inrp'](UP_PROJ, graph, self.intermediate_size // self.num_shards, self.mlp_bias,1,64,True)
        down_proj = self.proto_utils['Inrp'](DOWN_PROJ, graph, self.hidden_size // self.num_shards, self.mlp_bias,1,64,True)

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
        up_out_reshape_dims = self.make_named_shape([TOKEN, DIM0, DIM1, HIDDEN], [self.specd_size, 1, 1, self.intermediate_size//self.num_shards])
        up_output_reshaped = self.make_reshape(graph, up_output, up_out_reshape_dims, HIDDEN, 64)
        
        ######## GATE PROJ ######
        gate_output = gate_proj(gate_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        swish_output = swish(gate_output, derive_name_from_layer_name=True)
        if self.specd_size == 1:
            # DIM0, HIDDEN, DIM1, TOKEN -> DIM0, DIM1, TOKEN, HIDDEN
            swish_out_reshape_dims = self.make_named_shape([DIM0, DIM1, TOKEN, HIDDEN], [1, 1, 1, self.intermediate_size//self.num_shards])
        else:
            # DIM0, HIDDEN, DIM1, TOKEN -> TOKEN, DIM0, DIM1, HIDDEN
            swish_out_reshape_dims = self.make_named_shape([TOKEN, DIM0, DIM1, HIDDEN], [self.specd_size, 1, 1, self.intermediate_size//self.num_shards])
        swish_output_reshaped = self.make_reshape(graph, swish_output, swish_out_reshape_dims, HIDDEN, 64)

        ######## DOWN BLOCK ##############
        # Add comm node after swish_mul eltwise - operation=0, sync_group=0
        swish_mul_output =  swish_mul(swish_output_reshaped + up_output_reshaped)
        
        # Add comm node in middle of MLP for token processing - operation=0, sync_group=0
        comm_node_after_swish_mul = self.proto_utils['Comm'](
            "swish_mul_comm_token",
            graph,
             out_group_axis=3,  # Grouping along the token dimension
            out_group_size=64,  # One element per group
            shard_idx=shard_idx,
            num_shards=self.num_shards,
            operation=0,
            sync_group=2,
            num_inputs=2
        )
        
        # Apply comm node after swish_mul
        comm_out_swish_mul = comm_node_after_swish_mul(swish_mul_output,  derive_name_from_layer_name=True)
        
        # DIM0, DIM1, TOKEN, HIDDEN -> TOKEN, HIDDEN, DIM0, DIM1
        swish_mul_out_reshape_dims = self.make_named_shape([TOKEN, HIDDEN, DIM0, DIM1], [self.specd_size, self.intermediate_size,  1, 1])
        down_input = self.make_reshape(graph, comm_out_swish_mul, swish_mul_out_reshape_dims, HIDDEN, 64)
        
        down_proj_output = down_proj(down_input, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        
        # Add comm node after down_proj in token processing - operation=0, sync_group=0
        comm_node_after_down_proj = self.proto_utils['Comm'](
            "down_proj_comm_token",
            graph,
             out_group_axis=3,  # Grouping along the token dimension
            out_group_size=64,  # One element per group
            shard_idx=shard_idx,
            num_shards=self.num_shards,
            operation=0,
            sync_group=3,
            num_inputs=2
        )
        
        # Apply comm node after down_proj
        comm_out_down_proj = comm_node_after_down_proj(down_proj_output, derive_name_from_layer_name=True)
        
        # DIM0, HIDDEN, DIM1, TOKEN -> DIM0, DIM1, TOKEN, HIDDEN
        down_out_reshape_dims = self.make_named_shape([DIM0, DIM1, TOKEN, HIDDEN], [1, 1, self.specd_size, self.hidden_size])
        down_proj_reshaped = self.make_reshape(graph, comm_out_down_proj, down_out_reshape_dims, HIDDEN, 64)

        final_out = final_residue(residual_nodes + down_proj_reshaped)
        return final_out
    
    def make_attention_mlp_connection_token(self, graph, input_nodes, shard_idx=0):
        # Override for sharded implementation
        attention_residual = self.proto_utils['Eltwise'](ATTENTION_RESIDUAL, graph, -1, 64,  self.proto_utils['EltwiseOperation'].SUM, True)
        residual_out = attention_residual(input_nodes)
        return residual_out
    
    
    def make_decoder_prompt(self, layer_idx):
        # Override for sharded implementation
        # Generate multiple files for each shard
        for shard_idx in range(self.num_shards):
            graph = self.proto_utils['DAG']()
            attention_outs = self.make_attention_prompt(graph, layer_idx = layer_idx, shard_idx=shard_idx)
            mlp_ins = self.make_attention_mlp_connection_prompt(graph,attention_outs, shard_idx=shard_idx)
            final_out = self.make_mlp_prompt(graph,mlp_ins, shard_idx=shard_idx)

            decoder_out = os.path.join(self.output_dir,'prmpt_prc_128',f'split_{layer_idx}_shard_{shard_idx}')
            caffemodel_dir = os.path.join(self.output_dir,'caffemodels')
            os.makedirs(decoder_out, exist_ok=True)
            os.makedirs(caffemodel_dir, exist_ok=True)  
            proto_out = os.path.join(decoder_out,f"split_{layer_idx}_shard_{shard_idx}.prototxt")
            out_caffemodel = os.path.join(caffemodel_dir,f'split_{layer_idx}_shard_{shard_idx}.caffemodel')
            out_so = os.path.join(decoder_out,f"split_{layer_idx}_shard_{shard_idx}_so.yaml")
            img_out = os.path.join(decoder_out,f"imgs") 
            self.get_proto(graph,save_to=proto_out)
            self.make_decoder_caffemodel(proto_out, layer_idx, out_caffemodel, shard_idx)
            self.get_so_yaml(graph,out_so, layer_idx, self.use_8bit_prompt)
            self.make_inputs(proto_out, img_out)
            if self.sdk:
                config_out = os.path.join(decoder_out, "dvrun_config.yaml")
                create_sdk_config(self.caffe, proto_out, out_caffemodel, img_out, out_so, config_out, self.license_key)
            else:
                config_out = os.path.join(decoder_out, "config_ara2.yaml")
                create_config(self.caffe, proto_out, out_caffemodel, img_out, out_so, config_out )
                create_run_py(decoder_out)
    
    def make_decoder_token(self, layer_idx, ctx):
        # Override for sharded implementation
        # Generate multiple files for each shard
        for shard_idx in range(self.num_shards):
            graph = self.proto_utils['DAG']()
            attention_outs = self.make_attention_token(graph, ctx, layer_idx=layer_idx, shard_idx=shard_idx)
            mlp_ins = self.make_attention_mlp_connection_token(graph,attention_outs, shard_idx=shard_idx)
            final_out = self.make_mlp_token(graph,mlp_ins, shard_idx=shard_idx)
            decoder_out = os.path.join(self.output_dir,'tkn_gen_'+ str(ctx),f'split_{layer_idx}_shard_{shard_idx}')
            os.makedirs(decoder_out, exist_ok=True) 
            proto_out = os.path.join(decoder_out,f"split_{layer_idx}_shard_{shard_idx}.prototxt")
            out_caffemodel = os.path.join(decoder_out,f'split_{layer_idx}_shard_{shard_idx}.caffemodel')
            out_so = os.path.join(decoder_out,f"split_{layer_idx}_shard_{shard_idx}_so.yaml")
            img_out = os.path.join(decoder_out,f"imgs") 
            self.get_proto(graph, save_to=proto_out)
            # self.make_decoder_caffemodel(proto_out, layer_idx, out_caffemodel)
            self.get_so_yaml(graph,out_so, layer_idx)
            self.make_inputs(proto_out, img_out)
            config_out = os.path.join(decoder_out, "config_ara2.yaml")
            if self.sdk:
                config_out = os.path.join(decoder_out, "dvrun_config.yaml")
                create_sdk_config(self.caffe, proto_out, self.caffe_model_layer_idx_map[layer_idx][shard_idx], img_out, out_so, config_out, self.license_key)
            else:
                # breakpoint()
                create_config(self.caffe, proto_out, self.caffe_model_layer_idx_map[layer_idx][shard_idx], img_out, out_so, config_out )
                create_run_py(decoder_out)

    def make_lmhead(self):
        for shard_idx in range(self.num_shards):
            lmhead_out = os.path.join(self.output_dir, f"lmhead/split_{self.num_hidden_layers}_shard_{shard_idx}")
            os.makedirs(lmhead_out, exist_ok=True)
            proto_out = os.path.join(lmhead_out, f"lmhead_shard_{shard_idx}.prototxt")
            out_caffemodel = os.path.join(lmhead_out,f"lmhead_shard_{shard_idx}.caffemodel") 
            out_so = os.path.join(lmhead_out, "lmhead_so.yaml")
            img_out = os.path.join(lmhead_out, "imgs") 
            graph = self.proto_utils['DAG']()
            self.make_lmhead_graph(graph, shard_idx)
            self.get_proto(graph, save_to=proto_out)
            self.make_lmhead_caffemodel(proto_out, out_caffemodel, shard_idx)
            self.get_so_yaml(graph,out_so, self.num_hidden_layers)
            self.make_inputs(proto_out, img_out)
            if self.sdk:
                config_out = os.path.join(lmhead_out, "dvrun_config.yaml")
                create_sdk_config(self.caffe, proto_out, out_caffemodel, img_out, out_so, config_out, self.license_key)
            else:
                config_out = os.path.join(lmhead_out, "config_ara2.yaml")
                create_config(self.caffe, proto_out, out_caffemodel, img_out, out_so, config_out )
                create_run_py(lmhead_out)
            # Override for sharded implementation
            # Simple interface that creates LM head structure for the specified shard
            # The actual complex sharding logic is handled by make_lmhead_graph and make_lmhead_caffemodel
            
    
    def make_lmhead_graph(self, graph, shard_idx=0):
        # Override for sharded implementation
        # Define the network topology for LM head with proper sharding
        
        # Define LM head layer with reduced output dimension (shard size)
        vocab_size = self.vocab_size
        shard_size = vocab_size // self.num_shards
        hidden_state_named_shape = self.make_named_shape([DIM0, DIM1, TOKEN, HIDDEN], [1,1,self.specd_size,self.hidden_size]) 
        input = self.proto_utils['Input'](INPUT, graph,hidden_state_named_shape,3,64, True)
        rms_norm = self.proto_utils['LayerNorm'](INPUT_RMS, graph, 3, 64, False, False, self.rms_norm_eps,True)
        rms_out = rms_norm(input(), derive_name_from_layer_name=True)
        if self.specd_size == 1:
            # DIM0, DIM1, TOKEN, HIDDEN -> DIM0, HIDDEN, DIM2, TOKEN
            rms_out_reshape_dims = self.make_named_shape([DIM0, HIDDEN, DIM2, TOKEN], [1,self.hidden_size, 1, 1])
        else:
            # DIM0, DIM1, TOKEN, HIDDEN -> TOKEN, HIDDEN, DIM0, DIM2, 
            rms_out_reshape_dims = self.make_named_shape([TOKEN, HIDDEN, DIM0, DIM2], [self.specd_size, self.hidden_size, 1, 1])  
        reshape = self.make_reshape(graph,rms_out, rms_out_reshape_dims, HIDDEN, 64)
        lm_head = self.proto_utils['Inrp'](
            LMHEAD, 
            graph,
            shard_size,  # Reduced output dimension for this shard
            False,        # No bias term for LM head (typically)
            1,            # Group
            64,           # Precision
            True         
        )
        out = lm_head(reshape, derive_name_from_layer_name=True, accumulation_dim_name=HIDDEN)
        
        # Add comm node as terminal layer for shard result aggregation
        comm_node = self.proto_utils['Comm'](
            "lmhead_comm",
            graph,
             out_group_axis=3,      # Group along the output dimension
            out_group_size=1,      # One element per group
            shard_idx=shard_idx,
            num_shards=self.num_shards,
            operation=1,           # GATHER operation
            sync_group=4,          # Default sync group
            num_inputs=2
        )
        final_out = comm_node(out, derive_name_from_layer_name=True)
        
    
    def make_lmhead_caffemodel(self, proto_out, out_caffemodel, shard_idx=0):
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
                shard_size = len(raw_weights_transposed_flattened) // self.num_shards 
                lm_head_data = raw_weights_transposed_flattened[shard_idx*shard_size: (shard_idx+1)*shard_size]
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


class Qwen2ShardedModel(ShardedModel):

    def __init__(self, config, hf_config, proto_utils, onnx_model, caffe, output_dir, sdk, num_shards = 1):
        super().__init__(config, hf_config, proto_utils, onnx_model, caffe, output_dir, sdk, num_shards)
    
    def get_activations_config(self):
        super().get_activations_config()
        if self.qxk_out_scale_32_layers == []:
            self.qxk_out_scale_32_layers = [i for i in range(self.num_hidden_layers)]
        if self.attention_scale_layers == []:
            self.attention_scale_layers = [0, self.num_hidden_layers-1]

class Qwen2_5VLShardedModel(Qwen2ShardedModel):

    def __init__(self, config, hf_config, proto_utils, onnx_model, caffe, output_dir:dict, sdk, num_shards = 1):
        super().__init__(config, hf_config, proto_utils, onnx_model, caffe, output_dir['llm'], sdk, num_shards)
        # From here onwards, self.output_dir is LLM-specific.
        #  Model.output_dir should also be made a dict.

        import sys
        from pathlib import Path
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
