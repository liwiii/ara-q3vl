import caffe
import numpy as np
from onnx import ModelProto, TensorProto
from pathlib import Path
import sys

from llm_compiler.qwen2_5_vl.constants import (
    ONNX_PATCH_EMBED_PROJ_WEIGHT,
    # CAFFE_PATCH_EMBED_CONV3D,
    ONNX_PATCH_EMBED_ROTATION_WEIGHT,
    # CAFFE_PATCH_EMBED_ROTATION
)
from llm_compiler.constants import *
from padding_util import pad_to_shape

class Qwen2_5_VL_VisionCaffemodelCreator:
  def __init__(self, prototxt_path:str, caffemodel_path:str, caffe_layer_weight_prefix:str, onnx_model:ModelProto, full_attn_layers:list, num_vision_layers:int, attention_bias:bool, hidden_size:int, num_attention_heads:int, num_key_value_heads:int, is_rotated:bool):
    self.prototxt_path = prototxt_path
    self.caffemodel_path = caffemodel_path
    self.caffe_layer_weight_prefix = caffe_layer_weight_prefix
    self.onnx_model = onnx_model
    self.is_rotated = is_rotated
    self.full_attn_layers = full_attn_layers
    self.num_vision_layers = num_vision_layers
    self.caffemodel = caffe.Net(prototxt_path, caffe.TEST)
    self.attention_bias = attention_bias
    self.hidden_size = hidden_size
    self.num_attention_heads = num_attention_heads
    self.num_key_value_heads = num_key_value_heads

    self.attention_bias = attention_bias
    self.onnx_initializer = {}
    for initializer in self.onnx_model.graph.initializer:
        if 'visual' in initializer.name:
            self.onnx_initializer[initializer.name] = initializer

            # Often bias field is not set in hf_config, but safetensors may have bias weights and hence onnx may have it
            if (QKV_BIAS in initializer.name):
                print(f'Found QKV_BIAS {QKV_BIAS} in {initializer.name}')
                self.attention_bias = True

  def make_caffemodel_for_mlp_in_layer(self, layer_id:int):
    padded_onnx_weight_to_caffe_layer_name = {
        f'visual.blocks.{layer_id}.mlp.gate_proj.MatMul.weight': f'{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_gate_proj',  # (1280, 3420) --> (3456, 1280, 1, 1)
        f'visual.blocks.{layer_id}.mlp.up_proj.MatMul.weight': f'{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_up_proj',  #  (1280, 3420)  -->  (3456, 1280, 1, 1)
        f'visual.blocks.{layer_id}.mlp.down_proj.MatMul.weight': f'{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_down_proj', #  (3420, 1280) -->  (1280, 3456, 1, 1)
    }
    if self.attention_bias:
        padded_onnx_weight_to_caffe_layer_name[f'visual.blocks.{layer_id}.mlp.gate_proj.Add.bias'] = f'{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_gate_proj'
        padded_onnx_weight_to_caffe_layer_name[f'visual.blocks.{layer_id}.mlp.up_proj.Add.bias'] = f'{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_up_proj'                
        padded_onnx_weight_to_caffe_layer_name[f'visual.blocks.{layer_id}.mlp.down_proj.Add.bias'] = f'{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_down_proj'

    for onnx_name, caffe_name in padded_onnx_weight_to_caffe_layer_name.items():
        print(f"{onnx_name}  Shape: {self.onnx_initializer[onnx_name].dims} Data Type: {TensorProto.DataType.Name(self.onnx_initializer[onnx_name].data_type)}")
        raw_weights = np.frombuffer(self.onnx_initializer[onnx_name].raw_data, dtype=np.float32)
        raw_weights_transposed_flattened = np.transpose(raw_weights.reshape(self.onnx_initializer[onnx_name].dims)).flatten()

        if onnx_name.endswith('.weight'):
            caffe_param = self.caffemodel.params[caffe_name][0]

            original_dims = self.onnx_initializer[onnx_name].dims
            original_dims_transposed =  [original_dims[1], original_dims[0]]
            input_dims_for_padding = original_dims_transposed + [1, 1]
        elif onnx_name.endswith('.bias'):
            if self.attention_bias:
                caffe_param = self.caffemodel.params[caffe_name][1]
                input_dims_for_padding = self.onnx_initializer[onnx_name].dims
            else:
                raise KeyError(f"Runtime error: self.attention_bias:{self.attention_bias}: but bias found in onnx layer {layer_id}:{onnx_name}")               
        else:
            raise KeyError(f"Unhandled onnx initializer key for layer {layer_id}:{onnx_name}")


        raw_weights_transposed_flattened_padded = pad_to_shape(raw_weights_transposed_flattened, input_dims_for_padding, caffe_param.data.shape)
        print(f"             Caffe shape: {caffe_param.data.shape}")
        caffe_param.data[...] = raw_weights_transposed_flattened_padded.reshape(caffe_param.data.shape)

  def make_caffemodel_for_layer(self, layer_id:int):
    attn_type = 'full' if layer_id in self.full_attn_layers else 'windowed'

    direct_onnx_weight_to_caffe_layer_name = {
       f'visual.blocks.{layer_id}.norm1.weight': f'{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_lnorm1',
       f'visual.blocks.{layer_id}.norm2.weight': f'{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_lnorm2',
       f'visual.blocks.{layer_id}.attn.proj.MatMul.weight': f'{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_{attn_type}_attn_o_proj',
    }
    if self.attention_bias:
        direct_onnx_weight_to_caffe_layer_name[f'visual.blocks.{layer_id}.attn.proj.Add.bias'] = f'{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_{attn_type}_attn_o_proj'

    for onnx_name, caffe_name in direct_onnx_weight_to_caffe_layer_name.items():
        print(f"{onnx_name}  Shape: {self.onnx_initializer[onnx_name].dims} Data Type: {TensorProto.DataType.Name(self.onnx_initializer[onnx_name].data_type)}")
        raw_weights = np.frombuffer(self.onnx_initializer[onnx_name].raw_data, dtype=np.float32)
        if onnx_name.endswith('.weight'):
            caffe_param = self.caffemodel.params[caffe_name][0]
        elif onnx_name.endswith('.bias'):
            if self.attention_bias:
                caffe_param = self.caffemodel.params[caffe_name][1]
            else:
                raise KeyError(f"Runtime error: self.attention_bias:{self.attention_bias}: but bias found in onnx layer {layer_id}:{onnx_name}")               
        else:
            raise KeyError(f"Unhandled onnx initializer key for layer {layer_id}:{onnx_name}")

        print(f"             Caffe shape: {caffe_param.data.shape}")
        caffe_param.data[...] = raw_weights.reshape(caffe_param.data.shape)

    # Split  visual.blocks.0.attn.qkv.MatMul.weight into 'qwen2_5_vl_attn_block_0_windowed_attn_q_proj' 'qwen2_5_vl_attn_block_0_windowed_attn_k_proj' 'qwen2_5_vl_attn_block_0_windowed_attn_v_proj'
    qkv_initializer_raw_weights =  np.frombuffer(self.onnx_initializer[f'visual.blocks.{layer_id}.attn.qkv.MatMul.weight'].raw_data, dtype=np.float32)
    q_weights, k_weights, v_weights = np.split(qkv_initializer_raw_weights, 3)

    caffe_param = self.caffemodel.params[f"{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_{attn_type}_attn_q_proj"][0]
    caffe_param.data[...] = q_weights.reshape(caffe_param.data.shape)
    caffe_param = self.caffemodel.params[f"{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_{attn_type}_attn_k_proj"][0]
    caffe_param.data[...] = k_weights.reshape(caffe_param.data.shape)    
    caffe_param = self.caffemodel.params[f"{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_{attn_type}_attn_v_proj"][0]
    caffe_param.data[...] = v_weights.reshape(caffe_param.data.shape)

    if self.attention_bias:
        qkv_bias_initializer =  np.frombuffer(self.onnx_initializer[f'visual.blocks.{layer_id}.attn.qkv.Add.bias'].raw_data, dtype=np.float32)
        flattened_qkv_bias =  np.transpose(qkv_bias_initializer.reshape(self.onnx_initializer[f'visual.blocks.{layer_id}.attn.qkv.Add.bias'].dims)).flatten()
        q_bias, k_bias, v_bias = np.split(flattened_qkv_bias, 3)

        caffe_param = self.caffemodel.params[f"{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_{attn_type}_attn_q_proj"][1]
        caffe_param.data[...] = q_bias.reshape(caffe_param.data.shape)
        caffe_param = self.caffemodel.params[f"{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_{attn_type}_attn_k_proj"][1]
        caffe_param.data[...] = k_bias.reshape(caffe_param.data.shape)    
        caffe_param = self.caffemodel.params[f"{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_{attn_type}_attn_v_proj"][1]
        caffe_param.data[...] = v_bias.reshape(caffe_param.data.shape)

    self.make_caffemodel_for_mlp_in_layer(layer_id)



  def _make_caffemodel_for_merger(self):
    direct_onnx_weight_to_caffe_layer_name = {
       'visual.merger.ln_q.weight': f'{self.caffe_layer_weight_prefix}_merger_layernorm',
       'visual.merger.mlp.0.MatMul.weight': f'{self.caffe_layer_weight_prefix}_merger_mlp_fc1',
       'visual.merger.mlp.1.MatMul.weight': f'{self.caffe_layer_weight_prefix}_merger_mlp_fc2',
    }
    direct_onnx_bias_to_caffe_layer_name = {
       'visual.merger.mlp.0.Add.bias': f'{self.caffe_layer_weight_prefix}_merger_mlp_fc1',
       'visual.merger.mlp.1.Add.bias': f'{self.caffe_layer_weight_prefix}_merger_mlp_fc2',
    }

    for onnx_name, caffe_name in direct_onnx_weight_to_caffe_layer_name.items():
        print(f"{onnx_name}  Shape: {self.onnx_initializer[onnx_name].dims} Data Type: {self.onnx_initializer[onnx_name].data_type}")
        raw_weights = np.frombuffer(self.onnx_initializer[onnx_name].raw_data, dtype=np.float32)
        caffe_param = self.caffemodel.params[caffe_name][0]
        print(f"             Caffe shape: {caffe_param.data.shape}")
        caffe_param.data[...] = raw_weights.reshape(caffe_param.data.shape)

    for onnx_name, caffe_name in direct_onnx_bias_to_caffe_layer_name.items():
        print(f"{onnx_name}  Shape: {self.onnx_initializer[onnx_name].dims} Data Type: {self.onnx_initializer[onnx_name].data_type}")
        raw_weights = np.frombuffer(self.onnx_initializer[onnx_name].raw_data, dtype=np.float32)
        caffe_param = self.caffemodel.params[caffe_name][1]
        print(f"             Caffe shape: {caffe_param.data.shape}")
        caffe_param.data[...] = raw_weights.reshape(caffe_param.data.shape)

  def _make_patch_embed(self):
    onnx_initializer = {}
    for initializer in self.onnx_model.graph.initializer:
        if 'visual.patch_embed' in initializer.name:
           onnx_initializer[initializer.name] = initializer

    rotation_weight_in_onnx = ONNX_PATCH_EMBED_ROTATION_WEIGHT in onnx_initializer.keys()
    if self.is_rotated != rotation_weight_in_onnx:
       raise AssertionError(f'is_rotated: {self.is_rotated}, but rotation_weight_in_onnx = {rotation_weight_in_onnx}')

    onnx_to_caffe_param_map = {}
    onnx_to_caffe_param_map[ONNX_PATCH_EMBED_PROJ_WEIGHT] = f'{self.caffe_layer_weight_prefix}_patch_embed_conv3d'
    if (self.is_rotated):
       onnx_to_caffe_param_map[ONNX_PATCH_EMBED_ROTATION_WEIGHT] =  f'{self.caffe_layer_weight_prefix}_patch_embed_rotation'

    for onnx_name, caffe_name in onnx_to_caffe_param_map.items():
        print(f"{onnx_name}  Shape: {self.onnx_initializer[onnx_name].dims} Data Type: {self.onnx_initializer[onnx_name].data_type}")
        raw_weights = np.frombuffer(self.onnx_initializer[onnx_name].raw_data, dtype=np.float32)
        caffe_param = self.caffemodel.params[caffe_name][0]
        print(f"             Caffe shape: {caffe_param.data.shape}")
        caffe_param.data[...] = raw_weights.reshape(caffe_param.data.shape)

  def make_caffemodel(self):
    self._make_patch_embed()
    for layer_id in range(self.num_vision_layers):
        self.make_caffemodel_for_layer(layer_id)
    self._make_caffemodel_for_merger()
    self.caffemodel.save_hdf5(self.caffemodel_path)
    print(f"Caffe model saved to {self.caffemodel_path}")