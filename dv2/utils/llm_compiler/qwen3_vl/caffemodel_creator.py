from pathlib import Path
from onnx import ModelProto, TensorProto
import numpy as np
import sys

sys.path.append(str(Path(__file__).absolute().parent.parent / 'qwen2_5_vl'))
from qwen2_5_vl.caffemodel_creator import Qwen2_5_VL_VisionCaffemodelCreator
from qwen2_5_vl.constants import (
    ONNX_PATCH_EMBED_PROJ_WEIGHT,
    ONNX_PATCH_EMBED_ROTATION_WEIGHT,
)

class Qwen3_VL_VisionCaffemodelCreator(Qwen2_5_VL_VisionCaffemodelCreator):
  def __init__(self, prototxt_path:str, caffemodel_path:str, caffe_layer_weight_prefix:str, onnx_model:ModelProto, full_attn_layers:list, num_vision_layers:int, attention_bias:bool, hidden_size:int, num_attention_heads:int, num_key_value_heads:int, is_rotated:bool, deepstack_layers_list:list):
    super().__init__(prototxt_path, caffemodel_path, caffe_layer_weight_prefix, onnx_model, full_attn_layers, num_vision_layers, attention_bias, hidden_size, num_attention_heads, num_key_value_heads, is_rotated)
    assert len(deepstack_layers_list) > 0
    self.deepstack_layers_list = deepstack_layers_list

    # Often bias field is not set in hf_config, but safetensors may have bias weights and hence onnx may have it
    for name in self.onnx_initializer.keys():
        if 'attn.qkv.Add.bias' in name:
            # This string is model dependent.
            self.attention_bias = True

  def _make_patch_embed(self):
    rotation_weight_in_onnx = ONNX_PATCH_EMBED_ROTATION_WEIGHT in self.onnx_initializer.keys()
    if self.is_rotated != rotation_weight_in_onnx:
       raise AssertionError(f'is_rotated: {self.is_rotated}, but rotation_weight_in_onnx = {rotation_weight_in_onnx}')

    onnx_to_caffe_param_map = {}
    onnx_to_caffe_param_map[ONNX_PATCH_EMBED_PROJ_WEIGHT] = f'{self.caffe_layer_weight_prefix}_patch_embed_artificial_conv1d'
    if (self.is_rotated):
       onnx_to_caffe_param_map[ONNX_PATCH_EMBED_ROTATION_WEIGHT] =  f'{self.caffe_layer_weight_prefix}_patch_embed_rotation'

    for onnx_name, caffe_name in onnx_to_caffe_param_map.items():
        print(f"{onnx_name}  Shape: {self.onnx_initializer[onnx_name].dims} Data Type: {self.onnx_initializer[onnx_name].data_type}")
        raw_weights = np.frombuffer(self.onnx_initializer[onnx_name].raw_data, dtype=np.float32)
        caffe_param = self.caffemodel.params[caffe_name][0]
        print(f"             Caffe shape: {caffe_param.data.shape}")
        caffe_param.data[...] = raw_weights.reshape(caffe_param.data.shape)

  def _make_caffemodel_for_merger(self):
    direct_onnx_weight_to_caffe_layer_name = {
       'visual.merger.norm.weight': f'{self.caffe_layer_weight_prefix}_merger_layernorm',
       'visual.merger.linear_fc1.weight': f'{self.caffe_layer_weight_prefix}_merger_mlp_fc1',
       'visual.merger.linear_fc2.weight': f'{self.caffe_layer_weight_prefix}_merger_mlp_fc2',
    }
    direct_onnx_bias_to_caffe_layer_name = {
       'visual.merger.norm.bias': f'{self.caffe_layer_weight_prefix}_merger_layernorm',
       'visual.merger.linear_fc1.bias': f'{self.caffe_layer_weight_prefix}_merger_mlp_fc1',
       'visual.merger.linear_fc2.bias': f'{self.caffe_layer_weight_prefix}_merger_mlp_fc2',
    }

    for deepstack_layer_index in range(len(self.deepstack_layers_list)):
      direct_onnx_weight_to_caffe_layer_name.update({
        f'model.visual.deepstack_merger_list.{deepstack_layer_index}.norm.weight': f'qwen3_vl_deepstack_merger_{self.deepstack_layers_list[deepstack_layer_index]}_layernorm',
        f'model.visual.deepstack_merger_list.{deepstack_layer_index}.linear_fc1.weight': f'qwen3_vl_deepstack_merger_{self.deepstack_layers_list[deepstack_layer_index]}_mlp_fc1',
        f'model.visual.deepstack_merger_list.{deepstack_layer_index}.linear_fc2.weight': f'qwen3_vl_deepstack_merger_{self.deepstack_layers_list[deepstack_layer_index]}_mlp_fc2',
      })

      direct_onnx_bias_to_caffe_layer_name.update({
        f'model.visual.deepstack_merger_list.{deepstack_layer_index}.norm.bias': f'qwen3_vl_deepstack_merger_{self.deepstack_layers_list[deepstack_layer_index]}_layernorm',
        f'model.visual.deepstack_merger_list.{deepstack_layer_index}.linear_fc1.bias': f'qwen3_vl_deepstack_merger_{self.deepstack_layers_list[deepstack_layer_index]}_mlp_fc1',
        f'model.visual.deepstack_merger_list.{deepstack_layer_index}.linear_fc2.bias': f'qwen3_vl_deepstack_merger_{self.deepstack_layers_list[deepstack_layer_index]}_mlp_fc2',
      })
    
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

  def make_caffemodel_for_mlp_in_layer(self, layer_id:int):
    padded_onnx_weight_to_caffe_layer_name = {
        f'visual.blocks.{layer_id}.mlp.linear_fc1.MatMul.weight': f'{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_mlp_fc1',  # (1280, 3420) --> (3456, 1280, 1, 1)
        f'visual.blocks.{layer_id}.mlp.linear_fc2.MatMul.weight': f'{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_mlp_fc2',  #  (1280, 3420)  -->  (3456, 1280, 1, 1)
    }
    if self.attention_bias:
        padded_onnx_weight_to_caffe_layer_name[f'visual.blocks.{layer_id}.mlp.linear_fc1.Add.bias'] = f'{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_mlp_fc1'
        padded_onnx_weight_to_caffe_layer_name[f'visual.blocks.{layer_id}.mlp.linear_fc2.Add.bias'] = f'{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_mlp_fc2'

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
        print(f"             Caffe shape: {caffe_param.data.shape}")
        caffe_param.data[...] = raw_weights_transposed_flattened.reshape(caffe_param.data.shape)


  def make_caffemodel_for_layer(self, layer_id:int):
    direct_onnx_weight_to_caffe_layer_name = {
       f'visual.blocks.{layer_id}.norm1.weight': f'{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_lnorm1',
       f'visual.blocks.{layer_id}.norm2.weight': f'{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_lnorm2',
       f'visual.blocks.{layer_id}.attn.proj.MatMul.weight': f'{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_full_attn_o_proj',
    }
    if self.attention_bias:
        direct_onnx_weight_to_caffe_layer_name[f'visual.blocks.{layer_id}.attn.proj.Add.bias'] = f'{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_full_attn_o_proj'

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

    qkv_initializer_raw_weights =  np.frombuffer(self.onnx_initializer[f'visual.blocks.{layer_id}.attn.qkv.MatMul.weight'].raw_data, dtype=np.float32)
    q_weights, k_weights, v_weights = np.split(qkv_initializer_raw_weights, 3)

    caffe_param = self.caffemodel.params[f"{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_full_attn_q_proj"][0]
    caffe_param.data[...] = q_weights.reshape(caffe_param.data.shape)
    caffe_param = self.caffemodel.params[f"{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_full_attn_k_proj"][0]
    caffe_param.data[...] = k_weights.reshape(caffe_param.data.shape)    
    caffe_param = self.caffemodel.params[f"{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_full_attn_v_proj"][0]
    caffe_param.data[...] = v_weights.reshape(caffe_param.data.shape)

    print(f'About to populate vision qkv bias: self.attention_bias {self.attention_bias}')
    if self.attention_bias:
        qkv_bias_initializer =  np.frombuffer(self.onnx_initializer[f'visual.blocks.{layer_id}.attn.qkv.Add.bias'].raw_data, dtype=np.float32)
        flattened_qkv_bias =  np.transpose(qkv_bias_initializer.reshape(self.onnx_initializer[f'visual.blocks.{layer_id}.attn.qkv.Add.bias'].dims)).flatten()
        q_bias, k_bias, v_bias = np.split(flattened_qkv_bias, 3)

        caffe_param = self.caffemodel.params[f"{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_full_attn_q_proj"][1]
        caffe_param.data[...] = q_bias.reshape(caffe_param.data.shape)
        caffe_param = self.caffemodel.params[f"{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_full_attn_k_proj"][1]
        caffe_param.data[...] = k_bias.reshape(caffe_param.data.shape)    
        caffe_param = self.caffemodel.params[f"{self.caffe_layer_weight_prefix}_attn_block_{layer_id}_full_attn_v_proj"][1]
        caffe_param.data[...] = v_bias.reshape(caffe_param.data.shape)

    self.make_caffemodel_for_mlp_in_layer(layer_id)
