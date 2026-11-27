import os
import glob
import torch

from convert_to_onnx_llm.builder import MistralModel
from safetensors import safe_open

class Qwen2_5VLModel(MistralModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, output_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.is_rotated = self._detect_if_safetensors_rotated(config)

        # Add Qwen2_5VLModel specific inputs.
        self.output_dir = output_dir
        self.vl_rope_type = extra_options['vl_rope_type']
        self.num_vision_layers = config.vision_config.depth if hasattr(config, "vision_config") and hasattr(config.vision_config, "depth") else 0
    
    def _detect_if_safetensors_rotated(self, config):
        safetensors_fileset = glob.glob(os.path.join(config._name_or_path, '*.safetensors'))
        print(f"safetensors_fileset:{safetensors_fileset}")
        rotation_weight_key = 'visual.patch_embed.rotation.weight'

        for filename in safetensors_fileset:
            with safe_open(filename, framework="pt", device="cpu") as f:
                if rotation_weight_key in f.keys():
                    print(f"visual.patch_embed.rotation present in {filename}")
                    self.rotation_tensor = f.get_tensor(rotation_weight_key)
                    return True

        print(f"visual.patch_embed.rotation.weight not found in {safetensors_fileset}")
        return False

    def make_vision_patch_embed(self, patch_embed):
        # For now, just copy over weights into .onnx.
        self.make_initializer(patch_embed.proj.weight, 'visual.patch_embed.proj.weight', to=self.io_dtype)

        # If model is rotated:
        if self.is_rotated:
            self.make_initializer(patch_embed.rotation.weight, 'visual.patch_embed.rotation.weight', to=self.io_dtype)

    def make_vision_block(self, vision_layer_id, vision_block):
        self.make_initializer(
            vision_block.norm1.weight,
            f"visual.blocks.{vision_layer_id}.norm1.weight",
             to=self.io_dtype)
        self.make_initializer(
            vision_block.norm2.weight,
            f"visual.blocks.{vision_layer_id}.norm2.weight",
            to=self.io_dtype)

        # Unlike language layer which has qkv_proj, the vision encoder has proj and qkv as two separate modules.

        # Also the attention for vision is different. For now we are just transmitting weights.

        self.make_initializer(
            vision_block.attn.proj.weight,
            f"visual.blocks.{vision_layer_id}.attn.proj.MatMul.weight",
             to=self.io_dtype)
        self.make_initializer(
            vision_block.attn.proj.bias,
            f"visual.blocks.{vision_layer_id}.attn.proj.Add.bias",
             to=self.io_dtype)

        self.make_initializer(
            vision_block.attn.qkv.weight,
            f"visual.blocks.{vision_layer_id}.attn.qkv.MatMul.weight",
             to=self.io_dtype)
        self.make_initializer(
            vision_block.attn.qkv.bias,
            f"visual.blocks.{vision_layer_id}.attn.qkv.Add.bias",
             to=self.io_dtype)

        self.make_mlp(vision_layer_id, vision_block.mlp, root_input='/not_sutructured/just_weights', onnx_layername_prefix='/visual/blocks')

    def make_vision_patch_merger(self, merger):
        self.make_initializer(merger.ln_q.weight, 'visual.merger.ln_q.weight', to=self.io_dtype)
        self.make_initializer(merger.mlp[0].bias, 'visual.merger.mlp.0.Add.bias', to=self.io_dtype)
        self.make_initializer(merger.mlp[0].weight, 'visual.merger.mlp.0.MatMul.weight', to=self.io_dtype)
        # merger.mlp[1] is Gelu
        self.make_initializer(merger.mlp[2].bias, 'visual.merger.mlp.1.Add.bias', to=self.io_dtype)
        self.make_initializer(merger.mlp[2].weight, 'visual.merger.mlp.1.MatMul.weight', to=self.io_dtype)


    def make_rotary_embedding_caches_from_scratch(self, **kwargs):
        # Ideally this should have reflected the transformers logic from 
        # $PYTHONPATH/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py
        # You can see attempt in builder_qwen2_5_vl_ideal.py
        #
        # For now we have obtained rope tables manually based on transformers logic
        # and using them here.

        rope_initial_cache_cos_matching_files = [f for f in os.listdir(self.output_dir) if f.endswith("cos.pt")]
        if len(rope_initial_cache_cos_matching_files) == 2:
            for file_name in rope_initial_cache_cos_matching_files:
                if 'vision_rotary_embeddings' in file_name:
                    vision_rope_initial_cache_cos = torch.load(os.path.join(self.output_dir, file_name), map_location="cpu")
                elif 'language_rotary_embeddings' in file_name:
                    language_initial_cache_cos = torch.load(os.path.join(self.output_dir, file_name))
        else:
            raise ValueError(f"Exactly two *cos.pt embedding file expected. Found {rope_initial_cache_cos_matching_files}.")

        rope_initial_cache_sin_matching_files = [f for f in os.listdir(self.output_dir) if f.endswith("sin.pt")]
        if len(rope_initial_cache_sin_matching_files) == 2:
            for file_name in rope_initial_cache_sin_matching_files:
                if 'vision_rotary_embeddings' in file_name:
                    vision_rope_initial_cache_sin = torch.load(os.path.join(self.output_dir, file_name), map_location="cpu")
                elif 'language_rotary_embeddings' in file_name:
                    language_initial_cache_sin = torch.load(os.path.join(self.output_dir, file_name))
        else:
            raise ValueError(f"Exactly two *sin.pt embedding file expected. Found {rope_initial_cache_sin_matching_files}.")

        return language_initial_cache_cos, language_initial_cache_sin, vision_rope_initial_cache_cos, vision_rope_initial_cache_sin
