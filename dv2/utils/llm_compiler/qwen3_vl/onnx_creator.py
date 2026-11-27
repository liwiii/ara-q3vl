import os
import glob
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).absolute().parents[2]))
from ort_genai.May2026.models.builders.qwen import Qwen3VLTextModel
from safetensors import safe_open
from transformers import (
    AutoConfig,
    Qwen3VLForConditionalGeneration,
)
from ort_genai.May2026.models.builders.base import Model

class Qwen3VLOnnxCreator(Qwen3VLTextModel):
    def __init__(self, config, io_dtype, onnx_dtype, ep, cache_dir, output_dir, extra_options):
        super().__init__(config, io_dtype, onnx_dtype, ep, cache_dir, extra_options)
        self.is_rotated = self._detect_if_safetensors_rotated(config)
        self.config = config
        self.extra_options = extra_options
        self.model_name_or_path = config._name_or_path
        # Add Qwen3VLOnnxCreator specific inputs.
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
        self.make_initializer(patch_embed.proj.bias, 'visual.patch_embed.proj.bias', to=self.io_dtype)

        # If model is rotated:
        if self.is_rotated:
            self.make_initializer(patch_embed.rotation.weight, 'visual.patch_embed.rotation.weight', to=self.io_dtype)

    def make_vision_pos_embed(self, module):
        self.make_initializer(module.weight, 'model.visual.pos_embed.weight', to=self.io_dtype)

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
        self.make_initializer(merger.linear_fc1.weight, 'visual.merger.linear_fc1.weight', to=self.io_dtype)
        self.make_initializer(merger.linear_fc1.bias, 'visual.merger.linear_fc1.bias', to=self.io_dtype)
        self.make_initializer(merger.linear_fc2.bias, 'visual.merger.linear_fc2.bias', to=self.io_dtype)
        self.make_initializer(merger.linear_fc2.weight, 'visual.merger.linear_fc2.weight', to=self.io_dtype)
        self.make_initializer(merger.norm.bias, 'visual.merger.norm.bias', to=self.io_dtype)
        self.make_initializer(merger.norm.weight, 'visual.merger.norm.weight', to=self.io_dtype)


    def make_vision_deepstack_patch_merger(self, name, module):
        self.make_initializer(module.linear_fc1.weight, f'{name}.linear_fc1.weight', to=self.io_dtype)
        self.make_initializer(module.linear_fc1.bias, f'{name}.linear_fc1.bias', to=self.io_dtype)
        self.make_initializer(module.linear_fc2.weight, f'{name}.linear_fc2.weight', to=self.io_dtype)
        self.make_initializer(module.linear_fc2.bias, f'{name}.linear_fc2.bias', to=self.io_dtype)
        self.make_initializer(module.norm.weight, f'{name}.norm.weight', to=self.io_dtype)
        self.make_initializer(module.norm.bias, f'{name}.norm.bias', to=self.io_dtype)
            
            


    def make_rotary_embedding_caches_from_scratch(self, **kwargs):
        # Ideally this should have reflected the transformers logic from 
        # $PYTHONPATH/transformers/models/qwen2_5_vl/modeling_qwen3`_vl.py
        # You can see attempt in builder_qwen2_5_vl_ideal.py
        #
        # For now we have obtained rope tables manually based on transformers logic
        # and using them here.

        rope_initial_cache_cos_matching_files = [f for f in os.listdir(self.output_dir) if f.endswith("cos.pt")]
        vision_rope_initial_cache_cos = None
        language_initial_cache_cos = None
        for file_name in rope_initial_cache_cos_matching_files:
            if 'vision_rotary_embeddings' in file_name:
                vision_rope_initial_cache_cos = torch.load(
                    os.path.join(self.output_dir, file_name),
                    map_location="cpu"
                )
            elif 'final_full_rope_cos' in file_name:
                language_initial_cache_cos = torch.load(
                    os.path.join(self.output_dir, file_name),
                    map_location="cpu"
                )
        if vision_rope_initial_cache_cos is None:
            raise ValueError("Could not find vision cos rope file")
        if language_initial_cache_cos is None:
            raise ValueError("Could not find language final cos rope file")

        rope_initial_cache_sin_matching_files = [f for f in os.listdir(self.output_dir) if f.endswith("sin.pt")]
        vision_rope_initial_cache_sin = None
        language_initial_cache_sin = None
        for file_name in rope_initial_cache_sin_matching_files:
            if 'vision_rotary_embeddings' in file_name:
                vision_rope_initial_cache_sin = torch.load(
                    os.path.join(self.output_dir, file_name),
                    map_location="cpu"
                )
            elif 'final_full_rope_sin' in file_name:
                language_initial_cache_sin = torch.load(
                    os.path.join(self.output_dir, file_name),
                    map_location="cpu"
                )
        if vision_rope_initial_cache_sin is None:
            raise ValueError("Could not find vision sin rope file")
        if language_initial_cache_sin is None:
            raise ValueError("Could not find language final sin rope file")

        return language_initial_cache_cos, language_initial_cache_sin, vision_rope_initial_cache_cos, vision_rope_initial_cache_sin

    def make_rope_initializers(self):
        extra_kwargs = {"num_hidden_layers": self.num_layers} if "num_hidden_layers" in self.extra_options else {}
        model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_name_or_path, cache_dir=self.cache_dir, token=self.hf_token, trust_remote_code=True, **extra_kwargs)
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).absolute().parents[1]/"convert_onnx_to_llm"))
        from generate_qwen3vl_rope_embeddings import generate_qwen3vl_rope_embeddings, generate_qwen3vl_vision_rope_embeddings
        if self.vl_rope_type is None:
            rope_arg = 'IMAGE'
        else:
            rope_arg = self.vl_rope_type
        print(f"builder.py: Calling generate_qwen3vl_rope_embeddings with rope_arg {rope_arg} --> {self.output_dir}")
        generate_qwen3vl_rope_embeddings(output_dir=self.output_dir, vl_rope_type=rope_arg, model=model)
        print(f"builder.py: Calling generate_qwen3vl_vision_rope_embeddings --> {self.output_dir}")
        generate_qwen3vl_vision_rope_embeddings(output_dir=self.output_dir, model=model, config = self.config)

        cos_cache_name, sin_cache_name = self.make_rotary_embedding_caches()

    def load_weights(self, input_path):
        # For quantized models (e.g., Quark, AWQ, GPTQ) or GGUF, use base class logic
        # which loads weights directly via QuantModel
        if self.quant_type is not None or input_path.endswith(".gguf"):
            return super().load_weights(input_path)

        print("Loading Qwen3VLForConditionalGeneration model...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_name_or_path,
            cache_dir=self.cache_dir,
            token=self.hf_token,
            trust_remote_code=self.hf_remote,
        )

        if self.is_rotated:
            old_forward = model.visual.patch_embed.forward
            rotation = torch.nn.Linear(model.config.vision_config.hidden_size, model.config.vision_config.hidden_size, False, model.device, dtype = torch.bfloat16)
            rotation.load_state_dict({
                "weight": self.rotation_tensor
            })
            model.visual.patch_embed.register_module('rotation', rotation)
            def new_forward(pixels):
                output = old_forward(pixels)
                return model.visual.patch_embed.rotation(output)
            model.visual.patch_embed.forward = new_forward
        
        return model

    def make_model(self, input_path):
        self.make_rope_initializers()        #We do this because we want static rope initializers instead of dynamic.
        
        super().make_model(input_path)
        
        