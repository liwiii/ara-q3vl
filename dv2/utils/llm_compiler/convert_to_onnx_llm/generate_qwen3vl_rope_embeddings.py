# generate_qwen3vl_rope_embeddings.py
#
# Qwen3-VL equivalent of generate_qwen2_5vl_rope_embeddings.py
# Key differences vs Qwen2.5-VL:
#   - Uses Qwen3VLForConditionalGeneration + AutoProcessor
#   - patch_size=16 (vs 14), spatial_merge_size=2 (same)
#   - mrope_section=[24,20,20] (vs [16,24,24])
#   - rope_theta=5_000_000 (vs 1_000_000)
#   - vision head_dim=64 (vs 80)
#   - MODEL_NAME changed to avoid colliding with qwen2.5 files

from contextlib import contextmanager
from datetime import date
from enum import Enum
import os
from typing import Optional
import torch
import mimetypes
from pathlib import Path
import re
from abc import abstractmethod
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLConfig, Qwen3VLProcessor
from qwen_vl_utils import process_vision_info
import logging
import argparse

from generate_qwen2_5vl_rope_embeddings import (
    get_temp_video, 
    get_temp_image, 
    VisionParams, 
    VideoParams, 
    ImageParams, 
    RopeOption, 
    ChatTemplateParams, 
    check_for_existence_and_return_path, 
    RopeFileHander,
    check_for_existence, 
    file_type_by_mime, 
    prepare_messages_for_video, 
    prepare_messages_for_text, 
    prepare_messages_for_image, 
    merge_cos_and_sin
)


logger = logging.getLogger(Path(__file__).name)
logging.basicConfig(level=logging.INFO)

#  <------------------------------ CONTEXT_LEN -------------------------->
#  System prompt                + Video token     +     placeholder Prompt
#  <--this should be multiple
#     of 128 bytes-->
#
#  72 tokens per frame. Inference has to crop video in such format.
#

FPS = 2
FRAMES = 16
# before variable length video, we used to support video of this length
NUM_SECONDS = 8
FRAME_HEIGHT = 12
FRAME_WIDTH = 12
CONTEXT_LEN = 4096
GENERATION_PROMPT_SIZE = 5
DEFAULT_SYS_PROMPT_SIZE = 15   # This coming from system prompt from chat jinja-template.
VIDEO_START_TOKEN_SIZE = 1 # Size of <|vision_start|>
VIDEO_END_TOKEN_SIZE = 1 # Size of <|vision_end|>

SYS_PROMPT_METADATA = 9 # "<|im_start|>system\n{sys_prompt}<|im_end|>\\n<|im_start|>user\\n" without {sys_prompt}

# Vision token (image or video) should begin at an index which is
# multiple of 128. e.g. [0] or [127]...
# System prompts of multiple of 128 tokens size of prompts are supported
SUPPORTED_SYS_PROMPT_SIZE_MULTIPLES = 128

TOKENS_PER_FRAME = (FRAME_HEIGHT * FRAME_WIDTH) // 2

# Qwen3-VL uses patch_size=16, spatial_merge_size=2
# so effective_patch_size = 32 (vs 28 for Qwen2.5-VL)
MODEL_NAME = "qwen_3_vl_4b" 

# ---- reuse the same param/helper classes (no HF-class dependency) ----

def grab_rope_parameters(vision_params: Optional[VisionParams], chat_template_params: ChatTemplateParams,
                         rope_file_handler: RopeFileHander,
                         rope_option: RopeOption,
                         model: Qwen3VLForConditionalGeneration,
                         cpu_or_cuda = "cpu",
                         output_dir = str):
    vision_config = model.model.config.vision_config
    modality = "TEXT" if vision_params is None else vision_params.get_modality()
    media_file_for_rope = None
    if vision_params is None:
        # text only rope
        logger.info("Constructing rope parameters for text")
        messages = prepare_messages_for_text(
                         chat_template_params)

    elif modality == 'VIDEO':

        # for loop to invoke for different values of sys_prompt_size based on SUPPORTED_SYS_PROMPT_SIZE_MULTIPLES
        # minimum sys prompt size is SUPPORTED_SYS_PROMPT_SIZE_MULTIPLES
        messages, media_file_for_rope = prepare_messages_for_video(video_params=vision_params, chat_template_params=chat_template_params,
                        #  video_path_for_rope=media_file_for_rope,
                         vision_config=vision_config)
        # logger.info(f"Using temporary video generated on the fly {media_file_for_rope}")

    elif modality == 'IMAGE':
        messages, media_file_for_rope = prepare_messages_for_image(
                         image_params=vision_params,
                         chat_template_params=chat_template_params,
                         vision_config=vision_config)
    else:
        logger.info("Unsupported Modality")
        exit(1)

    # For video, this will generate multiple files.
    # Call this script multiple times.

    # no text embeddings for now.
    cos_embeddings_file, sin_embeddings_file = rope_file_handler.get_cos_sin_file_path(chat_template_params=chat_template_params, modality = modality, rope_option=rope_option)

    if cpu_or_cuda.lower() != "cpu" and cpu_or_cuda.lower() != "cuda":
        logger.info(f"Error: unsupported cpu_or_cuda={cpu_or_cuda}")
        exit(1)
        
    effective_patch = (
        vision_config.patch_size *
        vision_config.spatial_merge_size
    )

    pixels_per_frame = effective_patch * effective_patch

    if vision_params:
        processor = Qwen3VLProcessor.from_pretrained(
            model.name_or_path,
            min_pixels=vision_params.frame_height * vision_params.frame_width * pixels_per_frame,
            max_pixels=vision_params.frame_height * vision_params.frame_width * pixels_per_frame,
            use_fast=True,
            padding_side="left",
            trust_remote_code=True,
        )
    else:
        processor = Qwen3VLProcessor.from_pretrained(
            model.name_or_path,
            use_fast=True,
            padding_side="left",
            trust_remote_code=True,
        )


    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    logger.info(f"messages:{messages}")
    image_inputs, video_inputs = process_vision_info(messages)
    # media file no longer needed
    if media_file_for_rope:
        media_file_for_rope.close()

    if image_inputs is not None:
        single_image = image_inputs[0]
        logger.info('-----Image input')
        logger.info(f"Format: {single_image.format}")            # File format (e.g. JPEG, PNG)
        logger.info(f"Mode: {single_image.mode}")                # Color mode (e.g. RGB, L)
        logger.info(f"Size: {single_image.size}")                # Dimensions (width, height)
        logger.info(f"Info: {single_image.info}")                # Dictionary with image info / metadata
        logger.info(f"Palette: {single_image.palette}")          # Palette info if available (usually for indexed images)

    if video_inputs is not None:
        single_video = video_inputs[0]
        logger.info('-----Video input')
        logger.info(f"Mode: {single_video.mode}")                # Color mode (e.g. RGB, L)
        logger.info(f"Size: {single_video.size}")                # Dimensions (width, height)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(cpu_or_cuda.lower())
    logger.info(f"inputs.input_ids.shape: {inputs.input_ids.shape}")
    # check that video token always starts after `sys_prompt_size` number of tokens
    if vision_params:
        vision_positions = (
            inputs.input_ids[0] == vision_params.get_tkn_id(model.config)
        ).nonzero(as_tuple=True)[0]

        if len(vision_positions) == 0:
            raise RuntimeError(
                f"Could not find vision token "
                f"{vision_params.get_tkn_id(model.config)} in input_ids"
            )


    # ---------------------------------------------------------
    # Qwen3-VL compatible rope index handling
    # ---------------------------------------------------------

    rope_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
    }

    if modality == "VIDEO":
        rope_kwargs["video_grid_thw"] = inputs.video_grid_thw

        if hasattr(inputs, "second_per_grid_ts"):
            rope_kwargs["second_per_grid_ts"] = inputs.second_per_grid_ts

    elif modality == "IMAGE":
        rope_kwargs["image_grid_thw"] = inputs.image_grid_thw

    position_ids, _ = model.model.get_rope_index(**rope_kwargs)

    # Dummy tensor with type set as desired.
    # model.model.language_model.rotary_emb in transformers will use it.
    dummy_hidden_states_tensor = torch.randn(1, 1, dtype=torch.float32, device=cpu_or_cuda.lower())
    if rope_option == RopeOption.VISION_ONLY:
        start_idx = vision_positions[0].item()

        # actual available vision tokens
        available_tokens = position_ids.shape[-1] - start_idx

        requested_tokens = min(
            vision_params.get_num_vision_tokens(),
            available_tokens
        )

        position_ids = position_ids[
            :,
            :,
            start_idx:start_idx + requested_tokens
        ]

        current_len = position_ids.shape[-1]

        if current_len < chat_template_params.context_size:
            pad_amount = chat_template_params.context_size - current_len

            position_ids = torch.nn.functional.pad(
                position_ids,
                pad=(0, pad_amount),
            )
        elif current_len > chat_template_params.context_size:
            position_ids = position_ids[:, :, :chat_template_params.context_size]

    cos_embeddings, sin_embeddings = model.model.language_model.rotary_emb(dummy_hidden_states_tensor, position_ids)


    logger.info(f"cos_embeddings.shape: {cos_embeddings.shape}")
    logger.info(f"sin_embeddings.shape: {sin_embeddings.shape}")

    mrope_section = model.config.text_config.rope_scaling['mrope_section']
    mrope_section = mrope_section * 2

    # Qwen3-VL rotary_emb already returns merged embeddings.
    # Only perform split/interleave if first dim == 3.

    if cos_embeddings.dim() >= 3 and cos_embeddings.shape[0] == 3:
        cos_chunks = cos_embeddings.split(mrope_section, dim=-1)
        sin_chunks = sin_embeddings.split(mrope_section, dim=-1)

        cos_embeddings = torch.cat(
            [chunk[i % 3] for i, chunk in enumerate(cos_chunks)],
            dim=-1
        ).unsqueeze(1)

        sin_embeddings = torch.cat(
            [chunk[i % 3] for i, chunk in enumerate(sin_chunks)],
            dim=-1
        ).unsqueeze(1)

    else:
        print("Qwen3-VL detected merged rotary embeddings; skipping mrope interleave")

        if cos_embeddings.dim() == 3:
            cos_embeddings = cos_embeddings.unsqueeze(1)

        if sin_embeddings.dim() == 3:
            sin_embeddings = sin_embeddings.unsqueeze(1)

    logger.info(f"After applying mrope_section: cos_embeddings.shape: {cos_embeddings.shape}")
    logger.info(f"After applying mrope_section: sin_embeddings.shape: {sin_embeddings.shape}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to {cos_embeddings_file}")
    torch.save(cos_embeddings.cpu(), cos_embeddings_file)
    logger.info(f"Writing to {sin_embeddings_file}")
    torch.save(sin_embeddings.cpu(), sin_embeddings_file)

def generate_qwen3vl_vision_rope_embeddings(output_dir:str, model:Qwen3VLForConditionalGeneration, config: Qwen3VLConfig):
    """
    Vision RoPE for Qwen3-VL.
    Identical logic to generate_qwen2_5_vl_vision_rope_embeddings but uses
    Qwen3-VL vision_config values (patch_size=16, spatial_merge_size=2).
    Output: vision_rotary_embeddings_cos.pt, vision_rotary_embeddings_sin.pt
    """
    logger.info("generating qwen3_vl vision rope embeddings..")
    # rope embedding is only used when doing cnn inference in group of 2 frames. So image input is enough for
    # dumping purpose
    vision_params = ImageParams(FRAME_HEIGHT, FRAME_WIDTH)
    vision_config = config.vision_config
    chat_template_params = ChatTemplateParams(CONTEXT_LEN, GENERATION_PROMPT_SIZE, DEFAULT_SYS_PROMPT_SIZE)
    effective_patch_size = vision_config.patch_size * vision_config.spatial_merge_size
    pixels_per_frame = effective_patch_size * effective_patch_size

    processor = Qwen3VLProcessor.from_pretrained(
        model.config._name_or_path,
        min_pixels=vision_params.frame_height * vision_params.frame_width * pixels_per_frame,
        max_pixels=vision_params.frame_height * vision_params.frame_width * pixels_per_frame,
        use_fast=True,
        padding_side="left",
        trust_remote_code=True,
    )

    messages, image_file = prepare_messages_for_image(
        image_params=vision_params,
        chat_template_params=chat_template_params,
        vision_config=vision_config)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    image_file.close()

    # Input arguments as per transforms 4.57.1
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    grid_thw = inputs.image_grid_thw
    rotary_pos_emb = model.visual.rot_pos_emb(grid_thw)

    print(f"Qwen3-VL rotary_pos_emb shape: {rotary_pos_emb.shape}")

    # Qwen3-VL no longer exposes get_window_index().
    # The returned embeddings are already ordered correctly.

    if rotary_pos_emb.dim() == 3:
        rotary_pos_emb = rotary_pos_emb.reshape(-1, rotary_pos_emb.shape[-1])

    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    vision_cos_file_name = Path(output_dir) / "vision_rotary_embeddings_cos.pt"
    vision_sin_file_name = Path(output_dir) / "vision_rotary_embeddings_sin.pt"
    logger.info(f"Saving vision cos in {vision_cos_file_name}")
    logger.info(f"Saving vision sin in {vision_sin_file_name}")
    torch.save(cos, vision_cos_file_name)
    torch.save(sin, vision_sin_file_name)

def generate_qwen3vl_rope_embeddings(output_dir:str, vl_rope_type:str, model:Qwen3VLForConditionalGeneration):
    """
    Drop-in replacement for generate_qwen2_5vl_rope_embeddings.
    Call from builder.py in place of the Qwen2.5-VL version.
    """
    logger.info("generating qwen3_vl language rope embeddings..")
    model_name_or_path = model.config._name_or_path
    rope_file_handler = RopeFileHander(MODEL_NAME, Path(output_dir))

    if vl_rope_type.upper() == "IMAGE":
        vision_params = ImageParams(FRAME_HEIGHT, FRAME_WIDTH)
        chat_template_params = ChatTemplateParams(CONTEXT_LEN, GENERATION_PROMPT_SIZE, DEFAULT_SYS_PROMPT_SIZE)
        rope_option = RopeOption.FULL_ROPE
        grab_rope_parameters(vision_params = vision_params, chat_template_params=chat_template_params, rope_file_handler=rope_file_handler, rope_option=rope_option, model = model, output_dir = output_dir)
    elif vl_rope_type.upper() == "VIDEO":
        vision_params = VideoParams(FPS, FRAME_HEIGHT, FRAME_WIDTH, NUM_SECONDS * FPS * (FRAME_HEIGHT * FRAME_WIDTH // 2))
        chat_template_params = ChatTemplateParams(CONTEXT_LEN, GENERATION_PROMPT_SIZE, DEFAULT_SYS_PROMPT_SIZE)
        rope_option = RopeOption.FULL_ROPE
        grab_rope_parameters(vision_params = vision_params, chat_template_params=chat_template_params, rope_file_handler=rope_file_handler, rope_option=rope_option, model = model, output_dir = output_dir)
    elif vl_rope_type.upper() == "MULTI_ROPE":
        # grab video ropes
        with rope_file_handler.staging_mode():
            for sys_prompt_size in range(SUPPORTED_SYS_PROMPT_SIZE_MULTIPLES, CONTEXT_LEN - TOKENS_PER_FRAME * 2, SUPPORTED_SYS_PROMPT_SIZE_MULTIPLES):
                chat_template_params = ChatTemplateParams(CONTEXT_LEN, GENERATION_PROMPT_SIZE, sys_prompt_size)
                max_num_video_tkns = 2 * TOKENS_PER_FRAME * ((CONTEXT_LEN - sys_prompt_size - VIDEO_END_TOKEN_SIZE - GENERATION_PROMPT_SIZE) // (2 * TOKENS_PER_FRAME))
                vision_params = VideoParams(FPS, FRAME_HEIGHT, FRAME_WIDTH, max_num_video_tkns)
                rope_option = RopeOption.VISION_ONLY
                grab_rope_parameters(vision_params = vision_params, chat_template_params=chat_template_params, rope_file_handler=rope_file_handler, rope_option=rope_option, model = model, output_dir = output_dir)

            # # # grab text rope only
            chat_template_params = ChatTemplateParams(CONTEXT_LEN, GENERATION_PROMPT_SIZE, DEFAULT_SYS_PROMPT_SIZE)
            vision_params = None
            rope_option = RopeOption.FULL_ROPE
            grab_rope_parameters(vision_params = vision_params, chat_template_params=chat_template_params, rope_file_handler=rope_file_handler, rope_option=rope_option, model = model, output_dir = output_dir)

        # Join the files. One for sin and one for cos.
        merge_cos_and_sin(rope_file_handler, chat_template_params)

if __name__ == "__main__":
    # this_file_basename = Path(__file__).name
    # today = date.today().strftime("%Y-%m-%d")
    # output_dir = f"__{this_file_basename}.manual_run.{today}"
    # Path(output_dir).mkdir(parents=True, exist_ok=True)    
    # vl_rope_type = "multi_rope"
    # model = Qwen3VLForConditionalGeneration.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
    # generate_qwen3vl_vision_rope_embeddings(output_dir, model, model.model.config)
    # generate_qwen3vl_rope_embeddings(output_dir, vl_rope_type, model)

    parser = argparse.ArgumentParser(description="Generate Qwen3-VL RoPE Embeddings.")
    this_file_basename = Path(__file__).name
    today = date.today().strftime("%Y-%m-%d")
    default_output_dir = f"__{this_file_basename}.manual_run.{today}"
    parser.add_argument(
        "-o", "--output_dir", 
        type=str, 
        default=default_output_dir,
        help="Path to the output directory (default: %(default)s)"
    )
    args = parser.parse_args()
    output_dir = args.output_dir.strip()
    Path(output_dir).mkdir(parents=True, exist_ok=True)    
    vl_rope_type = "multi_rope"
    model = Qwen3VLForConditionalGeneration.from_pretrained("/auto/worka/isatyashreyas.kusumanchi/hf_model/Qwen3-VL-4B-Instruct")
    generate_qwen3vl_vision_rope_embeddings(output_dir, model, model.model.config)
    generate_qwen3vl_rope_embeddings(output_dir, vl_rope_type, model)