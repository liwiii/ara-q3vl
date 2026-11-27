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

# The logic in this file invokes some parts of the model for a given input
# and grabs the position_ids.

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
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info
import logging
logger = logging.getLogger(Path(__file__).name)
logging.basicConfig(level=logging.INFO)

#  <------------------------------ CONTEXT_LEN -------------------------->
#  System prompt                + Video token     +     placeholder Prompt
#  <--this should be multiple
#     of 128 bytes-->
#
#  72 tokens per frame. Inference has to crop video in such format.
#

def get_temp_video(duration_seconds:int):
    import cv2
    import numpy as np
    import tempfile

    # Video settings
    width, height = 1024, 1024

    # This is much more than FPS to simulate a real user input.
    # Our model will still process at only FPS rate.
    fps = 16

    num_frames = fps * duration_seconds

    # Create a temporary file path with .mp4 suffix in the system tmp directory
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=True)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for .mp4
    out = cv2.VideoWriter(temp_file.name, fourcc, fps, (width, height))

    for _ in range(num_frames):
        # Create a dummy frame with ones
        frame = np.ones((height, width, 3), dtype=np.uint8)
        out.write(frame)

    out.release()

    return temp_file


def get_temp_image():
    from PIL import Image
    import tempfile

    # Create a dummy image (1024x1024) with a solid gray color
    image = Image.new('RGB', (1024, 1024), color=(128, 128, 128))  # RGB gray
    tmp_file = tempfile.NamedTemporaryFile(delete=True, suffix='.png')
    image.save(tmp_file.name)
    return tmp_file 

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

MODEL_NAME = "qwen_2_5_vl_3b"

class VisionParams:
    def __init__(self, frame_height, frame_width):
        self.frame_height = frame_height
        self.frame_width = frame_width

    def get_tokens_per_frame(self):
        return (self.frame_height * self.frame_width // 2)

    @abstractmethod
    def get_modality(self):
        pass

    @abstractmethod
    def get_num_vision_tokens(self):
        pass

    @abstractmethod
    def get_tkn_id(self, config: Qwen2_5_VLConfig):
        pass

class VideoParams(VisionParams):
    def __init__(self, fps, frame_height, frame_width, video_tokens):
        super().__init__(frame_height, frame_width)
        self.fps = fps
        self.video_tokens = video_tokens

    def get_modality(self):
        return "VIDEO"

    def get_tkn_id(self, config):
        return config.video_token_id

    def get_num_vision_tokens(self):
        return self.video_tokens

    def get_num_seconds(self):
        return (self.video_tokens // self.fps) // (self.get_tokens_per_frame())

class ImageParams(VisionParams):
    def __init__(self, frame_height, frame_width):
        super().__init__(frame_height, frame_width)

    def get_modality(self):
        return "IMAGE"

    def get_tkn_id(self, config):
        return config.image_token_id

    def get_num_vision_tokens(self):
        # 2 frames for image because qwen2.5 vl processor has the requirement
        return (self.frame_height * self.frame_width // 2) * 2

class RopeOption(Enum):
    FULL_ROPE = 1
    VISION_ONLY = 2,




class ChatTemplateParams:
    def __init__(self, context_len, generation_prompt_size, sys_prompt_size):
        self.context_size = context_len
        self.generation_prompt_size = generation_prompt_size
        self.sys_prompt_size = sys_prompt_size
        self.vision_start_tkn_size = 1
        self.vision_end_tkn_size = 1
        self.sys_prompt_metadata = 9 # "<|im_start|>system\n{sys_prompt}<|im_end|>\\n<|im_start|>user\\n" without {sys_prompt}

    def get_num_tokens_filled(self, vision_params: Optional[VisionParams]):
        total_tokens = self.sys_prompt_size + self.generation_prompt_size
        if vision_params:
            total_tokens += vision_params.get_num_vision_tokens() + self.vision_end_tkn_size
        return total_tokens

def check_for_existence_and_return_path(file_path:Path):
    original_file_path = file_path

    if file_path.exists():
        logger.info(f"Target file already exists: {file_path}")
        new_file_path = str(file_path) + '.old'
        logger.info(f"Renaming it to {new_file_path}")
        file_path.rename(new_file_path)

    return original_file_path

class RopeFileHander:
    def __init__(self, model_name, output_dir: Path):
        self.model_name = model_name
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok = True)

    @contextmanager
    def staging_mode(self):
        try:
            prev_dir = self.output_dir
            self.output_dir = self.output_dir / "stage_dir"
            self.output_dir.mkdir(exist_ok=True)
            yield self  
        finally:
            self.output_dir = prev_dir

    def get_cos_file_name(self, chat_template_params: ChatTemplateParams, modality, rope_option: RopeOption):
        if modality == "TEXT":
            return f"{self.model_name}_language_rotary_embeddings_context_len_{chat_template_params.context_size}_text_cos.pt"
        if rope_option == RopeOption.VISION_ONLY:
                option = "vision_only"
        elif rope_option == RopeOption.FULL_ROPE:
            option = "full_rope"
        else:
            logger.info(f"Unhandled {rope_option}!!")
            exit()
        return f"{self.model_name}_language_rotary_embeddings_context_len_{chat_template_params.context_size}_sys_size_{chat_template_params.sys_prompt_size}_{modality.lower()}_{option}_cos.pt"

    def get_sin_file_name(self,chat_template_params: ChatTemplateParams, modality: str, rope_option: RopeOption):
        if modality == "TEXT":
            return f"{self.model_name}_language_rotary_embeddings_context_len_{chat_template_params.context_size}_text_sin.pt"
        if rope_option == RopeOption.VISION_ONLY:
                option = "vision_only"
        elif rope_option == RopeOption.FULL_ROPE:
            option = "full_rope"
        else:
            logger.info(f"Unhandled {rope_option}!!")
            exit()
        return f"{self.model_name}_language_rotary_embeddings_context_len_{chat_template_params.context_size}_sys_size_{chat_template_params.sys_prompt_size}_{modality.lower()}_{option}_sin.pt"


    # def get_sin_file_path(self, chat_template_params):
    #     return self.output_dir / f"{self.model_name}_language_rotary_embeddings_context_len_{chat_template_params.context_size}_sin.pt"

    # def get_cos_file_path(self, chat_template_params):
    #     return self.output_dir / f"{self.model_name}_language_rotary_embeddings_context_len_{chat_template_params.context_size}_cos.pt"

    def get_cos_sin_file_path(self, chat_template_params: ChatTemplateParams, modality: str, rope_option: RopeOption, read = False):
        cos_file = self.output_dir / self.get_cos_file_name(chat_template_params, modality, rope_option)
        sin_file= self.output_dir / self.get_sin_file_name(chat_template_params, modality, rope_option)
        if not read:
            cos_file = check_for_existence_and_return_path(cos_file)
            sin_file = check_for_existence_and_return_path(sin_file)
        return cos_file, sin_file 

def check_for_existence(file_path:str):
    if os.path.exists(file_path):
        logger.info(f"Target file already exists: {file_path}")
        logger.info(f"Renaming it to {file_path + '.old'}")
        os.rename(file_path, file_path + '.old')

def file_type_by_mime(path:str):
    mime, _ = mimetypes.guess_type(path)
    if mime:
        if mime.startswith('image/'):
            return 'IMAGE'
        elif mime.startswith('video/'):
            return 'VIDEO'
    return 'unknown'

def prepare_messages_for_video(video_params: VideoParams, chat_template_params: ChatTemplateParams, vision_config: Qwen2_5_VLConfig):
    media_file_for_rope = get_temp_video(video_params.get_num_seconds())
    logger.info(f"Constructing rope parameters from {media_file_for_rope}")
    sys_prompt = (chat_template_params.sys_prompt_size - SYS_PROMPT_METADATA) * "😁"

    video_token_size = video_params.video_tokens
    logger.info(f"video_token_size: {video_token_size}")
    context_len = chat_template_params.context_size

    if video_token_size > context_len:
        logger.info(f"video_token_size {video_token_size} too high. context_len is {context_len}")
        exit()

    if (chat_template_params.get_num_tokens_filled(video_params)) > chat_template_params.context_size:
        logger.info(f"video_token_size + generation_prompt_size + video_token_size : {chat_template_params.get_num_tokens_filled(video_params)}")
        logger.info(f"number of tokens too high. context_len is {context_len}")
        exit()


    logger.info(f"Size without dummy letters:{chat_template_params.get_num_tokens_filled(video_params)}")
    logger.info(f"           =>dummy letters:{context_len - chat_template_params.get_num_tokens_filled(video_params)}")
    prompt = ("😁" * (context_len - chat_template_params.get_num_tokens_filled(video_params))).strip()
    logger.info(f"prompt {prompt}")

    messages_for_video = []
    messages_for_video.append({
        "role": "system",
        "content": [{"type": "text", "text": sys_prompt}]
    })


    messages_for_video.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": media_file_for_rope.name,
                    "fps": video_params.fps,
                    "resized_height": video_params.frame_height * vision_config.spatial_merge_size *  vision_config.patch_size,
                    "resized_width": video_params.frame_width * vision_config.spatial_merge_size *  vision_config.patch_size,
                },
                {"type": "text", "text": prompt},
            ],
        }
    )
    return messages_for_video, media_file_for_rope

def prepare_messages_for_text(chat_template_params: ChatTemplateParams):
    prompt = "😁" * (chat_template_params.context_size - chat_template_params.sys_prompt_size - chat_template_params.generation_prompt_size + chat_template_params.vision_start_tkn_size)
    messages= [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return messages


def prepare_messages_for_image(image_params: ImageParams, chat_template_params: ChatTemplateParams, vision_config: Qwen2_5_VLConfig):
    image_path_for_rope = get_temp_image()
    image_token_size =  image_params.get_num_vision_tokens()
    logger.info(f"image_token_size: {image_token_size}")
    context_len = chat_template_params.context_size

    if image_token_size > context_len:
        logger.info(f"image_token_size {image_token_size} too high. context_len is {context_len}")
        exit(1)

    if (chat_template_params.get_num_tokens_filled(image_params)) > context_len:
        logger.info(f"image_token_size + generation_prompt_size + image_token_size : {chat_template_params.get_num_tokens_filled(image_params)}")
        logger.info(f"number of tokens is too high. context_len is {context_len}")
        exit(1)


    logger.info(f"Size without dummy letters:{chat_template_params.get_num_tokens_filled(image_params)}")
    logger.info(f"           =>dummy letters:{context_len - chat_template_params.get_num_tokens_filled(image_params)}")
    prompt = ("😁" * (context_len - chat_template_params.get_num_tokens_filled(image_params))).strip()
    logger.info(f"prompt {prompt}")

    messages_for_image = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path_for_rope.name,
                    "resized_height": image_params.frame_height * vision_config.spatial_merge_size *  vision_config.patch_size,
                    "resized_width": image_params.frame_width * vision_config.spatial_merge_size *  vision_config.patch_size,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return messages_for_image, image_path_for_rope

def grab_rope_parameters(vision_params: Optional[VisionParams], chat_template_params: ChatTemplateParams,
                         rope_file_handler: RopeFileHander,
                         rope_option: RopeOption,
                         model: Qwen2_5_VLForConditionalGeneration,
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

    # We are not using flash_attention2 because this is just one time rope
    # generation.
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    # )
    if vision_params:
        effective_patch_size = vision_config.patch_size * vision_config.spatial_merge_size
        pixels_per_frame = effective_patch_size * effective_patch_size
        processor = Qwen2_5_VLProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                min_pixels=vision_params.frame_height * vision_params.frame_width * pixels_per_frame,
                max_pixels=vision_params.frame_height * vision_params.frame_width * pixels_per_frame,
                use_fast=True,
                padding_side="left",
                trust_remote_code=True,
            )
    else:
        processor = Qwen2_5_VLProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
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
        assert((inputs.input_ids[0] == vision_params.get_tkn_id(model.config)).nonzero()[0] == chat_template_params.sys_prompt_size)

    if modality == 'VIDEO':
        position_ids, mrope_position_deltas = model.model.get_rope_index(input_ids=inputs.input_ids,image_grid_thw=None, video_grid_thw=inputs.video_grid_thw,
            second_per_grid_ts=inputs.second_per_grid_ts, attention_mask=inputs.attention_mask)
    elif modality == 'IMAGE':
        position_ids, mrope_position_deltas = model.model.get_rope_index(input_ids=inputs.input_ids,image_grid_thw=inputs.image_grid_thw, video_grid_thw=None,
            second_per_grid_ts=None, attention_mask=inputs.attention_mask)
    elif modality == 'TEXT':
        position_ids, mrope_position_deltas = model.model.get_rope_index(input_ids=inputs.input_ids,image_grid_thw=None, video_grid_thw=None,
            second_per_grid_ts=None, attention_mask=inputs.attention_mask)
    else:
        logger.info("Error: unsupported modality")
        exit()

    print (f"position_ids: {position_ids}")
    print (f"mrope_position_deltas {mrope_position_deltas}")

    # Dummy tensor with type set as desired.
    # model.model.language_model.rotary_emb in transformers will use it.
    dummy_hidden_states_tensor = torch.randn(1, 1, dtype=torch.float32, device=cpu_or_cuda.lower())
    if rope_option == RopeOption.VISION_ONLY:
        position_ids = position_ids[:, :, chat_template_params.sys_prompt_size: chat_template_params.sys_prompt_size + vision_params.get_num_vision_tokens()]
        position_ids = torch.nn.functional.pad(position_ids, pad = (0, chat_template_params.context_size - vision_params.get_num_vision_tokens()))
    assert(position_ids.shape[-1] == chat_template_params.context_size)
    cos_embeddings, sin_embeddings = model.model.language_model.rotary_emb(dummy_hidden_states_tensor, position_ids)


    logger.info(f"cos_embeddings.shape: {cos_embeddings.shape}")
    logger.info(f"sin_embeddings.shape: {sin_embeddings.shape}")

    mrope_section = model.config.rope_scaling['mrope_section']
    mrope_section = mrope_section * 2
    unsqueeze_dim = 1  # Default from transformers.
    cos_embeddings = torch.cat([m[i % 3] for i, m in enumerate(cos_embeddings.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )
    sin_embeddings = torch.cat([m[i % 3] for i, m in enumerate(sin_embeddings.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
        unsqueeze_dim
    )

    logger.info(f"After applying mrope_section: cos_embeddings.shape: {cos_embeddings.shape}")
    logger.info(f"After applying mrope_section: sin_embeddings.shape: {sin_embeddings.shape}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Writing to {cos_embeddings_file}")
    torch.save(cos_embeddings.cpu(), cos_embeddings_file)
    logger.info(f"Writing to {sin_embeddings_file}")
    torch.save(sin_embeddings.cpu(), sin_embeddings_file)

def merge_cos_and_sin(rope_file_handler: RopeFileHander, chat_template_params: ChatTemplateParams):
    cos_files = []
    sin_files = []
    chat_template_params = ChatTemplateParams(CONTEXT_LEN, GENERATION_PROMPT_SIZE, DEFAULT_SYS_PROMPT_SIZE)
    with rope_file_handler.staging_mode():
        text_cos_file, text_sin_file = rope_file_handler.get_cos_sin_file_path(chat_template_params, "TEXT", RopeOption.FULL_ROPE, read = True)
    cos_files.extend([text_cos_file, text_cos_file])
    sin_files.extend([text_sin_file, text_sin_file])


    for sys_prompt_size in range(SUPPORTED_SYS_PROMPT_SIZE_MULTIPLES, CONTEXT_LEN - SUPPORTED_SYS_PROMPT_SIZE_MULTIPLES - TOKENS_PER_FRAME * 2, SUPPORTED_SYS_PROMPT_SIZE_MULTIPLES):
        chat_template_params = ChatTemplateParams(CONTEXT_LEN, GENERATION_PROMPT_SIZE, sys_prompt_size)
        with rope_file_handler.staging_mode():
            video_cos_file, video_sin_file = rope_file_handler.get_cos_sin_file_path(chat_template_params, "VIDEO", RopeOption.VISION_ONLY, read = True)
        cos_files.append(video_cos_file)
        sin_files.append(video_sin_file)

    final_cos = torch.cat(([torch.load(cos_file, weights_only = True) for cos_file in cos_files]), dim = -2)
    final_sin = torch.cat(([torch.load(sin_file, weights_only = True) for sin_file in sin_files]), dim = -2)

    merged_cos_file_name, merged_sin_file_name = rope_file_handler.get_cos_sin_file_path(chat_template_params, 'FINAL', RopeOption.FULL_ROPE)

    logger.info(f"Writing to {merged_cos_file_name}")
    torch.save(final_cos, merged_cos_file_name)
    logger.info(f"Writing to {merged_sin_file_name}")
    torch.save(final_sin, merged_sin_file_name)

def generate_qwen2_5_vl_vision_rope_embeddings(output_dir:str, model:Qwen2_5_VLForConditionalGeneration, config: Qwen2_5_VLConfig):
    """
    Generate and save vision rotary position embeddings (RoPE) for Qwen2.5-VL model.

    This function computes vision-specific rotary position embeddings by processing
    a sample image through the model's visual module, then saves the cos and sin
    components as PyTorch tensors to the specified output directory.

    Args:
        output_dir: Directory path where vision_rotary_embeddings_cos.pt and
                    vision_rotary_embeddings_sin.pt will be saved.
        model: Pre-loaded Qwen2.5-VL model instance.
        config: Model configuration containing vision settings.

    Output Files:
        - vision_rotary_embeddings_cos.pt: Cosine component of vision RoPE embeddings
        - vision_rotary_embeddings_sin.pt: Sine component of vision RoPE embeddings
    """
    logger.info("generating qwen2_5_vl vision rope embeddings..")
    # rope embedding is only used when doing cnn inference in group of 2 frames. So image input is enough for
    # dumping purpose
    vision_params = ImageParams(FRAME_HEIGHT, FRAME_WIDTH)
    vision_config = config.vision_config
    chat_template_params = ChatTemplateParams(CONTEXT_LEN, GENERATION_PROMPT_SIZE, DEFAULT_SYS_PROMPT_SIZE)
    effective_patch_size = vision_config.patch_size * vision_config.spatial_merge_size
    pixels_per_frame = effective_patch_size * effective_patch_size
    processor = Qwen2_5_VLProcessor.from_pretrained(
                "Qwen/Qwen2.5-VL-3B-Instruct",
                min_pixels=vision_params.frame_height * vision_params.frame_width * pixels_per_frame,
                max_pixels=vision_params.frame_height * vision_params.frame_width * pixels_per_frame,
                use_fast=True,
                padding_side="left",
                trust_remote_code=True,
            )
    messages = prepare_messages_for_image(
                         image_params=vision_params,
                         chat_template_params=chat_template_params,
                         vision_config=config.vision_config)
    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

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
    window_index, cu_window_seqlens = model.visual.get_window_index(grid_thw)
    seq_len = vision_params.frame_height * vision_params.frame_width * vision_config.spatial_merge_size * vision_config.spatial_merge_size
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // (vision_config.spatial_merge_size * vision_config.spatial_merge_size), vision_config.spatial_merge_size * vision_config.spatial_merge_size, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    cos = emb.cos()
    sin = emb.sin()

    vision_cos_file_name = Path(output_dir) / "vision_rotary_embeddings_cos.pt"
    vision_sin_file_name = Path(output_dir) / "vision_rotary_embeddings_sin.pt"
    logger.info(f"Saving vision cos in {vision_cos_file_name}")
    logger.info(f"Saving vision sin in {vision_sin_file_name}")
    torch.save(cos, vision_cos_file_name)
    torch.save(sin, vision_sin_file_name)
    
def generate_qwen2_5vl_rope_embeddings(output_dir:str, vl_rope_type:str, model:Qwen2_5_VLForConditionalGeneration):
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

# Main function for manually generating and experimenting with RoPE.
if __name__ == "__main__":
    this_file_basename = Path(__file__).name
    today = date.today().strftime("%Y-%m-%d")
    output_dir = f"__{this_file_basename}.manual_run.{today}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)    
    vl_rope_type = "multi_rope"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    generate_qwen2_5_vl_vision_rope_embeddings(output_dir, model, model.model.config)
    generate_qwen2_5vl_rope_embeddings(output_dir, vl_rope_type, model)
