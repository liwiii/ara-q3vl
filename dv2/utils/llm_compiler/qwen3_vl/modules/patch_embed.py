import onnx
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent))
sys.path.append(str(Path(__file__).absolute().parent.parent / 'qwen2_5_vl'))
from llm_compiler.model import Model
from llm_compiler.qwen2_5_vl.constants import (
    BATCH,
    CHANNEL_DIM,
    FIXED_HEIGHT,
    FIXED_WIDTH,
    TKN,
    HEIGHT_TEMPORAL_DIM,
    HEIGHT_WINDOWS,
    IN_CHANNELS,
    NUM_WINDOWS_WIDTH,
    PIXELS_HEIGHT,
    PIXELS_WIDTH,
    HIDDEN,
    # INPUT,
    PATCH_EMBED,
    N,
    DEFAULT_GROUP_SIZE,
    TEMPORAL_DIM,
    WIDTH_TEMPORAL_DIM,
    HEIGHT_WINDOWS,
    WIDTH_WINDOWS_AND_ALL_TEMPORAL,
    UNSQUEEZE,
)

from llm_compiler.qwen3_vl.constants import *

from proto_utils import Network
from proto_utils.layers import Conv, Input
from proto_utils.functional import *

class PatchEmbedNetwork(Network):
    def __init__(
        self,
        network_name: str,
        graph,
        hidden_dim,
        pixels_width,
        pixels_height,
        spatial_patch_size,
        spatial_merge_size,
        artificial_large_channel_num,
        is_rotated,
    ):
        super().__init__(network_name, graph)
        self.is_rotated = is_rotated
        self.pixels_width = pixels_width
        self.pixels_height = pixels_height
        self.hidden_dim = hidden_dim
        self.spatial_merge_size = spatial_merge_size
        self.artificial_large_channel_num = artificial_large_channel_num
        self.patch_embed = Conv(
            "artificial_conv1d",
            graph,
            num_output=hidden_dim,
            # bias True/False may come from onnx weights.
            bias_term=True,

            # qwen2.5vl
            # kernel_size=[2, 14, 14],
            # stride=[2, 14, 14],

            #qwen3vl 2, 16, 16 based on patch size
            # kernel_size=[2, spatial_patch_size, spatial_patch_size],
            # stride=[2, spatial_patch_size, spatial_patch_size],

            #qwen3vl implementing as 1x1 conv due to absence of 16x16x2 kernel
            kernel_size=[1, 1],
            stride=[1, 1],

            out_group_axis=-3,
            new_accumulation_dim_name=HIDDEN,
            is_dynamic=True,
            out_group_size=64,
        )

        if self.is_rotated:
            self.rotation = Conv(
                "rotation",
                graph,
                num_output=self.hidden_dim,
                bias_term=False,
                kernel_size=[1, 1],
                stride=[1, 1],
                out_group_axis=-3,
                new_accumulation_dim_name=HIDDEN,
                is_dynamic=True,
                out_group_size=64,
            )
    
    def forward(self, pixel_inputs):
        hidden_states = permute(
            self.graph, pixel_inputs, (TKN, ARTIFICIAL_LARGE_CHANNEL)
        )

        hidden_states = reshape(
            self.graph,
            hidden_states,
            [
                [(C, self.artificial_large_channel_num, ARTIFICIAL_LARGE_CHANNEL), (ARTIFICIAL_LARGE_CHANNEL, 1, UNSQUEEZE)],
                [
                    (
                        C,
                        self.artificial_large_channel_num//64,
                        ARTIFICIAL_LARGE_CHANNEL,
                    ),
                    (ARTIFICIAL_LARGE_CHANNEL, 1, UNSQUEEZE),
                ],
            ],
            ARTIFICIAL_LARGE_CHANNEL,
        )

        hidden_states = self.patch_embed(
            hidden_states,
            derive_name_from_layer_name=True,
            accumulation_dim_name=ARTIFICIAL_LARGE_CHANNEL,
        )

        hidden_states = reshape(
            self.graph,
            hidden_states,
            [
                [(UNSQUEEZE, self.hidden_dim, HIDDEN), (HIDDEN, 1, UNSQUEEZE)],
                [
                    (
                        UNSQUEEZE,
                        self.hidden_dim//64,
                        HIDDEN,
                    ),
                    (HIDDEN, 1, UNSQUEEZE),
                ],
            ],
            HIDDEN,
        )
        hidden_states = permute(self.graph, hidden_states, (TKN, HIDDEN))
        return hidden_states
