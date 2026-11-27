import onnx
import numpy as np

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

from proto_utils import Network
from proto_utils.layers import Conv, Input
from proto_utils.functional import ReshapeMaker, reshape_with_reshapemaker, permute

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
        is_rotated,
    ):
        super().__init__(network_name, graph)
        self.is_rotated = is_rotated
        self.pixels_width = pixels_width
        self.pixels_height = pixels_height
        self.hidden_dim = hidden_dim
        self.spatial_merge_size = spatial_merge_size
        self.patch_embed = Conv(
            "conv3d",
            graph,
            num_output=hidden_dim,
            bias_term=False,
            kernel_size=[2, spatial_patch_size, spatial_patch_size],
            stride=[2, spatial_patch_size, spatial_patch_size],
            out_group_axis=-4,
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
        hidden_states = self.patch_embed(
            pixel_inputs,
            derive_name_from_layer_name=True,
            accumulation_dim_name=IN_CHANNELS,
        )

        reshape_maker = ReshapeMaker(
            hidden_states, group_size=DEFAULT_GROUP_SIZE
        )

        if self.is_rotated:
            reshape_maker = ReshapeMaker(
                hidden_states, group_size=DEFAULT_GROUP_SIZE
            )
            reshape_maker.dim(N).resize(1).rename(N).drop()
            reshape_maker.dim(HIDDEN).resize(1).rename(BATCH)
            reshape_maker.dim(CHANNEL_DIM).resize(self.hidden_dim).rename(HIDDEN).group()
            reshape_maker.dim(PIXELS_HEIGHT).resize(1)
            reshape_maker.dim(PIXELS_WIDTH).resize(
                FIXED_HEIGHT
                * FIXED_WIDTH
                * self.spatial_merge_size
                * self.spatial_merge_size
            )
            hidden_states = reshape_with_reshapemaker(
                self.graph, hidden_states, reshape_maker
            )
            hidden_states = self.rotation(
                hidden_states,
                derive_name_from_layer_name=False,
                accumulation_dim_name=HIDDEN,
            )
        reshape_maker.reset_to(hidden_states)
        if not self.is_rotated:
            reshape_maker.dim(N).resize(1).rename(N).drop()
        reshape_maker.dim(HIDDEN if not self.is_rotated else BATCH).resize(
            FIXED_HEIGHT * self.hidden_dim
        ).rename(HIDDEN).group()
        reshape_maker.dim(CHANNEL_DIM if not self.is_rotated else HIDDEN).resize(
            self.spatial_merge_size
        ).rename(HEIGHT_TEMPORAL_DIM)
        reshape_maker.dim(PIXELS_HEIGHT).resize(FIXED_WIDTH).rename(PIXELS_WIDTH)
        reshape_maker.dim(PIXELS_WIDTH).resize(self.spatial_merge_size).rename(
            WIDTH_TEMPORAL_DIM
        )
        hidden_states = reshape_with_reshapemaker(
            self.graph, hidden_states, reshape_maker
        )
        hidden_states = permute(
            self.graph, hidden_states, (HEIGHT_TEMPORAL_DIM, PIXELS_WIDTH)
        )
        reshape_maker.reset_to(hidden_states)
        reshape_maker.dim(HIDDEN).resize(self.hidden_dim * FIXED_HEIGHT // 4).group()
        reshape_maker.dim(PIXELS_WIDTH).resize(4).rename(HEIGHT_WINDOWS)
        reshape_maker.dim(HEIGHT_TEMPORAL_DIM).resize(
            FIXED_WIDTH // 4
        ).rename(NUM_WINDOWS_WIDTH)
        reshape_maker.dim(WIDTH_TEMPORAL_DIM).resize(
            4 * self.spatial_merge_size * self.spatial_merge_size
        ).rename(WIDTH_WINDOWS_AND_ALL_TEMPORAL)

        hidden_states = reshape_with_reshapemaker(
            self.graph, hidden_states, reshape_maker
        )
        hidden_states = permute(
            self.graph, hidden_states, (NUM_WINDOWS_WIDTH, HEIGHT_WINDOWS)
        )
        reshape_maker.reset_to(hidden_states)
        reshape_maker.dim(HIDDEN).resize(1).rename(BATCH)
        reshape_maker.dim(NUM_WINDOWS_WIDTH).resize(1).rename(UNSQUEEZE)
        reshape_maker.dim(HEIGHT_WINDOWS).resize(self.hidden_dim).rename(HIDDEN).group()
        reshape_maker.dim(WIDTH_WINDOWS_AND_ALL_TEMPORAL).resize(
            FIXED_HEIGHT * FIXED_WIDTH * self.spatial_merge_size * self.spatial_merge_size
        ).rename(TKN)
        hidden_states = reshape_with_reshapemaker(
            self.graph, hidden_states, reshape_maker
        )
        hidden_states = permute(self.graph, hidden_states, (HIDDEN, TKN))

        return hidden_states
