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

from pathlib import Path
import sys

sys.path.append("../../../")

from proto_utils import Network
from proto_utils.layers import Conv, Input
from proto_utils.functional import permute, reshape
from constants import (
    HEIGHT,
    HIDDEN_DIM,
    N,
    TKN,
    UNSQUEEZE_0,
    UNSQUEEZE_1,
    IN_CHANNELS,
    TEMPORAL_DIM,
    PATCH_DIM_0,
    PATCH_DIM_1,
    WIDTH,
    WINDOWS_SIZE,
    HEIGHT_WINDOWS,
    WIDTH_WINDOWS,
    WINDOWS_SIZE_0,
    WINDOWS_SIZE_1,
    BATCH,
    UNSQUEEZE,
)

class Qwen2_5_VL_PatchEmbedGather(Network):
    def __init__(
        self,
        network_name,
        graph,
        num_tkns,
        hidden_dim,
        frame_height,
        frame_width,
        is_dynamic=True,
        is_rotated = False,
    ):
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.hidden_dim = hidden_dim
        super().__init__(network_name, graph)
        self.patch_embed = Conv(
            "conv3d",
            graph,
            num_output=self.hidden_dim,
            bias_term=False,
            kernel_size=[2, 14, 14],
            stride=[2, 14, 14],
            out_group_axis=-4,
            new_accumulation_dim_name=HIDDEN_DIM,
            is_dynamic=is_dynamic,
            out_group_size=64,
        )
        self.is_rotated = is_rotated
        if self.is_rotated:
            self.rotation = Conv(
                "rotation",
                graph,
                num_output=self.hidden_dim,
                bias_term=False,
                kernel_size=[1,1],
                stride=[1,1],
                out_group_axis=-3,
                new_accumulation_dim_name=HIDDEN_DIM,
                is_dynamic=is_dynamic,
                out_group_size=64,
            )

    def forward(self, pixels, **kwargs):
        # pixels = (3, 2, 336, 336)
        hidden_states = self.patch_embed(
            pixels, derive_name_from_layer_name=True, accumulation_dim_name=IN_CHANNELS
        )
        if self.is_rotated:
            hidden_states = reshape(self.graph, hidden_states, [
                [
                    (N, 1, N),
                    (HIDDEN_DIM, 1, BATCH),
                    (TEMPORAL_DIM, self.hidden_dim, HIDDEN_DIM),
                    (HEIGHT, 1, HEIGHT), # FIXME: these dimensions don't like their temporary names
                    (WIDTH, 4 * self.frame_height * self.frame_width, WIDTH)
                ],[
                    (N, 1, N),
                    (HIDDEN_DIM, 1, BATCH),
                    (TEMPORAL_DIM, self.hidden_dim // self.patch_embed.out_group_size, HIDDEN_DIM),
                    (HEIGHT, 1, HEIGHT),
                    (WIDTH, 4 * self.frame_height * self.frame_width, WIDTH)
                ]
            ], out_group_axis_name=HIDDEN_DIM, drop_dim=N)
            hidden_states = self.rotation(hidden_states, derive_name_from_layer_name=True, accumulation_dim_name = HIDDEN_DIM)
        hidden_states = reshape(
            self.graph, 
            hidden_states,
            [
                [
                    (N, 1, N), #FIXME: this works by magic even is self.rotation is true
                    (HIDDEN_DIM if not self.is_rotated else BATCH, self.frame_height * self.hidden_dim, HIDDEN_DIM), #FIXME: being a bad programmer for now, should become better one day
                    (TEMPORAL_DIM if not self.is_rotated else HIDDEN_DIM, 2, PATCH_DIM_0), 
                    (HEIGHT, self.frame_width, WIDTH),
                    (WIDTH, 2, PATCH_DIM_1)
                ],
                [   (N, 1, N),
                    (HIDDEN_DIM if not self.is_rotated else BATCH, self.frame_height * self.hidden_dim // self.patch_embed.out_group_size, HIDDEN_DIM),
                    (TEMPORAL_DIM if not self.is_rotated else HIDDEN_DIM, 2, PATCH_DIM_0),
                    (HEIGHT, self.frame_width, WIDTH),
                    (WIDTH, 2, PATCH_DIM_1)
                ]
            ],
            out_group_axis_name=HIDDEN_DIM,
            drop_dim=N
        )

        hidden_states = permute(self.graph, hidden_states, (PATCH_DIM_0, WIDTH))
        hidden_states = reshape(self.graph, hidden_states, [
            [
                (HIDDEN_DIM, self.hidden_dim * self.frame_height // 4, HIDDEN_DIM),
                (WIDTH, 4, WIDTH_WINDOWS),
                (PATCH_DIM_0, self.frame_width // 4, WINDOWS_SIZE_0),
                (PATCH_DIM_1, 16, WINDOWS_SIZE_1)
            ],
            [
                (HIDDEN_DIM, self.hidden_dim * self.frame_height // 4 // self.patch_embed.out_group_size, HIDDEN_DIM),
                (WIDTH, 4, WIDTH_WINDOWS),
                (PATCH_DIM_0, self.frame_width // 4, WINDOWS_SIZE_0),
                (PATCH_DIM_1, 16, WINDOWS_SIZE_1)
            ]
        ], out_group_axis_name=HIDDEN_DIM)
        hidden_states = permute(self.graph, hidden_states, (WINDOWS_SIZE_0, WIDTH_WINDOWS))
        hidden_states = reshape(self.graph, hidden_states, [
            [
                (HIDDEN_DIM, 1, BATCH),
                (WINDOWS_SIZE_0, 1, UNSQUEEZE),
                (WIDTH_WINDOWS, self.hidden_dim, HIDDEN_DIM),
                (WINDOWS_SIZE_1, self.frame_height * self.frame_width * 4, TKN)
            ],
            [
                (HIDDEN_DIM, 1, BATCH),
                (WINDOWS_SIZE_0, 1, UNSQUEEZE),
                (WIDTH_WINDOWS, self.hidden_dim // self.patch_embed.out_group_size, HIDDEN_DIM),
                (WINDOWS_SIZE_1, self.frame_height * self.frame_width * 4, TKN)
            ]
        ], out_group_axis_name=HIDDEN_DIM)
        hidden_states = permute(self.graph, hidden_states, (HIDDEN_DIM, TKN))

        return hidden_states

    def get_top_shapes(self, inp_shapes):
        return inp_shapes

    def get_nickname(self):
        return "attn"

    def get_dict(self):
        out_dict = {"visual.patch_embed.proj": self.patch_embed.layer_name}
        if self.is_rotated:
            out_dict.update({"visual.patch_embed.rotation": self.rotation.layer_name})
        return out_dict
    
    def get_caffemodel(self, source_proto, target_caffemodel, is_rotated):
        final_torch_to_caffe = self.get_dict()
        get_caffemodel(final_torch_to_caffe, source_proto, target_caffemodel, is_rotated)

