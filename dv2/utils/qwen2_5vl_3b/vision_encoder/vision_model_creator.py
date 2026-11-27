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

import sys
sys.path.append("../../../")
from merger import Qwen2_5_VL_Merger
from proto_utils import Network
from proto_utils.layers import Input
from patch_embed_gather import Qwen2_5_VL_PatchEmbedGather
from full_attn_vision import Full_Vision_Attn
from qwen2_vl_attention import Qwen2_5VL_Attn
from vision_attn_block import Qwen2_5_VL_AttentionBlock
from constants import (
    HEIGHT,
    IN_CHANNELS,
    N,
    TEMPORAL_DIM,
    UNSQUEEZE_0,
    UNSQUEEZE_1,
    TKN,
    WIDTH,
    WINDOWS_SIZE,
    BATCH,
    UNSQUEEZE,
    MAX_TOKENS,
    SIN_COS,
)
from proto_utils.functional import permute

class Qwen2_5_VL(Network):
    def __init__(
        self,
        network_name,
        graph,
        num_tkns,
        context_dim,
        frame_height,
        frame_width,
        is_dynamic,
        rope_table,
        is_rotated = False,
    ):
        super().__init__(network_name, graph)
        self.patch_embed = Qwen2_5_VL_PatchEmbedGather(
            "patch_embed",
            graph,
            num_tkns=num_tkns,
            hidden_dim=context_dim,
            frame_height=frame_height,
            frame_width=frame_width,
            is_dynamic=is_dynamic,
            is_rotated=is_rotated
        )
        self.is_rotated = is_rotated
        self.fullatt_block_indexes = [7, 15, 23, 31]
        self.num_blocks = 32
        self.attn_blocks = [
            (
                Qwen2_5_VL_AttentionBlock(
                    "attn_block",
                    self.graph,
                    attn=Full_Vision_Attn(
                        "full_attn", self.graph, rope_table=rope_table
                    ),layer_idx = layer_idx
                )
                if layer_idx in self.fullatt_block_indexes
                else Qwen2_5_VL_AttentionBlock(
                    "attn_block",
                    self.graph,
                    attn=Qwen2_5VL_Attn(
                        "windowed_attn",
                        self.graph,
                        rope_table=rope_table
                    ),layer_idx = layer_idx
                )
            )
            for layer_idx in range(self.num_blocks)
        ]
        self.merger = Qwen2_5_VL_Merger('merger', self.graph, is_dynamic = is_dynamic, frame_height=frame_height, frame_width=frame_width, context_dim=context_dim, hidden_dim=2048)

    def forward(self, pixels, **kwargs):
        hidden_state = self.patch_embed(pixels)
        for attn_block in self.attn_blocks:
            hidden_state = attn_block(hidden_state)
        return self.merger(hidden_state)
    
    def get_dict(self):
        final_torch_to_caffe = {}
        final_torch_to_caffe.update(self.patch_embed.get_dict())
        for block in self.attn_blocks:
            final_torch_to_caffe.update(block.get_dict())
        final_torch_to_caffe.update(self.merger.get_dict())
        return final_torch_to_caffe

# Standalone method.
def make_vision_model(is_rotated = False):
    from proto_utils.layers.input import Input
    from proto_utils.graph import DAG, NamedDim

    graph = DAG()
    frame_height = 12
    frame_width = 12
    num_tkns = frame_height * frame_width * 4
    context_dim = 1280
    hidden_state_named_shape = [
        NamedDim(N, 1),
        NamedDim(IN_CHANNELS, 3),
        NamedDim(TEMPORAL_DIM, 2),
        NamedDim(HEIGHT, frame_height * 28),
        NamedDim(WIDTH, frame_width * 28),
    ]
    is_dynamic = True
    inp = Input(
        "input",
        graph,
        is_dynamic=is_dynamic,
        inp_shape=hidden_state_named_shape,
        out_group_size=1,
        out_group_axis=-1,
    )()
    rope_table_named_shape = [
        NamedDim(BATCH, 1),
        NamedDim(UNSQUEEZE, 1),
        NamedDim(MAX_TOKENS, num_tkns),
        NamedDim(SIN_COS, context_dim * 2 // 16),
    ]
    rope_table = permute(
        graph,
        Input(
            "rope_table",
            graph,
            rope_table_named_shape,
            is_dynamic=False,
            out_group_size=None,
            out_group_axis=None,
        )(),
        (MAX_TOKENS, SIN_COS),
    )
    vision_model = Qwen2_5_VL(
        "qwen2_5_vl",
        graph,
        num_tkns=num_tkns,
        context_dim=context_dim,
        frame_height=frame_height,
        frame_width=frame_width,
        is_dynamic=True,
        rope_table=rope_table,
        is_rotated = is_rotated
    )

    return vision_model
