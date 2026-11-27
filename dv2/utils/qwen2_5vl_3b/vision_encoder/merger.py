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
import os

from llm_compiler.proto_utils.graph import DAG, NamedDim
from llm_compiler.proto_utils.layers.gelu import Gelu


from constants import (
    BATCH,
    HEIGHT_WINDOWS,
    HIDDEN_DIM,
    PATCH_DIM_0,
    TKN,
    UNSQUEEZE,
    WIDTH_WINDOWS,
)
from proto_utils import Network
from proto_utils.layers import Conv, Input, LayerNorm
from proto_utils.functional import permute, reshape


class Qwen2_5_VL_Merger(Network):
    def __init__(
        self,
        network_name,
        graph,
        is_dynamic,
        frame_height,
        frame_width,
        context_dim=1280,
        hidden_dim=3584,
    ):
        super().__init__(network_name, graph)
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.hidden_size = context_dim * 4
        self.hidden_dim = hidden_dim
        self.ln_q = LayerNorm(
            "ln_q",
            self.graph,
            out_group_axis=-1,
            out_group_size=64,
            uses_mean=False,
            is_dynamic=is_dynamic,
        )
        self.mlp_fc1 = Conv(
            "mlp_fc1",
            self.graph,
            num_output=self.hidden_size,
            bias_term=True,
            kernel_size=[1, 1],
            stride=[1, 1],
            out_group_axis=1,
            out_group_size=64,
            new_accumulation_dim_name=HIDDEN_DIM,
            is_dynamic=is_dynamic,
        )
        self.mlp_fc2 = Conv(
            "mlp_fc2",
            self.graph,
            num_output=hidden_dim,
            bias_term=True,
            kernel_size=[1, 1],
            stride=[1, 1],
            out_group_axis=1,
            out_group_size=64,
            new_accumulation_dim_name=HIDDEN_DIM,
            is_dynamic=is_dynamic,
        )
        self.gelu = Gelu("gelu", self.graph, out_group_axis=1, out_group_size=64, is_dynamic=is_dynamic)

    def forward(self, hidden_states, **kwargs):
        # (576, 1280)
        hidden_states = self.ln_q(hidden_states)
        num_tkns = hidden_states[0].shape[-2].dim
        hidden_states = reshape(
            self.graph,
            hidden_states,
            [
                [
                    (TKN, num_tkns // 4, TKN),  # merge 4 tokens
                    (HIDDEN_DIM, self.hidden_size, HIDDEN_DIM),
                ],
                [
                    (TKN, num_tkns // 4, TKN),  # merge 4 tokens
                    (HIDDEN_DIM, self.hidden_size // 64, HIDDEN_DIM),
                ],
            ],
            out_group_axis_name=HIDDEN_DIM
        )
        hidden_states = permute(self.graph, hidden_states, (HIDDEN_DIM, TKN))
        hidden_states = reshape(
            self.graph,
            hidden_states,
            [
                [(UNSQUEEZE, self.hidden_size, HIDDEN_DIM), (HIDDEN_DIM, 1, UNSQUEEZE)],
                [
                    (UNSQUEEZE, self.hidden_size // 64, HIDDEN_DIM),
                    (HIDDEN_DIM, 1, UNSQUEEZE),
                ],
            ],
            out_group_axis_name=HIDDEN_DIM
        )
        hidden_states = self.mlp_fc1(hidden_states, accumulation_dim_name = HIDDEN_DIM)
        # no permute needed since former conv is gonna be merged with below gelu
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.mlp_fc2(hidden_states, accumulation_dim_name = HIDDEN_DIM)
        hidden_states = reshape(
            self.graph,
            hidden_states,
            [
                [(UNSQUEEZE, self.hidden_dim, HIDDEN_DIM), (HIDDEN_DIM, 1, UNSQUEEZE)],
                [
                    (UNSQUEEZE, self.hidden_dim // 64, HIDDEN_DIM),
                    (HIDDEN_DIM, 1, UNSQUEEZE),
                ],
            ],
            out_group_axis_name=HIDDEN_DIM
        )
        hidden_states = permute(self.graph, hidden_states, (HIDDEN_DIM, TKN))
        hidden_states = reshape(
            self.graph,
            hidden_states,
            [
                [
                    (BATCH, self.frame_height // 4, HEIGHT_WINDOWS),
                    (UNSQUEEZE, self.frame_width // 4, WIDTH_WINDOWS),
                    (TKN, 4, PATCH_DIM_0),
                    (HIDDEN_DIM, 4 * self.hidden_dim, HIDDEN_DIM),
                ],
                [
                    (BATCH, self.frame_height // 4, HEIGHT_WINDOWS),
                    (UNSQUEEZE, self.frame_width // 4, WIDTH_WINDOWS),
                    (TKN, 4, PATCH_DIM_0),
                    (HIDDEN_DIM, 4 * self.hidden_dim // 64, HIDDEN_DIM),
                ],
            ],
            out_group_axis_name=HIDDEN_DIM
        )
        hidden_states = permute(self.graph, hidden_states, (WIDTH_WINDOWS, PATCH_DIM_0))
        hidden_states = reshape(
            self.graph,
            hidden_states,
            [
                [
                    (HEIGHT_WINDOWS, 1, BATCH),
                    (PATCH_DIM_0, 1, UNSQUEEZE),
                    (WIDTH_WINDOWS, num_tkns // 4, TKN),
                    (HIDDEN_DIM, self.hidden_dim, HIDDEN_DIM),
                ],
                [
                    (HEIGHT_WINDOWS, 1, BATCH),
                    (PATCH_DIM_0, 1, UNSQUEEZE),
                    (WIDTH_WINDOWS, num_tkns // 4, TKN),
                    (HIDDEN_DIM, self.hidden_dim // 64, HIDDEN_DIM),
                ]
            ],
            out_group_axis_name=HIDDEN_DIM
        )

        return hidden_states
    def get_dict(self):
        return {
            "visual.merger.ln_q": self.ln_q.layer_name,
            "visual.merger.mlp.0": self.mlp_fc1.layer_name,
            "visual.merger.mlp.2": self.mlp_fc2.layer_name,
            }
    
    def get_caffemodel(self, source_proto, target_caffemodel):
        final_torch_to_caffe = self.get_dict()
        get_caffemodel(final_torch_to_caffe, source_proto, target_caffemodel)


# TODO: group axis should be named
if __name__ == "__main__":
    from proto_utils.layers.input import Input

    graph = DAG()
    context_dim = 1280
    hidden_dim = 3584
    frame_height = 16
    frame_width = 16
    tkns = frame_height * frame_width * 4
    hidden_state_named_shape = [
        NamedDim(BATCH, 1),
        NamedDim(UNSQUEEZE, 1),
        NamedDim(TKN, tkns),
        NamedDim(HIDDEN_DIM, context_dim),
    ]
    is_dynamic = True
    inp = Input(
        "input",
        graph,
        is_dynamic=is_dynamic,
        inp_shape=hidden_state_named_shape,
        out_group_axis=-1,
        out_group_size=64,
    )()
    merger = Qwen2_5_VL_Merger('merger', graph, is_dynamic = True, frame_height = frame_height, frame_width=frame_width, context_dim = context_dim, hidden_dim = hidden_dim)
    merger.update_layer_names()
    out = merger(inp)
    proto_out = "prototxt/merger.prototxt"
    graph.get_proto(save_to=proto_out)
    graph.get_so_yaml(save_to="so/merger_so.yaml")
    target_caffemodel = "caffemodel/merger.caffemodel"
    merger.get_caffemodel(proto_out, target_caffemodel)
