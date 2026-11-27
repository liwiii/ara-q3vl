import sys
import os

from llm_compiler.proto_utils.graph import DAG, NamedDim
from llm_compiler.qwen2_5_vl.constants import HIDDEN, TKN, UNSQUEEZE, BATCH, HEIGHT_WINDOWS, WIDTH_WINDOWS, PATCH_DIM_0
from proto_utils import Network
from proto_utils.layers import Conv, Input, LayerNorm, Gelu
from proto_utils.functional import permute, reshape


class MergerNetwork(Network):
    def __init__(
        self,
        network_name,
        graph,
        is_dynamic,
        frame_height,
        frame_width,
        context_dim,
        hidden_dim,
    ):
        super().__init__(network_name, graph)
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.hidden_size = context_dim * 4
        self.hidden_dim = hidden_dim
        self.layernorm = LayerNorm(
            'layernorm',
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
            new_accumulation_dim_name=HIDDEN,
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
            new_accumulation_dim_name=HIDDEN,
            is_dynamic=is_dynamic,
        )
        self.gelu = Gelu("gelu", self.graph, out_group_axis=1, out_group_size=64, is_dynamic=is_dynamic)

    def forward(self, hidden_states, **kwargs):
        # (576, 1280)
        hidden_states = self.layernorm(hidden_states)
        num_tkns = hidden_states[0].shape[-2].dim
        hidden_states = reshape(
            self.graph,
            hidden_states,
            [
                [
                    (TKN, num_tkns // 4, TKN),  # merge 4 tokens
                    (HIDDEN, self.hidden_size, HIDDEN),
                ],
                [
                    (TKN, num_tkns // 4, TKN),  # merge 4 tokens
                    (HIDDEN, self.hidden_size // 64, HIDDEN),
                ],
            ],
            out_group_axis_name=HIDDEN
        )
        hidden_states = permute(self.graph, hidden_states, (HIDDEN, TKN))
        hidden_states = reshape(
            self.graph,
            hidden_states,
            [
                [(UNSQUEEZE, self.hidden_size, HIDDEN), (HIDDEN, 1, UNSQUEEZE)],
                [
                    (UNSQUEEZE, self.hidden_size // 64, HIDDEN),
                    (HIDDEN, 1, UNSQUEEZE),
                ],
            ],
            out_group_axis_name=HIDDEN
        )
        hidden_states = self.mlp_fc1(hidden_states, accumulation_dim_name = HIDDEN)
        # no permute needed since former conv is gonna be merged with below gelu
        hidden_states = self.gelu(hidden_states)
        hidden_states = self.mlp_fc2(hidden_states, accumulation_dim_name = HIDDEN)
        hidden_states = reshape(
            self.graph,
            hidden_states,
            [
                [(UNSQUEEZE, self.hidden_dim, HIDDEN), (HIDDEN, 1, UNSQUEEZE)],
                [
                    (UNSQUEEZE, self.hidden_dim // 64, HIDDEN),
                    (HIDDEN, 1, UNSQUEEZE),
                ],
            ],
            out_group_axis_name=HIDDEN
        )
        hidden_states = permute(self.graph, hidden_states, (HIDDEN, TKN))
        hidden_states = reshape(
            self.graph,
            hidden_states,
            [
                [
                    (BATCH, self.frame_height // 4, HEIGHT_WINDOWS),
                    (UNSQUEEZE, self.frame_width // 4, WIDTH_WINDOWS),
                    (TKN, 4, PATCH_DIM_0),
                    (HIDDEN, 4 * self.hidden_dim, HIDDEN),
                ],
                [
                    (BATCH, self.frame_height // 4, HEIGHT_WINDOWS),
                    (UNSQUEEZE, self.frame_width // 4, WIDTH_WINDOWS),
                    (TKN, 4, PATCH_DIM_0),
                    (HIDDEN, 4 * self.hidden_dim // 64, HIDDEN),
                ],
            ],
            out_group_axis_name=HIDDEN
        )
        hidden_states = permute(self.graph, hidden_states, (WIDTH_WINDOWS, PATCH_DIM_0))

        # This reshape can be done in dvm. As of now to match with existing
        # hardcoded prototxt flow, we are commenting it. So this reshape is done
        # on host.
        # hidden_states = reshape(
        #     self.graph,
        #     hidden_states,
        #     [
        #         [
        #             (HEIGHT_WINDOWS, 1, BATCH),
        #             (PATCH_DIM_0, 1, UNSQUEEZE),
        #             (WIDTH_WINDOWS, num_tkns // 4, TKN),
        #             (HIDDEN, self.hidden_dim, HIDDEN),
        #         ],
        #         [
        #             (HEIGHT_WINDOWS, 1, BATCH),
        #             (PATCH_DIM_0, 1, UNSQUEEZE),
        #             (WIDTH_WINDOWS, num_tkns // 4, TKN),
        #             (HIDDEN, self.hidden_dim // 64, HIDDEN),
        #         ]
        #     ],
        #     out_group_axis_name=HIDDEN
        # )

        return hidden_states