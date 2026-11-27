import sys
import os
from pathlib import Path

from proto_utils import Network
from proto_utils.layers import Conv, Input, Rope, BatchDot, SoftMax
from proto_utils.functional import ReshapeMaker, reshape_with_reshapemaker, permute, reshape

# At some point this constants.py should be moved here in qwen2_5_vl directory
sys.path.append(str(Path(__file__).absolute().parent.parent.parent / 'qwen2_5vl_3b'))
sys.path.append(str(Path(__file__).absolute().parent.parent.parent / 'qwen2_5vl_3b/vision_encoder'))
from vision_encoder.layer_names import ATTN_BATCHDOT, O_PROJ, Q_PROJ, K_PROJ, QKT_BATCHDOT, QKT_SOFTMAX, ROPE, V_PROJ
from vision_encoder.constants import MAX_TOKENS, SIN_COS

from llm_compiler.model import Model
from llm_compiler.qwen2_5_vl.constants import (
    BATCH,
    CHANNEL_DIM,
    FIXED_HEIGHT,
    FIXED_WIDTH,
    HEAD_DIM,
    NUM_HEAD,
    NUM_WINDOWS,
    TKN,
    HEIGHT_TEMPORAL_DIM,
    HEIGHT_WINDOWS,
    IN_CHANNELS,
    NUM_WINDOWS_WIDTH,
    PIXELS_HEIGHT,
    PIXELS_WIDTH,
    HIDDEN,
    # INPUT,
    # windowed_attn,
    N,
    # NUM_HEADS,
    DEFAULT_GROUP_SIZE,
    TEMPORAL_DIM,
    VISION_ROPE,
    WIDTH_TEMPORAL_DIM,
    HEIGHT_WINDOWS,
    WIDTH_WINDOWS_AND_ALL_TEMPORAL,
    UNSQUEEZE,
    WINDOWED_TKNS,
    WINDOWED_TKNS_ACC,
    TKN_ACC,
)


class FullAttnNetwork(Network):
    def __init__(
        self,
        network_name: str,
        graph,
        rope_table,
        hidden_dim,
        num_heads,
        dim,
        is_dynamic=True,
    ):
        super().__init__(network_name, graph)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads

        self.q_proj = Conv(
                    Q_PROJ,
                    graph,
                    num_output=dim,
                    bias_term=True,
                    kernel_size=[1, 1],
                    stride=[1, 1],
                    out_group_axis=-3,  # hidden dim
                    new_accumulation_dim_name=HIDDEN,
                    is_dynamic=is_dynamic,
                    out_group_size=self.head_dim, #80,  # head dim out group size
                )
        self.k_proj = Conv(
            K_PROJ,
            graph,
            num_output=dim,
            bias_term=True,
            kernel_size=[1, 1],
            stride=[1, 1],
            out_group_axis=-3,  # hidden dim
            new_accumulation_dim_name=HIDDEN,
            is_dynamic=is_dynamic,
            out_group_size=self.head_dim, #80,  # head dim out group size
        )
        self.v_proj = Conv(
            V_PROJ,
            graph,
            num_output=dim,
            bias_term=True,
            kernel_size=[1, 1],
            stride=[1, 1],
            out_group_axis=-1,  # group along tokens
            new_accumulation_dim_name=HIDDEN,
            is_dynamic=is_dynamic,
            out_group_size=64,  # token group size
        )
        self.o_proj = Conv(
            O_PROJ,
            graph,
            num_output=dim,
            bias_term=True,
            kernel_size=[1, 1],
            stride=[1, 1],
            out_group_axis=-3,  # group along hidden dim
            new_accumulation_dim_name=HIDDEN,
            is_dynamic=is_dynamic,
            out_group_size=64,  # token group size
        )

        # from proto_utils.graph import DAG, NamedDim
        # num_tkns = 12 * 12 * 4 # TODO: Fix hardcoding parameters
        # context_dim = self.hidden_dim  # TODO: This was 1280, I am using from hf_config.vision_config.hidden_dim

        # rope_table_named_shape = [
        #     NamedDim(BATCH, 1),
        #     NamedDim(UNSQUEEZE, 1),
        #     NamedDim(MAX_TOKENS, num_tkns),
        #     NamedDim(SIN_COS, context_dim * 2 // 16),
        # ]
        # self.rope_table = permute(
        #     graph,
        #     rope_table_input,
        #     # Input(
        #     #     "rope_table",
        #     #     graph,
        #     #     rope_table_named_shape,
        #     #     is_dynamic=False,
        #     #     out_group_size=None,
        #     #     out_group_axis=None,
        #     # )(),
        #     (MAX_TOKENS, SIN_COS),
        # )

        self.rope = Rope(
            ROPE,
            graph,
            # rope_table=self.rope_table,
            rope_table=rope_table,
            is_dynamic=True,
            out_group_axis=-2,
            out_group_size=self.head_dim, #80,
        )
        self.qk_batchdot = BatchDot(
            QKT_BATCHDOT,
            graph,
            multiplier=1 / ((self.dim / self.num_heads) ** (1 / 2)),
            is_dynamic=is_dynamic,
            # negative indexing used because it will also work if batch dimension is there. if we put 4 here, we have to later handle case for batch
            out_group_axis=-1,
            # Since qkt batchdot is followed by batchdot, we need to group across whole dimension
            out_group_size=9999,  # group along tokens
        )
        self.qkt_softmax = SoftMax(
            QKT_SOFTMAX,
            graph,
            is_dynamic=is_dynamic,
            out_group_axis=-1,
            out_group_size=64,
        )
        self.attention_batchdot = BatchDot(
            ATTN_BATCHDOT,
            graph,
            multiplier=1,
            is_dynamic=is_dynamic,
            out_group_axis=-1,
            out_group_size=self.head_dim, #80,  # group along head dim, since o_proj requires it
        )

    def apply_rope_q_proj(self, hidden_state):
        q_proj = self.q_proj(
            hidden_state,
            derive_name_from_layer_name=True,
            accumulation_dim_name=HIDDEN,
        )
        q_proj = reshape(
            self.graph,
            q_proj,
            [
                [
                    (HIDDEN, self.num_heads, NUM_HEAD),
                    (UNSQUEEZE, self.head_dim, HEAD_DIM),
                ],
                [
                    (HIDDEN, self.num_heads, NUM_HEAD),
                    (UNSQUEEZE, self.head_dim // self.q_proj.out_group_size, HEAD_DIM),
                ],
            ],
            out_group_axis_name=HEAD_DIM,
        )
        return self.rope(q_proj)

    def apply_rope_k_proj(self, hidden_state, num_tkns):
        k_proj = self.k_proj(
            hidden_state,
            derive_name_from_layer_name=True,
            accumulation_dim_name=HIDDEN,
        )
        k_proj = reshape(
            self.graph,
            k_proj,
            [
                [
                    (HIDDEN, self.num_heads, NUM_HEAD),
                    (UNSQUEEZE, self.head_dim, HEAD_DIM),
                    (TKN, num_tkns, TKN_ACC),
                ],
                [
                    (HIDDEN, self.num_heads, NUM_HEAD),
                    (UNSQUEEZE, self.head_dim // self.k_proj.out_group_size, HEAD_DIM),
                    (TKN, num_tkns, TKN_ACC),
                ],
            ],
            out_group_axis_name=HEAD_DIM,
        )
        return self.rope(k_proj)

    def get_v_proj_emb(self, hidden_state):
        v_proj = self.v_proj(
            hidden_state,
            derive_name_from_layer_name=True,
            accumulation_dim_name=HIDDEN,
        )
        v_proj = reshape(
            self.graph,
            v_proj,
            [
                [
                    (HIDDEN, self.num_heads, NUM_HEAD),
                    (UNSQUEEZE, self.head_dim, HEAD_DIM),
                ],
                [
                    (HIDDEN, self.num_heads, NUM_HEAD),
                    (UNSQUEEZE, self.head_dim, HEAD_DIM),
                ],
            ],
            out_group_axis_name=TKN
        )
        return permute(self.graph, v_proj, swap=(HEAD_DIM, TKN))

    def forward(self, hidden_state, **kwargs):
        num_tkns = hidden_state[0].shape[-2].dim
        hidden_state = permute(self.graph, hidden_state, (HIDDEN, TKN))
        # hidden_state (1, 1, 1280, 1024)
        hidden_state = reshape(
            self.graph,
            hidden_state,
            [
                [
                    (UNSQUEEZE, self.dim, HIDDEN),
                    (HIDDEN, 1, UNSQUEEZE),
                ],
                [
                    (UNSQUEEZE, self.dim // 64, HIDDEN),  # TODO: no hardcode!
                    (HIDDEN, 1, UNSQUEEZE),
                ],
            ],
            out_group_axis_name=HIDDEN
        )
        q_proj_emb = self.apply_rope_q_proj(hidden_state)
        k_proj_emb = self.apply_rope_k_proj(hidden_state, num_tkns)
        v_proj_emb = self.get_v_proj_emb(hidden_state)

        qkt = self.qk_batchdot(
            q_proj_emb + k_proj_emb, derive_name_from_layer_name=True
        )

        qkt_softmax = self.qkt_softmax(qkt, axis=-1)
        qkt_softmax = permute(self.graph, qkt_softmax, (TKN, TKN_ACC))

        attention_batchdot = self.attention_batchdot(
            qkt_softmax + v_proj_emb, derive_name_from_layer_name=True
        )
        attention_batchdot = permute(self.graph, attention_batchdot, (TKN, HEAD_DIM))
        attention_batchdot = reshape(
            self.graph,
            attention_batchdot,
            [
                [
                    (NUM_HEAD, self.dim, HIDDEN),
                    (HEAD_DIM, 1, UNSQUEEZE),
                ],
                [
                    (
                        NUM_HEAD,
                        self.dim // self.attention_batchdot.out_group_size,
                        HIDDEN,
                    ),
                    (HEAD_DIM, 1, UNSQUEEZE),
                ],
            ],
            out_group_axis_name=HIDDEN
        )
        o_proj = self.o_proj(
            attention_batchdot,
            accumulation_dim_name=HIDDEN,
            derive_name_from_layer_name=True,
        )
        o_proj = reshape(
            self.graph,
            o_proj,
            [
                [
                    (
                        HIDDEN,
                        1,
                        UNSQUEEZE,
                    ),  # hidden size is reshaped to 1 and becomes unsqueeze dimension
                    (UNSQUEEZE, self.dim, HIDDEN),
                ],
                [
                    (HIDDEN, 1, UNSQUEEZE),
                    (UNSQUEEZE, self.dim // self.o_proj.out_group_size, HIDDEN),
                ],
            ],
            out_group_axis_name=HIDDEN
        )
        o_proj = permute(self.graph, o_proj, swap=(HIDDEN, TKN))

        return o_proj