import os
import sys
from pathlib import Path

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
    DEFAULT_GROUP_SIZE,
    TEMPORAL_DIM,
    VISION_ROPE,
    WIDTH_TEMPORAL_DIM,
    HEIGHT_WINDOWS,
    WIDTH_WINDOWS_AND_ALL_TEMPORAL,
    UNSQUEEZE,
    WINDOWED_TKNS,
    WINDOWED_TKNS_ACC,
)

from proto_utils import Network
from proto_utils.layers import Conv, Input, Rope, BatchDot, SoftMax
from proto_utils.functional import ReshapeMaker, reshape_with_reshapemaker, permute, reshape

# At some point this constants.py should be moved here in qwen2_5_vl directory
sys.path.append(str(Path(__file__).absolute().parent.parent.parent / 'qwen2_5vl_3b'))
sys.path.append(str(Path(__file__).absolute().parent.parent.parent / 'qwen2_5vl_3b/vision_encoder'))
from vision_encoder.constants import HIDDEN_DIM, MAX_TOKENS, SIN_COS
from vision_encoder.layer_names import ATTN_BATCHDOT, O_PROJ, Q_PROJ, K_PROJ, QKT_BATCHDOT, QKT_SOFTMAX, ROPE, V_PROJ

class WindowedAttnNetwork(Network):
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
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.dim = dim
        self.q_proj = Conv(
            Q_PROJ,
            self.graph,
            num_output=dim,
            bias_term=True,
            kernel_size=[1, 1],
            stride=[1, 1],
            out_group_axis=-3,  # hidden dim
            new_accumulation_dim_name=HIDDEN,
            is_dynamic=is_dynamic,
            out_group_size=80,  # head dim out group size
        )
        self.k_proj = Conv(
            K_PROJ,
            self.graph,
            num_output=dim,
            bias_term=True,
            kernel_size=[1, 1],
            stride=[1, 1],
            out_group_axis=-3,  # hidden dim
            new_accumulation_dim_name=HIDDEN,
            is_dynamic=is_dynamic,
            out_group_size=self.head_dim,  # head dim out group size
        )
        self.v_proj = Conv(
            V_PROJ,
            self.graph,
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
            self.graph,
            num_output=dim,
            bias_term=True,
            kernel_size=[1, 1],
            stride=[1, 1],
            out_group_axis=-3,  # group along hidden dim
            new_accumulation_dim_name=HIDDEN,
            is_dynamic=is_dynamic,
            out_group_size=64,  # token group size
        )
        self.rope = Rope(
            VISION_ROPE,
            self.graph,
            rope_table=rope_table,
            is_dynamic=True,
            out_group_axis=-2,
            out_group_size=self.head_dim,
        )
        self.qk_batchdot = BatchDot(
            QKT_BATCHDOT,
            graph,
            multiplier=1 / ((self.dim / self.num_heads) ** (1 / 2)),
            is_dynamic=is_dynamic,
            out_group_axis=-1,
            out_group_size=64,  # group along tokens
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
            self.graph,
            multiplier=1,
            is_dynamic=is_dynamic,
            out_group_axis=-1,
            out_group_size=self.head_dim,  # group along head dim, since o_proj requires it
        )

    def apply_rope_q_proj(self, hidden_state):
        q_proj = self.q_proj(
            hidden_state,
            derive_name_from_layer_name=True,
            accumulation_dim_name=HIDDEN,
        )
        # intention: split hidden dim between heads
        # effectively, (UNSQUEEZE, HIDDEN) -> (NUM_HEAD, HEAD_DIM), (1, 1280) -> (16, 80)
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
        # intention: split hidden dim between heads
        # effectively, (UNSQUEEZE, HIDDEN) -> (NUM_HEAD, HEAD_DIM), (1, 1280) -> (16, 80)
        k_proj = reshape(
            self.graph,
            k_proj,
            [
                [
                    (HIDDEN, self.num_heads, NUM_HEAD),
                    (UNSQUEEZE, self.head_dim, HEAD_DIM),
                ],
                [
                    (HIDDEN, self.num_heads, NUM_HEAD),
                    (UNSQUEEZE, self.head_dim // 64, HEAD_DIM),
                ],
            ],
            out_group_axis_name=HEAD_DIM,
        )
        return self.rope(k_proj)

    def forward(self, hidden_state, **kwargs):
        num_tkns = hidden_state[0].shape[-2].dim
        hidden_state = permute(self.graph, hidden_state, (HIDDEN, TKN))

        # Intention: [1,1280,1,576], to prepare for q_proj, k_proj, v_proj conv
        # for which we need hidden dim to be 3rd last
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
                    (HIDDEN, 1, UNSQUEEZE),  # (hidden, 1, unsqueeze)
                ],
            ],
            out_group_axis_name=HIDDEN,
        )
        q_proj_emb = self.apply_rope_q_proj(hidden_state)

        # intention: split tokens (576) into 9 windows of size 64. To make space, we left shift the dimensions
        # and pop off the batch dim
        # effectively,(batch, num_head, head_dim, tkn) -> (num_head, head_dim, num_windows, windowed_tkns)  # (1, 16, 80, 576) -> (16, 80, 9, 64)
        q_proj_emb = reshape(
            self.graph,
            q_proj_emb,
            [
                [
                    (BATCH, self.num_heads, NUM_HEAD),
                    (NUM_HEAD, self.head_dim, HEAD_DIM),
                    (HEAD_DIM, num_tkns // 64, NUM_WINDOWS),
                    (TKN, 64, WINDOWED_TKNS),
                ],
                [
                    (BATCH, self.num_heads, NUM_HEAD),
                    (NUM_HEAD, self.head_dim // self.q_proj.out_group_size, HEAD_DIM),
                    (HEAD_DIM, num_tkns // 64, NUM_WINDOWS),
                    (TKN, 64, WINDOWED_TKNS),
                ],
            ],
            out_group_axis_name=HEAD_DIM,
        )
        # this effectively brings num_windows "out" so that remaining last two dim
        # are prepared for batchdot, which requires (head_dim, (windowed)tkn)
        q_proj_emb = permute(self.graph, q_proj_emb, (HEAD_DIM, NUM_WINDOWS))

        # same intention as q_proj. Just marking the token axis as accumulation for use int attention_batchdot
        # so that in the output of qkt_batchdot, we can identify the actual accumulation token axis.
        # Look into intention of qkt_batchdot for details
        k_proj_emb = self.apply_rope_k_proj(hidden_state, num_tkns)
        k_proj_emb = reshape(
            self.graph,
            k_proj_emb,
            [
                [
                    (BATCH, self.num_heads, NUM_HEAD),
                    (NUM_HEAD, self.head_dim, HEAD_DIM),
                    (HEAD_DIM, num_tkns // 64, NUM_WINDOWS),
                    (TKN, 64, WINDOWED_TKNS_ACC),
                ],
                [
                    (BATCH, self.num_heads, NUM_HEAD),
                    (NUM_HEAD, self.head_dim // self.q_proj.out_group_size, HEAD_DIM),
                    (HEAD_DIM, num_tkns // 64, NUM_WINDOWS),
                    (TKN, 64, WINDOWED_TKNS_ACC),
                ],
            ],
            out_group_axis_name=HEAD_DIM,
        )
        k_proj_emb = permute(self.graph, k_proj_emb, (HEAD_DIM, NUM_WINDOWS))

        v_proj_emb = self.v_proj(
            hidden_state,
            derive_name_from_layer_name=True,
            accumulation_dim_name=HIDDEN,
        )

        # same intention as q_proj, just group accross tokens, and mark token axis as accumulation token axis
        # for use in attention batchdot
        v_proj_emb = reshape(
            self.graph,
            v_proj_emb,
            [
                [
                    (BATCH, self.num_heads, NUM_HEAD),
                    (HIDDEN, self.head_dim, HEAD_DIM),
                    (UNSQUEEZE, num_tkns // 64, NUM_WINDOWS),
                    (TKN, 64, WINDOWED_TKNS_ACC),
                ],
                [
                    (BATCH, self.num_heads, NUM_HEAD),
                    (HIDDEN, self.head_dim, HEAD_DIM),
                    (UNSQUEEZE, num_tkns // 64, NUM_WINDOWS),
                    (TKN, 64 // 64, WINDOWED_TKNS_ACC),
                ],
            ],
            out_group_axis_name=WINDOWED_TKNS_ACC,
        )

        # intention: the input is q_proj_emb: (HEAD_DIM, WINDOWED_TKN), (HEAD_DIM, WINDOWED_TKN_ACC)
        # The accumulation axis is HEAD_DIM, and output is (WINDOWED_TKN, WINDOWED_TKN_ACC)
        # That is, k_proj_emb's token axis becomes the accumulation axis for attention batchdot
        # We made this naming clear when creating k_proj_emb
        qkt = self.qk_batchdot(
            q_proj_emb + k_proj_emb, derive_name_from_layer_name=True
        )
        qkt_softmax = self.qkt_softmax(qkt, axis=-1)
        qkt_softmax = permute(
            self.graph, qkt_softmax, (WINDOWED_TKNS, WINDOWED_TKNS_ACC)
        )
        # Intention: prepare v_proj_emb for attention batchdot by making last two dims as (head_dim, windowed_tkn_acc)
        # (NUM_HEADS, HEAD_DIM, NUM_WINDOWS, WINDOWED_TKN_ACC) -> (NUM_HEADS, NUM_WINDOWS, WINDOWED_TKN_ACC, HEAD_DIM)
        v_proj_emb = permute(self.graph, v_proj_emb, swap=(HEAD_DIM, NUM_WINDOWS))
        v_proj_emb = permute(self.graph, v_proj_emb, swap=(HEAD_DIM, WINDOWED_TKNS_ACC))

        # input will be (WINDOWED_TKN_ACC, WINDOWED_TKN), (WINDOWED_TKN_ACC, HEAD_DIM). We accumulate along WINDOWED_TKN_ACC, with output being (WINDOWED_TKN, HEAD_DIM)
        attention_batchdot = self.attention_batchdot(
            qkt_softmax + v_proj_emb, derive_name_from_layer_name=True
        )

        # Intention: We merge the windowed tokens together.
        # (NUM_WINDOWS, WINDOWED_TKNS) -> (UNSQUEEZE, TKN)
        # (9, 64) -> (1, 576)
        attention_batchdot = reshape(
            self.graph,
            attention_batchdot,
            [
                [
                    (NUM_WINDOWS, 1, UNSQUEEZE),
                    (WINDOWED_TKNS, num_tkns, TKN),
                ],
                [
                    (NUM_WINDOWS, 1, UNSQUEEZE),
                    (WINDOWED_TKNS, num_tkns, TKN),
                ],
            ],
            out_group_axis_name=HEAD_DIM,
        )
        # Intention: Merge NUM_HEADS and HEAD_DIM to finally bring the HIDDEN dim into 3rd last position for o_proj conv
        # (UNSQUEEZE, TKN, HEAD_DIM) -> (HEAD_DIM, HEAD_DIM, TKN) -> (HIDDEN, UNSQUEEZE, TKN)
        # rename NUM_HEADS as batch in the process, since merging 
        # is done.
        attention_batchdot = permute(self.graph, attention_batchdot, (TKN, HEAD_DIM))
        attention_batchdot = reshape(
            self.graph,
            attention_batchdot,
            [
                [
                    (NUM_HEAD, 1, BATCH),
                    (UNSQUEEZE, self.dim, HIDDEN),
                    (HEAD_DIM, 1, UNSQUEEZE),
                ],
                [
                    (NUM_HEAD, 1, BATCH),
                    (
                        UNSQUEEZE,
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

        # bring hidden dim onto second last dim. Instead of permute, we use reshape since its 
        # effcient when one of the dims is 1
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
        
        # finally, bring hidden dim to last dimension
        o_proj = permute(self.graph, o_proj, swap=(HIDDEN, TKN))

        return o_proj
