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
from layer_names import ATTN_BATCHDOT, K_PROJ, O_PROJ, Q_PROJ, QKT_BATCHDOT, QKT_SOFTMAX, ROPE, V_PROJ
from proto_utils.layers import Conv, Rope, Input, BatchDot, SoftMax
from proto_utils.functional import permute, reshape
from proto_utils.graph import NamedDim
from proto_utils import Network
from constants import HIDDEN_DIM, MAX_TOKENS, SIN_COS, TKN, BATCH, UNSQUEEZE, NUM_HEADS, HEAD_DIM, TKN_ACC


class Full_Vision_Attn(Network):
    def __init__(
        self,
        network_name,
        graph,
        rope_table,
        num_heads=16,
        dim=1280,
        max_vision_tokens=1024,
        is_dynamic=True,
    ):
        super().__init__(network_name, graph)
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.dim = dim
        self.q_proj = Conv(
            Q_PROJ,
            graph,
            num_output=dim,
            bias_term=True,
            kernel_size=[1, 1],
            stride=[1, 1],
            out_group_axis=-3,  # hidden dim
            new_accumulation_dim_name=HIDDEN_DIM,
            is_dynamic=is_dynamic,
            out_group_size=80,  # head dim out group size
        )
        self.k_proj = Conv(
            K_PROJ,
            graph,
            num_output=dim,
            bias_term=True,
            kernel_size=[1, 1],
            stride=[1, 1],
            out_group_axis=-3,  # hidden dim
            new_accumulation_dim_name=HIDDEN_DIM,
            is_dynamic=is_dynamic,
            out_group_size=80,  # head dim out group size
        )
        self.v_proj = Conv(
            V_PROJ,
            graph,
            num_output=dim,
            bias_term=True,
            kernel_size=[1, 1],
            stride=[1, 1],
            out_group_axis=-1,  # group along tokens
            new_accumulation_dim_name=HIDDEN_DIM,
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
            new_accumulation_dim_name=HIDDEN_DIM,
            is_dynamic=is_dynamic,
            out_group_size=64,  # token group size
        )
        self.rope_table = rope_table
        self.rope = Rope(
            ROPE,
            graph,
            rope_table=self.rope_table,
            is_dynamic=True,
            out_group_axis=-2,
            out_group_size=80,
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
            graph,
            multiplier=1,
            is_dynamic=is_dynamic,
            out_group_axis=-1,
            out_group_size=80,  # group along head dim, since o_proj requires it
        )

    def apply_rope_q_proj(self, hidden_state):
        q_proj = self.q_proj(
            hidden_state,
            derive_name_from_layer_name=True,
            accumulation_dim_name=HIDDEN_DIM,
        )
        q_proj = reshape(
            self.graph,
            q_proj,
            [
                [
                    (HIDDEN_DIM, self.num_heads, NUM_HEADS),
                    (UNSQUEEZE, self.head_dim, HEAD_DIM),
                ],
                [
                    (HIDDEN_DIM, self.num_heads, NUM_HEADS),
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
            accumulation_dim_name=HIDDEN_DIM,
        )
        k_proj = reshape(
            self.graph,
            k_proj,
            [
                [
                    (HIDDEN_DIM, self.num_heads, NUM_HEADS),
                    (UNSQUEEZE, self.head_dim, HEAD_DIM),
                    (TKN, num_tkns, TKN_ACC),
                ],
                [
                    (HIDDEN_DIM, self.num_heads, NUM_HEADS),
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
            accumulation_dim_name=HIDDEN_DIM,
        )
        v_proj = reshape(
            self.graph,
            v_proj,
            [
                [
                    (HIDDEN_DIM, self.num_heads, NUM_HEADS),
                    (UNSQUEEZE, self.head_dim, HEAD_DIM),
                ],
                [
                    (HIDDEN_DIM, self.num_heads, NUM_HEADS),
                    (UNSQUEEZE, self.head_dim, HEAD_DIM),
                ],
            ],
            out_group_axis_name=TKN
        )
        return permute(self.graph, v_proj, swap=(HEAD_DIM, TKN))

    def forward(self, hidden_state, **kwargs):
        num_tkns = hidden_state[0].shape[-2].dim
        hidden_state = permute(self.graph, hidden_state, (HIDDEN_DIM, TKN))
        # hidden_state (1, 1, 1280, 1024)
        hidden_state = reshape(
            self.graph,
            hidden_state,
            [
                [
                    (UNSQUEEZE, self.dim, HIDDEN_DIM),
                    (HIDDEN_DIM, 1, UNSQUEEZE),
                ],
                [
                    (UNSQUEEZE, self.dim // 64, HIDDEN_DIM),  # TODO: no hardcode!
                    (HIDDEN_DIM, 1, UNSQUEEZE),
                ],
            ],
            out_group_axis_name=HIDDEN_DIM
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
                    (NUM_HEADS, self.dim, HIDDEN_DIM),
                    (HEAD_DIM, 1, UNSQUEEZE),
                ],
                [
                    (
                        NUM_HEADS,
                        self.dim // self.attention_batchdot.out_group_size,
                        HIDDEN_DIM,
                    ),
                    (HEAD_DIM, 1, UNSQUEEZE),
                ],
            ],
            out_group_axis_name=HIDDEN_DIM
        )
        o_proj = self.o_proj(
            attention_batchdot,
            accumulation_dim_name=HIDDEN_DIM,
            derive_name_from_layer_name=True,
        )
        o_proj = reshape(
            self.graph,
            o_proj,
            [
                [
                    (
                        HIDDEN_DIM,
                        1,
                        UNSQUEEZE,
                    ),  # hidden size is reshaped to 1 and becomes unsqueeze dimension
                    (UNSQUEEZE, self.dim, HIDDEN_DIM),
                ],
                [
                    (HIDDEN_DIM, 1, UNSQUEEZE),
                    (UNSQUEEZE, self.dim // self.o_proj.out_group_size, HIDDEN_DIM),
                ],
            ],
            out_group_axis_name=HIDDEN_DIM
        )
        o_proj = permute(self.graph, o_proj, swap=(HIDDEN_DIM, TKN))

        return o_proj

    def get_top_shapes(self, inp_shapes):
        return inp_shapes

    def get_nickname(self):
        return "attn"

    def get_dict(self):
        i = self.layer_idx
        return {
            f"visual.blocks.{i}.attn.proj": self.o_proj.layer_name,  # Weight Term
            f"visual.blocks.{i}.attn.qkv": [self.q_proj.layer_name, self.k_proj.layer_name, self.v_proj.layer_name],  # Weight Term
        }
