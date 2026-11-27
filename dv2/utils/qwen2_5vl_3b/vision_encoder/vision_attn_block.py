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
#from get_caffemodel import get_caffemodel
from layer_names import DOWN_PROJ, FINAL_ADD, GATE_PROJ, LNORM_1, LNORM_2, RESIDUAL_ADD, SILU, SILU_MUL, UP_PROJ
from proto_utils import Network
from proto_utils.layers import Conv, Input, Eltwise, LayerNorm, Swish
from proto_utils.functional import permute, reshape
from full_attn_vision import Full_Vision_Attn
from qwen2_vl_attention import Qwen2_5VL_Attn
from proto_utils.layers.eltwise import Operation
from typing import Union

from constants import HIDDEN_DIM, TKN, BATCH, UNSQUEEZE, MAX_TOKENS, SIN_COS


class Qwen2_5_VL_AttentionBlock(Network):
    def __init__(
        self,
        network_name,
        graph,
        layer_idx,
        attn: Union[Full_Vision_Attn, Qwen2_5VL_Attn],
        is_dynamic=True,
    ):
        super().__init__(network_name, graph)
        self.layer_idx = layer_idx
        self.network_name += f"_{layer_idx}"
        self.residual_add = Eltwise(
            RESIDUAL_ADD,
            self.graph,
            -1,
            64,
            is_dynamic=is_dynamic,
            operation=Operation.SUM,
        )
        self.silu_mul = Eltwise(
            SILU_MUL,
            self.graph,
            -1,
            64,
            is_dynamic=is_dynamic,
            operation=Operation.PROD,
        )
        self.final_add = Eltwise(
            FINAL_ADD,
            self.graph,
            -1,
            64,
            is_dynamic=is_dynamic,
            operation=Operation.SUM,
        )
        self.attn = attn
        self.attn.layer_idx = layer_idx
        self.lnorm1 = LayerNorm(
            LNORM_1,
            self.graph,
            out_group_axis=-1,
            out_group_size=64,
            uses_mean=False,
            eps=1e-06,
            is_dynamic=True,
        )
        self.lnorm2 = LayerNorm(
            LNORM_2,
            self.graph,
            out_group_axis=-1,
            out_group_size=64,
            uses_mean=False,
            eps=1e-06,
            is_dynamic=True,
        )
        self.gate_proj = Conv(
            GATE_PROJ,
            self.graph,
            num_output=3456,
            bias_term=True,
            out_group_axis=-3,
            out_group_size=64,
            kernel_size=[1, 1],
            stride=[1, 1],
            new_accumulation_dim_name=HIDDEN_DIM,
        )
        self.up_proj = Conv(
            UP_PROJ,
            self.graph,
            num_output=3456,
            bias_term=True,
            out_group_axis=-3,
            out_group_size=64,
            kernel_size=[1, 1],
            stride=[1, 1],
            new_accumulation_dim_name=HIDDEN_DIM,
        )
        self.down_proj = Conv(
            DOWN_PROJ,
            self.graph,
            num_output=1280,
            bias_term=True,
            out_group_axis=-3,
            out_group_size=64,
            kernel_size=[1, 1],
            stride=[1, 1],
            new_accumulation_dim_name=HIDDEN_DIM,
        )
        self.act_fn = Swish(
            SILU,
            self.graph,
            out_group_axis=-3,
            out_group_size=64,
            is_dynamic=is_dynamic,
        )

    def forward(self, hidden_states, **kwargs):
        input_attn = self.lnorm1(hidden_states, derive_name_from_layer_name=True)
        hidden_states = self.residual_add(
            self.attn(input_attn) + hidden_states, derive_name_from_layer_name=True
        )
        norm2_hidden_states = self.lnorm2(hidden_states)
        # permute and reshape here
        norm2_hidden_states = permute(
            self.graph, norm2_hidden_states, (TKN, HIDDEN_DIM)
        )
        norm2_hidden_states = reshape(
            self.graph,
            norm2_hidden_states,
            [
                [(UNSQUEEZE, self.attn.dim, HIDDEN_DIM), (HIDDEN_DIM, 1, UNSQUEEZE)],
                [
                    (
                        UNSQUEEZE,
                        self.attn.dim // self.residual_add.out_group_size,
                        HIDDEN_DIM,
                    ),
                    (HIDDEN_DIM, 1, UNSQUEEZE),
                ],
            ],
            HIDDEN_DIM,
        )
        up_proj = self.up_proj(
            norm2_hidden_states,
            derive_name_from_layer_name=True,
            accumulation_dim_name=HIDDEN_DIM,
        )
        gate_proj = self.gate_proj(
            norm2_hidden_states,
            derive_name_from_layer_name=True,
            accumulation_dim_name=HIDDEN_DIM,
        )
        activated_gate_proj = self.act_fn(gate_proj, derive_name_from_layer_name=True)
        activated_gate_proj = reshape(self.graph, activated_gate_proj, [
            [
                (HIDDEN_DIM, 1, UNSQUEEZE), (UNSQUEEZE, 3456, HIDDEN_DIM)
            ],
            [
                (HIDDEN_DIM, 1, UNSQUEEZE), (UNSQUEEZE, 54, HIDDEN_DIM)
            ]
        ], out_group_axis_name=HIDDEN_DIM)
        up_proj = reshape(self.graph, up_proj, [
            [
                (HIDDEN_DIM, 1, UNSQUEEZE), (UNSQUEEZE, 3456, HIDDEN_DIM)
            ],
            [
                (HIDDEN_DIM, 1, UNSQUEEZE), (UNSQUEEZE, 54, HIDDEN_DIM)
            ]
        ], out_group_axis_name=HIDDEN_DIM)
        activated_gate_proj = permute(self.graph, activated_gate_proj,(HIDDEN_DIM, TKN))
        up_proj = permute(self.graph, up_proj,(HIDDEN_DIM, TKN))
        silu = self.silu_mul(
            activated_gate_proj + up_proj, derive_name_from_layer_name=True
        )
        silu = permute(self.graph, silu,(HIDDEN_DIM, TKN))
        silu = reshape(self.graph, silu, [
            [
                (HIDDEN_DIM, 1, UNSQUEEZE), (UNSQUEEZE, 3456, HIDDEN_DIM)
            ],
            [
                (HIDDEN_DIM, 1, UNSQUEEZE), (UNSQUEEZE, 54, HIDDEN_DIM)
            ]
        ], out_group_axis_name=HIDDEN_DIM)
        mlp_out = self.down_proj(
            silu, accumulation_dim_name=HIDDEN_DIM, derive_name_from_layer_name=True
        )
        mlp_out = reshape(
            self.graph,
            mlp_out,
            [
                [(UNSQUEEZE, self.attn.dim, HIDDEN_DIM), (HIDDEN_DIM, 1, UNSQUEEZE)],
                [
                    (
                        UNSQUEEZE,
                        self.attn.dim // self.residual_add.out_group_size,
                        HIDDEN_DIM,
                    ),
                    (HIDDEN_DIM, 1, UNSQUEEZE),
                ],
            ],
            out_group_axis_name=HIDDEN_DIM
        )
        mlp_out = permute(self.graph, mlp_out, (TKN, HIDDEN_DIM))
        return self.final_add(hidden_states + mlp_out, derive_name_from_layer_name=True)

    def get_top_shapes(self, inp_shapes):
        return inp_shapes

    def get_nickname(self):
        return "attn"
    
    def get_dict(self):
        i = self.layer_idx
        final_torch_to_caffe = {
            f"visual.blocks.{i}.mlp.down_proj": self.down_proj.layer_name,
            f"visual.blocks.{i}.mlp.gate_proj": self.gate_proj.layer_name,
            f"visual.blocks.{i}.mlp.up_proj": self.up_proj.layer_name,
            f"visual.blocks.{i}.norm1": self.lnorm1.layer_name,
            f"visual.blocks.{i}.norm2": self.lnorm2.layer_name
        }
        final_torch_to_caffe.update(self.attn.get_dict())
        return final_torch_to_caffe
