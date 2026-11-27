import os
import sys
from pathlib import Path

from proto_utils import Network

from llm_compiler.qwen2_5_vl.constants import HIDDEN, TKN, UNSQUEEZE
from llm_compiler.padding_util import next_multiple_of

# At some point this constants.py should be moved here in qwen2_5_vl directory
sys.path.append(str(Path(__file__).absolute().parent.parent.parent / 'qwen2_5vl_3b'))
sys.path.append(str(Path(__file__).absolute().parent.parent.parent / 'qwen2_5vl_3b/vision_encoder'))
from vision_encoder.layer_names import *

from proto_utils.layers.eltwise import Operation, Eltwise
from proto_utils.layers import Conv, Input, Rope, BatchDot, SoftMax
from proto_utils.layers.lnrm import LayerNorm
from proto_utils.layers.swish import Swish
from proto_utils.functional import permute, reshape

class AttnBlockNetwork(Network):

    def __init__(
        self,
        network_name: str,
        graph,
        attn,
        layer_idx,
        hf_config,
        is_dynamic=True,
    ):
        super().__init__(network_name, graph)
        self.attn = attn  # Can be full or windowed
        self.layer_idx = layer_idx
        self.network_name += f"_{layer_idx}"
        self.hf_config = hf_config

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
            num_output=next_multiple_of(hf_config.vision_config.intermediate_size, 64),  # 3456,
            bias_term=True,
            out_group_axis=-3,
            out_group_size=64,
            kernel_size=[1, 1],
            stride=[1, 1],
            new_accumulation_dim_name=HIDDEN,
        )
        self.up_proj = Conv(
            UP_PROJ,
            self.graph,
            num_output=next_multiple_of(hf_config.vision_config.intermediate_size, 64),  # 3456,
            bias_term=True,
            out_group_axis=-3,
            out_group_size=64,
            kernel_size=[1, 1],
            stride=[1, 1],
            new_accumulation_dim_name=HIDDEN,
        )
        self.down_proj = Conv(
            DOWN_PROJ,
            self.graph,
            num_output=hf_config.vision_config.hidden_size,
            bias_term=True,
            out_group_axis=-3,
            out_group_size=64,
            kernel_size=[1, 1],
            stride=[1, 1],
            new_accumulation_dim_name=HIDDEN,
        )
        self.act_fn = Swish(
            SILU,
            self.graph,
            out_group_axis=-3,
            out_group_size=64,
            is_dynamic=is_dynamic,
        )

    def forward(self, hidden_states, **kwargs):
        print(f"AttnBlock.forward received hidden_states: {hidden_states[0].shape}")
        input_attn = self.lnorm1(hidden_states, derive_name_from_layer_name=True)
        print(f"AttnBlock.forward input_attn: {input_attn[0].shape}")
        hidden_states = self.residual_add(
            self.attn(input_attn) + hidden_states, derive_name_from_layer_name=True
        )
        norm2_hidden_states = self.lnorm2(hidden_states)
        # permute and reshape here
        norm2_hidden_states = permute(
            self.graph, norm2_hidden_states, (TKN, HIDDEN)
        )
        norm2_hidden_states = reshape(
            self.graph,
            norm2_hidden_states,
            [
                [(UNSQUEEZE, self.attn.dim, HIDDEN), (HIDDEN, 1, UNSQUEEZE)],
                [
                    (
                        UNSQUEEZE,
                        self.attn.dim // self.residual_add.out_group_size,
                        HIDDEN,
                    ),
                    (HIDDEN, 1, UNSQUEEZE),
                ],
            ],
            HIDDEN,
        )
        up_proj = self.up_proj(
            norm2_hidden_states,
            derive_name_from_layer_name=True,
            accumulation_dim_name=HIDDEN,
        )
        gate_proj = self.gate_proj(
            norm2_hidden_states,
            derive_name_from_layer_name=True,
            accumulation_dim_name=HIDDEN,
        )
        activated_gate_proj = self.act_fn(gate_proj, derive_name_from_layer_name=True)
        activated_gate_proj = reshape(self.graph, activated_gate_proj, [
            [
                (HIDDEN, 1, UNSQUEEZE), (UNSQUEEZE, next_multiple_of(self.hf_config.vision_config.intermediate_size, 64), HIDDEN)
            ],
            [
                (HIDDEN, 1, UNSQUEEZE), (UNSQUEEZE,  next_multiple_of(self.hf_config.vision_config.intermediate_size, 64) // 64, HIDDEN)
            ]
        ], out_group_axis_name=HIDDEN)
        up_proj = reshape(self.graph, up_proj, [
            [
                (HIDDEN, 1, UNSQUEEZE), (UNSQUEEZE, next_multiple_of(self.hf_config.vision_config.intermediate_size, 64), HIDDEN)
            ],
            [
                (HIDDEN, 1, UNSQUEEZE), (UNSQUEEZE, next_multiple_of(self.hf_config.vision_config.intermediate_size, 64) // 64, HIDDEN)
            ]
        ], out_group_axis_name=HIDDEN)
        activated_gate_proj = permute(self.graph, activated_gate_proj,(HIDDEN, TKN))
        up_proj = permute(self.graph, up_proj,(HIDDEN, TKN))
        silu = self.silu_mul(
            activated_gate_proj + up_proj, derive_name_from_layer_name=True
        )
        silu = permute(self.graph, silu,(HIDDEN, TKN))
        silu = reshape(self.graph, silu, [
            [
                (HIDDEN, 1, UNSQUEEZE), (UNSQUEEZE, next_multiple_of(self.hf_config.vision_config.intermediate_size, 64), HIDDEN)
            ],
            [
                (HIDDEN, 1, UNSQUEEZE), (UNSQUEEZE, next_multiple_of(self.hf_config.vision_config.intermediate_size, 64) // 64, HIDDEN)
            ]
        ], out_group_axis_name=HIDDEN)
        mlp_out = self.down_proj(
            silu, accumulation_dim_name=HIDDEN, derive_name_from_layer_name=True
        )
        mlp_out = reshape(
            self.graph,
            mlp_out,
            [
                [(UNSQUEEZE, self.attn.dim, HIDDEN), (HIDDEN, 1, UNSQUEEZE)],
                [
                    (
                        UNSQUEEZE,
                        self.attn.dim // self.residual_add.out_group_size,
                        HIDDEN,
                    ),
                    (HIDDEN, 1, UNSQUEEZE),
                ],
            ],
            out_group_axis_name=HIDDEN
        )
        mlp_out = permute(self.graph, mlp_out, (TKN, HIDDEN))
        return self.final_add(hidden_states + mlp_out, derive_name_from_layer_name=True)
