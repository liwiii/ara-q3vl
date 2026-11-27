import os
import sys
from pathlib import Path

from proto_utils import Network

from llm_compiler.qwen2_5_vl.constants import HIDDEN, TKN, UNSQUEEZE

sys.path.append(str(Path(__file__).absolute().parent.parent.parent / 'qwen2_5vl_3b'))
sys.path.append(str(Path(__file__).absolute().parent.parent.parent / 'qwen2_5vl_3b/vision_encoder'))
from vision_encoder.layer_names import *

from proto_utils.layers.eltwise import Operation, Eltwise
from proto_utils.layers import Conv, Input, Rope, BatchDot, SoftMax, Gelu
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
        self.attn = attn  # always full
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

        # MLP is fc1->gelu->fc2
        self.mlp_fc1 = Conv(
            "mlp_fc1",
            self.graph,
            num_output=hf_config.vision_config.intermediate_size,
            bias_term=True,
            out_group_axis=-3,
            out_group_size=64,
            kernel_size=[1, 1],
            stride=[1, 1],
            new_accumulation_dim_name=HIDDEN,
            is_dynamic=is_dynamic,
        )
        self.gelu = Gelu("gelu", self.graph, out_group_axis=-3, out_group_size=64, is_dynamic=is_dynamic)
        self.mlp_fc2 = Conv(
            "mlp_fc2",
            self.graph,
            num_output=hf_config.vision_config.hidden_size,
            bias_term=True,
            out_group_axis=-3,
            out_group_size=64,
            kernel_size=[1, 1],
            stride=[1, 1],
 
            new_accumulation_dim_name=HIDDEN,
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

        fc1_hidden_states = self.mlp_fc1(norm2_hidden_states, accumulation_dim_name = HIDDEN)
        # no permute needed since former conv is gonna be merged with below gelu
        gelu_hidden_states = self.gelu(fc1_hidden_states)
        fc2_hidden_states = self.mlp_fc2(gelu_hidden_states, accumulation_dim_name = HIDDEN)

        mlp_out = fc2_hidden_states

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
