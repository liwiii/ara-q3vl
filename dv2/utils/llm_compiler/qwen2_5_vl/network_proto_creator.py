from pathlib import Path
import sys
import os

from llm_compiler.qwen2_5_vl.constants import (
    BATCH,
    FIXED_HEIGHT,
    FIXED_WIDTH,
    UNSQUEEZE,
)

# src/dv/sdk/utils/qwen2_5vl_3b/vision_encoder/constants.py
# At some point this constants.py should be moved here in qwen2_5_vl directory
sys.path.append(str(Path(__file__).absolute().parent.parent.parent / 'qwen2_5vl_3b'))
sys.path.append(str(Path(__file__).absolute().parent.parent.parent / 'qwen2_5vl_3b/vision_encoder'))
from vision_encoder.constants import HIDDEN_DIM, MAX_TOKENS, SIN_COS

from proto_utils import Network
from proto_utils.layers import Conv, Input
from .modules.patch_embed import PatchEmbedNetwork
from .modules.full_attn import FullAttnNetwork
from .modules.windowed_attn import WindowedAttnNetwork
from .modules.attn_block import AttnBlockNetwork
from .modules.merger import MergerNetwork
from proto_utils.functional import permute

class Qwen2_5_VL_VisionNetwork(Network):
    # Singleton members
    _rope_table_input = None
    _rope_table = None

    def __init__(self,
                 network_name: str,
                 graph,
                 # Ideally this should be transformers.Qwen2_5_VLConfig object, but we
                 # do not run --mode 2 with custom python environment with transformers.
                 hf_config:object,
                 hidden_dim,
                 pixels_width,
                 pixels_height,
                 spatial_merge_size,
                 is_rotated):
        super().__init__(network_name, graph)
        self.hf_config = hf_config
        self.patch_embed = PatchEmbedNetwork("patch_embed", graph, hidden_dim, pixels_width, pixels_height, hf_config.vision_config.spatial_patch_size, spatial_merge_size, is_rotated)
        spatial_merge_unit = spatial_merge_size * spatial_merge_size
        num_tkns = FIXED_HEIGHT * FIXED_WIDTH * spatial_merge_unit
        context_dim = hidden_dim
        from proto_utils.graph import NamedDim
        rope_table_named_shape = [
            NamedDim(BATCH, 1),
            NamedDim(UNSQUEEZE, 1),
            NamedDim(MAX_TOKENS, num_tkns),
            NamedDim(SIN_COS, context_dim * 2 // 16),
        ]
        head_dim = dim=self.hf_config.vision_config.hidden_size // self.hf_config.vision_config.num_heads
        if Qwen2_5_VL_VisionNetwork._rope_table_input is None:
            Qwen2_5_VL_VisionNetwork._rope_table_input =  Input(
                    "rope_table",
                    graph,
                    rope_table_named_shape,
                    is_dynamic=False,
                    out_group_size=head_dim*2,
                    out_group_axis=-1,
                    # No input type.
                ) ()  # Or layout = NCHW
            Qwen2_5_VL_VisionNetwork._rope_table = permute(
                graph,
                Qwen2_5_VL_VisionNetwork._rope_table_input,
                (MAX_TOKENS, SIN_COS),
            )                

        self.attn_block = []
        for attn_block_id in range(self.hf_config.vision_config.depth):
            if attn_block_id in self.hf_config.vision_config.fullatt_block_indexes:
                attn = FullAttnNetwork(f"full_attn", graph, rope_table=Qwen2_5_VL_VisionNetwork._rope_table, hidden_dim=hidden_dim, num_heads=self.hf_config.vision_config.num_heads, dim=self.hf_config.vision_config.hidden_size)
            else:
                attn = WindowedAttnNetwork(f"windowed_attn", graph, rope_table=Qwen2_5_VL_VisionNetwork._rope_table, hidden_dim=hidden_dim, num_heads=self.hf_config.vision_config.num_heads, dim=self.hf_config.vision_config.hidden_size)

            self.attn_block.append(AttnBlockNetwork(
                f"attn_block",
                graph,
                attn,
                attn_block_id,
                hf_config=hf_config
            ))

        # Prioritize hf_config.text_config.hidden_size over hf_config.hidden_size
        hidden_dim_for_merger = hf_config.text_config.hidden_size if hasattr(hf_config, 'text_config') and hasattr(hf_config.text_config, 'hidden_size') else hf_config.hidden_size
        self.merger = MergerNetwork("merger", graph, is_dynamic=True, frame_height=FIXED_HEIGHT, frame_width=FIXED_WIDTH, context_dim = context_dim, hidden_dim=hidden_dim_for_merger)


    def forward(self, inputs, **kwargs):
        self.update_layer_names()
        latest_layer = self.patch_embed(inputs)
        for attn_block_id in range(self.hf_config.vision_config.depth):
            latest_layer = self.attn_block[attn_block_id](latest_layer)
        latest_layer = self.merger(latest_layer)
