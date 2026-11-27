from pathlib import Path
import sys

from llm_compiler.qwen2_5_vl.constants import (
    BATCH,
    FIXED_HEIGHT,
    FIXED_WIDTH,
    UNSQUEEZE,
)

sys.path.append(str(Path(__file__).absolute().parent.parent.parent / 'qwen2_5vl_3b'))
sys.path.append(str(Path(__file__).absolute().parent.parent.parent / 'qwen2_5vl_3b/vision_encoder'))
from vision_encoder.constants import HIDDEN_DIM, MAX_TOKENS, SIN_COS

from proto_utils import Network
from proto_utils.layers import Conv, Input
from proto_utils.layers.eltwise import Operation, Eltwise

from proto_utils.functional import permute

sys.path.append(str(Path(__file__).absolute().parent.parent / 'qwen2_5_vl'))
sys.path.append(str(Path(__file__).absolute().parent.parent / 'qwen2_5_vl/modules'))
from qwen3_vl.modules.patch_embed import PatchEmbedNetwork
from qwen2_5_vl.modules.full_attn import FullAttnNetwork 
from qwen3_vl.modules.attn_block import AttnBlockNetwork
from qwen3_vl.modules.merger import MergerNetwork
from llm_compiler.qwen2_5_vl.network_proto_creator import Qwen2_5_VL_VisionNetwork
from qwen2_5_vl.constants import *
from qwen3_vl.constants import *

class Qwen3_VL_VisionNetwork(Qwen2_5_VL_VisionNetwork):
    def __init__(self,
                 network_name: str,
                 graph:object,
                 # Ideally this should be transformers.Qwen3_VLConfig object, but we
                 # do not run --mode 2 with custom python environment with transformers.
                 hf_config:object,
                 hidden_dim:int,
                 pixels_width:int,
                 pixels_height:int,
                 spatial_merge_size:int,
                 artificial_large_channel_num:int,
                 is_rotated:bool):
        hf_config.vision_config.fullatt_block_indexes = range(hf_config.vision_config.depth)
        super().__init__(network_name, graph, hf_config, hidden_dim, pixels_width, pixels_height, spatial_merge_size, is_rotated)
        self.patch_embed = PatchEmbedNetwork("patch_embed", graph, hidden_dim, pixels_width, pixels_height, hf_config.vision_config.spatial_patch_size, spatial_merge_size, artificial_large_channel_num, is_rotated)
        context_dim = hidden_dim
        self.post_patch_embed_residual = Eltwise(
            POST_PATCH_EMBED_RESIDUAL,
            self.graph,
            -1,
            64,
            is_dynamic=True,
            operation=Operation.SUM,
        )
        # Prioritize hf_config.text_config.hidden_size over hf_config.hidden_size
        hidden_dim_for_merger = hf_config.text_config.hidden_size if hasattr(hf_config, 'text_config') and hasattr(hf_config.text_config, 'hidden_size') else hf_config.hidden_size
        self.deepstack_merger = {}

        self.attn_block = []
        for attn_block_id in range(self.hf_config.vision_config.depth):
            # Ideally _rope_table should come from SingletonProvider._rope_table following ideal singleton design pattern
            attn = FullAttnNetwork(f"full_attn", graph, rope_table=Qwen2_5_VL_VisionNetwork._rope_table, hidden_dim=hidden_dim, num_heads=self.hf_config.vision_config.num_heads, dim=self.hf_config.vision_config.hidden_size)
            self.attn_block.append(AttnBlockNetwork(
                f"attn_block",
                graph,
                attn,
                attn_block_id,
                hf_config=hf_config
            ))
        self.merger = MergerNetwork("merger", graph, is_dynamic=True, frame_height=FIXED_HEIGHT, frame_width=FIXED_WIDTH, context_dim = context_dim, hidden_dim=hidden_dim_for_merger, is_deepstack = False)

        for attn_block_id in self.hf_config.vision_config.deepstack_visual_indexes:
            print(f"Creating deepstack merger for attn_block_id {attn_block_id}. TODO: Weights need to be copied")
            self.deepstack_merger[f'{attn_block_id}'] = MergerNetwork(f"deepstack_merger_{attn_block_id}", graph, is_dynamic=True, frame_height=FIXED_HEIGHT, frame_width=FIXED_WIDTH, context_dim = context_dim, hidden_dim=hidden_dim_for_merger, is_deepstack = True)

    def forward(self, inputs, **kwargs):
        self.update_layer_names()
        latest_layer = self.patch_embed(inputs[INPUT]())
        latest_layer = self.post_patch_embed_residual(latest_layer + inputs[POST_PATCH_EMBED_RESIDUAL_INPUT](), derive_name_from_layer_name=True)
        for attn_block_id in range(self.hf_config.vision_config.depth):
            # breakpoint()
            latest_layer = self.attn_block[attn_block_id](latest_layer)
            if attn_block_id in self.hf_config.vision_config.deepstack_visual_indexes:
                print(f"Adding deepstack merger for attn_block_id {attn_block_id}")
                self.deepstack_merger[f'{attn_block_id}'](latest_layer)

        latest_layer = self.merger(latest_layer)
