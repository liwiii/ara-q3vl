import sys
from onnx import ModelProto
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
sys.path.append(str(Path(__file__).absolute().parent.parent))
sys.path.append(str(Path(__file__).absolute().parent.parent / 'qwen2_5_vl'))

from qwen2_5_vl.model import Qwen2_5_VL_VisionModel
from qwen2_5_vl.constants import *
from proto_utils.layers import *

from llm_compiler.qwen3_vl.network_proto_creator import Qwen3_VL_VisionNetwork
from llm_compiler.qwen3_vl.caffemodel_creator import Qwen3_VL_VisionCaffemodelCreator

from .constants import *

class Qwen3_VL_VisionModel(Qwen2_5_VL_VisionModel):
    def __init__(
        self,
        config:dict,
        hf_config_json:dict,
        proto_utils:dict,
        onnx_model:ModelProto,
        graph:object,
        caffe:object,
        output_dir:str,
        sdk=False
    ):
        vision_config = hf_config_json['vision_config']
        hf_config_json['vision_config']['spatial_patch_size'] = hf_config_json['vision_config']['patch_size']
        hf_config_json['vision_config']['fullatt_block_indexes'] = list(range(hf_config_json['vision_config']['depth']))
        super().__init__(
            config, hf_config_json, proto_utils, onnx_model, graph, caffe, output_dir, sdk
        )

        self.pixels_width = (
            FIXED_WIDTH * self.hf_config_obj.vision_config.spatial_merge_size
        )
        self.pixels_height = (
            FIXED_HEIGHT * self.hf_config_obj.vision_config.spatial_merge_size
        )
        self.artificial_large_channel_num = vision_config['in_channels'] * vision_config['patch_size']  * vision_config['patch_size'] * vision_config['temporal_patch_size']
        input_shape = self.make_named_shape(
            [N, C, TKN, ARTIFICIAL_LARGE_CHANNEL],
            [1, 1, self.pixels_height * self.pixels_width, self.artificial_large_channel_num]
        )
        post_patch_embed_residual_input_shape = self.make_named_shape(
            [N, UNSQUEEZE, TKN, HIDDEN],
            [1, 1, self.pixels_height * self.pixels_width, self.hidden_dim]
        )

        self.network_input = {}  # Dict in Qwen3VL.

        # Inputs for vision should be dynamic but scale should not be marked as scale region.
        # We cannot make it a singleton because the out_group_axis and out_group_size is different from Qwen2.5VL.
        self.network_input[INPUT] = Input(
            INPUT, self.graph, input_shape, out_group_axis=-1, out_group_size=64, is_dynamic=True, scale_as_scale_region=False
        )

        # Qwen3VL specific
        self.network_input[POST_PATCH_EMBED_RESIDUAL_INPUT] =  Input(
            POST_PATCH_EMBED_RESIDUAL_INPUT, self.graph, post_patch_embed_residual_input_shape, out_group_axis=-1, out_group_size=64, is_dynamic=True, scale_as_scale_region=False
            # Shouldb be dynamic but scale should not be marked as region 
        )

        # Then it is initiated below as parent class of Qwen3_VL_VisionNetwork.
        # Maybe with depedency injection, the objects can be created just once.
        self.network = Qwen3_VL_VisionNetwork(  # This is second initiation
                 QWEN3_VL_PREFIX,
                 self.graph,
                 self.hf_config_obj,
                 self.hidden_dim,
                 self.pixels_width,
                 self.pixels_height,
                 self.spatial_merge_size,
                 self.artificial_large_channel_num,
                 self.is_rotated)

    def make_caffemodel(self, proto_path:str, caffemodel_path:str):
        print(f'Going to make Qwen3_VL_VisionCaffemodelCreator with self.attention_bias : {self.attention_bias}')
        caffemodel_creator = Qwen3_VL_VisionCaffemodelCreator(
            prototxt_path=proto_path, caffemodel_path=caffemodel_path, caffe_layer_weight_prefix=QWEN3_VL_PREFIX, onnx_model=self.onnx_model, full_attn_layers=self.hf_config_obj.vision_config.fullatt_block_indexes,num_vision_layers=self.num_vision_layers, attention_bias=self.attention_bias, hidden_size=self.hidden_size, num_attention_heads=self.num_attention_heads, num_key_value_heads=self.num_key_value_heads, is_rotated=self.is_rotated, deepstack_layers_list=self.hf_config['vision_config']['deepstack_visual_indexes'])
        caffemodel_creator.make_caffemodel()

    def forward_pass(self):
        latest_layer = self.network(self.network_input)
        return latest_layer
    
    def get_input_index_order(self):
        """
        As per App requirement
        """
        INPUT_SCALE = INPUT + "_scale"
        POST_PATCH_EMBED_RESIDUAL_INPUT_SCALE = POST_PATCH_EMBED_RESIDUAL_INPUT + "_scale"
        ROPE_TABLE = "rope_table"
        return {ROPE_TABLE: 0, INPUT: 1, INPUT_SCALE: 2, POST_PATCH_EMBED_RESIDUAL_INPUT: 3, POST_PATCH_EMBED_RESIDUAL_INPUT_SCALE: 4}
