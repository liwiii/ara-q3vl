import os
import numpy as np
from pathlib import Path
import sys
import yaml
import re

sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
sys.path.append(str(Path(__file__).absolute().parent.parent))

from llm_compiler.model import Model
from llm_compiler.qwen2_5_vl.constants import (
    ONNX_PATCH_EMBED_ROTATION_WEIGHT,
    CHANNEL_DIM,
    IN_CHANNELS,
    INPUT,
    N,
    PIXELS_HEIGHT,
    PIXELS_WIDTH,
    FIXED_HEIGHT,
    FIXED_WIDTH,
    VISION_SIN_CACHE,
    VISION_COS_CACHE,

    QWEN2_5_VL_PREFIX,
)

from proto_utils.layers import Input
from utils import create_config, create_run_py, create_sdk_config
from llm_compiler.qwen2_5_vl.network_proto_creator import Qwen2_5_VL_VisionNetwork
from llm_compiler.qwen2_5_vl.caffemodel_creator import Qwen2_5_VL_VisionCaffemodelCreator

def dict_to_obj(d):
    from types import SimpleNamespace
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})
    return d

def detect_if_onnx_is_rotated(onnx_model):
    for initializer in onnx_model.graph.initializer:
        if ONNX_PATCH_EMBED_ROTATION_WEIGHT in initializer.name:
           return True
    return False

# Used for sorting
def natural_key(text):
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', text)]

class Qwen2_5_VL_VisionModel(Model):

    def __init__(
        self,
        config,
        hf_config_json,
        proto_utils,
        onnx_model,
        graph,
        caffe,
        output_dir,
        sdk=False
    ):
        super().__init__(
            config, hf_config_json, proto_utils, onnx_model, caffe, output_dir, sdk
        )
        self.hf_config_obj = dict_to_obj(hf_config_json)
        # Override/set the parameters which should come from vision_config
        self.hidden_dim = self.hf_config_obj.vision_config.hidden_size
        self.num_heads = self.hf_config_obj.vision_config.num_heads
        self.num_attention_heads = self.hf_config_obj.vision_config.num_heads
        self.hidden_size = self.hf_config_obj.vision_config.hidden_size
        self.head_dim = self.hf_config_obj.vision_config.head_dim if hasattr(self.hf_config_obj.vision_config, 'head_dim') else self.hidden_size // (self.num_attention_heads)

        self.spatial_merge_size = self.hf_config_obj.vision_config.spatial_merge_size
        self.num_vision_layers = self.hf_config_obj.vision_config.depth
        self.pixels_width = (
            self.hf_config_obj.vision_config.spatial_patch_size
            * FIXED_WIDTH
            * self.hf_config_obj.vision_config.spatial_merge_size
        )
        self.pixels_height = (
            self.hf_config_obj.vision_config.spatial_patch_size
            * FIXED_HEIGHT
            * self.hf_config_obj.vision_config.spatial_merge_size
        )

        self.rope_onnx_initializer_names = {'sin': VISION_SIN_CACHE, 'cos':VISION_COS_CACHE}
        self.max_position_embeddings = FIXED_HEIGHT * FIXED_WIDTH * self.hf_config_obj.vision_config.spatial_merge_size * self.hf_config_obj.vision_config.temporal_patch_size
        self.graph = graph

        input_shape = self.make_named_shape(
            [N, IN_CHANNELS, CHANNEL_DIM, PIXELS_HEIGHT, PIXELS_WIDTH],
            [1, 3, 2, self.pixels_width, self.pixels_height],
        )

        # This is not a singleton because it is different from Qwen3VL input.
        # We dont instantiate Input(args)() so that it does not clash with subclasses.
        # If this still clashes with any subclass, it can be moved in forward pass.
        self.network_input = Input(
            INPUT, self.graph, input_shape, out_group_axis=1, out_group_size=9999, is_dynamic=True, scale_as_scale_region=False
        )

        self.is_rotated = detect_if_onnx_is_rotated(self.onnx_model)

        self.network = Qwen2_5_VL_VisionNetwork(
                 "qwen2_5_vl",
                 self.graph,
                 self.hf_config_obj,
                 self.hidden_dim,
                 self.pixels_width,
                 self.pixels_height,
                 self.spatial_merge_size,
                 self.is_rotated)

    def get_weights_config(self):
        super().get_weights_config()
        return 

    def get_model_info(self):
        super().get_model_info()

    def forward_pass(self):
        latest_layer = self.network(self.network_input(layout = None))
        return latest_layer

    def get_input_index_order(self):
        """
        As per App requirement
        """
        INPUT_SCALE = INPUT + "_scale"
        ROPE_TABLE = "rope_table"
        return {INPUT: 0, INPUT_SCALE: 1, ROPE_TABLE: 2}

    def get_proto(self, save_to=None):
        # This can also be done in model.py > get_proto for all LLMs
        # Go through all layers in self.graph.get_proto
        accumulated_tops = set()
        accumulated_bottoms = set()
        network_inputs = set()
        # Find those 'top' which are not 'bottom' anywhere
        for layer in self.graph.proto_nodes:
            layer = layer.layer
            assert len(layer) == 1
            # layer[0].top is google.protobuf.internal.containers.RepeatedScalarFieldContainer
            accumulated_tops.update(layer[0].top)
            assert len(layer[0].top) > 0
            accumulated_bottoms.update(layer[0].bottom)
            if layer[0].type == "Input":
                network_inputs.update(layer[0].top)

        print(f'len(accumulated_tops): {len(accumulated_tops)}')
        print(f'len(accumulated_bottoms): {len(accumulated_bottoms)}')
        assert(len(accumulated_bottoms) < len(accumulated_tops))
        outputs_from_graph = accumulated_tops -  accumulated_bottoms
        outputs_from_graph_sorted_list = sorted(outputs_from_graph, key=natural_key)
        print(f'outputs_from_graph_sorted_list: {outputs_from_graph_sorted_list}')
        # Input order is fixed
        input_order = self.get_input_index_order()
        for _layer in self.graph.proto_nodes:
            layer = _layer.layer
            if layer[0].name in network_inputs:
                layer[0].input_index = input_order[layer[0].name]
                print(f"Assigning input index {input_order[layer[0].name]} for {layer[0].name}")
        # Mark these outputs_from_graph.
        output_index = 0
        for layer in self.graph.proto_nodes:
            layer = layer.layer
            assert len(layer) == 1
            layer_tops = set(layer[0].top)
            if layer_tops <= outputs_from_graph:  # subset
                assert len(layer[0].top) == 1
                top = layer[0].top[0]
                print(f"Outputs of graph: top: {top} = output_index {outputs_from_graph_sorted_list.index(top)}")
                layer[0].output_index = outputs_from_graph_sorted_list.index(top)
            elif layer_tops.isdisjoint(outputs_from_graph):
                # No graph output from this layer
                pass
            else:
                raise ValueError(f"Some layer.top {layer[0].top} is in outputs_from_graph {outputs_from_graph}, and some are not.")

        if not save_to:
            print(''.join(str(node) for node in self.graph.proto_nodes))
        else:
            print(f"Writing {save_to}")
            with open(save_to, "w+") as f:
                f.write('platform_target: "Ara2"\n')
                f.write('is_converted: True\n')
                f.write(''.join(str(node) for node in self.graph.proto_nodes))

    def make_caffemodel(self, proto_path:str, caffemodel_path:str):
        caffemodel_creator = Qwen2_5_VL_VisionCaffemodelCreator(
            prototxt_path=proto_path, caffemodel_path=caffemodel_path, caffe_layer_weight_prefix=QWEN2_5_VL_PREFIX,onnx_model=self.onnx_model, full_attn_layers=self.hf_config_obj.vision_config.fullatt_block_indexes,num_vision_layers=self.num_vision_layers, attention_bias=self.attention_bias, hidden_size=self.hidden_size, num_attention_heads=self.num_attention_heads, num_key_value_heads=self.num_key_value_heads, is_rotated=self.is_rotated)
        caffemodel_creator.make_caffemodel()

    def get_so_yaml(self,graph, save_to=None):
        so_yaml_dict = {}
        for node_with_shape in graph.nodes_with_shape:
            if not node_with_shape.out_group_axis:
                continue
            else:
                so_yaml_dict[node_with_shape.name] = {
                    "outputAxis": node_with_shape.out_group_axis if node_with_shape.out_group_axis <=1 else node_with_shape.out_group_axis + 1,
                    "outputGroupSize": node_with_shape.out_group_size,
                }
        if save_to:
            with open(save_to, "w") as file:
                yaml.dump({"LayerNames":so_yaml_dict}, file)
        else:
            print(so_yaml_dict)

    def convert_model(self):
        # This should create .prototxt, caffemodel, so_yaml and run.py dv_config.yaml
        # The 'vision/default_split' subdirectory is needed so that the artifacts generated
        # are at same level at artifacts for llm so that relative directory access
        # and binary invocations for compilation is streamlined.
        vision_output_dir = Path(self.output_dir) / "vision/default_split"
        os.makedirs(vision_output_dir, exist_ok=True)
        proto_out = str(vision_output_dir / "vision.prototxt")
        out_caffemodel = str(vision_output_dir / "vision.caffemodel")
        out_so = str(vision_output_dir / "vision_so.yaml")
        img_out = str(vision_output_dir / "imgs")

        # If you want to skip fwd pass, skip both of the statements below.
        # If you skip forward_pass, but call get_proto, it will generate a blank proto.
        self.forward_pass()
        self.get_proto(save_to=proto_out)

        self.make_caffemodel(proto_path=proto_out, caffemodel_path=out_caffemodel)

        self.get_so_yaml(self.graph,out_so)

        self.make_inputs(proto_out, img_out)

        if self.sdk:
            config_out = os.path.join(vision_output_dir, "dvrun_config.yaml")
            create_sdk_config(self.caffe, proto_out, out_caffemodel, img_out, out_so, config_out, self.license_key)
        else:
            config_out = os.path.join(vision_output_dir, "config_ara2.yaml")
            create_config(self.caffe, proto_out, out_caffemodel, img_out, out_so, config_out )
            create_run_py(vision_output_dir)
    
    def dump_assets(self):
        rope_table, rope_table_quantized = self.dump_rope_table()
        assets_dir =  os.path.join(self.output_dir,"assets")
        os.makedirs(assets_dir, exist_ok=True)

        # This will go into analyser. So we need to present it as float32
        # This is not the case for LLMs because for LLMs it directly goes to MCP.
        # Vision dvm design and flow is like a cnn, there is no separate MCP. 
        # So this is just another input to the model, not treates specially like ROPE for LLMs
        # Customer will be using the .dat file which will be output from analyzer.
        # Now, to analyzer we have to convey that the 32 bits are actually int8.
        # Please bear with the complexity because for regular CNNs there is no concept of ROPE.
        rope_table_quantized.astype(dtype=np.float32).tofile(os.path.join(assets_dir, "vision_rope_table.bin"))
        rope_table_quantized.astype(dtype=np.int8).tofile(os.path.join(assets_dir, "vision_rope_table_int8.bin"))
        rope_table.tofile(os.path.join(assets_dir, "vision_rope_table_float.bin"))
