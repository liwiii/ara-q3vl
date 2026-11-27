import sys
from pathlib import Path
from types import SimpleNamespace

# The SDK attention blocks import vision_encoder.layer_names by package name.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from proto_utils import Network
from proto_utils.functional import permute
from proto_utils.graph import DAG, NamedDim
from proto_utils.layers import Input
from proto_utils.layers.eltwise import Eltwise, Operation
from llm_compiler.qwen2_5_vl.constants import FIXED_HEIGHT, FIXED_WIDTH
from llm_compiler.qwen2_5_vl.modules.full_attn import FullAttnNetwork
from llm_compiler.qwen3_vl.modules.attn_block import AttnBlockNetwork
from llm_compiler.qwen3_vl.modules.merger import MergerNetwork
from llm_compiler.qwen3_vl.modules.patch_embed import PatchEmbedNetwork
from llm_compiler.qwen3_vl.constants import POST_PATCH_EMBED_RESIDUAL_INPUT

PIXELS = 'vision_input'


def named_shape(**dimensions):
    return [NamedDim(name, size) for name, size in dimensions.items()]


class VisionEncoder(Network):
    def __init__(self, graph, vision, rope):
        super().__init__('qwen3_vl', graph)
        hidden = vision.hidden_size
        merge = vision.spatial_merge_size
        channels = vision.in_channels * vision.temporal_patch_size * vision.patch_size ** 2
        self.patch = PatchEmbedNetwork('patch_embed', graph, hidden, FIXED_WIDTH * merge,
                                       FIXED_HEIGHT * merge, vision.patch_size, merge, channels, False)
        self.add_position = Eltwise('post_patch_embed_residual', graph, -1, 64,
                                    is_dynamic=True, operation=Operation.SUM)
        self.blocks = []
        for i in range(vision.depth):
            attn = FullAttnNetwork('full_attn', graph, rope, hidden, vision.num_heads, hidden)
            block = AttnBlockNetwork('attn_block', graph, attn, i, SimpleNamespace(vision_config=vision))
            block.lnorm1.uses_mean = block.lnorm2.uses_mean = True
            self.blocks.append(block)

        def merger(name, postshuffle):
            module = MergerNetwork(name, graph, True, FIXED_HEIGHT, FIXED_WIDTH, hidden,
                                   vision.out_hidden_size, is_deepstack=postshuffle)
            module.layernorm.uses_mean = True
            module.layernorm.eps = 1e-6
            return module

        self.deepstack = {i: merger(f'deepstack_merger_{i}', True) for i in vision.deepstack_visual_indexes}
        self.merger = merger('merger', False)
        self.update_layer_names()

    def forward(self, pixels, position):
        hidden = self.add_position(self.patch(pixels) + position)
        outputs = []
        for i, block in enumerate(self.blocks):
            hidden = block(hidden)
            if i in self.deepstack:
                outputs.extend(self.deepstack[i](hidden))
        outputs.extend(self.merger(hidden))
        return outputs


def build_graph(config, proto, so_yaml):
    vision = SimpleNamespace(**config)
    graph = DAG()
    tokens = FIXED_HEIGHT * FIXED_WIDTH * vision.spatial_merge_size ** 2
    head = vision.hidden_size // vision.num_heads
    rope = Input('rope_table', graph, named_shape(batch=1, unsqueeze=1, max_tokens=tokens,
                 sin_cos=2 * head), -1, 2 * head)()
    rope = permute(graph, rope, ('max_tokens', 'sin_cos'))
    flat = vision.in_channels * vision.temporal_patch_size * vision.patch_size ** 2
    pixels = Input(PIXELS, graph, named_shape(n=1, c=1, tkn=tokens, artificial_large_channel=flat),
                   -1, 64, is_dynamic=True, scale_as_scale_region=False)()
    position = Input(POST_PATCH_EMBED_RESIDUAL_INPUT, graph,
                     named_shape(n=1, unsqueeze=1, tkn=tokens, hidden=vision.hidden_size),
                     -1, 64, is_dynamic=True, scale_as_scale_region=False)()
    outputs = VisionEncoder(graph, vision, rope)(pixels, position=position)
    # SDK ABI: rope, patch values/scales, positional residual values/scales.
    # Each DeepStack pair precedes the final merger pair, in block order.
    input_names = ['rope_table'] + [n.name for n in pixels + position]
    output_names = [n.name for n in outputs]
    for node in graph.proto_nodes:
        layer = node.layer[0]
        if layer.name in input_names:
            layer.input_index = input_names.index(layer.name)
        if layer.name in output_names:
            layer.output_index = output_names.index(layer.name)
    with proto.open('w') as f:
        f.write('platform_target: "Ara2"\nis_converted: true\n')
        for node in graph.proto_nodes:
            f.write(str(node))
    graph.get_so_yaml(str(so_yaml))
    return input_names, output_names
