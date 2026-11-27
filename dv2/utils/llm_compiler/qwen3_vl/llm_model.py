import sys
from onnx import ModelProto
from pathlib import Path

sys.path.append(str(Path(__file__).absolute().parent.parent.parent))
sys.path.append(str(Path(__file__).absolute().parent.parent))

from llm_compiler.model import Qwen2Model
from llm_compiler.constants import DIM0, DIM1, TOKEN, HIDDEN

from .constants import *

class Qwen3LLMModel(Qwen2Model):
    def __init__(self, config:dict, hf_config:dict, proto_utils:dict, onnx_model:ModelProto, caffe, output_dir:str, sdk:bool):
        super().__init__(config, hf_config, proto_utils, onnx_model, caffe, output_dir, sdk)
        self.use_qk_norm = True

        # Active only for VLM-LLM. Empty for pure LLM model.
        self.num_deepstack_inputs = len(hf_config.get('vision_config', {}).get('deepstack_visual_indexes', {}))

    def make_decoder_prompt_graph(self, layer_idx:int):
        graph = self.proto_utils['DAG']()
        attention_outs = self.make_attention_prompt(graph, layer_idx = layer_idx)
        mlp_ins = self.make_attention_mlp_connection_prompt(graph,attention_outs)
        default_decoder_out = self.make_mlp_prompt(graph,mlp_ins)
        if (layer_idx < self.num_deepstack_inputs) and (self.architecture == "Qwen3VLForConditionalGeneration"):
            # Only for VL-LLM, not for pure LLM
            print(f"Adding deepstack eltwise in layer {layer_idx}")
            deepstack_input_shape = self.make_named_shape([DIM0, DIM1, TOKEN, HIDDEN],[1,1,self.token_size,self.hidden_size])
            # Call is required to make it NodeWithShape[0] and NodeWithShape[1]-scale
            deepstack_input_ids = self.proto_utils['Input'](DEEPSTACK_INPUT, graph, deepstack_input_shape,-1,64, True, input_type='RQ')()
            deepstack_eltwise = self.proto_utils['Eltwise'](DEEPSTACK_RESIDUAL, graph, -1, 64,  self.proto_utils['EltwiseOperation'].SUM, True)
            deepstack_eltwise(deepstack_input_ids + default_decoder_out)
        return graph
