from onnx import ModelProto

from .llm_model import Qwen3LLMModel
from .vision_model import Qwen3_VL_VisionModel

# Not a very good design. Qwen3VLModel should contain Qwen3LLMModel and Qwen3VLVisionModel
# but in our current design, call chaining is setup that way.
class Qwen3VLModel(Qwen3LLMModel):

    def __init__(self, config:dict, hf_config:dict, proto_utils:dict, onnx_model:ModelProto, caffe:object, output_dir:dict, sdk:bool):
        super().__init__(config, hf_config, proto_utils, onnx_model, caffe, output_dir['llm'], sdk)
        # From here onwards, self.output_dir is LLM-specific.
        #  Model.output_dir should also be made a dict.

        self.vision_model = Qwen3_VL_VisionModel(
            config=self.config, hf_config_json=self.hf_config, proto_utils=self.proto_utils, onnx_model=self.onnx_model, graph=self.proto_utils["DAG"](), caffe=self.caffe, output_dir=output_dir['vision'], sdk=self.sdk
        )

    def dump_assets(self):
        self.vision_model.dump_assets()  # Vision
        super().dump_assets()  # LLM


    def convert_model(self):
        self.vision_model.convert_model() # Vision
        super().convert_model()  # LLM