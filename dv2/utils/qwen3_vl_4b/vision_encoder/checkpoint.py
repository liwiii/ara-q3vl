import argparse
import json
from pathlib import Path

from safetensors import safe_open


def parse_paths():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, type=Path, dest='checkpoint')
    parser.add_argument('-o', required=True, type=Path, dest='out')
    return parser.parse_args()


class Checkpoint:
    def __init__(self, directory):
        self.directory = directory
        config = json.loads((directory / 'config.json').read_text())
        self.vision = config['vision_config']
        self.text = config['text_config']
        expected = dict(depth=24, hidden_size=1024, num_heads=16, intermediate_size=4096,
                        patch_size=16, temporal_patch_size=2, spatial_merge_size=2,
                        in_channels=3, out_hidden_size=2560, deepstack_visual_indexes=[5, 11, 17],
                        num_position_embeddings=2304, hidden_act='gelu_pytorch_tanh')
        if (config.get('model_type') != 'qwen3_vl' or self.text['hidden_size'] != 2560
                or self.text['num_hidden_layers'] != 36
                or any(self.vision.get(k) != v for k, v in expected.items())):
            raise ValueError(f'{directory} is not an original Qwen3-VL-4B checkpoint')

        index = directory / 'model.safetensors.index.json'
        if index.exists():
            self.files = json.loads(index.read_text())['weight_map']
        else:
            name = 'model.safetensors'
            with safe_open(directory / name, framework='pt', device='cpu') as tensors:
                self.files = dict.fromkeys(tensors.keys(), name)

    def read(self, key):
        with safe_open(self.directory / self.files[key], framework='pt', device='cpu') as tensors:
            return tensors.get_tensor(key)
