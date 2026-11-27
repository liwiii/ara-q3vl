import os
from tempfile import NamedTemporaryFile

import torch

from checkpoint import Checkpoint, parse_paths


def extract(checkpoint, out):
    weight = checkpoint.read('model.language_model.embed_tokens.weight')
    expected = (checkpoint.text['vocab_size'], checkpoint.text['hidden_size'])
    if tuple(weight.shape) != expected or not weight.is_floating_point():
        raise ValueError(f'Invalid token embeddings: {weight.shape}, {weight.dtype}')
    out.mkdir(parents=True, exist_ok=True)
    dest = out / 'embed_tokens_qwen_vl_instruct.pt'
    with NamedTemporaryFile(dir=out, delete=False) as tmp:
        try:
            torch.save(weight, tmp)
            tmp.close()
            os.replace(tmp.name, dest)
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)


if __name__ == '__main__':
    args = parse_paths()
    extract(Checkpoint(args.checkpoint), args.out)
