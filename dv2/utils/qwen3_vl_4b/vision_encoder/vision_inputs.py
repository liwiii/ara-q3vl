from math import isqrt

import torch
from llm_compiler.qwen2_5_vl.constants import FIXED_HEIGHT, FIXED_WIDTH
from llm_compiler.qwen3_vl.constants import POST_PATCH_EMBED_RESIDUAL_INPUT
from vision_model_creator import PIXELS


def position_inputs(checkpoint):
    v = checkpoint.vision
    merge = v['spatial_merge_size']
    h, w = FIXED_HEIGHT * merge, FIXED_WIDTH * merge
    head = v['hidden_size'] // v['num_heads']
    # Hugging Face enumerates patches within each merge x merge tile first.
    def patch_order(grid):
        return grid.reshape(h // merge, merge, w // merge, merge, -1).permute(
            0, 2, 1, 3, 4).reshape(h * w, -1)

    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    coords = patch_order(torch.stack((rows, cols), dim=-1))
    inv_freq = 1.0 / (10000 ** (torch.arange(0, head // 2, 2).float() / (head // 2)))
    angle = (coords.unsqueeze(-1) * inv_freq).flatten(1).repeat(1, 2)
    # Mode 5 calibrates the floating-point graph before quantizing the table.
    # The analyzer therefore needs cos/sin in [-1, 1].
    rope = torch.cat((angle.cos(), angle.sin()), dim=-1)

    table = checkpoint.read('model.visual.pos_embed.weight')
    side = isqrt(v['num_position_embeddings'])
    y = torch.linspace(0, side - 1, h)
    x = torch.linspace(0, side - 1, w)
    lo_y, lo_x = y.long(), x.long()
    hi_y, hi_x = (lo_y + 1).clamp(max=side - 1), (lo_x + 1).clamp(max=side - 1)
    dy, dx = y - lo_y, x - lo_x
    # Keep interpolation weights and sums in checkpoint precision, as in Transformers.
    terms = []
    for row, wy in ((lo_y, 1 - dy), (hi_y, dy)):
        for col, wx in ((lo_x, 1 - dx), (hi_x, dx)):
            values = table[row[:, None] * side + col]
            terms.append(values * (wy[:, None] * wx).to(table.dtype).unsqueeze(-1))
    return rope, patch_order(sum(terms)).float()


def write_inputs(net, checkpoint, directory):
    directory.mkdir(parents=True, exist_ok=True)
    rope, position = position_inputs(checkpoint)
    pixels = torch.rand(net.blobs[PIXELS].data.shape, generator=torch.Generator().manual_seed(0)) * 2 - 1
    inputs = {'rope_table': rope, PIXELS: pixels, POST_PATCH_EMBED_RESIDUAL_INPUT: position}
    for name in (PIXELS, POST_PATCH_EMBED_RESIDUAL_INPUT):
        values = inputs[name].reshape(-1, 64)
        peak = values.abs().amax(dim=-1)
        # Same base-2 exponent convention as SDK get_scale_tensor_float.
        inputs[name + '_scale'] = torch.where(peak > 0, 6 - peak.log2().floor(), 0)
    for name in net.inputs:
        inputs[name].reshape(net.blobs[name].data.shape).float().numpy().tofile(directory / (name + '.bin'))
