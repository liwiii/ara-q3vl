import os
from tempfile import NamedTemporaryFile


def write_weights(net, checkpoint, dest):
    assigned, consumed = set(), set()

    def load(name):
        key = f'model.visual.{name}'
        consumed.add(key)
        return checkpoint.read(key)

    def assign(layer, values):
        layer = 'qwen3_vl_' + layer
        params = net.params[layer]
        if len(params) != len(values):
            raise ValueError(f'{layer}: expected {len(params)} parameter blobs')
        for param, tensor in zip(params, values):
            target = param.data
            tensor = tensor.float().numpy()
            # Flatten a (C,T,H,W) patch kernel into the 1x1 input channel axis.
            # All other linear kernels keep the original [out, in] order.
            if tensor.shape[0] != target.shape[0] or tensor.size != target.size:
                raise ValueError(f'{layer}: cannot map {tensor.shape} to {target.shape}')
            target[...] = tensor.reshape(target.shape)
        assigned.add(layer)

    def copy_pair(source, layer):
        assign(layer, (load(source + '.weight'), load(source + '.bias')))

    copy_pair('patch_embed.proj', 'patch_embed_artificial_conv1d')
    for i in range(checkpoint.vision['depth']):
        src = f'blocks.{i}.'
        dst = f'attn_block_{i}_'
        copy_pair(src + 'norm1', dst + 'lnorm1')
        copy_pair(src + 'norm2', dst + 'lnorm2')
        qkv = load(src + 'attn.qkv.weight').chunk(3, dim=0)
        biases = load(src + 'attn.qkv.bias').chunk(3, dim=0)
        for projection, weight, bias in zip('qkv', qkv, biases):
            assign(dst + f'full_attn_{projection}_proj', (weight, bias))
        copy_pair(src + 'attn.proj', dst + 'full_attn_o_proj')
        copy_pair(src + 'mlp.linear_fc1', dst + 'mlp_fc1')
        copy_pair(src + 'mlp.linear_fc2', dst + 'mlp_fc2')

    mergers = [('merger', 'merger')]
    for i, layer in enumerate(checkpoint.vision['deepstack_visual_indexes']):
        mergers.append((f'deepstack_merger_list.{i}', f'deepstack_merger_{layer}'))
    for src, dst in mergers:
        copy_pair(src + '.norm', dst + '_layernorm')
        copy_pair(src + '.linear_fc1', dst + '_mlp_fc1')
        copy_pair(src + '.linear_fc2', dst + '_mlp_fc2')

    all_visual = {key for key in checkpoint.files if key.startswith('model.visual.')}
    # The learned position embedding is materialized as a separate SDK input.
    unused = all_visual - consumed - {'model.visual.pos_embed.weight'}
    missing = set(net.params) - assigned
    if unused or missing:
        raise ValueError(f'Unmapped checkpoint tensors: {sorted(unused)}; Caffe layers: {sorted(missing)}')
    with NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        try:
            tmp.close()
            net.save(tmp.name)
            os.replace(tmp.name, dest)
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)
