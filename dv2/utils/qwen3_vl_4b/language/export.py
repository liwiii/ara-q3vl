"""Export the original dense checkpoint for Ara's Qwen3 language compiler."""
import argparse
import fcntl
import os
import tempfile
from pathlib import Path

import onnx
import onnx_ir as ir
import torch
import yaml
from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextRotaryEmbedding

from ort_genai.May2026.models.builders.qwen import Qwen3VLTextModel
from PIL import Image
from qwen3_vl_4b.vision_encoder.checkpoint import Checkpoint


class LanguageExporter(Qwen3VLTextModel):
    def __init__(self, config, weights, data):
        self.checkpoint = weights
        self.config = config
        self.data = data
        text = config.text_config
        text.architectures = config.architectures
        super().__init__(text, ir.DataType.FLOAT, ir.DataType.FLOAT, 'cpu',
                         str(Path(data.name).parent), {'exclude_embeds': True})
        self.num_deepstack = len(config.vision_config.deepstack_visual_indexes)

    def make_initializer(self, tensor, name, to=None):
        # Keep the SDK's small constants in the graph. Stream large FP32 weights
        # in row blocks, including the transposed tied output head.
        if tensor.numel() < 8192:
            return super().make_initializer(tensor, name, to=to)
        offset = self.data.tell()
        rows = max(1, 2097152 // tensor[0].numel())
        for chunk in tensor.detach().split(rows):
            self.data.write(chunk.float().contiguous().numpy())
        shape = ir.Shape(tensor.shape)
        value = self.make_value(name, ir.DataType.FLOAT, shape)
        path = Path(self.data.name)
        value.const_value = ir.ExternalTensor(path.name, offset, self.data.tell() - offset,
                                              ir.DataType.FLOAT, shape=shape, name=name, base_dir=path.parent)
        self.model.graph.register_initializer(value)

    def load_weights(self, input_path):
        with torch.device('meta'):
            model = Qwen3VLForConditionalGeneration(self.config)
        del model.model.visual
        text = model.model.language_model
        text.load_state_dict({k: self.checkpoint.read('model.language_model.' + k)
                              for k in text.state_dict()}, assign=True)
        model.lm_head.weight = text.embed_tokens.weight
        return model

    def make_node(self, op_type, inputs, outputs, *, name, domain='', **attrs):
        if op_type == 'GroupQueryAttention':
            # ORT 1.22 in aracompiler:0.2 predates the optional attention-sinks input.
            while inputs and inputs[-1] == '':
                inputs = inputs[:-1]
        if op_type != 'SimplifiedLayerNormalization' or domain:
            return super().make_node(op_type, inputs, outputs, name=name, domain=domain, **attrs)
        # The SDK uses an ORT-only RMSNorm in the default ONNX domain. Emit
        # equivalent opset-21 FP32 arithmetic for the unfused per-head norms.
        x, weight = inputs
        eps = name + '/epsilon'
        self.make_initializer(torch.tensor(attrs['epsilon']), eps)
        for op, args, result in (
            ('Mul', [x, x], 'square'),
            ('ReduceMean', [name + '/square', '/model/constants/INT64/[-1]'], 'mean'),
            ('Add', [name + '/mean', eps], 'variance'),
            ('Sqrt', [name + '/variance'], 'rms'),
            ('Div', [x, name + '/rms'], 'normalized'),
        ):
            value = name + '/' + result
            super().make_node(op, args, [value], name=value)
        super().make_node('Mul', [name + '/normalized', weight], outputs, name=name)

    def make_layer(self, layer_id, layer):
        super().make_layer(layer_id, layer)
        if layer_id < self.num_deepstack:
            shape = ['batch_size', 'sequence_length', self.hidden_size]
            residual = f'deepstack_{layer_id}'
            self.model.graph.inputs.append(self.make_value(residual, self.io_dtype, shape))
            # The SDK fuses the residual sum with the following layer's RMSNorm.
            name = f'/model/layers.{layer_id}/deepstack/Add'
            self.make_add(name, [self.layernorm_attrs['skip_input'], residual], self.io_dtype, shape)
            self.layernorm_attrs['skip_input'] = name + '/output_0'


CONTEXT = 4096
GRID = 12  # 384 pixels / 16-pixel patch / 2x2 spatial merge
VIDEO_PREFIXES = range(128, CONTEXT - 128 - GRID ** 2, 128)


def write_rope(builder, model_path, kind):
    config = builder.config
    with torch.device('meta'):
        model = Qwen3VLForConditionalGeneration(config).model
    rotary = Qwen3VLTextRotaryEmbedding(config.text_config)
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    pixels = GRID * config.vision_config.spatial_merge_size * config.vision_config.patch_size

    def sample_positions(modality, temporal_groups=1):
        messages = [{'role': 'user', 'content': [{'type': modality}, {'type': 'text', 'text': 'Describe.'}]}]
        if modality == 'image':
            media = {'images': [Image.new('RGB', (pixels, pixels))]}
        else:
            # Ara's fixed-video template reserves six system text tokens.
            messages.insert(0, {'role': 'system', 'content': [{'type': 'text', 'text': '😁' * 6}]})
            frames = temporal_groups * config.vision_config.temporal_patch_size
            media = dict(videos=[torch.zeros(frames, 3, pixels, pixels, dtype=torch.uint8)],
                         videos_kwargs=dict(do_sample_frames=False, do_resize=False,
                                            video_metadata=[dict(total_num_frames=frames, fps=2,
                                                                 frames_indices=list(range(frames)))]))
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        sample = processor(text=[prompt], return_tensors='pt', **media)
        grid_name = modality + '_grid_thw'
        ids = sample.input_ids
        if ids.shape[1] < CONTEXT:
            ids = torch.nn.functional.pad(ids, (0, CONTEXT - ids.shape[1]), value=0)
        positions, _ = model.get_rope_index(ids, **{grid_name: sample[grid_name]})
        return ids, positions

    if kind == 'MULTI_ROPE':
        # SDK table order: text twice, then a visual-span table per aligned prefix.
        text_positions = torch.arange(CONTEXT).expand(3, 1, -1)
        tables = [text_positions, text_positions]
        groups = (CONTEXT - VIDEO_PREFIXES.start - 6) // GRID ** 2
        ids, positions = sample_positions('video', groups)
        first = (ids[0] == config.video_token_id).nonzero()[0].item()
        for prefix in VIDEO_PREFIXES:
            count = ((CONTEXT - prefix - 6) // GRID ** 2) * GRID ** 2
            # Preserve real timestamp tokens between frames. Absolute text prefix
            # length shifts all three MRoPE axes by the same number of positions.
            span = positions[:, :, first:first + count] + prefix - first
            tables.append(torch.nn.functional.pad(span, (0, CONTEXT - count)))
    else:
        _, positions = sample_positions(kind.lower(), temporal_groups=8)
        tables = [positions[:, :, :CONTEXT]]

    cos, sin = rotary(torch.empty(1), torch.cat(tables, dim=-1))
    for name, table in [('cos_cache', cos), ('sin_cache', sin)]:
        builder.make_initializer(table[0, :, :config.text_config.head_dim // 2], name)


def export(model, out, kind, joint, specd):
    checkpoint = Checkpoint(model)
    config = Qwen3VLConfig.from_pretrained(model, local_files_only=True)
    n = config.text_config.num_hidden_layers
    run = out / 'kinara_llm'
    weights = {p: {'INT4': list(range(n))} for p in
               ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'gate_proj', 'down_proj')}
    weights['lm_head'] = {'INT4': [n]}
    settings = dict(model_repo=str(model), original_model_repo=str(model), onnx_model_repo=str(out),
                    out=str(run), is_vlm=True, unique_params=1, rope={'vl_rope_type': kind},
                    tie_word_embeddings=True, weight_config=weights, use_8bit_prompt=False,
                    prmpt_context_size=4096, prmpt_prc_sizes=[128], tkn_gen_sizes=[1024, 2048, 3072, 4096],
                    specd_size=specd, num_mcs_shards=1, attention_scale_layers=[], qxk_out_scale_32_layers=[])
    if joint:
        settings['vision_out'] = str(run / 'vision')
    if kind == 'MULTI_ROPE':
        settings.update(num_video_ropes=len(VIDEO_PREFIXES), img_tkn_id=config.image_token_id, video_tkn_id=config.video_token_id)
    with tempfile.TemporaryDirectory(dir=out) as tmp, tempfile.NamedTemporaryFile(
            dir=out, prefix='language-', suffix='.data', delete=False) as data:
        staging = Path(tmp)
        try:
            builder = LanguageExporter(config, checkpoint, data)
            write_rope(builder, model, kind)
            builder.make_model(str(model))
            data.flush()
            builder.model.graph.sort()
            builder.model.metadata_props['ara.external_weights'] = Path(data.name).name
            ir.save(builder.model, staging / 'model.onnx')
            # Validate the graph beside its external data before publishing it.
            os.link(data.name, staging / Path(data.name).name)
            onnx.checker.check_model(str(staging / 'model.onnx'), check_custom_domain=False)
            (staging / 'kinara_run_config.yaml').write_text(yaml.safe_dump(settings))
        except BaseException:
            Path(data.name).unlink(missing_ok=True)
            raise
        # Serialize publication with other exporters and consumers of this graph.
        # Retire only weights named by the previous export in this directory.
        lock = os.open(out, os.O_RDONLY)
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            graph_path = out / 'model.onnx'
            old_data = None
            if graph_path.is_file():
                old_graph = onnx.load(graph_path, load_external_data=False)
                old_data = next((p.value for p in old_graph.metadata_props
                                 if p.key == 'ara.external_weights'), None)
            (staging / 'model.onnx').replace(graph_path)
            (staging / 'kinara_run_config.yaml').replace(out / 'kinara_run_config.yaml')
            if (old_data and Path(old_data).name == old_data and old_data.startswith('language-')
                    and old_data.endswith('.data') and old_data != Path(data.name).name):
                (out / old_data).unlink(missing_ok=True)
        finally:
            os.close(lock)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('rope', choices=['IMAGE', 'VIDEO', 'MULTI_ROPE'])
    parser.add_argument('specd', type=int, choices=[1, 4])
    parser.add_argument('--joint', action='store_true')
    args = parser.parse_args()
    with torch.no_grad():
        export(args.model, args.out, args.rope, args.joint, args.specd)
