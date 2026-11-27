"""Build the Ara language splits from external ONNX tensors, one layer at a time."""
import argparse
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('GLOG_minloglevel', '2')

import numpy as np
import onnx
import yaml
from llm_compiler.caffe_loader import CaffeLoader
from llm_compiler.constants import EMBEDDINGS, LMHEAD_WEIGHT, LMHEAD_RMS, INPUT_RMS, LMHEAD
from llm_compiler.utils import quantize, get_scale_tensor_float
from llm_compiler.qwen3_vl.llm_model import Qwen3LLMModel


class Weight:
    def __init__(self, tensor, directory):
        self.name = tensor.name
        self.dims = tuple(tensor.dims)
        self.data_type = tensor.data_type
        self.tensor = tensor
        self.directory = directory

    @property
    def raw_data(self):
        external = {item.key: item.value for item in self.tensor.external_data}
        if not external:
            return self.tensor.raw_data
        return np.memmap(self.directory / external['location'], mode='r', dtype=np.uint8,
                         offset=int(external.get('offset', 0)), shape=(int(external['length']),))

    def array(self):
        return np.frombuffer(self.raw_data, dtype='<f4').reshape(self.dims)


class LanguageSplits(Qwen3LLMModel):
    def dump_embeddings_table(self):
        # dump_assets() also writes the unquantized embedding table, so use a
        # mapped view and quantize groups of 64 in bounded chunks.
        params = {w.name: w for w in self.onnx_model.graph.initializer}
        emb = params[EMBEDDINGS].array() if EMBEDDINGS in params else params[LMHEAD_WEIGHT].array().T
        quant = np.empty(emb.shape, dtype=np.int8)
        scale = np.empty((emb.shape[0], emb.shape[1] // 64), dtype=np.int8)
        for i in range(0, len(emb), 512):
            rows = slice(i, i + 512)
            q, s = quantize(emb[rows])
            quant[rows] = q.reshape(quant[rows].shape)
            scale[rows] = s.reshape(scale[rows].shape)
        return emb, quant, scale

    def dump_rope_table(self):
        initializers = {w.name: w for w in self.onnx_model.graph.initializer}
        # MULTI_ROPE stores two text tables followed by the video tables.
        tables = 2 + self.config['num_video_ropes'] if self.vl_rope_type == 'MULTI_ROPE' else 1
        parts = []
        for name in ('cos_cache', 'sin_cache'):
            half = initializers[name].array().reshape(tables, -1, self.head_dim // 2)
            if half.shape[1] < self.max_position_embeddings:
                raise ValueError(f'{name} contains only {half.shape[1]} positions per table')
            half = half[:, :self.max_position_embeddings, :]
            parts.extend((half, half))
        rope = np.concatenate(parts, axis=-1)
        return rope, np.clip(rope * 128, -128, 127).astype(np.int8)

    def make_inputs(self, proto, img_out_dir):
        # Keys and values use different group axes. Read the declared scale
        # shapes so the SDK receives exponents for the correct 64-value groups.
        net = self.caffe.Net(proto, self.caffe.TEST)
        dest = Path(img_out_dir) / 'chksum0'
        dest.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        for name in net.inputs:
            if name.endswith('_scale'):
                continue
            shape = net.blobs[name].data.shape
            if name == 'valid_till':
                values = np.full(shape, self.token_size - 1, dtype=np.float32)
            else:
                values = rng.random(shape, dtype=np.float32) * 2 - 1
            values.tofile(dest / (name + '.bin'))
            scale = name + '_scale'
            if scale in net.inputs:
                scale_shape = net.blobs[scale].data.shape
                axis, = (i for i in range(len(shape)) if shape[i] != scale_shape[i])
                group = shape[axis] // scale_shape[axis]
                get_scale_tensor_float(values, shape, scale_shape, group, axis).tofile(dest / (scale + '.bin'))

    def make_lmhead_caffemodel(self, proto, out):
        params = {w.name: w for w in self.onnx_model.graph.initializer}
        net = self.caffe.Net(proto, self.caffe.TEST)
        head = net.params[LMHEAD][0].data.reshape(self.vocab_size, self.hidden_size)
        head[:] = params[LMHEAD_WEIGHT].array().T
        norm, = (w for w in params.values() if LMHEAD_RMS in w.name)
        net.params[INPUT_RMS][0].data.flat = norm.array()
        net.save_hdf5(out)


def split(config):
    directory = Path(config['onnx_model_repo'])
    if len(config['prmpt_prc_sizes']) != 1:
        raise ValueError('The Ara compiler supports a single prompt batch size')
    graph = onnx.load(directory / 'model.onnx', load_external_data=False).graph
    # Old combined ONNX exports may include unrelated visual tensors with
    # matching projection suffixes. Only pass the language weights to the SDK.
    weights = [Weight(t, directory) for t in graph.initializer
               if t.name.startswith('model.layers.') or t.name in
               (EMBEDDINGS, LMHEAD_WEIGHT, 'cos_cache', 'sin_cache')]
    model = SimpleNamespace(graph=SimpleNamespace(initializer=weights))
    hf = json.loads((Path(config['model_repo']) / 'config.json').read_text())
    out = Path(config['out']) / 'llm'
    out.mkdir(parents=True, exist_ok=True)
    loader = CaffeLoader('/')
    engine = LanguageSplits(config, hf, loader.get_proto_utils(), model, loader.get_caffe(), str(out), sdk=True)
    engine.dump_assets()
    engine.convert_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=Path)
    config = yaml.safe_load(parser.parse_args().config.read_text())
    lock = os.open(config['onnx_model_repo'], os.O_RDONLY)
    try:
        fcntl.flock(lock, fcntl.LOCK_SH)
        split(config)
    finally:
        os.close(lock)
