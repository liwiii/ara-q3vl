import caffe
import yaml
from llm_compiler.utils import create_sdk_config

from checkpoint import Checkpoint, parse_paths
from get_caffemodel import write_weights
from vision_model_creator import build_graph
from vision_inputs import write_inputs


def convert(model, out):
    checkpoint = Checkpoint(model)
    assets = out / 'vision_encoder'
    assets.mkdir(parents=True, exist_ok=True)
    proto = assets / 'qwen3_vl_4b.prototxt'
    so = assets / 'qwen3_vl_4b_so.yaml'
    weights = out / 'vision.caffemodel'
    images = assets / 'inputs'
    inputs, outputs = build_graph(checkpoint.vision, proto, so)
    net = caffe.Net(str(proto), caffe.TEST)
    if list(net.inputs) != inputs or set(net.outputs) != set(outputs):
        raise ValueError(f'Unexpected Caffe vision interface: {net.inputs}, {net.outputs}')
    write_inputs(net, checkpoint, images / 'input0')
    write_weights(net, checkpoint, weights)
    del net
    config = out / 'dvrun_config.yaml'
    create_sdk_config(caffe, str(proto), str(weights), str(images), str(so), str(config))
    data = yaml.safe_load(config.read_text())
    data['model_name'] = 'qwen3_vl_4b_vision'
    config.write_text(yaml.safe_dump(data))


if __name__ == '__main__':
    args = parse_paths()
    convert(args.checkpoint, args.out.resolve())
