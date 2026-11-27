'''
Copyright (c) 2025, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

import unittest
from layers import Conv, Input, LayerNorm, SoftMax, Rope, BatchDot, Eltwise, Swish, Gelu
from layers.eltwise import Operation
from functional import permute, reshape
from graph import DAG, NamedDim

class LayerTests(unittest.TestCase):
    def test_conv(self):
        graph = DAG()
        named_shapes = [
            NamedDim('batch', 1),
            NamedDim('unsqueezed',1),
            NamedDim('hidden_dim',1024),
            NamedDim('num_tokens',1536)
        ]
        inp = Input("input", graph, is_dynamic=True, inp_shape=named_shapes, out_group_size=64, out_group_axis = -1)()
        conv = Conv("Convolution", graph, 2048, bias_term=True, kernel_size=[1], stride=[1], is_dynamic=True, out_group_axis=-1, new_accumulation_dim_name="projected_dim", out_group_size=64)
        _ = conv(inp, accumulation_dim_name = "hidden_dim")
        graph.get_proto()
        graph.get_so_yaml()
    
    def test_lnrm(self):
        graph = DAG()
        named_shapes = [
            NamedDim('batch', 1),
            NamedDim('unsqueezed',1),
            NamedDim('num_tokens',1536),
            NamedDim('hidden_dim',1024)
        ]
        inp = Input("input", graph, is_dynamic=True, inp_shape=named_shapes, out_group_axis=-1, out_group_size = 64)()
        lnrm = LayerNorm('layer_norm', graph, uses_mean = True, output_mean_var = False, eps = 1e-06, is_dynamic=True, out_group_axis=-1, out_group_size = 64)
        _s = lnrm(inp, axis = -1)
        graph.get_proto()
        graph.get_so_yaml()
    
    def test_permute(self):
        graph = DAG()
        named_shapes = [
            NamedDim('batch', 1),
            NamedDim('unsqueezed',1),
            NamedDim('num_tokens',1536),
            NamedDim('hidden_dim',1024)
        ]
        inp = Input("input", graph, is_dynamic=True, inp_shape=named_shapes, out_group_axis=-1, out_group_size = 64)()
        _ = permute(graph, inp, ('hidden_dim', 'num_tokens'))
        graph.get_proto()
        graph.get_so_yaml()
    
    def test_softmax(self):
        graph = DAG()
        named_shapes = [
            NamedDim('batch', 1),
            NamedDim('unsqueezed',1),
            NamedDim('num_tokens',1536),
            NamedDim('hidden_dim',1024)
        ]
        softmax = SoftMax('softmax', graph, is_dynamic=True, out_group_axis=-2, out_group_size=64)
        inp = Input("input", graph, is_dynamic=True, inp_shape=named_shapes, out_group_axis = -1, out_group_size = 64)()
        _ = softmax(inp, axis = 3)
        graph.get_proto()
        graph.get_so_yaml()
    
    def test_swish(self):
        graph = DAG()
        named_shapes = [
            NamedDim('batch', 1),
            NamedDim('unsqueezed',1),
            NamedDim('num_tokens',1536),
            NamedDim('hidden_dim',1024)
        ]
        swish = Swish('swish', graph, is_dynamic=True, out_group_axis=-2, out_group_size = 64)
        inp = Input("input", graph, is_dynamic=True, inp_shape=named_shapes, out_group_axis=-1, out_group_size = 64)()
        _ = swish(inp)
        graph.get_proto()
        graph.get_so_yaml()
    
    def test_gelu(self):
        graph = DAG()
        named_shapes = [
            NamedDim('batch', 1),
            NamedDim('unsqueezed',1),
            NamedDim('num_tokens',1536),
            NamedDim('hidden_dim',1024)
        ]
        swish = Gelu('gelu', graph, is_dynamic=True, out_group_axis=-2, out_group_size = 64)
        inp = Input("input", graph, is_dynamic=True, inp_shape=named_shapes, out_group_axis=-1, out_group_size = 64)()
        _ = swish(inp)
        graph.get_proto()
        graph.get_so_yaml()
    
    def test_eltwise(self):
        graph = DAG()
        named_shapes = [
            NamedDim('batch', 1),
            NamedDim('unsqueezed',1),
            NamedDim('num_tokens',1536),
            NamedDim('hidden_dim',1024)
        ]
        add = Eltwise('add', graph, is_dynamic=True, out_group_axis=-1, out_group_size=64, operation=Operation.SUM)
        mul = Eltwise('mul', graph, is_dynamic=True, out_group_axis=-1, out_group_size=64, operation=Operation.PROD)
        inp_0 = Input("input_0", graph, is_dynamic=True, inp_shape=named_shapes, out_group_axis=-1, out_group_size=64)()
        inp_1 = Input("input_1", graph, is_dynamic=True, inp_shape=named_shapes, out_group_axis=-1, out_group_size=64)()
        _ = add(inp_0 + inp_1, derive_name_from_layer_name=True)
        _ = mul(inp_0 + inp_1, derive_name_from_layer_name=True)
        graph.get_proto()
        graph.get_so_yaml()
    
    def test_rope(self):
        graph = DAG()
        named_shapes = [
            NamedDim('batch', 1),
            NamedDim('unsqueezed',1),
            NamedDim('num_tokens',1536),
            NamedDim('hidden_dim',1024)
        ]
        named_rope_shape = [
            NamedDim('batch', 1),
            NamedDim('unsqueezed',1),
            NamedDim('num_max_tokens',4096),
            NamedDim('cos_and_sin',128)
        ]
        rope_table = Input('rope_table', graph, is_dynamic=False, inp_shape=named_rope_shape, out_group_axis = None, out_group_size = None)()
        rope = Rope('rope', graph, rope_table=rope_table, out_group_axis = -1, out_group_size = 64)
        inp = Input("input", graph, is_dynamic=True, inp_shape=named_shapes, out_group_axis = -1, out_group_size = 64)()
        _ = rope(inp)
        graph.get_proto()
        graph.get_so_yaml()
    
    def test_batchdot(self):
        graph = DAG()
        named_q_proj_shapes = [
            NamedDim('batch', 1),
            NamedDim('unsqueezed',1),
            NamedDim('hidden_dim',1024),
            NamedDim('hidden_dim',1536)
        ]
        named_k_proj_shapes = [
            NamedDim('batch', 1),
            NamedDim('unsqueezed',1),
            NamedDim('hidden_dim',1024),
            NamedDim('hidden_dim',1536)
        ]
        q_proj = Input("q_proj", graph, is_dynamic=True, inp_shape=named_q_proj_shapes, out_group_axis = -2, out_group_size = 64)()
        k_proj = Input("k_proj", graph, is_dynamic=True, inp_shape=named_k_proj_shapes, out_group_axis = -2, out_group_size = 64)()
        qkt = BatchDot('qkt', graph, is_dynamic=True, out_group_axis = -2, out_group_size = 64)
        _ = qkt(q_proj + k_proj)
        graph.get_proto()
        graph.get_so_yaml()
    
    def test_reshape(self):
        graph = DAG()
        named_shapes = [
            NamedDim('batch', 1),
            NamedDim('unsqueezed',1),
            NamedDim('hidden_dim',1024),
            NamedDim('tkn',1536)
        ]
        inp = Input("input", graph, is_dynamic=True, inp_shape=named_shapes, out_group_axis = -2, out_group_size = 64)()
        _ = reshape(graph, inp, [[('unsqueezed', 8, 'heads'), ('hidden_dim', 1024//8, 'head_dim')], [('unsqueezed',8, 'heads'), ('hidden_dim', 1024//8//64, 'head_dim')]], out_group_axis_name='head_dim')
        # print(out[0], out[1])
        graph.get_proto()
        graph.get_so_yaml()
    
if __name__ == "__main__":
    unittest.main()