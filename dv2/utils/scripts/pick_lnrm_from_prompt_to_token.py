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

import sys
sys.path.append("/auto/regrt/sw/titash/llama2/arasw/bld6/dv2/python/")
sys.path.append("/auto/regrt/sw/titash/llama2/arasw/bld6/dv2/python/caffe/proto/")

import caffe
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

for i in range(0,32):

    net = caffe_pb2.NetParameter()
    net2 = caffe_pb2.NetParameter()

    tkn = "tkn_gen_256/split_"+str(i)+"/__run.default/quantizer/params/convergence.prototxt"
    prmpt = "prmpt_prc_256/split_"+str(i)+"/__run.default/quantizer/params/convergence.prototxt"

    with open(prmpt) as f:
        s = f.read()
        txtf.Merge(s, net)

    with open(tkn) as f:
        s = f.read()
        txtf.Merge(s, net2)


    layerNames = [l.name for l in net.layer]
    layerNames2 = [l.name for l in net2.layer]

    idx1 = layerNames.index("_model_layers_"+str(i)+"_input_layernorm_Mul_1_ara_LayerNorm_0")
    idx2 = layerNames.index("_model_layers_"+str(i)+"_post_attention_layernorm_Mul_1_ara_LayerNorm_0")

    idx12 = layerNames2.index("_model_layers_"+str(i)+"_input_layernorm_Mul_1_ara_LayerNorm_0")
    idx22 = layerNames2.index("_model_layers_"+str(i)+"_post_attention_layernorm_Mul_1_ara_LayerNorm_0")

    net2.layer[idx12].analyzer_param.layerParams.inShift  = net.layer[idx1].analyzer_param.layerParams.inShift
    net2.layer[idx12].analyzer_param.layerParams.meanShift  = net.layer[idx1].analyzer_param.layerParams.meanShift
    net2.layer[idx12].analyzer_param.layerParams.varShift  = net.layer[idx1].analyzer_param.layerParams.varShift
    net2.layer[idx12].analyzer_param.layerParams.multiplier  = net.layer[idx1].analyzer_param.layerParams.multiplier
    net2.layer[idx12].analyzer_param.layerParams.varianceFraction  = net.layer[idx1].analyzer_param.layerParams.varianceFraction
    net2.layer[idx12].analyzer_param.layerParams.stdFraction  = net.layer[idx1].analyzer_param.layerParams.stdFraction
    net2.layer[idx12].analyzer_param.layerParams.normShift  = net.layer[idx1].analyzer_param.layerParams.normShift
    net2.layer[idx12].analyzer_param.layerParams.meanDeviationShift  = net.layer[idx1].analyzer_param.layerParams.meanDeviationShift
    net2.layer[idx12].analyzer_param.layerParams.meanDeviationMultiplier  = net.layer[idx1].analyzer_param.layerParams.meanDeviationMultiplier
    net2.layer[idx12].analyzer_param.layerParams.varMultiplier  = net.layer[idx1].analyzer_param.layerParams.varMultiplier
    net2.layer[idx12].analyzer_param.layerParams.normMultiplier  = net.layer[idx1].analyzer_param.layerParams.normMultiplier
    net2.layer[idx12].analyzer_param.layerParams.gammaMultiplier  = net.layer[idx1].analyzer_param.layerParams.gammaMultiplier
    net2.layer[idx12].analyzer_param.layerParams.gammaShift  = net.layer[idx1].analyzer_param.layerParams.gammaShift
    net2.layer[idx12].analyzer_param.layerParams.finalMultiplier  = net.layer[idx1].analyzer_param.layerParams.finalMultiplier


    net2.layer[idx22].analyzer_param.layerParams.inShift  = net.layer[idx2].analyzer_param.layerParams.inShift
    net2.layer[idx22].analyzer_param.layerParams.meanShift  = net.layer[idx2].analyzer_param.layerParams.meanShift
    net2.layer[idx22].analyzer_param.layerParams.varShift  = net.layer[idx2].analyzer_param.layerParams.varShift
    net2.layer[idx22].analyzer_param.layerParams.multiplier  = net.layer[idx2].analyzer_param.layerParams.multiplier
    net2.layer[idx22].analyzer_param.layerParams.varianceFraction  = net.layer[idx2].analyzer_param.layerParams.varianceFraction
    net2.layer[idx22].analyzer_param.layerParams.stdFraction  = net.layer[idx2].analyzer_param.layerParams.stdFraction
    net2.layer[idx22].analyzer_param.layerParams.normShift  = net.layer[idx2].analyzer_param.layerParams.normShift
    net2.layer[idx22].analyzer_param.layerParams.meanDeviationShift  = net.layer[idx2].analyzer_param.layerParams.meanDeviationShift
    net2.layer[idx22].analyzer_param.layerParams.meanDeviationMultiplier  = net.layer[idx2].analyzer_param.layerParams.meanDeviationMultiplier
    net2.layer[idx22].analyzer_param.layerParams.varMultiplier  = net.layer[idx2].analyzer_param.layerParams.varMultiplier
    net2.layer[idx22].analyzer_param.layerParams.normMultiplier  = net.layer[idx2].analyzer_param.layerParams.normMultiplier
    net2.layer[idx22].analyzer_param.layerParams.gammaMultiplier  = net.layer[idx2].analyzer_param.layerParams.gammaMultiplier
    net2.layer[idx22].analyzer_param.layerParams.gammaShift  = net.layer[idx2].analyzer_param.layerParams.gammaShift
    net2.layer[idx22].analyzer_param.layerParams.finalMultiplier  = net.layer[idx2].analyzer_param.layerParams.finalMultiplier

    outFn = "tkn_gen_256/split_"+str(i)+"/__run.default/quantizer/params/convergence.prototxt"
    # print 'writing', outFn
    with open(outFn, 'w') as f:
        f.write(str(net2))
