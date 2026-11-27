'''
Copyright (c) 2022-25, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

#!/auto/tools/installs/python/3.7.5/bin/python3.7
import argparse
import mxnet as mx
#from mmdnn.conversion.mxnet.mxnet_graph import MXNetGraph
parser = argparse.ArgumentParser()
requiredNamed = parser.add_argument_group('required arguments')
requiredNamed.add_argument('-i', "--inputSymbol", help='Input Symbol Path', required=True)
requiredNamed.add_argument('-o', '--outputNetwork', help='Output Network Path', required=True)
requiredNamed.add_argument('-ch', '--channels', help='channels', required=True)
requiredNamed.add_argument('-hi', '--height', help='height', required=True)
requiredNamed.add_argument('-wi', '--width', help='width', required=True)
requiredNamed.add_argument('-label', '--label', help='label name', required=True)
args = parser.parse_args()
inFile = args.inputSymbol
outFile = args.outputNetwork
channels = int(args.channels)
height = int(args.height)
width = int(args.width)
label = tuple(str(args.label).split(','))
# with open('Annapurna/'+file, 'r') as input_json:
#         json_string = input_json.read()
#         symbol = mx.sym.load_json(json_string)
#         model = mx.mod.Module(symbol = symbol)
'''
        mxnet_graph = MXNetGraph(model)
        model.bind(for_training=False, data_shapes=[('data', (1,3,5,5))])
        arg_params=symbol.list_arguments()
#       model.params_initialized = True
        init = mx.init.Xavier(factor_type="in", magnitude=2.34)
        model.init_params(init)
#       for i in arg_params:
                #print(i)
                #i = mx.initializer.Xavier(rnd_type='gaussian', factor_type="avg", magnitude=2)
#       print(model.get_params())       
'''
sym = mx.symbol.load(inFile)
model = mx.mod.Module(symbol=sym, label_names=label)
input_layer = 'data'
model.bind(data_shapes=[(input_layer, (1, channels, height, width))])
# import pdb;pdb.set_trace()
model.init_params()
model.save_checkpoint(outFile, 0)
# import pdb;pdb.set_trace()
# model.arg_params['softmax_handarm_label'] = mx.nd.array([0])
#model.save_params("Annapurna/"+file+"-9999.params")
