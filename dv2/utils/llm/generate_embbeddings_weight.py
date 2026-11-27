#!/usr/bin/env python3

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

import onnx, os
import numpy as np
import onnx.numpy_helper
import argparse


# Function to find and dump the embedding table
def dump_embedding_table(model, initializer):
    # Search for the embedding layer's weights in the model
    for initializer in model.graph.initializer:
        if 'embedding' in initializer.name.lower() or 'embed' in initializer.name.lower():
            # Convert the tensor to numpy array
            embedding_table = onnx.numpy_helper.to_array(initializer)
            print(f"Embedding Table '{initializer.name}':")
            print(embedding_table)
            return embedding_table
    print("No embedding table found in the model.")
    return None



def run(args):
    try:
        # Load the ONNX model
        onnx_model = onnx.load(args.inet)
        # Get the model's graph
        graph = onnx_model.graph

        # Print all initializers (these are the parameters like weights and biases)
        print("Initializers in the model:")
        for initializer in graph.initializer:
            print(initializer.name)

        # Dump the embedding table
        embedding_table = dump_embedding_table(onnx_model, initializer)



        # Save the embedding table to a file if needed
        if embedding_table is not None:
            np.save('embedding_table_base.npy', embedding_table)
            print("Embedding table saved to 'embedding_table_base.npy'")

            arr_a = embedding_table
            #arr_a = np.load('embedding_table.npy')
            arr_a = arr_a.reshape((int(arr_a.size/64)), 64)
            arr_b = np.abs(arr_a)
            arr_c = np.max(arr_b, axis = 1)
            arr_c = np.log2(arr_c)
            arr_c = np.floor(arr_c) + 1
            arr_c = 7 - arr_c
            arr_c = arr_c.astype(np.int8)
            arr_c.tofile(os.path.join(args.output_dir,args.embbedded_wts_scale_file_name))
            arr_c = arr_c.reshape(-1,1)
            print(arr_a.shape, arr_c.shape)
            q_a = np.floor(arr_a*(np.power(2,arr_c.astype(np.int32))) + 0.5)
            print("q_a", q_a.shape)
            q_a =  np.clip(q_a, -128, 127).astype(np.int8)
            q_a.tofile(os.path.join(args.output_dir,args.embbedded_weights_file_name))
    
    except Exception as e:
        raise Exception("getGatherWts failed with " + str(e))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inet', required=True, help='path to onnx file')
    parser.add_argument('--output_dir', required=True, help='path to llm output directory')
    parser.add_argument('--embbedded_weights_file_name', required=True, help='file name for embbedded weights')
    parser.add_argument('--embbedded_wts_scale_file_name', required=True, help='file name for embbedded weights scale')
    # global args
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()