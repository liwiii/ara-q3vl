'''
Copyright © 2024 Intel Corporation
SPDX-License-Identifier: Apache 2.0
Origin: https://github.com/intel/intel-npu-acceleration-library

Copyright (c) 2025, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Modifications:

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import onnx
import onnxruntime as ort
import numpy as np

import sys
sys.path.append("/auto/regrt/sw/titash/llama2/arasw/bld6/dv2/python/")
sys.path.append("/auto/regrt/sw/titash/llama2/arasw/bld6/dv2/python/caffe/proto")
import caffe
#model_id = "/auto/worka/aakash/llm/omniquant/OmniQuant/llama_7b_w8a8_e1_with_dp_smooth_without_lwc_without_activation_quant_save_dir"
#model_id = "/auto/worka/abhash/models/llama2_epoch_20_e20_22_27_2bit_all_3"
model_id = "/auto/regrt/sw/dgundimeda/llama2_7b_chat_epoch_20_e20_22_27_2bit_all_3/"
device = "cpu" # use the accelerator that you have or use "cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, add_prefix_space=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
# Load model
model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

# Utility function to compute shift left and insert a value into a tensor
def lshift_insert(tensor, value):
    tensor = torch.roll(tensor, shifts=1, dims=-1)
    tensor[0, 0] = value
    return tensor

# Generate function 
@torch.no_grad()
def generate_with_static_shape(model, input_ids, attention_mask=None, max_length=None, use_past=True, pad_token_id=None, **kwargs):

    # Get sequence lenght
    batch, seq_lenght = input_ids.shape

    if pad_token_id is None:
        RuntimeError("pad_token_id is not set and needed for static shape generation")

    # Padding attention mask
    if attention_mask is None:
        attention_mask  = torch.ones_like(input_ids, dtype=torch.int32).to(model.device)
    attention_mask_padding  = torch.zeros((batch, max_length - seq_lenght), dtype=input_ids.dtype, device=input_ids.device)
    attention_mask = torch.cat(( attention_mask, attention_mask_padding), dim=-1)

    # Padding input_ids with left padding
    padding_input_ids = pad_token_id * torch.ones((batch, max_length - seq_lenght), dtype=input_ids.dtype, device=input_ids.device)
    input_ids = torch.cat((input_ids, padding_input_ids), dim=-1).to(model.device)

    # Set the proper position ids
    position_ids = kwargs.get('position_ids', None)
    if position_ids is None:
        position_ids = torch.tensor([list(range(seq_lenght)) + [0] * (max_length - seq_lenght) ], dtype=torch.int32).to(model.device)
    else:
        raise RuntimeError("Cannot set position_ids with in static shape generation")

    # past_key_values for KV-cache
    past_key_values = None

    prev_position_ids = position_ids
    prev_input_ids = input_ids

    
    #net = caffe.Net("/auto/regrt/sw/titash/qwen_converted/params/nnconvert.prototxt", "/auto/regrt/sw/titash/qwen_converted/params/nnconvert.caffemodel",caffe.TEST)
    sess = ort.InferenceSession("/auto/regrt/sw/titash/llama_2_a16_s/model.onnx")
    model = onnx.load("/auto/regrt/sw/titash/llama_2_a16_s/model.onnx")
    model.graph.output.extend([onnx.ValueInfoProto(name="/model/layers.0/self_attn/Unsqueeze_output_0")])
    model.graph.output.extend([onnx.ValueInfoProto(name="/model/layers.0/self_attn/Unsqueeze_1_output_0")])
    model.graph.output.extend([onnx.ValueInfoProto(name="/model/embed_tokens/Gather_output_0")])
    onnx.save(model,"/auto/regrt/sw/titash/llama2_chat_omniquant/model.onnx", save_as_external_data=True)
    sess = ort.InferenceSession("/auto/regrt/sw/titash/llama2_chat_omniquant/model.onnx") 
    net = caffe.Net("/auto/regrt/sw/titash/llama2_prompt.prototxt", "/auto/regrt/sw/titash/llama2_chat_a16_converted/params/nnconvert.caffemodel",caffe.TEST)
    cnt = 0
    
    for idx in range(seq_lenght, max_length):
        # Run the inference
        if idx != seq_lenght:
          #input_ids = torch.cat((input_ids,prev_input_ids[0][1:].reshape(1,255)), dim=-1)
          input_ids = torch.cat((torch.cat((prev_input_ids[0][:idx-1].reshape(1,idx-1), input_ids), dim =-1), prev_input_ids[0][idx-1: -1].reshape(1, 255-idx+1)),dim = -1)
          position_ids = torch.cat((torch.cat((prev_position_ids[0][:idx-1].reshape(1,idx-1), position_ids), dim =-1), prev_position_ids[0][idx-1: -1].reshape(1, 255-idx+1)),dim = -1)
          #position_ids = torch.cat((position_ids, prev_position_ids[0][1:].reshape(1,255)), dim=-1)
        
        prev_position_ids = position_ids
        prev_input_ids = input_ids
        past_key_values = None
        
        # import pdb;pdb.set_trace()
        onnx_inp = dict()
        onnx_inp['input_ids'] = input_ids.detach().cpu().numpy().astype(np.int64)
        onnx_inp['attention_mask'] = attention_mask.detach().cpu().numpy().astype(np.int64)
        onnx_inp['position_ids'] = position_ids.detach().cpu().numpy().astype(np.int64)
        #net.blobs['input_ids'].data[...] = input_ids.detach().cpu().numpy().astype(np.float32)
        #net.blobs['position_ids'].data[...] = position_ids.detach().cpu().numpy().astype(np.float32)
        arr = np.zeros((1,1,256,256)).astype(np.float32)
        maskarr = attention_mask.detach().cpu().numpy().astype(np.float32).reshape(256)
        oarr = np.zeros((256)).astype(np.float32)
        for i in range(0,256):
          oarr[i] = -3.4028234663852886e+38;
        for i in range(0,256):
          if maskarr[i] == 1:
            oarr[i] = 0.0;
          arr[0][0][i] = oarr
        new_arr = [idx]
        onnx_res = sess.run(None, onnx_inp)
        
        net.blobs['input_ids_embedded'].data[...] = onnx_res[3] #.numpy().astype(np.float32).reshape(1,1,256,128)
        net.blobs['position_ids_embedded'].data[...] = np.concatenate((onnx_res[1], onnx_res[2]),axis = 3).transpose(0,1,3,2) #.numpy().astype(np.float32).reshape(1,1,256,128)
        net.blobs['valid_till'].data[...] = np.array(new_arr).astype(np.float32)
        #net.blobs['input_ids'].data[...].tofile("/auto/regrt/sw/titash/qwen_omniquant_input/img0/input_ids.bin")
        #net.blobs['_model_layers_0_self_attn_Slice_4_output_0'].data[...].tofile("/auto/regrt/sw/titash/qwen_omniquant_input/img0/_model_layers_0_self_attn_Slice_4_output_0.bin")
        #net.blobs['_model_layers_0_self_attn_Unsqueeze_1'].data[...].tofile("/auto/regrt/sw/titash/qwen_omniquant_input/img0/_model_layers_0_self_attn_Unsqueeze_1.bin")
        #net.blobs['_model_layers_0_self_attn_Unsqueeze'].data[...].tofile("/auto/regrt/sw/titash/qwen_omniquant_input/img0/_model_layers_0_self_attn_Unsqueeze.bin")
        #net.blobs['position_ids'].data[...].tofile("/auto/regrt/sw//titash/qwen_input/img0/position_ids.bin")
        #net.blobs['_model_Where_3_output_0'].data[...].tofile("/auto/regrt/sw/titash/qwen_input/img0/_model_Where_3_output_0.bin")
        #out = net.forward(qdq=0, file = "scale_llama_" + str(cnt) + ".txt")
        import pdb;pdb.set_trace()
        out = net.forward(qdq = 0)
        cnt = cnt + 1
        #onnx_res = sess.run(None, onnx_inp)
        #out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values)


        ## Here I do greedy search as an example, but in general is where you want to select the next token with your fancy decoding algorithm
        #logits = torch.from_numpy(onnx_res[0])
        logits = torch.from_numpy(out["_lm_head_MatMul_ara_inrp"].reshape(1,32000,256).transpose(0,2,1))
        #logits = out.logits
        #logits = torch.from_numpy(onnx_res[0])
        #import pdb;pdb.set_trace()
        new_token = torch.argmax(logits[0, idx-1, :])
        yield new_token

        if not use_past:
            # Shift left input and position ids and set the new token and idx to the proper values
            input_ids = lshift_insert(input_ids, new_token)
            position_ids = lshift_insert(position_ids, idx)
        else:
            # Set input_ids and position_ids to their new value
            input_ids = torch.tensor([[new_token]], dtype=input_ids.dtype).to("cpu")
            position_ids = torch.tensor([[idx]], dtype=input_ids.dtype).to("cpu")

            # Select the proper KV cached keys for next inference
            #past_key_values = [[item[:, :, 1:, :] for item in layer_past] for layer_past in out.past_key_values]

        # Shift left attention mask and set the last value to one
        attention_mask = lshift_insert(attention_mask, 1)


#prompt = "List all numbers in the Fibonacci sequence:  1, 1, "
#prompt = "Give me a short introduction to large language model. "
#prompt = "中国的首都是哪里？"
prompt = "What's the Value of (2+2) ? "
max_length = 256

messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": prompt}
    ]
text = tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
    )

# Tokenize 
input_ids = tokenizer([text], return_tensors='pt')['input_ids'].to(device)
minput_ids = input_ids
#print(prompt, end="", flush=True)
out = prompt
results = generate_with_static_shape(model, input_ids=input_ids, max_length=max_length, use_past=True, pad_token_id=tokenizer.pad_token_id)


for new_token_id in results:

    minput_ids = torch.cat((minput_ids, torch.tensor(new_token_id).reshape(1,1)), dim=1)
    import pdb;pdb.set_trace()
    token = tokenizer.decode(minput_ids[0], skip_special_tokens=True)
    # Not very good as depending on the tokenizer it might or might not add spaces
    print(token)
    #print(token)

