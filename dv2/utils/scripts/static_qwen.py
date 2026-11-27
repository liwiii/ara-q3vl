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


#model_id = "/auto/worka/aakash/llm/omniquant/OmniQuant/llama_7b_w8a8_e1_with_dp_smooth_without_lwc_without_activation_quant_save_dir"
#model_id = "/auto/worka/aakash/llm/omniquant/OmniQuant/llama_7b_w8a8_e1_with_dp_smooth_save_dir"
#model_id = "/auto/worka/abhash/models/llama2_epoch_20_e20_22_27_2bit_all_3/"
model_id = "/auto/regrt/sw/dgundimeda/qwen1_7B_model_epoch_20_llamafied/"
device = "cpu" # use the accelerator that you have or use "cpu"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-7B", trust_remote_code=True)
#tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")
import pdb;pdb.set_trace()
tokenizer.pad_token_id = 151643
tokenizer.bos_token_id = 151643
tokenizer.eos_token_id = 151643
# Load model
"""
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-7B-Chat",
    torch_dtype=torch.float32,
    device_map="auto"
)
"""
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
    model = onnx.load("/auto/regrt/sw/titash/qwen1_e10_22_27_2/model.sim.onnx")
    model.graph.output.extend([onnx.ValueInfoProto(name="/model/layers.0/self_attn/Unsqueeze_output_0")])
    model.graph.output.extend([onnx.ValueInfoProto(name="/model/layers.0/self_attn/Unsqueeze_1_output_0")])
    model.graph.output.extend([onnx.ValueInfoProto(name="/model/embed_tokens/Gather_output_0")])
    onnx.save(model,"/auto/regrt/sw/titash/qwen1_e10_22_27_2/model2.onnx", save_as_external_data=True)
    #sess = ort.InferenceSession("/auto/regrt/sw/titash/qwen1_original/model.sim.onnx")
    sess = ort.InferenceSession("/auto/regrt/sw/titash/qwen1_e10_22_27_2/model2.onnx")
    #model.graph.output.extend([onnx.ValueInfoProto(name="/model/Where_3_output_0")])
    #onnx.save(model,"/auto/worka/titash/llama_7b_onnx_static_256_where/model.onnx", save_as_external_data=True)
    #sess = ort.InferenceSession("/auto/regrt/sw/titash/onnx_export/qwen_onnx.onnx")
    #sess = ort.InferenceSession("/auto/regrt/sw/titash/qwen_optimum_onnx/model.onnx")
  
    for idx in range(seq_lenght, max_length):

        # Run the inference
        if idx != seq_lenght:
          #input_ids = torch.cat((input_ids,prev_input_ids[0][1:].reshape(1,255)), dim=-1)
          input_ids = torch.cat((torch.cat((prev_input_ids[0][:idx-1].reshape(1,idx-1), input_ids), dim =-1), prev_input_ids[0][idx-1: -1].reshape(1, 255-idx+1)),dim = -1)
          position_ids = torch.cat((torch.cat((prev_position_ids[0][:idx-1].reshape(1,idx-1), position_ids), dim =-1), prev_position_ids[0][idx-1: -1].reshape(1, 255-idx+1)),dim = -1)
          #position_ids = torch.cat((position_ids, prev_position_ids[0][1:].reshape(1,255)), dim=-1)
        
        import pdb;pdb.set_trace()
        prev_position_ids = position_ids
        prev_input_ids = input_ids
        past_key_values = None

        onnx_inp = dict()
        onnx_inp['input_ids'] = input_ids.detach().cpu().numpy().astype(np.int64)
        onnx_inp['attention_mask'] = attention_mask.detach().cpu().numpy().astype(np.int64)
        onnx_inp['position_ids'] = position_ids.detach().cpu().numpy().astype(np.int64)
        """
        for i in range(0, 32):
            onnx_inp[f"past_key_in{i}"] = np.empty((1, 32, 0,128)).astype(np.float32)
            onnx_inp[f"past_value_in{i}"] = np.empty((1, 32, 0,128)).astype(np.float32)
        """
        #import pdb;pdb.set_trace()
        onnx_res = sess.run(None, onnx_inp)
        #out = model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values)


        ## Here I do greedy search as an example, but in general is where you want to select the next token with your fancy decoding algorithm
        #logits = out.logits
        logits = torch.from_numpy(onnx_res[0])
        
        new_token = torch.argmax(logits[0, idx-1, :])
        import pdb;pdb.set_trace()
        yield new_token

        if not use_past:
            # Shift left input and position ids and set the new token and idx to the proper values
            input_ids = lshift_insert(input_ids, new_token)
            position_ids = lshift_insert(position_ids, idx)
        else:
            # Set input_ids and position_ids to their new value
            input_ids = torch.tensor([[new_token]], dtype=input_ids.dtype).to(device)
            position_ids = torch.tensor([[idx]], dtype=input_ids.dtype).to(device)

            # Select the proper KV cached keys for next inference
            #past_key_values = [[item[:, :, 1:, :] for item in layer_past] for layer_past in out.past_key_values]

        # Shift left attention mask and set the last value to one
        attention_mask = lshift_insert(attention_mask, 1)


#prompt = "List all numbers in the Fibonacci sequence: 1, 1, 2, 3, "
prompt = "The number 42 is especially significant to fans of science fiction novelist Douglas Adams The Hitchhiker’s Guide to the Galaxy, because that number is the answer given by a supercomputer to the Ultimate Question of Life, the Universe, and Everything. "
#prompt = "中国的首都是哪里？"
#prompt = "Write a code to compute factorial of 23 in python "
#prompt = "Value of (2+2) is "
#prompt = "(2+2) 的值为 "
max_length = 256

# Tokenize 
input_ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
minput_ids = input_ids
#print(prompt, end="", flush=True)
out = prompt
results = generate_with_static_shape(model, input_ids=input_ids, max_length=max_length, use_past=True, pad_token_id=tokenizer.pad_token_id)

for new_token_id in results:

    minput_ids = torch.cat((minput_ids, torch.tensor(new_token_id).reshape(1,1)), dim=1)
    token = tokenizer.decode(minput_ids[0], skip_special_tokens=True)
    # Not very good as depending on the tokenizer it might or might not add spaces
    print(token)
    #print(token)

