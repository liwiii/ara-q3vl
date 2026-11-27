#!/bin/env/python3

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

from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse, functools
import sys, torch, random, os, json, re, yaml, onnx
import numpy as np

def input_embeddings(model_repo, prompt, num_total_tokens, embedding_path, prompt_out, tokenout):
  tokenizer = AutoTokenizer.from_pretrained(model_repo)
  tokenized_inputs = tokenizer(prompt)["input_ids"]
  tokens = tokenized_inputs
  valid_till = len(tokens)
  tokens.extend([num_total_tokens for x in range(128 - valid_till)])
  arr = np.load(embedding_path)
  arr = np.reshape(arr, [-1, 4096])  #change required if model used -- 4096 is for llama2/qwen1.5 else use 3584 for llama3/qwen
  output_arr = [list(arr[i]) for i in tokens]
  output_arr = np.array(output_arr)
  print(output_arr.shape)
  os.mkdir(prompt_out) if not os.path.exists(prompt_out) else print("")
  os.mkdir(tokenout) if not os.path.exists(tokenout) else print("")
  prmpt_inp_embeddings = os.path.join(prompt_out, "input_ids_embedded.bin")
  tkn_inp_embeddings = os.path.join(tokenout, "input_ids_embedded.bin")
  output_arr.tofile(prmpt_inp_embeddings)
  output_arr[valid_till-1, :].tofile(tkn_inp_embeddings)
  valid_till_prmpt = os.path.join(prompt_out, 'valid_till.bin')
  valid_till_tken = os.path.join(tokenout, 'valid_till.bin')
  np.array([valid_till]).astype(np.float32).tofile(valid_till_prmpt)
  np.array([valid_till]).astype(np.float32).tofile(valid_till_tken)
  return prmpt_inp_embeddings, valid_till_prmpt

def model_kv_dump(model_repo, input_prompt, valid_till_prmpt, llama_cfg):
  seed = 10
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  # model_repo = "/auto/worka/sriram/llm_app/sid_llm/arasw/sw1/candid/test/networks/dv_cfg0/caffe/modelzoo/llama2_no_embeddings_splits/model_download/titash_model_export/qwen_model"
  model_repo = model_repo
  model = AutoModelForCausalLM.from_pretrained(model_repo, trust_remote_code=True)
  tokenizer = AutoTokenizer.from_pretrained(model_repo, trust_remote_code=True)
  tokenizer.pad_token_id = llama_cfg['pad_token_id']
  tokenizer.bos_token_id = llama_cfg['bos_token_id']
  tokenizer.eos_token_id = llama_cfg['eos_token_id']
  device = "cpu"
  # max_length = 256
  max_length = llama_cfg['prmpt_context_size']
  model.eval()
  input_text = input_prompt
  messages = [
      {
          "role": "system",
          "content": "You are a helpful assistant."
      },
      {
          "role": "user",
          "content": input_text
      },
  ]
  text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  input_ids = tokenizer.encode(text, return_tensors="pt")
  valid_till = len(input_ids[-1].numpy().tolist())
  input_ids_list = [[valid_till] * 128]
  input_ids = torch.Tensor(input_ids_list)
  input_ids = input_ids.type(torch.int64)
  output = model.generate(input_ids, return_dict_in_generate=True, output_scores=True, use_cache=True, max_new_tokens=128)
  loaded_out = output.past_key_values
  assert len(loaded_out) == 32
  # output_dir = os.path.abspath(os.path.join(valid_till_prmpt,"../../prompt"))
  # os.makedirs(output_dir, exist_ok=True) if not os.path.exists(output_dir) else 'simply glide'
  print("Loaded Output:")
  pmpt_token = ['prompt', 'token']
  for item_type in pmpt_token:
    output_dir = os.path.abspath(os.path.join(valid_till_prmpt,"../../{}".format(item_type)))
    for count, item in enumerate(loaded_out):
      for in_count, each_v in enumerate(item):
        if (in_count == 0) and (item_type == "prompt"):
          filename = os.path.join(output_dir, f"past_keys_{count}.bin")
          padded_key = reshape_and_save(each_v.numpy(), filename)
          padded_key.tofile(filename)
          arr_a = padded_key.reshape((int(padded_key.size/64)), 64)
          arr_b = np.abs(arr_a)
          arr_c = np.max(arr_b, axis = 1)
          arr_c[:128 * 64] = np.log2(arr_c[:128 * 64])
          arr_c[:128 * 64] = np.floor(arr_c[:128 * 64]) + 1
          arr_c[:128 * 64] = 7 - arr_c[:128 * 64]
          arr_c.tofile(os.path.join(output_dir, f"past_keys_{count}" + "_scale.bin"))
        elif (in_count == 1) and (item_type == "prompt"):
          filename = os.path.join(output_dir, f"past_values_{count}.bin")
          padded_value = reshape_and_save(each_v.numpy(), filename)
          padded_value.tofile(filename)
          arr_a = padded_value.reshape((32, 64, 4096))
          arr_b = np.abs(arr_a)
          arr_c = np.max(arr_b, axis = 1)
          arr_c[:2, :] = np.log2(arr_c[:2, :])
          arr_c[:2, :] = np.floor(arr_c[:2, :]) + 1
          arr_c[:2, :] = 7 - arr_c[:2, :]
          arr_c.tofile(os.path.join(output_dir,f"past_values_{count}" + "_scale.bin"))
        if (in_count == 0) and (item_type == "token"):
          filename = os.path.join(output_dir, f"past_keys_{count}.bin")
          padded_key = reshape_and_save(each_v.numpy(), filename)
          padded_key.tofile(filename)
          arr_a = padded_key.reshape((int(padded_key.size/64)), 64)
          arr_b = np.abs(arr_a)
          arr_c = np.max(arr_b, axis = 1)
          arr_c[:128 * 64] = np.log2(arr_c[:128 * 64])
          arr_c[:128 * 64] = np.floor(arr_c[:128 * 64]) + 1
          arr_c[:128 * 64] = 7 - arr_c[:128 * 64]
          arr_c.tofile(os.path.join(output_dir, f"past_keys_{count}" + "_scale.bin"))
        elif (in_count == 1) and (item_type == "token"):
          filename = os.path.join(output_dir, f"past_values_{count}.bin")
          padded_value = reshape_and_save(each_v.numpy(), filename)
          padded_value.tofile(filename)
          #### last_values
          last_padded = padded_value.reshape((2048, 4096))
          last_pad_val = last_padded[0:64,:]
          last_pad_val.tofile(os.path.join(output_dir, f"past_values_{count}"  + "_last64" + ".bin"))
          ### values scale
          arr_a = padded_value.reshape((32, 64, 4096))
          arr_b = np.abs(arr_a)
          arr_c = np.max(arr_b, axis = 1)
          arr_c[:2, :] = np.log2(arr_c[:2, :])
          arr_c[:2, :] = np.floor(arr_c[:2, :]) + 1
          arr_c[:2, :] = 7 - arr_c[:2, :]
          arr_c.tofile(os.path.join(output_dir,f"past_values_{count}" + "_scale.bin"))
          #### last values scale
          last_val_scale = arr_c.reshape((32, 4096))
          last_scale_np = last_val_scale[0,:]
          last_scale_np.tofile(os.path.join(output_dir,f"past_values_{count}" + "_last64_scale.bin"))


def reshape_and_save(src_file, dest_file, shape=(1, 1, 2048, 4096)):
  data = src_file
  required_size = np.prod(shape)
  if data.size < required_size:
      padded_data = np.zeros(required_size, dtype=np.float32)
      padded_data[:data.size] = data.flatten()
  else:
      padded_data = data[:required_size]
  reshaped_data = padded_data.reshape(shape)
  reshaped_data.tofile(dest_file)
  return reshaped_data

class DBG_Hook(object):

  def __init__(self):
    self.qbits = 8
    self.count = 0

  def __call__(self, module, input, output, name, rope_out, rope_file_name):
    try:
      if ("rotary_emb" in name) and (self.count==0):
        print(module)
        out_first = output[0]
        out_second = output[1]
        cos = out_first.detach().cpu().numpy()
        sin = out_second.detach().cpu().numpy()
        rope_input = np.zeros((128, 256), dtype=np.float32)
        rope_total = np.zeros((2048, 256), dtype=np.float32)
        rope_total = np.concatenate((cos, sin), axis=1)
        self.adjust_and_store(rope_total[:128,:], 0.0078125, 0, rope_file_name, rope_out)
        # rope_input[:, 0:128] = cos[:, :]
        # rope_input[:, 128:] = sin[:, :]
        # rope_input.tofile(os.path.join(prompt_inp_folder,"position_ids_embedded.bin"))
        # rope_input[4, :].tofile(os.path.join(token_inp_folder,"position_ids_embedded.bin"))
        self.count +=1
    except Exception as e:
      print(e)

  def adjust_and_store(self, wt_array, scale, offset, output_file, outpt_fldr_name):
    adjusted_wt_array = np.floor((wt_array / scale) + offset + 0.5)
    clamped_wt_array = np.clip(adjusted_wt_array, -128, 127)
    dir = os.path.join(outpt_fldr_name)
    if not os.path.exists(dir):
        # Create the directory
        os.makedirs(dir)
    ofile = os.path.join(dir,output_file)
    clamped_wt_array.astype(np.int8).tofile(ofile)
def debug_hooks(model, rope_out, rope_file_name):
  dbg_hooks = []
  dbg_hook = DBG_Hook()
  layers = ["model.rotary_emb","rotary_emb"]
  for layer in layers:
    for name, module in model.named_modules():
      if layer in name:
        hook = module.register_forward_hook(functools.partial(dbg_hook, name=name, rope_out=rope_out, rope_file_name = rope_file_name))
        print(f"{name} hook registered.")
        dbg_hooks.append(hook)
        break
  return model


def gen_positonal_embedings(model_repo,rope_out,input_prompt, llama_cfg,rope_file_name):
  # Set same seed for numpy, torch, torch_cuda and random modules
  try:
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(model_repo)
    tokenizer = AutoTokenizer.from_pretrained(model_repo)
    tokenizer.pad_token_id = llama_cfg['pad_token_id']
    tokenizer.bos_token_id = llama_cfg['bos_token_id']
    tokenizer.eos_token_id = llama_cfg['eos_token_id']
    device = "cpu"

    model.eval()
    input_text = input_prompt
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": input_text
        },
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(text, return_tensors="pt")
    valid_till = len(input_ids[-1].numpy().tolist())
    input_ids_list = [[valid_till] * 128]
    input_ids = torch.Tensor(input_ids_list)
    input_ids = input_ids.type(torch.int64)
    model = debug_hooks(model, rope_out, rope_file_name)
    output = model.generate(input_ids, do_sample=False, max_new_tokens=128)
  except Exception as e:
    raise Exception("rope table generation failed with " + str(e))
  
def generate_input_embeddings(llama_cfg, embed_table, inp_prompt, inp_token):
  model = llama_cfg['model_repo']
  prompt = llama_cfg['prompt']
  total_num_tokens = llama_cfg['num_total_tokens']
  prmpt_inp_embed, valid_till_prmpt = input_embeddings(model_repo=model, prompt=prompt, num_total_tokens=total_num_tokens, embedding_path=embed_table, prompt_out=inp_prompt, tokenout=inp_token)
  model_kv_dump(model, prompt, valid_till_prmpt, llama_cfg)

def generate_embedded_wts(model):
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

def check_embedding_table(network, output):
  # Load the ONNX model
  onnx_model = network
  onnx_model = onnx.load(onnx_model)
  # Get the model's graph
  graph = onnx_model.graph
  # Print all initializers (these are the parameters like weights and biases)
  print("Initializers in the model:")
  for initializer in graph.initializer:
    print(initializer.name)
  embedding_table = generate_embedded_wts(onnx_model)
  # Save the embedding table to a file if needed
  embed_wts_outdir = os.path.join(output, 'llama')
  os.makedirs(embed_wts_outdir) if not os.path.exists(embed_wts_outdir) else print("simply dump")
  if embedding_table is not None:
    embedding_table_path = os.path.join(embed_wts_outdir, 'embedding_table.npy')
    np.save(embedding_table_path, embedding_table)
    print("Embedding table saved to 'embedding_table.npy'")
    return embedding_table_path  
  

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_repo', default="", required=False, help="Hugging face repo id/ local cache repo")
  parser.add_argument('--output_dumps', default=os.path.join(os.path.dirname(os.path.realpath(__file__)),"out"), required=False, help="output folder to dump prompt inputs")
  parser.add_argument('--onnx_model', required=False, help="onnx model to dump embed weights")
  parser.add_argument('--llm_config', required=True, help="llm config for generate inputs")
  parser.add_argument('--prompt', required=False, default=None, help="prompt input to run")
  parser.add_argument('--rope_file_name',required=True, help='rope file name to dump with')
  args = parser.parse_args()
  return args
  
if __name__ == '__main__':
  args = parse_args()
  args.output_dumps = os.path.abspath(args.output_dumps)
  # inp_prompt = os.path.join(args.output_dumps,'analyzer_inputs', 'prompt')
  # inp_token = os.path.join(args.output_dumps, 'analyzer_inputs', 'token')
  # ana_inp_fldr = os.path.join(args.output_dumps,'analyzer_inputs')
  os.mkdir(args.output_dumps) if not os.path.exists(args.output_dumps) else print("")
  # os.mkdir(ana_inp_fldr) if not os.path.exists(ana_inp_fldr) else print("")
  llm_config_path = args.llm_config
  # network = args.onnx_model
  output_embed = os.path.join(args.output_dumps,"embed_weights")
  with open(llm_config_path, 'r') as file:
    llm_config = yaml.safe_load(file)
  prompt = llm_config['prompt'] if not args.prompt else args.prompt
  print(prompt)
  # embed_table = check_embedding_table(network, output_embed)
  # generate_input_embeddings(llm_config, embed_table=embed_table, inp_prompt=inp_prompt, inp_token=inp_token)
  gen_positonal_embedings(args.model_repo,rope_out=args.output_dumps, input_prompt= prompt, llama_cfg=llm_config, rope_file_name = args.rope_file_name)
