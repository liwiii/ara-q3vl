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

#from ruamel import yaml
import yaml
with open("/auto/worka/titash/accuracy/experiments/qwen1.0/qwen_prompt.yaml") as f:
      my_dict = yaml.safe_load(f)

with open("/auto/regrt/sw/titash/qwen1.0/Quantizer/prompt_output_dbg_2/stats/so.yaml") as f1:
      my_dict2 = yaml.safe_load(f1)

for keys in my_dict['LayerNames']:
  if keys in my_dict2['LayerNames']:
    my_dict['LayerNames'][keys]['scale'] = my_dict2['LayerNames'][keys]['scale']
    my_dict['LayerNames'][keys]['offset'] = my_dict2['LayerNames'][keys]['offset']
    del my_dict['LayerNames'][keys]['scale2']
    del my_dict['LayerNames'][keys]['offset2']
    #new_dict['LayerNames'][keys] = {**my_dict['LayerNames'][keys], **my_dict2['LayerNames'][keys]}
  #else:
    #new_dict['LayerNames'][keys] = my_dict['LayerNames'][keys]
with open('/auto/worka/titash/accuracy/experiments/qwen1.0/qwen_prompt_try1.yaml', 'w+') as ff:
    yaml.dump(my_dict, ff)
with open('/auto/worka/titash/accuracy/experiments/qwen1.0/qwen_prompt_try1.yaml') as f:
        my_dict = yaml.safe_load(f)

with open("/auto/regrt/sw/titash/llama2/arasw/sw1/candid/test/networks/dv_cfg0/caffe/modelzoo/llama2_no_embeddings_splits/tx2.yaml") as f1:
        my_dict2 = yaml.safe_load(f1)

for i in range(0,32):
  k1 = "past_keys_"+str(i)
  del my_dict2['LayerNames'][k1]
  my_dict2['LayerNames'][k1] = {'scale': my_dict['LayerNames']["_model_layers_"+str(i)+"_self_attn_Add_1"]['scale'], 'offset':  my_dict['LayerNames']["_model_layers_"+str(i)+"_self_attn_Add_1"]['offset']}
  my_dict2['LayerNames']["_model_layers_"+str(i)+"_self_attn_2a_Transpose_3_concat"] = {'scale': my_dict['LayerNames']["_model_layers_"+str(i)+"_self_attn_Add_1"]['scale'], 'offset':  my_dict['LayerNames']["_model_layers_"+str(i)+"_self_attn_Add_1"]['offset']}
  v1 = "past_values_"+str(i)
  del my_dict2['LayerNames'][v1]
  my_dict2['LayerNames'][v1] = {'scale': my_dict['LayerNames']['_model_layers_'+str(i)+'_self_attn_v_proj_MatMul']['scale'], 'offset': my_dict['LayerNames']['_model_layers_'+str(i)+'_self_attn_v_proj_MatMul']['offset']}
  my_dict2['LayerNames']["_model_layers_"+str(i)+"_self_attn_Reshape_2_concat"] = {'scale': my_dict['LayerNames']['_model_layers_'+str(i)+'_self_attn_v_proj_MatMul']['scale'], 'offset': my_dict['LayerNames']['_model_layers_'+str(i)+'_self_attn_v_proj_MatMul']['offset']}

for keys in my_dict2['LayerNames']:
  if keys in my_dict['LayerNames']:
    my_dict2['LayerNames'][keys]['scale'] = my_dict['LayerNames'][keys]['scale']
    my_dict2['LayerNames'][keys]['offset'] = my_dict['LayerNames'][keys]['offset']
"""
for i in range(0,32):
  del my_dict2['LayerNames']["_model_layers_"+str(i)+"_self_attn_q_proj_Add_original"]
  del my_dict2['LayerNames']["_model_layers_"+str(i)+"_self_attn_q_proj_Add_original_bnorm"]
  del my_dict2['LayerNames']["_model_layers_"+str(i)+"_self_attn_v_proj_Add_original"]
  del my_dict2['LayerNames']["_model_layers_"+str(i)+"_self_attn_v_proj_Add_original_bnorm"]
  del my_dict2['LayerNames']["_model_layers_"+str(i)+"_self_attn_k_proj_Add_original"]
  del my_dict2['LayerNames']["_model_layers_"+str(i)+"_self_attn_k_proj_Add_original_bnorm"] 
"""
with open('/auto/worka/titash/accuracy/experiments/qwen1.0/qwen_token_try1.yaml', 'w+') as ff:
  yaml.dump(my_dict2, ff)
import pdb;pdb.set_trace()
