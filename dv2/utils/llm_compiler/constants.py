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


# DIMENSIONS
DIM0 = 'DIM0'
DIM1 = 'DIM1'
DIM2 = 'DIM2'
DIM3 = 'DIM3'
HIDDEN = 'HIDDEN'
TOKEN = 'TOKEN'
TOKEN_CACHE = 'TOKEN_CACHE'
NUM_HEADS = 'NUM_HEADS'
HEAD_DIM = 'HEAD_DIM'
SIN_COS = 'SIN_COS'

# Layers
INPUT_RMS = 'input_layernorm'
K_PROJ = 'k_proj'
V_PROJ = 'v_proj'
Q_PROJ = 'q_proj'
Q_NORM = 'q_norm'  # Qwen3 Q normalization
K_NORM = 'k_norm'  # Qwen3 K normalization
K_ROPE = 'k_rope'
Q_ROPE = 'q_rope'
Q_KT = 'QxKt'
SFMX = 'attention_softmax'
P_V = 'PxV'
O_PROJ = 'o_proj'
ATTENTION_RESIDUAL = 'attention_residual'
POST_ATTENTION_RMS = 'post_attention_layernorm'
UP_PROJ = 'up_proj'
GATE_PROJ = 'gate_proj'
GATE_PROJ_ACT = 'gate_proj_act'
UP_GATE_MUL = 'up_gate_mul'
DOWN_PROJ = 'down_proj'
MLP_RESIDUAL = 'mlp_residual'
LMHEAD = 'lm_head'
REQUANT = 'requant'

# Weight names
LMHEAD_RMS = 'final_norm_layernorm.weight'
LMHEAD_WEIGHT = 'lm_head.MatMul.weight'
QKV_WEIGHT = 'qkv_proj.MatMul.weight'
QKV_BIAS = 'qkv_proj.Add.bias'
Q_PROJ_WEIGHT = 'q_proj.MatMul.weight' 
K_PROJ_WEIGHT = 'k_proj.MatMul.weight' 
V_PROJ_WEIGHT = 'v_proj.MatMul.weight'
Q_PROJ_BIAS = 'q_proj.MatMul.bias' 
K_PROJ_BIAS = 'k_proj.MatMul.bias' 
V_PROJ_BIAS = 'v_proj.MatMul.bias' 
O_PROJ_WEIGHT = 'o_proj.MatMul.weight'
Q_NORM_WEIGHT = 'q_norm.layernorm.weight'
K_NORM_WEIGHT = 'k_norm.layernorm.weight'
UP_PROJ_WEIGHT = 'up_proj.MatMul.weight'
GATE_PROJ_WEIGHT = 'gate_proj.MatMul.weight'
DOWN_PROJ_WEIGHT = 'down_proj.MatMul.weight'
INPUT_LAYERNORM_WEIGHT = 'input_layernorm.weight' 
POST_ATTENTION_LAYERNORM_WEIGHT = 'post_attention_layernorm.weight'
EMBEDDINGS = 'model.embed_tokens.weight'
SIN_CACHE = 'sin_cache'
COS_CACHE = 'cos_cache'

# Inputs
INPUT = 'input_ids_embedded'
POSTION_EMBEDS = 'position_ids_embedded'
PAST_KEYS = 'past_keys'
PAST_VALUES = 'past_values'
PAST_VALUES_REQUANT = 'past_values_requant'
VALID_TILL = 'valid_till'

#Precisions
INT4 = 'INT4'
INT3 = 'INT3'
INT2 = 'INT2'
INT8 = 'INT8'
INT16 = 'INT16'
INT32 = 'INT32'

#GGUF Tensors
K_PROJ_GGUF = 'attn_k.weight'
V_PROJ_GGUF = 'attn_v.weight'
Q_PROJ_GGUF = 'attn_q.weight'
O_PROJ_GGUF = 'attn_output.weight'
UP_PROJ_GGUF = 'ffn_up.weight'
GATE_PROJ_GGUF = 'ffn_gate.weight'
DOWN_PROJ_GGUF = 'ffn_down.weight'