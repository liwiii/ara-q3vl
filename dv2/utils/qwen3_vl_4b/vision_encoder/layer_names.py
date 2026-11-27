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

# layer name constants used for caffmodel too
Q_PROJ = "q_proj"
K_PROJ = "k_proj"
V_PROJ = "v_proj"
O_PROJ = "o_proj"

# layer name constants used for proto only
ROPE = "rope"
QKT_BATCHDOT = "qkt_batchdot"
QKT_SOFTMAX = "qkt_softmax"
ATTN_BATCHDOT = "attn_batchdot"
RESIDUAL_ADD = "residual_add"
SILU_MUL = "silu_mul"
FINAL_ADD = "final_add"
LNORM_1 = "lnorm1"
LNORM_2 = "lnorm2"
