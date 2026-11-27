
/*
Copyright 2025-2026 NXP
License: LA_OPT_NXP_Software_License
See the LICENSE for more details.
NXP Confidential and Proprietary. This software is owned or controlled by NXP and may only be used strictly in accordance with the applicable license terms. By expressly accepting such terms or by downloading, installing, activating and/or otherwise using the software, you are agreeing that you have read, and that you agree to comply with and are bound by, such license terms. If you do not agree to be bound by the applicable license terms, then you may not retain, install, activate or otherwise use the software.

Copyright (c) 2018-23, Kinara, Inc. All rights reserved.
*/


#pragma once

#include "vpu_param.h"
#include "common_structs.h"

typedef struct Tf_Avgpool {
  dvnet_tensor value;
  int32_t ksize[MAX_OPERAND_SIZE]; int ksize_size;
  int32_t strides[MAX_OPERAND_SIZE]; int strides_size;
  Avgpoolpadding_Type padding_type;
  
}Tf_Avgpool;

REGISTER_KERNEL(Tf_Avgpool)