
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

typedef struct Onnx_Averagepool {
  dvnet_tensor X;
  int32_t ceil_mode;
  int32_t count_include_pad;
  int32_t kernel_shape[MAX_OPERAND_SIZE]; int kernel_shape_size;
  int32_t pads[MAX_OPERAND_SIZE]; int pads_size;
  int32_t strides[MAX_OPERAND_SIZE]; int strides_size;
  
}Onnx_Averagepool;

REGISTER_KERNEL(Onnx_Averagepool)