
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

typedef struct Mxnet_Upsampling {
  dvnet_tensor data[MAX_OPERAND_SIZE]; int data_size;
  int32_t scale;
  int32_t num_filter;
  UpSamplingsample_typeType sample_type;
  UpSamplingmulti_input_modeType multi_input_mode;
  int32_t num_args;
  int64_t workspace;
  
}Mxnet_Upsampling;

REGISTER_KERNEL(Mxnet_Upsampling)