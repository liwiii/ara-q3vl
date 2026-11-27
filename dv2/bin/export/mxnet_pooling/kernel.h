
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

typedef struct Mxnet_Pooling {
  dvnet_tensor data;
  dvnet_shape kernel;
  Poolingpool_typeType pool_type;
  int32_t global_pool;
  int32_t cudnn_off;
  Poolingpooling_conventionType pooling_convention;
  dvnet_shape stride;
  dvnet_shape pad;
  int32_t p_value;
  int32_t count_include_pad;
  PoolinglayoutType layout;
  
}Mxnet_Pooling;

REGISTER_KERNEL(Mxnet_Pooling)