
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

typedef struct Mxnet_Batchnorm {
  dvnet_tensor data;
  dvnet_tensor gamma;
  dvnet_tensor beta;
  dvnet_tensor moving_mean;
  dvnet_tensor moving_var;
  float eps;
  float momentum;
  int32_t fix_gamma;
  int32_t use_global_stats;
  int32_t output_mean_var;
  int32_t axis;
  int32_t cudnn_off;
  float min_calib_range;
  float max_calib_range;
  
}Mxnet_Batchnorm;

REGISTER_KERNEL(Mxnet_Batchnorm)