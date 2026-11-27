
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

typedef struct Onnx_Generateproposals {
  dvnet_tensor scores;
  dvnet_tensor deltas;
  dvnet_tensor im_info;
  float spatial_scale;
  int32_t pre_nms_topN;
  int32_t post_nms_topN;
  float nms_thresh;
  float min_size;
  int32_t angle_bound_on;
  int32_t angle_bound_lo;
  int32_t angle_bound_hi;
  float clip_angle_thresh;
  int32_t legacy_plus_one;
  
}Onnx_Generateproposals;

REGISTER_KERNEL(Onnx_Generateproposals)