'''
Copyright (c) 2018 - 2025, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

from enum import Enum


class UPSP_TYPE(Enum):
  '''
  Enum Containing various upsampling types.
  '''
  LINEAR_3D = 0
  NEAREST_3D = 1
  NEAREST_4D = 2
  BILINEAR_FALSE = 3
  BILINEAR_TRUE = 4
  NEAREST_5D = 5
  TRILINEAR_FALSE = 6
  TRILINEAR_TRUE = 7

fusedActivation = Enum('fusedActivation', 'relu relu6 none')