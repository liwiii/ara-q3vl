'''
Copyright (c) 2022-25, Kinara, Inc. All rights reserved.
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
import json
import sys
import cv2
import numpy as np


class Preprocessor():

  def __init__(self, brg2rgb, center_crop, resize_with_aspectratio, interpolation, input_shape, mean, scale):
    self.bgr2rgb = brg2rgb
    self.center_crop = center_crop
    self.resize_with_aspectratio = resize_with_aspectratio
    self.interpolation = interpolation
    self.input_shape = input_shape
    self.mean = mean
    self.scale = scale

  @staticmethod
  def fromJson(json):
    if json is None:
      return None
    return Preprocessor(json.get('bgr2rgb'), json.get('center_crop'), json.get('resize_with_aspectratio'), json.get('interpolation'), InputShape.fromJson(json.get('input_shape')),
                        Mean.fromJson(json.get('mean')), Scale.fromJson(json.get('scale')))

  def read_image(self, image):
    return cv2.imread(image)

  # def bgr2rgb(self, img):
  #   if self.input_shape:
  #     if not self.input_shape['c']:
  #       sys.exit('error: channels of input shape is required')
  #     if not self.input_shape['h']:
  #       sys.exit('error: height of input shape is required')
  #     if not self.input_shape['w']:
  #       sys.exit('error: width of input shape is required')
  #   else:
  #     sys.exit('error: input shape of network is required')
  #   # img = cv2.imread(image)
  #   if img.shape[2] == 3 and self.input_shape['c'] == 1:
  #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #   elif img.shape[2] == 1 and self.input_shape['c'] == 3:
  #     img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
  #   elif img.shape[2] == 4 and self.input_shape['c'] == 3:
  #     img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
  #   elif img.shape[2] == 4 and self.input_shape['c'] == 1:
  #     img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
  #   if self.bgr2rgb:
  #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  #   return img

  def preprocess(self, img):
    # img = self.bgr2rgb(image)
    if self.input_shape:
      if not self.input_shape['c']:
        sys.exit('error: channels of input shape is required')
      if not self.input_shape['h']:
        sys.exit('error: height of input shape is required')
      if not self.input_shape['w']:
        sys.exit('error: width of input shape is required')
    else:
      sys.exit('error: input shape of network is required')
    # img = cv2.imread(image)
    if img.shape[2] == 3 and self.input_shape['c'] == 1:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif img.shape[2] == 1 and self.input_shape['c'] == 3:
      img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4 and self.input_shape['c'] == 3:
      img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] == 4 and self.input_shape['c'] == 1:
      img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    if self.bgr2rgb:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2_interpol = cv2.INTER_LINEAR
    if self.interpolation:
      if self.interpolation == Interpolate.NN:
        cv2_interpol = cv2.INTER_NEAREST
      elif self.interpolation == Interpolate.CUBIC:
        cv2_interpol = cv2.INTER_CUBIC
      elif self.interpolation == Interpolate.AREA:
        cv2_interpol = cv2.INTER_AREA
      elif self.interpolation == Interpolate.LANCZOS4:
        cv2_interpol = cv2.INTER_LANCZOS4
    if self.resize_with_aspectratio and self.center_crop:
      img = self.apply_resize_with_aspectratio(img, self.input_shape['h'], self.input_shape['w'], cv2_interpol)
      img = self.apply_center_crop(img, self.input_shape['h'], self.input_shape['w'])
    elif self.resize_with_aspectratio:
      img = self.apply_resize_with_aspectratio(img, self.input_shape['h'], self.input_shape['w'], cv2_interpol)
    elif self.center_crop:
      img = self.apply_center_crop(img, self.input_shape['h'], self.input_shape['w'])
    else:
      img = cv2.resize(img, (self.input_shape['w'], self.input_shape['h']), interpolation=cv2_interpol)
    img = np.asarray(img, dtype='float32')
    if self.mean:
      if self.input_shape['c'] == 3:
        if ('r' in self.mean) and ('g' in self.mean) and ('b' in self.mean):
          if self.bgr2rgb:
            img[..., 2] -= self.mean['b']
            img[..., 1] -= self.mean['g']
            img[..., 0] -= self.mean['r']
          else:
            img[..., 2] -= self.mean['r']
            img[..., 1] -= self.mean['g']
            img[..., 0] -= self.mean['b']
      elif self.input_shape['c'] == 1:
        if 'r' in self.mean:
          img[..., 0] -= self.mean['r']
        elif 'g' in self.mean:
          img[..., 0] -= self.mean['g']
        elif 'b' in self.mean:
          img[..., 0] -= self.mean['b']
        else:
          sys.exit('error: mean value not provided')
      else:
        sys.exit('error: unsupported channels in input shape - ' + self.input_shape['c'])
    if self.scale:
      if self.input_shape['c'] == 3:
        if self.scale['r'] and self.scale['g'] and self.scale['b']:
          if self.bgr2rgb:
            img[..., 2] *= self.scale['b']
            img[..., 1] *= self.scale['g']
            img[..., 0] *= self.scale['r']
          else:
            img[..., 2] *= self.scale['r']
            img[..., 1] *= self.scale['g']
            img[..., 0] *= self.scale['b']
        else:
          sys.exit('error: insufficient scale values provided')
      elif self.input_shape['c'] == 1:
        if self.scale['r']:
          img[..., 0] *= self.scale['r']
        elif self.scale['g']:
          img[..., 0] *= self.scale['g']
        elif self.scale['b']:
          img[..., 0] *= self.scale['b']
        else:
          sys.exit('error: scale value not provided')
      else:
        sys.exit('error: unsupported channels in input shape - ' + self.input_shape['c'])
    if len(img.shape) < 3:
      img = img.reshape(list(img.shape) + [1] * (3 - len(img.shape)))
    return img

  def apply_resize_with_aspectratio(self, img, input_height, input_width, inter_pol=cv2.INTER_LINEAR):
    img_height, img_width = img.shape[0], img.shape[1]
    aspect = img_width * 1.0 / img_height
    if aspect > 1:
      new_height = int(aspect * input_height)
      img = cv2.resize(img, (input_width, new_height), interpolation=inter_pol)
    elif aspect < 1:
      new_width = int(input_width / aspect)
      img = cv2.resize(img, (new_width, input_height), interpolation=inter_pol)
    else:
      img = cv2.resize(img, (input_width, input_height), interpolation=inter_pol)
    return self.apply_center_crop(img, input_height, input_width)

  def apply_center_crop(self, img, input_height, input_width):
    img_height, img_width = img.shape[0], img.shape[1]
    left = int((img_width - input_width) / 2)
    right = int((img_width + input_width) / 2)
    top = int((img_height - input_height) / 2)
    bottom = int((img_height + input_height) / 2)
    img = img[top:bottom, left:right]
    return img


class Mean():

  def __init__(self, r, g, b):
    self.r = r
    self.g = g
    self.b = b

  @staticmethod
  def fromJson(json):
    if json is None:
      return None
    return Mean(json.get('r'), json.get('g'), json.get('b'))


class Scale():

  def __init__(self, r, g, b):
    self.r = r
    self.g = g
    self.b = b

  @staticmethod
  def fromJson(json):
    if json is None:
      return None
    return Scale(json.get('r'), json.get('g'), json.get('b'))


class InputShape():

  def __init__(self, c, h, w):
    self.c = c
    self.h = h
    self.w = w

  @staticmethod
  def fromJson(json):
    if json is None:
      return None
    return InputShape(json.get('c'), json.get('h'), json.get('w'))


class Interpolate(Enum):
  NN = 0
  LINEAR = 1
  CUBIC = 2
  AREA = 3
  LANCZOS4 = 4


#def fromJson(data, obj, **kwargs):
#  return obj.fromJson(json.loads(data, **kwargs))
if __name__ == '__main__':
  f = open('preprocess.json', 'r')
  j = json.load(f)
  p = Preprocessor.fromJson(j)
