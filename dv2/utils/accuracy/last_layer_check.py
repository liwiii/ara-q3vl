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

import logging,os
from ctypes import *
logger = logging.getLogger(__name__)



class ClassificationItem(Structure):
    _fields_ = [('label', c_int),
                ('score', c_float)]

class ClassificationOutput(Structure):
    _fields_ = [('numOutputs', c_int)]

class Boxoutput(Structure):
    _fields_ = [('xmin', c_float),
                ('ymin', c_float),
                ('xmax', c_float),
                ('ymax', c_float),
                ('imageId',c_int),
                ('label',c_int),
                ('score', c_float)]

class DetectionOutput(Structure):
    _fields_ = [('numBoxesDetected', c_int)]

class last_layer_check(object):
    def __init__(self,tool1, tool2):
        self.tool1 = tool1
        self.tool2 = tool2

    def check(self,map_a,map_b,isclassification,box_a,box_b):
        for i in map_a:
            if(i not in map_b):
                logger.error("Tags mismatch for tags" + str(i))
            else:
                if len(map_a[i])!=len(map_b[i]):
                  logger.error("no of entries not matching for tag = " + str(i))
                
                else:
                  confmatch = 0
                  boxmatch = 0
                  for aidx in range(0,len(map_a[i])):
                    for bidx in range(0,len(map_b[i])):
                      if (abs(map_a[i][aidx]-map_b[i][bidx]))<=0.01:
                        confmatch+=1
                        if not isclassification:
                          match =1
                          for j in range(0,4):
                            if(abs(box_a[i][aidx][j]-box_b[i][bidx][j])>0.00001):
                              match = 0
                          boxmatch+=match
                  if confmatch < len(map_a[i]):
                    logger.error("confidence value changes for tag = " + str(i))
                  if not isclassification:
                    if boxmatch < len(map_a[i]):
                      logger.error("location value mismatch for tag = " + str(i))


    def last_layer_check(self, analyzer_file, simulator_file, network_type):
        analyzer_result=[]
        simulator_result=[]
        # enabling strict failure condition for classification
        # TODO for SSD once dvsim and analyzer are in sync
        isclassification = False
        if(network_type == "classification"):
            analyzer_result = self.classification_result(analyzer_file)
            simulator_result = self.classification_result(simulator_file)
            isclassification = True
        else:
            analyzer_result,box_analyzer = self.detectionout_result(analyzer_file)
            simulator_result,box_simulator = self.detectionout_result(simulator_file)
        map_a = {}
        box_a = {}
        for i in range(1,len(analyzer_result),2):
            if analyzer_result[i] not in map_a:
              map_a[analyzer_result[i]] = []
              box_a[analyzer_result[i]] = []
            map_a[analyzer_result[i]].append(analyzer_result[i+1])
            if not isclassification:
              box_a[analyzer_result[i]].append(box_analyzer[int(i/2)])
        map_b = {}
        box_b = {}
        for i in range(1,len(simulator_result),2):
            if simulator_result[i] not in map_b:
              map_b[simulator_result[i]] = []
              box_b[simulator_result[i]] = []
            map_b[simulator_result[i]].append(simulator_result[i+1])
            if not isclassification:
              box_b[simulator_result[i]].append(box_simulator[int(i/2)])
        self.check(map_a,map_b,isclassification,box_a,box_b)
        

    def classification_result(self, result_file):
        with open(result_file, 'rb') as file:
            result = []
            x = ClassificationOutput()
            if(file.readinto(x) == sizeof(x)):
                result.append(x.numOutputs)
            y = ClassificationItem() 
            cnt = 0
            while(file.readinto(y) == sizeof(y) and cnt < x.numOutputs):
                result.append(y.label)
                result.append(y.score)
                cnt=cnt+1
            return result

    def detectionout_result(self,result_file):
        with open(result_file, 'rb') as file:
            result = []
            x = DetectionOutput()
            boxes = []
            if(file.readinto(x) == sizeof(x)):
                result.append(x.numBoxesDetected)
            y = Boxoutput() 
            cnt = 0
            while(file.readinto(y) == sizeof(y) and cnt < x.numBoxesDetected):
                result.append(y.label)
                result.append(y.score)
                boxes.append([y.xmin,y.ymin,y.xmax,y.ymax])
                cnt=cnt+1
            return result, boxes