'''
Copyright (c) 2018-22, Deep Vision, Inc. All rights reserved.
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

from numpy.random import RandomState, SeedSequence
from numpy.random import MT19937
import tensorflow.compat.v1 as tf 
import time
import sys as _sys
import os
import numpy as np
import collections 
import re
import warnings
import pandas as pd     
import json
_sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../../../analyzer/python')
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../python')
_sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../../python/caffe/proto')
import tensorflow.lite as lite
import caffe 
from tensorflow.python.framework import dtypes
from dv_logging import get_logger
import cv2
# tf.disable_v2_behavior()
logger = get_logger(__name__, 'DVOPVL')


class tflite2caffe :

    def compare_outputs(args, mse_logs) :
        
        if args.config:
            pre_dir = os.path.dirname(os.path.abspath(__file__))+'/../utils/examples/preprocessed/'
            images = os.listdir(pre_dir)
            output_model = os.path.join(args.output_dir,'params', args.output_model_name)
            caffe_net = caffe.Net(output_model+'.prototxt', output_model+'.caffemodel',caffe.TEST)
            
            out_node_names = args.output_node_name
            input_shape = []
            if args.input_shape != None:
                for x in args.input_shape:
                    shape = x.split(',')
                    shape = shape[1:]
                    input_shape.append([int(x) for x in shape]) 

            assert len(args.input_node_name) == 1 #currently single inputs in tflite are assumed by default
            for image in images:
                data_dict = dict()
                for i in range(len(args.input_node_name)):
                    assert input_shape[i][0] == 3
                    data_dict[args.input_node_name[i]] = cv2.resize(np.fromfile(os.path.join(pre_dir,image),np.float32).reshape(300,300,3),tuple(input_shape[i][1:])).reshape(tuple([1]+input_shape[i][1:]+[input_shape[i][0]]))# np.random.random_sample(tuple([1]+input_shape[i][1:] + [input_shape[i][0]])).astype(np.float32)
                for i in args.input_node_name :
                    caffe_net.blobs[i.replace('/','_')].data[...] = data_dict[i].transpose((0,3,1,2))
                caffe_out = caffe_net.forward()

                ip = lite.Interpreter(args.tflite_graph)
                ip.allocate_tensors()
                inp_id = ip.get_input_details()[0]["index"]
                out_det = ip.get_output_details()
                out_id0 = out_det[0]["index"]
                out_id1 = out_det[1]["index"]
                out_id2 = out_det[2]["index"]
                out_id3 = out_det[3]["index"]
            
                for name in data_dict:
                    img = data_dict[name]
                    ip.set_tensor(inp_id, img)
                    ip.invoke()
                    boxes = ip.get_tensor(out_id0)
                    classes = ip.get_tensor(out_id1)
                    scores = ip.get_tensor(out_id2)
                    num_det = ip.get_tensor(out_id3) 
                
                num = int(num_det[0])
                emax_max_tolerance = 0.0001 # Fix metric. 
                dets_stat = {
                'tfclass' : [],
                'caffeclass' : [],
                'tfscore' : [],
                'caffescore' : [],
                'tfbox': [],
                'caffebox': []  
                }

                exitcode = 0
                caffe_sorted = caffe_out['detection_out'][0][0]
                caffe_sorted = sorted(caffe_sorted, key=lambda a_entry: a_entry[2])[::-1]
                if num != caffe_out['detection_out'].shape[2] :
                    logger.error("validation failed, unequal no. of detections = tf:%s vs caffe:%s",num,caffe_out['detection_out'].shape[2] )
                    for i in range(num):
                        cl_tf = int(classes[0][i])+1
                        box_tf = boxes[0][i]
                        score_tf = scores[0][i]
                        logger.info("TF:cl:%s ; score:%s ; box:%s",cl_tf,score_tf,box_tf)
                    for i in range(caffe_out['detection_out'].shape[2]):
                        cl_caffe = int(caffe_sorted[i][1])
                        score_caffe = caffe_sorted[i][2]
                        box_caffe = caffe_sorted[i][3:]
                        logger.info("CAFFE:cl:%s ; score:%s ; box:%s",cl_caffe,score_caffe,box_caffe)

                    exitcode= -1
                else:
                    for i in range(num):
                        cl_caffe = int(caffe_sorted[i][1])
                        cl_tf = int(classes[0][i])+1
                        box_caffe = caffe_sorted[i][3:]
                        box_tf = boxes[0][i]
                        score_caffe = caffe_sorted[i][2]
                        score_tf = scores[0][i]

                        dets_stat['tfclass'].append(cl_tf)
                        dets_stat['caffeclass'].append(cl_tf)
                        dets_stat['tfscore'].append(score_tf)
                        dets_stat['caffescore'].append(score_caffe)
                        dets_stat['tfbox'].append(box_tf)
                        dets_stat['caffebox'].append(box_caffe)

                        if cl_caffe != cl_tf :
                            logger.error("validation failed, class mismatch = tf:%s vs caffe:%s",cl_tf,cl_caffe ) 
                            exitcode = -1
                        if abs(score_caffe - score_tf).max() > emax_max_tolerance:
                            logger.error("validation failed, score mismatch = tf:%s vs caffe:%s",score_tf,score_caffe ) 
                            exitcode = -1                            
                        if abs(box_caffe - box_tf).max() > emax_max_tolerance:
                            logger.error("validation failed, box mismatch = tf:%s vs caffe:%s",box_tf,box_caffe ) 
                            exitcode = -1                   
                    
                    df = pd.DataFrame.from_dict(dets_stat)
                    with open(str(output_model)+'.csv', 'w') as f : 
                        df.to_csv(f,index=False)
                
                if exitcode == -1 :
                    for name in data_dict:
                        np.save(output_model+'__'+name+'.npy',data_dict[name])
                    return exitcode

            logger.info("Validation Ok!")
            return exitcode
        else:
            # for k in range(1):
                # print("Currently running {}".format(str(k+1)))
            output_model = os.path.join(args.output_dir, 'params',args.output_model_name)
            caffe_net = caffe.Net(output_model+'.prototxt', output_model+'.caffemodel',caffe.TEST)
            
            ip = lite.Interpreter(args.input_weight, experimental_preserve_all_tensors = True)

            data_dict = dict()
            np.random.seed(round(time.time()*1000) % 2**31)

            for i in range(len(ip.get_input_details())):
                loc_name = 'loc_' + str(ip.get_input_details()[i]['index'])
                if args.dtype == "int8":
                    data_dict[loc_name] = np.random.randint(-128, high=127, size=tuple(ip.get_input_details()[i]['shape']), dtype=np.int8)
                else:
                    data_dict[loc_name] = np.random.random_sample(tuple(ip.get_input_details()[i]['shape'])).astype(np.float32)
                if len(data_dict[loc_name].shape) == 4:
                    caffe_net.blobs[loc_name].data[...] = data_dict[loc_name].transpose((0,3,1,2)) 
                else:
                    caffe_net.blobs[loc_name].data[...] = data_dict[loc_name]
                               

            caffe_out = caffe_net.forward()
            ip.allocate_tensors()
            
            output_details = ip.get_output_details()
        
            for index, name in enumerate(data_dict):
                inp_id = ip.get_input_details()[index]["index"]
                img = data_dict[name]
                ip.set_tensor(inp_id, img)
                #np.savetxt("/auto/work/saketh/dev/sw/candid/test/networks/dv_cfg0/tflite/layer/conv1x1/input_{}.txt".format(str(index)), img.flatten(), fmt="%d")
                ip.invoke()
                            
            if len(output_details) != len(caffe_out.keys()):
                logger.error("unequal number of outputs::tflite:%d,caffe:%d",len(output_details),len(caffe_out.keys()))
            
            error_amax = list()
            error_pamx = list()

            possible_suffix = ['', '_crop', '_crop_crop', '_padding_crop', '_padding', '_scale', '_reshape1', '_reshape',
                               '_splitOutput', '_output0' , '_output1', '_output2', '_state_output', '_statecell_output','_output0',
                               '_perm_caffe', '_reshape_caffe', '_perm1'] 
            
            for opdict in output_details:
                found = False
                loc_name = 'loc_' + str(opdict['index']) 
                for suffix in possible_suffix:
                    if (loc_name + suffix) in caffe_out:
                        loc_name = loc_name + suffix
                        found =True
                if not found:
                    mse_logs.append("output node %s not in caffe",loc_name)
                    exitcode = -1
                    continue
                caffe_data = caffe_out[loc_name]
                tflite_data = ip.get_tensor(opdict['index'])
                
                if len(tflite_data.shape) == 4:
                    tflite_data = tflite_data.transpose(0, 3, 1, 2)
                if caffe_data.shape == tflite_data.shape :
                    diff = abs(tflite_data - caffe_data)
                    MSE = np.square(diff).mean()
                    E_AMAX = diff.max()
                    E_PAMX = E_AMAX/np.abs(tflite_data).mean()
                    error_amax.append(E_AMAX)
                    error_pamx.append(E_PAMX)
                    mse_logs.append("\n\nOutput TFLite layer:    {}\nOutput Caffe Layer: {}\n--------------------\nMSE: {}\nE_AMAX: {}\nE_PAMX: {}\n".format(index, loc_name, MSE, E_AMAX, E_PAMX))
                    logger.debug("Error stats: amean: %s, mean: %s, min: %s, max: %s, std: %s", np.abs(diff).mean(), diff.mean(), diff.min(), diff.max(), diff.std())
                    logger.debug("TFLite blob stats: amean: %s, mean: %s, min: %s, max: %s, std: %s", np.abs(tflite_data).mean(), tflite_data.mean(), tflite_data.min(), tflite_data.max(), tflite_data.std())                    
                else:
                    mse_logs.append("Shapes Mismatch : %s :: %s || Layer_caffe  %s :: Layer_tflite :: %s".format(caffe_data.shape,tflite_data.shape,loc_name,index))
            # del caffe_net
            # del ip

            flag_count = 0
            emax_max_tolerance = 0.0001 # Fix metric. 
            for ind in range(len(error_amax)):
                if error_amax[ind] < emax_max_tolerance or error_pamx[ind] < emax_max_tolerance:
                    flag_count += 1
                    
            if flag_count < len(output_details) : 
                for key in data_dict: 
                    with open(output_model + '_' + key + '.npy', 'wb') as f:
                        np.save(f,data_dict[key])
                logger.error("Validation failed, EMAX > emax_max_tolerance, emax_max_tolerance = %s", emax_max_tolerance)
                if args.ignore_mse:
                    logger.warn("ignoring mse")
                    return 0
                return -1
            logger.info("Validation Ok!")
            return 0