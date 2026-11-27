#!/usr/bin/env python3

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

import argparse, csv, os, yaml, logging, sys, re, math, filecmp, numpy, enum, tabulate, json
import last_layer_check
csv.field_size_limit(sys.maxsize)
logger = logging.getLogger(__name__)

class Status(enum.Enum):
  passed = 0
  failed = 1
  skipped = 2
  detections_output_size_mismatch = 3

class Framework(str, enum.Enum):
  caffe = "caffe"
  mxnet = "mxnet"
  tensorflow = "tensorflow"
  pytorch = "pytorch"
  onnx = "onnx"
  torchscript = "torchscript"


class Accuracy():

  def __init__(self):
    self.config = {}
    self.args = {}
    self.images = []
    self.outputs_root = None

    self.layer_type = 'layer_type'
    self.layer_name = 'layer_name'
    self.layer_id = 'layer_id'
    self.layer_blobs = 'layer_blobs'
    self.layer_active = 'active'
    self.layer_fused_layers = 'fused_layers'
    self.layer_fused_parent = 'fused_parent'
    self.layer_float_output = 'is_float_output'
    self.layer_output_shape = "output_shape"
    self.layer_output_min_max = 'output_min_max'
    self.layer_fused_output = 'fused_output_layer'
    self.layer_is_input = 'is_input'
    self.layer_output_sign = 'output_sign'
    self.layer_min_error = 'min_error'
    self.layer_max_error = 'max_error'
    self.layer_quant_errors = 'has_quant_errors'
    self.output_type = 'output_type'

    self.analyzer_out = 'quantizer'
    self.converter_out = 'converter'
    self.simulator_out = 'simulator/layer_outputs'
    self.stats_dir = 'stats'
    self.srcnet_layer_index = 'index'
    self.srcnet_caffe_name = 'caffename'
    self.summary_file = 'summary.csv'
    self.mse_file = 'layer_mse.csv'
    self.quantization_errors = 'quantization_errors.csv'
    self.op_list_file = 'op_list.json'
    self.outputs_dir = 'outputs'
    self.inputs_dir = 'inputs'
    self.diff_bins = 32
    self.diff_bin_length = 'error_bin_length'

    self.layers_summary = 'layers_summary.txt'
    self.srcnet_dvnet_comparision = 'srcnet_dvnet_comparision.txt'
    self.dvnet_unfused_and_fused_comparision = 'dvnet_unfused_and_fused_comparision.txt'
    self.dvnet_float_and_quantized_comparision = 'dvnet_float_and_quantized_comparision.txt'
    self.dvnet_quantizer_and_simulator_comparision = 'dvnet_quantizer_and_simulator_comparision.txt'
    self.dvnet_float_and_quantized_diff_histogram = 'dvnet_float_and_quantized_diff_histogram.csv'
    self.flows_outputs_dir = 'data_comparisions'

    self.fused_layers = {}
    self.fused_parents = {}
    self.layer_min_max = {}
    self.src_dvnet_results = {}
    self.orig_fused_results = {}
    self.float_fixed_results = {}
    self.quantizer_simulator_results = {}
    self.layers_list = []
    self.dvnet_layers_meta = {}
    self.srcnet_id_to_name_map = {}
    self.srcnet_name_to_id_map = {}
    self.srcnet_to_dvnet_map = {}
    self.dvnet_to_srcnet_map = {}
    self.dvnet_id_to_name_map = {}
    self.dvnet_name_to_id_map = {}
    self.dvnet_orig_to_fused_map = {}
    self.dvnet_layer_blob_map = {}
    self.layer_results_summary = {}
    self.srcnet_meta = {}
    self.float_fixed_diff_histogram = {}

    self.verif_img_dir = ''

    logger = logging.getLogger()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

  def has_non_max_suppression(self, op_list: dict) -> bool:
      """
      Check whether NonMaxSuppression op exists in the op list.
      """
      return any(op == "NonMaxSuppression" for op in op_list)

  def load_and_normalize_vpu_cnn_outputs(self,a_file, b_file, output_shape_list, layer, is_float_layer):
      try:
          # Basic validation
          if not os.path.exists(a_file):
              raise FileNotFoundError(f"Analyzer output file not found: {a_file}")
          if not os.path.exists(b_file):
              raise FileNotFoundError(f"DVSIM output file not found: {b_file}")

          # Select dtype
          dtype = numpy.float32 if int(is_float_layer) == 1 else numpy.int8

          # Load raw data
          a_array = numpy.fromfile(a_file, dtype=dtype)
          b_array = numpy.fromfile(b_file, dtype=dtype)

          # Reshape
          a_array = a_array.reshape(
              output_shape_list[0], output_shape_list[1], output_shape_list[2]
          )
          b_array = b_array.reshape(
              output_shape_list[0], output_shape_list[1], output_shape_list[2]
          )

          # Detect invalid (-1 padded) rows
          a_array_invalid = (a_array[0] == -1).all(axis=1)
          b_array_invalid = (b_array[0] == -1).all(axis=1)

          a_cut = numpy.argmax(a_array_invalid) if a_array_invalid.any() else a_array.shape[1]
          b_cut = numpy.argmax(b_array_invalid) if b_array_invalid.any() else b_array.shape[1]

          # Trim
          a_trim = a_array[0, :a_cut+1, :]
          b_trim = b_array[0, :b_cut+1, :]

          # Lexicographic sort
          a_sort_idx = numpy.lexsort(a_trim.T[::-1])
          b_sort_idx = numpy.lexsort(b_trim.T[::-1])

          a_sorted = a_trim[a_sort_idx]
          b_sorted = b_trim[b_sort_idx]

          return a_sorted, b_sorted

      except Exception as e:
          logger.error(
              "Failed to load and normalize VPU CNN outputs | "
              f"analyzer_file={a_file}, dvsim_file={b_file}, "
              f"shape={output_shape_list}, layer={layer}, is_float_layer={is_float_layer}",
              exc_info=True,
          )



  def compare_raw_outputs(self, src, dst, results, image, layer, blob):
    if filecmp.cmp(src, dst, shallow=False):
      snr, mse, min_diff, max_diff = 1e+6, 0, 0, 0
    else:
      snr, mse, min_diff, max_diff = 1e-3, 1e+3, 1e+3, 1e+3
    if image not in results:
      results[image] = {}
    if layer not in self.quantizer_simulator_results[image]:
      results[image][layer] = {}
    if blob not in self.quantizer_simulator_results[image][layer]:
      results[image][layer][blob] = {}
    results[image][layer][blob]['snr'] = snr
    results[image][layer][blob]['mse'] = mse
    results[image][layer][blob]['min_diff'] = min_diff
    results[image][layer][blob]['max_diff'] = max_diff
    return results[image][layer][blob]

  def compare_float_outputs(self, src, dst, results, image, layer, blob, numpyload=False, histogram_map=None, layer_output_shape=[], is_vpu_output_comparsion=False):
    if numpyload:
      src_data = numpy.load(src)
      dst_data = numpy.load(dst)
      if self.config['network']['srcfw'] == 'tensorflow':
        if len(src_data.shape) == 4 :
          src_data = src_data.transpose(0,3,1,2)
        elif len(src_data.shape) == 3 :
          src_data = src_data.transpose(0,2,1)
    else:
      if is_vpu_output_comparsion:
        if not layer_output_shape:
          sys.exit("layer {} output shape is empty".format(layer))
        src_data,dst_data = self.load_and_normalize_vpu_cnn_outputs(src,dst, layer_output_shape[:-1], layer,True)
      else:
        with open(src, 'rb') as src_output:
          src_data = numpy.fromfile(src_output, dtype=numpy.float32)
        with open(dst, 'rb') as dst_output:
          dst_data = numpy.fromfile(dst_output, dtype=numpy.float32)
      if len(src_data) != len(dst_data) or len(src_data) == 0:
        if len(src_data) != len(dst_data):
          logger.error("number of elements are not matching between files %s %s", src, dst)
          return Status.detections_output_size_mismatch
        elif len(src_data) == 0:
          logger.error("no data found in file %s", src)
        return 0, 0
    src_data = numpy.nan_to_num(src_data)
    dst_data = numpy.nan_to_num(dst_data)
    src_all_zeros = not numpy.any(src_data)
    dst_all_zeros = not numpy.any(dst_data)
    diff = src_data - dst_data
    min_diff = diff.min()
    max_diff = diff.max()
    if histogram_map is not None:
      max_diff_val = abs(min_diff) if abs(min_diff) > abs(max_diff) else abs(max_diff)
      hist, bins = numpy.histogram(numpy.abs(diff), bins=self.diff_bins, range=(0,max_diff_val))
      if layer not in self.float_fixed_diff_histogram:
        self.float_fixed_diff_histogram[layer] = {}
      if blob not in self.float_fixed_diff_histogram[layer]:
        self.float_fixed_diff_histogram[layer][blob] = {}
      self.float_fixed_diff_histogram[layer][blob][image] = []
      self.float_fixed_diff_histogram[layer][blob][image].append(layer+'_'+ blob + '_' + image)
      self.float_fixed_diff_histogram[layer][blob][image].append(max_diff_val)
      self.float_fixed_diff_histogram[layer][blob][image].append(max_diff_val/self.diff_bins)
      self.float_fixed_diff_histogram[layer][blob][image].extend(hist)
    signal = src_data ** 2
    error = diff ** 2
    snr = numpy.nanmean(signal)/numpy.nanmean(error+0.000001)
    mse = numpy.sqrt(numpy.nanmean(diff ** 2))

    if image not in results:
      results[image] = {}
    if layer not in results[image]:
      results[image][layer] = {}
    if blob not in results[image][layer]:
      results[image][layer][blob] = {}

    results[image][layer][blob]['snr'] = snr
    results[image][layer][blob]['mse'] = mse
    results[image][layer][blob]['min_diff'] = min_diff
    results[image][layer][blob]['max_diff'] = max_diff
    results[image][layer][blob]['all_zeros'] = True if src_all_zeros or dst_all_zeros else False
    return results[image][layer][blob]

  def set_srcnet_meta(self):
    srcfw = self.config['network']['srcfw']
    if srcfw == Framework.tensorflow:
      self.srcnet_meta['colname'] = 'tfname'
      self.srcnet_meta['output'] = 'tf'
    elif srcfw == Framework.mxnet:
      self.srcnet_meta['colname'] = 'mxname'
      self.srcnet_meta['output'] = 'mxnet'
    elif srcfw == Framework.caffe:
      self.srcnet_meta['colname'] = 'caffename'
      self.srcnet_meta['output'] = 'caffe'
    elif srcfw == Framework.onnx:
      self.srcnet_meta['colname'] = 'onnxname'
      self.srcnet_meta['output'] = 'onnx'
    elif srcfw == Framework.pytorch:
      self.srcnet_meta['colname'] = 'pytorchname'
      self.srcnet_meta['output'] = 'pytorch'
    elif srcfw == Framework.torchscript:
      self.srcnet_meta['colname'] = 'torchscriptname'
      self.srcnet_meta['output'] = 'torchscript'
    else:
      sys.exit('error: unsupported framework {}'.format(srcfw))

  def parse_converter_csv(self, file):
    src_colname = self.srcnet_meta['colname']
    with open(file, newline='\n') as csvfile:
      layers = csv.DictReader(csvfile)
      layerid = 0
      for layer in layers:
        if 'Shape_Match' in layer and  layer["Shape_Match"] == "NO":
            continue
        if src_colname not in layer and 'onnxname' in layer:
          # Case: Tensorflow fallback to onnx within the converter
          src_colname = 'onnxname'
        if layer[src_colname] not in self.srcnet_name_to_id_map:
          self.srcnet_id_to_name_map[layerid] = layer[src_colname]
          self.srcnet_name_to_id_map[layer[src_colname]] = layerid
          layerid += 1
        self.srcnet_to_dvnet_map[layer[src_colname]] = layer[self.srcnet_caffe_name]
        self.dvnet_to_srcnet_map[layer[self.srcnet_caffe_name]] = layer[src_colname]

  def parse_srcnet_layer_meta(self):
    srcfw = self.config['network']['srcfw']
    if srcfw == Framework.tensorflow or srcfw == Framework.mxnet:
      converter_csv_dir = os.path.join(self.outputs_root, self.converter_out, self.stats_dir)
      if not os.path.exists(converter_csv_dir):
        sys.exit('error: converter stats directory does not exist')
      img_list = os.listdir(self.images)
      found_csv = False
      for img in img_list:
        converter_csv_file = os.path.join(converter_csv_dir, img + '.csv')
        if os.path.exists(converter_csv_file):
          self.parse_converter_csv(converter_csv_file)
          found_csv = True
          break
      if not found_csv:
        if 'data_format' in self.config['dvconvert'] and self.config['dvconvert']['data_format'] == 'TNC':
          logger.error("converter mse file is currently not generated for data_format - TNC")
        elif 'pcfg' in self.config['network'] and self.config['network']['pcfg']:
          logger.error("converter mse file is currently not generated for detection type network in tensorflow")
        else:
          logger.error('error: could not find converter stats file')
    else:
      logger.error("layer mapping is not available for framework %s", srcfw)

  def get_fused_layers_list(self, fused_layers):
    layers = fused_layers.split('\r')
    fused_layers = []
    for layer in layers:
      if layer:
        id_name = re.split('\s+', layer)
        name = id_name[1].split(':')[1]
        fused_layers.append(name)
    return fused_layers

  def get_layer_blobs_list(self, layer_blobs):
    blobs = layer_blobs.split("\r")
    layer_blobs = []
    for blob in blobs:
      if blob:
        id_name = re.split('\s+', blob)
        id = id_name[0].split(':')[1]
        name = id_name[1].split(':')[1]
        layer_blobs.append((id, name))
    return layer_blobs

  def parse_dvnet_quantization_errors(self):
    quantization_errors_file = os.path.join(self.outputs_root, self.analyzer_out, self.stats_dir, self.quantization_errors)
    if not os.path.exists(quantization_errors_file):
      sys.exit('error: quantization errors file does not exist in quantizer output directory')

    with open(quantization_errors_file, newline='\n') as csvfile:
      layers = csv.DictReader(csvfile, skipinitialspace=True)
      for layer in layers:
        if layer[self.layer_type] == 'DummyData' or layer[self.layer_type] == 'Split' or layer[self.layer_type] == 'DVShift' or layer[self.layer_type] == 'DVCast':
          continue
        self.dvnet_layers_meta[layer[self.layer_name]][self.layer_quant_errors] = True

  def parse_dvnet_layers_summary(self):
    summary_csv_file = os.path.join(self.outputs_root, self.analyzer_out, self.stats_dir, self.summary_file)
    if not os.path.exists(summary_csv_file):
      sys.exit('error: network summary file does not exist in quantizer output directory')

    with open(summary_csv_file, newline='\n') as csvfile:
      layers = csv.DictReader(csvfile, skipinitialspace=True)
      for layer in layers:
        if layer[self.layer_type] == 'DummyData' or layer[self.layer_type] == 'Split' or layer[self.layer_type] == 'DVShift' or layer[self.layer_type] == 'DVCast':
          continue
        self.layers_list.append(layer[self.layer_name])
        self.dvnet_id_to_name_map[int(layer[self.layer_id])] = layer[self.layer_name]
        self.dvnet_name_to_id_map[layer[self.layer_name]] = int(layer[self.layer_id])
        self.dvnet_layers_meta[layer[self.layer_name]] = {}
        self.dvnet_layers_meta[layer[self.layer_name]][self.layer_type] = layer[self.layer_type]
        self.dvnet_layers_meta[layer[self.layer_name]][self.layer_output_sign] = int(layer[self.layer_output_sign]) if int(layer[self.layer_active]) else 0
        self.dvnet_layers_meta[layer[self.layer_name]][self.layer_active] = int(layer[self.layer_active])
        self.dvnet_layers_meta[layer[self.layer_name]][self.layer_is_input] = int(layer[self.layer_is_input]) if int(layer[self.layer_active]) else 0
        self.dvnet_layers_meta[layer[self.layer_name]][self.layer_float_output] = int(layer[self.layer_float_output]) if int(layer[self.layer_active]) else 0
        self.dvnet_layers_meta[layer[self.layer_name]][self.layer_output_shape] = [int(p) for p in layer[self.layer_output_shape].split("_")] if int(layer[self.layer_active]) else 0
        if layer[self.layer_blobs].strip() and layer[self.layer_blobs].strip().replace('"',''):
          layer_blobs = layer[self.layer_blobs].strip().replace('"','')
          layer_blobs = self.get_layer_blobs_list(layer_blobs)
          self.dvnet_layer_blob_map[layer[self.layer_name]] = layer_blobs
        if int(layer[self.layer_active]):
          if layer[self.layer_output_min_max]=='':
            self.layer_min_max[layer[self.layer_name]] = {}
            self.layer_min_max[layer[self.layer_name]]['min'] = 0
            self.layer_min_max[layer[self.layer_name]]['max'] = 0
          else:
            min_max = re.split('\s+',layer[self.layer_output_min_max])
            min_val = float(min_max[0].split(':')[1])
            max_val = float(min_max[1].split(':')[1])
            self.layer_min_max[layer[self.layer_name]] = {}
            self.layer_min_max[layer[self.layer_name]]['min'] = min_val
            self.layer_min_max[layer[self.layer_name]]['max'] = max_val
        if layer[self.layer_fused_parent].strip() and layer[self.layer_fused_parent].strip().replace('"',''):
          fused_parent = layer[self.layer_fused_parent].strip().replace('"','')
          id_name = re.split('\s+', fused_parent)
          fused_parent_name = re.split(':', id_name[1])[1]
          self.dvnet_layers_meta[layer[self.layer_name]][self.layer_fused_parent] = fused_parent_name
          self.fused_layers[layer[self.layer_name]] = fused_parent_name
          self.fused_parents[fused_parent_name] = 1
        if layer[self.layer_fused_layers].strip() and layer[self.layer_fused_layers].strip().replace('"',''):
          fused_layers = layer[self.layer_fused_layers].strip().replace('"','')
          fused_layers = self.get_fused_layers_list(fused_layers)
          self.dvnet_layers_meta[layer[self.layer_name]][self.layer_fused_layers] = fused_layers
        if layer['output_type'] != '':
          self.output_type = layer['output_type']
        else:
          self.output_type = ''

          #self.fused_layers_comparision_map[layer[self.layer_name]] = mapped_layer

  def get_status_from_code(self, code):
    if code == Status.passed:
      return 'passed'
    if code == Status.failed:
      return 'failed'
    if code == Status.skipped:
      return 'skipped'
    if code == Status.detections_output_size_mismatch:
      return 'detections_output_size_mismatch'

  def emit_histogram(self, file, histogram):
    header = ['layer','abs_max_diff','bin_length']
    for i in range(self.diff_bins):
      header.append('bin_' + str(i))
    with open(file,'w') as csvfile:
      writer = csv.writer(csvfile)
      writer.writerow(header)
      for layer in self.layers_list:
        if layer in histogram:
          for blob in histogram[layer]:
            for image in histogram[layer][blob]:
              writer.writerow(histogram[layer][blob][image])
    csvfile.close()

  def get_results_status(self, results, thresholds):
    if (thresholds['mse'] and results['mse'] > thresholds['mse']) or \
        (thresholds['snr'] and results['snr'] < thresholds['snr']) or \
        (thresholds['abs_diff'] and (abs(results['min_diff']) > thresholds['abs_diff'] or abs(results['max_diff']) > thresholds['abs_diff'])) or \
        ('all_zeros' in results and results['all_zeros']):
      return Status.failed
    else:
      return Status.passed

  def compare_srcnet_and_dvnet(self, image, layer):
    status = []
    if layer in self.dvnet_to_srcnet_map and self.srcnet_to_dvnet_map[self.dvnet_to_srcnet_map[layer]] == layer:
      if layer in self.dvnet_layer_blob_map:
        for blob in self.dvnet_layer_blob_map[layer]:
          if blob[1] in self.dvnet_to_srcnet_map:
            dvnet_layer_output = os.path.join(self.outputs_root, self.converter_out, self.converter_out, 'debug', image, 'outputs', 'dvnet_fp32', blob[1].replace('/','_') + '.npy')
            srcnet_layer_output = os.path.join(self.outputs_root, self.converter_out, self.converter_out, 'debug', image, 'outputs', self.srcnet_meta['output'] + '_fp32', self.dvnet_to_srcnet_map[blob[1]].replace('/','_') + '.npy')
            if os.path.exists(dvnet_layer_output) and os.path.exists(srcnet_layer_output):
                status.append(self.get_results_status(self.compare_float_outputs(srcnet_layer_output, dvnet_layer_output, self.src_dvnet_results, image, layer, blob[1], True), self.config['thresholds']['source_dvnet_fp32_comparision']))
            else:
              status.append(Status.skipped)
          else:
            status.append(Status.skipped)
      else:
        status.append(Status.skipped)
    else:
      status.append(Status.skipped)
    return status

  def compare_unfused_and_fused_dvnet(self, image, layer):
    unfused_layer = layer
    status = []
    if self.dvnet_layers_meta[layer]['active']:
      if self.layer_fused_layers in self.dvnet_layers_meta[layer]:
        for l in reversed(self.dvnet_layers_meta[layer][self.layer_fused_layers]):
          if self.dvnet_layers_meta[l]['active']:
            continue
          else:
            unfused_layer = l
            break
      if layer in self.dvnet_layer_blob_map and unfused_layer in self.dvnet_layer_blob_map:
        if (len(self.dvnet_layer_blob_map[layer]) == len(self.dvnet_layer_blob_map[unfused_layer])):
          for idx in range(len(self.dvnet_layer_blob_map[layer])):
            dvnet_unfused_fp32_layer = os.path.join(self.outputs_root, self.analyzer_out, (self.inputs_dir if self.dvnet_layers_meta[layer][self.layer_is_input] else self.outputs_dir), image, 'dvnet_fp32', unfused_layer + '-' + str(self.dvnet_name_to_id_map[unfused_layer]) + '-' + self.dvnet_layer_blob_map[unfused_layer][idx][1] + '-' + self.dvnet_layer_blob_map[unfused_layer][idx][0])
            dvnet_fused_fp32_layer = os.path.join(self.outputs_root, self.analyzer_out, (self.inputs_dir if self.dvnet_layers_meta[layer][self.layer_is_input] else self.outputs_dir), image, 'fp32', layer + '-' + str(self.dvnet_name_to_id_map[layer]) + '-' + self.dvnet_layer_blob_map[layer][idx][1] + '-' + self.dvnet_layer_blob_map[layer][idx][0])
            if os.path.exists(dvnet_unfused_fp32_layer) and os.path.exists(dvnet_fused_fp32_layer):
              status.append(self.get_results_status(self.compare_float_outputs(dvnet_unfused_fp32_layer, dvnet_fused_fp32_layer, self.orig_fused_results, image, layer, self.dvnet_layer_blob_map[layer][idx][1]), self.config['thresholds']['dvnet_unfused_fused_fp32_comparision']))
            else:
              status.append(Status.skipped)
      else:
        status.append(Status.skipped)
    else:
      status.append(Status.skipped)
    return status

  def compare_float_and_quantized(self, image, layer):
    status = []
    if self.dvnet_layers_meta[layer]['active']:
      if self.layer_fused_layers in self.dvnet_layers_meta[layer]:
        last_layer = self.dvnet_layers_meta[layer][self.layer_fused_layers][len(self.dvnet_layers_meta[layer][self.layer_fused_layers])-1]
        if not self.dvnet_layers_meta[layer][self.layer_output_sign] and not self.dvnet_layers_meta[last_layer][self.layer_output_sign]:
          status.append(Status.skipped)
      if layer in self.dvnet_layer_blob_map:
        for blob in self.dvnet_layer_blob_map[layer]:
          dvnet_fp32_layer = os.path.join(self.outputs_root, self.analyzer_out, (self.inputs_dir if self.dvnet_layers_meta[layer][self.layer_is_input] else self.outputs_dir), image, 'fp32', layer + '-' + str(self.dvnet_name_to_id_map[layer]) + '-' + blob[1] + '-' + blob[0])
          dvnet_dequantized_layer = os.path.join(self.outputs_root, self.analyzer_out, (self.inputs_dir if self.dvnet_layers_meta[layer][self.layer_is_input] else self.outputs_dir), image, 'dequantized_fp32', layer + '-' + str(self.dvnet_name_to_id_map[layer]) + '-' + blob[1] + '-' + blob[0])
          if os.path.exists(dvnet_fp32_layer) and os.path.exists(dvnet_dequantized_layer):
            status.append(self.get_results_status(self.compare_float_outputs(dvnet_fp32_layer, dvnet_dequantized_layer, self.float_fixed_results, image, layer, blob[1], False, self.float_fixed_diff_histogram), self.config['thresholds']['dvnet_float_fixed_comparision']))
          else:
            status.append(Status.skipped)
      else:
        status.append(Status.skipped)
    else:
      status.append(Status.skipped)
    return status

  def compare_quantizer_and_simulator(self, image, layer):
    status = []
    if self.dvnet_layers_meta[layer]['active']:
      if self.layer_fused_parent in self.dvnet_layers_meta[layer] or self.dvnet_layers_meta[layer][self.layer_is_input]:
        status.append(Status.skipped)
      else:
        if layer in self.dvnet_layer_blob_map:
          for blob in self.dvnet_layer_blob_map[layer]:
            dvnet_quantizer_layer = os.path.join(self.outputs_root, self.analyzer_out, self.outputs_dir, image, 'quantized', layer + '-' + str(self.dvnet_name_to_id_map[layer]) + '-' + blob[1] + '-' + blob[0])
            dvnet_simulator_layer = os.path.join(self.outputs_root, self.simulator_out, image, 'quantized', layer + '-' + str(self.dvnet_name_to_id_map[layer]) + '-' + blob[1] + '-' + blob[0])
            if os.path.exists(dvnet_quantizer_layer) and os.path.exists(dvnet_simulator_layer):
              if self.output_type != '' and (self.output_type =="classification" or self.output_type== "detection"):
                llc = lo = last_layer_check.last_layer_check("analyzer","simulator")
                dvnet_quantizer_layer_dequantized = dvnet_quantizer_layer.replace("quantized","dequantized_fp32")
                tmp = llc.last_layer_check(dvnet_quantizer_layer_dequantized,dvnet_simulator_layer,self.output_type)
                if tmp != False:
                  status.append(Status.passed)
                else:
                  status.append(Status.failed)
              elif self.dvnet_layers_meta[layer][self.layer_float_output]:
                op_list_json_file = os.path.join(self.outputs_root, self.converter_out, self.stats_dir, self.op_list_file)
                if not os.path.exists(op_list_json_file):
                  sys.exit('error: op_list json file does not exist in converter stats directory')
                with open(op_list_json_file, "r") as file:
                  op_list = json.load(file)
                is_vpu_output_comparsion = self.has_non_max_suppression(op_list)
                results = self.compare_float_outputs(dvnet_quantizer_layer, dvnet_simulator_layer, self.quantizer_simulator_results, image, layer, blob[1], False, layer_output_shape=self.dvnet_layers_meta[layer][self.layer_output_shape], is_vpu_output_comparsion=is_vpu_output_comparsion)
                if results == Status.detections_output_size_mismatch:
                  status.append(Status.detections_output_size_mismatch)
                else:
                  if self.quantizer_simulator_results[image][layer][blob[1]]['mse'] > 1e-1:
                    status.append(Status.failed)
                  else:
                    status.append(Status.passed)
              else:
                self.compare_raw_outputs(dvnet_quantizer_layer, dvnet_simulator_layer, self.quantizer_simulator_results, image, layer, blob[1])
                if self.quantizer_simulator_results[image][layer][blob[1]]['mse'] != 0.0:
                  status.append(Status.failed)
                else:
                  status.append(Status.passed)
            else:
              status.append(Status.skipped)
        else:
          status.append(Status.skipped)
    else:
      status.append(Status.skipped)
    return status

  def compare_for_image(self, image):

    self.layer_results_summary['A'][image] = {}
    self.layer_results_summary['B'][image] = {}
    self.layer_results_summary['C'][image] = {}
    self.layer_results_summary['D'][image] = {}

    for layer in self.layers_list:
      self.layer_results_summary['A'][image][layer] = self.compare_srcnet_and_dvnet(image, layer)
      self.layer_results_summary['B'][image][layer] = self.compare_unfused_and_fused_dvnet(image, layer)
      self.layer_results_summary['C'][image][layer] = self.compare_float_and_quantized(image, layer)
      self.layer_results_summary['D'][image][layer] = self.compare_quantizer_and_simulator(image, layer)
  def get_results_summary(self, results):
    failed = 0
    skipped = 0
    passed = 0
    detections_output_size_mismatch = 0
    for layer in results:
      for idx in range(len(results[layer])):
        if results[layer][idx] == Status.passed:
          passed += 1
        elif results[layer][idx] == Status.failed:
          failed += 1
        elif results[layer][idx] == Status.detections_output_size_mismatch:
          detections_output_size_mismatch += 1
        else:
          skipped += 1
    if failed > 0:
      return Status.failed.name
    elif passed > 0:
      return Status.passed.name
    elif detections_output_size_mismatch > 0:
      return Status.detections_output_size_mismatch.name
    else:
      return Status.skipped.name

  def get_layer_summary(self, results, layer):
    failed = 0
    skipped = 0
    passed = 0
    images = os.listdir(self.images)
    for image in images:
      for idx in range(len(results[image][layer])):
        if results[image][layer][idx] == Status.passed:
          passed += 1
        elif results[image][layer][idx] == Status.failed:
          failed += 1
        else:
          skipped += 1
    if failed > 0:
      return Status.failed.name
    elif passed > 0:
      return Status.passed.name
    else:
      return Status.skipped.name

  def all_blobs_skipped(self, layer_results):
    for result in layer_results:
      if result != Status.skipped:
        return False
    return True

  def has_failures(self, layer_results):
    for result in layer_results:
      if result == Status.failed:
        return True
    return False

  def emit_error_metrics(self, key, results, file):
    table = []
    header = ['layer_image', 'blob_name', 'snr', 'mse', 'min_diff', 'max_diff']
    images = os.listdir(self.images)

    for layer in self.layers_list:
      layer_row = ['', '', '', '', '', '']
      for image in images:
        layer_row[0] += layer + '_' + image + ('**' if self.has_failures(self.layer_results_summary[key][image][layer]) else '')
        if self.all_blobs_skipped(self.layer_results_summary[key][image][layer]):
          layer_row[1] += '\n'
          layer_row[2] += self.get_status_from_code(self.layer_results_summary[key][image][layer][0]) + '\n'
          layer_row[3] += self.get_status_from_code(self.layer_results_summary[key][image][layer][0]) + '\n'
          layer_row[4] += self.get_status_from_code(self.layer_results_summary[key][image][layer][0]) + '\n'
          layer_row[5] += self.get_status_from_code(self.layer_results_summary[key][image][layer][0]) + '\n'
        elif results and image in results and layer in results[image]:
          for blob in results[image][layer]:
            layer_row[1] += blob + ('^^' if 'all_zeros' in results[image][layer][blob] and results[image][layer][blob]['all_zeros'] else '') + '\n'
            layer_row[2] += str(results[image][layer][blob]['snr']) + '\n'
            layer_row[3] += str(results[image][layer][blob]['mse']) + '\n'
            layer_row[4] += str(results[image][layer][blob]['min_diff']) + '\n'
            layer_row[5] += str(results[image][layer][blob]['max_diff']) + '\n'
        else:
          pass
        layer_row[0] += '\n'
      table.append(layer_row)
    with open(file, 'w') as f:
      f.write(tabulate.tabulate([["**", "layers with this suffix failed due to error metrics exceeding threshold limits"],
                                 ["^^", "layers with this suffix failed due to all zero layer outputs"],
                                 ["skipped", "indicates error metric checks are skipped either due to layer output files not present or layer got fused into another layer"],
                                 ], tablefmt="plain"))
      f.write('\n')
      f.write(tabulate.tabulate(table, header, tablefmt="grid"))

  def emit_summary_details(self, file):
    table = []
    header = ['layer', 'A', 'B', 'C', 'D']

    for layer in self.layers_list:
      layer_row = []
      layer_row.append(layer + ('**' if layer in self.dvnet_layers_meta and self.layer_quant_errors in self.dvnet_layers_meta[layer] else ''))
      layer_row.append(self.get_layer_summary(self.layer_results_summary['A'], layer))
      layer_row.append(self.get_layer_summary(self.layer_results_summary['B'], layer))
      layer_row.append(self.get_layer_summary(self.layer_results_summary['C'], layer))
      layer_row.append(self.get_layer_summary(self.layer_results_summary['D'], layer))
      table.append(layer_row)

    with open(file, 'w') as f:
      f.write(tabulate.tabulate([["A", "compare source and dvnet fp32 outputs"],
                                 ["B", "compare dvnet unfused and fused fp32 outputs"],
                                 ["C", "compare dvnet float and fixed outputs"],
                                 ["D", "compare compiler and simulator outputs"],
                                 ["**", "attention - layers with this suffix may have quantization issues"],
                                 ["skipped", "indicates error metric checks are skipped either due to layer output files not present or layer got fused into another layer"]
                                 ]))
      f.write('\n')
      f.write(tabulate.tabulate(table, header, tablefmt="grid"))
      f.write('\n')
      f.write('compare srcnet and dvnet: ' + os.path.join(self.flows_outputs_dir, self.srcnet_dvnet_comparision) + '\n')
      f.write('compare dvnet fused and unfused outputs: ' + os.path.join(self.flows_outputs_dir, self.dvnet_unfused_and_fused_comparision) + '\n')
      f.write('compare dvnet float and quantized outputs: ' + os.path.join(self.flows_outputs_dir, self.dvnet_float_and_quantized_comparision) + '\n')
      f.write('compare dvnet quantizer and simulator outputs: ' + os.path.join(self.flows_outputs_dir, self.dvnet_quantizer_and_simulator_comparision) + '\n')
      f.write('quantization errors: ' + os.path.join(self.outputs_root, self.analyzer_out, self.stats_dir, self.quantization_errors) + '\n')

  def compare(self):
    converter_dir = os.path.join(self.outputs_root, self.converter_out)
    quantizer_dir = os.path.join(self.outputs_root, self.analyzer_out)
    simulator_dir = os.path.join(self.outputs_root, self.simulator_out)
    if not os.path.exists(converter_dir):
      sys.exit('error: converter output directory does not exist {}'.format(converter_dir))
    if not os.path.exists(quantizer_dir):
      sys.exit('error: quantizer output directory does not exist {}'.format(quantizer_dir))
    if not os.path.exists(simulator_dir):
      sys.exit('error: simulator output directory does not exist {}'.format(simulator_dir))

    self.layer_results_summary['A'] = {}
    self.layer_results_summary['B'] = {}
    self.layer_results_summary['C'] = {}
    self.layer_results_summary['D'] = {}

    headers = ['image', 'A', 'B', 'C', 'D']
    table = []

    images = os.listdir(self.images)
    count = 0
    for image in images:
      image_result = []
      image_result.append(image)
      self.compare_for_image(image)
      image_result.append(self.get_results_summary(self.layer_results_summary['A'][image]))
      image_result.append(self.get_results_summary(self.layer_results_summary['B'][image]))
      image_result.append(self.get_results_summary(self.layer_results_summary['C'][image]))
      image_result.append(self.get_results_summary(self.layer_results_summary['D'][image]))
      table.append(image_result)
      count += 1

    if not os.path.exists(self.flows_outputs_dir):
      os.makedirs(self.flows_outputs_dir)
    self.emit_error_metrics('A', self.src_dvnet_results, os.path.join(self.flows_outputs_dir, self.srcnet_dvnet_comparision))
    self.emit_error_metrics('B', self.orig_fused_results, os.path.join(self.flows_outputs_dir, self.dvnet_unfused_and_fused_comparision))
    self.emit_error_metrics('C', self.float_fixed_results, os.path.join(self.flows_outputs_dir, self.dvnet_float_and_quantized_comparision))
    self.emit_error_metrics('D', self.quantizer_simulator_results, os.path.join(self.flows_outputs_dir, self.dvnet_quantizer_and_simulator_comparision))
    self.emit_summary_details(os.path.join(self.flows_outputs_dir, self.layers_summary))
    self.emit_histogram(os.path.join(self.flows_outputs_dir, self.dvnet_float_and_quantized_diff_histogram), self.float_fixed_diff_histogram)

    print("\nsummary")
    print(tabulate.tabulate(table, headers, tablefmt="grid"))
    print(tabulate.tabulate([["A", "compare source and dvnet fp32 outputs"],
                             ["B", "compare dvnet unfused and fused fp32 outputs"],
                             ["C", "compare dvnet float and fixed outputs"],
                             ["D", "compare compiler and simulator outputs"]
                             ]))
    print('layer summary file : ' + os.path.join(self.flows_outputs_dir, 'layers_summary.txt'))

  def parse_config(self):
    if not os.path.exists(self.args.config):
      raise Exception('error: config file not accessible - ' + self.args.config)
    self.args.config = os.path.abspath(self.args.config)
    self.config = yaml.load(open(self.args.config, 'r'), Loader=yaml.FullLoader)
    if self.args.images:
      self.images = self.args.images
    else:
      self.images = self.config['network']['images']['verify'] if os.path.isabs(self.config['network']['images']['verify']) else os.path.join(os.path.dirname(self.args.config), self.config['network']['images']['verify'])
    if not os.path.exists(self.images):
      logger.error("image directory does not exist - %s", self.images)
    if self.args.results:
      self.outputs_root = self.args.results
    else:
      self.outputs_root = self.config['out'] if os.path.isabs(self.config['out']) else os.path.join(os.path.dirname(self.args.config), self.config['out'])
    self.flows_outputs_dir = os.path.join(self.outputs_root, self.flows_outputs_dir)

  def parse_args(self):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to configuration file')
    parser.add_argument('--debug', default=False, required=False, action='store_true', help="mode of accuracy comparision")
    parser.add_argument('--images', default=None, required=False, help="image directory used for comparisions")
    parser.add_argument('--results', default=None, required=False, help="root directory of tool outputs")
    self.args = parser.parse_args()
    self.parse_config()
    self.set_srcnet_meta()
    self.parse_srcnet_layer_meta()
    self.parse_dvnet_layers_summary()
    self.parse_dvnet_quantization_errors()

if __name__ == '__main__':
  a = Accuracy()
  a.parse_args()
  a.compare()
