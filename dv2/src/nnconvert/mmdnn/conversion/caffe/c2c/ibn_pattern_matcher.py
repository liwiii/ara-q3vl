'''
Copyright (c) 2018-25, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

from mmdnn.conversion.caffe.c2c.common import BasePass


class IBNPatternMatching(BasePass):

  def __init__(self):
    pass

  def do_transformation(self):
    pass
    # while self.get_nodes_of_type('Slice'):
    #     slice_layer = self.get_nodes_of_type('Slice')[0]
    #     if [i._layer.type for i in slice_layer.successors()] not in [['InstanceNorm', 'BatchNorm']]:
    #         continue
    #     inst_layer = next(i for i in slice_layer.successors() if i._layer.type == 'InstanceNorm')
    #     batch_layer = next(i for i in slice_layer.successors() if i._layer.type == 'BatchNorm')
    #     if batch_layer.successors() != inst_layer.successors() \
    #         or len(batch_layer.successors()) != 1 \
    #         or batch_layer.successors()[0]._layer.type != 'Concat':
    #         continue
    #     concat_layer = batch_layer.successors()[0]
    #     parent = slice_layer.predecessors()[0]
    #     child = concat_layer.successors()
    #     split_axis_shape = parent._layer.input_param.shape[0].dim[slice_layer._layer.slice_param.axis]
    #     slice_point = [split_axis_shape//2]
    #     if not slice_point == slice_layer._layer.slice_param.slice_point:
    #         continue
    #     counter += 1
    #     ibn = mk_ibn_proto('IBN_'+str(counter))
    #     ibn.bottom[:] = parent._layer.top
    #     self.G.add_node(ibn)
    #     self.G.add_edge(parent, ibn)
    #     for i in child:
    #         i._layer.bottom[:] = ibn.top
    #         self.G.add_edges(tuple(ibn, i))
    #     for i in [slice_layer, inst_layer, batch_layer, concat_layer]:
    #         self.G.remove_node(i)
    #     import pdb; pdb.set_trace()
    #     self.caffe_net.params.setdefault('IBN_'+str(counter),self.caffe_net.params['InstanceNorm'])
    #     self.caffe_net.params['IBN_1'].append(self.caffe_net.params['BatchNorm'][0])
    #     self.caffe_net.params['IBN_1'].append(self.caffe_net.params['BatchNorm'][1])
    #     shape = self.caffe_net.params['BatchNorm'][1].data.shape
    #     size = self.caffe_net.params['BatchNorm'][1].data.size
    #     self.caffe_net.params['IBN_1'].add_blob(size)
    #     self.caffe_net.params['IBN_1'].add_blob(size)
    #     self.caffe_net.params['IBN_1'][-2].data.reshape(shape)
    #     self.caffe_net.params['IBN_1'][-1].data.reshape(shape)
    #     x = self.caffe_net.params['IBN_1'][-2].data
    #     x[x == x] = 1
    #     x = self.caffe_net.params['IBN_1'][-1].data
    #     x[x == x] = 0
    #     self.caffe_net.params.pop('BatchNorm')
    #     self.caffe_net.params.pop('InstanceNorm')
    #     self.inp_wts.setdefault('IBN_'+str(counter),self.inp_wts['InstanceNorm'])
    #     for i in range(2,6):
    #         self.inp_wts['IBN_1'].append(self.caffe_net.params['IBN_1'][i].data)
    #     self.inp_wts.pop('BatchNorm')
    #     self.inp_wts.pop('InstanceNorm')
