# Utils

## subgraph_util.py 
    Utility to take a graph from an input prototxt and get an output prototxt of the subgraph based on input and output nodenames



> 
    required arguments: 
    -inet inputPrototxt, --inputPrototxt inputPrototxt
                            Input Prototxt Path
    -onet outputPrototxt, --outputPrototxt outputPrototxt
                            Output Prototxt Path
    -ishape Inputnodeshape, --inputShape Inputnodeshape
                            Comma separated shape for input nodes. Add space for the shape of new node, Add + if single node has multiple inputs
    -inputnn INPUTNODE [INPUTNODE ...], --inputnodename INPUTNODE [INPUTNODE ...]
                            Input Caffe Model Path
    -outputnn OUTPUTNODE [OUTPUTNODE ...], --outputnodename OUTPUTNODE [OUTPUTNODE ...]
                            Input Caffe Model Path


Examples: 

`python3 subgraph_util.py -inet /auto/worka/ideepanshu/sw/candid/test/networks/dv_cfg0/onnx/modelzoo/mobilenetv2/__run.default/nnconvert.default/params/nnconvert.prototxt -onet output.prototxt -inputnn Add_15 -outputnn Conv_25 Conv_20 -ishape 1,1,1,1 11,11,3,3`


`python3 subgraph_util.py -inet /auto/worka/aditya/t3/sw/candid/test/networks/dv_cfg0/mxnet/modelzoo/Mobileface_detection/__run.default/nnconvert.default/params/nnconvert.prototxt -onet nn.prototxt -inputnn mfdetv10_conv1_fwd -outputnn yolov30_conv1_fwd yolov30_yolodetectionblockv31_conv5_fwd -ishape 1,2,3,4`

  
