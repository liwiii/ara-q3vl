# Utils

## dotMatcher.py 
    - utiliy to save dot files for patterns in an ONNX graph. 
    We may use these generated files to writePattern matchers which replaces these complex patterns with some custom implementation.



> 
    required arguments: 
    -inet INPUTONNX, --inputONNX INPUTONNX
                            Input onnx model Path
    -dfile DOTFILE, --dotFile DOTFILE
                            Output dotFile Path
    -start STARTNODE [STARTNODE ...], --startNode STARTNODE [STARTNODE ...]
                            Input Caffe Model Path
    -end ENDNODE [ENDNODE ...], --endNode ENDNODE [ENDNODE ...]
                            Input Caffe Model Path

    optional arguments:
    -h, --help            show this help message and exit
    -relabel_nodes, --relabel_nodes
                            relabel nodes to have simpler node names in subgraph
    -test, --testPatterns
                            loads the same pattern and print number of matches in original graph


Example: 

`python3 dotMatcher.py -inet /auto/worka/aditya/customer/ap/jira/slowfast_pyramid18.onnx  -dfile ./.t.dot -start Conv_17  -end Relu_18  -relabelNodes -test` 

`python3 dotMatcher.py -inet /auto/worka/aditya/customer/ap/jira/slowfast_pyramid18.onnx  -dfile ./.t.dot -start Conv_0 Conv_17  -end Relu_18  -relabelNodes -test`