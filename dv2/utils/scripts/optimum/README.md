cd /dv2 && unset PYTHONPATH && virtualenv -p python3 /dv2/venv_qwen && source /dv2/venv_qwen/bin/activate && pip install --upgrade pip && pip install optimum==1.20.0 transformers==4.40.0 torch==2.2.2 onnx==1.16.1 onnxruntime==1.18.1 onnxsim==0.4.36 protobuf==3.20.2 && \

git clone --recursive https://github.com/luchangli03/onnxsim_large_model.git && cd onnxsim_large_model && git checkout cbcf3fbf985a00494b0f136c92eaccd42031bf65 && cd ../../ && \
pip freeze -l > qwen_requirements.txt && deactivate && \
