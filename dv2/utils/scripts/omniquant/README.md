git clone --recursive https://github.com/KinaraAi/OmniQuant.git --branch kinara_omniquant_base

pip3 install -r omniquant_requirements.txt


cd repos/OmniQuant && pip3 install -r requirements.txt && pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121 --force-reinstall && \
pip freeze -l > omniquant_requirements.txt && deactivate"
