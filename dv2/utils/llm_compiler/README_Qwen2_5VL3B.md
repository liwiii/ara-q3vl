# LLM Compilation Tool README for Qwen2.5VL-3B

## Overview

To deal with VLM's which are a MultiModal demos, a new mode of "mode=5" is introduced to our existing tool `llm_model_gen`. This is required as we need multiple `dvm` files to run a multimodal model on Ara device.


## Generate vision dvm
Hugging Face / Safetensors --> `dvm`

## Generate vision dvm (Mode 5)

The `llm_model_config.yaml` should look like:

```yaml
model_dir: <path_to_model_dir> # Path to the safetensors / torch model
quantization: "float" 
out: <output_directory> 
```

Command to launch :

```bash
./llm_model_gen --config <llm_model_config.yaml> --mode 5
```

This process generates two essential output files inside the output directory specified in the configuration. These files are required when running the application:

model.dvm — referred to as the vision DVM throughout this document

embed_tokens_qwen_vl_instruct.pt

Generating the model.dvm can take few hours. Please allow the process to finish without interruption.

---

## Generate language dvm

The Language dvm generation flow is similar to already existing flows. You can refer the `README_LLM_generic_compilation.md`, which is adjacent to this `README_Qwen2_5VL3B.md`.

<br>
Hugging Face / Safetensors --> ONNX --> `dvm`

1. **Convert to ONNX (Mode 1)**

<br> `llm_model_config.yaml` should look like:

```yaml
model_dir: <path_to_model_dir> # Path to the safetensors/torch model
quantization: "float" 
out: <output_directory>
rope:
  vl_rope_type: 'MULTI_ROPE'  # Or IMAGE or VIDEO
```
**Valid values for vl_rope_type:**
- `MULTI_ROPE` — variable length video support
- `IMAGE` — image support
- `VIDEO` — limited length (8sec) video support.

MULTI_ROPE is supposed to be work for image, video, text-only - all use cases. The dvm generated will be only slightly larger (order on few MB). For Multichip sharded models, MULTI_ROPE is not yet supported

Command to launch :
```bash
./llm_model_gen --config <llm_model_config.yaml> --mode 1
```

This will generate `kinara_run_config.yaml` in `output_directory` along side the `.onnx` file.

2. **Generate splits and compile (Mode 2)**
<br>Generates prompt processing and token generation splits, then compile the model for inference.

```bash
./llm_model_gen --config <kinara_run_config.yaml> --mode 2 --num_of_cores 8
```

This will generate a model.dvm file at the path `<output_directory>/kinara_llm/llm/model.dvm`. We will refer to this as language `dvm` in the remainder of the document. It may take some time, please be patient.

More details on `--mode`s and customizing the configuration files can be found in adjacent `README_LLM_generic_compilation.md`. The focus of `README_Qwen2_5VL3B.md` is only Qwen2_5VL 3B.