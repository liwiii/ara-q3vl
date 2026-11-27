# LLM Compilation Tool README


***

## Table of Contents

- [Overview](#overview)
- [Usage](#usage)
- [Configuration](#configuration)
- [Supported Models \& Requirements](#supported-models--requirements)
- [Quantization Notes](#quantization-notes)
- [Important Limitations and Notes](#important-limitations-and-notes)
- [Example Workflow](#example-workflow)

***

## Overview

This tool (`llm_model_gen.py`) is an SDK component designed for efficient large language model (LLM) conversion and compilation into Kinara format for optimized inference. It supports multiple quantization schemes, including GPTQ, GGUF, and float, with fine-grained configuration capabilities.

***

## Usage

The tool provides four main modes corresponding to stages in the compilation pipeline.

1. **Convert to ONNX (Mode 1)**
Converts a Hugging Face model directory into ONNX format with quantization.

```bash
./llm_model_gen.py --config <conversion_config.yaml> --mode 1
```

2. **Generate splits and compile (Mode 2)**
Generates prompt processing and token generation splits, then compiles the model for inference.

```bash
./llm_model_gen.py --config <run_config.yaml> --mode 2 --num_of_cores 8
```

3. **Generate splits only (Mode 3)**
Generates required splits without compiling.

```bash
./llm_model_gen.py --config <run_config.yaml> --mode 3
```

4. **Compile existing splits only (Mode 4)**
Compiles previously generated splits.

```bash
./llm_model_gen.py --config <run_config.yaml> --mode 4
```


***

## Configuration

There are two primary configuration files used by the tool:

### 1. Conversion Config (`llm_model_config.yaml`)

Used for initial model conversion with an example:

```yaml
model_dir: <path_to_model_dir>
quantization: "gptq"                # Options: "gptq", "gguf", "float"
# gguf_file: <optional_gguf_file>   # Required if quantization is "gguf"
out: <output_directory>
num_mcs_shards: 1 #2 or 4 for supported multichip sharding models
```


### 2. Run Config (`kinara_run_config.yaml`)

Generated after conversion in the <output_directory>, editable for compilation:

```yaml
attention_scale_layers: [] # layer indexes for layers with scaling for K_proj and V_proj
model_repo: <path_to_converted_model>
onnx_model_repo: <path_to_onnx_model_dir>
original_model_repo: <path_to_original_model>
out: <final_output_dir>

prmpt_context_size: 4096 # Maximum Prompt Context Siize

prmpt_prc_sizes: #P rompt batch size, currently only a single batch size is supported.
- 128

qxk_out_scale_32_layers: [] # Set 32bit output scale for layers [recommended to set all layers for LLMs with attention bias]

tkn_gen_sizes: #Token gen granularity. Equal intervals supported. This is a trade-off between dvm size and token rate
- 1024
- 2048
- 3072
- 4096

use_8bit_prompt: false # Set true to run prompt-processing with 8bit weights, trade-off between model size and performence

weight_config: # All MatMuls bit precisions need to be listed for all the hidden layer indexes. Supported are INT2, INT3, INT4, INT8
  down_proj:
    INT4:
      - 1
      - 2
```

These parameters control batch sizes, context length, quantization settings, and advanced model tuning.

***

## Supported Models \& Requirements

- Llama2-7b
- Llama3.2-3b
- Llama3.1-8b
- Qwen2 (7b)
- Qwen2.5 (1.5b, 3b, 7b)

**Constraints:**

- Hidden size must be multiple of 64
- Number of hidden layers ≤ 36
- Maximum context length ≤ 8192 tokens

***

## Quantization Notes

- GPTQ supports INT2, INT3, INT4 (group size 64)
- GGUF supports bit widths 2, 3, 4 (values >4 treated as 4), and only single-file gguf models
- Float models default to 4-bit quantization, editable
- Tool dequantizes and requantizes models internally (overrides original model scales)
- Quantization config dumped to run config file for manual editing

***

## Important Limitations and Notes

- Model hidden size must be a multiple of 64
- Max 36 hidden layers supported
- Maximum context length supported is 8192 tokens
- Prompt and token quantization assumed to be identical except if `use_8bit_prompt` is set true
- Weight configs editable per layer (e.g., `down_proj`, `up_proj`) in the run config
- GGUF bit-width reading ignores original scale logic in favor of internal scale logic
- Multi-core parallelism supported via `--num_of_cores` flag for faster compile times. This denotes number of splits to be ran in parallel
- Splits can be reused and compiled independently with modes 3 and 4. 
- If splits are already compiled, stitching of llm dvm can be done with mode 6.

***

## Multichip Sharding
- set num_mcs_shards as 2 or 4 in input config or in kinara_run_config.yaml
- Supported models of Llama2, Llama3, Qwen2, and Qwen2.5 family can be compiled to be ran on multiple Ara Endpoints.
- Restriction imposed on hidden_size, num_key_value_heads * head_dim, num_attention_heads * head_dim, intermediate_size, and vocab_size, to be divisible by 64 when divided by num_shards.
- Models which don't pass this criteria fail in mode 2 and 3

## Example Workflow

1. Prepare conversion config (`llm_sample_new.yaml`).
2. Run mode 1 to convert your model to ONNX.
3. Edit `kinara_run_config.yaml` to tune prompt context size, quantization, and batch sizes.
4. Run mode 2 with specified CPU cores to generate splits and compile.
5. Optionally use modes 3 or 4 for iterative split generation and compilation without repeating conversion.

***


```


