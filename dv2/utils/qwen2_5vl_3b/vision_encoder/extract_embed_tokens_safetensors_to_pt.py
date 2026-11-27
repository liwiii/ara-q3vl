'''
Copyright (c) 2025, Kinara, Inc. All rights reserved.
Copyright 2025-2026 NXP

NXP Confidential and Proprietary. This software is owned or controlled by NXP
and may only be used strictly in accordance with the applicable license terms.
By expressly accepting such terms or by downloading, installing, activating
and/or otherwise using the software, you are agreeing that you have read, and
that you agree to comply with and are bound by, such license terms.  If you do
not agree to be bound by the applicable license terms, then you may not retain,
install, activate or otherwise use the software.
'''

import os, argparse
from transformers import Qwen2_5_VLForConditionalGeneration
import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script that Generates embedding table pt")
    
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to Model's safetensors"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Path output folder to dump embed_tokens "
    )
    args = parser.parse_args()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                args.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation = "eager"
            )
    output_file = os.path.join(args.output_path, "embed_tokens_qwen_vl_instruct.pt")
    torch.save(model.language_model.embed_tokens.weight, output_file)
    print(f"Embedding output table saved to : {output_file}")