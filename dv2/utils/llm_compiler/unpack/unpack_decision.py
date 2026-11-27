"""Module for determining whether unpack optimization should be triggered for a model.

This module analyzes model configuration file to determine if the unpack optimization
pathway should be enabled. The unpack optimization is currently validated based on:
- Architecture: Qwen2_5_VLForConditionalGeneration (3B variant) as of now.
- 8-bit prompt processing enabled. (see explanation below)
- Quantization scheme in the layers. 

The unpack pathway separates dynamic weights from static weights, allowing dynamic
layers (convolution, layer norm, attention softmax) to be processed at runtime while
keeping static weights in the main model.

Why do we need 8-bit prompt processing?
The customer can provide 4bit quantized models. However when Kinara SDK generates
prompt processing command stream, it must follow 8bit scheme. The dvm with unpack
support must follow 8bit command stream  for prompt processing because it is compute
bound. The token-generation is data bound hence we don't have such a restriction.
Despite prompt processing being 8bit, the weights used for dvm packaging are from
unpack splits (4bit). The 8bit weights from prompt processing splits are ignored
when stitching a dvm with unpack support.

Example:
    >>> from pathlib import Path
    >>> from unpack_decision import should_trigger_unpack
    >>> result = should_trigger_unpack(Path("kinara_run_config.yaml"))
    >>> if result:
    ...     print("Unpack optimization enabled")
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def detect_qwen2_5vl_version(hf_config: Dict[str, Any]) -> str:
    """Detect the Qwen2.5-VL model variant based on vision configuration.

    Analyzes the vision_config section of HuggingFace model config to determine
    if the model is the 3B or 7B variant based on output hidden size.

    Args:
        hf_config: HuggingFace model configuration dictionary containing vision_config.

    Returns:
        Model variant identifier: '3B', '7B', or 'unhandled' if not supported.
    """
    if 'vision_config' not in hf_config:
        logger.info("No vision_config found in config.json")
        return 'unhandled'

    out_hidden_size = hf_config['vision_config'].get('out_hidden_size')
    if out_hidden_size == 3584:
        return '7B'
    elif out_hidden_size == 2048:
        return '3B'
    else:  # Maybe None
        logger.info(f"Unsupported Qwen 2.5VL model with vision_config.out_hidden_size {out_hidden_size}")
        return 'unhandled'


def get_subkeys(d: Dict[str, Any], key: str) -> List[str]:
    """Get subkeys from a nested dictionary.

    Args:
        d: Dictionary to search.
        key: Key whose subkeys to retrieve.

    Returns:
        List of subkeys if key exists and value is a dict, empty list otherwise.
    """
    if key in d and isinstance(d[key], dict):
        return list(d[key].keys())
    return []


def should_trigger_unpack_from_dict(kinara_run_config: Dict[str, Any]) -> bool:
    """Determine if unpack optimization should be triggered from config dictionary.

    Performs a series of validation checks on the kinara run configuration:
    1. Verifies config.json exists and is readable
    2. Checks architecture is supported.
    4. Confirms 8-bit prompt processing is enabled
    5. Ensures all layers use quantizations which are verified and supported.

    Args:
        kinara_run_config: Kinara run configuration dictionary containing:
            - model_repo: Path to model repository with config.json
            - use_8bit_prompt: Whether 8-bit prompt processing is enabled
            - weight_config: Layer-wise weight quantization configuration

    Returns:
        True if all validation checks pass and unpack should be triggered,
        False otherwise.
    """
    hf_config_path = Path(kinara_run_config['model_repo']) / "config.json"
    if not hf_config_path.exists():
        logger.info(f"Could not find {hf_config_path}, skipping unpack")
        return False

    hf_config = {}
    with open(hf_config_path, 'r') as file:
        hf_config = json.load(file)
    if (not hf_config) or (len(hf_config) == 0):
        logger.info(f"Could not open {hf_config_path}, skipping unpack")
        return False
    
    # As of now we have tested it end to end with Qwen2_5_VLForConditionalGeneration 3B
    if (hf_config['architectures'][0] != 'Qwen2_5_VLForConditionalGeneration') or (detect_qwen2_5vl_version(hf_config) != '3B'):
        logger.info(f"Architecture not verified for unpack, skipping unpack")
        return False

    if not kinara_run_config.get('use_8bit_prompt', False):
        logger.info(f"Prompt processing not 8bit precision, skipping unpack")
        return False

    if 'weight_config' not in kinara_run_config:
        logger.info(f"weight precisions cannot be determined, skipping unpack")
        return False
    for layer, weight_config in kinara_run_config['weight_config'].items():
        # This will continue to expand as we validate and enable more combinations.

        # For all the layers, if all weights are in 4bit, then proceed.
        if  set(weight_config.keys()) != {'INT4'}:
            logger.info(f"{layer} has {list(weight_config.keys())}, skipping unpack")
            return False

        logger.info(f"{layer} has {list(weight_config.keys())}, OK wrt unpack")
    return True


def should_trigger_unpack(kinara_run_config_path: Path) -> bool:
    """Determine if unpack optimization should be triggered from config file path.

    Loads the kinara run configuration from a YAML file and delegates to
    should_trigger_unpack_from_dict for validation.

    Args:
        kinara_run_config_path: Path to the kinara run configuration YAML file.

    Returns:
        True if unpack optimization should be triggered, False otherwise.

    Raises:
        FileNotFoundError: If the YAML config file does not exist.
        yaml.YAMLError: If the YAML file cannot be parsed.
    """
    with open(kinara_run_config_path, 'r') as file:
        kinara_run_config = yaml.safe_load(file)
    return should_trigger_unpack_from_dict(kinara_run_config)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the unpack decision script.

    Returns:
        Parsed arguments containing:
            - yaml_file: Path to the YAML configuration file to analyze.
    """
    parser = argparse.ArgumentParser(
        description="Determine if unpack optimization should be triggered for a model"
    )
    parser.add_argument(
        '--yaml_file',
        required=True,
        help="Path to kinara run configuration YAML file"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    logger.info(f"For {args.yaml_file}")
    result = should_trigger_unpack(Path(args.yaml_file))
    logger.info(f"should_trigger_unpack: {result}")
