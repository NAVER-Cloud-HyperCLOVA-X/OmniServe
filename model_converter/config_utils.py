"""
Config extraction and generation utilities for model conversion.
"""

import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any
from copy import deepcopy


def _strip_json_comments(content: str) -> str:
    """
    Remove # style comments from JSON content.

    Note: Standard JSON doesn't support comments, but some config files
    may contain them. This function removes single-line # comments.

    Args:
        content: JSON string that may contain # comments

    Returns:
        JSON string with comments removed
    """
    # Remove # comments (everything from # to end of line)
    # Be careful not to remove # inside strings
    lines = content.split('\n')
    cleaned_lines = []

    for line in lines:
        # Find # that's not inside a string
        in_string = False
        escape_next = False
        result = []

        for i, char in enumerate(line):
            if escape_next:
                result.append(char)
                escape_next = False
                continue

            if char == '\\':
                result.append(char)
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                result.append(char)
                continue

            if char == '#' and not in_string:
                # Found comment, stop here
                break

            result.append(char)

        cleaned_lines.append(''.join(result))

    return '\n'.join(cleaned_lines)


def load_config(model_dir: str) -> Dict:
    """Load config.json from model directory."""
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        content = f.read()

    # Remove any # comments (non-standard JSON but sometimes present in configs)
    content = _strip_json_comments(content)

    return json.loads(content)


def save_config(config: Dict, output_dir: str) -> None:
    """Save config.json to output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config.json")


def extract_llm_config(
    config: Dict,
    model_type: str = "auto"
) -> Dict:
    """
    Extract LLM config from unified model config.

    Args:
        config: Original unified model config
        model_type: LLM model type ('llama', 'hyperclovax', 'auto')

    Returns:
        Standalone LLM config
    """
    # Get text_config if it exists
    if "text_config" in config:
        llm_config = deepcopy(config["text_config"])
    else:
        # Already a standalone LLM config
        return deepcopy(config)

    # Ensure essential fields
    if "architectures" not in llm_config:
        # Infer from model_type
        model_type_detected = llm_config.get("model_type", model_type)
        if model_type_detected == "hyperclovax":
            llm_config["architectures"] = ["HyperCLOVAXForCausalLM"]
            llm_config["auto_map"] = {
                "AutoConfig": "configuration_hyperclovax.HyperCLOVAXConfig",
                "AutoModel": "modeling_hyperclovax.HyperCLOVAXModel",
                "AutoModelForCausalLM": "modeling_hyperclovax.HyperCLOVAXForCausalLM"
            }
        elif model_type_detected == "llama":
            llm_config["architectures"] = ["LlamaForCausalLM"]

    # Clean up unnecessary fields from parent config
    fields_to_remove = [
        "bad_words_ids", "begin_suppress_tokens", "chunk_size_feed_forward",
        "cross_attention_hidden_size", "decoder_start_token_id", "diversity_penalty",
        "do_sample", "early_stopping", "encoder_no_repeat_ngram_size",
        "exponential_decay_length_penalty", "finetuning_task", "forced_bos_token_id",
        "forced_eos_token_id", "id2label", "label2id", "is_decoder", "is_encoder_decoder",
        "length_penalty", "max_length", "min_length", "no_repeat_ngram_size",
        "num_beam_groups", "num_beams", "num_return_sequences", "output_attentions",
        "output_hidden_states", "output_scores", "prefix", "problem_type",
        "remove_invalid_values", "repetition_penalty", "return_dict",
        "return_dict_in_generate", "sep_token_id", "suppress_tokens",
        "task_specific_params", "temperature", "tf_legacy_loss", "tie_encoder_decoder",
        "tokenizer_class", "top_k", "top_p", "torchscript", "typical_p",
        "use_bfloat16", "add_cross_attention", "pruned_heads"
    ]

    for field in fields_to_remove:
        llm_config.pop(field, None)

    return llm_config


def extract_vision_config(config: Dict) -> Dict:
    """
    Extract Vision Encoder config from unified model config.

    Args:
        config: Original unified model config

    Returns:
        Vision encoder config
    """
    if "vision_config" not in config:
        raise ValueError("No vision_config found in config")

    vision_config = deepcopy(config["vision_config"])

    # Clean up unnecessary fields
    fields_to_remove = [
        "bad_words_ids", "begin_suppress_tokens", "chunk_size_feed_forward",
        "cross_attention_hidden_size", "decoder_start_token_id", "diversity_penalty",
        "do_sample", "early_stopping", "encoder_no_repeat_ngram_size",
        "exponential_decay_length_penalty", "finetuning_task", "forced_bos_token_id",
        "forced_eos_token_id", "id2label", "label2id", "is_decoder", "is_encoder_decoder",
        "length_penalty", "max_length", "min_length", "no_repeat_ngram_size",
        "num_beam_groups", "num_beams", "num_return_sequences", "output_attentions",
        "output_hidden_states", "output_scores", "prefix", "problem_type",
        "remove_invalid_values", "repetition_penalty", "return_dict",
        "return_dict_in_generate", "sep_token_id", "suppress_tokens",
        "task_specific_params", "temperature", "tf_legacy_loss", "tie_encoder_decoder",
        "tie_word_embeddings", "tokenizer_class", "top_k", "top_p", "torchscript",
        "typical_p", "use_bfloat16", "add_cross_attention", "pruned_heads"
    ]

    for field in fields_to_remove:
        vision_config.pop(field, None)

    return vision_config


def extract_audio_config(config: Dict) -> Dict:
    """
    Extract Audio Encoder config from unified model config.

    Args:
        config: Original unified model config

    Returns:
        Audio encoder config
    """
    if "audio_config" not in config:
        raise ValueError("No audio_config found in config")

    audio_config = deepcopy(config["audio_config"])

    # Clean up unnecessary fields
    fields_to_remove = [
        "bad_words_ids", "begin_suppress_tokens", "chunk_size_feed_forward",
        "cross_attention_hidden_size", "decoder_start_token_id", "diversity_penalty",
        "do_sample", "early_stopping", "encoder_no_repeat_ngram_size",
        "exponential_decay_length_penalty", "finetuning_task", "forced_bos_token_id",
        "forced_eos_token_id", "id2label", "label2id", "is_decoder", "is_encoder_decoder",
        "length_penalty", "max_length", "min_length", "no_repeat_ngram_size",
        "num_beam_groups", "num_beams", "num_return_sequences", "output_attentions",
        "output_hidden_states", "output_scores", "prefix", "problem_type",
        "remove_invalid_values", "repetition_penalty", "return_dict",
        "return_dict_in_generate", "sep_token_id", "suppress_tokens",
        "task_specific_params", "temperature", "tf_legacy_loss", "tie_encoder_decoder",
        "tie_word_embeddings", "tokenizer_class", "top_k", "top_p", "torchscript",
        "typical_p", "use_bfloat16", "add_cross_attention", "pruned_heads"
    ]

    for field in fields_to_remove:
        audio_config.pop(field, None)

    return audio_config


def copy_tokenizer_files(src_dir: str, dst_dir: str) -> List[str]:
    """
    Copy tokenizer-related files from source to destination.

    Args:
        src_dir: Source model directory
        dst_dir: Destination directory

    Returns:
        List of copied file names
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    ]

    copied = []
    for filename in tokenizer_files:
        src_file = src_path / filename
        if src_file.exists():
            shutil.copy2(src_file, dst_path / filename)
            copied.append(filename)
            print(f"  Copied {filename}")

    return copied


def copy_model_files(
    src_dir: str,
    dst_dir: str,
    file_patterns: Optional[List[str]] = None
) -> List[str]:
    """
    Copy model-related files (e.g., modeling_*.py, configuration_*.py).

    Args:
        src_dir: Source model directory
        dst_dir: Destination directory
        file_patterns: List of file patterns to copy

    Returns:
        List of copied file names
    """
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    if file_patterns is None:
        file_patterns = [
            "modeling_*.py",
            "configuration_*.py",
            "generation_config.json",
            "preprocessor_config.json",
            "preprocessor.py",
            "processing_*.py",
            "processor_config.json",
            "video_preprocessor_config.json",
        ]

    copied = []
    for pattern in file_patterns:
        for src_file in src_path.glob(pattern):
            dst_file = dst_path / src_file.name
            shutil.copy2(src_file, dst_file)
            copied.append(src_file.name)
            print(f"  Copied {src_file.name}")

    return copied


def get_model_name_from_path(model_dir: str) -> str:
    """Extract model name from directory path."""
    return Path(model_dir).name
