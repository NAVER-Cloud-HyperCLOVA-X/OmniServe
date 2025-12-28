"""
Weight loading, filtering, and saving utilities for model conversion.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
from collections import defaultdict

import torch
from safetensors import safe_open
from safetensors.torch import save_file as save_safetensors


def load_safetensors_index(model_dir: str) -> Dict:
    """Load model.safetensors.index.json to get weight mapping."""
    index_path = Path(model_dir) / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Index file not found: {index_path}")

    with open(index_path, "r") as f:
        return json.load(f)


def get_weights_by_file(index: Dict) -> Dict[str, List[str]]:
    """Group weight keys by their shard file."""
    weights_by_file = defaultdict(list)
    for key, filename in index["weight_map"].items():
        weights_by_file[filename].append(key)
    return dict(weights_by_file)


def filter_weights_by_prefix(
    index: Dict,
    prefixes: List[str],
    exclude_prefixes: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """
    Filter weight keys by prefix and group by file.

    Args:
        index: The safetensors index dict
        prefixes: List of prefixes to include (e.g., ['model.language_model'])
        exclude_prefixes: List of prefixes to exclude

    Returns:
        Dict mapping shard filename to list of matching weight keys
    """
    weights_by_file = defaultdict(list)
    exclude_prefixes = exclude_prefixes or []

    for key, filename in index["weight_map"].items():
        # Check if key starts with any of the include prefixes
        include = any(key.startswith(prefix) for prefix in prefixes)
        # Check if key should be excluded
        exclude = any(key.startswith(prefix) for prefix in exclude_prefixes)

        if include and not exclude:
            weights_by_file[filename].append(key)

    return dict(weights_by_file)


def transform_key(key: str, key_mapping: Dict[str, str]) -> str:
    """
    Transform weight key according to mapping rules.

    Args:
        key: Original weight key
        key_mapping: Dict of prefix replacements

    Returns:
        Transformed key
    """
    for old_prefix, new_prefix in key_mapping.items():
        if key.startswith(old_prefix):
            return new_prefix + key[len(old_prefix):]
    return key


def iterate_weights(
    model_dir: str,
    weights_by_file: Dict[str, List[str]],
    key_mapping: Optional[Dict[str, str]] = None,
    device: str = "cpu"
) -> Iterator[Tuple[str, torch.Tensor]]:
    """
    Iterate over selected weights from safetensors files.
    Memory-efficient: loads one shard at a time.

    Args:
        model_dir: Path to model directory
        weights_by_file: Dict mapping shard filename to weight keys
        key_mapping: Optional key transformation mapping
        device: Device to load tensors on

    Yields:
        Tuple of (transformed_key, tensor)
    """
    key_mapping = key_mapping or {}
    model_path = Path(model_dir)

    for shard_file, keys in sorted(weights_by_file.items()):
        shard_path = model_path / shard_file
        print(f"  Loading shard: {shard_file} ({len(keys)} weights)")

        with safe_open(shard_path, framework="pt", device=device) as f:
            for key in keys:
                tensor = f.get_tensor(key)
                new_key = transform_key(key, key_mapping)
                yield new_key, tensor


def load_weights_dict(
    model_dir: str,
    weights_by_file: Dict[str, List[str]],
    key_mapping: Optional[Dict[str, str]] = None,
    device: str = "cpu"
) -> Dict[str, torch.Tensor]:
    """
    Load selected weights into a dictionary.

    Args:
        model_dir: Path to model directory
        weights_by_file: Dict mapping shard filename to weight keys
        key_mapping: Optional key transformation mapping
        device: Device to load tensors on

    Returns:
        Dict of weight name to tensor
    """
    weights = {}
    for key, tensor in iterate_weights(model_dir, weights_by_file, key_mapping, device):
        weights[key] = tensor
    return weights


def save_weights_pt(
    weights: Dict[str, torch.Tensor],
    output_path: str,
    prefix_strip: Optional[str] = None,
    dtype: Optional[torch.dtype] = None
) -> None:
    """
    Save weights as PyTorch .pt file.

    Args:
        weights: Dict of weight name to tensor
        output_path: Output file path
        prefix_strip: Optional prefix to strip from keys
        dtype: Optional dtype to convert tensors to (e.g., torch.bfloat16)
    """
    processed_weights = {}
    for k, v in weights.items():
        # Strip prefix if needed
        new_key = k[len(prefix_strip):] if prefix_strip and k.startswith(prefix_strip) else k
        # Convert dtype if needed
        if dtype is not None:
            v = v.to(dtype)
        processed_weights[new_key] = v

    torch.save(processed_weights, output_path)
    print(f"  Saved {len(processed_weights)} weights to {output_path}")


def save_weights_safetensors(
    weights: Dict[str, torch.Tensor],
    output_path: str,
    prefix_strip: Optional[str] = None
) -> None:
    """
    Save weights as safetensors file.

    Args:
        weights: Dict of weight name to tensor
        output_path: Output file path
        prefix_strip: Optional prefix to strip from keys
    """
    if prefix_strip:
        weights = {
            k[len(prefix_strip):] if k.startswith(prefix_strip) else k: v
            for k, v in weights.items()
        }

    save_safetensors(weights, output_path)
    print(f"  Saved {len(weights)} weights to {output_path}")


def save_weights_sharded(
    model_dir: str,
    weights_by_file: Dict[str, List[str]],
    output_dir: str,
    key_mapping: Optional[Dict[str, str]] = None,
    max_shard_size: int = 5 * 1024 * 1024 * 1024,  # 5GB default
    device: str = "cpu"
) -> Dict:
    """
    Save weights as sharded safetensors files (HuggingFace format).
    Memory-efficient: processes one shard at a time.

    Args:
        model_dir: Path to source model directory
        weights_by_file: Dict mapping shard filename to weight keys
        output_dir: Output directory path
        key_mapping: Optional key transformation mapping
        max_shard_size: Maximum shard size in bytes
        device: Device to load tensors on

    Returns:
        Index dict for model.safetensors.index.json
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    key_mapping = key_mapping or {}

    # Collect all weights with their sizes
    all_weights = []
    total_size = 0

    for key, tensor in iterate_weights(model_dir, weights_by_file, key_mapping, device):
        size = tensor.numel() * tensor.element_size()
        all_weights.append((key, tensor, size))
        total_size += size

    # Sort weights by key for consistent ordering
    all_weights.sort(key=lambda x: x[0])

    # Determine sharding
    num_shards = max(1, (total_size + max_shard_size - 1) // max_shard_size)

    # Build shards
    shards = []
    current_shard = {}
    current_size = 0

    for key, tensor, size in all_weights:
        if current_size + size > max_shard_size and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0

        current_shard[key] = tensor
        current_size += size

    if current_shard:
        shards.append(current_shard)

    # Save shards
    weight_map = {}
    num_shards = len(shards)

    for i, shard in enumerate(shards, 1):
        if num_shards == 1:
            shard_name = "model.safetensors"
        else:
            shard_name = f"model-{i:05d}-of-{num_shards:05d}.safetensors"

        shard_path = output_path / shard_name
        save_safetensors(shard, str(shard_path))
        print(f"  Saved shard: {shard_name} ({len(shard)} weights)")

        for key in shard:
            weight_map[key] = shard_name

    # Create index
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map
    }

    # Save index if multiple shards
    if num_shards > 1:
        index_path = output_path / "model.safetensors.index.json"
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)
        print(f"  Saved index: model.safetensors.index.json")

    return index


def get_total_params(weights: Dict[str, torch.Tensor]) -> int:
    """Get total number of parameters in weights dict."""
    return sum(t.numel() for t in weights.values())


def get_total_size_bytes(weights: Dict[str, torch.Tensor]) -> int:
    """Get total size in bytes of weights dict."""
    return sum(t.numel() * t.element_size() for t in weights.values())


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"
