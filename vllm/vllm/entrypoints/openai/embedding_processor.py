# Copyright (c) 2025 NAVER Cloud Corp.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Embedding Processor for Vision/Audio Embedding Injection.

This module handles the preprocessing of prompts with placeholder tokens,
replacing them with external embeddings (vision, audio, etc.) and converting
the result to prompt_embeds format.

Key features:
- No modification to vllm/v1/ core code
- Caching support for repeated inputs
- Multi-turn conversation support
- Extensible for multiple modalities (vision, audio, etc.)
- S3 storage support for large embeddings
"""

import hashlib
import io
import os
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse

import requests
import torch
from transformers import AutoModel, AutoConfig

from vllm.logger import init_logger

logger = init_logger(__name__)


# =============================================================================
# Configuration via Environment Variables
# =============================================================================
# VLLM_EMBEDDING_CACHE_MAX_ITEMS: Maximum number of cached embedding downloads
#   Default: 50, Range: 1-1000
#   Example: export VLLM_EMBEDDING_CACHE_MAX_ITEMS=100
#
# VLLM_EMBEDDING_CACHE_MAX_MEMORY_MB: Maximum memory for embedding cache in MB
#   Default: 1024 (1GB), Range: 100-8192
#   Example: export VLLM_EMBEDDING_CACHE_MAX_MEMORY_MB=2048
#
# VLLM_EMBEDDING_PROCESSOR_CACHE_SIZE: LRU cache size for processed embeddings
#   Default: 1000, Range: 100-10000
#   Example: export VLLM_EMBEDDING_PROCESSOR_CACHE_SIZE=2000
#
# GPU Memory Utilization Guidelines:
# ----------------------------------
# When using --enable-prompt-embeds with multimodal inputs:
#   - Short audio (<30s): --gpu-memory-utilization 0.8 (default)
#   - Long audio (30s-2min): --gpu-memory-utilization 0.7
#   - Very long audio (>2min): --gpu-memory-utilization 0.6
#   - Multiple concurrent requests: Reduce by 0.1 per expected concurrent request
#
# Memory estimation per embedding type:
#   - Image (729 tokens): ~6MB per embedding
#   - Audio (per second): ~0.5MB per second of audio
#   - Video (per frame): ~6MB per frame
#
# Example server start command for multimodal:
#   python api_server.py --model ... --enable-prompt-embeds \
#     --gpu-memory-utilization 0.7 --max-model-len 8192
# =============================================================================

# S3 connection singleton
_s3_connection = None

# Download cache for S3/URL embeddings (path/url -> (data, timestamp, size_bytes))
_download_cache: dict[str, tuple[any, float, int]] = {}
_download_cache_max_size = int(os.environ.get("VLLM_EMBEDDING_CACHE_MAX_ITEMS", "50"))
_download_cache_max_memory_mb = int(os.environ.get("VLLM_EMBEDDING_CACHE_MAX_MEMORY_MB", "1024"))
_download_cache_current_memory = 0  # Current memory usage in bytes
_download_cache_lock = threading.Lock()

# Validate environment variable ranges
_download_cache_max_size = max(1, min(1000, _download_cache_max_size))
_download_cache_max_memory_mb = max(100, min(8192, _download_cache_max_memory_mb))


def _estimate_tensor_size(data) -> int:
    """Estimate memory size of data in bytes."""
    if isinstance(data, torch.Tensor):
        return data.numel() * data.element_size()
    elif isinstance(data, dict):
        total = 0
        for v in data.values():
            if isinstance(v, torch.Tensor):
                total += v.numel() * v.element_size()
        return total
    return 0


def _get_cached_download(key: str):
    """Get cached download data if available."""
    with _download_cache_lock:
        if key in _download_cache:
            data, timestamp, size = _download_cache[key]
            logger.info(f"Cache HIT for: {key[:80]}...")
            return data
    return None


def _set_cached_download(key: str, data):
    """Cache download data with LRU eviction and memory limits."""
    import time
    global _download_cache_current_memory
    
    data_size = _estimate_tensor_size(data)
    max_memory_bytes = _download_cache_max_memory_mb * 1024 * 1024
    
    with _download_cache_lock:
        # Evict if count limit reached
        while len(_download_cache) >= _download_cache_max_size:
            sorted_keys = sorted(
                _download_cache.keys(),
                key=lambda k: _download_cache[k][1]  # Sort by timestamp
            )
            if sorted_keys:
                old_key = sorted_keys[0]
                _, _, old_size = _download_cache[old_key]
                del _download_cache[old_key]
                _download_cache_current_memory -= old_size
                logger.debug(f"Evicted (count): {old_key[:50]}... freed {old_size/1024/1024:.1f}MB")
        
        # Evict if memory limit reached
        while _download_cache_current_memory + data_size > max_memory_bytes and _download_cache:
            sorted_keys = sorted(
                _download_cache.keys(),
                key=lambda k: _download_cache[k][1]
            )
            if sorted_keys:
                old_key = sorted_keys[0]
                _, _, old_size = _download_cache[old_key]
                del _download_cache[old_key]
                _download_cache_current_memory -= old_size
                logger.debug(f"Evicted (memory): {old_key[:50]}... freed {old_size/1024/1024:.1f}MB")
        
        _download_cache[key] = (data, time.time(), data_size)
        _download_cache_current_memory += data_size
        logger.info(
            f"Cached download: {key[:60]}... "
            f"(items={len(_download_cache)}, memory={_download_cache_current_memory/1024/1024:.1f}MB)"
        )


def get_download_cache_stats() -> dict:
    """Get cache statistics including memory usage."""
    with _download_cache_lock:
        return {
            "item_count": len(_download_cache),
            "max_items": _download_cache_max_size,
            "memory_mb": _download_cache_current_memory / 1024 / 1024,
            "max_memory_mb": _download_cache_max_memory_mb,
            "keys": list(_download_cache.keys())[:10],  # First 10 keys
        }


def clear_download_cache():
    """Clear the download cache and reset memory counter."""
    global _download_cache_current_memory
    with _download_cache_lock:
        _download_cache.clear()
        _download_cache_current_memory = 0
        logger.info("Download cache cleared")


def get_s3_connection():
    """Get or create S3 connection singleton."""
    global _s3_connection
    if _s3_connection is None:
        try:
            from wbl_storage_utility.s3_util import S3Connection
            _s3_connection = S3Connection()
            logger.info("S3 connection initialized successfully")
        except ImportError:
            logger.warning(
                "wbl_storage_utility not installed. S3 features will not be available. "
                "Install with: pip install wbl-storage-utility"
            )
            _s3_connection = None
        except Exception as e:
            logger.warning(f"Failed to initialize S3 connection: {e}")
            _s3_connection = None
    return _s3_connection


def download_embeddings_from_s3(
    s3_path: str, 
    allow_dict: bool = False,
    use_cache: bool = True,
):
    """
    Download embeddings from S3 storage with caching support.
    
    Args:
        s3_path: S3 path in format 'bucket_name/path/to/file.pt'
        allow_dict: If True, allow dict format (for multi-modal embeddings)
        use_cache: If True, use cached data if available (default: True)
        
    Returns:
        torch.Tensor or dict containing the embeddings
        
    Raises:
        RuntimeError: If download fails or S3 is not configured
    """
    # Check cache first
    cache_key = f"s3://{s3_path}"
    if use_cache:
        cached_data = _get_cached_download(cache_key)
        if cached_data is not None:
            return cached_data
    
    s3_conn = get_s3_connection()
    if s3_conn is None:
        raise RuntimeError(
            "S3 connection not available. "
            "Please install wbl_storage_utility and configure S3 credentials."
        )
    
    # Parse s3_path: bucket_name/path/to/file.pt
    parts = s3_path.split("/", 1)
    if len(parts) != 2:
        raise ValueError(
            f"Invalid S3 path format: {s3_path}. "
            "Expected format: 'bucket_name/path/to/file.pt'"
        )
    
    bucket_name, object_path = parts
    
    # Create a temporary file for download
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
        tmp_path = tmp_file.name
    
    try:
        logger.info(f"Downloading embeddings from S3: {bucket_name}/{object_path}")
        
        success = s3_conn.download_file(
            storage_name=bucket_name,
            local_path=tmp_path,
            storage_path=object_path,
        )
        
        if not success:
            raise RuntimeError(f"Failed to download from S3: {s3_path}")
        
        # Load the data
        data = torch.load(tmp_path, map_location="cpu")
        
        if isinstance(data, dict):
            if allow_dict:
                logger.info(f"Downloaded dict data from S3: keys={list(data.keys())}")
                # Cache the result
                if use_cache:
                    _set_cached_download(cache_key, data)
                return data
            else:
                raise ValueError(
                    f"Expected torch.Tensor in S3 file, got dict. "
                    f"Use multimodal_embeddings_s3_path for dict format."
                )
        elif isinstance(data, torch.Tensor):
            logger.info(
                f"Downloaded embeddings from S3: shape={data.shape}, dtype={data.dtype}"
            )
            # Cache the result
            if use_cache:
                _set_cached_download(cache_key, data)
            return data
        else:
            raise ValueError(
                f"Expected torch.Tensor or dict in S3 file, got {type(data)}"
            )
        
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _extract_s3_path_from_url(url: str) -> str | None:
    """Extract S3 path from presigned URL for cache key."""
    try:
        from urllib.parse import urlparse, unquote
        parsed = urlparse(url)
        # URL format: https://bucket.object.storage.com/path/to/file.pt?X-Amz-...
        # or: https://object.storage.com/bucket/path/to/file.pt?X-Amz-...
        path = parsed.path.lstrip("/")
        # Remove query parameters for stable cache key
        if path:
            return unquote(path)
    except Exception:
        pass
    return None


def download_embeddings_from_url(
    url: str, 
    allow_dict: bool = False,
    use_cache: bool = True,
):
    """
    Download embeddings from a presigned URL with caching support.
    
    Args:
        url: Presigned URL to download the embeddings file
        allow_dict: If True, allow dict format (for multi-modal embeddings)
        use_cache: If True, use cached data if available (default: True)
        
    Returns:
        torch.Tensor or dict containing the embeddings
        
    Raises:
        RuntimeError: If download fails
    """
    # Extract S3 path from URL for stable cache key
    s3_path = _extract_s3_path_from_url(url)
    cache_key = f"url://{s3_path}" if s3_path else f"url://{url[:100]}"
    
    # Check cache first
    if use_cache:
        cached_data = _get_cached_download(cache_key)
        if cached_data is not None:
            return cached_data
    
    logger.info(f"Downloading embeddings from URL: {url[:100]}...")
    
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Load tensor from bytes
        buffer = io.BytesIO(response.content)
        data = torch.load(buffer, map_location="cpu")
        
        if isinstance(data, dict):
            if allow_dict:
                logger.info(f"Downloaded dict data from URL: keys={list(data.keys())}")
                # Cache the result
                if use_cache:
                    _set_cached_download(cache_key, data)
                return data
            else:
                raise ValueError(
                    f"Expected torch.Tensor in downloaded file, got dict. "
                    f"Use multimodal_embeddings_url for dict format."
                )
        elif isinstance(data, torch.Tensor):
            logger.info(
                f"Downloaded embeddings from URL: shape={data.shape}, dtype={data.dtype}"
            )
            # Cache the result
            if use_cache:
                _set_cached_download(cache_key, data)
            return data
        else:
            raise ValueError(
                f"Expected torch.Tensor or dict in downloaded file, got {type(data)}"
            )
        
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to download from URL: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load embeddings from URL: {e}") from e


def load_vision_embeddings(
    vision_embeddings: list[list[float]] | None = None,
    vision_embeddings_s3_path: str | None = None,
    vision_embeddings_url: str | None = None,
) -> list[list[float]]:
    """
    Load vision embeddings from one of the supported sources.
    
    Priority: vision_embeddings > vision_embeddings_url > vision_embeddings_s3_path
    
    Args:
        vision_embeddings: Direct embeddings as nested list
        vision_embeddings_s3_path: S3 path to embeddings file
        vision_embeddings_url: Presigned URL to embeddings file
        
    Returns:
        Embeddings as nested list of floats
        
    Raises:
        ValueError: If no embeddings source is provided or multiple are provided
        RuntimeError: If download fails
    """
    sources = [
        vision_embeddings is not None,
        vision_embeddings_s3_path is not None,
        vision_embeddings_url is not None,
    ]
    
    if sum(sources) == 0:
        raise ValueError(
            "No vision embeddings source provided. "
            "Please provide one of: vision_embeddings, vision_embeddings_s3_path, "
            "or vision_embeddings_url"
        )
    
    if sum(sources) > 1:
        raise ValueError(
            "Multiple vision embeddings sources provided. "
            "Please provide only one of: vision_embeddings, vision_embeddings_s3_path, "
            "or vision_embeddings_url"
        )
    
    # Direct embeddings - already in correct format
    if vision_embeddings is not None:
        return vision_embeddings
    
    # Download from presigned URL (preferred for security)
    if vision_embeddings_url is not None:
        tensor = download_embeddings_from_url(vision_embeddings_url)
        return tensor.tolist()
    
    # Download from S3 path
    if vision_embeddings_s3_path is not None:
        tensor = download_embeddings_from_s3(vision_embeddings_s3_path)
        return tensor.tolist()
    
    # Should never reach here
    raise ValueError("No valid vision embeddings source")


@dataclass
class MultiModalEmbeddings:
    """Container for multi-modal embeddings with discrete and continuous components."""
    discrete: torch.Tensor  # Token IDs, shape: (1, num_discrete_tokens) or (num_discrete_tokens,)
    continuous: torch.Tensor  # Embeddings, shape: (num_continuous_tokens, hidden_size)
    meta: dict | None = None  # Optional metadata


def load_multimodal_embeddings(
    multimodal_embeddings_s3_path: str | None = None,
    multimodal_embeddings_url: str | None = None,
) -> MultiModalEmbeddings:
    """
    Load multi-modal embeddings from S3 or URL.
    
    The file should contain a dict with:
    - 'discrete': Token IDs tensor
    - 'continuous': Embeddings tensor
    - 'meta': Optional metadata dict
    
    Args:
        multimodal_embeddings_s3_path: S3 path to embeddings file
        multimodal_embeddings_url: Presigned URL to embeddings file
        
    Returns:
        MultiModalEmbeddings object
        
    Raises:
        ValueError: If no source provided or invalid format
    """
    if multimodal_embeddings_s3_path is None and multimodal_embeddings_url is None:
        raise ValueError(
            "No multi-modal embeddings source provided. "
            "Please provide multimodal_embeddings_s3_path or multimodal_embeddings_url"
        )
    
    if multimodal_embeddings_s3_path is not None and multimodal_embeddings_url is not None:
        raise ValueError(
            "Multiple sources provided. Use only one of: "
            "multimodal_embeddings_s3_path or multimodal_embeddings_url"
        )
    
    # Download the data (allow dict format for multi-modal)
    if multimodal_embeddings_url is not None:
        data = download_embeddings_from_url(multimodal_embeddings_url, allow_dict=True)
    else:
        data = download_embeddings_from_s3(multimodal_embeddings_s3_path, allow_dict=True)
    
    # Handle dict format
    if isinstance(data, dict):
        if 'discrete' not in data or 'continuous' not in data:
            raise ValueError(
                "Multi-modal embeddings file must contain 'discrete' and 'continuous' keys"
            )
        
        discrete = data['discrete']
        continuous = data['continuous']
        meta = data.get('meta', None)
        
        # Flatten discrete if needed (1, N) -> (N,)
        if discrete.dim() == 2 and discrete.shape[0] == 1:
            discrete = discrete.squeeze(0)
        
        logger.info(
            "Loaded multi-modal embeddings: discrete=%s, continuous=%s",
            discrete.shape, continuous.shape
        )
        
        return MultiModalEmbeddings(
            discrete=discrete,
            continuous=continuous,
            meta=meta,
        )
    else:
        raise ValueError(
            f"Expected dict with 'discrete' and 'continuous' keys, got {type(data)}"
        )


@dataclass
class EmbeddingCacheEntry:
    """Cache entry for processed embeddings."""
    prompt_hash: str
    embeddings_hash: str
    combined_embeddings: torch.Tensor
    token_count: int


class EmbeddingCache:
    """LRU cache for processed embeddings."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, EmbeddingCacheEntry] = OrderedDict()
    
    def _make_key(self, prompt_hash: str, embeddings_hash: str) -> str:
        return f"{prompt_hash}:{embeddings_hash}"
    
    def get(
        self, prompt_hash: str, embeddings_hash: str
    ) -> Optional[EmbeddingCacheEntry]:
        key = self._make_key(prompt_hash, embeddings_hash)
        if key in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, entry: EmbeddingCacheEntry) -> None:
        key = self._make_key(entry.prompt_hash, entry.embeddings_hash)
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove oldest entry
                self.cache.popitem(last=False)
            self.cache[key] = entry
    
    def clear(self) -> None:
        self.cache.clear()


@dataclass
class ModalityEmbedding:
    """Container for modality-specific embeddings."""
    modality: str  # "vision", "audio", etc.
    embeddings: torch.Tensor  # Shape: (num_tokens, hidden_size)
    placeholder_token: str  # e.g., "<|ptoken|>", "<|atoken|>"


class EmbeddingProcessor:
    """
    Processor for injecting external embeddings into prompts.
    
    This processor handles:
    1. Loading embedding weights from the model
    2. Converting token IDs to embeddings
    3. Replacing placeholder tokens with external embeddings
    4. Caching for performance optimization
    
    Usage:
        processor = EmbeddingProcessor(model_path, dtype=torch.bfloat16)
        
        # Single modality
        combined = processor.process_with_embeddings(
            token_ids=[1, 2, 3, PLACEHOLDER, PLACEHOLDER, 4],
            placeholder_token_id=PLACEHOLDER,
            external_embeddings=vision_embeddings,
        )
        
        # Multiple modalities
        combined = processor.process_multi_modal(
            token_ids=[...],
            modality_embeddings=[
                ModalityEmbedding("vision", vision_emb, "<|ptoken|>"),
                ModalityEmbedding("audio", audio_emb, "<|atoken|>"),
            ],
            tokenizer=tokenizer,
        )
    """
    
    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cpu",
        cache_size: int = 1000,
    ):
        """
        Initialize the embedding processor.
        
        Args:
            model_path: Path to the model (HuggingFace format)
            dtype: Data type for embeddings (default: bfloat16)
            device: Device to store embedding table ("cpu" recommended for memory efficiency)
            cache_size: Maximum number of cache entries
        """
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.cache = EmbeddingCache(max_size=cache_size)
        
        # Load embedding weights
        self._embedding_weights: Optional[torch.Tensor] = None
        self._load_embedding_weights()
    
    def _load_embedding_weights(self) -> None:
        """Load only the embedding weights from the model (memory optimized)."""
        logger.info(
            "Loading embedding weights from %s (dtype=%s)",
            self.model_path, self.dtype
        )
        
        try:
            # Try to load directly from safetensors first (most memory efficient)
            loaded_from_safetensors = self._try_load_from_safetensors()
            if loaded_from_safetensors:
                return
            
            # Fallback: Load model with minimal memory footprint
            logger.info("Falling back to full model loading...")
            config = AutoConfig.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            
            model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=self.dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            # Extract embedding weights
            embed_layer = model.get_input_embeddings()
            if embed_layer is None:
                raise ValueError("Model does not have input embeddings layer")
            
            # Move weights directly instead of clone to save memory
            self._embedding_weights = embed_layer.weight.detach().to(
                device=self.device, dtype=self.dtype
            )
            
            # Free memory immediately
            del model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(
                "Loaded embedding weights: shape=%s, dtype=%s, device=%s",
                self._embedding_weights.shape,
                self._embedding_weights.dtype,
                self._embedding_weights.device,
            )
            
        except Exception as e:
            logger.error("Failed to load embedding weights: %s", e)
            raise
    
    def _try_load_from_safetensors(self) -> bool:
        """Try to load embedding weights directly from safetensors (memory efficient)."""
        try:
            from safetensors import safe_open
            import glob
            
            # Find safetensors files
            safetensor_files = glob.glob(os.path.join(self.model_path, "*.safetensors"))
            if not safetensor_files:
                return False
            
            # Common embedding weight names
            embed_key_patterns = [
                "model.embed_tokens.weight",
                "transformer.wte.weight", 
                "embeddings.word_embeddings.weight",
                "embed_tokens.weight",
            ]
            
            for sf_file in safetensor_files:
                with safe_open(sf_file, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        for pattern in embed_key_patterns:
                            if pattern in key or key.endswith("embed_tokens.weight"):
                                logger.info(f"Loading embedding from safetensors: {key}")
                                self._embedding_weights = f.get_tensor(key).to(
                                    device=self.device, dtype=self.dtype
                                )
                                logger.info(
                                    "Loaded embedding weights from safetensors: shape=%s, dtype=%s",
                                    self._embedding_weights.shape, self._embedding_weights.dtype
                                )
                                return True
            
            return False
        except ImportError:
            logger.debug("safetensors not available, using fallback")
            return False
        except Exception as e:
            logger.debug(f"Failed to load from safetensors: {e}")
            return False
    
    @property
    def embedding_weights(self) -> torch.Tensor:
        """Get the embedding weights tensor."""
        if self._embedding_weights is None:
            raise RuntimeError("Embedding weights not loaded")
        return self._embedding_weights
    
    @property
    def hidden_size(self) -> int:
        """Get the hidden size of embeddings."""
        return self.embedding_weights.shape[1]
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self.embedding_weights.shape[0]
    
    def _compute_hash(self, data) -> str:
        """Compute hash for caching (optimized for tensors and lists)."""
        if isinstance(data, torch.Tensor):
            # Fast hash for tensors using shape and sample values
            shape_str = str(data.shape)
            # Sample a few values to avoid converting entire tensor
            flat = data.flatten()
            if flat.numel() > 10:
                samples = flat[::flat.numel()//10][:10].tolist()
            else:
                samples = flat.tolist()
            data_str = f"{shape_str}:{samples}"
        elif isinstance(data, list) and len(data) > 100:
            # For large lists, sample instead of converting all
            sample_indices = [0, len(data)//4, len(data)//2, 3*len(data)//4, len(data)-1]
            samples = [data[i] for i in sample_indices if i < len(data)]
            data_str = f"len={len(data)}:{samples}"
        else:
            data_str = str(data)
        
        return hashlib.md5(data_str.encode('utf-8')).hexdigest()
    
    def embed_tokens(self, token_ids: list[int]) -> torch.Tensor:
        """
        Convert token IDs to embeddings using the cached embedding table.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Embeddings tensor of shape (seq_len, hidden_size)
        """
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        embeddings = torch.nn.functional.embedding(
            token_tensor, self.embedding_weights
        )
        return embeddings
    
    def process_with_embeddings(
        self,
        token_ids: list[int],
        placeholder_token_id: int,
        external_embeddings: list[list[float]],
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Process token IDs and inject external embeddings at placeholder positions.
        
        Args:
            token_ids: List of token IDs
            placeholder_token_id: Token ID for placeholder (e.g., <|ptoken|>)
            external_embeddings: External embeddings to inject
                Shape: (num_placeholders, hidden_size)
            use_cache: Whether to use caching
            
        Returns:
            Combined embeddings tensor of shape (seq_len, hidden_size)
        """
        import time
        start_time = time.perf_counter()
        
        # Check cache
        if use_cache:
            prompt_hash = self._compute_hash(token_ids)
            embeddings_hash = self._compute_hash(external_embeddings)
            cached = self.cache.get(prompt_hash, embeddings_hash)
            if cached is not None:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                logger.info(
                    "CACHE HIT: embedding processing completed in %.2f ms "
                    "(seq_len=%d, num_placeholders=%d)",
                    elapsed_ms, cached.token_count, len(external_embeddings)
                )
                return cached.combined_embeddings.clone()
        
        # Convert token IDs to embeddings
        token_embeddings = self.embed_tokens(token_ids)
        
        # Convert external embeddings to tensor
        external_tensor = torch.tensor(
            external_embeddings,
            dtype=self.dtype,
            device=self.device,
        )
        
        # Find placeholder positions
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        placeholder_mask = token_tensor == placeholder_token_id
        placeholder_positions = placeholder_mask.nonzero(as_tuple=False).squeeze(-1)
        
        # Validate counts match
        num_placeholders = placeholder_positions.numel()
        num_externals = external_tensor.shape[0]
        if num_placeholders != num_externals:
            raise ValueError(
                f"Mismatch: found {num_placeholders} placeholder tokens "
                f"but got {num_externals} external embeddings"
            )
        
        # Replace placeholder embeddings with external embeddings
        combined = token_embeddings.clone()
        if num_placeholders > 0:
            combined[placeholder_positions] = external_tensor
        
        # Cache the result
        if use_cache:
            entry = EmbeddingCacheEntry(
                prompt_hash=prompt_hash,
                embeddings_hash=embeddings_hash,
                combined_embeddings=combined.clone(),
                token_count=len(token_ids),
            )
            self.cache.put(entry)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "CACHE MISS: embedding processing completed in %.2f ms "
                "(seq_len=%d, num_placeholders=%d, cache_size=%d)",
                elapsed_ms, len(token_ids), num_placeholders, len(self.cache.cache)
            )
        
        return combined
    
    def process_multi_modal(
        self,
        token_ids: list[int],
        modality_embeddings: list[ModalityEmbedding],
        tokenizer,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Process token IDs with multiple modality embeddings.
        
        This method supports injecting different types of embeddings
        (vision, audio, etc.) at different placeholder positions.
        
        Args:
            token_ids: List of token IDs
            modality_embeddings: List of ModalityEmbedding objects
            tokenizer: Tokenizer to resolve placeholder token IDs
            use_cache: Whether to use caching
            
        Returns:
            Combined embeddings tensor of shape (seq_len, hidden_size)
        """
        # Convert token IDs to embeddings
        combined = self.embed_tokens(token_ids)
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        
        # Process each modality
        for modality_emb in modality_embeddings:
            # Get placeholder token ID
            placeholder_token_id = tokenizer.convert_tokens_to_ids(
                modality_emb.placeholder_token
            )
            
            # Find placeholder positions for this modality
            placeholder_mask = token_tensor == placeholder_token_id
            placeholder_positions = placeholder_mask.nonzero(as_tuple=False).squeeze(-1)
            
            # Validate counts
            num_placeholders = placeholder_positions.numel()
            num_embeddings = modality_emb.embeddings.shape[0]
            if num_placeholders != num_embeddings:
                raise ValueError(
                    f"Mismatch for modality '{modality_emb.modality}': "
                    f"found {num_placeholders} placeholder tokens "
                    f"but got {num_embeddings} embeddings"
                )
            
            # Replace embeddings
            if num_placeholders > 0:
                external_tensor = modality_emb.embeddings.to(
                    dtype=self.dtype, device=self.device
                )
                combined[placeholder_positions] = external_tensor
            
            logger.debug(
                "Processed modality '%s': %d embeddings injected",
                modality_emb.modality, num_placeholders
            )
        
        return combined
    
    def process_multimodal_with_expansion(
        self,
        token_ids: list[int],
        multimodal_data: MultiModalEmbeddings,
        discrete_placeholder_id: int,
        continuous_placeholder_id: int,
    ) -> torch.Tensor:
        """
        Process token IDs with multi-modal embeddings, expanding single placeholders.
        
        This handles the new format where:
        - A single <|DISCRETE_IMAGE_PAD|> is expanded to multiple discrete token embeddings
        - A single <|IMAGE_PAD|> is expanded to multiple continuous embeddings
        
        Args:
            token_ids: List of token IDs (with single placeholders)
            multimodal_data: MultiModalEmbeddings with discrete and continuous data
            discrete_placeholder_id: Token ID for <|DISCRETE_IMAGE_PAD|>
            continuous_placeholder_id: Token ID for <|IMAGE_PAD|>
            
        Returns:
            Combined embeddings tensor of shape (expanded_seq_len, hidden_size)
        """
        import time
        start_time = time.perf_counter()
        
        # Handle both dict and object-style access
        if isinstance(multimodal_data, dict):
            discrete_data = multimodal_data.get("discrete")
            continuous_data = multimodal_data.get("continuous")
        else:
            discrete_data = getattr(multimodal_data, "discrete", None)
            continuous_data = getattr(multimodal_data, "continuous", None)
        
        # Get discrete token embeddings from embedding table
        if discrete_data is not None:
            discrete_token_ids = discrete_data.squeeze().to(torch.long)
            discrete_embeddings = torch.nn.functional.embedding(
                discrete_token_ids.to(self.device),
                self.embedding_weights
            )
        else:
            discrete_embeddings = None
        
        # Get continuous embeddings (already in embedding format)
        if continuous_data is not None:
            continuous_embeddings = continuous_data.to(
                dtype=self.dtype, device=self.device
            )
        else:
            continuous_embeddings = None
        
        if discrete_embeddings is not None and continuous_embeddings is not None:
            logger.info(
                "Multi-modal embeddings: discrete=%d tokens -> %s embeddings, "
                "continuous=%s embeddings",
                discrete_embeddings.shape[0], discrete_embeddings.shape,
                continuous_embeddings.shape
            )
        elif discrete_embeddings is not None:
            logger.info(
                "Multi-modal embeddings: discrete=%d tokens -> %s embeddings",
                discrete_embeddings.shape[0], discrete_embeddings.shape
            )
        elif continuous_embeddings is not None:
            logger.info(
                "Multi-modal embeddings: continuous=%s embeddings",
                continuous_embeddings.shape
            )
        
        # Pre-calculate output size for efficient memory allocation
        output_size = 0
        discrete_count = 0
        continuous_count = 0
        regular_token_ids = []
        regular_positions = []
        
        for i, token_id in enumerate(token_ids):
            if token_id == discrete_placeholder_id and discrete_embeddings is not None:
                output_size += discrete_embeddings.shape[0]
                discrete_count += 1
            elif token_id == continuous_placeholder_id and continuous_embeddings is not None:
                output_size += continuous_embeddings.shape[0]
                continuous_count += 1
            else:
                output_size += 1
                regular_token_ids.append(token_id)
                regular_positions.append(output_size - 1)
        
        # Pre-allocate output tensor
        combined = torch.empty(
            (output_size, self.hidden_size),
            dtype=self.dtype,
            device=self.device
        )
        
        # Batch process regular tokens at once (more efficient than one-by-one)
        if regular_token_ids:
            regular_tensor = torch.tensor(regular_token_ids, dtype=torch.long, device=self.device)
            regular_embeddings = torch.nn.functional.embedding(regular_tensor, self.embedding_weights)
        
        # Fill the output tensor
        output_idx = 0
        regular_idx = 0
        for token_id in token_ids:
            if token_id == discrete_placeholder_id and discrete_embeddings is not None:
                size = discrete_embeddings.shape[0]
                combined[output_idx:output_idx + size] = discrete_embeddings
                output_idx += size
            elif token_id == continuous_placeholder_id and continuous_embeddings is not None:
                size = continuous_embeddings.shape[0]
                combined[output_idx:output_idx + size] = continuous_embeddings
                output_idx += size
            else:
                combined[output_idx] = regular_embeddings[regular_idx]
                regular_idx += 1
                output_idx += 1
        
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Multi-modal expansion completed in %.2f ms: "
            "input_tokens=%d -> output_embeddings=%d",
            elapsed_ms, len(token_ids), combined.shape[0]
        )
        
        return combined
    
    def get_embeddings_for_tokens(self, token_ids: list[int]) -> torch.Tensor:
        """
        Get embeddings for a list of token IDs.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Embeddings tensor of shape (len(token_ids), hidden_size)
        """
        if not token_ids:
            return torch.empty((0, self.hidden_size), dtype=self.dtype, device=self.device)
        
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=self.device)
        embeddings = torch.nn.functional.embedding(token_tensor, self.embedding_weights)
        return embeddings
    
    def process_discrete_region(
        self,
        token_ids: list[int],
        multimodal_data: dict,
        start_id: int,
        end_id: int,
        placeholder_id: int,
    ) -> torch.Tensor:
        """
        Process a discrete image region, expanding placeholders with discrete embeddings.
        
        Args:
            token_ids: Token IDs for this region (including start/end tokens)
            multimodal_data: Dict containing 'discrete' key with token IDs
            start_id: Token ID for <|discrete_image_start|>
            end_id: Token ID for <|discrete_image_end|>
            placeholder_id: Token ID for <|DISCRETE_IMAGE_PAD|>
            
        Returns:
            Embeddings tensor for the region
        """
        # Get discrete data
        discrete_data = multimodal_data.get("discrete")
        if discrete_data is None:
            # No discrete data, just return regular embeddings
            return self.get_embeddings_for_tokens(token_ids)
        
        # Get discrete token embeddings from embedding table
        discrete_token_ids = discrete_data.squeeze().to(torch.long)
        discrete_embeddings = torch.nn.functional.embedding(
            discrete_token_ids.to(self.device),
            self.embedding_weights
        )
        
        # Build output: start + discrete_embeddings + end
        parts = []
        for token_id in token_ids:
            if token_id == placeholder_id:
                # Replace single placeholder with all discrete embeddings
                parts.append(discrete_embeddings)
            else:
                # Regular token
                token_embedding = self.get_embeddings_for_tokens([token_id])
                parts.append(token_embedding)
        
        return torch.cat(parts, dim=0)
    
    def process_continuous_region(
        self,
        token_ids: list[int],
        multimodal_data: dict,
        start_id: int,
        end_id: int,
        placeholder_id: int,
    ) -> torch.Tensor:
        """
        Process a continuous image region, expanding placeholders with continuous embeddings.
        
        Args:
            token_ids: Token IDs for this region (including start/end tokens)
            multimodal_data: Dict containing 'continuous' key with embeddings
            start_id: Token ID for <|image_start|>
            end_id: Token ID for <|image_end|>
            placeholder_id: Token ID for <|IMAGE_PAD|>
            
        Returns:
            Embeddings tensor for the region
        """
        # Get continuous data
        continuous_data = multimodal_data.get("continuous")
        if continuous_data is None:
            # No continuous data, just return regular embeddings
            return self.get_embeddings_for_tokens(token_ids)
        
        # Get continuous embeddings (already in embedding format)
        continuous_embeddings = continuous_data.to(
            dtype=self.dtype, device=self.device
        )
        
        # Build output: start + continuous_embeddings + end
        parts = []
        for token_id in token_ids:
            if token_id == placeholder_id:
                # Replace single placeholder with all continuous embeddings
                parts.append(continuous_embeddings)
            else:
                # Regular token
                token_embedding = self.get_embeddings_for_tokens([token_id])
                parts.append(token_embedding)
        
        return torch.cat(parts, dim=0)
    
    def to_embeds_prompt(
        self, embeddings: torch.Tensor, cache_salt: Optional[str] = None
    ) -> dict:
        """
        Convert embeddings tensor to EmbedsPrompt format for vLLM.
        
        Args:
            embeddings: Embeddings tensor of shape (seq_len, hidden_size)
            cache_salt: Optional cache salt for the prompt
            
        Returns:
            EmbedsPrompt dictionary
        """
        # Convert to list for serialization
        embeds_list = embeddings.tolist()
        
        result = {"prompt_embeds": embeds_list}
        if cache_salt is not None:
            result["cache_salt"] = cache_salt
        
        return result
    
    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self.cache.clear()
        logger.info("Embedding cache cleared")


# Global processor instance (lazy initialization)
_processor_instance: Optional[EmbeddingProcessor] = None

# Default cache size from environment variable
_default_processor_cache_size = int(os.environ.get("VLLM_EMBEDDING_PROCESSOR_CACHE_SIZE", "1000"))
_default_processor_cache_size = max(100, min(10000, _default_processor_cache_size))


def get_embedding_processor(
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    cache_size: int | None = None,
) -> EmbeddingProcessor:
    """
    Get or create the global embedding processor instance.
    
    This function ensures only one embedding processor is created
    for memory efficiency.
    
    Args:
        model_path: Path to the model
        dtype: Data type for embeddings (default: bfloat16)
        device: Device to store embedding table (default: "cpu")
        cache_size: LRU cache size (default from VLLM_EMBEDDING_PROCESSOR_CACHE_SIZE env var)
    """
    global _processor_instance
    
    if cache_size is None:
        cache_size = _default_processor_cache_size
    
    if _processor_instance is None:
        logger.info(
            "Initializing EmbeddingProcessor: "
            "download_cache_max_items=%d, download_cache_max_memory_mb=%d, "
            "processor_cache_size=%d",
            _download_cache_max_size, _download_cache_max_memory_mb, cache_size
        )
        _processor_instance = EmbeddingProcessor(
            model_path=model_path,
            dtype=dtype,
            device=device,
            cache_size=cache_size,
        )
    
    return _processor_instance


def reset_embedding_processor() -> None:
    """Reset the global embedding processor instance."""
    global _processor_instance
    _processor_instance = None


def get_embedding_config() -> dict:
    """Get current embedding processor configuration."""
    return {
        "download_cache_max_items": _download_cache_max_size,
        "download_cache_max_memory_mb": _download_cache_max_memory_mb,
        "processor_cache_size": _default_processor_cache_size,
        "download_cache_stats": get_download_cache_stats(),
    }

