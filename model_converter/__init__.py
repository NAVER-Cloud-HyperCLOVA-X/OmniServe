"""
Omni Model Converter

A tool for extracting individual components (Vision Encoder, Audio Encoder, LLM)
from unified Omni models.

Supports:
- Track B: Omni model -> VE, AE, LLM
- Track A: VE+LLM model -> VE, LLM
"""

from .converter import OmniModelConverter
from .weight_utils import load_safetensors_index, filter_weights_by_prefix, save_weights_pt
from .config_utils import extract_llm_config, extract_vision_config, extract_audio_config

__version__ = "1.0.0"
__all__ = [
    "OmniModelConverter",
    "load_safetensors_index",
    "filter_weights_by_prefix",
    "save_weights_pt",
    "extract_llm_config",
    "extract_vision_config",
    "extract_audio_config",
]
