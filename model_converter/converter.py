"""
Main conversion logic for extracting model components.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set
from enum import Enum

import torch

from .weight_utils import (
    load_safetensors_index,
    filter_weights_by_prefix,
    load_weights_dict,
    save_weights_pt,
    save_weights_sharded,
    get_total_params,
    get_total_size_bytes,
    format_size,
)
from .config_utils import (
    load_config,
    save_config,
    extract_llm_config,
    extract_vision_config,
    extract_audio_config,
    copy_tokenizer_files,
    copy_model_files,
    get_model_name_from_path,
)


class Track(Enum):
    """Model track type."""
    A = "a"  # VE + LLM (no audio)
    B = "b"  # VE + AE + LLM (full omni)


class OmniModelConverter:
    """
    Converter for extracting individual components from unified Omni models.

    Supports:
    - Track B: Omni model -> VE, AE, LLM
    - Track A: VE+LLM model -> VE, LLM
    """

    # Weight prefix mappings
    PREFIXES = {
        "vision_model": "model.vision_model.",
        "mm_projector": "model.mm_projector.",
        "discrete_vision_model": "model.discrete_vision_model.",
        "audio_model": "model.audio_model.",
        "audio_projector": "model.audio_projector.",
        "discrete_audio_model": "model.discrete_audio_model.",
        "video_audio_compressor": "model.video_audio_compressor.",
        "language_model": "model.language_model.",
    }

    # Key transformation for LLM (remove model.language_model. prefix)
    LLM_KEY_MAPPING = {
        "model.language_model.model.": "model.",
        "model.language_model.lm_head.": "lm_head.",
        "model.language_model.": "",  # fallback for any other keys
    }

    def __init__(self, model_dir: str, track: Track = Track.B):
        """
        Initialize converter.

        Args:
            model_dir: Path to unified model directory
            track: Model track (A or B)
        """
        self.model_dir = Path(model_dir)
        self.track = track
        self.model_name = get_model_name_from_path(model_dir)

        # Load config and index
        self.config = load_config(model_dir)
        self.index = load_safetensors_index(model_dir)

        print(f"Loaded model: {self.model_name}")
        print(f"Track: {track.value.upper()}")
        print(f"Total weights: {len(self.index['weight_map'])}")
        print(f"Total size: {format_size(self.index['metadata']['total_size'])}")

    def _get_available_components(self) -> Set[str]:
        """Get list of available components in the model."""
        available = set()
        weight_keys = set(self.index["weight_map"].keys())

        for component, prefix in self.PREFIXES.items():
            if any(k.startswith(prefix) for k in weight_keys):
                available.add(component)

        return available

    def extract_vision_encoder(
        self,
        output_dir: str,
        include_projector: bool = True,
        include_discrete: bool = True,
        dtype: torch.dtype = torch.bfloat16
    ) -> str:
        """
        Extract Vision Encoder weights.

        Args:
            output_dir: Output directory
            include_projector: Include mm_projector weights
            include_discrete: Include discrete_vision_model (ta_tok) weights
            dtype: Dtype to save weights in (default: bfloat16)

        Returns:
            Path to output directory
        """
        print("\n=== Extracting Vision Encoder ===")

        output_path = Path(output_dir) / "ve" / self.model_name
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Extract vision_model weights
        print("Extracting vision_model weights...")
        weights_by_file = filter_weights_by_prefix(
            self.index, [self.PREFIXES["vision_model"]]
        )
        weights = load_weights_dict(self.model_dir, weights_by_file)
        save_weights_pt(
            weights,
            str(output_path / "vision_weights.pt"),
            prefix_strip=self.PREFIXES["vision_model"],
            dtype=dtype
        )
        print(f"  Params: {get_total_params(weights):,}")

        # 2. Extract mm_projector weights
        if include_projector:
            print("Extracting mm_projector weights...")
            weights_by_file = filter_weights_by_prefix(
                self.index, [self.PREFIXES["mm_projector"]]
            )
            if weights_by_file:
                weights = load_weights_dict(self.model_dir, weights_by_file)
                save_weights_pt(
                    weights,
                    str(output_path / "mm_projector_weights.pt"),
                    prefix_strip=self.PREFIXES["mm_projector"],
                    dtype=dtype
                )
                print(f"  Params: {get_total_params(weights):,}")

        # 3. Extract discrete_vision_model (ta_tok) weights
        if include_discrete:
            print("Extracting discrete_vision_model (ta_tok) weights...")
            weights_by_file = filter_weights_by_prefix(
                self.index, [self.PREFIXES["discrete_vision_model"]]
            )
            if weights_by_file:
                weights = load_weights_dict(self.model_dir, weights_by_file)

                # Strip prefix and convert dtype for state_dict
                state_dict = {}
                prefix = self.PREFIXES["discrete_vision_model"]
                for k, v in weights.items():
                    new_key = k[len(prefix):] if k.startswith(prefix) else k
                    if dtype is not None:
                        v = v.to(dtype)
                    state_dict[new_key] = v

                # Save in the expected format: {"model": {"args": {...}, "sd": {...}}}
                # args contains model configuration, sd contains state_dict
                from easydict import EasyDict
                ta_tok_checkpoint = {
                    "model": {
                        "args": EasyDict({
                            "bottleneck": {
                                "name": "bottleneck",
                                "args": {
                                    "bottleneck_dim": 1536,
                                    "norm": "none",
                                    "regularizer": {
                                        "name": "simvq",
                                        "args": {
                                            "codebook_size": 65536,
                                            "commitment_loss_weight": 0.25,
                                            "codebook_loss_weight": 1.0,
                                            "entropy_loss_weight": 0.0,
                                            "entropy_loss_temperature": 0.01,
                                            "l2_normalized": True,
                                            "stochastic": True,
                                            "stochastic_temperature": 0.03,
                                            "top_k": 4,
                                            "top_k_prob": 0.5,
                                            "residual_weight": 0.1
                                        }
                                    }
                                }
                            },
                            "bottleneck_token_num": 729,
                            "input_size": 384,
                            "teacher": "google/siglip2-so400m-patch14-384",
                            "ckpt_path": "google/siglip2-so400m-patch14-384",
                            "pool_scale": 1,
                            "rand_scale": True
                        }),
                        "sd": state_dict
                    }
                }
                torch.save(ta_tok_checkpoint, str(output_path / "ta_tok.pth"))
                print(f"  Params: {get_total_params(weights):,}")

        print(f"Vision Encoder saved to: {output_path}")
        return str(output_path)

    def extract_audio_encoder(
        self,
        output_dir: str,
        include_projector: bool = True,
        include_discrete: bool = True,
        include_compressor: bool = True,
        include_qwen2_encoder: bool = True,
        dtype: torch.dtype = torch.bfloat16
    ) -> str:
        """
        Extract Audio Encoder weights (Track B only).

        Args:
            output_dir: Output directory
            include_projector: Include audio_projector weights
            include_discrete: Include discrete_audio_model weights
            include_compressor: Include video_audio_compressor weights
            include_qwen2_encoder: Save Qwen2 audio encoder in HF format
            dtype: Dtype to save weights in (default: bfloat16)

        Returns:
            Path to output directory
        """
        if self.track != Track.B:
            raise ValueError("Audio encoder extraction is only available for Track B")

        print("\n=== Extracting Audio Encoder ===")

        output_path = Path(output_dir) / "ae" / self.model_name
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Extract audio_model weights
        print("Extracting audio_model weights...")
        weights_by_file = filter_weights_by_prefix(
            self.index, [self.PREFIXES["audio_model"]]
        )
        weights = load_weights_dict(self.model_dir, weights_by_file)
        save_weights_pt(
            weights,
            str(output_path / "audio_weights.pt"),
            prefix_strip=self.PREFIXES["audio_model"],
            dtype=dtype
        )
        print(f"  Params: {get_total_params(weights):,}")

        # 1b. Also save as Qwen2AudioEncoder format if requested
        if include_qwen2_encoder:
            print("Saving Qwen2 Audio Encoder in HF format...")
            qwen2_path = output_path / "qwen2-audio-encoder-from-qwen2-audio-7b-instruct"
            qwen2_path.mkdir(parents=True, exist_ok=True)

            # Save weights with Qwen2 format key mapping and dtype conversion
            qwen2_weights = {}
            for k, v in weights.items():
                # Remove the model.audio_model. prefix if present
                if k.startswith("model.audio_model."):
                    new_key = k[len("model.audio_model."):]
                else:
                    new_key = k
                # Convert dtype
                if dtype is not None:
                    v = v.to(dtype)
                qwen2_weights[new_key] = v

            from safetensors.torch import save_file as save_safetensors
            save_safetensors(qwen2_weights, str(qwen2_path / "model.safetensors"))
            print(f"  Saved model.safetensors")

            # Save audio config
            audio_config = extract_audio_config(self.config)
            save_config(audio_config, str(qwen2_path))

            # Save audio preprocessor config (WhisperFeatureExtractor with 128 mel bins)
            # Note: The omni model's preprocessor_config.json is for vision, not audio
            audio_preprocessor_config = {
                "chunk_length": 30,
                "feature_extractor_type": "WhisperFeatureExtractor",
                "feature_size": 128,
                "hop_length": 160,
                "n_fft": 400,
                "n_samples": 480000,
                "nb_max_frames": 3000,
                "padding_side": "right",
                "padding_value": 0.0,
                "processor_class": "Qwen2AudioProcessor",
                "return_attention_mask": True,
                "sampling_rate": 16000
            }
            with open(qwen2_path / "preprocessor_config.json", "w") as f:
                json.dump(audio_preprocessor_config, f, indent=2)
            print(f"  Created preprocessor_config.json (WhisperFeatureExtractor, 128 mel bins)")

        # 2. Extract audio_projector weights
        if include_projector:
            print("Extracting audio_projector weights...")
            weights_by_file = filter_weights_by_prefix(
                self.index, [self.PREFIXES["audio_projector"]]
            )
            if weights_by_file:
                weights = load_weights_dict(self.model_dir, weights_by_file)
                save_weights_pt(
                    weights,
                    str(output_path / "audio_projector_weights.pt"),
                    prefix_strip=self.PREFIXES["audio_projector"],
                    dtype=dtype
                )
                print(f"  Params: {get_total_params(weights):,}")

        # 3. Extract discrete_audio_model weights
        if include_discrete:
            print("Extracting discrete_audio_model weights...")
            weights_by_file = filter_weights_by_prefix(
                self.index, [self.PREFIXES["discrete_audio_model"]]
            )
            if weights_by_file:
                weights = load_weights_dict(self.model_dir, weights_by_file)
                save_weights_pt(
                    weights,
                    str(output_path / "discrete_audio_weights.pt"),
                    prefix_strip=self.PREFIXES["discrete_audio_model"],
                    dtype=dtype
                )
                print(f"  Params: {get_total_params(weights):,}")

        # 4. Extract video_audio_compressor weights
        if include_compressor:
            print("Extracting video_audio_compressor weights...")
            weights_by_file = filter_weights_by_prefix(
                self.index, [self.PREFIXES["video_audio_compressor"]]
            )
            if weights_by_file:
                weights = load_weights_dict(self.model_dir, weights_by_file)
                save_weights_pt(
                    weights,
                    str(output_path / "video_audio_compressor_weights.pt"),
                    prefix_strip=self.PREFIXES["video_audio_compressor"],
                    dtype=dtype
                )
                print(f"  Params: {get_total_params(weights):,}")

        print(f"Audio Encoder saved to: {output_path}")
        return str(output_path)

    def extract_llm(
        self,
        output_dir: str,
        max_shard_size: int = 5 * 1024 * 1024 * 1024  # 5GB
    ) -> str:
        """
        Extract LLM weights in HuggingFace format.

        Args:
            output_dir: Output directory
            max_shard_size: Maximum shard size in bytes

        Returns:
            Path to output directory
        """
        print("\n=== Extracting LLM ===")

        output_path = Path(output_dir) / "llm" / self.model_name
        output_path.mkdir(parents=True, exist_ok=True)

        # 1. Extract language_model weights
        print("Extracting language_model weights...")
        weights_by_file = filter_weights_by_prefix(
            self.index, [self.PREFIXES["language_model"]]
        )

        # Save as sharded safetensors
        save_weights_sharded(
            str(self.model_dir),
            weights_by_file,
            str(output_path),
            key_mapping=self.LLM_KEY_MAPPING,
            max_shard_size=max_shard_size
        )

        # 2. Extract and save LLM config
        print("Saving LLM config...")
        llm_config = extract_llm_config(self.config)
        save_config(llm_config, str(output_path))

        # 3. Copy tokenizer files
        print("Copying tokenizer files...")
        copy_tokenizer_files(str(self.model_dir), str(output_path))

        # 4. Copy model implementation files (for custom models like HyperCLOVAX)
        print("Copying model implementation files...")
        copy_model_files(str(self.model_dir), str(output_path), [
            "modeling_*.py",
            "configuration_*.py",
            "generation_config.json",
        ])

        print(f"LLM saved to: {output_path}")
        return str(output_path)

    def copy_vision_decoder(self, output_dir: str) -> str:
        """
        Copy Vision Decoder directory (Track B only).

        Args:
            output_dir: Output directory

        Returns:
            Path to output directory
        """
        if self.track != Track.B:
            raise ValueError("Vision decoder is only available for Track B")

        print("\n=== Copying Vision Decoder ===")

        src_path = self.model_dir / "decoder" / "vision"
        if not src_path.exists():
            print(f"Warning: Vision decoder not found at {src_path}")
            return ""

        output_path = Path(output_dir) / "vd" / self.model_name
        shutil.copytree(src_path, output_path, dirs_exist_ok=True)

        print(f"Vision Decoder copied to: {output_path}")
        return str(output_path)

    def copy_audio_decoder(self, output_dir: str) -> str:
        """
        Copy Audio Decoder directory (Track B only).

        Args:
            output_dir: Output directory

        Returns:
            Path to output directory
        """
        if self.track != Track.B:
            raise ValueError("Audio decoder is only available for Track B")

        print("\n=== Copying Audio Decoder ===")

        src_path = self.model_dir / "decoder" / "audio"
        if not src_path.exists():
            print(f"Warning: Audio decoder not found at {src_path}")
            return ""

        output_path = Path(output_dir) / "ad" / self.model_name
        shutil.copytree(src_path, output_path, dirs_exist_ok=True)

        print(f"Audio Decoder copied to: {output_path}")
        return str(output_path)

    def convert(
        self,
        output_dir: str,
        components: Optional[List[str]] = None,
        max_shard_size: int = 5 * 1024 * 1024 * 1024
    ) -> Dict[str, str]:
        """
        Convert model by extracting specified components.

        Args:
            output_dir: Output directory
            components: List of components to extract ('ve', 'ae', 'llm', 'vd', 'ad', or 'all')
            max_shard_size: Maximum shard size for LLM in bytes

        Returns:
            Dict mapping component name to output path
        """
        if components is None or "all" in components:
            if self.track == Track.B:
                components = ["ve", "ae", "llm", "vd", "ad"]
            else:
                components = ["ve", "llm"]

        results = {}

        if "ve" in components:
            results["ve"] = self.extract_vision_encoder(output_dir)

        if "ae" in components:
            if self.track == Track.B:
                results["ae"] = self.extract_audio_encoder(output_dir)
            else:
                print("Warning: Audio encoder not available for Track A")

        if "llm" in components:
            results["llm"] = self.extract_llm(output_dir, max_shard_size)

        if "vd" in components:
            if self.track == Track.B:
                result = self.copy_vision_decoder(output_dir)
                if result:
                    results["vd"] = result
            else:
                print("Warning: Vision decoder not available for Track A")

        if "ad" in components:
            if self.track == Track.B:
                result = self.copy_audio_decoder(output_dir)
                if result:
                    results["ad"] = result
            else:
                print("Warning: Audio decoder not available for Track A")

        print("\n=== Conversion Complete ===")
        for component, path in results.items():
            print(f"  {component.upper()}: {path}")

        return results

    def verify(self, output_dir: str) -> bool:
        """
        Verify extracted components by loading them.

        Args:
            output_dir: Output directory containing extracted components

        Returns:
            True if verification passes
        """
        print("\n=== Verifying Extracted Components ===")
        success = True

        # Check VE
        ve_path = Path(output_dir) / "ve" / self.model_name
        if ve_path.exists():
            print("Checking Vision Encoder...")
            try:
                vision_weights = torch.load(ve_path / "vision_weights.pt")
                print(f"  vision_weights.pt: {len(vision_weights)} tensors")

                if (ve_path / "mm_projector_weights.pt").exists():
                    proj_weights = torch.load(ve_path / "mm_projector_weights.pt")
                    print(f"  mm_projector_weights.pt: {len(proj_weights)} tensors")
            except Exception as e:
                print(f"  Error: {e}")
                success = False

        # Check AE (Track B only)
        if self.track == Track.B:
            ae_path = Path(output_dir) / "ae" / self.model_name
            if ae_path.exists():
                print("Checking Audio Encoder...")
                try:
                    audio_weights = torch.load(ae_path / "audio_weights.pt")
                    print(f"  audio_weights.pt: {len(audio_weights)} tensors")
                except Exception as e:
                    print(f"  Error: {e}")
                    success = False

        # Check LLM
        llm_path = Path(output_dir) / "llm" / self.model_name
        if llm_path.exists():
            print("Checking LLM...")
            try:
                config = load_config(str(llm_path))
                print(f"  config.json: model_type={config.get('model_type', 'unknown')}")

                index_path = llm_path / "model.safetensors.index.json"
                if index_path.exists():
                    with open(index_path) as f:
                        llm_index = json.load(f)
                    print(f"  Shards: {len(set(llm_index['weight_map'].values()))}")
                    print(f"  Weights: {len(llm_index['weight_map'])}")
            except Exception as e:
                print(f"  Error: {e}")
                success = False

        return success
