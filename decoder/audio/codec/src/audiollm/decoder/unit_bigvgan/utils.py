# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import functools
import io
import json
import logging
import math
import os
from typing import Optional, Tuple

import librosa
import numpy as np
import pydub
import torch

from audiollm.decoder.unit_bigvgan.modules import BigVGAN
from audiollm.utils import audio_utils

logger = logging.getLogger(__name__)


def compile(model: BigVGAN):
    params_json = os.getenv("AUDIOLLM_TORCH_COMPILE_PARAMS")
    if not params_json:
        return model

    try:
        params = json.loads(params_json)
    except json.JSONDecodeError:
        logger.warning(
            "AUDIOLLM_TORCH_COMPILE_PARAMS environment variable is an invalid JSON. Skipping model compile..."
        )
        return model

    logger.info("Compiling model...")
    return torch.compile(model, **params)


def to_dtype(dtype_name: str):
    dtype_name = dtype_name.strip().lower()
    if dtype_name == "float32":
        return torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype_name}")


@functools.cache
def get_pad_multiple() -> Optional[int]:
    pad_multiple_str = os.getenv("AUDIOLLM_PAD_MULTIPLE")
    if not pad_multiple_str:
        return None

    try:
        pad_multiple = int(pad_multiple_str)
    except ValueError:
        logger.warning(
            "AUDIOLLM_PAD_MULTIPLE environment variable is not a valid int. Skipping padding..."
        )
        return None

    if pad_multiple <= 0:
        logger.warning(
            "AUDIOLLM_PAD_MULTIPLE environment variable is not a positive int. Skipping padding..."
        )
        return None

    return pad_multiple


@functools.cache
def _get_pad_token_id() -> Optional[int]:
    pad_token_id_str = os.getenv("AUDIOLLM_PAD_TOKEN_ID")
    if not pad_token_id_str:
        logger.warning(
            "AUDIOLLM_PAD_TOKEN_ID environment variable is not set. Skipping padding..."
        )
        return None

    try:
        pad_token_id = int(pad_token_id_str)
    except ValueError:
        logger.warning(
            "AUDIOLLM_PAD_TOKEN_ID environment variable is not a valid int. Skipping padding..."
        )
        return None

    if pad_token_id < 0:
        logger.warning(
            "AUDIOLLM_PAD_TOKEN_ID environment variable is a negative int. Skipping padding..."
        )
        return None

    return pad_token_id


def pad(unit: torch.Tensor) -> Tuple[torch.Tensor, float]:
    """Pad the `unit` tensor to AUDIOLLM_PAD_MULTIPLE environment variable.

    Args:
        unit: int tensor of shape [1, L]
    """

    pad_multiple = get_pad_multiple()
    if not pad_multiple:
        return unit, 1.0

    pad_token_id = _get_pad_token_id()
    if pad_token_id is None:
        return unit, 1.0

    overflow = unit.shape[1] % pad_multiple
    pad_amount = pad_multiple - overflow
    padded = torch.nn.functional.pad(
        unit, (0, pad_amount), mode="constant", value=pad_token_id
    )
    return padded, unit.shape[-1] / padded.shape[-1]


def unpad(x: torch.Tensor, original_portion: float) -> torch.Tensor:
    """Unpad the `x` tensor by retaining only the `original_portion`.

    Args:
        x: tensor of shape [..., T]
        original_portion: ratio of original unit length over padded unit length
    """
    return x[..., : math.ceil(x.shape[-1] * original_portion)]


@functools.cache
def get_warmup_port() -> Optional[int]:
    port_str = os.getenv("AUDIOLLM_INFERENCE_PORT")
    if not port_str:
        logger.warning(
            "AUDIOLLM_INFERENCE_PORT environment variable is not set. Skipping warmup..."
        )
        return None

    try:
        port = int(port_str)
    except ValueError:
        logger.warning(
            "AUDIOLLM_INFERENCE_PORT environment variable is not a valid int. Skipping warmup..."
        )
        return None

    if port <= 0:
        logger.warning(
            "AUDIOLLM_INFERENCE_PORT environment variable is not a positive int. Skipping warmup..."
        )
        return None

    return port


@functools.cache
def get_warmup_max_tokens() -> Optional[int]:
    max_tokens_str = os.getenv("AUDIOLLM_WARMUP_MAX_TOKENS")
    if not max_tokens_str:
        logger.warning(
            "AUDIOLLM_WARMUP_MAX_TOKENS environment variable is not set. Skipping warmup..."
        )
        return None

    try:
        max_tokens = int(max_tokens_str)
    except ValueError:
        logger.warning(
            "AUDIOLLM_WARMUP_MAX_TOKENS environment variable is not a valid int. Skipping warmup..."
        )
        return None

    if max_tokens <= 0:
        logger.warning(
            "AUDIOLLM_WARMUP_MAX_TOKENS environment variable is not a positive int. Skipping warmup..."
        )
        return None

    return max_tokens


@functools.cache
def _get_down_sample_rate() -> Optional[float]:
    down_sample_rate_str = os.getenv("AUDIOLLM_DOWN_SAMPLE_RATE")
    if not down_sample_rate_str:
        logger.warning(
            "AUDIOLLM_DOWN_SAMPLE_RATE environment variable is not set. Skipping down-sampling..."
        )
        return None

    try:
        down_sample_rate = float(down_sample_rate_str)
    except ValueError:
        logger.warning(
            "AUDIOLLM_DOWN_SAMPLE_RATE environment variable is not a valid float. Skipping down-sampling..."
        )
        return None

    if down_sample_rate <= 0:
        logger.warning(
            "AUDIOLLM_DOWN_SAMPLE_RATE environment variable is not a positive float. Skipping down-sampling..."
        )
        return None

    return down_sample_rate


def _detect_audio_format(audio: bytes) -> Optional[str]:
    """Detect audio format from file signature (magic bytes)."""
    if audio[:4] == b'RIFF':
        return 'wav'
    if audio[:4] == b'\x1a\x45\xdf\xa3':  # webm/mkv (EBML header)
        return 'webm'
    if audio[:4] == b'OggS':
        return 'ogg'
    if audio[:4] == b'fLaC':
        return 'flac'
    if audio[:3] == b'ID3' or audio[:2] == b'\xff\xfb':
        return 'mp3'
    if audio[:4] == b'\x00\x00\x00\x1c' or audio[:4] == b'\x00\x00\x00\x20':
        return 'mp4'  # m4a/aac
    return None


def _to_wav_file(audio: bytes) -> io.BytesIO:
    src = io.BytesIO(audio)
    fmt = _detect_audio_format(audio)

    if fmt:
        segment = pydub.AudioSegment.from_file(src, format=fmt)
    else:
        segment = pydub.AudioSegment.from_file(src)

    dest = io.BytesIO()
    segment.export(dest, format="wav")
    dest.seek(0)  # Reset position for subsequent reads
    return dest


def load_reference_audio(audio: bytes, sample_rate: float) -> np.ndarray:
    wav_file = _to_wav_file(audio)

    # Down-sample to reduce noise in final result.
    load_sr = _get_down_sample_rate()
    if load_sr is None:
        load_sr = sample_rate
    pcm, sr = librosa.load(wav_file, sr=load_sr, mono=True)
    pcm = librosa.resample(pcm, orig_sr=sr, target_sr=sample_rate)

    pcm = audio_utils.hpf_normalize(pcm, sample_rate, audio_utils.VOLUME_LEVEL)
    return pcm
