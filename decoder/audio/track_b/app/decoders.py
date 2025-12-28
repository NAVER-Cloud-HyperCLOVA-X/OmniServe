# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

"""Decode audio with POST API."""

import base64
import io
import logging
import wave
from typing import Any, BinaryIO, Dict, List

import aiohttp
import pydub

from . import exceptions, references, server_types


async def request(
    endpoint: str, model_name: str, units: List[int], speaker: references.Reference
) -> bytes:
    """Decode audio units into wav bytes."""
    data = {"unit": units, "format": "wav"}
    if isinstance(speaker, references.FinetunedReference):
        data["speaker"] = speaker.speaker_id
    else:
        data["ref_audio"] = base64.b64encode(speaker.raw_audio).decode("ascii")

    async with aiohttp.ClientSession() as session:
        async with session.post(f"{endpoint}/{model_name}", json=data) as response:
            try:
                response.raise_for_status()
            except aiohttp.ClientResponseError as e:
                logging.error("Error while requesting to model server: %s", e)
                raise exceptions.ServerError(
                    "Internal Server Error while requesting to model server."
                )
            audio = await response.read()
            return audio


def _get_pcm_from_wav(wav: BinaryIO) -> bytes:
    with wave.open(wav, "rb") as wave_file:
        num_frames = wave_file.getnframes()
        return wave_file.readframes(num_frames)


def convert_audio(wav: bytes, format: server_types.Format) -> bytes:
    src = io.BytesIO(wav)

    if format == "pcm":
        return _get_pcm_from_wav(src)

    segment = pydub.AudioSegment.from_wav(src)

    dest = io.BytesIO()
    # NOTE: format should be adts for aac
    export_kwargs: Dict[str, Any] = {"format": "adts" if format == "aac" else format}
    if format == "mp3":
        export_kwargs["bitrate"] = "320k"

    segment.export(dest, **export_kwargs)
    return dest.getvalue()
