# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

from typing import List, Literal, Optional

import pydantic

Format = Literal["wav", "mp3", "flac", "ogg", "aac", "pcm"]

Gender = Literal["f", "m"]
GENDERS = ("f", "m")

Age = Literal["0-10", "10", "20-30", "40-50", "60-70"]
AGES = ("0-10", "10", "20-30", "40-50", "60-70")


class Speaker(pydantic.BaseModel):
    id: Optional[str] = None
    gender: Optional[Gender] = None
    age: Optional[Age] = None
    ref_audio_base64: Optional[pydantic.Base64Bytes] = None
    ref_audio_url: Optional[str] = None


class Request(pydantic.BaseModel):
    units: List[int]
    format: Format = "wav"
    speaker: Optional[Speaker] = None


class Response(pydantic.BaseModel):
    audio: pydantic.Base64Bytes
