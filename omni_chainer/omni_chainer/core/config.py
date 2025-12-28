# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import os
from typing import Annotated, Any, Optional

from pydantic import (
  computed_field,
  AnyUrl,
  BeforeValidator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

def parse_cors(v: Any) -> list[str] | str:
  if isinstance(v, str) and not v.startswith("["):
    return [i.strip() for i in v.split(",") if i.strip()]
  elif isinstance(v, (list, str)):
    return v
  raise ValueError(v)

# NOTE: You can override the settings by setting the environment variables.
#       Example: export LLM_ENDPOINT=http://localhost:8002/llm
class Settings(BaseSettings):
  model_config = SettingsConfigDict(
    env_file=f"{os.curdir}/.env",
    env_ignore_empty=True,
    extra="ignore",
  )

  ENV_STAGE: str = "development"

  API_STR: str = ""
  # CORS variables
  FRONTEND_HOST: str = "http://localhost:5173"
  BACKEND_CORS_ORIGINS: Annotated[
    list[AnyUrl] | str, BeforeValidator(parse_cors)
  ] = []

  # BACKENDS - Configure via environment variables
  TRACK_A_LLM_ENDPOINT: str             = "http://localhost:10021/v1/chat/completions"
  TRACK_A_VISION_ENCODING_ENDPOINT: str = "http://localhost:10004/process_image_or_video"

  TRACK_B_LLM_ENDPOINT: str             = "http://localhost:10032/v1/chat/completions"
  TRACK_B_VISION_ENCODING_ENDPOINT: str = "http://localhost:10005/process_image_or_video"
  TRACK_B_VISION_DECODING_ENDPOINT: str = "http://localhost:10063/decode"
  TRACK_B_AUDIO_ENCODING_ENDPOINT: str  = "http://localhost:10002/process_audio"
  TRACK_B_AUDIO_DECODING_ENDPOINT: str  = "http://localhost:11180/predictions"

  WBL_S3_BUCKET_NAME: str = ""
  NCP_S3_ACCESS_KEY: str = ""
  NCP_S3_SECRET_KEY: str = ""

  BACKEND_TIMEOUT: float = 3600.0

  # Embedding cache control
  ENABLE_EMBEDDING_CACHE: bool = True

  OTLP_ENDPOINT: Optional[str] = None

  @computed_field
  @property
  def all_cors_origins(self) -> list[str]:
    return [str(origin).rstrip("/") for origin in self.BACKEND_CORS_ORIGINS] + [
      self.FRONTEND_HOST
    ]

settings = Settings()  # type: ignore
