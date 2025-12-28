# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

from typing import Optional

import pydantic
import pydantic_settings


# Load from environment variables
class Settings(pydantic_settings.BaseSettings):
    endpoint: str = "http://localhost:8081/predictions"
    zeroshot_model: str = "NCZSCosybigvganDecoder"
    finetuned_model: str = "NCCosybigvganDecoder"
    default_speaker: str = "fkms"
    speaker_config_path: Optional[str] = None

    s3_endpoint: str = pydantic.Field(default="", validation_alias="NCP_S3_ENDPOINT")
    s3_region: str = pydantic.Field(default="kr-standard", validation_alias="NCP_S3_REGION")
    s3_access_key: Optional[str] = pydantic.Field(default=None, validation_alias="NCP_S3_ACCESS_KEY")
    s3_secret_key: Optional[str] = pydantic.Field(default=None, validation_alias="NCP_S3_SECRET_KEY")
