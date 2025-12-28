# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0
#
# Portions of this file are adapted from:
# https://github.com/fastapi/full-stack-fastapi-template/blob/master/backend/app/api/main.py

from typing import Literal
from pydantic import BaseModel
from fastapi import APIRouter

from omni_chainer.api.routes.openai.v1 import chat
from omni_chainer.core.config import settings


api_router = APIRouter()

root_router = APIRouter(prefix="", tags=["Health Check"])


class HealthResponse(BaseModel):
  status: Literal["ok"] = "ok"


@root_router.get("/health")
async def health() -> HealthResponse:
  return HealthResponse()


api_router.include_router(root_router)
api_router.include_router(chat.track_a_router)
api_router.include_router(chat.track_b_router)
