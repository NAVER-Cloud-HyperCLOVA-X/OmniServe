# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

from typing import Tuple

import httpx
from fastapi import Header
from pydantic import BaseModel

from . import _tools as tools
from omni_chainer.models.openai.v1.chat.completions import (
  ChatCompletionRequest,
  ChatCompletionResponse,
  CustomChatCompletionContentPartImageParam,
)
from omni_chainer.core.config import settings
from omni_chainer.logging import logger
import omni_chainer.core.omni._tools as tools


class VisionEncodingRequest(BaseModel):
  media_url: str


class VisionEncodingResponse(BaseModel):
  latency: float
  s3_key: str


async def handle_stream_llm(request: ChatCompletionRequest, headers: Header) -> httpx.Response:
  client = await tools.get_http_client()

  request_json = tools.flatten_extra_body(request)

  request_headers = dict(headers) if headers else {}
  if request.model:
    request_headers["X-Gateway-Model-Name"] = request.model

  logger.debug(f"TRACK_A_LLM_ENDPOINT: {settings.TRACK_A_LLM_ENDPOINT}")
  logger.debug(f"sending payload: {request_json}, headers: {request_headers}")

  try:
    req = client.build_request("POST", settings.TRACK_A_LLM_ENDPOINT, json=request_json, headers=request_headers)
    response = await client.send(req, stream=True)
    tools.record_event("track_a_llm_request", request_json)
    tools.record_event("track_a_llm_response", "stream response")
    # 스트리밍 응답에서 에러 시 content를 먼저 읽어야 exception handler에서 접근 가능
    if response.is_error:
      await response.aread()
    response.raise_for_status()
  except httpx.ConnectError as e:
    raise httpx.HTTPError(message=f"Failed to connect track b llm endpoint. {e}")
  except httpx.ConnectTimeout as e:
    raise httpx.HTTPError(message=f"Timeout while connecting to track b llm endpoint. {e}")

  return response


async def handle_llm(request: ChatCompletionRequest, headers: Header) -> ChatCompletionResponse:
  client = await tools.get_http_client()

  request_json = tools.flatten_extra_body(request)

  request_headers = dict(headers) if headers else {}
  if request.model:
    request_headers["X-Gateway-Model-Name"] = request.model

  logger.debug(f"TRACK_A_LLM_ENDPOINT: {settings.TRACK_A_LLM_ENDPOINT}")
  logger.debug(f"sending payload: {request_json}, headers: {request_headers}")

  try:
    response = await client.post(settings.TRACK_A_LLM_ENDPOINT, json=request_json, headers=request_headers)
    tools.record_event("track_a_llm_request", request_json)
    tools.record_event("track_a_llm_response", response.json())
    response.raise_for_status()
  except httpx.ConnectError as e:
    raise httpx.HTTPError(message=f"Failed to connect track b llm endpoint. {e}")
  except httpx.ConnectTimeout as e:
    raise httpx.HTTPError(message=f"Timeout while connecting to track b llm endpoint. {e}")

  return ChatCompletionResponse(**response.json()), response.headers


async def handle_vision_encoding(media_content: CustomChatCompletionContentPartImageParam, headers: Header) -> Tuple[VisionEncodingResponse, Header]:
  client = await tools.get_http_client()

  request = VisionEncodingRequest(media_url=media_content.image_url["url"])
  logger.debug(f"TRACK_A_VISION_ENCODING_ENDPOINT: {settings.TRACK_A_VISION_ENCODING_ENDPOINT}")
  logger.debug(f"sending payload: {request}, headers: {headers}")

  try:
    response = await client.post(settings.TRACK_A_VISION_ENCODING_ENDPOINT, json=request.model_dump(), headers=headers)
    tools.record_event("track_a_vision_encoding_request", request.model_dump_json(exclude_none=True))
    tools.record_event("track_a_vision_encoding_response", response.json())
    response.raise_for_status()
  except httpx.ConnectError as e:
    raise httpx.HTTPError(message=f"Failed to connect track b llm endpoint. {e}")
  except httpx.ConnectTimeout as e:
    raise httpx.HTTPError(message=f"Timeout while connecting to track b llm endpoint. {e}")

  return VisionEncodingResponse(**response.json()), response.headers
