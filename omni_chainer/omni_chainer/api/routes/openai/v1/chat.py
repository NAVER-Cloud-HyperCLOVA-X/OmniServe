# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import urllib
from typing import Any
import logging
import urllib.parse
from http import HTTPStatus

from fastapi import APIRouter, HTTPException
from fastapi.requests import Request
from fastapi.responses import Response

from opentelemetry import trace
from openai.types.chat.chat_completion_audio import (
  ChatCompletionAudio as OpenAIChatCompletionAudio
)

from omni_chainer.core.config import settings
from omni_chainer.models.openai.v1.chat.completions import (
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionStreamResponse,
  TokenizeRequest,
  TokenizeResponse,
  ErrorResponse,
)
from omni_chainer.core.openai.v1.track_a import (
  handle_chat_completion as track_a_handle_chat_completion,
)
from omni_chainer.core.openai.v1.track_b import (
  handle_chat_completion as track_b_handle_chat_completion,
)
from omni_chainer.core.omni._tools import (
  get_http_client,
  get_trace_id_headers,
)


logger = logging.getLogger("omni_chainer")

track_a_router = APIRouter(prefix="/a", tags=["Track A"])
track_b_router = APIRouter(prefix="/b", tags=["Track B"])


@track_a_router.get("/v1/{path}")
async def proxy_get(path: str, raw_request: Request) -> dict:
  client = await get_http_client()
  url = urllib.parse.urlparse(settings.TRACK_A_LLM_ENDPOINT)
  response = await client.get(f"{url.scheme}://{url.netloc}/v1/{path}")
  return response.json()


@track_a_router.post(
  "/tokenize",
  summary="Proxy of track A tokenize",
  description="Proxy of track A tokenize.",
  responses={
    HTTPStatus.OK: {
      "description": "Successful Response",
      "content": {
        "application/json": {
          "schema": {"$ref": "#/components/schemas/TokenizeResponse"}
        }
      }
    },
    HTTPStatus.BAD_REQUEST: {"model": ErrorResponse},
    HTTPStatus.NOT_FOUND: {"model": ErrorResponse},
    HTTPStatus.UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
    HTTPStatus.INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
  },
)
async def proxy_post(request: TokenizeRequest, raw_request: Request) -> TokenizeResponse:
  client = await get_http_client()
  url = urllib.parse.urlparse(settings.TRACK_A_LLM_ENDPOINT)
  httpx_request = client.build_request(
      method="POST",
      url=f"{url.scheme}://{url.netloc}/tokenize",
      headers=raw_request.headers,
      content=await raw_request.body()
  )
  response = await client.send(httpx_request)
  return response.json()


@track_b_router.get("/v1/{path}")
async def proxy_get(path: str, raw_request: Request) -> dict:
  client = await get_http_client()
  url = urllib.parse.urlparse(settings.TRACK_B_LLM_ENDPOINT)
  response = await client.get(f"{url.scheme}://{url.netloc}/v1/{path}")
  return response.json()


@track_b_router.post(
  "/tokenize",
  summary="Proxy of track B tokenize",
  description="Proxy of track B tokenize.",
  responses={
    HTTPStatus.OK: {
      "description": "Successful Response",
      "content": {
        "application/json": {
          "schema": {"$ref": "#/components/schemas/TokenizeResponse"}
        }
      }
    },
    HTTPStatus.BAD_REQUEST: {"model": ErrorResponse},
    HTTPStatus.NOT_FOUND: {"model": ErrorResponse},
    HTTPStatus.UNPROCESSABLE_ENTITY: {"model": ErrorResponse},
    HTTPStatus.INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
  },
)
async def proxy_post(request: TokenizeRequest, raw_request: Request) -> TokenizeResponse:
  client = await get_http_client()
  url = urllib.parse.urlparse(settings.TRACK_B_LLM_ENDPOINT)
  httpx_request = client.build_request(
      method="POST",
      url=f"{url.scheme}://{url.netloc}/tokenize",
      headers=raw_request.headers,
      content=await raw_request.body()
  )
  response = await client.send(httpx_request)
  return response.json()


@track_a_router.post(
  "/v1/chat/completions",
  summary="Create chat completion",
  description="""
Creates a chat completion response. 

Returns different formats based on the `stream` parameter:
- `stream=false`: Returns ChatCompletionResponse as application/json
- `stream=true`: Returns ChatCompletionStreamResponse as text/event-stream (SSE)
""",
  responses={
    HTTPStatus.OK: {
      "description": "Successful Response",
      "content": {
        "application/json": {
          "schema": {"$ref": "#/components/schemas/ChatCompletionResponse"}
        },
        "text/event-stream": {
          "schema": {"$ref": "#/components/schemas/ChatCompletionStreamResponse"}
        }
      }
    },
    HTTPStatus.BAD_REQUEST: {"model": ErrorResponse},
    HTTPStatus.NOT_FOUND: {"model": ErrorResponse},
    HTTPStatus.INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
  },
)
async def completions(request: ChatCompletionRequest, raw_request: Request, response: Response) -> ChatCompletionResponse | ChatCompletionStreamResponse:
  response.headers.update(get_trace_id_headers())
  if request.stream:
    logger.debug(f"is_streaming: {request.stream}")
    return await track_a_handle_chat_completion(request, raw_request)

  chat_completion_response, headers = await track_a_handle_chat_completion(request, raw_request)
  
  return chat_completion_response


@track_b_router.post(
  "/v1/chat/completions",
  summary="Create chat completion",
  description="""
Creates a chat completion response. 

Returns different formats based on the `stream` parameter:
- `stream=false`: Returns ChatCompletionResponse as application/json
- `stream=true`: Returns ChatCompletionStreamResponse as text/event-stream (SSE)
""",
  responses={
    HTTPStatus.OK: {
      "description": "Successful Response",
      "content": {
        "application/json": {
          "schema": {"$ref": "#/components/schemas/ChatCompletionResponse"}
        },
        "text/event-stream": {
          "schema": {"$ref": "#/components/schemas/ChatCompletionStreamResponse"}
        },
      }
    },
    HTTPStatus.BAD_REQUEST: {"model": ErrorResponse},
    HTTPStatus.NOT_FOUND: {"model": ErrorResponse},
    HTTPStatus.INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
  },
)
async def completions(request: ChatCompletionRequest, raw_request: Request, response: Response) -> Any: # -> ChatCompletionResponse | ChatCompletionStreamResponse
  response.headers.update(get_trace_id_headers())
  if request.stream:
    logger.debug(f"is_streaming: {request.stream}")
    return await track_b_handle_chat_completion(request, raw_request)

  ret = await track_b_handle_chat_completion(request, raw_request)
  if isinstance(ret, tuple):
    chat_completion_response, headers = ret
    response.headers.update(headers)
    return chat_completion_response
  
  return ret
