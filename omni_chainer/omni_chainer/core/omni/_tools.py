# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import inspect
import json
from platform import java_ver
import httpx
from functools import wraps
from typing import Dict, Callable, Any

from pydantic import BaseModel
from opentelemetry import trace

from omni_chainer.core.config import settings
from omni_chainer.models.openai.v1.chat.completions import (
  ChatCompletionRequest,
)
from omni_chainer.logging import logger


_http_client: httpx.AsyncClient | None = None


# NOTE: do not remove 'async' keyword from this function.
async def get_http_client() -> httpx.AsyncClient:
  """Get or create the global HTTP client instance."""
  
  agent_header = "Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Mobile Safari/537.36"

  global _http_client
  if _http_client is None:
    _http_client = httpx.AsyncClient(timeout=settings.BACKEND_TIMEOUT, headers={"User-Agent": agent_header})
  return _http_client


async def close_http_client() -> None:
  """Close the global HTTP client and release resources."""
  global _http_client
  if _http_client is not None:
    await _http_client.aclose()
    _http_client = None


def get_trace_id_headers() -> Dict[str, str] | None:
  current_span = trace.get_current_span()
  if current_span.is_recording():
    return {
      "X-Omni-Trace-ID": str(format(current_span.get_span_context().trace_id, "x").zfill(32))
    }

  return {}


def trace_request(func: Callable) -> Callable:
  @wraps(func)
  async def wrapper(*args, **kwargs):
    tracer = trace.get_tracer(func.__module__)
    with tracer.start_as_current_span(func.__name__) as span:
      span.add_event(f"{func.__name__}_request", {
        "request": args[0].model_dump_json()
      })

      resp = await func(*args, **kwargs)

      if isinstance(resp, BaseModel):
        span.add_event(f"{func.__name__}_response", {
          "response": resp.model_dump_json()
        })
      elif isinstance(resp, tuple) and len(resp) == 2 and isinstance(resp[0], BaseModel):
        span.add_event(f"{func.__name__}_response", {
          "response": resp[0].model_dump_json()
        })

    return resp

  return wrapper


def flatten_extra_body(request: ChatCompletionRequest) -> dict:
  request_json = request.model_dump()
  extra_body = request_json.pop("extra_body", None)
  if extra_body is not None:
    logger.debug(f"flattening extra body: {extra_body}")
    request_json.update(**extra_body)

  # Remove None values to let vLLM use its defaults
  # This prevents unnecessary parameters from affecting model behavior
  request_json = {k: v for k, v in request_json.items() if v is not None and k not in request._need_to_filter_out_field}
  
  return request_json


def record_event(event_name: str, payload: str | Dict[str, Any]):
  current_frame = inspect.currentframe()
  caller_frame = current_frame.f_back
  caller_name = caller_frame.f_code.co_name

  tracer = trace.get_tracer(caller_name)
  with tracer.start_as_current_span(caller_name) as span:
    if span.is_recording():
      try:
        span.add_event(event_name, {
          "payload": payload if isinstance(payload, str) else json.dumps(payload)
        })
      except Exception as e:
        logger.warning(f"Failed to record event {event_name}: {e}")
