# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import sys
import traceback
import json
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

import httpx
import pydantic_core
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.requests import Request

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry import trace
from opentelemetry.trace import set_tracer_provider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

from omni_chainer.api.main import api_router
from omni_chainer.core.config import settings
from omni_chainer.core.omni._tools import (
  get_http_client,
  close_http_client,
  get_trace_id_headers,
)


if "--reload" in sys.argv or "dev" in sys.argv:
  if settings.ENV_STAGE == "production":
    settings.ENV_STAGE = "development"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
  """Manage application lifecycle: initialize and cleanup resources."""
  # Startup: Initialize the global HTTP client
  await get_http_client()
  yield
  # Shutdown: Close the global HTTP client
  await close_http_client()


app = FastAPI(
  title="Omni Chainer",
  debug=settings.ENV_STAGE == "development",
  openapi_url=f"{settings.API_STR}/openapi.json" if settings.ENV_STAGE == "development" else None,
  lifespan=lifespan,
)

if settings.OTLP_ENDPOINT is not None:
  resource = Resource.create(attributes={SERVICE_NAME: "omni-chainer"})
  tracer_provider = TracerProvider(resource=resource)
  set_tracer_provider(tracer_provider)
  
  otlp_exporter = OTLPSpanExporter(
    endpoint=f"{settings.OTLP_ENDPOINT}/v1/traces",
  )
  
  span_processor = BatchSpanProcessor(otlp_exporter)
  tracer_provider.add_span_processor(span_processor)
  
  FastAPIInstrumentor.instrument_app(app,
    excluded_urls=(
      "/health,"
      "/docs,"
      "/openapi.json,"
      "/a/v1/{path:path},"
      "/b/v1/{path:path}"
    ))
  HTTPXClientInstrumentor().instrument()


if settings.all_cors_origins:
  app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.all_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
  )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
  headers = get_trace_id_headers()

  tracer = trace.get_tracer(__name__)
  with tracer.start_as_current_span("http_status_exception_handler") as span:
    span.record_exception(exc)

  traceback.print_exc()

  return JSONResponse(
    content={
      "error": {
        "message": str(exc.detail),
        "type": "HTTPException",
        "params": None,
        "code": exc.status_code,
      }
    },
    status_code=exc.status_code,
    headers=headers,
  )


@app.exception_handler(httpx.HTTPError)
async def http_error_exception_handler(request: Request, exc: httpx.HTTPError) -> JSONResponse:
  headers = get_trace_id_headers()
  
  tracer = trace.get_tracer(__name__)
  with tracer.start_as_current_span("http_error_exception_handler") as span:
    span.record_exception(exc)

  traceback.print_exc()

  return JSONResponse(
    content={
      "error": {
        "message": str(exc),
        "type": "HTTPError",
        "params": None,
        "code": 500,
      }
    },

    status_code=500,
    headers=headers,
  )


@app.exception_handler(httpx.HTTPStatusError)
async def http_status_exception_handler(request: Request, exc: httpx.HTTPStatusError) -> JSONResponse:
  headers = get_trace_id_headers()

  tracer = trace.get_tracer(__name__)
  with tracer.start_as_current_span("http_status_exception_handler") as span:
    span.record_exception(exc)

  traceback.print_exc()

  try:
    error_response = exc.response.json()
  except json.JSONDecodeError:
    error_response = {
      "message": str(exc),
    }

  error_instance = error_response.get("error", None)

  if error_instance is not None:
    message = error_instance.get("message", "UnknownError")
    params = error_instance.get("params", None)
    error_type = error_instance.get("type", "UnknownError")
    status_code = error_instance.get("code", 500)
  elif (detail := error_response.get("detail", None)) is not None:
    message = detail
    params = None
    error_type = "EncoderError"
    status_code = 500
  else:
    message = "UnknownError"
    params = None
    error_type = "UnknownError"
    status_code = 500

  return JSONResponse(
    content={
      "error": {
        "message": message,
        "type": error_type,
        "params": params,
        "code": status_code,
      }
    },
    status_code=status_code,
    headers=headers,
  )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
  headers = get_trace_id_headers()

  tracer = trace.get_tracer(__name__)
  with tracer.start_as_current_span("validation_exception_handler") as span:
    span.record_exception(exc)

  # invalid_url 타입의 에러를 찾기
  for error in exc.errors():
    if error["type"] == "invalid_url":
      return JSONResponse(
        content={
          "error": {
            "message": "Invalid URL format",
            "type": "validation_error",
            "code": 422,
            "details": {
              "loc": error["loc"],
              "msg": error["msg"],
              "input": error["input"],
              "ctx": error["ctx"],
            }
          }
        },
        status_code=422,
        headers=headers,
      )

  # 그 외의 validation 에러는 기본 형식으로 반환
  traceback.print_exc()
  return JSONResponse(
    content={"detail": exc.errors()},
    status_code=422,
    headers=headers,
  )


app.include_router(api_router, prefix=settings.API_STR)
