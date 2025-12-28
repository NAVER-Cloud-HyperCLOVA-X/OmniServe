# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import asyncio
from typing import Tuple, List

from fastapi import HTTPException
from fastapi.requests import Request
from fastapi.responses import StreamingResponse
from starlette.datastructures import (
  Headers,
  MutableHeaders
)

from omni_chainer.core.config import settings
from omni_chainer.models.openai.v1.chat.completions import (
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatCompletionContentPartParam,
)
from omni_chainer.core.omni.track_a import (
  handle_llm,
  handle_stream_llm,
  handle_vision_encoding,
)
from omni_chainer.core.omni._tools import (
  get_trace_id_headers,
)
from ._tools import (
  get_media_contents,
  make_stream_response,
  validate_media_contents,
  ContentType,
  is_s3_embedding_url,
  get_cached_embedding,
  cache_embedding,
)
from omni_chainer.logging import logger


async def handle_chat_completion(track_a_request: ChatCompletionRequest, raw_request: Request) -> Tuple[ChatCompletionResponse, Headers] | StreamingResponse:
  headers = raw_request.headers.mutablecopy()
  if "Content-Length" in headers:
    del headers["Content-Length"]

  # ============================================================================
  # 모든 메시지에서 미디어 콘텐츠 수집 (멀티턴 지원)
  # ============================================================================
  # 이전: 마지막 메시지만 처리 → 멀티턴에서 이전 미디어 URL 처리 불가
  # 수정: 모든 메시지의 이미지/오디오 URL을 수집하여 처리
  # Track A: Vision Encoder만 사용 (이미지만 처리)
  # Track B: Vision Encoder + Audio Encoder 모두 사용 (이미지 + 오디오 처리)
  # ============================================================================
  all_media_contents: List[Tuple[ChatCompletionContentPartParam, ContentType]] = []

  for msg in track_a_request.messages:
    # msg가 dict일 수도 있고 Pydantic 모델일 수도 있음
    content = msg.get('content') if isinstance(msg, dict) else getattr(msg, 'content', None)
    if content is not None and isinstance(content, list):
      media_contents = await get_media_contents(msg)
      all_media_contents.extend(media_contents)

  if not await validate_media_contents(all_media_contents):
    raise HTTPException(status_code=400, detail="some of media contents are not available: check expired")

  # ============================================================================
  # 캐시를 활용한 미디어 Encoding (Track A: Vision만)
  # ============================================================================
  # 1. 이미 S3 URL (.pt)이면 건너뛰기
  # 2. 캐시에 있으면 캐시된 S3 경로 사용 (Encoder 호출 생략)
  # 3. 그 외에만 Encoder 호출 후 캐시에 저장
  #
  # Track A: Vision Encoder만 호출 (이미지)
  # Track B: Vision Encoder + Audio Encoder 호출 (이미지 + 오디오)
  # ============================================================================
  media_to_encode: List[Tuple[ChatCompletionContentPartParam, ContentType, str]] = []  # (content, type, original_url)

  for media_content, content_type in all_media_contents:
    if media_content is not None and content_type == ContentType.IMAGE_URL:
      url = media_content.image_url.get("url", "")

      # 1. 이미 S3 embedding URL이면 건너뛰기
      if is_s3_embedding_url(url):
        logger.debug(f"Skipping already S3 embedding: {url[:80]}...")
        continue

      # 2. 캐시에서 조회
      cached_s3_path = get_cached_embedding(url, track="track_a")
      if cached_s3_path:
        media_content.image_url["url"] = cached_s3_path
        logger.debug(f"Using cached embedding for: {url[:80]}...")
        continue

      # 3. 인코딩 필요
      media_to_encode.append((media_content, content_type, url))

  # Vision Encoder 호출 (인코딩이 필요한 미디어만)
  if media_to_encode:
    vision_encoding_future_list = [
      handle_vision_encoding(media_content, headers)
      for media_content, content_type, _ in media_to_encode
    ]

    vision_encoding_results = await asyncio.gather(*vision_encoding_future_list, return_exceptions=False)

    for (vision_encoding_result, _), (media_content, content_type, original_url) in zip(vision_encoding_results, media_to_encode):
      s3_path = f"s3://{settings.WBL_S3_BUCKET_NAME}/{vision_encoding_result.s3_key}"
      media_content.image_url["url"] = s3_path

      # 캐시에 저장
      cache_embedding(original_url, s3_path, track="track_a")

  if track_a_request.stream:
    return await make_stream_response(handle_stream_llm, track_a_request, headers)

  llm_response, resp_header = await handle_llm(track_a_request, headers=headers)

  headers = MutableHeaders()
  headers.update({
    **resp_header,
    **get_trace_id_headers()
  })
  if "Content-Length" in headers:
    del headers["Content-Length"]

  return llm_response, headers
