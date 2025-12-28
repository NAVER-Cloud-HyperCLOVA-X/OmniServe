# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import asyncio
import base64
import hashlib
import urllib.parse
from collections import OrderedDict
from enum import Enum
from http import HTTPStatus
from threading import Lock
from typing import Tuple, List, Callable, Optional

from httpx import HTTPStatusError, Response
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from starlette.datastructures import Headers
from wbl_storage_utility.s3_util.s3_connection import S3Connection

from omni_chainer.core.config import settings
from omni_chainer.core.omni._tools import (
  get_http_client,
  get_trace_id_headers,
)
from omni_chainer.models.openai.v1.chat.completions import (
  ChatCompletionContentPartParam,
  ChatCompletionMessageParam,
  ChatCompletionRequest
)
from omni_chainer.logging import logger

class ContentType(Enum):
  IMAGE_URL = "image_url"
  INPUT_AUDIO = "input_audio"

_content_types = [content_type.value for content_type in ContentType]

s3_connection = S3Connection()


# ============================================================================
# URL → S3 Embedding 캐시
# ============================================================================
# 외부 URL을 Vision/Audio Encoder로 처리한 결과(S3 embedding 경로)를 캐싱합니다.
# 같은 URL이 멀티턴 대화에서 반복 요청되면 Encoder 호출을 건너뛰고 캐시된 S3 경로를 사용합니다.
#
# 지원 미디어 타입:
# - IMAGE_URL: Vision Encoder → S3 embedding (.pt)
# - INPUT_AUDIO: Audio Encoder → S3 embedding (.pt)
#
# 캐시 동작:
# - Track A: 이미지 URL만 캐싱
# - Track B: 이미지 + 오디오 URL 모두 캐싱
# ============================================================================

class EmbeddingCache:
  """
  Thread-safe LRU 캐시: URL → S3 embedding 경로 매핑

  - 최대 10,000개 항목 저장 (문자열이므로 메모리 부담 적음)
  - LRU 방식으로 오래된 항목 자동 삭제
  - URL을 SHA256 해시로 저장하여 키 크기 최적화
  """

  def __init__(self, max_size: int = 10000):
    self._cache: OrderedDict[str, str] = OrderedDict()
    self._max_size = max_size
    self._lock = Lock()
    self._hits = 0
    self._misses = 0

  def _hash_url(self, url: str, track: str = "") -> str:
    """URL과 트랙 정보를 SHA256 해시로 변환"""
    key_source = f"{track}:{url}" if track else url
    return hashlib.sha256(key_source.encode('utf-8')).hexdigest()

  def get(self, url: str, track: str = "") -> Optional[str]:
    """
    캐시에서 S3 경로 조회

    Args:
      url: 원본 미디어 URL (http/https)
      track: 트랙 정보 ("track_a" 또는 "track_b")

    Returns:
      S3 embedding 경로 또는 None (캐시 미스)
    """
    key = self._hash_url(url, track)
    with self._lock:
      if key in self._cache:
        # LRU: 최근 사용된 항목을 맨 뒤로 이동
        self._cache.move_to_end(key)
        self._hits += 1
        return self._cache[key]
      self._misses += 1
      return None

  def set(self, url: str, s3_path: str, track: str = "") -> None:
    """
    캐시에 URL → S3 경로 저장

    Args:
      url: 원본 미디어 URL (http/https)
      s3_path: 변환된 S3 embedding 경로 (s3://bucket/path.pt)
      track: 트랙 정보 ("track_a" 또는 "track_b")
    """
    key = self._hash_url(url, track)
    with self._lock:
      if key in self._cache:
        # 이미 존재하면 값 업데이트 및 LRU 갱신
        self._cache.move_to_end(key)
        self._cache[key] = s3_path
      else:
        # 새 항목 추가
        if len(self._cache) >= self._max_size:
          # LRU: 가장 오래된 항목 삭제
          self._cache.popitem(last=False)
        self._cache[key] = s3_path

  def stats(self) -> dict:
    """캐시 통계 반환"""
    with self._lock:
      total = self._hits + self._misses
      hit_rate = (self._hits / total * 100) if total > 0 else 0
      return {
        "size": len(self._cache),
        "max_size": self._max_size,
        "hits": self._hits,
        "misses": self._misses,
        "hit_rate": f"{hit_rate:.1f}%"
      }


# 글로벌 캐시 인스턴스
embedding_cache = EmbeddingCache(max_size=10000)


def is_s3_embedding_url(url: str) -> bool:
  """S3에 저장된 embedding 파일(.pt)인지 확인"""
  return url.startswith("s3://") and url.endswith(".pt")


def get_cached_embedding(url: str, track: str = "") -> Optional[str]:
  """
  URL에 대한 캐시된 S3 embedding 경로 조회

  Args:
    url: 원본 미디어 URL
    track: 트랙 정보 ("track_a" 또는 "track_b")

  Returns:
    캐시된 S3 경로 또는 None
  """
  # 캐시 비활성화 시 None 반환
  if not settings.ENABLE_EMBEDDING_CACHE:
    return None

  # 이미 S3 URL이면 캐시 조회 불필요
  if is_s3_embedding_url(url):
    return url

  cached = embedding_cache.get(url, track)
  if cached:
    logger.debug(f"Cache HIT for URL ({track}): {url[:80]}... -> {cached[:80]}...")
  return cached


def cache_embedding(url: str, s3_path: str, track: str = "") -> None:
  """
  URL → S3 embedding 경로를 캐시에 저장

  Args:
    url: 원본 미디어 URL
    s3_path: 변환된 S3 embedding 경로
    track: 트랙 정보 ("track_a" 또는 "track_b")
  """
  # 캐시 비활성화 시 저장하지 않음
  if not settings.ENABLE_EMBEDDING_CACHE:
    return

  # S3 URL은 캐시하지 않음 (이미 처리됨)
  if is_s3_embedding_url(url):
    return

  embedding_cache.set(url, s3_path, track)
  logger.debug(f"Cached embedding ({track}): {url[:80]}... -> {s3_path[:80]}...")


def get_cache_stats() -> dict:
  """캐시 통계 조회"""
  return embedding_cache.stats()

async def get_media_contents(
  message: ChatCompletionMessageParam,
) -> List[Tuple[ChatCompletionContentPartParam | None, ContentType | None]]:
  """
  Try to update the last message with image encoding.
  Returns the updated headers if successful, otherwise returns None.
  """

  media_contents = []

  for content in message["content"]:
    if not isinstance(content, str) and content.type in _content_types:
      media_contents.append((content, ContentType(content.type)))

  return media_contents


async def validate_media_contents(media_contents: List[Tuple[ChatCompletionContentPartParam | None, ContentType | None]]) -> bool:
  """
  Validate the media contents.

  HEAD 요청에서:
  - 404: URL이 expire 되었거나 존재하지 않음 → False
  - 503/403 등: 권한 문제일 수 있지만 URL 자체는 유효할 수 있음 → True
  - 기타 에러: 네트워크 문제 등이므로 유효하다고 간주 → True
  """

  async def _validate_url(url: str) -> bool:
    parsed_url = urllib.parse.urlparse(url)
    client = await get_http_client()

    if parsed_url.scheme in ["http", "https"]:
      try:
        resp = await client.head(url)
        resp.raise_for_status()
        return True

      except HTTPStatusError as e:
        # 404: URL이 존재하지 않거나 expire됨 → 유일하게 False
        if resp.status_code == HTTPStatus.NOT_FOUND:
          logger.debug(f"URL validation failed (404 Not Found): {url[:80]}...")
          return False

        # 503 Service Unavailable, 403 Forbidden 등:
        # HEAD 요청에 대해 권한없음을 반환하는 서버가 많음
        # URL 자체는 유효할 수 있으므로 True로 처리
        logger.debug(f"URL validation passed despite {resp.status_code}: {url[:80]}...")
        return True

      except Exception as e:
        # 네트워크 에러 등 기타 예외: URL이 유효하다고 가정
        logger.debug(f"URL validation passed despite error ({type(e).__name__}): {url[:80]}...")
        return True

    elif parsed_url.scheme == "s3":
      ret = s3_connection.object_exists(storage_name=parsed_url.netloc, object_key=parsed_url.path.lstrip("/"))
      return ret

    # 알 수 없는 스킴은 일단 유효하다고 가정
    return True

  url_list = [None] * len(media_contents)
  for i, (media_content, content_type) in enumerate(media_contents):
    if content_type == ContentType.IMAGE_URL:
      url_list[i] = media_content.image_url["url"]
    elif content_type == ContentType.INPUT_AUDIO:
      url_string = media_content.input_audio.data
      if url_string.startswith("s3://") or url_string.startswith("https://") or url_string.startswith("http://"):
        url_list[i] = url_string
      else:
        url_list[i] = base64.urlsafe_b64decode(url_string).decode("utf-8")

  url_validation_results = await asyncio.gather(*[asyncio.create_task(_validate_url(url)) for url in url_list])

  return all(url_validation_results)


async def make_stream_response(
  handler: Callable[[ChatCompletionRequest, Headers], Response],
  request: ChatCompletionRequest,
  headers: Headers,
) -> StreamingResponse:
  stream_response = await handler(request, headers=headers)
  resp_headers = {
    **stream_response.headers,
    **get_trace_id_headers()
  }

  async def stream_response_generator():
    try:
      async for chunk in stream_response.aiter_bytes():
        yield chunk
    except Exception as e:
      raise HTTPException(status_code=500, detail={
        "error": {
          "where": "stream_response_generator",
          "code": 500,
          "message": str(e)
        }
      })
    finally:
      try:
        await stream_response.aclose()
      except Exception as e:
        raise HTTPException(status_code=500, detail={
          "error": {
            "where": "stream_response_generator",
            "code": 500,
            "message": str(e),
          }
        })
  
  return StreamingResponse(content=stream_response_generator(), headers=resp_headers)
