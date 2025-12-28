# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

from enum import Enum
from typing import List, Literal, Optional, Tuple

import httpx
from fastapi import Header
from pydantic import BaseModel, Field

from . import _tools as tools
from omni_chainer.core.config import settings
from omni_chainer.logging import logger
from omni_chainer.models.openai.v1.chat.completions import (
  ChatCompletionRequest,
  ChatCompletionResponse,
  CustomChatCompletionContentPartImageParam,
  CustomChatCompletionContentInputAudioParam,
)


class AudioEncodingRequest(BaseModel):
  media_url: str


class AudioEncodingResponse(BaseModel):
  latency: float
  s3_key: str


class VisionEncodingRequest(BaseModel):
  media_url: Optional[str] = Field(default=None, description="Image URL (mutually exclusive with image_base64)")
  media_base64: Optional[str] = Field(
    default=None, description="Base64 encoded image (mutually exclusive with image_url)"
  )
  anyres: Optional[bool] = Field(default=False, description="Use anyres processing")
  unpad: Optional[bool] = Field(default=False, description="Use unpad processing")
  num_queries_vis_abstractor: Optional[int] = Field(default=0, description="Number of visual abstractor queries")


class VisionEncodingResponse(BaseModel):
  latency: float
  s3_key: str


class VisionDecodingRequest(BaseModel):
  """이미지 생성 요청 - Vision Decoder API 스펙에 맞춤"""
  # vlm_output: discrete_image_token 문자열 전체
  # 예: "<|discrete_image_start|><|vision_ratio_1:1|><|vision27465|>...<|discrete_image_end|>"
  vlm_output: str = Field(
      description="VLM output string containing discrete vision tokens"
  )

  # 생성 파라미터 (Vision Decoder API와 동일하게)
  height: Optional[int] = Field(default=None, ge=64, le=2048, description="Image height (auto-detected if not provided)")
  width: Optional[int] = Field(default=None, ge=64, le=2048, description="Image width (auto-detected if not provided)")
  num_inference_steps: int = Field(default=30, ge=1, le=200, description="Number of inference steps")
  seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

  # S3 업로드 옵션
  upload_to_s3: bool = Field(default=True, description="Upload to S3")
  s3_prefix: str = Field(default="vision-decoder", description="S3 prefix")
  s3_expiration: int = Field(default=3600, ge=60, le=604800, description="Presigned URL expiration")


class VisionDecodingResponse(BaseModel):
  """이미지 생성 응답 - Vision Decoder API 응답 형식"""
  message: str = Field(description="Status message")
  num_images: int = Field(default=1, description="Number of images generated")
  height: int = Field(description="Image height")
  width: int = Field(description="Image width")
  num_tokens_parsed: int = Field(description="Number of vision tokens parsed")
  seed: Optional[int] = Field(default=None, description="Seed used")
  presigned_url: Optional[str] = Field(default=None, description="S3 presigned URL")
  s3_path: Optional[str] = Field(default=None, description="S3 full path")


# NOTE: ref_audio_url을 사용하면 zero-shot 모드로 동작하여 참조 오디오의 음색을 사용함
class Speaker(BaseModel):
  class VoiceType(Enum):
    MALE = "mhwj"
    FEMALE = "fkms"

  id: Optional[Literal["fkms", "mhwj"]] = None  # finetuned 모드용 speaker id
  gender: Optional[Literal["m", "f"]] = None
  age: Optional[Literal["0-10", "10", "20-30", "40-50", "60-70"]] = None
  # ref_audio_base64: Optional[str] = None
  ref_audio_url: Optional[str] = None  # zero-shot 모드용 참조 오디오 URL (S3, HTTP, file://)


class AudioDecodingRequest(BaseModel):
  units: List[int]
  format: Literal["wav", "mp3", "flac", "ogg", "aac", "pcm"]
  speaker: Optional[Speaker] = None


_AUDIO_FORMAT_TO_CONTENT_TYPE = {
  "wav": "audio/wav",
  "mp3": "audio/mpeg",
  "flac": "audio/flac",
  "ogg": "audio/ogg",
  "aac": "audio/aac",
  "pcm": "audio/pcm",
}

class AudioDecodingResponse(BaseModel):
  s3_key: str
  requested_format: Literal["wav", "mp3", "flac", "ogg", "aac", "pcm"]
  content_type: Literal["audio/wav", "audio/mpeg", "audio/flac", "audio/ogg", "audio/aac", "audio/pcm"]
  size_bytes: int


async def handle_audio_encoding(media_content: CustomChatCompletionContentInputAudioParam) -> Tuple[AudioEncodingResponse, Header]:
  client = await tools.get_http_client()

  logger.debug(f"AUDIO_ENCODING_ENDPOINT: {settings.TRACK_B_AUDIO_ENCODING_ENDPOINT}")
  logger.debug(f"sending payload: {media_content.input_audio.data}")

  request = AudioEncodingRequest(media_url=media_content.input_audio.data)

  try:
    response = await client.post(settings.TRACK_B_AUDIO_ENCODING_ENDPOINT, json=request.model_dump())
    tools.record_event("track_b_audio_encoding_request", request.model_dump_json())
    tools.record_event("track_b_audio_encoding_response", response.json())
    response.raise_for_status()

  except httpx.ConnectError as e:
    raise httpx.HTTPError(message=f"Failed to connect track b llm endpoint. {e}")
  except httpx.ConnectTimeout as e:
    raise httpx.HTTPError(message=f"Timeout while connecting to track b llm endpoint. {e}")

  return AudioEncodingResponse(**response.json()), response.headers


async def handle_stream_llm(request: ChatCompletionRequest, headers: Header) -> httpx.Response:
  client = await tools.get_http_client()

  request_json = tools.flatten_extra_body(request)

  request_headers = dict(headers) if headers else {}
  if request.model:
    request_headers["X-Gateway-Model-Name"] = request.model

  logger.debug(f"LLM_ENDPOINT: {settings.TRACK_B_LLM_ENDPOINT}")
  logger.debug(f"sending payload: {request_json}, headers: {request_headers}")

  try:
    req = client.build_request("POST", settings.TRACK_B_LLM_ENDPOINT, json=request_json, headers=request_headers)
    response = await client.send(req, stream=True)
    tools.record_event("track_b_llm_request", request_json)
    tools.record_event("track_b_llm_response", "stream response")
    # 스트리밍 응답에서 에러 시 content를 먼저 읽어야 exception handler에서 접근 가능
    if response.is_error:
      await response.aread()
    response.raise_for_status()
  except httpx.ConnectError as e:
    raise httpx.HTTPError(message=f"Failed to connect track b llm endpoint. {e}")
  except httpx.ConnectTimeout as e:
    raise httpx.HTTPError(message=f"Timeout while connecting to track b llm endpoint. {e}")

  return response


async def handle_llm(request: ChatCompletionRequest, headers: Header) -> Tuple[ChatCompletionResponse, Header]:
  client = await tools.get_http_client()

  request_json = tools.flatten_extra_body(request)

  request_headers = dict(headers) if headers else {}
  if request.model:
    request_headers["X-Gateway-Model-Name"] = request.model

  logger.debug(f"LLM_ENDPOINT: {settings.TRACK_B_LLM_ENDPOINT}")
  logger.debug(f"sending payload: {request_json}, headers: {request_headers}")

  try:
    response = await client.post(settings.TRACK_B_LLM_ENDPOINT, json=request_json, headers=request_headers)
    tools.record_event("track_b_llm_request", request_json)
    tools.record_event("track_b_llm_response", response.json())
    response.raise_for_status()
  except httpx.ConnectError as e:
    raise httpx.HTTPError(message=f"Failed to connect track b llm endpoint. {e}")
  except httpx.ConnectTimeout as e:
    raise httpx.HTTPError(message=f"Timeout while connecting to track b llm endpoint. {e}")

  return ChatCompletionResponse(**response.json()), response.headers


async def handle_vision_encoding(media_content: CustomChatCompletionContentPartImageParam) -> Tuple[VisionEncodingResponse, Header]:
  client = await tools.get_http_client()

  request = VisionEncodingRequest(media_url=media_content.image_url["url"])
  logger.debug(f"IMAGE_ENCODING_ENDPOINT: {settings.TRACK_B_VISION_ENCODING_ENDPOINT}")
  logger.debug(f"sending payload: {request.model_dump()}")

  try:
    response = await client.post(settings.TRACK_B_VISION_ENCODING_ENDPOINT, json=request.model_dump())
    tools.record_event("track_b_vision_encoding_request", request.model_dump_json())
    tools.record_event("track_b_vision_encoding_response", response.json())
    response.raise_for_status()
  except httpx.ConnectError as e:
    raise httpx.HTTPError(message=f"Failed to connect track b llm endpoint. {e}")
  except httpx.ConnectTimeout as e:
    raise httpx.HTTPError(message=f"Timeout while connecting to track b llm endpoint. {e}")

  return VisionEncodingResponse(**response.json()), response.headers


async def handle_vision_decoding(request: VisionDecodingRequest, headers: Header) -> Tuple[VisionDecodingResponse, Header]:
  """
  Vision Decoder API 호출하여 이미지 생성

  Vision Decoder API는 /decode 엔드포인트를 통해 JSON 응답 반환:
  {
    "message": "Image uploaded to S3: s3://<BUCKET>/vision-decoder/xxx.png",
    "num_images": 1,
    "height": 768,
    "width": 768,
    "num_tokens_parsed": 729,
    "seed": 42,
    "presigned_url": "https://...",
    "s3_path": "s3://<BUCKET>/vision-decoder/xxx.png"
  }
  """
  client = await tools.get_http_client()

  vlm_output_preview = request.vlm_output[:100] if len(request.vlm_output) <= 100 else request.vlm_output[:100] + "..."
  logger.debug(f"IMAGE_DECODING_ENDPOINT: {settings.TRACK_B_VISION_DECODING_ENDPOINT}")
  logger.debug(f"Sending vlm_output length: {len(request.vlm_output)}, preview: {vlm_output_preview}")
  logger.debug(f"Parameters: height={request.height}, width={request.width}, steps={request.num_inference_steps}, seed={request.seed}")

  try:
    response = await client.post(
      settings.TRACK_B_VISION_DECODING_ENDPOINT,
      json=request.model_dump(exclude_none=True),  # None 값 제외
      headers={"Content-Type": "application/json"},
      timeout=120.0  # 이미지 생성은 시간이 걸릴 수 있음
    )
    tools.record_event("track_b_vision_decoding_request", request.model_dump_json())
    tools.record_event("track_b_vision_decoding_response", response.json())
    response.raise_for_status()
  except httpx.ConnectError as e:
    raise httpx.HTTPError(message=f"Failed to connect track b llm endpoint. {e}")
  except httpx.ConnectTimeout as e:
    raise httpx.HTTPError(message=f"Timeout while connecting to track b llm endpoint. {e}")

  # Vision Decoder는 항상 JSON 응답 반환
  try:
    resp_data = response.json()
  except Exception as e:
    logger.error(f"Failed to parse Vision Decoder response as JSON: {e}")
    raise ValueError(f"Invalid Vision Decoder response: {e}")

  logger.debug(f"Vision Decoder response: message={resp_data.get('message', '')}, num_tokens={resp_data.get('num_tokens_parsed', 0)}")

  # VisionDecodingResponse 생성
  return VisionDecodingResponse(**resp_data), response.headers


def _validate_audio_units(units: List[int]) -> Tuple[bool, str]:
  """
  Audio units(토큰)의 유효성을 검사합니다.

  Audio Decoder는 특정 vocabulary size를 가지며, 이 범위를 벗어나는 토큰은
  처리할 수 없습니다.

  Returns:
    (is_valid, error_message)
  """
  if not units:
    return False, "Empty units list"

  # 최소/최대 토큰 수 검사
  MIN_UNITS = 3
  MAX_UNITS = 2000
  if len(units) < MIN_UNITS:
    return False, f"Too few units: {len(units)} < {MIN_UNITS}"
  if len(units) > MAX_UNITS:
    return False, f"Too many units: {len(units)} > {MAX_UNITS}"

  # 토큰 값 범위 검사 (Audio Decoder vocabulary: 0-6561)
  MIN_TOKEN_VALUE = 0
  MAX_TOKEN_VALUE = 6561
  invalid_tokens = [u for u in units if u < MIN_TOKEN_VALUE or u > MAX_TOKEN_VALUE]
  if invalid_tokens:
    return False, f"Invalid token values (out of range {MIN_TOKEN_VALUE}-{MAX_TOKEN_VALUE}): {invalid_tokens[:5]}{'...' if len(invalid_tokens) > 5 else ''}"

  return True, ""


async def handle_audio_decoding(
  units: List[int],
  format: Literal["wav", "mp3", "flac", "ogg", "aac", "pcm"],
  speaker: Optional[Speaker],
  headers: Header,
  ref_audio_url: Optional[str] = None,  # zero-shot 모드용 참조 오디오 URL
  fallback_to_finetuned: bool = True,  # zero-shot 실패 시 finetuned 모드로 재시도
) -> Tuple[AudioDecodingResponse, Header] | httpx.Response:
  """
  Audio Decoder API 호출하여 오디오 생성

  Args:
    units: 오디오 토큰 리스트
    format: 출력 포맷 (wav, mp3, flac, ogg, aac, pcm)
    speaker: Speaker 객체 (id 또는 ref_audio_url 포함)
    headers: HTTP 헤더
    ref_audio_url: 참조 오디오 URL (zero-shot 모드). speaker.ref_audio_url보다 우선함
    fallback_to_finetuned: zero-shot 실패 시 finetuned 모드로 재시도 여부

  Note:
    - ref_audio_url이 제공되면 zero-shot 모드로 동작 (참조 오디오의 음색 사용)
    - ref_audio_url이 없고 speaker.id가 있으면 finetuned 모드로 동작
    - zero-shot 실패 시 fallback_to_finetuned=True면 finetuned 모드로 재시도
  """
  # units 유효성 검사
  is_valid, error_msg = _validate_audio_units(units)
  if not is_valid:
    logger.error(f"Audio units validation failed: {error_msg}")
    raise ValueError(f"Invalid audio units: {error_msg}")

  client = await tools.get_http_client()

  # 원본 speaker 정보 저장 (fallback 시 사용)
  original_speaker = speaker
  use_zero_shot = bool(ref_audio_url)

  # ref_audio_url이 제공되면 zero-shot 모드로 동작 (id 사용 안 함)
  if ref_audio_url:
    # zero-shot 모드: ref_audio_url만 사용, id는 제외
    speaker = Speaker(ref_audio_url=ref_audio_url)
    logger.debug(f"Using zero-shot mode with ref_audio_url: {ref_audio_url[:80]}...")

  request = AudioDecodingRequest(units=units, format=format, speaker=speaker)
  logger.debug(f"AUDIO_DECODING_ENDPOINT: {settings.TRACK_B_AUDIO_DECODING_ENDPOINT}")
  logger.debug(f"sending payload: units_count={len(units)}, format={format}, speaker={request.speaker}")

  try:
    response = await client.post(
      settings.TRACK_B_AUDIO_DECODING_ENDPOINT,
      json=request.model_dump(exclude_none=True),
      headers=headers,
      timeout=120.0  # 오디오 생성은 시간이 걸릴 수 있음
    )
    tools.record_event("track_b_audio_decoding_request", request.model_dump_json())
    tools.record_event("track_b_audio_decoding_response", response.json())
    response.raise_for_status()
    ret = AudioDecodingResponse(**response.json())

    if ret.content_type != _AUDIO_FORMAT_TO_CONTENT_TYPE[format]:
      raise ValueError(f"Audio decoding response content-type mismatch: request format: {format}, expected: {_AUDIO_FORMAT_TO_CONTENT_TYPE[format]}, response: {ret.content_type}")

    return ret, response.headers

  except (httpx.ConnectError, httpx.ConnectTimeout) as e:
    raise httpx.HTTPError(message=f"Failed to connect track b audio decoding endpoint. {e}")

  except httpx.HTTPStatusError as e:
    # Zero-shot 모드에서 실패 시 Finetuned 모드로 fallback 시도
    if use_zero_shot and fallback_to_finetuned:
      logger.warning(f"Zero-shot audio decoding failed (status={e.response.status_code}), falling back to finetuned mode")

      # Finetuned 모드로 재시도 (기본 speaker id 사용)
      fallback_speaker = original_speaker if original_speaker and original_speaker.id else Speaker(id=Speaker.VoiceType.FEMALE.value)
      fallback_request = AudioDecodingRequest(units=units, format=format, speaker=fallback_speaker)
      logger.debug(f"Fallback to finetuned mode with speaker id: {fallback_speaker.id}")

      try:
        fallback_response = await client.post(
          settings.TRACK_B_AUDIO_DECODING_ENDPOINT,
          json=fallback_request.model_dump(exclude_none=True),
          headers=headers,
          timeout=120.0
        )
        fallback_response.raise_for_status()
        logger.info("Fallback to finetuned mode succeeded")
        return AudioDecodingResponse(**fallback_response.json()), fallback_response.headers
      except Exception as fallback_error:
        logger.error(f"Fallback to finetuned mode also failed: {fallback_error}")
        # 원래 에러를 다시 발생
        raise e

    # Fallback 없이 원래 에러 발생
    raise
