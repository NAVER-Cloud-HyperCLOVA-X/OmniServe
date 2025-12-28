# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import re
import json
import base64
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Tuple, Any, List, Optional

import httpx
from fastapi import HTTPException
from fastapi.requests import Request
from fastapi.responses import StreamingResponse
from starlette.datastructures import (
  Headers,
  MutableHeaders
)

from openai.types.chat import ChatCompletionAudio as OpenAIChatCompletionAudio
from omni_chainer.models.openai.v1.chat.completions import (
  ChatCompletionRequest,
  ChatCompletionResponse,
  ChatMessage,
  ToolCall,
)
from omni_chainer.core.config import settings
from omni_chainer.core.omni.track_b import (
  Speaker,
  handle_vision_encoding,
  handle_audio_encoding,
  handle_audio_decoding,
  handle_vision_decoding,
  handle_llm,
  handle_stream_llm,
  VisionDecodingRequest,
)
from ._tools import (
  get_media_contents,
  validate_media_contents,
  make_stream_response,
  ContentType,
  s3_connection,
  is_s3_embedding_url,
  get_cached_embedding,
  cache_embedding,
)
from omni_chainer.logging import logger


_AUDIO_TOKEN_PATTERN = re.compile(r"<\|audio(\d+)\|>")
_AUDIO_TOKEN_PREFIX = "<|audio"
_AUDIO_LONGEST_TOKEN = "<|audio6561|>"
_AUDIO_LONGEST_TOKEN_LENGTH = len(_AUDIO_LONGEST_TOKEN)
_SPEAKER_INFO_PATTERN = re.compile(r'\n\n(\{[^{}]+\})(?:\n\n)?$')  # \n\n{json}\n\n 이거나 \n\n{json} 이거나 (capture group으로 JSON만 추출)
_SPEAKER_INFO_TAG_PATTERN = re.compile(r'<speaker_info>(\{[^{}]+\})</speaker_info>')  # <speaker_info>{json}</speaker_info> 형식

# Transcript 패턴
# 내용 포함 전체 제거용 (둘 다 있을 때 user_transcript 제거)
_USER_TRANSCRIPT_FULL_PATTERN = re.compile(r'<user_transcript>.*?</user_transcript>\s*', re.DOTALL)
# 태그만 제거하고 내용 추출용
_USER_TRANSCRIPT_PATTERN = re.compile(r'<user_transcript>(.*?)</user_transcript>', re.DOTALL)
_ASSISTANT_TRANSCRIPT_PATTERN = re.compile(r'<assistant_transcript>(.*?)</assistant_transcript>', re.DOTALL)
_AUDIO_DECODER_CALL_PATTERN = re.compile(r'\s*<audio_decoder_call>\s*', re.DOTALL)

# Vision token 패턴 (discrete_image_token 검증용)
_VISION_TOKEN_PATTERN = re.compile(r"<\|vision\d+\|>")
_VISION_RATIO_PATTERN = re.compile(r"<\|vision_ratio_\d+:\d+\|>")

# .pt 임베딩 경로에서 원본 URL 추출을 위한 패턴
# 예: s3://wbl/.../video_https%3A%2F%2Fexample.com%2Faudio.wav.pt
#     s3://wbl/.../audio_https%3A%2F%2Fexample.com%2Faudio.wav.pt
_EMBEDDING_ORIGINAL_URL_PATTERN = re.compile(r'(?:video|audio)_(https?%3A%2F%2F.+)\.pt$')


def extract_original_url_from_embedding_path(embedding_path: str) -> Optional[str]:
  """
  .pt 임베딩 경로에서 원본 오디오/비디오 URL을 추출합니다.

  임베딩 파일명에는 원본 URL이 URL 인코딩되어 포함되어 있습니다.
  예: s3://wbl/.../video_https%3A%2F%2Fexample.com%2Faudio.wav.pt
      → https://example.com/audio.wav

  Args:
    embedding_path: S3 임베딩 경로 (예: s3://wbl/.../video_https%3A...pt)

  Returns:
    원본 URL 또는 None (추출 실패 시)
  """
  from urllib.parse import unquote

  # 파일명만 추출 (마지막 / 이후)
  filename = embedding_path.rsplit('/', 1)[-1] if '/' in embedding_path else embedding_path

  # 패턴 매칭
  match = _EMBEDDING_ORIGINAL_URL_PATTERN.search(filename)
  if match:
    encoded_url = match.group(1)
    try:
      # URL 디코딩 (이중 인코딩 처리)
      decoded_url = unquote(unquote(encoded_url))
      logger.debug(f"Extracted original URL from embedding path: {decoded_url[:80]}...")
      return decoded_url
    except Exception as e:
      logger.warning(f"Failed to decode URL from embedding path: {e}")
      return None

  return None


def has_audio_token(content: str) -> bool:
  return content.rfind(_AUDIO_TOKEN_PREFIX) != -1


def _strip_transcript_tags(content: str) -> str:
  """
  Audio-to-Audio 응답에서 transcript 관련 태그들을 정리합니다.
  
  처리:
  1. user_transcript와 assistant_transcript가 둘 다 있을 때만 user_transcript 내용 제거
  2. 항상: 모든 태그 자체는 제거 (내용은 유지)
     - <user_transcript>, </user_transcript>
     - <assistant_transcript>, </assistant_transcript>
     - <audio_decoder_call>
  
  예: "<user_transcript>A</user_transcript>\n<assistant_transcript>B</assistant_transcript>\n<audio_decoder_call>"
  → "B"
  
  예: "<assistant_transcript>B</assistant_transcript>"
  → "B"
  
  예: "<user_transcript>A</user_transcript>"
  → "A"
  """
  if not content:
    return content
  
  result = content
  
  has_user = _USER_TRANSCRIPT_PATTERN.search(content)
  has_assistant = _ASSISTANT_TRANSCRIPT_PATTERN.search(content)
  has_audio_decoder_call = _AUDIO_DECODER_CALL_PATTERN.search(content)
  
  # 처리할 태그가 있을 때만 로깅 (필터링 전 원본 content)
  if has_user or has_assistant or has_audio_decoder_call:
    logger.info(f"Stripping transcript tags - before: {content}")
  
  # 1. 둘 다 있을 때만 user_transcript 내용까지 완전히 제거
  if has_user and has_assistant:
    result = _USER_TRANSCRIPT_FULL_PATTERN.sub('', result)
  else:
    # user_transcript만 있을 때는 태그만 제거하고 내용 유지
    result = _USER_TRANSCRIPT_PATTERN.sub(r'\1', result)
  
  # 2. 항상: assistant_transcript 태그 제거 (내용은 유지)
  result = _ASSISTANT_TRANSCRIPT_PATTERN.sub(r'\1', result)
  
  # 3. 항상: <audio_decoder_call> 태그 제거
  result = _AUDIO_DECODER_CALL_PATTERN.sub('', result)
  
  # 앞뒤 공백 정리
  return result.strip()


def _remove_audio_tokens(content: str) -> str:
  """content에서 모든 audio token을 제거하고 정리

  Audio token (<|audio숫자|>)은 사용자에게 절대 노출되면 안 되는 토큰입니다.
  파싱 실패나 디코딩 실패 시에도 반드시 제거해야 합니다.
  """
  # <|audio숫자|> 패턴 제거
  cleaned = _AUDIO_TOKEN_PATTERN.sub("", content)
  # 연속된 공백/줄바꿈 정리
  cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
  return cleaned.strip()


def has_vision_token(discrete_image_token: str) -> bool:
  """discrete_image_token에 vision 토큰이 있는지 확인"""
  return bool(_VISION_TOKEN_PATTERN.search(discrete_image_token)) or \
         bool(_VISION_RATIO_PATTERN.search(discrete_image_token))


def find_tool_call_by_id(tool_calls: List[ToolCall], tool_call_id: str) -> ToolCall | None:
  """tool_call_id로 tool_call 찾기"""
  for tc in tool_calls:
    if tc.id == tool_call_id:
      return tc
  return None


async def process_vision_tool_call(
  tool_call: ToolCall,
  headers: Headers,
  request_id: str = "unknown"
) -> Tuple[str | None, str | None]:
  """
  Vision tool call을 처리하여 이미지를 생성하고 URL 반환

  vLLM의 tool_call arguments에서 discrete_image_token을 추출하여
  Vision Decoder API에 vlm_output으로 전달

  Returns:
    (presigned_url, s3_path) 또는 (None, None) if failed
  """
  try:
    args = json.loads(tool_call.function.arguments)
    discrete_image_token = args.get("discrete_image_token", "")

    # vision 토큰이 있는지 확인 (최소한 vision_ratio 또는 vision 토큰이 있어야 함)
    if not has_vision_token(discrete_image_token):
      token_preview = discrete_image_token[:100] if len(discrete_image_token) <= 100 else discrete_image_token[:100] + "..."
      logger.warning(f"[{request_id}] No vision tokens found in discrete_image_token: {token_preview}")
      return None, None

    token_preview = discrete_image_token[:100] if len(discrete_image_token) <= 100 else discrete_image_token[:100] + "..."
    logger.debug(f"[{request_id}] Processing vision tool call {tool_call.id}, token length: {len(discrete_image_token)}, preview: {token_preview}")

    # Vision decoding 요청 - Vision Decoder API 기본값 사용 (auto-detect height/width from ratio)
    request = VisionDecodingRequest(
      vlm_output=discrete_image_token,
      # height/width는 None으로 설정하여 Vision Decoder가 ratio token에서 자동 계산
      height=None,
      width=None,
      num_inference_steps=30,  # Vision Decoder API 기본값과 동일
      seed=None,  # Random seed
      upload_to_s3=True,
      s3_prefix="vision-decoder",
      s3_expiration=3600
    )
    resp, _ = await handle_vision_decoding(request, headers)

    # Vision Decoder 응답에서 URL과 S3 path 추출
    presigned_url = resp.presigned_url
    s3_path = resp.s3_path

    logger.debug(f"[{request_id}] Vision decoding completed for {tool_call.id}, s3_path: {s3_path}, size: {resp.width}x{resp.height}")

    return presigned_url, s3_path

  except Exception as e:
    logger.error(f"[{request_id}] Failed to process vision tool call {tool_call.id}: {e}")
    import traceback
    logger.error(traceback.format_exc())
    return None, None


async def process_vision_tool_calls_and_update(
  message: ChatMessage,
  headers: Headers
) -> None:
  """
  Vision tool_calls를 처리하고 discrete_image_token을 S3 URL로 교체
  
  content_parts를 활용하여 순서를 보존하면서 처리합니다.
  content_parts의 tool_call_ref에 해당하는 tool_call을 vision decoder로 보내서
  이미지를 생성한 다음 URL로 교체합니다.
  
  tool_calls는 유지하되, arguments.discrete_image_token 값만 S3 URL로 변경
  """
  if not message.tool_calls:
    return
  
  # Vision tool calls 수집
  vision_tool_calls = [
    tc for tc in message.tool_calls
    if tc.function.name == "t2i_model_generation"
  ]
  
  if not vision_tool_calls:
    return
  
  # content_parts가 있으면 그것을 활용하여 순서대로 처리
  if message.content_parts:
    logger.debug(f"Processing vision tool calls using content_parts (count: {len(message.content_parts)})")
    
    # content_parts에서 vision tool_call_ref 찾기
    # vLLM이 tool_call을 생성할 때 id를 재생성하므로, content_parts의 tool_call_id와 실제 tool_calls의 id가 일치하지 않을 수 있음
    # 따라서 함수 이름과 순서로 매칭
    vision_tool_call_indices = []
    vision_tool_call_refs = []  # content_parts의 tool_call_ref 정보 저장
    
    vision_tool_call_count = 0
    for i, part in enumerate(message.content_parts):
      # dict 또는 ContentPart 객체 모두 처리
      part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
      tool_call_id = part.get("tool_call_id") if isinstance(part, dict) else getattr(part, "tool_call_id", None)
      
      if part_type == "tool_call_ref":
        # content_parts의 tool_call_ref 순서대로 vision tool_call 찾기
        # vision_tool_calls의 순서와 content_parts의 tool_call_ref 순서가 일치한다고 가정
        if vision_tool_call_count < len(vision_tool_calls):
          tool_call = vision_tool_calls[vision_tool_call_count]
          if tool_call.function.name == "t2i_model_generation":
            vision_tool_call_indices.append(vision_tool_call_count)
            vision_tool_call_refs.append({
              "part_index": i,
              "part": part,
              "tool_call": tool_call
            })
            logger.debug(f"Matched vision tool_call_ref at content_parts[{i}] with tool_call[{vision_tool_call_count}] (id: {tool_call.id})")
            vision_tool_call_count += 1
          else:
            vision_tool_call_count += 1
        else:
          logger.warning(f"tool_call_ref at content_parts[{i}] has no matching vision tool_call")
    
    logger.debug(f"Matched {len(vision_tool_call_indices)} vision tool_call_refs with tool_calls")
    
    # vision tool_call_ref에 해당하는 tool_calls만 처리
    vision_tool_calls_to_process = [
      vision_tool_calls[i] for i in vision_tool_call_indices
    ]
    
    logger.debug(f"vision_tool_calls_to_process: {[tc.id for tc in vision_tool_calls_to_process]}")
    
    if vision_tool_calls_to_process:
      # 병렬로 vision 처리
      logger.debug(f"Processing {len(vision_tool_calls_to_process)} vision tool calls from content_parts")
      vision_tasks = [
        process_vision_tool_call(tc, headers) 
        for tc in vision_tool_calls_to_process
      ]
      vision_results = await asyncio.gather(*vision_tasks, return_exceptions=True)
      
      # 결과를 tool_calls에 반영 (discrete_image_token을 URL로 교체)
      for tc, result in zip(vision_tool_calls_to_process, vision_results):
        if isinstance(result, Exception):
          logger.error(f"Vision processing failed for {tc.id}: {result}")
          continue
        
        presigned_url, s3_key = result
        if presigned_url:
          # presigned URL을 직접 사용 (HTTP 접근 가능)
          # arguments 업데이트
          tc.function.arguments = json.dumps({
            "discrete_image_token": presigned_url
          }, ensure_ascii=False)
          
          logger.debug(f"Updated tool_call {tc.id}: discrete_image_token -> {presigned_url[:80]}...")
        elif s3_key:
          # presigned_url이 없으면 s3:// 형식 사용
          s3_url = f"s3://{settings.WBL_S3_BUCKET_NAME}/{s3_key}"
          tc.function.arguments = json.dumps({
            "discrete_image_token": s3_url
          }, ensure_ascii=False)
          logger.debug(f"Updated tool_call {tc.id}: discrete_image_token -> {s3_url}")
        else:
          logger.warning(f"Vision processing returned empty result for {tc.id}")
    
    # content_parts의 tool_call_ref를 URL로 교체 (매칭된 것만)
    for ref_info in vision_tool_call_refs:
      part = ref_info["part"]
      tool_call = ref_info["tool_call"]
      
      # arguments에서 URL 추출
      try:
        args = json.loads(tool_call.function.arguments)
        url = args.get("discrete_image_token", "")
        if url and (url.startswith("http://") or url.startswith("https://") or url.startswith("s3://")):
          # tool_call_ref를 image_url로 변환
          if isinstance(part, dict):
            part["type"] = "image_url"
            part["image_url"] = {"url": url}
            # tool_call_id는 제거 (더 이상 필요 없음)
            if "tool_call_id" in part:
              del part["tool_call_id"]
          else:
            # ContentPart 객체인 경우
            part.type = "image_url"
            part.image_url = {"url": url}
            part.tool_call_id = None
          logger.debug(f"Updated content_parts[{ref_info['part_index']}]: tool_call_ref -> image_url (tool_call id: {tool_call.id})")
      except Exception as e:
        logger.warning(f"Failed to update content_parts[{ref_info['part_index']}]: {e}")
  
  else:
    # content_parts가 없으면 기존 방식대로 처리
    logger.debug(f"Processing {len(vision_tool_calls)} vision tool calls (no content_parts)")
    vision_tasks = [
      process_vision_tool_call(tc, headers) 
      for tc in vision_tool_calls
    ]
    vision_results = await asyncio.gather(*vision_tasks, return_exceptions=True)
    
    # 결과를 tool_calls에 반영 (discrete_image_token을 URL로 교체)
    for tc, result in zip(vision_tool_calls, vision_results):
      if isinstance(result, Exception):
        logger.error(f"Vision processing failed for {tc.id}: {result}")
        continue
      
      presigned_url, s3_key = result
      if presigned_url:
        # presigned URL을 직접 사용 (HTTP 접근 가능)
        # arguments 업데이트
        tc.function.arguments = json.dumps({
          "discrete_image_token": presigned_url
        }, ensure_ascii=False)
        
        logger.debug(f"Updated tool_call {tc.id}: discrete_image_token -> {presigned_url[:80]}...")
      elif s3_key:
        # presigned_url이 없으면 s3:// 형식 사용
        s3_url = f"s3://{settings.WBL_S3_BUCKET_NAME}/{s3_key}"
        tc.function.arguments = json.dumps({
          "discrete_image_token": s3_url
        }, ensure_ascii=False)
        logger.debug(f"Updated tool_call {tc.id}: discrete_image_token -> {s3_url}")
      else:
        logger.warning(f"Vision processing returned empty result for {tc.id}")


# Audio token 유효성 검사 상수
_MIN_AUDIO_TOKENS = 3  # 최소 토큰 수 (너무 적으면 의미 없음)
_MAX_AUDIO_TOKENS = 2000  # 최대 토큰 수 (너무 많으면 처리 불가)
_MIN_UNIQUE_RATIO = 0.01  # 최소 고유 토큰 비율 (1% 미만이면 비정상)


def _validate_audio_tokens(tokens: list[int]) -> Tuple[bool, str]:
  """
  오디오 토큰의 유효성을 검사합니다.

  Returns:
    (is_valid, error_message)
  """
  if not tokens:
    return False, "Empty tokens"

  if len(tokens) < _MIN_AUDIO_TOKENS:
    return False, f"Too few tokens: {len(tokens)} < {_MIN_AUDIO_TOKENS}"

  if len(tokens) > _MAX_AUDIO_TOKENS:
    return False, f"Too many tokens: {len(tokens)} > {_MAX_AUDIO_TOKENS}"

  # 고유 토큰 비율 검사 (같은 토큰만 반복되면 비정상)
  unique_tokens = set(tokens)
  unique_ratio = len(unique_tokens) / len(tokens)
  if unique_ratio < _MIN_UNIQUE_RATIO:
    return False, f"Low unique token ratio: {unique_ratio:.2%} (tokens mostly repeated)"

  return True, ""


def extract_audio_info(content: str) -> Tuple[str, dict[str, Any], list[int]]:
  """
  오디오 토큰을 포함한 content에서 텍스트, speaker 정보, 토큰을 추출.

  예상 형식: text \n\n speaker_info_json \n\n audio_tokens
  하지만 형식이 맞지 않아도 오디오 토큰이 있으면 기본값으로 처리.

  Returns:
    (text_part, speaker_info, tokens) 또는 (None, None, None) if 파싱 실패
  """
  matches = _AUDIO_TOKEN_PATTERN.findall(content)
  if len(matches) == 0:
    return None, None, None

  tokens = [int(match) for match in matches]

  # 토큰 유효성 검사
  is_valid, error_msg = _validate_audio_tokens(tokens)
  if not is_valid:
    logger.warning(f"Invalid audio tokens: {error_msg}")
    return None, None, None

  # 오디오 토큰 부분을 제거하여 텍스트/speaker 정보 추출 시도
  # 토큰 패턴: <|audio1234|>
  text_content_last = content.find(_AUDIO_TOKEN_PREFIX)
  text_content = content[:text_content_last].rstrip()

  # 1. 먼저 <speaker_info>{json}</speaker_info> 태그 형식 시도 (vLLM 새 응답 형식)
  tag_match = _SPEAKER_INFO_TAG_PATTERN.search(text_content)
  # 2. 없으면 \n\n{json}\n\n 형식 시도 (기존 형식)
  pattern_match = _SPEAKER_INFO_PATTERN.search(text_content)

  if tag_match:
    try:
      speaker_info = json.loads(tag_match.group(1))  # capture group에서 JSON만 추출
      logger.debug(f"Parsed speaker_info from <speaker_info> tag: {speaker_info}")
    except json.JSONDecodeError:
      logger.warning(f"Failed to parse speaker_info JSON from tag: {tag_match.group(1)}")
      speaker_info = {}  # 파싱 실패 시 빈 dict (audio decoder가 default 사용)
    # <speaker_info> 태그 이전 부분만 텍스트로 사용
    text_content = text_content[:tag_match.start()]
  elif pattern_match:
    try:
      speaker_info = json.loads(pattern_match.group(1))  # capture group에서 JSON만 추출
      logger.debug(f"Parsed speaker_info from pattern: {speaker_info}")
    except json.JSONDecodeError:
      logger.warning(f"Failed to parse speaker_info JSON: {pattern_match.group(1)}")
      speaker_info = {}  # 파싱 실패 시 빈 dict (audio decoder가 default 사용)
    # speaker_info 이전 부분만 텍스트로 사용
    text_content = text_content[:pattern_match.start()]
  else:
    logger.debug(f"No speaker_info found in content, audio decoder will use default")
    speaker_info = {}  # speaker_info 없으면 빈 dict (audio decoder가 default 사용)
    # speaker_info가 없으면 오디오 토큰 전까지 전부 텍스트

  # 끝의 공백만 제거 (앞의 의미있는 공백은 보존)
  return text_content.rstrip(), speaker_info, tokens


async def handle_chat_completion(request: ChatCompletionRequest, raw_request: Request) -> Tuple[ChatCompletionResponse, Headers] | StreamingResponse | httpx.Response:

  headers = raw_request.headers.mutablecopy()
  if "Content-Length" in headers:
    del headers["Content-Length"]

  # ============================================================================
  # 모든 메시지에서 미디어 콘텐츠 수집 (멀티턴 지원)
  # ============================================================================
  # 이전: 마지막 메시지만 처리 → 멀티턴에서 이전 이미지/오디오 URL 처리 불가
  # 수정: 모든 메시지의 이미지 + 오디오 URL을 수집하여 처리
  # Track B: Vision Encoder (이미지) + Audio Encoder (오디오) 모두 지원
  # ============================================================================
  all_media_contents: List[Tuple[Any, ContentType]] = []

  for msg in request.messages:
    # msg가 dict일 수도 있고 Pydantic 모델일 수도 있음
    content = msg.get('content') if isinstance(msg, dict) else getattr(msg, 'content', None)
    if content is not None and isinstance(content, list):
      media_contents = await get_media_contents(msg)
      all_media_contents.extend(media_contents)

  if not await validate_media_contents(all_media_contents):
    raise HTTPException(status_code=400, detail="some of media contents are not available: check expired")

  # 입력 오디오 URL 추적 (Audio-to-Audio에서 참조 오디오로 사용)
  # 멀티턴 대화에서 여러 오디오가 있을 수 있으므로 리스트로 관리
  # 마지막(가장 최근) 오디오를 zero-shot 참조로 사용
  # - 로드밸런싱 환경에서 다른 서버로 요청이 갈 수 있음
  # - 이 경우 전체 chat history를 받지만 모든 오디오가 처음 보는 상황
  # - 마지막 오디오가 현재 화자의 가장 최근 목소리이므로 이를 참조로 사용
  input_audio_urls: List[str] = []

  # 모든 메시지에서 오디오 URL 수집 (zero-shot 참조용)
  for msg in request.messages:
    # msg가 dict일 수도 있고 Pydantic 모델일 수도 있음
    content = msg.get('content') if isinstance(msg, dict) else getattr(msg, 'content', None)
    if content is not None and isinstance(content, list):
      for part in content:
        part_dict = part if isinstance(part, dict) else (part.model_dump() if hasattr(part, 'model_dump') else {})
        if part_dict.get('type') == 'input_audio':
          audio_data = part_dict.get('input_audio', {}).get('data', '')

          # Case 1: 직접 URL인 경우
          if audio_data.startswith(("http://", "https://", "s3://", "file://")):
            # S3 .pt 임베딩 경로인 경우 원본 URL 추출 시도
            if audio_data.startswith("s3://") and audio_data.endswith(".pt"):
              original_url = extract_original_url_from_embedding_path(audio_data)
              if original_url:
                input_audio_urls.append(original_url)
                logger.debug(f"Extracted original audio URL from S3 embedding: {original_url[:80]}...")
            else:
              input_audio_urls.append(audio_data)
              logger.debug(f"Found audio URL in message: {audio_data[:80]}...")

          # Case 2: Base64 인코딩된 경우
          else:
            try:
              decoded_data = base64.b64decode(audio_data).decode("utf-8")
              # Case 2-1: Base64 인코딩된 S3 .pt 경로 (캐시 히트)
              if decoded_data.startswith("s3://") and decoded_data.endswith(".pt"):
                original_url = extract_original_url_from_embedding_path(decoded_data)
                if original_url:
                  input_audio_urls.append(original_url)
                  logger.debug(f"Extracted original audio URL from base64 S3 embedding: {original_url[:80]}...")
              # Case 2-2: Base64 인코딩된 직접 URL
              elif decoded_data.startswith(("http://", "https://", "file://")):
                input_audio_urls.append(decoded_data)
                logger.debug(f"Found base64 encoded audio URL in message: {decoded_data[:80]}...")
            except Exception:
              # Base64 디코딩 실패 = 실제 오디오 데이터 (raw bytes)
              pass

  # ============================================================================
  # 캐시를 활용한 미디어 인코딩 (Track B: Vision + Audio)
  # ============================================================================
  # 1. 이미 S3 URL (.pt)이면 건너뛰기
  # 2. 캐시에 있으면 캐시된 S3 경로 사용 (Encoder 호출 생략)
  # 3. 그 외에만 Encoder 호출 후 캐시에 저장
  #
  # Vision Encoder: 이미지 URL → S3 embedding (.pt)
  # Audio Encoder: 오디오 URL → S3 embedding (.pt)
  # ============================================================================
  media_to_encode: List[Tuple[Any, ContentType, str]] = []  # (content, type, original_url)
  media_already_encoded = []

  for media_content, content_type in all_media_contents:
    if content_type == ContentType.IMAGE_URL:
      url = media_content.image_url.get("url", "")

      # 1. 이미 S3 embedding URL이면 건너뛰기
      if is_s3_embedding_url(url):
        media_already_encoded.append((media_content, content_type))
        logger.debug(f"Skipping vision encoding for S3 embedding: {url[:80]}...")
        continue

      # 2. 캐시에서 조회
      cached_s3_path = get_cached_embedding(url, track="track_b")
      if cached_s3_path:
        media_content.image_url["url"] = cached_s3_path
        media_already_encoded.append((media_content, content_type))
        logger.debug(f"Using cached vision embedding for: {url[:80]}...")
        continue

      # 3. 인코딩 필요
      media_to_encode.append((media_content, content_type, url))

    elif content_type == ContentType.INPUT_AUDIO:
      # 오디오는 base64 인코딩된 S3 경로 또는 일반 URL일 수 있음
      # 실제 데이터가 들어올 수도 있으므로 유연하게 처리
      audio_data = media_content.input_audio.data

      # 1. S3 URL이 직접 들어온 경우 (base64 인코딩 없음)
      if audio_data.startswith("s3://") and audio_data.endswith(".pt"):
        # S3 embedding URL을 base64로 인코딩
        media_content.input_audio.data = base64.b64encode(audio_data.encode()).decode('utf-8')
        media_already_encoded.append((media_content, content_type))
        logger.debug(f"Using S3 embedding URL directly (encoded to base64): {audio_data[:80]}...")
        continue

      # 2. Base64 디코딩 시도
      try:
        decoded_data = base64.b64decode(audio_data).decode("utf-8")
        if decoded_data.startswith("s3://") and decoded_data.endswith(".pt"):
          # 이미 S3 embedding이 base64로 인코딩된 경우
          media_already_encoded.append((media_content, content_type))
          logger.debug(f"Skipping audio encoding for S3 embedding: {decoded_data[:80]}...")
          continue
      except Exception as e:
        # Base64 디코딩 실패 = 실제 오디오 데이터이거나 잘못된 형식
        logger.debug(f"Audio data is not base64 S3 URL, will encode: {str(e)[:100]}")

      # 3. 캐시에서 조회 (URL인 경우)
      if audio_data.startswith(("http://", "https://")):
        cached_s3_path = get_cached_embedding(audio_data, track="track_b")
        if cached_s3_path:
          media_content.input_audio.data = base64.b64encode(cached_s3_path.encode()).decode('utf-8')
          media_already_encoded.append((media_content, content_type))
          logger.debug(f"Using cached audio embedding for: {audio_data[:80]}...")
          continue

      # 4. 인코딩 필요 - 원본 오디오 URL 저장 (Audio-to-Audio에서 참조 오디오로 사용)
      # audio_data가 URL인 경우 저장 (http://, https://, s3://, file:// 등)
      original_url = audio_data if audio_data.startswith(("http://", "https://", "s3://", "file://")) else ""
      if original_url:
        input_audio_urls.append(audio_data)
        logger.debug(f"Saved input_audio_url for zero-shot audio decoding: {audio_data[:80]}...")
      media_to_encode.append((media_content, content_type, original_url))
    else:
      media_to_encode.append((media_content, content_type, ""))

  # 인코딩이 필요한 미디어만 처리
  media_encoding_future_list = []
  for media_content, content_type, _ in media_to_encode:
    if content_type == ContentType.IMAGE_URL:
      media_encoding_future_list.append(handle_vision_encoding(media_content))
    if content_type == ContentType.INPUT_AUDIO:
      media_encoding_future_list.append(handle_audio_encoding(media_content))

  if media_encoding_future_list:
    media_encoding_results = await asyncio.gather(*media_encoding_future_list, return_exceptions=False)
    for (encoding_result, _), (media_content, content_type, original_url) in zip(media_encoding_results, media_to_encode):
      s3_path = f"s3://{settings.WBL_S3_BUCKET_NAME}/{encoding_result.s3_key}"

      if content_type == ContentType.IMAGE_URL:
        media_content.image_url["url"] = s3_path
        # 캐시에 저장
        if original_url:
          cache_embedding(original_url, s3_path, track="track_b")
      elif content_type == ContentType.INPUT_AUDIO:
        media_content.input_audio.data = base64.b64encode(s3_path.encode()).decode('utf-8')
        # 캐시에 저장
        if original_url:
          cache_embedding(original_url, s3_path, track="track_b")

  # 수집된 오디오 URL 로깅
  if input_audio_urls:
    logger.debug(f"Collected {len(input_audio_urls)} audio URLs for zero-shot reference. Using last: {input_audio_urls[-1][:80]}...")

  if request.stream:
    # FIXME(junhee.yoo): if audio token is exist and stream=True, should I gather output tokens into buffer and processing it?
    return await make_stream_response(handle_stream_llm, request, headers)

  llm_resp, llm_resp_headers = await handle_llm(request, headers=headers)

  message = llm_resp.choices[0].message
  content = message.content

  # vLLM 응답에서 content_parts 확인
  if message.content_parts:
    logger.debug(f"Received content_parts from vLLM: {len(message.content_parts)} parts")
    for i, part in enumerate(message.content_parts):
      part_type = part.get("type") if isinstance(part, dict) else getattr(part, "type", None)
      logger.debug(f"  content_parts[{i}]: type={part_type}")
  else:
    logger.debug("No content_parts in vLLM response")

  # 1. Vision tool_calls 처리 (discrete_image_token -> S3 URL로 교체)
  # content_parts를 활용하여 순서를 보존하면서 처리
  # tool_calls는 유지하고, arguments만 업데이트
  await process_vision_tool_calls_and_update(message, headers)
  
  # 2. content_parts 처리 완료 후 정리
  # content_parts의 tool_call_ref는 이미 image_url로 변환되었으므로
  # 유저 응답에 포함시킬 수 있음 (선택적)
  # 현재는 유지하되, 필요시 제거 가능

  # 3. 레거시 Audio 토큰 처리 (content에 직접 포함된 경우)
  if content is not None and has_audio_token(content):
    logger.info(f"Audio token detected in content (length: {len(content)})")
    text_part, speaker_info, audio_tokens = extract_audio_info(content)
    logger.debug(f"extract_audio_info result - speaker_info: {speaker_info}, tokens: {len(audio_tokens) if audio_tokens else 0}")
    # extract_audio_info는 파싱 실패 시 None, None, None을 반환함
    if speaker_info is None or audio_tokens is None:
      logger.warning(f"Audio token detected but extraction failed. Removing audio tokens from content.")
      # Audio token을 강제로 제거 (사용자에게 노출 방지)
      message.content = _remove_audio_tokens(content)
    else:
      # speaker_info에 따라 3가지 TTS 모드 결정:
      # 1. Zero-shot TTS: {"is_ref_audio": true} AND input_audio_urls 존재
      #    → speaker=None, ref_audio_url=input_audio (입력 음성 클론)
      # 2. Prompt TTS: {"gender": "f", "age": "40-50"} (gender 존재)
      #    → speaker=Speaker(id="fkms/mhwj"), ref_audio_url=None (finetuned 음색)
      # 3. Default TTS: {} (빈 dict, speaker_info 없음)
      #    → speaker=None, ref_audio_url=None (audio decoder default 음색)

      # NOTE(junhee.yoo): extra_body can be None
      #                   replace this code block with flatten_extra_body later
      is_force_ref_audio = request.is_ref_audio
      if (extra_body := getattr(request, "extra_body", None)) is not None:
        is_force_ref_audio = extra_body.get("is_ref_audio", False)
        if is_force_ref_audio and not input_audio_urls:
          logger.warning(f"extra_body has is_ref_audio={is_force_ref_audio} but input_audio_urls is not provided, ignoring is_ref_audio")
          is_force_ref_audio = False

      is_ref_audio = speaker_info.get("is_ref_audio", False) or is_force_ref_audio
      has_gender = "gender" in speaker_info
      has_age = "age" in speaker_info

      # 케이스 1: Zero-shot TTS (is_ref_audio=true AND input_audio 있음)
      if is_ref_audio and input_audio_urls:
        speaker = None
        ref_audio_url = input_audio_urls[-1]
        logger.info(f"Using zero-shot cloning with is_ref_audio=true and ref_audio_url: {ref_audio_url[:80]}...")
      # is_ref_audio=true인데 입력 오디오가 없는 경우 - 경고 후 fallback
      elif is_ref_audio and not input_audio_urls:
        logger.warning("vLLM generated is_ref_audio=true but no input_audio_urls provided, falling back to default TTS")
        speaker = None
        ref_audio_url = None
      # 케이스 2: Prompt TTS (gender 정보 있음)
      elif has_gender or has_age:
        speaker = Speaker(age=speaker_info.get("age", None), gender=speaker_info.get("gender", None))
        ref_audio_url = None
        logger.debug(f"Using prompt TTS (finetuned) with speaker: {speaker}")
      # 케이스 3: Default TTS (speaker_info 없음)
      else:
        speaker = None
        ref_audio_url = None
        logger.debug("No speaker_info, using audio decoder default")

      try:
        resp, _ = await handle_audio_decoding(
          # NOTE: If audio format is not specified, default to wav when processing audio tokens
          audio_tokens, request.audio.get("format", "wav") if request.audio else "wav", speaker, headers,
          ref_audio_url=ref_audio_url,  # Audio-to-Audio: 마지막 입력 오디오를 참조로 사용
          fallback_to_finetuned=True  # zero-shot 실패 시 finetuned로 재시도
        )
        if ref_audio_url:
          logger.debug(f"Audio decoding with ref_audio_url (zero-shot mode): {ref_audio_url[:80]}...")
        message.content = text_part
        presigned_url = s3_connection.create_presigned_get(
          settings.WBL_S3_BUCKET_NAME,
          resp.s3_key, expiration=3600)  # FIXME(junhee.yoo): expiration time should be configurable

        encoded_url_key = base64.urlsafe_b64encode(presigned_url.encode()).decode("utf-8")
        message.audio = OpenAIChatCompletionAudio(
          data=encoded_url_key,
          transcript=text_part,
          id=encoded_url_key,
          expires_at=int((datetime.now(timezone.utc) + timedelta(seconds=3600)).timestamp())
        )
      except ValueError as e:
        # Units validation 실패 시 (잘못된 토큰) - 텍스트만 반환
        logger.warning(f"Audio token validation failed, returning text only: {str(e)[:200]}")
        message.content = text_part if text_part else _remove_audio_tokens(content)
      except Exception as e:
        # Audio Decoder 실패 시 텍스트만 반환 (500 에러 대신 graceful 처리)
        logger.error(f"Audio decoding failed, returning text only: {str(e)[:200]}")
        # Audio token이 절대 사용자에게 노출되지 않도록 보장
        message.content = text_part if text_part else _remove_audio_tokens(content)
        # audio 필드는 설정하지 않음

  # 4. audio_synthesis tool_call 처리
  audio_tool_calls = [
    tc for tc in (message.tool_calls or [])
    if tc.function.name == "audio_synthesis"
  ]

  if audio_tool_calls:
    for tc in audio_tool_calls:
      try:
        args = json.loads(tc.function.arguments)
        units = args.get("units", [])
        speaker_info = args.get("speaker_info", {})

        # speaker_info에 따라 Speaker 객체 생성
        # {"is_ref_audio": true} → zero-shot cloning (ref_audio_url 사용, speaker=None)
        # {"gender": "f", "age": "40-50"} → finetuned mode (speaker.id 사용)
        # {} (빈 dict) → audio decoder default 사용 (speaker=None)

        is_ref_audio = speaker_info.get("is_ref_audio", False)
        has_gender = "gender" in speaker_info

        if is_ref_audio or not has_gender:
          # Zero-shot cloning 모드 또는 speaker_info 없음 → speaker=None
          speaker = None
          ref_audio_url_for_synthesis = input_audio_urls[-1] if input_audio_urls else None
          if is_ref_audio and ref_audio_url_for_synthesis:
            logger.info(f"Using zero-shot cloning with is_ref_audio=true and ref_audio_url: {ref_audio_url_for_synthesis[:80]}...")
          elif not has_gender:
            logger.debug("No speaker_info (gender) in tool_call, audio decoder will use default")
        else:
          # Finetuned 모드 (gender 기반)
          voice_type = Speaker.VoiceType.MALE if speaker_info.get("gender") == "m" else Speaker.VoiceType.FEMALE
          speaker = Speaker(id=voice_type.value)
          ref_audio_url_for_synthesis = None  # finetuned 모드에서는 ref_audio_url 사용 안 함
          logger.debug(f"Using finetuned mode with speaker id: {voice_type.value}")

        resp, _ = await handle_audio_decoding(
          units, "wav", speaker, headers,
          ref_audio_url=ref_audio_url_for_synthesis,
          fallback_to_finetuned=True  # zero-shot 실패 시 finetuned로 재시도
        )
        if ref_audio_url_for_synthesis:
          logger.debug(f"Audio synthesis with ref_audio_url (zero-shot mode): {ref_audio_url_for_synthesis[:80]}...")
        presigned_url = s3_connection.create_presigned_get(
          settings.WBL_S3_BUCKET_NAME,
          resp.s3_key, expiration=3600
        )

        encoded_url_key = base64.urlsafe_b64encode(presigned_url.encode()).decode("utf-8")
        message.audio = OpenAIChatCompletionAudio(
          data=encoded_url_key,
          transcript=message.content or "",
          id=encoded_url_key,
          expires_at=int((datetime.now(timezone.utc) + timedelta(seconds=3600)).timestamp())
        )

        # audio tool_call 제거
        message.tool_calls = [t for t in message.tool_calls if t.id != tc.id]

      except ValueError as e:
        # Units validation 실패 시 - tool_call 제거하고 계속
        logger.warning(f"Audio tool call validation failed, removing tool_call: {str(e)[:200]}")
        message.tool_calls = [t for t in message.tool_calls if t.id != tc.id]
        # Audio token이 노출되지 않도록 content에서도 제거
        if message.content and has_audio_token(message.content):
          message.content = _remove_audio_tokens(message.content)

      except Exception as e:
        logger.error(f"Failed to process audio tool call: {e}")
        # Audio token이 노출되지 않도록 content에서도 제거
        if message.content and has_audio_token(message.content):
          message.content = _remove_audio_tokens(message.content)

  # 최종 응답에서 content_parts 처리
  # content_parts가 있고 image_url로 변환된 경우 유지 (유저에게 순서대로 반환)
  # 그렇지 않은 경우 제거
  if message.content_parts:
    # tool_call_ref가 남아있는지 확인 (처리되지 않은 경우)
    has_unprocessed_tool_call_ref = any(
      (part.get("type") if isinstance(part, dict) else getattr(part, "type", None)) == "tool_call_ref"
      for part in message.content_parts
    )
    if has_unprocessed_tool_call_ref:
      # 처리되지 않은 tool_call_ref가 있으면 제거
      logger.warning("Removing content_parts with unprocessed tool_call_ref")
      message.content_parts = None
    # image_url로 변환된 경우는 유지 (유저에게 순서대로 반환)

  # Safety check: content에 audio token이 남아있으면 제거
  # 위의 모든 처리를 거쳐도 audio token이 남아있는 경우를 최종적으로 방지
  if message.content and has_audio_token(message.content):
    logger.warning("Audio tokens found in final content, removing them for safety")
    message.content = _remove_audio_tokens(message.content)

  # Audio-to-Audio 응답에서 transcript 관련 태그들 정리
  # <user_transcript>, <assistant_transcript>, <audio_decoder_call> 태그 제거
  if message.content:
    message.content = _strip_transcript_tags(message.content)
  if message.audio and message.audio.transcript:
    message.audio.transcript = _strip_transcript_tags(message.audio.transcript)

  headers = MutableHeaders()
  headers.update(llm_resp_headers)
  if "Content-Length" in headers:
    del headers["Content-Length"]

  return llm_resp, headers
