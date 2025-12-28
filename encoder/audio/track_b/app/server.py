# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import base64
import hashlib
import io
import json
import logging
import os
import urllib.parse
from urllib.parse import ParseResult
from contextlib import asynccontextmanager
from time import time
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from wbl_storage_utility.s3_util import S3Connection

from app.model import Model


_S3_CONN = S3Connection()


def load_config(config_path: str = "config.json") -> dict:
    if not os.path.isfile(config_path):
        raise RuntimeError(f"Config file {config_path} does not exist")
    with open(config_path, "r") as f:
        return json.load(f)


CONFIG = load_config(os.getenv("CONFIG_PATH", "config.json"))


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# ---- App and Model ----


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("[Startup] Loading model...")

    # Config에서 모델 경로 및 설정 로드
    models_config = CONFIG.get("models_weight", {})
    continuous_audio_config_path = CONFIG.get("continuous_audio_config_path", "app/model/configs/audio_config.json")
    continuous_audio_weight_path = models_config.get("continuous_audio", {}).get("path", "app/model/weights/qwen2_audio_encoder.pt")
    discrete_audio_config = CONFIG.get("discrete_audio_config", {})
    discrete_audio_weight_path = models_config.get("discrete_audio", {}).get("path", "app/model/weights/cosyvoice_encoder.pt")
    audio_processor_path = CONFIG.get("audio_processor_path", "app/model/weights/qwen2-audio-encoder-from-qwen2-audio-7b-instruct")
    audio_projector_weight_path = CONFIG.get("audio_projector_weight_path", "app/model/weights/audio_projector_weights.pt")
    video_audio_compressor_weight_path = CONFIG.get("video_audio_compressor_weight_path", "app/model/weights/video_audio_compressor_weights.pt")
    llm_hidden_size = CONFIG.get("llm_hidden_size", 4096)
    # model_id: 환경변수 MODEL_ID > config.json > 기본값 순으로 적용
    model_id = os.getenv("MODEL_ID") or CONFIG.get("model_id", "default")

    app.state.model = Model(
        continuous_audio_config_path=continuous_audio_config_path,
        continuous_audio_weight_path=continuous_audio_weight_path,
        discrete_audio_config=discrete_audio_config,
        discrete_audio_weight_path=discrete_audio_weight_path,
        audio_processor_path=audio_processor_path,
        audio_projector_weight_path=audio_projector_weight_path,
        video_audio_compressor_weight_path=video_audio_compressor_weight_path,
        llm_hidden_size=llm_hidden_size,
        model_id=model_id,
    )
    warmup(app.state.model)
    logger.info("[Startup] Model warmup completed.")

    yield

    # Shutdown
    app.state.model = None
    logger.info("[Shutdown] Model reference removed.")


app = FastAPI(lifespan=lifespan)
app.state.model = None

# --- Data Models ---


class DictResponse(BaseModel):
    result: dict
    latency: float


class ResponseBase(BaseModel):
    latency: float


class MediaRequest(BaseModel):
    media_url: Optional[str] = Field(
        default=None, description="Media URL (audio or video, mutually exclusive with media_base64)"
    )
    media_base64: Optional[str] = Field(
        default=None, description="Base64 encoded media (mutually exclusive with media_url)"
    )


class InferenceResponse(ResponseBase):
    s3_key: str


# ---- Media utils ----


async def get_meta_from_s3(url: urllib.parse.ParseResult) -> dict:
    try:
        metadata = _S3_CONN.get_sync_client().head_object(Bucket=url.netloc, Key=url.path.lstrip("/"))
        logger.info(f"Metadata from S3: {metadata}")
        return metadata

    except Exception as e:
        logger.warning(f"Failed to get metadata from S3: {e}. Defaulting to empty dictionary.")
        return {}


async def get_meta_from_http(url: urllib.parse.ParseResult) -> dict:
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.head(url.geturl(), follow_redirects=True)
            response.raise_for_status()
            logger.info(f"Response headers from HTTP: {response.headers}")
            return response.headers

    except Exception as e:
        # Check if URL is from S3-compatible object storage
        s3_endpoint = os.getenv("NCP_S3_ENDPOINT", "")
        if s3_endpoint and s3_endpoint.replace("https://", "").replace("http://", "") in url.netloc:
            path = url.path.lstrip("/").split("/")
            bucket_name = path[0]
            key = "/".join(path[1:])

            # try with s3
            s3_url = urllib.parse.urlparse(f"s3://{bucket_name}/{key}")
            logger.info(f"Trying to get metadata from S3: {s3_url}")
            return await get_meta_from_s3(s3_url)

        logger.warning(f"Failed to get header info from URL: {e}. Defaulting to empty dictionary.")
        return {}


async def detect_media_type_from_url(url: str) -> str:
    """
    URL의 Content-Type 헤더를 확인하여 오디오인지 비디오인지 판단
    Returns: 'audio' or 'video'
    """

    try:
        url_parsed = urllib.parse.urlparse(url)
    except ValueError as e:
        raise ValueError(f"Invalid URL: {e}")

    content_type = None
    if url_parsed.scheme == "s3":
        metadata = await get_meta_from_s3(url_parsed)
        content_type = metadata.get("ContentType", "").lower()
    elif url_parsed.scheme == "http" or url_parsed.scheme == "https":
        metadata = await get_meta_from_http(url_parsed)
        content_type = metadata.get("Content-Type", "").lower()
    else:
        raise ValueError(f"Unsupported URL scheme: {url_parsed.scheme}")

    if content_type.startswith("audio/"):
        return "audio"
    elif content_type.startswith("video/"):
        return "video"
    else:
        # Content-Type이 명확하지 않은 경우, URL 확장자로 판단 시도
        url_lower = url.lower()
        audio_extensions = [".mp3", ".wav", ".flac", ".aac", ".ogg", ".m4a", ".wma", ".opus"]
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"]

        if any(url_lower.endswith(ext) for ext in audio_extensions):
            return "audio"
        elif any(url_lower.endswith(ext) for ext in video_extensions):
            return "video"
        else:
            # 기본값으로 비디오로 처리 (비디오에서 오디오 추출 가능)
            logger.warning(
                f"Could not determine media type from Content-Type '{content_type}' and URL extension. Defaulting to video."
            )
            return "video"


def normalize_media_url(media_url: str) -> str:
    """
    media_url이 base64 urlsafe 문자열인 경우 실제 URL로 디코드.
    이미 http(s)로 시작하면 그대로 반환.
    """
    if media_url.startswith(("http://", "https://")):
        return media_url

    try:
        padding = "=" * (-len(media_url) % 4)
        decoded = base64.urlsafe_b64decode(media_url + padding).decode()
        if decoded.startswith(("http://", "https://", "s3://")):
            return decoded
    except Exception:
        pass

    return media_url


def detect_media_type_from_base64(base64_str: str) -> str:
    """
    Base64 문자열에서 data URL 형식을 확인하여 오디오인지 비디오인지 판단
    Returns: 'audio' or 'video'
    """
    if base64_str.startswith("data:audio/"):
        return "audio"
    elif base64_str.startswith("data:video/"):
        return "video"
    else:
        # data URL 형식이 아닌 경우, 기본값으로 비디오로 처리
        logger.warning("Could not determine media type from base64 string. Defaulting to video.")
        return "video"


class AudioUtils:
    @staticmethod
    def get_sample_audios():
        # Dummy: return a list of audio paths for warmup
        warmup_paths = CONFIG.get("warmup_audio_paths", [])
        return warmup_paths


# ---- Warmup procedures ----


def warmup(model: Model):
    logger.info("Warm-up started ############")
    warmup_audios = AudioUtils.get_sample_audios()
    for idx, warmup_audio_path in enumerate(warmup_audios):
        if not os.path.exists(warmup_audio_path):
            logger.warning(f"Warmup audio path does not exist: {warmup_audio_path}")
            continue

        # 오디오를 base64로 변환
        with open(warmup_audio_path, "rb") as f:
            audio_bytes = f.read()
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        # base64 해시를 계산하여 캐시 키 생성
        base64_hash = hashlib.sha256(audio_base64.encode()).hexdigest()[:16]
        cache_key = (
            f"source/derived/embedding/audio/{model.model_id}/media_base64_{base64_hash}.pt"
        )

        # warmup 전에 캐시 삭제 (항상 실제 처리를 테스트하기 위해)
        model._delete_from_cache(cache_key)

        # process_audio_from_base64 테스트
        cache_result = model.process_audio_from_base64(base64_str=audio_base64)
        s3_key = cache_result.get("s3_key", "")

        if idx % 5 == 0:
            logger.info(
                f"Warm-up progress: {idx}th element completed. s3_key={s3_key}"
            )
    logger.info("Warm-up completed ############")


def livenessProbe():
    # 간단한 더미 오디오로 테스트
    if app.state.model is not None:
        return {"content": "ok"}
    return {"content": "error", "error": "Model not loaded"}


# ---- ROUTES ----


@app.post("/process_audio")
async def process_audio(req: MediaRequest) -> InferenceResponse:
    """
    미디어 URL 또는 Base64 인코딩된 미디어를 받아서 audio encoding 수행
    URL의 경우 Content-Type 헤더를 확인하여 자동으로 오디오/비디오를 판단
    비디오인 경우 음성만 추출하여 처리
    media_url과 media_base64 중 하나만 제공해야 함
    """
    start = time()
    try:
        # 입력 검증
        if (req.media_url is None) == (req.media_base64 is None):
            raise HTTPException(status_code=400, detail="Exactly one of media_url or media_base64 must be provided")

        # 미디어 타입 자동 감지 및 처리
        if req.media_url:
            media_url = normalize_media_url(req.media_url)
            # URL의 경우 Content-Type 헤더로 판단
            media_type = await detect_media_type_from_url(media_url)
            logger.info(f"Detected media type from URL: {media_type}")

            cache_result = app.state.model.process_media_from_url(
                media_url=media_url,
                media_type=media_type,
            )
        else:
            # Base64의 경우 data URL 형식으로 판단
            media_type = detect_media_type_from_base64(req.media_base64)
            logger.info(f"Detected media type from base64: {media_type}")

            cache_result = app.state.model.process_media_from_base64(
                base64_str=req.media_base64,
                media_type=media_type,
            )

        # cache_result는 {"s3_key": ...} 딕셔너리
        s3_key = cache_result.get("s3_key", "")

        took = time() - start
        logger.info(f"Media processed (type={media_type}): s3_key={s3_key}")

        return InferenceResponse(
            s3_key=s3_key,
            latency=round(took * 1000),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


# Liveness endpoint for container probe
@app.get("/live")
async def live_check():
    return livenessProbe()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": app.state.model is not None}

