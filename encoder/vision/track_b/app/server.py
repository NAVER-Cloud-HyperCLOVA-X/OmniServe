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
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from PIL import Image
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
    processor_path = CONFIG.get("processor_model_name_or_path", "configs/preprocessor_config.json")
    cont_config_path = CONFIG.get("cont_config_path", "configs/omni_cont_config.yaml")
    cont_weight_path = models_config.get("vision_module", {}).get("path", "weights/vision_weight.pt")
    disc_weight_path = models_config.get("discrete_vision", {}).get("path", "weights/ta_tok.pth")
    mm_projector_path = models_config.get("mm_projector", {}).get("path", "weights/mm_projector_weights.pt")
    llm_hidden_size = CONFIG.get("llm_hidden_size", 4096)
    # model_id: 환경변수 MODEL_ID > config.json > 기본값 순으로 적용
    model_id = os.getenv("MODEL_ID") or CONFIG.get("model_id", "default")

    app.state.model = Model(
        processor_model_name_or_path=processor_path,
        cont_config_path=cont_config_path,
        cont_weight_path=cont_weight_path,
        disc_weight_path=disc_weight_path,
        mm_projector_weight_path=mm_projector_path,
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


class ImageRequest(BaseModel):
    image_url: Optional[str] = Field(default=None, description="Image URL (mutually exclusive with image_base64)")
    image_base64: Optional[str] = Field(
        default=None, description="Base64 encoded image (mutually exclusive with image_url)"
    )
    anyres: Optional[bool] = Field(default=False, description="Use anyres processing")
    unpad: Optional[bool] = Field(default=False, description="Use unpad processing")
    num_queries_vis_abstractor: Optional[int] = Field(default=0, description="Number of visual abstractor queries")


class VideoRequest(BaseModel):
    video_url: Optional[str] = Field(default=None, description="Video URL (mutually exclusive with video_base64)")
    video_base64: Optional[str] = Field(
        default=None, description="Base64 encoded video (mutually exclusive with video_url)"
    )


class MediaRequest(BaseModel):
    media_url: Optional[str] = Field(
        default=None, description="Media URL (image or video, mutually exclusive with media_base64)"
    )
    media_base64: Optional[str] = Field(
        default=None, description="Base64 encoded media (mutually exclusive with media_url)"
    )
    anyres: Optional[bool] = Field(default=False, description="Use anyres processing (for images)")
    unpad: Optional[bool] = Field(default=False, description="Use unpad processing (for images)")
    num_queries_vis_abstractor: Optional[int] = Field(
        default=0, description="Number of visual abstractor queries (for images)"
    )


class InferenceResponse(ResponseBase):
    s3_key: str


class TokenLengthResponse(ResponseBase):
    vision_query_length: int


# ---- Media utils ----


async def get_meta_from_s3(url: urllib.parse.ParseResult) -> dict:
    bucket = url.netloc
    key = url.path.lstrip("/")
    logger.info(f"[S3] get_meta_from_s3 시작 - bucket={bucket}, key={key}")
    try:
        logger.info(f"[S3] head_object 호출 - Bucket={bucket}, Key={key}")
        metadata = _S3_CONN.get_sync_client().head_object(Bucket=bucket, Key=key)
        logger.info(f"[S3] head_object 성공 - ContentType={metadata.get('ContentType', 'N/A')}, ContentLength={metadata.get('ContentLength', 'N/A')}")
        logger.info(f"[S3] Metadata from S3: {metadata}")
        return metadata

    except Exception as e:
        logger.error(f"[S3] get_meta_from_s3 예외 발생 - bucket={bucket}, key={key}, error={e}", exc_info=True)
        logger.warning(f"[S3] Failed to get metadata from S3: {e}. Defaulting to empty dictionary.")
        return {}


async def get_meta_from_http(url: urllib.parse.ParseResult) -> dict:
    logger.info(f"[S3] get_meta_from_http 시작 - url={url.geturl()}")
    try:
        logger.info(f"[S3] HTTP HEAD 요청 - url={url.geturl()}")
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.head(url.geturl(), follow_redirects=True)
            logger.info(f"[S3] HTTP HEAD 응답 - status_code={response.status_code}")
            response.raise_for_status()
            logger.info(f"[S3] Response headers from HTTP: {response.headers}")
            return response.headers

    except Exception as e:
        logger.warning(f"[S3] HTTP HEAD 요청 실패 - url={url.geturl()}, error={e}")
        # Check if URL is from S3-compatible object storage
        s3_endpoint = os.getenv("NCP_S3_ENDPOINT", "")
        endpoint_host = s3_endpoint.replace("https://", "").replace("http://", "") if s3_endpoint else ""
        logger.info(f"[S3] S3 폴백 확인 - s3_endpoint={s3_endpoint}, endpoint_host={endpoint_host}, url.netloc={url.netloc}")
        if s3_endpoint and endpoint_host in url.netloc:
            path = url.path.lstrip("/").split("/")
            bucket_name = path[0]
            key = "/".join(path[1:])

            # try with s3
            s3_url = urllib.parse.urlparse(f"s3://{bucket_name}/{key}")
            logger.info(f"[S3] S3 폴백 URL 생성 - s3_url={s3_url}")
            logger.info(f"[S3] Trying to get metadata from S3: {s3_url}")
            return await get_meta_from_s3(s3_url)

        logger.warning(f"[S3] Failed to get header info from URL: {e}. Defaulting to empty dictionary.")
        return {}


async def detect_media_type_from_url(url: str) -> str:
    """
    URL의 Content-Type 헤더를 확인하여 이미지인지 비디오인지 판단
    Returns: 'image' or 'video'
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

    if content_type.startswith("image/"):
        return "image"
    elif content_type.startswith("video/"):
        return "video"
    else:
        # Content-Type이 명확하지 않은 경우, URL 확장자로 판단 시도
        url_lower = url.lower()
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"]
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"]

        if any(url_lower.endswith(ext) for ext in image_extensions):
            return "image"
        elif any(url_lower.endswith(ext) for ext in video_extensions):
            return "video"
        else:
            # 기본값으로 이미지로 처리 (기존 동작 유지)
            logger.warning(
                f"Could not determine media type from Content-Type '{content_type}' and URL extension. Defaulting to image."
            )
            return "image"


def detect_media_type_from_base64(base64_str: str) -> str:
    """
    Base64 문자열에서 data URL 형식을 확인하여 이미지인지 비디오인지 판단
    Returns: 'image' or 'video'
    """
    if base64_str.startswith("data:image/"):
        return "image"
    elif base64_str.startswith("data:video/"):
        return "video"
    else:
        # data URL 형식이 아닌 경우, 기본값으로 이미지로 처리
        logger.warning("Could not determine media type from base64 string. Defaulting to image.")
        return "image"


class ImageUtils:
    @staticmethod
    def get_sample_images():
        # Dummy: return a list of PIL images for warmup
        warmup_paths = CONFIG.get("warmup_image_paths", ["resources/tradeoff_sota.png"])
        images = []
        for path in warmup_paths:
            if os.path.exists(path):
                images.append(Image.open(path))
        if not images:
            # Create a dummy image if no warmup images available
            images.append(Image.new("RGB", (384, 384), color="white"))
        return images


# ---- Warmup procedures ----


def warmup(model: Model):
    logger.info("Warm-up started ############")
    warmup_images = ImageUtils.get_sample_images()
    for idx, warmup_image in enumerate(warmup_images):
        try:
            # 이미지를 base64로 변환
            buffer = io.BytesIO()
            warmup_image.save(buffer, format="PNG")
            image_bytes = buffer.getvalue()
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # base64 해시를 계산하여 캐시 키 생성
            base64_hash = hashlib.sha256(image_base64.encode()).hexdigest()[:16]
            cache_key = (
                f"source/derived/embedding/vision/{model.model_id}/image_base64_{base64_hash}.pt"
            )

            # warmup 전에 캐시 삭제 (항상 실제 처리를 테스트하기 위해)
            model._delete_from_cache(cache_key)

            # process_image_from_base64 테스트
            cache_result = model.process_image_from_base64(base64_str=image_base64)
            s3_key = cache_result.get("s3_key", "")

            if idx % 5 == 0:
                logger.info(
                    f"Warm-up progress: {idx}th element completed. s3_key={s3_key}"
                )
        except Exception as e:
            logger.warning(f"Warm-up failed for image {idx}: {e}")
    logger.info("Warm-up completed ############")


def livenessProbe():
    try:
        warmup_images = ImageUtils.get_sample_images()
        if warmup_images:
            user_img = warmup_images[0]
            vision_query_length = app.state.model.get_vision_query_length(user_img)
            return {"content": "ok", "vision_query_length": vision_query_length}
    except Exception as e:
        logger.error(f"Liveness probe failed: {e}")
        return {"content": "error", "error": str(e)}


# ---- ROUTES ----


@app.post("/process_image")
async def process_image(req: ImageRequest) -> InferenceResponse:
    """
    이미지 URL 또는 Base64 인코딩된 이미지를 받아서 vision encoding 수행
    image_url과 image_base64 중 하나만 제공해야 함
    """
    start = time()
    try:
        # 입력 검증
        if (req.image_url is None) == (req.image_base64 is None):
            raise HTTPException(status_code=400, detail="Exactly one of image_url or image_base64 must be provided")

        # URL 또는 Base64 처리
        if req.image_url:
            cache_result = app.state.model.process_image_from_url(
                image_url=req.image_url,
            )
        else:
            cache_result = app.state.model.process_image_from_base64(
                base64_str=req.image_base64,
            )

        # cache_result는 {"s3_key": ...} 딕셔너리
        s3_key = cache_result.get("s3_key", "")

        took = time() - start
        logger.info(f"Image processed: s3_key={s3_key}")

        return InferenceResponse(
            s3_key=s3_key,
            latency=round(took * 1000),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_video")
async def process_video(req: VideoRequest) -> InferenceResponse:
    """
    비디오 URL 또는 Base64 인코딩된 비디오를 받아서 vision encoding 수행
    video_url과 video_base64 중 하나만 제공해야 함
    """
    start = time()
    try:
        # 입력 검증
        if (req.video_url is None) == (req.video_base64 is None):
            raise HTTPException(status_code=400, detail="Exactly one of video_url or video_base64 must be provided")

        # URL 또는 Base64 처리
        if req.video_url:
            cache_result = app.state.model.process_video_from_url(
                video_url=req.video_url,
            )
        else:
            cache_result = app.state.model.process_video_from_base64(
                base64_str=req.video_base64,
            )

        # cache_result는 {"s3_key": ...} 딕셔너리
        s3_key = cache_result.get("s3_key", "")

        took = time() - start
        logger.info(f"Video processed: s3_key={s3_key}")

        return InferenceResponse(
            s3_key=s3_key,
            latency=round(took * 1000),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Video processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process_image_or_video")
async def process_image_or_video(req: MediaRequest) -> InferenceResponse:
    """
    미디어 URL 또는 Base64 인코딩된 미디어를 받아서 vision encoding 수행
    URL의 경우 Content-Type 헤더를 확인하여 자동으로 이미지/비디오를 판단
    media_url과 media_base64 중 하나만 제공해야 함
    """
    start = time()
    try:
        # 입력 검증
        if (req.media_url is None) == (req.media_base64 is None):
            raise HTTPException(status_code=400, detail="Exactly one of media_url or media_base64 must be provided")

        # 미디어 타입 자동 감지 및 처리
        if req.media_url:
            # URL의 경우 Content-Type 헤더로 판단
            media_type = await detect_media_type_from_url(req.media_url)
            logger.info(f"Detected media type from URL: {media_type}")

            if media_type == "image":
                cache_result = app.state.model.process_image_from_url(
                    image_url=req.media_url,
                )
            else:  # video
                cache_result = app.state.model.process_video_from_url(
                    video_url=req.media_url,
                )
        else:
            # Base64의 경우 data URL 형식으로 판단
            media_type = detect_media_type_from_base64(req.media_base64)
            logger.info(f"Detected media type from base64: {media_type}")

            if media_type == "image":
                cache_result = app.state.model.process_image_from_base64(
                    base64_str=req.media_base64,
                )
            else:  # video
                cache_result = app.state.model.process_video_from_base64(
                    base64_str=req.media_base64,
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
        logger.error(f"Media processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Liveness endpoint for container probe
@app.get("/live")
async def live_check():
    return livenessProbe()


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": app.state.model is not None}
