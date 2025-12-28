# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import base64
import hashlib
import io
import json
import logging
import os
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import requests
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision.transforms import ToTensor

# Setup logger first
logger = logging.getLogger(__name__)

from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoModel,
    AutoVideoProcessor,
)
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLVisionConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
)
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from wbl_storage_utility.s3_util import S3Connection

from .processing_vlm import HCXVisionV2Processor
from .qwen_vision_process import fetch_image, fetch_video, process_vision_info

# HCX 모델 관련 import
from .ta_tok import TextAlignedTokenizer

MODEL_MAPPING.register(Qwen2_5_VLVisionConfig, Qwen2_5_VisionTransformerPretrainedModel)
CONFIG_MAPPING.register("qwen2_5_vl_visual", Qwen2_5_VLVisionConfig)

S3_BUCKET_NAME=os.getenv("WBL_S3_BUCKET_NAME", "")


class VisionEncoderResult:
    """Vision encoder 처리 결과를 담는 클래스"""

    def __init__(
        self,
        continuous_feature: Optional[torch.Tensor] = None,
        discrete_tokens: Optional[torch.Tensor] = None,
        vision_query_length: int = 0,
        image_sizes: Optional[List[Dict[str, int]]] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        video_sizes: Optional[List[Dict[str, int]]] = None,
        video_duration: Optional[float] = None,
        video_fps: Optional[float] = None,
    ):
        self.continuous_feature = continuous_feature
        self.discrete_tokens = discrete_tokens
        self.vision_query_length = vision_query_length
        self.image_sizes = image_sizes
        self.image_grid_thw = image_grid_thw
        self.video_grid_thw = video_grid_thw
        self.video_sizes = video_sizes
        self.video_duration = video_duration
        self.video_fps = video_fps


def resolve_path(base_dir, path: str) -> str:
    """Convert relative path to absolute path."""
    if os.path.isabs(path):
        return path
    # Try relative to project root first
    abs_path = base_dir / path
    if abs_path.exists():
        return str(abs_path)
    # Try relative to current working directory
    if os.path.exists(path):
        return os.path.abspath(path)
    return str(abs_path)  # Return resolved path even if it doesn't exist yet


class Model:
    """Vision Encoder Model for processing images and videos."""

    def __init__(
        self,
        processor_model_name_or_path: str = "configs/preprocessor_config.json",
        cont_config_path: str = "configs/omni_cont_config.yaml",
        cont_weight_path: str = "weights/vision_weight.pt",
        disc_weight_path: str = "weights/ta_tok.pth",
        mm_projector_weight_path: str = "weights/mm_projector_weights.pt",
        llm_hidden_size: int = 4096,
        model_id: str = "default",
    ):
        """Initialize Vision Encoder Model.

        Args:
            processor_model_name_or_path: Path to processor config file or directory
            cont_config_path: Path to continuous vision model config YAML
            cont_weight_path: Path to continuous vision model weights
            disc_weight_path: Path to discrete vision model weights
            mm_projector_weight_path: Path to MM projector weights
            dtype: Model dtype (bf16, fp16, fp32)
            llm_hidden_size: Hidden dimension of the LLM (default: 4096)
        """
        # Setup device and dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_hidden_size = llm_hidden_size
        self.model_id = model_id
        # Convert relative paths to absolute paths
        # Get the base directory (assuming this file is in app/model/)
        base_dir = Path(__file__).parent.parent.parent  # Go up from app/model/__init__.py to project root

        # Resolve all paths
        processor_model_name_or_path = resolve_path(base_dir, processor_model_name_or_path)
        cont_config_path = resolve_path(base_dir, cont_config_path)
        cont_weight_path = resolve_path(base_dir, cont_weight_path)
        mm_projector_weight_path = resolve_path(base_dir, mm_projector_weight_path)
        disc_weight_path = resolve_path(base_dir, disc_weight_path)

        # Initialize s3 connection (optional - server can run without s3)
        self.s3_conn = S3Connection()
        logger.info("S3 connection established")

        # Load models
        self._load_continuous_vision_model(cont_config_path, cont_weight_path)
        self._load_mm_projector(mm_projector_weight_path)
        self._load_discrete_vision_model(disc_weight_path)
        self._load_image_processor(processor_model_name_or_path)
        self.dtype = next(self.continuous_vision_model.parameters()).dtype

    def _load_continuous_vision_model(self, config_path: str, weight_path: str) -> None:
        """Load continuous vision model (Qwen2.5-VL)."""
        logger.info("Loading continuous vision model...")

        # Validate weight file exists
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Vision model weight file not found: {weight_path}")

        # Load config
        with open(config_path, "r") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)

        # Normalize model type
        if config_dict.get("model_type") == "qwen2_5_vl":
            config_dict["model_type"] = "qwen2_5_vl_visual"

        # Initialize model from config
        if config_dict.get("model_type") == "qwen2_5_vl_visual":
            self.continuous_config = Qwen2_5_VLVisionConfig.from_dict(config_dict)
            self.continuous_config._attn_implementation = "flash_attention_2"
            self.continuous_vision_model = AutoModel.from_config(self.continuous_config, trust_remote_code=True)
        else:
            raise ValueError(f"Unsupported model type: {config_dict.get('model_type')}")

        # Load weights
        logger.info(f"Loading vision model weights: {weight_path}")
        state_dict = torch.load(weight_path, map_location="cpu")
        self.continuous_vision_model.load_state_dict(state_dict, strict=True)
        # FlashAttention은 fp16/bf16만 지원하므로 bfloat16으로 변환
        self.continuous_vision_model = self.continuous_vision_model.to(device=self.device, dtype=torch.bfloat16)
        self.continuous_vision_model.eval()

        # 최종 확인: 첫 번째 파라미터 dtype 확인
        first_param_dtype = next(self.continuous_vision_model.parameters()).dtype
        logger.info(f"Continuous vision model loaded with dtype: {first_param_dtype}")

    def _load_discrete_vision_model(self, weight_path: str) -> None:
        """Load discrete vision model (TextAlignedTokenizer)."""
        logger.info("Loading discrete vision model...")

        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Discrete vision model weight file not found: {weight_path}")

        logger.info(f"Loading discrete vision model weights: {weight_path}")
        self.discrete_vision_model = TextAlignedTokenizer.from_checkpoint(
            weight_path, load_teacher=False, input_type="indices"
        )
        self.discrete_vision_model = self.discrete_vision_model.to(self.device)
        self.discrete_vision_model.eval()
        logger.info("Discrete vision model loaded")

    def _load_image_processor(self, processor_path: str) -> None:
        """Load HCXVisionV2Processor (image/video only, no tokenizer needed)"""
        logger.info("Loading image processor...")

        # Load config file
        processor_path = os.path.abspath(processor_path)
        with open(processor_path, "r") as f:
            processor_config = json.load(f)

        # Extract image processor config (remove processor-level keys)
        image_processor_config = {
            k: v
            for k, v in processor_config.items()
            if k not in ["auto_map", "processor_class", "image_processor_type"]
        }

        # Initialize Qwen2VLImageProcessor from config
        image_processor = Qwen2VLImageProcessor(**image_processor_config)

        # video_processor도 동일한 config를 사용하여 초기화
        # AutoVideoProcessor를 사용하여 video_processor 로드 시도
        video_processor = AutoVideoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        logger.info("Video processor loaded from config")

        # Initialize HCXVisionV2Processor with image_processor and video_processor (no tokenizer)
        # This allows image/video-only processing without text
        self.processor = HCXVisionV2Processor(
            image_processor=image_processor,
            tokenizer=None,  # 이미지/비디오만 처리하므로 tokenizer 불필요
            video_processor=video_processor,
        )
        logger.info("Processor (HCXVisionV2Processor) loaded")
        logger.info(f"Processor: {self.processor}")

    def _load_mm_projector(self, weight_path: Optional[str]) -> None:
        """Load MM projector that maps vision features to LLM hidden dimension."""
        # Get input hidden size from vision config
        input_hidden_size = (
            self.continuous_config.out_hidden_size
            if hasattr(self.continuous_config, "out_hidden_size")
            else self.continuous_config.hidden_size
        )

        # Output hidden size should match LLM dimension
        output_hidden_size = self.llm_hidden_size

        logger.info(f"Loading mm projector: {input_hidden_size} -> {output_hidden_size} (LLM dim)")
        logger.info(f"MM projector weights path: {weight_path}")

        # Initialize and load projector
        self.mm_projector = nn.Linear(input_hidden_size, output_hidden_size)
        state_dict = torch.load(weight_path, map_location="cpu")
        self.mm_projector.load_state_dict(state_dict, strict=True)
        self.mm_projector = self.mm_projector.to(self.device)
        self.mm_projector.eval()
        logger.info("MM projector loaded successfully")

    def _download_media_from_s3(self, url: urllib.parse.ParseResult, timeout: int = 60) -> io.BytesIO:
        bucket = url.netloc
        key = url.path.lstrip("/")
        logger.info(f"[S3] _download_media_from_s3 시작 - bucket={bucket}, key={key}")
        try:
            logger.info(f"[S3] get_sync_client().get_object 호출 - Bucket={bucket}, Key={key}")
            resp = self.s3_conn.get_sync_client().get_object(Bucket=bucket, Key=key)
            logger.info(f"[S3] get_object 성공 - ContentLength={resp.get('ContentLength', 'N/A')}, ContentType={resp.get('ContentType', 'N/A')}")
            # S3 get_object returns StreamingBody which is not seekable.
            # Most media libraries require seekable objects,
            # so we need to read the content into memory and wrap it in BytesIO.
            body_data = resp["Body"].read()
            logger.info(f"[S3] Body 읽기 완료 - size={len(body_data) / 1024 / 1024:.2f} MB")
            return io.BytesIO(body_data)
        except Exception as e:
            logger.error(f"[S3] _download_media_from_s3 예외 발생 - bucket={bucket}, key={key}, error={e}", exc_info=True)
            raise

    def _download_media_from_http(self, url: urllib.parse.ParseResult, timeout: int = 60) -> io.BytesIO:
        logger.info(f"[S3] _download_media_from_http 시작 - url={url.geturl()}")
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        try:
            logger.info(f"[S3] HTTP GET 요청 - url={url.geturl()}, timeout={timeout}")
            response = requests.get(url.geturl(), timeout=timeout, stream=True, headers=headers, allow_redirects=True)
            logger.info(f"[S3] HTTP 응답 수신 - status_code={response.status_code}, content_length={response.headers.get('Content-Length', 'N/A')}")

            try:
                response.raise_for_status()
            except Exception as e:
                logger.warning(f"[S3] HTTP 응답 오류 - status_code={response.status_code}, error={e}")
                if response.status_code == 404:
                    raise ValueError(f"File not found: {url}")
                elif response.status_code == 403:
                    # Check if URL is from S3-compatible object storage
                    s3_endpoint = os.getenv("NCP_S3_ENDPOINT", "")
                    endpoint_host = s3_endpoint.replace("https://", "").replace("http://", "") if s3_endpoint else ""
                    logger.info(f"[S3] HTTP 403 - S3 폴백 시도 - s3_endpoint={s3_endpoint}, endpoint_host={endpoint_host}, url.netloc={url.netloc}")
                    if endpoint_host and endpoint_host in url.netloc:
                        path = url.path.lstrip("/").split("/")
                        bucket_name = path[0]
                        key = "/".join(path[1:])
                        s3_url = urllib.parse.urlparse(f"s3://{bucket_name}/{key}")
                        logger.info(f"[S3] S3 폴백 URL 생성 - s3_url={s3_url}")
                        return self._download_media_from_s3(s3_url)
                else:
                    raise e

            content_size = len(response.content)
            logger.info(f"[S3] _download_media_from_http 완료 - content_size={content_size / 1024 / 1024:.2f} MB")
            return io.BytesIO(response.content)
        except Exception as e:
            logger.error(f"[S3] _download_media_from_http 예외 발생 - url={url.geturl()}, error={e}", exc_info=True)
            raise

    def _download_media_from_url(self, url: str, timeout: int = 60) -> io.BytesIO:
        """URL에서 비디오 다운로드"""
        logger.info(f"[S3] _download_media_from_url 시작 - url={url}")
        try:
            url_parsed = urllib.parse.urlparse(url)
            logger.info(f"[S3] URL 파싱 완료 - scheme={url_parsed.scheme}, netloc={url_parsed.netloc}, path={url_parsed.path}")

            if url_parsed.scheme == "s3":
                logger.info(f"[S3] S3 스킴 감지 - S3에서 다운로드")
                return self._download_media_from_s3(url_parsed)

            if url_parsed.scheme == "http" or url_parsed.scheme == "https":
                logger.info(f"[S3] HTTP/HTTPS 스킴 감지 - HTTP에서 다운로드")
                return self._download_media_from_http(url_parsed)

            logger.error(f"[S3] 지원되지 않는 URL 스킴 - scheme={url_parsed.scheme}")
            raise ValueError(f"Unsupported URL scheme: {url_parsed.scheme}")

        except ValueError as e:
            logger.error(f"[S3] _download_media_from_url ValueError 발생 - url={url}, error={e}", exc_info=True)
            raise ValueError(f"Invalid URL: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"[S3] 미디어 다운로드 실패 (URL: {url}): {e}", exc_info=True)
            raise

    def _load_image_from_base64(self, base64_str: str) -> Image.Image:
        """Base64 문자열에서 이미지 로드"""
        try:
            img_bytes = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_bytes))

            # RGBA 이미지 처리 개선 (알파 채널을 고려한 RGB 변환)
            if img.mode == "RGBA":
                # 흰색 배경에 알파 채널을 고려하여 합성
                white_background = Image.new("RGB", img.size, (255, 255, 255))
                white_background.paste(img, mask=img.split()[3])  # 알파 채널을 마스크로 사용
                img = white_background
            elif img.mode != "RGB":
                img = img.convert("RGB")
            return img

        except Exception as e:
            logger.error(f"Base64 이미지 디코딩 실패: {e}")
            raise

    def _load_video_from_base64(self, base64_str: str) -> io.BytesIO:
        """Base64 문자열에서 비디오 로드"""
        try:
            video_bytes = base64.b64decode(base64_str)
            return io.BytesIO(video_bytes)
        except Exception as e:
            logger.error(f"Base64 비디오 디코딩 실패: {e}")
            raise

    def _check_cache_exists(self, storage_path: str, storage_name: str = S3_BUCKET_NAME) -> bool:
        """s3에서 캐시 파일 존재 여부만 확인 (빠른 확인용)"""
        logger.info(f"[S3] _check_cache_exists 시작 - storage_name={storage_name}, storage_path={storage_path}")
        try:
            # list_objects를 사용하여 파일 존재 여부 확인
            logger.info(f"[S3] list_objects 호출 - storage_name={storage_name}, prefix={storage_path}, max_keys=1")
            objects = self.s3_conn.list_objects(
                storage_name=storage_name,
                prefix=storage_path,
                max_keys=1
            )
            logger.info(f"[S3] list_objects 결과 - objects={objects}")
            # 정확히 일치하는 키가 있는지 확인
            exists = any(obj.get("Key") == storage_path for obj in objects)
            logger.info(f"[S3] _check_cache_exists 완료 - exists={exists}")
            return exists
        except Exception as e:
            logger.error(f"[S3] 캐시 존재 확인 실패 - storage_name={storage_name}, storage_path={storage_path}, error={e}", exc_info=True)
            return False

    def _get_from_cache(self, storage_path: str, storage_name: str = S3_BUCKET_NAME) -> Optional[Dict]:
        """s3에서 캐시된 결과 가져오기"""
        logger.info(f"[S3] _get_from_cache 시작 - storage_name={storage_name}, storage_path={storage_path}")

        # 파일 확장자에 따라 읽기 방식 결정
        is_tensor_file = storage_path.endswith(".pt")
        suffix = ".pt" if is_tensor_file else ".json"
        logger.info(f"[S3] 파일 타입 - is_tensor_file={is_tensor_file}, suffix={suffix}")

        # 임시 파일로 다운로드
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_path = tmp_file.name
        logger.info(f"[S3] 임시 파일 생성 - tmp_path={tmp_path}")

        try:
            logger.info(f"[S3] download_file 호출 - storage_name={storage_name}, storage_path={storage_path}, local_path={tmp_path}")
            success = self.s3_conn.download_file(
                storage_name=storage_name, local_path=tmp_path, storage_path=storage_path
            )
            logger.info(f"[S3] download_file 결과 - success={success}")

            if success and os.path.exists(tmp_path):
                file_size = os.path.getsize(tmp_path)
                logger.info(f"[S3] 다운로드 완료 - file_size={file_size / 1024 / 1024:.2f} MB")
                if is_tensor_file:
                    data = torch.load(tmp_path, map_location="cpu")
                else:
                    with open(tmp_path, "r") as f:
                        data = json.load(f)
                os.unlink(tmp_path)  # 임시 파일 삭제
                logger.info(f"[S3] _get_from_cache 완료 - 데이터 로드 성공")
                return data

            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            logger.warning(f"[S3] _get_from_cache 실패 - 파일 다운로드 실패 또는 파일 없음")
            return None
        except Exception as e:
            logger.error(f"[S3] _get_from_cache 예외 발생 - storage_name={storage_name}, storage_path={storage_path}, error={e}", exc_info=True)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _save_to_cache(self, storage_path: str, data: Any, storage_name: str = S3_BUCKET_NAME, is_tensor: bool = False):
        """s3에 결과 캐시

        Args:
            storage_path: s3 스토리지 경로 (key)
            data: 저장할 데이터 (Dict 또는 텐서)
            storage_name: s3 스토리지 네임스페이스 (환경변수 WBL_S3_BUCKET_NAME 참조)
            is_tensor: 텐서 데이터인지 여부 (텐서인 경우 .pt 파일로 저장)
        """
        logger.info(f"[S3] _save_to_cache 시작 - storage_name={storage_name}, storage_path={storage_path}, is_tensor={is_tensor}")
        
        # 임시 파일로 저장
        suffix = ".pt" if is_tensor else ".json"
        file_mode = "wb" if is_tensor else "w"
        logger.info(f"[S3] 임시 파일 생성 준비 - suffix={suffix}, file_mode={file_mode}")
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode=file_mode) as tmp_file:
                tmp_path = tmp_file.name
                logger.info(f"[S3] 임시 파일 생성 - tmp_path={tmp_path}")
                if is_tensor:
                    torch.save(data, tmp_path)
                    logger.info(f"[S3] 텐서 데이터 저장 완료")
                else:
                    json.dump(data, tmp_file)
                    logger.info(f"[S3] JSON 데이터 저장 완료")
                    
                # file 크기 출력 (MB 단위)
                file_size = os.path.getsize(tmp_path)
                logger.info(f"[S3] 임시 파일 크기: {file_size / 1024 / 1024:.2f} MB")

            # s3에 업로드
            logger.info(f"[S3] upload_wbl_asset 호출 - file_path={tmp_path}, key={storage_path}")
            s3_key = self.s3_conn.upload_wbl_asset(file_path=Path(tmp_path), key=storage_path)
            logger.info(f"[S3] upload_wbl_asset 완료 - 반환된 s3_key={s3_key}")

            # 임시 파일 삭제
            os.unlink(tmp_path)
            logger.info(f"[S3] 임시 파일 삭제 완료")

            logger.info(f"[S3] _save_to_cache 완료: {storage_name}/{storage_path} -> {s3_key}")
        except Exception as e:
            logger.error(f"[S3] _save_to_cache 예외 발생 - storage_name={storage_name}, storage_path={storage_path}, error={e}", exc_info=True)
            # 임시 파일이 존재하면 삭제
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def _delete_from_cache(self, storage_path: str, storage_name: str = S3_BUCKET_NAME):
        """s3에서 캐시 삭제

        Args:
            storage_path: s3 스토리지 경로 (key)
            storage_name: s3 스토리지 네임스페이스 (환경변수 WBL_S3_BUCKET_NAME 참조)
        """
        logger.info(f"[S3] _delete_from_cache 시작 - storage_name={storage_name}, storage_path={storage_path}")
        try:
            logger.info(f"[S3] delete_object 호출 - storage_name={storage_name}, object_key={storage_path}")
            success = self.s3_conn.delete_object(storage_name=storage_name, object_key=storage_path)
            logger.info(f"[S3] delete_object 결과 - success={success}")
            if success:
                logger.info(f"[S3] 캐시 삭제 완료: {storage_path}")
            else:
                logger.warning(f"[S3] 캐시 삭제 실패 (존재하지 않음): {storage_path}")
        except Exception as e:
            logger.error(f"[S3] _delete_from_cache 예외 발생 - storage_name={storage_name}, storage_path={storage_path}, error={e}", exc_info=True)
            raise

    def _process_image(
        self,
        image: Image.Image,
    ) -> VisionEncoderResult:
        """
        단일 이미지 처리

        Args:
            image: PIL Image
            anyres: anyres 기능 사용 여부
            unpad: unpad 기능 사용 여부
            num_queries_vis_abstractor: visual abstractor query 수
            possible_resolutions: 가능한 resolution 리스트

        Returns:
            VisionEncoderResult
        """
        # continuous vision model forward
        processed = self.processor(images=[image], return_tensors="pt")
        pixel_values_tensor = processed.pixel_values
        logger.info(f"pixel_values_tensor dtype: {pixel_values_tensor.dtype}, shape: {pixel_values_tensor.shape}")

        image_grid_thw = processed.image_grid_thw.to(self.device)
        image_sizes = [{"width": image.width, "height": image.height}]
        pixel_values_batch = pixel_values_tensor.to(device=self.device, dtype=self.dtype)
        logger.info(
            f"Before forward: pixel_values_batch shape={pixel_values_batch.shape}, dtype={pixel_values_batch.dtype}, device={pixel_values_batch.device}"
        )
        logger.info(
            f"image_grid_thw shape={image_grid_thw.shape}, dtype={image_grid_thw.dtype}, device={image_grid_thw.device}"
        )

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype, enabled=True):
            continuous_output = self.continuous_vision_model(pixel_values_batch, grid_thw=image_grid_thw)
            continuous_feature = self.mm_projector(continuous_output)

        # vision_query_length는 continuous_feature의 sequence length (첫 번째 dimension)
        vision_query_length = continuous_feature.shape[0] if len(continuous_feature.shape) > 0 else 0

        # Discrete vision model forward
        resized_image = image.resize((384, 384), Image.BICUBIC)
        discrete_tensor = ToTensor()(resized_image).unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype, enabled=True):
            discrete_output = self.discrete_vision_model(discrete_tensor)
            discrete_tokens = discrete_output["encoded"]

        return VisionEncoderResult(
            continuous_feature=continuous_feature,
            discrete_tokens=discrete_tokens,
            vision_query_length=vision_query_length,
            image_sizes=image_sizes,
            image_grid_thw=image_grid_thw,
        )

    def _process_video(
        self,
        video_bytes: io.BytesIO,
    ) -> VisionEncoderResult:
        """
        비디오 처리

        Args:
            video_bytes: 비디오 바이트 스트림

        Returns:
            VisionEncoderResult
        """
        if fetch_video is None or process_vision_info is None:
            raise RuntimeError("비디오 처리 모듈이 로드되지 않았습니다.")

        video_max_pixels = 378 * 378
        # video_max_pixels = 378 * 378 * 2

        # process_vision_info를 위한 messages 형식 구성
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_bytes,
                            "max_pixels": video_max_pixels,
                        }
                    ],
                }
            ],
        ]

        # process_vision_info로 비디오 처리 (fetch_video 내부 호출)
        _, videos, video_meta = process_vision_info(messages, return_video_kwargs=True)
        video_duration = round(videos[0].shape[0] / video_meta["fps"][0], 2)
        video_meta["video_duration"] = video_duration

        video = videos[0]
        logger.info(f"video dtype: {video.dtype}, shape: {video.shape}")

        # FlashAttention은 fp16/bf16만 지원하므로 dtype 변환
        if video.dtype == torch.float32:
            video = video.to(dtype=self.dtype)
            logger.info(f"video dtype converted to: {video.dtype}")

        # temporal patches를 chunk_temporal_patches씩 나눔
        continuous_feature_list = []
        max_batch = 32
        grid_thw_size = 0
        for t_start in range(0, video.shape[0], max_batch):
            t_end = min(t_start + max_batch, video.shape[0])
            video_chunk = video[t_start:t_end]

            # video_processor로 비디오 전처리
            # video_tensor는 (T, C, H, W) 형태
            # 중요: qwen 2.5 vl의 vit 는 mrope 가 아니라 2d rope 가 들어가기 때문에 이렇게 chunk해도 괜찮음.
            # grid_thw 활용하는 부분을 보면 .repeat(t) 를 수행하고 있음.
            processed = self.processor(videos=video_chunk, return_tensors="pt")
            pixel_values_batch = processed.pixel_values_videos.to(device=self.device, dtype=self.dtype)
            grid_thw_batch = processed.video_grid_thw.to(self.device)
            grid_thw_size += grid_thw_batch.shape[0]

            # logger.info(f"pixel_values_batch dtype: {pixel_values_batch.dtype}, shape: {pixel_values_batch.shape}")
            # logger.info(f"grid_thw_batch: {grid_thw_batch}")

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype, enabled=True):
                continuous_output_chunk = self.continuous_vision_model(pixel_values_batch, grid_thw=grid_thw_batch)
                continuous_feature_chunk = self.mm_projector(continuous_output_chunk)
                continuous_feature_list.append(continuous_feature_chunk)

        # 결과를 concat
        continuous_feature = torch.cat(continuous_feature_list, dim=0)
        grid_thw_total = torch.tensor(
            [[grid_thw_size, grid_thw_batch[0][1].item(), grid_thw_batch[0][2].item()]],
            dtype=grid_thw_batch.dtype,
            device="cpu",
        )

        # vision_query_length는 continuous_feature의 sequence length
        vision_query_length = continuous_feature.shape[0] if len(continuous_feature.shape) > 0 else 0

        # video_sizes 정보 구성 (첫 프레임 크기 사용)
        video_sizes = [{"width": video.shape[3], "height": video.shape[2]}]

        return VisionEncoderResult(
            continuous_feature=continuous_feature,
            vision_query_length=vision_query_length,
            video_grid_thw=grid_thw_total,
            video_sizes=video_sizes,
            video_duration=video_duration,
            video_fps=video_meta["fps"][0],
        )

    def process_image_from_url(
        self,
        image_url: str,
    ) -> Dict[str, str]:
        """
        URL에서 이미지를 다운로드하고 처리

        Returns:
            {"s3_key": str}
        """
        # 캐시 키 생성 (단일 파일)
        safe_url = quote_plus(image_url, safe="")
        cache_key = f"source/derived/embedding/vision/{self.model_id}/image_{safe_url}.pt"

        if self._check_cache_exists(cache_key):
            logger.info(f"캐시에서 결과 조회: {cache_key}")
            return {"s3_key": cache_key}

        # 이미지 다운로드 및 처리
        bytes_data = self._download_media_from_url(image_url)
        image = Image.open(bytes_data)

        if image.mode == "RGBA":
            # 흰색 배경에 알파 채널을 고려하여 합성
            white_background = Image.new("RGB", image.size, (255, 255, 255))
            white_background.paste(image, mask=image.split()[3])  # 알파 채널을 마스크로 사용
            image = white_background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        result = self._process_image(image)

        # 캐시 저장 (단일 딕셔너리로 저장)
        cache_data = {
            "discrete": result.discrete_tokens.detach().cpu(),
            "continuous": result.continuous_feature.detach().cpu(),
            "meta": {
                "vision_query_length": result.vision_query_length,
                "image_sizes": result.image_sizes,
                "image_grid_thw": result.image_grid_thw.detach().cpu(),
            },
        }

        self._save_to_cache(cache_key, cache_data, is_tensor=True)

        return {"s3_key": cache_key}

    def process_image_from_base64(
        self,
        base64_str: str,
    ) -> Dict[str, str]:
        """
        Base64 문자열에서 이미지를 로드하고 처리

        Returns:
            {"s3_key": str}
        """
        # base64 문자열의 해시를 사용하여 캐시 키 생성 (단일 파일)
        base64_hash = hashlib.sha256(base64_str.encode()).hexdigest()[:16]
        cache_key = f"source/derived/embedding/vision/{self.model_id}/image_base64_{base64_hash}.pt"

        if self._check_cache_exists(cache_key):
            logger.info(f"캐시에서 결과 조회: {cache_key}")
            return {"s3_key": cache_key}

        # 이미지 로드 및 처리
        image = self._load_image_from_base64(base64_str)
        result = self._process_image(image)

        # 캐시 저장 (단일 딕셔너리로 저장)
        cache_data = {
            "discrete": result.discrete_tokens.detach().cpu(),
            "continuous": result.continuous_feature.detach().cpu(),
            "meta": {
                "vision_query_length": result.vision_query_length,
                "image_sizes": result.image_sizes,
                "image_grid_thw": result.image_grid_thw.detach().cpu(),
            },
        }

        self._save_to_cache(cache_key, cache_data, is_tensor=True)

        return {"s3_key": cache_key}

    def process_video_from_url(
        self,
        video_url: str,
    ) -> Dict[str, str]:
        """
        URL에서 비디오를 다운로드하고 처리

        Returns:
            {"s3_key": str}
        """
        # 비디오 URL의 해시를 사용하여 캐시 키 생성 (단일 파일)
        safe_url = quote_plus(video_url, safe="")
        cache_key = f"source/derived/embedding/vision/{self.model_id}/video_{safe_url}.pt"

        if self._check_cache_exists(cache_key):
            logger.info(f"캐시에서 결과 조회: {cache_key}")
            return {"s3_key": cache_key}

        # 비디오 다운로드 및 처리
        video_bytes = self._download_media_from_url(video_url)
        result = self._process_video(video_bytes)

        # 캐시 저장 (단일 딕셔너리로 저장, discrete 없음)
        cache_data = {
            "continuous": result.continuous_feature.detach().cpu(),
            "meta": {
                "vision_query_length": result.vision_query_length,
                "video_sizes": result.video_sizes,
                "video_grid_thw": result.video_grid_thw.detach().cpu(),
                "video_duration": result.video_duration,
                "video_fps": result.video_fps,
            },
        }

        self._save_to_cache(cache_key, cache_data, is_tensor=True)

        return {"s3_key": cache_key}

    def process_video_from_base64(
        self,
        base64_str: str,
    ) -> Dict[str, str]:
        """
        Base64 문자열에서 비디오를 로드하고 처리

        Returns:
            {"s3_key": str}
        """
        # base64 문자열의 해시를 사용하여 캐시 키 생성 (단일 파일)
        base64_hash = hashlib.sha256(base64_str.encode()).hexdigest()[:16]
        cache_key = f"source/derived/embedding/vision/{self.model_id}/video_base64_{base64_hash}.pt"

        if self._check_cache_exists(cache_key):
            logger.info(f"캐시에서 결과 조회: {cache_key}")
            return {"s3_key": cache_key}

        # 비디오 로드 및 처리
        video_bytes = self._load_video_from_base64(base64_str)
        result = self._process_video(video_bytes)

        # 캐시 저장 (단일 딕셔너리로 저장, discrete 없음)
        cache_data = {
            "continuous": result.continuous_feature.detach().cpu(),
            "meta": {
                "vision_query_length": result.vision_query_length,
                "video_sizes": result.video_sizes,
                "video_grid_thw": result.video_grid_thw.detach().cpu(),
                "video_duration": result.video_duration,
                "video_fps": result.video_fps,
            },
        }

        self._save_to_cache(cache_key, cache_data, is_tensor=True)

        return {"s3_key": cache_key}
