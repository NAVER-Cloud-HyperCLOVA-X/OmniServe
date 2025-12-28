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
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import requests
import torch
import torch.nn as nn
import yaml
from PIL import Image

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

MODEL_MAPPING.register(Qwen2_5_VLVisionConfig, Qwen2_5_VisionTransformerPretrainedModel)
CONFIG_MAPPING.register("qwen2_5_vl_visual", Qwen2_5_VLVisionConfig)

S3_BUCKET_NAME=os.getenv("WBL_S3_BUCKET_NAME", "")


class VisionEncoderResult:
    """Vision encoder 처리 결과를 담는 클래스"""

    def __init__(
        self,
        continuous_feature: Optional[torch.Tensor] = None,
        vision_query_length: int = 0,
        image_sizes: Optional[List[Dict[str, int]]] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        video_sizes: Optional[List[Dict[str, int]]] = None,
        video_duration: Optional[float] = None,
        video_fps: Optional[float] = None,
    ):
        self.continuous_feature = continuous_feature
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
        cont_config_path: str = "configs/32b_cont_config.yaml",
        cont_weight_path: str = "weights/vision_weight.pt",
        mm_projector_weight_path: str = "weights/mm_projector_weights.pt",
        llm_hidden_size: int = 4096,
        model_id: str = "default",
    ):
        """Initialize Vision Encoder Model.

        Args:
            processor_model_name_or_path: Path to processor config file or directory
            cont_config_path: Path to continuous vision model config YAML
            cont_weight_path: Path to continuous vision model weights
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

        # Initialize s3 connection (optional - server can run without s3)
        self.s3_conn = S3Connection()
        logger.info("S3 connection established")

        # Load models
        self._load_continuous_vision_model(cont_config_path, cont_weight_path)
        self._load_mm_projector(mm_projector_weight_path)
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

    def _download_image_from_url(self, url: str, timeout: int = 30) -> Image.Image:
        """URL에서 이미지 다운로드"""
        try:
            # User-Agent 헤더 추가 (일부 서버에서 필요)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, timeout=timeout, stream=True, headers=headers, allow_redirects=True)
            response.raise_for_status()

            # Content-Type 확인
            content_type = response.headers.get("Content-Type", "").lower()
            if content_type and "text/html" in content_type:
                logger.error(f"이미지가 아닌 HTML 응답을 받았습니다 (URL: {url}, Content-Type: {content_type})")
                raise ValueError(f"Expected image but received HTML content. URL may be incorrect: {url}")

            # 이미지 로드
            img = Image.open(io.BytesIO(response.content))

            # RGBA 이미지 처리 개선 (알파 채널을 고려한 RGB 변환)
            if img.mode == "RGBA":
                # 흰색 배경에 알파 채널을 고려하여 합성
                white_background = Image.new("RGB", img.size, (255, 255, 255))
                white_background.paste(img, mask=img.split()[3])  # 알파 채널을 마스크로 사용
                img = white_background
            elif img.mode != "RGB":
                img = img.convert("RGB")

            return img

        except requests.exceptions.RequestException as e:
            logger.error(f"이미지 다운로드 네트워크 에러 (URL: {url}): {e}")
            raise ValueError(f"Image download network error: {e}")
        except (Image.UnidentifiedImageError, OSError) as e:
            logger.error(f"이미지 파싱 에러 (URL: {url}): {e}")
            raise ValueError(f"Invalid image format or corrupted image data: {e}")
        except Exception as e:
            logger.error(f"이미지 다운로드 실패 (URL: {url}): {type(e).__name__}: {e}")
            raise ValueError(f"Image download failed: {e}")

    def _download_video_from_url(self, url: str, timeout: int = 60) -> io.BytesIO:
        """URL에서 비디오 다운로드"""
        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            return io.BytesIO(response.content)
        except Exception as e:
            logger.error(f"비디오 다운로드 실패 (URL: {url}): {e}")
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
        try:
            objects = self.s3_conn.list_objects(
                storage_name=storage_name,
                prefix=storage_path,
                max_keys=1,
            )
            return any(obj.get("Key") == storage_path for obj in objects)
        except Exception as e:
            logger.warning(f"캐시 존재 확인 실패: {e}")
            return False

    def _get_from_cache(self, storage_path: str, storage_name: str = S3_BUCKET_NAME) -> Optional[Dict]:
        """s3에서 캐시된 결과 가져오기"""

        # 파일 확장자에 따라 읽기 방식 결정
        is_tensor_file = storage_path.endswith(".pt")
        suffix = ".pt" if is_tensor_file else ".json"

        # 임시 파일로 다운로드
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_path = tmp_file.name

        success = self.s3_conn.download_file(
            storage_name=storage_name, local_path=tmp_path, storage_path=storage_path
        )

        if success and os.path.exists(tmp_path):
            if is_tensor_file:
                data = torch.load(tmp_path, map_location="cpu")
            else:
                with open(tmp_path, "r") as f:
                    data = json.load(f)
            os.unlink(tmp_path)  # 임시 파일 삭제
            return data

        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return None

    def _save_to_cache(self, storage_path: str, data: Any, storage_name: str = S3_BUCKET_NAME, is_tensor: bool = False):
        """s3에 결과 캐시

        Args:
            storage_path: s3 스토리지 경로 (key)
            data: 저장할 데이터 (Dict 또는 텐서)
            storage_name: s3 스토리지 네임스페이스 (환경변수 WBL_S3_BUCKET_NAME 참조)
            is_tensor: 텐서 데이터인지 여부 (텐서인 경우 .pt 파일로 저장)
        """
        # 임시 파일로 저장
        suffix = ".pt" if is_tensor else ".json"
        file_mode = "wb" if is_tensor else "w"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode=file_mode) as tmp_file:
            tmp_path = tmp_file.name
            if is_tensor:
                torch.save(data, tmp_path)
            else:
                json.dump(data, tmp_file)

        # s3에 업로드
        s3_key = self.s3_conn.upload_wbl_asset(file_path=Path(tmp_path), key=storage_path)

        # 임시 파일 삭제
        os.unlink(tmp_path)

        logger.info(f"캐시 저장 완료: {storage_path} -> {s3_key}")

    def _delete_from_cache(self, storage_path: str, storage_name: str = S3_BUCKET_NAME):
        """s3에서 캐시 삭제"""
        success = self.s3_conn.delete_object(storage_name=storage_name, object_key=storage_path)
        if success:
            logger.info(f"캐시 삭제 완료: {storage_path}")
        else:
            logger.warning(f"캐시 삭제 실패 (존재하지 않음): {storage_path}")

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

        return VisionEncoderResult(
            continuous_feature=continuous_feature,
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

        safe_url = quote_plus(image_url, safe="")
        cache_key = f"source/derived/embedding/vision/{self.model_id}/image_{safe_url}.pt"
        if self._check_cache_exists(cache_key):
            logger.info(f"캐시에서 결과 조회: {cache_key}")
            return {"s3_key": cache_key}

        # 이미지 다운로드 및 처리
        image = self._download_image_from_url(image_url)
        result = self._process_image(image)

        # 캐시 저장 (단일 파일)
        cache_data = {
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
        # base64 문자열의 해시를 사용하여 캐시 키 생성
        base64_hash = hashlib.sha256(base64_str.encode()).hexdigest()[:16]
        cache_key = f"source/derived/embedding/vision/{self.model_id}/image_base64_{base64_hash}.pt"

        if self._check_cache_exists(cache_key):
            logger.info(f"캐시에서 결과 조회: {cache_key}")
            return {"s3_key": cache_key}

        # 이미지 로드 및 처리
        image = self._load_image_from_base64(base64_str)
        result = self._process_image(image)

        cache_data = {
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
        # 비디오 URL의 해시를 사용하여 캐시 키 생성
        safe_url = quote_plus(video_url, safe="")
        cache_key = f"source/derived/embedding/vision/{self.model_id}/video_{safe_url}.pt"

        if self._check_cache_exists(cache_key):
            logger.info(f"캐시에서 결과 조회: {cache_key}")
            return {"s3_key": cache_key}

        # 비디오 다운로드 및 처리
        video_bytes = self._download_video_from_url(video_url)
        result = self._process_video(video_bytes)

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
        # base64 문자열의 해시를 사용하여 캐시 키 생성
        base64_hash = hashlib.sha256(base64_str.encode()).hexdigest()[:16]
        cache_key = f"source/derived/embedding/vision/{self.model_id}/video_base64_{base64_hash}.pt"

        if self._check_cache_exists(cache_key):
            logger.info(f"캐시에서 결과 조회: {cache_key}")
            return {"s3_key": cache_key}

        # 비디오 로드 및 처리
        video_bytes = self._load_video_from_base64(base64_str)
        result = self._process_video(video_bytes)

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

    def get_vision_query_length(self, image: Image.Image) -> int:
        """Get vision query length for a given image (for liveness probe)."""
        try:
            result = self._process_image(image)
            return result.vision_query_length
        except Exception as e:
            logger.error(f"Failed to get vision query length: {e}")
            return 0
