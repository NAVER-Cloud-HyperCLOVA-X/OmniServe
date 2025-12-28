# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import base64
import hashlib
import io
import json
import logging
import os
import subprocess
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional

import librosa
import numpy as np
import requests
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, WhisperFeatureExtractor
from transformers.models.qwen2_audio import Qwen2AudioEncoderConfig
from wbl_storage_utility.s3_util import S3Connection
from urllib.parse import quote_plus

from .cosyvoice import CosyvoiceEncoder, DEFAULT_SAMPLE_RATE
from .mambamia_videoaudio_compressor import MambaMiaVideoAudioCompressor, MambaMiaVideoAudioCompressorConfig

# Setup logger first
logger = logging.getLogger(__name__)

S3_BUCKET_NAME=os.getenv("WBL_S3_BUCKET_NAME", "")


class AudioEncoderResult:
    """Audio encoder 처리 결과를 담는 클래스"""

    def __init__(
        self,
        discrete_tokens: Optional[torch.Tensor] = None,
        continuous_feature: Optional[torch.Tensor] = None,
        audio_length: Optional[float] = None,
        sample_rate: Optional[int] = None,
    ):
        self.discrete_tokens = discrete_tokens
        self.continuous_feature = continuous_feature
        self.audio_length = audio_length
        self.sample_rate = sample_rate


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
    """Audio Encoder Model for processing audio and extracting audio from video."""

    def __init__(
        self,
        continuous_audio_config_path: str = "app/model/configs/audio_config.json",
        continuous_audio_weight_path: str = "app/model/weights/audio_weights.pt",
        discrete_audio_config: Optional[Dict] = None,
        discrete_audio_weight_path: str = "app/model/weights/discrete_audio_weights.pt",
        audio_processor_path: str = "app/model/weights/qwen2-audio-encoder-from-qwen2-audio-7b-instruct",
        audio_projector_weight_path: str = "app/model/weights/audio_projector_weights.pt",
        video_audio_compressor_weight_path: str = "app/model/weights/video_audio_compressor_weights.pt",
        llm_hidden_size: int = 4096,
        model_id: str = "default",
    ):
        """Initialize Audio Encoder Model.

        Args:
            continuous_audio_config_path: Path to continuous audio encoder config JSON
            continuous_audio_weight_path: Path to continuous audio encoder model weights
            discrete_audio_config: Discrete audio config dict
            discrete_audio_weight_path: Path to discrete audio encoder model weights
            audio_processor_path: Path to audio processor (WhisperFeatureExtractor)
            audio_projector_weight_path: Path to audio projector weights
            video_audio_compressor_weight_path: Path to video audio compressor weights
            llm_hidden_size: Hidden dimension of the LLM (default: 4096)
            model_id: Model ID for cache key generation
        """
        # Setup device and dtype
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llm_hidden_size = llm_hidden_size
        self.model_id = model_id
        self.sample_rate = DEFAULT_SAMPLE_RATE

        # Convert relative paths to absolute paths
        # Get the base directory (assuming this file is in app/model/)
        base_dir = Path(__file__).parent.parent.parent  # Go up from app/model/__init__.py to project root

        # Resolve all paths
        continuous_audio_config_path = resolve_path(base_dir, continuous_audio_config_path)
        continuous_audio_weight_path = resolve_path(base_dir, continuous_audio_weight_path)
        discrete_audio_weight_path = resolve_path(base_dir, discrete_audio_weight_path)
        audio_processor_path = resolve_path(base_dir, audio_processor_path)
        audio_projector_weight_path = resolve_path(base_dir, audio_projector_weight_path)
        video_audio_compressor_weight_path = resolve_path(base_dir, video_audio_compressor_weight_path)

        # Initialize s3 connection (optional - server can run without s3)
        self.s3_conn = S3Connection()
        logger.info("S3 connection established")

        # Load models
        self._load_continuous_audio_encoder(continuous_audio_config_path, continuous_audio_weight_path)
        self._load_discrete_audio_encoder(discrete_audio_config, discrete_audio_weight_path)
        self._load_audio_processor(audio_processor_path)
        
        # Load audio projector (needs audio_config.d_model from continuous_audio_config)
        with open(continuous_audio_config_path, "r") as f:
            audio_config_dict = json.load(f)
        audio_d_model = audio_config_dict.get("d_model", 1280)
        self._load_audio_projector(audio_d_model, llm_hidden_size, audio_projector_weight_path)
        self._load_video_audio_compressor(llm_hidden_size, video_audio_compressor_weight_path)
        
        self.dtype = torch.bfloat16

    def _load_continuous_audio_encoder(self, config_path: str, weight_path: str) -> None:
        """Load continuous audio encoder model (Qwen2AudioEncoder)."""
        logger.info("Loading continuous audio encoder model...")

        # Load config
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # Initialize config object from dict
        audio_config = Qwen2AudioEncoderConfig.from_dict(config_dict)
        audio_config._attn_implementation = "flash_attention_2"
        self.continuous_audio_model = AutoModel.from_config(audio_config, trust_remote_code=True)

        # Load weights
        logger.info(f"Loading continuous audio encoder weights: {weight_path}")
        state_dict = torch.load(weight_path, map_location="cpu")
        
        self.continuous_audio_model.load_state_dict(state_dict, strict=True)
        self.continuous_audio_model = self.continuous_audio_model.to(self.device).to(torch.bfloat16)
        self.continuous_audio_model.eval()

        # 최종 확인: 첫 번째 파라미터 dtype 확인
        first_param_dtype = next(self.continuous_audio_model.parameters()).dtype
        logger.info(f"Continuous audio encoder model loaded with dtype: {first_param_dtype}")

    def _load_discrete_audio_encoder(self, discrete_audio_config: Optional[Dict], weight_path: str) -> None:
        """Load discrete audio encoder model (CosyvoiceEncoder)."""
        logger.info("Loading discrete audio encoder model...")

        logger.info(f"Loading discrete audio encoder weights: {weight_path}")
        self.discrete_audio_model = CosyvoiceEncoder.from_pretrained(weight_path)
        self.discrete_audio_model = self.discrete_audio_model.to(self.device).to(torch.bfloat16)
        self.discrete_audio_model.eval()
        
        # 최종 확인: 첫 번째 파라미터 dtype 확인
        first_param_dtype = next(self.discrete_audio_model.parameters()).dtype
        logger.info(f"Discrete audio encoder model loaded with dtype: {first_param_dtype}")

    def _load_audio_processor(self, processor_path: str) -> None:
        """Load audio processor (WhisperFeatureExtractor) for mel spectrogram generation."""
        logger.info(f"Loading audio processor from {processor_path}...")
        # qwen2audioencoder는 WhisperFeatureExtractor를 사용
        self.audio_processor = WhisperFeatureExtractor.from_pretrained(processor_path)
        logger.info("Audio processor loaded")

    def _load_audio_projector(
        self, 
        audio_d_model: int, 
        llm_hidden_size: int, 
        weight_path: str
    ) -> None:
        """Load audio projector (VLM_Mlp) for projecting audio features to LLM hidden size."""
        logger.info("Loading audio projector...")
        
        # Initialize audio projector
        self.audio_projector = VLM_Mlp(
            audio_d_model,
            hidden_features=audio_d_model,
            out_features=llm_hidden_size,
        )
        
        # Load weights
        logger.info(f"Loading audio projector weights: {weight_path}")
        state_dict = torch.load(weight_path, map_location="cpu")
        self.audio_projector.load_state_dict(state_dict, strict=True)
        self.audio_projector = self.audio_projector.to(self.device).to(torch.bfloat16)
        self.audio_projector.eval()
        
        # 최종 확인: 첫 번째 파라미터 dtype 확인
        first_param_dtype = next(self.audio_projector.parameters()).dtype
        logger.info(f"Audio projector loaded with dtype: {first_param_dtype}")


    def _load_video_audio_compressor(
        self, 
        llm_hidden_size: int, 
        weight_path: str
    ) -> None:
        """Load video audio compressor (MambaMiaVideoAudioCompressor)."""
        # 하위 호환: weight 파일이 없으면 compressor를 None으로 설정
        if not os.path.exists(weight_path):
            logger.warning(f"Video audio compressor weights not found: {weight_path}. Skipping compressor loading.")
            self.video_audio_compressor = None
            return

        logger.info("Loading video audio compressor...")
        
        # Initialize compressor config
        compressor_config = MambaMiaVideoAudioCompressorConfig(
            input_size=llm_hidden_size,
            output_size=llm_hidden_size,
            chunk_size=25,
            num_hidden_layers=1,
        )
        self.video_audio_compressor = MambaMiaVideoAudioCompressor(compressor_config)
        
        # Load weights
        logger.info(f"Loading video audio compressor weights: {weight_path}")
        state_dict = torch.load(weight_path, map_location="cpu")
        self.video_audio_compressor.load_state_dict(state_dict, strict=True)
        self.video_audio_compressor = self.video_audio_compressor.to(self.device).to(torch.bfloat16)
        self.video_audio_compressor.eval()
        
        # 최종 확인: 첫 번째 파라미터 dtype 확인
        first_param_dtype = next(self.video_audio_compressor.parameters()).dtype
        logger.info(f"Video audio compressor loaded with dtype: {first_param_dtype}")

    def _extract_audio_from_video(self, video_path_or_bytes: str | io.BytesIO) -> torch.Tensor:
        """비디오에서 오디오 추출

        Args:
            video_path_or_bytes: 비디오 파일 경로 또는 BytesIO 객체

        Returns:
            torch.Tensor: 오디오 파형 (1D tensor, sample_rate=16000)
        """
        # 임시 파일로 저장 (BytesIO인 경우)
        is_bytesio = isinstance(video_path_or_bytes, io.BytesIO)
        tmp_path = None
        audio_tmp_path = None
        
        if is_bytesio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_path = tmp_file.name
                tmp_file.write(video_path_or_bytes.getvalue())
            video_path = tmp_path
        else:
            video_path = video_path_or_bytes

        # ffmpeg가 있는지 확인
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True, timeout=5)
        
        # 임시 오디오 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_tmp_file:
            audio_tmp_path = audio_tmp_file.name
        
        # ffmpeg를 사용하여 오디오 추출
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vn",  # 비디오 스트림 제거
            "-acodec", "pcm_s16le",  # PCM 16-bit little-endian
            "-ac", "1",  # 모노
            "-ar", str(self.sample_rate),  # 샘플 레이트
            "-f", "wav",  # WAV 포맷
            "-y",  # 덮어쓰기
            audio_tmp_path
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5분 타임아웃
        )
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
        
        # 추출된 오디오 파일을 librosa로 로드
        audio, sr = librosa.load(audio_tmp_path, sr=self.sample_rate, mono=True)
        logger.info(f"Successfully extracted audio from video using ffmpeg: {len(audio)} samples at {sr}Hz")

        # torch tensor로 변환
        audio_tensor = torch.from_numpy(audio).float()
        
        # 임시 파일 정리
        if is_bytesio and tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        if audio_tmp_path and os.path.exists(audio_tmp_path):
            os.unlink(audio_tmp_path)
        
        return audio_tensor

    def _download_media_from_s3(self, url: urllib.parse.ParseResult, timeout: int = 60) -> io.BytesIO:
        resp = self.s3_conn.get_sync_client().get_object(Bucket=url.netloc, Key=url.path.lstrip("/"))
        # S3 get_object returns StreamingBody which is not seekable.
        # Most audio libraries (librosa, torchaudio) require seekable objects,
        # so we need to read the content into memory and wrap it in BytesIO.
        return io.BytesIO(resp["Body"].read())

    def _download_media_from_http(self, url: urllib.parse.ParseResult, timeout: int = 60) -> io.BytesIO:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url.geturl(), timeout=timeout, stream=True, headers=headers, allow_redirects=True)

        try:
            response.raise_for_status()
        except Exception as e:
            if response.status_code == 404:
                raise ValueError(f"File not found: {url}")
            elif response.status_code == 403:
                # Check if URL is from S3-compatible object storage
                s3_endpoint = os.getenv("NCP_S3_ENDPOINT", "")
                endpoint_host = s3_endpoint.replace("https://", "").replace("http://", "") if s3_endpoint else ""
                if endpoint_host and endpoint_host in url.netloc:
                    path = url.path.lstrip("/").split("/")
                    bucket_name = path[0]
                    key = "/".join(path[1:])
                    s3_url = urllib.parse.urlparse(f"s3://{bucket_name}/{key}")
                    logger.info(f"Trying to get metadata from S3: {s3_url}")
                    return self._download_media_from_s3(s3_url)
            else:
                raise e

        return io.BytesIO(response.content)

    def _download_media_from_url(self, url: str, timeout: int = 60) -> io.BytesIO:
        """URL에서 미디어 다운로드"""
        try:
            url_parsed = urllib.parse.urlparse(url)
        
            if url_parsed.scheme == "s3":
                return self._download_media_from_s3(url_parsed)

            elif url_parsed.scheme == "http" or url_parsed.scheme == "https":
                return self._download_media_from_http(url_parsed)
            else:
                raise ValueError(f"Invalid scheme: {url}")
        except ValueError as e:
            raise ValueError(f"Invalid URL: {e}")
        except requests.exceptions.RequestException as e:
            raise e

    def _load_audio_from_base64(self, base64_str: str) -> torch.Tensor:
        """Base64 문자열에서 오디오 로드"""
        # data URL 형식 처리
        base64_str = base64_str.split(",", 1)[1] if base64_str.startswith("data:") and "," in base64_str else base64_str

        audio_bytes = base64.b64decode(base64_str)
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=self.sample_rate, mono=True)
        audio_tensor = torch.from_numpy(audio).float()
        return audio_tensor

    def _load_audio_from_bytes(self, audio_bytes: io.BytesIO) -> torch.Tensor:
        """BytesIO에서 오디오 로드"""
        audio, sr = librosa.load(audio_bytes, sr=self.sample_rate, mono=True)
        audio_tensor = torch.from_numpy(audio).float()
        return audio_tensor

    def _check_cache_exists(self, storage_path: str, storage_name: str = S3_BUCKET_NAME) -> bool:
        """s3에서 캐시 파일 존재 여부만 확인 (빠른 확인용)"""
        try:
            # list_objects를 사용하여 파일 존재 여부 확인
            objects = self.s3_conn.list_objects(
                storage_name=storage_name,
                prefix=storage_path,
                max_keys=1
            )
            # 정확히 일치하는 키가 있는지 확인
            return any(obj.get("Key") == storage_path for obj in objects)
        except Exception as e:
            logger.warning(f"캐시 존재 확인 실패: {e}")
            return False

    def _get_from_cache(self, storage_path: str, storage_name: str = S3_BUCKET_NAME) -> Optional[Dict]:
        """s3에서 캐시된 결과 가져오기"""

        # 임시 파일로 다운로드
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as tmp_file:
            tmp_path = tmp_file.name

        success = self.s3_conn.download_file(
            storage_name=storage_name, local_path=tmp_path, storage_path=storage_path
        )

        if success and os.path.exists(tmp_path):
            data = torch.load(tmp_path, map_location="cpu")
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
                
            # file 크기 출력 (MB 단위)
            file_size = os.path.getsize(tmp_path)
            logger.info(f"File size: {file_size / 1024 / 1024} MB")

        # s3에 업로드
        s3_key = self.s3_conn.upload_wbl_asset(file_path=Path(tmp_path), key=storage_path)

        # 임시 파일 삭제
        os.unlink(tmp_path)

        logger.info(f"캐시 저장 완료: {storage_path} -> {s3_key}")

    def _delete_from_cache(self, storage_path: str, storage_name: str = S3_BUCKET_NAME):
        """s3에서 캐시 삭제

        Args:
            storage_path: s3 스토리지 경로 (key)
            storage_name: s3 스토리지 네임스페이스 (환경변수 WBL_S3_BUCKET_NAME 참조)
        """
        success = self.s3_conn.delete_object(storage_name=storage_name, object_key=storage_path)
        if success:
            logger.info(f"캐시 삭제 완료: {storage_path}")
        else:
            logger.warning(f"캐시 삭제 실패 (존재하지 않음): {storage_path}")

    def _compress_video_audio(self, continuous_feature: torch.Tensor) -> torch.Tensor:
        """비디오 오디오 feature를 MambaMia compressor로 압축

        Args:
            continuous_feature: 오디오 feature (T_total, hidden_size)

        Returns:
            torch.Tensor: 압축된 feature (num_queries, hidden_size)
        """
        if self.video_audio_compressor is None:
            return continuous_feature

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype, enabled=True):
            seq_len = continuous_feature.shape[0]
            
            # 배치 차원 추가: (1, T_total, hidden_size)
            batched_features = continuous_feature.unsqueeze(0)
            
            # Compressor 호출 (내부에서 자동 패딩 및 압축)
            compressed_batch = self.video_audio_compressor(batched_features)  # (1, n_chunk, hidden_size)
            
            # 배치 차원 제거
            compressed_feature = compressed_batch.squeeze(0)  # (n_chunk, hidden_size)
            
            logger.info(f"Video audio compression: {seq_len} -> {compressed_feature.shape[0]} tokens (chunk_size={self.video_audio_compressor.chunk_size})")
            
            return compressed_feature

    def _process_audio(self, audio_tensor: torch.Tensor) -> AudioEncoderResult:
        """
        오디오 처리

        Args:
            audio_tensor: 오디오 파형 (1D tensor)

        Returns:
            AudioEncoderResult
        """
        # 오디오 길이 계산
        audio_length = audio_tensor.shape[-1] / self.sample_rate

        # 오디오를 numpy array로 변환 (processor 입력용)
        audio_np = audio_tensor.cpu().numpy()

        # Continuous audio encoder: mel spectrogram 생성 및 처리
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype, enabled=True):
            # WhisperFeatureExtractor로 mel spectrogram 생성
            # 30초 단위로 chunking (qwen2audioencoder의 max_length가 30초)
            chunk_size = 30 * self.sample_rate
            chunks = []
            for i in range(0, len(audio_np), chunk_size):
                chunks.append(audio_np[i : i + chunk_size])

            # 각 chunk를 mel spectrogram으로 변환
            processed_chunks = self.audio_processor(
                chunks,
                sampling_rate=self.sample_rate,
                return_attention_mask=True,
                padding="max_length",
            )
            # input_features: (num_chunks, 128, 3000), attention_mask: (num_chunks, 3000)
            # preprocessor.py와 동일하게 numpy array를 torch.from_numpy로 변환 (기본적으로 float32)
            mel_spectrograms = torch.from_numpy(processed_chunks.input_features).to(device=self.device)
            attention_masks = torch.from_numpy(processed_chunks.attention_mask).to(device=self.device)

            # qwen2audioencoder forward
            # mel_spectrograms shape: (num_chunks, n_mels, max_mel_seq_len) = (num_chunks, 128, 3000)
            batch_size, num_mel_bins, max_mel_seq_len = mel_spectrograms.shape
            assert max_mel_seq_len == 3000, f"max_mel_seq_len should be 3000, but got {max_mel_seq_len}"
            audio_feat_lengths, audio_output_lengths = self.continuous_audio_model._get_feat_extract_output_lengths(
                attention_masks.sum(-1)
            )
            max_seq_len = (max_mel_seq_len - 2) // 2 + 1
            
            # Create 1D attention mask for Flash Attention 2
            # Flash Attention 2 expects 1D attention_mask: (batch_size, seq_len)
            # where 1 means valid token and 0 means padding
            seq_range = (
                torch.arange(0, max_seq_len, dtype=audio_feat_lengths.dtype, device=audio_feat_lengths.device)
                .unsqueeze(0)
                .expand(batch_size, max_seq_len)
            )
            lengths_expand = audio_feat_lengths.unsqueeze(1).expand(batch_size, max_seq_len)
            # Create 1D mask: 1 for valid positions, 0 for padding
            audio_masks_1d = (seq_range < lengths_expand).long()  # (batch_size, max_seq_len)

            # Forward pass - 모든 chunk를 한 번에 배치로 처리
            audio_outputs = self.continuous_audio_model(
                mel_spectrograms, attention_mask=audio_masks_1d
            )  # mel_spectrograms.shape: (num_chunks, 128, 3000), audio_masks_1d.shape: (num_chunks, 1500)
            
            # 각 chunk의 실제 길이만큼만 추출하여 concat
            audio_features = audio_outputs.last_hidden_state  # (batch_size, max_seq_len, d_model)
            continuous_features_list = []
            for b_idx in range(batch_size):
                output_length = audio_output_lengths[b_idx].item()
                actual_length = audio_features.shape[1]
                extract_length = min(output_length, actual_length)
                continuous_feature_chunk = audio_features[b_idx, :extract_length]  # (extract_length, d_model)
                continuous_features_list.append(continuous_feature_chunk)
            
            continuous_feature = torch.cat(continuous_features_list, dim=0)  # (T_total, d_model)
            continuous_feature = self.audio_projector(continuous_feature)  # (T_total, d_model) -> (T_total, hidden_size)

        # Discrete audio encoder: raw audio 처리
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype, enabled=True):
            # 오디오를 device로 이동
            audio_tensor_device = audio_tensor.to(self.device)
            
            # 80초 단위로 chunking (cosyvoice2의 제한)
            chunk_size = 80 * self.sample_rate
            audio_length_samples = audio_tensor_device.shape[-1]
            
            # modeling_vlm.py와 동일하게 처리: 80초보다 긴 경우만 chunking
            discrete_tokens_list = []
            if audio_length_samples > chunk_size:
                # 긴 오디오를 청킹하여 처리한 후 결과를 연결
                for i in range(0, audio_length_samples, chunk_size):
                    chunk = audio_tensor_device[i : i + chunk_size]
                    discrete_tokens_chunk = self.discrete_audio_model(chunk)
                    discrete_tokens_list.append(discrete_tokens_chunk)
                discrete_tokens = torch.cat(discrete_tokens_list, dim=-1)  # (1, T_tokens)
            else:
                # 짧은 오디오는 그대로 처리
                discrete_tokens = self.discrete_audio_model(audio_tensor_device)  # (1, T_tokens)

        return AudioEncoderResult(
            discrete_tokens=discrete_tokens.squeeze(0),  # (T_tokens,)
            continuous_feature=continuous_feature,
            audio_length=audio_length,
            sample_rate=self.sample_rate,
        )

    def process_media_from_url(
        self,
        media_url: str,
        media_type: str = "audio",
    ) -> Dict[str, str]:
        """
        URL에서 미디어를 다운로드하고 처리

        Args:
            media_url: 미디어 URL
            media_type: 'audio' or 'video'

        Returns:
            {"s3_key": str}
        """
        # 캐시 키 생성 (단일 파일)
        safe_url = quote_plus(media_url, safe="")
        cache_key = f"source/derived/embedding/audio/{self.model_id}/{media_type}_{safe_url}.pt"

        if self._check_cache_exists(cache_key):
            logger.info(f"캐시에서 결과 조회: {cache_key}")
            return {"s3_key": cache_key}

        # 미디어 다운로드
        media_bytes = self._download_media_from_url(media_url)

        # 오디오 추출
        media_bytes.seek(0)
        audio_tensor = (
            self._extract_audio_from_video(media_bytes)
            if media_type == "video"
            else self._load_audio_from_bytes(media_bytes)
        )

        # 오디오 처리
        result = self._process_audio(audio_tensor)

        # 캐시 저장 (단일 딕셔너리로 저장)
        cache_data = {
            "discrete": result.discrete_tokens.detach().cpu(),
            "continuous": result.continuous_feature.detach().cpu(),
            "meta": {
                "audio_length": result.audio_length,
                "sample_rate": result.sample_rate,
            },
        }

        self._save_to_cache(cache_key, cache_data, is_tensor=True)

        return {"s3_key": cache_key}

    def process_media_from_base64(
        self,
        base64_str: str,
        media_type: str = "audio",
    ) -> Dict[str, str]:
        """
        Base64 문자열에서 미디어를 로드하고 처리

        Args:
            base64_str: Base64 인코딩된 미디어
            media_type: 'audio' or 'video'

        Returns:
            {"s3_key": str}
        """
        # base64 문자열의 해시를 사용하여 캐시 키 생성 (단일 파일)
        base64_hash = hashlib.sha256(base64_str.encode()).hexdigest()[:16]
        cache_key = f"source/derived/embedding/audio/{self.model_id}/media_base64_{base64_hash}.pt"

        if self._check_cache_exists(cache_key):
            logger.info(f"캐시에서 결과 조회: {cache_key}")
            return {"s3_key": cache_key}

        # 오디오 추출
        is_video = media_type == "video"
        base64_str = base64_str.split(",", 1)[1] if base64_str.startswith("data:") and "," in base64_str else base64_str
        video_bytes = io.BytesIO(base64.b64decode(base64_str))
        audio_tensor = self._extract_audio_from_video(video_bytes) if is_video else self._load_audio_from_base64(base64_str)

        # 오디오 처리
        result = self._process_audio(audio_tensor)

        # 비디오의 경우 video_audio_compressor로 토큰 압축
        continuous_feature = result.continuous_feature
        if is_video:
            continuous_feature = self._compress_video_audio(continuous_feature)

        # 캐시 저장 (단일 딕셔너리로 저장)
        cache_data = {
            "discrete": result.discrete_tokens.detach().cpu(),
            "continuous": continuous_feature.detach().cpu(),
            "meta": {
                "audio_length": result.audio_length,
                "sample_rate": result.sample_rate,
            },
        }

        self._save_to_cache(cache_key, cache_data, is_tensor=True)

        return {"s3_key": cache_key}

    def process_audio_from_base64(
        self,
        base64_str: str,
    ) -> Dict[str, str]:
        """
        Base64 문자열에서 오디오를 로드하고 처리 (오디오 전용)

        Args:
            base64_str: Base64 인코딩된 오디오

        Returns:
            {"s3_key": str}
        """
        return self.process_media_from_base64(base64_str, media_type="audio")


class VLM_Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
