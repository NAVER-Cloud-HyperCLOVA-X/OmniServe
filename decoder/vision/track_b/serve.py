# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

"""
FastAPI server for Vision Token to Image Pipeline.

Usage:
    uvicorn serve:app --host 0.0.0.0 --port {port}

    # Or run directly:
    python serve.py --host 0.0.0.0 --port {port}
"""

import io
import logging
import os
import re
import sys
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vision-decoder-api")

# Add the current directory to path for importing pipeline
scripts_dir = str(Path(__file__).parent)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from pipeline import VisionTokenToImagePipeline

# S3 connection import (wbl_storage_utility 패키지 사용)
try:
    from wbl_storage_utility.s3_util import S3Connection
    print("✓ Loaded S3Connection from wbl_storage_utility")
except ImportError as e:
    print(f"⚠ Warning: Could not import wbl_storage_utility: {e}")
    print("  S3 upload feature will be disabled")
    S3Connection = None

# Global state
class ModelState:
    """전역 모델 상태"""
    pipeline: Optional[VisionTokenToImagePipeline] = None
    s3_connection = None

state = ModelState()

density = 768**2
factor = 16
_ratios = [(1, 1), (1, 2), (3, 4), (3, 5), (4, 5), (6, 9), (9, 16)]
for w, h in list(_ratios):
    if (h, w) not in _ratios:
        _ratios.append((h, w))

ratio2res = dict()  # (width, height)-ratio -> actual (width, height)
for w, h in list(_ratios):
    r = h / w
    res = (int(((density / r) ** 0.5 // factor) * factor), int(((density * r) ** 0.5 // factor) * factor))
    ratio2res[(w, h)] = res


def get_resolution(vlm_output: str) -> tuple[int, int]:
    mo = re.search(r"<\|vision_ratio_(\d+):(\d+)\|>", vlm_output)
    assert mo is not None
    return ratio2res[(int(mo.group(1)), int(mo.group(2)))]


class InferenceRequest(BaseModel):
    """Request model for vision token to image inference."""
    vision_tokens: List[int] = Field(
        ...,
        description="Vision tokens (expected length: 729)",
        min_length=1,
    )
    height: int = Field(default=768, ge=64, le=2048, description="Output image height")
    width: int = Field(default=768, ge=64, le=2048, description="Output image width")
    num_inference_steps: int = Field(default=50, ge=1, le=200, description="Number of denoising steps")
    guidance_scale: float = Field(default=0.75, ge=0.0, le=2.0, description="Autoguidance scale (0 = no guidance, requires transformer2)")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class VLMOutputRequest(BaseModel):
    """Request model for VLM output string to image inference."""
    vlm_output: str = Field(
        ...,
        description="VLM output string containing vision tokens like <|vision12345|>",
    )
    height: Optional[int] = Field(default=None, ge=64, le=2048, description="Output image height (auto-detected from ratio if not provided)")
    width: Optional[int] = Field(default=None, ge=64, le=2048, description="Output image width (auto-detected from ratio if not provided)")
    num_inference_steps: int = Field(default=50, ge=1, le=200, description="Number of denoising steps")
    guidance_scale: float = Field(default=0.75, ge=0.0, le=2.0, description="Autoguidance scale (0 = no guidance, requires transformer2)")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    
    # S3 업로드 옵션
    upload_to_s3: bool = Field(default=True, description="Upload image to S3 and return presigned URL")
    s3_key: Optional[str] = Field(default=None, description="S3 object key (filename)")
    s3_prefix: Optional[str] = Field(default="vision-decoder", description="S3 key prefix (folder path)")
    s3_expiration: int = Field(default=3600, ge=60, le=604800, description="Presigned URL expiration time in seconds")


class VLMOutputResponse(BaseModel):
    """Response model for VLM output to image generation."""
    message: str
    num_images: int = Field(default=1)
    height: int
    width: int
    num_tokens_parsed: int
    guidance_scale: float
    seed: Optional[int] = None
    presigned_url: Optional[str] = Field(default=None, description="S3 presigned URL (if uploaded)")
    s3_path: Optional[str] = Field(default=None, description="S3 full path (if uploaded)")


def parse_vision_tokens(vlm_output: str) -> Tuple[np.ndarray, int, int]:
    """
    Parse vision tokens from VLM output string.
    
    Extracts tokens like <|vision12345|> and returns as numpy array of integers.
    Also parses ratio token like <|vision_ratio_1:1|> and calculates image dimensions
    based on 0.6MP (600,000 pixels) constraint.
    Ignores special tokens like <|discrete_image_start|>, <|vision_eol|>.
    
    Args:
        vlm_output: VLM output string containing vision tokens
        
    Returns:
        tuple of (vision_tokens, width, height):
            - vision_tokens: numpy array of vision token IDs (expected length: 729)
            - width: calculated image width in pixels
            - height: calculated image height in pixels
    """
    # Parse vision tokens
    pattern = r'<\|vision(\d+)\|>'
    matches = re.findall(pattern, vlm_output)
    
    if not matches:
        raise ValueError("No vision tokens found in VLM output")
    
    vision_tokens = np.array([int(m) for m in matches], dtype=np.int64)[:729]
    
    width, height = get_resolution(vlm_output)
    
    return vision_tokens, width, height


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool


def sanitize_filename(filename: str) -> str:
    """
    파일명에서 특수문자를 제거하고 안전한 파일명으로 변환
    """
    name, ext = os.path.splitext(filename)
    safe_name = re.sub(r'[^a-zA-Z0-9\-_]', '_', name)
    safe_name = re.sub(r'_+', '_', safe_name)
    safe_name = safe_name.strip('_')
    return f"{safe_name}{ext}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    print("[INFO] Loading VisionTokenToImagePipeline...")
    model_dir = scripts_dir  # Use the same directory as the script
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Use bfloat16 for faster inference (16 sec vs 2 min without it)
    state.pipeline = VisionTokenToImagePipeline.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
    ).to(device)
    
    print("[INFO] Pipeline loaded successfully!")
    
    # S3 연결 초기화
    if S3Connection is not None:
        print("[INFO] Initializing S3 connection...")
        try:
            state.s3_connection = S3Connection()
            print("[INFO] S3 connection initialized successfully!")
        except Exception as e:
            print(f"[WARNING] S3 connection failed: {e}")
            print("  S3 upload feature will be disabled.")
            state.s3_connection = None
    else:
        print("[INFO] S3Connection not available, S3 upload feature disabled.")
        state.s3_connection = None
    
    yield
    
    # Cleanup
    if state.pipeline is not None:
        del state.pipeline
    if state.s3_connection is not None:
        del state.s3_connection
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[INFO] Pipeline unloaded.")


app = FastAPI(
    title="Vision Token to Image API",
    description="Convert vision tokens to images using diffusion model",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the service is healthy and model is loaded."""
    return HealthResponse(
        status="healthy",
        model_loaded=state.pipeline is not None,
    )


@app.post("/decode", response_model=VLMOutputResponse)
async def decode_output(request: VLMOutputRequest):
    """
    Generate an image from VLM output string.
    
    Parses vision tokens from VLM output like:
    `<|vision40191|><|vision19346|><|vision14822|>...`
    
    - **vlm_output**: VLM output string containing vision tokens
    - **height**: Output image height (auto-detected from ratio if not provided)
    - **width**: Output image width (auto-detected from ratio if not provided)
    - **num_inference_steps**: Number of denoising steps (default: 50)
    - **seed**: Random seed for reproducibility (optional)
    - **upload_to_s3**: Whether to upload to S3 (default: True)
    - **s3_prefix**: S3 key prefix/folder (default: "vision-decoder")
    - **s3_expiration**: Presigned URL expiration in seconds (default: 3600)
    
    Returns: VLMOutputResponse with presigned_url and s3_path
    """
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # S3 업로드가 요청되었지만 S3 연결이 없는 경우
    if request.upload_to_s3 and state.s3_connection is None:
        raise HTTPException(
            status_code=503, 
            detail="S3 upload requested but S3 connection not available"
        )
    
    try:
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())[:8]
        
        # 1. Parse vision tokens from VLM output string
        vtokens, parsed_width, parsed_height = parse_vision_tokens(request.vlm_output)
        
        logger.info(f"[{request_id}] === /decode request started ===")
        logger.info(f"[{request_id}] Parsed {len(vtokens)} vision tokens from VLM output")
        logger.info(f"[{request_id}] Calculated dimensions from ratio token: {parsed_width}x{parsed_height}")
        logger.info(f"[{request_id}] guidance_scale: {request.guidance_scale}")
        logger.info(f"[{request_id}] num_inference_steps: {request.num_inference_steps}")
        logger.info(f"[{request_id}] seed: {request.seed}")
        
        # Log which model will be used based on guidance_scale
        has_transformer2 = state.pipeline.transformer2 is not None
        if request.guidance_scale > 0 and has_transformer2:
            logger.info(f"[{request_id}] Model: transformer + transformer2 (autoguidance ENABLED)")
        elif request.guidance_scale > 0 and not has_transformer2:
            logger.warning(f"[{request_id}] Model: transformer only (autoguidance DISABLED - transformer2 not loaded, guidance_scale will be ignored)")
        else:
            logger.info(f"[{request_id}] Model: transformer only (autoguidance DISABLED - guidance_scale=0)")
        
        # 2. Set up generator for reproducibility
        generator = None
        seed = request.seed
        if seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(seed)
        
        # 만약 request 내부에 width, height가 들어오면, 이걸 사용하고, 아니면 parsed_width, parsed_height를 사용한다.
        if request.width is not None:
            parsed_width = int(request.width)
        if request.height is not None:
            parsed_height = int(request.height)
        
        # 3. Run inference with parsed tokens and calculated dimensions
        with torch.no_grad():
            result = state.pipeline(
                vision_tokens=vtokens,
                height=parsed_height,
                width=parsed_width,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                generator=generator,
            )
        
        # Get the PIL image
        image = result.images[0]
        
        # 4. S3 업로드 처리
        presigned_url = None
        s3_path = None
        message = "Image generated successfully"
        
        if request.upload_to_s3:
            # 파일명 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if request.s3_key:
                safe_filename = sanitize_filename(request.s3_key)
                key_name = f"{timestamp}_{Path(safe_filename).stem}.png"
            else:
                seed_str = f"_seed{seed}" if seed is not None else ""
                key_name = f"{timestamp}_{request.num_inference_steps}steps{seed_str}.png"
            
            # 임시 파일에 저장
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                image.save(tmp_path)
            
            try:
                # S3 업로드 및 presigned URL 생성
                presigned_url = state.s3_connection.upload_wbl_asset(
                    file_path=tmp_path,
                    key=key_name,
                    prefix=request.s3_prefix or "vision-decoder"
                )
                
                # S3 경로 생성 - configurable via environment variable
                prefix = request.s3_prefix or "vision-decoder"
                s3_bucket = os.getenv("WBL_S3_BUCKET_NAME", "")
                s3_path = f"s3://{s3_bucket}/{prefix}/{key_name}"
                
                message = f"Image uploaded to S3: {s3_path}"
                
            finally:
                # 임시 파일 삭제
                tmp_path.unlink(missing_ok=True)
        
        return VLMOutputResponse(
            message=message,
            num_images=1,
            height=parsed_height,
            width=parsed_width,
            num_tokens_parsed=len(vtokens),
            guidance_scale=request.guidance_scale,
            seed=seed,
            presigned_url=presigned_url,
            s3_path=s3_path,
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        error_msg = str(e)
        logger.error(f"[{request_id}] Exception occurred: {error_msg}")
        logger.error(f"[{request_id}] Traceback:\n{traceback.format_exc()}")
        if "s3" in error_msg.lower() or "upload" in error_msg.lower():
            raise HTTPException(status_code=500, detail=f"S3 upload failed: {error_msg}")
        else:
            raise HTTPException(status_code=500, detail=f"Inference failed: {error_msg}")


@app.post("/decode_base64")
async def decode_output_base64(request: VLMOutputRequest):
    """
    Generate an image from VLM output string and return as base64.
    
    Returns: JSON with base64-encoded PNG image and token info
    """
    import base64
    
    if state.pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Generate request ID for tracking
        request_id = str(uuid.uuid4())[:8]
        
        # 1. Parse vision tokens from VLM output string
        vtokens, parsed_width, parsed_height = parse_vision_tokens(request.vlm_output)
        
        logger.info(f"[{request_id}] === /decode_base64 request started ===")
        logger.info(f"[{request_id}] Parsed {len(vtokens)} vision tokens from VLM output")
        logger.info(f"[{request_id}] Calculated dimensions from ratio token: {parsed_width}x{parsed_height}")
        logger.info(f"[{request_id}] guidance_scale: {request.guidance_scale}")
        logger.info(f"[{request_id}] num_inference_steps: {request.num_inference_steps}")
        logger.info(f"[{request_id}] seed: {request.seed}")
        
        # Log which model will be used based on guidance_scale
        has_transformer2 = state.pipeline.transformer2 is not None
        if request.guidance_scale > 0 and has_transformer2:
            logger.info(f"[{request_id}] Model: transformer + transformer2 (autoguidance ENABLED)")
        elif request.guidance_scale > 0 and not has_transformer2:
            logger.warning(f"[{request_id}] Model: transformer only (autoguidance DISABLED - transformer2 not loaded, guidance_scale will be ignored)")
        else:
            logger.info(f"[{request_id}] Model: transformer only (autoguidance DISABLED - guidance_scale=0)")
        
        # 2. Set up generator for reproducibility
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
            generator.manual_seed(request.seed)
        
        # 만약 request 내부에 width, height가 들어오면, 이걸 사용
        if request.width is not None:
            parsed_width = int(request.width)
        if request.height is not None:
            parsed_height = int(request.height)
        
        # 3. Run inference with parsed tokens and calculated dimensions
        with torch.no_grad():
            result = state.pipeline(
                vision_tokens=vtokens,
                height=parsed_height,
                width=parsed_width,
                num_inference_steps=request.num_inference_steps,
                guidance_scale=request.guidance_scale,
                generator=generator,
            )
        
        # Get the PIL image
        image = result.images[0]
        
        # Convert to base64
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format="PNG")
        img_byte_arr.seek(0)
        img_base64 = base64.b64encode(img_byte_arr.getvalue()).decode("utf-8")
        
        return {
            "image_base64": img_base64,
            "format": "png",
            "height": parsed_height,
            "width": parsed_width,
            "guidance_scale": request.guidance_scale,
            "seed": request.seed,
            "num_tokens_parsed": len(vtokens),
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        import traceback
        error_msg = str(e)
        logger.error(f"[{request_id}] Exception occurred: {error_msg}")
        logger.error(f"[{request_id}] Traceback:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {error_msg}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vision Token to Image API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "serve:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
