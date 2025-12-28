#!/usr/bin/env python3
# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

"""
Inference script for Vision Token to Image Pipeline.

Required packages:
- torch==2.6.0
- numpy==2.3.5
- PIL==12.0.0
- fastapi==0.121.3
- diffusers==0.32.2
- transformers==4.49.0
"""

import logging
import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("vision-decoder-api.inference")

# Add scripts directory to path
scripts_dir = str(Path(__file__).parent)
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from pipeline import VisionTokenToImagePipeline


def inference(
    vision_tokens: np.ndarray,
    width: int,
    height: int,
    num_inference_steps: int = 30,
    guidance_scale: float = 0.0,
    generator: int = 42,
) -> Image.Image:
    
    logger.info(f"[Inference] Starting inference with guidance_scale={guidance_scale}")
    logger.info(f"[Inference] Image dimensions: {width}x{height}")
    logger.info(f"[Inference] num_inference_steps: {num_inference_steps}")
    logger.info(f"[Inference] generator seed: {generator}")
    logger.info(f"[Inference] vision_tokens shape: {vision_tokens.shape}")

    pipeline = VisionTokenToImagePipeline.from_pretrained(
        scripts_dir,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    # Log pipeline configuration
    logger.info(f"[Inference] Pipeline loaded. transformer2 available: {pipeline.transformer2 is not None}")
    
    result = pipeline(
        vision_tokens=vision_tokens,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    
    logger.info(f"[Inference] Inference completed successfully")

    return result.images[0]


if __name__ == "__main__":
    vision_tokens = np.load("sample_vision_tokens.npy").astype(np.int64)
    width = 768
    height = 768
    num_inference_steps = 50
    generator = 42
    guidance_scale = 0.0
    result = inference(vision_tokens, width, height, num_inference_steps, guidance_scale, generator)
    result.save("result.png")
