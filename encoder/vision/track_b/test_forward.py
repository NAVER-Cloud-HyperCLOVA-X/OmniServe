#!/usr/bin/env python3
# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

"""Test script to debug the forward pass issue"""
import sys
import os
import torch
import logging
from PIL import Image
import requests
from io import BytesIO

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add current directory to path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from app.model import Model
import json

# Load config
config_path = os.path.join(script_dir, "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

# Initialize model
logger.info("Initializing model...")
model = Model(
    processor_model_name_or_path=config["model"]["processor"]["path"],
    cont_config_path=config["model"]["continuous_vision"]["config_path"],
    cont_weight_path=config["model"]["continuous_vision"]["weight_path"],
    disc_weight_path=config["model"]["discrete_vision"]["weight_path"],
    mm_projector_weight_path=config["model"]["mm_projector"]["path"],
    dtype=config["dtype"],
    llm_hidden_size=config["llm_hidden_size"],
)

# Download test image
logger.info("Downloading test image...")
try:
    response = requests.get("https://picsum.photos/400/300", timeout=10)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    logger.info(f"Image loaded: {image.size}, mode: {image.mode}")
except Exception as e:
    logger.error(f"Failed to download image: {e}")
    sys.exit(1)

# Process image
logger.info("Processing image...")
try:
    result = model._process_image(image)
    logger.info("Success! Processing completed.")
    logger.info(
        f"Continuous features shape: {result.continuous_feature.shape if result.continuous_feature is not None else None}"
    )
    logger.info(
        f"Discrete tokens shape: {result.discrete_tokens.shape if result.discrete_tokens is not None else None}"
    )
except Exception as e:
    logger.error(f"Processing failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
