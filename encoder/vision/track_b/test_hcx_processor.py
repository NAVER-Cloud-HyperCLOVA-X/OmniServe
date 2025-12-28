#!/usr/bin/env python3
# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

"""Test script to test HCXVisionV2Processor"""
import sys
import os
import torch
import logging
from PIL import Image
import requests
from io import BytesIO
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add current directory to path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from app.model import Model


def main():
    # Load config
    logger.info("Loading config...")
    config_path = os.path.join(script_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Initialize model
    logger.info("Initializing model...")
    try:
        model = Model(
            processor_model_name_or_path=config["processor_model_name_or_path"],
            cont_config_path=config["cont_config_path"],
            cont_weight_path=config["models_weight"]["vision_module"]["path"],
            disc_weight_path=config["models_weight"]["discrete_vision"]["path"],
            mm_projector_weight_path=config["models_weight"]["mm_projector"]["path"],
            dtype=config["dtype"],
            llm_hidden_size=config["llm_hidden_size"],
        )
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Download test image
    logger.info("Downloading test image...")
    try:
        response = requests.get("https://picsum.photos/400/300", timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        logger.info(f"Image loaded: {image.size}, mode: {image.mode}")
    except Exception as e:
        logger.error(f"Failed to download image: {e}")
        return 1

    # Test image processor directly
    logger.info("Testing HCXVisionV2Processor directly...")
    try:
        processed = model.image_processor(images=[image], return_tensors="pt")
        logger.info(f"Processor output keys: {processed.keys()}")
        logger.info(
            f"pixel_values shape: {processed.pixel_values.shape if hasattr(processed, 'pixel_values') else 'N/A'}"
        )
        logger.info(
            f"pixel_values dtype: {processed.pixel_values.dtype if hasattr(processed, 'pixel_values') else 'N/A'}"
        )
        if hasattr(processed, "image_grid_thw"):
            logger.info(f"image_grid_thw: {processed.image_grid_thw}")
        logger.info("HCXVisionV2Processor test passed!")
    except Exception as e:
        logger.error(f"HCXVisionV2Processor test failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Process image through full pipeline
    logger.info("Processing image through full pipeline...")
    try:
        result = model._process_image(image)
        logger.info("Success! Processing completed.")
        logger.info(
            f"Continuous features shape: {result.continuous_feature.shape if result.continuous_feature is not None else None}"
        )
        logger.info(
            f"Discrete tokens shape: {result.discrete_tokens.shape if result.discrete_tokens is not None else None}"
        )
        logger.info(f"Vision query length: {result.vision_query_length}")
        logger.info("Full pipeline test passed!")
        return 0
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
