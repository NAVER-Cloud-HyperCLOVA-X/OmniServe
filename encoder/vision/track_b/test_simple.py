#!/usr/bin/env python3
# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import sys
import json
import os

# Add current directory to path for local imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from app.model import Model

# Load config
with open("config.json", "r") as f:
    config = json.load(f)

# Initialize model
print("Initializing model...")
model = Model(
    processor_model_name_or_path=config["processor_model_name_or_path"],
    cont_config_path=config["cont_config_path"],
    cont_weight_path=config["models_weight"]["vision_module"]["path"],
    disc_weight_path=config["models_weight"]["discrete_vision"]["path"],
    mm_projector_weight_path=config["models_weight"]["mm_projector"]["path"],
    dtype=config["dtype"],
    llm_hidden_size=config["llm_hidden_size"],
)
print("Model initialized!")

# Test processor
from PIL import Image
import requests
from io import BytesIO

print("Downloading test image...")
response = requests.get("https://picsum.photos/400/300", timeout=10)
image = Image.open(BytesIO(response.content)).convert("RGB")
print(f"Image: {image.size}")

print("Testing processor...")
processed = model.image_processor(images=[image], return_tensors="pt")
print(f"Keys: {list(processed.keys())}")
print(f"pixel_values shape: {processed.pixel_values.shape}")
print(f"pixel_values dtype: {processed.pixel_values.dtype}")
if hasattr(processed, "image_grid_thw"):
    print(f"image_grid_thw: {processed.image_grid_thw}")
print("Processor test passed!")

print("Testing full pipeline...")
result = model._process_image(image)
print(f"Continuous features: {result.continuous_feature.shape if result.continuous_feature is not None else None}")
print("Full pipeline test passed!")
