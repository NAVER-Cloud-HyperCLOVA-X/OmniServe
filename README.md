# OmniServe - Multimodal LLM Inference System

A multimodal LLM inference system supporting vision and audio modalities with encoding and decoding capabilities. Designed to run HyperCLOVAX-SEED models with OpenAI-compatible API.

## Overview

### Supported Models

| Model | Type | Modalities | Parameters |
|-------|------|------------|------------|
| [HyperCLOVAX-SEED-Think-32B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B) | VLM | Image/Video → Text | 32B |
| [HyperCLOVAX-SEED-Omni-8B](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B) | OMNI | Image/Video/Audio ↔ Text/Image/Audio | 8B |

### Key Features

- **OpenAI-compatible API**: Drop-in replacement for OpenAI's chat completions API
- **Multi-GPU inference**: Distributed architecture for optimal performance
- **Full multimodal support**: Text, image, video, and audio input/output
- **Docker-based deployment**: Easy setup with Docker Compose
- **Reasoning mode**: Chain-of-thought reasoning for complex tasks

## Architecture

OmniServe orchestrates multimodal inference by routing requests through encoders, LLM, and decoders.

### VLM (Vision Language Model)

```
                         User Request
                       (Image/Video/Text)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            OmniServe                                    │
│                  POST /a/v1/chat/completions                            │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     [1] INPUT ENCODING                           │   │
│  │                                                                  │   │
│  │                   ┌─────────────────┐                            │   │
│  │                   │  Vision Encoder │                            │   │
│  │                   │  (Image/Video)  │                            │   │
│  │                   └────────┬────────┘                            │   │
│  │                            │ embeddings                          │   │
│  └────────────────────────────┼─────────────────────────────────────┘   │
│                               ▼                                         │
│                       ┌──────────────┐                                  │
│                       │  LLM (32B)   │◀──── text                        │
│                       │    (vLLM)    │                                  │
│                       └──────┬───────┘                                  │
│                              │                                          │
│                              ▼                                          │
│                        Text Response                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                           Response
                            (Text)
```

**Data Flow:** User Request → OmniServe → Vision Encoder → LLM → Text Response

### OMNI (Multimodal Model)

```
                         User Request
                    (Image/Audio/Video/Text)
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            OmniServe                                    │
│                  POST /b/v1/chat/completions                            │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                     [1] INPUT ENCODING                           │   │
│  │                                                                  │   │
│  │    ┌─────────────────┐               ┌─────────────────┐         │   │
│  │    │  Vision Encoder │               │  Audio Encoder  │         │   │
│  │    │  (Image/Video)  │               │     (Audio)     │         │   │
│  │    └────────┬────────┘               └────────┬────────┘         │   │
│  │             │                                 │                  │   │
│  │             └────────────┬────────────────────┘                  │   │
│  │                          │ embeddings                            │   │
│  └──────────────────────────┼───────────────────────────────────────┘   │
│                             ▼                                           │
│                     ┌──────────────┐                                    │
│                     │   LLM (8B)   │◀──── text                          │
│                     │    (vLLM)    │                                    │
│                     └──────┬───────┘                                    │
│                            │                                            │
│  ┌─────────────────────────┼────────────────────────────────────────┐   │
│  │                  [2] OUTPUT DECODING                             │   │
│  │                         │                                        │   │
│  │          ┌──────────────┼──────────────┐                         │   │
│  │          ▼              ▼              ▼                         │   │
│  │    ┌───────────┐  ┌───────────┐  ┌───────────┐                   │   │
│  │    │   Text    │  │  Vision   │  │   Audio   │                   │   │
│  │    │           │  │  Decoder  │  │  Decoder  │                   │   │
│  │    └───────────┘  └─────┬─────┘  └─────┬─────┘                   │   │
│  │                         │              │                         │   │
│  │                         ▼              ▼                         │   │
│  │                    Image URL      Audio URL                      │   │
│  │                      (S3)           (S3)                         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                         Response
                   (Text / Image URL / Audio URL)
```

**Data Flow:** User Request → OmniServe → Encoders → LLM → Decoders → Response

### Components

| Component | Model | Description |
|-----------|-------|-------------|
| **OmniServe** | VLM, OMNI | Orchestration layer that routes requests to encoders, LLM, and decoders |
| **Vision Encoder** | VLM, OMNI | Converts images/videos to embeddings for LLM input |
| **Audio Encoder** | OMNI | Converts audio to embeddings for LLM input |
| **LLM (vLLM)** | VLM, OMNI | High-performance language model inference server |
| **Vision Decoder** | OMNI | Generates images from LLM output tokens |
| **Audio Decoder** | OMNI | Generates audio from LLM output tokens |

## Requirements

### Hardware

| Model | Service | GPU | VRAM Required |
|-------|---------|-----|---------------|
| VLM | Vision Encoder | 1x GPU | ~8GB |
| VLM | VLM (32B) | 2x GPU | ~60GB |
| VLM | **Total** | **3x GPU** | **~68GB** |
| OMNI | Vision/Audio Encoder | 1x GPU | ~12GB |
| OMNI | Omni LLM (8B) | 1x GPU | ~16GB |
| OMNI | Vision/Audio Decoder | 1x GPU | ~20GB |
| OMNI | **Total** | **3x GPU** | **~48GB** |

**Recommended**: NVIDIA A100 40GB or 80GB GPUs

### Software

- Docker & Docker Compose
- NVIDIA Driver 525+
- CUDA 12.1+
- Python 3.10+ (for model conversion)
- `jq` (for test scripts)

## Quick Start

### Step 1: Clone and Setup

```bash
git clone https://github.com/NAVER-Cloud-HyperCLOVA-X/OmniServe.git
cd OmniServe
cp .env.example .env
```

### Step 2: Install Dependencies

```bash
# Install HuggingFace CLI and model converter dependencies
pip install huggingface_hub safetensors torch easydict

# (Optional) OpenAI client for running the examples below
pip install openai
```

### Step 3: Download Models

```bash
# VLM model (~60GB, ~30 minutes)
huggingface-cli download naver-hyperclovax/HyperCLOVAX-SEED-Think-32B \
    --local-dir ./models/HyperCLOVAX-SEED-Think-32B

# OMNI model (~16GB, ~10 minutes)
huggingface-cli download naver-hyperclovax/HyperCLOVAX-SEED-Omni-8B \
    --local-dir ./models/HyperCLOVAX-SEED-Omni-8B
```

### Step 4: Convert Models

The models need to be split into individual components (vision encoder, audio encoder, LLM, etc.):

```bash
# VLM: Extract Vision Encoder + LLM
python convert_model.py \
    --input ./models/HyperCLOVAX-SEED-Think-32B \
    --output ./track_a \
    --track a

# OMNI: Extract all components (VE, AE, LLM, VD, AD)
python convert_model.py \
    --input ./models/HyperCLOVAX-SEED-Omni-8B \
    --output ./track_b \
    --track b
```

After conversion, you'll have the following directory structure:

```
./track_a/
├── llm/HyperCLOVAX-SEED-Think-32B/    # LLM weights
└── ve/HyperCLOVAX-SEED-Think-32B/     # Vision Encoder weights

./track_b/
├── llm/HyperCLOVAX-SEED-Omni-8B/      # LLM weights
├── ve/HyperCLOVAX-SEED-Omni-8B/       # Vision Encoder weights
├── ae/HyperCLOVAX-SEED-Omni-8B/       # Audio Encoder weights
├── vd/HyperCLOVAX-SEED-Omni-8B/       # Vision Decoder weights
└── ad/HyperCLOVAX-SEED-Omni-8B/       # Audio Decoder weights
```

### Step 5: Configure Environment

Edit `.env` with your model paths:

```bash
# VLM Models
VLM_MODEL_PATH=./track_a/llm/HyperCLOVAX-SEED-Think-32B
VLM_ENCODER_VISION_MODEL_PATH=./track_a/ve/HyperCLOVAX-SEED-Think-32B

# OMNI Models
OMNI_MODEL_PATH=./track_b/llm/HyperCLOVAX-SEED-Omni-8B
OMNI_ENCODER_VISION_MODEL_PATH=./track_b/ve/HyperCLOVAX-SEED-Omni-8B
OMNI_ENCODER_AUDIO_MODEL_PATH=./track_b/ae/HyperCLOVAX-SEED-Omni-8B
OMNI_DECODER_VISION_MODEL_PATH=./track_b/vd/HyperCLOVAX-SEED-Omni-8B
OMNI_DECODER_AUDIO_TORCHSERVE_MODEL_PATH=./track_b/ad/HyperCLOVAX-SEED-Omni-8B

# S3 Storage (required for image/audio generation)
NCP_S3_ENDPOINT=https://your-s3-endpoint.com
NCP_S3_ACCESS_KEY=your-access-key
NCP_S3_SECRET_KEY=your-secret-key
NCP_S3_BUCKET_NAME=your-bucket
```

### Step 6: Build and Run

```bash
# VLM only (Track A - Vision Language Model)
docker compose --profile track-a build
docker compose --profile track-a up -d

# OMNI only (Track B - Full multimodal)
docker compose --profile track-b build
docker compose --profile track-b up -d

# VLM + OMNI (Both tracks)
docker compose --profile track-a --profile track-b build
docker compose --profile track-a --profile track-b up -d
```

### Step 7: Wait for Services

Models take time to load (~5 minutes):

```bash
# Check container status
docker compose --profile track-a --profile track-b ps

# Check logs
docker compose logs -f vlm          # VLM LLM (Track A)
docker compose logs -f omni         # OMNI LLM (Track B)
```

### Step 8: Verify Deployment

Run the test suite to verify all services are working:

```bash
# Run full test suite (10 tests total)
#
# Note:
# - OmniServe (omni-chainer) listens on http://localhost:8000 by default (see docker-compose.yml)
#   so set OMNI_CHAINER_ENDPOINT explicitly when running locally.
OMNI_CHAINER_ENDPOINT=http://localhost:8000 bash scripts/check_server.sh

# With verbose output
OMNI_CHAINER_ENDPOINT=http://localhost:8000 VERBOSE=true bash scripts/check_server.sh
```

If you started **VLM only (Track A)**, the OMNI tests will fail. For **10/10 tests**, start both tracks:

```bash
docker compose --profile track-a --profile track-b up -d
```

**Test Coverage:**
| Test | Model | Description |
|------|-------|-------------|
| Image Input | VLM | Image understanding |
| Non-Reasoning | VLM | Text generation |
| Reasoning | VLM | Chain-of-thought reasoning |
| Audio Input | OMNI | Speech recognition |
| Image Input | OMNI | Image understanding |
| Image Output | OMNI | Text-to-image generation |
| Image to Image | OMNI | Image transformation |
| Video Input | OMNI | Video understanding |
| Audio Output | OMNI | Text-to-speech |
| Audio to Audio | OMNI | Voice response |

Expected output: `10/10 tests passed`

## Usage Examples

### VLM: Image Understanding

```bash
curl -X POST http://localhost:8000/a/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "track_a_model",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {
        "role": "user",
        "content": [
          {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
          {"type": "text", "text": "Describe this image in detail."}
        ]
      }
    ],
    "max_tokens": 512,
    "extra_body": {"chat_template_kwargs": {"thinking": false}}
  }'
```

### VLM: Reasoning Mode

```bash
curl -X POST http://localhost:8000/a/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "track_a_model",
    "messages": [
      {"role": "user", "content": "Solve step by step: 3x + 7 = 22"}
    ],
    "max_tokens": 1024,
    "extra_body": {
      "thinking_token_budget": 500,
      "chat_template_kwargs": {"thinking": true}
    }
  }'
```

### OMNI: Audio to Text

```bash
curl -X POST http://localhost:8000/b/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "track_b_model",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "input_audio", "input_audio": {"data": "<base64_audio_url>", "format": "mp3"}},
          {"type": "text", "text": "What is being said?"}
        ]
      }
    ],
    "max_tokens": 256,
    "chat_template_kwargs": {"skip_reasoning": true}
  }'
```

### OMNI: Text to Image

```bash
curl -X POST http://localhost:8000/b/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "track_b_model",
    "messages": [
      {"role": "user", "content": "Draw a picture of a sunset over mountains"}
    ],
    "tools": [{
      "type": "function",
      "function": {
        "name": "t2i_model_generation",
        "description": "Generates an image.",
        "parameters": {"type": "object", "required": ["discrete_image_token"], "properties": {"discrete_image_token": {"type": "string"}}}
      }
    }],
    "max_tokens": 7000,
    "chat_template_kwargs": {"skip_reasoning": true}
  }'
```

### OMNI: Text to Audio

```bash
curl -X POST http://localhost:8000/b/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "track_b_model",
    "messages": [
      {"role": "user", "content": "Say hello in a cheerful voice"}
    ],
    "max_tokens": 1000,
    "chat_template_kwargs": {"skip_reasoning": true}
  }'
```

## API Reference

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /a/v1/chat/completions` | VLM: Vision chat completions |
| `POST /b/v1/chat/completions` | OMNI: Multimodal chat completions |
| `GET /health` | Health check |

### Request Format

The API follows OpenAI's chat completions format. See [OpenAI API Reference](https://platform.openai.com/docs/api-reference/chat) for details.

**Model-specific parameters:**

| Parameter | Model | Description |
|-----------|-------|-------------|
| `extra_body.chat_template_kwargs.thinking` | VLM | Enable reasoning mode |
| `extra_body.thinking_token_budget` | VLM | Max tokens for reasoning |
| `chat_template_kwargs.skip_reasoning` | OMNI | Skip reasoning (recommended) |
| `tools` | OMNI | Enable image generation |

## Service Management

### Stop Services

```bash
# Stop VLM only (Track A)
docker compose --profile track-a down

# Stop OMNI only (Track B)
docker compose --profile track-b down

# Stop all services (VLM + OMNI)
docker compose --profile track-a --profile track-b down

# Stop and remove volumes
docker compose --profile track-a --profile track-b down -v
```

### Rebuild

```bash
# Rebuild VLM only (Track A)
docker compose --profile track-a build --no-cache
docker compose --profile track-a up -d

# Rebuild OMNI only (Track B)
docker compose --profile track-b build --no-cache
docker compose --profile track-b up -d

# Rebuild all services (VLM + OMNI)
docker compose --profile track-a --profile track-b build --no-cache
docker compose --profile track-a --profile track-b up -d
```

### Service List

| Service | Profile | Description |
|---------|---------|-------------|
| `vlm` | track-a | VLM LLM (vLLM server) |
| `vlm-encoder-vision-api` | track-a | VLM Vision Encoder |
| `omni-chainer` | default | Orchestration layer |
| `omni` | track-b | OMNI LLM (vLLM server) |
| `omni-encoder-audio-api` | track-b | OMNI Audio Encoder |
| `omni-encoder-vision-api` | track-b | OMNI Vision Encoder |
| `omni-decoder-vision-api` | track-b | OMNI Vision Decoder |
| `omni-decoder-audio-api` | track-b | OMNI Audio Decoder API |
| `omni-decoder-audio-torchserve` | track-b | OMNI Audio Decoder Backend |

## Model Converter

Extract individual components (Vision Encoder, Audio Encoder, LLM, etc.) from unified HuggingFace models.

```bash
# OMNI: Extract all components
python convert_model.py \
    --input /path/to/omni/model \
    --output /path/to/output \
    --track b

# VLM: Extract VE + LLM
python convert_model.py \
    --input /path/to/vlm/model \
    --output /path/to/output \
    --track a
```

See [model_converter/README.md](model_converter/README.md) for details.

## Troubleshooting

### Models not loading

- Check GPU memory: `nvidia-smi`
- Check logs: `docker compose logs -f <service_name>`
- Ensure model paths in `.env` are correct and accessible

### Out of memory

- Reduce batch size or max tokens
- Use tensor parallelism for larger models
- Check if other processes are using GPU memory

### Connection refused

- Wait for services to fully load (~5 minutes)
- Check if containers are running: `docker compose ps`
- Verify port configurations in `.env`

### Image/Audio generation not working

- Ensure S3 credentials are configured in `.env`
- Check Vision/Audio Decoder logs
- Verify S3 bucket exists and is accessible

## License

OmniServe - Multimodal LLM Inference System
Copyright (c) 2025-present NAVER Cloud Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
