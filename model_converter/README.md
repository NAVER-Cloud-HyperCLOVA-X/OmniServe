# Omni Model Converter

A tool for extracting individual components (Vision Encoder, Audio Encoder, LLM, Vision Decoder, Audio Decoder) from unified Omni models.

## Supported Models

- **VLM**: Vision Encoder + LLM (VE+LLM)
- **OMNI**: Vision Encoder + Audio Encoder + LLM + Vision Decoder + Audio Decoder (Full Omni)

## Installation

```bash
pip install torch safetensors easydict
```

## Usage

### Basic Usage

```bash
# OMNI: Extract all components (VE, AE, LLM, VD, AD)
python convert_model.py \
    --input /path/to/omni/model \
    --output /path/to/output \
    --track b

# VLM: Extract VE + LLM
python convert_model.py \
    --input /path/to/ve_llm/model \
    --output /path/to/output \
    --track a
```

### Extract Specific Components

```bash
# Extract LLM only
python convert_model.py \
    --input /path/to/model \
    --output /path/to/output \
    --track b \
    --components llm

# Extract VE and LLM only
python convert_model.py \
    --input /path/to/model \
    --output /path/to/output \
    --track b \
    --components ve llm
```

### Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input`, `-i` | Input model directory | (required) |
| `--output`, `-o` | Output directory | (required) |
| `--track`, `-t` | Model type: `a` (VLM) or `b` (OMNI) | `b` |
| `--components`, `-c` | Components to extract (`all`, `ve`, `ae`, `llm`, `vd`, `ad`) | `all` |
| `--max-shard-size` | Maximum LLM shard size | `5GB` |
| `--verify` | Verify extraction after conversion | - |

## Output Structure

### OMNI Extraction Result

```
output/
├── ve/
│   └── {model_name}/
│       ├── vision_weights.pt          # Vision Encoder weights
│       ├── mm_projector_weights.pt    # Vision Projector weights
│       └── ta_tok.pth                 # Discrete Vision (TA-Tok) weights
├── ae/
│   └── {model_name}/
│       ├── audio_weights.pt                    # Audio Encoder weights
│       ├── audio_projector_weights.pt          # Audio Projector weights
│       ├── discrete_audio_weights.pt           # Discrete Audio weights
│       ├── video_audio_compressor_weights.pt   # MambaMia Compressor weights
│       └── qwen2-audio-encoder-from-qwen2-audio-7b-instruct/
│           ├── config.json
│           ├── model.safetensors
│           └── preprocessor_config.json
├── vd/
│   └── {model_name}/
│       ├── scheduler/
│       ├── transformer/
│       ├── transformer2/
│       ├── vae/
│       ├── token_embedder/
│       └── model_index.json
├── ad/
│   └── {model_name}/
│       ├── NCCosybigvganDecoder.mar
│       └── NCZSCosybigvganDecoder.mar
└── llm/
    └── {model_name}/
        ├── config.json
        ├── model-00001-of-XXXXX.safetensors
        ├── model.safetensors.index.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── modeling_*.py
        └── configuration_*.py
```

### VLM Extraction Result

```
output/
├── ve/
│   └── {model_name}/
│       ├── vision_weights.pt
│       └── mm_projector_weights.pt
└── llm/
    └── {model_name}/
        ├── config.json
        ├── model-*.safetensors
        └── ...
```

## Loading Extracted Models

### Loading LLM (HuggingFace)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/path/to/output/llm/{model_name}"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto"
)
```

### Loading Vision Encoder

```python
import torch

ve_path = "/path/to/output/ve/{model_name}"

# Load Vision weights
vision_weights = torch.load(f"{ve_path}/vision_weights.pt")
mm_projector_weights = torch.load(f"{ve_path}/mm_projector_weights.pt")

# Apply weights to model
vision_model.load_state_dict(vision_weights)
mm_projector.load_state_dict(mm_projector_weights)
```

### Loading Audio Encoder (OMNI)

```python
import torch

ae_path = "/path/to/output/ae/{model_name}"

# Load Audio weights
audio_weights = torch.load(f"{ae_path}/audio_weights.pt")
audio_projector_weights = torch.load(f"{ae_path}/audio_projector_weights.pt")

# Load Qwen2 Audio Encoder (HuggingFace format)
from transformers import AutoModel
qwen2_encoder = AutoModel.from_pretrained(
    f"{ae_path}/qwen2-audio-encoder-from-qwen2-audio-7b-instruct"
)
```

## Uploading to HuggingFace Hub

### 1. Login

```bash
huggingface-cli login
```

### 2. Upload

```python
from huggingface_hub import HfApi

api = HfApi()

# Upload LLM
api.upload_folder(
    folder_path="/path/to/output/llm/{model_name}",
    repo_id="your-org/model-llm",
    repo_type="model"
)

# Upload Vision Encoder
api.upload_folder(
    folder_path="/path/to/output/ve/{model_name}",
    repo_id="your-org/model-vision-encoder",
    repo_type="model"
)

# Upload Audio Encoder (OMNI)
api.upload_folder(
    folder_path="/path/to/output/ae/{model_name}",
    repo_id="your-org/model-audio-encoder",
    repo_type="model"
)
```

### Upload via CLI

```bash
# Upload LLM
huggingface-cli upload your-org/model-llm /path/to/output/llm/{model_name}

# Upload Vision Encoder
huggingface-cli upload your-org/model-vision-encoder /path/to/output/ve/{model_name}
```

## Server Deployment Example

### Serving LLM with vLLM

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/output/llm/{model_name} \
    --trust-remote-code \
    --tensor-parallel-size 4 \
    --port 8000
```

### Using in Python

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="/path/to/output/llm/{model_name}",
    trust_remote_code=True,
    tensor_parallel_size=4
)

sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(["Hello, world!"], sampling_params)
```

## Model Structure

### OMNI Weight Prefixes

| Prefix | Component | Description |
|--------|-----------|-------------|
| `model.vision_model.` | Vision Encoder | Qwen2.5-VL Vision Transformer |
| `model.mm_projector.` | Vision Projector | Vision-to-LLM projection |
| `model.discrete_vision_model.` | TA-Tok | Discrete vision tokenizer |
| `model.audio_model.` | Audio Encoder | Qwen2 Audio Encoder |
| `model.audio_projector.` | Audio Projector | Audio-to-LLM projection |
| `model.discrete_audio_model.` | CosyVoice2 | Discrete audio tokenizer |
| `model.video_audio_compressor.` | MambaMia | Video-audio compressor |
| `model.language_model.` | LLM | Language model backbone |

### VLM Weight Prefixes

| Prefix | Component | Description |
|--------|-----------|-------------|
| `model.vision_model.` | Vision Encoder | Qwen2.5-VL Vision Transformer |
| `model.mm_projector.` | Vision Projector | Vision-to-LLM projection |
| `model.language_model.` | LLM | HyperCLOVAX backbone |

## Python API

```python
from model_converter import OmniModelConverter
from model_converter.converter import Track

# Initialize converter
converter = OmniModelConverter(
    model_dir="/path/to/omni/model",
    track=Track.B  # Track.B for OMNI, Track.A for VLM
)

# Full conversion
results = converter.convert(
    output_dir="/path/to/output",
    components=["ve", "ae", "llm"]
)

# Extract individual components
ve_path = converter.extract_vision_encoder("/path/to/output")
ae_path = converter.extract_audio_encoder("/path/to/output")
llm_path = converter.extract_llm("/path/to/output")

# Verify
success = converter.verify("/path/to/output")
```
