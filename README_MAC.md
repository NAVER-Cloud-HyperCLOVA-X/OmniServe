# OmniServe for Apple Silicon (Mac)

This guide explains how to run OmniServe (Track A & Track B) on macOS with Apple Silicon (M1/M2/M3). Since Docker for Mac does not support GPU acceleration natively, we run the services directly on the host using PyTorch's MPS backend and vLLM's Metal support.

## Prerequisites

1. **Python 3.10+**: Ensure you have Python installed. We recommend using `pyenv` or `conda` to manage your environment.
2. **PyTorch with MPS Support**: PyTorch natively supports the Metal Performance Shaders (MPS) backend on Apple Silicon.
3. **vLLM with Metal Support**: You need to install a vLLM build compiled for Metal (`vllm-metal` or from source).

## Installation

### 1. Setup Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### 2. Install PyTorch (MPS)

PyTorch for Mac comes with MPS support by default in the standard installation:
```bash
pip install torch torchvision torchaudio
```

### 3. Install vLLM (Metal)

To run the LLM models natively on the Mac GPU, you must use vLLM with the Metal backend. You can either install the `vllm-metal` community build or build from source:

```bash
# Option A: Install via pre-built wheel (if available)
pip install vllm-metal

# Option B: Build from source
# Make sure you have Xcode Command Line Tools installed (xcode-select --install)
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -e .
cd ..
```

### 4. Install Microservice Dependencies

Install the requirements for the encoders, decoders, and the Omni Chainer:

```bash
pip install -r encoder/vision/track_b/requirements.txt
pip install -r encoder/audio/track_b/requirements.txt
pip install -r decoder/vision/track_b/requirements.txt
pip install -r decoder/audio/track_b/requirements.txt
pip install -e omni_chainer
```

## Running the Services

After editing your `.env` file to point to your converted model paths (see the main `README.md`), you can launch all services natively using the provided script:

```bash
./run_mac.sh
```

This script will start the vLLM server (using Metal), the vision/audio encoders and decoders (using PyTorch MPS), and the `omni-chainer` orchestration layer in the background.

Logs for each service will be written to the `logs/` directory.

### Stopping the Services

To stop all running services, simply press `Ctrl+C` in the terminal where you ran `./run_mac.sh`.
