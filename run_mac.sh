#!/bin/bash
# OmniServe Mac Native Launcher
# Copyright (c) 2025-present

# Ensure PYTHONPATH is set so that the mac_support utility is found globally
export PYTHONPATH="$(pwd):$(pwd)/util:$PYTHONPATH"

# Load env file
if [ -f .env ]; then
  source .env
else
  echo ".env file not found! Please configure paths before running."
  # Do not exit here to allow execution
fi

export VLLM_WORKER_MULTIPROC_METHOD=spawn
export OMP_NUM_THREADS=8

# Function to kill all processes on script exit
cleanup() {
  echo "Stopping services..."
  kill $(jobs -p) 2>/dev/null
}
trap cleanup EXIT INT TERM

# Create logs directory
mkdir -p logs

echo "Starting Track B (OMNI) Native Services on Apple Silicon (MPS)..."

# 1. Start vLLM Omni (Track B LLM)
if [ -n "$OMNI_MODEL_PATH" ]; then
  echo "Starting vLLM (Omni)..."
  python3 -m vllm.entrypoints.openai.api_server \
    --model ${OMNI_MODEL_PATH} \
    --host 0.0.0.0 --port ${OMNI_PORT:-10032} \
    --trust-remote-code \
    --served-model-name track_b_model \
    --enable-prompt-embeds \
    --chat-template ./omni_chainer/tests/chat_template/track_b/chat_template.jinja \
    --reasoning-parser deepseek_v3 \
    --reasoning-config '{"think_start_str": "<think>", "think_end_str": "</think>"}' \
    > logs/vllm_omni.log 2>&1 &
fi

# 2. Start Omni Audio Encoder
if [ -n "$OMNI_ENCODER_AUDIO_MODEL_PATH" ]; then
  echo "Starting Audio Encoder..."
  export MODEL_ID=${OMNI_ENCODER_AUDIO_MODEL_ID}
  python3 -m uvicorn encoder.audio.track_b.app.main:app \
    --host 0.0.0.0 --port ${OMNI_ENCODER_AUDIO_API_PORT:-10002} \
    > logs/encoder_audio.log 2>&1 &
fi

# 3. Start Omni Vision Encoder
if [ -n "$OMNI_ENCODER_VISION_MODEL_PATH" ]; then
  echo "Starting Vision Encoder..."
  export MODEL_ID=${OMNI_ENCODER_VISION_MODEL_ID}
  python3 -m uvicorn encoder.vision.track_b.app.main:app \
    --host 0.0.0.0 --port ${OMNI_ENCODER_VISION_API_PORT:-10064} \
    > logs/encoder_vision.log 2>&1 &
fi

# 4. Start Omni Vision Decoder
if [ -n "$OMNI_DECODER_VISION_MODEL_PATH" ]; then
  echo "Starting Vision Decoder..."
  python3 decoder/vision/track_b/serve.py \
    --port ${OMNI_DECODER_VISION_API_PORT:-10063} \
    > logs/decoder_vision.log 2>&1 &
fi

# 5. Start Omni Audio Decoder (FastAPI + TorchServe mocked/native)
if [ -n "$OMNI_DECODER_AUDIO_MODEL_PATH" ]; then
  echo "Starting Audio Decoder..."
  python3 -m uvicorn decoder.audio.track_b.app.main:app \
    --host 0.0.0.0 --port ${OMNI_DECODER_AUDIO_API_PORT:-11180} \
    > logs/decoder_audio.log 2>&1 &
fi

# 6. Start Omni-Chainer
echo "Starting Omni Chainer..."
export TRACK_B_LLM_ENDPOINT=http://127.0.0.1:${OMNI_PORT:-10032}/v1/chat/completions
export TRACK_B_AUDIO_ENCODING_ENDPOINT=http://127.0.0.1:${OMNI_ENCODER_AUDIO_API_PORT:-10002}/process_audio
export TRACK_B_VISION_ENCODING_ENDPOINT=http://127.0.0.1:${OMNI_ENCODER_VISION_API_PORT:-10064}/process_image_or_video
export TRACK_B_VISION_DECODING_ENDPOINT=http://127.0.0.1:${OMNI_DECODER_VISION_API_PORT:-10063}/decode
export TRACK_B_AUDIO_DECODING_ENDPOINT=http://127.0.0.1:${OMNI_DECODER_AUDIO_API_PORT:-11180}/predictions

python3 -m uvicorn omni_chainer.omni_chainer.app:app \
  --host 0.0.0.0 --port ${OMNI_CHAINER_API_PORT:-8000} \
  > logs/omni_chainer.log 2>&1 &

echo "Services started. View logs in the 'logs/' directory."
echo "Press Ctrl+C to stop all services."
wait
