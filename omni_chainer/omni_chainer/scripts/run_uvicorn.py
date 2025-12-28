# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import uvicorn


def main():
  uvicorn.run("omni_chainer.main:app", host="0.0.0.0", port=8000, reload=True)

