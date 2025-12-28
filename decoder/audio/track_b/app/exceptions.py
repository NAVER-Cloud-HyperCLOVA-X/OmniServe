# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import fastapi


class ServerError(fastapi.HTTPException):
    def __init__(self, detail: str = "Internal Server Error") -> None:
        super().__init__(500, detail)


class BadRequestException(fastapi.HTTPException):
    def __init__(self, detail: str = "Bad Request") -> None:
        super().__init__(400, detail)
