# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import logging
import threading
from typing import Any, Callable, Dict, Optional

import requests

from audiollm.decoder.unit_bigvgan import utils

logger = logging.getLogger(__name__)


def _send_warmup_request(url: str, request: Dict[str, Any]):
    response = requests.post(url, json=request)
    response.raise_for_status()


PayloadGetter = Callable[[int], Dict[str, Any]]


def _run_warmup(
    url: str,
    payload_getter: PayloadGetter,
    max_tokens: int,
    pad_multiple: Optional[int] = None,
):
    logger.info("Sending warmup requests to %s...", url)
    if pad_multiple:
        for num_tokens in reversed(range(pad_multiple, max_tokens, pad_multiple)):
            _send_warmup_request(url, payload_getter(num_tokens))
    _send_warmup_request(url, payload_getter(max_tokens))
    logger.info("Finished sending warmup requests to %s.", url)


def launch_warmup_thread(model_name: str, payload_getter: PayloadGetter):
    """Launch a background thread that sends warmup requests."""
    max_tokens = utils.get_warmup_max_tokens()
    if not max_tokens:
        return

    port = utils.get_warmup_port()
    if not port:
        return

    url = f"http://localhost:{port}/predictions/{model_name}"
    pad_multiple = utils.get_pad_multiple()

    warmup_thread = threading.Thread(
        target=_run_warmup,
        args=(url, payload_getter, max_tokens, pad_multiple),
        daemon=True,  # killed at shutdown
    )
    warmup_thread.start()
