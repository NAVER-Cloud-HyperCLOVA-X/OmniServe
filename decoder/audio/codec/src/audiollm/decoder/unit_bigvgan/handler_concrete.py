# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0
#
# Portions of this code are adapted from pytorch/serve
# Source: https://github.com/pytorch/serve
# License: Apache-2.0

"""
TorchServe handler for UnitBigVGANDecoder with runtime-selectable output formats.

Supported formats: mp3, wav, flac, ogg, aac, pcm.
For 'pcm', raw 16-bit little-endian PCM samples are returned (no container).
"""

import importlib
import io
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from pydub import AudioSegment
from ts.torch_handler import base_handler

from audiollm.decoder.unit_bigvgan import utils, warmup

_InferenceItem = Tuple[torch.Tensor, str, str]


logger = logging.getLogger(__name__)

# Map of supported audio formats to their MIME types
FORMAT_MIME_MAP: Dict[str, str] = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "ogg": "audio/ogg",
    "aac": "audio/aac",
    "pcm": "audio/pcm",
}
DEFAULT_FORMAT = "wav"


def _extract_payload(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decode and parse the incoming request record to a JSON payload.
    """
    body = record.get("data") or record.get("body")
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    if isinstance(body, str):
        return json.loads(body)
    return body  # assume it's already a dict


def _get_warmup_payload(num_tokens: int):
    return {"unit": [1] * num_tokens}


class ConcreteUnitBigVGANHandler(base_handler.BaseHandler):
    """
    TorchServe handler for the UnitBigVGANDecoder model.

    Expects each request in the batch to be a JSON object with:
        - unit: List[int] (required)
        - speaker: str      (optional, default 'fkms')
        - format: str       (optional, default 'mp3')
            Supported formats include 'pcm' to return raw PCM samples.
    """

    def _load_pickled_model(self, model_dir, model_file, model_pt_path):
        """
        Loads the pickle file from the given model path.

        Args:
            model_dir (str): Points to the location of the model artifacts.
            model_file (.py): the file which contains the model class.
            model_pt_path (str): points to the location of the model pickle file.

        Raises:
            RuntimeError: It raises this error when the model.py file is missing.

        Returns:
            serialized model file: Returns the pickled pytorch model file
        """
        model_def_path = os.path.join(model_dir, model_file)
        if not os.path.isfile(model_def_path):
            raise RuntimeError("Missing the model.py file")

        config_path = os.path.join(model_dir, "config.json")
        if not os.path.isfile(config_path):
            raise RuntimeError("Missing the config.json file")

        spk_list_path = os.path.join(model_dir, "spk_list.txt")
        if not os.path.isfile(spk_list_path):
            raise RuntimeError("Missing the spk_list.txt file")

        module = importlib.import_module(model_file.split(".")[0])
        model_class_definitions = base_handler.list_classes_from_module(module)
        if len(model_class_definitions) != 1:
            raise ValueError(
                "Expected only one class as model definition. {}".format(
                    model_class_definitions
                )
            )

        model_class = model_class_definitions[0]
        model = model_class(model_pt_path, config_path, spk_list_path)
        return model

    def initialize(self, context: Any) -> None:
        """
        Load model and verify its type. Cache vocabulary info.
        """
        super().initialize(context)

        self._dtype = utils.to_dtype(os.getenv("AUDIOLLM_DTYPE", "float32"))
        self.model.to(self._dtype)

        self.model.compile()

        # Retrieve vocabulary size (0 => skip validation)
        self._vocab = int(getattr(self.model.model.h, "num_units", 0))
        logger.info(
            f"Initialized ConcreteUnitBigVGANHandler on {self.device}, vocab={self._vocab}"
        )

        warmup.launch_warmup_thread(context.model_name, _get_warmup_payload)

    def preprocess(self, data: List[Dict[str, Any]]) -> List[_InferenceItem]:
        """
        Convert raw input records to a list of (unit_tensor, speaker, fmt).

        Raises:
            ValueError: on invalid unit indices or unsupported formats.
        """
        batch: List[_InferenceItem] = []

        for record in data:
            payload = _extract_payload(record)

            # Required: unit sequence
            units_list = payload.get("unit")
            if not isinstance(units_list, list):
                raise ValueError(
                    "Missing or invalid 'unit' field; must be a list of ints."
                )

            units = torch.tensor(units_list, dtype=torch.long, device=self.device)

            # Validate indices if vocab size is known
            if self._vocab > 0:
                mask = (units < 0) | (units >= self._vocab)
                if mask.any():
                    bad_idxs = units[mask].tolist()
                    raise ValueError(
                        f"Unit indices out of range [0-{self._vocab - 1}]: {bad_idxs}"
                    )

            # Optional: speaker and format
            speaker = str(payload.get("speaker", "fkms"))
            fmt = str(payload.get("format", DEFAULT_FORMAT)).lower()
            if fmt not in FORMAT_MIME_MAP:
                raise ValueError(
                    f"Unsupported format '{fmt}'. Choose from {list(FORMAT_MIME_MAP)}"
                )

            batch.append((units, speaker, fmt))

        return batch

    @torch.inference_mode()
    def inference(
        self, batch: List[_InferenceItem], *args: Any, **kwargs: Any
    ) -> List[Tuple[torch.Tensor, str]]:
        """
        Run model inference on each (units, speaker) tuple.

        Returns:
            List of (waveform_tensor, fmt) tuples.
        """
        del args, kwargs

        results: List[Tuple[torch.Tensor, str]] = []
        for units, speaker, fmt in batch:
            # Model returns tensor of shape (1, T)
            wav_tensor = self.model(units, speaker)
            results.append((wav_tensor.to(torch.float32), fmt))
        return results

    def postprocess(self, outputs: List[Tuple[torch.Tensor, str]]) -> List[bytes]:
        """
        Convert each waveform tensor to the requested audio format bytes.

        For 'pcm', returns raw little-endian int16 samples without container.
        """
        responses: List[bytes] = []
        sample_rate = getattr(self.model, "SAMPLE_RATE", 24000)

        for wav_tensor, fmt in outputs:
            # Squeeze to (T,) and move to CPU
            wav = wav_tensor.squeeze().cpu().numpy()

            # Convert to int16 PCM samples
            pcm = (wav * 32767.0).astype(np.int16)

            if fmt == "pcm":
                # raw PCM data, no headers
                responses.append(pcm.tobytes())
                continue

            # wrap PCM in a container via pydub
            segment = AudioSegment(
                pcm.tobytes(),
                frame_rate=sample_rate,
                sample_width=pcm.dtype.itemsize,
                channels=1,
            )

            buf = io.BytesIO()
            export_kwargs: Dict[str, Any] = {"format": fmt}
            if fmt == "mp3":
                export_kwargs["bitrate"] = "320k"

            segment.export(buf, **export_kwargs)
            responses.append(buf.getvalue())

        return responses

    def handle(self, data: List[Dict[str, Any]], context: Any) -> List[bytes]:
        """
        Override to set per-request MIME types, then return audio bytes.
        """
        output_bytes = super().handle(data, context)

        for idx, record in enumerate(data):
            payload = _extract_payload(record)
            fmt = str(payload.get("format", DEFAULT_FORMAT)).lower()
            mime = FORMAT_MIME_MAP.get(fmt, "application/octet-stream")
            context.set_response_content_type(idx, mime)

        return output_bytes
