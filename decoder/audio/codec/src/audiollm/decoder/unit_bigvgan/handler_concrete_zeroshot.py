# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0
#
# Portions of this code are adapted from pytorch/serve
# Source: https://github.com/pytorch/serve
# License: Apache-2.0

"""
TorchServe handler for UnitBigVGANDecoder in zero-shot speaker settings.
"""

import base64
import binascii
import importlib
import io
import json
import logging
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pydub
import torch
from ts.torch_handler import base_handler

from audiollm.decoder.unit_bigvgan import utils, warmup
from audiollm.utils import audio_utils

logger = logging.getLogger(__name__)


_MIME = "audio/wav"


_Request = Tuple[List[int], bytes]
_InferenceItem = Tuple[torch.Tensor, torch.Tensor]


def _extract_payload(record: Dict[str, Any]) -> _Request:
    """
    Decode and parse the incoming request record to a JSON payload.
    """
    body = record.get("data") or record.get("body")
    if isinstance(body, (bytes, bytearray)):
        body = body.decode("utf-8")
    if isinstance(body, str):
        body = json.loads(body)

    units_list = body.get("unit")
    if not isinstance(units_list, list) or not all(
        isinstance(unit, int) for unit in units_list
    ):
        raise ValueError("Missing or invalid 'unit' field; must be a list of ints.")

    ref_audio = body.get("ref_audio")
    if not isinstance(ref_audio, str):
        raise ValueError("Missing 'ref_audio' field; must be a base64 string.")

    try:
        ref_audio_bytes = base64.b64decode(ref_audio.encode("ascii"), validate=True)
    except binascii.Error:
        raise ValueError("Invalid 'ref_audio' fields; must be a base64 string.")

    return units_list, ref_audio_bytes


def _get_warmup_payload(num_tokens: int):
    # dummy wav binary
    return {
        "unit": [1] * num_tokens,
        "ref_audio": "UklGRiwAAABXQVZFZm10IBAAAAABAAEACAAAAAgAAAABAAgAZGF0YQgAAAAwMTIzNDU2Nw==",
    }


def _get_reference_mel_spectrogram(ref_audio: bytes, h: dict) -> torch.Tensor:
    pcm = utils.load_reference_audio(ref_audio, h.sampling_rate)
    pcm = torch.from_numpy(pcm).unsqueeze(0)

    mel = audio_utils.compute_mel_spectrogram(
        pcm,
        h.n_fft,
        h.num_mels,
        h.sampling_rate,
        h.hop_size,
        h.win_size,
        h.fmin,
        h.fmax,
    )
    return mel


class ConcreteZSUnitBigVGANHandler(base_handler.BaseHandler):
    """
    TorchServe handler for the UnitBigVGANDecoder zero-shot model.

    Expects each request in the batch to be a JSON object with:
        - unit: List[int] (required)
        - ref_audio: Base64 str (required)
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

        module = importlib.import_module(model_file.split(".")[0])
        model_class_definitions = base_handler.list_classes_from_module(module)
        if len(model_class_definitions) != 1:
            raise ValueError(
                "Expected only one class as model definition. {}".format(
                    model_class_definitions
                )
            )

        model_class = model_class_definitions[0]
        model = model_class(model_pt_path, config_path)
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
            f"Initialized ConcreteZSUnitBigVGANHandler on {self.device}, vocab={self._vocab}"
        )

        warmup.launch_warmup_thread(context.model_name, _get_warmup_payload)

    def preprocess(self, data: List[Dict[str, Any]]) -> List[_InferenceItem]:
        """
        Convert raw input records to a list of (unit_tensor, ref_mel).

        Raises:
            ValueError: on invalid unit indices.
        """
        batch: List[_InferenceItem] = []

        for record in data:
            units, ref_audio = _extract_payload(record)

            units = torch.tensor(units, dtype=torch.long, device=self.device)

            # Validate indices if vocab size is known
            if self._vocab > 0:
                mask = (units < 0) | (units >= self._vocab)
                if mask.any():
                    bad_idxs = units[mask].tolist()
                    raise ValueError(
                        f"Unit indices out of range [0-{self._vocab - 1}]: {bad_idxs}"
                    )

            ref_mel = (
                _get_reference_mel_spectrogram(ref_audio, self.model.model.h)
                .to(self.device)
                .to(self._dtype)
            )

            batch.append((units, ref_mel))

        return batch

    @torch.inference_mode()
    def inference(
        self, batch: List[_InferenceItem], *args: Any, **kwargs: Any
    ) -> List[torch.Tensor]:
        """
        Run model inference on each (units, ref_mel) tuple.

        Returns:
            List of waveform_tensors.
        """
        del args, kwargs

        results: List[torch.Tensor] = []
        for units, ref_mel in batch:
            # Model returns tensor of shape (1, T)
            wav_tensor = self.model(units, ref_mel)
            results.append(wav_tensor.to(torch.float32))
        return results

    def postprocess(self, outputs: List[torch.Tensor]) -> List[bytes]:
        """
        Convert each waveform tensor to audio bytes.
        """
        responses: List[bytes] = []
        sample_rate = getattr(self.model, "SAMPLE_RATE", 24000)

        for wav_tensor in outputs:
            # Squeeze to (T,) and move to CPU
            wav = wav_tensor.squeeze().cpu().numpy()

            # Convert to int16 PCM samples
            pcm = (wav * 32767.0).astype(np.int16)

            # wrap PCM in a container via pydub
            segment = pydub.AudioSegment(
                pcm.tobytes(),
                frame_rate=sample_rate,
                sample_width=pcm.dtype.itemsize,
                channels=1,
            )

            buf = io.BytesIO()
            segment.export(buf, format="wav")
            responses.append(buf.getvalue())

        return responses

    def handle(self, data: List[Dict[str, Any]], context: Any) -> List[bytes]:
        """
        Override to set per-request MIME types, then return audio bytes.
        """
        output_bytes = super().handle(data, context)

        for idx, record in enumerate(data):
            del record
            context.set_response_content_type(idx, _MIME)

        return output_bytes
