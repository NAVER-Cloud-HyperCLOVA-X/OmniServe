# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import torch
import torch.nn as nn

from audiollm.decoder.unit_bigvgan import utils
from audiollm.decoder.unit_bigvgan.modules import BigVGAN

torch.backends.cudnn.benchmark = False


class ConcreteUnitbigvganDecoder(nn.Module):
    SAMPLE_RATE: int = 24000

    def __init__(self, ckpt_path: str, config_path: str, spk_list_path: str):
        super().__init__()
        self.model, self._spk_emb = self._load_model(
            ckpt_path=ckpt_path, config_path=config_path
        )
        with open(spk_list_path, "r") as f:
            speakers = [line.strip() for line in f]
        self.speaker_map = {spk: i for i, spk in enumerate(speakers)}

    def _load_model(self, ckpt_path: str, config_path: str):
        model = BigVGAN.from_pretrained(ckpt_path=ckpt_path, config_path=config_path)
        return model, model.spk_emb

    def compile(self):
        self.model = utils.compile(self.model)

    def forward(self, unit: torch.LongTensor, speaker: str = "fkms") -> torch.Tensor:
        assert len(unit.size()) < 3
        if len(unit.size()) == 2:
            assert unit.size(0) == 1, (
                "the underlying decoder does not support batch inference yet"
            )
        else:
            unit = unit.unsqueeze(0)

        # pad
        padded_unit, original_portion = utils.pad(unit)

        speaker_id = torch.tensor([self.speaker_map[speaker]], dtype=torch.long)
        speaker_id = speaker_id.unsqueeze(0).to(unit.device)
        spk_emb = self._spk_emb(speaker_id)

        # padded_out: shape [1, 1, T]
        padded_out, hidden = self.model(padded_unit, spk_emb)
        del hidden

        # unpad
        out = utils.unpad(padded_out, original_portion)
        return out
