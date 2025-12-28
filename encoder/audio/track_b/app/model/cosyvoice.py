# Portions of this code are adapted from:
# - CosyVoice (Mddct: Dinghao Zhou) - Apache-2.0
# - xingchensong/S3Tokenizer: https://github.com/xingchensong/S3Tokenizer (Apache-2.0)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from typing import Optional, Tuple

import librosa
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


DEFAULT_SAMPLE_RATE = 16000  # NOTE: 당분간 고정할 예정.


@dataclass
class ModelConfig:
    n_mels: int = 128
    n_audio_ctx: int = 1500
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 6
    n_codebook_size: int = 3**8

    use_sdpa: bool = True


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, scaling=None):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    if scaling is not None:
        t = t * scaling
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64

    return torch.cat((freqs_cis, freqs_cis), dim=-1)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    real = torch.view_as_real(freqs_cis)
    cos, sin = real[:, :, 0], real[:, :, 1]
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    D = xq.shape[-1]
    half_l, half_r = xq[:, :, :, : D // 2], xq[:, :, :, D // 2 :]
    xq_r = torch.cat((-half_r, half_l), dim=-1)

    D = xk.shape[-1]

    half_l, half_r = xk[:, :, :, : D // 2], xk[:, :, :, D // 2 :]
    xk_r = torch.cat((-half_r, half_l), dim=-1)

    return xq * cos + xq_r * sin, xk * cos + xk_r * sin


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class Linear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(
            x,
            self.weight.to(x.dtype),
            None if self.bias is None else self.bias.to(x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        return super()._conv_forward(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype)
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, n_state: int, n_head: int, use_sdpa: bool = False):
        super().__init__()
        self.n_head = n_head
        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

        self.use_sdpa = use_sdpa

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk = self.qkv_attention(q, k, v, mask)
        return self.out(wv), qk

    def qkv_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        _, _, D = q.shape
        scale = (D // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3) * scale
        k = k.view(*k.shape[:2], self.n_head, -1)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if not self.use_sdpa:
            k = k.permute(0, 2, 3, 1) * scale
            qk = q @ k  # (B, n_head, T, T)
            if mask is not None:
                qk = qk + mask
            qk = qk.float()
            w = torch.nn.functional.softmax(qk, dim=-1).to(q.dtype)
            return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach()
        else:
            k = k.permute(0, 2, 1, 3) * scale
            assert mask is not None
            output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, scale=1.0
            )
            output = (
                output.transpose(1, 2).contiguous().view(q.size(0), -1, D)
            )  # (batch, time1, d_model)
            return output, None


class FSQCodebook(torch.nn.Module):
    def __init__(self, dim: int, level: int = 3):
        super().__init__()
        self.project_down = torch.nn.Linear(dim, 8)
        self.level = level
        self.embed = None

    @torch.inference_mode()
    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "... d -> (...) d")
        return x

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_shape = x.shape
        # pre-process
        x = self.preprocess(x)
        # quantize
        h = self.project_down(x).float()
        h = h.tanh()
        h = h * 0.9990000128746033
        h = h.round() + 1
        # h = ((self.level - 1) * h).round()  # range [-k, k]
        powers = torch.pow(self.level, torch.arange(2**self.level, device=x.device, dtype=h.dtype))
        mu = torch.sum(h * powers.unsqueeze(0), dim=-1)
        ind = mu.reshape(x_shape[0], x_shape[1]).int()
        return ind

    @torch.inference_mode()
    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("There is no official up project component provided")


class FSQVectorQuantization(torch.nn.Module):
    """Vector quantization implementation (inference-only).
    Args:
        dim (int): Dimension
        codebook_size (int): Codebook size
    """

    def __init__(
        self,
        dim: int,
        codebook_size: int,
    ):
        super().__init__()
        assert 3**8 == codebook_size
        self._codebook = FSQCodebook(dim=dim, level=3)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._codebook.encode(x)

    @torch.inference_mode()
    def decode(self, embed_ind: torch.Tensor) -> torch.Tensor:
        quantize = self._codebook.decode(embed_ind)
        quantize = rearrange(quantize, "b n d -> b d n")
        return quantize


class FSMNMultiHeadAttention(MultiHeadAttention):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        kernel_size: int = 31,
        use_sdpa: bool = False,
    ):
        super().__init__(n_state, n_head)

        self.fsmn_block = torch.nn.Conv1d(
            n_state, n_state, kernel_size, stride=1, padding=0, groups=n_state, bias=False
        )
        self.left_padding = (kernel_size - 1) // 2
        self.right_padding = kernel_size - 1 - self.left_padding
        self.pad_fn = torch.nn.ConstantPad1d((self.left_padding, self.right_padding), 0.0)

        self.use_sdpa = use_sdpa

    def forward_fsmn(self, inputs: torch.Tensor, mask: Optional[torch.Tensor] = None):
        b, t, _, _ = inputs.size()
        inputs = inputs.view(b, t, -1)
        if mask is not None and mask.size(2) > 0:  # time2 > 0
            inputs = inputs * mask
        x = inputs.transpose(1, 2)
        x = self.pad_fn(x)
        x = self.fsmn_block(x)
        x = x.transpose(1, 2)
        x += inputs
        return x * mask

    def qkv_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_pad: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ):
        _, _, D = q.shape
        scale = (D // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1)
        k = k.view(*k.shape[:2], self.n_head, -1)
        v = v.view(*v.shape[:2], self.n_head, -1)

        if freqs_cis is not None:
            q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        fsm_memory = self.forward_fsmn(v, mask_pad)

        q = q.permute(0, 2, 1, 3) * scale
        v = v.permute(0, 2, 1, 3)

        if not self.use_sdpa:
            k = k.permute(0, 2, 3, 1) * scale
            qk = q @ k  # (B, n_head, T, T)
            if mask is not None:
                qk = qk + mask
            qk = qk.float()
            w = torch.nn.functional.softmax(qk, dim=-1).to(q.dtype)
            return (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2), qk.detach(), fsm_memory
        else:
            k = k.permute(0, 2, 1, 3) * scale
            assert mask is not None
            output = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                scale=1.0,
            )
            output = (
                output.transpose(1, 2).contiguous().view(q.size(0), -1, D)
            )  # (batch, time1, d_model)
            return output, None, fsm_memory

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_pad: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wv, qk, fsm_memory = self.qkv_attention(q, k, v, mask, mask_pad, freqs_cis)
        return self.out(wv) + fsm_memory, qk


class ResidualAttentionBlock(torch.nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        kernel_size: int = 31,
        use_sdpa: bool = False,
    ):
        super().__init__()

        self.attn = FSMNMultiHeadAttention(n_state, n_head, kernel_size, use_sdpa=use_sdpa)
        self.attn_ln = LayerNorm(n_state, eps=1e-6)

        n_mlp = n_state * 4

        self.mlp = torch.nn.Sequential(
            Linear(n_state, n_mlp), torch.nn.GELU(), Linear(n_mlp, n_state)
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mask_pad: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ):
        x = x + self.attn(self.attn_ln(x), mask=mask, mask_pad=mask_pad, freqs_cis=freqs_cis)[0]

        x = x + self.mlp(self.mlp_ln(x))
        return x


class AudioEncoderV2(torch.nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        stride: int,
        use_sdpa: bool,
    ):
        super().__init__()
        self.stride = stride

        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, stride=stride, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)
        self.freqs_cis = precompute_freqs_cis(64, 1024 * 2)
        self.blocks = torch.nn.ModuleList(
            [ResidualAttentionBlock(n_state, n_head, use_sdpa=use_sdpa) for _ in range(n_layer)]
        )

    def forward(self, x: torch.Tensor, x_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x : torch.Tensor, shape = (batch_size, n_mels, T)
            the mel spectrogram of the audio
        x_len: torch.Tensor, shape = (batch_size,)
            length of each audio in x
        """
        mask = self.make_non_pad_mask(x_len).unsqueeze(1)
        x = torch.nn.functional.gelu(self.conv1(x * mask))
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // self.stride + 1
        mask = self.make_non_pad_mask(x_len).unsqueeze(1)
        x = torch.nn.functional.gelu(self.conv2(x * mask))
        x_len = (x_len + 2 - 1 * (3 - 1) - 1) // 2 + 1
        mask = self.make_non_pad_mask(x_len).unsqueeze(1)
        x = x.permute(0, 2, 1)  # (B, T // 2, n_state)
        freqs_cis = self.freqs_cis.to(x.device)
        mask_pad = mask.transpose(1, 2)
        mask = self.mask_to_bias(mask, x.dtype)

        tmp = torch.view_as_real(freqs_cis)
        cos, sin = tmp[:, :, 0], tmp[:, :, 1]

        cos = torch.cat((cos, cos), dim=-1)
        sin = torch.cat((sin, sin), dim=-1)
        cos = cos.unsqueeze(0).unsqueeze(2)
        sin = sin.unsqueeze(0).unsqueeze(2)

        for block in self.blocks:
            x = block(x, mask.unsqueeze(1), mask_pad, freqs_cis[: x.size(1)])

        return x, x_len

    @staticmethod
    def make_non_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
        """Make mask tensor containing indices of non-padded part.
        The sequences in a batch may have different lengths. To enable
        batch computing, padding is need to make all sequence in same
        size. To avoid the padding part pass value to context dependent
        block such as attention or convolution , this padding part is
        masked.
        1 for non-padded part and 0 for padded part.
        Parameters
        ----------
            lengths (torch.Tensor): Batch of lengths (B,).
        Returns:
        -------
            torch.Tensor: Mask tensor containing indices of padded part (B, max_T).
        Examples:
            >>> import torch
            >>> import s3tokenizer
            >>> lengths = torch.tensor([5, 3, 2])
            >>> masks = s3tokenizer.make_non_pad_mask(lengths)
            masks = [[1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0],
                    [1, 1, 0, 0, 0]]
        """
        batch_size = lengths.size(0)
        max_len = max_len if max_len > 0 else lengths.max().item()
        seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
        seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
        seq_length_expand = lengths.unsqueeze(-1)
        mask = seq_range_expand >= seq_length_expand
        return ~mask

    @staticmethod
    def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """Convert bool-tensor to float-tensor for flash attention.
        Parameters
        ----------
            lengths (torch.Tensor): Batch of lengths (B, ?).
        Returns:
        -------
            torch.Tensor: Mask tensor containing indices of padded part (B, ?).
        Examples:
            >>> import torch
            >>> import s3tokenizer
            >>> lengths = torch.tensor([5, 3, 2])
            >>> masks = self.make_non_pad_mask(lengths)
            masks = [[1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 0],
                    [1, 1, 0, 0, 0]]
            >>> new_masks = self.mask_to_bias(masks, torch.float32)
            new_masks =
                [[-0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
                [-0.0000e+00, -0.0000e+00, -0.0000e+00, -1.0000e+10, -1.0000e+10],
                [-0.0000e+00, -0.0000e+00, -1.0000e+10, -1.0000e+10, -1.0000e+10]]
        """
        assert mask.dtype == torch.bool
        assert dtype in [torch.float32, torch.bfloat16, torch.float16]
        mask = mask.to(dtype)

        # attention mask bias
        # NOTE(Mddct): torch.finfo jit issues
        #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
        mask = (1.0 - mask) * -1.0e10
        return mask


class CosyvoiceEncoder(nn.Module):
    """S3 tokenizer of the CosyVoice2 implementation (inference-only).
    Args:
        config (ModelConfig): Config
    """

    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        self.encoder = AudioEncoderV2(
            self.config.n_mels,
            self.config.n_audio_state,
            self.config.n_audio_head,
            self.config.n_audio_layer,
            2,
            self.config.use_sdpa,
        )
        self.quantizer = FSQVectorQuantization(
            self.config.n_audio_state,
            self.config.n_codebook_size,
        )

    def forward(self, wav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mel = self.mel_spectrogram(wav, n_mels=self.config.n_mels)
        mel_len = torch.tensor([mel.shape[-1]]).to(self.device)
        return self.quantize(mel, mel_len)

    @torch.inference_mode()
    def quantize(
        self, mel: torch.Tensor, mel_len: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden, code_len = self.encoder(mel, mel_len)
        code = self.quantizer.encode(hidden)
        return code

    @staticmethod
    def mel_spectrogram(
        wav: torch.Tensor,
        n_mels: int = 80,
        padding: int = 0,
    ) -> torch.Tensor:
        """
        This method is based on the whisper.log_mel_spectrogram().
        So, don't use this as a general mel spectrogram function.
        """
        device = wav.device
        if padding > 0:
            wav = torch.nn.functional.pad(wav, (0, padding))

        window = torch.hann_window(400).to(device)
        stft = torch.stft(wav, 400, 160, window=window, return_complex=True)
        mag = stft[..., :-1].abs() ** 2

        filters = torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=400, n_mels=n_mels)).to(
            device
        )
        mel_spec = filters @ mag

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    @property
    def device(self):
        return next(self.parameters()).device

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    @classmethod
    def from_pretrained(
        cls, model_path: str = None
    ):
        import os
        if model_path is None:
            model_path = os.getenv("COSYVOICE_MODEL_PATH", "weights/cosyvoice_tokenizer.pt")
        model = cls()
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=True)
        model.eval()
        model.freeze()
        return model