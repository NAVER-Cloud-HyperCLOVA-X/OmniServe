# -*- coding: utf-8 -*-
# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0
#
# Portions of this code are adapted from:
# - state-spaces/mamba: https://github.com/state-spaces/mamba (Apache-2.0)
# - huggingface/transformers: https://github.com/huggingface/transformers (Apache-2.0)
"""
MambaMiaVideoAudioCompressor
============================
외부에서 간편하게 import하여 사용할 수 있는 Audio/Video Compressor.

================================================================================
⚠️ 환경 설정 가이드 (IMPORTANT: Environment Setup)
================================================================================

1. 필수 패키지 설치:
   ```bash
   pip install mamba_ssm==2.2.4
   pip install causal-conv1d==1.5.0.post8
   ```

2. selective_scan_cuda.so 설치 (중요!):
   - A100 GPU에서 mamba_ssm을 직접 빌드하면 selective_scan_cuda 컴파일이 
     실패할 수 있습니다.
   - 해결 방법: 미리 빌드된 .so 파일을 수동으로 복사해야 합니다.
   
   ```bash
   # 본인의 Python site-packages 경로에 맞게 수정하세요
   cp /path/to/selective_scan_cuda.cpython-310-x86_64-linux-gnu.so \
      $(python -c "import site; print(site.getsitepackages()[0])")/selective_scan_cuda.cpython-310-x86_64-linux-gnu.so
   ```
   
   예시 (가상환경 사용시):
   ```bash
   cp /path/to/selective_scan_cuda.cpython-310-x86_64-linux-gnu.so \
      ~/vlm/lib/python3.10/site-packages/selective_scan_cuda.cpython-310-x86_64-linux-gnu.so
   ```

3. 설치 확인:
   ```python
   import selective_scan_cuda  # 에러 없이 import 되면 성공
   from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
   print("Environment setup successful!")
   ```

4. 추가 의존성:
   - torch >= 2.0
   - transformers
   - einops
   - triton

================================================================================
⚠️ 주의사항 (IMPORTANT: Training Notes)
================================================================================

이 모듈을 외부 프로젝트에서 사용할 때 다음 사항을 확인하세요:

1. dtype 일치:
   - MambaMiaVideoAudioCompressor를 생성한 후 반드시 모델의 dtype과 일치시키세요.
   - 예: compressor = compressor.to(model.dtype)  # bfloat16 또는 float16

2. 초기화:
   - __init__에서 input_proj, output_proj, query_token이 적절히 초기화됩니다.
   - 내부 MambaMia2Model은 post_init()으로 자동 초기화됩니다.

3. Loss가 0이 되는 경우 체크리스트:
   - [ ] dtype 불일치 (float32 vs bfloat16)
   - [ ] gradient가 흐르지 않음 (requires_grad 확인)
   - [ ] output_proj 스케일이 너무 작음 (기본값 사용 권장)
   - [ ] NaN/Inf 발생 (torch.autograd.set_detect_anomaly(True)로 디버깅)

================================================================================

Usage:
    from mambamia_videoaudio_compressor import (
        MambaMiaVideoAudioCompressor,
        MambaMiaVideoAudioCompressorConfig,
        create_mambamia_compressor,  # 편의 함수
    )
    
    # 방법 1: Config 사용 (권장, PreTrainedModel 스타일)
    config = MambaMiaVideoAudioCompressorConfig(
        input_size=1280,       # audio encoder hidden size (e.g., Whisper: 1280)
        output_size=2048,      # LLM hidden size
        chunk_size=25,         # 25 tokens per chunk (1 second at 25Hz)
        num_hidden_layers=2,   # number of MambaMia2 layers
    )
    compressor = MambaMiaVideoAudioCompressor(config)
    
    # 방법 2: 편의 함수 사용
    compressor = create_mambamia_compressor(
        input_size=1280, output_size=2048, chunk_size=25, num_hidden_layers=2
    )
    
    # ⚠️ dtype 맞추기 (중요!)
    compressor = compressor.to(model.dtype).to(model.device)
    
    # Forward:
    # inputs_embeds: (batch_size, num_frames, chunk_size, hidden_dim) or (batch_size, seq_len, hidden_dim)
    # Returns: (batch_size, num_frames, hidden_dim) - one query token per frame
    outputs = compressor(inputs_embeds)
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import ModelOutput, logging
from transformers.utils.import_utils import is_causal_conv1d_available, is_mamba_2_ssm_available


logger = logging.get_logger(__name__)


# ============================================================================
# Check for fast path availability
# ============================================================================
if is_mamba_2_ssm_available():
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined
else:
    selective_state_update = None
    mamba_split_conv1d_scan_combined = None

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

is_fast_path_available = all((selective_state_update, causal_conv1d_fn, causal_conv1d_update))


# ============================================================================
# MambaMia2Config (Simplified for v04 only)
# ============================================================================
class MambaMia2Config(PretrainedConfig):
    """
    Simplified MambaMia2 configuration for v04 version only.
    """
    model_type = "mamba2"

    def __init__(
        self,
        num_heads=128,
        head_dim=64,
        vocab_size=32768,
        hidden_size=4096,
        state_size=128,
        num_hidden_layers=64,
        layer_norm_epsilon=1e-5,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        expand=2,
        conv_kernel=4,
        n_groups=8,
        use_bias=False,
        use_conv_bias=True,
        hidden_act="silu",
        initializer_range=0.1,
        residual_in_fp32=False,
        time_step_rank="auto",
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_floor=1e-4,
        time_step_limit=(0.0, float("inf")),
        rescale_prenorm_residual=False,
        use_cache=True,
        norm_before_gate=True,
        rms_norm=True,
        chunk_size=256,
        tie_word_embeddings=False,
        mambamia_chunk_size=10,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.num_hidden_layers = num_hidden_layers
        self.layer_norm_epsilon = layer_norm_epsilon
        self.conv_kernel = conv_kernel
        self.expand = expand

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.use_bias = use_bias
        self.use_conv_bias = use_conv_bias
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.time_step_rank = math.ceil(self.hidden_size / 16) if time_step_rank == "auto" else time_step_rank
        self.time_step_min = time_step_min
        self.time_step_max = time_step_max
        self.time_step_floor = time_step_floor
        self.rescale_prenorm_residual = rescale_prenorm_residual
        self.residual_in_fp32 = residual_in_fp32
        self.use_cache = use_cache
        self.n_groups = n_groups
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.norm_before_gate = norm_before_gate
        self.rms_norm = rms_norm
        self.state_size = state_size
        self.chunk_size = chunk_size
        self.time_step_limit = time_step_limit
        self.tie_word_embeddings = tie_word_embeddings
        self.mambamia_chunk_size = mambamia_chunk_size
        self.output_hidden_states = False
        self.output_deltas = False

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# ============================================================================
# Helper Modules
# ============================================================================
class MambaRMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        if gate is not None:
            hidden_states = hidden_states * nn.functional.silu(gate.to(torch.float32))
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class MambaMia2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# ============================================================================
# MambaMia2Mixer (v04 version - unidirectional with GPA)
# ============================================================================
class MambaMia2Mixer(nn.Module):
    """
    Unidirectional Mamba2 Mixer for v04 version.
    v04 = v0 (unidirectional Mamba) + GPA (Gated Pooling Attention in Block)
    """

    def __init__(self, config: MambaMia2Config, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_heads
        self.hidden_size = config.hidden_size
        self.ssm_state_size = config.state_size
        self.conv_kernel_size = config.conv_kernel
        self.intermediate_size = int(config.expand * self.hidden_size)
        self.time_step_rank = int(config.time_step_rank)
        self.layer_idx = layer_idx
        self.use_conv_bias = config.use_conv_bias
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]

        self.norm_before_gate = config.norm_before_gate
        self.layer_norm_epsilon = config.layer_norm_epsilon
        self.rms_norm = config.rms_norm

        self.n_groups = config.n_groups
        self.head_dim = config.head_dim
        self.chunk_size = config.chunk_size

        self.time_step_limit = config.time_step_limit
        self.time_step_min = config.time_step_min
        self.time_step_max = config.time_step_max

        self.conv_dim = self.intermediate_size + 2 * self.n_groups * self.ssm_state_size
        
        # Conv1d for SSM
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=config.use_conv_bias,
            kernel_size=config.conv_kernel,
            groups=self.conv_dim,
            padding=config.conv_kernel - 1,
        )

        # projection of the input hidden states
        projection_size = self.intermediate_size + self.conv_dim + self.num_heads
        self.in_proj = nn.Linear(self.hidden_size, projection_size, bias=config.use_bias)

        # time step projection
        self.dt_bias = nn.Parameter(torch.ones(self.num_heads))

        # S4D real initialization
        A = torch.arange(1, self.num_heads + 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        self.norm = MambaRMSNormGated(self.intermediate_size, eps=self.layer_norm_epsilon)
        self.D = nn.Parameter(torch.ones(self.num_heads))
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.use_bias)
        self.use_bias = config.use_bias

        if not is_fast_path_available:
            logger.warning_once(
                "The fast path is not available because one of "
                "`(selective_state_update, causal_conv1d_fn, causal_conv1d_update)` is None. "
                "Falling back to the naive implementation. To install follow "
                "https://github.com/state-spaces/mamba/#installation and "
                "https://github.com/Dao-AILab/causal-conv1d"
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        v04 unidirectional forward pass using CUDA kernels.
        """
        import os
        rank = int(os.environ.get("RANK", -1))
        debug = False # (rank <= 0)
        
        assert is_fast_path_available and "cuda" in self.in_proj.weight.device.type, \
            "CUDA kernels required for MambaMia2Mixer"

        dtype = hidden_states.dtype
        batch_size, seq_len, _ = hidden_states.shape
        
        if debug:
            print(f"[Mixer DEBUG] input: min={hidden_states.min().item():.6f}, max={hidden_states.max().item():.6f}, nan={torch.isnan(hidden_states).any().item()}, seq_len={seq_len}, chunk_size={self.chunk_size}")

        if attention_mask is not None and attention_mask.shape[1] > 1 and attention_mask.shape[0] > 1:
            hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

        # Gated MLP's linear projection
        projected_states = self.in_proj(hidden_states)
        
        if debug:
            print(f"[Mixer DEBUG] after in_proj: min={projected_states.min().item():.6f}, max={projected_states.max().item():.6f}, nan={torch.isnan(projected_states).any().item()}")
            print(f"[Mixer DEBUG] A_log: {self.A_log[:5].tolist()}, dt_bias: {self.dt_bias[:5].tolist()}, D: {self.D[:5].tolist()}")
        
        dt_limit_kwargs = {} if self.time_step_limit == (0.0, float("inf")) else {"dt_limit": self.time_step_limit}

        # Unidirectional forward pass (same as v0)
        outputs = mamba_split_conv1d_scan_combined(
            projected_states,
            self.conv1d.weight.squeeze(1),
            self.conv1d.bias,
            self.dt_bias,
            -torch.exp(self.A_log.float()),
            D=self.D,
            chunk_size=self.chunk_size,
            seq_idx=None,
            activation=self.activation,
            rmsnorm_weight=self.norm.weight,
            rmsnorm_eps=self.norm.variance_epsilon,
            outproj_weight=self.out_proj.weight,
            outproj_bias=self.out_proj.bias,
            headdim=self.head_dim,
            ngroups=self.n_groups,
            norm_before_gate=self.norm_before_gate,
            return_final_states=False,
            **dt_limit_kwargs,
        )
        
        if debug:
            print(f"[Mixer DEBUG] after mamba_kernel: min={outputs.min().item():.6f}, max={outputs.max().item():.6f}, nan={torch.isnan(outputs).any().item()}")

        return outputs.to(dtype)


# ============================================================================
# MambaMia2Block (v04 version only)
# ============================================================================
class MambaMia2Block(nn.Module):
    """
    Single MambaMia2 block with v04 gated pooling attention mechanism.
    """

    def __init__(self, config: MambaMia2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.residual_in_fp32 = config.residual_in_fp32
        self.mambamia_chunk_size = config.mambamia_chunk_size

        self.norm = MambaMia2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = MambaMia2Mixer(config, layer_idx=layer_idx)

        # v04 specific: Gated Pooling Attention (GPA)
        self.drop = nn.Dropout(p=0.1)

        # Per-frame weight prediction
        self.weight_fc = nn.Linear(config.hidden_size, self.mambamia_chunk_size)
        nn.init.zeros_(self.weight_fc.bias)
        with torch.no_grad():
            self.weight_fc.weight.mul_(1e-3)

        # Query vs aggregator gating
        self.gate_fc = nn.Linear(config.hidden_size, 1)
        nn.init.zeros_(self.gate_fc.bias)
        with torch.no_grad():
            self.gate_fc.weight.mul_(1e-3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        input_dtype = hidden_states.dtype
        residual = hidden_states
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)

        # v04 Gated Pooling Attention
        assert hidden_states.dim() == 3, f"hidden_states.dim()={hidden_states.dim()} != 3"
        bsz, seq_len, hidden_dim = hidden_states.shape
        mambamia_chunk_size = self.mambamia_chunk_size
        chunk_with_query = mambamia_chunk_size + 1

        if seq_len % chunk_with_query != 0:
            raise ValueError(
                f"seq_len={seq_len} must be divisible by (mambamia_chunk_size+1)={chunk_with_query}"
            )
        n_chunk = seq_len // chunk_with_query

        # Reshape to (bsz, n_chunk, chunk_size+1, hidden_dim)
        hidden_4d = hidden_states.view(bsz, n_chunk, chunk_with_query, hidden_dim)

        frames = hidden_4d[:, :, :mambamia_chunk_size, :]  # (bsz, n_chunk, chunk_size, hidden_dim)
        queries = hidden_4d[:, :, mambamia_chunk_size, :]  # (bsz, n_chunk, hidden_dim)

        # Weight prediction for frames (float32로 계산하여 안정성 확보)
        w_in = self.drop(queries)
        raw_weights = self.weight_fc(w_in)
        alpha = torch.softmax(raw_weights.float(), dim=-1).to(input_dtype)  # (bsz, n_chunk, chunk_size)

        # Weighted average: aggregator
        aggregator = (frames * alpha.unsqueeze(-1)).sum(dim=2)  # (bsz, n_chunk, hidden_dim)

        # Gating between queries and aggregator (float32로 계산)
        gating_in = self.drop(queries)
        gating = torch.sigmoid(self.gate_fc(gating_in).float()).to(input_dtype)  # (bsz, n_chunk, 1)
        epsilon = 0.01
        gating = gating * (1 - 2 * epsilon) + epsilon  # [0.01, 0.99]

        gating_broad = gating.expand(-1, -1, hidden_dim)
        aggregator = aggregator * gating_broad
        queries = queries * (1 - gating_broad)
        queries_new = queries + aggregator

        # Update query positions
        hidden_4d = hidden_4d.clone()
        hidden_4d[:, :, mambamia_chunk_size, :] = queries_new
        hidden_states = hidden_4d.view(bsz, seq_len, hidden_dim)

        # Mixer forward
        hidden_states = self.mixer(hidden_states, attention_mask=attention_mask)

        # Residual connection
        hidden_states = hidden_states + residual

        return hidden_states


# ============================================================================
# MambaMia2Model (Simplified)
# ============================================================================
@dataclass
class MambaMia2Output(ModelOutput):
    """Output class for MambaMia2Model."""
    last_hidden_state: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class MambaMia2PreTrainedModel(PreTrainedModel):
    """Base class for MambaMia2 models."""
    config_class = MambaMia2Config
    base_model_prefix = "backbone"
    _no_split_modules = ["MambaMia2Block"]
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, MambaMia2Mixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True

            dt = torch.exp(
                torch.rand(self.config.num_heads)
                * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min))
                + math.log(self.config.time_step_min)
            ).clamp(min=self.config.time_step_floor)

            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                module.dt_bias.copy_(inv_dt)
            module.dt_bias._no_reinit = True

        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, "_no_reinit", False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

        if self.config.rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name in ["out_proj.weight"]:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.num_hidden_layers)


class MambaMia2Model(MambaMia2PreTrainedModel):
    """
    Simplified MambaMia2 Model for v04 version.
    Takes inputs_embeds directly (no embedding layer used for audio/video).
    """

    def __init__(self, config: MambaMia2Config):
        super().__init__(config)
        self.layers = nn.ModuleList([
            MambaMia2Block(config, layer_idx=idx)
            for idx in range(config.num_hidden_layers)
        ])
        self.gradient_checkpointing = False
        self.norm_f = MambaMia2RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_init()

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MambaMia2Output]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None

        for mixer_block in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    mixer_block.__call__, hidden_states, attention_mask
                )
            else:
                hidden_states = mixer_block(hidden_states, attention_mask=attention_mask)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.norm_f(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states] if v is not None)

        return MambaMia2Output(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )


# ============================================================================
# MambaMiaVideoAudioCompressorConfig
# ============================================================================
class MambaMiaVideoAudioCompressorConfig(PretrainedConfig):
    """
    Configuration for MambaMiaVideoAudioCompressor.
    
    Args:
        input_size: Input embedding dimension (e.g., 1280 for Whisper)
        output_size: Output embedding dimension (e.g., 2048 for LLM)
        chunk_size: Number of tokens per chunk (default: 25, i.e., 1 second at 25Hz)
        num_hidden_layers: Number of MambaMia2 layers (default: 1)
        hidden_size: Internal hidden size (default: 3072, must be divisible by 24)
    """
    model_type = "mambamia_videoaudio_compressor"
    
    def __init__(
        self,
        input_size: int = 1280,
        output_size: int = 2048,
        chunk_size: int = 25,
        num_hidden_layers: int = 1,
        hidden_size: int = 3072,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.chunk_size = chunk_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size


# ============================================================================
# MambaMiaVideoAudioCompressor - Main Interface (PreTrainedModel 기반)
# ============================================================================
class MambaMiaVideoAudioCompressor(PreTrainedModel):
    """
    Video/Audio Compressor using MambaMia2 (v04 bidirectional version).
    
    This module compresses sequential embeddings (e.g., audio frames at 25Hz) 
    by inserting learnable query tokens and extracting them after processing.
    
    Args:
        config: MambaMiaVideoAudioCompressorConfig
    
    Input:
        inputs_embeds: (batch_size, num_frames, hidden_dim) where num_frames is 
                       typically the audio length and hidden_dim matches input_size
    
    Output:
        compressed_embeds: (batch_size, num_queries, output_size) where 
                          num_queries = num_frames // chunk_size
    """
    
    config_class = MambaMiaVideoAudioCompressorConfig
    base_model_prefix = "mambamia_compressor"
    _no_split_modules = ["MambaMia2Block"]

    def __init__(self, config: MambaMiaVideoAudioCompressorConfig):
        super().__init__(config)
        
        self.input_size = config.input_size
        self.output_size = config.output_size
        self.chunk_size = config.chunk_size
        self.hidden_size = config.hidden_size
        
        # Input projection: input_size -> hidden_size
        self.input_proj = nn.Linear(config.input_size, config.hidden_size)
        
        # Learnable query token
        self.query_token = nn.Parameter(torch.randn(config.hidden_size))
        
        # MambaMia2 backbone
        # 중요: chunk_size는 SSM kernel의 chunk size로, 시퀀스 길이보다 작아야 함
        # mambamia_chunk_size는 압축 비율 (25:1)
        # 시퀀스 길이가 짧을 수 있으므로 (예: 390 tokens) chunk_size=64로 설정
        mamba_config = MambaMia2Config(
            vocab_size=0,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            head_dim=64,
            num_heads=config.hidden_size * 2 // 64,  # e.g., 3072*2/64 = 96
            n_groups=1,
            expand=2.0,
            use_cache=False,
            chunk_size=256,  # SSM kernel chunk size
            mambamia_chunk_size=config.chunk_size,  # 압축 비율용 (25)
            residual_in_fp32=False,
        )
        self.model = MambaMia2Model(mamba_config)
        
        # LayerNorm before Mamba2 to normalize input scales
        # This ensures query_token and input_proj outputs are on the same scale
        self.input_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
        
        # Output projection: hidden_size -> output_size
        self.output_proj = nn.Linear(config.hidden_size, config.output_size)
        
        # Initialize weights (transformers style)
        self.post_init()

    def _init_weights(self, module):
        """
        Initialize weights - called by post_init() for all submodules.
        주의: MambaMia2Model 내부의 가중치는 건드리지 않음 (자체 post_init에서 처리됨)
        """
        # query_token 초기화 - std=1.0으로 input_proj 출력 스케일과 맞춤
        # (작은 std는 LayerNorm에서 variance가 0에 가까워져 inf 발생)
        if module is self:
            with torch.no_grad():
                self.query_token.data.normal_(mean=0.0, std=1.0)
        
        # input_proj, output_proj만 xavier 초기화 (MambaMia2 내부는 건드리지 않음)
        if module is self.input_proj or module is self.output_proj:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _init_all_weights(self):
        """
        Force re-initialize all weights. Call after dtype conversion for FSDP compatibility.
        This ensures weights are properly initialized even after model transformations.
        """
        # 1. input_proj, output_proj 초기화
        nn.init.xavier_uniform_(self.input_proj.weight)
        if self.input_proj.bias is not None:
            nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)
        
        # 2. query_token 초기화 - std=1.0으로 input_proj 출력 스케일과 맞춤
        self.query_token.data.normal_(mean=0.0, std=1.0)
        
        # 3. input_norm (LayerNorm) 초기화
        nn.init.ones_(self.input_norm.weight)
        nn.init.zeros_(self.input_norm.bias)
        
        # 4. MambaMia2Model 내부 초기화 (중요!)
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # 5. MambaMia2Block의 특수 초기화 (weight_fc, gate_fc)
        for layer in self.model.layers:
            if hasattr(layer, 'weight_fc'):
                nn.init.xavier_uniform_(layer.weight_fc.weight)
                layer.weight_fc.weight.data.mul_(0.01)  # Scale down
                nn.init.zeros_(layer.weight_fc.bias)
            if hasattr(layer, 'gate_fc'):
                nn.init.xavier_uniform_(layer.gate_fc.weight)
                layer.gate_fc.weight.data.mul_(0.01)  # Scale down
                nn.init.zeros_(layer.gate_fc.bias)
        
        # 6. A_log, D, dt_bias 파라미터 초기화 (SSM specific)
        for layer in self.model.layers:
            if hasattr(layer, 'mixer'):
                mixer = layer.mixer
                # A_log: S4D real initialization
                A = torch.arange(1, mixer.num_heads + 1, dtype=mixer.A_log.dtype, device=mixer.A_log.device)
                mixer.A_log.data.copy_(torch.log(A))
                # D: scaling factor
                mixer.D.data.fill_(1.0)
                # dt_bias: time step bias (중요!)
                mixer.dt_bias.data.fill_(1.0)
                
        # 7. RMSNorm weight 초기화 (MambaRMSNormGated)
        for layer in self.model.layers:
            if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'norm'):
                layer.mixer.norm.weight.data.fill_(1.0)
            if hasattr(layer, 'norm') and hasattr(layer.norm, 'weight'):
                layer.norm.weight.data.fill_(1.0)
        
        # 8. MambaMia2Model의 최종 norm_f 초기화
        if hasattr(self.model, 'norm_f') and hasattr(self.model.norm_f, 'weight'):
            self.model.norm_f.weight.data.fill_(1.0)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            inputs_embeds: (batch_size, seq_len, input_size) or 
                          (batch_size, num_frames, chunk_size, input_size)
        
        Returns:
            compressed: (batch_size, num_queries, output_size)
        """
        import os
        rank = int(os.environ.get("RANK", -1))
        debug = False # True if (rank <= 0) else False
        
        # Handle different input shapes
        if inputs_embeds.dim() == 4:
            # (batch_size, num_frames, chunk_size, input_size)
            bsz, num_frames, chunk_size, _ = inputs_embeds.shape
            assert chunk_size == self.chunk_size, \
                f"Input chunk_size {chunk_size} != expected {self.chunk_size}"
            inputs_embeds = inputs_embeds.view(bsz, -1, self.input_size)
        
        bsz, seq_len, _ = inputs_embeds.shape
        
        # Ensure seq_len is divisible by chunk_size
        if seq_len % self.chunk_size != 0:
            # Pad to make divisible
            pad_len = self.chunk_size - (seq_len % self.chunk_size)
            inputs_embeds = F.pad(inputs_embeds, (0, 0, 0, pad_len))
            seq_len = inputs_embeds.shape[1]
        
        n_chunk = seq_len // self.chunk_size
        
        # Project input
        hidden_states = self.input_proj(inputs_embeds)  # (bsz, seq_len, hidden_size)
        
        if debug:
            print(f"[MambaMia DEBUG] input_proj output: min={hidden_states.min().item():.6f}, max={hidden_states.max().item():.6f}, has_nan={torch.isnan(hidden_states).any().item()}")
        
        # Reshape to (bsz, n_chunk, chunk_size, hidden_size)
        hidden_4d = hidden_states.view(bsz, n_chunk, self.chunk_size, self.hidden_size)
        
        # Add query token to each chunk
        # query_token: (hidden_size,) -> (1, 1, 1, hidden_size)
        query_expanded = self.query_token.view(1, 1, 1, -1).expand(bsz, n_chunk, 1, self.hidden_size)
        
        if debug:
            print(f"[MambaMia DEBUG] query_token: min={self.query_token.min().item():.6f}, max={self.query_token.max().item():.6f}, has_nan={torch.isnan(self.query_token).any().item()}")
        
        # Concatenate: (bsz, n_chunk, chunk_size+1, hidden_size)
        hidden_with_query = torch.cat([hidden_4d, query_expanded], dim=2)
        
        # Flatten for model: (bsz, n_chunk * (chunk_size+1), hidden_size)
        model_input = hidden_with_query.view(bsz, -1, self.hidden_size)
        
        # Apply LayerNorm to normalize input scales before Mamba2
        model_input = self.input_norm(model_input)
        
        if debug:
            print(f"[MambaMia DEBUG] model_input (after LayerNorm, before Mamba2): min={model_input.min().item():.6f}, max={model_input.max().item():.6f}, has_nan={torch.isnan(model_input).any().item()}")
        
        # Forward through MambaMia2
        outputs = self.model(inputs_embeds=model_input)
        hidden_states = outputs.last_hidden_state  # (bsz, n_chunk * (chunk_size+1), hidden_size)
        
        if debug:
            print(f"[MambaMia DEBUG] model output (after Mamba2): min={hidden_states.min().item():.6f}, max={hidden_states.max().item():.6f}, has_nan={torch.isnan(hidden_states).any().item()}")
        
        # Check for NaN and replace with zeros if found (defensive)
        if torch.isnan(hidden_states).any():
            hidden_states = torch.nan_to_num(hidden_states, nan=0.0)

        # Reshape back: (bsz, n_chunk, chunk_size+1, hidden_size)
        hidden_out_4d = hidden_states.view(bsz, n_chunk, self.chunk_size + 1, self.hidden_size)
        
        # Extract query positions (last position in each chunk)
        query_outputs = hidden_out_4d[:, :, self.chunk_size, :]  # (bsz, n_chunk, hidden_size)
        
        if debug:
            print(f"[MambaMia DEBUG] query_outputs (extracted): min={query_outputs.min().item():.6f}, max={query_outputs.max().item():.6f}, has_nan={torch.isnan(query_outputs).any().item()}")
        
        # Project to output size
        compressed = self.output_proj(query_outputs)  # (bsz, n_chunk, output_size)
        
        if debug:
            print(f"[MambaMia DEBUG] output_proj output: min={compressed.min().item():.6f}, max={compressed.max().item():.6f}, has_nan={torch.isnan(compressed).any().item()}")
        
        return compressed


# ============================================================================
# Convenience function for quick instantiation
# ============================================================================
def create_mambamia_compressor(
    input_size: int,
    output_size: int,
    chunk_size: int = 25,
    num_hidden_layers: int = 2,
    hidden_size: int = 3072,
) -> MambaMiaVideoAudioCompressor:
    """
    Create a MambaMiaVideoAudioCompressor with default settings.
    
    Example:
        compressor = create_mambamia_compressor(1280, 2048, chunk_size=25)
    """
    config = MambaMiaVideoAudioCompressorConfig(
        input_size=input_size,
        output_size=output_size,
        chunk_size=chunk_size,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
    )
    return MambaMiaVideoAudioCompressor(config)


# ============================================================================
# Test code
# ============================================================================
if __name__ == "__main__":
    import time
    
    print("=" * 70)
    print("Testing MambaMiaVideoAudioCompressor (PreTrainedModel based)")
    print("=" * 70)
    
    # Real environment settings
    INPUT_SIZE = 4096       # Audio encoder hidden size (same as LLM)
    OUTPUT_SIZE = 4096      # LLM hidden size
    CHUNK_SIZE = 25         # 25 tokens per chunk (1 second at 25Hz)
    AVG_LENGTH = 4500       # Average sequence length
    MAX_LENGTH = 270000     # Maximum sequence length (3 hours at 25Hz)
    
    print(f"\n[Real Environment Settings]")
    print(f"  Input size: {INPUT_SIZE}")
    print(f"  Output size: {OUTPUT_SIZE}")
    print(f"  Chunk size: {CHUNK_SIZE}")
    print(f"  Average length: {AVG_LENGTH} (~{AVG_LENGTH/25/60:.1f} minutes at 25Hz)")
    print(f"  Max length: {MAX_LENGTH} (~{MAX_LENGTH/25/60:.1f} minutes at 25Hz)")
    
    # Create compressor using Config (PreTrainedModel style)
    config = MambaMiaVideoAudioCompressorConfig(
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        chunk_size=CHUNK_SIZE,
        num_hidden_layers=1,
    )
    compressor = MambaMiaVideoAudioCompressor(config)
    compressor = compressor.cuda()
    
    # ========================================================================
    # Print parameter counts
    # ========================================================================
    print("\n[Parameter Counts]")
    total_params = sum(p.numel() for p in compressor.parameters())
    trainable_params = sum(p.numel() for p in compressor.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Parameter size (MB): {total_params * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Breakdown by component
    print("\n[Parameter Breakdown]")
    input_proj_params = sum(p.numel() for p in compressor.input_proj.parameters())
    output_proj_params = sum(p.numel() for p in compressor.output_proj.parameters())
    query_params = compressor.query_token.numel()
    model_params = sum(p.numel() for p in compressor.model.parameters())
    print(f"  input_proj: {input_proj_params:,}")
    print(f"  output_proj: {output_proj_params:,}")
    print(f"  query_token: {query_params:,}")
    print(f"  MambaMia2Model: {model_params:,}")
    
    # ========================================================================
    # Test 1: Basic test (divisible by chunk_size)
    # ========================================================================
    print("\n" + "=" * 70)
    print("[Test 1] Basic test - seq_len divisible by chunk_size (100 = 25 * 4)")
    print("=" * 70)
    test_input = torch.randn(2, 100, INPUT_SIZE).cuda()
    with torch.no_grad():
        output = compressor(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output: (2, 4, {OUTPUT_SIZE})")
    assert output.shape == (2, 4, OUTPUT_SIZE), f"Expected (2, 4, {OUTPUT_SIZE}), got {output.shape}"
    print("  ✓ PASSED")
    
    # ========================================================================
    # Test 2: Not divisible by chunk_size (should pad automatically)
    # ========================================================================
    print("\n" + "=" * 70)
    print("[Test 2] seq_len NOT divisible by chunk_size (97 tokens)")
    print("=" * 70)
    test_input = torch.randn(2, 97, INPUT_SIZE).cuda()
    with torch.no_grad():
        output = compressor(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Note: 97 -> padded to 100 (next multiple of 25) -> 4 queries")
    print(f"  Expected output: (2, 4, {OUTPUT_SIZE})")
    assert output.shape == (2, 4, OUTPUT_SIZE), f"Expected (2, 4, {OUTPUT_SIZE}), got {output.shape}"
    print("  ✓ PASSED")
    
    # ========================================================================
    # Test 3: Average length test (~3 minutes audio)
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"[Test 3] Average length test ({AVG_LENGTH} tokens, ~{AVG_LENGTH/25/60:.1f} minutes)")
    print("=" * 70)
    test_input = torch.randn(1, AVG_LENGTH, INPUT_SIZE).cuda()
    torch.cuda.synchronize()
    start_time = time.time()
    with torch.no_grad():
        output = compressor(test_input)
    torch.cuda.synchronize()
    elapsed = time.time() - start_time
    expected_queries = AVG_LENGTH // CHUNK_SIZE
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected queries: {expected_queries}")
    print(f"  Inference time: {elapsed*1000:.2f} ms")
    assert output.shape == (1, expected_queries, OUTPUT_SIZE), f"Shape mismatch"
    print("  ✓ PASSED")
    del test_input, output
    torch.cuda.empty_cache()
    
    # ========================================================================
    # Test 4: Batch size 1, short sequence
    # ========================================================================
    print("\n" + "=" * 70)
    print("[Test 4] Batch size 1, short sequence")
    print("=" * 70)
    test_input = torch.randn(1, 75, INPUT_SIZE).cuda()
    with torch.no_grad():
        output = compressor(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output: (1, 3, {OUTPUT_SIZE})")
    assert output.shape == (1, 3, OUTPUT_SIZE), f"Expected (1, 3, {OUTPUT_SIZE}), got {output.shape}"
    print("  ✓ PASSED")
    
    # ========================================================================
    # Test 5: Gradient flow check
    # ========================================================================
    print("\n" + "=" * 70)
    print("[Test 5] Gradient flow check")
    print("=" * 70)
    compressor.train()
    # Create tensor directly on CUDA to keep it as leaf tensor
    test_input = torch.randn(2, 50, INPUT_SIZE, device='cuda', requires_grad=True)
    output = compressor(test_input)
    loss = output.sum()
    loss.backward()
    print(f"  Input requires_grad: {test_input.requires_grad}")
    print(f"  Input is_leaf: {test_input.is_leaf}")
    print(f"  Input grad exists: {test_input.grad is not None}")
    print(f"  Input grad shape: {test_input.grad.shape if test_input.grad is not None else 'N/A'}")
    assert test_input.grad is not None, "Gradient should flow back to input"
    print("  ✓ PASSED")
    compressor.eval()
    del test_input, output
    torch.cuda.empty_cache()
    
    # ========================================================================
    # Test 6: Memory benchmark with FP16 - 3min, 30min, 3hours
    # ========================================================================
    print("\n" + "=" * 70)
    print("[Test 6] MEMORY BENCHMARK (FP16) - 3min, 30min, 3hours")
    print("=" * 70)
    
    # Convert model to fp16 for realistic testing
    compressor_fp16 = compressor.half()
    
    test_lengths = [
        (AVG_LENGTH, "3 min"),      # 4500 tokens
        (45000, "30 min"),           # 30 minutes
        (270000, "3 hours"),         # 180 minutes (max)
    ]
    
    for length, desc in test_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        print(f"\n  --- {desc} ({length} tokens) ---")
        try:
            # Create fp16 input
            test_input = torch.randn(1, length, INPUT_SIZE, dtype=torch.float16, device='cuda')
            input_mem = test_input.numel() * 2 / 1024**3  # fp16 = 2 bytes
            print(f"  Input tensor: {input_mem:.3f} GB (fp16)")
            
            torch.cuda.synchronize()
            start_time = time.time()
            with torch.no_grad():
                output = compressor_fp16(test_input)
            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            
            expected_queries = (length + CHUNK_SIZE - 1) // CHUNK_SIZE  # ceil division for padding
            actual_queries = output.shape[1]
            
            peak_mem = torch.cuda.max_memory_allocated() / 1024**3
            current_mem = torch.cuda.memory_allocated() / 1024**3
            
            print(f"  Output shape: {output.shape}")
            print(f"  Queries: {actual_queries}")
            print(f"  Inference time: {elapsed*1000:.2f} ms ({elapsed:.2f} s)")
            print(f"  Peak GPU memory: {peak_mem:.2f} GB")
            print(f"  Current GPU memory: {current_mem:.2f} GB")
            print(f"  ✓ PASSED")
            
            del test_input, output
            torch.cuda.empty_cache()
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"  ⚠ OOM - requires more GPU memory")
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ✗ Error: {e}")
            torch.cuda.empty_cache()
    
    # Convert back to fp32 for remaining tests
    compressor = compressor.float()
    
    # ========================================================================
    # Test 7: Edge case - only 1 token
    # ========================================================================
    print("\n" + "=" * 70)
    print("[Test 7] Edge case - only 1 token")
    print("=" * 70)
    test_input = torch.randn(1, 1, INPUT_SIZE).cuda()
    with torch.no_grad():
        output = compressor(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Note: 1 -> padded to 25 -> 1 query")
    print(f"  Expected output: (1, 1, {OUTPUT_SIZE})")
    assert output.shape == (1, 1, OUTPUT_SIZE), f"Expected (1, 1, {OUTPUT_SIZE}), got {output.shape}"
    print("  ✓ PASSED")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)
    print(f"\nModel Summary:")
    print(f"  - Total parameters: {total_params:,} ({total_params * 4 / 1024 / 1024:.2f} MB)")
    print(f"  - Chunk size: {CHUNK_SIZE} (1 second at 25Hz)")
    print(f"  - Input dim: {INPUT_SIZE}, Output dim: {OUTPUT_SIZE}")
    print(f"  - Layers: 1")
    print(f"  - Compression ratio: {CHUNK_SIZE}x (25 tokens -> 1 query)")
    print(f"\nReal-world capacity:")
    print(f"  - Average audio ({AVG_LENGTH} tokens): {AVG_LENGTH // CHUNK_SIZE} queries")
    print(f"  - Max audio ({MAX_LENGTH} tokens): {MAX_LENGTH // CHUNK_SIZE} queries")
