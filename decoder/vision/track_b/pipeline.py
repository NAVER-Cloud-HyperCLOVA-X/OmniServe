# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0
#
# Portions of this code are adapted from:
# - black-forest-labs/flux: https://github.com/black-forest-labs/flux (Apache-2.0)
# - csuhan/Tar: https://github.com/csuhan/Tar (Apache-2.0)
# - huggingface/diffusers: https://github.com/huggingface/diffusers (Apache-2.0)

"""
HuggingFace Diffusers-Compatible Vision Token to Image Pipeline.

Usage:
    from diffusers_vtoken_pipeline import VisionTokenToImagePipeline
    
    pipeline = VisionTokenToImagePipeline.from_pretrained("path/to/model")
    image = pipeline(vision_tokens).images[0]
"""

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DiffusionPipeline
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import BaseOutput
from einops import rearrange, repeat
from PIL import Image
from tqdm import tqdm

# Configure logger
logger = logging.getLogger("vision-decoder-api.pipeline")

# Flash Attention support detection
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    FLASH_ATTN_AVAILABLE = torch.cuda.is_available()
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    sdpa_kernel = None
    SDPBackend = None


# =============================================================================
# Part 1: Utility Layers (RoPE, Attention, etc.)
# =============================================================================

def rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    """Rotary Position Embedding computation."""
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary position embedding to query and key."""
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
    """Attention with rotary position embedding and Flash Attention optimization."""
    q, k = apply_rope(q, k, pe)
    
    # Use Flash Attention when available for better memory efficiency and speed
    if FLASH_ATTN_AVAILABLE and q.is_cuda:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            x = F.scaled_dot_product_attention(q, k, v)
    else:
        # Fallback to default SDPA (will use best available backend)
        x = F.scaled_dot_product_attention(q, k, v)
    
    x = rearrange(x, "B H L D -> B L (H D)")
    return x


@torch.no_grad()
def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000, time_factor: float = 1000.0) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class EmbedND(nn.Module):
    """N-dimensional position embedding."""
    def __init__(self, dim: int, theta: int, axes_dim: List[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat([rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)], dim=-3)
        return emb.unsqueeze(1)


class MLPEmbedder(nn.Module):
    """MLP for timestep and vector embeddings."""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(nn.Module):
    """Query-Key normalization."""
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


@dataclass
class ModulationOut:
    shift: torch.Tensor
    scale: torch.Tensor
    gate: torch.Tensor


class Modulation(nn.Module):
    """Adaptive layer normalization modulation."""
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: torch.Tensor) -> Tuple[ModulationOut, Optional[ModulationOut]]:
        out = self.lin(F.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)
        return (ModulationOut(*out[:3]), ModulationOut(*out[3:]) if self.is_double else None)


class SingleStreamBlock(nn.Module):
    """Single stream transformer block (parallel attention and MLP)."""
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, qk_scale: Optional[float] = None):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        head_dim = hidden_size // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(head_dim)
        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

    def forward(self, x: torch.Tensor, vec: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        mod, _ = self.modulation(vec)
        x_mod = (1 + mod.scale) * self.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = self.norm(q, k, v)

        attn = attention(q, k, v, pe=pe)
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + mod.gate * output


class LastLayer(nn.Module):
    """Final projection layer with adaptive normalization."""
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x


# =============================================================================
# Part 2: VisionTransformer - Diffusers ModelMixin Compatible
# =============================================================================

class VisionTransformer(ModelMixin, ConfigMixin):
    """
    Vision Transformer for vision token to image generation.
    
    This model is fully compatible with HuggingFace Diffusers and can be:
    - Saved with `model.save_pretrained()`
    - Loaded with `VisionTransformer.from_pretrained()`
    - Uploaded to HuggingFace Hub
    """
    
    @register_to_config
    def __init__(
        self,
        in_channels: int = 16,
        vec_in_dim: int = 1536,
        context_in_dim: int = 1536,
        hidden_size: int = 1920,
        mlp_ratio: float = 4.0,
        num_heads: int = 24,
        depth: int = 0,
        depth_single_blocks: int = 35,
        axes_dim: Tuple[int, int, int] = (8, 36, 36),
        theta: int = 10_000,
        qkv_bias: bool = True,
        guidance_embed: bool = False,
        use_patchify: bool = False,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.context_in_dim = context_in_dim
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.use_patchify = use_patchify
        
        if hidden_size % num_heads != 0:
            raise ValueError(f"Hidden size {hidden_size} must be divisible by num_heads {num_heads}")
        
        pe_dim = hidden_size // num_heads
        axes_dim_list = list(axes_dim)
        if sum(axes_dim_list) != pe_dim:
            raise ValueError(f"Got {axes_dim_list} but expected positional dim {pe_dim}")
        
        # Position embedding
        self.pe_embedder = EmbedND(dim=pe_dim, theta=theta, axes_dim=axes_dim_list)
        
        # Input projections
        self.img_in = nn.Linear(in_channels + context_in_dim, hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=hidden_size)
        self.vector_in = MLPEmbedder(vec_in_dim, hidden_size)
        
        # Single stream blocks only
        self.single_blocks = nn.ModuleList([
            SingleStreamBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth_single_blocks)
        ])
        
        # Output layer
        self.final_layer = LastLayer(hidden_size, 1, self.out_channels)
    
    def forward(
        self,
        img: torch.Tensor,
        img_ids: torch.Tensor,
        timesteps: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the transformer.
        
        Args:
            img: Input tensor (B, L, in_channels + context_in_dim)
            img_ids: Position IDs tensor (B, L, 3)
            timesteps: Sigma/timestep tensor (B,)
            y: Vision pooler output tensor (B, vec_in_dim)
        
        Returns:
            Output tensor (B, L, out_channels)
        """
        if img.ndim != 3:
            raise ValueError("Input img tensor must have 3 dimensions.")
        
        # Project input
        img = self.img_in(img)
        
        # Time and vector embedding
        vec = self.time_in(
            timestep_embedding(timesteps, 256).to(dtype=self.time_in.in_layer.weight.dtype, device=img.device)
        )
        vec = vec + self.vector_in(y)
        
        # Position embedding
        pe = self.pe_embedder(img_ids)
        
        # Single stream blocks
        for block in self.single_blocks:
            img = block(img, vec=vec, pe=pe)
        
        # Final projection
        img = self.final_layer(img, vec)
        
        return img


# =============================================================================
# Part 3: VisionTokenEmbedder - Diffusers ModelMixin Compatible
# =============================================================================

class VisionTokenEmbedder(ModelMixin, ConfigMixin):
    """
    Vision Token Embedder that converts discrete vision tokens to embeddings.
    """

    @register_to_config
    def __init__(
        self,
        vocab_size: int = 65536,
        embedding_dim: int = 1536,
        token_length: int = 729,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.token_length = token_length
        
        # Register embeddings as buffer (not trainable, but saved with model)
        self.register_buffer("vocab_embeddings", torch.zeros(vocab_size, embedding_dim))
        self.register_buffer("uncond_embedding", torch.zeros(1, embedding_dim))
    
    def load_vocab_embeddings(self, embeddings: torch.Tensor):
        """Load vocabulary embeddings from a tensor."""
        if embeddings.shape != (self.vocab_size, self.embedding_dim):
            raise ValueError(
                f"Expected embeddings shape ({self.vocab_size}, {self.embedding_dim}), "
                f"got {embeddings.shape}"
            )
        self.vocab_embeddings.copy_(embeddings)
    
    def forward(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Convert vision tokens to embeddings.
        
        Args:
            tokens: Vision token IDs (B, L)
        
        Returns:
            Dictionary with vision_last_hidden_state and vision_pooler_output
        """
        hidden_states = self.vocab_embeddings[tokens]
        pooler_output = hidden_states.mean(dim=1)
        
        return {
            "vision_last_hidden_state": hidden_states,
            "vision_pooler_output": pooler_output,
        }
    
    def get_uncond_embeddings(self, batch_size: int, token_length: int) -> Dict[str, torch.Tensor]:
        """Get unconditional embeddings for classifier-free guidance."""
        uncond_hidden = self.uncond_embedding.expand(batch_size, token_length, -1)
        uncond_pooler = uncond_hidden.mean(dim=1)
        
        return {
            "vision_last_hidden_state": uncond_hidden,
            "vision_pooler_output": uncond_pooler,
        }
    
    @classmethod
    def from_numpy(cls, npy_path: str) -> "VisionTokenEmbedder":
        """Create embedder from numpy file."""
        embeddings = torch.from_numpy(np.load(npy_path)).float()
        vocab_size, embedding_dim = embeddings.shape
        
        embedder = cls(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            token_length=729,  # Default, can be overridden
        )
        embedder.load_vocab_embeddings(embeddings)
        return embedder


# =============================================================================
# Part 4: VisionTokenToImagePipeline - Diffusers DiffusionPipeline Compatible
# =============================================================================

@dataclass
class VisionTokenToImagePipelineOutput(BaseOutput):
    """Output class for VisionTokenToImagePipeline."""
    images: List[Image.Image]


class VisionTokenToImagePipeline(DiffusionPipeline):
    """
    HuggingFace Diffusers Pipeline for Vision Token to Image generation.

    Supports autoguidance when transformer2 is provided and guidance_scale > 0.
    """
    
    # Define model components for diffusers
    model_cpu_offload_seq = "token_embedder->transformer->transformer2->vae"
    _optional_components = ["transformer2"]
    
    def __init__(
        self,
        transformer: VisionTransformer,
        vae: AutoencoderKL,
        scheduler: FlowMatchEulerDiscreteScheduler,
        token_embedder: VisionTokenEmbedder,
        transformer2: Optional[VisionTransformer] = None,
    ):
        super().__init__()
        
        # Register modules (this enables save_pretrained/from_pretrained)
        self.register_modules(
            transformer=transformer,
            transformer2=transformer2,
            vae=vae,
            scheduler=scheduler,
            token_embedder=token_embedder,
        )
        
        self.vae_scale_factor = 8  # VAE uses 8x downsampling
        self._use_autoguidance = transformer2 is not None
    
    def _prepare_latents(
        self,
        batch_size: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Prepare random latents for diffusion."""
        latent_h = height // self.vae_scale_factor
        latent_w = width // self.vae_scale_factor
        latent_channels = 16  # VAE has 16 latent channels
        
        shape = (batch_size, latent_channels, latent_h, latent_w)
        latents = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        
        return latents
    
    def _prepare_img_ids(
        self,
        batch_size: int,
        img_h: int,
        img_w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare position IDs for the transformer."""
        img_ids = torch.zeros(img_h, img_w, 3)
        img_ids[..., 1] = img_ids[..., 1] + torch.arange(img_h)[:, None]
        img_ids[..., 2] = img_ids[..., 2] + torch.arange(img_w)[None, :]
        img_ids = repeat(img_ids, "h w c -> b (h w) c", b=batch_size)
        return img_ids.to(device=device, dtype=dtype)
    
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to images using VAE."""
        scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
        shift_factor = getattr(self.vae.config, "shift_factor", 0.0)
        
        latents = latents / scaling_factor + shift_factor
        images = self.vae.decode(latents).sample
        
        return images
    
    def postprocess(self, images: torch.Tensor) -> List[Image.Image]:
        """Convert tensor images to PIL images."""
        images = images.clamp(-1, 1)
        images = (images + 1) / 2
        images = images.float().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).astype(np.uint8)
        
        return [Image.fromarray(img) for img in images]
    
    @torch.no_grad()
    def __call__(
        self,
        vision_tokens: Union[np.ndarray, torch.Tensor, List[int]],
        height: int = 768,
        width: int = 768,
        num_inference_steps: int = 50,
        guidance_scale: float = 0.0,
        generator: Optional[Union[torch.Generator, int]] = None,
        output_type: str = "pil",
        return_dict: bool = True,
    ) -> Union[VisionTokenToImagePipelineOutput, Tuple]:
        """
        Generate images from vision tokens.
        
        Args:
            vision_tokens: Vision token IDs. Can be:
                - numpy array of shape (L,) or (B, L)
                - torch tensor of shape (L,) or (B, L)
                - List of integers
            height: Output image height (must be divisible by 16)
            width: Output image width (must be divisible by 16)
            num_inference_steps: Number of diffusion steps
            guidance_scale: Autoguidance scale (0 = no guidance, requires transformer2)
            generator: Random generator for reproducibility (int seed or torch.Generator)
            output_type: Output type ("pil" or "pt" for tensor)
            return_dict: Whether to return a dataclass or tuple
        
        Returns:
            VisionTokenToImagePipelineOutput with generated images
        """
        # Log pipeline call parameters
        logger.info(f"[Pipeline] __call__ invoked with guidance_scale={guidance_scale}, num_inference_steps={num_inference_steps}")
        logger.info(f"[Pipeline] Image dimensions: {width}x{height}")
        logger.info(f"[Pipeline] transformer2 available: {self.transformer2 is not None}")
        logger.info(f"[Pipeline] _use_autoguidance flag: {self._use_autoguidance}")
        
        device = self._execution_device
        dtype = self.transformer.dtype if hasattr(self.transformer, 'dtype') else torch.bfloat16
        
        # Prepare generator
        if isinstance(generator, int):
            logger.info(f"[Pipeline] Using seed: {generator}")
            generator = torch.Generator(device=device).manual_seed(generator)
        
        # Prepare tokens
        if isinstance(vision_tokens, list):
            vision_tokens = np.array(vision_tokens, dtype=np.int64)
        if isinstance(vision_tokens, np.ndarray):
            vision_tokens = torch.from_numpy(vision_tokens)
        if vision_tokens.ndim == 1:
            vision_tokens = vision_tokens.unsqueeze(0)
        
        vision_tokens = vision_tokens.long().to(device)
        batch_size = vision_tokens.shape[0]
        
        # Get vision embeddings
        vision_cond = self.token_embedder(vision_tokens)
        
        # Prepare latents
        latents = self._prepare_latents(
            batch_size, height, width,
            dtype=dtype, device=device,
            generator=generator,
        )
        
        # Prepare for denoising
        _, _, h, w = latents.shape
        use_patchify = self.transformer.use_patchify
        
        if use_patchify:
            img_h, img_w = h // 2, w // 2
        else:
            img_h, img_w = h, w
        
        img_ids = self._prepare_img_ids(batch_size, img_h, img_w, device, dtype)
        
        # Prepare vision conditioning
        vision_hidden = vision_cond["vision_last_hidden_state"].to(dtype)
        vision_pooler = vision_cond["vision_pooler_output"].to(dtype)
        
        # Reshape vision condition for spatial concatenation
        cond_h = cond_w = int(vision_hidden.shape[1] ** 0.5)
        vision_spatial = rearrange(vision_hidden, "b (h w) c -> b c h w", h=cond_h, w=cond_w)
        vision_spatial = F.interpolate(vision_spatial, size=(img_h, img_w), mode="bilinear")
        vision_spatial = rearrange(vision_spatial, "b d h w -> b (h w) d")
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        
        # Determine autoguidance usage ONCE before loop (no overhead inside loop)
        use_autoguidance = self._use_autoguidance and guidance_scale > 0
        
        # Log before denoising starts
        if use_autoguidance:
            logger.info(f"[Pipeline] ✓ Denoising: transformer + transformer2 (guidance_scale={guidance_scale})")
        elif guidance_scale > 0 and not self._use_autoguidance:
            logger.warning(f"[Pipeline] ✗ transformer2 not available, guidance_scale={guidance_scale} ignored")
        else:
            logger.info(f"[Pipeline] Denoising: transformer only (guidance_scale={guidance_scale})")
        
        # Denoising loop - no logging inside for performance
        for i, t in enumerate(tqdm(timesteps, desc="Denoising", leave=False)):
            # Prepare input
            if use_patchify:
                x_t = rearrange(latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
            else:
                x_t = rearrange(latents, "b c h w -> b (h w) c")
            
            # Concatenate with vision condition
            x_t = torch.cat((x_t, vision_spatial), dim=2)

            # Convert timesteps to sigma
            t_batch = torch.full((batch_size,), t.item(), device=device, dtype=torch.long)
            sigma = t_batch.float() / self.scheduler.config.num_train_timesteps
            
            # Forward pass
            pred = self.transformer(
                img=x_t,
                img_ids=img_ids,
                timesteps=sigma,
                y=vision_pooler,
            )

            # Apply autoguidance (condition checked once before loop)
            if use_autoguidance:
                pred2 = self.transformer2(img=x_t, img_ids=img_ids, timesteps=sigma, y=vision_pooler)
                pred = pred + guidance_scale * (pred - pred2)
            
            # Unpatchify prediction
            if use_patchify:
                pred = rearrange(pred, "b (h w) (c ph pw) -> b c (h ph) (w pw)", h=h//2, w=w//2, ph=2, pw=2)
            else:
                pred = rearrange(pred, "b (h w) c -> b c h w", h=h, w=w)
            
            # Scheduler step
            latents = self.scheduler.step(pred, t, latents, generator=generator).prev_sample

        # Decode latents
        images = self.decode_latents(latents)

        # Postprocess
        if output_type == "pil":
            images = self.postprocess(images)

        if not return_dict:
            return (images,)

        return VisionTokenToImagePipelineOutput(images=images)
