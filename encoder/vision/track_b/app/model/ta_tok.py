# Omni Chainer - Multimodal LLM Inference System
# Copyright (c) 2025-present NAVER Cloud Corp.
# Apache-2.0

import copy
import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.transforms import Resize
from transformers import AutoConfig, AutoModel, Siglip2VisionConfig, Siglip2VisionModel


# def make(model_spec, args=None, load_sd=False) -> torch.nn.Module:
def models_make(model_spec, args=None, load_sd=False) -> torch.nn.Module:
    if args is not None:
        model_args = copy.deepcopy(model_spec["args"])
        model_args.update(args)
    else:
        model_args = model_spec["args"]
    model_params = inspect.signature(models[model_spec["name"]]).parameters
    if "kwargs" not in model_params:
        model_args = {k: v for k, v in model_args.items() if k in model_params}
    model = models[model_spec["name"]](**model_args)
    if load_sd:
        if (
            ("abs_pe" in model_spec["sd"])
            and hasattr(model, "abs_pe")
            and model_spec["sd"]["abs_pe"].shape != model.abs_pe.shape
        ):
            del model_spec["sd"]["abs_pe"]
        msg = model.load_state_dict(model_spec["sd"], strict=False)
        print(msg)
    return model


class Bottleneck(nn.Module):
    def __init__(
        self, bottleneck_dim: int, input_dim: int, output_dim: int, token_nums: int, regularizer=None, **kwargs
    ):
        super().__init__()
        self.token_nums = token_nums
        self.input_dim = input_dim
        self.output_dim = output_dim
        if bottleneck_dim > 0:
            self.bottleneck_dim = bottleneck_dim
        else:
            assert (
                self.input_dim == self.output_dim
            ), "input_dim and output_dim must be the same when bottleneck_dim is not specified"
            self.bottleneck_dim = self.input_dim

        self.project_dim = self.bottleneck_dim

        if self.bottleneck_dim > 0:
            self.in_linear = nn.Linear(self.input_dim, self.project_dim)
            self.out_linear = nn.Linear(self.bottleneck_dim, self.output_dim)
        else:
            self.in_linear = self.out_linear = lambda x: x

        regularizer["args"]["dim"] = self.bottleneck_dim
        regularizer["args"]["token_nums"] = self.token_nums
        self.regularizer = models_make(regularizer)

    def project_in(self, x):
        assert len(x.shape) == 3, "Input shape must be (batch, n_tokens, e_dim)"
        z = self.in_linear(x)
        return z

    def project_out(self, z_cat):
        z = self.out_linear(z_cat)
        return z

    def decode(self, bottleneck_rep):
        regularized_z = self.regularizer.decode(bottleneck_rep)
        return self.project_out(regularized_z)

    def forward(self, x):
        z = self.project_in(x)
        projected_z = z
        regularized_output = self.regularizer(z)
        x_hat = self.project_out(regularized_output["regularized_z"])
        bottleneck_rep = regularized_output.pop("bottleneck_rep")
        return {
            "output": x_hat,
            "bottleneck_rep": bottleneck_rep,
            "projected_z": projected_z,
            **regularized_output,
        }


class SimVectorQuantizer(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        l2_normalized=False,
        same_index_shape=True,
        stochastic=False,
        stochastic_temperature=1.0,
        **kwargs,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.dim = dim
        assert isinstance(l2_normalized, bool)
        self.l2_normalized = l2_normalized
        self.stochastic = stochastic
        self.eval_deterministic = False
        self.default_stochastic_temperature = stochastic_temperature

        if self.stochastic:
            if stochastic_temperature > 0:  # fixed temperature
                self.stochastic_temperature_inv = 1 / stochastic_temperature
            else:  # set stochastic_temperature < 0 to use learnable temperature
                self.stochastic_temperature_inv = nn.Parameter(torch.tensor(10.0))

        # for clear inference code, we remove the codebook init from LLM's embedding
        self.embedding = nn.Embedding(self.codebook_size, self.dim)
        self.embedding_proj = nn.Linear(self.dim, self.dim)

        self.same_index_shape = same_index_shape

    def set_eval_deterministic(self, deterministic=True):
        self.eval_deterministic = deterministic

    def set_stochastic_temperature(self, temperature):
        self.stochastic_temperature_inv = 1 / temperature

    @torch.autocast(device_type="cuda", enabled=False)
    def get_emb(self):
        emb = self.embedding_proj(self.embedding.weight)
        if self.l2_normalized:
            emb = F.normalize(emb, p=2, dim=-1)
        # assert emb.dtype == torch.float32, f"Embedding weight dtype is {emb.dtype}, expected float32"
        return emb

    @torch.autocast(device_type="cuda", enabled=False)
    def forward(self, z):
        emb = self.get_emb()
        z = z.to(emb)
        # z = z.float()
        assert len(z.shape) == 3, "Input shape must be (batch, n_tokens, e_dim)"
        if self.l2_normalized:
            z = F.normalize(z, p=2, dim=-1)

        z_flattened = rearrange(z, "b n d -> (b n) d")

        if self.stochastic:
            # sample the softmaxed cosine similarity
            assert self.l2_normalized, "Stochastic sampling requires l2 normalization"
            cos_sim = torch.einsum("bd,nd->bn", z_flattened, emb)
            probs = F.softmax(cos_sim * self.stochastic_temperature_inv, dim=-1)
            if self.eval_deterministic and not self.training:
                q_indices = torch.argmax(probs, dim=-1)
            else:
                q_indices = torch.multinomial(probs, 1).squeeze(-1)
        else:
            d = (
                torch.sum(z_flattened**2, dim=1, keepdim=True)
                + torch.sum(emb**2, dim=1)
                - 2 * torch.einsum("bd,dn->bn", z_flattened, rearrange(emb, "n d -> d n"))
            )
            q_indices = torch.argmin(d, dim=1)

        quantized = F.embedding(
            q_indices,
            emb,
            self.embedding.padding_idx,
            self.embedding.max_norm,
            self.embedding.norm_type,
            self.embedding.scale_grad_by_freq,
            self.embedding.sparse,
        ).view(
            z.shape
        )  # (b, n, d)

        # preserve gradients
        quantized = z + (quantized - z).detach()

        if self.same_index_shape:
            q_indices = q_indices.reshape(quantized.shape[0], quantized.shape[1])

        return_dict = {
            "unregularized_z": z,  # but l2 normalized if l2_normalized=True
            "emb": emb,  # but l2 normalized if l2_normalized=True
            "regularized_z": quantized,
            "bottleneck_rep": q_indices,
        }
        return return_dict

    def get_codebook_entry(self, indices, shape=None):
        # shape specifying (batch, height, width, channel)
        indices_shape = indices.shape
        indices_flatten = rearrange(indices, "... -> (...)")

        # get quantized latent vectors
        emb = self.get_emb()
        z_q = F.embedding(indices_flatten, emb)
        # z_q = self.embedding(indices_flatten)
        if self.l2_normalized:
            z_q = F.normalize(z_q, p=2, dim=-1)

        if shape is not None:
            z_q = z_q.reshape(shape)
        else:
            z_q = z_q.reshape([*indices_shape, self.dim])
        return z_q

    def decode(self, indices):
        return self.get_codebook_entry(indices)


models = {"simvq": SimVectorQuantizer, "bottleneck": Bottleneck}


class ScalingLayer(nn.Module):
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        super().__init__()
        self.register_buffer("shift", torch.Tensor(mean)[None, :, None, None])
        self.register_buffer("scale", torch.Tensor(std)[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale

    def inv(self, inp):
        return inp * self.scale + self.shift


class TextAlignedTokenizer(nn.Module):
    def __init__(
        self,
        bottleneck,
        bottleneck_token_num=256,
        input_size=384,
        teacher="google/siglip2-so400m-patch14-384",
        input_type="quant",  # choose from ['quant', 'rec', 'indices']
        pool_scale=1,  # choose from [1, 2, 3]
        decoder_depth=3,
        select_layer_id=-2,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.bottleneck_token_num = bottleneck_token_num
        self.teacher = teacher
        self.input_type = input_type
        self.pool_scale = pool_scale
        self.decoder_depth = decoder_depth
        self.select_layer_id = select_layer_id

        self.bottleneck_dim = bottleneck["args"]["bottleneck_dim"]

        self.encoder_config = AutoConfig.from_pretrained(teacher)
        self.encoder = AutoModel.from_config(self.encoder_config).vision_model

        self.encoder_hidden_dim = self.encoder.config.hidden_size

        self.decoder_config = Siglip2VisionConfig()
        self.decoder_config.update(
            {
                "patch_size": 1,
                "num_hidden_layers": self.decoder_depth,
                "num_channels": self.bottleneck_dim,
                "hidden_size": self.encoder_hidden_dim,
            }
        )
        self.decoder = Siglip2VisionModel(self.decoder_config)

        self.encode_task_layer = nn.Sequential(nn.Linear(self.encoder_hidden_dim, self.encoder_hidden_dim), nn.Tanh())
        self.decode_task_layer = nn.Sequential(
            nn.Linear(self.encoder_hidden_dim, self.encoder_hidden_dim),
            nn.Tanh(),
            nn.Linear(self.encoder_hidden_dim, self.encoder_hidden_dim),
        )

        bottleneck_args = {
            "token_nums": self.bottleneck_token_num,
            "input_dim": self.encoder_hidden_dim,
            "output_dim": self.bottleneck_dim,
        }
        # self.bottleneck = models.make(bottleneck, args=bottleneck_args)
        self.bottleneck = models_make(bottleneck, args=bottleneck_args)

        self.scale_layer = ScalingLayer(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.image_resize = Resize((self.input_size, self.input_size))

    def set_vq_eval_deterministic(self, deterministic=True):
        self.bottleneck.regularizer.set_eval_deterministic(deterministic)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @classmethod
    def from_checkpoint(cls, ckpt, load_teacher=True, **kwargs):
        ckpt = torch.load(ckpt, map_location="cpu", weights_only=False)
        ckpt_kwargs = ckpt["model"]["args"]
        print(ckpt_kwargs)
        model = cls(**kwargs, **ckpt_kwargs)
        sd = ckpt["model"]["sd"]
        if not load_teacher:
            sd = {k: v for k, v in sd.items() if not k.startswith("teacher")}
        model.load_state_dict(sd, strict=True)
        return model

    def encode(self, x, **kwargs):
        if x.ndim == 5:
            x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.scale_layer(x)
        if tuple(x.shape[-2:]) != (self.input_size, self.input_size):
            x = self.image_resize(x)
        vq_feats = self.encoder(x, output_hidden_states=True).hidden_states[self.select_layer_id]

        pool_scale = self.pool_scale
        pool_scale = kwargs.get("pool_scale", pool_scale)
        if pool_scale != 1:
            vq_feats = self.avg_pool(vq_feats, pool_scale)
        vq_feats = self.encode_task_layer(vq_feats.to(x))

        bottleneck_out = self.bottleneck(vq_feats)
        z = bottleneck_out.pop("output")

        return {"encoded": z, "pool_scale": pool_scale, "vq_feats": vq_feats, **bottleneck_out}

    def avg_pool(self, z, pool_scale=1):
        if z.ndim == 3:
            b, n, c = z.shape
            p = int(n**0.5)
            z = rearrange(z, "b (p1 p2) c -> b c p1 p2", p1=p, p2=p)
        else:
            b, c, p, _ = z.shape
        p_s = int(p // pool_scale)
        z = F.avg_pool2d(z, kernel_size=(pool_scale, pool_scale), stride=(pool_scale, pool_scale)).contiguous()
        z = rearrange(z, "b c p1 p2 -> b (p1 p2) c")
        return z

    def decode(self, z):
        if z.ndim == 4:
            z = rearrange(z, "b c p1 p2 -> b (p1 p2) c")
        attention_mask = torch.ones(z.shape[:2], dtype=torch.int, device=z.device)
        p = int(z.shape[1] ** 0.5)
        spatial_shape = torch.tensor([[p, p]] * z.shape[0], device=self.device)
        z = self.decoder(z, attention_mask, spatial_shape, output_hidden_states=True).last_hidden_state
        z = self.decode_task_layer(z)
        return z

    def decode_from_bottleneck(self, bottleneck_rep):
        z = self.bottleneck.decode(bottleneck_rep)  # (b, n, c)
        p = int(z.shape[1] ** 0.5)
        z = rearrange(z, "b (p1 p2) c -> b c p1 p2", p1=p, p2=p)
        return self.decode(z)

    def forward(self, data, **kwargs):
        # data: video in shape (b, c, t, h, w)
        encode_output = self.encode(data, **kwargs)
        vq_feats = encode_output["encoded"]
        p = int(vq_feats.shape[1] ** 0.5)
        vq_feats = rearrange(vq_feats, "b (h w) c -> b c h w", h=p, w=p)
        pred_feats = self.decode(vq_feats)

        if self.input_type == "quant":
            z = encode_output["regularized_z"]  # [b, n, c]
        elif self.input_type == "indices":
            z = encode_output["bottleneck_rep"]  # [b, n]
        elif self.input_type == "rec":
            z = pred_feats  # [b, n, c]
        encode_output["encoded"] = z
        return encode_output
