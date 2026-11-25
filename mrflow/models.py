# Adapted from CTFlow (https://github.com/WongJiayi/CTFlow); modified for MRFlow.
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
import xformers
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging
from einops import rearrange
from timm.layers.drop import DropPath
from timm.layers.mlp import Mlp

approx_gelu = lambda: nn.GELU(approximate="tanh")

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_2d_sincos_pos_embed(
    embed_dim, grid_size, cls_token=False, extra_tokens=0, scale=1.0, base_size=None
):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if not isinstance(grid_size, tuple):
        grid_size = (grid_size, grid_size)

    grid_h = np.arange(grid_size[0], dtype=np.float32) / scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / scale
    if base_size is not None:
        grid_h *= base_size / grid_size[0]
        grid_w *= base_size / grid_size[1]
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate(
            [np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0
        )
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length, scale=1.0):
    pos = np.arange(0, length)[..., None] / scale
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


class PatchEmbed3D(nn.Module):
    """Video to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self,
        patch_size=(2, 4, 4),
        in_chans=3,
        embed_dim=96,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.flatten = flatten

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # (B C T H W)
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCTHW -> BNC
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        enable_flashattn: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if enable_flashattn:
            print(
                "[WARNING] FlashAttention cannot be used. Set enable_flashattn to False."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        qkv_permute_shape = (2, 0, 3, 1, 4)
        qkv = qkv.view(qkv_shape).permute(qkv_permute_shape)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        dtype = q.dtype
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)  # translate attn to float32
        attn = attn.to(torch.float32)
        attn = attn.softmax(dim=-1)
        attn = attn.to(dtype)  # cast back attn to original dtype
        attn = self.attn_drop(attn)
        x = attn @ v

        x_output_shape = (B, N, C)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, qk_norm=False):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    @torch._dynamo.disable
    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)
        # LayerNorm autocasts to fp32; xformers needs q, k, v in one dtype.
        q, k = self.q_norm(q).to(v.dtype), self.k_norm(k).to(v.dtype)

        attn_bias = None
        if mask is not None:
            attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        x = xformers.ops.memory_efficient_attention(
            q, k, v, p=self.attn_drop.p, attn_bias=attn_bias
        )

        x = x.view(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        )
        freqs = freqs.to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        uncond_prob,
        act_layer=nn.GELU(approximate="tanh"),
        token_num=120,
    ):
        super().__init__()
        self.y_proj = Mlp(
            in_features=in_channels,
            hidden_features=hidden_size,
            out_features=hidden_size,
            act_layer=act_layer,
            drop=0,
        )
        # A Parameter, not a buffer, so the optimizer and EMA see it.
        self.y_embedding = nn.Parameter(
            torch.randn(token_num, in_channels) / in_channels**0.5
        )
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids.to(caption.device).bool()
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    @torch._dynamo.disable
    def forward(self, caption, train, force_drop_ids=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption


class T2IFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, num_patch, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(
            torch.randn(2, hidden_size) / hidden_size**0.5
        )
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class STDiTBlock(nn.Module):
    """
    STDiT: Spatio-Temporal Diffusion Transformer.

    Args:
        hidden_size (int): Hidden size of the model.
        num_heads (int): Number of attention heads.
        d_s (int): Spatial patch size.
        d_t (int): Temporal patch size.
        mlp_ratio (float): Ratio of hidden to mlp hidden size.
        drop_path (float): Drop path rate.
        enable_flashattn (bool): Enable FlashAttention.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        d_s=None,
        d_t=None,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flashattn=False,
        uncond=False,
        qk_norm=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_flashattn = enable_flashattn

        self.attn_cls = Attention
        self.mha_cls = MultiHeadCrossAttention

        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.attn = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            enable_flashattn=False,
        )
        if uncond:
            self.cross_attn = self.mha_cls(hidden_size, num_heads, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=int(hidden_size * mlp_ratio),
            act_layer=approx_gelu,
            drop=0,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(
            torch.randn(6, hidden_size) / hidden_size**0.5
        )

        # temporal attention
        self.d_s = d_s
        self.d_t = d_t

        self.attn_temp = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=qk_norm,
            enable_flashattn=self.enable_flashattn,
        )

    def forward(self, x, t, y=None, mask=None, tpe=None):
        """
        Args:
            x (torch.Tensor): noisy input tensor of shape [B, N, C]
            y (torch.Tensor): conditional input tensor of shape [B, N, C]
            t (torch.Tensor): input tensor; of shape [B, C]
            mask (torch.Tensor): input tensor; of shape [B, N]
            tpe (torch.Tensor): input tensor; of shape [B, C]
        """
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)

        # spatial branch
        x_s = rearrange(x_m, "B (T S) C -> (B T) S C", T=self.d_t, S=self.d_s)
        x_s = self.attn(x_s)
        x_s = rearrange(x_s, "(B T) S C -> B (T S) C", T=self.d_t, S=self.d_s)
        x = x + self.drop_path(gate_msa * x_s)

        # temporal branch
        x_t = rearrange(x, "B (T S) C -> (B S) T C", T=self.d_t, S=self.d_s)
        if tpe is not None:
            x_t = x_t + tpe
        x_t = self.attn_temp(x_t)
        x_t = rearrange(x_t, "(B S) T C -> B (T S) C", T=self.d_t, S=self.d_s)
        x = x + self.drop_path(gate_msa * x_t)

        # cross attn
        if y is not None:
            x = x + self.cross_attn(x, y, mask)

        # mlp
        x = x + self.drop_path(
            gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp))
        )

        return x


class STDiT(nn.Module):
    def __init__(
        self,
        input_size=(1, 32, 32),  # T, H, W
        in_channels=4,
        out_channels=4,
        patch_size=(1, 2, 2),  # T, H, W
        hidden_size=1152,  #
        depth=28,  # Number of layers
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        drop_path=0.0,
        no_temporal_pos_emb=False,
        caption_channels=4096,  # 0 to disable
        model_max_length=120,
        space_scale=1.0,
        time_scale=1.0,
        enable_flashattn=False,
        num_modality_classes=0,  # 0 to disable
        num_plane_classes=0,  # 0 to disable
        qk_norm=False,  # LayerNorm on q and k in every attention, bounds the logits
        pooled_channels=0,  # pooled report vector added to the timestep embedding; 0 to disable
        y_norm=False,  # RMSNorm on the projected caption tokens (Sana)
        y_norm_scale_factor=1.0,  # initial RMSNorm gain; Sana uses 0.01
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.input_size = input_size
        num_patches = np.prod([input_size[i] // patch_size[i] for i in range(3)])
        self.num_patches = num_patches
        self.num_temporal = input_size[0] // patch_size[0]
        self.num_spatial = num_patches // self.num_temporal
        self.num_heads = num_heads
        self.no_temporal_pos_emb = no_temporal_pos_emb
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.enable_flashattn = enable_flashattn
        self.space_scale = space_scale
        self.time_scale = time_scale

        if caption_channels == 0:
            print("Warning: caption_channels is 0, disabling text conditioning.")

        self.register_buffer("pos_embed", self.get_spatial_pos_embed())
        self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.y_embedder = (
            CaptionEmbedder(
                in_channels=caption_channels,
                hidden_size=hidden_size,
                uncond_prob=class_dropout_prob,
                act_layer=approx_gelu,
                token_num=model_max_length,
            )
            if caption_channels > 0
            else None
        )
        # Sana's fix for a non-T5 text encoder in this block: the projected caption tokens are
        # RMS-normalized before any cross-attention sees them, null report included.
        self.y_norm_scale_factor = y_norm_scale_factor
        self.caption_norm = nn.RMSNorm(hidden_size, eps=1e-5) if y_norm else None

        # Modality and plane class embeddings, added to the timestep embedding.
        self.modality_embedder = (
            nn.Embedding(num_modality_classes, hidden_size)
            if num_modality_classes > 0
            else None
        )
        self.plane_embedder = (
            nn.Embedding(num_plane_classes, hidden_size) if num_plane_classes > 0 else None
        )
        # The pooled report vector is also added to the timestep embedding.
        self.pooled_embedder = (
            nn.Linear(pooled_channels, hidden_size) if pooled_channels > 0 else None
        )

        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                STDiTBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flashattn=self.enable_flashattn,
                    d_t=self.num_temporal,
                    d_s=self.num_spatial,
                    uncond=(caption_channels > 0),
                    qk_norm=qk_norm,
                )
                for i in range(self.depth)
            ]
        )
        self.final_layer = T2IFinalLayer(
            hidden_size, np.prod(self.patch_size), self.out_channels
        )

        # init model
        self.initialize_weights()
        self.initialize_temporal()

        # sequence parallel related configs
        self.sp_rank = None

    def forward(self, x, timestep, y=None, mask=None, modality_id=None, plane_id=None,
                force_drop_ids=None, pooled=None):
        """
        Forward pass of STDiT.
        Args:
            x (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]
            timestep (torch.Tensor): diffusion time steps; of shape [B]
            y (torch.Tensor): representation of prompts; of shape [B, 1, N_token, C]
            mask (torch.Tensor): mask for selecting prompt tokens; of shape [B, N_token]
            modality_id (torch.Tensor): modality class ids; of shape [B], dtype long
            plane_id (torch.Tensor): plane class ids; of shape [B], dtype long
            force_drop_ids (torch.Tensor): per-sample null-report mask; of shape [B], bool
            pooled (torch.Tensor): pooled report vector; of shape [B, pooled_channels]

        Returns:
            x (torch.Tensor): output latent representation; of shape [B, C, T, H, W]
        """


        # embedding
        x = self.x_embedder(x)  # [B, N, C]
        # print(x.shape, self.num_temporal, self.num_spatial)
        x = rearrange(
            x, "B (T S) C -> B T S C", T=self.num_temporal, S=self.num_spatial
        )
        x = x + self.pos_embed
        x = rearrange(x, "B T S C -> B (T S) C")

        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]

        # Class conditioning is added to the timestep embedding, so it reaches the AdaLN modulation
        # in every block (through t_block) and the output modulation (through final_layer).
        B = x.shape[0]
        for embedder, ids, name in (
            (self.modality_embedder, modality_id, "modality_id"),
            (self.plane_embedder, plane_id, "plane_id"),
        ):
            if embedder is None:
                continue
            assert ids is not None, f"{name} is required by this config"
            assert ids.shape == (B,), f"{name} must be [{B}], got {tuple(ids.shape)}"
            t = t + embedder(ids).to(t.dtype)

        # The null report drops the token sequence and the pooled vector together, so CFG keeps
        # its three branches. Drawn here rather than inside y_embedder so both routes agree.
        if force_drop_ids is None and self.training and self.y_embedder is not None \
                and self.y_embedder.uncond_prob > 0:
            force_drop_ids = torch.rand(B, device=x.device) < self.y_embedder.uncond_prob
        assert (pooled is None) == (self.pooled_embedder is None), \
            "pooled must be given exactly when pooled_channels > 0"
        if self.pooled_embedder is not None:
            keep = 1.0 if force_drop_ids is None else (~force_drop_ids.bool())[:, None].to(t.dtype)
            t = t + self.pooled_embedder(pooled.to(t.dtype)) * keep
        if mask is not None and force_drop_ids is not None:
            # A dropped report is the whole learned null sequence, not its first len(report) rows.
            mask = torch.where(force_drop_ids.bool()[:, None], torch.ones_like(mask), mask)

        t0 = self.t_block(t)  # [B, C]
        if self.y_embedder is not None and y is not None:
            y = self.y_embedder(y, self.training, force_drop_ids)  # [B, 1, N_token, C]
            if self.caption_norm is not None:
                y = self.caption_norm(y.float()).to(y.dtype)  # fp32, as Sana computes it

            if mask is not None:
                if mask.shape[0] != y.shape[0]:
                    mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
                mask = mask.squeeze(1).squeeze(1)
                y = (
                    y.squeeze(1)
                    .masked_select(mask.unsqueeze(-1) != 0)
                    .view(1, -1, x.shape[-1])
                )
                y_lens = mask.sum(dim=1).tolist()
            else:
                y_lens = [y.shape[2]] * y.shape[0]  # N_token * B
                y = y.squeeze(1).view(1, -1, x.shape[-1])
        else:
            y = None
            y_lens = None

        # blocks
        for i, block in enumerate(self.blocks):
            if i == 0:
                tpe = self.pos_embed_temporal
            else:
                tpe = None
            x = block(x=x, t=t0, y=y, mask=y_lens, tpe=tpe)
        # x.shape: [B, N, C]

        # final process
        x = self.final_layer(x, t)  # [B, N, C=T_p * H_p * W_p * C_out]
        x = self.unpatchify(x)  # [B, C_out, T, H, W]

        return x

    def unpatchify(self, x):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        return x

    def unpatchify_old(self, x):
        c = self.out_channels
        t, h, w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        pt, ph, pw = self.patch_size

        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = rearrange(x, "n t h w r p q c -> n c t r h p w q")
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def get_spatial_pos_embed(self, grid_size=None):
        if grid_size is None:
            grid_size = self.input_size[1:]
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            (grid_size[0] // self.patch_size[1], grid_size[1] // self.patch_size[2]),
            scale=self.space_scale,
        )
        pos_embed = (
            torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        )
        return pos_embed

    def get_temporal_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.hidden_size,
            self.input_size[0] // self.patch_size[0],
            scale=self.time_scale,
        )
        pos_embed = (
            torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        )
        return pos_embed

    def reset_temporal_pos_embed(self, new_num_frames: int):
        """
        Re-computes `model.pos_embed_temporal` for a different temporal length.
        Assumes your model uses `self.input_size[0]` as the frame-count.
        """
        # 1) Update the frame-count that your embedding factory uses:
        self.input_size[0] = new_num_frames

        # 2) Compute a fresh tensor
        new_embed = self.get_temporal_pos_embed()
        # make sure it lives on the same device & dtype
        new_embed = new_embed.to(
            self.pos_embed_temporal.device, self.pos_embed_temporal.dtype
        )

        # 3) Overwrite the existing buffer
        #    register_buffer will replace the old one if the name matches
        self.register_buffer("pos_embed_temporal", new_embed, persistent=False)
        self.num_patches = np.prod(
            [self.input_size[i] // self.patch_size[i] for i in range(3)]
        )
        self.num_temporal = new_num_frames // self.patch_size[0]
        self.num_spatial = self.num_patches // self.num_temporal

        for block in self.blocks:
            block.d_t = self.num_temporal
            block.d_s = self.num_spatial

    def freeze_not_temporal(self):
        for n, p in self.named_parameters():
            if "attn_temp" not in n:
                p.requires_grad = False

    def freeze_text(self):
        for n, p in self.named_parameters():
            if "cross_attn" in n:
                p.requires_grad = False

    def initialize_temporal(self):
        for block in self.blocks:
            nn.init.constant_(block.attn_temp.proj.weight, 0)
            nn.init.constant_(block.attn_temp.proj.bias, 0)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        if self.y_embedder is not None:
            nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
            nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)
        if self.caption_norm is not None:
            nn.init.constant_(self.caption_norm.weight, self.y_norm_scale_factor)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        # Zero-out the class embeddings, so a checkpoint that predates them predicts exactly as it
        # did (t is unchanged) while every row still receives gradients and can diverge.
        for embedder in (self.modality_embedder, self.plane_embedder, self.pooled_embedder):
            if embedder is not None:
                nn.init.constant_(embedder.weight, 0)
        if self.pooled_embedder is not None:
            nn.init.constant_(self.pooled_embedder.bias, 0)


@dataclass
class DiffuserSTDiTModelOutput(BaseOutput):
    """
    The output of [`DiffuserSTDiT`].

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, num_frames, height, width)` or `(batch size, num_vector_embeds - 1, num_latent_pixels)` if [`Transformer2DModel`] is discrete):
            The hidden states output conditioned on the `encoder_hidden_states` input. If discrete, returns probability
            distributions for the unnoised latent pixels.
    """

    sample: torch.FloatTensor


class DiffuserSTDiT(ModelMixin, ConfigMixin):
    """
    STDiT: Spatio-Temporal Diffusion Transformer.

    Parameters:
        input_size (tuple): Input size of the video. Default: (1, 32, 32).
        in_channels (int): Number of input video channels. Default: 4.
        out_channels (int): Number of output video channels. Default: 4.
        patch_size (tuple): Patch token size. Default: (1, 2, 2).
        hidden_size (int): Hidden size of the model. Default: 1152.
        depth (int): Number of layers. Default: 28.
        num_heads (int): Number of attention heads. Default: 16.
        mlp_ratio (float): Ratio of hidden to mlp hidden size. Default: 4.0.
        class_dropout_prob (float): Probability of dropping class tokens. Default: 0.1.
        drop_path (float): Drop path rate. Default: 0.0.
        no_temporal_pos_emb (bool): Disable temporal positional embeddings. Default: False.
        caption_channels (int): Number of caption channels. Default: 4096.
        model_max_length (int): Maximum length of the model. Default: 120.
        space_scale (float): Spatial scale. Default: 1.0.
        time_scale (float): Temporal scale. Default: 1.0.
        enable_flashattn (bool): Enable FlashAttention. Default: False.
    """

    @register_to_config
    def __init__(
        self,
        input_size=(1, 32, 32),  # T, H, W
        in_channels=4,
        out_channels=4,
        patch_size=(1, 2, 2),  # T, H, W
        hidden_size=1152,  #
        depth=28,  # Number of layers
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        drop_path=0.0,
        no_temporal_pos_emb=False,
        caption_channels=4096,  # 0 to disable
        model_max_length=120,
        space_scale=1.0,
        time_scale=1.0,
        enable_flashattn=False,
        num_modality_classes=0,
        num_plane_classes=0,
        qk_norm=False,
        pooled_channels=0,
        y_norm=False,
        y_norm_scale_factor=1.0,
    ):

        super().__init__()

        self.model = STDiT(
            input_size=input_size,
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            class_dropout_prob=class_dropout_prob,
            drop_path=drop_path,
            no_temporal_pos_emb=no_temporal_pos_emb,
            caption_channels=caption_channels,
            model_max_length=model_max_length,
            space_scale=space_scale,
            time_scale=time_scale,
            enable_flashattn=enable_flashattn,
            num_modality_classes=num_modality_classes,
            num_plane_classes=num_plane_classes,
            qk_norm=qk_norm,
            pooled_channels=pooled_channels,
            y_norm=y_norm,
            y_norm_scale_factor=y_norm_scale_factor,
        )

    def forward(
        self,
        x,
        timestep,
        encoder_hidden_states=None,
        cond_image=None,
        mask=None,
        return_dict=True,
        modality_id=None,
        plane_id=None,
        force_drop_ids=None,
        pooled=None,
        *args,
        **kwargs,
    ):
        """
        Args:
            x (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]
            timestep (torch.Tensor): diffusion time steps; of shape [B]
            y (torch.Tensor): representation of prompts; of shape [B, 1, N_token, C]
            mask (torch.Tensor): mask for selecting prompt tokens; of shape [B, N_token]
            return_dict (bool): return a dictionary or not. Default: True.
            modality_id (torch.Tensor): modality class ids; of shape [B], dtype long
            plane_id (torch.Tensor): plane class ids; of shape [B], dtype long
            force_drop_ids (torch.Tensor): per-sample null-report mask; of shape [B], bool
            pooled (torch.Tensor): pooled report vector; of shape [B, pooled_channels]
        """
        if type(timestep) == int or timestep.ndim == 0:
            timestep = torch.ones(x.shape[0], device=x.device) * timestep

        encoder_hidden_states = (
            encoder_hidden_states.unsqueeze(1)
            if encoder_hidden_states is not None
            else None
        )

        if cond_image is not None:
            assert (
                x.shape == cond_image.shape
            ), "x and cond_image must have the same shape"
            x = torch.cat([x, cond_image], dim=1)  # B x 2C x T x H x W

        output = self.model(
            x,
            timestep,
            y=encoder_hidden_states,
            mask=mask,
            modality_id=modality_id,
            plane_id=plane_id,
            force_drop_ids=force_drop_ids,
            pooled=pooled,
        )
        if not return_dict:
            return (output,)

        return DiffuserSTDiTModelOutput(sample=output)

    def reset_temporal_pos_embed(self, new_num_frames: int):
        """
        Re-computes `model.pos_embed_temporal` for a different temporal length.
        Assumes your model uses `self.input_size[0]` as the frame-count.
        """
        self.model.reset_temporal_pos_embed(new_num_frames)
