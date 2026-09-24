"""The masked ViT blocks the encoder and decoder are built from, and the positional encoder.

A batch is padded to its longest event, so every block takes `mask`, `(B, T)` with True at real
tokens: attention is confined to real tokens, the MLP output is zeroed elsewhere, and the
normalisation statistics exclude padding. Two normalisations exist under the same parameter
shapes. `TokenLayerNorm` is a layer norm per token. `GlobalMaskedNorm` takes one mean and one
standard deviation over every real token of the batch, which is what the published PoLAr-MAE
checkpoints trained with, so a token's features depend on the other events in its batch and
extraction depends on batch composition. The default is the first.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "Attention",
    "Block",
    "GlobalMaskedNorm",
    "LearnedPositionalEncoder",
    "Mlp",
    "TokenLayerNorm",
    "Transformer",
    "attention_mask",
    "make_norm",
]

NORMS = ("layer", "global")


def _tiny(dtype: torch.dtype) -> float:
    if dtype in (torch.float32, torch.float64, torch.bfloat16):
        return 1e-13
    if dtype == torch.float16:
        return 1e-4
    raise TypeError(f"unsupported dtype {dtype}")


class GlobalMaskedNorm(nn.Module):
    """One scalar mean and standard deviation over all real tokens and channels of the batch."""

    def __init__(self, size: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, 1, size))
        self.beta = nn.Parameter(torch.zeros(1, 1, size))
        self.size = int(size)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if mask is None:
            mask = x.new_ones(x.shape[:2], dtype=torch.bool)
        m = mask.unsqueeze(-1)
        n = m.sum() * self.size
        mean = (x * m).sum() / n
        centered = (x - mean) * m
        std = torch.sqrt((centered * centered).sum() / n + _tiny(x.dtype))
        return self.gamma * (x - mean) / (std + _tiny(x.dtype)) + self.beta


class TokenLayerNorm(nn.Module):
    """Layer norm over the channels of each token, with the same `gamma`/`beta` shapes as
    `GlobalMaskedNorm` so the two are interchangeable in a state dict."""

    def __init__(self, size: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, 1, size))
        self.beta = nn.Parameter(torch.zeros(1, 1, size))
        self.size = int(size)
        self.eps = float(eps)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        return F.layer_norm(x, (self.size,), eps=self.eps) * self.gamma + self.beta


def make_norm(size: int, norm: str) -> nn.Module:
    if norm == "layer":
        return TokenLayerNorm(size)
    if norm == "global":
        return GlobalMaskedNorm(size)
    raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")


class _Identity(nn.Module):
    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        return x


class MaskedDropPath(nn.Module):
    """Stochastic depth per token, active in training only; padded tokens are zeroed."""

    def __init__(self, drop_prob: float):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        rnd = x.new_empty((x.shape[0], x.shape[1], 1)).bernoulli_(keep).div_(keep)
        x = x * rnd
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop) if drop > 0.0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def attention_mask(mask: Tensor | None, dtype: torch.dtype) -> Tensor | None:
    """`(B, T)` token mask to a `(B, 1, T, T)` additive mask: 0 between real tokens, -1e9
    where either side is padding. Kept 4-D and in the query dtype so
    `F.scaled_dot_product_attention` takes a fused kernel."""
    if mask is None:
        return None
    pair = mask.unsqueeze(1).unsqueeze(2) & mask.unsqueeze(1).unsqueeze(3)
    return (~pair).to(dtype).masked_fill(~pair, -1e9)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: Tensor, attn_mask: Tensor | None = None) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None if attn_mask is None else attn_mask.to(q.dtype),
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        return self.proj_drop(self.proj(out.transpose(1, 2).reshape(B, N, C)))


class Block(nn.Module):
    """Pre-norm attention and MLP with residuals, both masked."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm: str = "layer",
    ):
        super().__init__()
        self.drop_path = MaskedDropPath(drop_path) if drop_path > 0.0 else _Identity()
        self.norm1 = make_norm(dim, norm)
        self.attn = Attention(
            dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop
        )
        self.norm2 = make_norm(dim, norm)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x: Tensor, attn_mask: Tensor | None, mask: Tensor | None) -> Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x, mask), attn_mask), mask)
        ffn = self.mlp(self.norm2(x, mask))
        if mask is not None:
            ffn = ffn * mask.unsqueeze(-1).to(ffn.dtype)
        return x + self.drop_path(ffn, mask)


class Transformer(nn.Module):
    """`depth` blocks with stochastic depth rising linearly to `drop_path`, the positional
    encoding added before every block, and a final norm when `postnorm` is set."""

    def __init__(
        self,
        *,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        postnorm: bool = False,
        norm: str = "layer",
    ):
        super().__init__()
        self.embed_dim = int(embed_dim)
        rates = [float(r) for r in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=rates[i],
                    norm=norm,
                )
                for i in range(depth)
            ]
        )
        self.norm = make_norm(embed_dim, norm) if postnorm else _Identity()
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, pos: Tensor, mask: Tensor | None) -> Tensor:
        attn_mask = attention_mask(mask, x.dtype)
        for block in self.blocks:
            x = block(x + pos, attn_mask, mask)
        return self.norm(x, mask)


class LearnedPositionalEncoder(nn.Module):
    """MLP `3 -> 128 -> embed_dim` on a group centre's normalised coordinates."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.pos_enc = nn.Sequential(nn.Linear(3, 128), nn.GELU(), nn.Linear(128, embed_dim))

    def forward(self, pos: Tensor) -> Tensor:
        return self.pos_enc(pos[..., :3])
