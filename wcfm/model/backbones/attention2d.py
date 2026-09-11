"""Sparse attention over 2-D coordinates.

Both the flash and the dense path inject the positional encoding the same way -- head-dim
wide, into q and k after the qkv projection, never into v -- so they compute the same operator
and differ only in the kernel and in whether the features are padded.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.features.ops.convert import cat_to_pad_tensor
from warpconvnet.nn.encodings import SinusoidalEncoding
from warpconvnet.nn.modules.attention import (
    Attention,
    ToSpatialFeatures,
    offset_to_mask,
    zero_out_points,
)
from warpconvnet.nn.modules.base_module import BaseSpatialModule

try:
    import flash_attn
except ImportError:  # pragma: no cover - the CPU test environment has no flash-attn
    flash_attn = None


class ToAttentionSmart(BaseSpatialModule):
    """WarpConvNet's `ToAttention` with the encoding shaped for both attention paths."""

    def __init__(
        self,
        out_channels: int,
        use_encoding: bool = False,
        num_encoding_channels: int | None = None,
        encoding_range: float | None = None,
        num_heads: int = 1,
        concat_input: bool = True,
        num_spatial_features: int = 3,
        out_type: Literal["nested", "cat"] = "cat",
    ):
        super().__init__()
        self.out_type = out_type
        self.use_encoding = use_encoding
        if use_encoding:
            assert num_encoding_channels is not None, "num_encoding_channels must be provided"
            assert encoding_range is not None, "encoding_range must be provided"
            assert out_channels % num_heads == 0, "out_channels must be divisible by num_heads"
            # Injected into q and k after the qkv projection, where a head-dim-wide tensor
            # broadcasts across heads; one width serves both paths.
            pos_out = out_channels // num_heads
            in_feats = num_encoding_channels * num_spatial_features + (
                num_spatial_features if concat_input else 0
            )
            self.encoding = nn.Sequential(
                SinusoidalEncoding(
                    num_channels=num_encoding_channels,
                    data_range=encoding_range,
                    concat_input=concat_input,
                ),
                nn.Linear(in_feats, pos_out),
            )

    def forward(self, x: Geometry):
        features_cat, offsets = x.features, x.offsets
        features = cat_to_pad_tensor(features_cat, offsets)  # [B, N, C]
        coordinates = x.coordinate_tensor  # [M, D]
        num_points = offsets.diff()  # [B]
        if self.use_encoding:
            pos_enc = cat_to_pad_tensor(self.encoding(coordinates), offsets)  # [B, N, pos_out]
        else:
            pos_enc = None
        mask = offset_to_mask(features, offsets, features.shape[1])  # [B, 1, N, N]
        return features, pos_enc, mask, num_points

    def forward_flash(self, x: Geometry):
        """The encoding alone, on the concatenated `[M, C]` layout the varlen kernel reads."""
        if not self.use_encoding:
            return None
        return self.encoding(x.coordinate_tensor)  # [M, pos_out]


class SpatialFeatureAttention2D(Attention):
    """Attention over `(x, y)` coordinates; flash on/off, encoding on/off."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        num_encoding_channels: int = 32,
        encoding_range: float = 1.0,
        use_encoding: bool = True,
        enable_flash: bool = True,
        use_batched_qkv: bool = True,
        **kwargs,
    ):
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            enable_flash=enable_flash,
            use_batched_qkv=use_batched_qkv,
        )
        self.to_attn = ToAttentionSmart(
            out_channels=dim,
            use_encoding=use_encoding,
            num_encoding_channels=num_encoding_channels,
            encoding_range=encoding_range,
            num_heads=num_heads,
            concat_input=True,
            num_spatial_features=2,
            out_type="cat",
        )
        self.from_attn = ToSpatialFeatures()

    def forward(self, x: Geometry) -> Geometry:
        if not self.enable_flash:
            features, pos_enc, mask, num_points = self.to_attn(x)
            B, N, C = features.shape
            qkv = self.qkv(features).reshape(B, N, 3, self.num_heads, C // self.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            if pos_enc is not None:
                q = q + pos_enc.unsqueeze(1)
                k = k + pos_enc.unsqueeze(1)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn_bias = torch.zeros(mask.shape, dtype=attn.dtype, device=attn.device)
                # dtype-aware sentinel: -1e9 is outside fp16 range
                attn_bias.masked_fill_(mask.logical_not(), torch.finfo(attn.dtype).min)
                attn = attn + attn_bias
            attn = self.attn_drop(attn.softmax(dim=-1))
            y = (attn @ v).transpose(1, 2).reshape(B, N, C)
            y = self.proj_drop(self.proj(y))
            if num_points is not None:
                y = zero_out_points(y, num_points)
            return self.from_attn(y, x)

        if flash_attn is None:  # pragma: no cover
            raise RuntimeError("flash_attention=True but flash_attn is not importable")
        pos_enc_cat = self.to_attn.forward_flash(x)  # [M, head_dim]
        feats, offsets = x.features, x.offsets
        M, C = feats.shape[:2]
        qkv = self.qkv(feats).reshape(M, 3, self.num_heads, C // self.num_heads)
        # Into q and k only, never v, so position acts as an attention bias and never leaks
        # into aggregated content. Before the fp16 cast, so the encoding keeps full precision.
        if pos_enc_cat is not None:
            pe = pos_enc_cat.unsqueeze(1)
            qkv = torch.stack([qkv[:, 0] + pe, qkv[:, 1] + pe, qkv[:, 2]], dim=1)
        if qkv.dtype not in (torch.float16, torch.bfloat16):
            qkv = qkv.to(torch.float16)
        max_seqlen = int(offsets.diff().max())
        attn_offsets = offsets.to(device=qkv.device, dtype=torch.int32)
        out_feat = flash_attn.flash_attn_varlen_qkvpacked_func(
            qkv,
            attn_offsets,
            max_seqlen=max_seqlen,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            softmax_scale=self.scale,
        )
        out_feat = out_feat.reshape(M, C).to(feats.dtype)
        out_feat = self.proj_drop(self.proj(out_feat))
        return x.replace(batched_features=out_feat.to(feats.dtype))
