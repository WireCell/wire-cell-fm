"""Sparse building blocks: the conv, transposed-conv, residual and attention-bottleneck
modules the U-Net is assembled from.
"""

from __future__ import annotations

from torch import nn
from warpconvnet.geometry.base.geometry import Geometry
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.modules.activations import GELU, ReLU
from warpconvnet.nn.modules.normalizations import LayerNorm
from warpconvnet.nn.modules.sequential import Sequential
from warpconvnet.nn.modules.sparse_conv import SparseConv2d

from .attention2d import SpatialFeatureAttention2D


class ConvBlock2D(Sequential):
    """`SparseConv2d -> LayerNorm -> ReLU`, the ReLU optional for a residual's second conv."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, bias=False, relu=True):
        super().__init__(
            SparseConv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, bias=bias),
            LayerNorm(out_ch),
            ReLU(inplace=True) if relu else nn.Identity(),
        )


class ConvTrBlock2D(nn.Module):
    """Transposed sparse convolution: the decoder's upsampling, guided by a skip's geometry."""

    def __init__(self, in_ch, out_ch, kernel_size=2, stride=2, bias=False):
        super().__init__()
        self.deconv = SparseConv2d(
            in_ch, out_ch, kernel_size=kernel_size, stride=stride, transposed=True, bias=bias
        )
        self.norm_act = Sequential(LayerNorm(out_ch), ReLU(inplace=True))

    def forward(self, x_sparse: Voxels, out_spatial_sparsity: Voxels) -> Voxels:
        # The output coordinates are the skip's. An injection at the skip therefore changes
        # where the decoder emits, not only what it reads -- which is how a mask token at a
        # removed coordinate becomes a prediction there.
        return self.norm_act(self.deconv(x_sparse, out_spatial_sparsity))


class ResidualSparseBlock2D(nn.Module):
    """ResNet BasicBlock: `Conv-LN-ReLU, Conv-LN, + identity, ReLU`. Stride is always 1 here;
    spatial downsampling is a separate `ConvBlock2D`."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = ConvBlock2D(in_ch, out_ch, kernel_size=1, stride=stride, relu=False)
        self.conv1 = ConvBlock2D(in_ch, out_ch, kernel_size=kernel_size, stride=stride)
        self.conv2 = ConvBlock2D(out_ch, out_ch, kernel_size=kernel_size, stride=1, relu=False)
        self.act = ReLU(inplace=True)

    def forward(self, x_sparse: Voxels) -> Voxels:
        identity = x_sparse if self.downsample is None else self.downsample(x_sparse)
        out = self.conv2(self.conv1(x_sparse))
        out += identity
        return self.act(out)


class BottleneckSparseAttention2D(nn.Module):
    """Transformer block on a sparse geometry, in one of two forms.

    Stream-norm (default): `1x1 in -> norm -> attention -> residual -> norm -> MLP -> residual
    -> 1x1 out`, as WarpConvNet's `StreamNormBlock`. The residual adds the normalised tensor and
    the block ends in a projection, so a stack of them has no identity path from its input to
    its output.

    `prenorm`: `x + attention(norm(x))`, then `x + MLP(norm(x))`, at `attn_channels` with no
    projections; `channels` must equal `attn_channels`. `PreNormBottleneck` stacks these
    between one projection on each side."""

    def __init__(
        self,
        channels: int,
        attn_channels: int,
        heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mlp_ratio: float = 2.0,
        encoding: bool = True,
        encoding_range: float = 1.0,
        encoding_channels: int = 32,
        flash: bool = True,
        qk_norm: bool = False,
        prenorm: bool = False,
    ):
        super().__init__()
        if prenorm and channels != attn_channels:
            raise ValueError(f"a prenorm block runs at one width: {channels} != {attn_channels}")
        self.prenorm = prenorm
        if not prenorm:
            self.pre_proj = SparseConv2d(channels, attn_channels, kernel_size=1)
        self.norm1 = LayerNorm(attn_channels)
        self.norm2 = LayerNorm(attn_channels)
        self.attn = SpatialFeatureAttention2D(
            dim=attn_channels,
            num_heads=heads,
            qkv_bias=True,
            qk_scale=None,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            use_encoding=encoding,
            num_encoding_channels=encoding_channels,
            encoding_range=encoding_range,
            enable_flash=flash,
            use_batched_qkv=True,
            qk_norm=qk_norm,
        )
        hidden = int(attn_channels * mlp_ratio)
        self.mlp = Sequential(
            SparseConv2d(attn_channels, hidden, kernel_size=1),
            GELU(),
            SparseConv2d(hidden, attn_channels, kernel_size=1),
        )
        if not prenorm:
            self.post_proj = SparseConv2d(attn_channels, channels, kernel_size=1)

    def forward(self, x: Geometry) -> Geometry:
        if self.prenorm:
            x = x + self.attn(self.norm1(x))
            return x + self.mlp(self.norm2(x))
        x2 = self.pre_proj(x)
        x2n = self.norm1(x2)
        x2 = x2n + self.attn(x2n)
        x2n = self.norm2(x2)
        x2 = x2n + self.mlp(x2n)
        return self.post_proj(x2)


class PreNormBottleneck(nn.Module):
    """`in_proj -> depth x prenorm BottleneckSparseAttention2D -> norm -> out_proj`.

    The residual stream runs from `in_proj` to `norm` through every block unnormalised, so the
    input's gradient reaches the encoder through two projections whatever the depth. `block_kw`
    goes to each block."""

    def __init__(self, channels: int, attn_channels: int, depth: int, **block_kw):
        super().__init__()
        self.in_proj = SparseConv2d(channels, attn_channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *(
                BottleneckSparseAttention2D(attn_channels, attn_channels, prenorm=True, **block_kw)
                for _ in range(depth)
            )
        )
        self.norm = LayerNorm(attn_channels)
        self.out_proj = SparseConv2d(attn_channels, channels, kernel_size=1)

    def forward(self, x: Geometry) -> Geometry:
        return self.out_proj(self.norm(self.blocks(self.in_proj(x))))
