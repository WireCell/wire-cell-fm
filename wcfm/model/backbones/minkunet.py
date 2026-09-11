"""The U-Net with an attention bottleneck: one class, with taps and role-typed injection.

`attention=False` puts a residual block where the attention is, `spatial_encoding` and
`flash_attention` select the bottleneck's variants, and `inject_roles` says which tokens the
backbone holds. There are no dense adapters and no classifier heads: nothing here consumes a
dense tensor, and a supervised head is a term.

The reconstruction heads are not part of the backbone. A charge or occupancy head belongs to
the term that scores it. The backbone emits features at the coordinates it was asked to, and
reports which ones it placed.

Module names -- `conv0 .. block8`, `final`, `bottleneck.attn.to_attn.encoding` -- match an
`ml-dune-model` state dict, so a converted checkpoint loads without remapping them. The two
mask tokens are the exception: `tokens.enc0__masked` and `tokens.enc1__masked` are keyed by
what the token means rather than by where it sits, and are the keys a conversion renames.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar

import torch
from torch import nn
from warpconvnet.geometry.types.voxels import Voxels
from warpconvnet.nn.functional.transforms import cat
from warpconvnet.nn.modules.sparse_conv import SparseConv2d

from .base import Backbone, FeatureBundle, Injection, InjectionGroup, inject_into_skip
from .blocks import (
    BottleneckSparseAttention2D,
    ConvBlock2D,
    ConvTrBlock2D,
    ResidualSparseBlock2D,
)


class MinkUNetAttention(Backbone):
    """Sparse encoder (2 strided stages), attention bottleneck, sparse decoder with skips.

    Taps, in forward order, with their stride relative to the input:

        enc0             1   after conv0: the full-res skip (and injection site)
        enc1             2   after block1: the half-res skip (and injection site)
        bottleneck       4   after block2, before the bottleneck
        bottleneck_attn  4   after the bottleneck
        dec_half         2   after block6
        dec_full         1   after block8, before `final`

    `out` is `final(dec_full)`, in `out_dim` channels.
    """

    TAPS: ClassVar[tuple[str, ...]] = (
        "enc0",
        "enc1",
        "bottleneck",
        "bottleneck_attn",
        "dec_half",
        "dec_full",
    )
    TAP_STRIDE: ClassVar[dict[str, int]] = {
        "enc0": 1,
        "enc1": 2,
        "bottleneck": 4,
        "bottleneck_attn": 4,
        "dec_half": 2,
        "dec_full": 1,
    }
    INJECT_TAPS: ClassVar[tuple[str, ...]] = ("enc0", "enc1")
    GRAD_GROUPS: ClassVar[dict[str, tuple[str, ...]]] = {
        "conv0": ("conv0.",),
        "enc1": ("conv1.", "block1."),
        "enc2": ("conv2.", "block2."),
        "bottleneck": ("bottleneck.",),
        "dec1": ("convtr5.", "block6."),
        "dec2": ("convtr7.", "block8."),
        "final": ("final.",),
        "tokens": ("tokens.",),
    }

    def __init__(
        self,
        *,
        in_ch: int = 1,
        widths: Iterable[int] = (32, 32, 64, 64),
        out_dim: int = 64,
        attention: bool = True,
        spatial_encoding: bool = True,
        flash_attention: bool = True,
        encoding_dim: int = 32,
        encoding_range: float = 125.0,
        attn_ch: int = 128,
        heads: int = 4,
        inject_roles: Iterable[str] = ("masked",),
    ):
        super().__init__()
        stem, e1, e2, dec = (int(w) for w in widths)
        self.out_dim = int(out_dim)
        self.inject_roles = tuple(inject_roles)

        self.conv0 = ConvBlock2D(in_ch, stem, kernel_size=3, stride=1)
        self.conv1 = ConvBlock2D(stem, e1, kernel_size=2, stride=2)
        self.block1 = ResidualSparseBlock2D(e1, e1, kernel_size=3)
        self.conv2 = ConvBlock2D(e1, e1, kernel_size=2, stride=2)
        self.block2 = ResidualSparseBlock2D(e1, e2, kernel_size=3)
        if attention:
            self.bottleneck: nn.Module = BottleneckSparseAttention2D(
                channels=e2,
                attn_channels=attn_ch,
                heads=heads,
                encoding=spatial_encoding,
                flash=flash_attention,
                encoding_range=encoding_range,
                encoding_channels=encoding_dim,
            )
        else:
            self.bottleneck = ResidualSparseBlock2D(e2, e2, kernel_size=3)
        self.convtr5 = ConvTrBlock2D(e2, dec, kernel_size=2, stride=2)
        self.block6 = ResidualSparseBlock2D(dec + e1, dec, kernel_size=3)
        self.convtr7 = ConvTrBlock2D(dec, dec, kernel_size=2, stride=2)
        self.block8 = ResidualSparseBlock2D(dec + stem, dec, kernel_size=3)
        self.final = SparseConv2d(dec, self.out_dim, kernel_size=1, bias=True)

        # One token per (skip, role), in that skip's width. Built after the network, as the
        # old tokens were, so every other parameter draws from the same point in the RNG
        # stream whether or not a run injects.
        self._skip_width = {"enc0": stem, "enc1": e1}
        # Every tap's width, so a term reading one can size its head. `dec_half` is block6's
        # output and is what the occupancy head reads: one question per 2x2 full-res block.
        self._tap_width = {
            "enc0": stem,
            "enc1": e1,
            "bottleneck": e2,
            "bottleneck_attn": e2,
            "dec_half": dec,
            "dec_full": dec,
        }
        self.tokens = nn.ParameterDict()
        for tap in self.INJECT_TAPS:
            for role in self.inject_roles:
                token = nn.Parameter(torch.zeros(self._skip_width[tap]))
                nn.init.trunc_normal_(token, std=0.02)
                self.tokens[self._token_key(tap, role)] = token

    def tap_dim(self, tap: str) -> int:
        if tap not in self._tap_width:
            raise ValueError(f"{type(self).__name__} has no tap {tap!r}; it has {self.TAPS}")
        return int(self._tap_width[tap])

    @staticmethod
    def _token_key(tap: str, role: str) -> str:
        return f"{tap}__{role}"  # ParameterDict keys may not contain "."

    def _tokens_at(self, tap: str) -> dict[str, torch.Tensor]:
        return {role: self.tokens[self._token_key(tap, role)] for role in self.inject_roles}

    def _skip(
        self, skip: Voxels, tap: str, inject: Injection | None
    ) -> tuple[Voxels, list[InjectionGroup]]:
        if inject is None or not inject.at(tap):
            return skip, []
        return inject_into_skip(skip, tap, self.TAP_STRIDE[tap], inject, self._tokens_at(tap))

    def forward(
        self, xs: Voxels, inject: Injection | None = None, taps: Iterable[str] = ()
    ) -> FeatureBundle:
        taps = self.check_request(inject, taps)

        enc0 = self.conv0(xs)
        enc1 = self.block1(self.conv1(enc0))
        bott = self.block2(self.conv2(enc1))
        bott_attn = self.bottleneck(bott)

        # Injection happens at the skips, so a token changes both what the decoder reads and
        # where it emits: the transposed convolution takes its output geometry from the skip.
        skip1, inj1 = self._skip(enc1, "enc1", inject)
        skip0, inj0 = self._skip(enc0, "enc0", inject)

        dec_half = self.block6(cat(self.convtr5(bott_attn, skip1), skip1))
        dec_full = self.block8(cat(self.convtr7(dec_half, skip0), skip0))
        out = self.final(dec_full)

        values = {
            "enc0": enc0,
            "enc1": enc1,
            "bottleneck": bott,
            "bottleneck_attn": bott_attn,
            "dec_half": dec_half,
            "dec_full": dec_full,
        }
        return FeatureBundle(
            out=out,
            taps={t: values[t] for t in taps},
            injected=Injection(inj1 + inj0) if inject is not None else None,
        )
