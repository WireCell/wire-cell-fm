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

from .base import Backbone, FeatureBundle, Injection, InjectionGroup, inject_into_skip, upcast
from .blocks import (
    BottleneckSparseAttention2D,
    ConvBlock2D,
    ConvTrBlock2D,
    ResidualSparseBlock2D,
)


class MinkUNetAttention(Backbone):
    """Sparse encoder (2 strided stages, 3 with `stage3`), attention bottleneck, sparse decoder
    with skips.

    Taps, in forward order, with their stride relative to the input (`stage3` in brackets):

        enc0             1      after conv0: the full-res skip (and injection site)
        enc1             2      after block1: the half-res skip (and injection site)
        enc2             (4)    after block2: the quarter-res skip; only with `stage3`
        bottleneck       4 (8)  before the bottleneck blocks
        bottleneck_attn  4 (8)  after the bottleneck blocks
        dec_quarter      (4)    after block5; only with `stage3`
        dec_half         2      after block6
        dec_full         1      after block8, before `final`

    `out` is `final(dec_full)`, in `out_dim` channels.

    `bottleneck_depth` stacks that many bottleneck blocks. `stage3` adds a third strided stage
    (`conv3`, `block3`, at the bottleneck's width) before the bottleneck. A decoder then opens
    with `convtr4` and `block5`, which return to stride 4 on `enc2`'s rows and, under `unet`,
    concatenate `enc2`. Tokens are never injected at `enc2`: the terms inject the `masked` role
    at `enc0` and the `candidate` role at `enc1`, and the decoder reaches their rows through
    those skips.

    `decoder` picks what follows the bottleneck. `unet` concatenates each skip into the
    decoder. `plain` runs the same decoder on its own input, taking only its output coordinates
    from the skips. `none` builds no decoder: every pixel of `enc0` takes the features of its
    parents at the `upcast` taps, concatenated, and `out` is `final` of that, on `enc0`'s rows.
    `dec_half` and `dec_full` do not exist then.

    `inject_at` picks where tokens enter. `skips`: into the skips after the encoder, so only
    the decoder reads them, and the `enc*` taps are the encoder's outputs without them.
    `input`: into `enc0` after `conv0` and into `enc1` before `conv2`, so they pass through
    every later stage, and the `enc*` taps include them. Without the `unet` decoder nothing
    reads a skip, so injection there is refused.
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
        "enc3": ("conv3.", "block3."),
        "bottleneck": ("bottleneck.",),
        "dec0": ("convtr4.", "block5."),
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
        decoder: str = "unet",
        upcast: Iterable[str] = ("bottleneck_attn",),
        inject_at: str = "skips",
        bottleneck_depth: int = 1,
        stage3: bool = False,
    ):
        super().__init__()
        if decoder not in ("unet", "plain", "none"):
            raise ValueError(f"decoder must be 'unet', 'plain' or 'none', got {decoder!r}")
        if inject_at not in ("skips", "input"):
            raise ValueError(f"inject_at must be 'skips' or 'input', got {inject_at!r}")
        if bottleneck_depth < 1:
            raise ValueError(f"bottleneck_depth must be at least 1, got {bottleneck_depth}")
        stem, e1, e2, dec = (int(w) for w in widths)
        self.stage3 = stage3
        taps = type(self).TAPS
        if stage3:
            # Instance attributes: the class tables describe the two-stage network.
            self.TAP_STRIDE = type(self).TAP_STRIDE | {
                "enc2": 4,
                "bottleneck": 8,
                "bottleneck_attn": 8,
                "dec_quarter": 4,
            }
            taps = ("enc0", "enc1", "enc2", "bottleneck", "bottleneck_attn", "dec_quarter")
            taps += ("dec_half", "dec_full")
        if decoder == "none":
            taps = tuple(t for t in taps if not t.startswith("dec_"))
        self.TAPS = taps
        self.out_dim = int(out_dim)
        self.inject_roles = tuple(inject_roles)
        self.decoder = decoder
        self.inject_at = inject_at
        self.upcast = tuple(upcast) if decoder == "none" else ()
        unknown = sorted(set(self.upcast) - set(self.TAPS))
        if unknown or (decoder == "none" and not self.upcast):
            raise ValueError(f"upcast must be a non-empty subset of {self.TAPS}, got {unknown}")

        self.conv0 = ConvBlock2D(in_ch, stem, kernel_size=3, stride=1)
        self.conv1 = ConvBlock2D(stem, e1, kernel_size=2, stride=2)
        self.block1 = ResidualSparseBlock2D(e1, e1, kernel_size=3)
        self.conv2 = ConvBlock2D(e1, e1, kernel_size=2, stride=2)
        self.block2 = ResidualSparseBlock2D(e1, e2, kernel_size=3)
        if stage3:
            self.conv3 = ConvBlock2D(e2, e2, kernel_size=2, stride=2)
            self.block3 = ResidualSparseBlock2D(e2, e2, kernel_size=3)

        def bottleneck_block() -> nn.Module:
            if not attention:
                return ResidualSparseBlock2D(e2, e2, kernel_size=3)
            return BottleneckSparseAttention2D(
                channels=e2,
                attn_channels=attn_ch,
                heads=heads,
                encoding=spatial_encoding,
                flash=flash_attention,
                encoding_range=encoding_range,
                encoding_channels=encoding_dim,
            )

        blocks = [bottleneck_block() for _ in range(bottleneck_depth)]
        # A single block stays unwrapped so depth-1 checkpoints keep their `bottleneck.*` keys.
        self.bottleneck: nn.Module = blocks[0] if bottleneck_depth == 1 else nn.Sequential(*blocks)
        # Built in forward order: initial weights depend on it through the RNG stream.
        tap_width = {"enc0": stem, "enc1": e1, "bottleneck": e2, "bottleneck_attn": e2}
        if stage3:
            tap_width |= {"enc2": e2, "dec_quarter": e2}
        if decoder == "none":
            final_in = sum(tap_width[t] for t in self.upcast)
        else:
            unet = decoder == "unet"
            if stage3:
                self.convtr4 = ConvTrBlock2D(e2, e2, kernel_size=2, stride=2)
                self.block5 = ResidualSparseBlock2D(e2 + e2 * unet, e2, kernel_size=3)
            self.convtr5 = ConvTrBlock2D(e2, dec, kernel_size=2, stride=2)
            self.block6 = ResidualSparseBlock2D(dec + e1 * unet, dec, kernel_size=3)
            self.convtr7 = ConvTrBlock2D(dec, dec, kernel_size=2, stride=2)
            self.block8 = ResidualSparseBlock2D(dec + stem * unet, dec, kernel_size=3)
            tap_width |= {"dec_half": dec, "dec_full": dec}
            final_in = dec
        self.final = SparseConv2d(final_in, self.out_dim, kernel_size=1, bias=True)

        # One token per (skip, role), in that skip's width. Built after the network, as the
        # old tokens were, so every other parameter draws from the same point in the RNG
        # stream whether or not a run injects.
        self._skip_width = {"enc0": stem, "enc1": e1}
        # Every tap's width, so a term reading one can size its head. `dec_half` is block6's
        # output and is what the occupancy head reads: one question per 2x2 full-res block.
        self._tap_width = tap_width
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
        device_type = xs.feature_tensor.device.type
        if torch.is_autocast_enabled(device_type) and (
            torch.get_autocast_dtype(device_type) == torch.bfloat16
        ):
            # A bf16 run trains on garbage or skips every step as non-finite, depending on the
            # head; `test_bf16_autocast_is_refused` pins the refusal.
            raise RuntimeError(
                "MinkUNetAttention does not run under bf16 autocast: warpconvnet's bf16 sparse "
                "convolution is numerically wrong. Set run.precision=32-true."
            )
        at_input = self.inject_at == "input"
        if inject and not at_input and self.decoder != "unet":
            raise ValueError(
                f"decoder={self.decoder!r} reads no skip, so tokens injected there would reach "
                "nothing; set inject_at='input'"
            )

        inj0: list[InjectionGroup] = []
        inj1: list[InjectionGroup] = []
        enc0 = self.conv0(xs)
        if at_input:
            enc0, inj0 = self._skip(enc0, "enc0", inject)
        enc1 = self.block1(self.conv1(enc0))
        if at_input:
            enc1, inj1 = self._skip(enc1, "enc1", inject)
        enc2 = bott = self.block2(self.conv2(enc1))
        if self.stage3:
            bott = self.block3(self.conv3(enc2))
        bott_attn = self.bottleneck(bott)
        # Only the names in `self.TAPS` are served: `enc2` and `dec_quarter` without `stage3`
        # are `bottleneck` and `bottleneck_attn` under another name.
        values = {
            "enc0": enc0,
            "enc1": enc1,
            "enc2": enc2,
            "bottleneck": bott,
            "bottleneck_attn": bott_attn,
        }

        if self.decoder == "none":
            parents = [upcast(values[t], enc0, self.TAP_STRIDE[t]) for t in self.upcast]
            out = self.final(enc0.replace(batched_features=torch.cat(parents, dim=1)))
        else:
            # A token at a skip changes both what the decoder reads and where it emits: the
            # transposed convolution takes its output geometry from the skip.
            skip1, skip0 = enc1, enc0
            if not at_input:
                skip1, inj1 = self._skip(enc1, "enc1", inject)
                skip0, inj0 = self._skip(enc0, "enc0", inject)
            dec_quarter = bott_attn
            if self.stage3:
                up2 = self.convtr4(bott_attn, enc2)
                dec_quarter = self.block5(cat(up2, enc2) if self.decoder == "unet" else up2)
            up1 = self.convtr5(dec_quarter, skip1)
            dec_half = self.block6(cat(up1, skip1) if self.decoder == "unet" else up1)
            up0 = self.convtr7(dec_half, skip0)
            dec_full = self.block8(cat(up0, skip0) if self.decoder == "unet" else up0)
            out = self.final(dec_full)
            values |= {"dec_quarter": dec_quarter, "dec_half": dec_half, "dec_full": dec_full}

        return FeatureBundle(
            out=out,
            taps={t: values[t] for t in taps},
            injected=Injection(inj1 + inj0) if inject is not None else None,
        )
