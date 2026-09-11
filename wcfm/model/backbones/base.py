"""What a backbone is, within `wcfm.model`. This is not a framework interface.

One return type, always: `FeatureBundle`. A term that needs the half-resolution decoder stage
asks for it by tap name, so the shape of the return does not change with what a run trains.

Injection is role-typed. A token is keyed by `(tap, role)`, a request names its role, and the
backbone reports what it actually injected. Tokens keyed by resolution alone would hand DINO's
placeholders and occupancy's candidates the identical learned vector at one skip -- two
different questions, no way to tell them apart, and labels that stay correct, so it fails
silently.

Coordinates in a request may be at any stride that divides the tap's: the masker removes
pixels at full resolution (`stride=1`) and the backbone projects them onto the half-res skip,
while a region masker enumerates occupancy candidates already at `stride=2` and they land on
that skip untouched. `inject_into_skip` is the one place that arithmetic lives, and the dedupe
against the skip -- a requested coordinate the skip already carries is not injected, since the
encoder computed a feature there -- is reported back through `FeatureBundle.injected`, so a
term scores exactly the set that was placed and nothing it merely asked for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import ClassVar

import torch
from torch import Tensor, nn
from warpconvnet.geometry.coords.integer import IntCoords
from warpconvnet.geometry.features.cat import CatFeatures
from warpconvnet.geometry.types.voxels import Voxels

ROLES: tuple[str, ...] = ("masked", "candidate")


@dataclass
class InjectionGroup:
    """One request: put the `role` token at these coordinates, at this tap.

    `coords` is per image, `[N_b, 2]`, in units of `stride` relative to full resolution.
    """

    tap: str
    coords: list[Tensor]
    role: str
    stride: int = 1

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"unknown injection role {self.role!r}; expected one of {ROLES}")
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1, got {self.stride}")


@dataclass
class Injection:
    groups: list[InjectionGroup] = field(default_factory=list)

    def at(self, tap: str) -> list[InjectionGroup]:
        return [g for g in self.groups if g.tap == tap]

    def roles(self) -> set[str]:
        return {g.role for g in self.groups}

    def taps(self) -> set[str]:
        return {g.tap for g in self.groups}

    def __bool__(self) -> bool:
        return bool(self.groups)

    @staticmethod
    def merge(parts: Iterable[Injection | None]) -> Injection | None:
        """The union of several terms' requests; `None` when nothing asked for anything."""
        groups = [g for part in parts if part is not None for g in part.groups]
        return Injection(groups) if groups else None


@dataclass
class FeatureBundle:
    out: Voxels
    """What terms consume by default: the final feature map."""
    taps: dict[str, Voxels] = field(default_factory=dict)
    """Named intermediates; empty unless requested."""
    injected: Injection | None = None
    """What was actually injected, after the skip dedupe: one group per `(tap, role)`, in the
    tap's own units."""


class Backbone(nn.Module, ABC):
    """The contract a term codes against. `out_dim` is what a head is built from."""

    out_dim: int
    TAPS: ClassVar[tuple[str, ...]] = ()
    TAP_STRIDE: ClassVar[dict[str, int]] = {}
    INJECT_TAPS: ClassVar[tuple[str, ...]] = ()
    # module attribute name -> parameter-name prefixes, for the gradient taxonomy
    GRAD_GROUPS: ClassVar[dict[str, tuple[str, ...]]] = {}
    inject_roles: tuple[str, ...] = ()

    def tap_dim(self, tap: str) -> int:
        """Channel width at `tap`. A term reading a tap builds its head from this rather than
        from `out_dim`: the two coincide on the shipped widths, and a subclass is free to make
        them differ, which would be a silent shape bug at the first forward."""
        raise NotImplementedError(
            f"{type(self).__name__} does not publish tap widths; a term that reads a tap "
            "needs tap_dim()"
        )

    @property
    def supports_injection(self) -> bool:
        return bool(self.inject_roles) and bool(self.INJECT_TAPS)

    @abstractmethod
    def forward(
        self, xs: Voxels, inject: Injection | None = None, taps: Iterable[str] = ()
    ) -> FeatureBundle: ...

    def check_request(self, inject: Injection | None, taps: Iterable[str]) -> tuple[str, ...]:
        """Refuse what this backbone cannot honour, before a forward silently ignores it."""
        taps = tuple(taps)
        unknown = sorted(set(taps) - set(self.TAPS))
        if unknown:
            raise ValueError(f"{type(self).__name__} has no tap {unknown}; it has {self.TAPS}")
        if inject:
            if not self.supports_injection:
                raise ValueError(
                    f"{type(self).__name__} holds no mask tokens (inject_roles is empty) and "
                    f"cannot inject; a term requested roles {sorted(inject.roles())}"
                )
            missing = sorted(inject.roles() - set(self.inject_roles))
            if missing:
                raise ValueError(
                    f"{type(self).__name__} has no token for roles {missing}; its "
                    f"inject_roles are {list(self.inject_roles)}"
                )
            bad_taps = sorted(inject.taps() - set(self.INJECT_TAPS))
            if bad_taps:
                raise ValueError(
                    f"{type(self).__name__} does not inject at {bad_taps}; it injects at "
                    f"{self.INJECT_TAPS}"
                )
        return taps


# --------------------------------------------------------------------- coordinate arithmetic


def _empty_like(coords: Tensor, coord_dim: int) -> Tensor:
    return coords.new_zeros(0, coord_dim)


def project_coords(coords_per_image: list[Tensor], skip: Voxels, factor: int) -> list[Tensor]:
    """Coordinates onto a skip's grid, minus those the skip already carries.

    Floor-divide by `factor` (WarpConvNet's striding convention), deduplicate, then drop any
    coordinate present in that image's slice of the skip.

    Each `(x, y)` is packed into a single integer `y*W + x` for the membership test, so `W`
    must exceed the largest `x` in both operands. Sizing it from the skip alone is wrong
    whenever a projected coordinate sits further right than anything left in the skip, which is
    exactly what masking produces, since the masker removed the pixels the skip no longer has:
    the wrapped key can equal a real skip key, the coordinate is dropped as a duplicate it is
    not, and no token lands there. `tests/test_model_inject.py` pins it.

    `skip.offsets` is the authority on the batch size: WarpConvNet's strided conv drops
    trailing empty images from its offsets, so the request list may be longer than the skip.
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    B = len(skip.offsets) - 1
    coord_dim = skip.coordinate_tensor.shape[1]
    skip_coords = skip.coordinate_tensor
    requests = [
        coords_per_image[b] if b < len(coords_per_image) else _empty_like(skip_coords, coord_dim)
        for b in range(B)
    ]

    x_maxes: list[Tensor] = []
    if skip_coords.shape[0] > 0:
        x_maxes.append(skip_coords[:, 0].max().to(torch.int64))
    x_maxes += [
        torch.div(c[:, 0].max(), factor, rounding_mode="floor").to(torch.int64)
        for c in requests
        if c.shape[0] > 0
    ]
    W = int(torch.stack(x_maxes).max().item()) + 2 if x_maxes else 1  # one sync per call

    out: list[Tensor] = []
    for b in range(B):
        c = requests[b]
        if c.shape[0] == 0:
            out.append(_empty_like(skip_coords, coord_dim))
            continue
        low = torch.div(c, factor, rounding_mode="floor").to(c.dtype) if factor > 1 else c
        low = torch.unique(low, dim=0)
        s, e = int(skip.offsets[b]), int(skip.offsets[b + 1])
        sk = skip_coords[s:e]
        if sk.shape[0] > 0:
            sk_keys = sk[:, 1].long() * W + sk[:, 0].long()
            keys = low[:, 1].long() * W + low[:, 0].long()
            sk_sorted, _ = sk_keys.sort()
            pos = torch.searchsorted(sk_sorted, keys).clamp(max=sk_sorted.shape[0] - 1)
            low = low[sk_sorted[pos] != keys]
        out.append(low.to(skip_coords.dtype))
    return out


def inject_into_skip(
    skip: Voxels,
    tap: str,
    tap_stride: int,
    inject: Injection | None,
    tokens: dict[str, Tensor],
) -> tuple[Voxels, list[InjectionGroup]]:
    """A skip with the requested coordinates appended, each carrying its role's token.

    Groups at this tap are first unioned per role -- the DINO term and the charge term both ask
    for the masked coordinates, and one token per coordinate is the point -- then deduped
    against the skip. A coordinate requested under two different roles at one tap is refused:
    the two roles are two questions, and one token cannot answer both.

    Returns the augmented skip and one `InjectionGroup` per role actually placed, in the tap's
    units, so a term can score exactly that set. Preserves `tensor_stride`: the half-res skip
    has `(2, 2)` and losing it breaks the transposed convolution's checks.
    """
    groups = inject.at(tap) if inject is not None else []
    if not groups:
        return skip, []

    B = len(skip.offsets) - 1
    coords = skip.coordinate_tensor
    feats = skip.feature_tensor
    coord_dim, C = coords.shape[1], feats.shape[1]
    device = coords.device

    # Per role: project every group onto this tap, union per image, dedupe against the skip.
    by_role: dict[str, list[Tensor]] = {}
    for role in sorted({g.role for g in groups}):  # sorted: identical order on every rank
        if role not in tokens:
            raise ValueError(f"no token for role {role!r} at tap {tap!r}")
        per_image: list[list[Tensor]] = [[] for _ in range(B)]
        for g in (g for g in groups if g.role == role):
            if tap_stride % g.stride:
                raise ValueError(
                    f"request at stride {g.stride} cannot be projected onto tap {tap!r} at "
                    f"stride {tap_stride}: not divisible"
                )
            for b, c in enumerate(project_coords(g.coords, skip, tap_stride // g.stride)):
                per_image[b].append(c)
        merged: list[Tensor] = []
        for b in range(B):
            parts = [c for c in per_image[b] if c.shape[0] > 0]
            if not parts:
                merged.append(_empty_like(coords, coord_dim))
            elif len(parts) == 1:
                merged.append(parts[0])
            else:
                merged.append(torch.unique(torch.cat(parts, dim=0), dim=0))
        by_role[role] = merged

    roles = list(by_role)
    if len(roles) > 1:
        _refuse_role_collisions(by_role, tap)

    new_coords: list[Tensor] = []
    new_feats: list[Tensor] = []
    for b in range(B):
        s, e = int(skip.offsets[b]), int(skip.offsets[b + 1])
        parts_c = [coords[s:e]]
        parts_f = [feats[s:e]]
        for role in roles:
            c = by_role[role][b]
            if c.shape[0] == 0:
                continue
            parts_c.append(c)
            parts_f.append(tokens[role].to(feats.dtype).unsqueeze(0).expand(c.shape[0], -1))
        new_coords.append(torch.cat(parts_c, dim=0))
        new_feats.append(torch.cat(parts_f, dim=0))

    counts = torch.tensor([c.shape[0] for c in new_coords], dtype=torch.int64, device=device)
    new_offsets = torch.cat([torch.zeros(1, dtype=torch.int64, device=device), counts.cumsum(0)])
    all_coords = torch.cat(new_coords, dim=0) if new_coords else _empty_like(coords, coord_dim)
    all_feats = torch.cat(new_feats, dim=0) if new_feats else feats.new_zeros(0, C)
    ts = skip.batched_coordinates.tensor_stride
    augmented = Voxels(
        batched_coordinates=IntCoords(all_coords, offsets=new_offsets, tensor_stride=ts),
        batched_features=CatFeatures(all_feats, offsets=new_offsets),
        offsets=new_offsets,
    )
    injected = [
        InjectionGroup(tap=tap, coords=by_role[role], role=role, stride=tap_stride)
        for role in roles
    ]
    return augmented, injected


def _refuse_role_collisions(by_role: dict[str, list[Tensor]], tap: str) -> None:
    B = len(next(iter(by_role.values())))
    for b in range(B):
        stacks = [c for c in (by_role[r][b] for r in by_role) if c.shape[0] > 0]
        if len(stacks) < 2:
            continue
        allc = torch.cat(stacks, dim=0)
        if torch.unique(allc, dim=0).shape[0] != allc.shape[0]:
            raise ValueError(
                f"tap {tap!r}, image {b}: a coordinate was requested under two roles "
                f"{sorted(by_role)}. Two roles are two questions and one token cannot answer "
                "both."
            )
