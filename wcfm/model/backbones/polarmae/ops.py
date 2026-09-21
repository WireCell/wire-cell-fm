"""Point-set operations for the PoLAr-MAE tokenizer and losses, in plain torch.

The conventions are pytorch3d's, whose compiled kernels the original model calls:

- distances are squared Euclidean over every coordinate the caller passes;
- `knn_points` pads a short row with index 0 and distance `inf`;
- `ball_query` returns the first `K` points strictly inside the radius in ascending index
  order and pads with -1; `sample_farthest_points` pads with -1 as well;
- a `lengths` argument masks the padded tail of each event in a `(B, N, D)` batch.

Distance matrices are computed in chunks of query rows so the `(B, chunk, P)` intermediate
stays under `CHUNK_ELEMENTS` floats on dense events. `grid_ball_query` is `ball_query` for
clouds on a lattice, which every pixel cloud is: it reads the candidates out of an index image
instead of a distance matrix and returns the same indices.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

__all__ = [
    "ball_query",
    "chamfer_distance",
    "cnms",
    "grid_ball_query",
    "knn_points",
    "masked_gather",
    "sample_farthest_points",
    "sq_dists",
]

#: Upper bound on the elements of one `(B, chunk, P)` distance block.
CHUNK_ELEMENTS = 1 << 28


def sq_dists(p1: Tensor, p2: Tensor) -> Tensor:
    """`(B, c, D) x (B, P, D) -> (B, c, P)` squared Euclidean distances, clamped at zero.

    Computed in fp32 whatever autocast the caller runs under. The grouping runs inside the
    training forward, and autocast would send `bmm` to bf16: coordinates are of order one and
    the production radius squared is 7e-5, so a bf16 product puts most of a group outside its
    own ball and cnms retains three times the centres.
    `tests/test_model_polarmae_ops.py::test_queries_ignore_autocast` pins it.
    """
    with torch.autocast(device_type=p1.device.type, enabled=False):
        p1 = p1.float()
        p2 = p2.float()
        d = (
            (p1 * p1).sum(-1, keepdim=True)
            + (p2 * p2).sum(-1).unsqueeze(1)
            - 2.0 * torch.bmm(p1, p2.transpose(1, 2))
        )
        return d.clamp_min(0.0)


def _exact_sq_dists(p1: Tensor, p2: Tensor) -> Tensor:
    """`sq_dists` formed as pytorch3d's kernels form it: the difference per coordinate, squared
    and summed in coordinate order, in fp32. `sq_dists` expands the square through `bmm`, and
    the cancellation there is enough to move a point that sits on the radius of a ball, which
    on a pixel lattice happens at every `(3, 4)` and `(5, 0)` offset of a radius-5 ball. A
    membership test needs this form; a nearest-neighbour search does not."""
    with torch.autocast(device_type=p1.device.type, enabled=False):
        diff = p1.float().unsqueeze(2) - p2.float().unsqueeze(1)  # (B, c, P, D)
        d2 = torch.zeros_like(diff[..., 0])
        for d in range(diff.shape[-1]):
            dd = diff[..., d]
            d2 = d2 + dd * dd
        return d2


def _chunk_rows(B: int, P: int) -> int:
    return max(1, min(1024, CHUNK_ELEMENTS // max(1, B * P)))


def _radius2(radius: float) -> float:
    """`radius * radius` as pytorch3d's kernels form it: the radius rounded to fp32 first, then
    squared in fp32. A distance compared against the double product instead moves the points
    that sit exactly on the radius, which on a pixel lattice are the `(3, 4)` and `(5, 0)`
    offsets of a radius-5 ball."""
    return float(torch.tensor(float(radius), dtype=torch.float32) ** 2)


def _valid(lengths: Tensor | None, B: int, P: int, device) -> Tensor:
    if lengths is None:
        return torch.ones(B, P, dtype=torch.bool, device=device)
    return torch.arange(P, device=device).unsqueeze(0) < lengths.to(device).unsqueeze(1)


def knn_points(
    p1: Tensor,
    p2: Tensor,
    lengths1: Tensor | None = None,
    lengths2: Tensor | None = None,
    K: int = 1,
    return_sorted: bool = True,
) -> tuple[Tensor, Tensor]:
    """The `K` nearest points of `p2` for every point of `p1`: `(dists, idx)`, both
    `(B, P1, K)`.

    A row of `p2` shorter than `K` fills the tail with distance `inf` and index 0, and rows of
    `p1` beyond `lengths1` get index 0 throughout. Differentiable in the distances, which is
    what `chamfer_distance` relies on.
    """
    B, P1, _ = p1.shape
    P2 = p2.shape[1]
    device = p1.device
    valid2 = _valid(lengths2, B, P2, device)
    K = int(K)
    Keff = min(K, P2)
    dists_out = p1.new_empty((B, P1, K), dtype=torch.float32)
    idx_out = torch.empty((B, P1, K), dtype=torch.int64, device=device)
    step = _chunk_rows(B, P2)
    for s in range(0, P1, step):
        e = min(s + step, P1)
        d2 = sq_dists(p1[:, s:e], p2)
        d2 = d2.masked_fill(~valid2.unsqueeze(1), float("inf"))
        dists, idx = torch.topk(d2, Keff, dim=2, largest=False, sorted=return_sorted)
        if Keff < K:
            pad = K - Keff
            dists = torch.cat([dists, dists.new_full((B, e - s, pad), float("inf"))], dim=2)
            idx = torch.cat([idx, idx.new_zeros((B, e - s, pad))], dim=2)
        idx = idx.masked_fill(torch.isinf(dists), 0)
        dists_out[:, s:e] = dists
        idx_out[:, s:e] = idx
    if lengths1 is not None:
        valid1 = _valid(lengths1, B, P1, device)
        idx_out = idx_out.masked_fill(~valid1.unsqueeze(-1), 0)
    return dists_out, idx_out


@torch.no_grad()
def ball_query(
    p1: Tensor,
    p2: Tensor,
    K: int,
    radius: float,
    lengths1: Tensor | None = None,
    lengths2: Tensor | None = None,
) -> Tensor:
    """Indices into `p2` of the first `K` points strictly within `radius` of each point of
    `p1`, in ascending `p2` index order, `(B, P1, K)`, -1 where fewer than `K` were found.

    The order is what makes the result reproducible: a point cloud sorted by coordinate hands
    every group its lowest-index members, and `cnms` inherits that when it truncates.
    """
    B, P1, D = p1.shape
    P2 = p2.shape[1]
    device = p1.device
    valid2 = _valid(lengths2, B, P2, device)
    K = int(K)
    Keff = min(K, P2)
    r2 = _radius2(radius)
    ar = torch.arange(P2, device=device)
    idx_out = torch.empty((B, P1, K), dtype=torch.int64, device=device)
    step = max(1, _chunk_rows(B, P2) // D)
    for s in range(0, P1, step):
        e = min(s + step, P1)
        c = e - s
        d2 = _exact_sq_dists(p1[:, s:e], p2)
        within = (d2 < r2) & valid2.unsqueeze(1)
        # Points inside the ball score their own index, the rest score past `P2`, so the `Keff`
        # smallest scores are the first `Keff` members in index order.
        score = torch.where(within, ar.view(1, 1, P2), ar.view(1, 1, P2) + P2)
        vals, _ = torch.topk(score, Keff, dim=2, largest=False, sorted=True)
        idx = torch.where(vals < P2, vals, torch.full_like(vals, -1))
        if Keff < K:
            idx = torch.cat([idx, idx.new_full((B, c, K - Keff), -1)], dim=2)
        idx_out[:, s:e] = idx
    if lengths1 is not None:
        valid1 = _valid(lengths1, B, P1, device)
        idx_out = idx_out.masked_fill(~valid1.unsqueeze(-1), -1)
    return idx_out


@torch.no_grad()
def grid_ball_query(
    p1: Tensor,
    p2: Tensor,
    K: int,
    radius: float,
    pitch: float,
    lengths1: Tensor | None = None,
    lengths2: Tensor | None = None,
) -> Tensor:
    """`ball_query` for clouds whose first two coordinates lie on a square lattice of `pitch`
    and whose remaining coordinates are the same for every point.

    The valid points of `p2` are written into a `(B, H, W)` index image over the batch's
    bounding box, each query point reads the `(2w + 1)^2` cells around its own, `w` being
    `ceil(radius / pitch)`, and the candidates are tested with the same fp32 distance and the
    same strict comparison as `ball_query`. Same result, and no `(B, P1, P2)` intermediate:
    the cost is `B * P1 * (2w + 1)^2` whatever the size of the event.

    Two rules the image imposes. The valid points of an event of `p2` must occupy distinct
    lattice sites, or a site would keep one of its points and drop the others in silence; the
    function raises when the count disagrees. And a coordinate has to round to its site, so
    `p / pitch` must be within 0.5 of an integer.
    `tests/test_model_polarmae_ops.py::test_grid_ball_query_matches_ball_query` pins the
    equality.
    """
    with torch.autocast(device_type=p1.device.type, enabled=False):
        p1 = p1.float()
        p2 = p2.float()
        B, P1, D = p1.shape
        P2 = p2.shape[1]
        device = p1.device
        K = int(K)
        idx_out = torch.full((B, P1, K), -1, dtype=torch.int64, device=device)
        valid1 = _valid(lengths1, B, P1, device)
        valid2 = _valid(lengths2, B, P2, device)
        if K == 0 or not bool(valid1.any()) or not bool(valid2.any()):
            return idx_out
        w = int(math.ceil(float(radius) / float(pitch)))
        pix1 = torch.round(p1[..., :2] / pitch).to(torch.int64)
        pix2 = torch.round(p2[..., :2] / pitch).to(torch.int64)
        lo = torch.minimum(pix1[valid1].amin(0), pix2[valid2].amin(0)) - w
        hi = torch.maximum(pix1[valid1].amax(0), pix2[valid2].amax(0)) + w
        H, W = (int(hi[0] - lo[0]) + 1, int(hi[1] - lo[1]) + 1)
        img = torch.full((B, H, W), -1, dtype=torch.int64, device=device)
        rows2 = torch.arange(B, device=device).unsqueeze(1).expand(B, P2)[valid2]
        q2 = (pix2 - lo)[valid2]
        img[rows2, q2[:, 0], q2[:, 1]] = torch.arange(P2, device=device).expand(B, P2)[valid2]
        n_sites = int(img.ge(0).sum())
        if n_sites != int(valid2.sum()):
            raise ValueError(
                f"grid_ball_query needs one point per lattice site; {int(valid2.sum())} points "
                f"occupy {n_sites} sites"
            )
        off = torch.arange(-w, w + 1, device=device)
        dy, dx = torch.meshgrid(off, off, indexing="ij")
        dy = dy.reshape(-1)
        dx = dx.reshape(-1)
        n_off = int(dy.numel())
        r2 = _radius2(radius)
        # Padding rows of `p1` hold zeros that may fall outside the image; they are masked at
        # the end, so their cells only have to be in range.
        q1 = pix1 - lo
        q1[..., 0].clamp_(w, H - 1 - w)
        q1[..., 1].clamp_(w, W - 1 - w)
        Keff = min(K, n_off)
        step = max(1, CHUNK_ELEMENTS // max(1, B * n_off * D))
        rows = torch.arange(B, device=device).view(B, 1, 1)
        for s in range(0, P1, step):
            e = min(s + step, P1)
            cy = q1[:, s:e, 0].unsqueeze(-1) + dy
            cx = q1[:, s:e, 1].unsqueeze(-1) + dx
            cand = img[rows, cy, cx]  # (B, c, n_off)
            nb = masked_gather(p2, cand)  # (B, c, n_off, D)
            diff = p1[:, s:e].unsqueeze(2) - nb
            d2 = torch.zeros_like(diff[..., 0])
            for d in range(D):
                dd = diff[..., d]
                d2 = d2 + dd * dd
            within = cand.ge(0) & (d2 < r2)
            score = torch.where(within, cand, torch.full_like(cand, P2))
            vals, _ = score.sort(dim=-1)
            vals = vals[..., :Keff]
            idx_out[:, s:e, :Keff] = torch.where(vals < P2, vals, torch.full_like(vals, -1))
        return idx_out.masked_fill(~valid1.unsqueeze(-1), -1)


@torch.no_grad()
def sample_farthest_points(
    points: Tensor, lengths: Tensor, K: int, start_idxs: Tensor | None = None
) -> Tensor:
    """Farthest point sampling: `(B, K)` indices into `points`, -1 past `min(lengths, K)`.

    Position 0 is `start_idxs` (0 when not given); every following pick is the valid point
    farthest from the picks so far, ties going to the lowest index.
    """
    points = points[..., :3].float()
    B, P, _ = points.shape
    device = points.device
    lengths = lengths.to(device=device, dtype=torch.int64)
    K = int(K)
    idx_out = torch.full((B, K), -1, dtype=torch.int64, device=device)
    if B == 0 or K == 0:
        return idx_out
    k_n = lengths.clamp(min=0, max=K)
    valid = torch.arange(P, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    rows = torch.arange(B, device=device)
    if start_idxs is None:
        sel = torch.zeros(B, dtype=torch.int64, device=device)
    else:
        sel = start_idxs.to(device=device, dtype=torch.int64).clamp(min=0, max=max(P - 1, 0))
    write0 = k_n > 0
    idx_out[write0, 0] = sel[write0]
    closest = torch.full((B, P), float("inf"), device=device)
    neg = torch.tensor(float("-inf"), device=device)
    for i in range(1, K):
        sel_pts = points[rows, sel]
        d = ((points - sel_pts.unsqueeze(1)) ** 2).sum(-1)
        closest = torch.minimum(closest, d)
        sel = torch.where(valid, closest, neg).argmax(dim=1)
        write = i < k_n
        idx_out[write, i] = sel[write]
    return idx_out


def masked_gather(points: Tensor, idx: Tensor) -> Tensor:
    """Gather `points[b, idx[b, ...]]` where `idx` is `(B, K)` or `(B, G, K)` and may hold -1.

    A -1 gathers point 0; the caller masks those slots by `idx.eq(-1)` rather than by value.
    """
    if idx.shape[0] != points.shape[0]:
        raise ValueError("points and idx must have the same batch dimension")
    D = points.shape[-1]
    idx = idx.clamp(min=0)
    if idx.ndim == 3:
        B, G, K = idx.shape
        flat = idx.reshape(B, G * K)
        out = points.gather(1, flat.unsqueeze(-1).expand(-1, -1, D))
        return out.reshape(B, G, K, D)
    if idx.ndim == 2:
        return points.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))
    raise ValueError(f"idx format is not supported {tuple(idx.shape)}")


@torch.no_grad()
def cnms(
    centroids: Tensor,
    radius: float,
    overlap_factor: float,
    K: int,
    lengths: Tensor,
    pitch: float | None = None,
) -> tuple[Tensor, Tensor]:
    """Centrality-based non-maximum suppression over candidate group centres.

    Every centroid counts the candidates within `2 * radius * overlap_factor` of it (at most
    `K`, in index order), through `grid_ball_query` when `pitch` is given and `ball_query`
    otherwise. Candidates are visited in descending count; a candidate not yet
    suppressed is retained and suppresses everything in its ball. Returns the centroids
    reordered retained-first, in ascending index order within each half, and the retained count
    per event.

    The greedy pass runs in rounds instead of one candidate at a time: a round retains every
    unresolved candidate that no unresolved candidate of higher count can still suppress, then
    suppresses their balls. The retained set equals the sequential visit's, and the number of
    rounds is the depth of the suppression chain rather than the number of centres.
    `tests/test_model_polarmae_ops.py` pins the equality against a sequential reference. The
    retained set is what PoLAr-MAE's compiled kernel produces under a stable candidate order;
    that kernel's own caller sorts the counts unstably, and most counts tie, so its set is not
    reproducible even between its CPU and GPU.
    """
    B, P, D = centroids.shape
    device = centroids.device
    lengths = lengths.to(device=device, dtype=torch.int64)
    query_radius = 2.0 * float(radius) * float(overlap_factor)
    if pitch is None:
        idx = ball_query(
            centroids, centroids, K=K, radius=query_radius, lengths1=lengths, lengths2=lengths
        )
    else:
        idx = grid_ball_query(
            centroids,
            centroids,
            K=K,
            radius=query_radius,
            pitch=pitch,
            lengths1=lengths,
            lengths2=lengths,
        )
    counts = idx.ge(0).sum(-1)  # (B, P)
    # Columns past the fullest neighbour list hold only -1. On a pixel lattice a radius-5 ball
    # has at most 81 sites, so with `K=256` the two scatter passes of every round would
    # otherwise run over three times the entries that carry anything.
    idx = idx[..., : max(int(counts.max()), 1)]
    K = idx.shape[-1]
    _, order = counts.sort(dim=-1, descending=True, stable=True)
    rank = torch.empty_like(order)
    rank.scatter_(1, order, torch.arange(P, device=device).expand(B, P))
    rank = rank.to(torch.int32)

    nb_valid = idx.ge(0)
    idx0 = idx.clamp(min=0).reshape(B, P * K)
    unresolved = torch.arange(P, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    retain = torch.zeros(B, P, dtype=torch.bool, device=device)
    none = torch.tensor(P + 1, dtype=torch.int32, device=device)
    while bool(unresolved.any()):
        r = torch.where(unresolved, rank, none)
        edge = torch.where(nb_valid, r.unsqueeze(-1).expand(B, P, K), none)
        best = torch.full((B, P), int(none), dtype=torch.int32, device=device)
        best.scatter_reduce_(1, idx0, edge.reshape(B, P * K), reduce="amin")
        now = unresolved & (best >= rank)
        retain |= now
        # A retained centre suppresses the members of its ball that come after it in the order.
        # The ones before it were decided at their own turn, and with a truncated neighbour
        # list a centre can sit in a later centre's ball without the reverse holding.
        r_now = torch.where(now, rank, none)
        edge = torch.where(nb_valid, r_now.unsqueeze(-1).expand(B, P, K), none)
        hit = torch.full((B, P), int(none), dtype=torch.int32, device=device)
        hit.scatter_reduce_(1, idx0, edge.reshape(B, P * K), reduce="amin")
        unresolved &= ~(now | (hit < rank))

    reorder = torch.argsort((~retain).to(torch.int8), dim=1, stable=True)
    centroids = centroids.gather(1, reorder.unsqueeze(-1).expand(-1, -1, D))
    return centroids, retain.sum(dim=1)


def chamfer_distance(x: Tensor, y: Tensor, x_lengths: Tensor, y_lengths: Tensor) -> Tensor:
    """Bidirectional Chamfer distance between padded point sets, `(N, P1, D)` and `(N, P2, D)`.

    Each direction is the mean over valid points of the squared distance to the nearest valid
    point on the other side; the two are summed and averaged over the batch, which is
    pytorch3d's default reduction. A set with no valid points contributes zero.
    """

    def one_way(a, b, la, lb):
        N, P, _ = a.shape
        d, _ = knn_points(a, b, lengths1=la, lengths2=lb, K=1)
        d = d[..., 0]
        valid = torch.arange(P, device=a.device).unsqueeze(0) < la.unsqueeze(1)
        d = torch.where(valid & torch.isfinite(d), d, torch.zeros_like(d))
        return d.sum(1) / la.clamp(min=1)

    per_event = one_way(x, y, x_lengths, y_lengths) + one_way(y, x, y_lengths, x_lengths)
    return per_event.sum() / max(int(x.shape[0]), 1)
