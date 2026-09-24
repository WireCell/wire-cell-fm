"""The PoLAr-MAE port at production size on a device, under the precision a run uses.

The CPU suite covers the architecture on a few thousand parameters. What only a device
reaches is the full-size model under `bf16-mixed` autocast: the tokenizer's forced fp32
region, the fused attention kernel with the additive mask, and one step of `PointMaeModule`
with both terms. ``wcfm test --gpu`` runs this suite.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("torch")
pytest.importorskip("warpconvnet")

import torch  # noqa: E402

from wcfm.data.voxels import voxels_from  # noqa: E402
from wcfm.model.augment.transforms import FeatureLogTransform  # noqa: E402
from wcfm.model.backbones import PolarMAEBackbone  # noqa: E402
from wcfm.model.modules import PointMaeModule  # noqa: E402
from wcfm.model.modules.ssl import _inference_context  # noqa: E402
from wcfm.model.terms import ChamferTerm, EnergyTerm  # noqa: E402

from .fake_backbone import make_batch  # noqa: E402

pytestmark = pytest.mark.gpu

SHARD = "/gpfs01/lbne/users/fm/cffm-data/shards_fhdh_sparse_200k_mixed_apa0W/shard_00000.h5"


@pytest.fixture
def device():
    return torch.device("cuda")


def dense_batch(n_pixels: int, device, seed: int = 0):
    """One event of `n_pixels` distinct pixels on the production canvas, positive charge."""
    g = torch.Generator().manual_seed(seed)
    ch = torch.randint(0, 960, (n_pixels * 2,), generator=g)
    tk = torch.randint(0, 1425, (n_pixels * 2,), generator=g)
    coords = torch.unique(torch.stack([ch, tk], 1), dim=0)[:n_pixels].to(torch.int32)
    q = torch.rand(coords.shape[0], 1, generator=g) * 500.0 + 1.0
    return voxels_from(
        coords.to(device), q.to(device), torch.tensor([0, coords.shape[0]], dtype=torch.int64)
    )


def test_the_full_size_backbone_runs_under_bf16_autocast(device):
    torch.manual_seed(0)
    bb = PolarMAEBackbone().to(device).eval()
    xs = make_batch((3000, 800, 12000), width=960, height=1425, seed=1).voxels.to(device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        fb = bb(xs, None, ("local",))
    assert fb.out.feature_tensor.shape == (xs.coordinate_tensor.shape[0], 384)
    assert torch.isfinite(fb.out.feature_tensor).all()
    assert torch.equal(fb.out.coordinate_tensor, xs.coordinate_tensor)


def test_one_step_of_pointmae_under_bf16_autocast(device):
    torch.manual_seed(0)
    m = PointMaeModule(
        backbone=PolarMAEBackbone(),
        terms={"chamfer": ChamferTerm(), "energy": EnergyTerm()},
        normalize=FeatureLogTransform(3.75, 83861.2),
    ).to(device)
    batch = make_batch((4000, 2500), width=960, height=1425, seed=2).to(device)
    ctx = _inference_context(m, 0, device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = m.training_step(batch, ctx)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert all(p.grad is not None for p in m.parameters())
    assert torch.isfinite(m.backbone.mask_token.grad).all()


def test_grouping_cost_on_a_dense_event(device, capsys):
    """Not an assertion on speed: the number a submit decision needs, printed."""
    bb = PolarMAEBackbone(context_length=4096).to(device).eval()
    xs = dense_batch(28_000, device)
    points, lengths, _ = bb.points_from(xs)
    torch.cuda.synchronize()
    t = time.perf_counter()
    with torch.no_grad():
        g = bb.grouping(points, lengths)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t
    with capsys.disabled():
        print(
            f"\n[polarmae] grouping of a {int(lengths[0])}-pixel event: {dt:.3f} s, "
            f"{int(g.emb_mask.sum())} tokens, "
            f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak"
        )
    assert g.emb_mask.sum() > 0


def production_batch(n_events: int, device, first: int = 0):
    """`n_events` consecutive events of a production shard as one `Batch`, raw ADC."""
    h5py = pytest.importorskip("h5py")
    with h5py.File(SHARD) as f:
        coords = torch.from_numpy(f["coords"][()]).to(torch.int32)
        feats = torch.from_numpy(f["features"][()]).to(torch.float32)
        off = torch.from_numpy(f["offsets"][()]).to(torch.int64)
    s, e = int(off[first]), int(off[first + n_events])
    from wcfm.data.voxels import Batch

    offsets = off[first : first + n_events + 1] - s
    return Batch(voxels_from(coords[s:e].to(device), feats[s:e].to(device), offsets), {})


def _timed(fn, reps: int = 3) -> float:
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


@pytest.mark.needs_data
def test_step_cost_on_production_events(device, capsys):
    """Not an assertion on speed: the per-rank step on 16 real events, printed. The grouping
    is timed with the dense and the grid query, then one training step with backward under
    bf16 autocast, which is what a rank of `polarmae_pm4w_ddp6` does."""
    torch.manual_seed(0)
    batch = production_batch(16, device)
    m = PointMaeModule(
        backbone=PolarMAEBackbone(context_length=2048),
        terms={"chamfer": ChamferTerm(), "energy": EnergyTerm()},
        normalize=FeatureLogTransform(3.75, 83861.2),
    ).to(device)
    bb = m.backbone
    points, lengths, _ = bb.points_from(batch.voxels)
    grouping = bb.grouping
    pitch = grouping.pitch
    lines = [f"16 events, {int(lengths.sum())} pixels, longest {int(lengths.max())}"]
    with torch.no_grad():
        for name, value in (("dense", None), ("grid", pitch)):
            grouping.pitch = value
            torch.cuda.reset_peak_memory_stats()
            dt = _timed(lambda: grouping(points, lengths))
            g = grouping(points, lengths)
            lines.append(
                f"grouping {name:5s}: {dt:.3f} s, {int(g.emb_mask.sum())} tokens, "
                f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak"
            )
    grouping.pitch = pitch
    from wcfm.model.backbones.polarmae.ops import cnms, grid_ball_query

    xyz = points[..., :3]
    r = grouping.group_radius
    with torch.no_grad():
        t_c = _timed(
            lambda: cnms(xyz, radius=r, overlap_factor=0.5, K=256, lengths=lengths, pitch=pitch)
        )
        centres, n = cnms(xyz, radius=r, overlap_factor=0.5, K=256, lengths=lengths, pitch=pitch)
        centres = centres[:, : int(n.max())]
        t_m = _timed(
            lambda: grid_ball_query(
                centres, xyz, K=256, radius=r, pitch=pitch, lengths1=n, lengths2=lengths
            )
        )
        idx = grid_ball_query(
            centres, xyz, K=256, radius=r, pitch=pitch, lengths1=n, lengths2=lengths
        )
        idx = idx[:, :, : max(int(idx.ge(0).sum(2).max()), 32)]
        t_f = _timed(lambda: grouping._reduce(points, idx))
    lines.append(f"  of which cnms {t_c:.3f} s, member query {t_m:.3f} s, fps reduce {t_f:.3f} s")
    ctx = _inference_context(m, 0, device)

    def step():
        m.zero_grad(set_to_none=True)
        b = production_batch(16, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m.training_step(b, ctx)
        out.loss.backward()

    torch.cuda.reset_peak_memory_stats()
    dt = _timed(step)
    lines.append(
        f"training step + backward: {dt:.3f} s, "
        f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak"
    )
    with capsys.disabled():
        print("\n[polarmae] " + "\n[polarmae] ".join(lines))
