"""One CUDA device: mixed precision, and the loop on a real device.

Run by ``wcfm test --gpu``. Two things live here that a CPU cannot reach.

**`bf16-mixed`.** ``run.precision`` defaults to ``32-true`` (the old repo is fp32 throughout),
but bf16 is supported and no CPU test can exercise autocast, so it is verified here. The plan
says to port ``tests/test_amp_support.py`` from the old repo; that file measures a real forward and
backward of the ``attn_mae`` backbone, which does not exist here until Stage 3, so what is
ported is its *measurement* -- warpconvnet takes its compute dtype from the autocast context
(``nn/functional/sparse_conv/helper.py:251``), and "should" is not "does". It asserts only what
must hold for AMP to be usable at all: the autocast dtype actually reaches the output, outputs
and gradients are finite, and master weights stay fp32. Speed and drift are reported, not
asserted -- they are the input to a decision, not a pass or a fail.

**The loop on a device.** ``Batch.to(device)`` is explicit because Fabric is told
``move_to_device=False`` (spike (b)), and an explicit move is the kind of thing that works on
CPU by accident.
"""

from __future__ import annotations

import time

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    return torch.device("cuda")


# ------------------------------------------------------------------ mixed precision


def _voxels(device, n_per_image: int = 512, batch: int = 8, seed: int = 0):
    """A batch shaped like a crop, built the way ``data/voxels.py`` builds one."""
    from wcfm.data.voxels import offsets_from_counts, voxels_from

    g = torch.Generator().manual_seed(seed)
    coords, feats = [], []
    for _ in range(batch):
        xy = (torch.randn(n_per_image, 2, generator=g) * 50 + 500).round().int()
        coords.append(xy)
        feats.append(torch.rand(n_per_image, 1, generator=g))
    offsets = offsets_from_counts([c.shape[0] for c in coords])
    return voxels_from(
        torch.cat(coords).to(device), torch.cat(feats).to(device), offsets
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_a_sparse_convolution_takes_its_compute_dtype_from_autocast(device, dtype):
    """The claim the old ``test_amp_support.py`` exists to check, kept alive without the
    backbone: warpconvnet reads the autocast context rather than the input dtype
    (``nn/functional/sparse_conv/helper.py:251``, "use the autocast dtype rather than the
    tensor's storage dtype"). If a torch or warpconvnet bump broke this, every AMP run would
    silently compute in fp32 -- correct, and with none of the speed.

    **``bias=False`` is load-bearing here.** With a bias, the convolution's bf16 output is
    added to an fp32 bias, and ``torch.add`` runs in the widest input dtype -- so the *output*
    comes back fp32 while the *compute* was bf16. Cluster 2250 failed on exactly that: the
    first version of this test asserted on the output of a conv with a bias and read the
    promotion as a broken stack. The bias-carrying case is asserted below, as fp32, so the
    distinction is recorded rather than rediscovered.
    """
    from warpconvnet.nn.modules.sparse_conv import SparseConv2d

    torch.manual_seed(0)
    conv = (
        SparseConv2d(in_channels=1, out_channels=16, kernel_size=3, bias=False)
        .to(device)
        .train()
    )
    xs = _voxels(device)

    with torch.autocast("cuda", dtype=dtype):
        out = conv(xs)
    features = out.batched_features.batched_tensor

    assert features.dtype == dtype, (
        f"autocast({dtype}) produced {features.dtype} with bias=False: the compute dtype did "
        "not follow the context, so mixed precision is inert"
    )
    assert torch.isfinite(features.float()).all()


def test_a_bias_promotes_the_output_back_to_fp32_and_that_is_not_a_failure(device):
    """The other half of the finding above, pinned so nobody reads it as a regression: the
    output dtype of a conv *with* a bias is fp32 under autocast, because the fp32 bias wins
    the type promotion on the add. The compute was still bf16."""
    from warpconvnet.nn.modules.sparse_conv import SparseConv2d

    torch.manual_seed(0)
    conv = SparseConv2d(in_channels=1, out_channels=16, kernel_size=3, bias=True)
    conv = conv.to(device).train()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = conv(_voxels(device))
    assert out.batched_features.batched_tensor.dtype is torch.float32


def test_amp_leaves_master_weights_and_gradients_in_fp32(device):
    """``bf16-mixed`` is mixed: the compute is bf16 and the parameters are not. A parameter
    that came back bf16 would lose the small-update accumulation the fp32 master copy is for,
    which shows up as a run that plateaus rather than as an error."""
    from warpconvnet.nn.modules.sparse_conv import SparseConv2d

    torch.manual_seed(0)
    conv = SparseConv2d(in_channels=1, out_channels=16, kernel_size=3).to(device).train()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = conv(_voxels(device))
        loss = out.batched_features.batched_tensor.float().pow(2).mean()
    loss.backward()

    for name, param in conv.named_parameters():
        assert param.dtype is torch.float32, f"{name} is {param.dtype}, not fp32"
        if param.grad is not None:
            assert param.grad.dtype is torch.float32, f"{name}.grad is {param.grad.dtype}"
            assert torch.isfinite(param.grad).all(), f"{name}.grad is not finite under AMP"


def test_amp_output_drift_and_timing_are_reported_not_asserted(device):
    """The numbers that were the input to choosing ``bf16-mixed``. Reported so a regression is
    visible in the job log, not asserted, because a threshold on either would be a number
    nobody can justify -- and because a *drift* that grew would be interesting long before it
    was wrong."""
    from warpconvnet.nn.modules.sparse_conv import SparseConv2d

    torch.manual_seed(0)
    conv = SparseConv2d(in_channels=1, out_channels=32, kernel_size=3).to(device).train()
    xs = _voxels(device, n_per_image=2000, batch=16)

    results = {}
    for label, dtype in (("fp32", None), ("bf16", torch.bfloat16), ("fp16", torch.float16)):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        ctx = (
            torch.autocast("cuda", dtype=dtype)
            if dtype is not None
            else torch.autocast("cuda", enabled=False)
        )
        with ctx:
            out = conv(xs)
            features = out.batched_features.batched_tensor
        torch.cuda.synchronize()
        results[label] = {
            "s": time.perf_counter() - started,
            "gb": torch.cuda.max_memory_allocated() / 1024**3,
            "features": features.detach().float(),
        }

    reference = results["fp32"]["features"]
    lines = []
    for label, r in results.items():
        drift = float((r["features"] - reference).abs().max())
        lines.append(f"  {label:5s} {r['s'] * 1e3:7.2f} ms  {r['gb']:5.2f} GB  drift {drift:.3e}")
        assert torch.isfinite(r["features"]).all(), f"{label} produced non-finite features"
    # Reaches the job log because stage 1 runs with `-s`: pytest captures and discards stdout
    # from a passing test, so without it "reported, not asserted" reports to nobody.
    print("\n[amp] forward at production-ish shape:\n" + "\n".join(lines))


# ------------------------------------------------------------------ the loop, on a device


def test_the_whole_loop_runs_on_cuda_under_autocast(shared_tmp):
    """End to end at ``bf16-mixed`` -- not the default any more, but the one precision a CPU
    test cannot reach -- so the guards, the schedules and the checkpoint are exercised under
    autocast rather than only in fp32."""
    from lightning_fabric import Fabric

    from wcfm.engine.trainer import Trainer

    from .test_engine import cfg
    from .toy import ToyModule, toy_loader

    seen: set[tuple[bool, torch.dtype]] = set()

    class Recording(ToyModule):
        def training_step(self, batch, ctx):
            out = super().training_step(batch, ctx)
            seen.add(self._autocast)
            return out

    module = Recording()
    before = module.net.weight.detach().clone()
    trainer = Trainer(
        cfg(shared_tmp, optim__epochs=2, run__save_every=1, run__precision="bf16-mixed"),
        module,
        fabric=Fabric(accelerator="cuda", devices=1, precision="bf16-mixed"),
        loader=toy_loader(steps=4),
        run_dir=shared_tmp / "gpu_run",
    )
    result = trainer.fit()

    assert result["steps"] == 8
    assert result["skipped_nonfinite"] == 0, "bf16 autocast produced a non-finite gradient"
    assert seen == {(True, torch.bfloat16)}, (
        f"the forward saw autocast {seen}, expected (True, bfloat16). `bf16-mixed` is an "
        "autocast inside the wrapper's forward, so a module computing through `self` gets no "
        "autocast at all whatever the config says -- which is how this went unnoticed "
        "alongside the DDP bug. Note the OUTPUT dtype cannot answer this: "
        "`_FabricModule.forward` calls `precision.convert_output`, which casts the result "
        "back to fp32 by design (wrappers.py:138)."
    )
    assert module.net.weight.device.type == "cuda"
    assert module.net.weight.dtype is torch.float32, "master weights stay fp32 under -mixed"
    assert not torch.equal(module.net.weight.detach().cpu(), before.cpu())
    assert (shared_tmp / "gpu_run" / "checkpoints" / "checkpoint_epoch2.pt").exists()


def test_the_batch_is_moved_explicitly_rather_than_by_fabric(shared_tmp):
    """``setup_dataloaders(move_to_device=False)`` is spike (b)'s answer, so the move is
    ``Batch.to`` and nothing else. On a CPU run both spellings look identical."""
    from lightning_fabric import Fabric

    from wcfm.engine.trainer import Trainer

    from .test_engine import cfg
    from .toy import ToyModule, toy_loader

    seen: list[str] = []

    class Recording(ToyModule):
        def training_step(self, batch, ctx):
            seen.append(self.rows_of(batch).device.type)
            return super().training_step(batch, ctx)

    Trainer(
        cfg(shared_tmp, optim__epochs=1),
        Recording(),
        fabric=Fabric(accelerator="cuda", devices=1, precision="32-true"),
        loader=toy_loader(steps=2),
        run_dir=shared_tmp / "move_run",
    ).fit()

    assert seen and set(seen) == {"cuda"}, f"batches arrived on {set(seen)}, not cuda"


def test_peak_memory_is_read_and_never_reset_by_the_collector(shared_tmp):
    """``debug.py:454`` called ``reset_peak_memory_stats()`` un-gated every step, which
    redefined "peak" as "peak since the last step" -- so the number meant to catch a run
    approaching the card's limit reported a typical step instead."""
    from wcfm.metrics.base import StepRecord
    from wcfm.metrics.collectors import Throughput

    torch.cuda.reset_peak_memory_stats()
    big = torch.zeros(64 * 1024 * 1024 // 4, device="cuda")  # 64 MB
    peak_before = torch.cuda.max_memory_allocated()
    del big

    out = Throughput(cadence="step").compute(
        StepRecord(step=0, epoch=1, scalars={"n_samples": 4}, timing={"step": 0.1})
    )
    assert out["peak_mem_gb"] * 1024**3 == pytest.approx(peak_before, rel=1e-6)
    assert torch.cuda.max_memory_allocated() == peak_before, "the collector reset the peak"
