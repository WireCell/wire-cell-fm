"""Marker handling: a test skips, with a reason, when what it needs is absent.

``gpu`` and ``distributed`` want devices; ``needs_data`` wants the production mounted, which is
a different thing -- ``/gpfs01`` is reachable from the login node and the sparse readers are
pure IO, so those run happily off a GPU node and only skip when the mount is not there.

The default run is ``pytest -m "not gpu and not distributed"``. The GPU suites have a named
trigger, ``wcfm test --gpu``, and are required before a stage lands."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _cuda_count() -> int:
    try:
        import torch

        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    except Exception:  # torch absent on a CPU runner is a legitimate state
        return 0


PRODUCTION_ROOT = Path("/gpfs01/lbne/users/fm/cffm-data")


def _world_size() -> int:
    """Ranks this process was launched with. Set by ``torchrun``, absent otherwise."""
    try:
        return int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        return 1


def pytest_collection_modifyitems(config, items):
    n = _cuda_count()
    world = _world_size()
    has_data = PRODUCTION_ROOT.is_dir()
    skip_gpu = pytest.mark.skip(reason="needs a CUDA device")
    skip_dist = pytest.mark.skip(
        reason=f"needs 2+ ranks; WORLD_SIZE={world}. "
        "Run: torchrun --nproc_per_node=2 -m pytest -m distributed"
    )
    skip_data = pytest.mark.skip(reason=f"needs the production under {PRODUCTION_ROOT}")
    for item in items:
        # `distributed` is about RANKS, not devices. DDP's gradient reduction and the arming
        # of its reducer behave identically over gloo on a CPU -- measured 2026-09-08, with a
        # control: computing through the unwrapped module diverges by 3.9e-02 on two CPU ranks
        # exactly as it did on two L40S. So these skip when the process was not launched with
        # two ranks, whatever hardware is present, and the suite stops being hostage to a
        # scarce 2-GPU slot.
        if "distributed" in item.keywords and world < 2:
            item.add_marker(skip_dist)
        elif "gpu" in item.keywords and n < 1:
            item.add_marker(skip_gpu)
        if "needs_data" in item.keywords and not has_data:
            item.add_marker(skip_data)


# ---------------------------------------------------------------------------
# Running the `distributed` suite means `torchrun --nproc_per_node=2 -m pytest`,
# so every rank is a separate pytest process with its own fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def shared_tmp(tmp_path):
    """A directory **every rank sees**, unlike ``tmp_path``.

    pytest derives ``basetemp`` per process, so under ``torchrun`` each rank gets a different
    ``tmp_path``. A distributed checkpoint test on ``tmp_path`` therefore has rank 0 writing
    ``latest.pt`` somewhere no other rank can read -- and it does not error: every other rank
    finds no checkpoint, takes the fresh-run path, and the test passes for the wrong reason or
    diverges into a hang on the next collective.

    ``_CONDOR_SCRATCH_DIR`` is per *job*, not per rank, so it is the shared root on the
    cluster. **Off the cluster it must not fall back to ``tmp_path``**: that was the first
    version of this fixture and it was wrong under a local ``torchrun``, where the ranks are
    still separate processes with separate ``basetemp``s. The symptom was exactly the one the
    fixture exists to prevent -- ``test_only_rank_zero_writes_and_both_ranks_resume_from_it``
    passed on the cluster (where ``_CONDOR_SCRATCH_DIR`` is set) and failed locally, taking
    the other rank into a hang with it.

    So when the process was launched with two or more ranks, the root is derived from
    something every rank already agrees on. ``TORCHELASTIC_RUN_ID`` and ``MASTER_PORT`` are
    both set by ``torchrun`` and identical across its children; a single-rank run keeps
    ``tmp_path``, which is correct there because there is nothing to share with.
    """
    if scratch := os.environ.get("_CONDOR_SCRATCH_DIR"):
        root = Path(scratch) / "shared_tmp"
    elif _world_size() > 1:
        shared_id = (
            os.environ.get("TORCHELASTIC_RUN_ID")
            or os.environ.get("MASTER_PORT")
            or "local"
        )
        root = Path(os.environ.get("TMPDIR", "/tmp")) / f"wcfm-dist-{shared_id}"
    else:
        return tmp_path

    # The test's own name, so two tests in one run do not share a directory.
    current = os.environ.get("PYTEST_CURRENT_TEST", "t")
    name = current.split("::")[-1].split(" ")[0].replace("/", "_").replace("[", "_").rstrip("]")
    shared = root / name
    shared.mkdir(parents=True, exist_ok=True)
    return shared


@pytest.fixture
def two_rank_fabric():
    """A 2-rank ``Fabric``, with the world size **asserted** rather than assumed.

    This is spike (c)'s control pattern. A `distributed` test that quietly runs at
    ``world_size == 1`` asserts that a collective produced agreement while no collective ran
    at all -- indistinguishable from passing correctly, and the worst outcome available, since
    reporting the distributed suite green is exactly the false confidence the README's "what
    the suite does not mean" section exists to prevent. So: fail loudly.
    """
    from lightning_fabric import Fabric
    from lightning_fabric.strategies import DDPStrategy

    # CUDA when the job has it, gloo on a CPU otherwise. Not a fallback for convenience: DDP's
    # reduction and its per-forward arming are backend-independent, so the CPU path tests the
    # same thing and turns a scarce 2-GPU slot into confirmation rather than the only signal.
    accelerator = "cuda" if _cuda_count() >= 2 else "cpu"
    fabric = Fabric(
        accelerator=accelerator,
        devices=2,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision="32-true",
    )
    fabric.launch()  # a verified no-op under torchrun; spike (c)
    assert fabric.world_size == 2, (
        f"world_size={fabric.world_size}, expected 2. This suite is meaningless on one rank: "
        "every assertion here is about ranks agreeing, and one rank always agrees with itself. "
        "Launch it with `torchrun --nproc_per_node=2 -m pytest -m distributed`."
    )
    return fabric


@pytest.fixture
def two_rank_accelerator():
    """Which backend the distributed suite is running over, for tests that report it."""
    return "cuda" if _cuda_count() >= 2 else "cpu"
