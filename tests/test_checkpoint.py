"""Checkpoint v2, and the resume rules.

The test that matters most is the buffer round trip. The old repo created the DINO centring
buffer lazily (``loss.py:73,121-126``) and never saved it, so a resumed run silently restarted
centring -- the loss recovers and nothing in the metrics says why. A resume test that only
compared parameters would pass against that bug, so this one compares a buffer.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from wcfm.engine.checkpoint import (  # noqa: E402
    SCHEMA_VERSION,
    Checkpoint,
    load_checkpoint,
    resolve_resume,
    rng_state,
    save_checkpoint,
    set_rng_state,
    should_save,
)

from .toy import ToyModule  # noqa: E402

pytestmark = pytest.mark.stack


def _ckpt(module: ToyModule, epoch: int = 3, step: int = 30) -> Checkpoint:
    return Checkpoint(
        epoch=epoch,
        step=step,
        cfg={"optim": {"epochs": 10}},
        model=module.state_dict(),
        optimizer={},
        rng={"rank0": rng_state(0)},
        meta={"world_size": 1},
    )


def test_round_trip_restores_buffers_not_only_parameters(tmp_path):
    module = ToyModule()
    with torch.no_grad():
        module.centre += 7.0
    path = save_checkpoint(tmp_path / "checkpoint_epoch3.pt", _ckpt(module))

    fresh = ToyModule()
    assert float(fresh.centre.sum()) == 0.0
    fresh.load_state_dict(load_checkpoint(path).model)
    assert torch.equal(fresh.centre, module.centre), (
        "the centring buffer must survive a checkpoint; this is the bug the v2 schema exists "
        "to fix, and only a buffer comparison catches it"
    )


def test_the_write_is_atomic(tmp_path):
    """No temporary file survives, and a reader polling the directory never sees a partial
    one. The preemption handler writes exactly when a kill is already in progress."""
    path = save_checkpoint(tmp_path / "latest.pt", _ckpt(ToyModule()))
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp*")), "the temporary write path was left behind"


def test_a_wrong_schema_version_is_refused_by_name(tmp_path):
    """A legacy ``torch.save`` from the old repo is a v0 file. This framework does not read
    one, and the error has to say so rather than fail on a missing key."""
    path = tmp_path / "legacy.pt"
    torch.save({"epoch": 5, "model": {}}, path)
    with pytest.raises(ValueError, match="ml-dune-model"):
        load_checkpoint(path)


def test_rng_state_round_trips_and_reproduces_the_stream():
    """The point of saving RNG state: the numbers after a resume are the numbers that would
    have come next, not a fresh stream that merely looks random."""
    torch.manual_seed(1234)
    state = rng_state(0)
    expected = torch.randn(5)

    torch.manual_seed(9999)  # move the stream somewhere else entirely
    assert set_rng_state(state) is True
    assert torch.equal(torch.randn(5), expected)


def test_missing_rng_state_is_a_degraded_resume_not_a_broken_one():
    """A checkpoint written before this existed, or by a different rank count, leaves the
    streams alone and reports that it did -- the caller records which happened."""
    assert set_rng_state(None) is False
    assert set_rng_state({}) is False


def test_rng_state_is_keyed_per_rank():
    """A single saved state restored on every rank would silently correlate the ranks'
    augmentation streams after the first resume."""
    assert rng_state(3)["rank"] == 3


# ------------------------------------------------------------------ resume


def test_auto_prefers_latest_then_falls_back_to_the_highest_epoch(tmp_path):
    """An eviction between the SIGTERM handler and its write must still resume from the last
    periodic checkpoint rather than starting a forty-hour run over."""
    assert resolve_resume("auto", tmp_path) is None

    (tmp_path / "checkpoint_epoch9.pt").touch()
    (tmp_path / "checkpoint_epoch10.pt").touch()
    (tmp_path / "checkpoint_epoch2.pt").touch()
    assert resolve_resume("auto", tmp_path).name == "checkpoint_epoch10.pt", (
        "sorted lexically, epoch9 would beat epoch10"
    )

    (tmp_path / "latest.pt").touch()
    assert resolve_resume("auto", tmp_path).name == "latest.pt"


def test_none_refuses_to_resume_even_with_a_checkpoint_present(tmp_path):
    """Which makes re-running a finished run from scratch a config change, not an ``rm``."""
    (tmp_path / "latest.pt").touch()
    assert resolve_resume("none", tmp_path) is None


def test_an_explicit_path_that_does_not_exist_raises(tmp_path):
    """Silently starting from epoch 0 because a path had a typo in it is the failure this
    avoids -- it costs a whole run to notice."""
    with pytest.raises(FileNotFoundError):
        resolve_resume(str(tmp_path / "nope.pt"), tmp_path)


# ------------------------------------------------------------------ cadence


def test_save_at_fires_alongside_save_every():
    """Cadence is a list as well as an interval, so every point of a sweep is checkpointed --
    and therefore probed -- at the same epochs regardless of its own ``epochs``."""
    assert should_save(10, 100, save_every=10, save_at=[])
    assert not should_save(11, 100, save_every=10, save_at=[])
    assert should_save(11, 100, save_every=10, save_at=[11, 50])
    assert should_save(50, 100, save_every=0, save_at=[50])
    assert not should_save(51, 100, save_every=0, save_at=[50])


def test_the_final_epoch_always_saves():
    assert should_save(37, 37, save_every=10, save_at=[])


def test_the_schema_version_is_pinned():
    """A bump has to be a deliberate edit here, not a side effect of adding a field."""
    assert SCHEMA_VERSION == 2
