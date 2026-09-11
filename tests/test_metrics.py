"""The metrics stream: absent-never-null, the resume truncation, and cadence.

The writer is deliberately torch-free so these run in the config-only environment. The
protocol half (``StepRecord``, ``fires``) needs ``torch`` for its ``Tensor`` annotation, so
those tests import it inside the body and are marked ``stack``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from wcfm.metrics.writer import MetricsWriter, _jsonable, read_stream


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_a_key_that_was_not_measured_is_absent_rather_than_null(tmp_path):
    """The old writer used ``0.0`` for "disabled" and ``None`` for "not measured" in the same
    columns (``model.py:656-657`` with ``loss.py:210,233``), so a legitimately measured zero
    and a switched-off metric were indistinguishable. Here zero is written and ``None`` is
    dropped, so a reader can tell the two apart."""
    with MetricsWriter(tmp_path) as writer:
        writer.write("step", {"step": 0, "loss": 0.0, "never_measured": None})

    row = _read(tmp_path / "step.jsonl")[0]
    assert row["loss"] == 0.0, "a measured zero is a value, not a sentinel"
    assert "never_measured" not in row


def test_nonfinite_values_are_dropped_so_the_stream_stays_parseable(tmp_path):
    """``json.dumps`` writes NaN as a bare ``NaN``, which is valid Python and invalid JSON --
    one diverged step would make the whole file unreadable by every other tool. The engine's
    skip counter reports the divergence instead, which is the number you actually want."""
    with MetricsWriter(tmp_path) as writer:
        writer.write("step", {"step": 0, "loss": float("nan"), "grad": float("inf"), "lr": 1e-4})

    text = (tmp_path / "step.jsonl").read_text()
    assert "NaN" not in text and "Infinity" not in text
    row = json.loads(text)
    assert row == {"step": 0, "lr": 1e-4}


def test_schema_json_records_the_names_actually_written(tmp_path):
    """Which is what lets a reader distinguish "never measured in this run" from "measured,
    and it was zero"."""
    with MetricsWriter(tmp_path) as writer:
        writer.write("step", {"step": 0, "lr": 1e-4})
        writer.write("epoch", {"epoch": 1, "loss": 0.3})

    schema = json.loads((tmp_path / "schema.json").read_text())
    assert schema["streams"]["step"] == ["lr", "step"]
    assert schema["streams"]["epoch"] == ["epoch", "loss"]


def test_a_stream_nothing_writes_to_has_no_file(tmp_path):
    """A run killed inside its first epoch writes no epoch record.

    An empty file is worse than an absent one, because a reader polling a run directory cannot
    tell "this run recorded no epochs" from "it has not finished its first one".
    """
    with MetricsWriter(tmp_path) as writer:
        writer.write("step", {"step": 0, "lr": 1e-4})

    assert (tmp_path / "step.jsonl").exists()
    assert not (tmp_path / "epoch.jsonl").exists(), "an unwritten stream left an empty file"
    # And it appears the moment something is written to it.
    with MetricsWriter(tmp_path) as writer:
        writer.write("epoch", {"epoch": 1, "loss": 0.3})
    assert (tmp_path / "epoch.jsonl").exists()


def test_an_oversized_record_is_counted_and_still_written_whole(tmp_path):
    """Truncating would drop whichever keys happened to sort last. The count is what tells you
    the budget is wrong, and it lands in schema.json rather than in a log nobody reads."""
    with MetricsWriter(tmp_path, max_record_bytes=64) as writer:
        writer.write("step", {"step": 0, **{f"k{i}": float(i) for i in range(40)}})

    assert json.loads((tmp_path / "schema.json").read_text())["records_over_budget"] == 1
    assert len(_read(tmp_path / "step.jsonl")[0]) == 41, "written whole, not truncated"


def test_every_record_is_flushed_so_a_kill_costs_at_most_the_last_line(tmp_path):
    writer = MetricsWriter(tmp_path)
    writer.write("step", {"step": 0, "lr": 1.0})
    assert _read(tmp_path / "step.jsonl") == [{"step": 0, "lr": 1.0}], "not flushed"
    writer.close()


def test_resume_drops_records_at_or_beyond_the_resume_step(tmp_path):
    """Without this a resumed run has two records for the same step and every plot silently
    averages the pre-eviction and post-eviction values."""
    with MetricsWriter(tmp_path) as writer:
        for step in range(6):
            writer.write("step", {"step": step, "lr": 1.0})

    with MetricsWriter(tmp_path, resume_step=4) as writer:
        writer.write("step", {"step": 4, "lr": 0.5})

    steps = [r["step"] for r in _read(tmp_path / "step.jsonl")]
    assert steps == [0, 1, 2, 3, 4]
    assert steps == sorted(set(steps)), "a step appears twice"


def test_resume_discards_the_partial_last_line_of_a_killed_run(tmp_path):
    """A SIGKILL mid-write leaves half a line. Reopening must not choke on it."""
    (tmp_path / "step.jsonl").write_text('{"step":0,"lr":1.0}\n{"step":1,"lr":0.9')
    with MetricsWriter(tmp_path, resume_step=1):
        pass
    assert _read(tmp_path / "step.jsonl") == [{"step": 0, "lr": 1.0}]


def test_a_disabled_writer_is_a_no_op_so_no_call_site_branches_on_rank(tmp_path):
    """Non-zero ranks get one of these. The trainer never writes ``if rank == 0`` at a call
    site -- that asymmetry is where this codebase's DDP bugs came from (ADR 0006)."""
    with MetricsWriter(tmp_path, enabled=False) as writer:
        writer.write("step", {"step": 0, "lr": 1.0})
    assert not list(tmp_path.iterdir())


def test_an_unknown_stream_is_refused(tmp_path):
    with MetricsWriter(tmp_path) as writer:
        with pytest.raises(ValueError, match="unknown metrics stream"):
            writer.write("steps", {"step": 0})


@pytest.mark.parametrize(
    "value,expected",
    [
        (np.float32(1.5), 1.5),
        (np.int64(3), 3),
        (np.array([1.0, 2.0]), [1.0, 2.0]),
        (np.arange(100.0), None),  # too large for the stream; ArrayDump's job, not this one
        (True, True),
        ({"a": np.float64(2.0)}, {"a": 2.0}),
    ],
)
def test_numpy_scalars_and_small_arrays_become_plain_json(value, expected):
    assert _jsonable(value) == expected


# ------------------------------------------------------------------ reading back


def test_a_partial_trailing_line_is_skipped(tmp_path):
    """A killed run leaves half a line, and a reader can be pointed at a stream the writer
    never reopened to truncate."""
    (tmp_path / "s.jsonl").write_text('{"step":0,"loss":1.0}\n{"step":1,"los')
    assert read_stream(tmp_path / "s.jsonl") == [{"step": 0, "loss": 1.0}]


def test_a_missing_stream_reads_as_empty(tmp_path):
    """A stream nothing wrote has no file, so its absence is a normal reading."""
    assert read_stream(tmp_path / "nope.jsonl") == []


# ------------------------------------------------------------------ the protocol


@pytest.mark.stack
def test_cadence_is_a_pure_function_of_the_step():
    """No state, so two collectors on the same cadence always fire on the same steps and their
    columns line up in the stream."""
    pytest.importorskip("torch")
    from wcfm.metrics.base import Collector, fires

    class Every(Collector):
        cadence = 5

        def compute(self, rec):
            return {}

    every, per_step, per_epoch = Every(), Every(), Every()
    per_step.cadence = "step"
    per_epoch.cadence = "epoch"

    assert [s for s in range(11) if fires(every, s)] == [0, 5, 10]
    assert all(fires(per_step, s) for s in range(5))
    assert not fires(per_epoch, 3)
    assert fires(per_epoch, 3, end_of_epoch=True)


@pytest.mark.stack
def test_a_collector_declares_what_it_needs_and_the_engine_skips_it_when_absent():
    """Rather than letting it raise mid-epoch, forty hours into a run."""
    pytest.importorskip("torch")
    import torch

    from wcfm.metrics.base import Collector, StepRecord

    class NeedsFeat(Collector):
        needs = frozenset({"student/dec_full", "params"})

        def compute(self, rec):
            return {}

    record = StepRecord(step=0, epoch=1, observables={"student/dec_full": torch.zeros(2)})
    assert NeedsFeat().unmet(record) == set(), "'params' is a literal, not an observable"

    assert NeedsFeat().unmet(StepRecord(step=0, epoch=1)) == {"student/dec_full"}
