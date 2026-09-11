"""`wcfm sweep`: Hydra enumerates, Condor launches.

The enumeration is Hydra's own (`OverridesParser` + `BasicSweeper.split_arguments`), so these
tests are not about the cross product being right -- they are about the three things this
command adds on top of it:

* a **content-addressed run name** per point, so two invocations of the same sweep do not
  produce two directories holding an identical run;
* a **manifest** that records which runs were one campaign;
* a **declared seed axis**, so `--group-by-seed` stops guessing replica families from run names.

That last one is the concrete payoff, and `test_declared_replicas_beat_the_name_guess` is the
test that shows it: the run names `wcfm sweep` produces are `<sweep_id>_<hash8>`, which the
regex fallback cannot group at all.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("hydra")

from wcfm.cli.sweep import (  # noqa: E402
    SEED_KEYS,
    build_manifest,
    enumerate_points,
    load_manifest,
    point_hash,
    write_manifest,
)


def test_hydra_sweep_syntax_is_the_real_thing():
    """`a,b`, `range(...)` and `choice(...)` all work, because the parser is Hydra's."""
    points, axes = enumerate_points(["model=dino,hybrid", "run.seed=range(1,3)", "optim.lr=0.1"])
    assert axes == ["model", "run.seed"]
    assert len(points) == 4
    # A plain key=value is a fixed override carried into every point, not an axis.
    assert all("optim.lr=0.1" in p for p in points)
    assert {tuple(sorted(p)) for p in points} == {
        tuple(sorted(["model=dino", "run.seed=1", "optim.lr=0.1"])),
        tuple(sorted(["model=dino", "run.seed=2", "optim.lr=0.1"])),
        tuple(sorted(["model=hybrid", "run.seed=1", "optim.lr=0.1"])),
        tuple(sorted(["model=hybrid", "run.seed=2", "optim.lr=0.1"])),
    }


def test_nothing_swept_is_one_point_and_no_axes():
    points, axes = enumerate_points(["model=hybrid"])
    assert points == [["model=hybrid"]] and axes == []


def test_the_run_name_is_content_addressed_and_order_independent():
    """Two invocations that list the same axes in a different order must name the same point the
    same thing, or the second submission is a duplicate run in a new directory."""
    assert point_hash(["a=1", "b=2"]) == point_hash(["b=2", "a=1"])
    assert point_hash(["a=1", "b=2"]) != point_hash(["a=1", "b=3"])

    m = build_manifest("scan", ["model=dino,hybrid"])
    for name, entry in m["points"].items():
        assert name == f"scan_{entry['hash']}"
        assert entry["hash"] == point_hash(entry["overrides"])


def test_the_manifest_records_the_axes_and_every_point():
    m = build_manifest("scan", ["model=dino,hybrid", "run.seed=range(1,3)"])
    assert m["sweep_id"] == "scan"
    assert m["n_points"] == 4 and len(m["points"]) == 4
    assert m["axes"] == ["model", "run.seed"]
    assert m["seed_axis"] == "run.seed"
    assert m["science_axes"] == ["model"]
    for entry in m["points"].values():
        assert set(entry["axis_values"]) == {"model", "run.seed"}


def test_seeds_are_recognised_as_a_replica_axis_not_a_science_one():
    m = build_manifest("scan", ["model=dino,hybrid", "run.seed=range(1,4)"])
    # Six points, but only two distinct configurations -- three replicas each.
    groups = {e["replica_group"] for e in m["points"].values()}
    assert m["n_points"] == 6 and len(groups) == 2

    by_group: dict[str, set] = {}
    for e in m["points"].values():
        by_group.setdefault(e["replica_group"], set()).add(e["axis_values"]["model"])
    # Every member of a replica group agrees on every axis except the seed.
    assert all(len(models) == 1 for models in by_group.values())


def test_a_sweep_with_no_seed_axis_records_that_rather_than_defaulting():
    """The plan is explicit that one seed per point is not a result; `None` is the finding."""
    m = build_manifest("scan", ["model=dino,hybrid"])
    assert m["seed_axis"] is None
    assert all(e["replica_group"] is None for e in m["points"].values())


@pytest.mark.parametrize("key", SEED_KEYS)
def test_any_of_the_known_seed_keys_is_recognised(key):
    m = build_manifest("scan", [f"{key}=1,2", "model=hybrid"])
    assert m["seed_axis"] == key


def test_an_oversized_sweep_is_refused_rather_than_queued():
    """Each point is a GPU job."""
    with pytest.raises(SystemExit, match="over the --max"):
        build_manifest("big", ["a=range(1,20)", "b=range(1,20)"], max_points=64)


def test_the_manifest_round_trips(tmp_path):
    m = build_manifest("scan", ["model=dino,hybrid", "run.seed=range(1,3)"])
    path = write_manifest(tmp_path / "scan", m)
    assert not list((tmp_path / "scan").glob("*.tmp"))
    assert load_manifest(tmp_path / "scan") == m  # by directory
    assert load_manifest(path) == m  # or by file
    assert json.loads(path.read_text())["sweep_id"] == "scan"


def test_a_missing_manifest_says_where_they_live(tmp_path):
    with pytest.raises(SystemExit, match="wcfm sweep"):
        load_manifest(tmp_path / "nope")


# ------------------------------------------------------------------------------ the command


def test_run_name_by_hand_is_refused(capsys):
    """Every point would write into one run directory and overwrite the last."""
    from wcfm.cli.sweep import main

    assert main(["model=dino,hybrid", "run.name=mine"]) == 2
    assert "must not be given by hand" in capsys.readouterr().err


def test_no_overrides_is_refused(capsys):
    from wcfm.cli.sweep import main

    assert main([]) == 0  # usage
    capsys.readouterr()
    assert main(["--id", "x"]) == 2
    assert "needs at least one override" in capsys.readouterr().err


def test_dry_run_enumerates_and_submits_nothing(tmp_path, capsys, monkeypatch):
    """`WCFM_OUTPUT_BASE` is redirected here and in every other test that reaches `main`.

    A dry run still goes through `wcfm submit --dry-run`, which writes each point's `.sub` into
    a run directory under the output base -- so without this the suite leaves a directory per
    point on GPFS every time it runs. It did, eight of them, before this was noticed.

    Note the base comes from the **environment**, not from `cfg.run.output_root`: overriding
    the config key does nothing, which is the first thing I tried.
    """
    from wcfm.cli.sweep import main

    monkeypatch.setenv("WCFM_OUTPUT_BASE", str(tmp_path / "out"))

    rc = main(
        [
            "model=dino,hybrid",
            "--id=scan",
            f"--sweeps-root={tmp_path}",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "2 point(s) enumerated, nothing submitted" in out
    # And it wrote nothing, so a dry run cannot leave a manifest for a campaign that never ran.
    assert not (tmp_path / "scan").exists()


def test_a_sweep_without_seeds_warns_that_one_seed_is_not_a_result(
    tmp_path, capsys, monkeypatch
):
    from wcfm.cli.sweep import main

    monkeypatch.setenv("WCFM_OUTPUT_BASE", str(tmp_path / "out"))
    main(
        [
            "model=dino,hybrid",
            "--id=scan",
            f"--sweeps-root={tmp_path}",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert "NOT swept" in out and "not a result" in out


def test_a_sweep_with_seeds_reports_the_replica_structure(tmp_path, capsys, monkeypatch):
    from wcfm.cli.sweep import main

    monkeypatch.setenv("WCFM_OUTPUT_BASE", str(tmp_path / "out"))
    main(
        [
            "model=dino,hybrid",
            "run.seed=range(1,4)",
            "--id=scan",
            f"--sweeps-root={tmp_path}",
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert "replica axis" in out
    assert "2 distinct configuration(s)" in out and "3 replica(s)" in out


# --------------------------------------------------------- what the manifest buys `compare`


def _rows_for(manifest, score_by_point):
    from wcfm.eval.compare import build_rows

    merged = {}
    for name in manifest["points"]:
        merged[f"{name}:ep10:student"] = {
            "n_events": 100,
            "eval_set_id": "s",
            "event_key_hash": "h",
            "pid": {"mlp_feat": score_by_point(name)},
        }
    return merged, build_rows(merged)


def test_sweep_columns_add_one_column_per_declared_axis():
    from wcfm.eval.compare import sweep_columns

    m = build_manifest("scan", ["model=dino,hybrid", "run.seed=range(1,3)"])
    merged, _ = _rows_for(m, lambda _n: 0.5)
    cols = sweep_columns(m, merged)
    assert len(cols) == 4
    for values in cols.values():
        assert set(values) == {"model", "run.seed"}
        assert values["model"] in ("dino", "hybrid")


def test_a_run_outside_the_manifest_is_kept_without_axis_values():
    """A table may legitimately hold a baseline that was not part of the campaign."""
    from wcfm.eval.compare import build_rows, sweep_columns

    m = build_manifest("scan", ["model=dino,hybrid"])
    merged, _ = _rows_for(m, lambda _n: 0.5)
    merged["baseline:ep10:student"] = {
        "n_events": 100, "eval_set_id": "s", "event_key_hash": "h",
        "pid": {"mlp_feat": 0.1},
    }
    header, rows, _ = build_rows(merged, sweep_columns(m, merged))
    assert "model" in header
    by_run = {r["run"]: r for r in rows}
    assert by_run["baseline:ep10:student"]["model"] == "-"
    assert len(rows) == 3


def test_declared_replicas_beat_the_name_guess():
    """The concrete payoff, and the reason `--by-config` was never a replacement.

    `wcfm sweep` names points `<sweep_id>_<hash8>`. The regex fallback strips a `_seed<N>`
    suffix, which those names do not have -- so without the manifest every point is its own
    family and `--group-by-seed` collapses nothing. With it, the three seeds of one
    configuration collapse to one row carrying their spread.
    """
    from wcfm.eval.compare import build_rows, group_by_seed

    m = build_manifest("scan", ["model=dino,hybrid", "run.seed=range(1,4)"])
    scores = {}
    for name, e in m["points"].items():
        scores[name] = 0.6 if e["axis_values"]["model"] == "dino" else 0.8
    merged, _ = _rows_for(m, lambda n: scores[n])
    header, rows, _ = build_rows(merged)

    # Without the manifest: six families, nothing collapsed.
    _, guessed = group_by_seed(header, rows)
    assert len(guessed) == 6

    # With it: two configurations, three replicas each, and the row says which is which.
    _, declared = group_by_seed(header, rows, m)
    assert len(declared) == 2
    assert {r["n"] for r in declared} == {"3"}
    assert sorted(r["run"].split(":")[0] for r in declared) == ["model=dino", "model=hybrid"]
    by_cfg = {r["run"].split(":")[0]: r["pid_mlp"] for r in declared}
    assert by_cfg["model=dino"].startswith("0.6")
    assert by_cfg["model=hybrid"].startswith("0.8")


def test_replicas_of_different_epochs_are_not_collapsed_together():
    """Two epochs of one run are not replicas of each other."""
    from wcfm.eval.compare import build_rows, group_by_seed

    m = build_manifest("scan", ["model=dino", "run.seed=range(1,3)"])
    merged = {}
    for name in m["points"]:
        for epoch in (10, 20):
            merged[f"{name}:ep{epoch}:student"] = {
                "n_events": 100, "eval_set_id": "s", "event_key_hash": "h",
                "pid": {"mlp_feat": 0.5},
            }
    header, rows, _ = build_rows(merged)
    _, declared = group_by_seed(header, rows, m)
    assert len(declared) == 2  # one per epoch
    assert {r["n"] for r in declared} == {"2"}
