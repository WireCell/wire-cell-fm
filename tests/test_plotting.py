"""The reading, not the drawing: sparse streams, panel selection, and the CLI's refusals.

The one behaviour worth pinning is that a series is built from the records that carry the key.
A step stream holds `gradnorm/*` on one record in a hundred, so anything that indexes every
record by that key raises, and anything that collects the values without their steps plots them
against the wrong x by the cadence ratio.

Drawing itself needs matplotlib, which is in the `analysis` extra; those tests skip without it.
"""

from __future__ import annotations

import json

import pytest

from wcfm.plotting.probes import PROBE_PANELS, _value, probe_files
from wcfm.plotting.streams import families, keys_present, read_records, series, smooth

SPARSE = [
    {"step": 0, "loss": 1.0, "gradnorm/enc1/grad_norm": 0.5},
    {"step": 1, "loss": 0.9},
    {"step": 2, "loss": 0.8},
    {"step": 3, "loss": float("nan")},
    {"step": 4, "loss": 0.7, "gradnorm/enc1/grad_norm": 0.25, "gradnorm/dec1/grad_norm": 0.1},
]


def test_series_pairs_a_sparse_key_with_its_own_steps():
    xs, ys = series(SPARSE, "gradnorm/enc1/grad_norm")
    assert xs == [0.0, 4.0]
    assert ys == [0.5, 0.25]


def test_series_drops_nan_rather_than_plotting_it():
    xs, ys = series(SPARSE, "loss")
    assert xs == [0.0, 1.0, 2.0, 4.0]
    assert ys == [1.0, 0.9, 0.8, 0.7]


def test_series_of_an_absent_key_is_empty_not_an_error():
    assert series(SPARSE, "no_such_metric") == ([], [])


def test_families_enumerates_the_groups_a_run_recorded():
    assert families(keys_present(SPARSE), "gradnorm/") == ["dec1", "enc1"]


def test_smooth_keeps_the_length_and_the_ends():
    out = smooth([0.0, 1.0, 2.0, 3.0, 4.0], 3)
    assert len(out) == 5
    assert out[0] == pytest.approx(0.5)
    assert out[2] == pytest.approx(2.0)


def test_read_records_of_a_run_without_metrics_is_empty(tmp_path):
    assert read_records(tmp_path) == []


def test_value_reads_the_paths_the_merge_table_uses():
    entry = {"pid": {"mlp_feat": 0.37, "chance": {"uniform": {"all_acc": 0.14}}}}
    assert _value(entry, "pid.mlp_feat") == pytest.approx(0.37)
    assert _value(entry, "pid.chance.uniform.all_acc") == pytest.approx(0.14)
    assert _value(entry, "pid.svm_feat") is None


def test_every_panel_declares_a_feat_role():
    """`raw` and `chance` are optional -- some probes record no baseline -- but a panel with no
    `feat` series has nothing to say."""
    assert all("feat" in roles for _, _, roles in PROBE_PANELS)


def test_panel_titles_are_unique():
    """The title is the only thing that tells two panels apart on the page."""
    titles = [t for t, _, _ in PROBE_PANELS]
    assert len(titles) == len(set(titles))


def test_a_constant_chance_reads_as_itself():
    assert _value({}, 0.0) == 0.0


def test_probe_files_accepts_a_run_or_a_probes_directory(tmp_path):
    run = tmp_path / "run"
    (run / "probes").mkdir(parents=True)
    (run / "probes" / "pid_ep10.json").write_text("{}")
    assert probe_files([run]) == probe_files([run / "probes"])


def _write_probe(root, epoch, value):
    path = root / "probes"
    path.mkdir(parents=True, exist_ok=True)
    entry = {
        "pid": {
            "mlp_feat": value,
            "mlp_raw": 0.19,
            "chance": {"uniform": {"m_f1": 0.14}},
            "headline_classes": ["Track", "Shower"],
            "per_class_f1": {
                "mlp_feat": {"Track": value + 0.1, "Shower": value - 0.1},
                "mlp_raw": {"Track": 0.2, "Shower": 0.1},
            },
        },
    }
    (path / f"pid_ep{epoch}.json").write_text(json.dumps({f"{root.name}:ep{epoch}:student": entry}))


def test_plot_probes_writes_the_trajectory_and_per_class_figures(tmp_path):
    """No `knn_pixel` in the entries, so the k-NN recall figure is skipped rather than empty."""
    pytest.importorskip("matplotlib")
    from wcfm.plotting import plot_probes

    run = tmp_path / "demo"
    for epoch, value in ((10, 0.30), (20, 0.35)):
        _write_probe(run, epoch, value)
    written = plot_probes([run], tmp_path / "plots")
    assert [p.name for p in written] == ["probes.png", "pid_per_class.png"]
    assert all(p.stat().st_size > 0 for p in written)


def _write_stream(root, per_parameter: bool):
    (root / "metrics").mkdir(parents=True)
    with (root / "metrics" / "step.jsonl").open("w") as fh:
        for step in range(0, 1000, 100):
            rec = {
                "step": step,
                "loss": 1.0 / (step + 1),
                "gradnorm/enc1/grad_norm": 0.5,
                "gradnorm/enc1/grad_to_param": 0.01,
            }
            if per_parameter:
                rec["gradnorm/enc1/param/conv1.0.weight/grad_norm"] = 0.3
                rec["gradnorm/enc1/param/block1.conv2.0.weight/grad_norm"] = 0.4
            fh.write(json.dumps(rec) + "\n")
    (root / "metrics" / "epoch.jsonl").write_text("")


def test_gradnorm_draws_a_line_per_parameter_when_the_stream_carries_them(tmp_path):
    pytest.importorskip("matplotlib")
    from wcfm.plotting.diagnostics import plot_diagnostics

    run = tmp_path / "demo"
    _write_stream(run, per_parameter=True)
    written = plot_diagnostics([run], tmp_path / "plots", only=("gradnorm",))
    assert [p.name for p in written] == ["gradnorm.png"]


def test_gradnorm_falls_back_to_the_group_line(tmp_path):
    pytest.importorskip("matplotlib")
    from wcfm.plotting.diagnostics import plot_diagnostics

    run = tmp_path / "demo"
    _write_stream(run, per_parameter=False)
    written = plot_diagnostics([run], tmp_path / "plots", only=("gradnorm",))
    assert [p.name for p in written] == ["gradnorm.png"]


def test_plot_probes_refuses_a_run_with_no_probe_json(tmp_path):
    pytest.importorskip("matplotlib")
    from wcfm.plotting import plot_probes

    with pytest.raises(SystemExit, match="no probe JSONs"):
        plot_probes([tmp_path], tmp_path / "plots")


def test_cli_needs_an_out_dir_for_more_than_one_run(tmp_path):
    from wcfm.cli.plot import main

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert main([str(a), str(b)]) == 2


def test_cli_rejects_an_unknown_figure(tmp_path):
    from wcfm.cli.plot import main

    run = tmp_path / "run"
    run.mkdir()
    assert main([str(run), "--figures=nonesuch"]) == 2
