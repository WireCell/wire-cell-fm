"""`wcfm eval merge` / `compare`: the table, and what it refuses to tabulate.

The table produces no new data, so the tests here are about what it *declines* to show and what
it refuses to show it beside:

* two results scored on different event sets raise, rather than tabulating side by side under a
  warning nobody reads;
* a metric present in the files but not matched by any column is named, because otherwise it
  vanishes from the table and "no column" is indistinguishable from "probe never ran";
* seed replicas collapse to mean and spread, because a difference means nothing until it clears
  the seed-to-seed spread.
"""

from __future__ import annotations

import json

import pytest

from wcfm.eval.compare import (
    build_rows,
    check_comparability,
    dig,
    epoch_of,
    group_by_seed,
    load_all,
    render,
    seed_group_of,
    write_csv,
)


def _entry(**kw):
    base = {
        "n_events": 100,
        "eval_set_id": "set-a",
        "event_key_hash": "hhh",
        "sample": "in-sample",
        "rows": "all",
        "raw_charge_transform": "trained",
        "pool_per_class": 10000,
        "pid": {"svm_feat": 0.5, "mlp_feat": 0.6, "mlp_raw": 0.4, "delta_mlp": 0.2},
    }
    base.update(kw)
    return base


def _write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return str(p)


def test_dig_handles_dotted_list_and_alternative_paths():
    entry = {"a": {"b": {"c": 1}}, "overlap": {"sweep": {"0.2": {"f1": 0.7}}}}
    assert dig(entry, "a.b.c") == 1
    assert dig(entry, "a.b.missing") is None
    # A literal key containing a dot: a dotted path would split "0.2" into "0" and "2".
    assert dig(entry, ["overlap", "sweep", "0.2", "f1"]) == 0.7
    assert dig(entry, ("nope.here", "a.b.c")) == 1


def test_epoch_and_seed_family_parsing():
    assert epoch_of("run:ep100:student") == 100
    assert epoch_of("run:latest:student") == -1
    assert seed_group_of("hybrid_b100_seed3:ep100:student") == "hybrid_b100:ep100:student"
    assert seed_group_of("hybrid_b100_s7:ep100:student") == "hybrid_b100:ep100:student"
    # No seed suffix: its own family. Nothing in a result file says two runs are replicas.
    assert seed_group_of("hybrid_b100:ep100:student") == "hybrid_b100:ep100:student"


def test_later_files_win_per_metric_so_one_probe_can_be_rerun(tmp_path):
    a = _write(tmp_path, "a.json", {"r:ep1:student": _entry()})
    b = _write(tmp_path, "b.json", {"r:ep1:student": {"vertex": {"f1_mlp_feat": 0.9}}})
    merged = load_all([a, b])
    assert merged["r:ep1:student"]["pid"]["mlp_feat"] == 0.6  # kept
    assert merged["r:ep1:student"]["vertex"]["f1_mlp_feat"] == 0.9  # added


def test_a_bad_or_missing_file_is_skipped_not_fatal(tmp_path, capsys):
    good = _write(tmp_path, "g.json", {"r:ep1:student": _entry()})
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    merged = load_all([good, str(bad), str(tmp_path / "nope.json")])
    assert set(merged) == {"r:ep1:student"}
    out = capsys.readouterr().out
    assert "not valid JSON" in out and "not found" in out


def test_different_eval_sets_raise_rather_than_tabulating(tmp_path):
    """The plan's hard axis. There is no reading of such a table that is correct, so a warning
    above it -- which is what v1 printed -- is not a useful thing to do."""
    merged = {
        "a:ep1:student": _entry(),
        "b:ep1:student": _entry(eval_set_id="set-b"),
    }
    with pytest.raises(SystemExit, match="different eval set"):
        check_comparability(merged)


def test_a_changed_event_key_hash_also_raises():
    merged = {
        "a:ep1:student": _entry(),
        "b:ep1:student": _entry(event_key_hash="other"),
    }
    with pytest.raises(SystemExit, match="event-key hash"):
        check_comparability(merged)


def test_soft_axes_warn_because_they_invalidate_some_columns_not_all():
    merged = {
        "a:ep1:student": _entry(),
        "b:ep1:student": _entry(raw_charge_transform="log10_1p", pool_per_class=5000),
    }
    warnings = check_comparability(merged)
    text = " ".join(warnings)
    assert "raw-charge transform" in text
    assert "pool_per_class" in text


def test_a_uniform_table_warns_about_nothing():
    merged = {"a:ep1:student": _entry(), "b:ep2:student": _entry()}
    assert check_comparability(merged) == []


def test_only_populated_columns_survive_and_missing_metrics_are_named():
    merged = {"a:ep1:student": _entry()}
    header, rows, notes = build_rows(merged)
    assert "pid_mlp" in header
    # No vertex results, so no vertex column and a note saying which probes had none.
    assert "vtx_f1" not in header
    assert any("no results for" in n and "vertex" in n for n in notes)
    assert rows[0]["pid_mlp"] == "0.6000"


def test_a_metric_with_no_matching_column_is_named_rather_than_vanishing():
    """The failure this catches: a probe writes results, COLUMNS does not match its keys, and
    the metric silently leaves the table -- indistinguishable from never having run."""
    merged = {"a:ep1:student": _entry(overlap={"some_renamed_key": 0.5})}
    _, _, notes = build_rows(merged)
    assert any("no column matched" in n and "overlap" in n for n in notes)


def test_missing_values_show_as_a_dash_so_a_partial_run_still_tabulates():
    merged = {
        "a:ep1:student": _entry(),
        "b:ep1:student": _entry(pid={"svm_feat": 0.5}),  # no mlp
    }
    header, rows, _ = build_rows(merged)
    by_run = {r["run"]: r for r in rows}
    assert by_run["b:ep1:student"]["pid_mlp"] == "-"
    assert by_run["a:ep1:student"]["pid_mlp"] == "0.6000"


def test_rows_sort_by_run_then_epoch():
    merged = {
        "b:ep2:student": _entry(),
        "a:ep10:student": _entry(),
        "a:ep2:student": _entry(),
    }
    _, rows, _ = build_rows(merged)
    assert [r["run"] for r in rows] == ["a:ep2:student", "a:ep10:student", "b:ep2:student"]


def test_group_by_seed_reports_mean_and_spread():
    merged = {
        "hy_seed1:ep1:student": _entry(pid={"mlp_feat": 0.60}),
        "hy_seed2:ep1:student": _entry(pid={"mlp_feat": 0.70}),
        "hy_seed3:ep1:student": _entry(pid={"mlp_feat": 0.80}),
    }
    header, rows, _ = build_rows(merged)
    gh, grows = group_by_seed(header, rows)
    assert gh[:2] == ["run", "n"]
    assert len(grows) == 1
    row = grows[0]
    assert row["run"] == "hy:ep1:student" and row["n"] == "3"
    assert row["pid_mlp"].startswith("0.7") and "+-" in row["pid_mlp"]


def test_a_single_replica_reports_no_spread_rather_than_zero():
    """A zero spread would read as "measured, and the replicas agreed exactly"."""
    merged = {"solo:ep1:student": _entry()}
    header, rows, _ = build_rows(merged)
    _, grows = group_by_seed(header, rows)
    assert "+-" not in grows[0]["pid_mlp"]


def test_group_by_seed_surfaces_replicas_that_disagree_on_a_text_column():
    merged = {
        "hy_seed1:ep1:student": _entry(rows="all"),
        "hy_seed2:ep1:student": _entry(rows="pooled"),
    }
    header, rows, _ = build_rows(merged)
    _, grows = group_by_seed(header, rows)
    assert grows[0]["rows"] == "all/pooled"


def test_render_and_csv_agree_on_the_same_rows(tmp_path):
    merged = {"a:ep1:student": _entry()}
    header, rows, _ = build_rows(merged)
    text = render(header, rows)
    assert "run" in text.splitlines()[0]
    assert "0.6000" in text
    md = render(header, rows, markdown=True)
    assert md.splitlines()[0].startswith("|")

    out = write_csv(tmp_path / "t.csv", header, rows)
    import csv as _csv

    with open(out) as f:
        got = list(_csv.DictReader(f))
    assert got[0]["pid_mlp"] == "0.6000"


# ------------------------------------------------------------------------------ config diff


def test_flatten_uses_dotted_keys_and_indexes_lists():
    from wcfm.eval.config_diff import flatten

    flat = flatten({"a": {"b": 1}, "t": [{"w": 0.5}, {"w": 1.0}]})
    assert flat["a.b"] == 1
    # Indexed, so "the second term's weight changed" is one key rather than two lists to align.
    assert flat["t[0].w"] == 0.5 and flat["t[1].w"] == 1.0


def test_differing_keys_treats_absence_as_a_difference():
    from wcfm.eval.config_diff import differing_keys

    a = {"model.name": "hybrid", "model.terms.charge.weight": 0.2}
    b = {"model.name": "hybrid"}
    # "this run had no charge term at all" is the most interesting difference there is.
    assert differing_keys({"a": a, "b": b}) == ["model.terms.charge.weight"]
    assert differing_keys({"a": a, "b": dict(a)}) == []
    assert differing_keys({"a": a}) == []


def test_run_name_is_excluded_from_the_default_diff():
    from wcfm.eval.config_diff import differing_keys

    a = {"run.name": "x", "run.seed": 1}
    b = {"run.name": "y", "run.seed": 1}
    assert differing_keys({"a": a, "b": b}) == []
    # ...but --all shows it, because "the seed is the only difference" is what a replica table
    # needs to establish.
    assert differing_keys({"a": a, "b": b}, include_noise=True) == ["run.name"]


def test_load_run_config_names_the_path_it_wanted(tmp_path):
    from wcfm.eval.config_diff import load_run_config

    with pytest.raises(SystemExit, match="config.yaml"):
        load_run_config(tmp_path / "nope")
