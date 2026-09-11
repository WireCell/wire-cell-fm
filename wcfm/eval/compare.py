"""Merge probe results into one table, across checkpoints, runs and objectives.

Every probe writes JSON keyed `<run>:<epoch tag>:<source>`, so result files from different
epochs, runs and objectives merge without any bookkeeping. Rows sort by run then epoch; a
missing metric shows as `-` rather than breaking the table, so a partial run still tabulates.

This produces no new data. Every number is read straight out of the JSONs, which are the durable
result, and the table is a view that rebuilds in seconds. That is why the Condor DAG treats this
step as non-fatal.

A differing `eval_set_id` or event-key hash is an error. The rows are then not measurements of
the same thing and there is no correct reading of the table, so a warning printed above it is
not useful. The softer axes -- charge transform, pool size, extraction size, sweep settings --
stay warnings, because they invalidate some columns rather than all of them.

`sample` is a column. Until a production ships a held-out split every result is `in-sample`, and
a column saying so is what stops the number being read as a validation score later.

`--group-by-seed` collapses replicas into mean and spread, which is the only honest way to read
a two-run difference: the seed-to-seed spread is the scale a real difference has to clear.

`--sweep <id>` and `--by-config` are complementary. `--sweep` reads a campaign's manifest
(`wcfm sweep`) and adds one column per declared axis, while `--by-config` reads what the runs'
own `config.yaml` files resolved to. The manifest says what was intended and the configs say
what happened, so a disagreement is a finding: a point whose config does not match its manifest
entry did not run what the campaign thinks it ran. Only `--sweep` can say which runs were one
campaign, or which are seed replicas.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

__all__ = [
    "COLUMNS",
    "build_rows",
    "check_comparability",
    "dig",
    "load_all",
    "sweep_columns",
]

# (column, dotted path into the entry, decimals). Deltas are feat - raw: the part of the score
# the representation is responsible for. A tuple of paths is tried in order, which is how
# results recorded under superseded key names still tabulate alongside current ones.
COLUMNS: list[tuple[str, Any, int]] = [
    ("events", "n_events", 0),
    ("sample", "sample", -1),
    ("rows", "rows", -1),
    ("pid_svm", "pid.svm_feat", 4),
    ("pid_mlp", "pid.mlp_feat", 4),
    ("pid_raw", "pid.mlp_raw", 4),
    ("d_pid", "pid.delta_mlp", 4),
    ("pid_miou", "pid.miou_mlp_feat", 4),
    # Overlap: F1 on the contaminated call at the headline threshold (0.2). Alternative paths
    # (the tuple form) are for RENAMED keys only. Never point a column at two different
    # measurements -- an MAE and an F1 answer different questions, and a column that silently
    # holds either is unreadable. Same rule for instance and vertex below.
    ("ov_f1", "overlap.f1_mlp_feat", 4),
    ("ov_f1_raw", "overlap.f1_mlp_raw", 4),
    ("d_ov_f1", "overlap.delta_f1_mlp", 4),
    # The delta's sign is not the same for both heads -- on the mixed production the SVM makes
    # the features win the same call the MLP makes them lose. The headline stays the MLP; this
    # column is here so that disagreement is on the table rather than one head's answer standing
    # in for "the" result.
    ("d_ov_f1_svm", "overlap.delta_f1_svm", 4),
    # A random guess on the same pixels, so the F1 columns above can be read. Reached through
    # the sweep because only the scores are flattened to the top of the probe's entry.
    ("ov_chance", ["overlap", "sweep", "0.2", "chance", "uniform", "f1"], 4),
    # Instance: macro over particle-size bins of the margin over chance. NOT an accuracy --
    # chance runs from 0.00 in the small bins to 0.72 in the largest, so a raw accuracy is not
    # comparable across bins and the pooled figure answers the opposite question.
    ("inst_mgn", "instance.macro_margin_feat", 4),
    ("inst_mgn_raw", "instance.macro_margin_raw", 4),
    ("d_inst_mgn", "instance.delta_macro_margin", 4),
    # Vertex: F1 on the near/far call at the headline radius (20 px).
    ("vtx_f1", "vertex.f1_mlp_feat", 4),
    ("vtx_f1_raw", "vertex.f1_mlp_raw", 4),
    ("d_vtx_f1", "vertex.delta_f1_mlp", 4),
    # Coin-flip F1 on the same pixels. Worth reading before the delta: near-vertex pixels are
    # ~5% of the image and both sides land close to this floor, which a bare F1 of 0.13 against
    # 0.16 does not show.
    ("vtx_chance", ["vertex", "sweep", "20", "chance", "f1"], 4),
    ("knn10", "event_knn.feat.10.accuracy", 4),
    ("d_knn10", "event_knn.delta_accuracy.10", 4),
    ("knn10_f1", "event_knn.feat.10.macro_f1", 4),
    # Always predicting the most common flavor. The majority floor rather than the uniform one
    # because the column beside it is an accuracy, and that is the floor an accuracy must clear.
    ("evt_chance", "event_knn.chance.majority.accuracy", 4),
    # Non-parametric pixel k-NN. NOT leakage-free -- a relative tracking curve, not comparable
    # with the trained-head columns.
    ("knnpix", "knn_pixel.overall_accuracy", 4),
    ("knnpix_f1", "knn_pixel.macro_f1", 4),
    # Spectrum: the collapse measures.
    ("pr", "spectrum.participation_ratio", 2),
    ("rankme", "spectrum.rankme", 2),
    ("d90", "spectrum.dims_for_90pct", 0),
    ("st_cos", "spectrum.teacher_cosine_mean", 4),
    ("sep", "spectrum.separation_ratio", 4),
]

# (display name, the namespaces a probe writes under, the column-path prefixes that read it).
# Tells "this probe did not run" apart from "this probe ran and COLUMNS does not match what it
# writes" -- the second is silent otherwise.
METRIC_PROBES = [
    ("PID", ("pid",), ("pid.",)),
    ("overlap", ("overlap",), ("overlap.",)),
    ("instance", ("instance",), ("instance.",)),
    ("vertex", ("vertex",), ("vertex.",)),
    ("event flavor", ("event_knn",), ("event_knn.",)),
    ("kNN PID", ("knn_pixel",), ("knn_pixel.",)),
    ("spectrum", ("spectrum",), ("spectrum.",)),
]


def dig(entry: dict, path):
    """Follow a dotted path; `None` if any step is missing or not a mapping.

    `path` may be `"a.b.c"` (dotted, the usual form), `["a", "0.2"]` (an explicit key
    list, for keys that contain a dot -- the sweeps are keyed by their threshold and overlap's
    headline is the literal key `"0.2"`, which a dotted path would split), or a tuple of
    alternatives tried in order, for keys that were renamed.
    """
    if isinstance(path, tuple):
        for alt in path:
            v = dig(entry, alt)
            if v is not None:
                return v
        return None
    cur = entry
    for part in path if isinstance(path, list) else path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def epoch_of(label: str) -> int:
    """Sort key from the epoch tag (`ep100` -> 100); -1 when unparseable."""
    m = re.search(r"ep(\d+)", label)
    return int(m.group(1)) if m else -1


def run_of(label: str) -> str:
    return label.split(":")[0]


def source_of(label: str) -> str:
    parts = label.split(":")
    return parts[2] if len(parts) > 2 else ""


def load_all(paths) -> dict:
    """Merge result files. Later files win per (label, metric), so re-running one metric into a
    new JSON updates the table without discarding the others."""
    merged: dict[str, dict] = {}
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"[skip] {path}: not found")
            continue
        try:
            data = json.loads(path.read_text())
        except ValueError as exc:
            print(f"[skip] {path}: not valid JSON ({exc})")
            continue
        if not isinstance(data, dict):
            print(f"[skip] {path}: not a probe result JSON")
            continue
        for label, entry in data.items():
            if isinstance(entry, dict):
                merged.setdefault(label, {}).update(entry)
    return merged


def check_comparability(merged: dict) -> list[str]:
    """Refuse a table whose rows are not measurements of the same thing; warn where they are
    only partly so. Returns the warnings, and raises on the two hard axes.

    A differing `eval_set_id` or event-key hash means the rows were scored on different events
    and no column is readable, so a warning above the table is not a useful thing to print.
    Everything else invalidates some columns and leaves others standing.
    """
    hard = {}
    for field, what in (
        ("eval_set_id", "eval set"),
        ("event_key_hash", "event-key hash"),
    ):
        groups: dict[Any, list[str]] = {}
        for label, entry in merged.items():
            v = entry.get(field)
            if v:
                groups.setdefault(v, []).append(label)
        if len(groups) > 1:
            hard[what] = groups
    if hard:
        what, groups = next(iter(hard.items()))
        raise SystemExit(
            f"these results were scored against a different {what} and cannot be tabulated "
            "together -- the rows are not measurements of the same events:\n"
            + "\n".join(
                f"  {str(k)[:16]}: {len(v)} row(s), e.g. {sorted(v)[0]}"
                for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))
            )
            + "\nScore them against one eval set (`--eval-set-root`), or tabulate them apart."
        )

    warnings: list[str] = []

    def _warn_field(getter, what: str, tail: str) -> None:
        groups: dict[Any, list[str]] = {}
        for label, entry in merged.items():
            v = getter(entry)
            if v is not None and v != "":
                groups.setdefault(v, []).append(label)
        if len(groups) > 1:
            warnings.append(
                f"mixed {what} in this table -- {tail}: "
                + ", ".join(
                    f"{k} ({len(v)} row(s))"
                    for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))
                )
            )

    _warn_field(
        lambda e: e.get("raw_charge_transform"),
        "raw-charge transform",
        "the raw and delta columns are NOT comparable across these groups",
    )
    _warn_field(
        lambda e: e.get("pool_per_class"),
        "pool_per_class",
        "a balanced score over a different pool is a different measurement",
    )
    _warn_field(
        lambda e: e.get("n_events"),
        "extraction size (n_events)",
        "rare types rest on very different event counts",
    )
    _warn_field(
        lambda e: e.get("rows"),
        "row space",
        "scores should agree, but a `pooled` store cannot answer a probe whose pool it omitted",
    )
    _warn_field(
        lambda e: e.get("sample"),
        "sample",
        "an in-sample score and a held-out one are different claims",
    )
    # The threshold metrics define their positive class with a parameter, so two entries scored
    # at different settings answer different questions while their F1 columns line up.
    for path, what in (
        ("overlap.headline_threshold", "overlap threshold"),
        ("vertex.headline_radius_px", "vertex radius (px)"),
        ("vertex.t0_ticks_assumed", "vertex t0 (ticks)"),
    ):
        _warn_field(
            lambda e, _p=path: dig(e, _p),
            what,
            "these rows define the positive class differently",
        )
    return warnings


def report_coverage(merged: dict, active) -> list[str]:
    """Name any metric present in the files that produced no column.

    The table keeps only populated columns, so a probe whose result keys do not match COLUMNS
    vanishes from it entirely rather than showing `-`, which is indistinguishable from never
    having run the probe. Naming the two cases apart is what stops a metric going unreported.
    """
    paths = []
    for _, path, _ in active:
        alts = path if isinstance(path, tuple) else (path,)
        paths.extend(".".join(a) if isinstance(a, list) else a for a in alts)

    missing, mismatched = [], []
    for label, namespaces, prefixes in METRIC_PROBES:
        present = any(ns in e for e in merged.values() for ns in namespaces)
        shown = any(p.startswith(prefixes) for p in paths)
        if present and not shown:
            mismatched.append(label)
        elif not present:
            missing.append(label)

    out = []
    if mismatched:
        out.append(
            "results present but no column matched, so these metrics are NOT in the table: "
            + ", ".join(mismatched)
            + " -- COLUMNS in wcfm/eval/compare.py is out of date with what the probe writes"
        )
    if missing:
        out.append(f"no results for: {', '.join(missing)}")
    return out


def _fmt(v, nd: int) -> str:
    if v is None:
        return "-"
    if nd < 0:
        return str(v)
    return f"{v:.{nd}f}" if nd else f"{int(v)}"


def build_rows(merged: dict, extra: dict[str, dict[str, Any]] | None = None):
    """Keep only columns that at least one run populated. `extra` adds per-label columns."""
    active = [
        (name, path, nd)
        for name, path, nd in COLUMNS
        if any(dig(e, path) is not None for e in merged.values())
    ]
    notes = report_coverage(merged, active)
    labels = sorted(merged, key=lambda s: (run_of(s), epoch_of(s), s))
    extra_cols = sorted({k for v in (extra or {}).values() for k in v})

    rows = []
    for label in labels:
        entry = merged[label]
        row = {"run": label}
        for name, path, nd in active:
            row[name] = _fmt(dig(entry, path), nd)
        for k in extra_cols:
            v = (extra or {}).get(label, {}).get(k)
            row[k] = "-" if v is None else str(v)
        rows.append(row)
    return ["run"] + [c[0] for c in active] + extra_cols, rows, notes


def _replica_groups(manifest: dict | None) -> dict[str, str]:
    """`{run_name: replica_group}` from a sweep manifest, empty without one."""
    if not manifest or not manifest.get("seed_axis"):
        return {}
    out = {}
    for name, entry in (manifest.get("points") or {}).items():
        group = entry.get("replica_group")
        if group:
            # Named by the axis values rather than by the hash, so the collapsed row says what
            # configuration it is rather than making a reader look the hash up.
            values = {k: v for k, v in (entry.get("axis_values") or {}).items()
                      if k != manifest["seed_axis"]}
            label = ",".join(f"{k}={v}" for k, v in sorted(values.items())) or manifest["sweep_id"]
            out[name] = label
    return out


def sweep_columns(manifest: dict, merged: dict) -> dict[str, dict]:
    """One column per declared axis, for the runs a manifest covers.

    A row whose run is not in the manifest gets no values rather than being dropped: a table may
    legitimately hold a baseline that was not part of the campaign.
    """
    points = manifest.get("points") or {}
    axes = manifest.get("axes") or []
    if not axes:
        return {}
    return {
        label: {k: (points[run_of(label)]["axis_values"] or {}).get(k, "-") for k in axes}
        for label in merged
        if run_of(label) in points
    }


_SEED_SUFFIX = re.compile(r"[_-](?:seed|s)\d+$")


def seed_group_of(label: str) -> str:
    """The replica family a label belongs to: its run name with a seed suffix stripped.

    `hybrid_b100_seed3:ep100:student` and `hybrid_b100_seed7:ep100:student` are one family.
    A run whose name carries no seed suffix is its own family, which is the honest reading --
    nothing else in the result file says two runs are replicas.
    """
    run = _SEED_SUFFIX.sub("", run_of(label))
    return f"{run}:ep{epoch_of(label)}:{source_of(label)}"


def group_by_seed(
    header: list[str], rows: list[dict], manifest: dict | None = None
) -> tuple[list[str], list[dict]]:
    """Collapse seed replicas into `mean +- spread`.

    With a sweep manifest, replicas are declared; without one they are guessed. The guess is
    `seed_group_of`, which strips a `_seed<N>` suffix off the run name: fine for runs named that
    way, and silently wrong for a campaign that named its points any other way, including the
    `<sweep_id>_<hash8>` names `wcfm sweep` produces. Given a manifest, two points are replicas
    exactly when every axis but the seed agrees, which is what `replica_group` records. Pass one
    whenever there is one.

    The spread is the sample standard deviation over replicas, and it is the point: a difference
    between two runs means nothing until it clears the seed-to-seed spread. A family with one
    member reports its value and an empty spread rather than a zero, since a zero spread would
    read as "measured, and the replicas agreed exactly".
    """
    import statistics

    families: dict[str, list[dict]] = {}
    declared = _replica_groups(manifest)
    for row in rows:
        run = run_of(row["run"])
        if run in declared:
            # `<group>:<epoch>:<source>` -- replicas of one configuration, scored at the same
            # epoch on the same branch. Two epochs of one run are not replicas of each other.
            key = f"{declared[run]}:ep{epoch_of(row['run'])}:{source_of(row['run'])}"
        else:
            key = seed_group_of(row["run"])
        families.setdefault(key, []).append(row)

    out = []
    for family, members in sorted(families.items()):
        row = {"run": family, "n": str(len(members))}
        for col in header[1:]:
            vals = []
            for m in members:
                try:
                    vals.append(float(m[col]))
                except (TypeError, ValueError):
                    pass
            if not vals:
                # Non-numeric columns (sample, rows) collapse to their common value, or to the
                # set when the replicas disagree -- which is itself worth seeing.
                seen = {m[col] for m in members if m[col] != "-"}
                row[col] = next(iter(seen)) if len(seen) == 1 else ("/".join(sorted(seen)) or "-")
            elif len(vals) == 1:
                row[col] = f"{vals[0]:.4g}"
            else:
                row[col] = f"{statistics.mean(vals):.4g}+-{statistics.stdev(vals):.2g}"
        out.append(row)
    return ["run", "n"] + header[1:], out


def render(header: list[str], rows: list[dict], markdown: bool = False) -> str:
    widths = {c: max(len(c), *(len(str(r.get(c, "-"))) for r in rows)) for c in header}
    lines = []
    if markdown:
        lines.append("| " + " | ".join(c.ljust(widths[c]) for c in header) + " |")
        lines.append("|" + "|".join("-" * (widths[c] + 2) for c in header) + "|")
        for r in rows:
            lines.append(
                "| " + " | ".join(str(r.get(c, "-")).ljust(widths[c]) for c in header) + " |"
            )
    else:
        lines.append("  ".join(c.ljust(widths[c]) for c in header))
        for r in rows:
            lines.append("  ".join(str(r.get(c, "-")).ljust(widths[c]) for c in header))
    return "\n".join(lines)


def write_csv(path: Path | str, header: list[str], rows: list[dict]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows([{c: r.get(c, "-") for c in header} for r in rows])
    return out
