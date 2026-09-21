"""Probe results as curves over epoch, drawn from the same JSONs `wcfm eval merge` tabulates.

The merge layer supplies the whole reading: `load_all` merges the files, `dig` follows a path
into an entry, and `epoch_of`/`run_of`/`source_of` decompose the `<run>:<epoch tag>:<source>`
key every probe writes. Nothing is re-flattened here: a second flattening drifts from the
table, and then a figure and a table disagree about the same run.

Three figures:

- `probes`: one panel per headline metric. Each draws the features, the same head on raw
  charge, and the chance floor where the probe recorded them. A feature curve alone is
  unreadable: whether a score is good depends on where the raw floor and chance sit.
- `pid_per_class`: the PID head's F1 per particle type. The macro average hides where a
  representation fails, which is the rare types.
- `knn_recall`: the untrained pixel k-NN's per-type recall, which asks whether a type is
  separable in feature space at all.

A panel whose paths no run populated is skipped rather than drawn empty, which is how a run
extracted without per-pixel truth -- no `overlap`, no `instance` -- still produces a figure of
the probes it does have.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from wcfm.eval.compare import dig, epoch_of, load_all, run_of, source_of

from .style import ROLE, ROLE_LABEL, figure, run_colors, save

__all__ = ["PROBE_PANELS", "plot_probes", "probe_files"]

#: A path into a probe entry: dotted, an explicit key list (for keys containing a dot), a tuple
#: of alternatives, a callable over the entry, or a constant.
Path_t = str | list[str] | tuple | float | Callable[[dict], Any]


def _uniform_over(classes_path: str) -> Callable[[dict], Any]:
    """Chance for a balanced multi-class call the probe did not record a floor for."""

    def chance(entry: dict):
        classes = dig(entry, classes_path)
        return 1.0 / len(classes) if isinstance(classes, list) and classes else None

    return chance


#: (title, ylabel, {role: path}). Paths are the same dotted form `wcfm eval merge` uses. The
#: PID probe's `mlp_feat` and `svm_feat` are macro-F1 over the six headline types. The instance
#: margin already has chance subtracted, so its reference is the constant 0.
PROBE_PANELS: list[tuple[str, str, dict[str, Path_t]]] = [
    (
        "PID macro-F1 (MLP head)",
        "macro-F1",
        {
            "feat": "pid.mlp_feat",
            "raw": "pid.mlp_raw",
            "chance": "pid.chance.uniform.m_f1",
        },
    ),
    (
        "PID macro-F1 (SVM head)",
        "macro-F1",
        {
            "feat": "pid.svm_feat",
            "raw": "pid.svm_raw",
            "chance": "pid.chance.uniform.m_f1",
        },
    ),
    (
        "PID macro-IoU (MLP head)",
        "mIoU",
        {
            "feat": "pid.miou_mlp_feat",
            "raw": "pid.miou_mlp_raw",
            "chance": "pid.chance.uniform.m_iou",
        },
    ),
    (
        "overlap F1 (t = 0.2, MLP head)",
        "F1",
        {
            "feat": "overlap.f1_mlp_feat",
            "raw": "overlap.f1_mlp_raw",
            "chance": ["overlap", "sweep", "0.2", "chance", "uniform", "f1"],
        },
    ),
    (
        "instance macro margin",
        "margin over chance",
        {
            "feat": "instance.macro_margin_feat",
            "raw": "instance.macro_margin_raw",
            "chance": 0.0,
        },
    ),
    (
        "vertex F1 (r = 20 px, MLP head)",
        "F1",
        {
            "feat": "vertex.f1_mlp_feat",
            "raw": "vertex.f1_mlp_raw",
            "chance": ["vertex", "sweep", "20", "chance", "f1"],
        },
    ),
    (
        "event flavour accuracy (k-NN, k = 10)",
        "accuracy",
        {
            "feat": "event_knn.feat.10.accuracy",
            "raw": "event_knn.raw.10.accuracy",
            "chance": "event_knn.chance.majority.accuracy",
        },
    ),
    (
        "event flavour macro-F1 (k-NN, k = 10)",
        "macro-F1",
        {
            "feat": "event_knn.feat.10.macro_f1",
            "raw": "event_knn.raw.10.macro_f1",
            "chance": "event_knn.chance.uniform.macro_f1",
        },
    ),
    (
        "pixel k-NN PID accuracy",
        "accuracy",
        {
            "feat": "knn_pixel.overall_accuracy",
            "chance": _uniform_over("knn_pixel.classes"),
        },
    ),
    (
        "pixel k-NN PID macro-F1",
        "macro-F1",
        {
            "feat": "knn_pixel.macro_f1",
            "chance": _uniform_over("knn_pixel.classes"),
        },
    ),
]


def probe_files(run_dirs: list[Path | str]) -> list[Path]:
    """Every `probes/*.json` under the given run directories, sorted.

    A run directory or a `probes/` directory both work: a campaign's files are sometimes
    collected somewhere that is not a run.
    """
    found: list[Path] = []
    for d in run_dirs:
        path = Path(d)
        probes = path if path.name == "probes" else path / "probes"
        found += sorted(probes.glob("*.json"))
    return found


def _value(entry: dict, path: Path_t):
    if isinstance(path, float):
        return path
    v = path(entry) if callable(path) else dig(entry, path)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def _curves(
    run_dirs: list[Path | str], sources: tuple[str, ...]
) -> dict[str, list[tuple[int, dict]]]:
    """`{curve key: [(epoch, entry), ...]}`, one curve per run, or per (run, source) when more
    than one source is drawn. Refuses to return nothing silently."""
    files = probe_files(run_dirs)
    if not files:
        roots = ", ".join(str(Path(d)) for d in run_dirs)
        raise SystemExit(
            f"no probe JSONs under {roots}. `wcfm eval submit <run>` writes them into "
            "<run>/probes; a run whose probe stage never succeeded has nothing to plot."
        )
    everything = load_all(files)
    merged = everything
    if sources:
        merged = {k: v for k, v in everything.items() if source_of(k) in sources}
    if not merged:
        have = ", ".join(sorted({source_of(k) for k in everything})) or "nothing"
        raise SystemExit(
            f"no probe results for source(s) {', '.join(sources)}; these files carry {have}"
        )

    found_sources = sorted({source_of(k) for k in merged})
    curves: dict[str, list[tuple[int, dict]]] = {}
    for label, entry in merged.items():
        run, src, ep = run_of(label), source_of(label), epoch_of(label)
        if ep < 0:
            continue
        key = run if len(found_sources) < 2 else f"{run}:{src}"
        curves.setdefault(key, []).append((ep, entry))
    for entries in curves.values():
        entries.sort(key=lambda t: t[0])

    # A run that contributed nothing must be named. A comparison figure that quietly becomes a
    # single-run plot is read as "the runs agree", which is the opposite of what happened.
    drew = {key.split(":")[0] for key in curves}
    named = [Path(d).name for d in run_dirs if Path(d).name != "probes"]
    silent = [n for n in named if n not in drew]
    if silent:
        print(f"[no probe results] {', '.join(silent)}: not drawn")
    return curves


def _series(entries: list[tuple[int, dict]], path: Path_t) -> tuple[list[int], list[float]]:
    pts = [(ep, _value(e, path)) for ep, e in entries]
    pts = [(ep, v) for ep, v in pts if v is not None]
    return [ep for ep, _ in pts], [v for _, v in pts]


def _style_axis(ax, title: str, ylabel: str, legend: bool = True):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, framealpha=0.85)


def plot_probes(
    run_dirs: list[Path | str],
    out_dir: Path | str,
    *,
    sources: tuple[str, ...] = ("student",),
    fmt: str = "png",
) -> list[Path]:
    """Draw the three probe figures for one or more runs into `out_dir`.

    `sources` filters the branch; `("student", "teacher")` overlays both, and `()` keeps every
    branch found. Returns the paths written; a figure with no data is not written.
    """
    curves = _curves(run_dirs, sources)
    colors = run_colors(sorted(curves))
    written: list[Path] = []
    for path in (
        _trajectories(curves, colors, out_dir, fmt),
        _per_class(
            curves,
            colors,
            out_dir,
            fmt,
            name="pid_per_class",
            classes_path="pid.headline_classes",
            feat=lambda c: ["pid", "per_class_f1", "mlp_feat", c],
            raw=lambda c: ["pid", "per_class_f1", "mlp_raw", c],
            ylabel="F1",
            suptitle="PID F1 per type (MLP head, balanced pool)",
        ),
        _per_class(
            curves,
            colors,
            out_dir,
            fmt,
            name="knn_recall",
            classes_path="knn_pixel.classes",
            feat=lambda c: ["knn_pixel", "per_class_accuracy", c],
            raw=None,
            ylabel="recall",
            suptitle="pixel k-NN PID recall per type (untrained, balanced pool)",
        ),
    ):
        if path is not None:
            written.append(path)
    return written


def _trajectories(curves, colors, out_dir, fmt) -> Path | None:
    """One panel per entry of `PROBE_PANELS`, three roles per curve."""
    scored = [e for entries in curves.values() for _, e in entries]
    live = [
        (title, ylabel, roles)
        for title, ylabel, roles in PROBE_PANELS
        if any(_value(e, roles["feat"]) is not None for e in scored)
    ]
    if not live:
        return None
    fig, axes = figure(len(live), ncols=3, height=2.8, width=4.4)
    for ax, (title, ylabel, roles) in zip(axes, live, strict=True):
        for key, entries in sorted(curves.items()):
            for role, path in roles.items():
                xs, ys = _series(entries, path)
                if not xs:
                    continue
                # One run on the axes: the legend is about the roles. Several: the run name is
                # the point, and the roles are told apart by line style.
                if len(curves) == 1:
                    label = ROLE_LABEL[role]
                else:
                    label = key if role == "feat" else f"{key} ({ROLE_LABEL[role]})"
                ax.plot(xs, ys, color=colors[key], label=label, **ROLE[role])
        _style_axis(ax, title, ylabel)
    return save(fig, out_dir, "probes", fmt)


def _per_class(
    curves,
    colors,
    out_dir,
    fmt,
    *,
    name: str,
    classes_path: str,
    feat: Callable[[str], Path_t],
    raw: Callable[[str], Path_t] | None,
    ylabel: str,
    suptitle: str,
) -> Path | None:
    """One panel per class named at `classes_path`, features and (optionally) raw charge.

    Each panel scales to its own range, so the shape of a trajectory is visible where the
    numbers are small. Panel heights therefore do not compare: read the axis.
    """
    classes: list[str] = []
    for entries in curves.values():
        for _, e in entries:
            found = dig(e, classes_path)
            if isinstance(found, list) and found:
                classes = [str(c) for c in found]
                break
        if classes:
            break
    if not classes:
        return None

    fig, axes = figure(len(classes), ncols=3, height=2.8, width=4.4)
    drew = False
    for ax, cls in zip(axes, classes, strict=True):
        for key, entries in sorted(curves.items()):
            roles = {"feat": feat(cls)} if raw is None else {"feat": feat(cls), "raw": raw(cls)}
            for role, path in roles.items():
                xs, ys = _series(entries, path)
                if not xs:
                    continue
                drew = True
                if len(curves) == 1:
                    label = ROLE_LABEL[role]
                else:
                    label = key if role == "feat" else f"{key} ({ROLE_LABEL[role]})"
                ax.plot(xs, ys, color=colors[key], label=label, **ROLE[role])
        _style_axis(ax, cls, ylabel)
    if not drew:
        from .style import pyplot

        pyplot().close(fig)
        return None
    fig.suptitle(suptitle, fontsize=11)
    return save(fig, out_dir, name, fmt)
