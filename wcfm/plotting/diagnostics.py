"""Training diagnostics: the figures a run's own metrics streams support.

Panels are declared, not discovered, where the meaning of a key is fixed -- loss, the
schedules, throughput. Where the names depend on what the module was built from --
`gradnorm/<group>/`, `spectrum/<branch>/<tap>/` -- the series are enumerated from the stream
with `streams.families`, so a run carrying a head this module has never heard of still plots
it, and one carrying none skips the panel instead of drawing an empty frame.

Every figure takes a list of run directories and draws one colour per run.
"""

from __future__ import annotations

from pathlib import Path

from .streams import families, keys_present, read_records, series, smooth
from .style import KEY_DASHES, figure, run_colors, save

__all__ = ["DIAGNOSTIC_FIGURES", "plot_diagnostics"]

#: A panel is (title, ylabel, [(legend label, key)], log y). Each list feeds one figure, by
#: name: an entry added to one of them must not change what another draws.
_LOSS_SPLIT: list[tuple[str, str, list[tuple[str, str]], bool]] = [
    (
        "loss split (step)",
        "loss",
        [("masked", "loss_masked"), ("unmasked", "loss_unmasked")],
        False,
    ),
]

_SCHEDULE_PANELS: list[tuple[str, str, list[tuple[str, str]], bool]] = [
    ("learning rate", "lr", [("lr", "lr")], True),
    ("weight decay", "wd", [("weight decay", "weight_decay")], False),
    ("teacher momentum", "m", [("momentum", "teacher_momentum")], False),
]

#: Where a collapsing objective shows first, and long before a checkpoint is extracted: the
#: teacher's output entropy falling towards zero is the whole distribution moving onto one mode.
_COLLAPSE_PANELS: list[tuple[str, str, list[tuple[str, str]], bool]] = [
    (
        "output entropy",
        "nats",
        [("teacher", "teacher_entropy"), ("student", "student_entropy")],
        False,
    ),
    ("teacher-student KL", "nats", [("KL", "kl")], False),
    ("pairs per step", "views", [("pairs", "n_pairs")], False),
    ("voxels per step", "voxels", [("voxels", "n_voxels")], False),
]

_THROUGHPUT_PANELS: list[tuple[str, str, list[tuple[str, str]], bool]] = [
    ("samples/s", "samples/s", [("samples/s", "throughput/samples_per_s")], False),
    ("voxels/s", "voxels/s", [("voxels/s", "throughput/voxels_per_s")], False),
    (
        "data wait",
        "fraction of step",
        [("step stream", "throughput/data_wait_frac")],
        False,
    ),
    ("peak memory", "GiB", [("peak", "throughput/peak_mem_gb")], False),
]

#: Which figures `plot_diagnostics` draws, in order.
DIAGNOSTIC_FIGURES = ("loss", "collapse", "schedules", "throughput", "gradnorm", "spectrum")


def plot_diagnostics(
    run_dirs: list[Path | str],
    out_dir: Path | str,
    *,
    smooth_window: int = 200,
    fmt: str = "png",
    only: tuple[str, ...] = DIAGNOSTIC_FIGURES,
) -> list[Path]:
    """Draw the diagnostic figures for one or more runs into `out_dir`.

    Returns the paths written. A figure whose every panel was empty is not written at all, so
    the returned list is what a caller can report without checking the disk.
    """
    runs = {Path(d).name: Path(d) for d in run_dirs}
    step = {name: read_records(path, "step") for name, path in runs.items()}
    epoch = {name: read_records(path, "epoch") for name, path in runs.items()}
    missing = [n for n, recs in step.items() if not recs and not epoch[n]]
    if missing:
        raise SystemExit(
            f"no metrics stream for {', '.join(missing)}: a run writes metrics/step.jsonl and "
            "metrics/epoch.jsonl, and a plot is a view over them"
        )
    empty = [n for n, recs in step.items() if not recs]
    if empty:
        print(f"[no step stream] {', '.join(empty)}: only the epoch panels will carry it")
    colors = run_colors(list(runs))
    written: list[Path] = []

    if "loss" in only:
        p = _loss_figure(step, epoch, colors, out_dir, smooth_window, fmt)
        written += [p] if p else []
    if "collapse" in only:
        p = _panel_figure(step, colors, _COLLAPSE_PANELS, out_dir, "collapse", smooth_window, fmt)
        written += [p] if p else []
    if "schedules" in only:
        p = _panel_figure(step, colors, _SCHEDULE_PANELS, out_dir, "schedules", smooth_window, fmt)
        written += [p] if p else []
    if "throughput" in only:
        p = _panel_figure(
            step, colors, _THROUGHPUT_PANELS, out_dir, "throughput", smooth_window, fmt
        )
        written += [p] if p else []
    if "gradnorm" in only:
        p = _gradnorm_figure(step, colors, out_dir, fmt)
        written += [p] if p else []
    if "spectrum" in only:
        p = _spectrum_figure(step, colors, out_dir, fmt)
        written += [p] if p else []
    return written


def _draw(ax, xs, ys, *, color, label, window, dashes=()):
    """One series: the raw trace behind, the rolling mean in front.

    Only traces long enough for the window to mean something get the pair; a per-epoch series
    of 100 points is already the trend.
    """
    dash = {"dashes": dashes} if dashes else {}
    if len(ys) > 4 * max(window, 1):
        ax.plot(xs, ys, color=color, linewidth=0.6, alpha=0.25, **dash)
        ax.plot(xs, smooth(ys, window), color=color, linewidth=1.6, label=label, **dash)
    else:
        ax.plot(xs, ys, color=color, linewidth=1.6, marker="o", markersize=2.5, label=label, **dash)


def _draw_keys(ax, step, colors, keys, window):
    """Every (run, key) pair of one panel. Colour separates the runs, dashes the keys."""
    for run, recs in step.items():
        for i, (lab, key) in enumerate(keys):
            xs, ys = series(recs, key)
            if not xs:
                continue
            label = lab if len(step) == 1 else (f"{run} {lab}" if len(keys) > 1 else run)
            dashes = KEY_DASHES[i % len(KEY_DASHES)] if len(keys) > 1 else ()
            _draw(ax, xs, ys, color=colors[run], label=label, window=window, dashes=dashes)


def _finish(ax, title, xlabel, ylabel, logy=False):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, framealpha=0.85)


def _loss_figure(step, epoch, colors, out_dir, window, fmt) -> Path | None:
    """Per-step loss, per-epoch loss, and whatever split the objective recorded."""
    panels = [("loss (step)", "loss", [("total", "loss")], False)]
    present = set().union(*(keys_present(r) for r in step.values())) if step else set()
    title, ylabel, keys, logy = _LOSS_SPLIT[0]
    split = [(lab, key) for lab, key in keys if key in present]
    if split:
        panels.append((title, ylabel, split, logy))
    n = len(panels) + 1
    fig, axes = figure(n)
    for ax, (title, ylabel, keys, logy) in zip(axes, panels, strict=False):
        _draw_keys(ax, step, colors, keys, window)
        _finish(ax, title, "step", ylabel, logy)
    ax = axes[-1]
    for run, recs in epoch.items():
        xs, ys = series(recs, "loss", x="epoch")
        if xs:
            ax.plot(xs, ys, color=colors[run], marker="o", markersize=3, linewidth=1.6, label=run)
    _finish(ax, "loss (epoch mean)", "epoch", "loss")
    return save(fig, out_dir, "loss", fmt)


def _panel_figure(step, colors, panels, out_dir, name, window, fmt) -> Path | None:
    present = set().union(*(keys_present(r) for r in step.values())) if step else set()
    live = [p for p in panels if any(key in present for _, key in p[2])]
    if not live:
        return None
    fig, axes = figure(len(live))
    for ax, (title, ylabel, keys, logy) in zip(axes, live, strict=True):
        _draw_keys(ax, step, colors, keys, window)
        _finish(ax, title, "step", ylabel, logy)
    return save(fig, out_dir, name, fmt)


def _gradnorm_figure(step, colors, out_dir, fmt) -> Path | None:
    """Gradient norm and grad-to-parameter ratio, one panel per group the run recorded.

    Two blocks of panels: `gradnorm/<group>/grad_norm` on top, `gradnorm/<group>/grad_to_param`
    below. The ratio is what says whether a group is learning; a norm alone does not, until the
    weight scale is known.

    Inside a norm panel, a single run is drawn one line per parameter from
    `gradnorm/<group>/param/<name>/grad_norm` where the stream carries it, with the group total
    behind in grey. With several runs the parameter lines would need a second colour axis, so
    only the group totals are drawn, one colour per run.

    The y axis is symlog, not log, and that is the point of the figure. A group can reach the
    stream at exactly 0.0, and a log axis masks every non-positive value: those records would
    be dropped silently and the panel would show only the steps that still had a gradient,
    which reads as a noisy curve rather than as a floor.
    """
    present = set().union(*(keys_present(r) for r in step.values())) if step else set()
    groups = [g for g in families(present, "gradnorm/") if f"gradnorm/{g}/grad_norm" in present]
    if not groups:
        return None
    ratio_groups = [g for g in groups if f"gradnorm/{g}/grad_to_param" in present]
    ncols = 3
    n_norm = -(-len(groups) // ncols) * ncols  # pad the norm block to whole rows
    fig, axes = figure(n_norm + len(ratio_groups), ncols=ncols, height=2.6, width=4.2)
    for ax in axes[len(groups) : n_norm]:
        ax.set_visible(False)

    one_run = len(step) == 1
    for ax, group in zip(axes[: len(groups)], groups, strict=True):
        floor = 1.0
        for run, recs in step.items():
            xs, ys = series(recs, f"gradnorm/{group}/grad_norm")
            if not xs:
                continue
            floor = _floor(floor, ys)
            params = sorted(families(present, f"gradnorm/{group}/param/", depth=1))
            params = [q for q in params if f"gradnorm/{group}/param/{q}/grad_norm" in present]
            if one_run and params:
                ax.plot(xs, ys, color="#888888", linewidth=1.0, alpha=0.6, label="group total")
                for q in params:
                    pxs, pys = series(recs, f"gradnorm/{group}/param/{q}/grad_norm")
                    if pxs:
                        floor = _floor(floor, pys)
                        ax.plot(pxs, pys, linewidth=1.0, label=q)
            else:
                ax.plot(xs, ys, color=colors[run], linewidth=1.2, label=run)
        ax.set_yscale("symlog", linthresh=max(floor, 1e-12))
        _finish(ax, f"{group} grad norm", "step", "L2 norm")

    for ax, group in zip(axes[n_norm:], ratio_groups, strict=True):
        floor = 1.0
        for run, recs in step.items():
            xs, ys = series(recs, f"gradnorm/{group}/grad_to_param")
            if xs:
                floor = _floor(floor, ys)
                ax.plot(xs, ys, color=colors[run], linewidth=1.2, label=run)
        ax.set_yscale("symlog", linthresh=max(floor, 1e-12))
        _finish(ax, f"{group} grad / param", "step", "ratio")
    return save(fig, out_dir, "gradnorm", fmt)


def _floor(floor: float, ys: list[float]) -> float:
    """The smallest non-zero magnitude seen so far: the symlog linear threshold."""
    nonzero = [abs(y) for y in ys if y != 0.0]
    return min(floor, min(nonzero)) if nonzero else floor


def _spectrum_figure(step, colors, out_dir, fmt) -> Path | None:
    """Participation ratio and RankMe per `spectrum/<branch>/<tap>` the run recorded.

    These are the online counterpart of `probe_spectrum`: how many directions of the embedding
    carry variance. A collapsing objective shows here long before a probe is extracted.
    """
    present = set().union(*(keys_present(r) for r in step.values())) if step else set()
    taps = [t for t in families(present, "spectrum/", depth=2)]
    panels = [
        (tap, metric)
        for tap in taps
        for metric in ("participation_ratio", "rankme")
        if f"spectrum/{tap}/{metric}" in present
    ]
    if not panels:
        return None
    fig, axes = figure(len(panels), ncols=2, height=2.8)
    for ax, (tap, metric) in zip(axes, panels, strict=True):
        for run, recs in step.items():
            xs, ys = series(recs, f"spectrum/{tap}/{metric}")
            if xs:
                ax.plot(
                    xs, ys, color=colors[run], linewidth=1.4, marker="o", markersize=2, label=run
                )
        _finish(ax, f"{tap} {metric}", "step", metric.replace("_", " "))
    return save(fig, out_dir, "spectrum", fmt)
