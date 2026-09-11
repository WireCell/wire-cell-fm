"""Argument handling shared by every probe, and the runner that scores several in one process.

Several probes score in one process so that `load_features` builds the truth subset once per
store rather than once per probe. Feature blocks are mmapped, so a second reader pays page-cache
hits, but the subset is real work and it is shared here.

Every stage runs even if an earlier one fails, and the exit status is non-zero if any did. A
sweep should not lose five completed measurements because the sixth aborted, nor report success
when it is missing metrics.

A probe takes no argument that changes the population it scores. `pool_per_class`,
`max_queries`, `val_pixels`, `max_pixels_per_class` and `vertex_t0_ticks` live on the
extraction, are recorded in provenance, and a probe checks them rather than accepting them, so
that `pools.npz` and `PoolSpec` stay the only definition of what was measured. What is left here
is the arguments that do not change the population: which branch, which tap, which device.
"""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

__all__ = ["STAGES", "probe_argv", "run_stages"]

#: stage -> (module, function, output filename template, title)
STAGES: dict[str, tuple[str, str, str]] = {
    "pid": ("probe_pid", "pid_{tag}.json", "particle type, trained head"),
    "knn": ("probe_knn_pid", "pixelknn_{tag}.json", "particle type, untrained k-NN"),
    "overlap": ("probe_overlap", "overlap_{tag}.json", "is a pixel's charge shared?"),
    "instance": ("probe_instance", "instance_{tag}.json", "do neighbours share a particle?"),
    "vertex": ("probe_vertex", "vertex_{tag}.json", "is a pixel near the interaction point?"),
    "event": ("probe_event", "event_{tag}.json", "interaction flavor, pooled k-NN"),
    "spectrum": ("probe_spectrum", "spectrum_{tag}.json", "how many directions carry signal?"),
}

DEFAULT_STAGES = "pid,knn,overlap,instance,vertex,event,spectrum"


def add_common(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The arguments every probe takes, and only those that cannot change the population."""
    ap.add_argument(
        "features",
        nargs="+",
        help="feature store directory/ies, i.e. <run>/features/epoch<N>",
    )
    ap.add_argument("--source", default="student", help="which branch to probe")
    ap.add_argument("--tap", default="out", help="which tap (default: the final map)")
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed for the heads and for any tie-break; the POPULATION's seed is the "
        "extraction's and is recorded in provenance (default: 0)",
    )
    ap.add_argument("--device", default="cpu", help="torch device for the heads (default: cpu)")
    return ap


def probe_argv(name: str, argv: list[str] | None, *, default_out: str, extra=None):
    """Parse one probe's arguments. `extra` adds probe-specific, population-neutral flags."""
    ap = argparse.ArgumentParser(prog=f"wcfm eval probe {name}")
    add_common(ap)
    ap.add_argument("--out", default=default_out, help="output JSON path")
    if extra is not None:
        extra(ap)
    return ap.parse_args(argv)


def _tag_of(store_root: Path) -> str:
    name = Path(store_root).name
    return f"ep{name[len('epoch'):]}" if name.startswith("epoch") else name


def run_stages(
    stores: list[str],
    *,
    stages: str = DEFAULT_STAGES,
    out_dir: str = ".",
    source: str = "student",
    tap: str = "out",
    seed: int = 0,
    device: str = "cpu",
) -> int:
    """Score every named stage against every store. Returns a process exit status.

    Non-zero if any stage raised, and the traceback is printed where it happened rather than
    collected at the end -- a stage that failed 40 minutes in should say so at minute 40.
    """
    import importlib

    wanted = [s for s in stages.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in STAGES]
    if unknown:
        raise SystemExit(f"unknown probe stage(s) {unknown}; known: {sorted(STAGES)}")

    failed: list[str] = []
    for store in stores:
        store_root = Path(store)
        tag = _tag_of(store_root)
        for stage in wanted:
            module_name, out_template, title = STAGES[stage]
            out = Path(out_dir) / out_template.format(tag=tag)
            print(
                f"\n{'=' * 78}\n== {module_name} ({title})\n"
                f"== {store_root} -> {out}\n{'=' * 78}",
                flush=True,
            )
            started = time.time()
            try:
                mod = importlib.import_module(f"{__package__}.{module_name}")
                argv = [
                    str(store_root),
                    f"--out={out}",
                    f"--source={source}",
                    f"--tap={tap}",
                    f"--seed={seed}",
                    f"--device={device}",
                ]
                mod.main(argv)
            except (Exception, SystemExit):  # noqa: BLE001 -- one stage must not end the job
                # `SystemExit` is caught deliberately and is NOT redundant with `Exception`:
                # it does not derive from it. Every probe reports a missing truth tier with
                # `fx.require`, which raises `SystemExit` -- so without this an extraction
                # lacking `pixel_energyfrac` would take the whole job down at the overlap
                # stage and discard the five results that had already been written, which is
                # the exact failure this runner exists to prevent.
                traceback.print_exc()
                failed.append(f"{stage}:{tag}")
            print(f"[{time.time() - started:.0f}s total for {stage}]", flush=True)

    if failed:
        print(f"\nFAILED stages: {failed}")
        return 1
    return 0
