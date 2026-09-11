"""Result-file conventions: how an entry is keyed, and what provenance it carries.

Every probe keys its JSON `<run>:<epoch tag>:<source>`, so result files from different metrics,
epochs and trainings merge into one table with no bookkeeping.

The label comes from the store's layout -- a directory per checkpoint under a run's
`features/` -- rather than from a filename, so run identity is not a property of where somebody
put a file. The header carries `eval_set_id` and the event-key hash, which is what turns "are
these two numbers comparable" into something `check_comparability` can refuse.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .features import Features, raw_charge_kind

__all__ = ["run_header", "run_label", "write_json"]


def run_label(store_root: Path | str, source: str) -> str:
    """`<run>:<epoch tag>:<source>`, e.g. `hybrid_baseline_mixed_b100:ep100:student`.

    Derived from the store's layout, `<run>/features/epoch<N>/`. The epoch directory is named
    `epochN` and the tag is `epN`, because that is the key every recorded result file and every
    `merge` table is written against.
    """
    path = Path(store_root)
    name = path.name
    tag = f"ep{name[len('epoch'):]}" if name.startswith("epoch") else name
    # `<run>/features/<epochN>` -- parents[0] is `features`, parents[1] is the run.
    run = path.parents[1].name if len(path.parents) >= 2 else path.parent.name
    return f"{run}:{tag}:{source}"


def write_json(results: dict, out_path: Path | str) -> None:
    """Write results incrementally, so a long multi-checkpoint run is crash-safe.

    Atomic: written to a temp beside the target and renamed over it, because a plain
    `open(path, "w")` truncates first and leaves the file unreadable for as long as the dump
    takes. Epoch jobs run concurrently and each one globs the whole directory, so a reader
    landing in that window would get a decode error on a file that is perfectly good a moment
    later. `os.replace` is atomic within a filesystem, and the temp is in the same directory to
    keep it so.
    """
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(out.parent), prefix=f".{out.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(results, f, indent=2, sort_keys=True)
        os.replace(tmp, out)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def run_header(fx: Features, seed: int, per_class: int) -> dict[str, Any]:
    """The provenance block recorded next to every metric in the output JSON.

    A merged table is refused on `eval_set_id`, `event_key_hash` and `rows`: two numbers scored
    on different event sets are not comparable, and without these three nothing records enough
    to notice.
    """
    prov = fx.provenance
    return {
        "features_dir": str(fx.path),
        "feature_source": fx.source,
        "tap": fx.tap,
        "n_events": fx.n_events,
        "n_pixels": fx.n_pixels,
        "feature_dim": int(fx.feat.shape[1]),
        "truth_channels": sorted(fx.truth),
        "seed": seed,
        "pool_per_class": per_class,
        "raw_charge_transform": raw_charge_kind(fx),
        "rows": prov.get("rows", "all"),
        "eval_set_id": prov.get("eval_set_id", ""),
        "event_key_hash": prov.get("event_key_hash", ""),
        "sample": prov.get("sample", "in-sample"),
        "checkpoint_sha256": prov.get("checkpoint_sha256", ""),
        "provenance": prov,
    }
