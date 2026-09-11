"""Offline gradient probe: are two terms pulling the backbone in the same direction?

Per-term gradient norms and pairwise cosines on the backbone parameters, over a fixed set of
events. `TermGrad` is the in-loop counterpart, on a cadence during training; this one runs on a
finished checkpoint and is the reason a comparison between two runs can be made after the fact.

The backbone is the thing the terms compete over. A term's own head is not contested, so a
cosine involving it would measure nothing. The cosine is the readable quantity: two terms with
large gradients that cancel look healthy in every per-term loss curve and train nothing, which
is the failure a loss plot hides best. A negative cosine says the objective is fighting itself.

It runs on the GPU, inside `wcfm eval extract`, because it needs the model rather than a feature
file. Everything under `wcfm/eval/probes/` is CPU-only and reads features off disk, which is the
invariant that lets the DAG put every probe on a CPU slot, so this lives beside extraction
instead of among them.

`wcfm/eval/` is a framework package and may not import `wcfm.model`. Per-term gradients are
model vocabulary, so this reaches them through the optional `term_gradients` hook with
`getattr`, the way extraction reaches `inference_step` and the metrics layer reaches
`grad_taxonomy`. The arithmetic here -- norms, cosines, accumulation -- knows only that it was
handed a dict of flat vectors.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["GRADIENTS_FILE", "gradient_report", "term_gradients"]

GRADIENTS_FILE = "gradients.json"

#: Batches to accumulate over. The plan says "a few hundred events"; at the production batch
#: size that is a handful of batches, and the number is recorded in the output rather than
#: assumed, because a cosine over 2 batches and one over 20 are not the same measurement.
DEFAULT_BATCHES = 8


def term_gradients(module, batch, *, step: int = 0):
    """Per-term flat gradient vectors, via the optional hook."""
    hook = getattr(module, "term_gradients", None)
    if hook is None:
        raise TypeError(
            f"{type(module).__name__} does not implement `term_gradients(batch)`, so its "
            "objective cannot be decomposed into per-term gradients. The hook is optional on "
            "the training contract."
        )
    return hook(batch, step=step)


def gradient_report(
    module,
    loader,
    *,
    device: str = "cuda",
    max_batches: int = DEFAULT_BATCHES,
    step: int = 0,
) -> dict[str, Any]:
    """Accumulate per-term gradients over `max_batches` and report norms and pairwise cosines.

    Gradients are **summed** across batches before the cosine is taken, not averaged per batch
    and then combined: the quantity of interest is the direction the objective actually pulls
    over the event set, and a mean of per-batch cosines answers a different, noisier question.
    The per-batch spread is reported alongside so that distinction is visible rather than
    assumed.
    """
    import torch

    totals: dict[str, torch.Tensor] = {}
    per_batch: dict[str, list[float]] = {}
    n_batches = n_events = 0

    for batch in loader:
        if n_batches >= max_batches:
            break
        batch = batch.to(device)
        grads = term_gradients(module, batch, step=step)
        if not grads:
            continue
        for name, g in grads.items():
            g = g.detach().float()
            totals[name] = g.clone() if name not in totals else totals[name] + g
            per_batch.setdefault(name, []).append(float(g.norm()))
        n_batches += 1
        n_events += int(getattr(batch, "batch_size", 0) or 0)

    if not totals:
        return {
            "error": "no term produced a gradient over the sampled batches",
            "n_batches": n_batches,
        }

    names = sorted(totals)
    norms = {n: float(totals[n].norm()) for n in names}
    cosines: dict[str, float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            na, nb = norms[a], norms[b]
            cosines[f"{a}|{b}"] = (
                float(torch.dot(totals[a], totals[b]) / (na * nb)) if na > 0 and nb > 0 else 0.0
            )

    import statistics

    return {
        "terms": names,
        "n_batches": n_batches,
        "n_events": n_events,
        "n_parameters": int(next(iter(totals.values())).numel()),
        "scope": "backbone",
        "norm": norms,
        # Per-batch norm spread, so a cosine taken over the summed gradient can be read against
        # how much the individual batches varied.
        "norm_per_batch_mean": {n: statistics.mean(v) for n, v in per_batch.items()},
        "norm_per_batch_stdev": {
            n: (statistics.stdev(v) if len(v) > 1 else 0.0) for n, v in per_batch.items()
        },
        # The headline. Negative means the two terms are pulling the backbone apart, which no
        # per-term loss curve shows.
        "cosine": cosines,
        "min_cosine": min(cosines.values()) if cosines else None,
    }


def write_gradients(store_root: Path | str, report: dict) -> Path:
    path = Path(store_root) / GRADIENTS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)
    return path
