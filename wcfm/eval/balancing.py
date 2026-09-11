"""How the scored population is constructed: an event-level split, then class-balanced pools.

Every detail of the draw is load-bearing: a pool drawn differently makes a new probe number
incomparable with one already recorded, and nothing raises. `tests/test_eval_balancing.py` pins
these against stored index arrays.

- `RandomState`, not `default_rng`. The two are different streams, so moving to the better API
  would silently redraw every pool.
- `int(train_frac * n_events)` truncates, and the split is over events, never pixels. A
  pixel-level split leaks: neighbouring pixels of one track land on both sides, and even an
  untrained backbone scores well on the result.
- Classes are drawn in the order given and the result is shuffled within a class only, so the
  pool stays grouped by class. Concatenation order is part of what a recorded number means.
- A class with fewer than `per_class` candidates contributes all of them rather than sampling
  with replacement, so a pool holds at most `per_class * len(classes)` and rare classes stay
  honestly rare.

Nothing here imports torch: pools are drawn from truth, which is feature-independent, and that
is what lets extraction draw them once for every probe rather than each probe redrawing its own.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def event_split(n_events: int, seed: int, train_frac: float = 0.8) -> np.ndarray:
    """Boolean `[n_events]`, `True` = train. Seeded permutation of events."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_events)
    n_train = int(train_frac * n_events)
    is_train = np.zeros(n_events, dtype=bool)
    is_train[perm[:n_train]] = True
    return is_train


def pixel_split(pixel_event: np.ndarray, seed: int, train_frac: float = 0.8) -> np.ndarray:
    """The per-pixel train mask induced by the event-level split.

    Takes `pixel_event`, the event index of each pixel, rather than a loaded feature object, so
    that this module reads without one.
    """
    pixel_event = np.asarray(pixel_event)
    n_events = int(pixel_event.max()) + 1 if pixel_event.size else 0
    return event_split(n_events, seed, train_frac)[pixel_event]


def balanced_pool(
    candidates: np.ndarray,
    y: np.ndarray,
    classes: Sequence,
    per_class: int,
    seed: int,
) -> np.ndarray:
    """Up to `per_class` indices per class, drawn from `candidates` (seeded).

    Returns positions in the same index space as `candidates`, grouped by class.
    """
    rng = np.random.RandomState(seed)
    candidates = np.asarray(candidates)
    y = np.asarray(y)
    picked = []
    for c in classes:
        ci = candidates[y[candidates] == c]
        if len(ci) > per_class:
            ci = rng.choice(ci, per_class, replace=False)
        else:
            ci = ci.copy()
            rng.shuffle(ci)
        picked.append(ci)
    return np.concatenate(picked) if picked else np.zeros(0, dtype=np.int64)


def pixel_event_index(offsets: np.ndarray) -> np.ndarray:
    """`[N_pix]` event index per pixel, from the CSR `[n_events + 1]` offsets."""
    offsets = np.asarray(offsets)
    counts = np.diff(offsets)
    return np.repeat(np.arange(len(counts), dtype=np.int64), counts)
