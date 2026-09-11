"""The evaluation format: what an extraction writes and what a probe reads.

Five properties hold, and each of them is what keeps a probe number meaningful:

- Feature blocks are uncompressed fp16 `.npy`, one array per `(source, tap)`, so a probe that
  wants one column of one tap pays a page-cache read and nothing else. Compression buys about
  1.08x on these arrays, costs a minute of single-threaded inflate per read, and forecloses
  mmap.
- Truth and geometry are written once per eval set. A checkpoint's own directory holds only
  what depends on the checkpoint, which is why scoring 40 epochs of one run does not write 40
  byte-identical copies of the truth columns.
- An eval set has an `eval_set_id` and a SHA-256 over its sorted event keys. The hash is what
  makes "these two files scored the same events" checkable rather than assumed; pinning the set
  by a convention such as `max_images // batch_size` makes `batch_size` silently change which
  events were scored.
- A differing `eval_set_id` or key hash is an error, not a warning. Two numbers over different
  events are not two measurements of the same thing, and a table printing them side by side is
  wrong rather than noisy. Softer axes -- the charge transform, the pool size -- stay warnings,
  because the feature columns remain meaningful.
- The row space is recorded. Only the pooled rows are written by default, and a pooled block is
  indistinguishable from a full one once it is an `[N, D]` array on disk, so `Provenance.rows`
  states which it is, `pools.npz` carries the `row_index` that joins either back to truth, and
  a strided tap carries its own coordinates. A probe indexing truth through `row_index` is
  correct without knowing which.

Nothing in this module imports torch: a probe host, a merge step and a test all need to read
these files, and only the extractor needs a model.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = 2

#: Truth and geometry live here, once, beside the features of every checkpoint scored on it.
EVAL_SET_FILE = "eval_set.json"
PROVENANCE_FILE = "provenance.json"
POOLS_FILE = "pools.npz"


def event_key_hash(keys: list[str] | np.ndarray) -> str:
    """SHA-256 over the sorted event keys, newline-joined.

    Sorted so that two extractions that visited the same events in a different order -- a
    different `num_workers`, a reshuffled shard set -- still hash equal. Order is a property of
    the reader, and membership is what makes two results comparable.
    """
    items = sorted(str(k) for k in np.asarray(keys).ravel().tolist())
    h = hashlib.sha256()
    for k in items:
        h.update(k.encode())
        h.update(b"\n")
    return h.hexdigest()


@dataclass
class EvalSet:
    """The events a set of results was scored on, and the truth for them.

    `id` names the set and `key_hash` proves it: a set built by a different filter that reuses
    the name is caught by the hash.
    """

    id: str
    key_hash: str
    n_events: int
    sample: str = "in-sample"
    """`in-sample` until a production ships a held-out split; carried as a column by merge so a
    result never implies a validation set it did not have."""
    truth_arrays: list[str] = field(default_factory=list)
    format_version: int = FORMAT_VERSION

    @classmethod
    def create(
        cls,
        root: Path | str,
        *,
        id: str,
        event_keys: list[str] | np.ndarray,
        truth: dict[str, np.ndarray],
        geometry: dict[str, np.ndarray] | None = None,
        sample: str = "in-sample",
    ) -> EvalSet:
        """Write truth and geometry once. Returns the descriptor the checkpoints refer to."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        arrays = {**truth, **(geometry or {})}
        for name, arr in arrays.items():
            _write_npy(root / f"{name}.npy", np.asarray(arr))
        np.save(root / "event_keys.npy", np.asarray(event_keys).astype("U"))
        me = cls(
            id=id,
            key_hash=event_key_hash(event_keys),
            n_events=int(len(event_keys)),
            sample=sample,
            truth_arrays=sorted(arrays),
        )
        (root / EVAL_SET_FILE).write_text(json.dumps(asdict(me), indent=2) + "\n")
        return me

    @classmethod
    def load(cls, root: Path | str) -> EvalSet:
        path = Path(root) / EVAL_SET_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist: an eval set is created once by `wcfm eval extract` "
                "and shared by every checkpoint scored on it"
            )
        return cls(**json.loads(path.read_text()))

    def read(self, root: Path | str, name: str) -> np.ndarray:
        """Memory-map one truth or geometry column."""
        if name not in self.truth_arrays:
            raise KeyError(
                f"eval set {self.id!r} has no array {name!r}; it has {self.truth_arrays}"
            )
        return np.load(Path(root) / f"{name}.npy", mmap_mode="r")

    def verify(self, root: Path | str) -> None:
        """Re-hash the stored keys. Catches a set edited in place after results were written."""
        keys = np.load(Path(root) / "event_keys.npy")
        actual = event_key_hash(keys)
        if actual != self.key_hash:
            raise ValueError(
                f"eval set {self.id!r} at {root} does not match its recorded key hash "
                f"({actual[:12]} != {self.key_hash[:12]}): the event list changed after the set "
                "was created, so results written against it are not comparable to new ones"
            )


@dataclass
class Provenance:
    """What a probe needs in order to say whether two numbers may be compared.

    Extraction wall time and throughput are here because the most expensive step in the
    pipeline was uninstrumented: a run that got slower had no record saying so.
    """

    eval_set_id: str
    event_key_hash: str
    checkpoint: str
    checkpoint_sha256: str
    sources: list[str]
    """Every branch written into this directory, e.g. `["student", "teacher"]`.

    Plural because one pass writes both. The store indexes features by `(source, tap)` and
    holds one provenance file, so a scalar `source` would name one branch while the directory
    held two.
    """
    taps: list[str]
    rows: str = "all"
    """The row space every feature array in this directory is indexed by.

    `all`: row `i` is pixel `i` of the eval set, so a truth column joins positionally.
    `pooled`: only the rows in the union of the pools were written, and row `i` is eval-set
    pixel `pools["row_index"][i]`. Recorded rather than inferred, because the two are
    indistinguishable from an `[N, D]` array: a probe that joins a pooled block positionally
    against a truth column gets a plausible answer that is entirely wrong.
    """
    tap_strides: dict[str, int] = field(default_factory=dict)
    """Each written tap's stride relative to full resolution. Per-pixel truth is positionally
    joinable only at stride 1; a strided tap has its own coordinates, in `coords__<tap>.npy`.
    """
    git_sha: str = ""
    max_images: int = -1
    batch_size: int = -1
    seed: int = -1
    charge_transform: str = ""
    charge_transform_params: dict[str, Any] = field(default_factory=dict)
    """The transform's actual parameters, not just its printed form.

    `charge_transform` is a display string and is what `check_comparability` groups on. The
    raw-charge baseline needs the numbers, because it reproduces the exact input the backbone
    was fed (`FeatureLogTransform(min_val, max_val)`), and parsing them back out of
    `"log[0.7,150.0]"` would be one float repr away from a silently different baseline. Empty
    means the run recorded no transform, and the baseline says so rather than guessing one.
    """
    apa: int = -1
    view: str = ""
    """Which wire plane these pixels are. Only `probe_vertex` needs them, since projecting the
    true 3-D vertex into the image is undefined without them, but they are a property of the
    extraction, so they are recorded here rather than passed to that probe by hand."""
    pool_spec: dict[str, Any] = field(default_factory=dict)
    """Every constant the pool draws consumed, from `wcfm.eval.pools.PoolSpec`. A probe checks
    the ones it is written against before scoring, so a changed constant fails loudly instead of
    scoring the stored pool under a name that no longer describes it."""
    pool_per_class: int = 0
    pool_seed: int = -1
    extract_seconds: float = 0.0
    events_per_second: float = 0.0
    format_version: int = FORMAT_VERSION
    extra: dict[str, Any] = field(default_factory=dict)


class FeatureStore:
    """One checkpoint's extraction output: features per `(source, tap)`, pools, provenance.

    Only what depends on the checkpoint lives here. Truth is in the :class:`EvalSet` beside it,
    which is the whole point of the split.
    """

    def __init__(self, root: Path | str):
        self.root = Path(root)

    # ------------------------------------------------------------------ writing

    def write_features(self, source: str, tap: str, array: np.ndarray) -> Path:
        """Write one `[N, D]` feature block as uncompressed fp16.

        fp16 because these are activations read for linear probes and neighbour searches, where
        the third decimal does not change a score, and it halves a ~9 GB resident set.
        """
        arr = np.asarray(array)
        if arr.ndim != 2:
            raise ValueError(f"features for {source}/{tap} are {arr.ndim}-D; expected [N, D]")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{_slug(source)}__{_slug(tap)}.npy"
        _write_npy(path, arr.astype(np.float16, copy=False))
        return path

    def write_pools(self, *, row_index: np.ndarray, **pools: np.ndarray) -> Path:
        """The balanced pools, drawn at extraction time.

        Drawing here rather than in each probe is what makes the population the same one for
        every probe, instead of each probe redrawing its own and the suite comparing
        measurements taken over different samples.

        `row_index` is mandatory and is the join back to truth: written row `i` is eval-set
        pixel `row_index[i]`. Under `rows="all"` it is `arange(n_pixels)`, so a reader never
        needs to branch on the row space: it indexes truth through `row_index` unconditionally
        and is correct either way. Every other array here holds row-space positions, so
        `feat[pool]` and `truth[row_index[pool]]` line up.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / POOLS_FILE
        arrays = {k: np.asarray(v) for k, v in pools.items()}
        arrays["row_index"] = np.asarray(row_index)
        np.savez(path, **arrays)
        return path

    def write_event_means(self, source: str, tap: str, array: np.ndarray) -> Path:
        """One mean vector per event, `[n_events, D]`.

        `probe_event` scores events rather than pixels, so this is the only thing it reads, and
        pooling here while the full feature block is still in memory is what keeps its ~20M
        sampled pixels out of the row space under `rows="pooled"`. Written fp32 rather than
        fp16 because a mean over up to 2,000 rows has already thrown away the variance fp16 was
        cheap for, and the array is `n_events` tall rather than `n_pixels`.

        NaN marks an event with no sampled pixels; the probe drops those rather than reading
        the NaN as a vector at the origin.
        """
        arr = np.asarray(array)
        if arr.ndim != 2:
            raise ValueError(f"event means for {source}/{tap} are {arr.ndim}-D; expected [N, D]")
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{_slug(source)}__{_slug(tap)}.events.npy"
        _write_npy(path, arr.astype(np.float32, copy=False))
        return path

    def event_means(self, source: str, tap: str) -> np.ndarray:
        path = self.root / f"{_slug(source)}__{_slug(tap)}.events.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"no per-event means for source={source!r} tap={tap!r} at {path}. They are "
                "written for stride-1 taps when the event pool is non-empty; an extraction "
                "with no per-pixel truth draws no pools at all."
            )
        return np.load(path, mmap_mode="r")

    def write_coords(self, tap: str, coords: np.ndarray) -> Path:
        """A strided tap's own coordinates, `[N_tap, 2]` in that tap's units.

        Only stride-1 taps are positionally joinable to per-pixel truth. A strided tap has fewer
        rows sitting on a coarser grid, so it carries the coordinates a reader would otherwise
        have to guess. Named `<tap>.coords.npy` rather than through the `__` separator so that
        `available` cannot mistake one for a feature block.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / f"{_slug(tap)}.coords.npy"
        _write_npy(path, np.asarray(coords).astype(np.int32, copy=False))
        return path

    def coords(self, tap: str) -> np.ndarray:
        path = self.root / f"{_slug(tap)}.coords.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"no coordinates for tap {tap!r} at {path}; stride-1 taps do not write them "
                "because their rows are the eval set's pixels"
            )
        return np.load(path, mmap_mode="r")

    def write_provenance(self, prov: Provenance) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / PROVENANCE_FILE
        path.write_text(json.dumps(asdict(prov), indent=2) + "\n")
        return path

    # ------------------------------------------------------------------ reading

    def provenance(self) -> Provenance:
        path = self.root / PROVENANCE_FILE
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist: either extraction did not finish, or this directory "
                "was written by an older layout"
            )
        d = json.loads(path.read_text())
        if int(d.get("format_version", 1)) != FORMAT_VERSION:
            raise ValueError(
                f"{path} is format v{d.get('format_version')}, this reader is v{FORMAT_VERSION}; "
                "re-extract rather than reading it as though the layouts agreed"
            )
        return Provenance(**d)

    def features(self, source: str, tap: str) -> np.ndarray:
        """Memory-map one feature block. No decompression, no copy."""
        path = self.root / f"{_slug(source)}__{_slug(tap)}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"no features for source={source!r} tap={tap!r} at {path}; this checkpoint has "
                f"{sorted(self.available())}"
            )
        return np.load(path, mmap_mode="r")

    def available(self) -> set[tuple[str, str]]:
        out = set()
        for p in self.root.glob("*__*.npy"):
            if p.name.endswith((".coords.npy", ".events.npy")):
                continue
            source, _, tap = p.stem.partition("__")
            out.add((source, tap))
        return out

    def pools(self) -> dict[str, np.ndarray]:
        path = self.root / POOLS_FILE
        if not path.exists():
            return {}
        with np.load(path) as z:
            return {k: z[k] for k in z.files}


def check_comparability(entries: dict[str, Provenance]) -> list[str]:
    """Refuse a table whose rows are not measurements of the same thing.

    Returns the warnings, and raises on the two axes that make numbers incomparable rather than
    merely noisy: a differing `eval_set_id` or a differing event-key hash. Warning about those
    instead prints a line above numbers nobody can interpret, and the line scrolls away.

    The softer axes stay warnings: a mixed charge transform invalidates the raw and delta
    columns but not the feature columns, and a mixed pool size means the balanced scores answer
    slightly different questions.
    """
    if not entries:
        return []
    by_set: dict[str, list[str]] = {}
    by_hash: dict[str, list[str]] = {}
    for label, p in entries.items():
        by_set.setdefault(p.eval_set_id, []).append(label)
        by_hash.setdefault(p.event_key_hash, []).append(label)

    if len(by_set) > 1:
        raise ValueError(
            "these results were scored on different eval sets and cannot be tabulated together: "
            + "; ".join(f"{k!r}: {sorted(v)}" for k, v in sorted(by_set.items()))
        )
    if len(by_hash) > 1:
        raise ValueError(
            "these results share an eval_set_id but not their event keys, so the set changed "
            "under them: "
            + "; ".join(f"{k[:12]}: {sorted(v)}" for k, v in sorted(by_hash.items()))
        )

    warnings: list[str] = []
    for field_name, msg in (
        ("charge_transform", "the raw and delta columns are NOT comparable across these groups"),
        ("pool_per_class", "a balanced score over different pool sizes is a different measurement"),
    ):
        groups: dict[Any, list[str]] = {}
        for label, p in entries.items():
            v = getattr(p, field_name)
            if v:
                groups.setdefault(v, []).append(label)
        if len(groups) > 1:
            warnings.append(
                f"mixed {field_name} in this table -- {msg}: "
                + "; ".join(
                    f"{k!r}: {len(v)} run(s), e.g. {sorted(v)[0]}" for k, v in groups.items()
                )
            )
    return warnings


def _slug(name: str) -> str:
    """Filenames are the index, so a tap or source name may not smuggle a separator into one."""
    if not name or "__" in name or "/" in name or name != name.strip():
        raise ValueError(
            f"{name!r} is not usable in a filename: names must be non-empty, trimmed, and "
            "contain neither '__' (the source/tap separator) nor '/'"
        )
    return name


def _write_npy(path: Path, arr: np.ndarray) -> None:
    """Write through a temp file and rename.

    The DAG's PRE script decides whether to re-extract by reading these files. A reader must
    never see a half-written one, and a rename within a directory is atomic -- which is also
    why the old "is the file older than the checkpoint, and has it settled" heuristic is gone.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Through a file handle: given a *path* without a `.npy` suffix numpy appends one, so the
    # temp file would be written next to the name we then try to rename.
    with open(tmp, "wb") as fh:
        np.save(fh, arr)
    tmp.replace(path)
