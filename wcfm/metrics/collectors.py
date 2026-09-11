"""Generic collectors: they know nothing about DINO, or about any model.

`Spectrum`, `GradNorm`, `Throughput`, `ArrayDump` and `TermGrad`. Each is pointed at
observables whose names the model chose -- `"student/dec_full"` is a string, and nothing here
parses it -- so the same five cover a model this framework has never seen.

The eigendecomposition runs online and the covariance it runs on is not kept. It is cheap to
compute and expensive to store: a stream carrying a D x D matrix per logged step is most of
what a history file weighs. `ArrayDump` is the one collector that writes matrices, and it is
configured at a very low cadence for that reason.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import Tensor

from wcfm.metrics.base import Cadence, Collector, Reduction, StepRecord

log = logging.getLogger(__name__)

__all__ = ["ArrayDump", "GradNorm", "Spectrum", "TermGrad", "Throughput"]


class Throughput(Collector):
    """Samples/s, voxels/s, the data-wait fraction, and peak memory.

    Peak memory is read and never reset. Calling `reset_peak_memory_stats()` on every step
    redefines "peak" as "peak since the last step", so the number that should catch a run
    approaching the card's limit reports a typical step instead.
    """

    needs = frozenset()
    reduce = {"samples_per_s": "sum", "voxels_per_s": "sum", "peak_mem_gb": "max"}

    def __init__(self, cadence: Cadence = "step"):
        self.cadence = cadence

    def compute(self, rec: StepRecord) -> dict[str, float | np.ndarray]:
        out: dict[str, float | np.ndarray] = {}
        step_time = rec.timing.get("step", 0.0)
        if step_time > 0.0:
            if (n := rec.scalars.get("n_samples")) is not None:
                out["samples_per_s"] = float(n) / step_time
            if (v := rec.scalars.get("n_voxels")) is not None:
                out["voxels_per_s"] = float(v) / step_time
        total = rec.timing.get("epoch_elapsed", 0.0)
        if total > 0.0:
            out["data_wait_frac"] = rec.timing.get("data_wait", 0.0) / total
        if torch.cuda.is_available():
            out["peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1024**3
        return out


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """`(sum L)^2 / sum L^2`.

    The clip and the `denom > 0` fallback are load-bearing: a covariance over a few hundred
    rows in a 128-dimensional space is near-singular, and `eigh` returns small negative
    eigenvalues for it as a matter of course. Dropping either turns a rounding artefact into
    a NaN in the stream.
    """
    v = eigenvalues.clip(0)
    denom = float((v**2).sum())
    return float(v.sum() ** 2 / denom) if denom > 0 else 1.0


def rankme(eigenvalues: np.ndarray) -> float:
    """`exp(-sum p log p)` over the normalised spectrum: the entropy of the eigenvalue
    distribution, exponentiated, so it reads on the same `[1, D]` scale as the participation
    ratio and can be plotted against it.

    The two disagree in a useful way. The participation ratio is dominated by the largest
    eigenvalue, so it falls the moment one direction runs away; RankMe weighs the tail, so it
    falls when many small directions collapse. A run losing rank slowly moves RankMe first.
    """
    v = eigenvalues.clip(0)
    total = float(v.sum())
    if total <= 0:
        return 1.0
    p = v / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


class Spectrum(Collector):
    """Second-order structure of a feature matrix: is the representation collapsing?

    Emits, per observable it is pointed at: the participation ratio and RankMe, the top-k
    eigenvalues, the condition number, order statistics of the per-dimension variance and of
    the row norms, and the off-diagonal fraction of the covariance.

    `reduce` is empty: these are statistics over one rank's shard, and rank 0's are the ones
    written. So every emission carries `n_rows`. Without it the number is uninterpretable --
    a participation ratio over 200 rows and one over 65 000 are not comparable, and nothing
    downstream would say so.

    The row sample is seeded from the step, so two runs at the same config see the same rows
    at the same step and their spectra can be compared. An unseeded sample makes every
    comparison noisy in a way that looks like a real difference.
    """

    needs = frozenset()
    reduce: dict[str, str] = {}

    def __init__(
        self,
        cadence: Cadence = 500,
        max_rows: int = 65536,
        top_k: int = 4,
        keys: list[str] | None = None,
    ):
        self.cadence = cadence
        self.max_rows = int(max_rows)
        self.top_k = int(top_k)
        # None means "every observable handed to me". Naming keys is how a run narrows an
        # expensive collector to the one tap it is actually watching.
        self.keys = list(keys) if keys else None

    def compute(self, rec: StepRecord) -> dict[str, float | np.ndarray]:
        out: dict[str, float | np.ndarray] = {}
        for name, tensor in rec.observables.items():
            if self.keys is not None and name not in self.keys:
                continue
            out.update({f"{name}/{k}": v for k, v in self._one(name, tensor, rec.step).items()})
        return out

    def _one(self, name: str, tensor: Tensor, step: int) -> dict[str, float | np.ndarray]:
        rows = _as_rows(tensor)
        if rows is None:
            log.debug("Spectrum skipped %s: not a 2-D feature matrix", name)
            return {}
        rows = _subsample(rows, self.max_rows, seed=step)
        n_rows = int(rows.shape[0])
        if n_rows < 2:
            # `torch.cov` needs two rows. Returning early on a data-dependent test is safe
            # here only because `compute` performs no collective and `reduce` is empty: in
            # front of either, a rank that took this branch would hang the job. The return
            # is an empty dict rather than a partial row.
            return {}

        rows = rows.detach().float()
        cov = torch.cov(rows.T)
        eigenvalues = torch.linalg.eigvalsh(cov).cpu().numpy()  # ascending, real
        variances = torch.diagonal(cov)
        norms = rows.norm(dim=-1)

        frobenius_sq = float((cov**2).sum())
        diagonal_sq = float((variances**2).sum())
        positive = eigenvalues.clip(0)
        top = positive[::-1][: self.top_k]

        out: dict[str, float | np.ndarray] = {
            "n_rows": float(n_rows),
            "dim": float(cov.shape[0]),
            "participation_ratio": participation_ratio(eigenvalues),
            "rankme": rankme(eigenvalues),
            "eig_top": np.asarray(top, dtype=float),
            # max/min over the positive spectrum. A near-singular covariance has a smallest
            # eigenvalue at the level of fp noise, so the guard is what keeps this from
            # dividing by zero and emitting an inf the writer would then drop.
            "condition_number": float(positive.max() / positive.min())
            if positive.min() > 0
            else float("nan"),
            "off_diagonal_frac": (frobenius_sq - diagonal_sq) / frobenius_sq
            if frobenius_sq > 0
            else 0.0,
        }
        out.update(_order_stats("var", variances))
        out.update(_order_stats("norm", norms))
        return out


class GradNorm(Collector):
    """Per-module gradient norm and grad-to-parameter ratio, over a declared taxonomy.

    Biases and the parameters of norm modules are excluded. Their gradients live on a
    different scale from a convolution's, so mixing them into a group makes the group's number
    track the bias count instead of the depth. `include_bias_and_norm=True` turns the filter
    off for anyone who wants the raw picture.

    The taxonomy comes from the module through `getattr`, so it stays out of the protocol: a
    module may define `grad_taxonomy() -> dict[str, tuple[str, ...]]` mapping a group name to
    parameter-name prefixes. Without one, parameters are grouped by their first path token.

    `reduce` is empty, and that is a claim worth checking. These gradients are read from
    `.grad` after backward, so DDP's own all-reduce has already run and every rank holds the
    same values -- including for a parameter skipped this iteration, which
    `find_unused_parameters` marks ready with a zero gradient rather than leaving out of the
    reduction. Declaring `"mean"` here would buy an extra collective per firing for a value
    the ranks already agree on.

    One caveat the numbers carry: the engine collects after the optimizer step, and
    `clip_gradients` rescales in place, so with `optim.clip_grad_norm > 0` these are post-clip
    norms. The pre-clip total is the engine's own `grad_norm` column, which `clip_gradients`
    returns.
    """

    needs = frozenset({"grads"})
    reduce: dict[str, str] = {}

    def __init__(
        self,
        cadence: Cadence = 100,
        dump_per_parameter_every: int = 0,
        include_bias_and_norm: bool = False,
    ):
        self.cadence = cadence
        # A separate, longer cadence for the per-parameter dump: the per-group number is
        # cheap enough to watch continuously and the dump is not.
        self.dump_per_parameter_every = int(dump_per_parameter_every)
        self.include_bias_and_norm = bool(include_bias_and_norm)

    def compute(self, rec: StepRecord) -> dict[str, float | np.ndarray]:
        taxonomy = rec.taxonomy
        groups: dict[str, list[tuple[str, Tensor, Tensor]]] = {}
        per_parameter: dict[str, float] = {}
        dumping = self.dump_per_parameter_every > 0 and (
            rec.step % self.dump_per_parameter_every == 0
        )

        for name, param in rec.named_parameters():
            if param.grad is None:
                continue
            if not self.include_bias_and_norm and _is_bias_or_norm(name):
                continue
            group = _group_of(name, taxonomy)
            groups.setdefault(group, []).append((name, param.detach(), param.grad.detach()))
            if dumping:
                per_parameter[name] = float(param.grad.detach().norm())

        out: dict[str, float | np.ndarray] = {}
        for group, entries in sorted(groups.items()):
            grad_sq = sum(float((g**2).sum()) for _, _, g in entries)
            param_sq = sum(float((p**2).sum()) for _, p, _ in entries)
            grad_norm = grad_sq**0.5
            out[f"{group}/grad_norm"] = grad_norm
            # The ratio is what says whether a group is learning: a norm of 1e-3 means
            # nothing until you know whether the weights are 1e-1 or 1e-6.
            out[f"{group}/grad_to_param"] = grad_norm / param_sq**0.5 if param_sq > 0 else 0.0
        for name, norm in per_parameter.items():
            out[f"param/{name}/grad_norm"] = norm
        return out


class ArrayDump(Collector):
    """The full covariance, and a feature sample, as `.npy` beside the stream.

    Very low cadence by construction: this is the one collector that writes bytes proportional
    to D^2. The stream records a filename and the array lands in `metrics/arrays/`, so the
    covariance stays recoverable for a heatmap without the step stream carrying megabytes of
    matrix.

    Returning the filename rather than the array is load-bearing: `MetricsWriter` drops any
    array over 64 elements, so a D x D matrix returned through the normal path would vanish
    silently.

    `rec.arrays_dir` is set on the global-zero rank only, so this writes from one rank and
    emits nothing on the others.
    """

    needs = frozenset()
    reduce: dict[str, str] = {}

    def __init__(
        self,
        cadence: Cadence = 5000,
        max_rows: int = 4096,
        keys: list[str] | None = None,
        dump_sample: bool = True,
    ):
        self.cadence = cadence
        self.max_rows = int(max_rows)
        self.keys = list(keys) if keys else None
        self.dump_sample = bool(dump_sample)

    def compute(self, rec: StepRecord) -> dict[str, float | np.ndarray]:
        if rec.arrays_dir is None:
            log.debug("ArrayDump skipped: no arrays_dir on the record")
            return {}
        rec.arrays_dir.mkdir(parents=True, exist_ok=True)

        out: dict[str, float | np.ndarray] = {}
        for name, tensor in rec.observables.items():
            if self.keys is not None and name not in self.keys:
                continue
            rows = _as_rows(tensor)
            if rows is None:
                continue
            rows = _subsample(rows, self.max_rows, seed=rec.step).detach().float()
            if rows.shape[0] < 2:
                continue
            slug = name.replace("/", "_")

            cov_name = f"cov_{slug}_step{rec.step}.npy"
            np.save(rec.arrays_dir / cov_name, torch.cov(rows.T).cpu().numpy())
            out[f"{name}/cov_file"] = cov_name  # type: ignore[assignment]

            if self.dump_sample:
                sample_name = f"sample_{slug}_step{rec.step}.npy"
                np.save(rec.arrays_dir / sample_name, rows.cpu().numpy())
                out[f"{name}/sample_file"] = sample_name  # type: ignore[assignment]
        return out


# ---------------------------------------------------------------------------- helpers

_NORM_TOKENS = ("norm", "bn", "batchnorm", "layernorm", "groupnorm")


def _is_bias_or_norm(name: str) -> bool:
    """Both filters by name, since a collector is handed `named_parameters` alone.

    Walking `named_modules()` would identify norm layers exactly. The test on the name is
    weaker and would miss a norm layer called `stem2`, so a module with an unusual naming
    scheme should declare `grad_taxonomy` and put its norm parameters in a group of their own
    rather than rely on this.
    """
    lowered = name.lower()
    if lowered.endswith("bias"):
        return True
    return any(token in lowered for token in _NORM_TOKENS)


def _group_of(name: str, taxonomy: dict[str, tuple[str, ...]] | None) -> str:
    """Which reported group a parameter name belongs to. Longest prefix wins.

    Nesting then means the obvious thing: a more specific prefix beats a more general one. On
    first match in declaration order,
    `{"backbone": ("model.student",), "head": ("model.student_head",)}` sends every head
    parameter to `backbone`, since `"model.student_head.weight"` starts with `"model.student"`
    -- the head group reports nothing and the backbone norm quietly averages the head into
    itself. Where the prefixes are disjoint the two rules agree.
    """
    if taxonomy:
        best_group, best_len = "other", -1
        for group, prefixes in taxonomy.items():
            for prefix in prefixes:
                if name.startswith(prefix) and len(prefix) > best_len:
                    best_group, best_len = group, len(prefix)
        return best_group
    return name.split(".")[0]


def _as_rows(tensor: Tensor) -> Tensor | None:
    """An `[N, D]` view of an observable, or `None` if it is not one.

    A model may hand over a scalar buffer or a 3-D tap; flattening all but the last dimension
    is the only interpretation that does not require knowing what the tap means.
    """
    if not isinstance(tensor, Tensor) or tensor.ndim < 2:
        return None
    return tensor.reshape(-1, tensor.shape[-1])


def _subsample(rows: Tensor, max_rows: int, seed: int) -> Tensor:
    """At most `max_rows` rows, chosen deterministically from `seed`.

    `O(N D^2 + D^3)` is well under a millisecond at D of 64 or 128, but only if N is bounded,
    and a full epoch of pixels is millions of rows. The generator is on the CPU regardless of
    where the rows live, so the same step selects the same indices whether the run is on a GPU
    or not, which is what makes a CPU test of this meaningful.
    """
    n = int(rows.shape[0])
    if n <= max_rows:
        return rows
    generator = torch.Generator().manual_seed(int(seed))
    index = torch.randperm(n, generator=generator)[:max_rows]
    return rows.index_select(0, index.to(rows.device))


def _order_stats(prefix: str, values: Tensor) -> dict[str, float]:
    """Order statistics, not a mean: a collapsing representation shows up as a spread between
    the median and the extremes long before it moves the average."""
    flat = values.detach().float().flatten()
    if flat.numel() == 0:
        return {}
    quantiles = torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0], device=flat.device)
    q = torch.quantile(flat, quantiles).tolist()
    return {
        f"{prefix}_min": q[0],
        f"{prefix}_p05": q[1],
        f"{prefix}_median": q[2],
        f"{prefix}_p95": q[3],
        f"{prefix}_max": q[4],
        f"{prefix}_mean": float(flat.mean()),
    }


class TermGrad(Collector):
    """Do the objective's terms pull the shared parameters in the same direction?

    Emits, over the parameters the terms share (the backbone): each term's gradient norm, its
    share of the total, the pairwise cosines, and the pairwise sign-conflict fraction.

    This is the failure a loss curve hides best. Two terms with large gradients that cancel
    leave every per-term loss looking healthy while the model learns nothing from either, and
    nothing else in the metrics layer would show it. A negative cosine says the objective is
    fighting itself; a sign-conflict fraction near 0.5 says the two are merely unrelated,
    which is a much less alarming reading of the same small cosine.

    The vectors arrive already reduced. `StepRecord.term_grads` is filled by the module, which
    takes them with `torch.autograd.grad(..., retain_graph=True)` before the real backward and
    all-reduces them itself. That primitive leaves `.grad` untouched and fires no accumulator
    hook, so it reduces nothing on its own and the real backward still reduces correctly
    afterwards. The module has to be the one that reduces them: `cos(mean(g_a), mean(g_b))` is
    the conflict in the gradient the optimizer applies, while a mean of per-rank cosines is a
    different and noisier quantity.

    So `reduce` is empty, and that is a claim worth checking, like `GradNorm`'s. Reducing here
    would double-reduce, or average cosines that were never comparable.

    `cadence` defaults to 500 because a firing step costs about 1.5x a normal one at three
    terms over six views.
    """

    needs = frozenset({"term_grads"})
    reduce: dict[str, Reduction] = {}

    def __init__(self, cadence: Cadence = 500, top_pairs: int = 0):
        self.cadence = cadence
        # 0 means "every pair". With three terms there are three pairs, so a cap only matters
        # if the objective grows; it is here so that growth does not silently widen the stream.
        self.top_pairs = int(top_pairs)

    def compute(self, rec: StepRecord) -> dict[str, float | np.ndarray]:
        grads = rec.term_grads or {}
        # Absent, never null: a step where the module did not produce them writes no columns
        # rather than a row of zeros that reads as "measured, and they did not conflict".
        if len(grads) < 1:
            return {}

        names = sorted(grads)
        norms = {n: float(grads[n].norm()) for n in names}
        total = sum(norms.values())

        out: dict[str, float | np.ndarray] = {"n_terms": float(len(names))}
        for n in names:
            out[f"norm/{n}"] = norms[n]
            # What fraction of the total gradient magnitude this term is responsible for. A
            # term at 0.99 is the objective; the others are decoration at this step.
            out[f"share/{n}"] = norms[n] / total if total > 0 else 0.0

        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
        if self.top_pairs:
            pairs = pairs[: self.top_pairs]
        cosines = []
        for a, b in pairs:
            ga, gb = grads[a], grads[b]
            na, nb = norms[a], norms[b]
            cos = float((ga * gb).sum() / (na * nb)) if na > 0 and nb > 0 else 0.0
            out[f"cos/{a}|{b}"] = cos
            cosines.append(cos)
            # The fraction of coordinates where the two terms disagree in sign. A cosine near
            # zero is ambiguous -- orthogonal or cancelling -- and this separates the two:
            # ~0.5 is unrelated, well above 0.5 is opposed coordinate by coordinate.
            both = (ga != 0) & (gb != 0)
            n_both = int(both.sum())
            out[f"conflict/{a}|{b}"] = (
                float(((ga[both] * gb[both]) < 0).sum()) / n_both if n_both else 0.0
            )
        if cosines:
            # The headline: the worst pair. A single number a run can be watched on.
            out["min_cos"] = min(cosines)
        return out
