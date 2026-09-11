"""The generic collectors: Spectrum, GradNorm, ArrayDump.

Two properties get most of the attention, because both are the kind of thing that produces a
plausible number rather than an error:

* **The spectral quantities agree with the old offline computation.** ``debug.py:335`` stored a
  D x D covariance per logged step and ``plot_histories.py:25-31`` took the participation ratio
  from it afterwards. Moving that online is the whole saving, so the online value is checked
  against the offline formula on the same matrix, and against closed forms on matrices whose
  spectrum is known by construction.
* **The row sample is deterministic given the step.** An unseeded sample makes every
  run-to-run comparison noisy in a way that reads as a real difference.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from wcfm.metrics.base import StepRecord  # noqa: E402
from wcfm.metrics.collectors import (  # noqa: E402
    ArrayDump,
    GradNorm,
    Spectrum,
    Throughput,
    participation_ratio,
    rankme,
)

pytestmark = pytest.mark.stack


def legacy_pr(matrix: np.ndarray) -> float:
    """``plot_histories.py:18-31``, verbatim: eigh then ``(sum L)^2 / sum L^2``."""
    values, _ = np.linalg.eigh(matrix)
    v = values.clip(0)
    denom = float((v**2).sum())
    return float(v.sum() ** 2 / denom) if denom > 0 else 1.0


def _rows(n: int, d: int, seed: int = 0, scale: torch.Tensor | None = None) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    return x * scale if scale is not None else x


# ------------------------------------------------------------------ Spectrum


def test_the_online_participation_ratio_equals_the_old_offline_one():
    """Same matrix, same number. This is what makes it safe to stop storing the matrix."""
    rows = _rows(500, 16, seed=3)
    cov = torch.cov(rows.T)

    out = Spectrum(cadence="step")._one("f", rows, step=0)
    assert out["participation_ratio"] == pytest.approx(legacy_pr(cov.numpy()), rel=1e-5)


def test_participation_ratio_and_rankme_are_d_for_an_isotropic_spectrum():
    """Both are normalised to read on ``[1, D]``, so a perfectly spread 8-dimensional
    covariance must give exactly 8 -- the ceiling a healthy representation approaches."""
    eigenvalues = np.ones(8)
    assert participation_ratio(eigenvalues) == pytest.approx(8.0)
    assert rankme(eigenvalues) == pytest.approx(8.0)


def test_both_are_one_for_a_rank_one_spectrum():
    """The collapse floor: all variance in one direction."""
    eigenvalues = np.array([0.0, 0.0, 0.0, 5.0])
    assert participation_ratio(eigenvalues) == pytest.approx(1.0)
    assert rankme(eigenvalues) == pytest.approx(1.0)


def test_negative_eigenvalues_from_rounding_do_not_become_nan():
    """A covariance over a few hundred rows in 128 dimensions is near-singular and ``eigh``
    returns small negative eigenvalues for it routinely. The ``.clip(0)`` is why this is a
    number rather than a NaN that the writer then drops as absent."""
    eigenvalues = np.array([-1e-18, -1e-19, 1.0, 2.0])
    assert np.isfinite(participation_ratio(eigenvalues))
    assert np.isfinite(rankme(eigenvalues))


def test_an_all_zero_spectrum_falls_back_to_one_rather_than_dividing_by_zero():
    assert participation_ratio(np.zeros(4)) == 1.0
    assert rankme(np.zeros(4)) == 1.0


def test_rankme_weighs_the_tail_where_the_participation_ratio_does_not():
    """Why both ship, stated as the relationship rather than a threshold.

    On a spectrum with one dominant direction and a long small tail, the participation ratio
    is pinned near 1 by the leading eigenvalue -- it is a ratio of sums dominated by the
    largest term -- while RankMe, being an entropy, counts every direction that carries any
    variance at all. So RankMe sits above it, and further above it the longer the tail: a run
    losing rank slowly moves RankMe first. On an isotropic spectrum the two agree exactly,
    which is what makes them comparable on one axis.
    """
    short_tail = np.array([1e-3] * 40 + [1.0])
    long_tail = np.array([1e-2] * 200 + [1.0])

    assert participation_ratio(short_tail) < 1.1, "pinned near the collapse floor"
    assert rankme(short_tail) > participation_ratio(short_tail)

    gap_short = rankme(short_tail) - participation_ratio(short_tail)
    gap_long = rankme(long_tail) - participation_ratio(long_tail)
    assert gap_long > gap_short, "a longer tail widens the disagreement"

    isotropic = np.ones(16)
    assert rankme(isotropic) == pytest.approx(participation_ratio(isotropic))


def test_a_collapsed_representation_reads_lower_than_a_healthy_one():
    """The end-to-end statement the collector exists to make."""
    healthy = _rows(2000, 12, seed=1)
    collapsed = healthy.clone()
    collapsed[:, 1:] = 0.0  # everything in one direction

    spectrum = Spectrum(cadence="step")
    assert spectrum._one("f", collapsed, 0)["participation_ratio"] < 1.1
    assert spectrum._one("f", healthy, 0)["participation_ratio"] > 10.0


def test_the_row_count_is_always_reported():
    """The plan's DDP rule for feature statistics is rank-0 local on that rank's shard with
    the row count recorded -- a participation ratio over 200 rows and one over 65 000 are not
    comparable, and without ``n_rows`` nothing downstream would say so."""
    out = Spectrum(cadence="step")._one("f", _rows(300, 8), 0)
    assert out["n_rows"] == 300.0
    assert out["dim"] == 8.0
    assert Spectrum.reduce == {}, "rank-0 local: no engine-side collective"


def test_the_row_sample_is_deterministic_given_the_step_and_varies_across_steps():
    rows = _rows(5000, 8, seed=7)
    spectrum = Spectrum(cadence="step", max_rows=100)

    first = spectrum._one("f", rows, step=42)
    again = spectrum._one("f", rows, step=42)
    other = spectrum._one("f", rows, step=43)

    assert first["participation_ratio"] == again["participation_ratio"]
    assert first["n_rows"] == 100.0
    assert first["participation_ratio"] != other["participation_ratio"]


def test_fewer_than_two_rows_emits_nothing_rather_than_something_partial():
    """``torch.cov`` needs two rows. ``debug.py:332-333`` handled this with a bare ``return``,
    the early-return shape the plan cites as what hangs a job -- harmless in a collector only
    because ``compute`` performs no collective and ``reduce`` is empty. Absent, never null."""
    assert Spectrum(cadence="step")._one("f", torch.randn(1, 8), 0) == {}


def test_a_non_matrix_observable_is_skipped_not_crashed_on():
    """A model may hand over a scalar buffer -- ``observables`` returns opaque names and the
    framework does not know what any of them mean."""
    record = StepRecord(step=0, epoch=1, observables={"scalar": torch.tensor(1.0)})
    assert Spectrum(cadence="step").compute(record) == {}


def test_a_three_dimensional_tap_is_flattened_to_rows():
    """Flattening all but the last dimension is the only reading that does not require
    knowing what the tap means."""
    out = Spectrum(cadence="step")._one("f", torch.randn(4, 25, 8), 0)
    assert out["n_rows"] == 100.0 and out["dim"] == 8.0


def test_keys_narrows_an_expensive_collector_to_the_tap_being_watched():
    record = StepRecord(
        step=0,
        epoch=1,
        observables={"a": _rows(50, 4), "b": _rows(50, 4, seed=2)},
    )
    out = Spectrum(cadence="step", keys=["a"]).compute(record)
    assert any(k.startswith("a/") for k in out)
    assert not any(k.startswith("b/") for k in out)


def test_the_off_diagonal_fraction_separates_correlated_from_independent_features():
    independent = _rows(4000, 6, seed=11)
    correlated = independent.clone()
    correlated[:, 1] = correlated[:, 0]  # a duplicated feature is pure off-diagonal mass

    spectrum = Spectrum(cadence="step")
    assert spectrum._one("f", independent, 0)["off_diagonal_frac"] < 0.1
    assert spectrum._one("f", correlated, 0)["off_diagonal_frac"] > 0.2


def test_order_statistics_are_emitted_for_variance_and_norm():
    out = Spectrum(cadence="step")._one("f", _rows(400, 8, seed=5), 0)
    for prefix in ("var", "norm"):
        for stat in ("min", "p05", "median", "p95", "max", "mean"):
            assert f"{prefix}_{stat}" in out
    assert out["var_min"] <= out["var_median"] <= out["var_max"]


def test_the_top_eigenvalues_come_back_descending():
    out = Spectrum(cadence="step", top_k=3)._one("f", _rows(500, 8, seed=6), 0)
    top = out["eig_top"]
    assert len(top) == 3
    assert list(top) == sorted(top, reverse=True)


# ------------------------------------------------------------------ GradNorm


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(4, 4)
        self.decoder = torch.nn.Linear(4, 4)
        self.norm = torch.nn.LayerNorm(4)

    def forward(self, x):
        return self.decoder(self.norm(self.encoder(x)))


def _with_grads() -> _Net:
    net = _Net()
    net(torch.randn(8, 4)).pow(2).mean().backward()
    return net


def _record(net: _Net, step: int = 0, taxonomy=None) -> StepRecord:
    return StepRecord(step=step, epoch=1, named_parameters=net.named_parameters, taxonomy=taxonomy)


def test_parameters_group_by_first_path_token_without_a_declared_taxonomy():
    """``debug.py:429``'s default, kept."""
    out = GradNorm(cadence="step").compute(_record(_with_grads()))
    assert "encoder/grad_norm" in out and "decoder/grad_norm" in out
    assert "encoder/grad_to_param" in out


def test_biases_and_norm_parameters_are_excluded_by_default():
    """Carried deliberately from ``debug.py:424-427``: their gradients live on a different
    scale from a convolution's, so mixing them in makes a group's norm track the bias count
    rather than the depth."""
    net = _with_grads()
    default = GradNorm(cadence="step").compute(_record(net))
    assert not any(k.startswith("norm/") for k in default)

    everything = GradNorm(cadence="step", include_bias_and_norm=True).compute(_record(net))
    assert any(k.startswith("norm/") for k in everything)
    assert everything["encoder/grad_norm"] != default["encoder/grad_norm"], (
        "the bias contributes to the group norm once the filter is off"
    )


def test_a_declared_taxonomy_overrides_the_default_grouping():
    """By ``getattr`` on the module, not by a seventh method on ``TrainingModule``."""
    taxonomy = {"body": ("encoder.", "decoder."), "head": ("nothing.",)}
    out = GradNorm(cadence="step").compute(_record(_with_grads(), taxonomy=taxonomy))
    assert "body/grad_norm" in out
    assert "encoder/grad_norm" not in out
    assert "head/grad_norm" not in out, "a group with no matching parameter emits nothing"


def test_a_parameter_matching_no_declared_group_lands_in_other():
    taxonomy = {"body": ("encoder.",)}
    out = GradNorm(cadence="step").compute(_record(_with_grads(), taxonomy=taxonomy))
    assert "other/grad_norm" in out, "the decoder is unclaimed and must not disappear"


def test_the_grad_to_param_ratio_is_the_norm_over_the_parameter_norm():
    net = _with_grads()
    out = GradNorm(cadence="step").compute(_record(net))
    weight = net.encoder.weight
    expected = float(weight.grad.detach().norm()) / float(weight.detach().norm())
    assert out["encoder/grad_to_param"] == pytest.approx(expected, rel=1e-5)


def test_a_parameter_without_a_gradient_is_skipped():
    net = _Net()  # never backwarded
    assert GradNorm(cadence="step").compute(_record(net)) == {}


def test_the_per_parameter_dump_has_its_own_longer_cadence():
    """The point of the per-group number is that it is cheap enough to watch continuously; the
    per-parameter dump is not, so it gets a separate interval rather than the same one."""
    net = _with_grads()
    collector = GradNorm(cadence="step", dump_per_parameter_every=10)

    off = collector.compute(_record(net, step=3))
    assert not any(k.startswith("param/") for k in off)

    on = collector.compute(_record(net, step=20))
    assert "param/encoder.weight/grad_norm" in on


def test_gradnorm_declares_no_reduction_because_ddp_already_agreed():
    """These are read from ``.grad`` after backward, so DDP's all-reduce has run and every
    rank holds the same values. Declaring ``"mean"`` would buy a collective per firing for a
    number the ranks already agree on."""
    assert GradNorm.reduce == {}
    assert GradNorm.needs == frozenset({"grads"})


# ------------------------------------------------------------------ ArrayDump


def test_the_stream_gets_a_filename_and_the_array_lands_beside_it(tmp_path):
    """Returning the array itself would vanish: ``MetricsWriter`` drops any array over 64
    elements, so a D x D matrix through the normal path disappears silently."""
    record = StepRecord(
        step=500,
        epoch=1,
        observables={"student/dec": _rows(200, 8, seed=4)},
        arrays_dir=tmp_path / "arrays",
    )
    out = ArrayDump(cadence=500).compute(record)

    assert out["student/dec/cov_file"] == "cov_student_dec_step500.npy"
    assert out["student/dec/sample_file"] == "sample_student_dec_step500.npy"

    cov = np.load(tmp_path / "arrays" / "cov_student_dec_step500.npy")
    assert cov.shape == (8, 8)
    assert np.allclose(cov, cov.T, atol=1e-6), "a covariance is symmetric"
    assert np.load(tmp_path / "arrays" / "sample_student_dec_step500.npy").shape == (200, 8)


def test_the_dumped_covariance_is_the_one_the_spectrum_summarised(tmp_path):
    """So a heatmap and the scalar in the stream describe the same matrix."""
    rows = _rows(300, 6, seed=8)
    record = StepRecord(step=0, epoch=1, observables={"f": rows}, arrays_dir=tmp_path)
    ArrayDump(cadence=1).compute(record)

    dumped = np.load(tmp_path / "cov_f_step0.npy")
    assert participation_ratio(np.linalg.eigvalsh(dumped)) == pytest.approx(
        Spectrum(cadence="step")._one("f", rows, 0)["participation_ratio"], rel=1e-5
    )


def test_no_arrays_dir_emits_nothing_rather_than_guessing_a_path(tmp_path):
    record = StepRecord(step=0, epoch=1, observables={"f": _rows(50, 4)}, arrays_dir=None)
    assert ArrayDump(cadence=1).compute(record) == {}


def test_the_sample_dump_can_be_turned_off(tmp_path):
    record = StepRecord(step=0, epoch=1, observables={"f": _rows(50, 4)}, arrays_dir=tmp_path)
    out = ArrayDump(cadence=1, dump_sample=False).compute(record)
    assert "f/cov_file" in out and "f/sample_file" not in out
    assert not list(tmp_path.glob("sample_*"))


# ------------------------------------------------------------------ the shipped set


@pytest.mark.parametrize("preset", ["default", "full"])
def test_the_shipped_presets_instantiate_and_name_only_collectors_that_exist(preset):
    """These are the configs a real run selects, so an unresolvable ``_target_`` in one is a
    run that dies at construction. Checked by instantiating every entry, not by reading it.

    Cadences are deliberately not asserted here -- `tests/test_config.py` owns the one that
    is a decision (`default` leaves the costly collectors off) and pinning the rest would
    make routine retuning look like a regression.
    """
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(f"conf/metrics/{preset}.yaml")
    built = {name: instantiate(entry) for name, entry in cfg.collectors.items()}

    assert set(built) == {"throughput", "gradnorm", "spectrum", "arrays", "termgrad"}
    assert "term_grads" in built["termgrad"].needs
    assert isinstance(built["spectrum"], Spectrum)
    assert isinstance(built["gradnorm"], GradNorm)
    assert isinstance(built["arrays"], ArrayDump)
    assert isinstance(built["throughput"], Throughput)
    assert built["spectrum"].max_rows == 65536


def test_no_collector_declares_a_reduction_it_does_not_need():
    """Every ``reduce`` entry is a collective per firing. Only ``Throughput``'s are genuinely
    rank-local sums and maxima."""
    assert Throughput.reduce == {
        "samples_per_s": "sum",
        "voxels_per_s": "sum",
        "peak_mem_gb": "max",
    }
    assert Spectrum.reduce == {} and GradNorm.reduce == {} and ArrayDump.reduce == {}
