"""`TermGrad`: the in-loop gradient-conflict collector, unblocked by spike (e).

Two halves, and only one of them is about arithmetic:

* the **collector** is pure -- it reads already-reduced vectors off the record and computes
  norms, shares, cosines and sign-conflict fractions. Tested here.
* the **request path** -- engine asks only when a firing collector needs it, module produces
  it before the real backward and all-reduces it -- is what could hang a job, and the
  rank-uniformity that stops it is asserted in `tests/test_engine_distributed.py`.

The property the collector rests on and cannot check for itself: `rec.term_grads` arrives
**already reduced**. `cos(mean(g_a), mean(g_b))` is the conflict in the gradient the optimizer
applies; a mean of per-rank cosines is a different quantity. Spike (e) established that
`autograd.grad` returns rank-local vectors, so the module must reduce them -- and the collector
must not reduce again, which is why `reduce` is empty.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from wcfm.metrics.base import StepRecord, fires  # noqa: E402
from wcfm.metrics.collectors import TermGrad  # noqa: E402


def _rec(**grads):
    return StepRecord(
        step=500,
        epoch=0,
        term_grads={k: torch.tensor(v, dtype=torch.float32) for k, v in grads.items()},
    )


def test_two_terms_pulling_together():
    out = TermGrad().compute(_rec(dino=[1.0, 0.0], charge=[2.0, 0.0]))
    assert out["cos/charge|dino"] == pytest.approx(1.0)
    assert out["norm/dino"] == pytest.approx(1.0)
    assert out["norm/charge"] == pytest.approx(2.0)
    # Share of total gradient magnitude: charge is twice dino, so 2/3 and 1/3.
    assert out["share/charge"] == pytest.approx(2 / 3)
    assert out["share/dino"] == pytest.approx(1 / 3)
    assert out["n_terms"] == 2.0


def test_two_terms_fighting_are_visible_as_a_negative_cosine():
    """The failure a loss curve hides best: both per-term losses look healthy and the shared
    parameters receive nothing."""
    out = TermGrad().compute(_rec(a=[1.0, 0.0], b=[-1.0, 0.0]))
    assert out["cos/a|b"] == pytest.approx(-1.0)
    assert out["min_cos"] == pytest.approx(-1.0)
    assert out["conflict/a|b"] == pytest.approx(1.0)


def test_sign_conflict_separates_orthogonal_from_opposed():
    """A cosine near zero is ambiguous. ~0.5 conflict is unrelated; near 1.0 is opposed
    coordinate by coordinate, and the two deserve different reactions."""
    # Orthogonal: cosine 0, but they never disagree on a shared coordinate.
    ortho = TermGrad().compute(_rec(a=[1.0, 1.0, 0.0, 0.0], b=[0.0, 0.0, 1.0, 1.0]))
    assert ortho["cos/a|b"] == pytest.approx(0.0)
    assert ortho["conflict/a|b"] == pytest.approx(0.0)  # no coordinate is non-zero in both

    # Also cosine 0, but half the shared coordinates are opposed.
    mixed = TermGrad().compute(_rec(a=[1.0, 1.0], b=[1.0, -1.0]))
    assert mixed["cos/a|b"] == pytest.approx(0.0)
    assert mixed["conflict/a|b"] == pytest.approx(0.5)


def test_three_terms_report_every_pair_and_the_worst_of_them():
    out = TermGrad().compute(_rec(a=[1.0, 0.0], b=[0.0, 1.0], c=[-1.0, 0.0]))
    assert {"cos/a|b", "cos/a|c", "cos/b|c"} <= set(out)
    assert out["min_cos"] == pytest.approx(-1.0)  # a vs c
    assert out["n_terms"] == 3.0


def test_a_zero_gradient_does_not_divide_by_zero():
    out = TermGrad().compute(_rec(a=[0.0, 0.0], b=[1.0, 0.0]))
    assert out["cos/a|b"] == 0.0
    assert out["share/a"] == 0.0


def test_no_term_grads_writes_no_columns():
    """Absent, never null: a step where the module produced none must not write a row of zeros
    that reads as "measured, and they did not conflict"."""
    assert TermGrad().compute(StepRecord(step=1, epoch=0)) == {}
    assert TermGrad().compute(StepRecord(step=1, epoch=0, term_grads={})) == {}


def test_a_single_term_reports_a_norm_and_no_cosine():
    out = TermGrad().compute(_rec(only=[3.0, 4.0]))
    assert out["norm/only"] == pytest.approx(5.0)
    assert out["share/only"] == pytest.approx(1.0)
    assert not any(k.startswith("cos/") for k in out)
    assert "min_cos" not in out


def test_reduce_is_empty_and_that_is_a_claim():
    """The vectors were reduced by the module. Reducing here would either double-reduce or
    average cosines that were never comparable -- see spike (e)."""
    assert TermGrad().reduce == {}


def test_it_declares_the_need_that_makes_the_engine_ask():
    assert "term_grads" in TermGrad().needs


def test_unmet_needs_are_satisfied_only_when_the_record_carries_them():
    """The engine skips a collector whose needs a record cannot satisfy, rather than letting it
    raise mid-epoch."""
    t = TermGrad()
    assert t.unmet(StepRecord(step=1, epoch=0)) == {"term_grads"}
    assert t.unmet(_rec(a=[1.0])) == set()


def test_cadence_zero_disables_it_so_full_metrics_does_not_pay_for_it():
    """It is the only collector that makes the MODEL do extra work, so `conf/metrics/full.yaml`
    ships it at cadence 0."""
    assert not fires(TermGrad(cadence=0), 0)
    assert not fires(TermGrad(cadence=0), 500)
    assert fires(TermGrad(cadence=500), 500)
    assert not fires(TermGrad(cadence=500), 501)
