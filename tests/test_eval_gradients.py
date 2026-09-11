"""The offline gradient probe.

Two halves, tested apart because only one of them needs a GPU:

* the **arithmetic** -- norms, pairwise cosines, accumulation across batches -- is framework
  side and knows only that it was handed flat vectors, so it is tested against a fake module
  here on CPU;
* the **hook** -- `SslModule.term_gradients` -- needs a real sparse-conv forward and is marked
  `gpu`, in `tests/test_model_gpu.py`.

The property worth pinning is the accumulation order: gradients are summed across batches and
the cosine taken once, not averaged per batch. The two answer different questions and the second
is noisier, so a test that passed either way would be worthless.
"""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from wcfm.eval.gradients import (  # noqa: E402
    GRADIENTS_FILE,
    gradient_report,
    term_gradients,
    write_gradients,
)


class _FakeBatch:
    batch_size = 4

    def to(self, _device):
        return self


class _FakeModule:
    """A module whose per-term gradients are dictated by the test.

    `gradient_report` reaches `term_gradients` with `getattr`, so nothing here needs to be an
    `nn.Module` -- which is the point of the hook being optional and duck-typed.
    """

    def __init__(self, per_batch):
        self._per_batch = list(per_batch)
        self.calls = 0

    def term_gradients(self, batch, *, step=0):
        out = self._per_batch[min(self.calls, len(self._per_batch) - 1)]
        self.calls += 1
        return {k: torch.tensor(v, dtype=torch.float32) for k, v in out.items()}


def _loader(n):
    return [_FakeBatch() for _ in range(n)]


def test_two_terms_pulling_together_score_a_positive_cosine():
    module = _FakeModule([{"dino": [1.0, 0.0], "charge": [1.0, 0.0]}])
    rep = gradient_report(module, _loader(1), device="cpu", max_batches=1)
    assert rep["terms"] == ["charge", "dino"]
    assert rep["cosine"]["charge|dino"] == pytest.approx(1.0)
    assert rep["norm"]["dino"] == pytest.approx(1.0)


def test_two_terms_fighting_score_a_negative_cosine():
    """The failure a loss plot hides best: both per-term losses look healthy and the backbone
    is being pulled apart."""
    module = _FakeModule([{"dino": [1.0, 0.0], "charge": [-1.0, 0.0]}])
    rep = gradient_report(module, _loader(1), device="cpu", max_batches=1)
    assert rep["cosine"]["charge|dino"] == pytest.approx(-1.0)
    assert rep["min_cosine"] == pytest.approx(-1.0)


def test_orthogonal_terms_score_zero():
    module = _FakeModule([{"a": [1.0, 0.0], "b": [0.0, 1.0]}])
    rep = gradient_report(module, _loader(1), device="cpu", max_batches=1)
    assert rep["cosine"]["a|b"] == pytest.approx(0.0)


def test_gradients_are_summed_across_batches_before_the_cosine():
    """Not averaged per batch and then combined. The quantity of interest is the direction the
    objective pulls over the whole event set; a mean of per-batch cosines is a different and
    noisier number, and here the two disagree in sign.

    Batch 1: a and b agree.  Batch 2: a is unchanged, b flips and is much larger.
    Per-batch cosines average to something positive; the summed vectors point apart.
    """
    module = _FakeModule(
        [
            {"a": [1.0, 0.0], "b": [1.0, 0.0]},
            {"a": [1.0, 0.0], "b": [-3.0, 0.0]},
        ]
    )
    rep = gradient_report(module, _loader(2), device="cpu", max_batches=2)
    # sum(a) = [2, 0], sum(b) = [-2, 0]  ->  cosine -1
    assert rep["cosine"]["a|b"] == pytest.approx(-1.0)
    assert rep["n_batches"] == 2


def test_the_per_batch_norm_spread_is_reported_beside_the_summed_norm():
    """So a cosine over the summed gradient can be read against how much the batches varied."""
    module = _FakeModule([{"a": [1.0, 0.0]}, {"a": [3.0, 0.0]}])
    rep = gradient_report(module, _loader(2), device="cpu", max_batches=2)
    assert rep["norm"]["a"] == pytest.approx(4.0)  # |[4, 0]|
    assert rep["norm_per_batch_mean"]["a"] == pytest.approx(2.0)  # (1 + 3) / 2
    assert rep["norm_per_batch_stdev"]["a"] > 0


def test_max_batches_bounds_the_pass():
    module = _FakeModule([{"a": [1.0, 0.0]}])
    rep = gradient_report(module, _loader(50), device="cpu", max_batches=3)
    assert rep["n_batches"] == 3 and module.calls == 3


def test_events_are_counted_so_two_reports_can_be_compared():
    module = _FakeModule([{"a": [1.0, 0.0]}])
    rep = gradient_report(module, _loader(2), device="cpu", max_batches=2)
    assert rep["n_events"] == 8  # 2 batches x batch_size 4


def test_a_single_term_reports_a_norm_and_no_cosine():
    module = _FakeModule([{"only": [2.0, 0.0]}])
    rep = gradient_report(module, _loader(1), device="cpu", max_batches=1)
    assert rep["norm"]["only"] == pytest.approx(2.0)
    assert rep["cosine"] == {} and rep["min_cosine"] is None


def test_a_zero_gradient_does_not_divide_by_zero():
    module = _FakeModule([{"a": [0.0, 0.0], "b": [1.0, 0.0]}])
    rep = gradient_report(module, _loader(1), device="cpu", max_batches=1)
    assert rep["cosine"]["a|b"] == 0.0


def test_no_gradients_at_all_is_a_recorded_error_not_a_crash():
    module = _FakeModule([{}])
    rep = gradient_report(module, _loader(2), device="cpu", max_batches=2)
    assert "error" in rep and rep["n_batches"] == 0


def test_a_module_without_the_hook_says_the_hook_is_optional():
    class NoHook:
        pass

    with pytest.raises(TypeError, match="optional on the training contract"):
        term_gradients(NoHook(), _FakeBatch())


def test_the_report_is_written_atomically_and_reads_back(tmp_path):
    module = _FakeModule([{"a": [1.0, 0.0], "b": [0.0, 1.0]}])
    rep = gradient_report(module, _loader(1), device="cpu", max_batches=1)
    path = write_gradients(tmp_path, rep)
    assert path.name == GRADIENTS_FILE
    assert not list(tmp_path.glob("*.tmp"))
    assert json.loads(path.read_text())["cosine"]["a|b"] == 0.0


def test_the_scope_is_recorded_because_a_cosine_needs_to_name_its_parameters():
    """A cosine over the backbone and one over "everything including each term's own head" are
    different measurements, and the second is close to meaningless -- a head is not contested."""
    module = _FakeModule([{"a": [1.0, 0.0]}])
    rep = gradient_report(module, _loader(1), device="cpu", max_batches=1)
    assert rep["scope"] == "backbone"
    assert rep["n_parameters"] == 2
