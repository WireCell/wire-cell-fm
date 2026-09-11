"""The pool draws, and the two properties the whole `rows="pooled"` layout rests on.

The properties, in order of what they cost if they are wrong:

1. **Every pool is inside the row space.** If a pool holds an eval-set index the store did not
   write, `searchsorted` maps it to a neighbouring row and the probe scores the wrong pixel
   without any array being the wrong shape. Nothing downstream can detect it.
2. **The draw does not depend on the features.** It is the claim that makes drawing at
   extraction time legal at all.
3. **Row space and eval-set space agree.** `feat[pool]` and `truth[row_index[pool]]` name the
   same pixel, under both row spaces.
"""

from __future__ import annotations

import numpy as np
import pytest

from wcfm.eval.pools import (
    PoolSpec,
    draw_pools,
    instance_truth_mask,
    overlap_contamination,
    truth_mask,
)
from wcfm.eval.taxonomy import PID_CLASSES

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _fake(n_events: int = 40, per_event: int = 120, seed: int = 0):
    """Truth and geometry with the shape extraction produces, and no features anywhere."""
    rng = np.random.RandomState(seed)
    n_pixels = n_events * per_event
    offsets = np.arange(n_events + 1, dtype=np.int64) * per_event
    truth = {
        "pixel_labels": rng.randint(0, len(PID_CLASSES), n_pixels).astype(np.int8),
        "pixel_energyfrac": rng.uniform(0.4, 1.0, n_pixels).astype(np.float32),
        "pixel_trackid": rng.randint(0, 9, n_pixels).astype(np.int32),
        "labels": rng.randint(0, 4, n_events).astype(np.int64),
    }
    geometry = {
        "positions": np.stack(
            [rng.randint(0, 800, n_pixels), rng.randint(0, 2000, n_pixels)], axis=1
        ).astype(np.int32),
        "charges": rng.uniform(1, 200, n_pixels).astype(np.float32),
        "offsets": offsets,
    }
    return truth, geometry


def test_pools_are_drawn_for_every_probe_whose_truth_is_present():
    truth, geometry = _fake()
    pools = draw_pools(truth=truth, geometry=geometry, spec=PoolSpec(per_class=50))
    for name in ("pid_train", "pid_val", "knn_pool", "event_sample", "overlap_val"):
        assert name in pools, name
    assert "overlap_train_0" in pools and "overlap_train_2" in pools
    assert "instance_queries" in pools
    # No apa/view were given, so the vertex projection cannot run and the pools are absent
    # rather than empty -- "this probe cannot run here" is not "it ran and found nothing".
    assert not any(k.startswith("vertex") for k in pools)


def test_absent_truth_leaves_a_note_rather_than_an_empty_pool():
    truth, geometry = _fake()
    del truth["pixel_energyfrac"]
    del truth["pixel_trackid"]
    spec = PoolSpec(per_class=50)
    pools = draw_pools(truth=truth, geometry=geometry, spec=spec)
    assert not any(k.startswith("overlap") for k in pools)
    assert not any(k.startswith("instance") for k in pools)
    assert "overlap" in spec.notes and "instance" in spec.notes


def test_no_pixel_labels_draws_nothing():
    truth, geometry = _fake()
    del truth["pixel_labels"]
    spec = PoolSpec()
    assert draw_pools(truth=truth, geometry=geometry, spec=spec) == {}
    assert "all" in spec.notes


def test_the_draw_is_a_function_of_truth_alone():
    """Property 2: same truth, same seed, same pools -- and no feature ever entered."""
    truth, geometry = _fake()
    a = draw_pools(truth=truth, geometry=geometry, spec=PoolSpec(per_class=50))
    b = draw_pools(truth=truth, geometry=geometry, spec=PoolSpec(per_class=50))
    assert sorted(a) == sorted(b)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def test_a_different_seed_draws_a_different_population():
    truth, geometry = _fake()
    a = draw_pools(truth=truth, geometry=geometry, spec=PoolSpec(per_class=50, seed=42))
    b = draw_pools(truth=truth, geometry=geometry, spec=PoolSpec(per_class=50, seed=7))
    assert not np.array_equal(a["pid_train"], b["pid_train"])


def test_pid_pools_are_disjoint_because_the_split_is_by_event():
    truth, geometry = _fake()
    pools = draw_pools(truth=truth, geometry=geometry, spec=PoolSpec(per_class=50))
    assert not set(pools["pid_train"].tolist()) & set(pools["pid_val"].tolist())


def test_pools_map_into_row_space_and_back(tmp_path):
    """Properties 1 and 3, through the real `_draw_pools`, under both row spaces."""
    from wcfm.eval.extract import _draw_pools

    truth, geometry = _fake()
    spec = PoolSpec(per_class=50)
    n_pixels = int(geometry["offsets"][-1])

    all_rows, all_pools, all_sample = _draw_pools(
        truth=truth, geometry=geometry, rows="all", spec=spec, apa=-1, view=""
    )
    np.testing.assert_array_equal(all_rows, np.arange(n_pixels))

    pooled_rows, pooled_pools, pooled_sample = _draw_pools(
        truth=truth, geometry=geometry, rows="pooled", spec=PoolSpec(per_class=50),
        apa=-1, view="",
    )
    assert sorted(all_pools) == sorted(pooled_pools)
    # The event sample comes back separately and in EVAL-SET space in both cases -- it is
    # pooled into per-event vectors, never written as rows.
    np.testing.assert_array_equal(all_sample, pooled_sample)
    assert "event_sample" not in pooled_pools

    labels = truth["pixel_labels"]
    for name in all_pools:
        # Property 1: in range, so no index silently lands on a neighbour.
        assert pooled_pools[name].max(initial=-1) < len(pooled_rows)
        # Property 3: the same pixel is named in both spaces.
        np.testing.assert_array_equal(
            labels[all_rows[all_pools[name]]],
            labels[pooled_rows[pooled_pools[name]]],
            err_msg=name,
        )


def test_pooled_row_space_is_exactly_the_union_of_the_pools():
    from wcfm.eval.extract import _draw_pools

    truth, geometry = _fake()
    rows, pools, _ = _draw_pools(
        truth=truth, geometry=geometry, rows="pooled", spec=PoolSpec(per_class=50),
        apa=-1, view="",
    )
    covered = np.unique(np.concatenate([rows[p] for p in pools.values() if len(p)]))
    np.testing.assert_array_equal(covered, rows)


def test_spec_check_refuses_a_pool_drawn_under_other_constants():
    spec = PoolSpec()
    spec.check("probe_overlap", overlap_thresholds=(0.2, 0.1, 0.3))
    with pytest.raises(ValueError, match="written against"):
        spec.check("probe_overlap", overlap_thresholds=(0.2, 0.4))
    with pytest.raises(ValueError, match="written against"):
        spec.check("probe_instance", instance_max_queries=5)


def test_spec_round_trips_through_json_shaped_dict():
    spec = PoolSpec(per_class=123, overlap_thresholds=(0.5, 0.6))
    back = PoolSpec.from_dict(spec.as_dict())
    assert back.per_class == 123
    assert back.overlap_thresholds == (0.5, 0.6)
    # Tolerates a dict carrying keys this version does not know, so a newer store still reads.
    PoolSpec.from_dict({**spec.as_dict(), "something_new": 1})


def test_truth_derived_quantities_match_the_old_definitions():
    labels = np.array([0, 1, 2, 0, 3], dtype=np.int64)
    frac = np.array([0.5, 0.25, 1.0, 0.1, 0.8], dtype=np.float32)
    tid = np.array([0, 5, -7, 3, 0], dtype=np.int32)

    np.testing.assert_array_equal(truth_mask(labels), [False, True, True, False, True])
    ov = overlap_contamination(labels, frac)
    # Untruthed pixels are forced to 0 rather than left at 1 - frac.
    assert ov[0] == 0.0 and ov[3] == 0.0
    assert ov[1] == pytest.approx(0.75) and ov[2] == pytest.approx(0.0)

    inst, mask = instance_truth_mask(labels, tid)
    np.testing.assert_array_equal(inst, [0, 5, 7, 3, 0])
    # trackid 0 is "no truth", and a label-0 pixel never carries instance truth.
    np.testing.assert_array_equal(mask, [False, True, True, False, False])


def test_event_sample_keeps_small_events_whole_and_thins_large_ones():
    from wcfm.eval.pools import _event_sample

    offsets = np.array([0, 3, 3, 20], dtype=np.int64)  # sizes 3, 0 (empty), 17
    idx = _event_sample(offsets, max_per_event=5, seed=0)
    first, last = idx[idx < 3], idx[idx >= 3]
    np.testing.assert_array_equal(np.sort(first), [0, 1, 2])  # kept whole
    assert len(last) == 5  # thinned to the cap
    assert set(last.tolist()) <= set(range(3, 20))


def test_knn_pool_spans_events_rather_than_the_largest_one():
    """The per-image cap is the whole point of `collect`: without it one event fills a class."""
    from wcfm.eval.pools import _knn_pool

    n_events, per_event = 30, 100
    offsets = np.arange(n_events + 1, dtype=np.int64) * per_event
    labels = np.ones(n_events * per_event, dtype=np.int64)  # every pixel is class 1 (Track)
    pool = _knn_pool(
        pixel_labels=labels, offsets=offsets, max_per_class=300, max_per_image=2, seed=0
    )
    events = pool // per_event
    assert len(np.unique(events)) > 10, "the pool came from too few events"
    assert np.bincount(events).max() <= 2


def test_pooling_saves_rows_when_events_are_larger_than_the_caps():
    """What `rows="pooled"` is actually for.

    On a small eval set the caps do not bind and the union is everything -- which is honest,
    not a bug. The saving appears once events are big enough that the caps bite, which is the
    production case: ~5,500 pixels per event against a 2,000-pixel event cap.
    """
    from wcfm.eval.extract import _draw_pools

    truth, geometry = _fake(n_events=12, per_event=4000)
    n_pixels = int(geometry["offsets"][-1])
    rows, _, sample = _draw_pools(
        truth=truth,
        geometry=geometry,
        rows="pooled",
        spec=PoolSpec(per_class=50, instance_max_queries=200),
        apa=-1,
        view="",
    )
    assert len(rows) < n_pixels
    # The event sample is capped per event and is not part of the row space.
    assert len(sample) == 12 * 2000


def test_event_means_average_only_the_sampled_pixels():
    from wcfm.eval.pools import mean_pool

    offsets = np.array([0, 4, 4, 8], dtype=np.int64)  # sizes 4, 0, 4
    feats = np.arange(8, dtype=np.float32).reshape(8, 1)
    sample = np.array([0, 1, 6, 7], dtype=np.int64)  # two from event 0, two from event 2
    pooled = mean_pool(feats, sample, offsets, n_events=3)
    assert pooled.shape == (3, 1)
    assert pooled[0, 0] == pytest.approx(0.5)
    assert np.isnan(pooled[1, 0])  # the empty event has no mean, and says so
    assert pooled[2, 0] == pytest.approx(6.5)
