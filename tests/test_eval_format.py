"""Evaluation format v2: the properties that make two numbers comparable, or refuse to.

No torch here on purpose -- the format is readable by a probe host, a merge step and a test
without a model, and a test that needed one would be evidence the split had not been made.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from wcfm.eval.format import (
    EvalSet,
    FeatureStore,
    Provenance,
    check_comparability,
    event_key_hash,
)


def keys(n: int = 6) -> list[str]:
    return [f"evt{i:04d}" for i in range(n)]


def make_set(tmp_path, *, id="v1_W", ks=None, sample="in-sample") -> tuple[EvalSet, object]:
    ks = keys() if ks is None else ks
    root = tmp_path / "evalset"
    es = EvalSet.create(
        root,
        id=id,
        event_keys=ks,
        truth={"pid": np.arange(len(ks)) % 3},
        geometry={"coords": np.zeros((len(ks), 2), dtype=np.int32)},
        sample=sample,
    )
    return es, root


def prov(**kw) -> Provenance:
    base = dict(
        eval_set_id="v1_W",
        event_key_hash=event_key_hash(keys()),
        checkpoint="checkpoint_epoch10.pt",
        checkpoint_sha256="a" * 64,
        sources=["student", "teacher"],
        taps=["dec_full"],
    )
    base.update(kw)
    return Provenance(**base)


# ------------------------------------------------------------------- the event-key hash


def test_the_hash_is_over_membership_not_order():
    """Two extractions that visited the same events in a different order -- a different
    `num_workers`, a reshuffled shard set -- are the same measurement."""
    assert event_key_hash(["b", "a", "c"]) == event_key_hash(["a", "b", "c"])


def test_the_hash_changes_when_one_event_changes():
    assert event_key_hash(["a", "b"]) != event_key_hash(["a", "b", "c"])
    assert event_key_hash(["a", "b"]) != event_key_hash(["a", "z"])


def test_a_set_edited_in_place_after_results_were_written_is_caught(tmp_path):
    """The failure v1 could not see: the name still matches, the events do not."""
    es, root = make_set(tmp_path)
    es.verify(root)
    np.save(root / "event_keys.npy", np.asarray(keys(5)).astype("U"))
    with pytest.raises(ValueError, match="does not match its recorded key hash"):
        es.verify(root)


# ------------------------------------------------------------------ truth written once


def test_truth_lives_with_the_eval_set_not_the_checkpoint(tmp_path):
    """~1.37 GB of byte-identical columns per checkpoint in v1. A checkpoint's directory must
    hold only what depends on the checkpoint."""
    es, root = make_set(tmp_path)
    store = FeatureStore(tmp_path / "ep10")
    store.write_features("student", "dec_full", np.zeros((6, 8)))
    store.write_provenance(prov())

    written = {p.name for p in (tmp_path / "ep10").iterdir()}
    assert not any("pid" in n or "coords" in n for n in written), written
    assert es.read(root, "pid").tolist() == [0, 1, 2, 0, 1, 2]


def test_features_are_fp16_and_memory_mapped(tmp_path):
    """Uncompressed so a read is a page-cache hit; fp16 because these feed linear probes and
    neighbour searches, where the third decimal does not move a score."""
    store = FeatureStore(tmp_path / "ep10")
    store.write_features("student", "dec_full", np.random.rand(4, 3).astype(np.float64))
    got = store.features("student", "dec_full")
    assert got.dtype == np.float16
    assert isinstance(got, np.memmap)


def test_both_branches_of_one_pass_are_separate_arrays(tmp_path):
    store = FeatureStore(tmp_path / "ep10")
    store.write_features("student", "dec_full", np.zeros((2, 2)))
    store.write_features("teacher", "dec_full", np.ones((2, 2)))
    store.write_features("student", "enc1", np.zeros((2, 4)))
    assert store.available() == {
        ("student", "dec_full"),
        ("teacher", "dec_full"),
        ("student", "enc1"),
    }
    assert store.features("teacher", "dec_full").tolist() == [[1, 1], [1, 1]]


def test_a_missing_block_names_what_is_there(tmp_path):
    store = FeatureStore(tmp_path / "ep10")
    store.write_features("student", "dec_full", np.zeros((2, 2)))
    with pytest.raises(FileNotFoundError, match="dec_half"):
        store.features("student", "dec_half")


def test_a_name_that_would_break_the_filename_index_is_refused(tmp_path):
    """The filename *is* the index, so a tap called `a__b` would parse back as source `a`."""
    store = FeatureStore(tmp_path / "ep10")
    with pytest.raises(ValueError, match="not usable in a filename"):
        store.write_features("student", "dec__full", np.zeros((2, 2)))


def test_a_partly_written_file_is_never_visible(tmp_path):
    """The DAG's PRE script reads these to decide whether to re-extract, which is why the old
    'older than the checkpoint, and has it settled' heuristic could go."""
    store = FeatureStore(tmp_path / "ep10")
    store.write_features("student", "dec_full", np.zeros((2, 2)))
    assert not list((tmp_path / "ep10").glob("*.tmp"))


# ---------------------------------------------------------------------- pools


def test_pools_are_drawn_once_at_extraction(tmp_path):
    """Every probe scores the same population by construction, instead of each redrawing and
    the suite comparing measurements over different samples."""
    store = FeatureStore(tmp_path / "ep10")
    store.write_pools(
        row_index=np.arange(5), train=np.array([0, 1, 2]), val=np.array([3, 4])
    )
    assert sorted(store.pools()) == ["row_index", "train", "val"]
    assert store.pools()["val"].tolist() == [3, 4]


def test_pools_always_carry_the_row_index_that_joins_them_to_truth(tmp_path):
    """A pooled block's row `i` is eval-set pixel `row_index[i]`, and a reader that indexes
    truth through it is correct whether or not the block was pooled."""
    store = FeatureStore(tmp_path / "ep10")
    store.write_pools(row_index=np.array([7, 9, 11]), train=np.array([0, 2]))
    assert store.pools()["row_index"].tolist() == [7, 9, 11]
    with pytest.raises(TypeError):
        store.write_pools(train=np.array([0]))  # row_index is not optional


def test_no_pools_is_empty_not_an_error(tmp_path):
    store = FeatureStore(tmp_path / "ep10")
    store.write_features("student", "dec_full", np.zeros((2, 2)))
    assert store.pools() == {}


# ------------------------------------------------------------- comparability


def test_different_eval_sets_are_an_error_not_a_warning(tmp_path):
    """v1 warned and printed the table anyway. Two numbers over different events are not two
    measurements of the same thing."""
    with pytest.raises(ValueError, match="different eval sets"):
        check_comparability({"a": prov(), "b": prov(eval_set_id="other")})


def test_a_reused_id_over_changed_events_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="share an eval_set_id but not their event keys"):
        check_comparability({"a": prov(), "b": prov(event_key_hash=event_key_hash(keys(5)))})


def test_the_same_set_scored_twice_is_comparable():
    assert check_comparability({"ep10": prov(), "ep20": prov(checkpoint="e20.pt")}) == []


def test_a_mixed_charge_transform_warns_because_the_feature_columns_survive():
    """It invalidates the raw and delta columns, not the feature ones -- so the table is still
    worth printing, with the caveat attached."""
    got = check_comparability(
        {"a": prov(charge_transform="trained"), "b": prov(charge_transform="log10_1p")}
    )
    assert len(got) == 1 and "charge_transform" in got[0]


def test_a_mixed_pool_size_warns():
    got = check_comparability({"a": prov(pool_per_class=500), "b": prov(pool_per_class=1000)})
    assert len(got) == 1 and "pool_per_class" in got[0]


def test_an_empty_table_is_fine():
    assert check_comparability({}) == []


# ----------------------------------------------------------------- provenance


def test_provenance_records_the_cost_of_the_step_that_dominates(tmp_path):
    """Extraction is the most expensive stage in the pipeline and v1 did not record its cost,
    so a run that got slower left no evidence."""
    store = FeatureStore(tmp_path / "ep10")
    store.write_provenance(prov(extract_seconds=612.5, events_per_second=16.3))
    got = store.provenance()
    assert got.extract_seconds == 612.5 and got.events_per_second == 16.3


def test_a_v1_directory_is_refused_rather_than_misread(tmp_path):
    d = tmp_path / "ep10"
    d.mkdir()
    (d / "provenance.json").write_text(json.dumps({"format_version": 1}))
    with pytest.raises(ValueError, match="format v1, this reader is v2"):
        FeatureStore(d).provenance()


def test_a_missing_provenance_says_which_two_things_it_could_mean(tmp_path):
    with pytest.raises(FileNotFoundError, match="did not finish"):
        FeatureStore(tmp_path / "nothing").provenance()


def test_the_sample_field_is_carried_so_a_result_cannot_imply_a_holdout_it_lacks(tmp_path):
    es, root = make_set(tmp_path, sample="in-sample")
    assert EvalSet.load(root).sample == "in-sample"
    es2, root2 = make_set(tmp_path / "b", id="v2_W", sample="held-out")
    assert EvalSet.load(root2).sample == "held-out"
