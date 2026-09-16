"""The evaluation DAG, and `wcfm diff`.

The DAG is generated, not hand-written, so the tests are about the structure it generates:
which node depends on which, which nodes ask for a GPU, and the two policies that are the whole
reason it replaces `collect_probes.sh` -- retries on the nodes that can fail transiently, and a
merge that cannot take a campaign down with it.

`build_dag` returns `{path: contents}` and writes nothing, so all of this is checkable without a
filesystem and `--dry-run` exercises the same code path a real submission does. That last point
is the one worth keeping: a dry run that builds the plan a different way proves nothing about
the real one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from wcfm.cli.jobpack import ARCHIVE_NAME
from wcfm.eval.dag import DagPlan, build_dag, checkpoints_of, epoch_of_checkpoint

#: `build_dag` copies the three job scripts out of `gridutils/eval/`, so the plan has to
#: name a real checkout. The archive itself is written by `wcfm eval submit`, not here.
REPO = Path(__file__).resolve().parents[1]


def _run_dir(tmp_path, epochs=(1, 5, 10)):
    run = tmp_path / "runs" / "hybrid_a"
    (run / "checkpoints").mkdir(parents=True)
    (run / "config.yaml").write_text("run:\n  name: hybrid_a\n")
    for e in epochs:
        (run / "checkpoints" / f"checkpoint_epoch{e}.pt").write_bytes(b"x" * (e + 1))
    (run / "checkpoints" / "latest.pt").write_bytes(b"x")
    return run


def _plan(tmp_path, **kw):
    run = kw.pop("run_dir", None) or _run_dir(tmp_path)
    base = dict(
        run_dir=run,
        out_root=run / "features",
        repo=REPO,
        repo_archive=run / "features" / "dag" / "job" / ARCHIVE_NAME,
        pyenv=tmp_path / "uvenv",
        cache=tmp_path / "cache",
        checkpoints=checkpoints_of(run),
        eval_set_root=run / "features" / "eval_set",
        stages="pid,spectrum",
        sources="student,teacher",
        taps="",
        rows="all",
        max_images=1000,
    )
    base.update(kw)
    return DagPlan(**base)


def test_latest_is_excluded_even_though_it_is_a_checkpoint(tmp_path):
    """It duplicates some `checkpoint_epochN.pt` under a name whose meaning changes between
    runs, so a features directory called `latest` would mean a different epoch in each."""
    run = _run_dir(tmp_path)
    found = checkpoints_of(run)
    assert [epoch_of_checkpoint(p) for p in found] == [1, 5, 10]
    assert not any("latest" in p.name for p in found)


def test_asking_for_an_epoch_with_no_checkpoint_says_which_exist(tmp_path):
    run = _run_dir(tmp_path)
    with pytest.raises(SystemExit, match=r"no checkpoint for epochs \[7\]"):
        checkpoints_of(run, "5,7")
    assert [epoch_of_checkpoint(p) for p in checkpoints_of(run, "10,1")] == [1, 10]


def test_the_dag_is_one_extract_probe_pair_per_epoch_plus_one_merge(tmp_path):
    files = build_dag(_plan(tmp_path))
    dag = next(v for k, v in files.items() if k.name == "eval.dag")

    for e in (1, 5, 10):
        assert f"JOB extract_ep{e} " in dag
        assert f"JOB probes_ep{e} " in dag
        assert f"PARENT extract_ep{e} CHILD probes_ep{e}" in dag
    assert "JOB merge " in dag
    # Every probe node is a parent of the single merge, so the table sees the whole campaign.
    assert "PARENT probes_ep1 probes_ep5 probes_ep10 CHILD merge" in dag


def test_epochs_are_independent_so_one_free_gpu_still_makes_progress(tmp_path):
    """No PARENT edge between epochs -- that is what lets a queue with one GPU work through a
    campaign instead of blocking on it."""
    dag = next(v for k, v in build_dag(_plan(tmp_path)).items() if k.name == "eval.dag")
    for a in (1, 5, 10):
        for b in (1, 5, 10):
            if a != b:
                assert f"PARENT extract_ep{a} CHILD extract_ep{b}" not in dag


def test_only_extraction_asks_for_a_gpu(tmp_path):
    """The point of the split: the GPU pass runs once per checkpoint, every metric stays on a
    CPU slot."""
    files = build_dag(_plan(tmp_path))
    subs = {k.name: v for k, v in files.items() if k.suffix == ".sub"}
    def gpus(sub: str) -> str:
        return re.search(r"request_gpus\s*=\s*(\S+)", sub).group(1)

    assert gpus(subs["extract_ep5.sub"]) == "1"
    assert gpus(subs["probes_ep5.sub"]) == "0"
    assert gpus(subs["merge.sub"]) == "0"
    # And only the GPU node carries a device requirement.
    assert "Requirements" in subs["extract_ep5.sub"]
    assert "Requirements" not in subs["probes_ep5.sub"]


def test_every_working_node_retries_and_merge_cannot_fail_the_dag(tmp_path):
    dag = next(v for k, v in build_dag(_plan(tmp_path, retry=3)).items() if k.name == "eval.dag")
    assert "RETRY extract_ep5 3" in dag
    assert "RETRY probes_ep5 3" in dag
    # The merge produces no new data -- its table rebuilds from the JSONs in seconds -- so
    # failing a campaign's completed probe jobs over it would be the wrong trade.
    assert "SCRIPT POST merge" in dag
    assert "RETRY merge" not in dag


def test_the_pre_script_compares_checkpoint_hashes_not_timestamps(tmp_path):
    """v1's "are the features newer than the checkpoint, and has the file stopped growing" is
    defeated by a re-run in one direction and by a slow GPFS write in the other."""
    files = build_dag(_plan(tmp_path))
    dag = next(v for k, v in files.items() if k.name == "eval.dag")
    stale = next(v for k, v in files.items() if k.name == "stale.sh")

    assert "SCRIPT PRE extract_ep5" in dag
    assert "checkpoint_sha256" in stale
    assert "sha256sum" in stale
    # Exit 1 from a PRE script is DAGMan's "skip this node", which is what "already current"
    # has to mean here.
    assert "exit 1" in stale


def test_probe_json_goes_to_one_directory_per_run_not_per_epoch(tmp_path):
    """`wcfm eval merge <run>/probes/*.json` is then the whole trajectory."""
    files = build_dag(_plan(tmp_path))
    subs = {k.name: v for k, v in files.items() if k.suffix == ".sub"}
    for e in (1, 5, 10):
        assert str(tmp_path / "runs" / "hybrid_a" / "probes") in subs[f"probes_ep{e}.sub"]


def test_each_extract_node_is_pinned_to_its_own_epoch(tmp_path):
    """Otherwise every node would re-extract every checkpoint and the DAG would be a loop with
    extra steps."""
    subs = {k.name: v for k, v in build_dag(_plan(tmp_path)).items() if k.suffix == ".sub"}
    assert "'--epochs=5'" in subs["extract_ep5.sub"]
    assert "'--epochs=1'" not in subs["extract_ep5.sub"]


def test_all_nodes_share_one_eval_set_so_the_epochs_are_comparable(tmp_path):
    subs = {k.name: v for k, v in build_dag(_plan(tmp_path)).items() if k.suffix == ".sub"}
    root = str(tmp_path / "runs" / "hybrid_a" / "features" / "eval_set")
    for e in (1, 5, 10):
        assert f"'--eval-set-root={root}'" in subs[f"extract_ep{e}.sub"]


def test_build_dag_writes_nothing(tmp_path):
    """`--dry-run` prints what a real submission queues because it runs the same builder."""
    files = build_dag(_plan(tmp_path))
    assert files
    assert not (tmp_path / "runs" / "hybrid_a" / "features" / "dag").exists()


def test_initialdir_is_set_on_every_node(tmp_path):
    """It is how a queued node is traced back to its run directory (`condor_q -af Iwd`), and
    the directory relative paths in the `.sub` resolve against."""
    subs = [v for k, v in build_dag(_plan(tmp_path)).items() if k.suffix == ".sub"]
    assert subs and all("initialdir" in s for s in subs)


def test_every_node_transfers_the_archive_and_returns_nothing(tmp_path):
    """The two halves of turning file transfer on, and the second is the dangerous one.

    Without `transfer_output_files = ""`, Condor copies everything created at the top of the
    scratch directory back to `initialdir` when the job exits -- and `initialdir` is the
    directory these jobs have already rsynced their real output into. The duplicate would land
    exactly where the correct result goes, so it would look right.
    """
    plan = _plan(tmp_path)
    subs = [v for k, v in build_dag(plan).items() if k.suffix == ".sub"]
    assert subs
    for sub in subs:
        assert "should_transfer_files   = YES" in sub
        assert f"transfer_input_files    = {plan.repo_archive}" in sub
        assert 'transfer_output_files   = ""' in sub
        assert "request_disk" in sub


def test_the_nodes_execute_staged_copies_of_the_job_scripts(tmp_path):
    """Condor transfers the `executable` at job start like any other input, so a node naming
    the live checkout would run whatever that file said hours later -- and bash reads a script
    incrementally, so an edit to a running node's script truncates it silently."""
    plan = _plan(tmp_path)
    files = build_dag(plan)
    job_dir = plan.out_root / "dag" / "job"
    for script in ("evaljob.sh", "probesjob.sh", "mergejob.sh"):
        assert files[job_dir / script] == (REPO / "gridutils" / "eval" / script).read_text()
    subs = [v for k, v in files.items() if k.suffix == ".sub"]
    assert all(f"executable              = {job_dir}/" in s for s in subs)
    assert not any(str(REPO / "gridutils" / "eval") in s for s in subs)


def test_each_node_is_told_the_archive_name_not_a_repo_path(tmp_path):
    """The worker unpacks it from its own scratch; a path into the checkout would not exist
    there, and the whole point is that the checkout is not read at run time."""
    plan = _plan(tmp_path)
    subs = [v for k, v in build_dag(plan).items() if k.suffix == ".sub"]
    args = [re.search(r'arguments += +"(.*)"', s).group(1) for s in subs]
    assert args and all(a.startswith(ARCHIVE_NAME + " ") for a in args)


def test_the_git_identity_reaches_the_workers(tmp_path):
    """They have no `.git` -- the archive excludes it -- so without this every eval job writes
    a null git block into its provenance, on exactly the runs whose provenance matters."""
    plan = _plan(tmp_path, git_env="WCFM_GIT_SHA=abc123 WCFM_GIT_BRANCH=main WCFM_GIT_DIRTY=0")
    subs = [v for k, v in build_dag(plan).items() if k.suffix == ".sub"]
    assert subs and all("WCFM_GIT_SHA=abc123" in s for s in subs)


# --------------------------------------------------------------------------------- wcfm diff


def _write_cfg(path, text):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.yaml").write_text(text)
    return path


def test_diff_reports_only_what_differs(tmp_path, capsys):
    from wcfm.cli.diff import main

    cfg = "run:\n  name: {name}\nmodel:\n  terms:\n    charge:\n      weight: {w}\n"
    a = _write_cfg(tmp_path / "a", cfg.format(name="a", w=0.2))
    b = _write_cfg(tmp_path / "b", cfg.format(name="b", w=0.5))
    assert main([str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "model.terms.charge.weight" in out
    assert "0.2" in out and "0.5" in out
    # run.name differs by construction and is excluded from the default view.
    assert "run.name" not in out


def test_diff_reports_an_absent_key_rather_than_skipping_it(tmp_path, capsys):
    from wcfm.cli.diff import main

    a = _write_cfg(tmp_path / "a", "model:\n  terms:\n    occupancy:\n      weight: 1.0\n")
    b = _write_cfg(tmp_path / "b", "model:\n  terms: {}\n")
    main([str(a), str(b)])
    out = capsys.readouterr().out
    assert "model.terms.occupancy.weight" in out
    assert "<absent>" in out


def test_diff_of_identical_configs_says_so(tmp_path, capsys):
    from wcfm.cli.diff import main

    text = "model:\n  name: hybrid\n"
    a, b = _write_cfg(tmp_path / "a", text), _write_cfg(tmp_path / "b", text)
    assert main([str(a), str(b)]) == 0
    assert "identical" in capsys.readouterr().out


def test_diff_all_shows_the_by_construction_keys(tmp_path, capsys):
    from wcfm.cli.diff import main

    a = _write_cfg(tmp_path / "a", "run:\n  name: a\n  seed: 1\n")
    b = _write_cfg(tmp_path / "b", "run:\n  name: b\n  seed: 1\n")
    main([str(a), str(b), "--all"])
    assert "run.name" in capsys.readouterr().out


def test_diff_needs_exactly_two_runs(tmp_path, capsys):
    from wcfm.cli.diff import main

    assert main([str(tmp_path / "a")]) == 2
    assert "exactly two" in capsys.readouterr().err


def test_diff_code_says_which_archive_is_missing(tmp_path, capsys):
    from wcfm.cli.diff import main

    text = "model:\n  name: hybrid\n"
    a, b = _write_cfg(tmp_path / "a", text), _write_cfg(tmp_path / "b", text)
    assert main([str(a), str(b), "--code"]) == 2
    err = capsys.readouterr().err
    assert "no code archive" in err
    # And it names the fallback, plus the reason the fallback is weaker.
    assert "run_metadata.json" in err and "lower bound" in err


def test_diff_code_compares_the_two_archives(tmp_path, capfd):
    """`--code` looked for a `code/` directory that nothing in this repository ever wrote --
    inherited from the old one -- so it could not succeed on any wcfm run. The archive each
    Condor run keeps is the tree that actually executed."""
    import tarfile

    from wcfm.cli.diff import main

    text = "model:\n  name: hybrid\n"
    runs = []
    for name, body in (("a", "one\n"), ("b", "two\n")):
        run = _write_cfg(tmp_path / name, text)
        src = tmp_path / f"src_{name}"
        (src / "wcfm").mkdir(parents=True)
        (src / "wcfm" / "engine.py").write_text(body)
        job = run / "job"
        job.mkdir(parents=True)
        with tarfile.open(job / ARCHIVE_NAME, "w:gz") as tar:
            tar.add(src / "wcfm", arcname="wcfm")
        runs.append(run)

    assert main([str(runs[0]), str(runs[1]), "--code"]) == 0
    # `capfd`, not `capsys`: `diff` is a subprocess writing to the real file descriptor.
    out = capfd.readouterr().out
    assert "-one" in out and "+two" in out
