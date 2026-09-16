"""`wcfm eval`: the offline evaluation pipeline -- `extract`, `probe`, `merge` and `submit`.

`extract` scores a run's checkpoints on a fixed set of events and writes the store the rest of
the pipeline reads. Three of its properties are what make a result mean something on its own:

- The run's own `config.yaml` supplies the data, so the events come from the same reader, with
  the same charge transform, that the run trained on, rather than from flags a caller repeats
  by hand at every epoch.
- `--max-images` caps events, never batches. The cap lands on the event count and the reader
  keeps its short tail (`drop_last=False`), so the scored set is the first `max_images` events
  of a fixed stream at any batch size. Capping batches instead lets `--batch-size` choose the
  population. One residual case remains: a cap that does not bind -- `--max-images` at or above
  everything available -- where the sharded reader's own implicit tail drop is still batch-size
  dependent. Extraction detects exactly that and says so, rather than leaving it to be
  discovered from a hash mismatch two runs later.
- Every checkpoint of a run shares one eval set, written once. Point `--eval-set-root` at a
  shared location to make two runs comparable by construction; the key hash then refuses the
  pairing if the sets ever drift apart.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

__all__ = ["main"]

USAGE = """usage: wcfm eval <subcommand> [options]

  wcfm eval extract <run_dir> [options]     score checkpoints, write a feature store
  wcfm eval probe <store> [options]         run the probe suite on an extraction
  wcfm eval merge <results.json...>         tabulate probe results
  wcfm eval compare <results.json...>       merge, plus sweep and seed-replica views
  wcfm eval submit <run_dir> [options]      queue the extract -> probes -> merge DAG

probe options:
  --stages=pid,knn,...  which probes            (default: every stage)
  --out-dir=P           where the JSONs go      (default: the store's ../../probes)
  --source=NAME         branch to score         (default: student)
  --tap=NAME            tap to score            (default: out)
  --seed=N              seeds the HEADS, not the population  (default: 0)
  --device=cpu|cuda     (default: cpu)

submit options:
  --epochs=1,5,10       which checkpoints        (default: all in checkpoints/)
  --repo=P              the checkout to package for the jobs (default: $WCFM_REPO or this one)
  --stages=pid,knn,...  which probes             (default: every stage)
  --sources=a,b         branches to extract      (default: student,teacher)
  --taps=a,b            extra taps               (default: none)
  --rows=all|pooled     row space                (default: all)
  --max-images=N        cap on EVENTS            (default: 10000)
  --eval-set-root=P     share an eval set across runs
  --retry=N             Condor RETRY per node    (default: 2)
  --dry-run             write the DAG, submit nothing

merge/compare options:
  --csv=PATH            also write the table as CSV
  --markdown            emit a markdown table
  --group-by-seed       collapse seed replicas into mean +- spread
  --sweep=ID|PATH       add one column per axis declared in a sweep manifest, and use its
                        declared seed axis for --group-by-seed instead of guessing
  --by-config           add one column per config key that differs across the runs
  --runs-root=P         where to find <run>/config.yaml for --by-config (default: cwd)

extract options:
  --epochs=1,5,10     which checkpoints (default: all in checkpoints/, latest.pt excluded)
  --sources=a,b       branches to write            (default: student,teacher; missing dropped)
  --taps=a,b          named intermediates as well as the final map  (default: none)
  --max-images=N      cap on EVENTS, not batches   (default: 10000)
  --batch-size=N      throughput only   (default: the run's PER-RANK batch, because extraction
                      is one process; warpconvnet's hash table packs the batch index into 9 bits
                      and refuses more than 512 images in one forward)
  --num-workers=N     loader workers               (default: 4)
  --rows=all|pooled   write every pixel, or only the pooled rows      (default: all)
  --pool-per-class=N  balanced pool size per class (default: 10000, probe_pid's own)
  --pool-seed=N       seed for the split and pools (default: 42)
  --device=cuda|cpu   (default: cuda if available)
  --eval-set-root=P   share an eval set across runs (default: <run_dir>/features/eval_set)
  --out-root=P        where feature stores go       (default: <run_dir>/features)
  --gradients=N       also run the offline gradient probe over N batches (default: 0=off)
  --dry-run           resolve and print the plan, touch no GPU

`run_dir` is the run's own directory -- the one holding checkpoints/, features/ and config.yaml.
"""


def _flags(argv: list[str]) -> tuple[dict[str, str], list[str]]:
    flags, rest = {}, []
    for a in argv:
        if a.startswith("--"):
            key, _, value = a[2:].partition("=")
            flags[key] = value if value != "" else "true"
        else:
            rest.append(a)
    return flags, rest


def _checkpoints(run_dir: Path, epochs: str) -> list[Path]:
    """Which checkpoints to score.

    `latest.pt` is excluded even when it is the only file: it is a duplicate of some
    `checkpoint_epochN.pt` under a name that changes meaning between runs, and a features
    directory called `latest` would silently mean a different epoch in each of them.
    """
    ckpt_dir = run_dir / "checkpoints"
    found = sorted(
        ckpt_dir.glob("checkpoint_epoch*.pt"),
        key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0),
    )
    if not epochs or epochs == "true":
        return found
    want = {int(e) for e in epochs.split(",") if e.strip()}
    by_epoch = {int("".join(c for c in p.stem if c.isdigit()) or 0): p for p in found}
    missing = sorted(want - set(by_epoch))
    if missing:
        raise SystemExit(
            f"wcfm eval extract: {run_dir.name} has no checkpoint for epochs {missing}; it has "
            f"{sorted(by_epoch)}. `save_at` in the run config is what makes a sweep's points "
            "land on the same epochs."
        )
    return [by_epoch[e] for e in sorted(want)]


def _charge_transform(cfg) -> tuple[str, dict]:
    """The charge transform, as a comparison string and as its actual parameters.

    The string is what `check_comparability` groups on -- the raw and delta columns are only
    comparable across runs that fed the backbone the same transform. The parameters are what
    the raw-charge baseline rebuilds the real input from, and it needs the floats rather than
    their printed form.
    """
    norm = cfg.get("model", {}).get("normalize") if hasattr(cfg, "get") else None
    if not norm or not norm.get("enabled", True):
        return "none", {}
    lo, hi = norm.get("min_val"), norm.get("max_val")
    if lo is None or hi is None:
        return "none", {}
    return f"log[{lo},{hi}]", {"kind": "log", "min_val": float(lo), "max_val": float(hi)}


def _git_sha(run_dir: Path) -> str:
    import json

    path = run_dir / "run_metadata.json"
    if not path.exists():
        return ""
    try:
        return str((json.loads(path.read_text()).get("git") or {}).get("sha", "") or "")
    except (ValueError, AttributeError):
        return ""


def _extract(argv: list[str]) -> int:
    from omegaconf import OmegaConf

    from wcfm.config.io import per_rank_batch_size

    # Imported here, not at module scope: `wcfm.eval.extract` pulls in torch, and
    # `wcfm eval --help` has to work in the config-only environment where there is none.
    from wcfm.eval.extract import DEFAULT_POOL_PER_CLASS

    flags, rest = _flags(argv)
    if not rest:
        print("wcfm eval extract: needs a run directory", file=sys.stderr)
        return 2
    run_dir = Path(rest[0])
    cfg_path = run_dir / "config.yaml"
    if not cfg_path.exists():
        print(
            f"wcfm eval extract: no config.yaml at {cfg_path}. Extraction reads the run's own "
            "resolved config so the events and the charge transform match the ones it trained "
            "on; a run directory without it cannot be scored faithfully.",
            file=sys.stderr,
        )
        return 2
    cfg = OmegaConf.load(cfg_path)

    checkpoints = _checkpoints(run_dir, flags.get("epochs", ""))
    if not checkpoints:
        print(
            f"wcfm eval extract: no checkpoint_epoch*.pt in {run_dir / 'checkpoints'}",
            file=sys.stderr,
        )
        return 2

    max_images = int(flags.get("max-images", 10000))
    num_workers = int(flags.get("num-workers", 4))
    out_root = Path(flags.get("out-root", run_dir / "features"))
    eval_set_root = Path(flags.get("eval-set-root", out_root / "eval_set"))
    sources = tuple(s for s in flags.get("sources", "student,teacher").split(",") if s)
    taps = tuple(t for t in flags.get("taps", "").split(",") if t)
    rows = flags.get("rows", "all")
    pool_per_class = int(flags.get("pool-per-class", DEFAULT_POOL_PER_CLASS))
    pool_seed = int(flags.get("pool-seed", 42))
    gradient_batches = int(flags.get("gradients", 0))

    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data, resolve=True))
    if "batch-size" in flags:
        data_cfg.global_batch_size = int(flags["batch-size"])
    else:
        # Extraction is ONE process, so it must not inherit a batch sized for the whole world.
        # `global_batch_size` grows with `launch.devices`, so it is divided by the number of devices 
        # to get the per-rank batch size.
        devices = int(OmegaConf.select(cfg, "launch.devices", default=1) or 1)
        data_cfg.global_batch_size = per_rank_batch_size(
            int(cfg.data.global_batch_size), devices
        )
    batch_size = int(data_cfg.global_batch_size)
    # The cap on the READ. The loop caps the event count itself as well, because the sharded
    # reader rounds `n_subset` up to a whole shard -- so this is an optimisation, not the rule.
    if max_images > 0:
        data_cfg.n_subset = max_images
    # Unconditionally, not just under `rows=pooled`: the pools are drawn from `pixel_labels`
    # whatever the row space is, and drawing them at extraction time is what makes every probe
    # score the same population. Without this the default invocation writes a `pools.npz` with
    # nothing in it but `row_index`, and the suite is back to each probe redrawing its own.
    data_cfg.return_pixel_truth = True

    device = flags.get("device") or ("cuda" if _cuda() else "cpu")
    seed = int(cfg.get("run", {}).get("seed", 42))
    charge_transform, charge_params = _charge_transform(cfg)
    # `apa`/`view` describe the production, so they come off the data config and are recorded
    # in provenance: `probe_vertex` cannot project the true vertex into the image without them,
    # and a caller asked to repeat them per epoch will eventually not.
    apa = int(data_cfg.get("apa", -1))
    view = str(data_cfg.get("view", "") or "")

    print(f"run          {run_dir}")
    print(f"checkpoints  {[p.name for p in checkpoints]}")
    print(f"eval set     {eval_set_root}  (max_images={max_images}, batch_size={batch_size})")
    print(f"sources      {list(sources)}   taps {list(taps) or ['out only']}   rows={rows}")
    print(
        f"pools        per_class={pool_per_class} seed={pool_seed}  "
        f"pixel_truth={bool(data_cfg.return_pixel_truth)}  drop_last=False"
    )
    grad_note = f"   gradients over {gradient_batches} batches" if gradient_batches else ""
    print(f"device       {device}{grad_note}")
    print(f"geometry     apa={apa} view={view or '?'}   charge {charge_transform}")
    if flags.get("dry-run"):
        print("dry run: nothing read, nothing written")
        return 0

    from wcfm.data.build import build_loader
    from wcfm.eval.extract import extract

    git_sha = _git_sha(run_dir)
    for ckpt in checkpoints:
        epoch = "".join(c for c in ckpt.stem if c.isdigit())
        store_root = out_root / f"epoch{epoch}"
        # A fresh loader per checkpoint: `shuffle=False` makes the event sequence identical on
        # every pass, which is what lets one eval set serve them all.
        loader = build_loader(
            data_cfg, num_workers=num_workers, shuffle=False, seed=seed, drop_last=False
        )
        res = extract(
            ckpt,
            store_root=store_root,
            eval_set_root=eval_set_root,
            loader=loader,
            sources=sources,
            taps=taps,
            max_images=max_images,
            rows=rows,
            pool_per_class=pool_per_class,
            pool_seed=pool_seed,
            device=device,
            batch_size=batch_size,
            seed=seed,
            charge_transform=charge_transform,
            charge_transform_params=charge_params,
            gradient_batches=gradient_batches,
            apa=apa,
            view=view,
            git_sha=git_sha,
            progress=True,
        )
        print(
            f"  {ckpt.name}: {res.n_events} events, {res.n_pixels} pixels, "
            f"{res.n_rows} rows, {res.provenance.extract_seconds:.1f} s "
            f"({res.provenance.events_per_second:.1f} ev/s) -> {store_root}"
        )
    print(f"eval set {eval_set_root} ({checkpoints[0].parent.parent.name})")
    return 0


def _cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _probe(argv: list[str]) -> int:
    from wcfm.eval.probes.runner import DEFAULT_STAGES, run_stages

    flags, rest = _flags(argv)
    if not rest:
        print("wcfm eval probe: needs a feature store directory", file=sys.stderr)
        return 2
    stores = [str(Path(r)) for r in rest]
    # Default beside the run, not beside the store: probe JSONs for every epoch belong in one
    # directory, which is what makes `wcfm eval merge <run>/probes/*.json` the whole trajectory.
    default_out = Path(stores[0]).parent.parent / "probes"
    return run_stages(
        stores,
        stages=flags.get("stages", DEFAULT_STAGES),
        out_dir=flags.get("out-dir", str(default_out)),
        source=flags.get("source", "student"),
        tap=flags.get("tap", "out"),
        seed=int(flags.get("seed", 0)),
        device=flags.get("device", "cpu"),
    )


def _merge(argv: list[str], *, allow_views: bool) -> int:
    from wcfm.eval.compare import (
        build_rows,
        check_comparability,
        group_by_seed,
        load_all,
        render,
        run_of,
        write_csv,
    )

    flags, rest = _flags(argv)
    if not rest:
        print("wcfm eval merge: needs probe result JSON file(s)", file=sys.stderr)
        return 2
    merged = load_all(rest)
    if not merged:
        print("wcfm eval merge: no results found", file=sys.stderr)
        return 2

    # Raises on the two axes that make the table meaningless, before anything is printed: a
    # warning above numbers nobody can interpret scrolls away.
    for w in check_comparability(merged):
        print(f"[warn] {w}")

    manifest = None
    extra: dict[str, dict] = {}
    if allow_views and flags.get("sweep"):
        from wcfm.cli.sweep import load_manifest
        from wcfm.eval.compare import sweep_columns

        manifest = load_manifest(_sweep_path(flags["sweep"]))
        extra.update(sweep_columns(manifest, merged))
        covered = sum(1 for label in merged if run_of(label) in (manifest["points"] or {}))
        print(
            f"[sweep] {manifest['sweep_id']}: {manifest['n_points']} point(s), axes "
            f"{manifest['axes']}; {covered} of {len(merged)} row(s) in this table belong to it"
        )
        missing = sorted(set(manifest["points"]) - {run_of(label) for label in merged})
        if missing:
            # A campaign with a point that produced no result is the thing a manifest exists to
            # make visible; a table built only from the JSONs that exist cannot show a gap.
            print(f"[sweep] {len(missing)} point(s) have NO results here: {missing}")

    if allow_views and flags.get("by-config"):
        for label, cols in _config_columns(merged, Path(flags.get("runs-root", "."))).items():
            extra.setdefault(label, {}).update(cols)

    header, rows, notes = build_rows(merged, extra or None)
    for n in notes:
        print(f"[note] {n}")
    if allow_views and flags.get("group-by-seed"):
        if manifest is None:
            print(
                "[note] --group-by-seed without --sweep: replica families are GUESSED by "
                "stripping a `_seed<N>` suffix off the run name. Pass --sweep for a campaign "
                "whose points were named by `wcfm sweep`, where the seed axis is declared."
            )
        header, rows = group_by_seed(header, rows, manifest)

    print(
        "\nraw baseline = [channel, tick, log_charge] per pixel (wcfm/eval/rawcharge.py)."
        "  d_ = feat - raw.\n"
    )
    print(render(header, rows, markdown=bool(flags.get("markdown"))))
    if flags.get("csv"):
        print(f"\nwrote {write_csv(flags['csv'], header, rows)}")
    if allow_views and flags.get("by-config") and not extra:
        print(
            "\n[note] --by-config found nothing to add: every run in this table resolved to "
            "the same config, or their config.yaml files were not under --runs-root."
        )
    return 0


def _sweep_path(value: str) -> Path:
    """A sweep id or a path. An id is resolved under $WCFM_SWEEPS (or ./sweeps)."""
    p = Path(value)
    if p.exists():
        return p
    return Path(os.environ.get("WCFM_SWEEPS", "sweeps")) / value


def _config_columns(merged: dict, runs_root: Path) -> dict:
    """One column per config key that differs across the runs in the table.

    Reads each run's own `config.yaml`. A run whose config is not found is simply left out of
    the comparison rather than failing the table -- the probe numbers are still valid, and the
    table is a view.
    """
    from wcfm.eval.compare import run_of
    from wcfm.eval.config_diff import differing_keys, load_run_config

    configs = {}
    for label in merged:
        run = run_of(label)
        try:
            configs[run] = load_run_config(runs_root / run)
        except SystemExit:
            continue
    keys = differing_keys(configs)
    if not keys:
        return {}
    return {
        label: {k: configs[run_of(label)].get(k, "-") for k in keys}
        for label in merged
        if run_of(label) in configs
    }


def _submit(argv: list[str]) -> int:
    import shutil
    import subprocess

    from wcfm.cli.jobpack import ARCHIVE_NAME, check_repo, git_environment, pack_repo
    from wcfm.eval.dag import DagPlan, build_dag, checkpoints_of

    flags, rest = _flags(argv)
    if not rest:
        print("wcfm eval submit: needs a run directory", file=sys.stderr)
        return 2
    run_dir = Path(rest[0]).resolve()
    if not (run_dir / "config.yaml").exists():
        print(
            f"wcfm eval submit: no config.yaml at {run_dir}. Extraction reads the run's own "
            "resolved config, so a directory without one cannot be scored.",
            file=sys.stderr,
        )
        return 2

    checkpoints = checkpoints_of(run_dir, flags.get("epochs", ""))
    if not checkpoints:
        print(f"wcfm eval submit: no checkpoint_epoch*.pt in {run_dir}", file=sys.stderr)
        return 2

    # The checkout is packaged, not read in place: every node transfers one archive of this
    # tree (`wcfm/cli/jobpack.py`) and unpacks it into its own scratch. A DAG's later nodes
    # start hours after submission, and Condor reads `transfer_input_files` at job start, so
    # the archive -- written once, below -- is what pins them.
    repo = Path(
        flags.get("repo") or os.environ.get("WCFM_REPO") or Path(__file__).resolve().parents[2]
    ).resolve()
    try:
        # Before the DAG is built, not after: `build_dag` copies the three job scripts out of
        # this tree, and a `--dry-run` that cannot package is not a dry run of anything.
        check_repo(repo)
    except FileNotFoundError as exc:
        print(f"wcfm eval submit: {exc}", file=sys.stderr)
        return 2
    out_root = Path(flags.get("out-root", run_dir / "features"))
    job_dir = out_root / "dag" / "job"
    user = os.environ.get("USER", "unknown")
    plan = DagPlan(
        run_dir=run_dir,
        out_root=out_root,
        repo=repo,
        repo_archive=job_dir / ARCHIVE_NAME,
        git_env=git_environment(repo),
        # `$USER` in a default has to be interpolated here: nothing expands it downstream, so a
        # literal one reaches Condor as a directory named `$USER`.
        pyenv=Path(os.environ.get("WCFM_PYENV", f"/gpfs01/lbne/users/fm/{user}/uvenv")),
        # `WCFM_CACHE_DIR`, the same name `wcfm submit` and `wcfm test` read. The default stays
        # the run's own sibling, which is where existing extractions already cache.
        cache=Path(os.environ.get("WCFM_CACHE_DIR", str(run_dir.parent / ".cache"))),
        checkpoints=checkpoints,
        eval_set_root=Path(flags.get("eval-set-root", out_root / "eval_set")),
        stages=flags.get("stages", "pid,knn,overlap,instance,vertex,event,spectrum"),
        sources=flags.get("sources", "student,teacher"),
        taps=flags.get("taps", ""),
        rows=flags.get("rows", "all"),
        max_images=int(flags.get("max-images", 10000)),
        retry=int(flags.get("retry", 2)),
    )

    files = build_dag(plan)
    dag_path = next(p for p in files if p.name == "eval.dag")

    print(f"run:      {run_dir}")
    print(f"repo:     {repo}")
    print(f"epochs:   {plan.epochs}")
    print(f"stages:   {plan.stages}")
    print(f"dag:      {dag_path}")
    if flags.get("dry-run"):
        print("\ndry run: nothing written, nothing submitted\n")
        print(files[dag_path])
        return 0

    try:
        archive = pack_repo(repo, job_dir)
    except FileNotFoundError as exc:
        print(f"wcfm eval submit: {exc}", file=sys.stderr)
        return 2
    print(f"archive:  {archive} ({archive.stat().st_size / 1e6:.1f} MB)")

    (dag_path.parent / "logs").mkdir(parents=True, exist_ok=True)
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        if path.suffix == ".sh":
            path.chmod(0o755)
    print(f"\nwrote {len(files)} files under {dag_path.parent}")

    if not shutil.which("condor_submit_dag"):
        print(
            "wcfm eval submit: condor_submit_dag not on PATH; submit from a login node with\n"
            f"  condor_submit_dag {dag_path}",
            file=sys.stderr,
        )
        return 2
    completed = subprocess.run(["condor_submit_dag", str(dag_path)], check=False)
    print(f"\nWatch with:  condor_q -nobatch ; tail -F {dag_path.parent}/logs/*.out")
    return completed.returncode


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    sub, rest = argv[0], argv[1:]
    if sub == "extract":
        return _extract(rest)
    if sub == "probe":
        return _probe(rest)
    if sub == "merge":
        return _merge(rest, allow_views=False)
    if sub == "compare":
        return _merge(rest, allow_views=True)
    if sub == "submit":
        return _submit(rest)
    print(
        f"wcfm eval: unknown subcommand {sub!r}. "
        "Known: extract, probe, merge, compare, submit.",
        file=sys.stderr,
    )
    return 2
