"""`wcfm env-check`: what stack is this, really.

Prints versions of the four pinned GPU packages -- built once by `gridutils/build_env.sh`, and
never to be resolved by pip or uv -- and the framework dependencies this repo adds, plus the
CUDA arch actually detected. The same block is recorded in every run's `run_metadata.json`, so
an environment is a fact about the run rather than about the login node it was submitted from.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import pathlib
import platform
import sys
import traceback

import wcfm

PINNED = ("torch", "warpconvnet", "flash_attn", "torch_scatter")
FRAMEWORK = ("hydra", "omegaconf", "lightning_fabric", "numpy", "h5py")


def _installed_version(name: str) -> str | None:
    """The version from package metadata, for a package that is present but will not import."""
    for candidate in (name, name.replace("_", "-")):
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _version(name: str) -> str:
    try:
        mod = importlib.import_module(name)
    except Exception as e:  # noqa: BLE001 - any import failure is the finding
        # A compiled extension linking libcuda.so.1 cannot load where no GPU driver is
        # installed, which is every login node. That is not a missing package, and reporting it
        # as ABSENT sends people to reinstall a stack that is already correct..
        if "libcuda" in "".join(traceback.format_exception(e)):
            found = _installed_version(name)
            if found is not None:
                return f"{found} (installed; needs a GPU driver to import)"
        return f"ABSENT ({type(e).__name__})"
    return str(getattr(mod, "__version__", "?"))


def _plugins() -> list[str]:
    try:
        from wcfm.config.store import register_plugins

        return register_plugins()
    except Exception as e:  # noqa: BLE001
        return [f"ERROR ({type(e).__name__})"]


def collect() -> dict:
    info: dict = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        # The layout is flat, so `import wcfm` resolves to the working tree when the cwd is the
        # repo root and to site-packages otherwise. Record which, or a job that ran a stray
        # checkout is indistinguishable from one that ran the installed copy.
        "wcfm_from": str(pathlib.Path(wcfm.__file__).resolve().parent),
        "pinned": {n: _version(n) for n in PINNED},
        "framework": {n: _version(n) for n in FRAMEWORK},
        # Which config-schema plugins the entry point found. `[]` here with a `wcfm/model`
        # directory present means the egg-info next to `wcfm/` is missing or stale.
        "config_plugins": _plugins(),
    }
    try:
        import torch

        info["cuda"] = {
            "available": torch.cuda.is_available(),
            "version": torch.version.cuda,
            "devices": [
                {
                    "name": torch.cuda.get_device_name(i),
                    "arch": "{}.{}".format(*torch.cuda.get_device_capability(i)),
                }
                for i in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
    except Exception as e:  # noqa: BLE001
        info["cuda"] = {"available": False, "error": type(e).__name__}
    return info


def main(argv: list[str] | None = None) -> int:
    info = collect()
    print(f"python {info['python']}  ({info['platform']})")
    print(f"wcfm    {info['wcfm_from']}")
    print("pinned GPU stack (never resolve these):")
    for k, v in info["pinned"].items():
        print(f"  {k:14s} {v}")
    print("framework:")
    for k, v in info["framework"].items():
        print(f"  {k:14s} {v}")
    plugins = info["config_plugins"] or "NONE -- run `uv pip install -e . --no-deps`"
    print(f"config schema plugins: {plugins}")
    cuda = info["cuda"]
    if cuda.get("available"):
        print(f"cuda {cuda['version']}:")
        for d in cuda["devices"]:
            print(f"  {d['name']}  sm_{d['arch'].replace('.', '')}")
    else:
        print("cuda: not available")
    return 0
