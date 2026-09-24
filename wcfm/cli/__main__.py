"""The `wcfm` entry point. An unknown subcommand says so and lists the ones there are."""

from __future__ import annotations

import sys

COMMANDS = {
    "env-check": "wcfm.cli.env_check",
    "train": "wcfm.cli.train",
    "submit": "wcfm.cli.submit",
    "sweep": "wcfm.cli.sweep",
    "metrics": "wcfm.cli.metrics",
    "eval": "wcfm.cli.eval",
    "plot": "wcfm.cli.plot",
    "diff": "wcfm.cli.diff",
    "test": "wcfm.cli.test_gpu",
    "datagen": "wcfm.cli.datagen",
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: wcfm <command> [args]\n\ncommands:")
        for name in COMMANDS:
            print(f"  {name}")
        return 0
    name, rest = argv[0], argv[1:]
    if name not in COMMANDS:
        print(f"wcfm: unknown command {name!r}; try one of {', '.join(COMMANDS)}", file=sys.stderr)
        return 2
    import importlib

    return importlib.import_module(COMMANDS[name]).main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
