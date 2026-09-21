"""The enforceable rule: framework packages may not import ``wcfm.model``.

Walks every module under the framework packages with ``ast`` -- no imports are executed, so
this runs on a CPU runner with nothing installed beyond the standard library -- and fails on
any ``import wcfm.model...`` or ``from wcfm.model... import`` (including relative imports that
resolve into ``nn``).
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "wcfm"
FRAMEWORK = ("config", "data", "engine", "metrics", "eval", "cli", "plotting")


def _module_name(path: Path) -> str:
    rel = path.relative_to(SRC.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative(module: str, level: int, name: str | None) -> str:
    base = module.split(".")
    # A relative import from a package's __init__ counts the package itself as level 1.
    base = base[: len(base) - level + 1] if level else base
    return ".".join(base + ([name] if name else []))


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    mod = _module_name(path)
    if path.name != "__init__.py":
        mod = mod.rsplit(".", 1)[0]  # package containing this module, for relative resolution
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                found.append(_resolve_relative(mod, node.level, node.module))
            elif node.module:
                found.append(node.module)
    return found


def test_framework_does_not_import_nn():
    offenders: list[str] = []
    for pkg in FRAMEWORK:
        for path in (SRC / pkg).rglob("*.py"):
            for imp in _imports(path):
                if imp == "wcfm.model" or imp.startswith("wcfm.model."):
                    offenders.append(f"{path.relative_to(SRC.parent)} imports {imp}")
    assert not offenders, "framework packages import wcfm.model:\n  " + "\n  ".join(offenders)


def test_framework_packages_exist():
    missing = [pkg for pkg in FRAMEWORK if not (SRC / pkg / "__init__.py").exists()]
    assert not missing, f"framework packages without __init__.py: {missing}"
