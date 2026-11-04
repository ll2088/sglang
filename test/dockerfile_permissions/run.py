#!/usr/bin/env python3
"""Standalone entry point for the Dockerfile permission checks.

This wrapper exists so CI can run the checks without pytest and without
installing sglang. It imports only the stdlib-only test module and exits with
its status code.

Usage:
    python3 test/dockerfile_permissions/run.py
"""

import sys
from pathlib import Path

# Make the sibling test module importable regardless of the caller's CWD.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Keep the repository root off sys.path so a stray `import sglang` cannot
# resolve to the in-tree package and start a dependency install.
_REPO_ROOT = str(_HERE.parent.parent)
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
if "sglang" not in sys.modules:
    sys.modules["sglang"] = None

# Importing the test module must not require pytest or sglang; the module is
# stdlib-only by design so this wrapper can run under a bare python3.
from test_dockerfile_permissions import _run_standalone  # noqa: E402


def _print_env() -> None:
    """Print the interpreter and module path so CI logs show the real entry."""
    print(f"python: {sys.executable}")
    print(f"version: {sys.version.split()[0]}")
    print(f"module: {Path(__file__).resolve()}")
    print(f"cwd: {Path.cwd()}")
    # Report whether pytest is importable without importing it: importing pytest
    # here would make the standalone runner depend on a third-party package and
    # could load repository-level plugins. Use importlib.util.find_spec, which
    # only inspects the module search path.
    import importlib.util

    if importlib.util.find_spec("pytest") is not None:
        print("pytest: importable")
    else:
        print("pytest: not importable (stdlib-only run)")


def _print_keyword_hits() -> None:
    """Print per-README keyword hit counts for CI log parity with grep."""
    from test_dockerfile_permissions import readme_keyword_hits_per_file

    for name, count in readme_keyword_hits_per_file().items():
        print(f"readme_keyword_hits[{name}]={count}")


if __name__ == "__main__":
    if "--check-env" in sys.argv[1:]:
        _print_env()
        sys.exit(0)
    if "--keyword-hits" in sys.argv[1:]:
        _print_keyword_hits()
        sys.exit(0)
    _print_env()
    sys.exit(_run_standalone())
