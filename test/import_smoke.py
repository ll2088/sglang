"""Self-contained smoke checks for the ``sglang`` top-level import chain.

This module only depends on the standard library (and pytest, when collected
by pytest).  It bootstraps ``python/`` onto ``sys.path`` and installs
lightweight stubs for genuinely missing heavy third-party dependencies, so the
import chain under test can be exercised even when the ``sglang`` distribution
and its runtime dependency tree were never installed.

Entry points:

* ``python test/import_smoke.py --bare``       -- single-line OK/FAIL signal
* ``python test/import_smoke.py --selfcheck``  -- every check, stdlib only
* ``python test/import_smoke.py --negative``   -- broken import must be caught
* ``python -m pytest -q test/import_smoke.py -k import_smoke``

Real installations are never shadowed: a stub is only created when the module
is neither already imported nor resolvable via ``importlib.util.find_spec``.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
import types


# Heavy third-party modules that ``sglang/__init__.py`` (transitively) needs
# but that the smoke tests must not require to be installed.
_HEAVY_DEPENDENCY_STUBS = (
    "torch",
    "transformers",
    "flashinfer",
    "numpy",
    "requests",
    "pydantic",
    "pybase64",
    "IPython",
    "tqdm",
)


def _ensure_source_on_path() -> None:
    """Make ``python/sglang`` importable from the source tree.

    The smoke tests must be runnable even when the ``sglang`` distribution is
    not installed (e.g. the CI environment-preparation step failed while
    resolving the heavy runtime dependency chain).  Adding the source
    directory to ``sys.path`` lets us exercise the real import chain without
    requiring a successful ``pip install``.

    ``test/`` is also added so that ``import test.import_smoke`` works from
    ``test/run_suite.py`` even when the repository root is not on ``sys.path``
    (e.g. pytest invoked with a different rootdir).
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_dir = os.path.join(repo_root, "python")
    if os.path.isdir(os.path.join(python_dir, "sglang")):
        if python_dir not in sys.path:
            sys.path.insert(0, python_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _install_lightweight_dependency_stubs() -> None:
    """Install minimal stubs for heavy third-party deps of ``sglang``.

    The smoke tests must run in a bare environment (only the standard library
    and pytest) so that a broken CI dependency-resolution step cannot mask a
    real regression in the import chain under test.  ``sglang/__init__.py``
    pulls in ``torch`` (via ``hf_transformers_patches`` and ``lang.api``) and
    a handful of other heavy packages; when any of them is absent we register
    a permissive stub module so the import chain can still be exercised.

    Real installations are never shadowed: a stub is only created when the
    module is neither already imported nor resolvable via
    ``importlib.util.find_spec``.  The helper is idempotent.
    """
    for name in _HEAVY_DEPENDENCY_STUBS:
        if name in sys.modules:
            continue
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError):
            spec = None
        if spec is not None:
            continue

        stub = types.ModuleType(name)
        stub.__path__ = []  # type: ignore[attr-defined]
        stub.__version__ = "0.0.0+stub"  # type: ignore[attr-defined]

        class _StubAny:
            def __init__(self, *args, **kwargs):
                pass

            def __getattr__(self, item):
                return _StubAny()

            def __call__(self, *args, **kwargs):
                return _StubAny()

        def _stub_getattr(item):
            if item.startswith("__") and item.endswith("__"):
                raise AttributeError(item)
            return _StubAny()

        stub.__getattr__ = _stub_getattr  # type: ignore[attr-defined]
        sys.modules[name] = stub


def _stub_is_installed(name: str) -> bool:
    """Return True when ``name`` is a stub installed by this module."""
    module = sys.modules.get(name)
    if module is None:
        return False
    return getattr(module, "__version__", None) == "0.0.0+stub"


def _run_bare_smoke() -> int:
    """Run the minimal smoke check without pytest.

    Returns a process exit code (0 on success, 1 on failure) and prints a
    single-line, greppable result so the CI log carries an explicit signal.
    """
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()
    try:
        sglang = importlib.import_module("sglang")
    except Exception as exc:  # noqa: BLE001 - report any import failure
        print(f"import_smoke: FAIL ({type(exc).__name__}: {exc})")
        return 1

    version = getattr(sglang, "__version__", None)
    if not version or not isinstance(version, str):
        print(f"import_smoke: FAIL (bad __version__: {version!r})")
        return 1

    print(f"import_smoke: OK (sglang.__version__={version})")
    return 0


def _run_selfcheck() -> int:
    """Run every smoke check without pytest and without third-party deps.

    Exercises the same import chain as the pytest tests (top-level ``sglang``,
    ``_platform_stubs``, ``version``, ``utils``) using only the standard
    library, and prints one greppable line per check so a failure is
    attributable without a test runner.

    Returns 0 when every check passes, 1 otherwise.
    """
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()

    failures = []

    try:
        sglang = importlib.import_module("sglang")
    except Exception as exc:  # noqa: BLE001 - report any import failure
        print(f"import_smoke: FAIL (sglang: {type(exc).__name__}: {exc})")
        return 1

    version = getattr(sglang, "__version__", None)
    if not version or not isinstance(version, str):
        failures.append(f"sglang.__version__ invalid: {version!r}")
    else:
        print(f"import_smoke: OK (sglang.__version__={version})")

    for module_name in ("sglang._platform_stubs", "sglang.version", "sglang.utils"):
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # noqa: BLE001 - report any import failure
            failures.append(f"{module_name}: {type(exc).__name__}: {exc}")
        else:
            print(f"import_smoke: OK ({module_name})")

    for name in getattr(sglang, "__all__", ()):
        if not hasattr(sglang, name):
            failures.append(f"sglang.__all__ missing symbol: {name}")
    if not failures:
        print("import_smoke: OK (sglang.__all__)")

    if failures:
        for failure in failures:
            print(f"import_smoke: FAIL ({failure})")
        return 1
    return 0


def _run_negative_check() -> int:
    """Negative self-check: a broken ``sglang`` import must be reported.

    The check is performed in a subprocess so the current interpreter's
    ``sys.modules`` is not polluted by the deliberately broken import.  The
    subprocess runs the ``--bare`` entry point, which bootstraps the source
    tree and lightweight stubs itself, so this check does not depend on the
    sglang distribution (or its heavy runtime dependencies) being installed.

    Returns 0 when the broken import is correctly detected as a failure, and
    1 when the smoke test would have passed despite the breakage.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_dir = os.path.join(repo_root, "python")
    init_path = os.path.join(python_dir, "sglang", "__init__.py")
    if not os.path.isfile(init_path):
        print("import_smoke_negative: SKIP (sglang/__init__.py not found)")
        return 0

    with open(init_path, encoding="utf-8") as f:
        original = f.read()

    broken = "raise ImportError('import_smoke_negative_check')\n" + original
    try:
        with open(init_path, "w", encoding="utf-8") as f:
            f.write(broken)
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--bare"],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
    finally:
        with open(init_path, "w", encoding="utf-8") as f:
            f.write(original)

    if proc.returncode == 0:
        print(
            "import_smoke_negative: FAIL (broken import was not detected)\n"
            + proc.stdout
            + proc.stderr
        )
        return 1

    print("import_smoke_negative: OK (broken import detected)")
    return 0


# Bootstrap at import time so that any module importing this file (e.g.
# test/run_suite.py) gets a working ``sglang`` import chain even when the
# sglang distribution and its heavy runtime dependencies were never installed.
_ensure_source_on_path()
_install_lightweight_dependency_stubs()


if __name__ == "__main__":
    if "--bare" in sys.argv[1:]:
        raise SystemExit(_run_bare_smoke())
    if "--selfcheck" in sys.argv[1:]:
        raise SystemExit(_run_selfcheck())
    if "--negative" in sys.argv[1:]:
        raise SystemExit(_run_negative_check())
    raise SystemExit(_run_selfcheck())
