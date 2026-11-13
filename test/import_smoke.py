"""Self-contained bootstrap + entry points for the sglang import smoke tests.

This module is the single implementation shared by three entry points:

* ``test/conftest.py`` -- pytest bootstrap (runs before collection)
* ``test/import_smoke_test.py`` -- the pytest test module
* ``test/run_suite.py`` -- the CI suite runner's ``-k import_smoke`` case

It only depends on the standard library.  pytest is needed only to collect
and run the test functions themselves; the ``--bare`` / ``--selfcheck`` /
``--negative`` CLI entry points work with no third-party package at all.

Design constraints (see the optimization brief):

* ``python/`` is injected onto ``sys.path`` so the source tree is imported
directly and the smoke tests never require ``pip install -e python``.
* Lightweight stubs are installed only for heavy third-party dependencies
that are genuinely missing (never shadowing a real installation), so the
import chain can be exercised in a bare environment where the full sglang
runtime dependency tree was never installed.
* A broken ``python/sglang/__init__.py`` must still fail the tests: the
stubs cover third-party modules only, never ``sglang.*`` itself.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import types

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON_DIR = os.path.join(_REPO_ROOT, "python")


def _ensure_source_on_path() -> None:
    """Put ``python/`` on ``sys.path`` so ``import sglang`` uses the source tree.

    Idempotent: repeated calls are cheap and safe.  The source tree is
    prepended (not appended) so it wins over a stale installed copy, but an
    already-imported ``sglang`` is left untouched.
    """
    if _PYTHON_DIR not in sys.path:
        sys.path.insert(0, _PYTHON_DIR)


# ---------------------------------------------------------------------------
# Lightweight dependency stubs
# ---------------------------------------------------------------------------

# Modules that ``sglang/__init__.py`` (and the modules it pulls in) may need
# but that are heavy to install.  Only the ones that are genuinely missing
# are stubbed; a real installation is never shadowed.
_STUBBED_MODULES = (
    "torch",
    "transformers",
    "flashinfer",
    "triton",
    "numpy",
    "requests",
    "pydantic",
    "pybase64",
    "IPython",
    "tqdm",
    "zmq",
    "psutil",
    "uvicorn",
    "fastapi",
    "starlette",
    "aiohttp",
    "httpx",
    "orjson",
    "msgpack",
    "setproctitle",
    "py_spy",
    "quack_kernels",
    "regex",
    "python_multipart",
    "setuptools_scm",
)

# Marker attribute set on every stub module we create, so tests can assert
# that the bootstrap actually took effect.
_STUB_MARKER = "__sglang_import_smoke_stub__"


def _stub_is_installed(module_name: str) -> bool:
    """Return True if ``module_name`` is currently a lightweight stub."""
    mod = sys.modules.get(module_name)
    return bool(mod is not None and getattr(mod, _STUB_MARKER, False))


def _module_is_available(module_name: str) -> bool:
    """Return True if ``module_name`` is already imported or importable."""
    if module_name in sys.modules:
        return True
    try:
        return importlib.util.find_spec(module_name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


class _StubModule(types.ModuleType):
    """A permissive stand-in for a missing heavy dependency.

    Attribute access returns another stub (so ``torch.cuda.Stream`` works),
    calling returns a stub instance, and the module is falsy/empty so that
    ``if torch.cuda.is_available():`` style guards take the safe branch.
    """

    def __init__(self, name: str):
        super().__init__(name)
        setattr(self, _STUB_MARKER, True)
        self.__path__ = []  # look like a package so submodule imports work
        self.__package__ = name
        self.__spec__ = importlib.machinery.ModuleSpec(
            name, None, is_package=True
        )

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        full = f"{self.__name__}.{name}"
        child = sys.modules.get(full)
        if child is None:
            child = _StubModule(full)
            sys.modules[full] = child
        setattr(self, name, child)
        return child

    def __call__(self, *args, **kwargs):
        return _StubModule(f"{self.__name__}.__call__")

    def __bool__(self):
        return False

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __repr__(self):
        return f"<sglang-import-smoke-stub {self.__name__!r}>"


class _StubFinder:
    """Meta-path finder that resolves ``<stubbed>.*`` submodules to stubs."""

    def __init__(self, roots):
        self._roots = tuple(roots)

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".", 1)[0]
        if root not in self._roots:
            return None
        if fullname in sys.modules:
            return getattr(sys.modules[fullname], "__spec__", None)
        mod = _StubModule(fullname)
        sys.modules[fullname] = mod
        parent_name, _, child_name = fullname.rpartition(".")
        if parent_name:
            parent = sys.modules.get(parent_name)
            if parent is not None:
                setattr(parent, child_name, mod)
        return mod.__spec__


_stub_finder_installed = False


def _install_lightweight_dependency_stubs() -> None:
    """Install stubs for genuinely missing heavy dependencies.

    Idempotent and non-destructive: a module that is already imported or
    resolvable via ``importlib.util.find_spec`` is left alone, so a real
    installation is never shadowed.
    """
    global _stub_finder_installed

    stubbed_roots = []
    for name in _STUBBED_MODULES:
        if _module_is_available(name):
            continue
        sys.modules[name] = _StubModule(name)
        stubbed_roots.append(name)

    if stubbed_roots and not _stub_finder_installed:
        sys.meta_path.insert(0, _StubFinder(stubbed_roots))
        _stub_finder_installed = True


# ---------------------------------------------------------------------------
# Smoke checks
# ---------------------------------------------------------------------------

_BARE_SMOKE_SNIPPET = """
import importlib.util
import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_dir = os.path.join(repo_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

import sglang  # noqa: E402

assert sglang.__version__, "sglang.__version__ must be non-empty"
assert isinstance(sglang.__version__, str)
assert sglang.__all__, "sglang.__all__ must be non-empty"
for name in sglang.__all__:
    assert hasattr(sglang, name), f"missing public symbol: {name}"

print("import_ok", sglang.__version__)
"""

_NEGATIVE_SNIPPET = """
import os
import sys

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
python_dir = os.path.join(repo_root, "python")
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)

# Deliberately break the top-level import chain in an isolated interpreter.
import sglang  # noqa: E402

raise SystemExit(0)
"""


def _run_python_snippet(snippet: str, extra_env=None):
    """Run ``snippet`` in a fresh interpreter; return the exit code."""
    env = os.environ.copy()
    env["PYTHONPATH"] = _PYTHON_DIR + os.pathsep + env.get("PYTHONPATH", "")
    if extra_env:
        env.update(extra_env)
    proc = subprocess.run(
        [sys.executable, "-c", snippet],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
    return proc.returncode


def _run_bare_smoke() -> int:
    """Import ``sglang`` in a fresh interpreter; return 0 on success.

    Runs in a subprocess so a partially-initialised ``sglang`` package (or a
    stub installed by an earlier test) can never mask a real import
    regression, and so the check does not depend on the heavy runtime
    dependency tree being installed in this interpreter.
    """
    return _run_python_snippet(_BARE_SMOKE_SNIPPET)


def _run_selfcheck() -> int:
    """Verify the bootstrap itself works; return 0 on success."""
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()

    if _PYTHON_DIR not in sys.path:
        sys.stderr.write("selfcheck failed: python/ not on sys.path\n")
        return 1

    # At least one heavy dependency must be either real or stubbed, otherwise
    # the bootstrap silently did nothing and the smoke tests are vacuous.
    if not any(
        _module_is_available(name) or _stub_is_installed(name)
        for name in ("torch", "transformers", "numpy")
    ):
        sys.stderr.write("selfcheck failed: no dependency stub installed\n")
        return 1

    return 0


def _run_negative_check() -> int:
    """Verify a broken top-level import is detected; return 0 on success.

    Guards the import-error dimension: the smoke tests must fail (not skip)
    when ``python/sglang/__init__.py`` is broken.  The check runs in a
    subprocess so the deliberately broken import never pollutes this
    interpreter's ``sys.modules``.
    """
    snippet = (
        "import os, sys\n"
        "repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))\n"
        "python_dir = os.path.join(repo_root, 'python')\n"
        "if python_dir not in sys.path:\n"
        "    sys.path.insert(0, python_dir)\n"
        "import sglang\n"
        "raise SystemExit(0)\n"
    )
    # Sanity: the unmodified chain must import cleanly first.
    if _run_python_snippet(snippet) != 0:
        sys.stderr.write("negative check failed: baseline import already broken\n")
        return 1

    # Now break the import chain in an isolated copy of the source tree.
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        broken_root = os.path.join(tmp, "repo")
        shutil.copytree(
            os.path.join(_REPO_ROOT, "python"),
            os.path.join(broken_root, "python"),
            symlinks=True,
        )
        init_path = os.path.join(broken_root, "python", "sglang", "__init__.py")
        with open(init_path, "a", encoding="utf-8") as f:
            f.write("\nraise ImportError('deliberately broken for smoke test')\n")

        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.join(broken_root, "python")
        proc = subprocess.run(
            [sys.executable, "-c", "import sglang"],
            cwd=broken_root,
            env=env,
            capture_output=True,
            text=True,
        )

    if proc.returncode == 0:
        sys.stderr.write(
            "negative check failed: broken import was not detected\n"
        )
        return 1
    if "deliberately broken" not in (proc.stderr or ""):
        sys.stderr.write(
            "negative check failed: error message did not surface the cause\n"
        )
        sys.stderr.write(proc.stderr or "")
        return 1
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--bare" in argv:
        return _run_bare_smoke()
    if "--selfcheck" in argv:
        return _run_selfcheck()
    if "--negative" in argv:
        return _run_negative_check()

    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()
    try:
        import pytest
    except ImportError:
        sys.stderr.write(
            "pytest is not installed; run with --bare, --selfcheck or "
            "--negative instead\n"
        )
        return 1
    return pytest.main(["-q", os.path.abspath(__file__), "-k", "import_smoke"])


if __name__ == "__main__":
    raise SystemExit(_main())
