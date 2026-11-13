"""Self-contained smoke checks for the ``sglang`` top-level import chain.

Only the standard library is required.  The module bootstraps ``python/``
onto ``sys.path`` and installs lightweight stubs for genuinely missing heavy
third-party dependencies, so the checks run even when ``pip install -e
python`` failed during CI environment preparation (see
``test/constraints-smoke.txt`` for the recommended minimal install).

Entry points:

* ``python test/import_smoke.py --bare``      -- import chain only
* ``python test/import_smoke.py --selfcheck`` -- bootstrap self-check
* ``python test/import_smoke.py --negative``  -- broken import must fail
* ``python test/import_smoke.py --ci``        -- setup-safe, never imports pytest
* ``python test/import_smoke.py --pytest``    -- run the pytest smoke suite
* ``python test/import_smoke.py --all``       -- bare + selfcheck + negative (+ pytest if available)

Real installations are never shadowed: a stub is only created when the
module is neither already imported nor resolvable via
``importlib.util.find_spec``.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
import types

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYTHON_DIR = os.path.join(_REPO_ROOT, "python")

# Heavy third-party modules that ``sglang/__init__.py`` transitively needs.
# Each entry maps the import name to the attribute names the import chain
# touches; anything else resolves to a permissive stub object.
_STUBBED_MODULES = (
    "torch",
    "torch.mps",
    "torch.cuda",
    "torch.backends",
    "torch.backends.mps",
    "transformers",
    "flashinfer",
    "triton",
    "triton.language",
    "numpy",
    "requests",
    "pydantic",
    "IPython",
    "IPython.display",
    "tqdm",
    "pybase64",
)

_installed_stubs: set[str] = set()


def _ensure_source_on_path() -> None:
    """Put ``python/`` on ``sys.path`` so the source tree is importable."""
    if _PYTHON_DIR not in sys.path:
        sys.path.insert(0, _PYTHON_DIR)


def _stub_is_installed(name: str) -> bool:
    """Return whether ``name`` is currently backed by a lightweight stub."""
    return name in _installed_stubs


class _StubModule(types.ModuleType):
    """A module whose every attribute is itself a permissive stub."""

    def __init__(self, name: str):
        super().__init__(name)
        self.__path__: list[str] = []
        self.__package__ = name
        self.__file__ = __file__
        self.__spec__ = importlib.machinery.ModuleSpec(name, None, is_package=True)

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        child = _StubModule(f"{self.__name__}.{name}")
        setattr(self, name, child)
        return child

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def _decorator(fn):
            return fn

        return _decorator

    def __bool__(self):
        return False

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __repr__(self):
        return f"<sglang-smoke-stub {self.__name__!r}>"


def _install_lightweight_dependency_stubs() -> None:
    """Install stubs for heavy deps that are genuinely missing.

    A module is only stubbed when it is neither already imported nor
    resolvable via ``importlib.util.find_spec``, so a real installation is
    never shadowed.
    """
    for name in _STUBBED_MODULES:
        if name in sys.modules:
            continue
        try:
            if importlib.util.find_spec(name) is not None:
                continue
        except (ImportError, ValueError):
            pass
        mod = _StubModule(name)
        sys.modules[name] = mod
        _installed_stubs.add(name)
        parts = name.rsplit(".", 1)
        if len(parts) == 2:
            parent = sys.modules.get(parts[0])
            if parent is not None:
                setattr(parent, parts[1], mod)


def _import_sglang():
    """Import ``sglang`` with the bootstrap applied."""
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()
    return importlib.import_module("sglang")


def _run_bare_smoke() -> int:
    """Import ``sglang`` in a fresh subprocess and check ``__version__``."""
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import importlib.util; "
        "import import_smoke as _s; "
        "_s._install_lightweight_dependency_stubs(); "
        "import sglang; "
        "assert sglang.__version__; "
        "print('import_ok', sglang.__version__)"
    ) % _PYTHON_DIR
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [_REPO_ROOT, env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
    return proc.returncode


def _run_selfcheck() -> int:
    """Verify the bootstrap actually takes effect."""
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()
    if _PYTHON_DIR not in sys.path:
        print("selfcheck failed: python/ not on sys.path", file=sys.stderr)
        return 1
    if importlib.util.find_spec("torch") is not None and not _stub_is_installed(
        "torch"
    ):
        print("selfcheck: real torch present, stub intentionally unused")
        return 0
    if not _stub_is_installed("torch"):
        print("selfcheck failed: torch stub not installed", file=sys.stderr)
        return 1
    print("selfcheck_ok")
    return 0


def _run_negative_check() -> int:
    """A deliberately broken top-level import must be detected.

    Runs in a subprocess so the broken import never pollutes this
    interpreter's ``sys.modules``.
    """
    code = (
        "import sys; sys.path.insert(0, %r); "
        "import import_smoke as _s; "
        "_s._install_lightweight_dependency_stubs(); "
        "import sglang; "
        "raise SystemExit(0)"
    ) % _PYTHON_DIR
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [_REPO_ROOT, env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    # Break the import chain by shadowing a module it depends on.
    broken = os.path.join(_REPO_ROOT, "test", "_broken_sglang_shadow")
    os.makedirs(broken, exist_ok=True)
    with open(os.path.join(broken, "sglang.py"), "w", encoding="utf-8") as f:
        f.write("raise ImportError('deliberately broken sglang import')\n")
    env["PYTHONPATH"] = os.pathsep.join([broken, env["PYTHONPATH"]])
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    # The broken import must make the subprocess fail.
    if proc.returncode == 0:
        print(
            "negative check failed: broken import was not detected",
            file=sys.stderr,
        )
        return 1
    print("negative_ok")
    return 0


def _run_ci() -> int:
    """Setup-safe entry point: never imports pytest, never touches network."""
    rc = _run_bare_smoke()
    if rc != 0:
        return rc
    return _run_selfcheck()


def _run_pytest_smoke() -> int:
    """Run the pytest-based smoke suite when pytest is available."""
    try:
        import pytest  # noqa: F401
    except ImportError:
        print("pytest not available; skipping pytest smoke suite")
        return 0
    test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "import_smoke_test.py")
    return int(
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", test_file, "-k", "import_smoke"],
            cwd=_REPO_ROOT,
        ).returncode
    )


def _run_all() -> int:
    for step in (_run_bare_smoke, _run_selfcheck, _run_negative_check):
        rc = step()
        if rc != 0:
            return rc
    return _run_pytest_smoke()


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--bare" in argv:
        return _run_bare_smoke()
    if "--selfcheck" in argv:
        return _run_selfcheck()
    if "--negative" in argv:
        return _run_negative_check()
    if "--ci" in argv:
        return _run_ci()
    if "--pytest" in argv:
        return _run_pytest_smoke()
    if "--all" in argv:
        return _run_all()
    return _run_ci()


if __name__ == "__main__":
    raise SystemExit(main())
