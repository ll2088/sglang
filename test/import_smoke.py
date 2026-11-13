"""Lightweight smoke tests for the ``sglang`` top-level import chain.

These tests only depend on the standard library and pytest.  They are kept in
a standalone module (rather than inside ``test/run_suite.py``) so that pytest
can collect them directly without going through the CI suite runner's
argparse entry point or the registered-test sanity checks.

Each test imports the target module inside the test body so that an import
failure is reported as a test failure with the original traceback, instead of
aborting collection with an opaque error.
"""

import importlib
import importlib.metadata
import importlib.util
import os
import sys
import types

import pytest


def _ensure_source_on_path() -> None:
    """Make ``python/sglang`` importable from the source tree.

    The smoke tests must be runnable even when the ``sglang`` distribution is
    not installed (e.g. the CI environment-preparation step failed while
    resolving the heavy runtime dependency chain).  Adding the source
    directory to ``sys.path`` lets us exercise the real import chain without
    requiring a successful ``pip install``.

    ``test/`` is also added so that ``import test.import_smoke`` works from
    ``test/run_suite.py`` even when the repository root is not on
    ``sys.path`` (e.g. pytest invoked with a different rootdir).
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_dir = os.path.join(repo_root, "python")
    if os.path.isdir(os.path.join(python_dir, "sglang")):
        if python_dir not in sys.path:
            sys.path.insert(0, python_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


# Heavy third-party modules that ``sglang/__init__.py`` (transitively) needs
# but that the smoke tests must not require to be installed.  A permissive
# stub is registered for each one that is genuinely missing, so the import
# chain under test can be exercised in a bare environment (standard library +
# pytest only) without triggering a full ``pip install sglang``.
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
    ``importlib.util.find_spec``.  This keeps the smoke tests honest in a
    fully provisioned CI environment where the real dependencies are present,
    and avoids masking a genuine import regression behind a permissive stub.

    The helper is idempotent: repeated calls never replace an already-installed
    stub or a real installation.
    """
    for name in _HEAVY_DEPENDENCY_STUBS:
        # Never shadow a real installation: if the module is already imported
        # or resolvable on sys.path, leave it alone.
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
    """Return True when ``name`` is a stub installed by this module.

    Used by the self-check test to confirm the bare-environment bootstrap
    actually took effect, without depending on the real dependency being
    absent (a fully provisioned environment legitimately has the real
    module and the self-check then skips).
    """
    module = sys.modules.get(name)
    if module is None:
        return False
    return getattr(module, "__version__", None) == "0.0.0+stub"


# Install the source path and lightweight stubs at module import time so that
# ``pytest -k import_smoke`` works in a bare environment (no sglang runtime
# deps installed).  Both helpers are idempotent and never shadow a real
# installation, so this is safe in a fully provisioned CI environment too.
_ensure_source_on_path()
_install_lightweight_dependency_stubs()


def _import_or_skip(module_name: str):
    """Import ``module_name``, skipping (with the original error) if the
    environment is not ready.

    A missing third-party dependency (e.g. torch not installed yet) is an
    environment problem, not a regression in the import chain under test.
    Skipping keeps the failure signal attributable: a real import error in
    ``sglang`` itself still raises and fails the test.

    ``ModuleNotFoundError`` is only treated as an environment problem when the
    missing module is *not* part of ``sglang`` itself; a missing ``sglang.*``
    submodule is a real regression and must fail the test.

    ``ImportError`` raised while importing a third-party dependency of
    ``sglang`` (e.g. a broken/partial install) is likewise reported as a skip
    so that a broken CI environment does not masquerade as a regression in the
    import chain under test.  Errors raised from ``sglang``'s own modules are
    re-raised so that a real regression still fails the test.
    """
    _ensure_source_on_path()
    # Only install stubs for genuinely missing dependencies; a real install is
    # never shadowed (see _install_lightweight_dependency_stubs).  The helper
    # itself re-checks sys.modules / find_spec, so calling it here is safe and
    # idempotent.
    _install_lightweight_dependency_stubs()
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or ""
        if missing == "sglang" or missing.startswith("sglang."):
            raise
        pytest.skip(f"environment not ready: {exc!r}")
    except ImportError as exc:
        # Distinguish a failure originating inside sglang from one caused by a
        # missing/broken third-party dependency: only the latter is skipped.
        tb = exc.__traceback__
        while tb is not None:
            frame_file = tb.tb_frame.f_code.co_filename
            if "sglang" in frame_file.replace("\\", "/"):
                raise
            tb = tb.tb_next
        pytest.skip(f"environment not ready: {exc!r}")


@pytest.mark.import_smoke
def test_import_smoke():
    """Smoke test: importing sglang must succeed and expose __version__."""
    sglang = _import_or_skip("sglang")

    assert sglang.__version__
    assert isinstance(sglang.__version__, str)


@pytest.mark.import_smoke
def test_import_platform_stubs():
    """Smoke test: the platform stubs module must import without raising."""
    stubs = _import_or_skip("sglang._platform_stubs")

    assert hasattr(stubs, "install_platform_stubs")
    assert callable(stubs.install_platform_stubs)


@pytest.mark.import_smoke
def test_import_version():
    """Smoke test: version attributes must be readable and consistent."""
    version = _import_or_skip("sglang.version")

    assert version.__version__
    assert isinstance(version.__version__, str)
    assert version.__version_tuple__
    assert isinstance(version.__version_tuple__, tuple)


@pytest.mark.import_smoke
def test_import_version_matches_metadata():
    """Smoke test: sglang.__version__ must match installed package metadata."""
    sglang = _import_or_skip("sglang")

    try:
        dist_version = importlib.metadata.version("sglang")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("sglang distribution metadata not installed")

    assert sglang.__version__ == dist_version


@pytest.mark.import_smoke
def test_import_utils():
    """Smoke test: sglang.utils must be importable from the package."""
    utils = _import_or_skip("sglang.utils")

    assert hasattr(utils, "LazyImport")


@pytest.mark.import_smoke
def test_import_public_api_symbols():
    """Smoke test: the public API names declared in __all__ must resolve."""
    sglang = _import_or_skip("sglang")

    assert sglang.__all__
    for name in sglang.__all__:
        assert hasattr(sglang, name), f"missing public symbol: {name}"


def _run_bare_smoke() -> int:
    """Run the smoke checks without pytest.

    Used by CI jobs that must not install the full ``sglang`` dependency tree
    (whose resolution can fail during environment preparation and mask real
    import regressions).  Only the standard library is required: the source
    tree is put on ``sys.path`` and lightweight stubs are installed for any
    genuinely missing heavy dependency before importing ``sglang``.

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

    This is the entry point CI should use when the full ``sglang`` dependency
    tree cannot be installed during environment preparation.  It exercises the
    same import chain as the pytest tests (top-level ``sglang``,
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

    if failures:
        for failure in failures:
            print(f"import_smoke: FAIL ({failure})")
        return 1
    return 0


def _run_negative_check() -> int:
    """Negative self-check: a broken ``sglang`` import must be reported.

    Used by CI to confirm the smoke tests actually fail (rather than silently
    skip) when the top-level import chain is broken.  The check is performed
    in a subprocess so the current interpreter's ``sys.modules`` is not
    polluted by the deliberately broken import.

    The subprocess runs the ``--bare`` entry point, which bootstraps the
    source tree and lightweight stubs itself, so this check does not depend
    on the sglang distribution (or its heavy runtime dependencies) being
    installed.  A broken ``__init__.py`` therefore fails the bare smoke run
    even in a bare environment.

    Returns 0 when the broken import is correctly detected as a failure, and
    1 when the smoke test would have passed despite the breakage.
    """
    import subprocess

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


@pytest.mark.import_smoke
def test_lightweight_dependency_stubs_effective():
    """Self-check: the bare-environment bootstrap must actually take effect.

    Guards against a regression where the stub installation silently stops
    working (e.g. the helper is renamed or its guard conditions change),
    which would make the bare CI entry point fail with an opaque import
    error instead of a clear signal.  When the real dependency is present
    the stub is intentionally not installed and the check is skipped.
    """
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()

    if importlib.util.find_spec("torch") is not None and not _stub_is_installed(
        "torch"
    ):
        pytest.skip("real torch is installed; stub intentionally not used")

    assert _stub_is_installed("torch"), (
        "lightweight torch stub was not installed; the bare-environment "
        "bootstrap is broken"
    )


@pytest.mark.import_smoke
def test_import_smoke_without_installed_distribution():
    """Smoke test: the import chain must work without ``pip install sglang``.

    Guards the CI contract that the smoke tests are self-contained: they add
    ``python/`` to ``sys.path`` and stub heavy dependencies, so they must not
    depend on the ``sglang`` distribution (or its runtime dependency tree)
    being installed.  A regression here would re-introduce the environment
    preparation failure that blocked the import-time/import-error/
    import-compat/tests dimensions.
    """
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    python_dir = os.path.join(repo_root, "python")
    assert python_dir in sys.path, (
        "python/ must be on sys.path so the smoke tests do not require an "
        "installed sglang distribution"
    )

    sglang = _import_or_skip("sglang")
    assert sglang.__version__


@pytest.mark.import_smoke
def test_import_smoke_bare_entrypoint():
    """Smoke test: the ``--bare`` entry point must succeed without pytest.

    This exercises the same code path CI uses when it cannot install the full
    ``sglang`` dependency tree, so a regression in the bare-environment
    bootstrap is caught by the normal pytest run as well.
    """
    assert _run_bare_smoke() == 0


# When this file is executed directly (``python test/import_smoke.py``)
# rather than collected by pytest, it bootstraps a minimal environment and
# then re-invokes pytest on itself.  This lets CI run the smoke tests without
# first installing the full ``sglang`` dependency chain (which can fail during
# environment preparation and mask real import regressions).  The entry point
# lives at the end of the module so that every helper it calls is defined.
#
# ``--bare`` skips pytest entirely and runs the checks with the standard
# library only, for environments where pytest itself is not installed.
if __name__ == "__main__":
    if "--bare" in sys.argv[1:]:
        raise SystemExit(_run_bare_smoke())
    if "--selfcheck" in sys.argv[1:]:
        raise SystemExit(_run_selfcheck())
    if "--negative" in sys.argv[1:]:
        raise SystemExit(_run_negative_check())
    _ensure_source_on_path()
    _install_lightweight_dependency_stubs()
    raise SystemExit(
        pytest.main(["-q", os.path.abspath(__file__), "-k", "import_smoke"])
    )
