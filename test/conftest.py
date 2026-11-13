"""Pytest bootstrap for the lightweight import smoke tests.

``test/import_smoke_test.py`` is designed to run in a bare environment (only
the standard library and pytest) so that a failed CI dependency-resolution
step cannot mask a real regression in the ``sglang`` top-level import chain.

That contract only holds if the bootstrap runs *before* pytest imports any
module that pulls in ``sglang`` (e.g. ``test/run_suite.py``, which imports
``sglang.test.ci.ci_register`` at module scope).  This conftest performs the
bootstrap at collection time, before test modules are imported, so the smoke
tests stay runnable even when ``pip install -e python`` failed and neither the
``sglang`` distribution metadata nor its heavy runtime dependencies are
available.

Both helpers are idempotent and never shadow a real installation, so this is
safe in a fully provisioned CI environment as well.
"""

import importlib.util
import os
import sys


def _load_smoke_module():
    """Import ``test/import_smoke_test.py`` by file path.

    ``test/`` has no ``__init__.py``, so the module is loaded by path rather
    than by package import.  Loading it executes its module-level bootstrap
    (``_ensure_source_on_path`` + ``_install_lightweight_dependency_stubs``),
    which is exactly what we want before any other test module is imported.
    """
    smoke_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "import_smoke_test.py")
    if not os.path.isfile(smoke_path):
        return None
    existing = sys.modules.get("import_smoke_test")
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location("import_smoke_test", smoke_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules["import_smoke_test"] = module
    spec.loader.exec_module(module)
    return module


# Run the bootstrap as early as possible: at conftest import time, before
# pytest collects and imports the test modules.  This keeps the smoke tests
# self-contained even when the sglang distribution (and its heavy runtime
# dependencies) were never installed.
_smoke = _load_smoke_module()
if _smoke is not None:
    _smoke._ensure_source_on_path()
    _smoke._install_lightweight_dependency_stubs()
