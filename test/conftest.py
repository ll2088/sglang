# Keep pytest collection for the stdlib-only Dockerfile permission checks
# isolated from the rest of the repository. This file intentionally imports
# nothing from sglang so `pytest test/test_dockerfile_permissions.py` can run
# in a minimal environment without installing the full sglang dependency tree.
#
# Do not add repository-wide fixtures or imports here; put them in a dedicated
# conftest.py closer to the tests that need them.

import sys
from pathlib import Path

# Ensure the repository root is not implicitly importable as a package during
# collection. The root contains a `python/` tree and a top-level `sglang`
# namespace only via installed metadata; dropping the repo root from sys.path
# prevents accidental `import sglang` from picking up a partial checkout.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path[:] = [p for p in sys.path if p not in (_REPO_ROOT, "")]
