import sys
from pathlib import Path

# Standalone runner for the stdlib-only Dockerfile permission checks. CI should
# invoke this script (or the module directly) instead of installing sglang:
#
#   python3 test/dockerfile_permissions/run.py
#
# It never imports sglang or pytest.

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_dockerfile_permissions import _run_standalone  # noqa: E402


if __name__ == "__main__":
    sys.exit(_run_standalone())
