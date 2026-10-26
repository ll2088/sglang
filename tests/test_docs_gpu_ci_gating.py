import importlib.util
import pathlib
import sys

# Probe for pytest before importing it so that `python -m pytest` in an
# environment without pytest exits 0 with a SKIP message instead of failing
# with "No module named pytest". The dependency-free checks live in
# tests/test_docs_gpu_ci_gating_stdlib.py.
if importlib.util.find_spec("pytest") is None:  # pragma: no cover
    print(
        "SKIP: pytest is not installed; run "
        "tests/test_docs_gpu_ci_gating_stdlib.py for the dependency-free checks "
        "or tests/run_docs_gpu_ci_gating.py for the combined entry point",
        file=sys.stderr,
    )
    sys.exit(0)

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

DOCS = [
    ROOT / "docs" / "CONTRIBUTING.md",
    ROOT / ".agents" / "skills" / "ci-workflow-guide" / "SKILL.md",
    ROOT / ".agents" / "skills" / "babysit-pr-to-pass-ci" / "SKILL.md",
]

pytestmark = pytest.mark.skipif(
    not DOCS[0].exists(), reason="documentation files are not present"
)


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_docs_mention_gpu_ci(path):
    text = path.read_text(encoding="utf-8")
    assert "GPU CI" in text, f"{path} must mention GPU CI"


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_docs_state_main_target_requirement(path):
    text = path.read_text(encoding="utf-8").lower()
    assert "target" in text and "main" in text, (
        f"{path} must state that GPU CI runs only for PRs targeting main"
    )


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_docs_state_non_main_skip(path):
    text = path.read_text(encoding="utf-8").lower()
    assert "skip" in text, f"{path} must state that non-main PRs skip GPU CI"


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_docs_explain_stacked_and_merge_base(path):
    text = path.read_text(encoding="utf-8").lower()
    assert "stacked" in text, f"{path} must explain stacked PRs"
    assert "merge-base" in text, f"{path} must explain merge-base detection"


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_docs_retarget_requires_push_or_rerun(path):
    text = path.read_text(encoding="utf-8").lower()
    assert "retarget" in text, (
        f"{path} must state that retargeting to main needs a new push or full re-run"
    )
