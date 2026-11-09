"""Compatibility checks for the anthropic dependency bump (0.20 -> 2.6).

The repository does not import the anthropic SDK anywhere; it is only an
optional runtime dependency listed in pyproject.toml and reported by
check_env.py. These tests pin that contract so a future bump cannot silently
break the declared version constraint or the environment report.
"""

import importlib
import importlib.metadata
import re
from pathlib import Path

import pytest

# Resolve the python/ package root without assuming a fixed depth, so the test
# works whether it is collected from the repo root or from python/.
_HERE = Path(__file__).resolve()
PYPROJECT = next(
    (p for p in _HERE.parents if (p / "pyproject.toml").is_file()),
    None,
)
assert PYPROJECT is not None, "pyproject.toml not found above the test file"
REPO_ROOT = PYPROJECT.parent


def _declared_anthropic_constraint() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^\s*"anthropic([^"]*)"', text, re.MULTILINE)
    assert match is not None, "anthropic dependency not found in pyproject.toml"
    return match.group(1)


def test_anthropic_constraint_is_declared():
    # The constraint itself is managed by the dependency bump; here we only
    # pin that the dependency is still declared so check_env keeps reporting it.
    constraint = _declared_anthropic_constraint()
    assert constraint, constraint


def test_anthropic_installed_version_if_present():
    try:
        version = importlib.metadata.version("anthropic")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("anthropic is not installed in this environment")
    parts = tuple(int(p) for p in re.findall(r"\d+", version)[:3])
    assert parts >= (2, 6, 0), version


def test_anthropic_sdk_importable_if_present():
    # The SDK is optional; when installed, its public client entry point must
    # still be importable under the new major version. Import lazily so this
    # module never pulls in the SDK (or sglang) at collection time.
    try:
        anthropic = importlib.import_module("anthropic")
    except ImportError:
        pytest.skip("anthropic is not installed in this environment")
    assert hasattr(anthropic, "Anthropic")
    assert hasattr(anthropic, "AsyncAnthropic")


def test_declared_constraint_allows_2_6():
    # The constraint is managed by the dependency bump; assert it admits 2.6.0
    # without rewriting pyproject.toml here.
    constraint = _declared_anthropic_constraint()
    match = re.search(r">=\s*(\d+)\.(\d+)\.(\d+)", constraint)
    assert match is not None, constraint
    assert tuple(int(p) for p in match.groups()) <= (2, 6, 0), constraint


def test_check_env_reports_anthropic():
    # check_env imports torch and sglang.srt.utils, which are not available in
    # a bare test environment. Verify the contract statically instead of
    # importing the module, so this test does not depend on the full runtime.
    source = (REPO_ROOT / "sglang" / "check_env.py").read_text(encoding="utf-8")
    assert '"anthropic"' in source
    assert "PACKAGE_LIST" in source
    assert "importlib.metadata.version" in source
    # The version-reporting helper must still exist and be callable.
    assert "def get_package_versions" in source


@pytest.mark.parametrize(
    "module_path",
    [
        "sglang/benchmark/endpoint.py",
        "sglang/benchmark/datasets/agentic_trace.py",
        "sglang/benchmark/datasets/openai_dataset.py",
        "sglang/benchmark/utils.py",
        "sglang/check_env.py",
        "sglang/cli/generate.py",
    ],
)
def test_modules_import_without_anthropic_api_usage(module_path):
    # Scan the source statically: importing these modules pulls in torch and
    # other heavy runtime deps that are not provisioned in a bare test env.
    path = REPO_ROOT / module_path
    assert path.is_file(), module_path
    source = path.read_text(encoding="utf-8")
    assert "import anthropic" not in source
    assert "from anthropic" not in source


def test_anthropic_sdk_public_surface():
    # Pin the public entry points the repository would rely on if it ever
    # adopted the SDK, so a future major bump is caught here. Import lazily so
    # collection never requires the SDK to be installed.
    try:
        anthropic = importlib.import_module("anthropic")
    except ImportError:
        pytest.skip("anthropic is not installed in this environment")
    assert hasattr(anthropic, "Anthropic")
    assert hasattr(anthropic, "AsyncAnthropic")
    assert hasattr(anthropic, "APIStatusError")
    assert hasattr(anthropic, "APIError")


def test_repo_has_no_anthropic_sdk_imports():
    # The SDK is an optional runtime dependency only; no module may import it,
    # so the 0.20 -> 2.6 breaking changes cannot affect this repository.
    root = REPO_ROOT / "sglang"
    if not root.is_dir():
        pytest.skip("sglang package sources are not present in this checkout")
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"^\s*(import anthropic|from anthropic)", text, re.MULTILINE):
            offenders.append(str(path.relative_to(root)))
    assert not offenders, offenders
