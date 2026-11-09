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

PYPROJECT = Path(__file__).resolve().parents[3] / "pyproject.toml"


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
    # still be importable under the new major version.
    anthropic = pytest.importorskip("anthropic")
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
    check_env = importlib.import_module("sglang.check_env")
    assert "anthropic" in check_env.PACKAGE_LIST
    versions = check_env.BaseEnv().get_package_versions()
    assert "anthropic" in versions
    assert isinstance(versions["anthropic"], str)
    # Either a real version string or the explicit not-found marker.
    assert versions["anthropic"] == "Module Not Found" or re.match(
        r"\d+\.\d+", versions["anthropic"]
    ), versions["anthropic"]


@pytest.mark.parametrize(
    "module_name",
    [
        "sglang.benchmark.endpoint",
        "sglang.benchmark.datasets.agentic_trace",
        "sglang.benchmark.datasets.openai_dataset",
        "sglang.benchmark.utils",
        "sglang.check_env",
        "sglang.cli.generate",
    ],
)
def test_modules_import_without_anthropic_api_usage(module_name):
    module = importlib.import_module(module_name)
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "import anthropic" not in source
    assert "from anthropic" not in source


def test_repo_has_no_anthropic_sdk_imports():
    # The SDK is an optional runtime dependency only; no module may import it,
    # so the 0.20 -> 2.6 breaking changes cannot affect this repository.
    root = PYPROJECT.parent / "sglang"
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if re.search(r"^\s*(import anthropic|from anthropic)", text, re.MULTILINE):
            offenders.append(str(path.relative_to(root)))
    assert not offenders, offenders
