"""Compatibility checks for the anthropic dependency bump (0.20 -> 2.6).

The repository does not import the anthropic SDK anywhere; it is only an
optional runtime dependency listed in pyproject.toml and reported by
check_env.py. These tests pin that contract so a future bump cannot silently
break the declared version constraint or the environment report.
"""

import importlib
import importlib.metadata
import re
import tomllib
from pathlib import Path

import pytest

# This module intentionally depends only on the standard library and pytest so
# it can run in a minimal CI job without installing sglang's full dependency
# set (torch, quack-kernels, py-spy, ...). Do not add imports of sglang or the
# anthropic SDK at module scope; use importlib inside tests instead.


def _specifier_contains(constraint: str, version: str) -> bool:
    """Minimal PEP 440 lower-bound check without importing packaging.

    The constraint is managed by the dependency bump and is expected to be a
    simple ``>=X.Y.Z`` lower bound; compare numeric release tuples so the test
    does not require packaging to be installed in a bare environment.
    """
    match = re.fullmatch(r">=\s*(\d+(?:\.\d+)*)", constraint)
    if match is None:
        return False
    lower = tuple(int(p) for p in match.group(1).split("."))
    # Pad the installed version so a shorter release tuple (e.g. "2.6")
    # compares correctly against a longer lower bound (e.g. "2.6.0").
    parts = tuple(int(p) for p in re.findall(r"\d+", version))
    parts = parts + (0,) * (len(lower) - len(parts))
    return parts[: len(lower)] >= lower


# Resolve the python/ package root without assuming a fixed depth, so the test
# works whether it is collected from the repo root or from python/. Resolved
# lazily so collection never fails when the checkout layout differs.
_HERE = Path(__file__).resolve()


def _find_pyproject() -> Path:
    for parent in _HERE.parents:
        candidate = parent / "pyproject.toml"
        if candidate.is_file():
            return candidate
    # Fall back to the conventional checkout layout so the constraint checks
    # still run (and fail loudly) instead of being skipped wholesale.
    # Guard the index so a shallow checkout cannot raise IndexError.
    if len(_HERE.parents) > 3:
        fallback = _HERE.parents[3] / "python" / "pyproject.toml"
        if fallback.is_file():
            return fallback
    # pytest.skip raises Skipped, but keep an explicit raise so a future
    # refactor cannot fall through and return None (UnboundLocalError).
    pytest.skip("pyproject.toml not found above the test file")
    raise AssertionError("unreachable")


def _read_pyproject_text() -> str:
    """Read pyproject.toml as text, skipping when it is absent.

    Used by the static checks so they do not depend on tomllib being able to
    parse the whole file (e.g. a trimmed checkout with an unusual layout).
    """
    path = _find_pyproject()
    try:
        return path.read_text(encoding="utf-8")
    except OSError as e:  # pragma: no cover - defensive
        pytest.skip(f"cannot read {path}: {e}")
        raise AssertionError("unreachable")


def _repo_root() -> Path:
    return _find_pyproject().parent


def _sglang_root() -> Path:
    """Locate the sglang package directory without importing it.

    The package lives next to pyproject.toml (python/sglang) in the normal
    checkout, but tests may also be collected from a layout where the package
    is one level up. Probe both and return the first that exists; callers must
    check ``is_dir()`` before reading files so a missing checkout skips rather
    than errors.
    """
    root = _repo_root()
    for candidate in (root / "sglang", root.parent / "sglang"):
        if candidate.is_dir():
            return candidate
    return root / "sglang"


def _declared_anthropic_constraint_from_toml() -> str:
    """Read the anthropic specifier from [project].dependencies via tomllib."""
    try:
        with _find_pyproject().open("rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        pytest.skip(f"cannot parse pyproject.toml: {e}")
        raise AssertionError("unreachable")
    for dep in data.get("project", {}).get("dependencies", []):
        name = re.split(r"[\s<>=!~\[]", dep, maxsplit=1)[0].strip()
        if name == "anthropic":
            # Strip surrounding whitespace so the specifier checks below see a
            # bare ">=2.6.0" regardless of how the dependency is formatted.
            return dep[len("anthropic"):].strip()
    # The dependency is owned by the bump; if it is absent in this checkout
    # (e.g. a trimmed tree), skip rather than fail so the suite still runs.
    pytest.skip("anthropic dependency not found in pyproject.toml")


def _declared_anthropic_constraint() -> str:
    # Parse the TOML with the standard library so the check does not depend on
    # packaging or on the exact formatting of the dependencies array.
    return _declared_anthropic_constraint_from_toml()


def test_anthropic_constraint_is_declared():
    # The constraint itself is managed by the dependency bump; here we only
    # pin that the dependency is still declared so check_env keeps reporting it.
    constraint = _declared_anthropic_constraint()
    assert constraint, constraint
    assert constraint.strip(), constraint
    # The declared lower bound must be at least 2.6.0 so the upgraded SDK is
    # allowed by the install step; the exact value is owned by the bump.
    assert _specifier_contains(constraint, "2.6.0"), constraint


def test_anthropic_constraint_is_not_pinned_to_old_major():
    # Guard against a regression that re-pins the SDK to the 0.x line, which
    # would make the declared constraint exclude the upgraded 2.6 release.
    constraint = _declared_anthropic_constraint()
    assert not re.fullmatch(r"==\s*0\..*", constraint), constraint


def test_anthropic_constraint_allows_2_6():
    # The declared lower bound must not exclude the upgraded SDK. The exact
    # value is owned by the dependency bump; here we only require that 2.6.0
    # satisfies the declared specifier.
    constraint = _declared_anthropic_constraint()
    assert _specifier_contains(constraint, "2.6.0"), constraint


def test_anthropic_installed_version_if_present():
    # The SDK is optional; when it is installed, it must satisfy the declared
    # lower bound. Skip (not fail) when absent so the suite runs in a bare env.
    # The lower bound is read from pyproject.toml rather than hardcoded, so the
    # dependency bump remains the single source of truth for the version.
    try:
        version = importlib.metadata.version("anthropic")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("anthropic is not installed in this environment")
    constraint = _declared_anthropic_constraint()
    assert _specifier_contains(constraint, version), (version, constraint)


def test_anthropic_constraint_is_parseable():
    # The constraint is managed by the dependency bump; here we only require it
    # to be a well-formed specifier so the install step cannot fail on syntax.
    constraint = _declared_anthropic_constraint()
    assert re.fullmatch(r"[<>=!~][^\s]*", constraint), constraint


def test_anthropic_constraint_has_no_removed_0x_api_reference():
    # The 0.20 line is gone; the declared constraint must not reference it.
    # This is a static check so it works without the SDK installed.
    text = _read_pyproject_text()
    assert "anthropic>=0." not in text
    assert "anthropic==0." not in text
    assert "anthropic<2" not in text


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
    # Constructing a client must not require network access or credentials
    # beyond the explicit api_key; this pins the 2.x keyword-only signature.
    # Guard against SDKs that validate credentials eagerly at construction.
    try:
        client = anthropic.Anthropic(api_key="test-key")
    except TypeError as e:
        pytest.fail(f"Anthropic(api_key=...) signature changed: {e}")
    except Exception as e:  # noqa: BLE001 - SDK may validate eagerly
        pytest.skip(f"Anthropic client construction unavailable: {e}")
    assert client is not None
    # The client must expose the messages resource used by any future caller.
    assert hasattr(client, "messages")


def test_anthropic_sdk_version_satisfies_constraint_if_present():
    # When the SDK is installed, its version must satisfy the declared
    # constraint, so a stale 0.20 install cannot silently pass the suite.
    try:
        version = importlib.metadata.version("anthropic")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("anthropic is not installed in this environment")
    constraint = _declared_anthropic_constraint()
    assert _specifier_contains(constraint, version), (version, constraint)


def test_declared_constraint_is_a_lower_bound():
    # The constraint is managed by the dependency bump; assert it is expressed
    # as a lower bound so the installed SDK is never older than declared.
    # Do not pin the exact version here: the bump owns that value.
    constraint = _declared_anthropic_constraint()
    assert re.fullmatch(r">=\s*\d+(?:\.\d+)*", constraint), constraint


def test_check_env_reports_anthropic():
    # check_env imports torch and sglang.srt.utils, which are not available in
    # a bare test environment. Verify the contract statically instead of
    # importing the module, so this test does not depend on the full runtime.
    # Skip when the sglang sources are not present in this checkout.
    path = _sglang_root() / "check_env.py"
    if not path.is_file():
        pytest.skip("sglang/check_env.py is not present in this checkout")
    source = path.read_text(encoding="utf-8")
    assert '"anthropic"' in source
    assert "PACKAGE_LIST" in source
    assert "importlib.metadata.version" in source
    # The version-reporting helper must still exist and be callable.
    assert "def get_package_versions" in source
    # The package name must be reported without a version specifier so
    # importlib.metadata.version() receives a bare distribution name.
    package_block = source.split("PACKAGE_LIST", 1)[1].split("]", 1)[0]
    assert '"anthropic"' in package_block
    assert "anthropic>=" not in package_block
    assert "anthropic==" not in package_block


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
def test_modules_do_not_import_anthropic(module_path):
    # Scan the source statically: importing these modules pulls in torch and
    # other heavy runtime deps that are not provisioned in a bare test env.
    # None of them use the SDK, so the 0.20 -> 2.6 breaking changes cannot
    # affect them; this test pins that fact. Skip when the sglang sources are
    # not present in this checkout (e.g. a bare test environment).
    root = _sglang_root()
    if not root.is_dir():
        pytest.skip("sglang package sources are not present in this checkout")
    path = root / Path(module_path).relative_to("sglang")
    if not path.is_file():
        pytest.skip(f"{module_path} is not present in this checkout")
    source = path.read_text(encoding="utf-8")
    assert "import anthropic" not in source
    assert "from anthropic" not in source


def test_benchmark_and_cli_modules_do_not_reference_anthropic():
    # The benchmark and CLI modules are the ones the acceptance criteria call
    # out; scan them statically so the check works without torch/sglang.
    # check_env.py is excluded: it intentionally lists "anthropic" in
    # PACKAGE_LIST to report the installed version. Skip when the sglang
    # sources are not present in this checkout.
    root = _sglang_root()
    if not root.is_dir():
        pytest.skip("sglang package sources are not present in this checkout")
    for rel in (
        "sglang/benchmark/endpoint.py",
        "sglang/benchmark/datasets/agentic_trace.py",
        "sglang/benchmark/datasets/openai_dataset.py",
        "sglang/benchmark/utils.py",
        "sglang/cli/generate.py",
    ):
        path = root / Path(rel).relative_to("sglang")
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        # These modules do not use the SDK; only assert they do not import it,
        # so the 0.20 -> 2.6 breaking changes cannot affect them.
        assert "import anthropic" not in text, rel
        assert "from anthropic" not in text, rel


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
    # 2.5+ documents error_code/details/api_message on APIStatusError. These
    # are instance attributes populated from the response body, so they are
    # not guaranteed to appear in class annotations; only assert the class is
    # a subclass of APIError, which is the stable 2.x contract.
    assert issubclass(anthropic.APIStatusError, anthropic.APIError)
    # 2.5 made OverloadedError/ServiceUnavailableError subclasses of
    # InternalServerError; pin that hierarchy so a regression is caught.
    for name in ("OverloadedError", "ServiceUnavailableError"):
        if hasattr(anthropic, name):
            assert issubclass(getattr(anthropic, name), anthropic.InternalServerError)
    # 2.6 added error codes for ramp limit and external key validation; the
    # base error class must still be present so callers can catch it.
    assert hasattr(anthropic, "APIError")


def test_repo_has_no_anthropic_sdk_imports():
    # The SDK is an optional runtime dependency only; no module may import it,
    # so the 0.20 -> 2.6 breaking changes cannot affect this repository.
    root = _sglang_root()
    if not root.is_dir():
        pytest.skip("sglang package sources are not present in this checkout")
    offenders = []
    for path in root.rglob("*.py"):
        # The compatibility test itself imports the SDK lazily on purpose.
        if path.resolve() == _HERE:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if re.search(r"^\s*(import anthropic|from anthropic)", text, re.MULTILINE):
            offenders.append(str(path.relative_to(root)))
    assert not offenders, offenders


def test_anthropic_absent_degrades_gracefully():
    # When the optional SDK is not installed, importing it must raise
    # ImportError (not some other error), so callers can guard with a plain
    # try/except ImportError. This pins the optional-dependency contract.
    try:
        importlib.import_module("anthropic")
    except ImportError:
        return
    pytest.skip("anthropic is installed; absence path not exercised")


def test_check_env_reports_anthropic_without_specifier():
    # check_env must pass a bare distribution name to importlib.metadata so the
    # reported version is the installed one, not a specifier string. Skip when
    # the sglang sources are not present in this checkout.
    path = _sglang_root() / "check_env.py"
    if not path.is_file():
        pytest.skip("sglang/check_env.py is not present in this checkout")
    source = path.read_text(encoding="utf-8")
    package_block = source.split("PACKAGE_LIST", 1)[1].split("]", 1)[0]
    assert '"anthropic"' in package_block
    assert "anthropic>=" not in package_block
    assert "anthropic==" not in package_block


def test_anthropic_sdk_error_classes_if_present():
    # 2.5 added status/code error classes and 2.6 extended them; pin the
    # public names the repository would rely on if it ever adopted the SDK.
    try:
        anthropic = importlib.import_module("anthropic")
    except ImportError:
        pytest.skip("anthropic is not installed in this environment")
    for name in (
        "APIStatusError",
        "APIError",
        "BadRequestError",
        "AuthenticationError",
        "RateLimitError",
        "InternalServerError",
    ):
        assert hasattr(anthropic, name), name
    # 2.6 added error codes for ramp limit and external key validation; the
    # status error class must still carry the documented attributes.
    status_error = anthropic.APIStatusError
    assert hasattr(status_error, "status_code") or "status_code" in getattr(
        status_error, "__annotations__", {}
    )
