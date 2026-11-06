"""Create/reuse a project-local .venv and install sglang into it.

CI previously ran ``pip install -e python`` with the system interpreter while
all verification commands activated ``.venv``, so the install and the checks
disagreed on the Python environment (and system pip may be blocked by PEP 668).
This helper makes the install target the same interpreter the checks use.

Usage (from the repository root):

    python python/test/unit/bootstrap_venv.py

It is intentionally dependency-free and does not import sglang or torch.
"""

import os
import re
import shutil
import subprocess
import sys
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
VENV_DIR = REPO_ROOT / ".venv"
PYTHON_DIR = REPO_ROOT / "python"


def _venv_python():
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _python_version(candidate):
    """Return (major, minor) for an interpreter, or None if unusable."""
    try:
        result = subprocess.run(
            [
                candidate,
                "-c",
                "import sys; print('%d.%d' % sys.version_info[:2])",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    try:
        major, minor = (int(part) for part in result.stdout.strip().split("."))
    except ValueError:
        return None
    return major, minor


def _is_supported(version):
    return version is not None and (3, 11) <= version < (3, 14)


def _supported_python():
    # The interpreter running this script may be too new (e.g. 3.14) for the
    # pinned torch/transformers wheels. Prefer an explicit override, then any
    # supported python3.x on PATH, and only then the current interpreter.
    override = os.environ.get("SGLANG_BOOTSTRAP_PYTHON")
    candidates = []
    if override:
        candidates.append(override)
    for minor in (13, 12, 11):
        found = shutil.which(f"python3.{minor}")
        if found:
            candidates.append(found)
    candidates.append(sys.executable)
    # Print every candidate and its detected version so CI logs show why a
    # particular interpreter was (or was not) selected.
    print("python interpreter candidates:", flush=True)
    for candidate in candidates:
        version = _python_version(candidate)
        shown = "unknown" if version is None else f"{version[0]}.{version[1]}"
        print(f"  {candidate} -> {shown}", flush=True)
    for candidate in candidates:
        version = _python_version(candidate)
        if _is_supported(version):
            print(
                f"selected {candidate} (Python {version[0]}.{version[1]})",
                flush=True,
            )
            return candidate
    raise SystemExit(
        "no supported Python (3.11-3.13) found for the venv; install one "
        "(e.g. python3.12) or set SGLANG_BOOTSTRAP_PYTHON to its path. "
        "Refusing to fall back to an unsupported interpreter."
    )


def _ensure_venv():
    python = _venv_python()
    if python.exists():
        # A stale .venv may have been created from an unsupported interpreter
        # (e.g. a system Python 3.14); rebuild it instead of reusing it.
        if _is_supported(_python_version(os.fspath(python))):
            return python
        print(
            f"existing virtualenv at {VENV_DIR} uses an unsupported Python; "
            "recreating it",
            flush=True,
        )
        shutil.rmtree(VENV_DIR)
    base_python = _supported_python()
    print(f"creating virtualenv at {VENV_DIR} with {base_python}", flush=True)
    # Use the supported interpreter explicitly instead of the one running this
    # script, so a system Python 3.14 cannot leak into the venv.
    subprocess.run(
        [base_python, "-m", "venv", os.fspath(VENV_DIR)],
        check=True,
        cwd=os.fspath(REPO_ROOT),
    )
    if not python.exists():
        raise SystemExit(f"virtualenv creation did not produce {python}")
    return python


def _run(python, args):
    # Always invoke pip as `<venv python> -m pip` so the install cannot land in
    # the system interpreter (PEP 668) or a different environment.
    # Guard every invocation with the supported-range assertion so a stray
    # Python 3.14 can never start resolving dependencies (cp314 wheels).
    version = _python_version(os.fspath(python))
    if not _is_supported(version):
        shown = "unknown" if version is None else f"{version[0]}.{version[1]}"
        raise SystemExit(
            f"refusing to run {python} (Python {shown}): compressed-tensors "
            "0.20.x requires Python >= 3.11 and the pinned torch/transformers "
            "wheels require Python < 3.14"
        )
    cmd = [os.fspath(python), "-m", *args]
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd, cwd=os.fspath(REPO_ROOT), capture_output=True, text=True
    )
    # Echo the full output so CI logs are not truncated by the runner and the
    # actual pip resolver error is visible when the install fails.
    if result.stdout:
        print(result.stdout, flush=True)
    if result.stderr:
        print(result.stderr, file=sys.stderr, flush=True)
    if result.returncode != 0:
        raise SystemExit(
            f"command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )


def _check_python_version(python):
    # compressed-tensors 0.20.x requires Python >= 3.11; fail early with a
    # clear message instead of letting pip resolve against an older runtime.
    # Upper bound: the pinned torch/transformers wheels in this repo are not
    # published for Python 3.14, so reject it before pip starts resolving.
    version = _python_version(os.fspath(python))
    if not _is_supported(version):
        shown = "unknown" if version is None else f"{version[0]}.{version[1]}"
        raise SystemExit(
            f"compressed-tensors 0.20.x requires Python >= 3.11 and the "
            f"pinned torch/transformers wheels require Python < 3.14, "
            f"but {python} is {shown}; create the venv with a "
            "Python 3.11-3.13 interpreter (e.g. python3.12 -m venv .venv)"
        )
    print(f"using Python {version[0]}.{version[1]} at {python}", flush=True)


def _check_compressed_tensors_version(python):
    # The pin in python/pyproject.toml is the source of truth; assert the
    # installed distribution matches it so a stale system install cannot
    # silently satisfy the import check.
    result = subprocess.run(
        [
            os.fspath(python),
            "-c",
            "import importlib.metadata as m; print(m.version('compressed-tensors'))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    version = result.stdout.strip()
    if version != "0.20.1":
        # Dump diagnostics before failing so CI logs show what pip resolved.
        for args in (
            ["-m", "pip", "show", "compressed-tensors"],
            ["-m", "pip", "list"],
        ):
            subprocess.run(
                [os.fspath(python), *args], check=False, cwd=os.fspath(REPO_ROOT)
            )
        raise SystemExit(
            f"expected compressed-tensors 0.20.1, but {python} has {version}"
        )
    return version


def _report_environment(python):
    # Keep the install and the verification commands on the same interpreter.
    for args in (
        ["-c", "import sys; print(sys.executable)"],
        ["-V"],
        ["-m", "pip", "-V"],
        ["-m", "pip", "show", "compressed-tensors"],
    ):
        subprocess.run([os.fspath(python), *args], check=False, cwd=os.fspath(REPO_ROOT))


def main():
    # Do not require the interpreter running this script to be supported: it
    # may be a system Python 3.14. _ensure_venv picks a supported interpreter
    # to create the venv, and the venv interpreter is validated below.
    print(f"bootstrap invoked with {sys.executable} (Python "
          f"{sys.version_info[0]}.{sys.version_info[1]})", flush=True)
    python = _ensure_venv()
    # Re-check the venv interpreter: a venv created from a different base
    # interpreter (e.g. a stale .venv) must not bypass the version gate.
    _check_python_version(python)
    # Print the interpreter that will receive the install so CI logs make it
    # obvious whether the system interpreter leaked in.
    _run(python, ["-c", "import sys; print(sys.executable)"])
    _run(python, ["-V"])
    _run(python, ["-m", "pip", "-V"])
    _run(python, ["-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    # Fail fast before pip resolves anything if the venv interpreter drifted
    # out of the supported range (e.g. a stale .venv from Python 3.14).
    _run(
        python,
        [
            "-c",
            "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14), "
            "sys.version; print(sys.executable, sys.version.split()[0])",
        ],
    )
    # Editable install of the project; the pin in python/pyproject.toml decides
    # the compressed-tensors version, so no version is passed here.
    _run(python, ["-m", "pip", "install", "-e", os.fspath(PYTHON_DIR)])
    version = _check_compressed_tensors_version(python)
    _run(
        python,
        [
            "-c",
            "import sys, compressed_tensors; "
            "assert (3, 11) <= sys.version_info[:2] < (3, 14), sys.version; "
            "print(sys.executable, compressed_tensors.__version__)",
        ],
    )
    _report_environment(python)
    # Surface the compressed-tensors pin from every install variant so CI logs
    # make it obvious that no 0.18.0 entry remains.
    for name in (
        "pyproject.toml",
        "pyproject_cpu.toml",
        "pyproject_npu.toml",
        "pyproject_xpu.toml",
        "pyproject_other.toml",
    ):
        path = PYTHON_DIR / name
        if not path.is_file():
            print(f"{name}: missing", flush=True)
            continue
        pins = sorted(
            set(
                re.findall(
                    r'"compressed-tensors==([0-9][^"]*)"',
                    path.read_text(encoding="utf-8"),
                )
            )
        )
        print(f"{name}: compressed-tensors pins {pins}", flush=True)
    print(
        f"bootstrap complete (compressed-tensors {version}); "
        f"activate with: . {VENV_DIR}/bin/activate",
        flush=True,
    )


if __name__ == "__main__":
    main()
