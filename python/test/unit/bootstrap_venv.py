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


def _ensure_venv():
    python = _venv_python()
    if python.exists():
        return python
    print(f"creating virtualenv at {VENV_DIR}", flush=True)
    venv.EnvBuilder(with_pip=True, clear=False, symlinks=os.name != "nt").create(
        os.fspath(VENV_DIR)
    )
    if not python.exists():
        raise SystemExit(f"virtualenv creation did not produce {python}")
    return python


def _run(python, args):
    cmd = [os.fspath(python), "-m", *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=os.fspath(REPO_ROOT))


def _check_python_version(python):
    # compressed-tensors 0.20.x requires Python >= 3.11; fail early with a
    # clear message instead of letting pip resolve against an older runtime.
    result = subprocess.run(
        [os.fspath(python), "-c", "import sys; print('%d.%d' % sys.version_info[:2])"],
        check=True,
        capture_output=True,
        text=True,
    )
    major, minor = (int(part) for part in result.stdout.strip().split("."))
    if (major, minor) < (3, 11):
        raise SystemExit(
            f"compressed-tensors 0.20.x requires Python >= 3.11, "
            f"but {python} is {major}.{minor}"
        )


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
    python = _ensure_venv()
    _check_python_version(python)
    _run(python, ["pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    # Editable install of the project; the pin in python/pyproject.toml decides
    # the compressed-tensors version, so no version is passed here.
    _run(python, ["pip", "install", "-e", os.fspath(PYTHON_DIR)])
    _run(python, ["-c", "import compressed_tensors; print(compressed_tensors.__version__)"])
    _report_environment(python)
    print(f"bootstrap complete; activate with: . {VENV_DIR}/bin/activate", flush=True)


if __name__ == "__main__":
    main()
