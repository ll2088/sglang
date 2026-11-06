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


def _current_version():
    return (sys.version_info[0], sys.version_info[1])


def _supported_python():
    # The interpreter running this script may be too new (e.g. 3.14) for the
    # pinned torch/transformers wheels. Prefer an explicit override, then any
    # supported python3.x on PATH, and only then the current interpreter.
    # The current interpreter is only a last resort: if it is outside the
    # supported range the loop below rejects it and we fail loudly instead of
    # letting pip resolve cp314 wheels.
    override = os.environ.get("SGLANG_BOOTSTRAP_PYTHON")
    candidates = []
    if override:
        candidates.append(override)
    # Prefer an explicit PYTHON (e.g. set by actions/setup-python) before
    # probing PATH, so CI pins the interpreter deterministically.
    env_python = os.environ.get("PYTHON")
    if env_python:
        candidates.append(env_python)
    # Probe PATH for a supported interpreter in strict 3.13 -> 3.12 -> 3.11
    # order. shutil.which returns the first match, so probe each minor
    # explicitly and keep the first supported one. The unversioned python3 is
    # deliberately NOT considered: on CI it is the system Python 3.14, which
    # would resolve cp314 wheels and break the install.
    for minor in (13, 12, 11):
        found = shutil.which(f"python3.{minor}")
        if found:
            candidates.append(found)
    # Never fall back to the interpreter running this script: on CI it is the
    # system Python 3.14, which would resolve cp314 wheels. Only an explicit
    # override, PYTHON, or a versioned python3.x on PATH may be selected; if
    # none is supported we fail loudly below instead of silently using 3.14.
    if _is_supported(_current_version()):
        print(
            f"note: invoking interpreter {sys.executable} is supported but is "
            "not used as a fallback; select it explicitly via "
            "SGLANG_BOOTSTRAP_PYTHON or PYTHON if intended",
            flush=True,
        )
    # Deduplicate while preserving order so the same path is not probed twice
    # (e.g. PYTHON and python3.12 pointing at the same binary).
    seen = set()
    deduped = []
    for candidate in candidates:
        key = os.path.realpath(os.fspath(candidate))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    candidates = deduped
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
    available = []
    for minor in (11, 12, 13, 14):
        found = shutil.which(f"python3.{minor}")
        if found:
            available.append(f"python3.{minor} -> {found}")
    raise SystemExit(
        "no supported Python (3.11-3.13) found for the venv; install one "
        "(e.g. python3.12) or set SGLANG_BOOTSTRAP_PYTHON to its path. "
        "Refusing to fall back to an unsupported interpreter. "
        f"Available interpreters: {available or 'none'}"
    )


def _ensure_venv():
    python = _venv_python()
    if python.exists():
        # A stale .venv may have been created from an unsupported interpreter
        # (e.g. a system Python 3.14); rebuild it instead of reusing it.
        # Check both the venv interpreter and the base interpreter recorded in
        # pyvenv.cfg, since a venv can report a supported version while its
        # base (and thus pip's wheel tags) is not. Also resolve the venv
        # interpreter symlink target so a link to system Python 3.14 is caught.
        venv_version = _python_version(os.fspath(python))
        base_version = _venv_base_version()
        resolved_version = _python_version(os.path.realpath(os.fspath(python)))
        if (
            _is_supported(venv_version)
            and (base_version is None or _is_supported(base_version))
            and (resolved_version is None or _is_supported(resolved_version))
        ):
            print(
                f"reusing virtualenv at {VENV_DIR} with {python} "
                f"(Python {_python_version(os.fspath(python))[0]}."
                f"{_python_version(os.fspath(python))[1]})",
                flush=True,
            )
            return python
        print(
            f"existing virtualenv at {VENV_DIR} uses an unsupported Python "
            f"(venv={venv_version}, base={base_version}, "
            f"resolved={resolved_version}); recreating it",
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
    # Validate the freshly created venv interpreter before returning it, so a
    # broken symlink or an unexpected base interpreter fails here rather than
    # during pip resolution.
    created_version = _python_version(os.fspath(python))
    if not _is_supported(created_version):
        shown = (
            "unknown"
            if created_version is None
            else f"{created_version[0]}.{created_version[1]}"
        )
        raise SystemExit(
            f"virtualenv at {VENV_DIR} was created with an unsupported "
            f"Python ({shown}); expected 3.11-3.13"
        )
    return python


def _venv_base_version():
    """Return (major, minor) of the base interpreter from pyvenv.cfg."""
    cfg = VENV_DIR / "pyvenv.cfg"
    if not cfg.is_file():
        return None
    for line in cfg.read_text(encoding="utf-8").splitlines():
        key, _, value = line.partition("=")
        if key.strip() == "version":
            parts = value.strip().split(".")
            try:
                return (int(parts[0]), int(parts[1]))
            except (IndexError, ValueError):
                return None
    return None


def _pip_install_args(python, args):
    # Constrain pip's resolver to the venv interpreter's own ABI so a stray
    # system Python 3.14 pip cannot download cp314 wheels. --only-binary
    # avoids source builds that would silently target the wrong interpreter.
    # The tags are derived from the selected venv interpreter, never fixed to
    # a single minor, so 3.11/3.12/3.13 all resolve their own wheels.
    version = _python_version(os.fspath(python))
    if version is None:
        raise SystemExit(
            f"cannot determine the Python version of {python}; refusing to "
            "run pip without an ABI constraint"
        )
    py_tag = f"{version[0]}.{version[1]}"
    abi_tag = f"cp{version[0]}{version[1]}"
    constrained = list(args)
    if "--only-binary" not in constrained:
        constrained += ["--only-binary", ":all:"]
    if "--python-version" not in constrained:
        constrained += ["--python-version", py_tag]
    if "--implementation" not in constrained:
        constrained += ["--implementation", "cp"]
    if "--abi" not in constrained:
        constrained += ["--abi", abi_tag]
    # Also pin the platform tag so a resolver running under a different
    # interpreter cannot pick a wheel for another ABI. Derive it from the
    # venv interpreter instead of hard-coding x86_64 so ARM hosts still work.
    if "--platform" not in constrained:
        platform_tag = _manylinux_platform_tag(python)
        if platform_tag is not None:
            constrained += ["--platform", platform_tag]
    return constrained


def _manylinux_platform_tag(python):
    """Return a manylinux platform tag for the venv interpreter, or None."""
    result = subprocess.run(
        [
            os.fspath(python),
            "-c",
            "import sysconfig; print(sysconfig.get_platform())",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    platform = result.stdout.strip()
    # sysconfig reports e.g. "linux-x86_64"; map it to a manylinux tag that
    # matches the ABI we already constrained above.
    if platform.startswith("linux-"):
        arch = platform.split("-", 1)[1]
        return f"manylinux2014_{arch}"
    return None


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
    # Refuse to let pip resolve wheels for an interpreter other than the venv
    # one. A system Python 3.14 leaking into the install path shows up as
    # cp314 wheels in the resolver output; fail before that happens.
    if args[:2] == ["-m", "pip"] and "install" in args:
        _assert_no_cp314_wheels(python)
        args = _pip_install_args(python, args)
        # Always upgrade pip first so Requires-Python metadata is honored
        # before resolving the project's dependencies.
        subprocess.run(
            [os.fspath(python), "-m", "pip", "install", "--upgrade", "pip"],
            check=True,
            cwd=os.fspath(REPO_ROOT),
        )
    cmd = [os.fspath(python), "-m", *args]
    # Print the full command (including the ABI constraints) so CI logs show
    # exactly which interpreter and wheel tags pip is using.
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
        # Print interpreter diagnostics so a wrong-Python failure is obvious.
        for diag in (
            ["-c", "import sys; print(sys.executable, sys.version)"],
            ["-V"],
            ["-m", "pip", "-V"],
            ["-m", "pip", "debug", "--verbose"],
        ):
            subprocess.run(
                [os.fspath(python), *diag],
                check=False,
                cwd=os.fspath(REPO_ROOT),
            )
        raise SystemExit(
            f"command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )


def _assert_no_cp314_wheels(python):
    # The pinned torch/transformers wheels are not published for Python 3.14;
    # if pip would resolve cp314 artifacts the install is running under the
    # wrong interpreter. Check the interpreter's own wheel tag support rather
    # than parsing pip output, so this gate is deterministic.
    result = subprocess.run(
        [
            os.fspath(python),
            "-c",
            "import sysconfig; print(sysconfig.get_platform()); "
            "print(sysconfig.get_config_var('SOABI') or ''); "
            "print(sysconfig.get_config_var('EXT_SUFFIX') or '')",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = result.stdout.strip().splitlines()
    soabi = lines[-2] if len(lines) >= 2 else ""
    ext_suffix = lines[-1] if lines else ""
    # Reject any interpreter whose ABI tag is cp314 or newer: the pinned
    # torch/transformers wheels stop at cp313, so resolving under cp314 would
    # download incompatible artifacts (e.g. pybase64-*-cp314, regex-*-cp314).
    for tag in (soabi, ext_suffix):
        match = re.search(r"cp3(\d+)", tag)
        if match and int(match.group(1)) >= 14:
            raise SystemExit(
                f"refusing to install with {python}: interpreter reports "
                f"{tag}, which would resolve cp3{match.group(1)} wheels; "
                "use Python 3.11-3.13"
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
    # Also assert the interpreter's own ABI tag is not cp314, so a venv that
    # reports 3.13 but links a 3.14 base cannot slip through.
    _assert_no_cp314_wheels(python)
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
    # Fail fast when the invoking interpreter is outside [3.11, 3.14) AND no
    # supported interpreter can be found; _ensure_venv already raises in that
    # case, but print the actual interpreter path/version first so CI logs
    # make the cause obvious.
    if not _is_supported(_current_version()):
        print(
            f"warning: invoking interpreter {sys.executable} is Python "
            f"{sys.version_info[0]}.{sys.version_info[1]}, outside the "
            "supported 3.11-3.13 range; a supported interpreter will be "
            "selected for the venv",
            flush=True,
        )
    # Refuse to run the install with an interpreter whose ABI tag would
    # resolve cp314 wheels. This is checked on the invoking interpreter too,
    # so a CI setup that bypasses the venv still fails loudly instead of
    # silently downloading incompatible wheels.
    _assert_no_cp314_wheels(sys.executable)
    python = _ensure_venv()
    # Hard gate: the venv interpreter must be 3.11-3.13 before any pip call.
    # Fail here with diagnostics rather than letting pip resolve cp314 wheels.
    _check_python_version(python)
    # Re-check the venv interpreter: a venv created from a different base
    # interpreter (e.g. a stale .venv) must not bypass the version gate.
    _check_python_version(python)
    # Print the interpreter that will receive the install before any pip
    # command runs, so CI logs make it obvious that the system Python 3.14
    # did not leak in.
    print(f"venv interpreter: {python}", flush=True)
    # Print the interpreter that will receive the install so CI logs make it
    # obvious whether the system interpreter leaked in.
    _run(
        python,
        [
            "-c",
            "import sys; print(sys.executable, sys.version.split()[0])",
        ],
    )
    _run(python, ["-V"])
    _run(python, ["-m", "pip", "-V"])
    _run(python, ["-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    # Print the venv interpreter and pip version so CI logs prove the install
    # path is the 3.11-3.13 venv, not the system Python 3.14.
    _run(python, ["-c", "import sys; print('venv python:', sys.executable, sys.version.split()[0])"])
    _run(python, ["-m", "pip", "-V"])
    # Fail fast right after the venv is populated: every later step assumes a
    # 3.11-3.13 interpreter, and a wrong one would resolve cp314 wheels.
    # Also assert the venv's own pip reports the same interpreter, so a
    # mismatched pip cannot silently resolve for another Python.
    _run(
        python,
        [
            "-c",
            "import sys, subprocess; "
            "out = subprocess.check_output([sys.executable, '-m', 'pip', '-V'], text=True); "
            "assert sys.executable in out, (sys.executable, out); "
            "print('pip interpreter ok:', out.strip())",
        ],
    )
    _run(
        python,
        [
            "-c",
            "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 14), "
            "sys.version; print('venv python ok:', sys.version.split()[0])",
        ],
    )
    # Run the pin consistency gate before resolving the project so a stale
    # 0.18.0 entry fails fast with a clear message instead of a resolver error.
    # Invoke it with the venv interpreter so the gate runs in the same
    # environment that will receive the install.
    _run(python, [os.fspath(PYTHON_DIR / "test" / "unit" / "check_compressed_tensors_pins.py")])
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
    # Print the interpreter that will receive the editable install, so CI logs
    # make it obvious that the system Python 3.14 did not leak in.
    _run(
        python,
        [
            "-c",
            "import sys; print('install interpreter:', sys.executable, "
            "sys.version.split()[0])",
        ],
    )
    # Editable install of the project; the pin in python/pyproject.toml decides
    # the compressed-tensors version, so no version is passed here. Use the
    # venv interpreter explicitly (never bare `pip`) so the system Python 3.14
    # cannot resolve cp314 wheels.
    _run(python, ["-m", "pip", "install", "-e", os.fspath(PYTHON_DIR)])
    version = _check_compressed_tensors_version(python)
    _run(
        python,
        [
            "-c",
            "import sys, compressed_tensors; "
            "assert (3, 11) <= sys.version_info[:2] < (3, 14), sys.version; "
            "assert compressed_tensors.__version__ == '0.20.1', "
            "compressed_tensors.__version__; "
            "print(sys.executable, compressed_tensors.__version__)",
        ],
    )
    # Assert pip did not resolve any cp314 wheel: the venv interpreter must be
    # 3.11-3.13 and the installed compressed-tensors must be the 0.20.1 pin.
    _run(
        python,
        [
            "-c",
            "import sys, importlib.metadata as m; "
            "assert (3, 11) <= sys.version_info[:2] < (3, 14), sys.version; "
            "assert m.version('compressed-tensors') == '0.20.1', "
            "m.version('compressed-tensors'); "
            "print('wheel tag ok:', sys.version.split()[0])",
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
