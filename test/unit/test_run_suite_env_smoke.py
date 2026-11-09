import os
import subprocess
import sys


UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(UNIT_DIR)
REPO_ROOT = os.path.dirname(TEST_DIR)


# Log lines that only appear when the full sglang dependency set is being
# resolved. Their presence means the lightweight setup entry was not hit.
FULL_DEP_MARKERS = (
    "Collecting py-spy",
    "Collecting pybase64",
    "Collecting quack-kernels",
    "Collecting sglang==",
)


def _run_setup(script, *args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [TEST_DIR, env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    return subprocess.run(
        ["bash", script, *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )


def test_setup_env_rejects_editable_install():
    """Passing an editable arg must fail fast instead of resolving sglang."""
    for script in (
        os.path.join(REPO_ROOT, "setup_env.sh"),
        os.path.join(UNIT_DIR, "setup_env.sh"),
    ):
        proc = _run_setup(script, "-e", "python")
        assert proc.returncode != 0, (
            f"{script} accepted an editable install\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        combined = proc.stdout + proc.stderr
        for marker in FULL_DEP_MARKERS:
            assert marker not in combined, (
                f"{script} resolved full sglang deps: {marker}"
            )


def test_setup_env_does_not_resolve_full_deps():
    """The setup entries must not emit full sglang dependency logs.

    This is the CI-path smoke check: it asserts the environment preparation
    step does not fall back to `pip install -e python` (which downloads
    py-spy / pybase64 / quack-kernels). The editable guard is exercised so
    the check runs without network access.
    """
    for script in (
        os.path.join(REPO_ROOT, "setup_env.sh"),
        os.path.join(UNIT_DIR, "setup_env.sh"),
    ):
        proc = _run_setup(script, "--editable=python")
        combined = proc.stdout + proc.stderr
        for marker in FULL_DEP_MARKERS:
            assert marker not in combined, (
                f"{script} resolved full sglang deps: {marker}"
            )
        assert "[setup]" in combined, (
            f"{script} produced no setup diagnostic\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
