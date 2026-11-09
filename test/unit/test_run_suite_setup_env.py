import os
import subprocess
import sys


UNIT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.dirname(UNIT_DIR)
REPO_ROOT = os.path.dirname(TEST_DIR)


def _run_setup(script, *args):
    """Run a setup script with a fake python that records pip invocations."""
    return subprocess.run(
        ["bash", script, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHON=sys.executable),
    )


def test_root_setup_rejects_editable_before_delegating():
    """`setup_env.sh -e` must fail fast instead of resolving sglang deps."""
    script = os.path.join(REPO_ROOT, "setup_env.sh")
    proc = _run_setup(script, "-e")
    assert proc.returncode == 2, proc.stderr
    assert "refusing editable install" in proc.stderr
    # The delegate must not have run: no pip install line in stdout.
    assert "pip install" not in proc.stdout


def test_root_setup_rejects_long_editable_flag():
    script = os.path.join(REPO_ROOT, "setup_env.sh")
    proc = _run_setup(script, "--editable")
    assert proc.returncode == 2, proc.stderr
    assert "refusing editable install" in proc.stderr


def test_unit_setup_rejects_editable():
    script = os.path.join(UNIT_DIR, "setup_env.sh")
    proc = _run_setup(script, "--editable")
    assert proc.returncode == 2, proc.stderr
    assert "refusing editable install" in proc.stderr


def test_setup_scripts_are_syntactically_valid():
    """Both entries must parse as bash so CI fails loudly on a typo."""
    for script in (
        os.path.join(REPO_ROOT, "setup_env.sh"),
        os.path.join(UNIT_DIR, "setup_env.sh"),
    ):
        proc = subprocess.run(
            ["bash", "-n", script], capture_output=True, text=True
        )
        assert proc.returncode == 0, proc.stderr
