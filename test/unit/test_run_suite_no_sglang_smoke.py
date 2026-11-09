import os
import subprocess
import sys
import textwrap


REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
TEST_DIR = os.path.join(REPO_ROOT, "test")


SMOKE_SCRIPT = textwrap.dedent(
    """
    import sys
    import types

    # Simulate a CI environment where the full sglang runtime is NOT installed:
    # block any real sglang import so only the stubs injected by the unit test
    # can satisfy run_suite's module-level imports.
    class _Blocker:
        def find_module(self, name, path=None):
            if name == "sglang" or name.startswith("sglang."):
                return self
            return None

        def load_module(self, name):
            raise ImportError("sglang runtime intentionally unavailable: " + name)

    sys.meta_path.insert(0, _Blocker())

    import run_suite

    # Stub injection must be idempotent: importing twice must not raise and
    # must keep the same stub objects in sys.modules.
    import importlib
    importlib.reload(run_suite)

    assert run_suite.EXIT_OK == 0
    assert run_suite.EXIT_TEST_FAILURE == 1
    assert run_suite.EXIT_COLLECT_FAILURE == 2
    assert run_suite.EXIT_NO_TESTS == 3
    assert len(
        {
            run_suite.EXIT_OK,
            run_suite.EXIT_TEST_FAILURE,
            run_suite.EXIT_COLLECT_FAILURE,
            run_suite.EXIT_NO_TESTS,
        }
    ) == 4

    # Collection failure must be reported, not swallowed.
    def boom(files, sanity_check=True):
        raise RuntimeError("smoke: bad registration")

    run_suite.collect_tests = boom
    import argparse

    args = argparse.Namespace(
        hw="cpu",
        suite="base-a-test-cpu",
        nightly=False,
        timeout_per_file=10,
        timeout_override=[],
        timeout_from_est_time=False,
        continue_on_error=False,
        auto_partition_id=None,
        auto_partition_size=None,
        enable_retry=False,
        max_attempts=2,
        retry_wait_seconds=0,
        retry_timeout_increase=0,
        partition_model_file=None,
        only_file_list=None,
        fork_worker_batch_size=1,
    )
    code = run_suite.run_a_suite(args)
    assert code == run_suite.EXIT_COLLECT_FAILURE, code
    print("SMOKE_OK")
    """
)


def test_run_suite_imports_without_sglang_runtime():
    """run_suite must import and expose exit codes with no sglang installed.

    This is the CI-path smoke test: it runs in a subprocess with a meta_path
    blocker that makes any real `sglang` import fail, proving the stub
    injection in test_run_suite_exit_codes.py is what makes the unit tests
    runnable in a lightweight environment.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [TEST_DIR, env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    proc = subprocess.run(
        [sys.executable, "-c", SMOKE_SCRIPT],
        cwd=TEST_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "smoke subprocess failed\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    assert "SMOKE_OK" in proc.stdout
