import os
import subprocess
import sys
import textwrap


REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
TEST_DIR = os.path.join(REPO_ROOT, "test")


# Self-contained smoke test: runs in a subprocess where any real `sglang`
# import is blocked, so it exercises run_suite.py's collection diagnostics and
# exit-code branches without installing the full sglang runtime. This keeps the
# verification job runnable in a minimal environment (pytest only).
STANDALONE_SCRIPT = textwrap.dedent(
    """
    import argparse
    import enum
    import sys
    import types

    # Block any real sglang import so only the stubs below can satisfy
    # run_suite's module-level imports.
    class _Blocker:
        def find_module(self, name, path=None):
            if name == "sglang" or name.startswith("sglang."):
                return self
            return None

        def load_module(self, name):
            raise ImportError("sglang runtime intentionally unavailable: " + name)

    sys.meta_path.insert(0, _Blocker())

    class HWBackend(enum.Enum):
        CPU = "cpu"
        CUDA = "cuda"
        AMD = "amd"
        MUSA = "musa"
        NPU = "npu"
        XPU = "xpu"
        MLX = "mlx"
        MPS = "mps"
        PPU = "ppu"

    class CIRegistry:
        pass

    def collect_tests(files, sanity_check=True):
        raise RuntimeError("collect_tests stub not patched")

    def auto_partition(tests, part_id, part_size, live_est=None):
        return tests

    ci_register = types.ModuleType("sglang.test.ci.ci_register")
    ci_register._run_suite_stub = True
    ci_register.CIRegistry = CIRegistry
    ci_register.HWBackend = HWBackend
    ci_register.auto_partition = auto_partition
    ci_register.collect_tests = collect_tests
    ci_utils = types.ModuleType("sglang.test.ci.ci_utils")
    ci_utils.run_unittest_files = lambda *a, **k: 0
    sys.modules.setdefault("sglang", types.ModuleType("sglang"))
    sys.modules.setdefault("sglang.test", types.ModuleType("sglang.test"))
    sys.modules.setdefault("sglang.test.ci", types.ModuleType("sglang.test.ci"))
    sys.modules["sglang.test.ci.ci_register"] = ci_register
    sys.modules["sglang.test.ci.ci_utils"] = ci_utils

    import run_suite

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

    def _args(**overrides):
        defaults = dict(
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
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    # 1) Collection failure -> dedicated code + readable diagnostic.
    def boom(files, sanity_check=True):
        raise RuntimeError("smoke: bad registration")

    run_suite.collect_tests = boom
    code = run_suite.run_a_suite(_args())
    assert code == run_suite.EXIT_COLLECT_FAILURE, code

    # 2) No tests -> dedicated code.
    run_suite.collect_tests = lambda files, sanity_check=True: []
    code = run_suite.run_a_suite(_args())
    assert code == run_suite.EXIT_NO_TESTS, code

    # 3) Execution failure -> dedicated code, distinct from collection.
    class _FakeTest:
        def __init__(self):
            self.filename = "/tmp/test_a.py"
            self.backend = HWBackend.CPU
            self.effective_suite = "base-a-test-cpu"
            self.nightly = False
            self.disabled = None
            self.est_time = 1.0

    run_suite.collect_tests = lambda files, sanity_check=True: [_FakeTest()]
    run_suite.run_unittest_files = lambda *a, **k: 1
    code = run_suite.run_a_suite(_args())
    assert code == run_suite.EXIT_TEST_FAILURE, code
    assert code != run_suite.EXIT_COLLECT_FAILURE

    # 4) All pass -> 0.
    run_suite.run_unittest_files = lambda *a, **k: 0
    code = run_suite.run_a_suite(_args())
    assert code == run_suite.EXIT_OK, code

    print("STANDALONE_OK")
    """
)


def test_run_suite_standalone_without_sglang_runtime():
    """run_suite exit-code branches must work with no sglang installed.

    Runs in a subprocess with a meta_path blocker that makes any real
    `sglang` import fail, proving the collection-diagnostic and exit-code
    branches are exercisable in a minimal (pytest-only) environment.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [TEST_DIR, env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)
    proc = subprocess.run(
        [sys.executable, "-c", STANDALONE_SCRIPT],
        cwd=TEST_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "standalone subprocess failed\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}"
    )
    assert "STANDALONE_OK" in proc.stdout
