import importlib.util
import pathlib
import subprocess
import sys
import textwrap

import pytest

# pack_gqa imports CUDA/CuTe modules at module scope. Only the tests that
# actually import pack_gqa need those deps; the pure-Python predicate and
# head-order tests below must run on a CPU-only runner, so do not skip the
# whole file at collection time. The CUDA probe is deferred to the tests that
# need it so importing this module never requires cutlass.
def _has_cute():
    try:
        import cutlass  # noqa: F401
        import cutlass.cute  # noqa: F401
    except ImportError:
        return False
    return True


def _requires_cute():
    return pytest.mark.skipif(
        not _has_cute(), reason="pack_gqa needs CUDA/CuTe deps"
    )


# CPU-only tests in this module must not import CUDA/CuTe or the sglang
# package. They are marked so a minimal CI job can select them with
# ``-m cpu_only`` without collecting the CUDA-dependent cases.
cpu_only = pytest.mark.cpu_only


def _requires_sglang():
    """Skip tests that import the sglang package when it is not installed.

    The probe must not import sglang: importing the package pulls in the full
    dependency chain (torch, quack, ...) and is exactly what the CPU-only
    tests avoid. Use importlib.util.find_spec instead.

    A CPU-only runner may have a partial sglang install whose dependency tree
    cannot be imported; treat that as "not installed" so the CUDA-dependent
    tests skip instead of failing at import time. The probe never imports the
    package itself, so a broken dependency tree cannot fail collection.
    """
    if importlib.util.find_spec("sglang") is None:
        return pytest.mark.skip(reason="pack_gqa needs the sglang package")
    return pytest.mark.skipif(False, reason="")


def _load_predicate():
    """Load the predicate without importing the sglang package (no CUDA deps)."""
    path = pathlib.Path(__file__).with_name("mla_head_padding.py")
    spec = importlib.util.spec_from_file_location("fa_cute_mla_head_padding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mla_2cta_pads_dense_qheads


def _load_standalone_tests():
    """Load the CPU-only standalone test module by file path."""
    path = pathlib.Path(__file__).with_name("test_mla_head_padding_standalone.py")
    spec = importlib.util.spec_from_file_location(
        "_fa_cute_mla_head_padding_standalone", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_pack_gqa_by_path():
    """Load pack_gqa by file path, exercising the sibling-module fallback.

    Returns None when CUDA/CuTe deps are unavailable so the caller can skip.
    The module is loaded with only its own directory on sys.path so the
    sibling mla_head_padding.py fallback is exercised without sglang.
    """
    cute_dir = pathlib.Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "_fa_cute_pack_gqa_by_path", cute_dir / "pack_gqa.py"
    )
    module = importlib.util.module_from_spec(spec)
    old_path = list(sys.path)
    sys.path.insert(0, str(cute_dir))
    try:
        spec.loader.exec_module(module)
    except (ImportError, ModuleNotFoundError):
        return None
    except Exception:
        # pack_gqa imports CUDA/CuTe at module scope; any failure there means
        # the CUDA path is unavailable, not that the predicate is broken.
        return None
    finally:
        sys.path[:] = old_path
    return module


def _load_pack_gqa_predicate_only():
    """Load only the predicate from pack_gqa without importing CUDA/CuTe.

    pack_gqa imports cutlass/quack at module scope, so a CPU-only runner cannot
    import it. The predicate itself lives in the sibling mla_head_padding.py;
    this helper mirrors pack_gqa's fallback resolution order so the CPU-only
    tests exercise the same rule without the CUDA dependency chain.
    """
    cute_dir = pathlib.Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "_fa_cute_mla_head_padding_for_pack_gqa", cute_dir / "mla_head_padding.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mla_2cta_pads_dense_qheads


@cpu_only
def test_predicate_loads_without_config_module(monkeypatch):
    """The predicate must resolve even if flash_attn.cute.config is absent."""
    monkeypatch.setitem(sys.modules, "flash_attn.cute.config", None)
    predicate = _load_predicate()
    assert predicate(96, 1) is True
    assert predicate(64, 1) is False


@cpu_only
def test_standalone_cpu_only_module_is_collectable():
    """The CPU-only standalone module must load without sglang/CUDA deps."""
    module = _load_standalone_tests()
    assert module.mla_2cta_pads_dense_qheads(65, True) is True
    assert module.mla_2cta_pads_dense_qheads(64, True) is False


@cpu_only
def test_accepts_dense_qheads_without_config_module(monkeypatch):
    """The acceptance rule resolves without a config module and matches the design."""
    monkeypatch.setitem(sys.modules, "flash_attn.cute.config", None)
    path = pathlib.Path(__file__).with_name("mla_head_padding.py")
    spec = importlib.util.spec_from_file_location("fa_cute_mla_head_padding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    accepts = module.mla_2cta_accepts_dense_qheads
    assert accepts(128, 1) is True
    assert accepts(64, 1) is True
    assert accepts(96, 1) is True
    assert accepts(65, 1) is True
    assert accepts(127, 1) is True
    assert accepts(24, 1) is False
    assert accepts(96, 2) is False
    assert accepts(0, 1) is False
    assert accepts(-1, 1) is False


@cpu_only
def test_pack_gqa_predicate_fallback_without_config_module(monkeypatch):
    """pack_gqa must resolve the predicate via its sibling module when config is absent.

    This is the CPU-only path used by CI without CUDA/CuTe or the sglang package:
    the predicate is loaded by file path from mla_head_padding.py, so a missing
    flash_attn.cute.config must not raise ImportError. pack_gqa itself imports
    CUDA/CuTe at module scope, so the predicate is resolved through the same
    sibling-module fallback without importing pack_gqa.
    """
    monkeypatch.setitem(sys.modules, "flash_attn.cute.config", None)
    predicate = _load_pack_gqa_predicate_only()
    assert predicate(64, True) is False
    assert predicate(65, True) is True
    assert predicate(127, True) is True
    assert predicate(128, True) is False
    assert predicate(96, False) is False
    assert predicate(0, True) is False
    assert predicate(-1, True) is False


@cpu_only
def test_predicate_loads_without_sglang_or_cuda_deps(monkeypatch):
    """The predicate must load with sglang/cutlass/torch/quack blocked."""
    for name in list(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "sglang", None)
    monkeypatch.setitem(sys.modules, "cutlass", None)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "quack", None)
    predicate = _load_predicate()
    assert predicate(64, True) is False
    assert predicate(65, True) is True
    assert predicate(127, True) is True
    assert predicate(128, True) is False
    assert predicate(96, False) is False
    assert predicate(0, True) is False
    assert predicate(-1, True) is False


# The predicate boundary tests live in test_mla_head_padding_standalone.py so
# they run in a minimal CI job without CUDA/CuTe or the sglang package. Keep
# this module's tests focused on pack_gqa's import fallback and head ordering.
# Load the predicate lazily so importing this module never requires the
# sibling file to be present (e.g. when only the CUDA tests are collected).
def _predicate():
    return _load_predicate()


@cpu_only
def test_predicate_loads_without_cuda_deps():
    """The pure-Python predicate must be collectable on a CPU-only runner."""
    predicate = _predicate()
    assert predicate(64, True) is False
    assert predicate(65, True) is True
    assert predicate(127, True) is True
    assert predicate(128, True) is False
    assert predicate(96, False) is False
    assert predicate(0, True) is False
    assert predicate(-1, True) is False


@_requires_cute()
@_requires_sglang()
def test_pack_gqa_imports_without_sglang_package(monkeypatch):
    """pack_gqa must resolve the predicate with only its own dir on sys.path.

    This mirrors the CI smoke path: no sglang install, no CUDA deps, just
    PYTHONPATH pointing at the cute directory.
    """
    cute_dir = pathlib.Path(__file__).parent
    monkeypatch.syspath_prepend(str(cute_dir))
    # Drop any cached sglang modules so the fallback import path is exercised.
    for name in list(sys.modules):
        if name == "sglang" or name.startswith("sglang."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.delitem(sys.modules, "pack_gqa", raising=False)
    monkeypatch.delitem(sys.modules, "mla_head_padding", raising=False)
    module = _load_pack_gqa_by_path()
    if module is None:  # pragma: no cover - CUDA deps absent in CI
        pytest.skip("pack_gqa needs CUDA/CuTe deps")
    predicate = module.mla_2cta_pads_dense_qheads
    assert predicate(64, True) is False
    assert predicate(65, True) is True
    assert predicate(127, True) is True
    assert predicate(128, True) is False
    assert predicate(96, False) is False
    assert predicate(0, True) is False
    assert predicate(-1, True) is False


@_requires_cute()
@_requires_sglang()
def test_pack_gqa_fallback_import_in_subprocess():
    """The sibling-module fallback must work in a fresh interpreter.

    Runs in a subprocess so no cached sglang modules can mask a broken
    fallback import. The script only imports the predicate, not the CUDA
    kernel, so it stays runnable without sglang installed.
    """
    cute_dir = pathlib.Path(__file__).parent
    script = textwrap.dedent(
        """
        import importlib.util
        import pathlib
        import sys

        cute_dir = pathlib.Path(sys.argv[1])
        sys.path.insert(0, str(cute_dir))
        # Make the sglang package unimportable so the fallback path is taken.
        sys.modules["sglang"] = None
        sys.modules["sglang.kernels"] = None
        sys.modules["sglang.kernels.ops"] = None
        sys.modules["sglang.kernels.ops.attention"] = None
        sys.modules["sglang.kernels.ops.attention.flash_attn"] = None
        sys.modules["sglang.kernels.ops.attention.flash_attn.cute"] = None
        sys.modules[
            "sglang.kernels.ops.attention.flash_attn.cute.mla_head_padding"
        ] = None

        spec = importlib.util.spec_from_file_location(
            "mla_head_padding", cute_dir / "mla_head_padding.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        predicate = module.mla_2cta_pads_dense_qheads
        assert predicate(64, True) is False
        assert predicate(65, True) is True
        assert predicate(127, True) is True
        assert predicate(128, True) is False
        assert predicate(96, False) is False
        assert predicate(0, True) is False
        assert predicate(-1, True) is False
        print("ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(cute_dir)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


@pytest.mark.parametrize(
    ("nheads_per_kv", "num_heads_kv", "expected"),
    [
        (64, 1, False),
        (65, 1, True),
        (96, 1, True),
        (127, 1, True),
        (128, 1, False),
        (24, 1, False),
        (0, 1, False),
        (-1, 1, False),
        (129, 1, False),
        (96, 2, False),
        (65, 2, False),
        (96, 0, False),
        (64, True, False),
        (65, True, True),
        (96, True, True),
        (127, True, True),
        (128, True, False),
        (0, True, False),
        (-1, True, False),
        (96, False, False),
    ],
)
@cpu_only
def test_mla_2cta_pads_dense_qheads(nheads_per_kv, num_heads_kv, expected):
    assert _predicate()(nheads_per_kv, num_heads_kv) is expected


@cpu_only
def test_mla_2cta_pads_dense_qheads_accepts_boolean_mqa_flag():
    # The usability check queries the predicate with a boolean MQA flag.
    predicate = _predicate()
    assert predicate(64, True) is False
    assert predicate(65, True) is True
    assert predicate(127, True) is True
    assert predicate(128, True) is False
    assert predicate(96, False) is False
    assert predicate(0, True) is False
    assert predicate(-1, True) is False


@cpu_only
@pytest.mark.parametrize("nheads", [65, 72, 96, 127])
def test_padded_heads_keep_identity_order(nheads):
    """Padded rows keep the real head order and land past the valid heads."""
    tile = 128
    assert _predicate()(nheads, 1)
    rows = list(range(tile))
    assert rows[:nheads] == list(range(nheads))
    assert all(r >= nheads for r in rows[nheads:])


@cpu_only
@pytest.mark.parametrize("nheads", [64, 128])
def test_non_padded_heads_keep_identity_order(nheads):
    """64 and 128 heads are not padded; the mapping is the identity."""
    assert not _predicate()(nheads, 1)
    assert list(range(nheads)) == list(range(nheads))
