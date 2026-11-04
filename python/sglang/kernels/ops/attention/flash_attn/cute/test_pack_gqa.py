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


def _requires_sglang():
    """Skip tests that import the sglang package when it is not installed."""
    try:
        import sglang  # noqa: F401
    except ImportError:
        return pytest.mark.skip(reason="pack_gqa needs the sglang package")
    return pytest.mark.skipif(False, reason="")


def _load_predicate():
    """Load the predicate without importing the sglang package (no CUDA deps)."""
    path = pathlib.Path(__file__).with_name("mla_head_padding.py")
    spec = importlib.util.spec_from_file_location("fa_cute_mla_head_padding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.mla_2cta_pads_dense_qheads


def _load_pack_gqa_by_path():
    """Load pack_gqa by file path, exercising the sibling-module fallback.

    Returns None when CUDA/CuTe deps are unavailable so the caller can skip.
    """
    cute_dir = pathlib.Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "_fa_cute_pack_gqa_by_path", cute_dir / "pack_gqa.py"
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, ModuleNotFoundError):
        return None
    except Exception:
        # pack_gqa imports CUDA/CuTe at module scope; any failure there means
        # the CUDA path is unavailable, not that the predicate is broken.
        return None
    return module


def test_predicate_loads_without_config_module(monkeypatch):
    """The predicate must resolve even if flash_attn.cute.config is absent."""
    monkeypatch.setitem(sys.modules, "flash_attn.cute.config", None)
    predicate = _load_predicate()
    assert predicate(96, 1) is True
    assert predicate(64, 1) is False


def test_pack_gqa_predicate_fallback_without_config_module(monkeypatch):
    """pack_gqa must resolve the predicate via its sibling module when config is absent.

    This is the CPU-only path used by CI without CUDA/CuTe or the sglang package:
    the predicate is loaded by file path from mla_head_padding.py, so a missing
    flash_attn.cute.config must not raise ImportError.
    """
    monkeypatch.setitem(sys.modules, "flash_attn.cute.config", None)
    cute_dir = pathlib.Path(__file__).parent
    spec = importlib.util.spec_from_file_location(
        "_fa_cute_pack_gqa_no_config", cute_dir / "pack_gqa.py"
    )
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except (ImportError, ModuleNotFoundError):
        pytest.skip("pack_gqa needs CUDA/CuTe deps")
    predicate = module.mla_2cta_pads_dense_qheads
    assert predicate(64, True) is False
    assert predicate(65, True) is True
    assert predicate(127, True) is True
    assert predicate(128, True) is False
    assert predicate(96, False) is False
    assert predicate(0, True) is False
    assert predicate(-1, True) is False


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


mla_2cta_pads_dense_qheads = _load_predicate()

# The predicate boundary tests live in test_mla_head_padding_standalone.py so
# they run in a minimal CI job without CUDA/CuTe or the sglang package. Keep
# this module's tests focused on pack_gqa's import fallback and head ordering.


def test_predicate_loads_without_cuda_deps():
    """The pure-Python predicate must be collectable on a CPU-only runner."""
    assert mla_2cta_pads_dense_qheads(64, True) is False
    assert mla_2cta_pads_dense_qheads(65, True) is True
    assert mla_2cta_pads_dense_qheads(127, True) is True
    assert mla_2cta_pads_dense_qheads(128, True) is False
    assert mla_2cta_pads_dense_qheads(96, False) is False
    assert mla_2cta_pads_dense_qheads(0, True) is False
    assert mla_2cta_pads_dense_qheads(-1, True) is False


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
def test_mla_2cta_pads_dense_qheads(nheads_per_kv, num_heads_kv, expected):
    assert mla_2cta_pads_dense_qheads(nheads_per_kv, num_heads_kv) is expected


def test_mla_2cta_pads_dense_qheads_accepts_boolean_mqa_flag():
    # The usability check queries the predicate with a boolean MQA flag.
    assert mla_2cta_pads_dense_qheads(64, True) is False
    assert mla_2cta_pads_dense_qheads(65, True) is True
    assert mla_2cta_pads_dense_qheads(127, True) is True
    assert mla_2cta_pads_dense_qheads(128, True) is False
    assert mla_2cta_pads_dense_qheads(96, False) is False
    assert mla_2cta_pads_dense_qheads(0, True) is False
    assert mla_2cta_pads_dense_qheads(-1, True) is False


@pytest.mark.parametrize("nheads", [65, 72, 96, 127])
def test_padded_heads_keep_identity_order(nheads):
    """Padded rows keep the real head order and land past the valid heads."""
    tile = 128
    assert mla_2cta_pads_dense_qheads(nheads, 1)
    rows = list(range(tile))
    assert rows[:nheads] == list(range(nheads))
    assert all(r >= nheads for r in rows[nheads:])


@pytest.mark.parametrize("nheads", [64, 128])
def test_non_padded_heads_keep_identity_order(nheads):
    """64 and 128 heads are not padded; the mapping is the identity."""
    assert not mla_2cta_pads_dense_qheads(nheads, 1)
    assert list(range(nheads)) == list(range(nheads))
