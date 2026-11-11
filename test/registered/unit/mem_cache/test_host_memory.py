"""Exercise container and ancestor budgets using synthetic procfs/cgroup files."""

import functools
import mmap
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache import host_memory
from sglang.srt.mem_cache.hicache_storage import HiCacheStorageConfig
from sglang.srt.mem_cache.storage.backend_factory import StorageBackendFactory
from sglang.test.ci.ci_register import register_cpu_ci

_cgroup_memory_headroom = host_memory._cgroup_memory_headroom

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestHostMemory(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.proc = self.root / "proc"
        (self.proc / "self").mkdir(parents=True)
        self.mount = self.root / "cgroup mount"
        self.mount.mkdir()

    def configure(self, membership="/task/engine", mount_root="/", v1=False):
        controllers = "memory" if v1 else ""
        (self.proc / "self/cgroup").write_text(f"0:{controllers}:{membership}\n")
        filesystem = "cgroup" if v1 else "cgroup2"
        options = "rw,memory" if v1 else "rw"
        escaped = str(self.mount).replace(" ", r"\040")
        (self.proc / "self/mountinfo").write_text(
            f"1 0 0:1 {mount_root} {escaped} rw - {filesystem} cgroup {options}\n"
        )

    def memory(
        self, path, usage, maximum="max", high="max", v1=False, stat=None, minimum=0
    ):
        directory = self.mount / path
        directory.mkdir(parents=True, exist_ok=True)
        files = (
            {"memory.limit_in_bytes": maximum, "memory.usage_in_bytes": usage}
            if v1
            else {
                "memory.max": maximum,
                "memory.high": high,
                "memory.current": usage,
                "memory.min": minimum,
            }
        )
        prefix = "total_" if v1 else ""
        stat = {f"{prefix}active_file": 0, f"{prefix}inactive_file": 0} | (stat or {})
        files["memory.stat"] = "".join(
            f"{key} {value}\n" for key, value in stat.items()
        )
        for name, value in files.items():
            (directory / name).write_text(str(value))

    def test_v2_parent_and_high_limits(self):
        self.configure()
        self.memory("task/engine", 100)
        for maximum, high, usage, expected in [
            (1000, "max", 300, 700),
            (1000, 600, 300, 300),
            ("max", 600, 700, 0),
            ("max", "max", 300, None),
        ]:
            with self.subTest(maximum=maximum, high=high):
                self.memory("task", usage, maximum, high)
                self.assertEqual(
                    host_memory._cgroup_memory_headroom(self.proc), expected
                )
        self.memory("task", 100, 1000)
        self.memory("task/engine", 100, 250)
        self.assertEqual(host_memory._cgroup_memory_headroom(self.proc), 150)

    def test_mount_subtree_and_cgroup_namespace(self):
        self.memory("engine", 100, 900)
        self.memory("", 300, 1000)
        for membership in ["/host/task/engine", "/engine"]:
            with self.subTest(membership=membership):
                self.configure(membership, mount_root="/host/task")
                self.assertEqual(host_memory._cgroup_memory_headroom(self.proc), 700)

    def test_v1_parent_limit_and_unlimited_sentinel(self):
        self.configure(v1=True)
        self.memory("task/engine", 100, 2**63 - 4096, v1=True)
        self.memory("task", 400, 1000, v1=True)
        self.assertEqual(host_memory._cgroup_memory_headroom(self.proc), 600)

    def test_reclaimable_page_cache_is_not_used(self):
        """Page cache charged to the cgroup, such as a checkpoint just read, is
        reclaimed under the limit and must not shrink the headroom. Shared
        memory is counted in file/cache but cannot be reclaimed."""
        v2 = {"file": 700, "shmem": 200, "active_file": 200, "inactive_file": 300}
        v1 = {
            "cache": 700,
            "shmem": 200,
            "active_file": 0,
            "inactive_file": 0,
            "total_active_file": 200,
            "total_inactive_file": 300,
        }
        for is_v1, stat, usage, expected in [
            (False, v2, 900, 600),
            (True, v1, 900, 600),
            # v1 usage is approximate and may trail the cache counters.
            (True, v1, 400, 1000),
        ]:
            with self.subTest(v1=is_v1, usage=usage):
                self.configure(v1=is_v1)
                self.memory("task/engine", usage, 1000, v1=is_v1, stat=stat)
                self.assertEqual(
                    host_memory._cgroup_memory_headroom(self.proc), expected
                )

    def test_protected_sibling_cache_stays_charged(self):
        self.configure()
        self.memory("task", 850, 900, stat={"inactive_file": 600})
        self.memory("task/engine", 250, 1000)
        self.memory("task/sibling", 600, stat={"inactive_file": 600})
        self.assertEqual(self.available(), 650)

        # Parent reclaim must preserve the sibling's protected cache.
        (self.mount / "task/sibling/memory.min").write_text("600")
        self.assertEqual(self.available(), 50)

    def test_independent_engines_have_separate_allowances(self):
        # Both engines see the same host RAM but have different charged usage.
        for task, usage, expected in [("a", 300, 700), ("b", 600, 400)]:
            with self.subTest(task=task):
                self.configure(f"/{task}/engine")
                self.memory(f"{task}/engine", 0)
                self.memory(task, usage, 1000)
                with (
                    patch.object(
                        host_memory.psutil,
                        "virtual_memory",
                        return_value=Mock(available=2000),
                    ),
                    patch.object(
                        host_memory,
                        "_cgroup_memory_headroom",
                        return_value=host_memory._cgroup_memory_headroom(self.proc),
                    ),
                ):
                    self.assertEqual(
                        host_memory.available_host_memory_bytes(), expected
                    )

    def test_host_availability_is_also_a_bound(self):
        for cgroup, allow_fallback, expected in [
            (None, False, 100),
            (200, False, 100),
            (50, False, 50),
            (50, True, 50),
        ]:
            with (
                self.subTest(cgroup=cgroup, allow_fallback=allow_fallback),
                patch.object(
                    host_memory.psutil,
                    "virtual_memory",
                    return_value=Mock(available=100),
                ),
                patch.object(
                    host_memory, "_cgroup_memory_headroom", return_value=cgroup
                ),
            ):
                self.assertEqual(
                    host_memory.available_host_memory_bytes(
                        allow_cgroup_fallback=allow_fallback
                    ),
                    expected,
                )

    def available(self, host_available=5000, *, allow_cgroup_fallback=False):
        """Public sizing entry point over the synthetic procfs."""
        with (
            patch.object(
                host_memory.psutil,
                "virtual_memory",
                return_value=Mock(available=host_available),
            ),
            patch.object(
                host_memory,
                "_cgroup_memory_headroom",
                functools.partial(_cgroup_memory_headroom, self.proc),
            ),
        ):
            return host_memory.available_host_memory_bytes(
                allow_cgroup_fallback=allow_cgroup_fallback
            )

    def without_cgroupfs(self, v1=False):
        # A container sharing the host cgroup namespace, no cgroupfs mounted.
        self.configure("/system.slice/engine.scope", v1=v1)
        (self.proc / "self/mountinfo").write_text("1 0 0:1 / /proc rw - proc proc rw\n")

    def test_unavailable_cgroup_requires_explicit_sizing(self):
        for v1 in [False, True]:
            for mounted in [False, True]:
                with self.subTest(v1=v1, mounted=mounted):
                    if mounted:
                        self.configure("/system.slice/engine.scope", v1=v1)
                    else:
                        self.without_cgroupfs(v1=v1)
                    with self.assertRaisesRegex(RuntimeError, "set --hicache-size"):
                        self.available()
                    with self.assertLogs(host_memory.logger, level="WARNING"):
                        self.assertEqual(
                            self.available(allow_cgroup_fallback=True), 5000
                        )

    def test_missing_counters_for_known_limit_requires_explicit_sizing(self):
        for name in ("memory.current", "memory.stat"):
            with self.subTest(name=name):
                self.configure()
                self.memory("task/engine", 100, 1000)
                (self.mount / "task/engine" / name).unlink()
                with self.assertRaisesRegex(RuntimeError, "set --hicache-size"):
                    self.available()
                with self.assertLogs(host_memory.logger, level="WARNING"):
                    self.assertEqual(self.available(allow_cgroup_fallback=True), 5000)

    def test_split_host_buffer_views_are_contiguous(self):
        buf = host_memory.alloc_host_buffer(3 * mmap.PAGESIZE)
        views = host_memory.split_host_buffer(buf, [mmap.PAGESIZE, 2 * mmap.PAGESIZE])
        self.assertEqual(len(views), 2)
        self.assertEqual(views[0].data_ptr() + views[0].nbytes, views[1].data_ptr())
        self.assertEqual(sum(v.nbytes for v in views), buf.nbytes)
        with self.assertRaises(ValueError):
            host_memory.split_host_buffer(buf, [buf.nbytes + 1])

    def test_alloc_host_buffer_rounds_to_page(self):
        buf = host_memory.alloc_host_buffer(1)
        self.assertEqual(buf.nbytes, mmap.PAGESIZE)
        self.assertEqual(buf.dtype, torch.int8)

    def test_parse_hugepage_size(self):
        self.assertIsNone(host_memory.parse_hugepage_size(None))
        self.assertIsNone(host_memory.parse_hugepage_size("none"))
        self.assertEqual(host_memory.parse_hugepage_size("2MB"), 2 << 20)
        self.assertEqual(host_memory.parse_hugepage_size("1GB"), 1 << 30)
        with self.assertRaises(ValueError):
            host_memory.parse_hugepage_size("4KB")

    def _hugepage_pool_bytes(self, size: int) -> int:
        """Return the host's free HugeTLB pool bytes for ``size``, or 0."""
        try:
            with open(f"/sys/kernel/mm/hugepages/hugepages-{size // 1024}kB/free_hugepages") as f:
                return int(f.read().strip()) * size
        except (FileNotFoundError, ValueError):
            return 0

    def test_hugepage_shortfall_raises(self):
        # Requesting more HugeTLB pages than the host has must fail explicitly.
        # Skip only when the host has no HugeTLB pool at all, so the assertion
        # still runs on machines that can actually back the mapping.
        if self._hugepage_pool_bytes(1 << 30) == 0:
            self.skipTest("host has no 1 GiB HugeTLB pool")
        with self.assertRaisesRegex(RuntimeError, "HugeTLB"):
            host_memory.alloc_host_buffer(1 << 40, hugepage_size=1 << 30)

    def test_hugepage_explicit_request_without_pool_raises(self):
        # Explicitly requesting HugeTLB pages must never silently fall back to
        # ordinary pages: if the pool is absent or too small, raise.
        if self._hugepage_pool_bytes(1 << 30) > 0:
            self.skipTest("host has a 1 GiB HugeTLB pool; shortfall covered above")
        with self.assertRaisesRegex(RuntimeError, "HugeTLB"):
            host_memory.alloc_host_buffer(mmap.PAGESIZE, hugepage_size=1 << 30)

    def test_hugepage_disabled_uses_ordinary_pages(self):
        # Without the flag, allocation must succeed on ordinary pages.
        buf = host_memory.alloc_host_buffer(mmap.PAGESIZE)
        self.assertEqual(buf.nbytes, mmap.PAGESIZE)
        self.assertEqual(buf.dtype, torch.int8)

    def test_madvise_dontfork_is_called(self):
        calls = []
        real_mmap = mmap.mmap

        class RecordingMmap(real_mmap):
            def madvise(self, option, start=0, length=0):
                calls.append(option)
                return super().madvise(option, start, length)

        with patch.object(host_memory.mmap, "mmap", RecordingMmap):
            host_memory.alloc_host_buffer(mmap.PAGESIZE)
        self.assertIn(mmap.MADV_DONTFORK, calls)

    def test_madvise_dontfork_isolates_child(self):
        # A forked child must not inherit the host mapping; writing in the
        # child must not be visible in the parent's buffer.
        if not hasattr(os, "fork"):
            self.skipTest("fork() is unavailable on this platform")
        buf = host_memory.alloc_host_buffer(mmap.PAGESIZE)
        buf[0] = 7
        pid = os.fork()
        if pid == 0:
            try:
                # Child: the mapping is excluded from fork, so touching it
                # must not corrupt the parent's page. Exit without flushing.
                os._exit(0)
            except BaseException:
                os._exit(1)
        _, status = os.waitpid(pid, 0)
        self.assertEqual(os.waitstatus_to_exitcode(status), 0)
        self.assertEqual(int(buf[0]), 7)

    def test_alloc_host_buffer_releases_mmap(self):
        # Dropping the tensor view must release the underlying mmap without
        # leaking file descriptors or emitting ResourceWarning.
        import gc
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ResourceWarning)
            buf = host_memory.alloc_host_buffer(mmap.PAGESIZE)
            del buf
            gc.collect()
        self.assertFalse(
            [w for w in caught if issubclass(w.category, ResourceWarning)],
            "alloc_host_buffer leaked an mmap",
        )

    def test_storage_backend_factory_uses_host_pool(self):
        # The factory path must still create and release a backend after the
        # host pool allocation path was reworked.
        config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name="test",
        )
        backend = StorageBackendFactory.create_backend("file", config, None)
        self.assertIsNotNone(backend)
        backend.clear()

    def test_host_registration_ranges_align_to_pages(self):
        page = mmap.PAGESIZE
        block = 4 * page
        sizes = [8 * block, 8 * block]
        total = sum(sizes)
        ranges = host_memory.host_registration_ranges(
            sizes, 8, total, max_chunk_bytes=5 * block, page_size=page
        )
        self.assertEqual(ranges[0][0], 0)
        self.assertEqual(ranges[-1][1], total)
        for start, end in ranges:
            self.assertEqual(start % page, 0)
            self.assertEqual(end % page, 0)
            self.assertLessEqual(end - start, 5 * block)
        for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:]):
            self.assertEqual(prev_end, next_start)

    def test_host_registration_ranges_rejects_misaligned_pool(self):
        with self.assertRaises(ValueError):
            host_memory.host_registration_ranges(
                [mmap.PAGESIZE], 1, mmap.PAGESIZE + 1
            )

    def test_pin_unpin_chunked_round_trip(self):
        page = mmap.PAGESIZE
        block = 4 * page
        sizes = [4 * block]
        buf = host_memory.alloc_host_buffer(sum(sizes))
        cudart = Mock()
        cudart.cudaHostRegister.return_value = Mock(value=0)
        cudart.cudaHostUnregister.return_value = Mock(value=0)
        with patch.object(torch.cuda, "cudart", return_value=cudart):
            ranges = host_memory.pin_tensor_chunked(
                buf, sizes, 4, page_size=page
            )
            host_memory.unpin_tensor_chunked(buf, ranges)
        self.assertEqual(cudart.cudaHostRegister.call_count, len(ranges))
        self.assertEqual(cudart.cudaHostUnregister.call_count, len(ranges))
        registered = [c.args[0] for c in cudart.cudaHostRegister.call_args_list]
        unregistered = [c.args[0] for c in cudart.cudaHostUnregister.call_args_list]
        self.assertEqual(registered, unregistered)

    def test_pin_chunked_rolls_back_on_failure(self):
        page = mmap.PAGESIZE
        block = 4 * page
        sizes = [4 * block]
        buf = host_memory.alloc_host_buffer(sum(sizes))
        cudart = Mock()
        cudart.cudaHostRegister.side_effect = [
            Mock(value=0),
            Mock(value=1),
        ]
        cudart.cudaHostUnregister.return_value = Mock(value=0)
        with patch.object(torch.cuda, "cudart", return_value=cudart):
            with self.assertRaises(RuntimeError):
                host_memory.pin_tensor_chunked(buf, sizes, 4, page_size=page)
        self.assertEqual(cudart.cudaHostUnregister.call_count, 1)

    def test_pin_unpin_chunked_reuses_buffer(self):
        # After unpin, the same buffer can be pinned again with identical ranges.
        page = mmap.PAGESIZE
        block = 4 * page
        sizes = [4 * block]
        buf = host_memory.alloc_host_buffer(sum(sizes))
        cudart = Mock()
        cudart.cudaHostRegister.return_value = Mock(value=0)
        cudart.cudaHostUnregister.return_value = Mock(value=0)
        with patch.object(torch.cuda, "cudart", return_value=cudart):
            first = host_memory.pin_tensor_chunked(buf, sizes, 4, page_size=page)
            host_memory.unpin_tensor_chunked(buf, first)
            second = host_memory.pin_tensor_chunked(buf, sizes, 4, page_size=page)
            host_memory.unpin_tensor_chunked(buf, second)
        self.assertEqual(first, second)
        self.assertEqual(cudart.cudaHostRegister.call_count, 2 * len(first))
        self.assertEqual(cudart.cudaHostUnregister.call_count, 2 * len(first))

    def test_pin_chunked_does_not_pin_unlisted_region(self):
        # Only the ranges returned by host_registration_ranges are registered.
        page = mmap.PAGESIZE
        block = 4 * page
        sizes = [4 * block]
        buf = host_memory.alloc_host_buffer(sum(sizes))
        cudart = Mock()
        cudart.cudaHostRegister.return_value = Mock(value=0)
        cudart.cudaHostUnregister.return_value = Mock(value=0)
        with patch.object(torch.cuda, "cudart", return_value=cudart):
            ranges = host_memory.pin_tensor_chunked(buf, sizes, 4, page_size=page)
            host_memory.unpin_tensor_chunked(buf, ranges)
        base = buf.data_ptr()
        registered = [c.args[0] for c in cudart.cudaHostRegister.call_args_list]
        self.assertEqual(registered[0], base)
        self.assertEqual(registered[-1] + (ranges[-1][1] - ranges[-1][0]), base + buf.nbytes)


if __name__ == "__main__":
    unittest.main()
