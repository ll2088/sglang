"""Unit tests for the opt-in rank-dependent safetensors file shuffle."""

import os
import unittest
from unittest.mock import patch

from sglang.multimodal_gen.runtime.loader import weight_utils
from sglang.multimodal_gen.runtime.loader.weight_utils import (
    _get_shuffle_rank,
    _shuffle_files,
    _shuffle_files_enabled_from_env,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

ENV = "SGLANG_SHUFFLE_SAFETENSORS_FILES"
FILES = [f"model-{i:05d}-of-00008.safetensors" for i in range(1, 9)]


class _FakeBackend:
    name = "fake"

    def iter_weights(self, files, **kwargs):
        for path in files:
            yield path, None


class TestShuffleSafetensorsFiles(CustomTestCase):
    def test_env_switch_defaults_to_off(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV, None)
            self.assertFalse(_shuffle_files_enabled_from_env())
        for value, expected in (("1", True), ("true", True), ("0", False), ("", False)):
            with patch.dict(os.environ, {ENV: value}):
                self.assertIs(_shuffle_files_enabled_from_env(), expected, value)

    def test_order_is_reproducible_and_input_independent(self):
        with patch.object(weight_utils, "_get_shuffle_rank", return_value=3):
            first = _shuffle_files(FILES)
            second = _shuffle_files(list(reversed(FILES)))
        self.assertEqual(first, second)
        self.assertEqual(sorted(first), sorted(FILES))

    def test_ranks_get_different_orders(self):
        orders = set()
        for rank in range(4):
            with patch.object(weight_utils, "_get_shuffle_rank", return_value=rank):
                orders.add(tuple(_shuffle_files(FILES)))
        self.assertGreater(len(orders), 1)

    def test_rank_falls_back_to_zero_outside_distributed(self):
        with patch("torch.distributed.is_initialized", return_value=False):
            self.assertEqual(_get_shuffle_rank(), 0)
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=5),
        ):
            self.assertEqual(_get_shuffle_rank(), 5)

    def test_iterator_shuffles_only_when_enabled(self):
        calls = []

        def fake_shuffle(files):
            calls.append(list(files))
            return sorted(files, reverse=True)

        files = list(FILES)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(ENV, None)
            with (
                patch.object(weight_utils, "_shuffle_files", side_effect=fake_shuffle),
                patch.object(
                    weight_utils, "_scan_safetensors_files", return_value=([], {})
                ),
                patch.object(
                    weight_utils, "select_weight_reader", return_value=_FakeBackend()
                ),
            ):
                default_order = [
                    name
                    for name, _ in weight_utils.safetensors_weights_iterator(files)
                ]
                self.assertEqual(default_order, files)
                self.assertEqual(calls, [])

                shuffled_order = [
                    name
                    for name, _ in weight_utils.safetensors_weights_iterator(
                        files, shuffle_safetensors_files=True
                    )
                ]
        self.assertEqual(calls, [files])
        self.assertEqual(shuffled_order, sorted(files, reverse=True))


if __name__ == "__main__":
    unittest.main()
