"""Minimal conftest for the kernel tests that must run without sglang.

These tests only need numpy and pytest.  Keeping this file free of torch and
sglang imports lets CI run them in a small dedicated job instead of installing
the full runtime, which is what previously failed during environment setup.
"""

import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_sglang: test needs the sglang package to be importable",
    )
