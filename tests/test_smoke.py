"""Smoke test: verify the package imports and exposes a version."""

import commercebench


def test_package_imports():
    assert commercebench.__version__ == "0.1.0"
