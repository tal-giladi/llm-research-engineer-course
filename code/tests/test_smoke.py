"""Smoke test: the package imports and torch is available.

Real per-algorithm tests are added as each module is built (see tests/ alongside).
"""
import importlib


def test_package_imports():
    m = importlib.import_module("llmre")
    assert hasattr(m, "__version__")


def test_torch_available():
    import torch

    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert tuple(x.shape) == (2, 3)
    assert x.sum().item() == 15.0
