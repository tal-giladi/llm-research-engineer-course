"""Parity test: our scalar autodiff must match torch.autograd.

We build the same small expression with our :class:`llmre.model.micrograd.Value`
and with ``torch`` tensors (``requires_grad=True``), run both backward passes,
and check the forward value and every input gradient agree to 1e-6.
"""
import torch

from llmre.model.micrograd import Value


def test_micrograd_matches_torch():
    # Expression: L = tanh(a*b + b) * (a + b), with a = -3.0, b = 2.0.
    # It exercises +, *, and tanh, and reuses a and b at several nodes so the
    # gradient must correctly accumulate over multiple paths.
    a = Value(-3.0)
    b = Value(2.0)
    L = (a * b + b).tanh() * (a + b)
    L.backward()

    ta = torch.tensor(-3.0, requires_grad=True, dtype=torch.float64)
    tb = torch.tensor(2.0, requires_grad=True, dtype=torch.float64)
    tL = torch.tanh(ta * tb + tb) * (ta + tb)
    tL.backward()

    assert abs(L.data - tL.item()) < 1e-6
    assert abs(a.grad - ta.grad.item()) < 1e-6
    assert abs(b.grad - tb.grad.item()) < 1e-6


def test_micrograd_reused_node_accumulates():
    # x used twice: y = x*x + x  => dy/dx = 2x + 1. At x = 4.0 that is 9.0.
    x = Value(4.0)
    y = x * x + x
    y.backward()
    assert abs(y.data - 20.0) < 1e-6
    assert abs(x.grad - 9.0) < 1e-6
