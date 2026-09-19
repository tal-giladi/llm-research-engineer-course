"""A tiny scalar-valued reverse-mode autodiff engine (micrograd-style).

This module exists to demystify what ``tensor.backward()`` does in PyTorch. It
implements a single :class:`Value` type that wraps one Python ``float`` and
records, for every arithmetic operation, how to push a gradient back to its
inputs (the local derivative / vector-Jacobian product for that op). Calling
:meth:`Value.backward` topologically sorts the recorded graph and applies the
chain rule once per node, exactly as autograd does — only here every "tensor"
is a rank-0 scalar, so there are no shapes to track.

Everything here is plain Python ``float`` arithmetic; there are no tensors, no
dtypes, and no devices. It is meant for teaching and for the parity test in
``tests/test_micrograd.py`` that checks these gradients against
``torch.autograd`` on a small expression.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable


class Value:
    """A single scalar node in a reverse-mode autodiff graph.

    Attributes:
        data (float): the scalar value produced at this node (rank-0, a plain
            Python float — no shape, dtype, or device).
        grad (float): d(output)/d(this node), accumulated during
            :meth:`backward`. Starts at ``0.0``.
        _prev (tuple[Value, ...]): the input nodes this value was computed from.
        _backward (Callable[[], None]): closure that, given this node's
            ``grad`` already filled in, adds this node's contribution to each
            input's ``grad`` (the local derivative times the incoming grad).
    """

    def __init__(self, data: float, _children: Iterable["Value"] = (), _op: str = ""):
        self.data: float = float(data)
        self.grad: float = 0.0
        self._prev: tuple[Value, ...] = tuple(_children)
        self._op: str = _op
        self._backward: Callable[[], None] = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g})"

    # --- addition: z = a + b ; dz/da = 1, dz/db = 1 ------------------------
    def __add__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            # local derivative of a+b w.r.t. each input is 1, so the incoming
            # gradient flows straight through to both inputs, added on.
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    # --- multiplication: z = a * b ; dz/da = b, dz/db = a ------------------
    def __mul__(self, other: "Value | float") -> "Value":
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            # product rule: the other factor is the local derivative.
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    # --- tanh: z = tanh(a) ; dz/da = 1 - tanh(a)^2 -------------------------
    def tanh(self) -> "Value":
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    # --- convenience wrappers so ordinary numbers can appear either side ---
    def __radd__(self, other: "Value | float") -> "Value":
        return self + other

    def __rmul__(self, other: "Value | float") -> "Value":
        return self * other

    def __neg__(self) -> "Value":
        return self * -1.0

    def __sub__(self, other: "Value | float") -> "Value":
        return self + (-(other if isinstance(other, Value) else Value(other)))

    def backward(self) -> None:
        """Fill ``.grad`` on every node with d(self)/d(node), via the chain rule.

        Builds a topological order of the graph (every node appears after all
        of its inputs), seeds the output's own gradient to ``1.0`` (d(self)/
        d(self) = 1), then walks the order in reverse, calling each node's
        ``_backward`` closure so it hands its gradient to its inputs. This is
        the scalar version of PyTorch's backward sweep.
        """
        topo: list[Value] = []
        visited: set[int] = set()

        def build(node: "Value") -> None:
            if id(node) not in visited:
                visited.add(id(node))
                for child in node._prev:
                    build(child)
                topo.append(node)

        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
