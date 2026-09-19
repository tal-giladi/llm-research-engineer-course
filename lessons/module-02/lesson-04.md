# 02.4 · Autograd internals

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="lesson-03.md">02.3 · Backprop through linear + softmax + cross-entropy by hand</a> — the VJP of a concrete op and the fan-in summation of gradients. <a href="lesson-02.md">02.2</a>'s computational-graph picture (nodes = ops, edges = tensors) is used throughout. A first acquaintance with <a href="../module-00/lesson-03.md">00.3 · Autograd: backward and grad</a> helps.</p>
<p><strong>You will learn:</strong> exactly how PyTorch builds the computational graph <em>dynamically</em> as you run the forward pass — each result tensor's <code>.grad_fn</code>, what a <strong>leaf</strong> tensor with <code>requires_grad</code> is, and how <code>.backward()</code> runs a <strong>topological</strong> sweep applying each node's local VJP. Then <code>retain_graph</code> and <code>create_graph</code> briefly. Finally you build a <strong>from-scratch scalar autodiff engine</strong> (micrograd-style) supporting <code>+</code>, <code>*</code>, and <code>tanh</code> with a working <code>.backward()</code>, and verify it reproduces PyTorch's gradients.</p>
<p><strong>Why this matters for ML:</strong> until now <code>.backward()</code> has been a black box that happens to print the right numbers. After this lesson it is not magic — it is a graph, a topological sort, and one local-derivative rule per operation, which you will have written yourself in about 60 lines. Every training loop in the rest of the course calls <code>.backward()</code>; knowing precisely what it does is what lets you debug vanishing gradients, <code>retain_graph</code> errors, and memory blowups later.</p>
</div>

## 1. Intuition: the graph is built while you compute, not before

Some frameworks (the older TensorFlow 1.x, Theano) make you *define* a computation graph first, compile it, then feed numbers through. PyTorch is **define-by-run**: there is no separate graph-building step. As each ordinary Python line executes a tensor operation, PyTorch quietly records that operation into a graph *on the fly*. Run an `if` and only one branch is recorded; run a Python `for` loop and the graph grows one iteration's worth per pass. The graph is a side effect of doing the forward computation, and it is rebuilt from scratch on every forward pass.

This is why PyTorch feels like "just Python": you write the math, and the graph assembles itself behind the tensors. The backward pass then walks that recorded graph. Our whole job this lesson is to see the recording mechanism and the walk clearly enough to rebuild both.

## 2. Leaves, `grad_fn`, and how the graph is stored

Three pieces of state on every tensor make the graph work.

- **`requires_grad`** — a boolean flag. Set it on the tensors you want gradients for (your parameters and inputs). Any tensor computed from a `requires_grad=True` tensor inherits the flag, so the flag propagates forward through the graph automatically.
- **leaf tensor** — a tensor you created directly (a parameter, an input) rather than as the result of an op. Leaves with `requires_grad=True` are where gradients *accumulate*: after `.backward()`, their `.grad` is filled. Non-leaf (intermediate) tensors do not keep a `.grad` by default.
- **`.grad_fn`** — on every *non-leaf* tensor, a reference to the operation that produced it, together with references back to its inputs. This is the edge structure of the graph. A leaf has `grad_fn = None` (nothing produced it); an intermediate has a `grad_fn` like `<AddBackward0>` or `<MulBackward0>` naming the op and holding the recipe to push a gradient back to that op's inputs.

You can read all of this directly:

```python
import torch

x = torch.tensor(2.0, requires_grad=True)   # a LEAF: grad_fn is None
y = 3 * x + 1                                # intermediate
z = y ** 2                                   # intermediate (the root we'll backward from)

print(x.is_leaf, x.grad_fn)                  # True  None
print(y.is_leaf, y.grad_fn)                  # False <AddBackward0 object ...>
print(z.is_leaf, z.grad_fn)                  # False <PowBackward0 object ...>
```

`z.grad_fn` is `PowBackward0` (the backward of `**`); follow its stored input references and you reach `y`'s `AddBackward0`, and from there `x`. That linked structure — result tensor → `grad_fn` → input tensors → their `grad_fn`s → … → leaves — **is** the computational graph of lesson 02.2, stored as objects hanging off the tensors.

<div class="callout pt"><p>Each <code>grad_fn</code> also holds the <strong>saved tensors</strong> its VJP needs (recall from 02.1 that the backward of <code>y**2</code> needs the value of <code>y</code>). Those saved activations are what the forward pass caches and what dominate training memory. You can inspect them: <code>z.grad_fn._saved_self</code> is the <code>y</code> that <code>PowBackward0</code> stashed. Freeing the graph (see §4) is what releases them.</p></div>

## 3. What `.backward()` does: seed, topo-sort, sweep

When you call `z.backward()` on a scalar `z`, PyTorch runs a precise algorithm:

1. **Seed.** Set the gradient of the root to $1$ (this is the length-1 VJP seed of lesson 02.2: $\frac{dz}{dz} = 1$).
2. **Topological sort.** Order the graph nodes so that every node comes *after* all nodes that produced its inputs — equivalently, so that when we process a node in reverse order, its own output-gradient has already been fully accumulated from everything downstream.
3. **Reverse sweep.** Walk the sorted nodes in reverse. At each node, take its already-accumulated output gradient, apply the node's local VJP (its `grad_fn`), and **add** the result into each input's gradient.

Two subtleties, both of which we met in 02.2:

- **Accumulation, not assignment.** A tensor used in several places receives a gradient contribution along each path; the sweep *adds* them (fan-in summation). This is why `.grad` accumulates across `.backward()` calls too, and why you must call `optimizer.zero_grad()` each step — otherwise this step's gradient piles onto last step's.
- **Topological order is mandatory.** You cannot finalize a node's gradient until every downstream contribution has arrived. The sort guarantees that ordering. Process out of order and a node would apply its VJP with an incomplete gradient.

That is the entire algorithm. The rest of PyTorch's autograd is efficiency and coverage (hundreds of ops' VJPs, CUDA kernels, in-place-op tracking) around this core.

## 4. `retain_graph` and `create_graph`, briefly

Two flags you will eventually hit:

- **`retain_graph=True`.** By default, `.backward()` **frees** the graph (and its saved tensors) as it sweeps, because a training step needs the graph only once and freeing it reclaims memory. If you need to call `.backward()` a *second* time on the *same* graph (e.g. two losses sharing a subgraph), the first call must pass `retain_graph=True`, or the second raises "Trying to backward through the graph a second time." It is not something you want routinely — it keeps activations alive.
- **`create_graph=True`.** Makes the backward pass itself a tracked, differentiable computation, so the *gradients* have their own `grad_fn` and you can differentiate them again — i.e. compute **second-order** derivatives (Hessian-vector products, some meta-learning and penalty terms). It is more expensive in time and memory because the backward graph is now retained and recorded.

You will not need either in ordinary pretraining; recognize the error messages and know which flag answers them.

## 5. From scratch: a scalar autodiff engine

Now we prove we understand all of the above by building it. We write a tiny `Value` class — one scalar per node — that records a graph as it computes and whose `.backward()` runs the seed/topo-sort/sweep of §3. It supports `+`, `*`, and `tanh`, which is enough to build a neuron. This is the micrograd design (Andrej Karpathy's teaching engine), reconstructed to match PyTorch's semantics.

The key idea: each operation, besides computing its forward value, **stores a closure** `_backward` that knows that op's local derivative and adds the incoming gradient (times that local derivative) into its inputs' `.grad`. That closure is our hand-written `grad_fn`.

```python
import math

class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = float(data)     # the scalar value (rank-0, plain float)
        self.grad = 0.0             # d(root)/d(self), accumulated in backward
        self._prev = tuple(_children)   # input nodes -> the graph edges
        self._op = _op                  # label, for debugging
        self._backward = lambda: None   # this node's local VJP; default no-op (leaf)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")
        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1: incoming grad flows straight through
            self.grad  += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")
        def _backward():
            # product rule: local derivative w.r.t. each factor is the OTHER factor
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")
        def _backward():
            # d/dx tanh(x) = 1 - tanh(x)^2, and we saved t = tanh(x)
            self.grad += (1.0 - t * t) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # 1) build a topological order: every node after all its inputs
        topo, visited = [], set()
        def build(v):
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)
        # 2) seed the root, 3) sweep in reverse applying each local VJP
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
```

Read it against §3: `build` produces the topological order (a node is appended only after all its `_prev` inputs), `self.grad = 1.0` is the seed, and the reversed loop is the sweep. Each `_backward` closure is exactly one op's VJP, and it uses `+=` so a reused node accumulates over all its paths — the fan-in summation. Every derivative rule here (`1`, the other factor, $1 - t^2$) is one you derived in lessons 02.1 and 02.3.

## 6. Verifying it against PyTorch

We build the same small expression twice — once with `Value`, once with torch tensors — and check both the forward value and every gradient agree.

```python
import torch

# our engine
a = Value(-3.0)
b = Value(2.0)
L = (a * b + b).tanh() * (a + b)     # exercises *, +, and tanh; a and b reused
L.backward()
print(L.data, a.grad, b.grad)
# 0.999329299739067 -1.0020112011051188 -0.9966473983730152

# PyTorch, same expression
ta = torch.tensor(-3.0, requires_grad=True, dtype=torch.float64)
tb = torch.tensor( 2.0, requires_grad=True, dtype=torch.float64)
tL = torch.tanh(ta * tb + tb) * (ta + tb)
tL.backward()
print(tL.item(), ta.grad.item(), tb.grad.item())
# 0.999329299739067 -1.0020112011051188 -0.9966473983730152
```

The two implementations agree to full double precision: forward value $0.99932930$, $\frac{\partial L}{\partial a} = -1.00201120$, $\frac{\partial L}{\partial b} = -0.99664740$. Our 60-line engine does what `torch.autograd` does, on this expression, exactly. That is the proof that `.backward()` is not magic — it is the graph, the topological sort, and one local rule per op that you just wrote.

Trace one gradient to be sure. The forward computes $n = a\cdot b + b = (-3)(2) + 2 = -4$, then $\tanh(-4) = -0.99933$, then $s = a + b = -1$, then $L = \tanh(n)\cdot s = (-0.99933)(-1) = 0.99933$. On the way back: the final `*` sends $s = -1$ into the $\tanh$ branch and $\tanh(n) = -0.99933$ into the $s$ branch; $\tanh$ multiplies its incoming $-1$ by $1 - \tanh^2(n) = 1 - 0.99866 = 0.00134$; and $a$ collects contributions through *both* the product $a\cdot b$ (via $b = 2$) and the sum $a + b$ — the accumulation §5's `+=` handles. The arithmetic lands on $-1.00201$, matching torch.

## 7. This engine lives in the codebase

The `Value` class above is shipped as `llmre/model/micrograd.py`, and the parity check against torch is `tests/test_micrograd.py`:

```python
# code/tests/test_micrograd.py (excerpt)
from llmre.model.micrograd import Value
import torch

def test_micrograd_matches_torch():
    a = Value(-3.0); b = Value(2.0)
    L = (a * b + b).tanh() * (a + b)
    L.backward()
    ta = torch.tensor(-3.0, requires_grad=True, dtype=torch.float64)
    tb = torch.tensor( 2.0, requires_grad=True, dtype=torch.float64)
    tL = torch.tanh(ta * tb + tb) * (ta + tb)
    tL.backward()
    assert abs(L.data - tL.item()) < 1e-6
    assert abs(a.grad - ta.grad.item()) < 1e-6
    assert abs(b.grad - tb.grad.item()) < 1e-6
```

Run it from `code/`:

```
py -m pytest -q code/tests/test_micrograd.py
```

It passes. From here, adding a `pow`, an `exp`, or a `relu` to `Value` is just one more forward line plus its one-line local derivative — the same pattern extends to the full set of ops PyTorch supports, only PyTorch's versions operate on whole tensors (so each `_backward` is a tensor VJP, e.g. the $W^\top\mathbf{g}$ of lesson 02.3) and run on the GPU.

## 8. Under the hood: how PyTorch differs from our toy

Our engine captures the *algorithm* faithfully; PyTorch differs in engineering, not in principle.

- **Tensors, not scalars.** Each PyTorch node is a whole tensor, so each `_backward` is a tensor VJP (a matmul, a reduction, a broadcast-aware add) rather than a float multiply. The topo-sort and reverse sweep are identical.
- **C++ engine and op coverage.** The graph walk runs in a C++ autograd engine, and there are hundreds of registered backward formulas (like the fused `cross_entropy` backward from 02.3). Many are hand-written closed forms for speed and stability, not naive compositions.
- **Memory management.** PyTorch frees the graph and saved tensors after the sweep (unless `retain_graph=True`), and offers activation checkpointing to trade compute for memory. Our toy just lets Python's garbage collector reclaim the `Value` objects.
- **In-place and non-differentiable ops.** PyTorch tracks in-place modifications and errors if one would corrupt a value some `_backward` still needs. Our toy never mutates `data`, so it sidesteps the issue entirely.

None of these change the mental model: **build a graph on the forward pass, seed the scalar root with $1$, topologically sort, sweep backward applying each op's local VJP, accumulate into leaves.** That sentence is `loss.backward()`.

## Exercise

Add a `def relu(self)` method to `Value`. Forward: $\text{relu}(x) = \max(0, x)$. Backward: the local derivative is $1$ where $x > 0$ and $0$ where $x \le 0$. Then verify against `torch.relu` on a small expression that mixes it with `+` and `*`.

<details><summary>Optional hint</summary>

Mirror the `tanh` method: compute the forward value, make the `out` node with `(self,)` as its child, and define a `_backward` closure that adds `local_derivative * out.grad` into `self.grad`.

</details>

<details><summary>Stronger hint</summary>

The local derivative is `1.0 if self.data > 0 else 0.0`. So `self.grad += (1.0 if self.data > 0 else 0.0) * out.grad`. The forward value is `max(0.0, self.data)`.

</details>

<details><summary>Solution</summary>

```python
def relu(self):
    out = Value(max(0.0, self.data), (self,), "relu")
    def _backward():
        self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
    out._backward = _backward
    return out
```

Verification:

```python
import torch
x = Value(-2.0); w = Value(3.0)
y = (x * w).relu() + x          # (-2*3)=-6 -> relu 0 -> +x = -2
y.backward()
print(y.data, x.grad, w.grad)   # -2.0  1.0  0.0

tx = torch.tensor(-2.0, requires_grad=True)
tw = torch.tensor(3.0,  requires_grad=True)
ty = torch.relu(tx * tw) + tx
ty.backward()
print(ty.item(), tx.grad.item(), tw.grad.item())  # -2.0  1.0  0.0
```

Both give `y = -2.0`, `x.grad = 1.0` (the relu branch is dead since $x\cdot w = -6 \le 0$, so only the `+x` path contributes), `w.grad = 0.0` (its only path runs through the dead relu). They match.

</details>

## Debugging exercise

This code raises `RuntimeError: Trying to backward through the graph a second time`. Why, and what is the one-word fix (and the better fix)?

```python
import torch
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2
y.backward()      # ok
z = y * 3         # reuse y ...
z.backward()      # RuntimeError
```

<details><summary>Solution</summary>

The first `y.backward()` freed the graph that produced `y` (including the saved tensors `PowBackward0` needs). `z = y * 3` builds a new node *on top of the freed `y` subgraph*, so `z.backward()` tries to walk through the already-freed part and fails.

- **Quick fix:** `y.backward(retain_graph=True)` on the first call keeps the graph alive for the second backward.
- **Better fix:** usually you did not mean to backward twice at all — you meant to compute one loss and back-propagate once. Restructure so there is a single scalar loss and a single `.backward()`, or detach `y` (`y.detach()`) if the second computation should not propagate gradients into the first. `retain_graph=True` is correct only when you genuinely need two backward passes over shared structure, and it costs memory.

</details>

## Check yourself

<details><summary>What distinguishes a leaf tensor from an intermediate one, and which one carries a <code>.grad</code> after backward?</summary>

A leaf is created directly (a parameter or input) and has `grad_fn = None`; an intermediate is the output of an op and has a `grad_fn`. By default only leaves with `requires_grad=True` accumulate and keep a `.grad` after `.backward()`. Intermediates' gradients are computed and used during the sweep but not retained (unless you call `.retain_grad()`).

</details>

<details><summary>Why must <code>.backward()</code> visit nodes in topological (reverse) order rather than any order?</summary>

Because a node's local VJP needs its *output* gradient fully accumulated first, and the output gradient is the sum of contributions from every downstream node. Reverse topological order guarantees all downstream nodes are processed before the node itself, so its incoming gradient is complete when its `_backward` runs.

</details>

<details><summary>In the <code>Value</code> engine, why does <code>_backward</code> use <code>+=</code> and not <code>=</code>?</summary>

Because a node can feed multiple operations (like `a` feeding both `a*b` and `a+b`). Each downstream op contributes a gradient term along its edge, and the chain rule sums them. `+=` accumulates those contributions; `=` would overwrite all but the last, giving a wrong gradient. This is the same reason PyTorch accumulates into `.grad` and you must `zero_grad()` between steps.

</details>

<details><summary>You call <code>loss.backward()</code> twice in a loop without <code>zero_grad()</code>. What happens to the parameter gradients?</summary>

They accumulate: after two backward passes each `.grad` holds the *sum* of the two steps' gradients, so the optimizer steps on stale, doubled-up gradients. The fix is `optimizer.zero_grad()` (or `param.grad = None`) at the top of each iteration. Gradient accumulation across micro-batches deliberately exploits this behavior — but only when you intend it (Module 7).

</details>

## Next

You can now explain `.backward()` end to end — the dynamically built graph on the tensors, the leaf/`grad_fn` structure, the seed–topo-sort–sweep algorithm, and the per-op local VJPs — and you have written a working autodiff engine that matches PyTorch. That closes the gradient machinery of the course. With derivatives, VJPs, the softmax+cross-entropy output gradient, and autograd internals in hand, the next module turns gradients into learning: how to actually step the parameters downhill — SGD, momentum, and Adam.

Continue to [03.1 · Gradient descent, SGD, momentum](../module-03/lesson-01.md). See also the [curriculum map](../../curriculum/course-outline.md) for how the gradient machinery feeds the optimization and Transformer modules.
