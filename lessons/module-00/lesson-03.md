# 00.3 · Autograd: backward and grad

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="lesson-01.md">00.1 · Tensors: shape, dtype, device</a> and <a href="lesson-02.md">00.2 · Ops, broadcasting, views vs copies</a> — you should be comfortable with elementwise ops and with scalar (rank-0) tensors.</p>
<p><strong>You will learn:</strong> how PyTorch automatically computes derivatives. Specifically: <code>requires_grad=True</code>, the dynamic computational graph PyTorch records as you compute, calling <code>.backward()</code> to run the chain rule, and reading the result off <code>.grad</code>. You will differentiate two examples by hand and confirm PyTorch prints the same numbers. Then <code>torch.no_grad()</code>, <code>.detach()</code>, and why gradients <strong>accumulate</strong> (so you must zero them each step).</p>
<p><strong>Why this matters for ML:</strong> training a neural network <em>is</em> repeatedly computing the gradient of a loss with respect to millions of parameters and stepping each one downhill. Doing that by hand is impossible; autograd does it for you, exactly and automatically. Every training loop in this course — up to and including GPT-2 — rests on the three lines <code>loss.backward()</code>, read <code>.grad</code>, then step and zero. This lesson is where those lines start to make sense.</p>
</div>

## 1. Intuition: the derivative tells each number which way to nudge

Training asks a single question over and over: "if I nudge this parameter a tiny bit, does the loss go up or down, and how fast?" That is precisely what a **derivative** is — the rate of change of the loss with respect to one input. If the derivative of the loss with respect to a weight is $+5$, then increasing that weight increases the loss, so to *reduce* the loss we move the weight the other way. Do that for every parameter at once and you are doing gradient descent (next lesson).

The problem is that a language model computes the loss through millions of chained operations. Working out the derivative by hand through all of them is hopeless. **Autograd** solves this by watching every operation you perform, building a record of them (a graph), and then walking that record backwards applying the chain rule mechanically. You write the forward computation as ordinary Python; PyTorch silently prepares to differentiate it.

<div class="callout key"><p><strong>Autograd</strong> = automatic differentiation. As you compute forward, PyTorch records each operation and how to differentiate it. When you call <code>.backward()</code> on the final scalar, it walks that record in reverse, multiplying local derivatives together via the chain rule, and deposits the derivative of that scalar with respect to each leaf tensor into the leaf's <code>.grad</code>.</p></div>

## 2. requires_grad and the graph

By default a tensor is inert — PyTorch does not track operations on it. You opt a tensor into differentiation by setting `requires_grad=True`. From then on, any tensor computed *from* it inherits a link back, and PyTorch records the operation that produced it. This growing record is the **computational graph**: nodes are tensors, edges are operations, and each edge knows how to compute its own local derivative.

It is a **dynamic** graph — it is built fresh, on the fly, every time you run the forward computation, by the ordinary execution of your Python code. There is no separate "compile the model" step. This is why you can use normal `if`/`for`/`while` in a model and the graph simply reflects whatever path actually ran. (Frameworks that build a static graph once, ahead of time, trade this flexibility for other optimizations; PyTorch chose flexibility, and that choice is a big part of why it dominates research.)

## 3. The simplest example, by hand then by autograd

Take $y = x^2$ and ask for $\frac{dy}{dx}$ at $x = 3$. By calculus, $\frac{dy}{dx} = 2x$, so at $x = 3$ it is $2 \cdot 3 = 6$. Now the same thing in PyTorch:

```python
import torch

x = torch.tensor(3.0, requires_grad=True)   # a leaf we want the gradient of
y = x**2                                     # forward: y = 9.0, graph records "square"
y.backward()                                 # walk the graph backward
x.grad                                       # tensor(6.)
```

Line by line: `x` is a scalar tensor with `requires_grad=True`, so it is a **leaf** we care about. Computing `y = x**2` runs the forward pass (`y` holds `9.0`) *and* records that `y` came from squaring `x`. Calling `y.backward()` triggers the backward pass: PyTorch knows the local derivative of "square" is $2x$, evaluates it at the stored $x = 3$, and writes $6$ into `x.grad`. The printed `tensor(6.)` matches our hand calculation exactly.

Three things to internalise from this tiny example:

- You call `.backward()` on the **final scalar** (here `y`). Backward starts from a single number — in real training that number is the loss.
- The gradient lands in `x.grad`, on the **leaf** tensor, not on `y`.
- `x.grad` answers "how does `y` change as `x` changes right here" — the slope at the current value, not a formula.

## 4. A two-node graph, worked by hand and matched

One operation is not convincing; the power of autograd is chaining. Let us build a miniature version of what a single neuron does: multiply an input by a weight, add a bias, then measure squared error against a target.

$$
z = w x + b, \qquad \text{loss} = (z - \text{target})^2.
$$

Pick concrete numbers: $w = 2$, $x = 3$, $b = 1$, $\text{target} = 10$. Forward:

$$
z = 2\cdot 3 + 1 = 7, \qquad \text{loss} = (7 - 10)^2 = (-3)^2 = 9.
$$

Now the derivatives, by the chain rule. First the derivative of the loss with respect to $z$:

$$
\frac{d\,\text{loss}}{dz} = 2(z - \text{target}) = 2(7 - 10) = -6.
$$

Then push that back through $z = wx + b$ to each input. The local derivatives are $\frac{\partial z}{\partial w} = x$, $\frac{\partial z}{\partial b} = 1$, $\frac{\partial z}{\partial x} = w$. Multiply each by $\frac{d\,\text{loss}}{dz} = -6$ (that is the chain rule — the derivative of a chain is the product of the local derivatives):

$$
\frac{d\,\text{loss}}{dw} = -6 \cdot x = -6 \cdot 3 = -18,
$$
$$
\frac{d\,\text{loss}}{db} = -6 \cdot 1 = -6,
$$
$$
\frac{d\,\text{loss}}{dx} = -6 \cdot w = -6 \cdot 2 = -12.
$$

So increasing $w$ sharply reduces the loss (gradient $-18$, the steepest), and $x$ (which we would not train, but can still differentiate) has gradient $-12$. Now PyTorch:

```python
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
x = torch.tensor(3.0)              # note: no requires_grad -> treated as a constant
target = torch.tensor(10.0)

z = w * x + b                      # 7.0
loss = (z - target)**2            # 9.0
loss.backward()

w.grad    # tensor(-18.)
b.grad    # tensor(-6.)
```

Exactly the hand-computed numbers. `loss.backward()` walked the graph `loss ← z ← (w, b)` in reverse, multiplying local derivatives, and deposited $-18$ into `w.grad` and $-6$ into `b.grad`. Because we left `x` with `requires_grad=False`, PyTorch treated it as a constant and did not compute or store a gradient for it — which is exactly how you tell autograd "these are the things to train, and everything else is fixed data."

<div class="callout pt"><p>Gradients only appear on <strong>leaf</strong> tensors that had <code>requires_grad=True</code>. Intermediate tensors like <code>z</code> do participate in the backward pass (their local derivatives are used), but they do not keep a <code>.grad</code> by default — PyTorch frees that to save memory. In a real model, the leaves are the model's parameters, and that is precisely the set of tensors you want gradients for.</p></div>

## 5. Turning autograd off: no_grad and detach

Autograd's bookkeeping costs time and memory — every recorded operation stores what it needs for the backward pass. During training you want that. But at other times (running the model to make predictions, or doing arithmetic that must *not* be part of the gradient) you want it off. There are two tools.

**`torch.no_grad()`** is a context manager that disables graph recording for everything inside it:

```python
w = torch.tensor(2.0, requires_grad=True)
with torch.no_grad():
    y = w * 3                # computed, but NOT recorded
y.requires_grad             # False
```

You wrap evaluation and generation in `torch.no_grad()` so PyTorch does not waste memory building a graph you will never call `.backward()` on. You will also wrap the optimizer's parameter update in it next lesson, because *changing* the weights is not part of the function whose gradient we want.

**`.detach()`** returns a tensor that shares the same data but is cut out of the graph — a value with the history clipped off:

```python
w = torch.tensor(2.0, requires_grad=True)
d = w.detach()              # same number, but no link back to w
d.requires_grad             # False
```

Use `.detach()` when you want to use a value as a plain constant — for logging, or for feeding something into a computation whose gradient must not flow back into `w`.

## 6. Why gradients accumulate — and zero_grad

Here is the single most surprising thing about autograd for newcomers: **`.backward()` adds to `.grad`; it does not overwrite it.** Watch:

```python
p = torch.tensor(1.0, requires_grad=True)

(p * 2).backward()
p.grad          # tensor(2.)     d(2p)/dp = 2

(p * 2).backward()
p.grad          # tensor(4.)     2 got ADDED to the existing 2, not replaced
```

The second `backward()` computed the same gradient, $2$, and **added** it to the $2$ already sitting in `p.grad`, giving $4$. PyTorch does this on purpose — accumulation is what makes it easy to sum gradients from several pieces of a batch (you will use this deliberately for "gradient accumulation" in [module 7](../module-07/lesson-02.md)). But it means that in an ordinary training loop, if you do not clear the gradients between steps, step 2's gradient is polluted by step 1's leftover, step 3's by both, and training silently goes wrong.

The fix is to **zero the gradients** before each `backward()`. In a real loop the optimizer does it for you with `optimizer.zero_grad()` (next lesson); manually it is `p.grad.zero_()` or `p.grad = None`.

<div class="callout warn"><p>Forgetting <code>zero_grad()</code> is one of the most common training bugs, and an insidious one: the code runs without error, the loss just behaves strangely (often failing to converge, because each step's gradient is contaminated by all previous steps'). If a training loop "runs but does not learn", the first thing to check is that gradients are being zeroed every step.</p></div>

## Debugging exercise: the silently detached tensor

The following is supposed to compute a gradient, but `x.grad` comes back as `None`. Diagnose why before reading on.

```python
x = torch.tensor(3.0, requires_grad=True)
y = x.detach() ** 2       # intended: y = x^2
y.backward()              # RuntimeError, or...
print(x.grad)             # None
```

<details><summary>What is wrong, and what does PyTorch do?</summary>

`x.detach()` returns a tensor cut out of the graph — it has no link back to `x` and `requires_grad=False`. So `y = x.detach()**2` is computed from a *constant*, not from `x`; the graph has no path from `y` back to `x`. Calling `y.backward()` fails outright with `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`, because `y` is not attached to anything differentiable. Even if you forced it, `x.grad` would stay `None` — no gradient can flow to `x` through a detached value.

The fix is to remove the stray `.detach()`: `y = x ** 2`, after which `y.backward()` sets `x.grad` to `tensor(6.)`. The general lesson: a `.detach()` (or a `with torch.no_grad():`) anywhere on the path between your loss and a parameter silently severs the gradient. When a parameter's `.grad` is unexpectedly `None` or zero, hunt for a detach, a `no_grad`, or a place where the tensor was converted to a non-differentiable dtype or to a NumPy array and back.

</details>

## Check yourself

<details><summary>For <code>y = x**3</code> with <code>x = torch.tensor(2.0, requires_grad=True)</code>, what will <code>x.grad</code> be after <code>y.backward()</code>?</summary>

$\frac{dy}{dx} = 3x^2 = 3 \cdot 2^2 = 12$. So `x.grad` is `tensor(12.)`.

</details>

<details><summary>In the two-node example, why does <code>x</code> not get a gradient while <code>w</code> and <code>b</code> do?</summary>

`x` was created with `requires_grad=False` (the default), so autograd treats it as a constant and does not track or store a gradient for it. `w` and `b` had `requires_grad=True`, marking them as leaves we want to differentiate, so their `.grad` is populated. This is exactly how you separate "parameters to train" from "fixed input data".

</details>

<details><summary>You call <code>loss.backward()</code> twice in a row (without zeroing) on the same computation. What happens to the parameter gradients?</summary>

They double. `.backward()` accumulates into `.grad` rather than overwriting, so the second call adds the same gradient again. To avoid this in a training loop you must zero the gradients (e.g. `optimizer.zero_grad()`) before each backward pass.

</details>

<details><summary>Why do we wrap model evaluation / generation in <code>with torch.no_grad():</code>?</summary>

Because we are not going to call `.backward()` there, so building the computational graph would waste memory and time storing intermediate values for a backward pass that never happens. `torch.no_grad()` disables that recording, making the forward pass cheaper and lighter.

</details>

<details><summary>A parameter's <code>.grad</code> is <code>None</code> after <code>loss.backward()</code>, even though the parameter clearly feeds into the loss "on paper". Name two likely causes.</summary>

(1) The path from loss to parameter passes through a `.detach()` or a `with torch.no_grad():` block, severing the graph. (2) The parameter was not actually a leaf with `requires_grad=True`, or was replaced/reassigned in a way that broke its connection (e.g. converted to NumPy and back, or overwritten by an in-place non-tracked op). Both cut the gradient path so no gradient reaches the parameter.

</details>

## Next

You can now let PyTorch compute exact gradients of any scalar with respect to your parameters, and you know the two rules that bite newcomers: gradients accumulate, and a stray detach kills them. The last piece is to package parameters into a model and put gradients to work: `nn.Module`, `nn.Linear`, an optimizer, and the canonical five-line training loop you will reuse for the entire course — culminating in fitting a real line to noisy data and watching the loss fall.

Continue to [00.4 · nn.Module and a minimal training loop](lesson-04.md).
