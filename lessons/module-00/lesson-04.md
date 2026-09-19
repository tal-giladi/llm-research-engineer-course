# 00.4 · nn.Module and a minimal training loop

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="lesson-03.md">00.3 · Autograd: backward and grad</a> (you must understand <code>.backward()</code>, reading <code>.grad</code>, and why gradients accumulate), building on <a href="lesson-01.md">00.1</a> and <a href="lesson-02.md">00.2</a>.</p>
<p><strong>You will learn:</strong> how PyTorch packages parameters into a model with <code>nn.Module</code>; what <code>nn.Linear</code> is and why its weight and bias carry gradients; how <code>.parameters()</code> hands the whole set to an optimizer; and <strong>the</strong> five-step training loop that every model in this course — up to GPT-2 — reuses unchanged. You will run a complete example that fits $y = 2x + 1$ from noisy data and watch the loss fall and the learned weight approach 2 and bias approach 1.</p>
<p><strong>Why this matters for ML:</strong> everything after this module is variations on this one loop. The model gets bigger (a Transformer instead of a single line), the data gets richer (token sequences instead of numbers), the optimizer gets fancier (AdamW instead of SGD), but the skeleton — predict, measure loss, zero grads, backward, step — is identical. If you can read and write this loop cold, you can read every training script in the course.</p>
</div>

## 1. Intuition: bundle the parameters, then repeat a fixed recipe

Last lesson we differentiated a loss with respect to a couple of loose scalars `w` and `b`. A real model has millions of parameters, and tracking them as loose variables would be unmanageable. `nn.Module` is PyTorch's container for a model: it holds the parameters, knows how to run the forward computation, and can hand you every parameter at once so an optimizer can update them together.

Training is then a loop of the same five steps, forever:

1. **predict** — run the model forward on a batch of inputs.
2. **loss** — measure how wrong the predictions are against the targets.
3. **zero_grad** — clear last step's gradients (recall they accumulate).
4. **backward** — autograd fills every parameter's `.grad`.
5. **step** — the optimizer nudges every parameter downhill using its `.grad`.

That is the whole recipe. Let us build each piece.

## 2. nn.Linear: a learnable affine map

The single most common building block is `nn.Linear`. `nn.Linear(in_features, out_features)` represents the operation

$$
\mathbf{y} = \mathbf{x} W^\top + \mathbf{b},
$$

where $W$ is a learnable weight matrix of shape `(out_features, in_features)` and $\mathbf{b}$ is a learnable bias vector of shape `(out_features,)`. It is exactly the $wx + b$ of the previous lesson, generalised to vectors and matrices. "Learnable" means its `W` and `b` are created with `requires_grad=True`, so autograd tracks them and the optimizer can update them.

```python
import torch
import torch.nn as nn

layer = nn.Linear(1, 1)          # 1 input feature -> 1 output feature
layer.weight.shape               # torch.Size([1, 1])   the W matrix
layer.bias.shape                 # torch.Size([1])      the b vector
layer.weight.requires_grad       # True  -- it is a parameter, so it carries grad
```

`nn.Linear(1, 1)` has exactly one weight and one bias — two numbers — which is what makes it perfect for fitting a line. In [module 5](../module-05/lesson-04.md) the very same `nn.Linear` becomes the QKV and output projections of attention and the two layers of the MLP, just with large `in`/`out` features; nothing about it changes.

<div class="callout pt"><p>A <code>Parameter</code> is just a tensor with <code>requires_grad=True</code> that an <code>nn.Module</code> registers as "part of the model". Registration is what makes it show up in <code>.parameters()</code> and get moved by <code>.to(device)</code>. The distinction between a plain tensor and a parameter is bookkeeping: parameters are the tensors the optimizer is allowed to change.</p></div>

## 3. .parameters() and the optimizer

`model.parameters()` returns an iterator over every registered parameter tensor. You hand that to an **optimizer**, which is the object that knows how to turn `.grad` into an update. The simplest is **SGD** (stochastic gradient descent): for each parameter $\theta$ it does

$$
\theta \leftarrow \theta - \eta \, \theta.\text{grad},
$$

where $\eta$ (the **learning rate**, `lr`) is the step size — how far to move downhill each step. That is the entire SGD update: subtract the learning rate times the gradient. We derive SGD, momentum, and the Adam family properly in [module 3](../module-03/lesson-01.md); here you only need that `optimizer.step()` applies this rule to every parameter, and `optimizer.zero_grad()` clears every parameter's `.grad`.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
```

The optimizer holds a *reference* to the parameters, so when it calls `step()` it mutates them in place. It reads each parameter's `.grad` (filled by `backward()`) and writes the updated value back.

## 4. The canonical training loop

Here is the complete, runnable example. We fabricate 100 noisy points that lie roughly on the line $y = 2x + 1$, then fit a single `nn.Linear(1, 1)` to recover the slope 2 and intercept 1. Read the five-step loop in the middle — it is the shape of every training loop in the course.

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

# --- fake data: y = 2x + 1 + small noise ---
N = 100
x = torch.randn(N, 1)                       # (100, 1) inputs
y = 2 * x + 1 + 0.1 * torch.randn(N, 1)     # (100, 1) targets, true slope 2, intercept 1

# --- model, loss, optimizer ---
model = nn.Linear(1, 1)                      # learns one weight + one bias
loss_fn = nn.MSELoss()                       # mean squared error
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# --- THE training loop (memorise these five lines) ---
for step in range(1001):
    pred = model(x)                          # 1. predict           (100,1)
    loss = loss_fn(pred, y)                  # 2. measure loss      scalar
    optimizer.zero_grad()                    # 3. clear old grads
    loss.backward()                          # 4. fill .grad on every param
    optimizer.step()                         # 5. nudge every param downhill

    if step % 200 == 0:
        print(f"step {step:4d}  loss {loss.item():.4f}")

print(f"learned weight {model.weight.item():.4f}  bias {model.bias.item():.4f}")
```

Running it prints (these are the real numbers from this exact code with seed 42):

```
step    0  loss 2.3003
step  200  loss 0.0078
step  400  loss 0.0078
step  600  loss 0.0078
step  800  loss 0.0078
step 1000  loss 0.0078
learned weight 2.0012  bias 1.0036
```

The loss falls from $2.30$ to $0.0078$ and flattens; the learned weight is $2.0012 \approx 2$ and the bias $1.0036 \approx 1$ — the model recovered the true line. The residual loss of $0.0078$ is not a failure: it is roughly the variance of the noise we added ($0.1$ standard deviation, so $\approx 0.1^2 = 0.01$ per point), which is the best any straight line can do against noisy data. The model fit the signal and correctly refused to fit the noise.

### Reading the five lines

- **`pred = model(x)`** — calling the module runs its forward pass. `model(x)` computes $xW^\top + b$, shape `(100, 1)`. (You call `model(x)`, never `model.forward(x)` directly — the `__call__` wrapper runs hooks PyTorch needs.)
- **`loss = loss_fn(pred, y)`** — `MSELoss` computes the mean of $(\text{pred} - y)^2$ over all 100 points, giving one scalar. It is a tensor, so it carries the graph back to the parameters.
- **`optimizer.zero_grad()`** — clears `weight.grad` and `bias.grad` to zero. Remember from lesson 3: gradients accumulate, so without this every step would pile onto the last and the update would be wrong.
- **`loss.backward()`** — autograd walks from `loss` back through `MSELoss` and the `Linear` to fill `model.weight.grad` and `model.bias.grad`.
- **`optimizer.step()`** — SGD subtracts `lr * grad` from each parameter. This is the only line that *changes* the parameters, and it is why the next `pred` is a little better.

<div class="callout key"><p>This five-line body — <code>pred = model(x)</code>; <code>loss = loss_fn(pred, y)</code>; <code>optimizer.zero_grad()</code>; <code>loss.backward()</code>; <code>optimizer.step()</code> — is the canonical training loop. It does not change for the rest of the course. Training GPT-2 in <a href="../module-07/lesson-01.md">module 7</a> is this exact loop with a Transformer as <code>model</code>, a batch of token sequences as <code>x</code>, cross-entropy as <code>loss_fn</code>, and AdamW as <code>optimizer</code>.</p></div>

## 5. Under the hood — what each step costs and does

**`.parameters()` is the contract between model and optimizer.** The optimizer never knows what a `Linear` or a Transformer is; it only sees a flat list of tensors, each with a `.grad`. That decoupling is why the same `torch.optim.SGD` trains a one-line model and a billion-parameter one — it just loops over more tensors.

**The order of the five steps matters.** `zero_grad` must come before `backward` (or after `step`), never between `backward` and `step`, or you would erase the gradients before the optimizer uses them. `step` must come after `backward`, because it reads the `.grad` that `backward` just wrote. A different order does not error — it silently trains wrong.

**Memory during training is several copies of the parameters.** Each parameter needs storage for its value, its `.grad` (same size), and, for richer optimizers, extra per-parameter state (Adam keeps two more tensors per parameter). For this two-parameter model that is nothing, but it is the accounting that decides whether GPT-2 fits in your GPU — we make that memory budget precise in [module 7](../module-07/lesson-04.md).

<div class="hw">
<p><strong>Hardware for this lesson.</strong> Minimum: any CPU. Recommended: same — no GPU needed. Expected runtime: well under one second for all 1001 steps. GPU memory: n/a. GPU-hours: 0. <strong>CPU-only: yes</strong> — this is the last lesson that trains instantly on a laptop; the compute budget grows steadily from module 6 onward.</p>
</div>

## Exercise

The learning rate `lr` controls the step size. In the loop above it is `0.1` and training converges smoothly. Change it and observe what happens: try `lr=1.0` and `lr=2.0`. What goes wrong, and why?

<em>Optional hint:</em> print the loss every step (not every 200) and watch the first ten values for each learning rate.

<em>Stronger hint:</em> the SGD update is $\theta \leftarrow \theta - \eta\,\nabla$. If $\eta$ is too large, a step can *overshoot* the minimum and land somewhere with a *larger* gradient, so the next step overshoots even more. Look at whether the loss decreases, oscillates, or grows.

<details><summary>Solution</summary>

With `lr=0.1` the loss falls monotonically to ~0.0078. With `lr=1.0` training **diverges**: each step overshoots the minimum and lands farther out, so the loss *grows* — after 50 steps it is around $1.3\times10^4$ and the learned weight has blown up to about $-72$ instead of $2$. With `lr=2.0` it diverges even faster, overflowing to `inf` (and then `nan`) within a few dozen steps, with the weight around $-1.3\times10^{25}$.

The mechanism: gradient descent assumes small steps down a slope. If the step size is larger than the curvature can tolerate, the update jumps past the bottom of the bowl to a point with an even steeper opposite gradient; the next step jumps back past even farther, and the oscillation amplifies exponentially. This is why the learning rate is the single most important hyperparameter to tune, and why [module 3](../module-03/lesson-03.md) introduces warmup and decay schedules and gradient clipping specifically to keep large models from diverging early in training.

```python
# lr=0.1  -> loss shrinks to 0.0078, weight -> 2.001   (converges)
# lr=1.0  -> loss grows to ~1.3e4,   weight -> -72      (diverges)
# lr=2.0  -> loss -> inf/nan,        weight -> -1.3e25   (blows up)
```

</details>

## Check yourself

<details><summary>Why must <code>optimizer.zero_grad()</code> be called each step, and where in the five-line loop does it go?</summary>

Because `loss.backward()` *accumulates* into `.grad` rather than overwriting it (lesson 3), so without zeroing, each step's gradient would be added to all previous steps' leftovers and the update would be wrong. It goes before `backward()` (equivalently, right after the previous `step()`), never between `backward()` and `step()` — otherwise it would erase the gradients the optimizer is about to use.

</details>

<details><summary>What does <code>model.parameters()</code> return, and why does the optimizer need it?</summary>

An iterator over every registered parameter tensor of the model (here, the `Linear`'s weight and bias). The optimizer needs it to know which tensors to update: it holds references to them, reads each one's `.grad` after `backward()`, and applies the update rule ($\theta \leftarrow \theta - \eta\,\theta.\text{grad}$ for SGD) in place. The optimizer knows nothing about the model's architecture — just this flat list of tensors.

</details>

<details><summary>After training, the learned weight is 2.0012 and bias 1.0036, and the loss plateaus at 0.0078 rather than 0. Is that a bug?</summary>

No. The data was generated as $2x + 1$ plus Gaussian noise of standard deviation $0.1$. No straight line can fit the noise, so the best achievable mean squared error is about the noise variance, $\approx 0.1^2 = 0.01$. A residual loss of $0.0078$ and weight/bias within a few thousandths of the true 2 and 1 means the model recovered the signal and correctly ignored the noise — exactly right.

</details>

<details><summary>Why do we call <code>model(x)</code> and not <code>model.forward(x)</code>?</summary>

`nn.Module.__call__` runs the forward pass *and* invokes any registered hooks and bookkeeping PyTorch relies on. Calling `.forward()` directly skips that machinery. `model(x)` is the supported way to run a module; `forward` is what you *define*, not what you *call*.

</details>

<details><summary>Trace the shapes: with <code>x</code> of shape (100, 1) and <code>nn.Linear(1, 1)</code>, what shape is <code>pred</code>, and what shape is <code>loss</code>?</summary>

`Linear(1, 1)` maps the last dimension from 1 to 1, so `pred = model(x)` has shape `(100, 1)` — one prediction per input. `MSELoss` reduces all of them to a single mean, so `loss` is a scalar (rank-0 tensor, shape `()`), which is what `.backward()` requires.

</details>

## Next

That completes Module 0. You can now create and inspect tensors; do the arithmetic, reductions, matmuls, and broadcasts of a forward pass; let autograd compute exact gradients; and assemble an `nn.Module` and train it with the canonical five-line loop. Every later module stands on these mechanics. The next module steps back to the *why*: the probability and information theory that define what a language model is actually predicting, and the cross-entropy loss that the training loop above will minimise.

Continue to [01.1 · Random variables, expectation, variance](../module-01/lesson-01.md).
