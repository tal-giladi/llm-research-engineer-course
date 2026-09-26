# 03.1 · Gradient descent, SGD, momentum

<div class="prereq">
<p><strong>Prerequisites:</strong> the gradient $\nabla L$ as the vector of partial derivatives from <a href="#/lessons/module-02/lesson-01">02.1 · Derivatives, partials, chain rule, gradients</a>, and how <code>loss.backward()</code> fills in <code>.grad</code> from <a href="#/lessons/module-02/lesson-03">02.3 · Backprop by hand</a>. Tensors, dtype, device, and in-place ops from <a href="#/lessons/module-00/lesson-01">00.1</a>. The cross-entropy loss that these optimizers actually minimize from <a href="#/lessons/module-01/lesson-04">01.4 · The language-modeling objective</a>.</p>
<p><strong>You will learn:</strong> the gradient-descent update $\theta \leftarrow \theta - \eta\nabla L$ and exactly what each symbol does; what the learning rate $\eta$ controls; why we use <em>stochastic</em> / minibatch gradients (cheap, noisy estimates of the true gradient) instead of the full-batch gradient; and <strong>momentum</strong> as an exponential moving average of gradients, $v \leftarrow \mu v + g;\ \theta \leftarrow \theta - \eta v$. Every rule is worked by hand on tiny numbers and checked against Python.</p>
<p><strong>Why this matters for ML:</strong> training a language model <em>is</em> running one of these update rules millions of times. Everything later in the course — Adam (03.2), warmup and cosine schedules (03.3), the pretraining loop (Module 7) — is a refinement of the three ideas in this lesson. If you understand the plain update and momentum, the rest is bookkeeping on top.</p>
</div>

## 1. Intuition: walking downhill on the loss surface

Training a model means choosing its parameters $\theta$ (all the weight matrices and biases, flattened conceptually into one long vector of numbers) so that a **loss** $L(\theta)$ — a single number measuring how wrong the model is — is as small as possible. You cannot solve for the best $\theta$ directly: $L$ is a function of hundreds of millions of parameters wired through a deep nonlinear network. So instead you do the one thing you *can* do cheaply: from wherever you are, figure out which direction is downhill and take a small step that way. Repeat a few hundred thousand times.

The gradient $\nabla L(\theta)$, from lesson 02.1, is the vector that points in the direction of **steepest increase** of $L$. So $-\nabla L(\theta)$ points in the direction of steepest *decrease*. Gradient descent is nothing more than "take a small step in the direction $-\nabla L$, then recompute the gradient and do it again." The whole subtlety is in *how big* a step and *how* you use the gradient — which is what separates plain descent, SGD, momentum, and Adam.

<div class="callout key"><p>Every optimizer in deep learning answers one question: given the current gradient (and some memory of past gradients), how should I change the parameters? Plain gradient descent's answer is the simplest possible one — step straight downhill by a fixed fraction of the gradient.</p></div>

## 2. Full-batch gradient descent

### 2.1 The update rule

The gradient-descent update, applied to every parameter at once, is

$$
\theta \;\leftarrow\; \theta - \eta\,\nabla L(\theta).
$$

Naming every symbol:

- $\theta$ — the current parameter vector (all the model's weights). The arrow $\leftarrow$ means "overwrite $\theta$ with the right-hand side"; this is an in-place update you repeat every step.
- $\nabla L(\theta)$ — the gradient of the loss at the current $\theta$: a vector the *same shape* as $\theta$, whose $i$-th entry is $\partial L / \partial \theta_i$, how much the loss rises per unit increase in parameter $i$. This is exactly what `loss.backward()` computes and stores in each parameter's `.grad` (lesson 02.3).
- $\eta$ — the **learning rate** (Greek "eta"), a small positive scalar, e.g. $0.1$ or $3\times 10^{-4}$. It sets the step size: the new $\theta$ moves a fraction $\eta$ of the way along the negative gradient.

"Full-batch" means the gradient $\nabla L$ is computed over the **entire** training set: $L(\theta) = \frac{1}{N}\sum_{n=1}^{N} \ell(\theta; x_n)$ averaged over all $N$ examples, and its gradient is the average of the per-example gradients. That is the *true* gradient of the training loss.

### 2.2 What the learning rate controls

The learning rate is the single most important hyperparameter in training. Intuitively:

- **Too small:** each step barely moves; training is correct but crawls, wasting compute.
- **Too large:** a step overshoots the downhill valley and can land somewhere *higher* than where it started; loss oscillates or diverges to `inf`.
- **Just right:** steady, fast decrease.

We will make this precise with a worked example, and lesson 03.3 is entirely about *scheduling* $\eta$ over training rather than holding it fixed.

### 2.3 Worked example: minimize $f(x) = x^2$

Take the simplest possible loss, $f(x) = x^2$, a parabola with its minimum at $x = 0$. Its derivative (the 1-D gradient) is $f'(x) = 2x$. Start at $x = 5$ with learning rate $\eta = 0.1$. The update is $x \leftarrow x - \eta\, f'(x) = x - 0.1\cdot 2x = x - 0.2x = 0.8x$.

**Step 0.** At $x = 5$: gradient $f'(5) = 2\cdot 5 = 10$. New $x = 5 - 0.1\cdot 10 = 5 - 1 = 4.0$.

**Step 1.** At $x = 4$: gradient $f'(4) = 8$. New $x = 4 - 0.1\cdot 8 = 4 - 0.8 = 3.2$.

**Step 2.** At $x = 3.2$: gradient $f'(3.2) = 6.4$. New $x = 3.2 - 0.1\cdot 6.4 = 3.2 - 0.64 = 2.56$.

So the iterates are $5 \to 4.0 \to 3.2 \to 2.56$, marching toward the minimum at $0$. Notice each step multiplies $x$ by $0.8$: the steps get *smaller* as we approach the minimum, because the gradient itself shrinks as $x \to 0$. That automatic slowdown near the bottom is a feature of gradient descent on a smooth loss.

```python
import torch

x = torch.tensor(5.0, requires_grad=True)
eta = 0.1
for step in range(3):
    loss = x**2                 # f(x) = x^2
    loss.backward()             # fills x.grad = 2x
    with torch.no_grad():       # update without tracking this as an op
        x -= eta * x.grad       # x <- x - eta * f'(x)
    print(f"step {step}: grad={x.grad.item():.1f}, x_new={x.item():.4f}")
    x.grad = None               # clear grad before the next backward
# step 0: grad=10.0, x_new=4.0000
# step 1: grad=8.0,  x_new=3.2000
# step 2: grad=6.4,  x_new=2.5600
```

The printed values match the hand computation exactly.

<div class="callout warn"><p>Two `torch` details you must get right or the loop silently breaks. (1) Wrap the parameter update in <code>torch.no_grad()</code>: the update <code>x -= eta * x.grad</code> is arithmetic on a leaf tensor, not part of the model's forward pass, and autograd must not record it. (2) Reset <code>x.grad</code> to <code>None</code> (or zero it) after each step — <code>backward()</code> <em>adds</em> into <code>.grad</code>, so without clearing it you would accumulate the sum of all past gradients. This is why every optimizer below has a <code>zero_grad()</code> method.</p></div>

### 2.4 Why not just use full-batch descent forever?

Because computing $\nabla L$ over the *entire* dataset is astronomically expensive for a real LLM: one gradient would require a forward and backward pass over billions of tokens before you could take a single step. You would take a handful of steps a day. The fix is the next idea.

## 3. Stochastic and minibatch gradient descent (SGD)

### 3.1 The gradient as an average, and a cheap estimate of it

The true loss is an average over all $N$ training examples, so the true gradient is an average of per-example gradients:

$$
\nabla L(\theta) = \frac{1}{N}\sum_{n=1}^{N} \nabla \ell(\theta; x_n).
$$

An average over $N$ terms can be *estimated* by an average over a small random subset. Pick a random **minibatch** $B$ of, say, 32 or 500{,}000 tokens' worth of examples, and compute

$$
g \;=\; \frac{1}{|B|}\sum_{n \in B} \nabla \ell(\theta; x_n) \;\approx\; \nabla L(\theta).
$$

This $g$ is a **noisy but unbiased** estimate of the true gradient: unbiased because a random subset's average has the same expected value as the full average (lesson 01.1 on expectation), noisy because any particular small sample deviates from that expectation. **Stochastic gradient descent** just plugs this estimate into the same update rule:

$$
\theta \;\leftarrow\; \theta - \eta\, g.
$$

"Stochastic" strictly means a batch of one example; "minibatch SGD" means a batch of many; in practice everyone says "SGD" for the minibatch version, and that is what we mean for the rest of the course.

### 3.2 Why noisy-but-cheap wins

The trade is: each SGD step uses a gradient that is *wrong* in detail (it points roughly, not exactly, downhill), but you get to take a step after processing only $|B|$ examples instead of $N$. For a fixed compute budget you take thousands of approximate steps instead of a few exact ones, and that overwhelmingly wins. Two bonuses:

- The noise itself helps. A little randomness in the gradient lets training escape flat regions and poor sharp minima that a perfectly exact gradient could get stuck in.
- Larger batches give *less* noisy gradients (the estimate averages over more samples, so its variance falls like $1/|B|$), but cost proportionally more per step. Choosing the batch size (Module 7) is a compute-vs-noise trade.

<div class="callout key"><p>SGD = plain gradient descent with the true gradient replaced by a cheap random-minibatch estimate $g$. Same update $\theta \leftarrow \theta - \eta g$, but affordable. Every LLM is trained this way; nobody ever computes a full-dataset gradient.</p></div>

### 3.3 The from-scratch SGD optimizer

Here is the SGD class this course uses (`code/src/llmre/optim/sgd.py`), first without momentum. It owns a list of parameter tensors and mutates them in place — exactly the pattern of the loop in §2.3, generalized to many parameters.

```python
class SGD:
    def __init__(self, params, lr, momentum=0.0):
        self.params = list(params)         # leaf tensors with requires_grad=True
        self.lr = lr
        self.momentum = momentum
        self.velocities = [None] * len(self.params)

    @torch.no_grad()
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad                     # the minibatch gradient estimate
            # (momentum handled in §4; plain SGD uses update = g)
            p.add_(g, alpha=-self.lr)      # p <- p - lr * g

    @torch.no_grad()
    def zero_grad(self):
        for p in self.params:
            p.grad = None
```

The test `code/tests/test_optim.py::test_sgd_plain_matches_torch` checks this reproduces `torch.optim.SGD` step-for-step on a small linear-regression problem.

## 4. Momentum

### 4.1 The problem momentum solves

Plain SGD treats every step independently: it looks only at the current gradient and forgets everything before it. On a loss surface shaped like a long narrow valley — steep across the valley, gently sloped along it — this is slow. The gradient mostly points *across* the valley (the steep direction), so SGD zig-zags back and forth across the valley walls while creeping only slowly along the valley floor toward the minimum. The steep direction limits how large $\eta$ can be (too big and it diverges across the valley), which in turn starves the gentle direction of progress.

**Momentum** fixes this by giving the optimizer memory. Instead of stepping along the raw gradient, it steps along a running average of recent gradients. Across the valley, successive gradients point in *opposite* directions and cancel in the average; along the valley, they consistently point the same way and *accumulate*. The optimizer builds up speed ("momentum") in the consistent direction and damps the oscillation.

### 4.2 The mathematics: an exponential moving average of gradients

Momentum keeps a **velocity** vector $v$ (same shape as $\theta$), updates it, and steps along it:

$$
v \;\leftarrow\; \mu\, v + g, \qquad\qquad \theta \;\leftarrow\; \theta - \eta\, v.
$$

Symbols:

- $g$ — the current minibatch gradient (as in SGD).
- $\mu$ — the **momentum coefficient** in $[0, 1)$, typically $0.9$. It sets how much of the past velocity carries over. $\mu = 0$ recovers plain SGD ($v = g$); $\mu = 0.9$ means "keep 90% of the accumulated velocity and add the new gradient."
- $v$ — the velocity, a decaying running sum of past gradients. Initialized so that on the very first step $v = g$ (there is no history yet).

Unrolling the recursion shows $v$ is an **exponential moving average** (EMA): a gradient from $k$ steps ago contributes $\mu^k g$, so recent gradients dominate and old ones fade geometrically. With $\mu = 0.9$, a gradient's influence halves roughly every 7 steps. The velocity therefore smooths out the per-step noise (§3) and accumulates any *consistent* component of the gradient.

<div class="callout key"><p>Momentum replaces "step along this step's gradient" with "step along a decaying average of all recent gradients." Consistent directions add up (acceleration); oscillating directions cancel (damping). One extra buffer $v$ per parameter, and one extra hyperparameter $\mu$.</p></div>

### 4.3 The from-scratch update

The full `step()` with momentum from `code/src/llmre/optim/sgd.py`:

```python
if self.momentum != 0.0:
    buf = self.velocities[i]
    if buf is None:
        buf = g.clone()                    # first step: v = g (no history yet)
        self.velocities[i] = buf
    else:
        buf.mul_(self.momentum).add_(g)    # v <- mu * v + g
    update = buf
else:
    update = g
p.add_(update, alpha=-self.lr)             # theta <- theta - lr * v
```

`buf.mul_(self.momentum).add_(g)` is the two-operation in-place form of $v \leftarrow \mu v + g$: first scale the buffer by $\mu$, then add the new gradient. Initializing `buf = g.clone()` on the first step (rather than starting from a zero buffer and blending) is exactly what `torch.optim.SGD` does, which is why `test_sgd_momentum_matches_torch` matches to $10^{-6}$.

### 4.4 Worked example: momentum converges faster on an ill-conditioned quadratic

Take the "long narrow valley" concretely: $f(x, y) = \tfrac{1}{2}(x^2 + 100\,y^2)$. The curvature is $1$ in $x$ and $100$ in $y$ — a **condition number** of $100$, a stretched bowl. The gradient is $\nabla f = (x,\ 100y)$. Start at $(1, 1)$.

The steep $y$-direction (curvature $100$) forces a small learning rate: plain gradient descent is stable only for $\eta < 2/100 = 0.02$. Use $\eta = 0.0198$ for **both** optimizers, so the comparison is fair — same step size, the only difference is momentum. We measure how many steps each needs to drive the loss below $10^{-3}$.

| optimizer | $\mu$ | steps to loss $< 10^{-3}$ |
|---|---|---|
| plain SGD | $0$ | **269** |
| SGD + momentum | $0.9$ | **57** |

Momentum reaches the target about $4.7\times$ faster with the *same* learning rate. The reason is precisely §4.1: the slow $x$-direction has a small, consistent gradient that momentum accumulates into real speed, while the fast oscillating $y$-direction gets damped. Both start identically (step 1 loss $= 48.50$ for each, since $v = g$ on the first step); by step 20 the momentum run is already at loss $0.07$ while plain SGD is still at $22.5$.

```python
import torch

def run(momentum, eta, steps=300, thresh=1e-3):
    p = torch.tensor([1.0, 1.0]); v = torch.zeros(2)
    curv = torch.tensor([1.0, 100.0])          # f = 0.5*(x^2 + 100 y^2)
    for s in range(steps):
        g = curv * p                           # grad = (x, 100y)
        if momentum > 0:
            v = momentum * v + g               # v <- mu v + g
            p = p - eta * v
        else:
            p = p - eta * g
        loss = 0.5 * (curv * p * p).sum().item()
        if loss < thresh:
            return s + 1
    return None

print(run(0.0, 0.0198))   # 269
print(run(0.9, 0.0198))   # 57
```

<div class="callout pt"><p><code>torch.optim.SGD(params, lr=eta, momentum=0.9)</code> implements exactly the update in §4.3. Its default <code>momentum=0</code> gives plain SGD. PyTorch also offers <code>nesterov=True</code> (a lookahead variant) and <code>dampening</code>; our from-scratch class matches the common <code>dampening=0, nesterov=False</code> setting that everyone actually uses.</p></div>

## 5. Tensor shapes, dtype, device

These optimizers are **shape-agnostic** and **elementwise**. For each parameter tensor `p` (any shape — a `(768, 768)` weight matrix, a `(768,)` bias, whatever; dtype `float32` in normal training, on CPU or GPU):

- `p.grad` has the **same shape, dtype, and device** as `p` — one gradient number per parameter number.
- the velocity buffer `v` (momentum) also has the same shape, dtype, and device as `p`.
- the update `p.add_(v, alpha=-lr)` is elementwise: no shape ever changes, nothing is reduced or broadcast. The optimizer never needs to know whether a tensor is a matrix or a vector.

The memory cost follows directly: plain SGD stores nothing beyond the parameters and their gradients. Momentum stores **one extra tensor the size of the parameters** (the velocity), i.e. it roughly doubles the optimizer's memory footprint beyond params+grads. Keep this number in mind — in 03.2 you will see Adam store *two* such buffers, and Module 9 is largely about who holds those buffers when the model is split across GPUs.

## 6. Under the hood: what one `step()` costs

A single optimizer step, per parameter tensor of $P$ elements:

- **Compute:** a handful of elementwise passes over $P$ numbers — for momentum, one multiply-scale, one add (the velocity update), and one fused multiply-add (the parameter update). That is $O(P)$ work, utterly negligible next to the forward/backward pass that *produced* the gradient (which for a Transformer is dominated by the big matrix multiplies of Module 5). Optimizer step time is essentially memory-bandwidth-bound, not compute-bound: it just streams the parameter, gradient, and velocity tensors through the ALUs once.
- **Memory:** as in §5 — plain SGD adds nothing; momentum adds one parameter-sized buffer.
- **The in-place ops matter.** `mul_`, `add_`, `addcdiv_` (used in 03.2) modify tensors in place rather than allocating new ones. On a real model with billions of parameters, allocating fresh buffers every step would thrash memory; in-place updates are why the optimizer's memory stays flat across training.

This is the payoff of writing the optimizer from scratch: there is no hidden magic. `optimizer.step()` in any framework is exactly the arithmetic above, looped over the parameter tensors.

## Exercise

Implement plain gradient descent (no momentum) on $f(x) = x^2$ starting from $x_0 = 5$, but with learning rate $\eta = 1.0$ instead of $0.1$. Predict the sequence of iterates *before* running it, then run it. What happens, and why?

<details><summary>Optional hint</summary>

The update is $x \leftarrow x - \eta\, f'(x) = x - 1.0\cdot 2x = x - 2x = -x$. Write out $x_0, x_1, x_2$.

</details>

<details><summary>Stronger hint</summary>

With $\eta = 1.0$ the update multiplies $x$ by $(1 - 2\eta) = -1$ each step. What does repeatedly multiplying by $-1$ do to $5$?

</details>

<details><summary>Solution</summary>

The iterates are $5 \to -5 \to 5 \to -5 \to \dots$ — the optimizer bounces between $+5$ and $-5$ forever, never approaching the minimum at $0$. The learning rate is exactly at the stability boundary: for $f(x)=x^2$, gradient descent converges only when $|1 - 2\eta| < 1$, i.e. $0 < \eta < 1$. At $\eta = 1$ the factor is $-1$ (perpetual oscillation); for $\eta > 1$ the iterates *grow* and diverge to $\pm\infty$. This is the concrete version of "learning rate too large" from §2.2: the step overshoots the minimum by more than it should, landing at equal or greater height.

```python
import torch
x = torch.tensor(5.0, requires_grad=True)
for _ in range(4):
    loss = x**2; loss.backward()
    with torch.no_grad(): x -= 1.0 * x.grad
    print(x.item()); x.grad = None
# -5.0, 5.0, -5.0, 5.0
```

</details>

## Debugging exercise

A student writes this training loop and finds the loss decreases far too slowly, as if the learning rate were tiny. The learning rate is $0.1$. What is the bug?

```python
opt = SGD([w], lr=0.1, momentum=0.9)
for x, y in batches:
    loss = mse(model(x, w), y)
    loss.backward()
    opt.step()
```

<details><summary>Solution</summary>

There is no `opt.zero_grad()` in the loop. `backward()` *accumulates* into `w.grad`, so on step $k$ the gradient is the **sum** of the gradients from all $k$ batches so far, not the current one. Early on this makes the effective step wildly too large (and momentum compounds it); the symptoms are erratic, and the accumulated-gradient bug is a classic. The fix is to clear gradients each iteration:

```python
for x, y in batches:
    opt.zero_grad()          # <-- reset w.grad to None first
    loss = mse(model(x, w), y)
    loss.backward()
    opt.step()
```

This is precisely why every optimizer ships a `zero_grad()` and why the warning box in §2.3 exists.

</details>

## Check yourself

<details><summary>In the update $\theta \leftarrow \theta - \eta\nabla L$, why is there a minus sign?</summary>

$\nabla L$ points in the direction of steepest *increase* of the loss. We want to *decrease* the loss, so we step in the opposite direction, $-\nabla L$. The minus sign turns "uphill" into "downhill."

</details>

<details><summary>What makes SGD "stochastic," and what is the cost and benefit versus full-batch descent?</summary>

SGD replaces the true gradient (an average over all $N$ training examples) with an average over a small random minibatch. The estimate is unbiased but noisy. The cost is that each step points only *roughly* downhill; the benefit is that a step is affordable after processing $|B| \ll N$ examples, so you take vastly more steps per unit compute. The noise also helps escape poor minima.

</details>

<details><summary>Momentum with $\mu = 0.9$: how much does a gradient from 10 steps ago still contribute to the current velocity?</summary>

The velocity is an EMA in which a gradient from $k$ steps ago is weighted by $\mu^k$. For $\mu = 0.9$ and $k = 10$: $0.9^{10} \approx 0.349$, so about 35% of its original weight — still non-trivial. This "memory" is what accumulates consistent directions and averages away noise.

</details>

<details><summary>How much extra memory does momentum need compared to plain SGD, for a model with $P$ parameters?</summary>

One extra tensor of $P$ numbers — the velocity buffer $v$, the same shape/dtype/device as the parameters. Plain SGD needs no optimizer state beyond the parameters and their gradients; momentum adds one parameter-sized buffer.

</details>

<details><summary>On the ill-conditioned quadratic in §4.4, both optimizers used the same $\eta = 0.0198$. Why couldn't we just make plain SGD faster by raising $\eta$?</summary>

The steep direction (curvature 100) sets the stability limit: plain gradient descent diverges for $\eta \ge 2/100 = 0.02$. Raising $\eta$ to speed up the slow direction would blow up the steep one. Momentum sidesteps this by accumulating speed in the slow, consistent direction without needing a larger $\eta$.

</details>

## Next

You now have the core update rule, its cheap stochastic form, and momentum's memory trick. But momentum still uses a single global learning rate for every parameter — and a parameter whose gradients are consistently large needs a smaller effective step than one whose gradients are tiny. The next lesson adds **per-parameter adaptive scaling** (RMSProp), combines it with momentum and a bias correction to get **Adam**, and then fixes Adam's weight-decay bug to arrive at **AdamW**, the optimizer that trains essentially every modern LLM.

Continue to [03.2 · RMSProp, Adam, AdamW, weight decay](lessons/module-03/lesson-02.md).
