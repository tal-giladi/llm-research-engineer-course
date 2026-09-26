# 03.3 · Warmup, cosine decay, gradient clipping

<div class="prereq">
<p><strong>Prerequisites:</strong> AdamW, bias correction, and the fact that Adam's first steps have size $\approx\eta$ from an unreliable early $\hat v$, from <a href="#/lessons/module-03/lesson-02">03.2 · RMSProp, Adam, AdamW, weight decay</a>. The learning rate $\eta$ and the update rule from <a href="#/lessons/module-03/lesson-01">03.1</a>. The gradient $\nabla L$ and <code>.grad</code> from <a href="#/lessons/module-02/lesson-01">02.1</a>.</p>
<p><strong>You will learn:</strong> the <strong>learning-rate schedule</strong> used by real LLM runs — linear <em>warmup</em> for the first $W$ steps, then <em>cosine decay</em> down to a floor — with the exact formula and a worked table of the learning rate at steps $0$, $W$, the midpoint, and the end; <em>why</em> warmup is needed (Adam's early variance estimates are unstable and a large early $\eta$ can diverge); and <strong>gradient clipping</strong> by global L2 norm, $g \leftarrow g\cdot\min(1, c/\lVert g\rVert)$, worked on a concrete gradient.</p>
<p><strong>Why this matters for ML:</strong> these three techniques are on essentially every production LLM training run — GPT-3, LLaMA, OLMo, and the pretraining loop you build in Module 7. They are not optional polish; without warmup and clipping, large-batch Adam training routinely diverges in the first few hundred steps. This lesson is the difference between a run that trains and a run that explodes.</p>
</div>

## 1. Intuition: the learning rate should change over training

Lesson 03.1 held the learning rate $\eta$ fixed. Real runs never do. The right step size is different at different phases of training:

- **At the very start**, the parameters are random and Adam's per-parameter scale estimate $\hat v$ is built from just one or two noisy gradients (03.2), so it is unreliable — and Adam's first steps are already size $\approx\eta$ regardless of gradient magnitude. A large $\eta$ here can throw parameters wildly off and diverge. So we want $\eta$ to start *near zero* and ramp up: **warmup**.
- **In the middle**, once things are stable, we want $\eta$ at its peak to make fast progress.
- **Near the end**, we want $\eta$ to shrink toward zero so the optimizer settles into a good minimum instead of bouncing around it: **decay**.

The standard schedule that does all three is **linear warmup followed by cosine decay to a floor**. It is a pure function of the step number, so the training loop just asks "what is $\eta$ at step $s$?" each iteration and writes the answer into the optimizer.

<div class="callout key"><p>The learning rate is <em>scheduled</em>, not fixed: ramp it up linearly from ~0 over the first $W$ steps (warmup), then glide it down a cosine curve to a small floor over the rest of training. This one schedule is the near-universal default for LLM pretraining.</p></div>

## 2. The schedule: linear warmup + cosine decay

### 2.1 The exact formula

Let $s$ be the current step, $W$ the number of warmup steps, $T$ the total number of steps, $\eta_{\max}$ the peak learning rate, and $\eta_{\min}$ the floor. The scheduled learning rate is

$$
\eta(s) =
\begin{cases}
\eta_{\max}\cdot\dfrac{s+1}{W} & s < W \quad\text{(linear warmup)}\\[2ex]
\eta_{\min} + \tfrac{1}{2}\big(1 + \cos(\pi\, p)\big)\,(\eta_{\max} - \eta_{\min}), \quad p = \dfrac{s - W}{T - W} & W \le s \le T \quad\text{(cosine decay)}\\[2ex]
\eta_{\min} & s > T \quad\text{(floor)}
\end{cases}
$$

Reading each piece:

- **Warmup** ($s < W$): a straight line from $\eta_{\max}/W$ at step $0$ up to $\eta_{\max}$. The $s+1$ (rather than $s$) makes the first step's rate nonzero and puts the peak exactly at $s = W$.
- **Cosine decay** ($W \le s \le T$): $p$ is the *progress* through the decay phase, running $0 \to 1$. The factor $\tfrac{1}{2}(1 + \cos(\pi p))$ runs smoothly from $1$ (at $p = 0$, since $\cos 0 = 1$) down to $0$ (at $p = 1$, since $\cos\pi = -1$). So the rate glides from $\eta_{\max}$ down to $\eta_{\min}$ along a half-cosine — fast decay in the middle, gentle near both ends.
- **Floor** ($s > T$): held at $\eta_{\min}$ if the loop runs past the planned horizon.

### 2.2 Worked table

Take $W = 100$, $T = 1000$, $\eta_{\max} = 6\times 10^{-4}$ (a typical GPT peak), $\eta_{\min} = 6\times 10^{-5}$ (one tenth of the peak, a common choice). The midpoint of the decay phase is $s = (W + T)/2 = 550$, where $p = 0.5$ and $\cos(\pi/2) = 0$, so the factor is exactly $\tfrac{1}{2}$.

| step $s$ | phase | $\eta(s)$ | how it is computed |
|---|---|---|---|
| $0$ | warmup start | $6.0\times 10^{-6}$ | $\eta_{\max}\cdot\tfrac{1}{100} = 6\mathrm{e}{-4}/100$ |
| $100$ ($=W$) | end of warmup / peak | $6.0\times 10^{-4}$ | cosine with $p = 0$: $\eta_{\min} + 1\cdot(\eta_{\max}-\eta_{\min}) = \eta_{\max}$ |
| $550$ (midpoint) | mid-decay | $3.30\times 10^{-4}$ | $\eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min}) = 6\mathrm{e}{-5} + 0.5\cdot 5.4\mathrm{e}{-4}$ |
| $1000$ ($=T$) | end / floor | $6.0\times 10^{-5}$ | cosine with $p = 1$: $\eta_{\min} + 0 = \eta_{\min}$ |

The rate climbs linearly to the peak at step $100$, then eases down the cosine to the floor at step $1000$. These four numbers are produced exactly by the code below.

### 2.3 The from-scratch schedule

`code/src/llmre/optim/schedule.py`:

```python
import math

def cosine_warmup_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    # Phase 1: linear warmup, 0 -> max_lr, peak exactly at step == warmup_steps.
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    # Phase 3: past the horizon, clamp to the floor.
    if step >= max_steps:
        return min_lr
    # Phase 2: cosine decay from max_lr (progress 0) to min_lr (progress 1).
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))   # 1 -> 0
    return min_lr + coeff * (max_lr - min_lr)
```

```python
for s in [0, 100, 550, 1000]:
    print(s, cosine_warmup_lr(s, 100, 1000, 6e-4, 6e-5))
# 0    6e-06
# 100  0.0006
# 550  0.00033
# 1000 6e-05
```

The training loop (Module 7) calls this each step and writes the result into the optimizer: `for g in opt.param_groups: g['lr'] = cosine_warmup_lr(step, ...)`. Our from-scratch `AdamW` exposes `self.lr` directly, so it is just `opt.lr = cosine_warmup_lr(step, ...)` before each `opt.step()`. The test `test_cosine_warmup_endpoints_and_shape` checks the peak lands at $s = W$, the floor at $s = T$, and warmup is monotonically increasing.

<div class="callout pt"><p>PyTorch ships this as <code>torch.optim.lr_scheduler.LambdaLR</code> (or <code>CosineAnnealingLR</code> + a warmup wrapper): you pass a function of the step and call <code>scheduler.step()</code> each iteration, and it overwrites the optimizer's <code>lr</code> exactly as our loop does. Writing it as a plain function of <code>step</code> makes the whole schedule inspectable — you can print the entire curve before training starts.</p></div>

## 3. Why warmup?

Warmup exists because of two facts from lesson 03.2, both worst at the start of training:

1. **Adam's early steps are size $\approx\eta$ regardless of gradient.** At $t = 1$ the ratio $\hat m/\sqrt{\hat v} = \operatorname{sign}(g)$ exactly (03.2 §3.4), so every parameter moves by about $\eta$ in some direction. If $\eta$ is at its full peak value, that is a large, coordinated jump from a random initialization.
2. **The second-moment estimate $\hat v$ is unreliable early.** $\hat v$ is an EMA with $\beta_2 = 0.999$, so it takes hundreds of steps to warm up into a trustworthy magnitude estimate. In the first handful of steps it is based on one or two noisy gradients, so the per-parameter scaling — the whole point of Adam — is essentially guessing.

Put together: at the start, Adam takes large steps ($\approx\eta$) using an unreliable per-parameter scale, from a random point where the loss surface is steep and badly conditioned. That is a recipe for divergence — the loss spikes to `inf` and the run is dead. Warmup defuses it by making $\eta$ start near zero and grow linearly, so the early, ill-informed steps are *tiny*. By the time $\eta$ reaches its peak (step $W$), $\hat v$ has warmed up and the parameters have moved off the pathological initial point. This effect is stronger the larger the batch size and the larger the model, which is why every large run uses it.

<div class="callout warn"><p>Skipping warmup on a large-batch Adam/AdamW run is one of the most common ways to make training diverge in the first few hundred steps. The symptom is a loss that decreases briefly then rockets to <code>inf</code> or <code>nan</code>. If you see that, the first two things to try are: add/lengthen warmup, and turn on gradient clipping (§4).</p></div>

<div class="callout paper"><p>Warmup + cosine decay is documented in the <strong>GPT-3</strong> paper (Brown et al. 2020) and used by nearly every open LLM since (LLaMA, OLMo, ...). See the <a href="#/papers/index">paper curriculum</a>: GPT-3 warms the learning rate up over the first ~375M tokens and then cosine-decays it to 10% of the peak — exactly the $\eta_{\min} = \eta_{\max}/10$ shape of the worked table in §2.2.</p></div>

## 4. Gradient clipping by global L2 norm

### 4.1 The problem and the fix

Even with warmup, an occasional bad minibatch — a batch with a few pathological examples, or a rare token combination — can produce a gradient far larger than usual. One such gradient, fed into the update, can undo thousands of good steps in an instant (a loss "spike"). **Gradient clipping** caps the size of the whole gradient before the optimizer uses it, so no single step can be catastrophically large.

The standard form clips by the **global L2 norm**: treat *all* the model's gradients as one long vector, measure its length, and if that length exceeds a threshold $c$, scale the entire vector down to length $c$:

$$
g \;\leftarrow\; g \cdot \min\!\left(1,\ \frac{c}{\lVert g\rVert}\right),
$$

where $\lVert g\rVert = \sqrt{\sum_i g_i^2}$ is the L2 norm over *every* gradient component in the model, and $c$ (typically $1.0$) is the clip threshold. If $\lVert g\rVert \le c$, the factor is $1$ and nothing changes. If $\lVert g\rVert > c$, the factor is $c/\lVert g\rVert < 1$ and every component is scaled by the same amount — so the gradient's **direction is preserved**, only its magnitude is capped. That last point is why we use the *global* norm and one shared scale, not per-tensor clipping: scaling each tensor separately would bend the overall gradient direction, changing which way "downhill" points.

### 4.2 Worked example

Suppose the entire model's gradient (flattened) is $g = [3, 4]$ and the clip threshold is $c = 1.0$. The norm is $\lVert g\rVert = \sqrt{3^2 + 4^2} = \sqrt{25} = 5$. Since $5 > 1$, we scale by $c/\lVert g\rVert = 1/5 = 0.2$:

$$
g \leftarrow [3, 4]\cdot 0.2 = [0.6, 0.8], \qquad \lVert[0.6, 0.8]\rVert = \sqrt{0.36 + 0.64} = \sqrt{1} = 1.
$$

The clipped gradient has norm exactly $1$ (the cap) and points the same way as $[3, 4]$ (still the $3\!:\!4$ ratio). A gradient already under the threshold — say $[0.1, 0.2]$ with norm $\approx 0.22 < 1$ — would be left untouched.

### 4.3 The from-scratch implementation

`code/src/llmre/optim/clip.py`:

```python
def clip_grad_norm_(params, max_norm):
    grads = [p.grad for p in params if p.grad is not None]
    if len(grads) == 0:
        return 0.0
    # Global L2 norm: L2-norm of the vector of per-tensor L2 norms
    #   sqrt(sum_i ||g_i||^2) == sqrt(sum of every g^2 in the model).
    total_norm = torch.norm(
        torch.stack([torch.norm(g.detach(), 2) for g in grads]), 2
    )
    clip_coef = max_norm / (total_norm + 1e-6)   # tiny eps guards /0
    if clip_coef < 1.0:                          # only ever shrink
        for g in grads:
            g.mul_(clip_coef)
    return float(total_norm)                     # pre-clip norm (for logging)
```

Two design points. The function returns the **pre-clip** norm — the training loop logs it to watch for spikes (a sudden jump in gradient norm is an early warning of instability). And the global norm is computed as the L2 norm of the *vector of per-tensor norms*, which equals the L2 norm over all components at once: $\sqrt{\sum_j \lVert g_j\rVert^2} = \sqrt{\sum_j \sum_i g_{ji}^2}$. This runs from scratch; PyTorch provides the same operation as `torch.nn.utils.clip_grad_norm_`, and the test `test_clip_grad_norm_matches_torch` checks our version returns the same norm and produces the same rescaled gradients (to floating-point precision, including PyTorch's identical $10^{-6}$ epsilon).

<div class="callout key"><p>Clip by <em>global</em> L2 norm: one scale factor $\min(1, c/\lVert g\rVert)$ applied to every gradient, so the whole update is capped in size but unchanged in direction. Clip threshold $c = 1.0$ is the near-universal default. It runs after <code>backward()</code> and before <code>optimizer.step()</code>.</p></div>

## 5. Where these sit in the training loop

All three techniques slot into the loop between `backward()` and `step()`, in this order:

```python
for step in range(max_steps):
    opt.zero_grad()
    loss = compute_loss(model, get_batch())     # forward
    loss.backward()                             # fill .grad
    clip_grad_norm_(model.parameters(), 1.0)    # (1) cap the gradient
    opt.lr = cosine_warmup_lr(step, W, T, max_lr, min_lr)  # (2) set this step's lr
    opt.step()                                  # (3) AdamW update with that lr
```

That is the skeleton of every pretraining loop in this course. Module 7 fills in batching, gradient accumulation, checkpointing, and throughput accounting around exactly these lines.

## 6. Tensor shapes, dtype, cost

- **Schedule:** pure Python `float` arithmetic on the integer `step` — no tensors, no device, negligible cost. Called once per step.
- **Clipping:** computes one scalar norm by reducing over every gradient component ($O(P)$ work for $P$ parameters, a single streaming pass), then, only if clipping triggers, one elementwise scale of each gradient tensor (another $O(P)$ pass). The per-tensor norms are stacked into a tiny vector on the gradients' device; nothing changes shape. Cost is trivial next to the backward pass that produced the gradients, and memory overhead is essentially zero (one scalar).

## Exercise

You are training with $\eta_{\max} = 3\times 10^{-4}$ and want the learning rate at step $0$ to be about $1\times 10^{-6}$ (a gentle start). Using the warmup formula $\eta(0) = \eta_{\max}\cdot\frac{1}{W}$, roughly what warmup length $W$ do you need?

<details><summary>Hint</summary>

Set $\eta_{\max}/W = 10^{-6}$ and solve for $W$.

</details>

<details><summary>Solution</summary>

$W = \eta_{\max}/\eta(0) = (3\times 10^{-4})/(10^{-6}) = 300$ steps. So a warmup of a few hundred steps gives a step-0 rate around $10^{-6}$. Real runs often warm up over hundreds to a couple thousand steps for exactly this reason — long enough that $\hat v$ has stabilized and the early steps are tiny.

</details>

## Debugging exercise

A run uses warmup + cosine decay and AdamW, but every few thousand steps the loss suddenly spikes upward by a lot and then slowly recovers. Warmup is present and correct. What single technique from this lesson most directly addresses these intermittent spikes, and why not just lower the learning rate instead?

<details><summary>Solution</summary>

**Gradient clipping** (§4). The spikes are caused by occasional outlier minibatches producing an unusually large gradient; clipping the global norm to $c = 1.0$ caps exactly those rare huge updates while leaving all the normal steps untouched. Lowering the learning rate instead would slow down *every* step — the 99.9% that were fine — to defend against the 0.1% that were not, wasting compute and hurting convergence. Clipping is targeted: it only acts when $\lVert g\rVert > c$. Logging the pre-clip norm (which `clip_grad_norm_` returns) would confirm the diagnosis — you would see the norm spike on the bad steps.

</details>

## Check yourself

<details><summary>At the end of warmup (step $s = W$), what is the learning rate, and does the warmup branch or the cosine branch produce it?</summary>

It is exactly $\eta_{\max}$. At $s = W$ the condition $s < W$ is false, so the cosine branch runs with progress $p = (W - W)/(T - W) = 0$; the factor $\tfrac{1}{2}(1+\cos 0) = 1$, giving $\eta_{\min} + 1\cdot(\eta_{\max}-\eta_{\min}) = \eta_{\max}$. The two branches meet continuously at the peak.

</details>

<details><summary>Why does warmup specifically help Adam, as opposed to plain SGD?</summary>

Because of Adam's second-moment estimate. Early in training $\hat v$ is built from one or two noisy gradients, so the per-parameter scaling is unreliable, and Adam's first steps are size $\approx\eta$ regardless of gradient magnitude. Warmup keeps $\eta$ tiny until $\hat v$ has warmed up. Plain SGD has no such adaptive estimate to destabilize, so it is far less sensitive to a large initial rate (though warmup can still help it).

</details>

<details><summary>A gradient has global norm $10$ and the clip threshold is $c = 2$. By what factor is each component scaled, and what is the resulting norm?</summary>

Factor $= c/\lVert g\rVert = 2/10 = 0.2$; each component is multiplied by $0.2$, so the resulting norm is $10\cdot 0.2 = 2 = c$. The direction is unchanged.

</details>

<details><summary>Why clip by the <em>global</em> norm across all parameters rather than clipping each tensor's gradient separately?</summary>

A single global scale factor shrinks every gradient component by the same amount, preserving the overall gradient *direction* (which way is downhill). Clipping each tensor independently would rescale different tensors by different factors, bending the combined gradient into a different direction than the true one — corrupting the step, not just capping it.

</details>

<details><summary>In the training loop, does gradient clipping go before or after <code>optimizer.step()</code>? Before or after <code>loss.backward()</code>?</summary>

After `loss.backward()` (it needs the gradients to exist in `.grad`) and before `optimizer.step()` (it must cap the gradients before they are used to update parameters). The order is `backward()` → `clip_grad_norm_()` → set lr → `step()`.

</details>

## Next

You now have the full optimization toolkit: the update rule and momentum (03.1), adaptive per-parameter scaling and decoupled weight decay in AdamW (03.2), and the warmup + cosine schedule with gradient clipping that keep a real run stable (03.3). Every training loop for the rest of the course uses exactly these pieces. The next module leaves optimization behind and turns to how text becomes numbers the model can consume — **tokenization**: characters, bytes, Unicode, and Byte-Pair Encoding from scratch.

Continue to [04.1 · Characters, bytes, Unicode, the vocab problem](lessons/module-04/lesson-01.md).
