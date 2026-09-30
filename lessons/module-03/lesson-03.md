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

The whole curve, for the values used in the worked table below ($W = 100$, $T = 1000$, $\eta_{\max} = 6\times 10^{-4}$, $\eta_{\min} = 6\times 10^{-5}$), with the loop running on to step $1100$ so the floor is visible:

<svg viewBox="0 0 640 300" width="100%" style="max-width:640px;display:block;margin:1em auto;font-family:sans-serif;font-size:12px" role="img" aria-label="Learning rate eta(s) versus step s: linear warmup to 6e-4 at s=100, cosine decay to 6e-5 at s=1000, then flat floor">
<rect x="70.0" y="20" width="50.0" height="230" fill="#f59e0b" fill-opacity="0.12"/>
<rect x="570.0" y="20" width="50.0" height="230" fill="#888" fill-opacity="0.12"/>
<line x1="70" y1="250.0" x2="620" y2="250.0" stroke="#888" stroke-opacity="0.25"/>
<text x="64" y="254.0" text-anchor="end" fill="#888">0</text>
<line x1="70" y1="215.2" x2="620" y2="215.2" stroke="#888" stroke-opacity="0.25"/>
<text x="64" y="219.2" text-anchor="end" fill="#888">1e-4</text>
<line x1="70" y1="180.3" x2="620" y2="180.3" stroke="#888" stroke-opacity="0.25"/>
<text x="64" y="184.3" text-anchor="end" fill="#888">2e-4</text>
<line x1="70" y1="145.5" x2="620" y2="145.5" stroke="#888" stroke-opacity="0.25"/>
<text x="64" y="149.5" text-anchor="end" fill="#888">3e-4</text>
<line x1="70" y1="110.6" x2="620" y2="110.6" stroke="#888" stroke-opacity="0.25"/>
<text x="64" y="114.6" text-anchor="end" fill="#888">4e-4</text>
<line x1="70" y1="75.8" x2="620" y2="75.8" stroke="#888" stroke-opacity="0.25"/>
<text x="64" y="79.8" text-anchor="end" fill="#888">5e-4</text>
<line x1="70" y1="40.9" x2="620" y2="40.9" stroke="#888" stroke-opacity="0.25"/>
<text x="64" y="44.9" text-anchor="end" fill="#888">6e-4</text>
<text x="70.0" y="266" text-anchor="middle" fill="#888">0</text>
<line x1="70.0" y1="250" x2="70.0" y2="254" stroke="#888"/>
<text x="120.0" y="266" text-anchor="middle" fill="#888">100</text>
<line x1="120.0" y1="250" x2="120.0" y2="254" stroke="#888"/>
<text x="270.0" y="266" text-anchor="middle" fill="#888">400</text>
<line x1="270.0" y1="250" x2="270.0" y2="254" stroke="#888"/>
<text x="345.0" y="266" text-anchor="middle" fill="#888">550</text>
<line x1="345.0" y1="250" x2="345.0" y2="254" stroke="#888"/>
<text x="470.0" y="266" text-anchor="middle" fill="#888">800</text>
<line x1="470.0" y1="250" x2="470.0" y2="254" stroke="#888"/>
<text x="570.0" y="266" text-anchor="middle" fill="#888">1000</text>
<line x1="570.0" y1="250" x2="570.0" y2="254" stroke="#888"/>
<line x1="70" y1="250" x2="620" y2="250" stroke="#888"/><line x1="70" y1="20" x2="70" y2="250" stroke="#888"/>
<polyline points="70.0,247.9 72.5,237.5 75.0,227.0 77.5,216.5 80.0,206.1 82.5,195.6 85.0,185.2 87.5,174.7 90.0,164.3 92.5,153.8 95.0,143.4 97.5,132.9 100.0,122.5 102.5,112.0 105.0,101.5 107.5,91.1 110.0,80.6 112.5,70.2 115.0,59.7 117.5,49.3 120.0,40.9 122.5,40.9 125.0,41.0 127.5,41.0 130.0,41.1 132.5,41.3 135.0,41.4 137.5,41.6 140.0,41.8 142.5,42.1 145.0,42.3 147.5,42.6 150.0,43.0 152.5,43.3 155.0,43.7 157.5,44.1 160.0,44.6 162.5,45.0 165.0,45.5 167.5,46.0 170.0,46.6 172.5,47.2 175.0,47.8 177.5,48.4 180.0,49.0 182.5,49.7 185.0,50.4 187.5,51.2 190.0,51.9 192.5,52.7 195.0,53.5 197.5,54.3 200.0,55.2 202.5,56.1 205.0,57.0 207.5,57.9 210.0,58.9 212.5,59.9 215.0,60.9 217.5,61.9 220.0,62.9 222.5,64.0 225.0,65.1 227.5,66.2 230.0,67.3 232.5,68.5 235.0,69.6 237.5,70.8 240.0,72.0 242.5,73.3 245.0,74.5 247.5,75.8 250.0,77.1 252.5,78.4 255.0,79.7 257.5,81.0 260.0,82.4 262.5,83.8 265.0,85.1 267.5,86.5 270.0,88.0 272.5,89.4 275.0,90.8 277.5,92.3 280.0,93.8 282.5,95.2 285.0,96.7 287.5,98.2 290.0,99.8 292.5,101.3 295.0,102.8 297.5,104.4 300.0,105.9 302.5,107.5 305.0,109.1 307.5,110.6 310.0,112.2 312.5,113.8 315.0,115.4 317.5,117.0 320.0,118.7 322.5,120.3 325.0,121.9 327.5,123.5 330.0,125.2 332.5,126.8 335.0,128.4 337.5,130.1 340.0,131.7 342.5,133.4 345.0,135.0 347.5,136.6 350.0,138.3 352.5,139.9 355.0,141.6 357.5,143.2 360.0,144.8 362.5,146.5 365.0,148.1 367.5,149.7 370.0,151.3 372.5,153.0 375.0,154.6 377.5,156.2 380.0,157.8 382.5,159.4 385.0,160.9 387.5,162.5 390.0,164.1 392.5,165.6 395.0,167.2 397.5,168.7 400.0,170.2 402.5,171.8 405.0,173.3 407.5,174.8 410.0,176.2 412.5,177.7 415.0,179.2 417.5,180.6 420.0,182.0 422.5,183.5 425.0,184.9 427.5,186.2 430.0,187.6 432.5,189.0 435.0,190.3 437.5,191.6 440.0,192.9 442.5,194.2 445.0,195.5 447.5,196.7 450.0,198.0 452.5,199.2 455.0,200.4 457.5,201.5 460.0,202.7 462.5,203.8 465.0,204.9 467.5,206.0 470.0,207.1 472.5,208.1 475.0,209.1 477.5,210.1 480.0,211.1 482.5,212.1 485.0,213.0 487.5,213.9 490.0,214.8 492.5,215.7 495.0,216.5 497.5,217.3 500.0,218.1 502.5,218.8 505.0,219.6 507.5,220.3 510.0,221.0 512.5,221.6 515.0,222.2 517.5,222.8 520.0,223.4 522.5,224.0 525.0,224.5 527.5,225.0 530.0,225.4 532.5,225.9 535.0,226.3 537.5,226.7 540.0,227.0 542.5,227.4 545.0,227.7 547.5,227.9 550.0,228.2 552.5,228.4 555.0,228.6 557.5,228.7 560.0,228.9 562.5,229.0 565.0,229.0 567.5,229.1 570.0,229.1 572.5,229.1 575.0,229.1 577.5,229.1 580.0,229.1 582.5,229.1 585.0,229.1 587.5,229.1 590.0,229.1 592.5,229.1 595.0,229.1 597.5,229.1 600.0,229.1 602.5,229.1 605.0,229.1 607.5,229.1 610.0,229.1 612.5,229.1 615.0,229.1 617.5,229.1 620.0,229.1" fill="none" stroke="#2563eb" stroke-width="2.5"/>
<circle cx="70.0" cy="247.9" r="4" fill="#dc2626"/>
<text x="80.0" y="236" text-anchor="start" fill="#dc2626">s=0: 6.0e-6</text>
<circle cx="120.0" cy="40.9" r="4" fill="#dc2626"/>
<text x="128.0" y="30.9" text-anchor="start" fill="#dc2626">s=W=100: peak 6.0e-4</text>
<circle cx="345.0" cy="135.0" r="4" fill="#dc2626"/>
<text x="353.0" y="125.0" text-anchor="start" fill="#dc2626">s=550: 3.3e-4</text>
<circle cx="570.0" cy="229.1" r="4" fill="#dc2626"/>
<text x="570.0" y="176" text-anchor="end" fill="#dc2626">s=T=1000: floor 6.0e-5</text>
<text x="95.0" y="36" text-anchor="middle" fill="#b45309">warmup</text>
<text x="457.5" y="242" text-anchor="middle" fill="#888">cosine decay</text>
<text x="595.0" y="242" text-anchor="middle" fill="#888">floor</text>
<text x="345.0" y="292" text-anchor="middle" fill="#888">step s</text>
<text x="16" y="135.0" text-anchor="middle" fill="#888" transform="rotate(-90 16 135.0)">η(s)</text>
</svg>

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

<div class="callout warn"><p><strong>Always use warmup — this is a bug to avoid, not a setting to experiment with.</strong> Production LLM runs never turn warmup off; what they tune is only its <em>length</em> (and the peak $\eta_{\max}$). A missing or too-short warmup on a large-batch Adam/AdamW run is one of the most common <em>accidental</em> causes of a run blowing up in its first few hundred steps. The symptom is a loss that decreases briefly and then rockets to <code>inf</code> or <code>nan</code>. If you see that, the first two fixes to try are: add or lengthen warmup, and turn on gradient clipping (§4).</p></div>

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
