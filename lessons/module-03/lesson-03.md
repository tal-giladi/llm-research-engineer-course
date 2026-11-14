# 03.3 · Warmup, cosine decay, gradient clipping, batch ramp

<div class="prereq">
<p><strong>Prerequisites:</strong> AdamW, bias correction, and the fact that Adam's first steps have size $\approx\eta$ from an unreliable early $\hat v$, from <a href="#/lessons/module-03/lesson-02">03.2 · RMSProp, Adam, AdamW, weight decay</a>. The learning rate $\eta$ and the update rule from <a href="#/lessons/module-03/lesson-01">03.1</a>. The gradient $\nabla L$ and <code>.grad</code> from <a href="#/lessons/module-02/lesson-01">02.1</a>.</p>
<p><strong>You will learn:</strong> the <strong>learning-rate schedule</strong> used by real LLM runs — linear <em>warmup</em> for the first $W$ steps, then <em>cosine decay</em> down to a floor — with the exact formula and a worked table of the learning rate at steps $0$, $W$, the midpoint, and the end; <em>why</em> warmup is needed (Adam's early variance estimates are unstable and a large early $\eta$ can diverge); and <strong>gradient clipping</strong> by global L2 norm, $g \leftarrow g\cdot\min(1, c/\lVert g\rVert)$, worked on a concrete gradient; and <strong>batch-size warmup</strong> (batch ramp): why the useful batch size grows during training (the gradient noise scale $B_{\text{simple}} = \operatorname{tr}(\Sigma)/\lVert G\rVert^2$), how to measure it, and how to implement a ramp with gradient accumulation and a token-based learning-rate schedule.</p>
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

## 5. Batch-size warmup (batch ramp)

Warmup changes the *learning rate* over training. Many large runs also change the *batch size*: they start with a small global batch and grow it during the first part of training. This is called **batch-size warmup**, **batch ramp-up**, or just a **batch ramp**. It is less universal than learning-rate warmup (plenty of runs keep the batch fixed), but most of the largest publicly documented pretraining runs use some form of it, so you need to know what it is, why it works, and how to build it.

### 5.1 Intuition: early gradients agree, late gradients argue

Recall from 03.1 that a minibatch gradient is an *average* of per-example gradients. Averaging $B$ examples does one thing: it cancels out the part of the per-example gradients that disagree with each other (the noise). How much averaging you need depends on how much the examples disagree.

- **At the start of training**, the model is terrible at everything in the same way. It has not even learned that "the" is common. Almost every example says the same thing: "raise the probability of common tokens, lower the rare ones." The per-example gradients all point roughly the same way. Averaging 8 of them gives nearly the same direction as averaging 8,000. A big batch spends 1,000× more tokens to buy almost no better a step.
- **Later in training**, the easy patterns are learned. What is left are subtle, example-specific things: this document wants a rare word, that one wants a different one. Per-example gradients now point in many directions and mostly cancel. The true downhill direction is a small signal buried in large noise. Now you *need* a big batch to average the noise away, or each step is mostly a random walk.

So the batch size worth paying for grows as the model learns. A batch ramp follows that: small batch (more, cheaper steps) while the gradients agree, large batch (fewer, cleaner steps) once they disagree.

<div class="callout key"><p>The useful batch size is not a constant. It grows as the loss falls, because per-example gradients disagree more and more. A batch ramp starts small and grows the global batch so tokens are not wasted on oversized batches early in training.</p></div>

### 5.2 The mathematics: the gradient noise scale

McCandlish et al. (2018) made this precise. Name the pieces:

- $G = \nabla L(\theta)$ — the **true gradient**: the average gradient over the whole data distribution. A vector with one entry per parameter.
- $g_i$ — the gradient from a single example $i$. Its average over all examples is $G$.
- $\Sigma$ — the **covariance** of the per-example gradients: how much the $g_i$ scatter around $G$. $\operatorname{tr}(\Sigma)$ (the trace, the sum of the diagonal) is the total scatter: the sum over all parameters of the variance of that parameter's gradient component.
- $G_B = \frac{1}{B}\sum_{i=1}^{B} g_i$ — the minibatch gradient you actually compute, averaging $B$ random examples.

Averaging $B$ independent samples divides the variance by $B$, so the minibatch gradient's expected squared length is the true signal plus noise shrunk by $B$:

$$
\mathbb{E}\big[\lVert G_B\rVert^2\big] = \lVert G\rVert^2 + \frac{\operatorname{tr}(\Sigma)}{B}.
$$

The batch size at which the noise term equals the signal term is the **(simple) gradient noise scale**:

$$
B_{\text{simple}} = \frac{\operatorname{tr}(\Sigma)}{\lVert G\rVert^2}.
$$

The paper's main quantity $B_{\text{noise}}$ also weights both terms by the curvature (Hessian); $B_{\text{simple}}$ drops that weighting and is what people measure in practice. With the learning rate tuned for each batch size, the paper shows the loss improvement from one step is

$$
\Delta L_{\text{opt}}(B) = \frac{\Delta L_{\max}}{1 + B_{\text{noise}}/B},
$$

where $\Delta L_{\max}$ is the improvement an infinitely large batch (the exact gradient $G$) would give. Read the fraction $\frac{1}{1 + B_{\text{noise}}/B}$ as "how much of a perfect step you get":

- $B \ll B_{\text{noise}}$: the fraction is about $B/B_{\text{noise}}$. Doubling the batch nearly doubles progress per step. Larger batches are efficient.
- $B \gg B_{\text{noise}}$: the fraction is about $1$. Doubling the batch doubles the tokens per step and buys almost nothing. Larger batches are wasted compute.

$B_{\text{noise}}$ is close to the **critical batch size**: the best trade between number of steps (wall-clock time) and number of examples (compute). And because $\lVert G\rVert^2$ shrinks as the model approaches a minimum while $\operatorname{tr}(\Sigma)$ shrinks much more slowly, $B_{\text{simple}}$ **grows during training**. That growth is the whole reason for a batch ramp.

### 5.3 Worked examples by hand

**Noise scale from four per-example gradients.** Take a 2-parameter model and four examples.

*Early in training* the per-example gradients are $[1.1, 0.9],\ [0.9, 1.1],\ [1.0, 1.2],\ [1.0, 0.8]$. Their mean is $G = [1.0, 1.0]$, so $\lVert G\rVert^2 = 1 + 1 = 2$. Variance of the first component: deviations $0.1, -0.1, 0, 0$, squares sum to $0.02$, divided by $4$ is $0.005$. Second component: deviations $-0.1, 0.1, 0.2, -0.2$, squares sum to $0.10$, divided by $4$ is $0.025$. So $\operatorname{tr}(\Sigma) = 0.005 + 0.025 = 0.03$ and

$$
B_{\text{simple}} = 0.03 / 2 = 0.015.
$$

Far below $1$: a single example already gives an almost perfect direction.

*Late in training* the per-example gradients are $[1, -1],\ [-1, 1],\ [1.2, -0.8],\ [-0.8, 1.2]$. They point in opposite directions. The mean is $G = [0.1, 0.1]$, $\lVert G\rVert^2 = 0.02$. First component: deviations $0.9, -1.1, 1.1, -0.9$, squares $0.81 + 1.21 + 1.21 + 0.81 = 4.04$, divided by $4$ is $1.01$. The second component is the same, $1.01$. So $\operatorname{tr}(\Sigma) = 2.02$ and

$$
B_{\text{simple}} = 2.02 / 0.02 = 101.
$$

Same model size, same data, but the batch worth paying for went from about $1$ to about $100$.

**What a batch buys.** Suppose $B_{\text{noise}} = 10$ early and $500$ late. Compare $B = 32$ and $B = 512$ (16× the tokens per step):

| phase | $B_{\text{noise}}$ | fraction at $B=32$ | fraction at $B=512$ | 16× tokens buys |
|---|---|---|---|---|
| early | $10$ | $\frac{1}{1 + 10/32} = 0.76$ | $\frac{1}{1 + 10/512} = 0.98$ | $1.29\times$ progress |
| late | $500$ | $\frac{1}{1 + 500/32} = 0.06$ | $\frac{1}{1 + 500/512} = 0.51$ | $8.4\times$ progress |

Early, the big batch is a terrible deal: 16× the compute for 1.29× the progress per step. Late, it is a good deal. Hence: small early, large late.

**Measuring the noise scale without per-example gradients.** You normally cannot afford per-example gradients for a large model. But you can measure $\lVert G_B\rVert^2$ at two batch sizes, $B_{\text{small}}$ and $B_{\text{big}}$, and solve the two equations from §5.2 for the two unknowns (McCandlish et al., Appendix A.1):

$$
\lVert G\rVert^2 \approx \frac{B_{\text{big}}\lVert G_{B_{\text{big}}}\rVert^2 - B_{\text{small}}\lVert G_{B_{\text{small}}}\rVert^2}{B_{\text{big}} - B_{\text{small}}}, \qquad
\operatorname{tr}(\Sigma) \approx \frac{\lVert G_{B_{\text{small}}}\rVert^2 - \lVert G_{B_{\text{big}}}\rVert^2}{1/B_{\text{small}} - 1/B_{\text{big}}}.
$$

Example: with $B_{\text{small}} = 4$ you measure $\lVert G_4\rVert^2 = 2.0$; with $B_{\text{big}} = 64$, $\lVert G_{64}\rVert^2 = 1.0625$. Then

$$
\lVert G\rVert^2 = \frac{64\cdot 1.0625 - 4\cdot 2.0}{64 - 4} = \frac{68 - 8}{60} = 1.0, \qquad
\operatorname{tr}(\Sigma) = \frac{2.0 - 1.0625}{0.25 - 0.015625} = \frac{0.9375}{0.234375} = 4.0,
$$

so $B_{\text{simple}} = 4.0/1.0 = 4$. (Check: plug back in. $1 + 4/4 = 2$ and $1 + 4/64 = 1.0625$.) In a data-parallel run this is almost free: each GPU's local gradient is a small-batch gradient and the all-reduced gradient is the big-batch one, so you get both numbers every step.

### 5.4 The schedule

Real runs use one of two shapes. Both are **publicly documented**:

- **Linear ramp in tokens.** GPT-3 (Brown et al. 2020, Appendix B) increases the batch linearly from 32k tokens to the full batch (3.2M tokens for the 175B model) over the first 4–12 billion tokens, depending on model size.
- **Step doubling.** Gopher (Rae et al. 2021) increases its batch from 3M to 6M tokens partway through training. Llama 3 405B (2024) starts at 4M tokens with 4,096-token sequences, doubles to 8M after 252M tokens (switching to 8,192-token sequences), and doubles again to 16M after 2.87T tokens. DeepSeek-V3 (2024) grows its batch from 3,072 to 15,360 sequences over the first 469B tokens. Merrill et al. (2025, AI2) measure the critical batch size during OLMo training and double the batch each time it has grown past the current one.

Not every run does this: LLaMA 1 and 2, for example, used a fixed 4M-token batch. Whether a run ramps is a design choice, not a requirement like learning-rate warmup.

The linear version, written like the learning-rate formula. Let $n$ be the number of tokens seen so far, $B_0$ the starting batch, $B_{\text{final}}$ the full batch (both in sequences), $R$ the ramp length in tokens, and $k$ the **granularity**:

$$
B(n) = \begin{cases}
\operatorname{clamp}\!\Big(k\Big\lfloor \tfrac{1}{k}\big(B_0 + (B_{\text{final}} - B_0)\,\tfrac{n}{R}\big)\Big\rfloor,\ B_0,\ B_{\text{final}}\Big) & n < R\\[1ex]
B_{\text{final}} & n \ge R
\end{cases}
$$

Why round down to a multiple of $k$? Because a batch is not one tensor. It is built from **micro-batches** of $b$ sequences — the most that fits in one GPU's memory — on each of $W$ GPUs, summed over $G$ accumulation steps (Module 7 builds this): $B = b \cdot G \cdot W$. The micro-batch $b$ is fixed by memory, so the only knob that changes during a ramp is $G$, and $B$ must be a multiple of $k = b\cdot W$.

**Worked table.** Sequences of $1{,}024$ tokens, micro-batch $b = 8$ on one GPU (so $k = 8$), ramp from $B_0 = 8$ to $B_{\text{final}} = 64$ sequences over $R = 1{,}000{,}000$ tokens:

| tokens seen $n$ | raw $B_0 + 56\cdot n/R$ | rounded down to $\times 8$ | accumulation $G = B/8$ | tokens per step |
|---|---|---|---|---|
| $0$ | $8$ | $8$ | $1$ | $8{,}192$ |
| $250{,}000$ | $22$ | $16$ | $2$ | $16{,}384$ |
| $500{,}000$ | $36$ | $32$ | $4$ | $32{,}768$ |
| $750{,}000$ | $50$ | $48$ | $6$ | $49{,}152$ |
| $\ge 1{,}000{,}000$ | — | $64$ | $8$ | $65{,}536$ |

Train on $4{,}194{,}304$ tokens ($= 64$ full batches). Without a ramp that is $64$ steps of $64$ sequences. With the ramp it is $95$ steps: $47$ steps during the ramp (18 at batch 8, 9 at 16, 6 at 24, 4 at 32, 4 at 40, 3 at 48, 3 at 56) consume about $1.06$M tokens, which the fixed batch would have spent in only about $16$ steps. Same tokens, same compute, about $31$ more optimizer updates, all of them in the phase where small batches lose almost nothing (§5.3). That is the trade: **more steps for the same compute, early, where they are cheap in quality.**

### 5.5 Tensor shapes, dtype, device

The schedule itself is plain Python `int` arithmetic, like the learning-rate schedule. The tensors it controls:

- Each micro-batch `x, y`: shape `(b, T)` = `(8, 1024)`, dtype `int64` token ids, on the training device. **This never changes during the ramp**, so peak activation memory never changes either.
- What changes is how many micro-batches feed one optimizer step, $G$: $1$ at the start, $8$ at the end in the table above. Gradients are accumulated in each parameter's `.grad` (same shape and dtype as the parameter, e.g. `float32`) across the $G$ backward passes.
- For the noise-scale estimate, the per-example gradients in §5.3 form a `(B, P)` matrix (examples × parameters). That is exactly what you never materialize for a real model; the two-batch-size estimator only needs two scalars, $\lVert G_{B_{\text{small}}}\rVert^2$ and $\lVert G_{B_{\text{big}}}\rVert^2$.

### 5.6 The from-scratch implementation

`code/src/llmre/optim/batch_schedule.py`:

```python
def batch_size_at(tokens_seen, start_batch, final_batch, ramp_tokens, granularity=1):
    if start_batch % granularity or final_batch % granularity:
        raise ValueError("start_batch and final_batch must be multiples of granularity")
    if tokens_seen >= ramp_tokens:
        return final_batch
    raw = start_batch + (final_batch - start_batch) * tokens_seen / ramp_tokens
    rounded = int(raw // granularity) * granularity   # whole micro-batches only
    return max(start_batch, min(final_batch, rounded))


def grad_accum_for(batch_size, micro_batch_size, world_size=1):
    per_step = micro_batch_size * world_size          # sequences per accumulation step
    if batch_size % per_step:
        raise ValueError(f"batch_size {batch_size} is not a multiple of {per_step}")
    return batch_size // per_step                     # G


def gradient_noise_scale(g_small_sq, g_big_sq, b_small, b_big):
    g_sq = (b_big * g_big_sq - b_small * g_small_sq) / (b_big - b_small)       # |G|^2
    trace_sigma = (g_small_sq - g_big_sq) / (1.0 / b_small - 1.0 / b_big)     # tr(Sigma)
    return g_sq, trace_sigma, trace_sigma / g_sq                              # B_simple
```

```python
for n in [0, 250_000, 500_000, 750_000, 1_000_000]:
    b = batch_size_at(n, 8, 64, 1_000_000, granularity=8)
    print(n, b, grad_accum_for(b, micro_batch_size=8))
# 0         8  1
# 250000   16  2
# 500000   32  4
# 750000   48  6
# 1000000  64  8

print(gradient_noise_scale(2.0, 1.0625, 4, 64))   # (1.0, 4.0, 4.0)
```

The same file has `plan_batch_ramp(...)`, which simulates the whole run and returns one row per step (step, tokens seen, batch, $G$, tokens). Print it before you launch: it is how the $95$-step count above was produced. The tests `test_batch_ramp_schedule_and_accumulation` and `test_gradient_noise_scale_recovers_known_value` check the worked table, the step count, and that the estimator recovers a known noise scale from synthetic per-example gradients.

Inside the training loop, the ramp only changes how many micro-batches you accumulate. Two things must switch from *steps* to *tokens*:

```python
tokens_seen = 0
while tokens_seen < total_tokens:
    B = batch_size_at(tokens_seen, B0, B_final, ramp_tokens, granularity=b * world)
    G = grad_accum_for(B, micro_batch_size=b, world_size=world)
    opt.zero_grad()
    for _ in range(G):
        x, y = get_batch()                        # (b, T) int64 — fixed shape
        loss = compute_loss(model, x, y)
        (loss / G).backward()                     # divide by THIS step's G
    clip_grad_norm_(model.parameters(), 1.0)
    # The lr schedule is now a function of tokens, not steps (W_tok, T_tok in tokens).
    opt.lr = cosine_warmup_lr(tokens_seen, W_tok, T_tok, max_lr, min_lr)
    opt.step()
    tokens_seen += B * seq_len
```

- **Divide by the current $G$.** Each micro-batch loss is a mean over $b$ sequences. Summing $G$ of them and dividing by $G$ gives the mean over all $B$ sequences, so the gradient has the same scale at every batch size.
- **Schedule the learning rate in tokens.** With a fixed batch, "step 100" and "6.5M tokens" mean the same thing. With a ramp they do not: the first 100 steps of the table's run cover about $4.5$M tokens instead of $6.6$M. GPT-3 states its learning-rate warmup (375M tokens) and decay in tokens for this reason. `cosine_warmup_lr` does not care about units, so pass it tokens and token-valued $W$ and $T$.
- **Checkpoint `tokens_seen`.** It is now part of the training state. Resume without it and the run restarts its ramp at the small batch.

<div class="callout pt"><p>Neither <code>torch.optim</code> nor <code>torch.utils.data.DataLoader</code> has a batch ramp: a DataLoader's batch size is fixed when you build it. Production training frameworks implement it the way above, by varying the number of accumulated micro-batches per step; Megatron-LM, for example, has a <code>--rampup-batch-size</code> option. In a hand-written loop it is the few lines shown.</p></div>

### 5.7 Should the learning rate change when the batch grows?

A larger batch gives a less noisy gradient, which can tolerate a larger step. Two rules of thumb exist: scale $\eta$ linearly with $B$ (common for SGD) or with $\sqrt{B}$ (common for Adam). What is documented:

- **Merrill et al. (2025)** raise the base learning rate by $\sqrt{2}$ each time they double the batch (square-root rule), and report slightly better loss than the original OLMo run with 43% fewer gradient steps.
- **GPT-3** describes its batch ramp and its learning-rate schedule separately and does not say the learning rate was rescaled during the ramp.
- **Smith et al. (2017)** show the link from the other side: for SGD the gradient noise depends on the ratio $\eta/B$, so *increasing the batch size* during training can replace *decaying the learning rate*, reaching the same accuracy in fewer steps.

So: **publicly documented** that both choices have been used; **reasonable practice** to rescale $\eta$ with $\sqrt{B}$ under Adam when the batch jumps by a large factor, and to watch the loss and gradient norm right after each jump.

### 5.8 Cost

- **Memory:** unchanged. The micro-batch shape is fixed; only the number of accumulation steps changes.
- **Throughput (tokens per second per GPU):** unchanged in the simple loop, because each GPU always processes full micro-batches. Time per optimizer step grows with $G$.
- **The real constraint:** the starting batch must be at least one micro-batch per GPU, $B_0 \ge b\cdot W$. On a large cluster with a small $B_0$ you cannot keep every GPU busy, so how small a batch you can start with is limited by the cluster's size (this is **inference**, from $B = b\cdot G\cdot W$; the papers do not describe how they handled it).
- **Benefit:** more optimizer updates for the same token budget during the phase where small batches are nearly as good per step (§5.3), or equivalently the same loss in less compute. Merrill et al. report slightly better loss than their original fixed-batch OLMo run with 43% fewer gradient steps.

<div class="callout paper"><p><strong>Read:</strong> McCandlish et al. 2018, <a href="https://arxiv.org/abs/1812.06162">An Empirical Model of Large-Batch Training</a>. Read §2 (the noise scale, eqs. 2.7–2.11) and Appendix A.1 (the two-batch-size estimator); skim the per-task experiments. Then Merrill et al. 2025, <a href="https://arxiv.org/abs/2505.23971">Critical Batch Size Revisited</a>, for a modern LLM measurement and a batch-warmup recipe. Optional: Smith et al. 2017, <a href="https://arxiv.org/abs/1711.00489">Don't Decay the Learning Rate, Increase the Batch Size</a>. Sources for §5.4: <a href="https://arxiv.org/abs/2005.14165">GPT-3</a> Appendix B, <a href="https://arxiv.org/abs/2112.11446">Gopher</a> §3.2, <a href="https://arxiv.org/abs/2407.21783">Llama 3</a> §3.4.1, <a href="https://arxiv.org/abs/2412.19437">DeepSeek-V3</a> §4.2. Checked 2026-11-14.</p></div>

### 5.9 Code this after reading

Run `code/experiments/noise_scale_toy.py` (CPU, a few seconds). It trains a linear regression with SGD and measures $B_{\text{simple}}$ with the two-batch-size estimator at several points:

```text
step    0  loss  14.5636  |G|^2   29.2846  tr(Sigma)    431.42  B_simple      14.7
step   25  loss   1.3251  |G|^2    2.3543  tr(Sigma)     46.73  B_simple      19.9
step   50  loss   0.2241  |G|^2    0.2000  tr(Sigma)      7.32  B_simple      36.6
step  100  loss   0.1254  |G|^2    0.0023  tr(Sigma)      4.12  B_simple    1768.9
step  300  loss   0.1248  |G|^2    0.0022  tr(Sigma)      3.85  B_simple    1778.2
```

Both $\lVert G\rVert^2$ and $\operatorname{tr}(\Sigma)$ fall, but the signal falls far faster, so the noise scale grows about 100× as the loss approaches its floor. Then change it: add a batch ramp to the SGD loop (start at 8, grow to 256) and compare the loss after the same number of *examples* against fixed batches of 8 and 256.

### 5.10 Common mistakes

- **Learning-rate schedule still in steps.** Warmup and cosine decay end at a different point in training than planned. Use tokens.
- **Dividing the loss by a fixed $G$.** Divide by the current step's $G$, or the gradient's scale jumps every time the batch grows.
- **Batch not a multiple of $b\cdot W$.** The last accumulation step is a partial micro-batch or the batch silently differs across GPUs. Round down to the granularity.
- **Not saving `tokens_seen`.** A resumed run drops back to the starting batch.
- **Ramping the micro-batch instead of $G$.** Growing $b$ changes peak memory mid-run and can OOM hours into training. Keep $b$ fixed.

## 6. Where these sit in the training loop

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

That is the skeleton of every pretraining loop in this course. Module 7 fills in batching, gradient accumulation, checkpointing, and throughput accounting around exactly these lines. A run with a batch ramp uses the token-based version in §5.6.

## 7. Tensor shapes, dtype, cost

- **Schedule:** pure Python `float` arithmetic on the integer `step` — no tensors, no device, negligible cost. Called once per step.
- **Clipping:** computes one scalar norm by reducing over every gradient component ($O(P)$ work for $P$ parameters, a single streaming pass), then, only if clipping triggers, one elementwise scale of each gradient tensor (another $O(P)$ pass). The per-tensor norms are stacked into a tiny vector on the gradients' device; nothing changes shape. Cost is trivial next to the backward pass that produced the gradients, and memory overhead is essentially zero (one scalar).
- **Batch ramp:** Python `int` arithmetic once per step; micro-batch tensors keep their `(b, T)` shape, only the number of them per step changes (§5.5, §5.8).

## Exercise

You are training with $\eta_{\max} = 3\times 10^{-4}$ and want the learning rate at step $0$ to be about $1\times 10^{-6}$ (a gentle start). Using the warmup formula $\eta(0) = \eta_{\max}\cdot\frac{1}{W}$, roughly what warmup length $W$ do you need?

<details><summary>Hint</summary>

Set $\eta_{\max}/W = 10^{-6}$ and solve for $W$.

</details>

<details><summary>Solution</summary>

$W = \eta_{\max}/\eta(0) = (3\times 10^{-4})/(10^{-6}) = 300$ steps. So a warmup of a few hundred steps gives a step-0 rate around $10^{-6}$. Real runs often warm up over hundreds to a couple thousand steps for exactly this reason — long enough that $\hat v$ has stabilized and the early steps are tiny.

</details>

### Exercise: batch ramp

A run uses 4 GPUs with micro-batch $b = 4$ sequences each, and ramps the global batch linearly from $16$ to $128$ sequences over $2{,}000{,}000$ tokens. After $1{,}300{,}000$ tokens, what batch size does `batch_size_at` return, and how many accumulation steps $G$ does each GPU run?

<details><summary>Hint</summary>

The granularity is $b\cdot W$. Compute the raw linear value first, then round it down.

</details>

<details><summary>Stronger hint</summary>

Granularity $k = 4\cdot 4 = 16$. Raw value $= 16 + (128 - 16)\cdot 1.3/2.0$. Round down to a multiple of $16$, then $G = B/(b\cdot W)$.

</details>

<details><summary>Solution</summary>

Raw $= 16 + 112\cdot 0.65 = 88.8$. Rounded down to a multiple of $16$: $80$. Then $G = 80/16 = 5$: each of the 4 GPUs runs 5 micro-batches of 4 sequences ($4\cdot 5\cdot 4 = 80$). Check in code: `batch_size_at(1_300_000, 16, 128, 2_000_000, 16)` → `80`, `grad_accum_for(80, 4, 4)` → `5`.

</details>

## Debugging exercise

A run uses warmup + cosine decay and AdamW, but every few thousand steps the loss suddenly spikes upward by a lot and then slowly recovers. Warmup is present and correct. What single technique from this lesson most directly addresses these intermittent spikes, and why not just lower the learning rate instead?

<details><summary>Solution</summary>

**Gradient clipping** (§4). The spikes are caused by occasional outlier minibatches producing an unusually large gradient; clipping the global norm to $c = 1.0$ caps exactly those rare huge updates while leaving all the normal steps untouched. Lowering the learning rate instead would slow down *every* step — the 99.9% that were fine — to defend against the 0.1% that were not, wasting compute and hurting convergence. Clipping is targeted: it only acts when $\lVert g\rVert > c$. Logging the pre-clip norm (which `clip_grad_norm_` returns) would confirm the diagnosis — you would see the norm spike on the bad steps.

</details>

### Debugging exercise: batch ramp

You add a batch ramp ($8 \to 64$ sequences, micro-batch $8$) to a working loop. The loss curve looks fine, but the logged pre-clip gradient norm climbs in a staircase: it jumps up each time the batch size changes and stays there, ending about $8\times$ higher than in the old run. Clipping now triggers on every step late in training. The accumulation code is:

```python
G = grad_accum_for(B, micro_batch_size=8)
for _ in range(G):
    x, y = get_batch()
    compute_loss(model, x, y).backward()
```

What is wrong, why does the loss still look fine, and what is the fix?

<details><summary>Hint</summary>

What is the gradient of a *sum* of $G$ mean losses, compared with the gradient of their mean?

</details>

<details><summary>Solution</summary>

The loss is never divided by $G$, so `.grad` holds the *sum* of $G$ micro-batch gradients instead of their mean: $G$ times too large. $G$ grows from $1$ to $8$ during the ramp, which is exactly the staircase. The loss looks fine because Adam divides by $\sqrt{\hat v}$, which cancels a constant gradient scale (03.2), so the update barely changes. But clipping now acts on the inflated norm and fires constantly, and every batch jump mixes old and new scales in $\hat m$ and $\hat v$ for a while. SGD would have taken $8\times$ larger steps and probably diverged. Fix: `(compute_loss(model, x, y) / G).backward()` with the current step's $G$.

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

<details><summary>Why can a run use a small batch early in training without losing much, but not late?</summary>

Early, per-example gradients mostly agree, so the gradient noise scale $B_{\text{simple}} = \operatorname{tr}(\Sigma)/\lVert G\rVert^2$ is small and a small batch already gives nearly the true direction. As the loss falls, $\lVert G\rVert^2$ shrinks much faster than $\operatorname{tr}(\Sigma)$, so the noise scale grows and a larger batch is needed to average the noise away.

</details>

<details><summary>With $B_{\text{noise}} = 100$, what fraction of a perfect step do you get at $B = 100$ and at $B = 1000$?</summary>

$\frac{1}{1 + 100/100} = 0.5$ and $\frac{1}{1 + 100/1000} \approx 0.91$. Ten times the tokens per step for under twice the progress: past $B_{\text{noise}}$, extra batch is mostly wasted.

</details>

<details><summary>During a batch ramp, which part of the batch changes: the micro-batch size, the number of accumulation steps, or the number of GPUs? Why?</summary>

The number of accumulation steps $G$. The micro-batch $b$ is set by GPU memory, and changing it mid-run would change peak memory and could OOM; the number of GPUs is fixed for the job. $B = b\cdot G\cdot W$, so $G$ is the knob.

</details>

<details><summary>Why should the learning-rate schedule be expressed in tokens when the batch ramps?</summary>

With a fixed batch, steps and tokens are proportional, so it does not matter. With a ramp, early steps consume fewer tokens than late ones, so a warmup of $W$ steps covers fewer tokens than planned and the cosine decay ends at a different point in training. Tokens are the unit that stays meaningful.

</details>

## Next

You now have the full optimization toolkit: the update rule and momentum (03.1), adaptive per-parameter scaling and decoupled weight decay in AdamW (03.2), and the warmup + cosine schedule with gradient clipping that keep a real run stable, plus the batch ramp that large runs use to spend their early tokens well (03.3). Every training loop for the rest of the course uses exactly these pieces. The next module leaves optimization behind and turns to how text becomes numbers the model can consume — **tokenization**: characters, bytes, Unicode, and Byte-Pair Encoding from scratch.

Continue to [04.1 · Characters, bytes, Unicode, the vocab problem](lessons/module-04/lesson-01.md).
