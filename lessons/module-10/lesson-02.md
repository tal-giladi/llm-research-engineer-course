# 10.2 · Kaplan scaling laws; fit a curve

<div class="prereq">
<p><strong>Prerequisites:</strong> parameter counting and the $6ND$ compute rule from <a href="lesson-01.md">10.1 · FLOPs & parameter counting</a>; the training loop from <a href="../module-07/lesson-01.md">07.1</a> and <code>get_batch</code> from <a href="../module-04/lesson-03.md">04.3</a>; cross-entropy loss as the training signal from <a href="../module-01/lesson-04.md">01.4</a>. Logarithms and a straight-line fit ($y = mx + b$) — no more than that.</p>
<p><strong>You will learn:</strong> what a <strong>power law</strong> is and why it becomes a straight line on <strong>log-log axes</strong>; the Kaplan et al. 2020 empirical laws $L(N)=(N_c/N)^{\alpha_N}$ (and the analogous laws in data $D$ and compute $C$); and — hands on — how to <strong>fit one</strong>: train several tiny GPTs of increasing size on one toy corpus, record their final loss, and recover the exponent by linear regression in log-log space with <code>fit_power_law</code>.</p>
<p><strong>Why this matters for ML:</strong> scaling laws are how labs <em>predict the future</em>. Fit the power law on cheap small runs, then extrapolate the straight line to say what loss a 100× bigger model will reach — <em>before</em> building it. This is the empirical backbone of every modern pretraining decision, and the tool that makes 10.3's compute-optimal allocation possible.</p>
</div>

## 1. Intuition: a straight line in the right coordinates

Train a language model, plot its final loss against how many parameters it has, and you get a curve that keeps dropping but bends — big early gains, diminishing returns. That curve looks hopeless to extrapolate. But plot the *same data* with **both axes on a log scale**, and it straightens into a line you can extend with a ruler. That straightening is the signature of a **power law**: loss falls as a fixed *fraction* every time you *multiply* the resource by a fixed factor.

Kaplan et al. (2020) measured this across seven orders of magnitude of model size, data, and compute and found the same clean straight line every time. The remarkable empirical fact is not that bigger is better — everyone expected that — but that the improvement is *so regular* you can write it as a two-parameter formula and trust the extrapolation.

<div class="callout key"><p>A power law $L = c\,x^{-\alpha}$ is a straight line in log-log coordinates with slope $-\alpha$. "Loss drops by a constant factor per constant multiplicative increase in $N$" — that is what makes small-scale runs predictive of large-scale ones.</p></div>

## 2. Mathematics: the power law and its log-log line

Kaplan's laws, in the *decay* convention (loss falls as a positive power of the resource):

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad
L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \qquad
L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}.
$$

Every symbol:
- $N$ — non-embedding parameter count (10.1); $D$ — training tokens; $C$ — compute in FLOPs.
- $L$ — the cross-entropy loss (nats/token), the quantity we minimise.
- $\alpha_N, \alpha_D, \alpha_C > 0$ — the **scaling exponents**. Larger $\alpha$ = steeper improvement. Kaplan reports small values, e.g. $\alpha_N \approx 0.076$.
- $N_c, D_c, C_c$ — constants with units of the resource; $N_c$ is the scale at which the fitted $L(N)$ would hit $1$.

Each law holds "all else abundant" — $L(N)$ is measured with enough data that data is not the bottleneck, and vice versa.

**Why log-log is a line.** Take a general power law $L = c\,x^{-\alpha}$ (here $c = N_c^{\alpha}$ folds the constant in). Take logs of both sides:

$$
\log L = \log c - \alpha \log x.
$$

Read that as $\underbrace{\log L}_{y} = \underbrace{(-\alpha)}_{\text{slope}} \underbrace{\log x}_{X} + \underbrace{\log c}_{\text{intercept}}$. Plotting $\log L$ against $\log x$ gives a straight line whose **slope is $-\alpha$** and whose **intercept is $\log c$**. Fitting the power law is therefore just fitting a line to log-transformed data — ordinary least squares.

## 3. Numerical example: fit an exponent from two points by hand

Suppose a sweep gives two measurements: a $1\text{M}$-parameter model reaches loss $4.0$, and a $100\text{M}$-parameter model reaches loss $3.0$. Assume $L(N) = c\,N^{-\alpha_N}$ and solve for the exponent from just these two points.

The slope in log-log space is rise over run. Using base-10 logs (any base works — the exponent is base-independent):

$$
\text{slope} = \frac{\log_{10} L_2 - \log_{10} L_1}{\log_{10} N_2 - \log_{10} N_1}
= \frac{\log_{10} 3.0 - \log_{10} 4.0}{\log_{10}(10^{8}) - \log_{10}(10^{6})}
= \frac{0.4771 - 0.6021}{8 - 6} = \frac{-0.1249}{2} = -0.0625.
$$

The slope is $-\alpha_N$, so $\alpha_N \approx 0.0625$. Now the coefficient $c$, from $L_1 = c\,N_1^{-\alpha_N}$ at $N_1 = 10^6$:

$$
(10^6)^{-0.0625} = 10^{-0.375} = 0.4217, \qquad c = \frac{L_1}{0.4217} = \frac{4.0}{0.4217} \approx 9.48.
$$

So the fitted law is $L(N) \approx 9.48\,N^{-0.0625}$. Sanity check at $N_2 = 10^8$: $9.48 \cdot (10^8)^{-0.0625} = 9.48 \cdot 10^{-0.5} = 9.48 \cdot 0.316 = 3.00$. ✓

Running `fit_power_law([1e6, 1e8], [4.0, 3.0])` returns `(0.06247, 9.4815)` — matching the hand computation. With only two points the "fit" is exact (a line through two points); with more points, least squares finds the best line through the scatter, which is the real use.

## 4. From scratch: `fit_power_law` via log-log least squares

The fit is short — take logs, run a degree-1 least-squares polynomial fit, negate the slope:

```python
import numpy as np

def fit_power_law(x, y):
    """Fit y = coefficient * x**(-exponent) by least squares in log-log space."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    log_x, log_y = np.log(x), np.log(y)
    slope, intercept = np.polyfit(log_x, log_y, deg=1)   # log_y = slope*log_x + intercept
    exponent = -float(slope)          # slope is -alpha in the decay convention
    coefficient = float(np.exp(intercept))
    return exponent, coefficient
```

Two design points worth stating explicitly. First, we fit in **log space, not linear space**: minimising squared error on $\log L$ treats a 10% error at loss 5 the same as a 10% error at loss 2, which is what you want for a multiplicative law — a linear-space fit would let the large-loss points dominate. Second, we **negate the slope** so a decaying law (loss falls as $x$ grows) reports a *positive* exponent, matching the $(N_c/N)^{\alpha}$ convention. This is the version in `llmre.evaluation.scaling`; its docstring states the exact model it fits.

## 5. Fit a real curve: a tiny GPT sweep

Now the hands-on part the brief asks for: fit a scaling law on models you actually train. The script `code/experiments/scaling_sweep.py` trains six tiny GPTs of increasing width (`n_embd` from 8 to 64, two layers each) on one fixed toy corpus, records each model's final loss, and calls `fit_power_law`. The core loop:

```python
from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT
from llmre.data.loader import get_batch
from llmre.evaluation.scaling import fit_power_law

Ns, Ls = [], []
for w in [8, 16, 24, 32, 48, 64]:
    cfg = GPTConfig(vocab_size=256, block_size=32, n_layer=2, n_head=4, n_embd=w)
    model = GPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95))
    for _ in range(600):                       # a few hundred steps is plenty for a toy
        x, y = get_batch(data, 32, 32)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    Ns.append(model.num_params(non_embedding=True))
    Ls.append(final_loss(model, data))         # mean loss over a few held-out batches

exponent, coefficient = fit_power_law(Ns, Ls)
```

Running it (seeded, ~1–2 minutes on a laptop CPU) produces a table like this and the fit below it:

```
 n_embd   N (non-emb)  final loss
      8         1,536      2.6931
     16         6,144      2.4489
     24        13,824      2.2707
     32        24,576      2.1435
     48        55,296      1.9017
     64        98,304      1.7331

fitted power law:  L(N) = 4.86 * N**(-0.0700)
fitted exponent alpha_N = 0.0700
```

Loss falls monotonically as width grows, and the six points lie close to a log-log straight line — a real, if tiny, scaling law fitted from scratch. The fitted exponent (**$\alpha_N \approx 0.07$** on this toy setup) happens to land near Kaplan's reported $\alpha_N \approx 0.076$, but do not read anything into that coincidence: six 100k-parameter models on a repeated toy corpus cannot estimate the real exponent. What the sweep *does* show honestly is the mechanism — bigger model, lower loss, straight in log-log — and that `fit_power_law` recovers a sensible slope from real training runs.

<div class="callout warn"><p>A toy sweep like this is a <em>demonstration</em>, not a measurement of "the" scaling exponent. Real scaling studies span many orders of magnitude of $N$, use a real corpus large enough that data is never the bottleneck, tune each run's learning-rate schedule to its size, and average over seeds. The exponent from six tiny models on a 5-sentence corpus is qualitative only.</p></div>

## 6. Under the hood: least squares, and what the fit assumes

`np.polyfit(log_x, log_y, 1)` solves a tiny linear least-squares problem: find slope $m$ and intercept $b$ minimising $\sum_i (\,b + m\log x_i - \log y_i\,)^2$. That has the closed-form normal-equations solution (module 2's linear algebra), which is why the fit is instant and has no hyperparameters. The fit *assumes* the log-log relationship really is linear — that a single power law holds over the range you measured. That assumption is what Kaplan verified empirically and what can break: near the **irreducible loss** (the entropy of the data itself, below which no model can go — the $E$ term of 10.3) the curve flattens and bends *away* from the straight line, so extrapolating a pure power law too far predicts impossibly low loss. A good fit reports the range it was fit over and does not extrapolate past where the line stays straight.

## Research connection

<div class="callout paper"><p><strong>Scaling Laws for Neural Language Models</strong> (Kaplan et al., 2020) — <a href="../../papers/index.md">paper #3 in the reading list</a>. This is the paper you have just reproduced in miniature. <strong>Read:</strong> the abstract, the main power-law figure (loss vs $N$, $D$, $C$ as straight log-log lines), and equation set for $L(N), L(D), L(C)$. <strong>What to understand:</strong> loss follows power laws over many orders of magnitude; larger models are more sample-efficient; how they split a compute budget between $N$ and $D$. <strong>Skip on a first pass:</strong> the batch-size critical-value derivations and the detailed fitting-methodology appendices. Note the conclusion about compute allocation — Kaplan favours spending most of extra compute on <em>bigger models</em>. Lesson 10.3 (Chinchilla) revises exactly that conclusion, so hold it loosely.</p></div>

## Exercise

A sweep of three models gives: $N = 10^6 \to L = 3.60$; $N = 10^7 \to L = 3.00$; $N = 10^8 \to L = 2.50$. **(a)** Using the first and last points, estimate $\alpha_N$ by the two-point log-log slope. **(b)** Predict the loss of a $10^9$-parameter model by extrapolating the line. **(c)** Why should you distrust that prediction?

<details><summary>Hint</summary>
(a) slope $= (\log_{10}L_{\text{last}} - \log_{10}L_{\text{first}})/(\log_{10}N_{\text{last}} - \log_{10}N_{\text{first}})$, then $\alpha_N = -\text{slope}$. (b) each 10× in $N$ multiplies $L$ by $10^{-\alpha_N}$; you are going one more decade. (c) think about the irreducible loss.
</details>

<details><summary>Stronger hint</summary>
(a) $(\log_{10} 2.50 - \log_{10} 3.60)/(8 - 6)$. (b) $L(10^9) = L(10^8)\cdot 10^{-\alpha_N}$. (c) a pure power law predicts loss $\to 0$ as $N\to\infty$, but text has nonzero entropy.
</details>

<details><summary>Solution</summary>

**(a)** $\log_{10} 2.50 = 0.3979$, $\log_{10} 3.60 = 0.5563$. Slope $= (0.3979 - 0.5563)/(8 - 6) = -0.1584/2 = -0.0792$, so $\alpha_N \approx 0.079$.

**(b)** Going from $10^8$ to $10^9$ is one more decade, multiplying loss by $10^{-0.0792} = 0.833$. So $L(10^9) \approx 2.50 \cdot 0.833 \approx 2.08$. (`fit_power_law` on all three points gives a very similar slope, since they are near-collinear in log-log.)

**(c)** The pure power law implies loss keeps falling toward $0$ forever, but language has an **irreducible entropy** $E > 0$ that no model can beat. As $N$ grows the true curve flattens toward $E$ and bends below the straight line, so a single-power-law extrapolation over-predicts the gains. This is precisely why Chinchilla (10.3) fits $L(N,D) = E + A/N^\alpha + B/D^\beta$ with an explicit floor $E$, instead of a bare power law.

</details>

## Common mistakes

- **Fitting in linear space.** Fit the *logs*; a linear-space least-squares fit lets the large-loss points dominate and distorts the exponent.
- **Reading the sign wrong.** In log-log the slope is $-\alpha$; a decaying loss has a *negative* slope and a *positive* exponent. `fit_power_law` negates for you — know that it does.
- **Extrapolating past the straight part.** Near the irreducible loss the line bends; extrapolating a pure power law there predicts impossibly low loss.
- **Treating a toy exponent as "the" exponent.** Six tiny models on a toy corpus estimate the *mechanism*, not the real $\alpha_N$.

## Check yourself

<details><summary>Why does a power law $L = c\,x^{-\alpha}$ become a straight line on log-log axes, and what are the line's slope and intercept?</summary>

Taking logs: $\log L = \log c - \alpha \log x$. With $\log x$ on the horizontal axis and $\log L$ on the vertical, this is $y = (-\alpha)X + \log c$ — a line of slope $-\alpha$ and intercept $\log c$.

</details>

<details><summary>A fit returns exponent $0.05$ and coefficient $8.0$. What loss does it predict at $N = 10^7$?</summary>

$L = 8.0 \cdot (10^7)^{-0.05} = 8.0 \cdot 10^{-0.35} = 8.0 \cdot 0.4467 \approx 3.57$.

</details>

<details><summary>You fit $\alpha_N = 0.076$. Roughly how much must you scale $N$ to halve the loss (ignoring the irreducible floor)?</summary>

Halving loss needs $x^{-\alpha_N} = 0.5$, i.e. $x = 0.5^{-1/0.076} = 2^{1/0.076} = 2^{13.2} \approx 9400$. You must grow $N$ by roughly four orders of magnitude to halve the loss — the brutal economics of scaling, and why exponents this small make frontier training so expensive.

</details>

<details><summary>Your six-point sweep bends downward at the largest model instead of staying straight. What might be happening?</summary>

Two common causes: (1) the largest model is starting to hit the irreducible loss / overfit the tiny corpus, so the curve flattens; or (2) the larger models are under-trained or mistuned (same step count and LR for every size), so their measured loss is artificially high or low. Either way the single-power-law fit is no longer valid over the whole range — refit only over the straight portion, or fix the training.

</details>

<div class="hw">
<p><strong>Hardware track — the toy sweep.</strong></p>
<p><strong>Minimum:</strong> any laptop CPU. The six models are ~1k–100k parameters each; the whole sweep of 6 × 600 steps runs in roughly 1–2 minutes on CPU, no GPU required. <strong>Recommended:</strong> same — a GPU is pointless at this size (kernel-launch overhead dominates). <strong>GPU memory:</strong> negligible (<50 MB). <strong>GPU-hours:</strong> 0. <strong>CPU-only:</strong> yes, this is designed for it. A <em>real</em> scaling study is the opposite extreme — dozens of runs from millions to billions of parameters, many GPU-days each; that is a cluster job, not a lesson exercise, which is exactly why we fit the mechanism on a toy.</p>
</div>

## Next

You can now fit a scaling law and read its exponent. But Kaplan's laws describe $L(N)$ and $L(D)$ *separately*; the real planning question couples them: given a fixed compute budget $C = 6ND$, how do you split it between model size and tokens to get the lowest loss? Chinchilla answers that with a single fit $L(N, D)$ and a constrained optimisation — and overturns Kaplan's "spend it on bigger models" conclusion.

Continue to [10.3 · Chinchilla compute-optimality](lesson-03.md).
