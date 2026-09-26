# 10.3 · Chinchilla compute-optimality

<div class="prereq">
<p><strong>Prerequisites:</strong> the $6ND$ compute rule from <a href="#/lessons/module-10/lesson-01">10.1 · FLOPs & parameter counting</a>; power laws, the irreducible loss, and <code>fit_power_law</code> from <a href="#/lessons/module-10/lesson-02">10.2 · Kaplan scaling laws</a>; the chain rule / setting a derivative to zero from <a href="#/lessons/module-02/lesson-01">02.1</a>.</p>
<p><strong>You will learn:</strong> the Chinchilla parametric loss $L(N,D) = E + A/N^\alpha + B/D^\beta$ and what each term means; how to <strong>derive the compute-optimal allocation</strong> — given a fixed budget $C = 6ND$, minimise $L$ to get the best model size $N^\*$ and token count $D^\*$; the headline result that $N$ and $D$ should scale <strong>together</strong> (~20 tokens per parameter), so GPT-3-era models were badly <strong>undertrained</strong>; and how this honestly contrasts with Kaplan's earlier conclusion.</p>
<p><strong>Why this matters for ML:</strong> this single result changed how every lab spends its compute. Before Chinchilla, the field over-invested in parameters and under-fed them data; after it, "train a smaller model on far more tokens" became the default. Being able to solve the allocation yourself — turning a dollar budget into a concrete $(N^\*, D^\*)$ — is the practical payoff of the entire module.</p>
</div>

## 1. Intuition: one budget, two ways to spend it

You have a fixed compute budget $C$ — a number of FLOPs you can afford. Because $C = 6ND$ (10.1), that budget is a *trade*: spend it on a **bigger model** (larger $N$) trained on **fewer tokens** (smaller $D$), or a **smaller model** on **more tokens**. Every point on the curve $D = C/(6N)$ is affordable; they differ only in how the fixed budget is split.

Which split gives the lowest loss? Too big a model sees too little data and never learns to use its capacity (data-starved); too small a model saturates — extra tokens stop helping because the model has run out of room (capacity-starved). The best split is in between, and Chinchilla's contribution was to *measure the loss surface* $L(N,D)$ precisely enough to find that optimum and show it is not where the field had been operating.

<div class="callout key"><p>A compute budget $C=6ND$ is a constraint curve, not a single point. Loss $L(N,D)$ varies along it. Compute-optimal training means finding the $(N^\*, D^\*)$ on that curve with the lowest loss — and Chinchilla found it sits at roughly <strong>20 tokens per parameter</strong>, far more data than GPT-3-era models used.</p></div>

## 2. Mathematics: the Chinchilla parametric loss

Kaplan's laws (10.2) describe $L(N)$ and $L(D)$ separately. Chinchilla (Hoffmann et al. 2022) fits *both at once* with one parametric surface:

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}.
$$

Every symbol:
- $E$ — the **irreducible loss**: the entropy of natural language itself (nats/token). No model, however large, trained on however much data, can beat it. This is the floor 10.2 warned a bare power law ignores.
- $A/N^{\alpha}$ — the **finite-model penalty**: extra loss from having only $N$ parameters. Vanishes as $N \to \infty$. $\alpha$ is how fast capacity helps.
- $B/D^{\beta}$ — the **finite-data penalty**: extra loss from training on only $D$ tokens. Vanishes as $D \to \infty$. $\beta$ is how fast data helps.
- $A, B > 0$ — scale constants; $\alpha, \beta > 0$ — the two exponents.

The structure is the whole idea: loss is a floor $E$ plus two independently vanishing penalties, one for too few parameters and one for too few tokens. Fit $E, A, B, \alpha, \beta$ once (from many training runs) and you can predict the loss of any $(N, D)$ — including ones you have not run.

## 3. Derivation: the compute-optimal allocation

We minimise $L(N, D)$ subject to the budget constraint $C = 6ND$. Since $E$ is a constant it does not affect where the minimum is; only the two penalties matter.

**Step 1 — use the constraint to eliminate $D$.** From $C = 6ND$, $D = C/(6N)$. Substitute:

$$
L(N) = E + \frac{A}{N^{\alpha}} + B\left(\frac{6N}{C}\right)^{\beta}
     = E + A\,N^{-\alpha} + B\,6^{\beta} C^{-\beta}\, N^{\beta}.
$$

Now $L$ depends on the single variable $N$ (the budget $C$ is fixed). As $N$ grows the first penalty $A N^{-\alpha}$ falls (bigger model) but the second $\propto N^{\beta}$ rises (fewer tokens left) — the trade-off made explicit.

**Step 2 — set the derivative to zero.**

$$
\frac{dL}{dN} = -\alpha A\,N^{-\alpha-1} + \beta B\,6^{\beta} C^{-\beta}\,N^{\beta-1} = 0.
$$

Move the negative term over and multiply through by $N^{\alpha+1}$:

$$
\alpha A = \beta B\,6^{\beta} C^{-\beta}\,N^{\alpha+\beta}
\quad\Longrightarrow\quad
N^{\alpha+\beta} = \frac{\alpha A}{\beta B}\left(\frac{C}{6}\right)^{\beta}.
$$

**Step 3 — solve for $N^\*$, then $D^\*$.**

$$
\boxed{\,N^\* = \left[\frac{\alpha A}{\beta B}\left(\frac{C}{6}\right)^{\beta}\right]^{\frac{1}{\alpha+\beta}}, \qquad D^\* = \frac{C}{6\,N^\*}.\,}
$$

Reading the exponents: $N^\* \propto C^{a}$ and $D^\* \propto C^{b}$ with

$$
a = \frac{\beta}{\alpha+\beta}, \qquad b = \frac{\alpha}{\alpha+\beta}, \qquad a + b = 1.
$$

Because $a + b = 1$ and both are near $\tfrac12$, **$N$ and $D$ grow at almost the same rate** as the budget grows — double the compute and you make the model $\approx\!1.4\times$ bigger *and* train on $\approx\!1.4\times$ more tokens, not one at the expense of the other. Chinchilla reports $a \approx 0.46$, $b \approx 0.54$: roughly equal scaling. That is the headline finding in one line of algebra.

## 4. Numerical example: allocate a budget of $C = 10^{21}$ FLOPs

We use the replication constants of Besiroglu et al. (2024) — $A = 482.01$, $B = 2085.43$, $\alpha = 0.3478$, $\beta = 0.3658$, $E = 1.82$ — which reproduce Chinchilla's ~20-tokens/param rule cleanly (see §6 on why we do not use the paper's originally-tabulated constants). Take $C = 10^{21}$ FLOPs.

$$
\alpha + \beta = 0.7136, \qquad \frac{C}{6} = \frac{10^{21}}{6} = 1.667\times 10^{20}.
$$

The prefactor:

$$
\frac{\alpha A}{\beta B} = \frac{0.3478 \cdot 482.01}{0.3658 \cdot 2085.43} = \frac{167.66}{762.85} = 0.2198.
$$

The budget factor:

$$
\left(\frac{C}{6}\right)^{\beta} = (1.667\times 10^{20})^{0.3658} = 2.4955\times 10^{7}.
$$

So

$$
N^\* = \left[0.2198 \cdot 2.4955\times 10^{7}\right]^{1/0.7136} = \left[5.485\times 10^{6}\right]^{1.4014} = 2.78\times 10^{9},
$$

$$
D^\* = \frac{C}{6\,N^\*} = \frac{10^{21}}{6 \cdot 2.78\times 10^{9}} = 6.00\times 10^{10}.
$$

**Compute-optimal for $10^{21}$ FLOPs: a $\approx 2.8$-billion-parameter model on $\approx 60$ billion tokens.** The tokens-per-parameter ratio:

$$
\frac{D^\*}{N^\*} = \frac{6.00\times 10^{10}}{2.78\times 10^{9}} \approx 21.6 \ \text{tokens per parameter}.
$$

There is the ~20× rule, fallen straight out of the arithmetic. Running `chinchilla_optimal(1e21, 482.01, 2085.43, 0.3478, 0.3658, 1.82)` returns `(2.778e9, 5.999e10)`, and the predicted loss $L(N^\*, D^\*) = 2.31$ nats.

Scaling up to **Gopher's actual budget** $C = 5.76\times 10^{23}$ FLOPs, the same formula gives $N^\* \approx 7.2\times 10^{10}$ (72B params) and $D^\* \approx 1.33\times 10^{12}$ (1.33T tokens), ratio $\approx 18$ — which is essentially Chinchilla's real recipe: a 70B model on 1.4T tokens, *four times smaller than the 280B Gopher trained on the same compute, and better*.

## 5. The code

`chinchilla_optimal` implements the boxed closed form directly:

```python
def chinchilla_optimal(compute_flops, A, B, alpha, beta, E=0.0):
    """Compute-optimal (N*, D*) for budget C = 6ND, minimising E + A/N^a + B/D^b."""
    half_budget = compute_flops / 6.0            # C/6 = N*D
    ratio = (alpha * A) / (beta * B)             # the alpha A / (beta B) prefactor
    N_star = (ratio * half_budget**beta) ** (1.0 / (alpha + beta))
    D_star = compute_flops / (6.0 * N_star)
    return float(N_star), float(D_star)
```

`E` is accepted (so the signature matches the full loss) but does not enter the optimisation — it is an additive constant. The test `test_chinchilla_optimal_matches_grid_minimum` in `code/tests/test_scaling.py` confirms this closed form agrees with a brute-force grid minimisation of $L(N, C/6N)$ over $N$, and reproduces $N^\* \approx 70$B, $D^\* \approx 1.4$T at Gopher's budget.

## 6. Kaplan vs Chinchilla, honestly

Both papers fit power laws to LM loss; they reach *different* compute-allocation conclusions, and it is worth being precise about why rather than declaring one simply "wrong".

- **Kaplan et al. (2020)** concluded that when compute grows, most of it should go into a **bigger model**, with token count growing only slowly ($D \propto C^{\sim 0.27}$). Following this, GPT-3 (175B) was trained on only ~300B tokens — under 2 tokens per parameter.
- **Chinchilla (Hoffmann et al. 2022)** re-ran the study with a learning-rate schedule matched to each run's token budget (Kaplan had used a fixed schedule, which penalised the data-heavy runs) and concluded $N$ and $D$ should scale **about equally** ($a \approx b \approx 0.5$), i.e. ~20 tokens per parameter. Under this view GPT-3 and Gopher were **badly undertrained**: for their compute they should have been several times smaller and fed several times more data.

The core technical disagreement is the learning-rate-schedule methodology, which shifted the estimated exponents. Chinchilla's conclusion is now the consensus for *training*-compute-optimality.

<div class="callout warn"><p><strong>PUBLIC vs INFERENCE.</strong> The parametric form $L=E+A/N^\alpha+B/D^\beta$ and the "$N$ and $D$ scale together, ~20 tokens/param" conclusion are <em>publicly documented</em> in Chinchilla. But the paper's <em>tabulated</em> constants ($A{=}406.4, B{=}410.7, \alpha{=}0.34, \beta{=}0.28, E{=}1.69$) were later shown by a replication (Besiroglu et al. 2024) to be slightly internally inconsistent with the paper's own ~20× rule — plugging them into the boxed formula gives a tokens/param ratio that drifts to 30–90×. We therefore use the replication's revised constants for the worked example, and treat the exact numbers as an <em>estimate with real uncertainty</em>, not gospel. The <em>shape</em> of the result — equal-ish scaling, ~20×, GPT-3 undertrained — is robust; the third significant figure of any specific $N^\*$ is not.</p></div>

<div class="callout warn"><p><strong>Compute-optimal to <em>train</em> ≠ optimal to <em>serve</em>.</strong> Chinchilla minimises <em>training</em> loss per training FLOP. But a model is trained once and served billions of times, so it is often worth "over-training" a <em>smaller</em> model far past its Chinchilla-optimal token count: it costs more to train but is cheaper to run forever. LLaMA (paper #9) did exactly this — 7B–65B models trained on ~1–1.4T tokens, well beyond 20×. Chinchilla tells you the training optimum; inference economics can rationally pull you away from it.</p></div>

## Research connection

<div class="callout paper"><p><strong>Training Compute-Optimal Large Language Models</strong> (Hoffmann et al., 2022) — <a href="#/papers/index">paper #4 in the reading list</a>, "Chinchilla". <strong>Read:</strong> the abstract, the IsoFLOP figure (loss-vs-model-size curves at fixed compute, whose minima trace the optimal frontier), and the parametric fit $L(N,D)=E+A/N^\alpha+B/D^\beta$. <strong>Understand:</strong> for fixed compute, $N$ and $D$ scale together (~20 tokens/param); prior models were oversized and underfed; three independent estimation approaches converge. <strong>Skip on a first pass:</strong> the per-approach uncertainty analysis and the downstream eval tables. Pair it with Kaplan (paper #3) and read them as a debate, not a contradiction.</p></div>

## Exercise

Use the constants $A = 482.01$, $B = 2085.43$, $\alpha = 0.3478$, $\beta = 0.3658$. **(a)** For a budget $C = 10^{19}$ FLOPs, compute $N^\*$ and $D^\*$ and the tokens/param ratio. **(b)** Suppose instead you are *given* a 6-billion-parameter model and $C = 10^{19}$ FLOPs. How many tokens does the budget allow, and how does that compare to the compute-optimal $D^\*$ — is this model too big or too small for the budget?

<details><summary>Hint</summary>
(a) plug into the boxed formula, or call <code>chinchilla_optimal(1e19, ...)</code>. (b) tokens allowed $= C/(6N) = C/(6\cdot 6\text{e}9)$; compare to $D^\*$ from (a) and to the ~20× rule.
</details>

<details><summary>Stronger hint</summary>
(a) $\alpha+\beta = 0.7136$; $C/6 = 1.667\text{e}18$. (b) $C/(6\cdot 6\text{e}9) \approx 2.78\text{e}8$ tokens $\Rightarrow$ tokens/param $\approx 0.046$ — compare to ~20.
</details>

<details><summary>Solution</summary>

**(a)** `chinchilla_optimal(1e19, 482.01, 2085.43, 0.3478, 0.3658)` gives $N^\* \approx 2.62\times 10^{8}$ (262M params) and $D^\* \approx 6.36\times 10^{9}$ (6.4B tokens), ratio $\approx 24$ tokens/param — in the ~20× band.

**(b)** With $N = 6\times 10^{9}$ fixed, the budget allows $D = C/(6N) = 10^{19}/(6\cdot 6\times 10^{9}) \approx 2.78\times 10^{8}$ tokens — only **0.046 tokens per parameter**, hundreds of times below optimal. The 6B model is *far too big* for a $10^{19}$-FLOP budget: it would see almost no data and train terribly. The compute-optimal choice at this budget is the 262M model of part (a). This is the Chinchilla lesson in miniature — a big model on a small budget is a waste.

</details>

## Common mistakes

- **Thinking "compute-optimal" means "as big as possible".** It means the *best split* of a fixed budget; past ~20 tokens/param a bigger model actually raises loss for that budget.
- **Treating 20 tokens/param as an exact law.** It is a robust ballpark from the fit; the precise ratio drifts with the (uncertain) constants and the compute scale.
- **Confusing training-optimal with deployment-optimal.** For a heavily-served model, over-training a smaller one (LLaMA-style) can be the right call even though it is not Chinchilla-optimal to train.
- **Dropping $E$ from the loss but expecting it to move the optimum.** $E$ is an additive constant; it changes the loss *value* but not where $(N^\*, D^\*)$ sits.

## Check yourself

<details><summary>In $L(N,D)=E+A/N^\alpha+B/D^\beta$, what does each of the three terms represent?</summary>

$E$ is the irreducible loss — the entropy of the data, a floor no model can beat. $A/N^\alpha$ is the finite-model penalty (loss from having only $N$ parameters), vanishing as $N\to\infty$. $B/D^\beta$ is the finite-data penalty (loss from only $D$ tokens), vanishing as $D\to\infty$.

</details>

<details><summary>Why does the constraint $C = 6ND$ turn a two-variable minimisation into a one-variable one?</summary>

The constraint pins $D$ to $N$: $D = C/(6N)$. Substituting removes $D$, leaving $L$ as a function of $N$ alone (with $C$ fixed), so a single derivative $dL/dN = 0$ finds the optimum.

</details>

<details><summary>Chinchilla finds $a \approx 0.46$, $b \approx 0.54$ with $a+b=1$. What does $a+b=1$ tell you, and what do the near-equal values mean?</summary>

$a+b=1$ is forced by the constraint $C \propto ND$: since $N^\* D^\* \propto C^{a+b}$ must equal $C^1$, the exponents sum to 1. Near-equal $a,b\approx 0.5$ means $N$ and $D$ should grow at almost the same rate as the budget — scale model and data together, not one at the expense of the other.

</details>

<details><summary>GPT-3 was 175B parameters trained on ~300B tokens. Roughly how many tokens/param is that, and what does Chinchilla say about it?</summary>

$300\text{B}/175\text{B} \approx 1.7$ tokens/param — more than ten times below the ~20× optimum. Chinchilla says GPT-3 was badly undertrained: for its compute it should have been several times smaller and trained on far more tokens, and would have reached lower loss.

</details>

<details><summary>Doubling the compute budget $C$, by roughly what factor should $N^\*$ and $D^\*$ each grow (using $a\approx b\approx 0.5$)?</summary>

$N^\* \propto C^{a} \approx C^{0.5}$ and $D^\* \propto C^{b} \approx C^{0.5}$, so each grows by $\approx 2^{0.5} \approx 1.41$. Doubling compute means a ~1.4× bigger model on ~1.4× more tokens.

</details>

<div class="hw">
<p><strong>Hardware track — this lesson.</strong></p>
<p><strong>Minimum / recommended:</strong> pure arithmetic — the allocation solver, the derivative, the worked example all run instantly on any CPU. <strong>Runtime:</strong> milliseconds. <strong>GPU memory / GPU-hours:</strong> 0. <strong>CPU-only:</strong> yes. The <em>point</em> of this lesson is that you decide $(N^\*, D^\*)$ on paper <em>before</em> committing GPU-hours — the expensive part (fitting the constants $E, A, B, \alpha, \beta$) is the many-run study a lab does once; consuming the result is free.</p>
</div>

## Next

You have completed the scaling-laws module: you can count parameters and FLOPs (10.1), fit a power law to a loss sweep (10.2), and allocate a compute budget compute-optimally (10.3). The Chinchilla optimum assumes an *abundant, clean* token supply — but where do trillions of high-quality tokens come from? The next module is data engineering: turning Common Crawl into training text, deduplicating it, and guarding against contamination — the other half of making the $D$ in $6ND$ real.

Continue to [11.1 · Common Crawl → clean text](lessons/module-11/lesson-01.md).
