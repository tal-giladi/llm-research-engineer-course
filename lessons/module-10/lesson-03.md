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

## 3. From training measurements to the compute-optimal allocation

The headline you already know is **roughly 20 training tokens per parameter**: a 1B-parameter model would be trained on about 20B tokens. How did the researchers get there, and why do we need $A$, $B$, $\alpha$ and $\beta$?

There are two separate jobs:

1. **Fit a loss predictor from experiments.** Learn which values of $E, A, B, \alpha, \beta$ make the formula match measured losses.
2. **Use that predictor to plan a training run.** Given a compute budget $C$, calculate the model size $N$ and training-token count $D$ with the lowest predicted loss.

The constants come from the first job. The derivative comes in the second job. The 20-token rule is an approximate result of the analysis, rather than an assumption used to choose the constants.

### 3.1 What do the experiments actually give us?

Imagine a spreadsheet in which each row records one training run:

| Quantity | Where it comes from | Example |
|---|---|---|
| $N_i$ | Count the model's parameters | 1 billion parameters |
| $D_i$ | Count the tokens processed during training | 20 billion tokens |
| $L_i$ | Measure the resulting cross-entropy loss | 2.5 nats/token |

The subscript $i$ labels the run; the example row is illustrative. Train other configurations with different model sizes and token budgets, and collect more rows. Compare them using a consistent data distribution, tokenizer and loss measurement. Here $D$ means **tokens processed during training**, rather than the vocabulary size or number of documents.

**Publicly documented:** Chinchilla used over 400 training runs, varying both model size and training tokens. Its parametric approach fitted the runs' final losses. The paper used smoothed training loss as an estimate of test loss in its abundant-data setting. See [Hoffmann et al., sections 3 and 3.3](https://arxiv.org/html/2203.15556v1#S3).

At this point we have observations $(N_i,D_i,L_i)$. The experiment has not directly measured an "$A$" or an "$\alpha$". Those are numbers we will infer from the pattern across runs.

### 3.2 Why introduce these five constants?

We choose the following *family of possible loss predictors*:

$$
\widehat L(N,D)=E+\frac{A}{N^\alpha}+\frac{B}{D^\beta}.
$$

The hat on $\widehat L$ means **predicted loss**; $L_i$ is a measured loss. This is a modelling assumption whose usefulness must be checked against experiments.

| Unknown constant | What changing it does to the prediction |
|---|---|
| $E$ | Raises or lowers the floor approached when both resources become very large |
| $A$ | Scales the loss penalty associated with limited model size |
| $\alpha$ (alpha) | Controls how quickly that model-size penalty falls as $N$ increases |
| $B$ | Scales the loss penalty associated with limited training tokens |
| $\beta$ (beta) | Controls how quickly that training-token penalty falls as $D$ increases |

For example, doubling $A$ doubles the model penalty at every $N$. With $\alpha=0.5$, multiplying $N$ by four halves that penalty, because $4^{-0.5}=1/2$. The same reasoning applies to $B$, $D$ and $\beta$.

These five numbers describe the **loss predictor**. They are different from the billions of learned weights inside a language model. We fit them after training the experimental models, using the losses those models produced. They also depend on the experimental setup and units: section 4's constants expect raw counts, so a 1B model means $N=10^9$, not $N=1$.

### 3.3 A tiny example: recover every constant by hand

To make "fit the constants" concrete, use these **invented, noise-free observations**. The tiny $N,D$ values are for arithmetic practice, not realistic language models or Chinchilla measurements.

| Run | Parameters $N$ | Training tokens $D$ | Measured loss $L$ |
|---|---:|---:|---:|
| 1 | 4 | 4 | 7.0 |
| 2 | 16 | 4 | 6.0 |
| 3 | 64 | 4 | 5.5 |
| 4 | 4 | 16 | 5.0 |
| 5 | 4 | 64 | 4.0 |

**First recover $\alpha$ and $A$.** Compare runs 1, 2 and 3. Only $N$ changes; $D=4$ stays fixed. Therefore $E+B/4^\beta$ is the same unknown offset in all three losses. Subtracting successive losses cancels that offset.

Each step multiplies $N$ by four. The loss decreases are $7-6=1$ and $6-5.5=0.5$: the second decrease is half the first. For a power law, equal resource multipliers produce equal penalty multipliers; successive decreases have that same multiplier. Thus the model penalty is multiplied by $0.5$ whenever $N$ is multiplied by four:

$$
4^{-\alpha}=0.5
\quad\Longrightarrow\quad
\alpha=\frac{\log 2}{\log 4}=0.5.
$$

Now use the first loss decrease to recover $A$:

$$
7-6=A\left(\frac{1}{\sqrt4}-\frac{1}{\sqrt{16}}\right)
=A\left(\frac12-\frac14\right)=\frac A4
\quad\Longrightarrow\quad A=4.
$$

**Next recover $\beta$ and $B$.** Compare runs 1, 4 and 5. Now $N=4$ stays fixed and $D$ multiplies by four. The decreases are $7-5=2$ and $5-4=1$. Again the second is half the first, so $4^{-\beta}=0.5$ and $\beta=0.5$. Then:

$$
7-5=B\left(\frac{1}{\sqrt4}-\frac{1}{\sqrt{16}}\right)
=\frac B4
\quad\Longrightarrow\quad B=8.
$$

**Finally recover $E$.** Substitute run 1 and the four constants we just found:

$$
7=E+\frac4{\sqrt4}+\frac8{\sqrt4}=E+2+4
\quad\Longrightarrow\quad E=1.
$$

We have recovered the complete predictor:

$$
\widehat L(N,D)=1+\frac4{\sqrt N}+\frac8{\sqrt D}.
$$

Check run 5: $1+4/\sqrt4+8/\sqrt{64}=1+2+1=4$, exactly its measured loss. **This is where constants come from: choose them so the formula reproduces the observed relationship between resources and loss.**

### 3.4 How does fitting work with real, noisy measurements?

Real losses do not line up this perfectly. Instead of solving five exact equations, a fitting program adjusts all five constants together to reduce prediction errors across many runs.

For each trial set of constants, it predicts each row's loss and compares it with the measured value. For example, predicting 6.8 for a row whose measured loss is 7.0 gives an error of $6.8-7.0=-0.2$. The program combines errors into a fitting objective, updates the constants, and repeats.

**The paper's fitting method:** it compared predicted and observed **log losses**, used a Huber objective to limit the influence of outliers, and used L-BFGS with multiple starting guesses to search for good constants. See [Hoffmann et al., section 3.3, equation 3](https://arxiv.org/html/2203.15556v1#S3.SS3).

This extends the curve fitting in lesson 10.2. We cannot simply take the log of the full prediction and fit one straight line: the logarithm of a **sum** does not split into a sum of logarithms. We need a joint nonlinear fit.

For the rest of this lesson, use the following fitted values:

$$
A=482.01,\qquad B=2085.43,\qquad
\alpha=0.3478,\qquad\beta=0.3658,\qquad E\approx1.82.
$$

**Source of these particular numbers:** they are the revised estimates from [Besiroglu et al. (2024), Table 1](https://arxiv.org/html/2404.10102v2#S3), which refitted data reconstructed from Chinchilla's plots. They are not the constants originally printed in the 2022 paper. Section 6 explains the discrepancy. We are using a published fit here; the small example above teaches how fitting works.

There are also **two different objectives** to keep separate. Fitting minimises the predictor's errors by changing $E,A,B,\alpha,\beta$. Planning minimises the predicted language-model loss by changing $N,D$, with those fitted constants held fixed. We now move to planning.

### 3.5 Turn the compute budget into a trade-off

Suppose the available training compute is $C$ FLOPs. Using lesson 10.1's approximation:

$$
C=6ND
\quad\Longrightarrow\quad
D=\frac{C}{6N}.
$$

For example, a toy budget of $C=1536$ allows $ND=256$. It can pay for $(N,D)=(4,64)$, $(8,32)$ or $(16,16)$. Doubling model size halves the affordable tokens.

Use our **toy fitted predictor** to compare those choices:

| Parameters $N$ | Tokens $D=256/N$ | Predicted loss $1+4/\sqrt N+8/\sqrt D$ |
|---:|---:|---:|
| 4 | 64 | $1+2+1=4.000$ |
| 8 | 32 | $1+1.414+1.414\approx3.828$ |
| 16 | 16 | $1+1+2=4.000$ |

The middle choice is better than its neighbours. The calculus below locates the minimum without having to try every possible model size.

### 3.6 Substitute the constraint, then differentiate

Return to the general formula. Keep $C,E,A,B,\alpha,\beta$ fixed and replace $D$ by $C/(6N)$:

$$
\begin{aligned}
\widehat L_C(N)
&=E+\frac{A}{N^\alpha}+\frac{B}{[C/(6N)]^\beta}\\
&=E+A N^{-\alpha}+B\left(\frac{6N}{C}\right)^\beta\\
&=E+A N^{-\alpha}+B\left(\frac6C\right)^\beta N^\beta.
\end{aligned}
$$

The subscript $C$ reminds us that we are considering one fixed budget. Increasing $N$ makes the model penalty fall, but makes the training-token penalty rise because fewer tokens are affordable. In the toy case, substituting $D=256/N$ gives $1+4/\sqrt N+\sqrt N/2$; at $N=8$ this is again about 3.828.

Use the power rule: the derivative of $kN^p$ is $kpN^{p-1}$ when $k$ is constant. The derivative of the constant $E$ is zero. Therefore:

$$
\frac{d\widehat L_C}{dN}
=-\alpha A N^{-\alpha-1}
+\beta B\left(\frac6C\right)^\beta N^{\beta-1}.
$$

At the optimum, increasing $N$ a little provides exactly as much benefit from capacity as it costs through reduced training tokens. Set the derivative to zero and move the negative term to the other side:

$$
\alpha A N^{-\alpha-1}
=\beta B\left(\frac6C\right)^\beta N^{\beta-1}.
$$

Multiplying both sides by $N$ makes the meaning easier to see:

$$
\boxed{\alpha\frac{A}{N^\alpha}
=\beta\frac{B}{D^\beta}.}
$$

**The exponent-weighted penalties balance.** The penalties themselves are equal only when $\alpha=\beta$. For the toy optimum $(8,32)$, each penalty is about 1.414, and each weighted penalty is $0.5\times1.414\approx0.707$. These are the two opposing effects that cancel in the derivative.

### 3.7 Solve that balance for the best model size and tokens

Starting from the derivative equation, multiply both sides by $N^{\alpha+1}$:

$$
\alpha A
=\beta B\left(\frac6C\right)^\beta N^{\alpha+\beta}.
$$

Divide by the constant on the right:

$$
N^{\alpha+\beta}
=\frac{\alpha A}{\beta B}\left(\frac C6\right)^\beta.
$$

Finally take the power $1/(\alpha+\beta)$ of both sides. A star means **the optimal choice**:

$$
\boxed{
N^{*}=
\left[\frac{\alpha A}{\beta B}
\left(\frac C6\right)^\beta\right]^{1/(\alpha+\beta)},
\qquad
D^{*}=\frac{C}{6N^{*}}.
}
$$

Check every step on the toy problem: $\alpha+\beta=1$, $\alpha A/(\beta B)=2/4=1/2$, and $(C/6)^\beta=\sqrt{256}=16$. Thus $N^{*}=[(1/2)\times16]^1=8$ and $D^{*}=256/8=32$, matching the table.

Why is this a minimum? With positive constants, the model penalty grows without bound as $N$ approaches zero; the token penalty grows without bound as $N$ becomes very large at fixed $C$. The equation above has exactly one positive solution, where the derivative changes from negative to positive. This establishes the minimum **of the fitted formula**; its accuracy for actual models still depends on the fit.

### 3.8 Where does "scale model and data together" come from?

The boxed formula chooses a configuration at one budget. Now imagine comparing the best configurations at several increasing budgets, keeping the fitted constants fixed.

The power of $C$ in $N^{*}$ is $\beta/(\alpha+\beta)$. Since $D^{*}=C/(6N^{*})$, its power is $1-\beta/(\alpha+\beta)=\alpha/(\alpha+\beta)$. Write:

$$
N^{*}\propto C^a,\qquad D^{*}\propto C^b,
\qquad
a=\frac{\beta}{\alpha+\beta},\qquad
b=\frac{\alpha}{\alpha+\beta}.
$$

Here $\propto$ means "is proportional to," leaving out a fixed multiplier. **$\alpha,\beta$ describe how penalties fall; $a,b$ describe how the optimal resource amounts grow with compute.** We calculate $a,b$ from the fitted exponents, rather than fitting five constants and then arbitrarily choosing two more.

Using this lesson's revised fit gives $a=0.3658/0.7136\approx0.513$ and $b=0.3478/0.7136\approx0.487$. Both are near $1/2$. With exactly $a=b=1/2$, four times the compute means twice the model size and twice the training tokens: $4^{1/2}=2$, and $2\times2=4$.

The original paper's parametric approach reported $a\approx0.46$, $b\approx0.54$; these differ from the revised fit used here, while still suggesting approximately equal scaling. See [Hoffmann et al., equation 4 and Table 2](https://arxiv.org/html/2203.15556v1#S3.SS4).

The exponents must sum to one because $N^{*}D^{*}=C/6$. For the revised fit, $0.513+0.487=1$. That requirement alone would also permit $a=0.9,b=0.1$; the **measurements** are what put both exponents near one half.

At one fixed budget, a bigger model forces fewer tokens. Across growing budgets, both optimal quantities can grow. Keep those two comparisons separate.

### 3.9 Where does the specific ratio of about 20 come from?

Near-equal scaling rates do not determine the actual ratio. To see this, suppose for a moment that $\alpha=\beta=p$. The optimum's balance condition becomes:

$$
\frac{A}{(N^{*})^p}=\frac{B}{(D^{*})^p}
\quad\Longrightarrow\quad
\frac{D^{*}}{N^{*}}=\left(\frac BA\right)^{1/p}.
$$

Our toy fit has $p=0.5$, $A=4$ and $B=8$, so its optimal ratio is $(8/4)^2=4$ tokens per parameter, matching $32/8=4$. It has equal scaling, but a ratio of four. **The coefficient values matter as well as the exponents.**

For the actual revised fit, $\alpha$ and $\beta$ are close but unequal, so the ratio varies slowly with the budget. Section 4 substitutes the fitted constants and $C=10^{21}$ into the boxed formula. It gives about 2.78B parameters and 60B tokens: $60/2.78\approx21.6$ tokens per parameter. We did not insert 20 anywhere in that calculation.

The original Chinchilla model used 70B parameters and 1.4T tokens, giving $1400/70=20$. The paper also examined training-curve minima and fixed-compute loss curves, alongside its parametric fit. These supported its approximate equal-scaling recommendation. See [Hoffmann et al., sections 3.1–3.4 and Table 1](https://arxiv.org/html/2203.15556v1#S3).

> **The complete chain:** train models with different $N,D$ → measure their losses → fit $E,A,B,\alpha,\beta$ → impose $C=6ND$ → minimise predicted loss → obtain $N^{*},D^{*}$ → calculate their ratio. "About 20 tokens per parameter" is the practical headline; the fit and optimisation explain how we get it and why it is approximate.

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

<div class="callout warn"><p><strong>Compute-optimal to <em>train</em> ≠ optimal to <em>serve</em>.</strong> Chinchilla minimises <em>training</em> loss per training FLOP. But a model is trained once and served billions of times, so it is often worth "over-training" a <em>smaller</em> model far past its Chinchilla-optimal token count: it costs more to train but is cheaper to run forever. LLaMA (paper #9) did exactly this — 7B–65B models trained on ~1–1.4T tokens, well beyond 20×. Chinchilla tells you the training optimum; inference economics can rationally pull you away from it.</p>
<p><strong>Why leave the 20× rule?</strong> Because the 20× rule answers a narrower question than the one a lab actually faces. It answers: <em>"I have $C$ FLOPs for training; which $N$ and $D$ give the lowest loss?"</em> The cost it minimises is training compute, $6ND$, and nothing else. A deployed model also costs compute every time it generates a token: one forward pass, $\approx 2N$ FLOPs per token. If the model will serve $T$ tokens over its lifetime, the bill you really pay is</p>
<p>$$\text{total cost} \;=\; \underbrace{6ND}_{\text{train once}} \;+\; \underbrace{2N\,T}_{\text{serve forever}}.$$</p>
<p>The serving term contains $N$ but not $D$: how many tokens the model was trained on leaves no trace in the finished weights, so it does not change the cost of running it. To reduce the serving term you must make $N$ smaller. To keep the same quality with a smaller $N$, the loss formula $E + A/N^\alpha + B/D^\beta$ says you must make $D$ bigger to compensate. A smaller model fed more tokens is exactly "over-training", and its tokens-per-parameter ratio rises above 20. The 20× rule is just the special case $T = 0$, a model nobody ever uses.</p>
<p><strong>How far does it move?</strong> Keep the quality fixed at the §4 optimum ($L = 2.31$, i.e. $N = 2.78\text{B}$, $D = 60\text{B}$ for $C = 10^{21}$), and for each expected lifetime serving volume $T$ pick the $N$ (and the $D$ it then needs) that minimises total cost. With the §4 constants:</p>
<table>
<tr><th>Tokens served $T$</th><th>Best $N$</th><th>Its $D$</th><th>Tokens / param</th><th>Total cost saved vs 20× model</th></tr>
<tr><td>0 (never served)</td><td>2.78B</td><td>60B</td><td>22</td><td>0%</td></tr>
<tr><td>$10^{11}$</td><td>1.88B</td><td>95B</td><td>50</td><td>7%</td></tr>
<tr><td>$10^{12}$</td><td>1.09B</td><td>249B</td><td>230</td><td>42%</td></tr>
<tr><td>$10^{13}$</td><td>0.71B</td><td>967B</td><td>~1,400</td><td>68%</td></tr>
</table>
<p>At $T = 10^{12}$, for example, the 1.09B model spends $1.63\times10^{21}$ FLOPs on training (63% more than the 20× model) but half as much on every served token, so its total is $3.8\times10^{21}$ against $6.6\times10^{21}$. The more a model will be used, the smaller and longer-trained it should be. That is why production models such as LLaMA sit at 100×–1,000×+ tokens per parameter: they are compute-optimal for <em>training plus serving</em>, not for training alone. This inference-aware version of the Chinchilla analysis is worked out in <a href="https://arxiv.org/abs/2401.00448">Sardana et al. 2023, "Beyond Chinchilla-Optimal"</a>. Two caveats: the cost model counts FLOPs only (real serving also pays for memory and latency, which also favour a smaller $N$), and the loss law is fitted near 20×, so the extreme rows are extrapolations.</p></div>

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
