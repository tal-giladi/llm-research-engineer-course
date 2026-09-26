# 19.2 · Experiment design & research writing

<div class="prereq">
<p><strong>Prerequisites:</strong> honest held-out evaluation, loss, and perplexity from <a href="#/lessons/module-13/lesson-01">13.1 · Loss, perplexity, zero-/few-shot</a>; fitting a trend and reasoning about noise from <a href="#/lessons/module-10/lesson-02">10.2 · Kaplan scaling laws</a>; mean/variance from <a href="#/lessons/module-01/lesson-01">01.1 · Random variables, expectation, variance</a>. The ReAct loop from <a href="#/lessons/module-19/lesson-01">19.1</a> for context on the agent you might use to run sweeps.</p>
<p><strong>You will learn:</strong> how to turn a vague idea into a falsifiable experiment; hypotheses, baselines, controls, and ablations; why a single-seed result is untrustworthy and how to report <strong>mean ± std</strong> with a <strong>confidence interval</strong>; checkpoint selection, hyperparameter sweeps, and experiment tracking; failure analysis, paper reproduction, and the value of negative results; and the discipline of scientific writing via a fixed <strong>experiment-report template</strong>.</p>
<p><strong>Why this matters for ML:</strong> this is the core of the job. A research engineer's output is not code or checkpoints — it is <em>trustworthy knowledge</em>: "does change X actually help, and by how much, and are we sure?" Most wrong conclusions in ML come not from bad models but from bad experimental hygiene — one seed, a moving baseline, a metric that drifted. Getting this right is what separates real progress from fooling yourself.</p>
</div>

## 1. Intuition: an experiment is a question with a fair test

You have an idea: "RMSNorm will train better than LayerNorm at this scale." That is not yet an experiment. An experiment is a *specific, falsifiable question* plus a *fair test* that could return "no." The entire discipline is about making the test fair — so that when you see a difference, it is caused by the thing you changed and nothing else, and it is larger than the noise.

Three failure modes to design against from the start:

- **Confounding** — you changed two things at once (norm *and* learning rate), so you cannot attribute the effect to either.
- **Noise** — the difference you see is smaller than the run-to-run variation from random seeds, so it means nothing.
- **A moving target** — the baseline, the data split, or the metric changed between runs, so the comparison is not like-for-like.

<div class="callout key"><p>An experiment answers <strong>one</strong> falsifiable question by changing <strong>one</strong> variable against a <strong>fixed</strong> baseline, and reports the result with enough replication to tell signal from noise. If you changed two things, you have no experiment — you have a story.</p></div>

## 2. The vocabulary of a clean experiment

- **Hypothesis** — what you expect and *why* (the mechanism), stated *before* you look at results. "RMSNorm removes the mean-centering step, which is cheap and, prior work suggests, harmless to quality; I expect equal or slightly better validation loss at lower cost." A hypothesis with a mechanism is falsifiable in a useful way.
- **Baseline** — the reference you compare against, run under *identical* conditions. Without a baseline a number is meaningless: "loss 3.1" is neither good nor bad until you know what the unmodified system scores.
- **Independent variable** — the *one* thing you change (the norm layer).
- **Controls** — everything you deliberately hold fixed (data, tokens, steps, batch size, learning-rate schedule, seed set, hardware). Controls are what make the comparison fair.
- **Ablation study** — remove or swap one component at a time to measure *its* contribution. "The model works" is not science; "removing component C costs 0.4 loss, removing D costs nothing" is. An ablation is a series of experiments that each change one part.

<div class="callout warn"><p>The cardinal sin is changing more than one variable per run. If you swap the norm <em>and</em> bump the learning rate and loss improves, you have learned nothing about the norm. Every confound is a conclusion you cannot draw.</p></div>

## 3. The reason single-seed results lie

Neural-net training is stochastic: the weight initialization, the data shuffling, and (on GPU) some kernels all depend on a random **seed**. Re-run the *exact same config* with a different seed and you get a *different* final loss. A single run is therefore a sample of size one from a distribution — you have no idea how wide that distribution is.

### 3.1 Mathematics: mean and confidence interval

Run a config with $n$ seeds, getting metric values $x_1, \dots, x_n$. Report two things.

The **sample mean** and **sample standard deviation**:

$$
\bar{x} = \frac{1}{n}\sum_{i=1}^{n} x_i,
\qquad
s = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2}.
$$

The divisor is $n-1$, not $n$ (**Bessel's correction**): with a handful of seeds you are *estimating* the spread from a small sample, and dividing by $n-1$ removes the bias that $n$ would introduce. Here $s$ measures how much a single run bounces around.

Better than $\bar x \pm s$, report a **confidence interval for the mean** — the range that, at a stated confidence (say 95%), plausibly contains the *true* mean you would get with infinite seeds:

$$
\bar{x} \pm t_{1-\alpha/2,\; n-1} \cdot \frac{s}{\sqrt{n}}.
$$

Here $t_{1-\alpha/2,\,n-1}$ is the two-sided **Student-t critical value** at $n-1$ degrees of freedom (the t-distribution, not the normal, because $n$ is tiny and $s$ is itself estimated), and $s/\sqrt{n}$ is the **standard error of the mean** — the spread shrinks like $\sqrt{n}$, which is exactly why more seeds tighten the interval. Two configs whose confidence intervals **overlap** are not distinguishable at that confidence: the difference is within noise.

### 3.2 Numerical example, worked by hand

Compare two configs, a **baseline A** and a **variant B**, each run for 3 seeds. Validation accuracy (%):

| seed | A (baseline) | B (variant) |
| --- | --- | --- |
| 0 | 72.6 | 71.4 |
| 1 | 70.1 | 72.9 |
| 2 | 70.7 | 71.7 |

Look only at **seed 0** and you would conclude *A wins* by 1.2 points. Now compute the statistics for A by hand. The mean:

$$
\bar{x}_A = \frac{72.6 + 70.1 + 70.7}{3} = \frac{213.4}{3} = 71.1333.
$$

Deviations from the mean: $+1.4667,\ -1.0333,\ -0.4333$. Squared and summed: $2.1511 + 1.0678 + 0.1878 = 3.4067$. Divide by $n-1 = 2$ and take the root:

$$
s_A = \sqrt{\tfrac{3.4067}{2}} = \sqrt{1.7033} = 1.3051.
$$

For a 95% interval at $n-1 = 2$ degrees of freedom, $t_{0.975,\,2} = 4.303$, so the margin is

$$
4.303 \cdot \frac{1.3051}{\sqrt{3}} = 3.2424,
\qquad \text{CI}_A = (67.89,\ 74.38).
$$

Doing the same for B gives $\bar{x}_B = 72.00$, $s_B = 0.7937$, $\text{CI}_B = (70.03,\ 73.97)$. The full picture:

| config | seed 0 | mean ± std | 95% CI |
| --- | --- | --- | --- |
| A (baseline) | 72.6 | 71.13 ± 1.31 | (67.89, 74.38) |
| B (variant) | 71.4 | 72.00 ± 0.79 | (70.03, 73.97) |

Two lessons jump out. First, the **single-seed conclusion flips**: seed 0 said A wins by 1.2, but the *mean over three seeds* says B wins by 0.87. A one-seed result pointed you the wrong way. Second, the **confidence intervals overlap almost entirely** (67.9–74.4 vs 70.0–74.0), so even the three-seed difference of 0.87 is well inside the noise — the honest conclusion is *"no detectable difference at 3 seeds; run more seeds if 0.87 points matters."* Reporting "B is better (72.0 vs 71.1)" without the spread would be a false claim.

<div class="callout key"><p>Never report a single-seed number as a result. Report <strong>mean ± std over ≥ 3 seeds</strong> and a confidence interval, and only claim a difference when the intervals separate. A result you cannot reproduce across seeds is not a result.</p></div>

### 3.3 From-scratch code

`code/src/llmre/agents/stats.py` gives you exactly these two computations, with no SciPy dependency (a small built-in t-table plus a normal fallback):

```python
from llmre.agents.stats import mean_std, confidence_interval

A = [72.6, 70.1, 70.7]
B = [71.4, 72.9, 71.7]

print(mean_std(A))                    # (71.1333..., 1.3051...)
print(confidence_interval(A, 0.95))   # (67.891..., 74.376...)
print(mean_std(B))                    # (72.0, 0.7937...)
print(confidence_interval(B, 0.95))   # (70.028..., 73.972...)
```

`mean_std` uses `ddof=1` (Bessel) and returns `0.0` std for a single value; `confidence_interval` uses the Student-t critical value for $n-1$ degrees of freedom and collapses to `(mean, mean)` for a single value — an explicit signal that one seed carries no uncertainty estimate. These are the functions the tests in `code/tests/test_agents.py` verify against the hand computation above.

<div class="callout pt"><p>Why a t-table and not just the normal <em>z</em>? With $n = 3$ seeds you have only 2 degrees of freedom and you are estimating $s$ from the same tiny sample, so the sampling distribution of the mean has fatter tails than a normal. $t_{0.975,2} = 4.303$ is far larger than the normal $z_{0.975} = 1.96$ — using $z$ here would report an interval less than half as wide as the truth and make you overconfident. The t-distribution is the correct small-sample choice; it converges to the normal as $n$ grows.</p></div>

## 4. Selecting checkpoints, sweeps, and tracking

**Checkpoint selection.** Training saves the model periodically. You do not report the *last* checkpoint — you report the one with the best *validation* metric (Module 13), chosen on a validation set and, ideally, confirmed on a held-out *test* set you touch once. Picking the checkpoint that happens to look best on the test set is a subtle form of overfitting to it.

**Hyperparameter sweeps.** To find a good learning rate you run a **sweep** — the same config at several learning rates — and plot metric vs hyperparameter. Two rules: sweep on the *validation* set, and beware that the winner of a sweep is partly luck (you took the max over several noisy runs), so the sweep's best number is optimistically biased. Re-run the chosen setting with fresh seeds before believing it.

**Experiment tracking.** Every run must be reproducible from its record: the exact config, the code commit hash, the data version, the seed, and the resulting metrics. A run you cannot reproduce is a run you cannot cite. This is why the report template below demands the commit and config, and why the paper-reading workflow files a summary per paper under `code/reports/` (see the <a href="#/papers/index">paper curriculum</a>).

## 5. Failure analysis, reproduction, and negative results

**Failure analysis** is where most of the learning is. When a run underperforms, do not just tweak and rerun — *look at the failures*. Which examples does the model get wrong? Is the loss curve spiking (a data or learning-rate problem) or plateauing (a capacity or data problem)? A histogram of per-example losses or a handful of read wrong outputs teaches more than another sweep.

**Paper reproduction** is the skill that makes you trusted. Reproducing a result means matching a published number under stated conditions — and it usually *fails the first time*, which surfaces the undocumented details (data preprocessing, tokenizer, exact schedule) that matter. Treat a reproduction like any experiment: baseline, controls, seeds, a report.

**Negative results are results.** "RMSNorm did not help at this scale (Δ = 0.87 ± overlapping CIs)" is genuine, publishable knowledge that saves the next person a week. The pressure to report only wins is how fields accumulate false beliefs. Write the negative result up with the same rigor as a positive one — a clean null with tight intervals is far more valuable than a noisy win.

## 6. The experiment-report template

Every experiment gets a written report, filed under `code/reports/`. A fixed template forces you to state the question before the answer and the limitations alongside the conclusion — the structure *is* the discipline. The full template lives at `code/reports/EXPERIMENT_TEMPLATE.md`; its ten sections:

- **QUESTION** — the single, specific, falsifiable question this experiment answers.
- **HYPOTHESIS** — what you expect and the mechanism why, stated before results.
- **METHOD** — exactly what you did: model, data, tokens, steps, hardware; enough to reproduce.
- **BASELINE** — the reference point, run under identical conditions.
- **VARIABLES** — the one independent variable and the controls held fixed.
- **RESULTS** — the numbers as mean ± std over N seeds, with a confidence interval.
- **ANALYSIS** — what the numbers mean; is the difference larger than seed noise?
- **LIMITATIONS** — what this does *not* show: confounds, untested scales, unmeasured metrics.
- **CONCLUSION** — the one-line answer to the QUESTION, qualified by analysis and limitations.
- **NEXT EXPERIMENT** — the most informative follow-up this result motivates.

<div class="callout key"><p>Write the QUESTION and HYPOTHESIS <em>before</em> you run anything. Filling RESULTS and CONCLUSION first, then back-writing a question that fits, is how you fool yourself — it converts a random finding into a "prediction" you never actually made.</p></div>

<div class="hw">
<p><strong>Hardware track.</strong> The statistics in this lesson are CPU-only and instantaneous. The <em>experiments</em> they analyze are not: a fair 3-seed comparison of two configs means running training six times. Budget accordingly — replication multiplies your compute by the number of seeds, which is exactly why you fix the seed count and the sweep grid up front rather than re-running ad hoc.</p>
</div>

## Exercise

You run config A for 5 seeds and get validation losses `[3.10, 3.14, 3.09, 3.12, 3.11]`, and a variant B for 5 seeds and get `[3.05, 3.07, 3.06, 3.08, 3.04]`. Using `mean_std` and `confidence_interval`, decide whether B is genuinely better than A at 95% confidence.

<p><em>Optional hint:</em> compute both means and both 95% confidence intervals, then check whether the intervals overlap.</p>

<p><em>Stronger hint:</em> B is better only if its interval sits <em>entirely below</em> A's — any overlap means "not distinguishable at this confidence." These spreads are much tighter than the §3.2 example, so the outcome may differ.</p>

<details><summary>Solution</summary>

```python
from llmre.agents.stats import mean_std, confidence_interval
A = [3.10, 3.14, 3.09, 3.12, 3.11]
B = [3.05, 3.07, 3.06, 3.08, 3.04]
print(mean_std(A), confidence_interval(A))  # ~ (3.112, 0.0192), (3.088, 3.136)
print(mean_std(B), confidence_interval(B))  # ~ (3.060, 0.0158), (3.040, 3.080)
```

A: mean 3.112, 95% CI ≈ (3.088, 3.136). B: mean 3.060, 95% CI ≈ (3.040, 3.080). B's entire interval (up to 3.080) sits **below** A's entire interval (down to 3.088) — they do **not** overlap. So here, unlike the §3.2 example, you *can* claim B is genuinely better at 95% confidence: the ~0.05-nat improvement is larger than the seed noise. The difference from §3.2 is that these runs are far more consistent across seeds (std ≈ 0.017 vs ≈ 1.0), which tightens the intervals enough to separate them. Lesson: whether a gap is "real" depends on the gap *and* the spread — you must compute both.

</details>

## Common mistakes

- **Reporting one seed.** The most common and most damaging error. One run tells you nothing about variance; the §3.2 example shows a single seed pointing to the *wrong* winner.
- **Overlapping CIs reported as a win.** If the intervals overlap, "B is better" is false. Say "no detectable difference" and, if you care about the gap, add seeds.
- **A moving baseline.** Re-running the baseline with a newer data split or code and comparing to an old variant number. Baseline and variant must run under identical, current conditions.
- **Tuning on the test set.** Selecting the checkpoint or hyperparameter that looks best on the set you will report — that set is now contaminated. Select on validation, report on a test set touched once.
- **Writing the conclusion first.** Deciding the answer, then choosing the metric, seeds, or checkpoint that supports it. Fix the protocol before you look.

## Debugging exercise

A colleague reports: "Variant B improves accuracy from 71.1% to 72.0%, so we should ship it," citing this code. What is wrong with the claim, and what one change to the code would reveal it?

```python
acc_A = run_training(config="A", seed=0).eval_accuracy   # 71.1
acc_B = run_training(config="B", seed=0).eval_accuracy   # 72.0
print(f"B improves accuracy by {acc_B - acc_A:.1f} points")
```

<details><summary>Answer</summary>

Both numbers come from **a single seed each** (`seed=0`), so the 0.9-point "improvement" could be entirely seed noise — exactly the situation in §3.2, where a single seed even pointed to the wrong winner. The claim is unsupported. The fix: run each config over several seeds and compare with confidence intervals:

```python
from llmre.agents.stats import mean_std, confidence_interval
A = [run_training(config="A", seed=s).eval_accuracy for s in range(5)]
B = [run_training(config="B", seed=s).eval_accuracy for s in range(5)]
print(mean_std(A), confidence_interval(A))
print(mean_std(B), confidence_interval(B))
# claim "B is better" only if B's CI sits clearly above A's
```

If the intervals overlap, the honest report is "no detectable difference at 5 seeds," not "ship it."

</details>

## Check yourself

<details><summary>Why divide by <code>n - 1</code> instead of <code>n</code> when computing the standard deviation of seed results?</summary>

Bessel's correction. With a small sample you are *estimating* the population spread, and using the sample mean (itself computed from the data) makes the sum of squared deviations systematically too small; dividing by $n-1$ instead of $n$ removes that bias. It matters most exactly when it bites hardest — tiny $n$, which is the seed-count regime.

</details>

<details><summary>Two configs have 95% confidence intervals (67.9, 74.4) and (70.0, 74.0). Can you claim one is better?</summary>

No. The intervals overlap heavily, so at 95% confidence the two are indistinguishable — the observed mean difference is within the seed noise. The honest statement is "no detectable difference at this seed count"; if the gap matters, run more seeds to tighten the intervals and see whether they separate.

</details>

<details><summary>Why report a confidence interval for the mean rather than just <code>mean ± std</code>?</summary>

`± std` describes how much a *single run* varies. The confidence interval describes how well you know the *true mean* — it uses the standard error $s/\sqrt{n}$, which shrinks as you add seeds, and the t critical value for the small-sample uncertainty. It is the interval that answers "could the difference between two configs be zero?", which is the question you actually care about when comparing them.

</details>

<details><summary>Why does the template make you write the QUESTION and HYPOTHESIS before running anything?</summary>

To prevent HARKing — hypothesizing after the results are known. If you look at the data first, you will always find *some* pattern and can back-write a question it "confirms," turning noise into a fake discovery. Committing to the question and predicted mechanism up front keeps the test honest and makes a negative result meaningful.

</details>

<details><summary>You did a learning-rate sweep and the best run scored 3.05. Why should you not report 3.05 as the config's performance?</summary>

Taking the maximum (or minimum-loss) over several noisy runs is optimistically biased: the winner is partly good luck, so its number overstates the true expected performance. Re-run the chosen learning rate with fresh seeds and report *that* mean ± CI — the sweep chooses the setting, but an independent replication estimates its performance.

</details>

## Next

You now have the full research-engineer toolkit: from-scratch models and training (Modules 0–14), preference learning and reasoning (15–17), tools and agents (18–19), and — here — the experimental discipline to tell whether any change to all of that actually helps. The capstone puts it together: design, train, evaluate, and *write up* a mini frontier LLM end to end, with every claim backed by an experiment report.

Continue to [20 · Mini frontier LLM](lessons/module-20/lesson-01.md).
