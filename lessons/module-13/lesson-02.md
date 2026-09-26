# 13.2 · Contamination, calibration, task eval

<div class="prereq">
<p><strong>Prerequisites:</strong> loss, perplexity, and log-likelihood multiple-choice scoring from <a href="#/lessons/module-13/lesson-01">13.1 · Loss, perplexity, zero-/few-shot</a>; benchmark contamination and near-duplicate detection from <a href="#/lessons/module-11/lesson-03">11.3 · Contamination, mixing, curriculum</a> and <a href="#/lessons/module-11/lesson-02">11.2 · Deduplication &amp; MinHash</a>; softmax confidence from <a href="#/lessons/module-01/lesson-04">01.4</a>.</p>
<p><strong>You will learn:</strong> how benchmark <strong>contamination</strong> silently inflates scores and how to reason about it; what <strong>calibration</strong> is — whether a model's confidence matches its accuracy — and how to compute the <strong>Expected Calibration Error (ECE)</strong> on a small bucketed example by hand; a map of the task-evaluation families you will meet later in the course (instruction-following, reasoning, tool-use, human/preference); and <strong>pass@k</strong> for code and reasoning, including the unbiased estimator and why the naive one is biased.</p>
<p><strong>Why this matters for ML:</strong> a single accuracy number hides more than it shows. A contaminated benchmark makes a mediocre model look brilliant; a miscalibrated model gives confident wrong answers that are dangerous to trust; and the wrong metric can reward the wrong behavior. Knowing what each number does and does <em>not</em> measure is the core skill of an evaluation engineer.</p>
</div>

## 1. Contamination: when the test leaks into training

The premise of every benchmark is that the model has never seen the test questions. Web-scraped pretraining corpora quietly break that premise. Benchmarks are published on the web — on GitHub, in papers, on leaderboards, in blog posts discussing them — and Common Crawl scoops all of that up. So the exact questions, and often their answers, end up in the training data. This is **benchmark contamination**, and it is the evaluation-side twin of the training-set deduplication problem from <a href="#/lessons/module-11/lesson-02">11.2</a>.

A contaminated benchmark does not measure task ability; it measures **memorization of the test set**. The model can score high on MMLU or GSM8K not because it reasons, but because it saw those exact items during pretraining and is recalling them. The inflated score transfers to nothing.

<div class="callout key"><p>Contamination inflates scores <strong>upward</strong> and silently. There is no error message — a contaminated model simply reports a higher number than it deserves, and you only discover it when the model fails on genuinely novel inputs that "should" have been easy given the benchmark.</p></div>

How practitioners detect and limit it (all of these are imperfect):

- **N-gram / substring overlap.** Flag any test item whose text appears (as a long enough n-gram) in the training corpus, and either remove those items from the benchmark or remove the documents from training. This is the standard decontamination pass; GPT-3, Llama, and others report doing it.
- **Near-duplicate detection.** Exact-match misses paraphrases and reformattings. The MinHash / Jaccard machinery from <a href="#/lessons/module-11/lesson-02">11.2</a> catches items that are close but not identical.
- **Canary strings.** Benchmark authors embed a unique random string ("canary GUID") in the released files so that anyone can grep their corpus for it and know the benchmark leaked in.
- **Held-out / freshly-collected test sets.** The most robust defense is a test set created *after* the training cutoff, or kept entirely private, so it cannot have been seen. This is why leaderboards increasingly rotate to new, time-stamped question sets.

<div class="callout warn"><p>Decontamination is best-effort, not a guarantee. Paraphrases, translations, reordered multiple-choice options, and reformatted problems slip past n-gram filters. Treat a headline benchmark number from a web-trained model as an <em>upper bound</em> on true ability, and trust a private or post-cutoff evaluation far more than a public one.</p></div>

<div class="callout paper"><p>The Llama and GPT-3 papers (see the <a href="#/papers/index">paper curriculum</a>) both document their decontamination procedures and report how much scores move once contaminated items are removed — worth reading for how seriously frontier labs treat this, and how much a "clean" number can differ from the raw one.</p></div>

## 2. Calibration: does confidence match accuracy?

Accuracy asks *how often is the model right?* Calibration asks a different, orthogonal question: *when the model says it is 80% sure, is it actually right about 80% of the time?* A model is **well-calibrated** if its stated confidence matches its empirical accuracy at every confidence level.

The model's confidence in a prediction is naturally read off the softmax: after choosing the most likely class, its **confidence** is the probability mass it put on that chosen class (the max softmax probability). Calibration compares that number to how often such predictions are correct.

The two properties are independent:

- A model can be **accurate but overconfident** — 70% correct, but claiming 99% sure every time.
- A model can be **inaccurate but honest** — 40% correct, and reporting ~40% confidence.

For anything where a human or a downstream system acts on the model's probability — flagging low-confidence answers for review, abstaining, routing to a human — calibration matters as much as accuracy. A confident wrong answer is worse than an unconfident one.

### 2.1 Expected Calibration Error (ECE)

ECE turns calibration into one number. The recipe:

1. Take $N$ predictions, each with a **confidence** $c_i \in [0,1]$ (the max softmax probability) and a **correctness** $y_i \in \{0,1\}$.
2. Sort them into $M$ equal-width confidence **bins** across $[0,1]$ (e.g. $M=10$: $(0.0,0.1], (0.1,0.2], \dots$).
3. In each bin $b$ compute the **accuracy** $\mathrm{acc}(b)$ (fraction correct among predictions in the bin) and the **average confidence** $\mathrm{conf}(b)$.
4. ECE is the average gap $|\mathrm{acc}(b) - \mathrm{conf}(b)|$, weighted by how many predictions fall in each bin:

$$
\mathrm{ECE} = \sum_{b=1}^{M} \frac{|B_b|}{N}\,\bigl|\,\mathrm{acc}(b) - \mathrm{conf}(b)\,\bigr|,
$$

where $|B_b|$ is the number of predictions in bin $b$. A perfectly calibrated model has $\mathrm{acc}(b) = \mathrm{conf}(b)$ in every bin, so $\mathrm{ECE} = 0$.

### 2.2 Worked example — ECE by hand

Ten predictions. Confidences and correctness (verified in Python):

| confidence $c_i$ | correct $y_i$ |
| --- | --- |
| 0.6, 0.6, 0.6, 0.6, 0.6 | 1, 1, 1, 0, 0 |
| 0.8, 0.8, 0.8, 0.8 | 1, 1, 0, 0 |
| 0.95 | 1 |

Sort into width-$0.1$ bins. Only three bins are populated:

- **Bin $(0.5, 0.6]$** — 5 predictions, all confidence $0.6$; 3 of 5 correct. $\mathrm{conf}=0.60$, $\mathrm{acc}=0.60$, gap $=0.00$, weight $=5/10=0.50$.
- **Bin $(0.7, 0.8]$** — 4 predictions, all confidence $0.8$; 2 of 4 correct. $\mathrm{conf}=0.80$, $\mathrm{acc}=0.50$, gap $=0.30$, weight $=4/10=0.40$.
- **Bin $(0.9, 1.0]$** — 1 prediction, confidence $0.95$; correct. $\mathrm{conf}=0.95$, $\mathrm{acc}=1.00$, gap $=0.05$, weight $=1/10=0.10$.

Weighted sum:

$$
\mathrm{ECE} = 0.50\cdot 0.00 + 0.40\cdot 0.30 + 0.10\cdot 0.05 = 0 + 0.12 + 0.005 = 0.125.
$$

The middle bin dominates: the model claims 80% confidence there but is right only half the time — overconfident. The first bin is perfectly calibrated (contributes 0); the last bin is slightly under-confident but rare, so it barely moves the total. The final ECE of $0.125$ means the model's confidence is off by about 12.5 percentage points on average.

<div class="callout warn"><p>ECE is a useful summary but a crude one. It depends on the number of bins (too few hides miscalibration, too many leaves bins nearly empty and noisy); it can read 0 for a badly wrong model whose over- and under-confident bins happen to cancel in the weighting; and it says nothing about <em>which direction</em> the model errs. Report it alongside a reliability diagram (accuracy vs confidence per bin), not on its own.</p></div>

### 2.3 In code

`expected_calibration_error(confidences, correct, n_bins=10)` in `llmre.evaluation.calibration` implements exactly section 2.1, and `code/tests/test_eval.py` checks the two anchor cases: it returns exactly $0$ for perfectly-calibrated synthetic data (e.g. 100 predictions at confidence $0.5$ with exactly half correct) and a positive value for miscalibrated data (100 predictions at confidence $0.9$ that are all wrong give $\mathrm{ECE}=0.9$).

```python
from llmre.evaluation.calibration import expected_calibration_error

confidences = [0.6]*5 + [0.8]*4 + [0.95]
correct     = [1,1,1,0,0] + [1,1,0,0] + [1]
expected_calibration_error(confidences, correct, n_bins=10)   # -> 0.125
```

## 3. A map of task-evaluation families

Perplexity and multiple-choice log-likelihood are only the start. As the course moves from a pretrained base model into fine-tuning and beyond, each stage has its own evaluation discipline. Here is the map, with forward links to where each is developed.

- **Instruction-following evaluation.** After supervised fine-tuning, you ask whether the model does what an instruction says. This is judged by held-out instruction sets and increasingly by a strong model acting as a judge (LLM-as-judge), which is fast but inherits the judge's biases. Developed in [Module 14](lessons/module-14/lesson-01.md).
- **Reasoning evaluation.** For math and logic with a single verifiable answer (GSM8K, MATH), you grade by **exact match** on the final answer — extract the model's answer and compare it to the gold answer. Objective and cheap, but blind to *how* the model got there (a right answer from wrong reasoning still scores). Developed in [Module 16](lessons/module-16/lesson-01.md).
- **Tool-use evaluation.** When the model calls tools/functions, you check whether it selected the right tool, formed valid arguments, and used the results correctly — often via task success in a sandboxed environment rather than a text match. Developed in [Module 18](lessons/module-18/lesson-01.md).
- **Human / preference evaluation.** For open-ended quality (helpfulness, style, safety) there is no single correct string. You collect **pairwise human preferences** — which of two responses is better — and aggregate them (e.g. win rates, Elo). This is the data that trains reward models and drives RLHF. Developed in [Module 15](lessons/module-15/lesson-01.md).

<div class="callout key"><p>The metric encodes what you reward. Exact-match rewards the final answer and ignores the reasoning; LLM-as-judge rewards whatever the judge prefers; preference win-rate rewards whatever annotators like. Choosing a metric is choosing an objective — pick the one that matches the behavior you actually want, and stay honest about what it leaves out.</p></div>

## 4. pass@k for code and reasoning

For tasks where an answer can be *checked* — code that must pass unit tests, math with a verifiable result — a model that is allowed several attempts is more useful than one judged on a single try. **pass@k** measures this: the probability that **at least one** of $k$ sampled attempts is correct.

The naive way to estimate it — sample $k$ times, score 1 if any passed — is a valid estimate but a noisy and **biased** one, especially for small $k$ and rare successes. The fix, from the Codex paper, is to sample a larger number $n \ge k$ of completions once, count how many are correct, and compute the expected pass@k **analytically**.

Let $n$ be the number of samples generated and $c$ the number of them that are correct. The **unbiased estimator** is

$$
\text{pass@}k = 1 - \frac{\dbinom{n-c}{k}}{\dbinom{n}{k}}.
$$

The fraction $\binom{n-c}{k} / \binom{n}{k}$ is the probability that a randomly chosen size-$k$ subset of the $n$ samples contains **none** of the $c$ correct ones (choosing all $k$ from the $n-c$ wrong ones); one minus that is the probability the subset contains at least one correct sample. Generating $n > k$ samples and using this formula gives a far lower-variance estimate than repeatedly drawing exactly $k$.

### 4.1 Worked example

Generate $n = 5$ samples for a problem; $c = 2$ turn out correct. Estimate pass@$k$ for $k = 3$ (verified in Python):

$$
\text{pass@}3 = 1 - \frac{\binom{5-2}{3}}{\binom{5}{3}} = 1 - \frac{\binom{3}{3}}{\binom{5}{3}} = 1 - \frac{1}{10} = 0.9.
$$

Reading it: of the $\binom{5}{3}=10$ ways to pick 3 of the 5 samples, only $\binom{3}{3}=1$ picks the 3 wrong ones and misses both correct samples; the other 9 contain at least one correct sample. So a random draw of 3 attempts succeeds 90% of the time.

<div class="callout warn"><p>pass@k needs a <strong>reliable verifier</strong>. It measures "can the model produce a passing answer within k tries", not "is the answer good" — a solution that passes weak unit tests but is wrong in an untested case still counts as a pass. A high pass@k with a leaky or incomplete checker overstates real capability, and comparisons are only meaningful when everyone uses the same n, the same k, and the same verifier.</p></div>

<div class="callout paper"><p>pass@k and its unbiased estimator come from the Codex paper, "Evaluating Large Language Models Trained on Code" (Chen et al., 2021). We return to verifiable rewards for reasoning in <a href="#/lessons/module-16/lesson-02">16.2 · RLVR &amp; verifiable rewards</a>, which reuses this same idea of a checkable answer.</p></div>

## 5. Staying honest about what a metric measures

Pulling the lesson together: every number you report has a blind spot, and naming it is part of the job.

- **Perplexity** measures token prediction on held-out text — not task ability, not truthfulness, not reasoning.
- **Multiple-choice accuracy** measures ranking among *given* options — not whether the model could produce the answer unaided, and it is sensitive to the scoring rule (§13.1) and to option ordering.
- **Exact-match** on reasoning rewards the final answer — a correct answer from broken reasoning still scores, and a correct answer phrased differently from the gold string wrongly fails.
- **pass@k** measures reachability within $k$ tries under a specific verifier — not solution quality.
- **A contaminated benchmark** measures memorization, not generalization.
- **ECE** summarizes calibration but can hide direction and cancel across bins.

None of these is wrong; each is partial. Report several, describe how each was computed (bins, normalization, $n$/$k$, decontamination), and keep a private or post-cutoff set for the number you actually trust.

## Exercise

You sample $n = 6$ completions of a coding problem and $c = 2$ pass the tests. Estimate pass@$2$ with the unbiased estimator.

<details><summary>Optional hint</summary>

pass@$k = 1 - \binom{n-c}{k}/\binom{n}{k}$. Here $n=6$, $c=2$, $k=2$, so $n-c = 4$.

</details>

<details><summary>Stronger hint</summary>

Compute $\binom{4}{2}$ and $\binom{6}{2}$, take the ratio, subtract from 1. $\binom{4}{2}=6$, $\binom{6}{2}=15$.

</details>

<details><summary>Solution</summary>

$$
\text{pass@}2 = 1 - \frac{\binom{4}{2}}{\binom{6}{2}} = 1 - \frac{6}{15} = 1 - 0.4 = 0.6.
$$

Of the 15 ways to pick 2 of the 6 samples, 6 pick both from the 4 failing samples (and thus miss every correct one); the remaining 9 include at least one of the 2 correct samples. So pass@2 $= 9/15 = 0.6$.

</details>

## Common mistakes

- **Trusting a public benchmark number from a web-trained model at face value** — assume some contamination and treat it as an upper bound.
- **Confusing accuracy with calibration** — a model can be very accurate and badly calibrated, or vice versa; they need separate measurement.
- **Reporting ECE with a single bin count** and no reliability diagram — the number is bin-dependent and can mask direction.
- **Estimating pass@k by literally sampling k and checking "any pass"** — biased and high-variance; sample $n > k$ and use the analytic estimator.
- **Grading reasoning by exact string match without normalizing** the answer format — "0.5", "1/2", and "½" are the same answer but fail a naive string compare.

## Debugging exercise

A pass@10 evaluation reports suspiciously high numbers. The code samples exactly 10 completions per problem and scores:

```python
passk = 1.0 if any(check(c) for c in completions) else 0.0   # averaged over problems
```

The team then also reports "pass@1" from the *same* 10 samples as `passk` above. What are the two problems?

<details><summary>Solution</summary>

1. **This is pass@10 estimated with n = k = 10 — the biased, high-variance estimator.** With only one draw of exactly $k$ samples per problem, the per-problem estimate is 0 or 1 and the average is noisy. The Codex estimator with a larger $n$ (e.g. sample $n=100$, report pass@10 analytically) is far more stable.

2. **"pass@1" cannot be read off the same "any of 10 passed" indicator.** That indicator *is* pass@10; reusing it as pass@1 reports the probability that one of ten attempts works as if it were the probability a single attempt works, massively overstating pass@1. pass@1 must be estimated as the fraction of individual samples that pass (i.e. $c/n$), or via the estimator with $k=1$: $1 - \binom{n-c}{1}/\binom{n}{1} = c/n$.

</details>

## Check yourself

<details><summary>A web-trained model scores 92% on a public multiple-choice benchmark but 61% on a privately-held set of freshly written questions of the same difficulty. What is the most likely explanation?</summary>

Contamination. The public benchmark almost certainly leaked into the pretraining corpus, so the 92% partly reflects memorization of those exact items. The private, post-hoc set could not have been seen, so 61% is the more honest estimate of true ability. The 31-point gap is the inflation contamination bought.

</details>

<details><summary>Model A is 85% accurate with ECE 0.20; model B is 80% accurate with ECE 0.02. Which would you deploy in a system that abstains when confidence is low?</summary>

Likely **B**. In an abstention system you act on the model's confidence, so calibration is critical: B's confidence is trustworthy (ECE 0.02), letting you reliably route its low-confidence cases to a human. A's 5-point accuracy edge is undermined by ECE 0.20 — its confidence is off by ~20 points, so it will confidently emit wrong answers and abstain on the wrong cases. (If the system never used confidence, A's higher accuracy would win — the right choice depends on how the number is used.)

</details>

<details><summary>In the ECE worked example, which bin contributed the most and why?</summary>

The $(0.7, 0.8]$ bin, contributing $0.40 \times 0.30 = 0.12$ of the total $0.125$. It held 4 predictions all claiming 0.80 confidence but only 50% correct — a 0.30 gap — and its weight (4/10) was large. It is the model's most overconfident, most populated region, so it dominates the ECE.

</details>

<details><summary>You sample n = 4 completions and 0 pass. What is pass@k for any k ≤ 4, and does the estimator agree with intuition?</summary>

With $c = 0$: $\text{pass@}k = 1 - \binom{n-0}{k}/\binom{n}{k} = 1 - \binom{n}{k}/\binom{n}{k} = 1 - 1 = 0$. Every size-$k$ subset consists entirely of failing samples, so none contains a correct one — pass@k is 0. This matches intuition: if none of your samples passed, no selection of them can pass.

</details>

<details><summary>Why is exact-match on the final answer a limited metric for reasoning tasks?</summary>

It scores only the final answer string, so it is blind to the reasoning path: a model that reaches the right number through invalid steps (or by guessing) scores full marks, and a model with correct reasoning but a differently-formatted answer scores zero unless the answer is normalized. It measures answer correctness under a fixed format, not reasoning quality — which is why process-based evaluation (Module 17) exists.

</details>

## Next

You now know what benchmark numbers hide — contamination, miscalibration, and the blind spots of each metric family — and you can compute ECE and pass@k by hand. The final lesson of the module turns all of this into engineering: a small, reproducible evaluation harness with fixed seeds, a fixed eval set, and regression tests, so that every experiment produces numbers you can trust and compare.

Continue to [13.3 · A reproducible eval harness](lessons/module-13/lesson-03.md).
