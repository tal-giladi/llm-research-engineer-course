# 16.5 · Where RL prompts come from, and how to build a verifiable set

<div class="prereq">
<p><strong>Prerequisites:</strong> GRPO and group-normalised advantages from <a href="#/lessons/module-16/lesson-01">16.1</a>; verifiers and the RLVR loop from <a href="#/lessons/module-16/lesson-02">16.2</a>; rejection sampling from <a href="#/lessons/module-16/lesson-03">16.3</a>; deduplication and decontamination of post-training data from <a href="#/lessons/module-14/lesson-04">14.4</a>.</p>
<p><strong>You will learn:</strong> what makes a prompt usable for RL with verifiable rewards; the public math, code and instruction-following sets in use today and their known problems; how to keep only problems whose answer a program can check; why GRPO learns nothing from problems that are too easy or too hard, worked out in numbers; how to measure difficulty with your own model and filter by it; and how to plug the result into TRL's <code>GRPOTrainer</code>.</p>
<p><strong>Why this matters for ML:</strong> in RLVR the prompt set <em>is</em> the curriculum. The verifier decides what counts as correct, and the difficulty of the prompts decides whether there is any learning signal at all. DAPO, Skywork-OR1, POLARIS and OLMo 3 all report that filtering prompts by the current model's pass rate was one of their most important changes.</p>
</div>

<div class="hw">
<p><strong>Hardware track:</strong> Parts 1 to 3 of the lab are CPU only (streams about 10,000 rows, a few minutes). Part 4 (measuring pass rates with your own model) runs on CPU with a 0.5B model for a handful of problems (minutes per problem), and on one GPU with vLLM for thousands. Checked on 2026-11-13 with Python 3.14, <code>datasets</code> 5.2.0, <code>math-verify</code> 0.9.0, <code>transformers</code> 5.17 and <code>trl</code> 1.17.1.</p>
</div>

## 1. Intuition: what an RL prompt needs

An SFT row needs a good answer. A preference row needs two answers and a judgment. An RLVR row needs **neither**: it needs a prompt and a way to check any answer the model produces. The model writes its own answers during training; the data only has to make checking possible.

So a usable RLVR prompt has three properties:

1. **A checkable answer.** One value (a number, a fraction, an interval, an expression) that a program can compare, or unit tests that a program can run, or constraints a program can verify ("at most 50 words", "exactly 3 bullet points").
2. **A correct reference.** Public sets were built by scraping, OCR and model generation. Some reference answers are wrong. A wrong reference punishes correct answers, which is worse than no data.
3. **The right difficulty for the model being trained.** Not solved every time, not failed every time. This depends on the model, so it cannot be a fixed property of the dataset.

There are three main families:

- **Math with a final answer.** The model reasons, then puts its answer in `\boxed{...}` (DeepSeek-R1, most open recipes) or on a last line such as `Answer: 42` (DAPO). The verifier extracts and compares it.
- **Code with tests.** The verifier runs the program on hidden test cases in a sandbox. Reward is 1 if all pass (or the fraction passed).
- **Instruction following with checkable constraints.** A normal request plus constraints such as "answer in all lowercase" or "end with the phrase 'Any other questions?'". IFEval defined 25 such constraint types; Tulu 3's RLVR set adds one random constraint to each prompt.

## 2. The public sets, as of November 2026

Every number below was read from the dataset card or the Hugging Face API on 2026-11-13.

**Math**

| Dataset (Hugging Face id) | Rows | Answer format and notes | License |
|---|---|---|---|
| `project-numina/NuminaMath-1.5` | 896,215 | `answer` field; can be `proof` or `notfound`; `question_type` marks proofs and MCQ. Not decontaminated. Was `AI-MO/…` (redirects) | Apache-2.0 |
| `agentica-org/DeepScaleR-Preview-Dataset` | 40,315 | `answer` in LaTeX; includes AIME 1984–2023 and AMC before 2023, which overlap common evals | MIT |
| `zwhe99/DeepMath-103K` | 103,022 | `final_answer` plus a `difficulty` score; decontaminated by semantic matching; 48 leaked answers fixed in May 2025 | MIT |
| `Skywork/Skywork-OR1-RL-Data` | 105k math + 14k code | Per-model difficulty for three R1-Distill sizes; deduplicated and AIME 24/25 and LiveCodeBench near-matches removed. Use a version from commit `b48ac2e` or later | No tag on the card; repo Apache-2.0 |
| `open-r1/DAPO-Math-17k-Processed` | 17,398 | Every answer converted to an integer. The original `BytedTsinghua-SIA/DAPO-Math-17k` repeats each prompt about 100 times (1,791,700 rows) | Apache-2.0 |
| `POLARIS-Project/Polaris-Dataset-53K` | 53,291 | `difficulty` = pass rate of R1-Distill-Qwen-7B | Apache-2.0 |
| `nvidia/OpenMathReasoning` | 5.68M samples, 306K problems | `expected_answer`; for some problems it is a majority vote of model answers, so it is noisy; `pass_rate_72b_tir` for difficulty | CC-BY-4.0 |
| `SynthLabsAI/Big-Math-RL-Verified` | gated | Filtered for uniquely verifiable, open-ended, closed-form answers. An ungated copy is `open-r1/Big-Math-RL-Verified-Processed` (215,608) | Apache-2.0 |
| `allenai/Dolci-Think-RL-7B` (OLMo 3) | 102,014 | Math, code, instruction following and chat mixed for one RL run | ODC-BY |

**Code:** `agentica-org/DeepCoder-Preview-Dataset` (24,974 problems with at least 5 tests each; its training data includes LiveCodeBench v5 problems from May 2023 to July 2024, so do not evaluate on those afterwards), `PrimeIntellect/verifiable-coding-problems` (144,169, no license tag), `open-r1/codeforces` (CC-BY-4.0; has a test split you must not train on), `KodCode/KodCode-V1` (487,432 synthetic problems with pytest tests; **CC-BY-NC-4.0**, non-commercial). `BAAI/TACO`, `codeparrot/apps` and `livecodebench/code_generation_lite` still use loading scripts, which `datasets` 4.0 and later no longer run: download their raw files instead.

**Instruction following:** `allenai/RLVR-IFeval` (14,973 prompts, each with one IFEval constraint and the name of the function that checks it), `allenai/IF_multi_constraints_upto5` (95,373 prompts with up to five constraints, from the IFBench work), and the evaluation sets to decontaminate against, `google/IFEval` (541) and `allenai/IFBench_test` (300).

<div class="callout warn"><p>Several of the most-used math sets contain old AIME and AMC problems, and AIME is what most reasoning papers report. Decontaminate against every benchmark you will report, and when a benchmark is released yearly, prefer evaluating on a year your data cannot contain (AIME 2025 for data built in early 2025, and so on).</p></div>

## 3. Is the answer checkable? The verifiability filter

NuminaMath-1.5 is a good example of raw material. In a random sample of 5,000 rows, 601 are labelled `proof` and 651 `MCQ`, and 105 have the answer `notfound`. A proof has no value to compare. A multiple-choice letter can be guessed one time in four or five, so the policy gets rewarded for guessing. Neither belongs in an RLVR set.

`answer_is_verifiable(problem, answer)` adds checks the labels miss:

- **no final answer** (empty or missing);
- **proof wording** in the problem ("prove", "show that"), which catches multi-part problems whose last part is a proof;
- **multi-part answers**: a top-level `,` or `;` as in `x_1=-9, x_2=2`. One string match cannot grade "both roots". A comma *inside* brackets is one answer: the point `(2, 3)` and the interval `[0, 1)` are kept;
- **very long answers** (over 40 characters after normalisation), which are usually descriptions rather than values;
- **multiple-choice letters** that slipped through the labels.

Then check the reference itself: if `math-verify` cannot parse the gold answer, the verifier will mark every response wrong, so drop the row.

### Comparing answers

`extract_boxed` returns the contents of the last `\boxed{...}`, counting braces so `\boxed{\frac{1}{2}}` works. `answers_match` normalises both answers (strips `$`, `\left`/`\right`, `\text{}`, `\$`, `^\circ`, a leading `x=`, thousands separators; turns `\frac{a}{b}` into `a/b`) and compares them as strings, then as numbers, so `\frac{1}{2}` matches `0.5`.

This is a teaching version. In real training use Hugging Face's `math-verify` (0.9.0), which parses LaTeX into SymPy and compares values, intervals, sets and expressions:

```python
from math_verify import parse, verify

gold = parse(r"\boxed{\frac{1}{2}}")
pred = parse(r"so the answer is \boxed{0.5}")
verify(gold, pred)          # True. Argument order matters: gold first.
```

Two practical notes, both from running it for this lesson. Its README asks for LaTeX inside a math environment (dollar signs or `\boxed{}`); in our tests bare `\frac{1}{2}` also parsed, but the lab wraps gold answers in dollar signs to follow the documented form. And its default timeouts use signals on Unix and subprocesses on Windows, which failed in our Windows sandbox; the lab passes `parsing_timeout=None` and `timeout_seconds=None`. A trainer that calls the verifier millions of times should run it in a worker process with its own timeout, because a pathological expression can make SymPy run for a very long time.

<div class="callout warn"><p><strong>Verifiers make mistakes in both directions.</strong> False negatives (a correct answer written as <code>0.50</code>, <code>\dfrac12</code> or <code>x = 1/2</code> marked wrong) hurt more as the policy improves, because more of its correct answers are written in varied forms ("From Accuracy to Robustness", 2025). False positives are worse: a verifier that accepts the last number in the text rewards models that list many numbers. Test your verifier on a few hundred real model outputs before training, and read the ones it rejects.</p></div>

## 4. Difficulty: why the right prompts are the ones the model sometimes solves

### 4.1 Intuition

GRPO (16.1) samples a group of $G$ answers per prompt and gives each answer an advantage relative to the group. If every answer in the group gets the same reward, every advantage is zero and the prompt contributes **no gradient**. It still costs the full price of generating $G$ long answers. That happens when the model always solves the prompt (too easy) or never does (too hard).

### 4.2 Mathematics

For a group of rewards $r_1, \dots, r_G \in \{0, 1\}$, the GRPO advantage is

$$
A_i = \frac{r_i - \bar r}{\sigma_r + \varepsilon}, \qquad \bar r = \frac{1}{G}\sum_{j=1}^G r_j, \quad \sigma_r = \sqrt{\frac{1}{G}\sum_{j=1}^G (r_j - \bar r)^2},
$$

where $\varepsilon$ is a small constant that avoids division by zero. If all $r_j$ are equal, $r_i - \bar r = 0$ for every $i$, so all $A_i = 0$.

Let $p$ be the model's true probability of solving a prompt (its **pass rate**). The probability that a group of $G$ independent samples is all-correct or all-wrong is

$$
P_{\text{no signal}}(p) = p^G + (1 - p)^G.
$$

### 4.3 Numerical example

Take $G = 8$, the default `num_generations` in TRL 1.17.1.

| Pass rate $p$ | $p^8$ | $(1-p)^8$ | $P_{\text{no signal}}$ |
|---|---|---|---|
| 0.5 | 0.0039 | 0.0039 | **0.8%** |
| 0.9 | 0.430 | $10^{-8}$ | **43%** |
| 0.95 | 0.663 | $4 \times 10^{-11}$ | **66%** |
| 0.05 | $4 \times 10^{-11}$ | 0.663 | **66%** |
| 1.0 or 0.0 | | | **100%** |

So a prompt the model solves 90% of the time wastes 43% of the groups spent on it. One group, worked by hand: rewards $(1, 0, 0, 0)$ with $G = 4$ give $\bar r = 0.25$, $\sigma_r = \sqrt{(0.75^2 + 3 \cdot 0.25^2)/4} = \sqrt{0.1875} \approx 0.433$, so $A_1 = 0.75 / 0.433 \approx 1.73$ and the three wrong answers get $-0.25/0.433 \approx -0.58$ each. Rewards $(1, 1, 1, 1)$ give $\bar r = 1$, $\sigma_r = 0$, and all four advantages are $0 / \varepsilon = 0$.

```python
from llmre.rl.data import group_has_signal
group_has_signal([1.0, 0.0, 0.0, 0.0])   # True
group_has_signal([1.0] * 8)              # False: this prompt taught nothing this step
```

### 4.4 Measuring difficulty, and the noise in the measurement

You estimate $p$ for each prompt by sampling $k$ answers from the **current** model and counting correct ones: `pass_rate(problem, gold, policy, k)`. Then `difficulty_filter(rows)` keeps $0 < \hat p < 1$.

The estimate is noisy. A prompt with true $p = 0.05$ gets $\hat p = 0$ from $k = 8$ samples with probability $0.95^8 \approx 0.66$, so two times in three you drop a prompt that would sometimes give signal. Larger $k$ helps (Skywork-OR1 used $k = 16$ for math), and so does re-estimating during training instead of once.

### 4.5 What the published recipes do

- **DAPO** (dynamic sampling): during training, over-sample prompts and discard any whose group is all-correct or all-wrong, until the batch is full of prompts with signal.
- **Skywork-OR1**: before training, 16 samples per math problem from each of three model sizes; for each model, drop problems with 0 or 16 correct. Between training stages, drop prompts the current model solved every time.
- **POLARIS**: pass rate from 8 samples of the model being trained. Removing only the 8/8 problems helped; removing everything above 4/8 left too little data and hurt. Their target distribution has most problems on the hard side.
- **FADE** (November 2026) argues that a pass rate tracked per prompt mixes up two things that change over time, the model's ability and the prompt's difficulty, and models them separately. Treat it as a recent research direction, not settled practice.

<div class="callout key"><p>Difficulty is a property of a <em>(prompt, model)</em> pair, not of a prompt. Filter with the model you are about to train, and refilter as it improves.</p></div>

## 5. Shapes and cost

**Shapes, dtype, device.** The data side is plain Python: a problem is a `dict` with `problem: str`, `answer: str`, and after scoring `pass_rate: float`. Rewards are Python floats. Inside GRPO the rewards become a `float32` tensor of shape `(B, G)` on the training device (`B` prompts, `G` samples each), and the advantages have the same shape.

**Cost.** Difficulty filtering costs generation: $N$ prompts × $k$ samples × average answer length. For 50,000 prompts, $k = 8$ and 2,000-token answers, that is $8 \times 10^8$ generated tokens. With vLLM on one modern GPU at an aggregate of a few thousand tokens per second for a 1.5B to 7B model, that is days of GPU time, comparable to a short RL run. That is why recipes filter once with a cheap model or reuse the difficulty columns that ship with Skywork-OR1 and POLARIS, then rely on dynamic sampling during training. The verifier itself is cheap for math (milliseconds) and expensive for code (a sandboxed process per test case).

## 6. Lab: build an RL prompt set from real data

`code/scripts/lab_rl_data.py`:

- **Part 1** streams 5,000 random NuminaMath-1.5 problems, applies the dataset's labels and `answer_is_verifiable`, and checks that `math-verify` can parse every kept gold answer.
- **Part 2** streams 5,000 Skywork-OR1 math problems and applies `difficulty_filter` using the shipped difficulty for each of three model sizes.
- **Part 3** decontaminates the kept NuminaMath problems against MATH-500, AIME 2024 and AIME 2025 (560 benchmark problems) with the 8-gram index from 14.4, and writes `rl_prompts.jsonl`.
- **Part 4** (optional) measures pass rates with your own model.

```bash
cd code
pip install -e . "datasets==5.2.0" "math-verify==0.9.0"
python scripts/lab_rl_data.py --n 5000 --out data/rl
```

You know it worked when `data/rl/rl_prompts.jsonl` holds a few thousand `{problem, answer, source}` rows and `data_card.json` has the counts below.

### What we measured (2026-11-13, seed 0)

**Part 1, NuminaMath-1.5:** of 5,000 problems, **3,223 (64%) were kept**.

| Dropped because | Rows |
|---|---|
| Dataset label: proof | 656 |
| Dataset label: multiple choice | 651 |
| Our filter: multi-part answer (`x=26,y=24`, `x_1=-9, x_2=2`, lists of points) | 158 |
| Our filter: answer longer than 40 characters (general solutions like `\frac{\pi}{3}+\pi k, k\in\mathbb{Z}`) | 123 |
| Dataset label: answer `notfound` | 105 |
| Our filter: no answer | 49 |
| Our filter: proof wording in the problem | 23 |
| Our filter: multiple-choice letter despite the label | 10 |
| `math-verify` could not parse the gold answer | 1 |

Reading the drops is how the filter was fixed. The first run rejected `\$93,750` as a two-part answer because of the escaped dollar sign and the comma; normalisation now strips `\$`, `\%` and degree signs, and that recovered 10 rows.

**Part 2, Skywork-OR1:** the same 5,000 problems, filtered for three models:

| Model the difficulty was measured on | Too easy (16/16) | Too hard (0/16) | Kept |
|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | 770 | 1,279 | **2,951** |
| DeepSeek-R1-Distill-Qwen-7B | 1,679 | 1,055 | **2,266** |
| DeepSeek-R1-Distill-Qwen-32B | 2,152 | 1,009 | **1,839** |

The 32B model already solves 43% of the problems every time, so its useful set is the smallest. 431 problems are never solved by the 1.5B model but are useful for the 32B one. Same data, three different training sets.

**Part 3, decontamination:** 0 of the 3,223 kept NuminaMath problems matched MATH-500, AIME 2024 or AIME 2025 at $\tau = 0.5$. Lesson 14.4's exercise found MATH-500 copies in NuminaMath-*CoT* (13 hits in 20,000), mostly from its `math` and `synthetic_math` sources. NuminaMath-1.5's rebuilt sources show none in this sample. Zero hits in 5,000 rows is evidence for this sample, not a guarantee for all 896,215: run it on everything you train on.

### Part 4: measure pass rates with your own model

```bash
python scripts/lab_rl_data.py --n 0 --policy Qwen/Qwen2.5-0.5B-Instruct --k 8 --m 20
```

This samples $k$ answers per problem with `transformers`, at temperature 1.0 with the instruction "Please reason step by step, and put your final answer within \boxed{}", grades them with `math-verify`, and applies `difficulty_filter`. On a CPU keep `--m` small: in our run a 0.5B model generated about 5 tokens per second on a laptop CPU, so one 500-token answer took about 100 seconds. On a GPU, replace the `generate` loop with vLLM (0.33.0), which batches all $k$ samples of all problems together.

**Our run (2026-11-13, `Qwen/Qwen2.5-0.5B-Instruct`, $k = 4$, 6 problems, CPU, about 15 minutes):** pass rates were 0, 0, 0, 0, 0.25 and 0. Five of six problems were "too hard" and would be dropped; one was kept. Reading the outputs explained both kinds of failure:

- The model was simply wrong on easy word problems at temperature 1.0 (it boxed `34` where the answer is `28`). A 0.5B instruct model is far too weak for this set; it is the wrong model to run RL on with these prompts, and the filter says so before you spend GPU time.
- One reference answer was `0.3333..`, a sloppy way to write $\frac{1}{6}$'s gap to $0.5$, which is $\frac{1}{3}$. The verifier cannot match a correct `\frac{1}{3}` against `0.3333..`. This is the "trust the reference" problem from Section 1 in a dataset that had already passed our filters.

This is why Part 4 matters even when a dataset ships difficulty scores: they were measured on someone else's model, and only your own samples show whether the prompts suit yours.

## 7. Into TRL's GRPOTrainer

TRL 1.17.1 calls each reward function with the batch's `completions` and every dataset column except `prompt` as keyword arguments, and expects one float per completion (or `None` to skip that function for a sample). So a dataset with `prompt` and `solution` columns and a function that reads `solution` is all you need:

```python
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer
from llmre.rl.data import extract_boxed, answers_match

SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
ds = load_dataset("json", data_files="data/rl/rl_prompts.jsonl", split="train")
ds = ds.map(lambda r: {"prompt": [{"role": "user", "content": r["problem"] + SUFFIX}],
                       "solution": r["answer"]})

def boxed_answer_reward(completions, solution, **kwargs):
    # conversational format: each completion is a list with one assistant message
    texts = [c[0]["content"] for c in completions]
    return [answers_match(extract_boxed(t), gold) for t, gold in zip(texts, solution)]

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-0.5B-Instruct",
    reward_funcs=boxed_answer_reward,
    args=GRPOConfig(output_dir="grpo-run", num_generations=8),
    train_dataset=ds,
)
trainer.train()
```

TRL also ships `trl.rewards.accuracy_reward(completions, solution, **kwargs)`, which uses `math-verify` (install `trl[math_verify]`) and reads the same `solution` column. In 1.17.1 the GRPO defaults are `num_generations=8`, `loss_type="dapo"`, `scale_rewards="group"` and `beta=0.0` (no KL term). Running this needs a GPU; on a CPU, `GRPOConfig` refuses its default bf16 setting unless you pass `use_cpu=True, bf16=False`, and training is too slow to be useful.

## Common mistakes

- **Keeping proof and multiple-choice problems.** Proofs have no checkable value; letters reward guessing.
- **Trusting reference answers blindly.** OpenMathReasoning's answers for some problems are a majority vote of model answers; NuminaMath's olympiad subset had extraction errors. Spot-check answers on a sample, and drop problems that *every* strong model "gets wrong" in the same way: often the reference is wrong.
- **Filtering difficulty with a different model than the one you train.** A set filtered for a 32B model is mostly too hard for a 1.5B one.
- **Estimating pass rates with too few samples.** $k = 4$ drops many prompts with small but useful pass rates.
- **Using the original DAPO-Math-17k file.** It repeats each prompt about 100 times. Use the deduplicated copy or deduplicate yourself.
- **Training on a set that contains your evaluation.** DeepScaleR includes old AIME; DeepCoder includes LiveCodeBench v5. Decontaminate, and say what you checked.
- **Letting format decide the reward.** If the prompt asks for `\boxed{}` and the verifier only reads `\boxed{}`, a correct answer without a box scores 0. Either reward the format separately (as DeepSeek-R1 did) or make sure SFT taught the format first.

## Exercise: a filter for code problems

Write `code_is_verifiable(row)` for `agentica-org/DeepCoder-Preview-Dataset` (config `primeintellect`). It should drop problems with fewer than 5 tests, problems whose tests have no expected output, and problems whose statement asks for several valid outputs ("print any of them"), which exact-match tests cannot grade. Report how many of 2,000 streamed rows it keeps.

<details><summary>Hint</summary>

Stream one row and print its keys and the first test. In this dataset `tests` is a JSON string; parse it with `json.loads`. Each test has an input and an expected output.

</details>

<details><summary>Stronger hint</summary>

"Any valid answer" problems are detected from the statement: search for phrases such as "print any", "output any", "if there are multiple answers". Codeforces handles them with a custom checker program; without one, your exact-match test would reject correct solutions (a false negative for every valid alternative).

</details>

<details><summary>Solution sketch</summary>

```python
import json, re
ANY = re.compile(r"(print|output) any|multiple (possible |valid )?(answers|solutions)", re.I)

def code_is_verifiable(row):
    tests = json.loads(row["tests"]) if isinstance(row["tests"], str) else row["tests"]
    reasons = []
    if len(tests) < 5:
        reasons.append("fewer than 5 tests")
    if any(not str(t.get("output", "")).strip() for t in tests):
        reasons.append("test without expected output")
    if ANY.search(row["problem"]):
        reasons.append("multiple valid outputs need a checker")
    return reasons
```

Our run on 2026-11-13 (2,000 shuffled rows, seed 0): **1,798 kept**, 194 dropped as "multiple valid outputs" (about 10%), 8 dropped for a test with an empty expected output, and 0 for having fewer than 5 tests (the card promises at least 5, so a non-zero count would mean your parsing is wrong). We read ten of the 194 and all ten were real "if there are several answers, print any of them" problems. Exact-match tests reject every valid alternative answer to such a problem, which is why Codeforces grades them with checker programs and why `open-r1/codeforces` ships generated checkers for about 30% of its problems.

</details>

## Debugging exercise

A GRPO run on a new math set shows the mean reward stuck near 0.92 from step 1, the loss near zero, and no improvement on the evaluation set after 500 steps. The logs report that about 60% of groups have zero reward variance. What is wrong with the data, and what do you do?

<details><summary>Answer</summary>

The set is too easy for this model: with $p \approx 0.92$ and $G = 8$, $P_{\text{no signal}} = 0.92^8 + 0.08^8 \approx 0.51$, and with many prompts at $p = 1$ it rises to the observed 60%. Most groups give zero advantage, so almost nothing is learned. Measure pass rates with the current model and drop prompts solved $k/k$ (and $0/k$), add harder problems, or turn on dynamic sampling so the batch is refilled with prompts that have signal. A mean reward near 1 at step 1 is itself the warning sign.

</details>

## Research connection

- **DeepSeekMath** ([Shao et al., 2024](https://arxiv.org/abs/2402.03300)): GRPO and its RL data (about 144K GSM8K- and MATH-style questions).
- **DeepSeek-R1** ([DeepSeek-AI, 2025](https://arxiv.org/abs/2501.12948)): rule-based accuracy and format rewards, and why they avoided learned reward models for reasoning.
- **Tulu 3** ([Lambert et al., 2024](https://arxiv.org/abs/2411.15124)), Section 6: RLVR on GSM8K, MATH and IFEval-style constraints.
- **DAPO** ([Yu et al., 2025](https://arxiv.org/abs/2503.14476)): dynamic sampling and converting answers to integers.
- **Skywork-OR1** ([He et al., 2025](https://arxiv.org/abs/2505.22312)): per-model difficulty estimation and stage-wise refiltering.
- **POLARIS** ([blog, 2025](https://hkunlp.github.io/blog/2025/Polaris/)): how much filtering is too much.
- **Generalizing Verifiable Instruction Following / IFBench** ([Pyatkin et al., 2025](https://arxiv.org/abs/2507.02833)) and **IFEval** ([Zhou et al., 2023](https://arxiv.org/abs/2311.07911)).
- **From Accuracy to Robustness** ([Huang et al., 2025](https://arxiv.org/abs/2505.22203)) on verifier errors, and **Spurious Rewards** ([Shao et al., 2025](https://arxiv.org/abs/2506.10947)), which shows that even random rewards raised one model family's MATH-500 score, so gains must be checked across model families.
- **One-shot RLVR** ([Wang et al., 2025](https://arxiv.org/abs/2504.20571)): a single well-chosen example took Qwen2.5-Math-1.5B from 36.0% to 73.6% on MATH-500. Prompt selection matters more than prompt count.
- **OLMo 3** ([Olmo Team, 2025](https://arxiv.org/abs/2512.13961)): the Dolci RL sets mixing math, code, instruction following and chat.

**What to read and what to skip.** In DAPO, read the dynamic-sampling section and the dataset paragraph; skim the token-level loss section if you already read F.2. In Skywork-OR1, read the data-preparation section and the multi-stage training description; skip the long ablation tables on entropy. In Spurious Rewards, read the abstract and the cross-model comparison: the lesson is "test on more than one model family".

**Code this after reading:** add `--k` and `--threshold-high` options to Part 4 and plot how many problems survive as you vary them.

## Check yourself

<details><summary>Why is a multiple-choice letter a bad RLVR answer even when it is correct?</summary>

A policy can earn reward by guessing: one time in four or five with no reasoning at all. That rewards guessing strategies and makes the reward a poor signal of correct reasoning.

</details>

<details><summary>With $G = 8$, what fraction of groups give no gradient for a prompt with pass rate 0.9?</summary>

$0.9^8 + 0.1^8 \approx 0.430 + 0.00000001 \approx 43\%$.

</details>

<details><summary>Skywork-OR1 ships difficulty for three model sizes. Why not one difficulty per problem?</summary>

Whether a problem gives signal depends on the model's pass rate on it, which differs by model. In our sample the 32B model solves 43% of problems every time, the 1.5B model only 15%, so their useful subsets are very different.

</details>

<details><summary>Your verifier marks a correct answer written as <code>x = 1/2</code> wrong against the gold <code>\frac{1}{2}</code>. Why does this matter more late in training than early?</summary>

Late in training more answers are correct, so a fixed rate of false negatives punishes a larger number of correct answers, and pushes the model toward the narrow formats the verifier accepts rather than toward correctness.

</details>

<details><summary>Why can 4 samples per prompt be too few for difficulty filtering?</summary>

A prompt with true pass rate 0.1 gets 0 correct out of 4 with probability $0.9^4 \approx 0.66$, so it is usually dropped as "too hard" although it would give signal in an 8-sample GRPO group about 57% of the time ($1 - 0.9^8 - 0.1^8$).

</details>

## Next

You can now build all three kinds of post-training data: SFT demonstrations (14.4), preference pairs (15.5) and verifiable RL prompts (this lesson). Module 17 uses them to build reasoning models, starting with chain-of-thought and self-consistency.

Continue to [17.1 · CoT, self-consistency, STaR](lessons/module-17/lesson-01.md).
