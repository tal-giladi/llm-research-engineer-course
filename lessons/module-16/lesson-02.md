# 16.2 · RLVR & verifiable rewards

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-16/lesson-01">16.1 · From PPO to GRPO</a> (the group-relative advantage and the clipped objective), the <a href="#/lessons/module-15/lesson-01">15.1 · reward model</a> (the <em>learned</em> reward this lesson replaces), and <code>softmax</code> / <code>log_softmax</code> / <code>torch.multinomial</code> from the earlier PyTorch lessons.</p>
<p><strong>You will learn:</strong> what <strong>RLVR</strong> (RL with Verifiable Rewards) is and why an <em>automatic checker</em> beats a learned reward model for reasoning; how a verifier is just a function <code>(problem, answer) -&gt; {0.0, 1.0}</code>; the exact-match and arithmetic verifiers in <code>llmre/reasoning/verifiers.py</code>; and the complete RLVR training loop in <code>llmre/reasoning/rlvr.py</code> — sample a group, verify, compute GRPO advantages, take a clipped step — watching a toy policy's correct-rate climb from chance to 1.0.</p>
<p><strong>Why this matters for ML:</strong> the single change that made RL for reasoning take off was swapping the fragile, gameable learned reward model for a cheap, exact, ungameable rule that <em>knows the right answer</em>. That is why math and code — domains where correctness is checkable by a program — are where reasoning RL first worked (DeepSeekMath, DeepSeek-R1). This lesson is the mechanism, end to end, small enough to run on a laptop.</p>
</div>

## 1. The problem with a learned reward

In RLHF (Module 15) the reward came from a **reward model**: a network trained on human preference pairs to output a scalar "how good is this completion." It works for open-ended helpfulness, where there is no ground truth to check against. But it has a structural weakness the policy will find and exploit: it is a **proxy**.

The reward model only approximates "what a human would prefer." The policy, optimized hard against it, learns to maximize the *proxy's score* — including in ways that diverge from what the proxy was meant to measure. It produces text that scores high with the reward model but is actually worse: over-long answers, confident-sounding nonsense, formatting tricks the RM happened to like. This is **reward hacking**, and it is the central failure mode of optimizing against a learned reward.

For **reasoning**, we have something RLHF does not: many problems have a **checkable ground-truth answer**. `3 + 4` is `7`. A function either passes its unit tests or it does not. A proof either type-checks or it does not. When the truth is checkable, there is no reason to *guess* the reward with a network — just *check the answer*.

<div class="callout key"><p><strong>RLVR = RL with Verifiable Rewards.</strong> Replace the learned reward model with an <em>automatic, rule-based checker</em> that knows the ground truth. The reward is cheap, exact, and cannot be hacked the way a learned proxy can — because it is not a proxy, it is the actual correctness of the answer.</p></div>

## 2. A verifier is a tiny function

In this codebase a verifier is any function

$$
\text{verifier} : (\text{problem},\ \text{answer}) \longrightarrow \{0.0,\ 1.0\},
$$

returning `1.0` when the answer is correct and `0.0` when it is wrong. That's it. No network, no parameters, no training. The reward is a Python `float`, and the training loop assembles many of them into a tensor.

`code/src/llmre/reasoning/verifiers.py` gives two.

### 2.1 `exact_match` — the outcome check

Most math RLVR rewards are, at bottom, "extract the final answer, compare it to the gold answer." That is `exact_match(pred, gold)`. It compares two ways and either match counts:

```python
def exact_match(pred: str, gold: str) -> float:
    pred, gold = str(pred), str(gold)
    p_num, g_num = _as_number(pred), _as_number(gold)
    if p_num is not None and g_num is not None:
        return 1.0 if p_num == g_num else 0.0        # numeric path: "3" == "3.0" == "+3"
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0   # string path: trimmed, lowercased
```

- **Numeric path:** if both sides parse as numbers, compare their float *values*, so `"3"`, `"3.0"`, and `"+3"` all match `3`. This matters because a model emits *strings*; you do not want to mark `"7.0"` wrong against gold `"7"`.
- **String path:** otherwise compare normalized strings (`_normalize` trims whitespace, lowercases, drops a leading `+`), so `" Cat "` matches `"cat"`.
- **Return:** a Python `float` in `{0.0, 1.0}`, not a tensor — rewards are cheap scalars the caller stacks into a tensor.

```python
from llmre.reasoning.verifiers import exact_match
print(exact_match("3", "3.0"))   # 1.0  (numeric)
print(exact_match("+7", "7"))    # 1.0  (leading plus)
print(exact_match(" Cat ", "cat"))  # 1.0  (string path, normalized)
print(exact_match("8", "9"))     # 0.0
```

### 2.2 `arithmetic_verifier` — a self-contained verifiable reward

For the toy loop we need a verifier that recomputes the ground truth *from the problem itself*, so there are no labels at all. `arithmetic_verifier(problem, answer)` parses a two-operand integer expression, computes the true result, and defers to `exact_match`:

```python
_ARITH_RE = re.compile(r"^\s*(-?\d+)\s*([+\-*x])\s*(-?\d+)\s*=?\s*$")

def arithmetic_verifier(problem: str, answer) -> float:
    m = _ARITH_RE.match(str(problem))
    if m is None:
        raise ValueError(f"cannot parse arithmetic problem: {problem!r}")
    a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
    truth = _OPS[op](a, b)                 # + - * ; "x" is also multiply
    return exact_match(str(answer), str(truth))
```

- It accepts `+ - * x` (with `x` as multiply), optional whitespace, and an optional trailing `=`, so `"3+4"`, `" 12 * 11 "`, and `"5 - 2 ="` all parse.
- Given *only* the problem string and the model's answer, it regenerates the correct answer and grades it. **No labels, no learned reward model** — this is the essence of a verifiable reward.
- It raises `ValueError` on anything it cannot parse, which the unit test checks (a verifier that silently returns 0 for malformed input would hide bugs).

```python
from llmre.reasoning.verifiers import arithmetic_verifier
print(arithmetic_verifier("3 + 4", 7))    # 1.0
print(arithmetic_verifier("3 + 4", 8))    # 0.0
print(arithmetic_verifier("12*11", 132))  # 1.0
print(arithmetic_verifier("6 x 7", "42")) # 1.0  (string answer, x = multiply)
```

<div class="callout warn"><p>A real math verifier is far hairier than this: it normalizes LaTeX, parses fractions and surds, handles $\pm$, units, and multiple equivalent forms, and often calls a computer-algebra system for symbolic equality. The <em>shape</em> of the signal is identical — <code>(problem, answer) -&gt; 0/1</code> — but the parsing is where the engineering (and the false-positive bugs of lesson <a href="#/lessons/module-16/lesson-03">16.3</a>) live.</p></div>

## 3. The RLVR loop, end to end

Now assemble the two halves — the verifier (this lesson) and GRPO (lesson [16.1](lessons/module-16/lesson-01.md)) — into a training loop that actually improves. `code/src/llmre/reasoning/rlvr.py` does this on a deliberately tiny problem so it runs on a CPU in under a second and you can *watch the correct-rate rise*.

### 3.1 A toy policy so the algorithm is unobscured

Wiring GRPO to a full GPT that emits answer *tokens* is engineering, not new algorithm. To keep the mechanism visible, the toy "policy" is a **table of logits**, one row per prompt, over a small discrete answer space — a categorical distribution $\pi(\text{answer} \mid \text{prompt})$:

```python
logits = torch.zeros(num_prompts, num_answers, requires_grad=True)  # (P, A) float32, CPU
opt = torch.optim.Adam([logits], lr=lr)
```

- **Shape / dtype / device:** `logits` is `(num_prompts, num_answers)` float32 on CPU; `logits[p]` is the distribution over the answer space for problem $p$. Everything stays on CPU.
- Initialized to all zeros, so `softmax` is uniform — the policy starts at *chance*, $\approx 1/\text{num\_answers}$. With the default 10-answer space, chance is $\approx 0.1$.

Swap this categorical table for a GPT that emits answer tokens and the outer loop is *unchanged* — only "sample an answer" and "score its log-prob" become "generate tokens" and "sum their log-probs."

### 3.2 The four steps, each mapped to 16.1

For every step, for every prompt, the loop does exactly the GRPO recipe:

```python
for _ in range(steps):
    # 1) SAMPLE a group of G answers per prompt from softmax(logits[p])
    sampled = torch.stack([_sample_group(logits[p], group_size) for p in range(num_prompts)])
    #   sampled: (num_prompts, group_size) long, indices into the answer space

    # 2) VERIFY: reward 1.0 if the sampled answer is correct, else 0.0
    rewards = torch.zeros(num_prompts, group_size)
    for p in range(num_prompts):
        for g in range(group_size):
            rewards[p, g] = verifier(problems[p], answers[sampled[p, g].item()])
    history.append(rewards.mean().item())        # fraction correct this step

    # 3) ADVANTAGE: group-relative, no critic  (lesson 16.1)
    advantages = grpo_advantages(rewards)        # (num_prompts, group_size)

    # 4) UPDATE: ascend the clipped objective (on-policy -> ratio 1)
    logp_all = F.log_softmax(logits, dim=-1)     # (num_prompts, num_answers)
    logp     = torch.gather(logp_all, 1, sampled)   # (num_prompts, group_size)
    obj  = grpo_objective(logp, logp.detach(), advantages, clip_eps=clip_eps)
    loss = -obj                                  # optimizer minimizes; we maximize the objective
    opt.zero_grad(); loss.backward(); opt.step()
```

Read the shapes as data flowing:

1. **Sample.** `_sample_group` softmaxes one prompt's logits and draws $G$ answers with `torch.multinomial(..., replacement=True)`. Result `(num_prompts, group_size)` of `long` indices into the answer list.
2. **Verify.** Loop the group, call the verifier on each sampled answer's *value* (`answers[idx]`), fill a `(num_prompts, group_size)` reward tensor of 0.0/1.0. Its mean over everything is the step's correct-rate, logged to `history`.
3. **Advantage.** Straight into `grpo_advantages` from 16.1 — standardize each prompt's group. Same shape.
4. **Update.** `log_softmax` gives log-probs over all answers; `gather` picks the log-prob of each *sampled* answer, shape `(num_prompts, group_size)` — aligned with the advantages. `grpo_objective` with `old_logprobs = logp.detach()` makes the ratio exactly 1 (on-policy), so this is the group-baselined policy gradient. `loss = -obj`, then Adam step.

<div class="callout pt"><p><code>torch.gather(logp_all, 1, sampled)</code> is the key line: it selects, for each prompt, the log-probabilities of the answers that were actually sampled (not all answers). Same idea as reading off the log-prob of the generated token in a real LLM policy-gradient step — there the "answers" are vocabulary tokens.</p></div>

### 3.3 Watch the correct-rate climb

Run the default configuration (four single-digit additions, 10-answer space, $G = 8$, 150 steps, seed 0):

```python
from llmre.reasoning.rlvr import run_rlvr
res = run_rlvr(seed=0)
print(res.initial_correct_rate, res.final_correct_rate)   # 0.15625  1.0
h = res.history
print(h[0], h[10], h[30], h[149])   # 0.156  0.750  1.000  1.000
```

It starts at `0.156` — a hair above the $1/10$ chance you'd expect from a uniform policy over ten answers — and by around step 30 it is sampling the verifier-accepted answer essentially every time. The reward taught the policy *which answer is right* purely by rewarding correct samples more; no answer key was ever handed to the optimizer, only a checker that says yes/no.

This is exactly what the unit test in `code/tests/test_reasoning_rl.py` asserts: `initial_correct_rate < 0.4`, `final_correct_rate > 0.9`, and the improvement `> 0.5` — a clear margin, reproducible from a fixed seed, and holding across seeds too.

<div class="hw">
<p><strong>Hardware track.</strong></p>
<p><strong>This toy:</strong> CPU-only, no GPU. Runtime &lt; 1 second, memory a few MB. Reproducible from <code>seed</code>. Run it in a REPL and re-run with different seeds; it always converges.</p>
<p><strong>Real reasoning RLVR (context, not a task here):</strong> the same algorithm on a real policy needs <em>many</em> GPUs. You sample a group of $G$ long completions per prompt (generation-bound), run policy forward/backward plus a frozen reference forward for the KL term, and repeat over large batches. Published reasoning-RL runs (DeepSeekMath, DeepSeek-R1) use large multi-GPU clusters and long wall-clock; the exact compute is lab-scale. The point of the toy is that the <em>algorithm</em> is identical — only the policy and the scale change.</p>
</div>

## 4. Why verifiable domains, and the honest caveats

Math and code led reasoning RL for one reason: **correctness is checkable by a program**, cheaply and exactly. That gives a reward with no proxy to hack and no human in the loop per sample. Where you *cannot* automatically verify — essay quality, open-ended helpfulness — RLVR does not directly apply, and you are back to a learned reward model or preference methods (Module 15).

Distinguishing the honesty levels the brief asks for:

- **Publicly documented:** DeepSeekMath and DeepSeek-R1 use rule-based / verifiable rewards (answer-correctness, and for R1 also format rules) instead of a learned reward model for the reasoning RL stage. This is stated in the papers.
- **Reasonable industry practice:** pairing a verifiable reward with GRPO; using unit tests as the verifier for code; combining an outcome-correctness reward with light format rewards.
- **Inference / speculation:** the precise verifier internals, reward shaping, and filtering any given lab uses are mostly not public — do not assume a specific recipe beyond "the reward is rule-based and checks correctness."

<div class="callout paper"><p><strong>Read:</strong> DeepSeekMath (Shao et al. 2024, paper #25) for GRPO + verifiable rewards, and DeepSeek-R1 (paper #26) for pure-RL-from-base with rule-based rewards — both in the <a href="#/papers/index">paper curriculum</a>. Note in R1 how the reward is answer-correctness plus format, no learned RM.</p></div>

## Exercise

Change `run_rlvr` to use a harder problem set — say two-digit sums like `"12 + 7"`, `"34 + 5"` — but keep the default 10-answer space `range(10)`. Predict what happens to the final correct-rate, then run it.

<details><summary>Hint</summary>

The policy can only output answers in `answer_space`. What are the true answers to `"12 + 7"` and `"34 + 5"`, and are they *in* `range(10)`?

</details>

<details><summary>Stronger hint</summary>

`12 + 7 = 19` and `34 + 5 = 39` are not in `range(10)`. No answer the policy can *sample* is ever verified correct, so every reward is 0.0 — and a unanimous all-zero group has advantage 0 (lesson 16.1). There is no gradient signal.

</details>

<details><summary>Solution</summary>

```python
from llmre.reasoning.rlvr import run_rlvr
res = run_rlvr(problems=("12 + 7", "34 + 5"), seed=0)
print(res.initial_correct_rate, res.final_correct_rate)   # 0.0 0.0
```

The correct-rate stays at 0.0: the right answers (19, 39) are outside the answer space, so no sampled answer is ever accepted, every group is unanimously wrong, `grpo_advantages` returns all zeros, and the policy never moves. The lesson: **RLVR can only teach answers the policy is capable of producing.** If the reward is never 1 for any reachable output, there is nothing to learn from — you must widen the answer space (e.g. `answer_space=range(100)`) so a correct answer is sampleable. This is the toy version of a very real failure: if a task is so hard the policy *never* stumbles onto a correct chain, verifiable RL has no signal to bootstrap from.

</details>

## Common mistakes

- **Answer unreachable by the policy.** If the correct answer is outside `answer_space` (or the model can never generate it), every reward is 0, every group is unanimous, and learning stalls — see the exercise.
- **Treating the verifier's output as a tensor.** It returns a Python `float`; the loop assembles floats into the `rewards` tensor. Don't call `.item()` on it.
- **Forgetting `replacement=True` semantics.** The group samples *with* replacement, so the same answer can appear several times in a group — that is fine and expected; the group is a Monte-Carlo estimate of the policy's distribution.
- **Passing a problem the verifier can't parse.** `arithmetic_verifier` raises `ValueError` on malformed input rather than returning 0 — catch it or fix the problem string; don't mask it.

## Check yourself

<details><summary>Why is a verifiable reward less hackable than a learned reward model?</summary>

A learned reward model is a *proxy* for correctness; the policy can find inputs that score high on the proxy without being genuinely better (reward hacking). A verifier checks the *actual* ground truth (the answer is right or it isn't), so there is no proxy gap to exploit — the only way to score is to be correct.

</details>

<details><summary>In the RLVR loop, what does <code>torch.gather(logp_all, 1, sampled)</code> produce, and why that op?</summary>

For each prompt it selects the log-probabilities of the *sampled* answers, giving a `(num_prompts, group_size)` tensor aligned with the advantages. We need the log-prob of each action actually taken (to form $\hat A \cdot \log\pi$), not the log-probs of all answers — `gather` picks exactly those entries by index.

</details>

<details><summary>The default run starts near 0.156. Where does that number come from?</summary>

The policy starts uniform over 10 answers, so a sampled answer is correct with probability $\approx 1/10 = 0.1$. `0.156` is the measured fraction correct in the first step's groups — chance, plus sampling noise from the finite group.

</details>

<details><summary>You set the answer space to <code>range(5)</code> but ask it to solve <code>"6 + 3"</code>. What happens?</summary>

`6 + 3 = 9` is not in `range(5)`, so no reachable answer is ever verified correct. Every reward is 0, every group is unanimous (advantage 0), and the correct-rate stays flat. RLVR can only reinforce answers the policy can actually produce.

</details>

## Next

You've seen RL *directly* optimize a policy against a verifier. There is a simpler, often-cheaper cousin that uses the same verifier without a policy-gradient loop at all: generate many candidates, **keep only the ones the verifier accepts, and fine-tune on those** — rejection sampling and STaR-style bootstrapping. It also raises the question of *what* you reward: only the final answer (outcome), or each step of the chain (process)?

Continue to [16.3 · Rejection sampling & verifiers](lessons/module-16/lesson-03.md).
