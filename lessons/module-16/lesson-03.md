# 16.3 · Rejection sampling & verifiers

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="lesson-02.md">16.2 · RLVR &amp; verifiable rewards</a> (a verifier is a <code>(problem, answer) -&gt; {0,1}</code> checker), <a href="lesson-01.md">16.1 · GRPO</a> (sampling a group of completions), and <a href="../module-14/lesson-01.md">14.1 · SFT</a> (fine-tuning on demonstrations).</p>
<p><strong>You will learn:</strong> <strong>rejection sampling</strong> / best-of-$n$ / STaR-style bootstrapping — generate many completions, keep the verifier-accepted ones, and SFT on them; when this beats a full RL loop and when it doesn't; the difference between an <strong>outcome</strong> reward model (ORM) and a <strong>process</strong> reward model (PRM) and the trade-offs between them; and the ways a verifier can betray you — false positives and gaming — with concrete pitfalls from the code you already have.</p>
<p><strong>Why this matters for ML:</strong> not every reasoning improvement needs a policy-gradient loop. The cheapest, most robust lever in the reasoning toolkit is often "sample a lot, keep what verifies, fine-tune on it." Knowing when to reach for rejection sampling versus GRPO, and understanding that <em>your verifier is now the definition of correct</em>, is core research-engineering judgment — a wrong verifier trains a wrong model, silently.</p>
</div>

## 1. You don't always need the RL loop

Lesson [16.2](lesson-02.md) optimized the policy *directly* against the verifier with GRPO — a gradient step per group, the policy's distribution nudged every iteration. That is powerful but has moving parts: ratios, clipping, a KL term, a reference model, sampling groups every step.

There is a blunter method that uses the *same verifier* and no policy-gradient machinery at all:

1. **Generate** many completions per prompt from the current model (a group, like GRPO — but we won't compute advantages).
2. **Filter** with the verifier: keep only the completions that reach a correct answer.
3. **SFT** the model on those kept completions — plain supervised fine-tuning (Module 14) on the model's *own* verified-correct outputs.
4. Optionally repeat: the fine-tuned model generates better candidates next round.

This is **rejection sampling** (a.k.a. rejection-sampling fine-tuning, or best-of-$n$ when you keep the single best per prompt). The self-improving, iterated form is **STaR** (Self-Taught Reasoner): the model bootstraps by learning from the chains it got right.

<div class="callout key"><p><strong>Rejection sampling / STaR:</strong> sample many, <em>keep only verifier-accepted</em> completions, SFT on them. Same verifier as RLVR, but the update is ordinary supervised learning on filtered self-generated data — no advantages, no clipping, no KL.</p></div>

### 1.1 Why it works, and the connection to GRPO

Both methods exploit the same fact: a model that is *right sometimes* generates correct chains at some nonzero rate, and a verifier can pick them out for free. Rejection sampling then imitates those correct chains directly. In fact it is the extreme, simplified cousin of the policy gradient: GRPO pushes up correct completions' probability *smoothly and by a graded amount* (the advantage); rejection-sampling SFT pushes up the kept (correct) completions with a *full imitation-learning gradient* and simply *discards* the rest, rather than pushing them down.

The same precondition therefore applies as in 16.2's exercise: **if the model never produces a correct completion for a prompt, there is nothing to keep** — the filtered set is empty and that prompt teaches nothing. Rejection sampling can only amplify reasoning the model can already occasionally do.

### 1.2 A tiny illustration with the tools you have

You can see the "generate a group → keep the verified ones" filter using the exact building blocks from `llmre/reasoning`:

```python
import torch, torch.nn.functional as F
from llmre.reasoning.verifiers import arithmetic_verifier

torch.manual_seed(0)
problem, answers = "3 + 4", list(range(10))
logits = torch.zeros(len(answers))                 # uniform toy policy over 0..9
probs  = F.softmax(logits, dim=-1)
group  = torch.multinomial(probs, 12, replacement=True)   # sample n=12 candidates

kept = [answers[i] for i in group.tolist()
        if arithmetic_verifier(problem, answers[i]) == 1.0]  # keep verifier-accepted
print(kept)   # e.g. [7, 7] -- only the correct answer 7 survives; you'd SFT on those
```

Every candidate not equal to 7 is *rejected*; the survivors (all `7`) are the supervised targets you would fine-tune on. Scale the "answer" up to a full worked chain-of-thought and this is exactly rejection-sampling fine-tuning.

### 1.3 Rejection sampling vs GRPO — when to reach for which

- **Rejection sampling** is simpler, very stable (it's just SFT), parallelizes trivially (generation then a standard fine-tune), and is a strong first move — often most of the gain for a fraction of the complexity. Its ceiling is lower: it throws away the information in *wrong* answers and can't push probability *down*, and iterating it can narrow diversity.
- **GRPO / RLVR** squeezes more out of the same verifier — it uses negative signal (wrong completions get negative advantage) and keeps optimizing past what pure imitation reaches — at the cost of a more delicate training loop.

A common, publicly-described pattern is to do **both**: rejection-sampling SFT to get a strong start, then RL (GRPO) on top. (DeepSeek-R1's pipeline, covered in [17.3](../module-17/lesson-03.md), interleaves SFT on filtered data with RL stages.)

## 2. Outcome vs process rewards (ORM vs PRM)

So far every verifier scored the **final answer**: right or wrong. That is an **outcome reward** — an **Outcome Reward Model (ORM)** when learned, or an outcome *verifier* like `arithmetic_verifier` when it's a rule. But a chain of reasoning has *steps*, and you can instead reward *each step*:

- **Outcome reward (ORM):** one scalar for the whole completion, from the final answer only. Cheap, and for verifiable tasks it's *exact* (the answer is checkably right). Its blind spot: it cannot tell a **lucky wrong-reasoning-right-answer** chain from a genuinely correct one, and it gives no signal about *where* a failed chain went wrong.
- **Process reward (PRM):** a score for *each intermediate step* of the chain — was this step valid given the previous ones? Denser signal: it can credit the good part of a chain that later derails, and it pushes toward *correct reasoning*, not just correct final tokens. Its cost: you need step-level labels or a learned PRM to produce them, which is expensive and itself a proxy that can be gamed.

<div class="callout key"><p><strong>Outcome</strong> = reward the final answer (cheap, exact when verifiable, but blind to <em>how</em>). <strong>Process</strong> = reward each step (dense, targets real reasoning, but needs step labels / a learned model and reintroduces proxy risk). Verifiable-outcome rewards are why Module 16's toy works; process rewards are the subject of <a href="../module-17/lesson-02.md">17.2</a>.</p></div>

The trade-off is genuinely open and task-dependent: outcome rewards are simpler and, crucially, *un-hackable when the verifier is exact*; process rewards give richer credit assignment but drag a learned, gameable component back in. We take PRMs up properly — including the "Let's Verify Step by Step" line of work — in [17.2 · Process reward & DeepSeekMath](../module-17/lesson-02.md).

## 3. Verifier design pitfalls: your checker *is* the reward

Here is the uncomfortable truth of RLVR and rejection sampling: **whatever your verifier accepts becomes the model's definition of "correct."** Optimize hard enough and the model will find every gap in the checker. A buggy verifier does not fail loudly — it silently trains the model to satisfy the bug.

### 3.1 False positives — accepting a wrong answer

A **false positive** is the verifier returning `1.0` for an answer that is actually wrong. Every false positive is poison: RLVR reinforces it, rejection sampling fine-tunes on it. Look for them in the checker you already have.

The `exact_match` numeric path compares float *values*. That is usually what you want (`"7.0" == "7"`), but floats also mean this:

```python
from llmre.reasoning.verifiers import exact_match
print(exact_match("7.0000000001", "7"))   # 0.0  -- fine, differs
print(exact_match("1e1", "10"))           # 1.0  -- "1e1" parses to 10.0: correct here, but
                                          #        watch for formats you didn't intend to accept
print(exact_match("inf", "inf"))          # 1.0  -- float("inf") == float("inf"); "inf" is a
                                          #        valid float in Python -> a degenerate accept
```

`float("inf")`, `float("nan")` (never equal — a *false negative*), scientific notation, and locale quirks all sneak through a naive numeric compare. A math verifier that strips a trailing period, or normalizes too aggressively, can accept `"12"` for gold `"1.2"`. Each is a false positive waiting to be exploited.

### 3.2 Gaming — the model exploits the checker, not the task

Even a *correct* verifier can be gamed if it checks less than you think:

- **Answer extraction is the weak point.** Real verifiers pull "the final answer" out of free-form text (e.g. the last number, or the contents of `\boxed{}`). A model can learn to emit a correct-looking boxed answer with nonsense reasoning, or print many numbers so the extractor grabs a right one. The verifier says `1.0`; the reasoning is garbage.
- **Code that passes weak tests.** If the verifier is a unit-test suite, the model can special-case the visible tests (`if input == known_case: return known_answer`) rather than solve the problem. The tests pass; the solution generalizes to nothing.
- **Format hacking.** If any part of the reward keys off format (as R1's does), the model over-produces the rewarded format.

<div class="callout warn"><p><strong>Verifier bugs are silent and self-amplifying.</strong> RLVR/rejection sampling optimize <em>toward whatever the verifier accepts</em>. A false positive isn't a one-off wrong grade — it becomes a training target the model is actively pushed to reproduce. Audit verifiers adversarially: try to <em>fool your own checker</em> before you train against it.</p></div>

### 3.3 Defenses

- **Make the parser strict, then test it against adversarial inputs** — the way `arithmetic_verifier` *raises* on unparseable problems instead of silently returning 0. Fail loud.
- **Prefer exact/symbolic checks** (value equality, CAS, comprehensive hidden tests) over fuzzy string matching wherever possible.
- **Separate answer-extraction from grading** and test extraction on its own; most gaming lives in extraction.
- **Hold out hidden tests** for code so the model can't special-case the visible ones.
- **Spot-check accepted completions by hand** during training — a rising correct-rate that comes from false positives is the failure mode you most need to catch early.

## Exercise

Find an input that makes `exact_match` return a **false positive** — accept two answers that a human grading a math problem would call different — using only its numeric path.

<details><summary>Hint</summary>

The numeric path does `float(pred) == float(gold)`. What strings does Python's `float()` accept besides plain decimals?

</details>

<details><summary>Stronger hint</summary>

`float("1e1")` is `10.0`; `float("inf")` and `float("+3")` are valid. Think about a format a student would never write as an answer but that `float()` happily parses to the same value.

</details>

<details><summary>Solution</summary>

Several work. Scientific notation collides with decimal:

```python
from llmre.reasoning.verifiers import exact_match
print(exact_match("1e2", "100"))     # 1.0  -- "1e2" -> 100.0
print(exact_match("1_000", "1000"))  # 1.0  -- Python float accepts underscores!
print(exact_match("inf", "inf"))     # 1.0  -- degenerate: "inf" is a legal float
```

`"1_000"` is the sharpest: Python's `float()` accepts underscore digit-group separators, so `float("1_000") == 1000.0`. If gold answers are ever formatted with separators (or a model learns to insert them), the numeric path can match strings a strict grader would reject — and, worse, `float("1,000")` raises rather than matching, so behavior is format-dependent and surprising. The lesson: **the numeric path inherits every quirk of `float()`.** A production verifier should normalize to a canonical numeric form deliberately, not lean on `float()`'s permissiveness. (None of these break the *toy* — `arithmetic_verifier` only ever passes `str(int)` as gold — but they show how a checker admits inputs you never intended once real model text flows through it.)

</details>

## Common mistakes

- **Assuming rejection sampling needs a correct answer for every prompt.** Prompts the model never solves contribute nothing; the kept set is empty. Like RLVR, it only amplifies reachable successes.
- **Trusting a rising correct-rate.** If the *verifier* is wrong, the rate rises because the model is learning to fool it. Always audit what "correct" is being measured.
- **Confusing outcome and process rewards.** An exact outcome verifier is un-gameable *on the answer* but still blind to the reasoning; a PRM scores steps but is a learned proxy again.
- **Over-normalizing in a verifier.** Every extra normalization (strip punctuation, coerce types, `float()` everything) is a new way to accept a wrong answer. Normalize deliberately and test adversarially.

## Check yourself

<details><summary>How does rejection-sampling fine-tuning differ from GRPO, given both use the same verifier?</summary>

Rejection sampling keeps only verifier-accepted completions and does ordinary SFT on them — no advantages, no clip, no KL; it discards wrong completions. GRPO uses the whole group: correct completions get positive advantage, wrong ones negative, and it takes a clipped policy-gradient step. Rejection sampling is simpler and very stable but throws away negative signal and has a lower ceiling.

</details>

<details><summary>Give one thing an outcome reward cannot distinguish that a process reward can.</summary>

An outcome reward gives the same score to a chain that reasons correctly and one that reasons wrongly but stumbles onto the right final answer (or a chain that's mostly right but derails at the last step). A process reward scores each step, so it can credit valid steps and penalize the invalid one, distinguishing lucky-right from genuinely-right.

</details>

<details><summary>Why is a false positive in a verifier especially dangerous during RL / rejection sampling?</summary>

Because training optimizes *toward whatever the verifier accepts*. A false positive isn't a single mis-grade — it becomes a target the model is actively pushed to reproduce, so the error compounds: the model learns to produce exactly the wrong-but-accepted outputs.

</details>

<details><summary>You SFT a model on its own verifier-accepted completions and the reported correct-rate climbs, but held-out accuracy drops. What's a likely cause?</summary>

The verifier has false positives (or is gameable): the model is learning to satisfy the checker rather than solve the task. The "accepted" completions include wrong reasoning the verifier mistakenly passed, so training on them hurts real accuracy. Audit the verifier and inspect accepted completions.

</details>

## Next

You now have the full Module 16 toolkit: GRPO (critic-free RL), RLVR (verifiable rewards), and rejection sampling — plus the judgment to distinguish outcome from process rewards and to distrust your own verifier. Module 17 turns these into modern reasoning models: chain-of-thought and self-consistency, process reward models and the DeepSeekMath training recipe in depth, and the DeepSeek-R1 case study that ties every piece together.

Continue to [17.1 · CoT, self-consistency, STaR](../module-17/lesson-01.md).
