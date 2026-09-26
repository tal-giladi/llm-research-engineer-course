# 17.1 · CoT, self-consistency, STaR

<div class="prereq">
<p><strong>Prerequisites:</strong> the evaluation harness and how we score answers from <a href="#/lessons/module-13/lesson-01">13.1 · Evaluation</a>; reasoning RL and verifiable rewards from <a href="#/lessons/module-16/lesson-01">16.1 · Reasoning RL</a>; sampling from a language model (temperature, top-k) from <a href="#/lessons/module-06/lesson-01">06.1 · Assembling GPT-2</a>.</p>
<p><strong>You will learn:</strong> what a <em>chain of thought</em> is and the two mechanical reasons it raises accuracy (more test-time compute, and problem decomposition); how <em>self-consistency</em> turns many sampled chains into one answer by majority vote — worked by hand and reproduced exactly with our <code>self_consistency</code> helper; and how <em>STaR</em> promotes chain-of-thought from a prompting trick into a self-improvement training loop.</p>
<p><strong>Why this matters for ML:</strong> every modern reasoning model — including the DeepSeek-R1 pipeline you will rebuild in miniature in 17.3 — is built on these three ideas stacked. CoT is the behaviour, self-consistency is the cheapest way to cash it in at inference, and STaR is the first "let the model teach itself to reason" loop that RL later generalises.</p>
</div>

## 1. Intuition: make the model think out loud

Ask a small language model "What is $17 \times 23$?" and force it to answer in one token and it will usually guess wrong. Ask the same model to *show its work* — "$17 \times 23 = 17 \times 20 + 17 \times 3 = 340 + 51 = 391$" — and it gets it right far more often. Nothing about the weights changed. The only difference is that we let the model emit intermediate steps before the final number.

That string of intermediate steps is a **chain of thought** (CoT). Wei et al. (2022) showed that simply prompting a large model with a few worked examples that include their reasoning makes it continue in the same "reason, then answer" style, and that this dramatically improves arithmetic, commonsense, and symbolic reasoning — with **no training at all**, purely a change in the prompt.

Two mechanical reasons explain why writing steps helps. Keep both in mind; they recur in every later lesson.

- **More test-time compute.** A Transformer does a *fixed* amount of computation per token: one forward pass, the same number of layers, no loops. A hard problem may need more sequential computation than one token's worth. Every CoT token is another full forward pass whose output feeds the next — so a 50-token chain buys the model roughly 50× the sequential compute of a blurt-it-out answer. The scratchpad *is* the extra thinking.
- **Decomposition.** Each step conditions on the steps already written. "$340 + 51$" is an easy sub-problem once "$17\times20=340$" and "$17\times3=51$" are on the page. CoT breaks one hard conditional $p(a \mid x)$ into a chain of easy ones.

<div class="callout key"><p>Chain-of-thought = let the model write intermediate steps before the final answer. It is a <em>prompting</em> change, not a training change, and it works because it buys more sequential compute and decomposes the problem into easier conditioned sub-steps.</p></div>

## 2. Mathematics: CoT as marginalising over reasoning paths

Write $x$ for the question, $a$ for the final answer, and $z$ for the reasoning chain (the intermediate steps). A model that must answer in one shot is asked to model $p(a \mid x)$ directly. A CoT model instead generates a chain $z$ and then the answer, and the probability of the answer is a sum over all the chains that could have produced it:

$$
p(a \mid x) = \sum_{z} p(a \mid x, z)\, p(z \mid x).
$$

Every symbol: $p(z \mid x)$ is how likely the model is to write reasoning chain $z$ given the question; $p(a \mid x, z)$ is how likely it is to state answer $a$ once that chain is on the page; the sum runs over all possible chains $z$. In words: the answer's probability is an average of "answer given this reasoning" weighted by "how likely this reasoning is."

We cannot actually enumerate every chain $z$ — there are exponentially many token sequences. Two practical strategies approximate this sum, and they define the rest of the lesson:

- **Greedy CoT** takes a single high-probability chain $\hat z$ (temperature near 0) and reads off $p(a \mid x, \hat z)$. One sample from the sum.
- **Self-consistency** (section 4) draws *many* chains $z_1, \dots, z_N$, extracts each one's answer, and lets them **vote** — a Monte-Carlo estimate of which answer the sum concentrates on.

## 3. Building a CoT prompt in code

Before we can vote, we need to turn the model's text into an answer. Our `llmre.reasoning.cot` module has three dependency-free helpers (they touch only Python strings — no GPU, no model), which we build on for the rest of the module.

`cot_prompt` assembles a few-shot prompt: a handful of worked exemplars in "question → reasoning → `#### answer`" form, then the new question with an open `A:` so the model continues in the same style.

```python
from llmre.reasoning.cot import cot_prompt

exemplars = [
    ("What is 2+3?", "2 plus 3 is 5.", "5"),
    ("What is 4+1?", "4 plus 1 is 5.", "5"),
]
print(cot_prompt("What is 17*23?", exemplars))
```

```
Q: What is 2+3?
A: 2 plus 3 is 5. #### 5

Q: What is 4+1?
A: 4 plus 1 is 5. #### 5

Q: What is 17*23?
A:
```

The exemplars teach the *format* (show steps, then `#### <answer>`); the model, primed by them, produces its own steps and its own `#### ...` line for the new question. The `####` marker is the GSM8K convention; you could equally end each exemplar with "The answer is 391."

## 4. Self-consistency: sample many chains, take the majority

A single greedy chain is one draw from a noisy process — if the model takes one wrong turn, the whole answer is wrong. **Self-consistency** (Wang et al. 2022) exploits a simple asymmetry: there are *many* wrong reasoning paths but they tend to disagree with each other, while *correct* paths tend to converge on the same answer. So if we sample several chains and take the answer the majority reach, the correct answer wins even when any single chain is unreliable.

The recipe:

1. Build one CoT prompt for the question.
2. Sample $N$ completions at a **non-zero temperature** (so the chains differ — temperature 0 would give $N$ identical chains and defeat the point).
3. Extract the final answer from each completion.
4. Return the answer that appears most often (majority vote).

### 4.1 Extracting each answer

`extract_answer` pulls the final answer out of a completion. It takes the **last** marker match, because a chain mentions numbers along the way and only the last one is the stated answer.

```python
from llmre.reasoning.cot import extract_answer

extract_answer("17*20=340, 17*3=51, 340+51=391. #### 391")   # -> "391"
extract_answer("First I get 12, but rechecking #### 18")      # -> "18"  (last marker)
extract_answer("no marker at all")                            # -> None
```

That second case is the whole point of "last match": the chain second-guesses itself ("First I get 12") and the parser correctly keeps the *final* committed answer, `18`, not the abandoned `12`.

### 4.2 The vote, worked by hand

Suppose we sample five chains for a GSM8K-style word problem and extract these answers:

$$
[\,18,\ 18,\ 12,\ 18,\ 26\,].
$$

Tally the votes: $18$ appears **three** times, $12$ once, $26$ once. The majority answer is $18$ with $3$ votes. Two of the five chains went astray (to $12$ and $26$) but disagreed with each other, while the three correct chains agreed — exactly the asymmetry self-consistency banks on. Note that greedy decoding might have returned the very first chain; if that chain had been the one that produced $12$, plain greedy CoT would be wrong here while self-consistency is right.

Our `self_consistency` helper does exactly this tally and returns `(answer, count)`:

```python
from llmre.reasoning.cot import self_consistency

self_consistency(["18", "18", "12", "18", "26"])   # -> ('18', 3)
```

<div class="callout pt"><p>Verified against the real helper: running the line above prints <code>('18', 3)</code>. The function takes any iterable of <em>hashable</em> answers — the strings <code>extract_answer</code> returns — counts them with a <code>collections.Counter</code>, and returns the top answer and its vote count.</p></div>

### 4.3 Ties are broken deterministically

What if two answers tie for the lead? A dictionary/hash iteration order would make the winner unpredictable across runs — a reproducibility bug. `self_consistency` breaks ties by **first appearance in the input**: among the answers sharing the top count, the one that appeared earliest wins.

$$
[\,18,\ 12,\ 18,\ 12\,] \;\Rightarrow\; 18\text{ and }12\text{ both have count }2 \;\Rightarrow\; \textbf{18 wins (seen first)}.
$$

```python
self_consistency(["18", "12", "18", "12"])   # -> ('18', 2)   first-seen wins the tie
self_consistency(["12", "18", "12", "18"])   # -> ('12', 2)   reorder -> deterministic flip
```

Both lines are verified: the winner flips with input order, never with hash order, so the same input always gives the same output.

### 4.4 Under the hood — cost

Self-consistency is a straight compute-for-accuracy trade. $N$ samples cost $N$ full generations, so a 40-sample self-consistency run is 40× the inference cost of one greedy chain. Accuracy rises quickly with the first few samples and then flattens (diminishing returns), so in practice you pick $N$ where the curve bends — often 5–40. This is your first taste of **test-time scaling**: spend more compute at inference to get a better answer, no retraining. R1-style long-CoT models (17.3) push the same lever a different way — one very long chain instead of many short ones.

## 5. STaR: bootstrap reasoning into the weights

Self-consistency spends extra compute *every time you answer*. STaR (**S**elf-**Ta**ught **R**easoner, Zelikman et al. 2022) asks a different question: can the model *train on its own correct reasoning* so a single cheap chain becomes reliable? It turns CoT from a prompting trick into a self-improvement loop, and it is the direct conceptual ancestor of the RL reasoning pipelines in Module 16 and R1.

The catch STaR solves: we have questions with known **final answers** (e.g. GSM8K), but we do *not* have gold **reasoning chains** to train on. STaR generates the chains itself and keeps only the ones that work.

### 5.1 The loop

Given a dataset of (question, correct-answer) pairs and a base model:

1. **Generate.** For each question, prompt the model (few-shot CoT) to produce a reasoning chain and a final answer.
2. **Filter by correctness.** Keep the chain only if its extracted answer matches the known correct answer. A right answer is a *cheap proxy* that the reasoning was probably sound — no human had to label the steps.
3. **Fine-tune.** Supervised fine-tune the model on the kept (question, chain, answer) triples. It learns to imitate its own successful reasoning.
4. **Repeat.** The fine-tuned model generates better chains next round, solving problems it previously missed. Iterate.

<div class="callout key"><p>STaR loop = <strong>generate</strong> rationales → <strong>filter</strong> to the ones that reach the correct answer → <strong>fine-tune</strong> on them → repeat. Correctness of the final answer is the free supervision signal that lets the model bootstrap reasoning without any human-written chains.</p></div>

### 5.2 Rationalization: the fix for problems it always misses

There is a gap in the loop. A problem the model *never* gets right produces *no* correct chains, so it never contributes training data — and stays unsolved forever. STaR's fix is **rationalization**: for a missed problem, feed the model the **correct answer as a hint** and ask it to produce a chain that arrives there. A chain generated with the answer in hand is far more likely to be valid, and once fine-tuned on it the model can often reach that answer on its own next round, hint removed. Rationalization is how STaR pulls itself up on the hard tail.

### 5.3 Why this is the seed of RL

Look at the STaR loop as a reward loop and the connection to Module 16 is exact. "Filter by correctness" is a **binary verifiable reward** (1 if the answer is right, else 0). "Fine-tune on the kept chains" is a crude policy-improvement step that up-weights high-reward trajectories. STaR is essentially the simplest possible policy-gradient method — keep the winners, imitate them — done as offline SFT rounds. Replace that with a proper on-policy RL update and a group-relative baseline and you have **GRPO** (17.2), the engine of DeepSeek-R1 (17.3).

<div class="callout paper"><p><strong>Research connection.</strong> The two founding papers of this lesson: <em>Chain-of-Thought Prompting Elicits Reasoning in Large Language Models</em> (Wei et al. 2022) and <em>STaR: Bootstrapping Reasoning With Reasoning</em> (Zelikman et al. 2022). Self-consistency is <em>Self-Consistency Improves Chain of Thought Reasoning</em> (Wang et al. 2022). See the reading guides in <a href="#/papers/index">the paper curriculum</a> (entries 22 and 23), which place them in the dependency graph leading to R1. The overall module map is in <a href="#/curriculum/course-outline">the curriculum outline</a>.</p></div>

## Common mistakes

- **Sampling self-consistency at temperature 0.** Zero temperature makes every chain identical, so the "vote" is $N$ copies of one answer — no diversity, no benefit. Self-consistency needs a non-zero temperature so the chains genuinely differ.
- **Extracting the first number instead of the last.** A chain says "I first thought 12… actually #### 18." Grabbing the first number gives the abandoned guess. `extract_answer` deliberately takes the *last* marker.
- **Training STaR on unfiltered chains.** If you fine-tune on chains regardless of correctness, you teach the model its own mistakes. The correctness filter is not optional — it *is* the supervision.
- **Forgetting rationalization.** Without it, STaR can only ever reinforce problems it already solves, and the hard tail never improves.

## Exercise

You sample four CoT chains for one question and `extract_answer` returns `["42", None, "42", "7"]`. One chain had no parseable answer (`None`). What does `self_consistency(["42", None, "42", "7"])` return, and is feeding `None` into the vote a good idea?

*Hint:* `None` is a perfectly hashable Python value, so `Counter` will happily count it.

*Stronger hint:* count each distinct value; `"42"` appears twice, `None` once, `"7"` once.

<details><summary>Solution</summary>

It returns `('42', 2)` — `"42"` has the majority with two votes; `None` and `"7"` have one each. The call succeeds because `None` is hashable, but counting `None` as if it were a real answer is usually a mistake: a chain that failed to produce any answer should be **dropped before voting**, not allowed to split the vote. In a real harness you would filter out `None` results first: `answers = [a for a in raw if a is not None]`, then call `self_consistency(answers)`. Here it happens not to change the winner, but with `["42", None, None, "7"]` the unfiltered vote would tie `None` with the reals and (by first-seen tie-break) could return `None`.

</details>

## Check yourself

<details><summary>CoT is "just a prompting change" yet it raises accuracy. Where does the extra problem-solving power come from if the weights are unchanged?</summary>

From two sources. (1) More test-time compute: a Transformer spends a fixed amount of computation per token, so emitting more tokens (the chain) gives the model more sequential forward passes to work with before it commits to an answer. (2) Decomposition: each step conditions on the previous ones, turning one hard conditional $p(a\mid x)$ into a sequence of easy sub-steps $p(a\mid x,z)$. No weight update is needed to use compute the model already had.

</details>

<details><summary>Why does self-consistency require a non-zero sampling temperature?</summary>

Majority voting only helps if the sampled chains actually differ. At temperature 0 decoding is (near) deterministic, so all $N$ chains are identical and the vote is $N$ copies of one answer — no better than a single greedy chain. A non-zero temperature produces diverse reasoning paths; correct ones tend to converge on the same answer while wrong ones scatter, and the majority picks out the convergent (usually correct) answer.

</details>

<details><summary>Trace it: <code>self_consistency(["A", "B", "A", "B", "C"])</code> returns what, and why is it reproducible?</summary>

`A` and `B` each appear twice, `C` once. The top count is 2, tied between `A` and `B`. The tie-break is first appearance in the input, and `A` appears before `B`, so it returns `('A', 2)`. It is reproducible because the tie is resolved by *input order*, not by dictionary/hash iteration order — the same list always yields the same result.

</details>

<details><summary>In STaR, why is filtering generated chains by final-answer correctness enough, given no one labelled the reasoning steps?</summary>

A correct final answer is a cheap, automatic proxy for "the reasoning was probably sound." It is not perfect — a chain can reach the right answer by luck or flawed logic — but at scale, chains that reach the correct answer are much more likely to contain valid reasoning than chains that don't, so fine-tuning on the filtered set improves reasoning on average. This is precisely a binary verifiable reward (1 if correct, else 0), which is why STaR is the conceptual seed of verifiable-reward RL.

</details>

<details><summary>What problem does STaR's <em>rationalization</em> step solve, and how?</summary>

It fixes the hard tail: problems the model never answers correctly generate no correct chains, so they never enter the fine-tuning set and never improve. Rationalization gives the model the correct answer as a hint and asks it to produce a chain that reaches it. Those hinted chains are usually valid; fine-tuning on them lets the model later solve the problem without the hint, so the loop can make progress on problems it initially always missed.

</details>

## Next

You can now make a model reason out loud, cash in that reasoning cheaply with a majority vote, and bootstrap it into the weights with STaR. But "the final answer is correct" is a blunt signal — it says nothing about *where* a long chain went wrong. The next lesson sharpens the reward from the whole outcome down to individual steps, and recaps the GRPO algorithm that turns a verifiable reward into a policy update.

Continue to [17.2 · Process reward & DeepSeekMath](lessons/module-17/lesson-02.md).
