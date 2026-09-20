# 17.3 · DeepSeek-R1 case study

<div class="prereq">
<p><strong>Prerequisites:</strong> process vs outcome reward and the GRPO recap from <a href="lesson-02.md">17.2 · Process reward & DeepSeekMath</a>; CoT, self-consistency, and the STaR loop from <a href="lesson-01.md">17.1 · CoT, self-consistency, STaR</a>; RLVR from <a href="../module-16/lesson-02.md">16.2 · RLVR</a>.</p>
<p><strong>You will learn:</strong> the two DeepSeek-R1 training paths — <strong>R1-Zero</strong> (pure RL on a base model, no SFT) and <strong>R1</strong> (cold-start SFT → reasoning RL → rejection-sampling SFT → final RL); what "reasoning emerges from RL" means and its failure modes; how R1's traces are <em>distilled</em> into small dense models; and how the runnable <code>mini_r1</code> maps stage-for-stage onto this pipeline so you can execute the control flow on a laptop.</p>
<p><strong>Why this matters for ML:</strong> R1 is the capstone of the whole reasoning track — it shows that large-scale RL with a <em>verifiable</em> reward, and nothing more, can grow long chain-of-thought and self-correction out of a base model. Understanding its stages (and which are documented vs inferred) is how you read any frontier reasoning report critically instead of taking the diagram on faith.</p>
</div>

## 1. Two paths, one insight

DeepSeek-R1 (DeepSeek-AI 2025) is really two results. The first, **R1-Zero**, is the striking one: take a base language model, apply *only* reinforcement learning with a rule-based verifiable reward — **no supervised fine-tuning first at all** — and watch reasoning behaviour emerge on its own. The second, **R1**, is the production system: the same RL engine wrapped in supervised stages before and after to fix R1-Zero's rough edges and make the model usable.

The shared insight, carried straight from 17.2: when the reward is *verifiable* (answer-correct + format checks, scored by a program, optimized with GRPO), you can run RL at enormous scale cheaply, and that scale is enough to *grow* reasoning rather than having to *demonstrate* it with human-written chains.

<div class="callout key"><p>R1-Zero: base model + verifiable-reward RL (GRPO), <em>no SFT</em> → reasoning emerges. R1: cold-start SFT → reasoning RL → rejection-sampling SFT → final RL, to make that reasoning readable and well-rounded. Both are driven by the same rule-based reward from 17.2.</p></div>

## 2. R1-Zero: reasoning from pure RL

### 2.1 The setup

Start from the DeepSeek-V3 base model. Do **no** SFT. Run GRPO directly, with a reward that is purely rule-based (**publicly documented** in the R1 report):

- **Accuracy reward** — parse the model's final answer and check it against the gold answer (for math) or run the code against tests. Correct → positive reward. This is the verifiable reward of 17.2 — a program, not a learned reward model.
- **Format reward** — the model must put its reasoning between `<think>...</think>` tags and its answer after, so the chain is machine-parseable. Well-formed → small reward.

No process reward model, no outcome reward model, no human preference data. Just a checker and GRPO's group baseline.

### 2.2 What emerges

Over RL training, two things rise together (the R1 report's headline figure, **publicly documented**):

- **Response length grows.** The model, entirely on its own, learns to write *longer* chains of thought — it discovers that spending more test-time compute (17.1) earns more reward on hard problems. No one told it to; longer correct chains simply got rewarded.
- **Accuracy grows**, and qualitatively new behaviours appear inside the chains: re-checking earlier steps, trying an alternative approach after a dead end, and the now-famous "wait, let me reconsider" self-correction the report calls an **aha moment**.

This is the important conceptual payoff: reasoning strategies were **not demonstrated** to the model. They were *incentivised* by a verifiable reward and *discovered* by search. STaR (17.1) hinted at this — filter for correct chains, imitate them — but R1-Zero shows on-policy RL can grow the behaviour from a base model with no reasoning examples at all.

### 2.3 Why R1-Zero alone is not enough

R1-Zero's chains work but read badly (**publicly documented** limitations in the report): they mix languages mid-chain, are poorly formatted, and are hard for a human to follow. Pure reward-hacking of a correctness signal optimises for *getting the answer*, not for *being readable or general*. That is exactly the gap the full R1 pipeline closes — and it is why cold-start SFT comes back.

## 3. R1: the multi-stage pipeline

The full R1 recipe (**publicly documented** as a four-stage pipeline; the exact data and hyperparameters are proprietary) wraps the R1-Zero RL engine in supervised stages:

1. **Cold-start SFT.** Fine-tune the base model on a small set of *curated* long chain-of-thought examples (thousands, human-friendly format). This gives RL a sane, readable starting point instead of a raw base model — fixing R1-Zero's readability problem before RL begins. (This is the "cold start" — priming the pump.)
2. **Reasoning RL.** Run GRPO with the verifiable reward (as in R1-Zero) on the cold-started model, now also rewarding language consistency so chains stop code-switching. Reasoning ability climbs while staying readable.
3. **Rejection-sampling SFT.** Use the RL'd model to *generate* many solutions, keep the correct and well-formed ones (this is best-of-$N$/rejection sampling from 17.2, and the filter is STaR's correctness filter from 17.1), and add non-reasoning data (writing, QA, safety). Fine-tune on this larger, broader curated set. This generalises the model beyond narrow verifiable tasks.
4. **Final RL.** A last RL stage over both reasoning (verifiable reward) and general (helpfulness/harmlessness) prompts, to align the all-round model.

Read the shape of it: **SFT to prime → RL to grow reasoning → SFT to broaden → RL to align.** Stages alternate supervised imitation and reward-driven search — supervision sets the format and breadth, RL discovers the capability.

<div class="callout key"><p>R1 = cold-start SFT (prime a readable format) → reasoning RL with verifiable reward (grow reasoning) → rejection-sampling SFT (broaden with self-generated correct traces + general data) → final RL (align). It is the STaR loop and DeepSeekMath's GRPO, industrialised and interleaved.</p></div>

## 4. Distillation: reasoning into small models

R1 is huge. To get its reasoning into models small enough to serve cheaply, DeepSeek used **distillation** (**publicly documented**): run R1 to generate a large set of high-quality reasoning traces, then plain-SFT *smaller dense* models (e.g. Qwen- and Llama-based, 1.5B–70B) on those traces. No RL on the small models — just supervised imitation of the big model's chains.

The reported finding matters: distilling R1's traces into a small dense model gives **better reasoning than running RL directly on that same small model**. The interpretation (partly **inference**): the large model's RL-discovered reasoning patterns are hard for a small model to *find* by RL on its own, but relatively easy to *imitate* once demonstrated. Search once at scale, then copy — a recurring theme (it is STaR's "generate then imitate" with R1 as the generator).

## 5. Mapping the pipeline to runnable code: `mini_r1`

You cannot run R1 on a laptop, but you *can* run its **control flow**. `llmre.reasoning.r1_pipeline.mini_r1` reproduces the *sequence of stages* on a toy task that finishes in milliseconds on CPU. It is a didactic skeleton, not a language model — the "policy" is a tiny table of logits, one row per problem, one column per candidate answer — but it exercises the exact two learning signals R1 uses.

### 5.1 The toy task and the verifier

Each "problem" has a few candidate answers; exactly one is correct. The **verifier** is a local, rule-based checker — the miniature of R1's accuracy reward, and deliberately *not* imported from Module 16's shared code:

```python
def toy_verifier(actions, correct):
    # Verifiable reward: 1.0 if the sampled answer matches the gold key, else 0.0.
    return (actions == correct).float()
```

The policy is a logit table (shape `(P, A)`, `float32`, CPU), initialised to zeros so the starting policy is uniform — it has learned nothing, mirroring a base model.

### 5.2 The stages, mapped

| R1 stage (this lesson) | `mini_r1` stage | What it does |
|---|---|---|
| base model | `base` | uniform policy, a reference point |
| cold-start SFT (§3.1) | `sft` | supervised cross-entropy on a *few curated* problems toward their known correct answer |
| reasoning RL (§3.2) | `rl` | for each problem sample a group, score with `toy_verifier`, GRPO group-relative advantage, policy-gradient update |

The `rl` stage is the heart of it — GRPO reduced to its core (from 17.2):

```python
probs   = softmax(logits)                       # current policy, (P, A)
actions = multinomial(probs, group_size)        # sample a group per problem, (P, G)
rewards = verifier(actions, correct)            # verifiable reward in {0,1}, (P, G)
adv     = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + eps)
loss    = -(adv.detach() * log_prob(actions)).mean()   # REINFORCE/GRPO update
```

That is the group-relative advantage of 17.2 (`(r - mean)/std`) driving a policy-gradient step, with the group mean as the critic-free baseline. A fully-solved problem yields an all-correct group, advantage $\approx 0$, and *no* update — so the RL stage never pushes a solved problem back toward a wrong answer, which is why toy accuracy is monotonic across the stage.

### 5.3 Run it

```python
from llmre.reasoning.r1_pipeline import mini_r1
from pprint import pprint
pprint(mini_r1())
```

A representative run (default settings, seed 0) prints per-stage metrics like:

```
base: accuracy 0.333, expected_reward 0.250
sft : accuracy 0.500, demo_loss 1.386 -> 0.071
rl  : accuracy 0.500 -> 1.000, expected_reward 0.477 -> 0.956
```

Read it as the R1 story in miniature. The **base** uniform policy is right on $1/3$ of problems (chance is $1/A = 1/4$; it lands at $2/6$ here). **Cold-start SFT** drives the demo problems' loss from $1.386$ (that is $\ln 4$, a uniform distribution over 4 answers) down to $0.071$ and lifts accuracy to $0.5$ — but only on the curated demos. Then **reasoning RL**, using nothing but the verifiable reward, lifts accuracy from $0.5$ to $1.0$ on *all* problems, including the ones SFT never saw — reasoning "emerging" from reward alone, exactly R1-Zero's phenomenon shrunk to a toy.

<div class="callout pt"><p>These numbers are produced by actually running <code>mini_r1()</code>. The RL stage's <code>accuracy_after &gt;= accuracy_before</code> holds by construction (solved problems stop being updated), and the test suite checks it across several seeds. The code is heavily commented to flag which line maps to which R1 stage.</p></div>

### 5.4 What the toy does and does not show

**Honest scope.** `mini_r1` faithfully reproduces the *pipeline structure* — cold-start SFT then verifiable-reward RL, with a critic-free group-relative update — and the *qualitative* result that RL improves problems SFT never touched. It does **not** reproduce R1's substance: there is no language model, no long chain of thought, no emergent self-correction, no distillation stage. The "reasoning" is choosing one of four answers, not writing a derivation. It is a control-flow scale model, useful for seeing the stages fit together, not a reproduction of the capability.

<div class="hw">
<p><strong>Hardware track.</strong></p>
<p><strong>The toy (<code>mini_r1</code>):</strong> CPU-only, no GPU needed. Runtime &lt; 1 second, memory a few MB (the policy is a <code>(6, 4)</code> float tensor). Runs anywhere Python + PyTorch install. This is all you need for the lesson and the tests.</p>
<p><strong>Real R1-scale training (for context — do NOT attempt):</strong> the base is DeepSeek-V3, a 671B-parameter (~37B active) MoE model. Pretraining V3 alone was reported at ~2.79M H800 GPU-hours; the R1 RL and SFT stages add large-scale generation and training on top, requiring a multi-node cluster of hundreds of high-memory GPUs, weeks of wall-clock time, and enormous verified-problem datasets. The point of <code>mini_r1</code> is precisely that you can understand and run the <em>structure</em> for free while the real thing is a datacenter-scale effort.</p>
</div>

## 6. Documented vs inferred — read the report critically

Frontier-lab honesty (a rule of this course): keep straight what the R1 report *states* versus what we *infer*.

- **Publicly documented (in the R1 report):** R1-Zero uses pure RL from a base model with rule-based accuracy + format rewards; reasoning behaviours and growing response length emerge over RL; the full R1 uses a cold-start-SFT → RL → rejection-sampling-SFT → RL pipeline; R1 traces are distilled into smaller dense models and this beats RL on those small models directly.
- **Inference / not claimed:** the exact cold-start dataset, the precise reward weights, hyperparameters, and data mixtures are proprietary and *not* fully disclosed — our stage descriptions are the documented structure, not a recipe you could copy line-for-line. The interpretation of *why* distillation beats small-model RL (large-model patterns are easier to imitate than to rediscover) is our reasoning, not a proven claim.
- **We do NOT claim** this is OpenAI's or Anthropic's reasoning-model pipeline. R1 is DeepSeek's *published* system; other labs' reasoning models (o-series, etc.) are not publicly documented at this level and may differ substantially. `mini_r1` is a teaching reconstruction of the R1 report's structure, nothing more.

<div class="callout paper"><p><strong>Research connection.</strong> The paper for this lesson is <em>DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning</em> (DeepSeek-AI 2025), entry 26 in <a href="../../papers/index.md">the paper curriculum</a>. It pulls together everything in Modules 16–17: GRPO (DeepSeekMath, 17.2), verifiable rewards (16.2), the STaR generate-filter-imitate loop (17.1), and process-vs-outcome supervision (17.2). See <a href="../../curriculum/course-outline.md">the curriculum outline</a> for how the reasoning track connects to the tool-use and agents modules that follow.</p></div>

## Common mistakes

- **Believing R1-Zero used no reward.** It used no *SFT* and no *learned reward model*, but it absolutely used a reward — a rule-based *verifiable* one (answer-correct + format). "Pure RL" means pure of supervision and reward *models*, not pure of reward.
- **Reading the four-stage diagram as a copyable recipe.** The stage *structure* is documented; the datasets, reward weights, and hyperparameters are not. Do not assert exact numbers.
- **Assuming distillation used RL on the small models.** It did not — distillation is plain SFT on R1's generated traces. That is the surprising part: imitation beat small-model RL.
- **Over-claiming from `mini_r1`.** It reproduces the pipeline's *control flow*, not R1's reasoning capability. It has no language model and no chain of thought.
- **Attributing R1's pipeline to other labs.** Only DeepSeek's R1 is documented at this level; do not claim it is how any other lab's reasoning model is trained.

## Exercise

In `mini_r1`, the cold-start SFT stage only fine-tunes on a *few* curated problems (`n_demos`), yet the later RL stage lifts accuracy on *all* problems to near-perfect. If you set `n_demos=0` (skip cold-start entirely), which R1 variant are you now simulating, and would you expect the RL stage to still work?

*Hint:* re-read section 2 on R1-Zero, and recall the RL stage samples and scores every problem regardless of whether SFT touched it.

<details><summary>Solution</summary>

With `n_demos=0` you skip cold-start SFT and go straight to verifiable-reward RL on the base (uniform) policy — that is **R1-Zero** (pure RL, no SFT). You would expect RL to still work, because the RL stage samples a group for *every* problem and the verifiable reward gives signal wherever the group is mixed (some correct, some wrong); it never needed the SFT demos. Running it confirms the RL stage still drives accuracy up from the uniform base — the toy analogue of R1-Zero growing reasoning from a base model with no SFT. (In the real system the difference cold-start makes is *readability*, which the toy has no notion of — the toy only measures whether the right answer is chosen, so R1-Zero-style and R1-style reach the same toy accuracy.)

</details>

## Check yourself

<details><summary>What is the single most important claim of R1-Zero, and what did it deliberately leave out?</summary>

That reasoning behaviour — long chains of thought, self-correction, growing response length — can *emerge* from pure reinforcement learning on a base model with only a rule-based verifiable reward. It deliberately left out any supervised fine-tuning (no curated CoT demonstrations first) and any learned reward model (ORM/PRM). The reasoning was incentivised by a checker and discovered by RL search, not demonstrated.

</details>

<details><summary>Why does the full R1 add cold-start SFT before the RL that R1-Zero showed already works?</summary>

Because R1-Zero's RL-only chains, while correct, read badly — mixed languages, poor formatting, hard to follow. Cold-start SFT on a small curated set of readable long-CoT examples gives the RL stage a clean, human-friendly starting format, so the reasoning that RL then grows is usable. Cold start trades a little supervision for a large gain in readability and controllability.

</details>

<details><summary>Map each <code>mini_r1</code> stage (<code>base</code>, <code>sft</code>, <code>rl</code>) to an R1 concept, and say which line of the RL update is the "critic-free baseline."</summary>

<code>base</code> = the untrained base model (uniform policy). <code>sft</code> = cold-start SFT (supervised cross-entropy on curated demo problems). <code>rl</code> = reasoning RL with a verifiable reward via GRPO. The critic-free baseline is the group mean subtracted in the advantage: <code>adv = (rewards - rewards.mean(-1)) / (rewards.std(-1) + eps)</code> — the group's own mean reward replaces PPO's learned value network.

</details>

<details><summary>Distilling R1's traces into a small dense model beat running RL on that small model directly. What does that suggest, and what part of that explanation is inference rather than documented?</summary>

It suggests that the reasoning patterns a large model discovers through RL are hard for a small model to *find* by RL on its own, but relatively easy to *imitate* once the large model has demonstrated them — so "search once at scale, then copy" is more effective than "search at small scale." The empirical result (distillation beats small-model RL) is **documented**; the *why* (patterns are easier to imitate than to rediscover) is our **inference/interpretation**, not a proven claim in the report.

</details>

<details><summary>Someone says "R1 proves OpenAI trains its reasoning models with GRPO and no SFT." What is wrong with that statement?</summary>

Two things. First, even for R1, "no SFT" is only true of R1-*Zero*; the full R1 uses cold-start SFT and rejection-sampling SFT. Second, R1 is *DeepSeek's* published system — it says nothing about how OpenAI (or any other lab) trains its reasoning models, which are not publicly documented at this level and may use entirely different methods. The correct framing: R1 is a documented example of one lab's verifiable-reward RL pipeline, not a claim about anyone else's.

</details>

## Next

You have reconstructed the full DeepSeek-R1 pipeline, run its control flow in `mini_r1`, and learned to separate what a frontier report documents from what you infer. That completes the reasoning track (Modules 16–17): from CoT prompting, through verifiable-reward RL and GRPO, to a modern reasoning model end to end. The course now turns from *how a model reasons* to *how a model acts* — calling tools and running as an agent.

Continue to [18.1 · Tool use](../module-18/lesson-01.md).
