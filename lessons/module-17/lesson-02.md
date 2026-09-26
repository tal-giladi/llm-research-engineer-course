# 17.2 · Process reward & DeepSeekMath

<div class="prereq">
<p><strong>Prerequisites:</strong> chain-of-thought and self-consistency from <a href="#/lessons/module-17/lesson-01">17.1 · CoT, self-consistency, STaR</a>; reinforcement learning with verifiable rewards (RLVR) from <a href="#/lessons/module-16/lesson-02">16.2 · RLVR</a>; the Bradley-Terry reward model and PPO from <a href="#/lessons/module-15/lesson-01">15.1 · RLHF</a>.</p>
<p><strong>You will learn:</strong> the difference between an <em>outcome</em> reward (score only the final answer, ORM) and a <em>process</em> reward (score every reasoning step, PRM); why "Let's Verify Step by Step" found PRMs win on hard math and what they cost; how a PRM reranks best-of-$N$ samples — worked by hand; and how DeepSeekMath's GRPO recipe applies a <em>verifiable</em> reward to math with no learned reward model and no value network.</p>
<p><strong>Why this matters for ML:</strong> the choice of reward granularity is the central design decision of a reasoning trainer. ORM is cheap and scalable but noisy on long chains; PRM is precise but expensive to label. DeepSeekMath's answer: skip the learned reward model entirely when the answer is checkable, and use GRPO — the exact engine DeepSeek-R1 (17.3) runs at scale.</p>
</div>

## 1. Intuition: grade the answer, or grade the working?

In 17.1 the only signal was "is the final answer correct?" — a single bit for the whole chain. That bit is a blunt instrument for a 30-step derivation. Consider two failure modes it cannot tell apart:

- A chain reasons perfectly for 29 steps and makes one arithmetic slip at the end. Final answer wrong → reward 0. The 29 good steps get no credit.
- A chain makes an error in step 3, then a second error that *cancels* it, and stumbles onto the right final number. Final answer right → reward 1. The model is rewarded for broken reasoning.

This is the tension between two kinds of reward model:

- **Outcome Reward Model (ORM)** — scores only the *final answer*. One number per solution. Cheap to label (you just need the gold answer) and to compute, but blind to *where* a chain went right or wrong.
- **Process Reward Model (PRM)** — scores *each step* of the chain. A vector of per-step correctness signals. It knows the chain broke at step 3, or that step 29 was the only slip. Far more informative — and far more expensive, because someone (or something) must label every step.

<div class="callout key"><p>ORM grades the destination; PRM grades every turn of the journey. ORM is cheap and scalable but noisy on long chains; PRM is precise but needs step-level labels. The rest of the lesson is about when the extra cost of PRM pays off, and how to sidestep <em>both</em> when the answer is machine-checkable.</p></div>

## 2. "Let's Verify Step by Step": process supervision wins on hard math

Lightman et al. (2023) ran the ORM-vs-PRM comparison head to head on the hard MATH dataset. They trained a PRM on **PRM800K** — 800k *human* step-level labels, each step marked good/neutral/bad — and an ORM on final-answer correctness, then used each as a **verifier** to rerank many sampled solutions (best-of-$N$, section 3).

Their headline result: as you sample more candidate solutions per problem, the PRM-selected answer beats the ORM-selected answer by a widening margin. Two reasons:

- **Better credit assignment.** The PRM can reject a solution the moment a step is wrong, even if the final answer happens to look plausible; the ORM cannot see inside.
- **Robust to lucky guesses.** On hard problems, the fraction of sampled chains that reach the right answer by *flawed* reasoning is non-trivial. The ORM rewards them; the PRM catches the broken step and down-weights them.

The trade-offs, stated honestly:

- PRM needs step-level labels. PRM800K is a large, expensive **human** annotation effort — the main cost of the approach.
- ORM needs only the gold final answer, which for math/code you often already have — essentially free.
- The gap between them grows with problem difficulty and with $N$: for easy problems or small $N$, ORM is often good enough.

This sets up the key question DeepSeekMath answers: on tasks where the answer is *automatically* checkable, do we need a learned reward model — ORM or PRM — at all?

## 3. Using a verifier: best-of-$N$ reranking, worked by hand

Both ORM and PRM are most simply used at **inference** as a *verifier* that reranks samples. Sample $N$ solutions, score each with the verifier, keep the top-scoring one. This is **best-of-$N$** (also called verifier-guided or rejection sampling).

The interesting part is how a PRM turns a *vector* of per-step scores into one number to rank by. Two standard aggregations:

- **Product** of per-step correctness probabilities — the model's estimate that *every* step is correct, assuming (roughly) independence: $\prod_t s_t$.
- **Minimum** per-step score — the solution is only as strong as its weakest step: $\min_t s_t$.

### 3.1 A worked example

Two candidate solutions to the same problem, each three steps, with PRM per-step correctness probabilities:

$$
\text{Solution A: } [\,0.90,\ 0.80,\ 0.95\,], \qquad
\text{Solution B: } [\,0.99,\ 0.40,\ 0.90\,].
$$

Both, say, produce the *same final answer*, so an **ORM would score them identically** and pick arbitrarily. The PRM sees more. Aggregate by **product**:

$$
\text{A: } 0.90 \times 0.80 \times 0.95 = 0.684, \qquad
\text{B: } 0.99 \times 0.40 \times 0.90 = 0.3564.
$$

Aggregate by **minimum**:

$$
\text{A: } \min(0.90, 0.80, 0.95) = 0.80, \qquad
\text{B: } \min(0.99, 0.40, 0.90) = 0.40.
$$

Both aggregations rank **A above B**, and for the same reason: B's step 2 is shaky ($0.40$) — the PRM suspects B took a wrong turn in the middle and merely recovered the right-looking answer. The ORM, seeing only the final answer, is blind to that. This is the "lucky guess / cancelling errors" case from section 1, caught.

<div class="callout pt"><p>Verified in Python: the four products/minima above are exactly <code>0.684</code>, <code>0.3564</code>, <code>0.80</code>, <code>0.40</code>. Product punishes many small doubts multiplicatively; minimum keys on the single weakest step. Product is the more common default; minimum is more conservative about any single bad step.</p></div>

### 3.2 Relation to self-consistency

Self-consistency (17.1) is best-of-$N$ with a *free* verifier: "how many other samples agree with you." A trained PRM/ORM is a *learned* verifier. They compose — you can majority-vote **and** weight each vote by its verifier score ("weighted self-consistency"). All three are test-time scaling: spend more inference compute to pick a better answer.

## 4. DeepSeekMath: verifiable reward + GRPO, no reward model

Lightman showed PRMs help but cost human labels. DeepSeekMath (Shao et al. 2024) took the opposite, cheaper route for tasks where the answer is *machine-checkable*: **drop the learned reward model entirely** and use a **rule-based verifiable reward**, optimized with a new, lightweight RL algorithm — **GRPO**.

### 4.1 The verifiable reward

For a math problem with a known gold answer, you do not need an ORM or a PRM to *judge* a solution — you can just **check** it: parse the model's final answer and compare to the gold answer (plus a format check that the chain is well-formed). Reward $1$ if correct, $0$ if not. This is the RLVR idea from 16.2: the reward is a *deterministic program*, not a learned network, so it is free to run, cannot be gamed by fooling a reward model, and has no reward-model training cost. The price is that it only applies where answers are checkable — math, code with tests, formal proofs — not open-ended writing.

### 4.2 GRPO recap: PPO without a critic

PPO (from 15.1) needs a learned **value network** (critic) the same size as the policy to estimate a baseline for the advantage — doubling model memory. **GRPO** (Group Relative Policy Optimization) removes it with one idea: for each prompt, sample a **group** of outputs and use the *group's own mean reward* as the baseline. No critic, roughly half the memory.

For a prompt, sample a group of $G$ outputs, get their rewards $r_1, \dots, r_G$, and standardise each within the group into an **advantage**:

$$
\hat A_i = \frac{r_i - \operatorname{mean}(r_{1:G})}{\operatorname{std}(r_{1:G})}.
$$

Every symbol: $r_i$ is output $i$'s (verifiable) reward; $\operatorname{mean}(r_{1:G})$ is the group's average reward — the baseline that replaces PPO's critic; $\operatorname{std}(r_{1:G})$ normalises the scale. An output better than its group-mates gets a positive advantage (push the policy toward it); worse than average gets negative (push away). These advantages plug into the same PPO-style clipped objective with a KL-to-reference penalty you saw in 15.1 — just with the group baseline instead of a value net.

### 4.3 A worked group

Sample a group of $G = 4$ solutions to one problem; the verifier returns rewards

$$
r = [\,1,\ 1,\ 0,\ 0\,] \quad(\text{two correct, two wrong}).
$$

Group mean $= (1+1+0+0)/4 = 0.5$. Using the population standard deviation over the group, $\operatorname{std} = 0.5$. The advantages are

$$
\hat A = \frac{[\,1,1,0,0\,] - 0.5}{0.5} = [\,+1,\ +1,\ -1,\ -1\,].
$$

The two correct solutions get advantage $+1$ (the update raises their probability), the two wrong ones get $-1$ (their probability is lowered) — all measured *relative to this group*, with no value network anywhere. If the whole group were correct ($r=[1,1,1,1]$) the mean would be $1$, every $r_i - \text{mean} = 0$, and the advantage would be zero: a fully-solved prompt produces no update. That self-limiting behaviour is exactly what keeps the miniature R1 in 17.3 from regressing on problems it has already learned.

<div class="callout pt"><p>Verified in Python: for <code>r=[1,1,0,0]</code>, mean is <code>0.5</code>, population std is <code>0.5</code>, and the standardised advantages are exactly <code>[+1,+1,-1,-1]</code>. (DeepSeekMath normalises by the group's standard deviation; a small epsilon in the denominator, and the biased-vs-unbiased std convention, are implementation details that do not change the sign or ranking of the advantages.)</p></div>

### 4.4 The DeepSeekMath recipe end to end

Putting it together, DeepSeekMath's reasoning-RL loop is: start from a math-pretrained base model; for each problem sample a group of chain-of-thought solutions; score each with the **rule-based verifiable reward** (answer correct? format valid?); compute group-relative advantages; take a GRPO clipped update with a KL penalty to the reference policy; repeat. No ORM, no PRM, no critic — just a checker and a group baseline. It is cheap enough to run at scale, which is precisely why DeepSeek-R1 builds directly on it.

<div class="callout paper"><p><strong>Research connection.</strong> This lesson rests on two papers: <em>Let's Verify Step by Step</em> (Lightman et al. 2023, the ORM-vs-PRM study and PRM800K) and <em>DeepSeekMath</em> (Shao et al. 2024, which introduces GRPO). See entries 24 and 25 in <a href="#/papers/index">the paper curriculum</a>; both are direct prerequisites for the DeepSeek-R1 reading (entry 26) in the next lesson.</p></div>

## Common mistakes

- **Thinking PRM is strictly better than ORM.** PRM wins on *hard* problems with *many* samples, but it needs expensive step labels; on easy tasks or small $N$, cheap ORM is often as good. Reward granularity is a cost/accuracy trade, not a free lunch.
- **Confusing a verifiable reward with a reward model.** A verifiable reward is a *program* that checks correctness ($1$/$0$); a reward model (ORM/PRM) is a *learned network* that predicts a score. DeepSeekMath's point is to use the former and skip the latter where possible.
- **Forgetting GRPO's advantage is group-relative.** The baseline is the mean of *this prompt's group*, not a global constant or a learned value. Standardising across the wrong axis (e.g. across the whole batch instead of within each prompt's group) breaks the method.
- **Assuming a correct final answer means correct reasoning.** It does not — cancelling errors and lucky guesses exist. That gap is the entire motivation for process supervision.

## Exercise

You are rerank­ing two solutions with the *same* final answer using a PRM. Solution C has per-step scores `[0.95, 0.95, 0.30]`; solution D has `[0.7, 0.7, 0.7]`. Which does **product** aggregation prefer, which does **minimum** prefer, and what does the disagreement tell you?

*Hint:* compute $\prod_t s_t$ and $\min_t s_t$ for each.

<details><summary>Solution</summary>

Product: C $= 0.95 \times 0.95 \times 0.30 = 0.27075$; D $= 0.7^3 = 0.343$. Product prefers **D** ($0.343 > 0.271$). Minimum: C $= 0.30$, D $= 0.70$. Minimum also prefers **D**. Here they agree, both punishing C's weak final step ($0.30$). The interesting lesson is *why* the aggregations can disagree in general: product rewards being *consistently* good and multiplies in every doubt, so a solution with one very bad step but many near-perfect steps can still beat a uniformly mediocre one; minimum ignores everything except the single worst step. Pick product when you care about overall chain reliability, minimum when a single broken step should be disqualifying.

</details>

## Check yourself

<details><summary>Give a concrete case where an ORM gives the wrong reward but a PRM would catch it.</summary>

A chain makes an error in step 3 and a second error later that cancels the first, arriving at the correct final number. The ORM sees only the correct final answer and rewards it (reward 1), reinforcing broken reasoning. A PRM scores step 3 as low, so its aggregated score (product or minimum) is small and the solution is correctly down-weighted despite the right answer. The ORM cannot see inside the chain; the PRM can.

</details>

<details><summary>What exactly does GRPO remove compared to PPO, and what replaces it?</summary>

GRPO removes PPO's learned value network (critic), which is roughly the same size as the policy and is used to compute the advantage baseline. It replaces that baseline with the mean reward of a *group* of outputs sampled for the same prompt: $\hat A_i = (r_i - \operatorname{mean}(r))/\operatorname{std}(r)$. This roughly halves memory (no second large network) while still giving each sample an advantage relative to a baseline.

</details>

<details><summary>For a group with rewards $[1, 0, 0, 0]$, what are the group-relative advantages (population std), and what does the sign pattern mean?</summary>

Mean $= 0.25$. Population std $= \sqrt{(0.75^2 + 3\cdot 0.25^2)/4} = \sqrt{0.1875} \approx 0.4330$. Advantages $\approx [(1-0.25)/0.433,\ (0-0.25)/0.433 \times 3] = [+1.732,\ -0.577,\ -0.577,\ -0.577]$. The one correct sample gets a large positive advantage (strongly up-weighted) and the three wrong samples each a small negative one — the update pushes the policy toward the single solution that worked.

</details>

<details><summary>Why can DeepSeekMath skip a learned reward model where InstructGPT/RLHF could not?</summary>

Because math answers are *machine-checkable*: a deterministic program can parse the final answer and compare it to the gold answer, giving an exact $1$/$0$ reward with no learned network. RLHF targets open-ended helpfulness/harmlessness, which has no automatic checker, so it must *learn* a reward model from human preferences (Bradley-Terry). Verifiable rewards apply only where correctness is programmatically decidable (math, code, proofs); elsewhere you still need a learned reward model.

</details>

## Next

You now have the two ingredients DeepSeek-R1 runs at scale: a verifiable reward (no reward model needed) and GRPO (no critic needed). The final lesson puts them together into the full R1 pipeline — pure-RL R1-Zero and the cold-start-plus-RL R1 — reconstructs the conceptual stages, and ties each one to the runnable `mini_r1` you can execute on a laptop.

Continue to [17.3 · DeepSeek-R1 case study](lessons/module-17/lesson-03.md).
