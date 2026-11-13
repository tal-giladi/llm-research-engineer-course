# 15.5 · Where preference data comes from, and how to build it

<div class="prereq">
<p><strong>Prerequisites:</strong> the Bradley-Terry reward model and the <code>{prompt, chosen, rejected}</code> rows it trains on from <a href="#/lessons/module-15/lesson-01">15.1</a>; DPO from <a href="#/lessons/module-15/lesson-03">15.3</a>; RLAIF and Constitutional AI from <a href="#/lessons/module-15/lesson-04">15.4</a>; building and cleaning an SFT set from <a href="#/lessons/module-14/lesson-04">14.4</a>.</p>
<p><strong>You will learn:</strong> the three ways preference data is collected (people compare, people rate, a model judges); the public datasets worth using today and their licenses; how to turn N rated responses into one pair (binarization) and why the choice of the rejected response matters; how to measure annotator agreement with Cohen's kappa; the two biases that quietly ruin preference data (length and position) and how to measure and remove them; on-policy data and the "delta" idea behind current open recipes; and how to evaluate the result with RewardBench 2.</p>
<p><strong>Why this matters for ML:</strong> a reward model or DPO run can only learn the preferences in its data. If "chosen" is usually the longer answer, the reward model learns "longer is better", and RL or DPO then makes every answer longer. Most failures blamed on RLHF are data failures that a few counts would have caught before training.</p>
</div>

<div class="hw">
<p><strong>Hardware track:</strong> CPU only. The lab streams 4,000 UltraFeedback prompts and 4,000 HelpSteer3 rows (a few hundred MB of downloads) and runs in a few minutes on a laptop. No GPU. Training a reward model on the output uses 15.1's command and hardware track. Checked on 2026-11-13 with Python 3.14 and <code>datasets</code> 5.2.0.</p>
</div>

## 1. Intuition: a preference is a comparison, and comparisons are noisy

SFT data says "here is a good answer". Preference data says "for this prompt, answer A is better than answer B". That is easier to give than a perfect answer (people can tell which of two essays is better without being able to write either), and it carries information SFT cannot: what makes one acceptable answer better than another.

It is also noisy. Two careful people often disagree about which of two good answers is better. InstructGPT reported that its trained labelers agreed with each other 72.6% of the time, and held-out labelers agreed with them 77.3% of the time. So roughly one pair in four is a matter of taste. A reward model that reaches 70 to 75% accuracy on such data may be near the ceiling the labels allow.

There are three ways to collect preferences:

1. **People compare two responses** and pick one, sometimes with a strength ("much better", "slightly better"). InstructGPT, Anthropic's HH-RLHF, Chatbot Arena votes and HelpSteer3 all work like this.
2. **People rate each response** on a scale (for example helpfulness 0 to 4), and pairs are built afterwards by comparing ratings. HelpSteer2 rated five attributes per response.
3. **A model judges.** An LLM rates or compares responses, guided by written principles: RLAIF and Constitutional AI (15.4). UltraFeedback used GPT-4 to rate four completions per prompt on four aspects. Tulu 3 used GPT-4o in the same way. Most open preference data today is model-judged, with people writing the guidelines and checking samples.

<div class="callout key"><p>A preference row has three parts, and each can come from a different place: the <strong>prompt</strong> (real users, SFT prompts, synthetic), the <strong>two responses</strong> (your own model, other models, or edited versions), and the <strong>judgment</strong> (people or a model). Which model wrote the responses turns out to matter as much as who judged them (Section 6).</p></div>

### The public sets, as of November 2026

Every number was read from the dataset card or the Hugging Face API on 2026-11-13.

| Dataset (Hugging Face id) | Rows | How it was made | License and caveats |
|---|---|---|---|
| `openbmb/UltraFeedback` | 63,967 prompts × 4 responses | 4 of 17 models answer; GPT-4 rates helpfulness, honesty, instruction following and truthfulness 1–5 | MIT. Contains TruthfulQA prompts (a benchmark) |
| `argilla/ultrafeedback-binarized-preferences-cleaned` | 60,917 pairs | UltraFeedback binarized by mean rating, TruthfulQA removed | MIT. The recommended UltraFeedback variant |
| `HuggingFaceH4/ultrafeedback_binarized` | 61,135 train pairs | Best vs random lower (used for Zephyr) | MIT. Its card notes wrong labels and TruthfulQA contamination |
| `nvidia/HelpSteer3` (config `preference`) | 38,459 train | Human pairwise preference with strength −3…3 and written reasoning; General, STEM, Code, Multilingual | CC-BY-4.0. Responses from about 20 permissively licensed models |
| `nvidia/HelpSteer2` | 20,324 | Human 0–4 ratings of 5 attributes; a preference subset was added later | CC-BY-4.0 |
| `Anthropic/hh-rlhf` | 160,800 train | Human pairwise preferences, helpfulness and harmlessness | MIT. Contains offensive content; the card says not for SFT |
| `allenai/llama-3.1-tulu-3-8b-preference-mixture` | 272,898 | On-policy + off-policy responses, GPT-4o judged | ODC-BY; parts non-commercial; generator terms apply |
| `allenai/Dolci-Instruct-DPO` (OLMo 3) | 259,922 | Delta-learning pairs plus GPT-judged pairs and multi-turn pairs | ODC-BY |
| `Skywork/Skywork-Reward-Preference-80K-v0.2` | 77,016 | Curated subset of public sets; v0.2 removed 4,957 pairs that overlapped RewardBench | No license tag on the card: check before use |
| `lmarena-ai/arena-human-preference-140k` | 135,634 | Real users' votes between two anonymous models | Prompts CC-BY-4.0; responses under each provider's terms |
| `PKU-Alignment/PKU-SafeRLHF` | 73,907 train | Harmlessness and helpfulness preferences with harm categories | CC-BY-NC-4.0: non-commercial |

For **evaluating** a reward model, use `allenai/reward-bench-2` (1,865 items, ODC-BY): each item has one correct and three incorrect responses, across factuality, precise instruction following, math, safety, focus and ties. A reward model scores an item only if it ranks the correct response above all three.

<div class="callout warn"><p>The same license warning as 14.4 applies: model-judged and model-written data can carry the terms of the models involved. Also check every set against the benchmarks you report. The Skywork v0.1 to v0.2 change and the TruthfulQA rows in UltraFeedback are both cases where a popular preference set contained an evaluation set.</p></div>

## 2. From ratings to pairs: binarization

UltraFeedback gives four responses per prompt, each with ratings. DPO and the Bradley-Terry loss of 15.1 need pairs. Turning N rated responses into one pair is called **binarization**.

Give each response $i$ a score $s_i$. In UltraFeedback that is the mean of its four aspect ratings. The chosen response is the best one, $c = \arg\max_i s_i$. For the rejected one there are two common choices:

- **best vs worst:** $r = \arg\min_i s_i$. Largest margin, but the worst response is often obviously bad (off-topic, broken), so the model learns little it did not already know.
- **best vs random lower:** $r$ is drawn uniformly from $\{i : s_i < s_c\}$. Smaller margins on average, more informative pairs. This is how `ultrafeedback_binarized` and Tulu 3's preference data were built.

Pairs where every response has the same score are dropped: there is no preference to learn. You can also require a minimum margin $s_c - s_r > m$ to drop near-ties, which are mostly judge noise.

**Numerical example.** Four responses with mean ratings $s = (4.5, 2.0, 3.0, 4.5)$. The maximum is 4.5, shared by responses 1 and 4; `binarize` takes the first, so $c = 1$. Responses with a strictly lower score are $\{2, 3\}$; response 4 is excluded because it is tied with the chosen one (a pair of two equally rated responses teaches nothing). Best vs worst gives $r = 2$ with margin $4.5 - 2.0 = 2.5$. Best vs random picks $r = 2$ or $r = 3$ with equal probability; the margin is 2.5 or 1.5. With `min_margin=3.0` no pair qualifies and the prompt is dropped.

```python
import random
from llmre.preference.data import binarize

binarize("p", ["a", "b", "c", "d"], [4.5, 2.0, 3.0, 4.5], strategy="best_vs_worst")
# {'prompt': 'p', 'chosen': 'a', 'rejected': 'b', 'score_chosen': 4.5, 'score_rejected': 2.0}
binarize("p", ["a", "b", "c", "d"], [4.5, 2.0, 3.0, 4.5], rng=random.Random(1))["rejected"]  # 'b' or 'c'
```

**Shapes, dtype, device.** Plain Python: `completions` is a `list[str]` of length N, `scores` a `list[float]` of length N, the output a `dict` of strings and floats. Nothing touches a tensor until 15.1's `encode_pairs`.

## 3. Pairwise judgments, and position bias

When a person or a model compares two responses, `pair_from_judgment(prompt, a, b, verdict)` turns the verdict `"A"`, `"B"` or `"tie"` into a row (ties give no row).

LLM judges have a well-documented **position bias**: they prefer whichever response is shown first (some models prefer the second) more often than the content justifies. The standard fix costs one extra call: ask twice with the order swapped, and keep the verdict only if it names the same underlying response both times. If the verdict flips with the order, the judge was deciding by position, and the pair is treated as a tie.

```python
from llmre.preference.data import debiased_verdict

def judge(prompt, first, second):          # a broken judge: always prefers the first
    return "A"

debiased_verdict(judge, "p", "x", "yy")    # 'tie': it picked x, then yy, so it was position, not content
```

The cost is two judge calls per pair instead of one. In return you remove the pairs where the judge had no real preference, which are exactly the noisy ones.

## 4. Do the labels mean anything? Annotator agreement

Before trusting labels, measure whether independent annotators agree. The simple measure is **percent agreement** $p_o$: the fraction of items where two annotators gave the same label. It is misleading when one label dominates: if both annotators say "A" 90% of the time at random, they agree $0.9^2 + 0.1^2 = 82\%$ of the time without reading anything.

**Cohen's kappa** corrects for this. With $p_1(k)$ and $p_2(k)$ the fraction of items each annotator gave label $k$,

$$
p_e = \sum_k p_1(k)\, p_2(k), \qquad \kappa = \frac{p_o - p_e}{1 - p_e}.
$$

$p_e$ is the agreement expected if both labelled independently with their own label frequencies. $\kappa = 1$ is perfect agreement, $\kappa = 0$ is chance level, negative is worse than chance.

**Numerical example.** Ten pairs, two annotators:

| Pair | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| Annotator 1 | A | A | A | B | B | A | A | B | A | A |
| Annotator 2 | A | A | B | B | A | A | A | B | A | A |

They agree on 8 of 10, so $p_o = 0.8$. Annotator 1 said A 7 times, annotator 2 also 7 times: $p_1(A) = p_2(A) = 0.7$, $p_1(B) = p_2(B) = 0.3$. Then $p_e = 0.7 \cdot 0.7 + 0.3 \cdot 0.3 = 0.58$ and

$$
\kappa = \frac{0.8 - 0.58}{1 - 0.58} = \frac{0.22}{0.42} \approx 0.52.
$$

80% agreement sounds high; a kappa of 0.52 says that about half of the agreement above chance is real. Common rules of thumb call 0.4 to 0.6 "moderate" and above 0.8 "almost perfect"; treat these as rough labels, not thresholds.

```python
from llmre.preference.data import percent_agreement, cohens_kappa
a = list("AAABBAABAA"); b = list("AABBAAABAA")
percent_agreement(a, b), cohens_kappa(a, b)     # (0.8, 0.5238...)
```

HelpSteer2 reported quadratic-weighted kappa for its 0–4 helpfulness ratings rising from 0.465 to 0.706 after the guidelines were rewritten, and to 0.791 after removing outlier annotations. Clear written guidelines are the cheapest way to raise agreement: say what counts as better (correctness first, then completeness, then style), how to treat a confident wrong answer versus an honest "I don't know", and when to choose "tie".

## 5. Length bias: the most common silent bug

Judges, human and model, tend to prefer longer answers: a longer answer looks more thorough. If "chosen" is the longer response in most pairs, a reward model can reach good accuracy just by measuring length, and RL against it then makes every answer longer. "A Long Way to Go" (Singhal et al., 2023) showed that a reward based only on length reproduces much of the measured gain of RLHF in their settings.

**Measure it.** `length_bias(pairs)` counts how often the chosen response is longer. A value near 0.5 means length tells you nothing about the label; far above 0.5 means it does.

**Remove it.** `length_balanced_subset(pairs)` keeps every pair from the minority side ("chosen is shorter"), samples the same number from the majority side, and keeps equal-length pairs. Length then no longer predicts the label. The cost is data: you drop part of the majority side. Two alternatives that keep all the data: length-normalised DPO (Tulu 3 used it) and length-controlled evaluation (AlpacaEval's length-controlled win rate).

## 6. On-policy data and the delta between responses

Who wrote the two responses matters:

- **Off-policy** responses come from other models (UltraFeedback's 17 models). They are easy to get, but they may differ from your model's own answers in ways that are easy to tell apart and irrelevant to quality, for example one model's style.
- **On-policy** responses come from the model you are about to train. A pair of two of your own model's answers teaches it to tell its own good outputs from its own bad ones, which is exactly the decision it faces at inference. Tulu 3 mixed both: for each prompt, one response from the Tulu 3 SFT model and others from a pool of models, all judged by GPT-4o on four aspects; chosen = highest mean rating, rejected = a random lower one. It found that more unique prompts helped, while repeating prompts with new response pairs did not.

**The delta-learning idea.** "The Delta Learning Hypothesis" (Geng et al., 2025) found that the *gap* between chosen and rejected drives preference tuning, even when both responses are weak. Pairs of a 3B model's answer (chosen) and a 1.5B model's answer (rejected) tuned an 8B model to match Tulu 3 on their 11 benchmarks, although supervised training on the 3B answers alone made the 8B model worse. OLMo 3's Dolci DPO sets use this: a large share of their pairs pair a stronger and a weaker model of the same family. SmolTalk2's preference config does the same with Qwen3-32B (chosen) and Qwen3-0.6B (rejected).

<div class="callout key"><p>Practical recipe, reasonable industry practice rather than one documented pipeline: take prompts that are not in your SFT set (or are, with fresh samples); sample 2 to 4 responses from your SFT model and 1 to 2 from other models; judge with a strong model in both orders; binarize best vs random lower with a minimum margin; check agreement on a human-labelled sample of 100 to 200 pairs; measure and correct length bias; decontaminate prompts as in 14.4; evaluate the reward model on RewardBench 2.</p></div>

## 7. Lab: build and audit preference pairs from real data

`code/scripts/lab_preference_data.py` has two parts. Part 1 streams 4,000 random UltraFeedback prompts, removes TruthfulQA prompts, binarizes both ways using the mean of the four aspect ratings, measures ties, margins and length bias, and writes `train.jsonl` and `validation.jsonl` in exactly the format 15.1's trainer reads. Part 2 streams 4,000 HelpSteer3 preference rows and measures the agreement between their first two annotators.

```bash
cd code
pip install -e . "datasets==5.2.0"
python scripts/lab_preference_data.py --n 4000 --out data/pref
```

You know it worked when `data/pref/train.jsonl` has about 3,400 rows of `{prompt, chosen, rejected}` and `data_card.json` contains the counts below. Then train a reward model on it with the command from 15.1 (`python -m llmre.preference.train --sft ... --train data/pref/train.jsonl --validation data/pref/validation.jsonl ...`).

### What we measured (2026-11-13, seed 0)

**UltraFeedback.** 316 of the 4,000 prompts (7.9%) came from TruthfulQA and were removed. 85 prompts had fewer than two rated responses (an "N/A" rating) and 29 had all responses tied. That left 3,570 pairs with either strategy:

| | best vs random lower | best vs worst |
|---|---|---|
| Pairs with margin ≤ 0.5 | 1,281 (36%) | 379 (11%) |
| Chosen is the longer response | **69.6%** | **73.2%** |
| Mean length, chosen / rejected (characters) | 1,480 / 1,074 | 1,480 / 928 |

The length bias is strong: in 7 of 10 pairs the chosen answer is longer. After `length_balanced_subset`, 2,172 pairs remain with exactly 50% chosen-longer; the mean lengths are still 1,350 vs 1,185, because balancing equalises the *count* of longer and shorter, not the size of the gap.

UltraFeedback also has an `overall_score` from GPT-4's written critique. In 1,759 of the 3,599 prompts with at least two rated responses (49%), the best response by mean aspect rating is not the best by `overall_score` (some of these are ties broken by order). That is why the Argilla cleaned version switched to the mean rating: the two scores in the same dataset disagree on the winner about half the time.

**HelpSteer3.** 3,957 of the 4,000 rows have three annotators. The first two annotators agree on the direction (which response is better) **98.6%** of the time, $\kappa = 0.972$. That is far above InstructGPT's 72.6%, and the reason is on the card: every sample was annotated by three to five people, only the three annotations that agree most are released, and a substantial part of the data was filtered away in quality assurance. Published agreement on a filtered release measures the release, not how hard the task is. Only 72 of the 4,000 rows are exact ties (overall preference 0).

**Common mistakes**

- **Training on the binarized set without measuring length bias.** It takes one function call and explains most "my model got verbose" reports.
- **Using best vs worst by default.** It maximises the margin but many rejected answers are trivially bad.
- **Splitting train and validation by row when there are several pairs per prompt.** The same prompt then appears on both sides and validation accuracy is inflated. Split by prompt. The lab keeps one pair per prompt, so a row split is a prompt split.
- **Trusting a single-order LLM judge.** Always ask in both orders, or randomise the order and measure the bias.
- **Comparing reward-model accuracy across datasets.** 70% on noisy human labels can be a better model than 90% on easy synthetic pairs.
- **Keeping chosen == rejected rows.** They give a zero gradient in both DPO and Bradley-Terry and usually signal a bug upstream; `validate_pair` catches them.

## 8. Exercise: does balancing length change what the reward model learns?

Train two reward models with 15.1's command on the lab output: one on `train.jsonl`, one on a length-balanced version of it. Evaluate both on a held-out set where the chosen response is the *shorter* one. Which one does better, and by how much?

<details><summary>Hint</summary>

Build the balanced training file with `length_balanced_subset(read_jsonl("data/pref/train.jsonl"))`. Build the "chosen is shorter" evaluation set by filtering `validation.jsonl` on `len(chosen) < len(rejected)`.

</details>

<details><summary>Stronger hint</summary>

The two training files have different sizes (about 3,400 vs 2,000 pairs). To compare fairly, also train a third model on a random subset of the original file with the same size as the balanced one. Otherwise you cannot tell whether a difference comes from balance or from size.

</details>

<details><summary>Solution sketch</summary>

On the "chosen is shorter" subset, a model that learned length as a shortcut scores below 50%, because length points the wrong way there. The balanced model should do better on this subset, while the same-size random model shows how much of the gap is due to size alone. With the course's tiny GPT and a few thousand pairs the numbers are noisy: run each setting with two or three seeds and report the spread, not one number. The finding to look for is the direction, not a specific value.

</details>

## 9. Debugging exercise

After DPO on a new preference set, average response length went from 180 to 410 tokens, and the team's human raters say quality did not change. The DPO loss went down normally. What do you check first, and what are two fixes?

<details><summary>Answer</summary>

Check `length_bias` on the training pairs. If the chosen response is longer in, say, 75% of pairs, DPO learned that "longer" separates chosen from rejected, which is the cheapest feature to exploit. Fixes: rebalance with `length_balanced_subset` (or regenerate pairs with a length-matched rejected response), or use a length-normalised DPO objective as Tulu 3 did. Then evaluate with a length-controlled metric so you can see whether quality, not length, improved.

</details>

## Research connection

- **InstructGPT** ([Ouyang et al., 2022](https://arxiv.org/abs/2203.02155)), Section 3 and Appendix B: how labelers were selected and trained, and the 72.6% / 77.3% agreement figures.
- **UltraFeedback** ([Cui et al., 2023](https://arxiv.org/abs/2310.01377)): GPT-4 rating four aspects for completions from many models.
- **HelpSteer2** ([Wang et al., 2024](https://arxiv.org/abs/2406.08673)), Appendix G: a complete annotation guideline, and kappa before and after rewriting it. **HelpSteer3** ([Wang et al., 2025](https://arxiv.org/abs/2505.11475)).
- **Tulu 3** ([Lambert et al., 2024](https://arxiv.org/abs/2411.15124)), Section 5: the on-policy preference pipeline and the prompt-vs-response-count ablation.
- **The Delta Learning Hypothesis** ([Geng et al., 2025](https://arxiv.org/abs/2507.06187)) and **OLMo 3** ([Olmo Team, 2025](https://arxiv.org/abs/2512.13961)).
- **A Long Way to Go: Investigating Length Correlations in RLHF** ([Singhal et al., 2023](https://arxiv.org/abs/2310.03716)) and **Length-Controlled AlpacaEval** ([Dubois et al., 2024](https://arxiv.org/abs/2404.04475)).
- **Constitutional AI** ([Bai et al., 2022](https://arxiv.org/abs/2212.08073)) and **RLAIF vs RLHF** ([Lee et al., 2023](https://arxiv.org/abs/2309.00267)): model-judged preferences.
- **RewardBench 2** ([Malik et al., 2025](https://arxiv.org/abs/2506.01937)): evaluating the reward model you train on this data.

**What to read and what to skip.** In InstructGPT, read the labeler section and the agreement numbers; skip the API prompt-category tables. In HelpSteer2, read the guidelines appendix in full: it is the best template for writing your own. In Tulu 3 Section 5, read the pipeline figure and the ablations on prompt count and on-policy data; skim the per-benchmark tables.

**Code this after reading:** add a `min_margin` sweep to the lab (0, 0.5, 1.0) and report how many pairs and how much length bias remain at each value.

## Check yourself

<details><summary>Two annotators agree on 90% of 200 pairs, but both chose "A" 95% of the time. Is that good agreement?</summary>

No. $p_e = 0.95^2 + 0.05^2 = 0.905$, so chance alone gives 90.5% agreement and $\kappa = (0.90 - 0.905)/(1 - 0.905) \approx -0.05$: no better than chance. The label "A" is probably the response shown first, so check position bias as well.

</details>

<details><summary>Why does best vs random lower often give better DPO data than best vs worst?</summary>

The worst of several responses is often obviously bad, so the model already ranks it last and learns little from the pair. A random lower response is more often a plausible answer that is still worse, which is the distinction the model needs to learn. The cost is smaller margins, so add a minimum margin to drop near-ties.

</details>

<details><summary>An LLM judge picks response x when x is shown first and response y when y is shown first. What does <code>debiased_verdict</code> return, and why?</summary>

`"tie"`. The judge chose by position, not content, so its verdict carries no information about which response is better. Dropping these pairs removes noise at the cost of one extra judge call per pair.

</details>

<details><summary>HelpSteer3's annotators agree 98.6% of the time, InstructGPT's labelers 72.6%. Is HelpSteer3's task easier?</summary>

Not necessarily. HelpSteer3 releases only the three most-agreeing of three to five annotations per sample and filtered out a substantial part of the data, so its agreement describes the filtered release. InstructGPT's figure is raw agreement between labelers. Compare agreement only between numbers measured the same way.

</details>

<details><summary>What is the "delta" in delta learning, and why can a pair of two weak responses help?</summary>

The quality gap between chosen and rejected. Preference objectives learn from the direction of that gap, not from the absolute quality of the chosen response. A pair where a 3B model's answer beats a 1.5B model's answer still points the 8B model in the direction of "better", even though imitating the 3B answer directly (SFT) would make the 8B model worse.

</details>

## Next

You can now build, audit and evaluate preference data and train 15.1's reward model or 15.3's DPO on it. Module 16 replaces the learned reward with a program that checks answers, and 16.5 shows where the prompts for that come from.

Continue to [16.1 · From PPO to GRPO](lessons/module-16/lesson-01.md).
