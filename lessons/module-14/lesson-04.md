# 14.4 · Where SFT data comes from, and how to build a clean set

<div class="prereq">
<p><strong>Prerequisites:</strong> the <code>messages</code> format and chat templates from <a href="#/lessons/module-14/lesson-01">14.1</a>; loss masking from <a href="#/lessons/module-14/lesson-02">14.2</a>; the pretraining data funnel, MinHash and n-gram contamination from <a href="#/lessons/module-11/lesson-01">11.1</a>, <a href="#/lessons/module-11/lesson-02">11.2</a> and <a href="#/lessons/module-11/lesson-03">11.3</a>.</p>
<p><strong>You will learn:</strong> the four places SFT data comes from (people, real user logs, existing NLP tasks, and stronger models); which public datasets are worth using today and what their licenses allow; how to convert three different dataset formats into one; the cleaning steps that matter for chat data (structure checks, boilerplate, truncation, duplicate prompts, benchmark contamination); how to mix sources to a token budget; and how to check whether a filter is helping or just throwing away good rows.</p>
<p><strong>Why this matters for ML:</strong> lessons 14.1 to 14.3 assumed a clean list of chats already existed. In practice the SFT set decides what the model can do after fine-tuning far more than the trainer settings do. LIMA showed 1,000 carefully chosen examples can beat much larger noisy sets. Tulu 3 and OLMo 3 spent most of their post-training effort on data, not on the algorithm.</p>
</div>

<div class="hw">
<p><strong>Hardware track:</strong> CPU only. The lab streams about 6,000 rows (roughly 50 MB of downloads), and the cleaning steps take under a minute on a laptop. No GPU, no GPU-hours. Checked on 2026-11-13 with Python 3.14, <code>datasets</code> 5.2.0 and the dataset revisions current on that date.</p>
</div>

## 1. Intuition: SFT data is a set of demonstrations you choose

SFT trains the model to imitate. Whatever the assistant turns in your set do, the model will learn to do: answer well, answer badly, refuse harmless requests, say "As an AI language model", stop mid-sentence, or quote a benchmark answer it memorised. So building an SFT set is mostly a job of *choosing* demonstrations and *removing* the ones you do not want imitated.

The data comes from four places:

1. **People write it.** Annotators or volunteers write both the request and the ideal answer. This is the highest quality and the most expensive. Examples: OpenAssistant (volunteers, 35 languages), Dolly 15k (Databricks employees), the 1,000 examples of LIMA, and the demonstration data of InstructGPT (about 13k prompts written or answered by about 40 contractors).
2. **Real user logs.** Prompts that real users typed into a chatbot. Real prompts are messy, multilingual and varied in a way synthetic prompts are not. Examples: WildChat (users of a free GPT-3.5/GPT-4 service who consented) and LMSYS-Chat-1M. The *responses* in these logs come from the chatbot of the time, so many teams keep the prompts and regenerate the answers with a better model. OLMo 3's Dolci set did exactly that with WildChat.
3. **Existing NLP tasks, rewritten as instructions.** FLAN turned translation, summarisation and QA datasets into instructions (lesson 14.1). Cheap and accurate, but the style is narrow (short answers).
4. **A stronger model writes it (synthetic data).** Self-Instruct and Alpaca asked a large model to invent instructions and answer them. Evol-Instruct rewrites instructions to be harder. Magpie gives an aligned model only the start of its chat template, up to the user slot, so the model writes a user request itself. Persona Hub conditions the generator on one of a billion personas to make the requests diverse. Today most open SFT data is synthetic: SmolTalk, the Nemotron post-training sets, OpenThoughts and most of Tulu 3.

<div class="callout key"><p>Modern open recipes combine the four: <strong>real or persona-seeded prompts</strong> for diversity, <strong>a strong open-weight model</strong> for the responses, <strong>a verifier</strong> where one exists (unit tests for code, answer matching for math, constraint checkers for format), and <strong>heavy filtering and decontamination</strong> afterwards. People are used for a small set of very high-quality examples and for checking samples of the rest.</p></div>

### What the public sets are, as of November 2026

Every number below was read from the dataset card or the Hugging Face API on 2026-11-13. Row counts are for the default configuration.

| Dataset (Hugging Face id) | Rows | How it was made | License |
|---|---|---|---|
| `allenai/tulu-3-sft-mixture` | 939,343 | Mix of 18 sources: persona-synthetic math/code/IF, FLAN v2, WildChat GPT-4, OASST1, safety sets, NuminaMath-TIR and more | ODC-BY, but the card says some parts are non-commercial (No Robots is CC-BY-NC) |
| `allenai/Dolci-Instruct-SFT` (OLMo 3) | 2,152,112 | Tulu 3 prompts plus WildChat with GPT-4.1 responses, precise instruction following, tool use, reasoning | ODC-BY |
| `allenai/Dolci-Think-SFT-Clean-v2` | 2,050,381 | A cleaned copy of the OLMo 3 32B reasoning SFT set (released 2026-11-03), with a table of removed rows and why | ODC-BY |
| `HuggingFaceTB/smoltalk` / `smoltalk2` | 1.04M / 8.6M | Synthetic (Magpie on Llama-3.1-405B; Qwen3-32B reasoning traces in v2) plus public sets | Apache-2.0 for new subsets; others keep their own license |
| `nvidia/Nemotron-Post-Training-Dataset-v1` | 25.7M | Responses from DeepSeek-R1-0528 and Qwen3-235B-A22B | CC-BY-4.0 |
| `open-thoughts/OpenThoughts3-1.2M` | 1.2M | Math, code and science with QwQ-32B reasoning traces | Apache-2.0 |
| `OpenAssistant/oasst2` | 128,575 messages | Human-written by volunteers, as message trees | Apache-2.0 |
| `databricks/databricks-dolly-15k` | 15,011 | Human-written by Databricks employees | CC-BY-SA-3.0 |
| `allenai/WildChat-4.8M` | 3.2M | Real user prompts with OpenAI model responses; toxic and minors' chats removed, PII scrubbed | ODC-BY (OpenAI terms apply to responses) |

Sets you will see referenced but should handle with care: `GAIR/lima` and `lmsys/lmsys-chat-1m` are gated (you accept terms first); Alpaca's 52k examples are licensed for research only; Magpie's sets inherit the Llama license; and `nvidia/Nemotron-Post-Training-Dataset-v2` is gated. NVIDIA's 2026 releases are split by domain (for example `nvidia/Nemotron-SFT-Math-v4`) instead of a "v3" bundle.

<div class="callout warn"><p><strong>Licenses are part of the data.</strong> A dataset's license covers the dataset. Synthetic responses can also carry the terms of the model that generated them (OpenAI, Llama, Gemma and Qwen outputs each have terms). Before training a model you will ship, read the card's license section and the license of every generator it names. "It is on Hugging Face" does not mean "I may use it commercially". This lesson is not legal advice; it tells you where to look.</p></div>

## 2. One schema: converting formats

Public SFT data comes in three shapes. You need one, the `messages` list of 14.1:

| Shape | Who uses it | Example fields |
|---|---|---|
| **messages** | Tulu 3, Dolci, SmolTalk, Nemotron, TRL's "conversational" format | `{"messages": [{"role": "user", "content": ...}, ...]}` |
| **ShareGPT** | OpenHermes and many community sets | `{"conversations": [{"from": "human", "value": ...}, {"from": "gpt", ...}]}` |
| **Instruction triple** | Alpaca, Dolly, FLAN-style sets | `{"instruction", "input"/"context", "output"/"response"}` |

`llmre.sft.data.to_messages` handles all three. The triple becomes a two-turn chat, with the instruction and its input joined by a blank line:

```python
from llmre.sft.data import to_messages

to_messages({"instruction": "Summarize.", "context": "Long text.", "response": "Short."})
# [{'role': 'user', 'content': 'Summarize.\n\nLong text.'},
#  {'role': 'assistant', 'content': 'Short.'}]

to_messages({"conversations": [{"from": "human", "value": "Hi"}, {"from": "gpt", "value": "Hello!"}]})
# [{'role': 'user', 'content': 'Hi'}, {'role': 'assistant', 'content': 'Hello!'}]
```

**Shapes, dtype, device.** Nothing here is a tensor. A row is a Python `dict`; a chat is a `list[dict[str, str]]`. Everything in this lesson runs on the CPU before tokenization. The chats become `torch.long` ids only in 14.2's collator.

OASST is the one common source that needs more work: it is stored as individual messages with `message_id` and `parent_id`, forming a tree (one prompt, several ranked replies, several follow-ups). To get chats you walk from each leaf up to the root. The Tulu 3 mixture ships 7,132 OASST1 conversations already converted to `messages` (`source = ai2-adapt-dev/oasst1_converted`).

## 3. Cleaning: what can go wrong in a chat

Pretraining filters (11.1) look at one document. Chat data has a structure, and most of its problems are structural or specific to assistant replies.

### 3.1 Structure checks

`validate_messages` returns a list of problems; an empty list means the chat is usable:

- every role is `system`, `user` or `assistant`; a system message, if present, is first;
- user and assistant turns alternate, starting with the user;
- no turn is empty;
- the chat **ends with an assistant turn**. A chat ending on a user turn has nothing to train on, because 14.2 masks every non-assistant token. It would add a row that contributes zero loss and still costs compute.

### 3.2 Content checks on the assistant turns

`response_issues` flags three things:

- **Length.** Empty or near-empty answers, and answers longer than a cap (runaway generations).
- **Boilerplate and refusals of harmless requests**, such as "As an AI language model, I cannot…". These phrases came from 2023-era chatbot outputs that leaked into many community sets. A model trained on them learns to say them, and to refuse requests it should answer.
- **Truncation.** A response cut off when the generator hit its token limit. The model learns to stop mid-sentence.

These are heuristics, and every heuristic has false positives. Section 6 measures them on real data, and the result is the most important practical lesson here.

### 3.3 Duplicate prompts

Public mixtures overlap: the same Alpaca or ShareGPT prompt appears in many community sets, and some sets contain the same prompt dozens of times. Two passes:

1. **Exact:** normalise the first user turn (unicode NFKC, lower-case, collapse whitespace), hash it with blake2b, keep the first chat per hash. Use blake2b, not Python's `hash()`, which is salted per process and changes between runs.
2. **Near:** MinHash from 11.2 on the first user turn. Prompts are short, so use 3-word shingles instead of the 5 to 10 used for web pages.

<div class="callout warn"><p>Deduplicating on the <em>prompt</em> is a choice. Two chats with the same prompt and different good answers are not useless: they show the model that several answers are fine. Tulu 3 found that adding more <em>unique prompts</em> helped, while repeating prompts with new responses did not. So the default is one chat per prompt, unless you have a reason.</p></div>

### 3.4 Benchmark decontamination

If a benchmark question (or a paraphrase with the same numbers) is in your SFT set, your evaluation in 13.3 measures memory, not ability. This is far more likely in SFT data than in pretraining data, because synthetic math sets are often seeded from benchmark training splits and some leak test items.

**Mathematics.** Let $G_n(t)$ be the set of lower-cased word $n$-grams of text $t$. Build one index of all benchmark $n$-grams, $B = \bigcup_{q \in \text{benchmark}} G_n(q)$. The contamination score of a training prompt $p$ is the fraction of its $n$-grams found in that index:

$$
c(p) = \frac{|G_n(p) \cap B|}{|G_n(p)|}.
$$

A prompt is removed when $c(p) \ge \tau$. Tulu 3 used $n = 8$ and flagged an item when more than half of its tokens had 8-gram matches, so $\tau = 0.5$ is a reasonable default. Notice the direction: lesson 11.3's `ngram_overlap` asked how much of *one test item* is inside *one training document*. Here we ask how much of *a training prompt* is benchmark text, against one big index. That is one set lookup per $n$-gram, so it scales to millions of rows.

**Numerical example.** Use $n = 3$ so the numbers stay small. The benchmark has one question: "Tom has 3 red apples and 2 green apples". Its 3-grams are

$$
B = \{\text{tom has 3},\ \text{has 3 red},\ \text{3 red apples},\ \text{red apples and},\ \text{apples and 2},\ \text{and 2 green},\ \text{2 green apples}\}, \quad |B| = 7.
$$

The training prompt "Tom has 3 red apples and 2 green pears" has 7 trigrams. The first six are in $B$; the last, "2 green pears", is not. So $c = 6/7 \approx 0.857 \ge 0.5$: removed. The prompt "Ana has 3 red apples" has 3 trigrams: "ana has 3" (no), "has 3 red" (yes), "3 red apples" (yes), so $c = 2/3 \approx 0.67$. With 3-grams this short, unrelated prompt is flagged, which is why real decontamination uses $n = 8$ or more: an 8-word run shared by chance is rare.

```python
from llmre.sft.data import build_benchmark_index, contamination_score

index = build_benchmark_index(["Tom has 3 red apples and 2 green apples"], n=3)
contamination_score("Tom has 3 red apples and 2 green pears", index, n=3)   # 0.857...
contamination_score("Ana has 3 red apples", index, n=3)                     # 0.666...
```

**Cost.** Building the index is $O(\sum_q |q|)$ set insertions; scoring is $O(|p|)$ lookups per prompt. GSM8K test plus MATH-500 is under 2,000 questions, so the index fits in a few MB of Python sets. Real pipelines over billions of tokens use the same idea with hashed $n$-grams in a Bloom filter or a suffix array (the open-instruct `decontamination` tool indexes into Elasticsearch).

<div class="callout warn"><p>$n$-gram matching misses paraphrases: "A farmer owns 3 red and 2 green apples…" shares few 8-grams with the original. Tulu 3 considered embedding-based matching and rejected it as too noisy. If you train on synthetic rewrites of benchmark <em>training</em> sets, you cannot fully rule out paraphrased test items. Report what you checked, and prefer benchmarks released after your data was built.</p></div>

## 4. Mixing sources to a token budget

An SFT mixture is a weighted sample of sources, like the pretraining mixture of 11.3. You decide the share of the **token budget** each source gets, because tokens (not rows) set the compute cost and the influence on the model: one 2,000-token code review counts as much as a hundred 20-token chit-chat replies.

**Mathematics.** With total budget $T$ tokens and non-negative weights $w_s$, source $s$ gets

$$
T_s = T \cdot \frac{w_s}{\sum_{s'} w_{s'}}.
$$

`mix_to_budget` shuffles each source with a fixed seed and takes chats until $T_s$ is reached. If a source runs out first it contributes everything it has and no more. Repeating a small high-quality source (for example two epochs of a hand-written set) can be the right call, but it should be a decision you write down, not a side effect.

**Numerical example.** $T = 600{,}000$, weights Tulu 3 $= 0.7$, Dolly $= 0.3$. Then $T_{\text{tulu}} = 420{,}000$ and $T_{\text{dolly}} = 180{,}000$. The lab below asked for exactly this and got 420,389 and 180,065 tokens: each source overshoots by less than one chat, because the last chat taken crosses the line.

Tokens are counted as characters divided by 4, the usual rough rule for English with a GPT-style BPE. For a real run, pass your tokenizer's length function to `mix_to_budget(count=...)`; for non-English or code-heavy data the rule can be off by a factor of two.

## 5. From-scratch implementation

All of it lives in `code/src/llmre/sft/data.py` and reuses Module 11's MinHash and $n$-gram code. The core of the cleaning loop:

```python
from llmre.sft.data import (to_messages, validate_messages, response_issues,
                            dedupe_by_prompt, near_dedupe_prompts,
                            build_benchmark_index, decontaminate, mix_to_budget)

def clean(rows, check_refusals=True, check_truncation=True):
    kept = []
    for row in rows:
        chat = to_messages(row)
        if validate_messages(chat):            # structural problems -> drop
            continue
        if response_issues(chat, check_refusals=check_refusals,
                           check_truncation=check_truncation):
            continue
        kept.append(chat)
    kept, _ = dedupe_by_prompt(kept)           # exact duplicate prompts
    kept, _ = near_dedupe_prompts(kept)        # MinHash, threshold 0.8
    return kept
```

The duplicate step keeps the *first* chat for each prompt, so the order of your input is a priority order: put the source you trust most first.

The `near_dedupe_prompts` here compares each new signature against every kept one, which is $O(n^2)$. That is fine for a few thousand rows. For a million rows, use the LSH banding of 11.2 or a tool built for it (`datatrove` 0.10.1 has MinHash dedup stages).

## 6. Lab: build a clean SFT set from real data

The script `code/scripts/lab_sft_data.py` streams 3,000 random rows each from `allenai/tulu-3-sft-mixture` and `databricks/databricks-dolly-15k`, cleans them, decontaminates against GSM8K test and MATH-500, mixes 70/30 to 600k tokens, and writes `sft_clean.jsonl` plus a `data_card.json` that records every drop and its reason.

```bash
cd code
pip install -e . "datasets==5.2.0"
python scripts/lab_sft_data.py --n 3000 --out data/sft
```

It takes a few minutes, most of it download. You know it worked when `data/sft/sft_clean.jsonl` exists, every line has a `messages` list ending in an assistant turn, and the data card's token totals are close to 420k and 180k.

### What the first runs taught us (results from 2026-11-13)

The first versions of this lab applied every filter to every row (the truncation counts are from an unshuffled run, the rest from a shuffled one). Here is what came out, and why the final script differs:

| Filter | First runs | What we found when we read the dropped rows | Final script |
|---|---|---|---|
| Boilerplate/refusal | 225 of 3,000 Tulu rows dropped | Most came from the **safety** subsets (WildGuardMix, WildJailbreak, CoCoNot), where refusing a harmful request is the behaviour being taught | Skip this filter for safety sources: 13 dropped |
| Truncation (original "ends in a letter" rule) | 160 Tulu and 167 Dolly rows | Almost all were fine: list items, URLs, `3 + 3 = 6`, sign-offs like "Thanks! - Jane". Dolly's human writers often skip the final period | Stricter rule (last line is prose of 8 or more words with no punctuation), and off for Dolly: 16 Tulu rows |
| Too short (under 2 characters) | 2 Dolly rows | One was the answer "5" to a counting question, which is correct | Minimum is 1 character |
| Exact duplicate prompts | 6 Tulu, 12 Dolly | Real duplicates | Kept |
| Near duplicate prompts | 3 Tulu, 31 Dolly | Dolly has many near-identical "classify each of these…" prompts | Kept |
| Contamination (8-gram, $\tau = 0.5$) | 0 and 0 | Expected: the Tulu sources sampled are named `*_decontaminated`, and Dolly has no math | Kept; Section 7's exercise finds real hits |

<div class="callout key"><p><strong>Every filter needs a precision check.</strong> Before trusting a filter, read 20 to 50 of the rows it drops. If most of them are fine, the filter is hurting you: it removes good data and, worse, it removes it unevenly (here it would have removed nearly all safety training data). The data card's <code>one_dropped_example_per_reason</code> field exists so you look.</p></div>

Also notice the stream is shuffled. Large mixtures are stored grouped by source: the first 3,000 rows of the Tulu 3 mixture are all OASST. `ds.shuffle(seed=0, buffer_size=10_000)` picks shards in random order and mixes rows in a buffer, which gave six different sources in our sample.

**Common mistakes**

- **Taking the first N rows of a mixture.** You get one source. Shuffle the stream, or sample per source.
- **Running one filter set over every source.** Safety sets need refusals; human sets lack final periods; code needs symbols. Configure filters per source.
- **Dropping without recording why.** Without a data card you cannot answer "why is the model worse at X?" a month later.
- **Deduplicating across the chat instead of the prompt.** Two chats that differ only in the answer are not exact duplicates as whole chats, so a whole-chat hash keeps both.
- **Decontaminating only the user turn of the first message.** Multi-turn chats can paste a benchmark question in turn 3. `decontaminate` checks every user turn.
- **Ignoring the generator's license.** See the warning in Section 1.

## 7. Exercise: find contamination in a set that was not cleaned

`project-numina/NuminaMath-CoT` (859,594 problems, Apache-2.0; the old id `AI-MO/NuminaMath-CoT` redirects there) was not decontaminated by its authors. Stream 20,000 of its problems, convert them with `to_messages` (they ship a `messages` field), and decontaminate against GSM8K test and MATH-500. How many hits do you get at $\tau = 0.5$ and at $\tau = 0.8$? Read five hits at each threshold: which are true leaks?

<details><summary>Hint</summary>

Reuse `load_rows` and the index-building lines from `lab_sft_data.py`. `NuminaMath-CoT` has a `source` column; count hits per source.

</details>

<details><summary>Stronger hint</summary>

Some sources in NuminaMath are rewrites of MATH and GSM8K *training* problems (`synthetic_math`), and some are scraped exam problems. A high score on a short problem can come from a common phrase like "what is the value of x such that"; a high score on a long problem almost never does. Print `len(prompt.split())` next to the score.

</details>

<details><summary>Solution sketch</summary>

```python
from collections import Counter
rows = load_rows("project-numina/NuminaMath-CoT", "train", 20_000)
chats = [to_messages(r) for r in rows]
for tau in (0.5, 0.8):
    clean, removed = decontaminate(chats, index, n=8, threshold=tau)
    print(tau, len(removed))
```

Our run on 2026-11-13 (20,000 shuffled rows, seed 0) gave **13 hits at $\tau = 0.5$** (12 from `synthetic_math`, 1 from `math`) and **2 hits at $\tau = 0.8$**. Reading them:

- The $\tau = 0.8$ hit from the `math` source is a MATH-500 test problem copied word for word (the $\sin \angle RPS$ diagram problem): a true leak.
- Most $\tau = 0.5$ hits from `synthetic_math` are rewrites of MATH problems that kept the original's instruction sentence ("Enter your answer in the form…", "Convert the point … to polar coordinates"). Some changed the numbers, so they are not the test item, but a model trained on them has seen the test item's exact template.
- The other $\tau = 0.8$ hit is a short counting problem ("arrange the letters of the word BANANA") whose wording matches a benchmark template almost entirely.

So the honest answer is: at 0.8 you remove clear copies, at 0.5 you also remove templated near-copies, and where you set the line is a judgment you write in the data card. Tulu 3 used 0.5.

</details>

## 8. Debugging exercise

A teammate fine-tunes on a cleaned mixture and reports: "The new model refuses far fewer harmful requests than the previous one, but our safety data is in the mixture." The data card shows 3,000 Tulu rows in, 2,751 out, with 225 dropped as "boilerplate/refusal phrase". What happened, and how do you confirm it?

<details><summary>Answer</summary>

The refusal filter was applied to the safety subsets, whose whole purpose is demonstrating refusals of harmful requests. It removed most of them, so the model saw almost no refusal demonstrations. Confirm by counting dropped rows per `source`: if nearly all refusal drops come from `wildguardmix`, `wildjailbreak` or `coconot`, that is the cause. Fix by turning the refusal check off for safety sources (`check_refusals=False`), and add a per-source count of drops to the data card so this shows up immediately next time.

</details>

## Research connection

- **LIMA** ([Zhou et al., 2023](https://arxiv.org/abs/2305.11206)): 1,000 carefully curated examples, no RL, and responses judged equal to or better than GPT-4's in 43% of cases. The argument that quality and diversity beat quantity for SFT.
- **Tulu 3** ([Lambert et al., 2024](https://arxiv.org/abs/2411.15124)), Sections 3 and 4: the most detailed public description of building an SFT mixture, including the decontamination rule used above and the percentage removed from each source (for example 11.3% of NuminaMath-TIR).
- **OLMo 3** ([Olmo Team, 2025](https://arxiv.org/abs/2512.13961)): the Dolci SFT sets; regenerating WildChat responses with a stronger model.
- **Magpie** ([Xu et al., 2024](https://arxiv.org/abs/2406.08464)) and **Persona Hub** ([Ge et al., 2024](https://arxiv.org/abs/2406.20094)): two ways to get diverse synthetic prompts without seed tasks.
- **Self-Instruct** ([Wang et al., 2022](https://arxiv.org/abs/2212.10560)) and **Evol-Instruct / WizardLM** ([Xu et al., 2023](https://arxiv.org/abs/2304.12244)).
- **OpenThoughts** ([Guha et al., 2025](https://arxiv.org/abs/2506.04178)): over 1,000 controlled experiments on how to build a reasoning SFT set (which questions, how many answers per question, which teacher).
- **The Dolci-Think-SFT-Clean-v2 card** (`allenai/Dolci-Think-SFT-Clean-v2`, November 2026): a public example of the kind of data card this lesson asks for. It lists every removal reason with a count (truncated answers, empty reasoning, leaked identity claims, literal control tokens, benchmark hits) and states its own limits.

**What to read and what to skip.** In Tulu 3, read Section 3 (prompt curation and decontamination) and Section 4.1 (SFT data); skim the per-skill ablation tables. In LIMA, read Sections 2 and 5 and skip the human-evaluation protocol details. In OpenThoughts, read the summary of findings and skip the per-ablation appendix.

**Code this after reading:** add a per-source drop count to the lab's data card, then rerun with the first 3,000 rows unshuffled and compare the two cards.

## Check yourself

<details><summary>Why do teams keep WildChat's prompts but often replace its responses?</summary>

The prompts are real requests from many users, which gives diversity that synthetic prompts lack. The responses came from GPT-3.5/GPT-4 at the time (and carry OpenAI's terms), and are weaker than what a current open model produces. Regenerating them, as OLMo 3's Dolci set did, keeps the diversity and raises answer quality.

</details>

<details><summary>A chat in your set ends with a user turn. Why remove it rather than keep it?</summary>

After 14.2's loss masking, only assistant tokens contribute to the loss. A chat ending on a user turn may still contain earlier assistant turns, but a chat with no final answer is usually a broken export, and if it has no assistant turn at all it adds zero loss while still costing compute and padding.

</details>

<details><summary>Your refusal filter drops 7% of a mixture. What do you do before accepting that?</summary>

Count the drops per source and read a sample of them. If they concentrate in safety sources, the filter is removing intended refusals; disable it there. If they are spread across general sources and really are boilerplate refusals of harmless requests, keep the filter.

</details>

<details><summary>With $n = 8$ and $\tau = 0.5$, a 30-word training prompt shares 12 of its 23 8-grams with the benchmark index. Is it removed?</summary>

A 30-word prompt has $30 - 8 + 1 = 23$ 8-grams. $c = 12/23 \approx 0.52 \ge 0.5$, so yes. Twelve shared 8-word runs is far more than chance; it is very likely a copy or near-copy of a benchmark item.

</details>

<details><summary>Why is the mixture budget set in tokens instead of rows?</summary>

Compute cost and influence on the gradient scale with tokens, not rows. Sources differ hugely in average length (short chit-chat vs long code reviews), so equal row counts would give very unequal training signal.

</details>

## Next

You now know where SFT data comes from and how to build a clean set. SFT only imitates demonstrations; it cannot learn from *preferences* ("response A is better than B"). Module 15 adds that: reward modeling, RLHF/PPO and DPO, and in 15.5, where preference data comes from.

Continue to [15.1 · Bradley-Terry & the reward model](lessons/module-15/lesson-01.md).
