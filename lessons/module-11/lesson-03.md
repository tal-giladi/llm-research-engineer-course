# 11.3 · Contamination, mixing, curriculum

<div class="prereq">
<p><strong>Prerequisites:</strong> shingling and n-grams from <a href="#/lessons/module-11/lesson-02">11.2 · Deduplication &amp; MinHash</a> (an n-gram is the same idea as a k-shingle); the quality funnel from <a href="#/lessons/module-11/lesson-01">11.1</a>; token budgets and compute-optimality from <a href="#/lessons/module-10/lesson-03">10.3 · Chinchilla compute-optimality</a>. A preview of evaluation from <a href="#/lessons/module-13/lesson-02">13.2 · Contamination, calibration, task eval</a> — this lesson is the <em>data-side</em> of a topic Module 13 revisits from the <em>eval-side</em>.</p>
<p><strong>You will learn:</strong> benchmark <strong>contamination / leakage</strong> — how test-set text in the training data inflates evaluation, and how to detect it with n-gram overlap; <strong>dataset mixing</strong> — weighting sources (web / code / books / math), why high-quality sources are upsampled, and how a total token budget splits across sources; and the <strong>data curriculum</strong> — ordering and annealing, e.g. saving higher-quality data for late in training. You will run a small n-gram contamination check that flags an overlapping document.</p>
<p><strong>Why this matters for ML:</strong> two models with identical architecture and compute can differ enormously in quality purely because of what data they saw and in what proportion. Contamination makes a mediocre model <em>look</em> great on paper (and is a real, recurring scandal in LLM leaderboards); mixing decides which capabilities the model even has; curriculum squeezes extra quality from the same tokens. These are among the highest-leverage decisions in a pretraining project — and the least visible in the model code.</p>
</div>

## 1. Benchmark contamination: the invisible cheat

You train a model, evaluate it on a benchmark like MMLU or GSM8K, and it scores
wonderfully. Then you discover the benchmark's questions (or their answers) were
sitting in your training crawl — the model didn't *reason*, it *remembered*. That
is **contamination** (also called leakage): test data leaking into training,
inflating evaluation.

It happens constantly because benchmarks are built from the public web, and your
training data *is* the public web. Question banks, quiz sites, GitHub solutions,
and forum answers all get crawled. Unless you actively remove them, you are very
likely training on some of your own test set.

<div class="callout warn"><p>Contamination is not a hypothetical. Public
leaderboards are regularly disrupted when a top model turns out to have trained on
benchmark data. The fix is a deliberate <strong>decontamination</strong> pass:
before training, scan the corpus for overlap with every benchmark you plan to
report, and remove matching training documents. If you skip it, you cannot trust
your own eval numbers — and neither can anyone else.</p></div>

This is the *data-side* of contamination — removing leakage before training.
Module 13 covers the *eval-side*: detecting after the fact that a model was likely
contaminated. Same phenomenon, two defenses.

## 2. Detecting contamination with n-gram overlap

The standard detector reuses the n-gram idea from the last lesson. Represent each
text as its set of word **n-grams** (n consecutive words), then ask: what
fraction of a *benchmark item's* n-grams also appear in a *training document*? A
high fraction means the benchmark text is present in training — leakage.

This is deliberately **directional** (asymmetric). We do not want symmetric
Jaccard here; we specifically ask "how much of the test item shows up in train",
because that is the leakage question. The measure:

$$
\text{overlap}(train, test) = \frac{\big|\,\text{ngrams}(test) \cap \text{ngrams}(train)\,\big|}{\big|\,\text{ngrams}(test)\,\big|}.
$$

- $1.0$: every test n-gram appears in the training document (e.g. the test item
  is a substring of a train doc) — clear contamination.
- $\approx 0$: the two share almost no length-$n$ word runs — clean.

The implementation (`code/src/llmre/data/contamination.py`) normalizes text to
lower-cased `\w+` word tokens first, so trailing punctuation and capitalization
(`"Paris,"` vs `"paris"`) do not block a real match — the same normalization GPT-3
and later reports use for their overlap checks:

```python
def ngram_overlap(train_text: str, test_text: str, n: int = 8) -> float:
    test_ngrams = ngrams(test_text, n)
    if not test_ngrams:
        return 0.0
    train_ngrams = ngrams(train_text, n)
    return len(test_ngrams & train_ngrams) / len(test_ngrams)
```

### Numerical example, worked by hand

Benchmark item (`test_text`): **"the mitochondria is the powerhouse of the
cell"** — 8 words. Its $n = 4$ word 4-grams (slide a 4-word window; $8 - 4 + 1 =
5$ of them):

```text
1: the mitochondria is the
2: mitochondria is the powerhouse
3: is the powerhouse of
4: the powerhouse of the
5: powerhouse of the cell
```

Now two training documents:

- **Contaminated:** *"Study notes. The mitochondria is the powerhouse of the
  cell, a fact every student memorizes."* After normalization this contains the
  exact run "the mitochondria is the powerhouse of the cell", so **all 5** test
  4-grams appear in it. Overlap $= 5/5 = 1.0$.
- **Clean:** *"photosynthesis converts sunlight into chemical energy inside
  chloroplasts."* It shares **none** of the 5 test 4-grams. Overlap $= 0/5 =
  0.0$.

Verify:

```python
from llmre.data.contamination import ngram_overlap
test = "the mitochondria is the powerhouse of the cell"
leak = "Study notes. The mitochondria is the powerhouse of the cell, a fact every student memorizes."
clean = "photosynthesis converts sunlight into chemical energy inside chloroplasts"
print(ngram_overlap(leak,  test, n=4))   # 1.0   -> flagged as contaminated
print(ngram_overlap(clean, test, n=4))   # 0.0   -> clean
```

A decontamination pass sets a threshold (say, flag any training document with
$\ge 0.5$ overlap against any benchmark item, or even a single shared long
n-gram) and **removes the offending training documents** — never the benchmark.
The unit test for this module asserts exactly these two poles: overlap $= 1.0$
when the test text is a substring of the train text, and $\approx 0$ for disjoint
text (`code/tests/test_dataeng.py`).

<div class="callout pt"><p>Choice of $n$ is a precision/recall knob. Small $n$
(say 4) catches paraphrases and partial leakage but fires on common phrases that
are not really contamination; large $n$ (13 is a common production value) fires
only on long verbatim runs — high precision, but misses lightly-edited leaks.
Real pipelines often flag on "shares any $n$-gram for large $n$" rather than a
fraction, because a single shared 13-word run is already very unlikely by
chance.</p></div>

## 3. Dataset mixing: weighting the sources

A pretraining corpus is not one pile — it is several **sources** with very
different character and value:

- **Web** — huge, diverse, variable quality (the bulk, after filtering).
- **Code** — teaches syntax, structure, and (evidence suggests) reasoning.
- **Books** — long-form, high-quality, coherent long-range structure.
- **Math** — dense reasoning, scarce but disproportionately valuable.
- (Also: Wikipedia/reference, academic papers, curated Q&A, etc.)

**Mixing** decides what fraction of the training tokens comes from each source.
You assign each source $s$ a weight $p_s$ with $\sum_s p_s = 1$, and given a total
token budget $T_{\text{total}}$, source $s$ contributes $p_s \cdot T_{\text{total}}$
tokens. The weights are a first-class design decision: they directly shape which
capabilities the model gets. Over-weight code and you get a better coder and
(often) reasoner; drop math to near-zero and the model is bad at math no matter
how big it is.

### Why upsample high-quality sources

Here is the tension. High-quality sources are usually *small*. You may have
900B tokens of filtered web but only 3B tokens of good math. If you weight
strictly by how much data exists, math is a rounding error and the model barely
learns it. So pipelines **upsample** scarce high-quality sources: give them a
weight larger than their share of raw tokens, which means the model sees those
tokens **more than once** (multiple epochs over that source) while seeing the
abundant web less than once.

### Numerical example — a token budget split

Total budget $T_{\text{total}} = 300\text{B}$ tokens. Chosen weights and the
tokens available per source:

| source | weight $p_s$ | tokens used $= p_s \cdot 300\text{B}$ | tokens available | epochs over source |
|---|---|---|---|---|
| web | 0.67 | 201.0B | 900B | 0.22 |
| code | 0.17 | 51.0B | 500B | 0.10 |
| books | 0.09 | 27.0B | 30B | 0.90 |
| math | 0.07 | 21.0B | 3B | **7.00** |

The weights sum to $1.00$, so the used tokens sum to $300\text{B}$. "Epochs over
source" is `tokens used / tokens available` — how many times the model passes over
that source's data. Web is seen only $0.22$ times (we have far more web than we
need, so we sample a fraction of it); **math is upsampled 7×** — its 3B tokens are
repeated seven times to fill its 21B-token allocation. That is the upsampling
lever made concrete: a scarce, valuable source punches above its raw size by being
repeated.

```python
budget = 300  # billions of tokens
weights = {"web": 0.67, "code": 0.17, "books": 0.09, "math": 0.07}
avail   = {"web": 900,  "code": 500,  "books": 30,   "math": 3}
for s, p in weights.items():
    used = budget * p
    print(f"{s:6s} used={used:5.1f}B  epochs={used / avail[s]:.2f}")
# web    used=201.0B  epochs=0.22
# code   used= 51.0B  epochs=0.10
# books  used= 27.0B  epochs=0.90
# math   used= 21.0B  epochs=7.00
```

<div class="callout warn"><p>Upsampling trades coverage for emphasis, and too much
of it backfires: repeating the same tokens many times is a form of duplication
(lesson 11.2!) and drives memorization. Empirically a few epochs over a
high-quality source is fine; many epochs starts to hurt. There is no free lunch —
you are choosing what the model over-learns.</p></div>

### How the weights are chosen

<p><strong>PUBLICLY DOCUMENTED:</strong> that frontier and open models mix
multiple sources with deliberate, non-proportional weights, and upsample
high-quality data. GPT-3's paper reports sampling Common Crawl <em>less</em> than
its size would suggest and higher-quality sources (WebText2, Books, Wikipedia)
<em>more</em>. The Pile and Dolma publish their per-source weights explicitly.</p>

<p><strong>REASONABLE INDUSTRY PRACTICE:</strong> tuning mixture weights by
training many small "proxy" models on candidate mixtures and picking the mixture
that minimizes validation loss (or maximizes downstream eval) — mixtures are
searched, not guessed. Increasing the code and math share to improve reasoning is
widely reported across labs.</p>

<p><strong>INFERENCE / SPECULATION:</strong> the <em>exact</em> mixture weights of
any closed frontier model (GPT-4, Claude, Gemini) are not public. Specific claims
like "GPT-4 was N% code" are speculation unless a lab documents them.</p>

## 4. Token budgets and Chinchilla

Mixing sits on top of the total budget, which Module 10 set: **Chinchilla**
(Hoffmann et al., 2022) says that for a fixed compute budget, tokens should scale
roughly with parameters — a rule of thumb of about **20 tokens per parameter**.
So a 7B-parameter model wants on the order of $7\text{B} \times 20 = 140\text{B}$
tokens (modern models train far past this for better inference economics, as
LLaMA did). Mixing then partitions *that* budget across sources. The two
decisions compose: Chinchilla sets *how many* tokens, mixing sets *which* tokens.

<div class="callout key"><p>Two knobs, two questions. <strong>Token budget</strong>
(Module 10): how many tokens total, from compute and the Chinchilla ratio.
<strong>Mixture weights</strong> (this lesson): what fraction of those tokens
comes from each source, with scarce high-quality sources upsampled. Both are set
before a single training step runs, and both move the final loss more than most
architecture choices.</p></div>

## 5. Data curriculum: order and annealing

So far, order did not matter — `get_batch` samples windows uniformly at random
(Module 4). A **data curriculum** deliberately changes *what* the model sees
*when* during training. The most important instance is **annealing**: near the
*end* of training, shift the mixture toward higher-quality data and decay the
learning rate, so the model finishes on its best material.

The intuition mirrors learning-rate decay (Module 3): early training makes big,
coarse updates where rough web data is fine; late training makes small, refining
updates where you want the highest-quality, most-trustworthy tokens to shape the
final weights.

Be careful about certainty here:

<p><strong>PUBLICLY DOCUMENTED:</strong> multi-phase pretraining with a
high-quality final phase is documented in open models. <strong>OLMo 2</strong>
(2024) describes a two-stage pretrain with a late high-quality data phase;
<strong>Llama 3</strong> (2024) documents adjusting the data mix and annealing on
high-quality data toward the end of pretraining. These are real, written-down
recipes — see <a href="#/papers/index">the paper curriculum</a>.</p>

<p><strong>REASONABLE INDUSTRY PRACTICE:</strong> pairing the quality-annealing
phase with learning-rate decay, and using the annealing phase to fold in freshly
curated or synthetic high-quality data, is common and sensible, and reported in
several technical reports.</p>

<p><strong>SPECULATION:</strong> stronger claims — e.g. a strict
"easy-to-hard" example ordering (classic curriculum learning) reliably helping
large-scale LM pretraining — are <em>not</em> settled. Much large-scale
pretraining is still essentially IID random sampling plus a late quality-anneal;
elaborate ordering schemes are researched but not established as standard. Do not
present them as known best practice.</p>

<div class="callout warn"><p>"Curriculum" in classic ML means ordering examples
easy→hard. In modern LLM pretraining the term is used loosely and mostly refers to
<em>phase-based mixture and learning-rate schedules</em> (especially a
high-quality final anneal), not a per-example difficulty sort. Keep the two senses
distinct so you don't overclaim what labs actually do.</p></div>

## 6. Where this closes the module

Put the whole module together. Module 4 packs documents into an id stream and
samples windows. Module 11 is everything that produces and orders those
documents:

```text
raw crawl ─► extract ─► langid ─► quality (11.1) ─► dedup (11.2)
          ─► decontaminate (11.3) ─► MIX sources by weight (11.3)
          ─► [ ordered documents, quality-annealed late (11.3) ]
          ─► BPE encode ─► pack ─► get_batch ─► model      (Module 4 / 7)
```

Every stage is a lever on the final loss that never appears in the model
definition. That is the lesson of the module: at the frontier, data engineering
*is* modeling.

## Exercise

You are decontaminating against a benchmark of 1,000 short questions before
training. You choose $n = 4$ for the n-gram overlap check and flag any training
document with overlap $\ge 0.5$ against any question. A teammate worries this will
delete large amounts of legitimate training data. (a) Why might small $n$
over-flag? (b) What is a safer choice, and what does it trade away?

<details><summary>Hint</summary>
Think about how common a specific 4-word run is versus a specific 13-word run in
ordinary English, and what "over-flagging" costs versus what "under-flagging"
costs.
</details>

<details><summary>Solution</summary>

(a) Short n-grams like 4-word runs occur by coincidence in unrelated text
("the fact that the", "as a result of"), so a low-$n$, fraction-based check fires
on documents that merely share common phrasing with a question — false positives
that delete clean training data.

(b) Use a larger $n$ (e.g. 8–13) and/or flag on "shares any single long n-gram"
rather than a fraction. A shared 13-word verbatim run is almost never a
coincidence, so precision is high. The trade-off is recall: lightly-edited or
paraphrased leaks (which break long exact runs) slip through. Decontamination
deliberately errs toward high precision on long n-grams because deleting good
training data is costly and long-run matches are the ones that truly inflate
scores.

</details>

## Common mistakes

- **Reporting eval numbers without a decontamination pass.** If benchmark text
  was in training, the score measures memorization, not capability — and the
  number is worthless to you and to readers.
- **Removing the benchmark instead of the training doc.** Decontamination deletes
  the *training* documents that overlap; the benchmark stays intact so the eval is
  still meaningful.
- **Weighting sources by raw size.** That buries scarce high-quality data (math,
  books) under abundant web; deliberate, non-proportional weights with upsampling
  are the norm.
- **Over-upsampling.** Repeating a small source too many times is duplication in
  disguise and drives memorization — a few epochs, not fifty.
- **Overclaiming curriculum.** A late quality-anneal is documented; a strict
  easy→hard example ordering is not established for large-scale pretraining. Mark
  the difference.

## Research connection

<div class="callout paper"><p>For token budgets and the tokens-per-parameter
rule, revisit <strong>Chinchilla</strong> (Hoffmann et al., 2022). For documented
data mixing and quality annealing, read the data sections of <strong>GPT-3</strong>
(source upsampling), <strong>OLMo 2</strong> (two-stage pretrain with a late
high-quality phase), and <strong>Llama 3</strong> (data-mix and annealing choices).
All are in <a href="#/papers/index">the paper curriculum</a>. Module 13
(<a href="#/lessons/module-13/lesson-02">13.2</a>) revisits contamination from the
evaluation side.</p></div>

## Check yourself

<details><summary>What exactly does <code>ngram_overlap(train, test, n)</code> return, and why is it directional rather than a symmetric Jaccard?</summary>

It returns the fraction of <em>test</em>'s n-grams that also appear in
<em>train</em>: $|\text{ngrams}(test) \cap \text{ngrams}(train)| /
|\text{ngrams}(test)|$. It is directional because the leakage question is "how
much of the benchmark item is present in training", which is asymmetric — a giant
training document naturally contains many n-grams, so normalizing by the small
test item's n-gram count is what makes "the test item is inside train" read as
$1.0$.

</details>

<details><summary>In the worked example, why is the overlap for the contaminated doc exactly 1.0?</summary>

The test item has 5 word 4-grams. The contaminated training document contains the
exact normalized run "the mitochondria is the powerhouse of the cell", so all 5 of
those 4-grams appear in it. $5/5 = 1.0$. The clean document shares none, giving
$0/5 = 0.0$.

</details>

<details><summary>You have 3B tokens of math and want it to be 7% of a 300B-token run. How many epochs over the math data is that, and what is the risk?</summary>

$0.07 \times 300\text{B} = 21\text{B}$ tokens needed from a 3B source, so
$21/3 = 7$ epochs — the math data is repeated 7 times. The risk is that heavy
repetition is duplication in disguise: it drives memorization of those exact
tokens and can hurt generalization if pushed too far.

</details>

<details><summary>Classify: "Llama 3 annealed on high-quality data at the end of pretraining." vs "GPT-4 was trained on 30% code."</summary>

The first is PUBLICLY DOCUMENTED — Llama 3's report describes late-stage data-mix
adjustment and annealing on high-quality data. The second is SPECULATION — the
exact data mixture of GPT-4 is not public, so any specific percentage is a guess
unless a lab documents it.

</details>

<details><summary>Why does data ordering not matter in Module 4's <code>get_batch</code>, and what changes with annealing?</summary>

<code>get_batch</code> samples windows uniformly at random from one packed stream,
so on average every part of the corpus is seen throughout training — order is
washed out. Annealing changes the <em>stream itself</em> over time (or switches
which source is sampled) so that late training draws from a higher-quality
mixture, usually alongside a decaying learning rate, letting the best data shape
the final weights.

</details>

## Next

You now have the full data pipeline: extraction, quality filtering,
deduplication, decontamination, mixing, and curriculum — everything that turns the
raw web into the ordered, weighted token stream a model trains on. With clean data
in hand, the next module returns to the *model*, upgrading the vanilla GPT-2 block
into the modern architecture (RoPE, RMSNorm, SwiGLU, GQA, MoE) used by current
frontier LLMs.

Continue to [12.1 · RoPE](lessons/module-12/lesson-01.md).
