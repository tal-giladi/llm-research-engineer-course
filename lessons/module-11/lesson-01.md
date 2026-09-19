# 11.1 · Common Crawl → clean text

<div class="prereq">
<p><strong>Prerequisites:</strong> the data pipeline that turns documents into training windows — packing, the <code>&lt;|endoftext|&gt;</code> separator, <code>get_batch</code> — from <a href="../module-04/lesson-03.md">04.3 · Special tokens, packing &amp; data loading</a>; a working BPE tokenizer from <a href="../module-04/lesson-02.md">04.2 · Byte-Pair Encoding from scratch</a>; comfort reading Python string code. No new math.</p>
<p><strong>You will learn:</strong> what actually sits between "the web" and the id stream Module 4 packs — the real pretraining data pipeline: Common Crawl WARC/WET files → text extraction → language identification → <strong>quality filtering</strong> (Gopher/C4-style heuristics) → toxicity and <strong>PII</strong> handling. You will build a small <code>quality_filter(text) -&gt; bool</code> from a few cheap metrics and watch it keep clean prose while rejecting too-short, symbol-spam, and repetitive junk.</p>
<p><strong>Why this matters for ML:</strong> at frontier scale, data quality moves the loss more than almost any architecture tweak. A one-line <code>load_dataset(...)</code> is <em>not</em> a data pipeline — it is the last step after someone else already did the filtering. If you want to train a real model, or understand why one model is better than another trained with the same FLOPs, you have to understand this stage. Most of the engineering effort in a pretraining project lives here, not in the model code.</p>
</div>

## 1. Intuition: `load_dataset(...)` is not a data pipeline

When you learned Module 4 you had a clean corpus already: some documents, encode
them, pack them, sample windows. That is the *tail* of the data pipeline. This
lesson is the *head* — where the documents come from and why most of what the
web offers never makes it into the tensor.

Here is the uncomfortable reality of pretraining data. The raw material is a
crawl of the public web: hundreds of terabytes of HTML, most of it navigation
menus, cookie banners, SEO spam, link farms, machine-translated gibberish,
adult content, and exact copies of the same article on a thousand mirror sites.
A frontier model is trained on the small, clean fraction that survives an
aggressive filtering funnel. The funnel is the product.

<div class="callout key"><p>The pretraining data pipeline is a funnel:
<strong>raw crawl → extracted text → language-filtered → quality-filtered →
deduplicated → decontaminated → mixed → tokenized → packed</strong>. Module 4
built the last two boxes. This module builds the middle ones. Each stage throws
away far more than it keeps; a common outcome is that a few percent of the raw
bytes reach the model.</p></div>

The classic mistake, and the reason the section title exists, is to think that
because `datasets.load_dataset("some/corpus")` returns clean-looking text in one
line, the data problem is solved. It is not — someone ran this entire funnel to
produce that corpus, and their filtering choices are baked into every token. When
you train your own model on your own crawl, that work is yours.

## 2. Where the text comes from: Common Crawl, WARC and WET

**Common Crawl** is a nonprofit that crawls the web every month or two and
publishes the result for free. It is the raw feedstock behind most open
pretraining corpora (C4, RefinedWeb, FineWeb, Dolma, and the web slice of nearly
every LLM). One monthly crawl is on the order of a few billion pages.

Common Crawl ships two file formats you need to know by name:

- **WARC** (Web ARChive) — the *full* record of each fetch: the HTTP request,
  the HTTP response headers, and the complete raw HTML body. Everything.
- **WET** (WARC Encapsulated Text) — a pre-extracted *plain-text* version, where
  Common Crawl has already stripped the HTML tags for you.

It is tempting to just take WET and skip the extraction step. In practice the
serious corpora start from **WARC and do their own extraction**, because WET's
generic tag-stripping keeps a lot of boilerplate (menus, footers, "related
articles") that a purpose-built extractor removes. The quality of your text
extraction sets a ceiling on everything downstream.

<div class="callout pt"><p>A WARC file is just gzipped concatenated records. You
stream it, not load it: iterate record by record with a library like
<code>warcio</code> and process each page as it goes past, because a single
crawl shard is far larger than RAM. This is the same streaming discipline as the
<code>memmap</code> note in <a href="../module-04/lesson-03.md">04.3</a> — at this
scale nothing fits in memory, so everything is a stream.</p></div>

## 3. Stage one — text extraction (boilerplate removal)

**Goal:** turn one page's raw HTML into just its main textual content — the
article body — and drop navigation, ads, cookie notices, sidebars, and footers
(collectively, *boilerplate*).

This is a genuinely hard problem (HTML is a mess of nested tags, inline scripts,
and templated cruft), and it is a *solved* problem in the sense that good open
tools exist. **You do not hand-write an HTML parser for this.** The two tools
worth knowing:

- **trafilatura** — a Python library specialized in extracting the main text
  (and optionally metadata) from a web page, with boilerplate removal tuned for
  exactly this use case.
- **resiliparse** — part of the ChatNoir web-data stack; a very fast HTML-to-text
  extractor used in several large corpora when throughput matters.

Both take raw HTML in and give clean-ish main text out. Conceptually:

```python
# Illustrative — you would run this over millions of records, streamed.
import trafilatura

def extract_text(raw_html: str) -> str | None:
    # Returns the main body text, or None if the page has no usable content.
    return trafilatura.extract(raw_html)
```

The reason we do not reimplement this in the course is the same reason we use a
real BPE later: the *mechanism* (walk the DOM, score blocks by text density,
keep the dense ones) is understandable, but a production-grade extractor encodes
years of edge-case handling. Know what it does and what its failure modes are
(it sometimes drops legitimate content, sometimes keeps a stubborn footer); do
not write your own.

## 4. Stage two — language identification

**Goal:** keep documents in the language(s) you are training on (say, English),
and route or drop the rest.

The standard tool is a **fastText language-id classifier**: a tiny, fast linear
model over character/word n-gram features that, given a string, returns a
predicted language label and a confidence score. You keep a document if its top
language is the one you want *and* the confidence clears a threshold (say
`0.65`). The threshold matters: low-confidence documents are often code-switched,
very short, or garbled, and are worth dropping regardless of the predicted
label.

You will not train a real langid model here, but the *shape* of the idea is easy
to illustrate with a crude frequency heuristic — how many of a document's words
are common English stop-words:

```python
# Illustrative toy — NOT how production langid works, but shows the shape:
# a cheap score in [0,1] that a threshold turns into a keep/drop decision.
_EN_STOP = {"the", "of", "and", "to", "in", "a", "is", "that", "it", "for"}

def english_score(text: str) -> float:
    toks = text.lower().split()
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in _EN_STOP)
    return hits / len(toks)          # fraction of stop-words

# keep_if English-like:
#   english_score(doc) > 0.05   ->   plausibly English prose
```

Real fastText langid is far more accurate and covers 176 languages, but the
control flow is identical: **score → threshold → keep/drop**. Notice the
stop-word idea reappears in the next stage as a *quality* signal too; presence of
common function words is a strong "this is real prose" signal.

## 5. Stage three — quality filtering (the heart of this lesson)

**Goal:** among the correctly-extracted, right-language documents, keep the ones
that look like *real writing* and drop machine junk, list-spam, and degenerate
text. This is where the well-known **Gopher** (Rae et al., 2021) and **C4**
(Raffel et al., 2020) heuristics live.

The insight is that you do not need a neural classifier to catch most junk — a
handful of cheap, interpretable statistics per document does the bulk of the
work. We implement a representative subset. All of them operate on a raw Python
`str` (one document) and return plain Python numbers on the CPU — there is no
tensor, dtype, or device here, because this all happens *before* tokenization.

### 5.1 The metrics

We use four signals (from `code/src/llmre/data/quality.py`):

1. **Word count** — too short (a nav stub, a title) teaches nothing; absurdly
   long is usually a log dump or concatenation artifact.
2. **Mean word length** — real English prose averages ~4–6 characters per word.
   Far below signals space-spam or tokenized junk; far above signals hash-like
   or URL-like garbage.
3. **Symbol ratio** — the fraction of non-whitespace characters that are neither
   letters nor digits. Clean prose is a few percent (just punctuation); markup
   spam and ASCII art are much higher.
4. **Repetition ratio** — `1 - unique_words / total_words`. Natural text repeats
   function words so this is never zero, but "buy buy buy …" spam pushes it near
   `1`.

### 5.2 Mathematics, every symbol named

Let a document tokenize (on `\w+` runs) into words $w_1, \dots, w_n$, so $n$ is
the word count. Let $|w_i|$ be the character length of word $i$.

**Mean word length:**

$$
\text{mwl} = \frac{1}{n}\sum_{i=1}^{n} |w_i|.
$$

**Symbol ratio.** Let $c_1, \dots, c_M$ be the *non-whitespace* characters of the
document ($M$ of them), and let $\mathbb{1}[\cdot]$ be $1$ when its condition
holds and $0$ otherwise. A character is a "symbol" if it is not alphanumeric:

$$
\text{sym} = \frac{1}{M}\sum_{j=1}^{M} \mathbb{1}\big[c_j \text{ is not a letter or digit}\big].
$$

**Repetition ratio.** Let $U$ be the number of *distinct* lower-cased words and
$n$ the total:

$$
\text{rep} = 1 - \frac{U}{n}.
$$

A document is **kept** only if it clears every threshold at once — one bad signal
is enough to drop it.

### 5.3 Numerical example, worked by hand

Take the junk document

```text
Great deal!!! Buy now, buy now >>> click <<<
```

Tokenizing on `\w+` gives the $n = 7$ words
`Great, deal, Buy, now, buy, now, click`.

**Mean word length.** Character lengths are $5, 4, 3, 3, 3, 3, 5$, summing to
$26$, so

$$
\text{mwl} = \frac{26}{7} = 3.714.
$$

**Symbol ratio.** Strip whitespace; there are $M = 36$ non-whitespace
characters. The non-alphanumeric ones are the ten characters
`! ! ! , > > > < < <`. So

$$
\text{sym} = \frac{10}{36} = 0.278.
$$

**Repetition ratio.** Lower-cased, the distinct words are
`great, deal, buy, now, click`, so $U = 5$ out of $n = 7$:

$$
\text{rep} = 1 - \frac{5}{7} = \frac{2}{7} = 0.286.
$$

With a symbol-ratio ceiling of $0.10$, this document is **rejected** on the
symbol ratio alone ($0.278 > 0.10$) — before we even weigh the repetition. That
matches the intuition: it is an ad, not prose.

Verify it:

```python
from llmre.data.quality import mean_word_length, symbol_ratio, repetition_ratio
t = "Great deal!!! Buy now, buy now >>> click <<<"
print(mean_word_length(t))   # 3.7142857142857144
print(symbol_ratio(t))       # 0.2777777777777778
print(repetition_ratio(t))   # 0.2857142857142857
```

### 5.4 From-scratch implementation

Here is the predicate, from the course codebase. It is deliberately small and
readable — a subset of the real Gopher/C4 rule set:

```python
def quality_filter(
    text: str,
    *,
    min_words: int = 20,
    max_words: int = 100_000,
    min_mean_word_length: float = 3.0,
    max_mean_word_length: float = 10.0,
    max_symbol_ratio: float = 0.10,
    max_repetition_ratio: float = 0.40,
) -> bool:
    """Return True to KEEP the document, False to drop it."""
    ws = words(text)
    n = len(ws)
    if n < min_words or n > max_words:
        return False
    mwl = mean_word_length(text)
    if mwl < min_mean_word_length or mwl > max_mean_word_length:
        return False
    if symbol_ratio(text) > max_symbol_ratio:
        return False
    if repetition_ratio(text) > max_repetition_ratio:
        return False
    return True
```

The full helpers `words`, `mean_word_length`, `symbol_ratio`, and
`repetition_ratio` are in `code/src/llmre/data/quality.py`, each documented with
its return type.

### 5.5 Accept the good, reject the junk

Four documents through the filter — one real paragraph and three kinds of junk:

```python
from llmre.data.quality import quality_filter

clean = ("The transformer architecture replaced recurrent networks for most "
         "language tasks because attention lets every position read every other "
         "position in a single step.")            # 23 words
print(quality_filter(clean))                        # True

print(quality_filter("Click here to win now."))     # False  (5 words < min_words)
print(quality_filter("buy " * 60))                  # False  (rep = 0.983)
print(quality_filter("### >>> @@@ *** ||| " * 5))   # False  (no words / all symbols)
```

Measured signals for these, to see *why* each verdict falls out:

| document | words | mwl | sym | rep | verdict |
|---|---|---|---|---|---|
| clean paragraph | 23 | 6.04 | 0.007 | 0.087 | **keep** |
| "Click here to win now." | 5 | 3.40 | 0.056 | 0.000 | drop (too short) |
| "buy " ×60 | 60 | 3.00 | 0.000 | 0.983 | drop (repetition) |
| "### >>> @@@ …" | 0 | 0.00 | 1.000 | 0.000 | drop (no words) |

The clean paragraph is the only one that clears all four gates. Each junk
document trips a different gate — which is exactly why you want several cheap
signals rather than one: different failure modes hide from different metrics.

<div class="callout warn"><p><strong>Filters are corpus-specific and lossy.</strong>
These thresholds are teaching values. Tuned too aggressively, a quality filter
throws away good text (code, poetry, and dialogue all violate "prose" heuristics);
tuned too loosely, junk leaks through and wastes compute. Real projects
<em>inspect samples of what each rule drops</em> before trusting it. Never ship a
filter you have not eyeballed the rejects of.</p></div>

## 6. Real quality filtering goes further: model-based scoring

The heuristics above are the cheap first pass. Modern corpora add a
**model-based quality classifier**: a small classifier trained to distinguish
"high-quality" reference text (for example, pages resembling Wikipedia or
well-edited articles) from random crawl, then used to score and rank every
document. GPT-3's data pipeline (Brown et al., 2020) did exactly this — a
classifier trained to recognize WebText-like pages, used to filter Common Crawl.
The FineWeb-Edu corpus took it further, using an LLM to label educational
quality and training a classifier on those labels.

The point for you: heuristics remove the obvious junk cheaply; a learned scorer
captures "is this well-written and informative", which no simple ratio can. Both
layers are normal. We build only the heuristic layer here because it is the part
you can fully understand and implement in an afternoon.

## 7. Toxicity filtering and PII

Two more filters run in this stage, both about *what should not be in the model*
rather than *what teaches it well*.

**Toxicity / safety filtering.** Crawls contain hate speech, explicit content,
and abuse. Pipelines run a toxicity classifier (or block-lists of domains and
terms) and drop or down-weight documents that score high. This is a policy
decision as much as a quality one, and the thresholds are chosen deliberately —
over-filtering removes legitimate discussion of sensitive topics, under-filtering
teaches the model things you do not want it to say.

**PII (Personally Identifiable Information).** PII is data that identifies a real
person: names tied to addresses, phone numbers, email addresses, government IDs,
credit-card numbers, and so on. It matters because a language model can
**memorize** and later **regurgitate** its training data verbatim — so PII in the
corpus is a privacy leak waiting to happen, and in many jurisdictions a legal
liability. Pipelines therefore **detect and redact or drop** PII: for example,
replacing detected emails/phone numbers with placeholder tokens, or removing
documents dense with personal data.

<div class="callout warn"><p>We deliberately do <strong>not</strong> build a real
PII detector in this course. A toy regex for emails would give a false sense of
safety — real PII detection is a specialized, high-stakes system (missing a
Social Security number is a serious failure), and a half-working one is worse than
none. Know that this stage exists, know why (memorization → regurgitation →
privacy/legal harm), and use a vetted tool if you ever run a real pipeline.</p></div>

## 8. Where this hands off to Module 4

Everything in this lesson runs *before* the code you already have. The output of
this funnel — clean, right-language, quality-passing, decontaminated text — is
exactly the "pile of documents" that <a href="../module-04/lesson-03.md">04.3</a>
assumed as its input. The hand-off is literally:

```text
raw WARC ─► extract ─► langid ─► quality_filter ─► dedup (11.2) ─► decontam (11.3)
        ─► [ documents ] ─► BPE encode ─► pack_documents ─► get_batch ─► model
                              └────────────  Module 4  ────────────┘
```

Only the surviving documents are tokenized and packed. Every earlier gate makes
the packed stream smaller and cleaner, which is the whole point: at fixed
compute, cleaner tokens buy lower loss.

## Exercise

You are handed a document that is one long comma-separated list of city names,
about 500 words, e.g. `"Paris, London, Tokyo, Berlin, Madrid, Rome, ..."`. Before
running any code, predict which of the four `quality_filter` gates (word count,
mean word length, symbol ratio, repetition ratio) it is most likely to fail, and
which it will pass.

<details><summary>Hint</summary>
Count what a comma-separated list does to each metric. City names are ordinary
words; there is one comma per word; and cities repeat rarely in a 500-item list.
</details>

<details><summary>Stronger hint</summary>
Word count: ~500, comfortably in range. Mean word length: city names are ~5–7
chars, in range. Repetition: a list of distinct cities is <em>low</em>
repetition. That leaves one metric — count the punctuation.
</details>

<details><summary>Solution</summary>

It most likely fails on **symbol ratio**, and passes the other three. With one
comma (and often a space) per word, roughly one non-whitespace symbol per ~6
letters gives a symbol ratio around $1/7 \approx 0.14$, above the $0.10$ ceiling.
Word count (~500) is fine, mean word length (~6) is fine, and repetition is
*low* because the cities are mostly distinct. This is a good example of why the
symbol-ratio gate exists: list-spam looks fine on every metric except its
punctuation density. (Real Gopher filters add a dedicated "fraction of lines
ending in a bullet/list marker" rule for exactly this pattern.)

</details>

## Common mistakes

- **Trusting a corpus because it loaded cleanly.** `load_dataset` returning tidy
  text means *someone else* filtered it; their choices are now yours, sight
  unseen.
- **Skipping extraction and using WET blindly.** Generic tag-stripping keeps
  boilerplate; the corpora that win re-extract from WARC.
- **Tuning filters without inspecting rejects.** A filter that quietly deletes
  all your code or all your dialogue can tank a downstream capability while every
  aggregate metric looks fine.
- **Rolling your own PII/toxicity detector.** High-stakes, specialized, easy to
  get subtly and dangerously wrong. Use vetted tools.

## Research connection

<div class="callout paper"><p>GPT-3's paper (Brown et al., 2020) documents the
Common-Crawl pipeline used for a frontier model: a learned quality classifier to
filter crawl toward high-quality reference text, plus fuzzy deduplication (next
lesson). Read its data section as the canonical "this is what the funnel looks
like at scale". See <a href="../../papers/index.md">the paper curriculum</a> (GPT-3,
and Llama 3 / OLMo 2 for modern data-quality and annealing details).</p></div>

## Check yourself

<details><summary>Why is "<code>dataset = load_dataset(...)</code> is not a data pipeline" the slogan of this lesson?</summary>

Because that one line only <em>loads</em> a corpus someone already built by running
the whole extraction → langid → quality → dedup → decontamination funnel. The
filtering choices — which set the quality ceiling of anything you train — are
already baked in. Training your own model on your own crawl means doing that work
yourself; it is where most of the engineering effort goes.

</details>

<details><summary>What is the difference between a Common Crawl WARC file and a WET file, and why do serious corpora start from WARC?</summary>

WARC holds the full fetch including the raw HTML body; WET holds Common Crawl's
own generic plain-text extraction. Serious corpora re-extract from WARC with a
purpose-built extractor (trafilatura/resiliparse) because WET's generic
tag-stripping leaves boilerplate (menus, footers) that hurts quality, and
extraction quality caps everything downstream.

</details>

<details><summary>A document scores mean word length 6.0, symbol ratio 0.03, repetition ratio 0.05, and has 12 words. With the default thresholds, is it kept?</summary>

No. Three metrics are fine, but the word count is $12 < \text{min\_words} = 20$,
and <code>quality_filter</code> requires <em>all</em> gates to pass. One failing
gate drops the document. (Short snippets are usually nav stubs or titles, not
training-worthy prose.)

</details>

<details><summary>Why is PII in the training corpus a problem specifically for a language model, and why don't we build a PII detector here?</summary>

Language models can memorize training data and regurgitate it verbatim at
generation time, so PII in the corpus becomes a privacy leak (and legal
liability). We don't build a detector because real PII detection is a
specialized, high-stakes system — a toy regex would give false confidence and
missing, say, a government ID is a serious failure. Know the stage exists; use a
vetted tool for the real thing.

</details>

<details><summary>You have two cheap signals that both catch spam. Why keep both instead of the stronger one?</summary>

Different junk hides from different metrics: "buy buy buy" trips repetition but
has a fine symbol ratio; markup spam trips symbol ratio but may have fine
repetition; a nav stub trips word count and neither of the others. Several cheap,
orthogonal gates catch more failure modes than any single one, at negligible
cost.

</details>

## Next

Clean, right-language, quality-passing documents still contain a huge amount of
*duplication* — the same article mirrored across sites, the same paragraph
templated onto thousands of pages. Duplicated data wastes compute and worsens
memorization. The next lesson builds exact and fuzzy deduplication, and the
MinHash trick that makes near-duplicate detection tractable at web scale.

Continue to [11.2 · Deduplication & MinHash](lesson-02.md).
