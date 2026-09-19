# 04.2 · Byte-Pair Encoding from scratch

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="lesson-01.md">04.1 · Characters, bytes, Unicode, the vocabulary problem</a> — that UTF-8 turns any string into a sequence of byte ids in $0..255$, that this base alphabet of 256 symbols has no out-of-vocabulary problem, and that we want a <em>subword</em> unit sitting between long byte sequences and an unbounded word vocabulary.</p>
<p><strong>You will learn:</strong> the Byte-Pair Encoding <strong>training</strong> algorithm — start from bytes, repeatedly count adjacent symbol pairs, merge the most frequent pair into a new symbol, record the merge, repeat — worked by hand on a tiny corpus with the first three merges shown explicitly. Then <strong>encode</strong> (apply the learned merges in order) and <strong>decode</strong> (expand ids back to bytes), why the two invert each other exactly, and how the token count shrinks as merges grow. Finally how GPT-2's byte-level BPE adds a regex pre-tokenization split.</p>
<p><strong>Why this matters for ML:</strong> this is the actual tokenizer that sits in front of GPT-2 and, in the same family, most models since. You will implement it in <code>llmre.tokenizer.bpe</code> and reuse it for every training run in the course. Understanding it removes the last piece of "magic" between raw text and the integer tensors the model consumes, and it explains concrete production facts — why a token is "about ¾ of a word", why leading spaces matter, why numbers tokenize oddly.</p>
</div>

## 1. Intuition: build a vocabulary by fusing what co-occurs

Start from the byte view of a corpus: a long sequence of symbols drawn from the 256 byte values. Some adjacent pairs of symbols show up far more often than others — in English text `t` is very often followed by `h`, `e` by `s`, a space by `t`. BPE's idea is simple and greedy:

> Find the most frequent adjacent pair in the whole corpus, and glue it into a single new symbol. Add that symbol to the vocabulary. Repeat.

Each glue operation is a **merge**. The first merge might fuse `e`+`s` into a new symbol `es`; a later one might fuse that `es` with `t` to make `est`; a later one still might fuse `est` with a following space to make the common word-ending `est `. After a few thousand merges, the symbols that survive are exactly the byte-chunks that recur in the language: whole common words, frequent prefixes and suffixes, and — as a fallback that never disappears — the raw bytes for anything rare.

The name is literally the algorithm: we repeatedly encode the most frequent **pair** of adjacent symbols (originally bytes) as one. It was a data-compression scheme first; using it to build tokenizer vocabularies is the twist that made it central to LLMs.

<div class="callout key"><p>BPE training is a loop of exactly four steps: (1) count every adjacent symbol pair in the corpus, (2) pick the most frequent pair, (3) mint a new symbol id for it and record the merge, (4) rewrite the corpus with that pair fused. Stop when the vocabulary reaches the target size. Encoding later <em>replays</em> the recorded merges; decoding expands ids back to bytes.</p></div>

## 2. The setup: a tiny corpus and its byte view

We will train on a deliberately tiny corpus so we can do the arithmetic by hand and then check it against Python:

```text
newest newest widest widest lowest
```

Every character here is ASCII, so — from [04.1](lesson-01.md) — each character is a single byte equal to its code point (`e` is 101, `s` is 115, `t` is 116, the space is 32, and so on). That is a convenience for reading the example: the initial symbol sequence is just the characters, one byte each. (On non-ASCII text the exact same algorithm runs, but the starting symbols would be multi-byte, as `é → [195, 169]` was last lesson.)

The corpus is **34 bytes** long (32 letters plus 2 spaces). That 34 is our starting sequence length; watch it fall as we merge.

## 3. The training algorithm, worked by hand

We target a small vocabulary: the 256 bytes plus a handful of merges. Let us do the first three merges explicitly.

### 3.1 Merge 0 — count pairs, pick the winner

List the adjacent pairs and their counts. The word `newest` is `n e w e s t`, giving pairs `ne, ew, we, es, st`; `widest` is `w i d e s t` giving `wi, id, de, es, st`; `lowest` is `l o w e s t` giving `lo, ow, we, es, st`. Tallying the frequent ones across all five words (two `newest`, two `widest`, one `lowest`):

| pair | occurrences | count |
|------|-------------|-------|
| `e s` | once in every word | **5** |
| `s t` | once in every word | **5** |
| `t ` (t + space) | end of the first four words | 4 |
| `w e` | `newest`×2, `lowest`×1 | 3 |

Both `es` and `st` occur 5 times — a tie. We break ties deterministically by taking the pair that appears first when scanning the corpus left to right; in `newest` the pair `e s` occurs before `s t`, so **`e s` wins**. We mint it as new symbol id **256** and record the merge `(101, 115) → 256` (those are the byte ids of `e` and `s`). Rewriting the corpus with every `e s` fused drops the length from 34 to **29** (five fusions, each removing one symbol).

### 3.2 Merge 1 — the new symbol can itself be merged

Now recount pairs *in the rewritten sequence*, where `es` (id 256) is a single symbol. In every word the `es` symbol is immediately followed by `t`, so the pair `es t` (that is, `(256, 116)`) occurs **5** times — the most frequent. We mint id **257** for it and record `(256, 116) → 257`. The symbol 257 expands to the bytes of `es` followed by `t`, i.e. the three bytes spelling **`est`**. Length falls from 29 to **24**.

This is the crucial move: **merges compound**. A merged symbol is a first-class symbol that later merges can consume, so BPE grows units of length 3, 4, 8, … out of pairs, without ever handling more than two symbols at a time.

### 3.3 Merge 2 — merges cross the space boundary

Recount again, with `est` (257) now a single symbol. The first four words are each `... est` followed by a space, so the pair `est ` — symbol 257 followed by the space byte 32 — occurs **4** times, the new winner. We mint id **258** and record `(257, 32) → 258`. Symbol 258 expands to the four bytes spelling **`est `** (with a trailing space). Length falls from 24 to **20**.

Notice the space got swallowed into the token. This is normal and important for byte-level BPE: spaces are ordinary bytes, so tokens routinely include a leading or trailing space. That is why, in a real GPT-2 vocabulary, ` the` (with its space) is a single common token distinct from `the`.

### 3.4 The three merges, verified

Here is the whole hand computation confirmed by running `py`:

```text
raw byte length            : 34
merge 0: (e, s)   count 5  -> id 256 = "es"    length 34 -> 29
merge 1: (es, t)  count 5  -> id 257 = "est"   length 29 -> 24
merge 2: (est, _) count 4  -> id 258 = "est "  length 24 -> 20   (_ is a space)
```

The recorded merge table is, in order:

$$
(101,115)\to 256,\qquad (256,116)\to 257,\qquad (257,32)\to 258,
$$

and the vocabulary now has $256 + 3 = 259$ entries, the last three being `es`, `est`, `est `. The order of the merge table is not decoration — encoding must replay it in exactly this order, because merge 1 can only fire after merge 0 has produced symbol 256.

<div class="callout warn"><p>BPE is <strong>greedy</strong>, not optimal. At each step it fuses the single most frequent pair; it never reconsiders. A different tie-break or a different corpus gives a different vocabulary. There is no claim that the resulting units are the theoretically best subwords — only that this cheap, deterministic procedure produces good-enough units in practice. Two implementations that break ties differently will produce different (both valid) tokenizers.</p></div>

## 4. Encoding: replay the merges in order

Training produced an ordered list of merges. To **encode** a new string we turn it into bytes and then apply those merges, always choosing the pair whose merge came *earliest* in training among the pairs currently present. Concretely, repeat: look at all adjacent pairs in the current id list; of those that are in the merge table, fuse the one with the smallest merge id first; stop when no present pair is mergeable.

Why "earliest first"? Because a later merge may depend on a symbol an earlier merge creates (merge 1 needed the `es` from merge 0). Replaying in learned order reproduces the exact symbols training built. Encoding the word `newest` with our three-merge tokenizer:

```text
"newest" -> bytes [110, 101, 119, 101, 115, 116]        # n e w e s t
apply (e,s)->256 :  [110, 101, 119, 256, 116]           # n e w es t
apply (es,t)->257:  [110, 101, 119, 257]                # n e w est
(no more pairs are in the merge table)  ==> [110, 101, 119, 257]
```

Verified with `py`: `encode("newest")` returns `[110, 101, 119, 257]`. The common ending `est` collapsed to one token; the rarer prefix `new` stayed as its three bytes `n`, `e`, `w`. The merge `(est, space) -> 258` did *not* fire here because the standalone word `newest` has no trailing space — a nice reminder that which tokens you get depends on the surrounding bytes. Words `widest` and `lowest` likewise end in the single token `257`, sharing that suffix exactly as we wanted.

## 5. Decoding: expand ids back to bytes

Decoding is the easy direction and needs no merge logic at all. Each id maps to a fixed byte string in the vocabulary (`256 → b"es"`, `257 → b"est"`, a plain byte id like `110 → b"n"`), so we look up every id, concatenate the byte strings, and UTF-8-decode the result:

```text
[110, 101, 119, 257] -> b"n" + b"e" + b"w" + b"est" = b"newest" -> "newest"
```

Because `encode` only ever emits ids whose byte expansions concatenate back to the original UTF-8 stream, decoding inverts encoding **exactly**, for any string — the round-trip guarantee we demanded in [04.1](lesson-01.md). Even a never-seen emoji survives: no merge applies, so it stays as its raw bytes and decodes right back.

<div class="callout key"><p><strong>Encode</strong> = bytes, then greedily replay learned merges in training order. <strong>Decode</strong> = map each id to its stored bytes, concatenate, UTF-8-decode. Decode needs only the id→bytes vocabulary; encode needs the ordered merge list. They compose to the identity on every string.</p></div>

## 6. The from-scratch implementation

All of the above is implemented in `llmre.tokenizer.bpe` (`code/src/llmre/tokenizer/bpe.py`). The core is two tiny helpers plus the class. First, counting adjacent pairs:

```python
def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    counts = {}
    for a, b in zip(ids, ids[1:]):          # every adjacent pair (a, b)
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts
```

`zip(ids, ids[1:])` walks the sequence one step out of phase with itself, so it yields exactly the adjacent pairs — `[1,2,3]` gives `(1,2)` then `(2,3)`. Second, rewriting the sequence with one pair fused:

```python
def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)              # found the pair -> emit the merged id
            i += 2                          # and skip past both symbols
        else:
            out.append(ids[i])
            i += 1
    return out
```

Training is the four-step loop from section 1, and encoding is the "earliest merge first" replay from section 4:

```python
def train(self, text: str, vocab_size: int) -> None:
    ids = list(text.encode("utf-8"))                 # start from bytes
    for i in range(vocab_size - 256):                # one iteration per merge
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)             # most frequent adjacent pair
        if stats[pair] < 2:
            break                                    # nothing repeats -> stop early
        new_id = 256 + i
        ids = merge(ids, pair, new_id)               # fuse it everywhere
        self.merges[pair] = new_id                   # record the merge (ordered)
        self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

def encode(self, text: str) -> list[int]:
    ids = list(text.encode("utf-8"))
    while len(ids) >= 2:
        stats = get_stats(ids)
        # of the pairs present, take the one learned earliest (smallest merge id)
        pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
        if pair not in self.merges:
            break                                    # no present pair is mergeable
        ids = merge(ids, pair, self.merges[pair])
    return ids
```

Two implementation notes worth internalizing. In `train`, `self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]` concatenates the *byte strings* of the two children, so every id — byte or merged — always knows the exact bytes it stands for; that is what makes `decode` a pure lookup. In `encode`, `min(..., key=lambda p: self.merges.get(p, float("inf")))` gives any pair that was never learned an effectively infinite rank, so it is chosen only when nothing mergeable remains — at which point the guard `if pair not in self.merges` breaks the loop.

These are plain-Python `int`s and `bytes`: no tensors, no dtype, no device. The output is a `list[int]`; only the data loader ([04.3](lesson-03.md)) turns lists of ids into a `torch.long` tensor for the model.

<div class="callout pt"><p>This from-scratch loop is $O(n)$ work <em>per merge</em> because it re-scans the whole sequence each time — fine for the lessons' tiny corpora, far too slow for gigabytes. Production tokenizers (<code>tiktoken</code>, HuggingFace <code>tokenizers</code>) run the identical algorithm with a Rust core, incremental pair counts, and the regex split of section 8. Same math, engineered for throughput. We implement the mechanism first, exactly as the course rule demands, and only then reach for the fast library.</p></div>

## 7. Token count shrinks as merges grow

The whole point of the merges is to trade a bigger vocabulary for shorter sequences. Encoding our training corpus back with tokenizers trained to different sizes shows the trade directly (verified with `py`):

| merges learned | vocab size $V$ | encoded length of the corpus |
|----------------|----------------|------------------------------|
| 0 | 256 | 34 (= raw bytes) |
| 1 | 257 | 29 |
| 2 | 258 | 24 |
| 3 | 259 | 20 |
| 5 | 261 | 16 |
| 9 | 265 | 8 |

Zero merges is exactly the byte-level tokenizer — 34 bytes, 34 tokens. Each merge fuses a frequent pair and shortens the encoding; by 9 merges this toy corpus is 8 tokens. On real text the curve flattens (later merges are rarer and save less), which is why real vocabularies stop in the tens of thousands rather than chasing ever-shorter sequences. This table is the concrete face of the $V$-vs-$T$ trade-off from [04.1](lesson-01.md): moving *down* the rows buys shorter sequences with a wider embedding table.

## 8. GPT-2's byte-level BPE and the regex pre-split

The tokenizer you just built *is* byte-level BPE, the GPT-2 family. Real GPT-2 adds one refinement worth knowing about, even though we will not reimplement its exact form.

Before counting pairs, GPT-2 first splits the text with a regular expression into chunks — roughly, runs of letters, runs of numbers, and runs of punctuation, with leading spaces attached to the following word. BPE merges are then only allowed *within* a chunk, never across chunk boundaries. The reason is to stop the greedy merger from learning silly units that straddle a word and its neighbour's punctuation (like `dog.` or `the "`), and to keep numbers and words from fusing. This is why GPT-2 tends to tokenize a leading-space-plus-word as one clean token (` the`, ` dog`) and why long numbers break into digit-chunks.

Two honest caveats to keep straight (per the course's frontier-lab honesty rule):

- The GPT-2 regex, the byte-level BPE design, and the vocabulary size 50,257 are **publicly documented** (the GPT-2 paper and OpenAI's released code).
- The precise merge list of any *proprietary* newer model is generally **not** documented; do not assume a specific vocabulary for GPT-4-class models. What is safe to say is that they use the same byte-level-BPE-with-regex-presplit *family*.

<div class="callout paper"><p><strong>Research connection.</strong> Byte-level BPE with a regex pre-tokenization split is introduced in the GPT-2 paper, "Language Models are Unsupervised Multitask Learners" (Radford et al., 2019). Section 2.2 explains why they went byte-level: it guarantees no out-of-vocabulary inputs while keeping the vocabulary modest. See the reading guide in <a href="../../papers/index.md">the paper curriculum</a> (paper #1), which points at the exact figure and the LM objective this tokenizer feeds. The tokenizer is the front door to everything that paper does.</p></div>

## 9. A unit test that pins the behavior down

The tests in `code/tests/test_tokenizer.py` back every claim we made. The two load-bearing ones:

```python
def test_roundtrip_unicode_emoji_and_accents():
    tok = BPETokenizer()
    tok.train("café au lait, café crème, déjà vu", vocab_size=300)
    for s in ["café déjà vu 😀", "naïve façade — 日本語 🚀", "", "a"]:
        assert tok.decode(tok.encode(s)) == s     # exact round-trip, unseen chars

def test_training_creates_merges_and_compresses():
    tok = BPETokenizer()
    corpus = "low low low low lower lowest newest newest widest widest widest"
    raw_len = len(corpus.encode("utf-8"))
    tok.train(corpus, vocab_size=256 + 20)
    assert len(tok.merges) > 0                     # it actually learned merges
    assert len(tok.encode(corpus)) < raw_len       # and shortened the corpus
```

The first is the guarantee from [04.1](lesson-01.md): decode inverts encode for arbitrary Unicode, including emoji and scripts never seen in training, thanks to the byte fallback. The second checks that training does real work — it learns at least one merge and the encoded corpus is strictly shorter than its raw byte count. There is also `test_vocab_size_is_256_plus_num_merges`, asserting `tok.vocab_size == 256 + len(tok.merges)`. Run them from `code/` with `py -m pytest -q`.

## Exercise

Extend the tokenizer with a `save`/`load` pair so a trained tokenizer can be written to disk and restored, and confirm the restored tokenizer encodes identically.

<details><summary>Optional hint</summary>

What is the minimal state that fully determines `encode` and `decode`? You do not need to store the whole `vocab` dict — it is derivable. Think about what `train` records.

</details>

<details><summary>Stronger hint</summary>

`decode` needs `vocab` (id → bytes), and `encode` needs `merges` (ordered pair → id). But `vocab` can be rebuilt from `merges` by replaying them over the base 256 bytes, exactly as `train` does. So the only thing you must persist is the ordered merge list. Write the pairs in order; on load, rebuild `vocab` by concatenating child bytes in that order.

</details>

<details><summary>Solution</summary>

```python
def save(self, path: str) -> None:
    # One line per merge, in learned order: "<a> <b>" whose new id is 256 + line#.
    with open(path, "w", encoding="utf-8") as f:
        for (a, b), new_id in sorted(self.merges.items(), key=lambda kv: kv[1]):
            f.write(f"{a} {b}\n")

def load(self, path: str) -> None:
    self.merges = {}
    self.vocab = {i: bytes([i]) for i in range(256)}
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            a, b = map(int, line.split())
            new_id = 256 + i
            self.merges[(a, b)] = new_id
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
```

The key insight is that the ordered merge list is the *entire* learned state; `vocab` is a derived cache rebuilt by replaying merges over the 256 base bytes. Verify with:

```python
a = BPETokenizer(); a.train(corpus, 320); a.save("t.bpe")
b = BPETokenizer(); b.load("t.bpe")
assert a.encode("some text") == b.encode("some text")
```

(This is essentially the format GPT-2's released `merges.txt` uses.)

</details>

## Debugging exercise

A teammate writes their own `encode` and complains it produces different, longer token lists than training implied. Here is their loop. Find the bug.

```python
def encode_buggy(self, text):
    ids = list(text.encode("utf-8"))
    for pair, new_id in self.merges.items():        # <-- apply merges "in order"
        ids = merge(ids, pair, new_id)
    return ids
```

<details><summary>What is wrong, and why does it sometimes still look right?</summary>

The loop applies each merge **exactly once, in dict-insertion order**, instead of repeatedly picking the earliest-applicable pair until nothing is mergeable. In modern Python `self.merges` does preserve insertion (training) order, so for simple inputs this often gives the right answer and hides the bug. But it is wrong in general: applying merge $k$ can create a *new* occurrence of the pair from an *earlier* merge $j < k$ that a single left-to-right pass has already gone past, so that earlier merge should fire again and never does. The result is an under-merged, longer encoding.

The correct `encode` (section 6) loops on the *current* sequence — `while len(ids) >= 2`, each time choosing the present pair with the smallest merge id — so a newly exposed earlier pair still gets merged. Fix it by replacing the single `for` pass with the repeat-until-stable loop.

</details>

## Check yourself

<details><summary>In the worked example, why does the second merge <code>(es, t) → est</code> only become possible after the first merge, and what does this tell you about the merge list's ordering?</summary>

The pair `(es, t)` involves the symbol `es`, which does not exist until merge 0 creates it. So merge 1 is defined in terms of merge 0's output. This is why the merge list is *ordered* and why `encode` must replay merges earliest-first: a later merge can depend on a symbol an earlier merge produced. Shuffle the order and you get a different (or broken) tokenizer.

</details>

<details><summary>You train two BPE tokenizers on the same corpus but they break ties between equally-frequent pairs differently. Are both valid? Will they encode text identically?</summary>

Both are valid tokenizers — each round-trips correctly and covers all input via the byte fallback. But they are *different* tokenizers: they may learn different merges and therefore encode the same text into different id sequences of possibly different lengths. BPE is greedy and tie-breaking is a free choice, so "the BPE vocabulary" of a corpus is not unique. This is why you must ship the exact merge list with a model.

</details>

<details><summary>Encode the word <code>widest</code> with the three-merge tokenizer from section 3. How many tokens, and what are they?</summary>

`widest` is bytes `w i d e s t`. Merge 0 fuses `e s → es`; merge 1 fuses `es t → est`. No trailing space, so merge 2 does not fire. Result: `w`, `i`, `d`, `est` — **4 tokens** `[119, 105, 100, 257]`. It shares the single suffix token `257` (`est`) with `newest` and `lowest`, which is the morphology-sharing we wanted.

</details>

<details><summary>Why does <code>decode</code> need only the id→bytes vocabulary and not the ordered merge list, while <code>encode</code> needs the order?</summary>

Decoding is a context-free lookup: each id expands to a fixed byte string regardless of its neighbours, so concatenating expansions reconstructs the bytes with no ordering decisions. Encoding must *choose* which pairs to fuse and in which sequence, and those choices depend on the training order (a later merge may need an earlier merge's symbol). So encode carries the ordered merge list; decode carries only the vocabulary.

</details>

<details><summary>On real text, why do the later merges each shorten sequences less than the early ones, and what does that imply about picking a vocabulary size?</summary>

Early merges capture the most frequent pairs (very common letters and words), so they fire on a large fraction of the corpus and remove many symbols. Later merges target progressively rarer pairs, firing on less text and saving fewer tokens each — diminishing returns. So sequence length falls fast at first and then flattens, which is why real vocabularies stop in the tens of thousands: past that, extra rows cost embedding parameters and gradient sparsity for little length savings.

</details>

## Next

You can now train, encode, and decode a byte-level BPE tokenizer, and you understand the trade-off its vocabulary size controls. But a tokenizer alone does not feed a model. The next lesson wires it into the data pipeline: special tokens like `<|endoftext|>` to separate documents, packing a whole corpus into one long id stream, and cutting that stream into the `(x, y)` next-token training windows the model actually trains on.

Continue to [04.3 · Special tokens, packing & data loading](lesson-03.md).
