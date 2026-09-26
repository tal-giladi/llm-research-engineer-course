# 11.2 · Deduplication & MinHash

<div class="prereq">
<p><strong>Prerequisites:</strong> the data funnel and where dedup sits in it from <a href="#/lessons/module-11/lesson-01">11.1 · Common Crawl → clean text</a>; basic hashing (a hash function maps data to a fixed-size integer; equal inputs give equal hashes) and set operations (union, intersection). A little probability from <a href="#/lessons/module-01/lesson-01">01.1 · Random variables, expectation, variance</a> helps but is not required.</p>
<p><strong>You will learn:</strong> why duplicated training data is actively harmful; <strong>exact</strong> deduplication by hashing whole documents; <strong>fuzzy / near-duplicate</strong> detection by representing a document as a set of k-shingles and measuring <strong>Jaccard similarity</strong>; how <strong>MinHash</strong> estimates Jaccard cheaply because, for a random hash, $P(\text{minhash}(A)=\text{minhash}(B)) = J(A,B)$; and how <strong>LSH banding</strong> finds near-duplicate pairs without comparing all $O(n^2)$ pairs. You will work a tiny MinHash example fully by hand and verify it.</p>
<p><strong>Why this matters for ML:</strong> the web is enormously redundant — the same article is mirrored, quoted, and templated across thousands of pages. Training on duplicates wastes compute (you pay to learn the same tokens many times), worsens <em>memorization</em> (the model overfits repeated text and can regurgitate it), and inflates evaluation via contamination. Every serious pretraining corpus is deduplicated, and MinHash + LSH is the standard way to do it at scale.</p>
</div>

## 1. Intuition: why duplicates hurt

Suppose the same 2,000-word article appears 500 times across your crawl (mirror
sites, scrapers, "content" farms). If you train on all 500 copies, three bad
things happen:

1. **Wasted compute.** You spend 500× the FLOPs teaching the model that one
   article instead of 500 different ones. At a fixed compute budget (Module 10),
   every duplicated token is a token you did *not* spend on new information.
2. **Memorization.** Text the model sees many times gets *memorized* rather than
   generalized. A heavily duplicated passage can be reproduced verbatim by the
   trained model — a privacy and copyright hazard, and evidence the model spent
   capacity storing instead of learning.
3. **Evaluation contamination.** If a duplicated document happens to overlap a
   benchmark item, the model has effectively seen the test — inflating scores.
   (That is the subject of the next lesson.)

Deduplication removes this redundancy. There are two flavors, and you need both.

<div class="callout key"><p><strong>Exact dedup</strong> removes documents that are
byte-for-byte identical: hash each document, drop repeated hashes.
<strong>Fuzzy dedup</strong> removes documents that are <em>nearly</em> identical
— same article with a changed date, an added footer, or minor edits — which exact
hashing misses entirely because one different byte gives a completely different
hash.</p></div>

## 2. Exact deduplication by hashing

The simplest and cheapest filter. To find byte-identical duplicates, hash each
document to a fixed-size fingerprint and keep only the first occurrence of each
fingerprint. Comparing 64-bit hashes is far cheaper than comparing full document
strings, and a set membership test is $O(1)$ on average.

```python
def dedup_exact(docs: list[str]) -> list[str]:
    seen: set[int] = set()
    kept: list[str] = []
    for d in docs:
        h = hash(d)                 # a stable content hash in a real pipeline
        if h not in seen:
            seen.add(h)
            kept.append(d)
    return kept

docs = ["hello world foo", "hello world foo", "a completely different doc"]
print(dedup_exact(docs))
# ['hello world foo', 'a completely different doc']   -- the exact copy is gone
```

The second `"hello world foo"` is dropped because its hash collides with the
first. This catches true copies at negligible cost.

<div class="callout warn"><p>In a real pipeline use a <em>stable</em> content hash
(e.g. <code>hashlib.sha1</code> / <code>blake2b</code> of the bytes), not Python's
built-in <code>hash()</code>: the built-in is salted per process, so its values
differ between runs and cannot be persisted or sharded. We use <code>hash()</code>
above only for a one-shot in-memory illustration.</p></div>

**The limitation.** Change one byte — a timestamp, a visitor counter, a "Share
on X" line — and the hash changes completely. Exact dedup then sees two different
documents where a human sees the same article. The web is full of exactly this,
so we need a similarity measure that tolerates small edits.

## 3. Fuzzy dedup: documents as sets of shingles

To measure *near*-duplication we first turn each document into a **set**, then
measure how much two sets overlap.

A **k-shingle** (a.k.a. k-gram) is a window of `k` consecutive units — here, `k`
consecutive words joined by spaces. Sliding the window across the document
produces its shingle set. Two documents that share long runs of words share many
shingles; a small edit changes only the handful of shingles that span the edit.

Example with `k = 3` on `"the quick brown fox jumps"`:

```python
from llmre.data.minhash import shingles
print(shingles("the quick brown fox jumps", k=3))
# {'the quick brown', 'quick brown fox', 'brown fox jumps'}
```

Three words, slid one at a time, give three shingles. The document is now the
*set* `{'the quick brown', 'quick brown fox', 'brown fox jumps'}` — order and
repetition collapsed away. That set representation is what we compare.

<div class="callout pt"><p>Why <em>sets</em> of shingles rather than the raw word
sequence? Because near-duplicate detection cares about "how much content do these
share", not "in what order". Two pages that reorder a few paragraphs are still
duplicates; their shingle sets still overlap heavily. Larger <code>k</code> makes
accidental overlap between unrelated documents vanishingly unlikely (a shared run
of 8 exact words is rarely a coincidence); typical values are 5–10 words.</p></div>

## 4. Jaccard similarity: the quantity we want

Given two sets $A$ and $B$ (shingle sets of two documents), their **Jaccard
similarity** is the size of their intersection over the size of their union:

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}.
$$

- $J = 1$: identical sets (every shingle shared).
- $J = 0$: disjoint (no shingle shared).
- In between: the fraction of all distinct shingles that both documents contain.

A dedup pass declares two documents near-duplicates when $J$ exceeds a threshold
(commonly around $0.8$). Every symbol here is a set count — nothing is a tensor;
this is CPU set arithmetic.

```python
from llmre.data.minhash import shingles, jaccard
a = shingles("the cat sat on the mat", k=2)
b = shingles("the cat sat on the rug", k=2)
print(jaccard(a, b))    # fraction of 2-word shingles shared
```

## 5. The problem: exact Jaccard is too expensive at scale

Computing $J$ exactly means storing every document's full shingle set and
intersecting sets. A single document can have thousands of shingles; a corpus has
*billions* of documents. Two costs blow up:

1. **Storage / comparison per pair.** Intersecting two big shingle sets is
   expensive, and you would do it for many pairs.
2. **Number of pairs.** Checking every pair of $n$ documents is $O(n^2)$ — for
   $n = 10^9$ that is $10^{18}$ comparisons, utterly infeasible.

MinHash attacks cost #1 (estimate $J$ from tiny fixed-size signatures); LSH
banding attacks cost #2 (avoid looking at most pairs at all). Take them in turn.

## 6. MinHash: estimating Jaccard from a tiny signature

Here is the beautiful fact MinHash rests on. Imagine applying a random
**permutation** (a shuffle) to the universe of all possible shingles, which
assigns every shingle a rank. Define the **MinHash** of a set as the shingle in
it with the smallest rank:

$$
\text{minhash}(A) = \arg\min_{x \in A} \pi(x),
$$

where $\pi$ is the random permutation. Now ask: what is the probability that two
sets have the *same* MinHash?

The element with the smallest rank in $A \cup B$ is equally likely to be any
element of the union (the permutation is random). The two sets share a MinHash
exactly when that overall-smallest element lies in *both* sets, i.e. in
$A \cap B$. So

$$
P\big(\text{minhash}(A) = \text{minhash}(B)\big)
= \frac{|A \cap B|}{|A \cup B|} = J(A, B).
$$

<div class="callout key"><p>For a single random permutation, the probability that
two sets collide under MinHash <em>equals</em> their Jaccard similarity. So each
independent permutation is a coin flip that comes up "match" with probability
exactly $J$. Average many such flips and you get an unbiased estimate of $J$ —
from tiny fixed-size signatures instead of the full sets.</p></div>

Use $m$ independent permutations. The **signature** of a set is the length-$m$
vector of its MinHash values, one per permutation. To estimate $J$ from two
signatures, count the fraction of the $m$ slots that are equal:

$$
\hat{J} = \frac{1}{m}\sum_{i=1}^{m} \mathbb{1}\big[\text{sig}_A[i] = \text{sig}_B[i]\big].
$$

This $\hat J$ is an unbiased estimator of $J$ (each slot matches with probability
$J$), and its standard deviation shrinks like $1/\sqrt{m}$ — so more permutations
means a tighter estimate. A signature of a few hundred integers replaces a set of
thousands of shingles, and comparing two documents is now $m$ integer
comparisons.

## 7. Tiny MinHash example, worked entirely by hand

Let us make the whole mechanism concrete with numbers small enough to check in
your head. We will use integer "shingles" and a universe $\{1,2,3,4,5,6\}$.

Two documents' shingle sets:

$$
A = \{1, 2, 3, 4\}, \qquad B = \{3, 4, 5, 6\}.
$$

**True Jaccard.** The intersection is $A \cap B = \{3, 4\}$ (size $2$); the union
is $A \cup B = \{1,2,3,4,5,6\}$ (size $6$). So

$$
J(A, B) = \frac{2}{6} = \frac{1}{3} \approx 0.3333.
$$

**Six permutations.** Each permutation is an ordering of the universe; the element
listed first has rank $0$ (smallest), and a set's MinHash is *its* member with
the smallest rank. We tabulate the MinHash of each set under six permutations:

| permutation (rank order) | minhash(A) | minhash(B) | match? |
|---|---|---|---|
| 1, 4, 2, 6, 3, 5 | 1 | 4 | no |
| 3, 1, 5, 2, 6, 4 | 3 | 3 | **yes** |
| 5, 6, 1, 3, 2, 4 | 1 | 5 | no |
| 4, 2, 6, 1, 5, 3 | 4 | 4 | **yes** |
| 2, 5, 1, 4, 6, 3 | 2 | 5 | no |
| 6, 3, 4, 1, 5, 2 | 3 | 6 | no |

Read the first row: the permutation ranks element `1` smallest. `1` is in $A$ so
$\text{minhash}(A)=1$; the smallest-ranked member of $B$ is `4`, so
$\text{minhash}(B)=4$; they differ, no match. A match happens exactly when the
overall-smallest element of the union is in *both* sets — i.e. is `3` or `4`, the
intersection. Rows 2 and 4 are the two where that happens.

**The estimate.** Two matches out of six permutations:

$$
\hat J = \frac{2}{6} = \frac{1}{3} = 0.3333,
$$

which here lands exactly on the true $J = 1/3$. With only six permutations that
is partly luck (the estimate has variance); the point is that averaging matches
recovers Jaccard. Push $m$ up and the estimate concentrates: a simulation over
$100{,}000$ random permutations of this same $A, B$ gives $\hat J \approx 0.3336$.

Verify the whole thing:

```python
from fractions import Fraction
A, B = {1, 2, 3, 4}, {3, 4, 5, 6}
print(Fraction(len(A & B), len(A | B)))    # 1/3  -- true Jaccard

import random
random.seed(0)
U = list(A | B)
matches = 0
N = 100_000
for _ in range(N):
    random.shuffle(U)
    rank = {e: i for i, e in enumerate(U)}
    mA = min(A, key=lambda e: rank[e])
    mB = min(B, key=lambda e: rank[e])
    matches += (mA == mB)
print(matches / N)                          # ~0.3336  -- MinHash estimate -> J
```

## 8. From permutations to hash functions (what the code does)

Real MinHash does not literally shuffle the universe of all shingles — that
universe is astronomically large. Instead it simulates a random permutation with
a **hash function**. A universal hash

$$
h(x) = (a \cdot x + b) \bmod p,
$$

with a large prime $p$ and random coefficients $a, b$, maps each shingle's base
hash to a pseudo-random value; the shingle with the *smallest* hash value plays
the role of "smallest rank". Using $m$ different $(a_i, b_i)$ pairs gives $m$
independent hash functions — the $m$ "permutations". This is exactly what
`MinHasher` does (`code/src/llmre/data/minhash.py`):

```python
class MinHasher:
    def __init__(self, m: int = 128, seed: int = 0) -> None:
        self.m = m
        rng = np.random.default_rng(seed)
        self.a = rng.integers(1, _MERSENNE_P, size=m, dtype=np.uint64)
        self.b = rng.integers(0, _MERSENNE_P, size=m, dtype=np.uint64)

    def signature(self, s: set) -> list[int]:
        base = np.array([_stable_hash(x) for x in s], dtype=np.uint64)  # (|s|,)
        # (m,1) coeffs against (1,|s|) hashes -> (m,|s|); min over the |s| axis.
        hashed = (self.a[:, None] * base[None, :] + self.b[:, None]) % _MERSENNE_P
        return hashed.min(axis=1).astype(np.int64).tolist()             # length m
```

The one real array here, `hashed`, has shape `(m, |s|)` and numpy dtype
`uint64` on the CPU: row $i$ is hash function $i$ applied to every shingle, and
`min(axis=1)` collapses each row to that function's MinHash. The returned
signature is a plain `list[int]` of length `m`. `_stable_hash` uses `blake2b`
(not Python's salted `hash()`) so signatures are reproducible across runs — which
is why the constructor takes a `seed`.

Estimating Jaccard is then just "fraction of equal slots":

```python
def estimate_jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    equal = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return equal / len(sig_a)
```

## 9. It works on real text — and $m$ controls accuracy

Take two overlapping documents (words 0–99 vs words 30–129), shingle them with
`k = 3`, and estimate their Jaccard at several signature sizes:

```python
from llmre.data.minhash import MinHasher, shingles, jaccard, estimate_jaccard
a = shingles(" ".join(f"word{i}" for i in range(100)), k=3)
b = shingles(" ".join(f"word{i}" for i in range(30, 130)), k=3)
print(jaccard(a, b))                       # 0.5312  -- ground truth

for m in (16, 64, 256, 1024):
    mh = MinHasher(m=m, seed=0)
    est = estimate_jaccard(mh.signature(a), mh.signature(b))
    print(m, round(est, 4))
```

Running it:

| $m$ | estimate | error vs true $0.5312$ |
|---|---|---|
| 16 | 0.3750 | 0.156 |
| 64 | 0.5156 | 0.016 |
| 256 | 0.5039 | 0.027 |
| 1024 | 0.5430 | 0.012 |

At $m = 16$ the estimate is noisy; by $m = 64$–$1024$ it hugs the true value, the
error trending down as $1/\sqrt{m}$ predicts (not monotonically — it is a random
estimate — but clearly tighter). This is the accuracy/cost knob: bigger
signatures cost more storage and comparison but estimate $J$ more precisely. The
unit test for this module asserts exactly this — that with $m \ge 128$ and a
fixed seed, `estimate_jaccard` lands within $\sim 0.1$ of the true `jaccard` for
overlapping shingle sets (`code/tests/test_dataeng.py`).

## 10. LSH banding: avoiding the $O(n^2)$ comparison

MinHash made *each* comparison cheap ($m$ integers instead of full sets), but we
still cannot afford to compare all $\binom{n}{2}$ pairs when $n$ is in the
billions. **Locality-Sensitive Hashing (LSH)** with **banding** avoids it by only
comparing documents that are *likely* similar.

Split each length-$m$ signature into $b$ **bands** of $r$ rows each ($m = b \cdot
r$). Hash each band to a bucket. Two documents become a **candidate pair** if
they land in the same bucket for *at least one* band — i.e. they share an entire
band of $r$ MinHash values. The intuition: two very similar documents (high $J$)
agree on most signature slots, so they are very likely to agree on all $r$ rows
of at least one band; two dissimilar documents almost never do.

The probability that a pair with Jaccard $J$ shares at least one band is

$$
P_{\text{candidate}}(J) = 1 - \big(1 - J^{\,r}\big)^{b}.
$$

This is an S-shaped curve in $J$: nearly $0$ below some threshold and nearly $1$
above it, with the threshold near $(1/b)^{1/r}$. Choosing $b$ and $r$ tunes where
that cutoff sits — how similar two documents must be before you bother to compare
them. Only candidate pairs get the (already cheap) signature comparison, so the
total work is roughly linear in $n$ instead of quadratic.

<div class="callout key"><p>The full pipeline: <strong>shingle</strong> each
document → <strong>MinHash</strong> to a length-$m$ signature → <strong>LSH
band</strong> to generate candidate pairs → compare only candidates and drop
those above the Jaccard threshold. Shingling captures content; MinHash makes each
comparison $O(m)$; banding makes the number of comparisons ~$O(n)$. We implement
the first two; banding is a bookkeeping layer on top.</p></div>

## Exercise

You have two documents. Document $A$ has shingle set of size $100$; document $B$
has size $100$; they share $60$ shingles. (a) What is their exact Jaccard? (b) If
you MinHash both with $m = 200$ hash functions, roughly how many signature slots
do you expect to match? (c) You want to flag pairs with $J \ge 0.6$ as
duplicates — is this pair flagged?

<details><summary>Hint</summary>
Union size is not $200$. Use $|A \cup B| = |A| + |B| - |A \cap B|$. Expected
matches $= m \cdot J$.
</details>

<details><summary>Solution</summary>

(a) $|A \cap B| = 60$ and $|A \cup B| = 100 + 100 - 60 = 140$, so
$J = 60/140 = 3/7 \approx 0.4286$.

(b) Each slot matches with probability $J$, so expected matches
$= 200 \times 0.4286 \approx 85.7$, i.e. about $86$ of the $200$ slots.

(c) No. $J \approx 0.43 < 0.6$, so this pair is *not* a duplicate under a
$0.6$ threshold — the documents share a lot but not enough. (This is why the
threshold is a deliberate choice: set it too low and you delete merely-related
documents.)

</details>

## Debugging exercise

A colleague's MinHash dedup "works on my machine" but flags *different* pairs
every time the pipeline runs, so results are not reproducible across the sharded
job. Their signature code is:

```python
def signature(self, s):
    return [min(hash((i, x)) for x in s) for i in range(self.m)]
```

What is the bug, and why does it only show up across processes?

<details><summary>Solution</summary>

They use Python's built-in <code>hash()</code>, which is <em>salted per process</em>
(PYTHONHASHSEED randomizes it) for strings/bytes. Within one process the
signatures are self-consistent, so a quick local test passes; but a different
worker process computes different hash values for the same shingle, so its
signatures — and thus which pairs collide — differ. The fix is a stable hash
(e.g. <code>hashlib.blake2b</code> of the encoded shingle), which is exactly why
<code>MinHasher</code> uses <code>_stable_hash</code> and a fixed <code>seed</code>
for the coefficients. Reproducible hashing is not optional in a sharded dedup
job.

</details>

## Research connection

<div class="callout paper"><p>GPT-3 (Brown et al., 2020) applied fuzzy
deduplication (MinHash-LSH) to its Common-Crawl data, and it is standard in every
major open corpus since (C4, RefinedWeb, Dolma, FineWeb). The empirical case that
dedup improves models and reduces memorization is made directly in
"Deduplicating Training Data Makes Language Models Better" (Lee et al., 2021).
See <a href="#/papers/index">the paper curriculum</a>; read GPT-3's data
section for the pipeline in context.</p></div>

## Check yourself

<details><summary>Why can't exact hashing catch a document that is the same article with a changed publication date?</summary>

A hash function maps any single-byte change to a completely different output, so
"…published 2023" and "…published 2024" hash to unrelated fingerprints. Exact
dedup sees two distinct documents. Fuzzy dedup handles it because the two shingle
sets still overlap in almost every shingle, giving Jaccard near $1$.

</details>

<details><summary>State the identity MinHash relies on, in words.</summary>

For a single random permutation of the shingle universe, the probability that two
sets have the same MinHash (their minimum-ranked element) equals their Jaccard
similarity. Equivalently, each independent hash/permutation is a Bernoulli trial
that succeeds with probability $J$, so the fraction of matching signature slots
estimates $J$.

</details>

<details><summary>In the tiny example ($A=\{1,2,3,4\}$, $B=\{3,4,5,6\}$), under the permutation ordering <code>5, 6, 1, 3, 2, 4</code>, what are minhash(A) and minhash(B)?</summary>

Ranks: 5→0, 6→1, 1→2, 3→3, 2→4, 4→5. For $A=\{1,2,3,4\}$ the smallest rank is
element `1` (rank 2), so minhash(A) = 1. For $B=\{3,4,5,6\}$ the smallest rank is
element `5` (rank 0), so minhash(B) = 5. They differ — no match — which matches
the table in section 7.

</details>

<details><summary>You double $m$ from 128 to 256. What happens to the estimate's accuracy and to the cost?</summary>

The estimator's standard deviation shrinks by $1/\sqrt{2} \approx 0.71$ (accuracy
improves like $1/\sqrt{m}$), while signature storage and per-comparison cost
double (linear in $m$). It is a direct accuracy-vs-cost trade; you pick $m$ for
the precision the dedup threshold needs.

</details>

<details><summary>What problem does LSH banding solve that MinHash alone does not?</summary>

MinHash makes each pairwise comparison cheap ($O(m)$ integers), but there are
still $O(n^2)$ pairs. LSH banding hashes signature bands into buckets so only
documents sharing a full band (likely-similar pairs) are ever compared, cutting
the number of comparisons to roughly linear in $n$.

</details>

## Next

Deduplication removes redundancy *within* the training corpus. But a subtler
leak remains: text from an *evaluation benchmark* sneaking into training, which
silently inflates your scores. The next lesson builds an n-gram contamination
check to catch it, then turns to how you *mix* multiple data sources (web, code,
books, math) with weights and token budgets, and how data ordering — the
curriculum — is used late in training.

Continue to [11.3 · Contamination, mixing, curriculum](lessons/module-11/lesson-03.md).
