"""Tests for Module 11 data engineering: quality, MinHash, contamination.

These back the three lessons of Module 11:
  * ``quality_filter`` keeps clean prose and rejects too-short / symbol-spam;
  * exact-dup detection via hashing, and MinHash ``estimate_jaccard`` landing
    within ~0.1 of the true Jaccard for overlapping shingle sets (m>=128, fixed
    seed);
  * ``ngram_overlap`` == 1.0 when the test text is a substring of the train
    text, and ~0 for disjoint text.
"""

from __future__ import annotations

from llmre.data.contamination import ngram_overlap
from llmre.data.minhash import MinHasher, estimate_jaccard, jaccard, shingles
from llmre.data.quality import quality_filter


# --------------------------------------------------------------------------- #
# Quality filtering
# --------------------------------------------------------------------------- #
_CLEAN = (
    "The transformer architecture replaced recurrent networks for most language "
    "tasks because attention lets every position read every other position in a "
    "single step. This makes training parallel across the sequence, which in turn "
    "makes it practical to scale models to billions of parameters on modern "
    "accelerators without the sequential bottleneck that limited earlier designs."
)


def test_quality_filter_accepts_clean_paragraph():
    assert quality_filter(_CLEAN) is True


def test_quality_filter_rejects_too_short():
    assert quality_filter("Click here now.") is False


def test_quality_filter_rejects_symbol_spam():
    assert quality_filter("#$%^&* " * 40 + "<<<>>> @@@ ||| ### $$$ %%% ^^^") is False


def test_quality_filter_rejects_repetition():
    # 60 words but almost all identical -> high repetition ratio.
    assert quality_filter("buy " * 60) is False


# --------------------------------------------------------------------------- #
# Exact + fuzzy dedup
# --------------------------------------------------------------------------- #
def test_exact_dedup_by_hash():
    docs = ["hello world foo", "hello world foo", "a completely different doc"]
    seen: set[int] = set()
    kept: list[str] = []
    for d in docs:
        h = hash(d)
        if h not in seen:
            seen.add(h)
            kept.append(d)
    assert kept == ["hello world foo", "a completely different doc"]


def test_minhash_estimates_jaccard_within_tolerance():
    text_a = " ".join(f"word{i}" for i in range(100))
    # ~70% overlap: b shares words 30..99 and adds 30 fresh ones.
    text_b = " ".join(f"word{i}" for i in range(30, 130))
    a, b = shingles(text_a, k=3), shingles(text_b, k=3)

    true_j = jaccard(a, b)
    mh = MinHasher(m=256, seed=0)
    est = estimate_jaccard(mh.signature(a), mh.signature(b))

    assert 0.0 < true_j < 1.0
    assert abs(est - true_j) < 0.1


def test_minhash_identical_sets_estimate_one():
    a = shingles("the quick brown fox jumps over the lazy dog", k=3)
    mh = MinHasher(m=128, seed=1)
    assert estimate_jaccard(mh.signature(a), mh.signature(a)) == 1.0


# --------------------------------------------------------------------------- #
# Contamination
# --------------------------------------------------------------------------- #
def test_ngram_overlap_substring_is_one():
    test_item = "the capital of france is paris"
    train_doc = "a geography note: the capital of france is paris, everyone knows it"
    assert ngram_overlap(train_doc, test_item, n=4) == 1.0


def test_ngram_overlap_disjoint_is_near_zero():
    train_doc = "quantum error correction protects logical qubits from decoherence noise"
    test_item = "the mitochondria is the powerhouse of the cell in biology"
    assert ngram_overlap(train_doc, test_item, n=4) < 0.05
