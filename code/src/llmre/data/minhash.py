"""Near-duplicate detection with shingling and MinHash (Module 11, lesson 11.2).

Exact deduplication (hash the whole document, drop repeats) misses the common
case on the web: two pages that are 95% identical but differ in a timestamp or a
boilerplate line. To catch those we measure the *Jaccard similarity* of two
documents' k-shingle sets and estimate it cheaply with MinHash.

Everything here is plain Python / numpy on the CPU. A shingle set is a Python
``set[str]``; a MinHash signature is a ``list[int]`` of length ``m`` (one
minimum per hash function). No torch tensors, no GPU. The hashing is made
deterministic by seeding numpy, so a signature is reproducible across runs.
"""

from __future__ import annotations

import numpy as np

# 61-bit Mersenne prime: modulus for the universal hash family h(x) = a*x + b mod p.
_MERSENNE_P = (1 << 61) - 1


def shingles(text: str, k: int = 5) -> set[str]:
    """Represent ``text`` as its set of overlapping k-word shingles.

    A *shingle* (a.k.a. k-gram) is a window of ``k`` consecutive words joined by
    single spaces. Two documents that share long runs of words share many
    shingles; small edits change only the few shingles that span them. The whole
    document collapses to the *set* of its shingles, discarding order and counts.

    Args:
        text: one document as a Python ``str``.
        k: shingle size in words. Larger ``k`` makes coincidental overlap
            between unrelated documents less likely (typical values 5-10).

    Returns:
        A ``set[str]`` of shingles. If the document has fewer than ``k`` words,
        returns a single shingle containing the whole (lower-cased) text so
        short docs still compare sensibly.
    """
    toks = text.lower().split()
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i : i + k]) for i in range(len(toks) - k + 1)}


def jaccard(a: set, b: set) -> float:
    """Exact Jaccard similarity ``|a ∩ b| / |a ∪ b|`` of two sets.

    This is the ground-truth quantity MinHash estimates. It ranges from ``0``
    (disjoint) to ``1`` (identical).

    Args:
        a: first set (e.g. a shingle set).
        b: second set.

    Returns:
        A ``float`` in ``[0, 1]``; ``1.0`` when both sets are empty (they are
        trivially identical).
    """
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union


class MinHasher:
    """Estimate Jaccard similarity via ``m`` min-wise independent hash functions.

    Each hash function is a universal hash ``h_i(x) = (a_i * x + b_i) mod p``
    over 64-bit token hashes, with random ``a_i, b_i`` fixed at construction. The
    MinHash *signature* of a set is, for each function, the minimum hash value
    over the set's elements. The key fact: for a random hash the probability that
    two sets share their minimum equals their Jaccard similarity, so the fraction
    of the ``m`` signature slots that agree is an unbiased estimate of Jaccard.
    More functions (larger ``m``) shrink the estimate's variance.

    Signatures are ``list[int]`` of length ``m`` on the CPU; construction is
    seeded, so the same seed gives identical coefficients and reproducible
    signatures.
    """

    def __init__(self, m: int = 128, seed: int = 0) -> None:
        """Draw ``m`` random hash-function coefficients.

        Args:
            m: number of hash functions = signature length. More reduces
                estimator variance (std shrinks like ``1/sqrt(m)``).
            seed: RNG seed making the coefficients (and thus all signatures)
                deterministic.
        """
        self.m = m
        rng = np.random.default_rng(seed)
        # a must be non-zero; draw in [1, p-1]. b in [0, p-1].
        self.a = rng.integers(1, _MERSENNE_P, size=m, dtype=np.uint64)
        self.b = rng.integers(0, _MERSENNE_P, size=m, dtype=np.uint64)

    def signature(self, s: set) -> list[int]:
        """Compute the length-``m`` MinHash signature of set ``s``.

        Each element is hashed to a stable 64-bit integer, the ``m`` universal
        hash functions are applied to every element, and the per-function minima
        form the signature.

        Args:
            s: a set of hashable elements (typically string shingles).

        Returns:
            A ``list[int]`` of length ``m``. For an empty set every slot is the
            modulus ``p`` (a sentinel larger than any real hash), so two empty
            sets still compare as identical.
        """
        if not s:
            return [int(_MERSENNE_P)] * self.m
        # Stable 64-bit base hash of each element (hashlib-free, deterministic
        # across runs because we do not use Python's salted hash()).
        base = np.array(
            [_stable_hash(x) for x in s], dtype=np.uint64
        )  # shape (|s|,)
        # Broadcast: (m, 1) coeffs against (|s|,) hashes -> (m, |s|) then min over axis 1.
        # Use Python ints via modular arithmetic to avoid uint64 overflow surprises:
        # numpy uint64 multiplication wraps mod 2**64, which is fine as a hash mix,
        # but we take an explicit mod p to keep values in the field.
        hashed = (self.a[:, None] * base[None, :] + self.b[:, None]) % _MERSENNE_P
        return hashed.min(axis=1).astype(np.int64).tolist()


def estimate_jaccard(sig_a: list[int], sig_b: list[int]) -> float:
    """Estimate Jaccard from two MinHash signatures.

    The estimate is the fraction of signature slots that are equal — an unbiased
    estimator of the true Jaccard similarity of the underlying sets.

    Args:
        sig_a: signature of the first set (length ``m``).
        sig_b: signature of the second set (same length ``m``).

    Returns:
        A ``float`` in ``[0, 1]``: ``(# equal slots) / m``.

    Raises:
        ValueError: if the signatures have different lengths.
    """
    if len(sig_a) != len(sig_b):
        raise ValueError(
            f"signatures differ in length: {len(sig_a)} vs {len(sig_b)}"
        )
    if not sig_a:
        return 1.0
    equal = sum(1 for x, y in zip(sig_a, sig_b) if x == y)
    return equal / len(sig_a)


def _stable_hash(x: object) -> int:
    """Deterministic 64-bit hash of an object's string form (blake2b, 8 bytes).

    Python's built-in ``hash`` is salted per process, so it cannot be used for a
    reproducible signature. blake2b gives the same value every run.
    """
    import hashlib

    h = hashlib.blake2b(str(x).encode("utf-8"), digest_size=8)
    return int.from_bytes(h.digest(), "little")
