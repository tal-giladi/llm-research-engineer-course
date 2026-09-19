"""Byte-level Byte-Pair Encoding (BPE) from scratch.

This is the tokenizer of Module 4 (`lessons/module-04/lesson-02.md`). It is the
same byte-level BPE family that GPT-2 uses: training starts from the 256 raw
UTF-8 byte values and repeatedly merges the most frequent adjacent pair into a
new symbol, so common character sequences collapse to a single token while
anything unseen still falls back to bytes. Because every string is first turned
into its UTF-8 bytes, **nothing is ever out-of-vocabulary** and
``decode(encode(s)) == s`` holds exactly for any Unicode string.

The implementation is written for readability, not speed: the training loop is
a plain re-count-and-merge over a Python list of ints (O(n) work per merge),
which is fine for the small corpora used in the lessons and tests. Production
tokenizers (``tiktoken``, HuggingFace ``tokenizers``) do the same algorithm with
a Rust core and a regex pre-tokenization split; see the lesson for that story.
"""

from __future__ import annotations


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """Count how often each adjacent pair of symbols occurs in ``ids``.

    Args:
        ids: a sequence of symbol ids (Python list of ``int``). During training
            these start as byte values ``0..255`` and grow as merges add new ids.

    Returns:
        A dict mapping each adjacent pair ``(a, b)`` to its integer count. For
        ``[1, 2, 1, 2, 3]`` this is ``{(1, 2): 2, (2, 1): 1, (2, 3): 1}``.
    """
    counts: dict[tuple[int, int], int] = {}
    for a, b in zip(ids, ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of ``pair`` in ``ids`` with the single ``new_id``.

    Args:
        ids: the current sequence of symbol ids (list of ``int``).
        pair: the adjacent pair ``(a, b)`` to fuse.
        new_id: the id of the new merged symbol.

    Returns:
        A new list with each non-overlapping ``a, b`` collapsed to ``new_id``.
        ``merge([1, 2, 1, 2, 3], (1, 2), 99)`` returns ``[99, 99, 3]``.
    """
    out: list[int] = []
    i = 0
    while i < len(ids):
        # If the pair starts here (and there is a next symbol), emit the merge.
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    """A byte-level BPE tokenizer trained from raw text.

    State after ``train``:
        merges: dict ``{(a, b): new_id}`` giving, for each learned merge, the two
            symbol ids that fuse and the id of their fused symbol. Insertion order
            is the merge order, which ``encode`` must respect.
        vocab: dict ``{id: bytes}`` mapping every symbol id to the raw ``bytes``
            it expands to. Ids ``0..255`` map to the single byte ``bytes([id])``;
            a merged id maps to the concatenation of its two children's bytes.

    Everything is plain Python (ints and bytes) — there are no tensors, no dtype
    and no device here. Token ids only become a ``torch.long`` tensor later, in
    the data loader (`llmre.data.loader`).
    """

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int) -> None:
        """Learn ``vocab_size - 256`` merges from ``text``.

        Args:
            text: the training corpus (a Python ``str``). It is encoded to UTF-8
                bytes; training operates on those byte ids.
            vocab_size: target vocabulary size. Must be ``>= 256`` (the byte
                floor). The number of merges learned is ``vocab_size - 256``
                (fewer if the sequence runs out of repeated pairs first).

        Returns:
            None. Populates ``self.merges`` and ``self.vocab`` in place.
        """
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be >= 256, got {vocab_size}")
        num_merges = vocab_size - 256

        # Start from the raw UTF-8 bytes as the initial symbol sequence.
        ids = list(text.encode("utf-8"))

        # Reset any previous training so train() is idempotent.
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break  # sequence collapsed to a single symbol; nothing left to merge
            # Most frequent adjacent pair. max() with a count key; ties break on
            # the first pair encountered, which is deterministic for a given text.
            pair = max(stats, key=stats.get)
            if stats[pair] < 2:
                break  # no pair repeats; further merges would not compress anything
            new_id = 256 + i
            ids = merge(ids, pair, new_id)
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text: str) -> list[int]:
        """Encode a string to a list of token ids.

        The text is turned into UTF-8 bytes, then the learned merges are applied
        **in the order they were learned**: at each step we fuse the pair whose
        merge index is lowest among the pairs currently present, because a later
        merge may depend on the symbol a earlier merge produced.

        Args:
            text: the string to encode (any Unicode).

        Returns:
            A list of ``int`` token ids, each in ``range(self.vocab_size)``.
        """
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats(ids)
            # Pick the present pair with the smallest merge index (earliest learned).
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # no remaining pair is mergeable
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back to a string.

        Each id is expanded to its raw ``bytes`` via ``self.vocab`` and the bytes
        are concatenated, then decoded as UTF-8. Because ``encode`` always emits
        ids whose bytes reconstitute a valid UTF-8 stream, this inverts ``encode``
        exactly. ``errors="replace"`` only guards against hand-crafted id lists
        that do not form valid UTF-8.

        Args:
            ids: a list of ``int`` token ids.

        Returns:
            The decoded ``str``.
        """
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        """Total number of symbols: ``256`` byte values plus one per merge."""
        return len(self.vocab)
