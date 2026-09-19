"""Tests for llmre.tokenizer.bpe — byte-level BPE from scratch.

These back the claims of Module 4's lessons: that the tokenizer round-trips any
Unicode string exactly, that training on a repetitive corpus actually learns
merges and shrinks the encoded length below the raw byte count, and that the
vocabulary size is exactly ``256 + num_merges``.
"""

from __future__ import annotations

from llmre.tokenizer.bpe import BPETokenizer


def test_roundtrip_ascii():
    tok = BPETokenizer()
    tok.train("the cat sat on the mat, the cat ran", vocab_size=280)
    s = "the cat sat on the mat"
    assert tok.decode(tok.encode(s)) == s


def test_roundtrip_unicode_emoji_and_accents():
    tok = BPETokenizer()
    tok.train("café au lait, café crème, déjà vu", vocab_size=300)
    # Strings the tokenizer never saw in training, with emoji and accents, must
    # still round-trip exactly via the UTF-8 byte fallback.
    for s in ["café déjà vu 😀", "naïve façade — 日本語 🚀", "", "a"]:
        assert tok.decode(tok.encode(s)) == s


def test_training_creates_merges_and_compresses():
    tok = BPETokenizer()
    corpus = "low low low low lower lowest newest newest widest widest widest"
    raw_len = len(corpus.encode("utf-8"))

    tok.train(corpus, vocab_size=256 + 20)

    # Training learned at least one merge...
    assert len(tok.merges) > 0
    # ...and encoding the corpus is now shorter than its raw byte length.
    encoded = tok.encode(corpus)
    assert len(encoded) < raw_len
    # Round-trip still exact after training.
    assert tok.decode(encoded) == corpus


def test_vocab_size_is_256_plus_num_merges():
    tok = BPETokenizer()
    corpus = "aaaaaaaa bbbbbbbb aaaa bbbb abababab"
    tok.train(corpus, vocab_size=256 + 10)
    assert tok.vocab_size == 256 + len(tok.merges)


def test_more_merges_never_increase_encoded_length():
    corpus = "the quick brown fox the quick brown fox the quick brown fox"
    lengths = []
    for extra in (0, 5, 15, 40):
        tok = BPETokenizer()
        tok.train(corpus, vocab_size=256 + extra)
        lengths.append(len(tok.encode(corpus)))
    # Each larger vocabulary encodes the same text in no more tokens than the
    # smaller one — more merges can only fuse, never split.
    assert all(lengths[i] >= lengths[i + 1] for i in range(len(lengths) - 1))
