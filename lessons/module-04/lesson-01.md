# 04.1 · Characters, bytes, Unicode, the vocabulary problem

<div class="prereq">
<p><strong>Prerequisites:</strong> the language-modeling objective from <a href="#/lessons/module-01/lesson-04">01.4 · The language-modeling objective &amp; perplexity</a> — a model puts a softmax distribution over a fixed set of next-token choices and is scored by the cross-entropy of the true token. The size of that "set of choices" is the vocabulary, and this lesson is about where it comes from.</p>
<p><strong>You will learn:</strong> why a language model cannot just use words as its units (out-of-vocabulary words, an enormous sparse vocabulary, no sharing between related word forms), why it cannot use raw characters either (sequences become far too long, wasting compute), and how <strong>UTF-8 bytes</strong> give a universal 256-symbol alphabet that can spell any text on Earth. That sets up the <em>subword</em> compromise — Byte-Pair Encoding — which the next lesson builds from scratch.</p>
<p><strong>Why this matters for ML:</strong> the tokenizer decides what a "token" is, and everything downstream is counted in tokens: context length, training-set size, the cost of a forward pass, the price of an API call. The <code>vocab_size</code> field of <code>GPTConfig</code> is the width of the model's input and output layers. Choose the units badly and you either blow up sequence length or cripple the model on unseen text. This is the first design decision in the whole pipeline.</p>
</div>

## 1. The problem: text is not numbers, but the model only eats numbers

A neural network multiplies matrices of floating-point numbers. Text is a string of abstract characters. So the very first thing any language model needs is a fixed, invertible rule that turns a string into a list of integers and back:

$$
\text{"hello"} \;\xrightarrow{\text{encode}}\; [\,15496,\ 220,\ \dots\,] \;\xrightarrow{\text{decode}}\; \text{"hello"}.
$$

Each integer is an index into a **vocabulary**: a numbered list of the atomic units the model knows. The model's input embedding table has one row per vocabulary entry, and its final output layer produces one logit per vocabulary entry — so if the vocabulary has $V$ entries, the model predicts a probability distribution over exactly those $V$ choices at every position (that distribution is the softmax you met in [01.4](lessons/module-01/lesson-04.md)).

Two numbers are in tension the moment we pick our units, and the whole lesson is about their trade-off:

- **Vocabulary size $V$** — how many distinct units exist. It sets the width of the embedding table and the output layer.
- **Sequence length $T$** — how many units a given piece of text becomes. It sets how long a context the model must process, and attention cost grows with $T^2$ (you will feel this in Module 5).

The three candidate choices below — words, characters, bytes — each push one of these to an extreme. Subword tokenization, next lesson, is the balance.

<div class="callout key"><p>A tokenizer is a fixed, invertible <code>string ↔ list[int]</code> codec. It is chosen and frozen <em>before</em> training. Its vocabulary size $V$ becomes the model's input/output width, and the number of tokens it emits per document sets the sequence length the model pays for.</p></div>

## 2. Why not a word vocabulary?

The obvious idea: let the units be words. Split on spaces, assign each distinct word an integer. `"the cat sat"` becomes three tokens. Sequences stay short and each token is meaningful. It fails for three concrete reasons.

**Out-of-vocabulary (OOV) words.** A vocabulary is fixed at training time. The moment the model meets a word that was not in that list — a new product name, a typo, a rare surname, `covfefe` — it has no integer for it. The classic patch is a single catch-all `<unk>` ("unknown") token, but then every unseen word collapses to the same id and the model literally cannot tell them apart or reproduce them. For a model that must generate fluent text and handle arbitrary user input, throwing away every novel word is fatal.

**Huge, sparse vocabulary.** English alone has hundreds of thousands of word forms; across all languages, code, URLs, and numbers the count is effectively unbounded. To cover even most English you would need a vocabulary in the hundreds of thousands. That is a gigantic embedding table and a gigantic output layer: with hidden size $C$ and vocabulary $V$, the embedding matrix alone is $V \times C$ parameters. At $V = 500{,}000$ and $C = 768$ that is $3.8 \times 10^8$ parameters just to look words up — and most rows are for rare words seen a handful of times, so they never get enough gradient signal to train well. Sparse, under-trained rows are wasted capacity.

**No morphology.** Words are made of reusable parts, and a word vocabulary throws that structure away. `run`, `runs`, `running`, `runner` are four unrelated integers with four independent embeddings; the model cannot see that they share a stem. Same for `nation / national / nationalize / nationalization`. A good tokenizer should let the model reuse the piece `run` across all its forms. Whole-word units make every inflection a fresh, unrelated symbol.

<div class="callout warn"><p>The <code>&lt;unk&gt;</code> escape hatch quietly breaks the round-trip: <code>decode(encode(s))</code> is no longer equal to <code>s</code> whenever <code>s</code> contains an unseen word. A tokenizer that cannot faithfully reproduce its input is unusable for generation. We will insist on exact round-tripping in the code we write.</p></div>

## 3. Why not pure characters?

Swing to the other extreme: let the units be individual characters. Now the vocabulary is tiny — roughly 100 symbols for English text — and there is no OOV problem for those characters, since every word is just a sequence of them. `run` and `running` now share the letters `r`, `u`, `n`. So characters fix everything words got wrong.

The price is **sequence length**, and it is steep. A 1,000-word passage is maybe 6,000 characters. If each character is one token, the model must process a length-6,000 sequence to see the same content that a word tokenizer covered in 1,000 tokens. Two costs follow:

- **Attention is quadratic in length.** As you will build in Module 5, each layer compares every position with every other, so work grows like $T^2$. Six times the length is roughly $6^2 = 36$ times the attention compute and memory for the same text. Sequence length is the single most expensive knob in a Transformer.
- **The model wastes capacity learning to spell.** With character units, a large share of the model's effort goes into re-deriving that `t-h-e` spells "the" — structure a coarser unit would hand it for free. Compute spent reassembling common words from letters is compute not spent on meaning.

There is also a subtler issue: a single "character" is not even well defined across the world's scripts (accents, emoji modifiers, and combining marks make "one character" ambiguous). We need a unit that is precise and universal. That is where bytes come in.

<div class="callout key"><p>Words → short sequences but an unbounded, sparse vocabulary with an OOV cliff. Characters → a tiny vocabulary with no OOV, but sequences so long that attention's $T^2$ cost explodes and the model burns capacity learning to spell. We want short-ish sequences <em>and</em> a bounded vocabulary <em>and</em> no OOV. That is the subword compromise.</p></div>

## 4. UTF-8: a universal 256-symbol alphabet

Before we build subwords, we need a rock-solid foundation that can represent *any* text with *no* OOV, ever. That foundation is **bytes**.

Every character in every script — Latin letters, Chinese characters, Arabic, emoji — has a **Unicode code point**, an integer identifier. `A` is U+0041 (decimal 65), `é` is U+00E9 (233), `世` is U+4E16 (19,990), `😀` is U+1F600 (128,512). There are about 150,000 assigned code points and room for over a million. That is far too many to use directly as a vocabulary — we would be back to the huge-vocabulary problem.

**UTF-8** is the standard encoding that maps each code point to a sequence of one to four **bytes**, where a byte is just an integer in $0..255$. ASCII characters (the common English set) are a single byte equal to their code point; everything else expands to two, three, or four bytes. Crucially, this means *any* string, in *any* language, is representable as a sequence drawn from a fixed alphabet of only **256 symbols** — the byte values $0$ through $255$. There is no such thing as an out-of-vocabulary byte.

In Python, `str.encode("utf-8")` gives you those bytes and `bytes.decode("utf-8")` inverts it exactly:

```python
for ch in ["A", "é", "世", "😀"]:
    print(ch, hex(ord(ch)), list(ch.encode("utf-8")))
```

which prints (verified with `py`):

```text
A  0x41    [65]
é  0xe9    [195, 169]
世 0x4e16  [228, 184, 150]
😀 0x1f600 [240, 159, 152, 128]
```

Read those four lines carefully, because they *are* UTF-8:

- `A` (U+0041) is plain ASCII, so it is the single byte `65` — the code point itself.
- `é` (U+00E9) is *not* ASCII, so it becomes **two** bytes `[195, 169]`. Note `233` never appears: UTF-8 does not store the code point directly for non-ASCII; it uses a multi-byte pattern.
- `世` (U+4E16) becomes **three** bytes.
- `😀` (U+1F600) becomes **four** bytes.

So a five-character string with one accent is more than five bytes:

```python
"café".encode("utf-8")   # -> b'caf\xc3\xa9'  = [99, 97, 102, 195, 169]
```

Four characters, **five** bytes: `c`, `a`, `f` are one byte each, and `é` is the two bytes `195, 169`. This length inflation is the one downside of bytes, and it is exactly what BPE will claw back next lesson.

<div class="callout pt"><p><code>"café".encode("utf-8")</code> returns a Python <code>bytes</code> object; wrapping it in <code>list(...)</code> gives the integers. These are ordinary Python <code>int</code>s in <code>0..255</code> — no tensors yet, no dtype, no device. Token ids only become a <code>torch.long</code> tensor much later, when the data loader (<a href="#/lessons/module-04/lesson-03">04.3</a>) hands batches to the model.</p></div>

### 4.1 Bytes as our safety net

Using raw bytes as the units gives us a character-style tokenizer with two superpowers: the vocabulary is exactly 256 (tiny and bounded), and **nothing is ever OOV**, because every possible string is by definition a sequence of bytes. The round-trip is exact for any input:

$$
\text{decode}(\text{encode}(s)) = s \quad\text{for every string } s.
$$

The catch is the same one characters had — sequences are long (even a bit longer than character count, because of multi-byte characters). So bytes alone are not the answer either. But they are the perfect *floor*: a base alphabet that can spell anything, on top of which we will learn larger, more efficient units. That is precisely what byte-level BPE does.

## 5. The subword compromise BPE strikes

Byte-Pair Encoding, built in the next lesson, starts from the 256 bytes and then **learns** a set of merges: it repeatedly finds the most frequent adjacent pair of symbols in a training corpus and fuses it into a single new symbol. Run this a few thousand times and you get a vocabulary that lands exactly in the sweet spot:

- **Common words become a single token.** `the`, `and`, ` of` (note the leading space) occur constantly, so their bytes get merged early into one unit. Frequent text is short.
- **Rare words split into a few subword pieces.** An unusual word like `tokenization` might become `token` + `ization`, reusing pieces the model already knows. Related forms share stems — exactly the morphology sharing whole words lacked.
- **Nothing is ever OOV.** Because the vocabulary still *contains all 256 bytes* as its foundation, any string the merges do not cover falls back to bytes. A never-before-seen emoji or a random hash is representable, always, as its raw bytes. The exact round-trip survives.

This is the deliberate control of the vocabulary-size-vs-sequence-length trade-off from section 1. Every merge you add grows $V$ by one and shrinks $T$ for text that uses that merge:

- **Few merges** (small $V$): close to byte-level — short vocabulary, long sequences.
- **Many merges** (large $V$): frequent words are single tokens — long vocabulary, short sequences, but a big embedding table and many rare, under-trained rows (the word-vocabulary problem creeping back).

Real models pick a point in between: GPT-2 uses $V = 50{,}257$, and the field mostly sits between about 32k and 128k. `GPTConfig.vocab_size` defaults to `50257` for exactly this reason — it is the GPT-2 tokenizer's size, which the next two lessons will let you reproduce in miniature.

<div class="callout key"><p>BPE = start from the 256 UTF-8 bytes (so no OOV, ever) and learn merges that fuse frequent adjacent pairs into new symbols. The result: frequent text is one-token-per-word (short sequences), rare text degrades gracefully into subword pieces and ultimately bytes (still no OOV), and $V$ is a tunable dial trading table size against sequence length.</p></div>

## Common mistakes

- **Confusing characters, code points, and bytes.** `"世"` is *one* character (one Unicode code point, U+4E16) but *three* UTF-8 bytes. `len("café")` is `4`; `len("café".encode("utf-8"))` is `5`. Byte-level tokenizers count in bytes, so their sequence lengths are driven by the byte count, not the character count.
- **Assuming one token ≈ one word.** For a subword tokenizer it is only *roughly* true for common English. Rare words, code, other languages, and numbers can be several tokens each. Never estimate cost by counting words; count tokens.
- **Forgetting the leading space.** Byte-level BPE treats ` the` (with its space) and `the` (at the start of a line) as different byte sequences, so they can be different tokens. The space is part of the data.

## Check yourself

<details><summary>Why does a pure word-level vocabulary force either an <code>&lt;unk&gt;</code> token or an impractically large table, and why are both bad?</summary>

Because the set of possible words is effectively unbounded (new names, typos, code, other languages). A fixed table must therefore either omit rare words — replacing them all with one `<unk>` id, which loses information and breaks exact reconstruction — or try to include them, ballooning to hundreds of thousands of rows, most of which are rare and under-trained (wasted parameters). Neither gives you faithful round-tripping with a manageable, well-trained table.

</details>

<details><summary>A document is 500 words ≈ 3,000 characters. Roughly how does switching from word units to character units change the attention cost, and why?</summary>

Length goes from about 500 tokens to about 3,000 tokens, a factor of 6. Self-attention cost scales like $T^2$, so the attention compute and memory grow by roughly $6^2 = 36\times$ for the same text. Sequence length is the dominant cost knob, which is why we do not want the very long sequences that character (or raw byte) units produce.

</details>

<details><summary>How many bytes is <code>"é"</code> in UTF-8, and why is it not simply the single byte <code>233</code> (its code-point value U+00E9 = 233)?</summary>

Two bytes: `[195, 169]`. UTF-8 stores only ASCII code points (0–127) as their raw single byte; any code point ≥ 128, including U+00E9, is encoded with a multi-byte pattern in which the leading byte signals the length and the continuation bytes carry the rest. So `233` never appears literally — `é` is the byte pair `195, 169`.

</details>

<details><summary>Why can a byte-level tokenizer guarantee <code>decode(encode(s)) == s</code> for absolutely any string, including one full of never-before-seen emoji?</summary>

Because its base vocabulary is the complete set of 256 byte values, and every string is <em>by definition</em> a sequence of UTF-8 bytes. There is no input that cannot be expressed in the alphabet, so there is never an unknown symbol to lose. Even if no learned merge applies, the string still encodes as its raw bytes and decodes back exactly.

</details>

<details><summary>You increase your BPE vocabulary from 8k to 64k merges. Name one thing that gets better and one that gets worse.</summary>

Better: sequences get shorter (more frequent words and word-pieces become single tokens), so the model processes the same text in fewer positions and attention is cheaper per document. Worse: the embedding table and output layer get wider ($V$ larger), costing more parameters and memory, and the extra tokens are rarer, so their embedding rows receive less gradient signal and train less well.

</details>

## Next

You now know why the units are neither words nor characters, and that UTF-8 gives a universal 256-byte floor with no OOV. The next lesson turns the "learn merges from frequent pairs" idea into a concrete algorithm — training, encoding, and decoding a byte-level BPE tokenizer from scratch, worked by hand on a tiny corpus and implemented in `llmre.tokenizer.bpe`.

Continue to [04.2 · Byte-Pair Encoding from scratch](lessons/module-04/lesson-02.md).
