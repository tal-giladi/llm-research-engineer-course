# 04.3 · Special tokens, packing & data loading

<div class="prereq">
<p><strong>Prerequisites:</strong> a working byte-level BPE tokenizer — <code>train</code> / <code>encode</code> / <code>decode</code> — from <a href="lesson-02.md">04.2 · Byte-Pair Encoding from scratch</a>; the language-modeling objective (each position predicts the next token) from <a href="../module-01/lesson-04.md">01.4 · The language-modeling objective &amp; perplexity</a>; tensors, <code>.shape</code>, and slicing from <a href="../module-00/lesson-01.md">Module 0</a>.</p>
<p><strong>You will learn:</strong> what a <em>special token</em> like <code>&lt;|endoftext|&gt;</code> is and why pretraining needs one; how a whole corpus becomes a single 1-D stream of token ids; how that stream is cut into fixed-length <code>block_size</code> windows; and exactly how the input <code>x</code> and target <code>y</code> of a training batch are the same slice shifted by one position — the entire supervision signal for pretraining.</p>
<p><strong>Why this matters for ML:</strong> the data loader is the thing your training loop calls every single step (Module 7). If the <code>(x, y)</code> shift is off by one, or documents bleed into each other, the model learns the wrong objective and you will chase a "mysteriously bad loss" for days. This is one of the most common and most invisible sources of training bugs.</p>
</div>

## 1. Intuition: from a pile of documents to training windows

You have a tokenizer. You have a corpus — say a few million documents of text. To pretrain a
language model you must turn that pile into a stream of `(input, target)` pairs where the target
is always "the next token". Three jobs stand between raw documents and the model:

1. **Mark boundaries.** The model must know where one document ends and the next begins, or it
   will learn to run a sentence from one article straight into an unrelated one.
2. **Pack.** Concatenate every document's ids into one long 1-D array of token ids. No padding,
   no wasted space — every token is training signal.
3. **Cut windows.** The model has a fixed context length `block_size` (call it $T$). Slice the
   long stream into $T$-length windows, and pair each window with the *same* window shifted one
   step to the right.

We take these in order.

## 2. Special tokens and the `<|endoftext|>` boundary

A **special token** is an id in the vocabulary that does not stand for any byte or subword — it
carries a *structural* meaning instead. The most fundamental one in pretraining is the
end-of-text separator, written `<|endoftext|>` (GPT-2's name for it). It is placed between
documents so the model sees an explicit "this document is over" signal.

Without it, if you concatenated *"The cat sat."* and *"Quantum error correction…"* directly, the
model would be trained to predict `Quantum` as the natural continuation of `The cat sat.` — a
nonsense dependency it would waste capacity learning. With a separator, the token after
`<|endoftext|>` is the start of a fresh, independent document, and (because the causal mask still
lets attention look back across the boundary) the model also learns that `<|endoftext|>` means
"reset — a new document starts now", which is exactly what lets a trained model *stop* generating
at the end of a sample.

<div class="callout key"><p>A special token is a reserved vocabulary id with structural, not
lexical, meaning. <code>&lt;|endoftext|&gt;</code> separates documents so the model does not learn
spurious cross-document continuations and learns where generation should stop. Chat models add
more special tokens (role markers, turn boundaries) — we build those in <a href="../module-14/lesson-01.md">Module 14</a>.</p></div>

Concretely: if your BPE tokenizer trained a vocabulary of `V` ordinary tokens, you reserve one
more id — say `V` itself — as the `<|endoftext|>` id, so the model's embedding table and output
head both have one extra row for it.

## 3. Packing: one corpus, one id stream

Packing is just "encode every document, then join the id lists with a separator between them".
Here is the implementation from the course codebase
(`code/src/llmre/data/loader.py`):

```python
import torch

def pack_documents(list_of_id_lists: list[list[int]], eot_id: int) -> torch.Tensor:
    """Concatenate encoded documents into one id stream, separated by eot_id."""
    stream: list[int] = []
    for doc in list_of_id_lists:
        stream.extend(doc)
        stream.append(eot_id)          # boundary marker after every document
    return torch.tensor(stream, dtype=torch.long)
```

### Numerical example

Two tiny "documents" already encoded to ids — `[10, 11, 12]` and `[20, 21]` — packed with
`eot_id = 0`:

```python
stream = pack_documents([[10, 11, 12], [20, 21]], eot_id=0)
# tensor([10, 11, 12,  0, 20, 21,  0])
```

The result is a **1-D `torch.long` tensor** of length $3 + 1 + 2 + 1 = 7$. The two zeros are the
document boundaries. `long` (64-bit integer) is the dtype every id tensor uses in this course,
because token ids are indices into the embedding table and PyTorch's embedding lookup requires an
integer index type.

<div class="callout pt"><p>Real corpora are far too large to hold as a Python list. In practice
you stream documents through the tokenizer and write the ids straight to a memory-mapped
<code>uint16</code> file on disk (a vocab under 65536 fits in 16 bits), then <code>np.memmap</code>
it at training time so the OS pages in only the windows you touch. The logic is identical to the
toy above; only the storage changes.</p></div>

## 4. Cutting windows: the one-position shift

Now the key step. The model reads a length-$T$ window `x` and, at *every* position $t$, must
predict the token that actually comes next. So the target window `y` is just `x` slid one step to
the left: `y[t] = x[t+1]`. One window therefore supplies $T$ next-token prediction problems at
once (position 0 predicts token 1, position 1 predicts token 2, …), and the causal mask from
[05.2](../module-05/lesson-02.md) guarantees position $t$ can only use tokens $\le t$ to make its
prediction — no cheating by peeking at the answer.

To build a window we need $T+1$ consecutive tokens from the stream: the first $T$ are `x`, and the
last $T$ (offset by one) are `y`. The batch is `batch_size` such windows sampled at random start
positions and stacked:

```python
def get_batch(data, block_size, batch_size, device="cpu"):
    high = data.numel() - block_size            # last valid start (need block_size+1 tokens)
    ix = torch.randint(low=0, high=high, size=(batch_size,))
    x = torch.stack([data[i     : i + block_size    ] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)
```

### Numerical example — watch the shift

Take a stream that is just `0, 1, 2, …, 11` so the ids double as positions, `block_size = 4`,
`batch_size = 2`, seeded for reproducibility:

```python
data = torch.arange(12)                          # tensor([0,1,2,...,11])
torch.manual_seed(0)
x, y = get_batch(data, block_size=4, batch_size=2)
# x = [[4, 5, 6, 7],       y = [[5, 6, 7, 8],
#      [7, 8, 9, 10]]           [8, 9, 10, 11]]
```

Both `x` and `y` have shape **`(batch_size, block_size) = (2, 4)`**, dtype `long`. Read row 0:
the input is `4 5 6 7`, the target is `5 6 7 8`. Position 0 sees `4` and must predict `5`;
position 1 sees `4 5` and must predict `6`; and so on. Formally, `y[:, :-1]` equals `x[:, 1:]` —
the target at each position (except the last) is literally the next input token:

```python
torch.equal(x[:, 1:], y[:, :-1])                 # True
```

That single boolean is the sanity check to run whenever you touch a data loader.

<div class="callout warn"><p><strong>The classic off-by-one.</strong> If you write
<code>y = data[i : i + block_size]</code> (forgetting the <code>+1</code>), the model is trained to
predict the token it can already see — trivial to fit, and the loss will plummet to near zero on
train while the model learns nothing useful. A suspiciously low training loss in the first few
steps almost always means the targets are not shifted.</p></div>

## 5. Tensor-shape summary

| Object | Shape | Dtype | Meaning |
|---|---|---|---|
| a document's ids | `(L_doc,)` list | int | one encoded document |
| packed `stream` | `(N,)` | `long` | whole corpus, docs joined by `eot_id` |
| `x` (a batch) | `(B, T)` | `long` | `B` input windows of length `T = block_size` |
| `y` (a batch) | `(B, T)` | `long` | same windows shifted one token left |

`x` then enters the model's embedding table (`(B, T)` → `(B, T, C)`, next module), and `y` is the
target passed to the cross-entropy loss you built in [01.4](../module-01/lesson-04.md).

## 6. Under the hood: why random windows, and the cost

`get_batch` samples **random** start positions rather than marching through the stream in order.
Two reasons. First, successive minibatches are then near-independent, which is the "stochastic" in
SGD — ordered windows would make consecutive gradients highly correlated and training less stable.
Second, it is trivially cheap: no shuffling of the giant array, just `batch_size` random integers
and a gather. The cost of one `get_batch` is `O(B · T)` memory copies — negligible next to the
forward/backward pass — which is why we can afford to re-sample every step. (With a `memmap`, the
only real cost is the disk pages the OS faults in for those windows.)

One consequence: because starts are uniform over the stream, a given token is seen in many
different context alignments across an epoch, and window boundaries fall in different places each
time — a mild, free form of data augmentation.

## Exercise

You have `data = torch.arange(100)` and call `get_batch(data, block_size=8, batch_size=16)`.

1. What are the shapes and dtype of `x` and `y`?
2. What is the largest start index `i` that could be sampled, and why that value?

<details><summary>Hint</summary>
A window needs <code>block_size + 1</code> tokens (the window plus its one-step shift). Count
backwards from the end of the stream.
</details>

<details><summary>Solution</summary>

1. Both `x` and `y` are shape `(16, 8)` with dtype `torch.long`.
2. `get_batch` uses `high = data.numel() - block_size = 100 - 8 = 92` as the *exclusive* upper
   bound for `torch.randint`, so the largest start is `i = 91`. At `i = 91`, `x = data[91:99]`
   and `y = data[92:100]` — the last target index is `99`, the final token, so nothing runs off
   the end. A start of `92` would make `y = data[93:101]` and index `100` does not exist.

</details>

## Check yourself

<details><summary>Why append <code>&lt;|endoftext|&gt;</code> between documents instead of just concatenating them?</summary>

So the model does not learn to predict the start of one document as the continuation of an
unrelated one, and so it has an explicit "document over" signal it can later emit to stop
generating. The separator is a real vocabulary id with its own embedding.

</details>

<details><summary>Given <code>x[0] = [4, 5, 6, 7]</code>, what is <code>y[0]</code>, and what does position 2 of this window learn to predict?</summary>

`y[0] = [5, 6, 7, 8]`. Position 2 sees tokens `4, 5, 6` (the causal mask blocks `7`) and is
trained to predict `y[0][2] = 7`.

</details>

<details><summary>A colleague's model hits a training loss near 0 within 10 steps on real text. What is the single most likely bug?</summary>

The targets are not shifted — `y` equals `x` (or is shifted the wrong way), so the model is
"predicting" a token it can already see. Check `torch.equal(x[:, 1:], y[:, :-1])`.

</details>

<details><summary>Why does the id tensor use dtype <code>long</code> and not <code>float32</code>?</summary>

Token ids are integer indices into the embedding table; PyTorch's embedding lookup and
cross-entropy target argument both require an integer (`long`) index type. Floats would be both
meaningless as indices and a type error.

</details>

## Next

The corpus is now a stream of ids, and `get_batch` hands the model `(x, y)` windows of shape
`(B, T)`. But the model cannot consume integer ids directly — it needs vectors. The next module
turns each id into a learned embedding vector, adds positional information, and builds the
attention and transformer machinery that transforms those vectors. That is where the id `x`
becomes the `(B, T, C)` activation that flows through the network.

Continue to [05.1 · Embeddings & positional encoding](../module-05/lesson-01.md).
