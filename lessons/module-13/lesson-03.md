# 13.3 · A reproducible eval harness

<div class="prereq">
<p><strong>Prerequisites:</strong> loss/perplexity and log-likelihood scoring from <a href="#/lessons/module-13/lesson-01">13.1 · Loss, perplexity, zero-/few-shot</a>; contamination, calibration, and metric honesty from <a href="#/lessons/module-13/lesson-02">13.2 · Contamination, calibration, task eval</a>; the model's <code>forward(idx, targets) -&gt; (logits, loss)</code> from <a href="#/lessons/module-06/lesson-01">06.1 · Assembling GPT-2</a>; the from-scratch <code>log_softmax</code> and <code>cross_entropy</code> from <a href="#/lessons/module-01/lesson-04">01.4</a>; seeds and reproducibility from <a href="#/lessons/module-07/lesson-03">07.3 · Checkpointing, resuming, seeds, reproducibility</a>.</p>
<p><strong>You will learn:</strong> how to turn the metrics of this module into a small, reusable <strong>evaluation harness</strong> — <code>evaluate_perplexity</code>, <code>sequence_loglikelihood</code>, and <code>multiple_choice_score</code> — with documented shapes and no autograd overhead; how to make evaluation <strong>reproducible</strong> (fixed seeds, fixed eval set, versioned config); and how <strong>regression testing</strong> locks metrics in so a change that quietly hurts them is caught automatically. You will see the harness run end-to-end on a tiny GPT trained a few steps.</p>
<p><strong>Why this matters for ML:</strong> research velocity is bounded by how quickly and reliably you can measure. An evaluation you cannot reproduce is worse than none — it produces numbers that drift between runs and lead you to false conclusions. A first-class harness is the instrument every experiment in the rest of the course reads from.</p>
</div>

## 1. Evaluation is a first-class discipline

In a software project you would never ship a change without a test suite. In an ML project the equivalent is the evaluation harness: the code that, given a model, produces the metrics you make decisions on. Treat it with the same seriousness as production code — documented interfaces, deterministic behavior, and its own tests.

Everything below builds on pieces you already have. The forward pass returns `(logits, loss)` from <a href="#/lessons/module-06/lesson-01">06.1</a>; the numerically stable `log_softmax` is from <a href="#/lessons/module-01/lesson-04">01.4</a>. The harness just orchestrates them over held-out data and returns plain Python floats and indices you can log, compare, and assert on.

<div class="callout key"><p>Three properties make a harness trustworthy: it is <strong>deterministic</strong> (same inputs → same numbers, every run), <strong>documented</strong> (every function states tensor shape/dtype/device), and <strong>tested</strong> (its own metrics are checked against known-answer cases). Lose any one and the numbers become unreliable.</p></div>

## 2. `evaluate_perplexity` over a held-out stream

The first tool is the held-out perplexity of <a href="#/lessons/module-13/lesson-01">13.1</a>, computed over a long token stream. The design:

- **Input.** A flat 1-D `torch.long` tensor `data` of held-out token ids, a `block_size` (the context length $T$ each window uses), a `batch_size`, and an optional `max_batches` to stop early on a huge stream.
- **Windows.** Cut `data` into non-overlapping windows of length `block_size`. Window $i$ gives input `x = data[i·T : i·T + T]` and target `y = data[i·T + 1 : i·T + 1 + T]` — the same window shifted one token right, exactly the next-token setup from <a href="#/lessons/module-07/lesson-01">07.1</a>.
- **Aggregation.** For each batch, `model(x, y)` returns the mean cross-entropy over that batch's $B \cdot T$ target tokens. We accumulate a **token-weighted** mean (multiply each batch's mean loss by its token count, sum, divide by total tokens) so the result does not depend on how the windows happen to divide into batches. The perplexity is `exp(mean_loss)`.
- **No grad.** The whole thing runs under `torch.no_grad()` with the model in `eval()` mode: evaluation builds no autograd graph (saving memory and time) and disables dropout so the numbers are deterministic.

<div class="callout pt"><p><code>torch.no_grad()</code> tells PyTorch not to record operations for backpropagation, so no computation graph is retained — this is why evaluation uses a fraction of the memory of a training step. <code>model.eval()</code> flips modules like dropout and (if present) batch-norm into inference behavior; for our GPT it disables dropout so a given input always produces the same logits. Forgetting either does not change correctness of a single run, but <code>eval()</code> is required for determinism and <code>no_grad()</code> for memory.</p></div>

Here is the core, from `llmre.evaluation.harness`:

```python
@torch.no_grad()
def evaluate_perplexity(model, data, block_size, batch_size=8, max_batches=None):
    model.eval()
    n_windows = (data.numel() - 1) // block_size
    total_nats, total_tokens = 0.0, 0
    for start in range(0, n_windows, batch_size):
        stop = min(start + batch_size, n_windows)
        xs, ys = [], []
        for i in range(start, stop):
            off = i * block_size
            xs.append(data[off : off + block_size])        # (block_size,)
            ys.append(data[off + 1 : off + 1 + block_size])
        x = torch.stack(xs)          # (B, block_size) long
        y = torch.stack(ys)          # (B, block_size) long
        _, loss = model(x, y)        # scalar mean nats over B*block_size
        n_tok = x.numel()
        total_nats += float(loss) * n_tok
        total_tokens += n_tok
        # (max_batches early-stop omitted here)
    return math.exp(total_nats / total_tokens)
```

The token-weighted accumulation (`float(loss) * n_tok`, then divide by `total_tokens`) is the subtle-but-important part: averaging the per-batch losses directly would over-weight a small final batch. Weighting by token count makes the result identical to computing one mean over all target tokens at once.

## 3. `sequence_loglikelihood`: the likelihood primitive

Task evaluation needs the total log-probability the model assigns to a sequence, $\sum_{t} \ln q(x_t \mid x_{<t})$. Feeding the whole sequence once gives every conditional in a single forward pass, because the logits at position $t$ are the model's prediction for token $t+1$:

```python
@torch.no_grad()
def sequence_loglikelihood(model, ids):        # ids: (L,) long, L >= 2
    model.eval()
    logits, _ = model(ids.unsqueeze(0))        # (1, L, V)
    logits = logits[0]                         # (L, V)
    logp = log_softmax(logits[:-1])            # (L-1, V): positions predict tokens 1..L-1
    targets = ids[1:]                          # (L-1,)
    chosen = logp[torch.arange(targets.numel()), targets]   # (L-1,)
    return float(chosen.sum())                 # <= 0
```

The index arithmetic is the whole trick: `logits[:-1]` drops the last position (which would predict a token past the end of the sequence), and `ids[1:]` is the set of tokens those positions predict. The first token is never scored — under a plain LM it has no prefix. The result is a single Python float $\le 0$; less negative means the model finds the sequence more probable.

## 4. `multiple_choice_score`: grading tasks

`multiple_choice_score` is the length-normalized log-likelihood scorer from <a href="#/lessons/module-13/lesson-01">13.1</a>. For each option it builds `[context; option]`, runs one forward pass, sums the log-probabilities of the **option** tokens only (using the same index arithmetic as §3, offset by the context length), optionally divides by the option length, and returns the argmax option index. The signature and shapes are documented on the function; the key line is that the option's tokens sit at absolute positions $L_c \dots L_c + L_o - 1$ and are predicted by logits at positions $L_c - 1 \dots L_c + L_o - 2$.

Because it takes any `model(seq) -> (logits, loss)`, the same scorer grades zero-shot and few-shot alike — the number of in-context examples just changes how many tokens are in `context_ids`.

## 5. Reproducibility: seeds, fixed eval set, versioned config

A metric you cannot reproduce is not a measurement. Three habits make evaluation deterministic and comparable across experiments (the training-side versions are in <a href="#/lessons/module-07/lesson-03">07.3</a>):

- **Fixed seeds.** Seed every source of randomness before an eval that uses any — `torch.manual_seed(...)`, and NumPy/Python if you sample. Perplexity and log-likelihood scoring are deterministic given `eval()` mode, but anything involving sampling (e.g. generating completions for pass@k) must be seeded to be reproducible.
- **Fixed eval set.** The held-out stream and the benchmark question set must be frozen and versioned. Comparing run B against run A is only valid if both were measured on the *same* data. Changing the eval set silently turns a comparison into noise.
- **Versioned config.** Record everything that affects the number: model checkpoint hash, tokenizer version, `block_size`, `batch_size`, the eval-set version/hash, the scoring rule (summed vs length-normalized), the number of few-shot examples, and library versions. Store it next to the result (a config file or the experiment log) so any number can be regenerated months later.

<div class="callout warn"><p>The most common silent reproducibility bug is comparing two numbers produced under different settings — a different <code>block_size</code>, a different normalization, a decontaminated vs raw eval set, or a changed tokenizer. The numbers look comparable and are not. Log the full config with every metric so mismatches are visible.</p></div>

## 6. Regression testing: lock the metrics in

The reason evaluation code has its own tests is the same reason production code does: to catch a change that quietly breaks things. An evaluation **regression test** pins a metric to a known value (or a known-answer synthetic case) so that a future refactor which changes the number fails loudly instead of silently shifting your results.

`code/tests/test_eval.py` does exactly this with known-answer cases that need no training:

- **Uniform model ⇒ perplexity = vocabulary size.** A toy model returning all-zero logits is uniform over $V$ tokens, so its held-out perplexity must equal $V$ exactly. This regression-tests `evaluate_perplexity` end-to-end: `evaluate_perplexity(UniformModel(V), data, block_size=16)` returns $V$.
- **Constructed logits ⇒ the favored option wins.** A toy model that puts a large logit on one token lets us assert `multiple_choice_score` picks the option built from that token, and that swapping the options flips the answer — pinning the scoring logic.
- **Length normalization changes the winner.** A model with two known per-token confidences reproduces the §13.1 flip: the summed rule picks the short option, the normalized rule picks the longer, higher-confidence one. This locks in the normalization behavior.
- **ECE is 0 for perfectly-calibrated synthetic data and positive for miscalibrated data.** 100 predictions at confidence $0.5$ with exactly half correct give $\mathrm{ECE}=0$; 100 predictions at confidence $0.9$ all wrong give $\mathrm{ECE}=0.9$.

<div class="callout key"><p>Known-answer tests are the backbone of a trustworthy harness: pick inputs whose correct metric you can compute by hand (uniform → perplexity $V$, all-wrong-at-0.9 → ECE 0.9), and assert the harness reproduces them. If a refactor changes one of these, you know instantly — before the wrong number reaches an experiment.</p></div>

## 7. Running the harness on a tiny trained model

To see all the pieces work together on a real (if tiny) `GPT`, `test_eval.py` trains a 2-layer model a few dozen steps on a fixed repeating pattern `0,1,2,…,V-1,0,1,…` — a sequence whose next token is fully learnable — then runs the harness:

```python
import torch
from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT
from llmre.evaluation.harness import evaluate_perplexity, multiple_choice_score

torch.manual_seed(0)                      # reproducibility: fix the seed
V, T = 16, 8
cfg = GPTConfig(vocab_size=V, block_size=T, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
model = GPT(cfg)

stream = torch.arange(V).repeat(60)       # the fixed eval/train pattern, length 960
opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
model.train()
for _ in range(60):                       # a few steps: enough to learn the pattern
    i = torch.randint(0, stream.numel() - T - 1, (1,)).item()
    x = stream[i : i + T].unsqueeze(0)
    y = stream[i + 1 : i + 1 + T].unsqueeze(0)
    _, loss = model(x, y)
    opt.zero_grad(); loss.backward(); opt.step()

ppl = evaluate_perplexity(model, stream, block_size=T, batch_size=8)   # finite, small
# context ends at token 4 -> the true next token is 5, which should score highest:
ctx = torch.tensor([0, 1, 2, 3, 4])
assert multiple_choice_score(model, ctx, [torch.tensor([5]), torch.tensor([11])]) == 0
```

Before training, the model is roughly uniform and `evaluate_perplexity` returns about $V = 16$ (the sanity check from <a href="#/lessons/module-13/lesson-01">13.1</a>). After a few dozen steps it has learned the deterministic pattern, perplexity drops well below $V$, and `multiple_choice_score` correctly prefers the true continuation (`5`) over a wrong one (`11`). The full test asserts perplexity is finite and positive and that the right option wins — a complete, reproducible evaluation on a real model in under a second on CPU.

<div class="hw">
<p><strong>Hardware track.</strong></p>
<p><strong>Minimum:</strong> any CPU — the harness itself is inference-only and the toy example runs in well under a second. <strong>CPU-only: fully sufficient</strong> for this lesson and for the test suite (<code>py -m pytest -q</code>).</p>
<p><strong>Recommended:</strong> the same CPU is fine; a GPU only helps when you point <code>evaluate_perplexity</code> at a real held-out corpus of millions of tokens through a full-size model, where it becomes a batched inference job (seconds to minutes depending on stream length and model size).</p>
<p><strong>Expected runtime:</strong> the tiny-GPT example and each harness test complete in &lt; 1 s on CPU. <strong>GPU memory:</strong> negligible for the toy model; for a real model, evaluation under <code>torch.no_grad()</code> uses far less than training (no optimizer state, no saved activations for backward). <strong>GPU-hours:</strong> ~0.</p>
</div>

## Exercise

`evaluate_perplexity` accumulates `total_nats += float(loss) * n_tok` and finally returns `exp(total_nats / total_tokens)`. Suppose a teammate "simplifies" it to average the per-batch losses directly — `losses.append(float(loss))` then `exp(mean(losses))`. On a stream whose last batch is smaller than the others, does this give the same answer? Why or why not?

<details><summary>Optional hint</summary>

Each batch's `loss` is already a mean over that batch's tokens. Averaging the means treats a small final batch as equal in weight to a full one.

</details>

<details><summary>Stronger hint</summary>

Compare the mean of batch means to the overall mean when batches have unequal token counts. Consider two batches with means 2.0 (over 800 tokens) and 4.0 (over 8 tokens).

</details>

<details><summary>Solution</summary>

It gives a **different** (wrong) answer whenever the batches contain unequal numbers of tokens. Each batch's `loss` is the mean over *its* tokens, so averaging the batch means weights every batch equally regardless of size. The token-weighted version weights each batch by its token count, which reproduces the single mean over all target tokens.

Concretely, with batch A = mean 2.0 over 800 tokens and batch B = mean 4.0 over 8 tokens: the correct token-weighted mean is $(2.0\cdot800 + 4.0\cdot8)/808 = 1632/808 \approx 2.02$, while the mean-of-means is $(2.0+4.0)/2 = 3.0$ — badly off, because the tiny 8-token batch is given the same weight as the 800-token one. This is exactly why the harness accumulates `float(loss) * n_tok`.

</details>

## Common mistakes

- **Running eval without `model.eval()`** — dropout stays active, so the same input gives different logits each run and the metric is non-deterministic.
- **Running eval without `torch.no_grad()`** — correctness is unaffected for one pass, but PyTorch retains the graph and evaluation can OOM on a real model.
- **Mean-of-batch-means instead of token-weighted mean** — biased whenever batches differ in token count.
- **Comparing metrics across different configs** — a different `block_size`, tokenizer, or scoring rule makes two numbers incomparable even though both are "perplexity" or "accuracy".
- **No regression test on the harness** — a refactor silently changes the number and you trust it anyway.

## Debugging exercise

This `evaluate_perplexity` variant returns a number that is far too *low* (too optimistic). What is wrong?

```python
@torch.no_grad()
def evaluate_perplexity(model, data, block_size, batch_size=8):
    model.eval()
    n_windows = (data.numel() - 1) // block_size
    losses = []
    for start in range(0, n_windows, batch_size):
        stop = min(start + batch_size, n_windows)
        xs = [data[i*block_size : i*block_size + block_size] for i in range(start, stop)]
        ys = [data[i*block_size : i*block_size + block_size] for i in range(start, stop)]  # <-- ?
        x, y = torch.stack(xs), torch.stack(ys)
        _, loss = model(x, y)
        losses.append(float(loss))
    return math.exp(sum(losses) / len(losses))
```

<details><summary>Solution</summary>

The **targets `ys` are not shifted**: they slice `data[i*block_size : i*block_size + block_size]`, the *same* window as the inputs `xs`, instead of the window shifted one token right (`data[i*block_size + 1 : i*block_size + 1 + block_size]`). So the model is asked to predict each token *from itself* — the target at every position is the current input token, which the model can read directly. That makes the task trivial, the loss near zero, and the perplexity far too low. The fix is to shift the target slice by one:

```python
ys = [data[i*block_size + 1 : i*block_size + 1 + block_size] for i in range(start, stop)]
```

(The mean-of-means at the end is also not token-weighted, a second, smaller bug — see the exercise.)

</details>

## Check yourself

<details><summary>Why does the harness run under torch.no_grad() and model.eval(), and what breaks if you omit each?</summary>

`torch.no_grad()` stops PyTorch from recording the computation graph, so evaluation uses far less memory (no saved activations for a backward pass) — omitting it risks out-of-memory on a real model. `model.eval()` switches dropout (and any batch-norm) to inference behavior; omitting it leaves dropout active, so the same input yields different logits across runs and the metric becomes non-deterministic. Neither changes the mathematical value of a single deterministic pass, but both are required for a correct, reproducible, memory-efficient harness.

</details>

<details><summary>What does the uniform-model regression test assert, and why is it a good known-answer case?</summary>

It asserts that `evaluate_perplexity` on a model returning all-zero logits (uniform over $V$ tokens) equals exactly $V$. It is a good known-answer case because the correct value is derivable by hand — a uniform distribution has cross-entropy $\ln V$, so perplexity $\exp(\ln V) = V$ — independent of the data or the harness's internal batching. If the harness ever returns something other than $V$ here, a bug (bad shift, wrong aggregation) is present.

</details>

<details><summary>You measured perplexity 18.4 on run A with block_size 512 and 21.7 on run B with block_size 128, same model. Can you conclude A's setting is better?</summary>

No. Perplexity depends on the context length available at each position: with a longer `block_size`, most positions have more preceding context to condition on, so per-token loss (and thus perplexity) is generally lower — independent of any real quality difference. The two numbers were produced under different configs and are not comparable. To compare fairly, fix `block_size` (and every other eval setting) across runs, which is exactly why the config must be versioned with the metric.

</details>

<details><summary>In sequence_loglikelihood, why is the first token of the sequence never included in the sum?</summary>

Under a plain autoregressive LM, the score of a token is its conditional log-probability given the preceding tokens, $\ln q(x_t \mid x_{<t})$. The first token has no preceding context, so there is no conditional to score. The code enforces this by predicting tokens $1 \dots L-1$ from logits at positions $0 \dots L-2$ (`logits[:-1]` against `ids[1:]`), leaving token 0 unscored.

</details>

## Next

You have a reproducible evaluation harness — perplexity, sequence log-likelihood, and length-normalized multiple-choice scoring — with fixed seeds, a frozen eval set, versioned configs, and its own regression tests. This is the instrument every later experiment reads from. The next module takes the base model you have been measuring and begins to shape its behavior with supervised fine-tuning, where instruction-following evaluation (previewed in <a href="#/lessons/module-13/lesson-02">13.2</a>) becomes the metric that matters.

Continue to [14.1 · Instruction data &amp; chat templates](lessons/module-14/lesson-01.md).
