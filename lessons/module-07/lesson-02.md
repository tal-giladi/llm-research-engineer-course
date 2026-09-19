# 07.2 · Micro-batch, global batch, gradient accumulation

<div class="prereq">
<p><strong>Prerequisites:</strong> the training loop from <a href="lesson-01.md">07.1 · The training loop</a>; that gradients <em>accumulate</em> across <code>backward()</code> calls (07.1, §5); the mean cross-entropy loss from <a href="../module-06/lesson-01.md">06.1</a>.</p>
<p><strong>You will learn:</strong> the precise vocabulary of batch sizes — sequence length $T$, micro-batch $b$, gradient-accumulation steps $G$, data-parallel workers $W$, global batch $B_{\text{global}} = b\cdot G\cdot W$, and tokens per optimizer step $= B_{\text{global}}\cdot T$; <em>why</em> gradient accumulation exists (a large effective batch that does not fit in memory); and how to implement it correctly by scaling each micro-batch loss by $1/G$.</p>
<p><strong>Why this matters for ML:</strong> large models are trained with global batches of hundreds of thousands to millions of tokens, far more than fit in one GPU's memory at once. Gradient accumulation is the universal trick that decouples the batch size your <em>math</em> wants from the batch size your <em>memory</em> allows. Every serious training config is, at heart, a choice of $b$, $G$, and $W$.</p>
</div>

## 1. Intuition: the batch you want vs. the batch that fits

In [07.1](lesson-01.md) each step used one small batch. But the *quality* of a gradient step depends on how many examples it averages over: a bigger batch gives a less noisy estimate of the true gradient, which lets you take larger, more confident steps and is part of why large-scale training uses enormous batches (millions of tokens per step for the biggest models).

The problem: a batch of a million tokens does not fit in a single GPU's memory — the activations alone (07.1, §6) would be enormous. So we split the batch we *want* into pieces small enough to fit, run them one at a time, and **add up their gradients** before taking a single optimizer step. The optimizer never knows the batch arrived in pieces; it sees one big accumulated gradient. That is **gradient accumulation**.

<div class="callout key"><p>Gradient accumulation decouples the <em>effective</em> batch size (what the optimizer sees, chosen for learning dynamics) from the <em>micro</em>-batch size (what one forward/backward can hold, chosen for memory). You get the gradient of a huge batch while only ever materializing a small one's activations.</p></div>

## 2. Mathematics: five quantities, one product

Define, precisely:

| symbol | name | meaning |
|---|---|---|
| $T$ | sequence length / block size | tokens per sequence |
| $b$ | micro-batch size | sequences per forward/backward on **one** worker |
| $G$ | gradient-accumulation steps | micro-batches summed before one optimizer step |
| $W$ | data-parallel workers | independent devices each processing their own micro-batches |
| $B_{\text{global}}$ | global batch size | total sequences per optimizer step |

They combine as

$$
B_{\text{global}} = b \cdot G \cdot W, \qquad
\text{tokens per step} = B_{\text{global}} \cdot T = b \cdot G \cdot W \cdot T.
$$

The **tokens per step** is the number that matters for scaling laws (Module 10) and for reading any training config: it is how much text the model learns from between weight updates. `llmre.training.metrics.tokens_per_step(b, G, W, T)` is exactly this product.

### Why the $1/G$ scaling is required

The loss for a full batch of $b\cdot G$ sequences is a **mean** over all its tokens (06.1). Split it into $G$ micro-batches of $b$ sequences each. Because every micro-batch has the *same* number of tokens, the full-batch mean equals the average of the $G$ micro-batch means:

$$
\mathcal{L}_{\text{full}} = \frac{1}{G}\sum_{g=1}^{G} \mathcal{L}_g.
$$

Gradients are linear, so

$$
\nabla \mathcal{L}_{\text{full}} = \frac{1}{G}\sum_{g=1}^{G} \nabla \mathcal{L}_g = \sum_{g=1}^{G} \nabla\!\left(\frac{\mathcal{L}_g}{G}\right).
$$

That last form is the recipe: call `(loss_g / G).backward()` on each micro-batch and let the `.grad`s accumulate (07.1, §5). After $G$ micro-steps, `.grad` holds exactly $\nabla\mathcal{L}_{\text{full}}$. **Forget the $1/G$ and your gradient is $G$ times too large** — equivalent to secretly multiplying the learning rate by $G$.

## 3. Numerical example: two micro-batches, worked by hand

Take $G = 2$ micro-batches. Suppose the per-token losses are $[2, 4]$ in micro-batch 1 and $[6, 0]$ in micro-batch 2.

- Full batch mean: $\dfrac{2 + 4 + 6 + 0}{4} = 3.0$.
- Micro means: $\mathcal{L}_1 = \dfrac{2+4}{2} = 3.0$, $\mathcal{L}_2 = \dfrac{6+0}{2} = 3.0$.
- Average of micro means: $\dfrac{1}{2}(3.0 + 3.0) = 3.0$. ✓

The full mean and the $\frac{1}{G}$-weighted sum of micro means agree exactly, because each micro-batch holds the same token count. This is not an approximation — it is an identity, and the from-scratch test `test_grad_accumulation_matches_full_batch` in `code/tests/test_training.py` checks the *gradients* match to $10^{-5}$ on a real tiny GPT.

<div class="callout warn"><p>The identity relies on <strong>equal-size</strong> micro-batches. If the last micro-batch is short (e.g. the tail of an epoch), a plain <code>1/G</code> average slightly mis-weights it — every token no longer carries equal weight. Production loops handle this by weighting each micro-batch by its token count, or simply by dropping the ragged tail. For fixed <code>b</code> (our case) it is exact.</p></div>

## 4. Tensor shapes: nothing new, just repeated

Gradient accumulation adds **no** new tensors. Each micro-step is an ordinary forward/backward on a `(b, T)` batch; the only thing that persists across micro-steps is the `.grad` on each parameter, which is the same shape as the parameter and simply gets added to $G$ times before `optimizer.step()` reads it and `zero_grad()` clears it.

```text
per micro-step:   x, y      (b, T)         int64
                  loss      ()             float32   (loss / G).backward()
across G steps:   p.grad    shape of p     float32   accumulates the sum
after step():     p         shape of p     float32   updated once
```

Peak *activation* memory is set by $b$ (one micro-batch), not by $B_{\text{global}}$ — that is the entire payoff. You pay for a batch of $b$ in memory but learn from a batch of $b\cdot G\cdot W$.

## 5. From-scratch PyTorch: the accumulation inner loop

This is the exact block from `llmre/training/loop.py` (07.1), now with the accumulation made explicit:

```python
optimizer.zero_grad()                      # clear grads ONCE, before the G micro-steps
step_loss = 0.0
for _ in range(cfg.grad_accum_steps):      # G micro-batches
    x, y = get_batch("train")              # (b, T), (b, T)
    _, loss = model(x, y)                  # scalar mean loss for THIS micro-batch
    step_loss += loss.item() / cfg.grad_accum_steps   # for logging, same scale
    (loss / cfg.grad_accum_steps).backward()          # accumulate (1/G)*grad

grad_norm = clip_grad_norm_(params, cfg.grad_clip)    # clip the SUMMED gradient
optimizer.step()                                       # one update for all G
```

Three things to notice:

- **`zero_grad()` is outside** the micro-loop — we *want* the $G$ gradients to accumulate, so we clear only once per optimizer step.
- **`(loss / G).backward()`** applies the $1/G$ scale from section 2, so the accumulated `.grad` equals the full-batch gradient.
- **Clip and step run once**, after all $G$ micro-batches, on the summed gradient — so clipping sees the true full-batch gradient norm, exactly as it would without accumulation.

Data-parallel workers $W$ are the third multiplier: each of $W$ devices runs this same loop on its own micro-batches, and an **all-reduce** averages their gradients before `step()` so all workers stay identical. We build that in [Module 9](../module-09/lesson-01.md); until then, $W = 1$ and $B_{\text{global}} = b\cdot G$.

## 6. Under the hood: accumulation is compute-for-memory

Gradient accumulation trades **wall-clock time for memory**. Running $G$ micro-batches sequentially takes ~$G$× as long as one micro-batch of the same size *would* (if it fit), because the GPU does the same total arithmetic either way — it just cannot do it all at once. What you save is peak activation memory, which scales with $b$, not $b\cdot G$.

There is a subtlety with normalization layers. Our LayerNorm (Module 5) normalizes each token independently, so it is unaffected by how the batch is split. **BatchNorm** would break — its statistics are computed across the batch dimension, so $G$ micro-batches of $b$ give different statistics than one batch of $b\cdot G$. This is one of several reasons transformers use LayerNorm/RMSNorm, never BatchNorm: they are invariant to how you slice the batch, which makes accumulation (and data parallelism) exact.

## Exercise

You are configuring a run. **Target:** 491,520 tokens per optimizer step. Your sequence length is $T = 1024$, you have $W = 8$ data-parallel GPUs, and profiling shows each GPU can hold a micro-batch of at most $b = 6$ sequences before running out of memory. Find the gradient-accumulation steps $G$ that hits the target exactly.

<details><summary>Hint</summary>
First convert the token target to a target in <em>sequences</em>: divide by $T$. Then use $B_{\text{global}} = b\cdot G\cdot W$.
</details>

<details><summary>Stronger hint</summary>
Global sequences $= 491520 / 1024 = 480$. Now solve $480 = 6\cdot G\cdot 8$ for $G$.
</details>

<details><summary>Solution</summary>

Convert tokens to sequences: $B_{\text{global}} = 491520 / T = 491520 / 1024 = 480$ sequences per step.

Solve for $G$:
$$
G = \frac{B_{\text{global}}}{b \cdot W} = \frac{480}{6 \cdot 8} = \frac{480}{48} = 10.
$$

So $G = 10$: each of the 8 GPUs runs 10 micro-batches of 6 sequences, giving $6\cdot 10\cdot 8 = 480$ sequences $= 480 \cdot 1024 = 491{,}520$ tokens per step. Check with `tokens_per_step(6, 10, 8, 1024)` → `491520`. ✓

If the target had not divided evenly (say 500,000 tokens), you would round $G$ to the nearest integer and accept a slightly different actual tokens/step — the schedule and scaling-law targets are not sensitive to a few percent.

</details>

## Common mistakes

- **Dropping the $1/G$.** The accumulated gradient is then $G$× too big — a silent $G$× learning-rate increase that usually diverges. The step-0 loss looks fine; step 5 explodes.
- **Zeroing grads inside the micro-loop.** Then only the *last* micro-batch's gradient survives; you paid for $G$ forwards and used one. The loss goes down (it is still a valid gradient), just from a batch $G$× smaller than intended.
- **Confusing $b$ with $B_{\text{global}}$ when setting the learning rate.** LR schedules are tuned to the *global* batch; if you change $G$ to fit memory, keep $B_{\text{global}}$ (and thus the LR) fixed.
- **Assuming BatchNorm-style layers accumulate correctly.** They do not. Transformers avoid them; if you add one, accumulation and data parallelism stop being exact.

## Check yourself

<details><summary>With $b = 12$, $G = 40$, $W = 8$, $T = 1024$, how many sequences and how many tokens per optimizer step?</summary>

Sequences: $B_{\text{global}} = 12 \cdot 40 \cdot 8 = 3840$. Tokens: $3840 \cdot 1024 = 3{,}932{,}160$ — about 3.9M tokens per step. (`tokens_per_step(12, 40, 8, 1024)`.)

</details>

<details><summary>Why do we scale each micro-batch loss by $1/G$ rather than, say, summing the raw losses?</summary>

Because the full-batch loss is a *mean*, and with equal-size micro-batches the full mean equals the average of the micro means. Summing raw losses (or raw gradients) gives $G$× the mean gradient, i.e. an unintended $G$× larger step. The $1/G$ makes the accumulated gradient equal the true full-batch gradient.

</details>

<details><summary>You halve your micro-batch $b$ to fit a longer context, but want the same global batch. What must change, and does peak activation memory go up or down?</summary>

Double $G$ to keep $B_{\text{global}} = b\cdot G\cdot W$ constant. Peak activation memory is set by the micro-batch, so halving $b$ lowers it (though a longer context raises the per-sequence activation cost — the two effects trade off).

</details>

<details><summary>Why is LayerNorm compatible with gradient accumulation but BatchNorm is not?</summary>

LayerNorm normalizes each token over its feature dimension, independently of other examples, so splitting the batch changes nothing. BatchNorm computes mean/variance across the batch dimension, so $G$ micro-batches of $b$ give different statistics than one batch of $b\cdot G$ — the result depends on how you slice, breaking the equivalence.

</details>

## Next

You can now size a global batch and split it to fit memory. Real runs also stop and restart — hardware fails, jobs get pre-empted, you want to resume from last night. Doing that *exactly*, so a resumed run is bit-for-bit identical, means saving more than the weights.

Continue to [07.3 · Checkpointing, resuming, seeds, reproducibility](lesson-03.md).
