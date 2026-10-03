# 07.2 · Micro-batch, global batch, gradient accumulation

<div class="prereq">
<p><strong>Prerequisites:</strong> the training loop from <a href="#/lessons/module-07/lesson-01">07.1 · The training loop</a>; that gradients <em>accumulate</em> across <code>backward()</code> calls (07.1, §5); the mean cross-entropy loss from <a href="#/lessons/module-06/lesson-01">06.1</a>.</p>
<p><strong>You will learn:</strong> the precise vocabulary of batch sizes — sequence length $T$, micro-batch $b$, gradient-accumulation steps $G$, data-parallel workers $W$, global batch $B_{\text{global}} = b\cdot G\cdot W$, and tokens per optimizer step $= B_{\text{global}}\cdot T$; <em>why</em> gradient accumulation exists (a large effective batch that does not fit in memory); and how to implement it correctly by scaling each micro-batch loss by $1/G$.</p>
<p><strong>Why this matters for ML:</strong> large models are trained with global batches of hundreds of thousands to millions of tokens, far more than fit in one GPU's memory at once. Gradient accumulation is the universal trick that decouples the batch size your <em>math</em> wants from the batch size your <em>memory</em> allows. Every serious training config is, at heart, a choice of $b$, $G$, and $W$.</p>
</div>

## 1. Intuition: the batch you want vs. the batch that fits

In [07.1](lessons/module-07/lesson-01.md) each step used one small batch. But the *quality* of a gradient step depends on how many examples it averages over: a bigger batch gives a less noisy estimate of the true gradient, which lets you take larger, more confident steps and is part of why large-scale training uses enormous batches (millions of tokens per step for the biggest models).

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

### What a "data-parallel worker" is (first look — built in Module 9)

$W$ is the one quantity in this table you have not met yet, so here is enough to read this lesson; the full treatment is [09.1 · Data parallelism, all-reduce, DDP](lessons/module-09/lesson-01.md).

So far everything ran as **one process driving one device** (one GPU, or the CPU). A **worker** is one such process. With **data parallelism** you launch $W$ of them at once — typically one per GPU, e.g. $W = 8$ on an 8-GPU machine — and each one:

1. holds a **full, identical copy** of the model and optimizer state;
2. reads a **different** slice of the training data (worker 0 gets different sequences from worker 1, …);
3. runs the ordinary forward/backward (and its own $G$ accumulation micro-steps) on its slice, producing its own gradient;
4. joins an **all-reduce**: a collective communication operation in which every worker contributes its gradient tensor and every worker receives back the **average** of all $W$ of them;
5. calls `optimizer.step()` with that averaged gradient. Because all workers started identical and applied the identical averaged gradient, they are still identical afterwards — no worker ever has to send its weights to another.

Tiny example: $W = 2$ workers, $b = 4$, $G = 2$. Each worker processes $4 \times 2 = 8$ sequences per step; the two together process $16$. After the all-reduce, each worker's gradient is the average over all 16 sequences — exactly what one device would get from a batch of 16 if it had the memory and the time. So $W$ multiplies the batch the same way $G$ does; the difference is that $G$ buys a bigger batch with **more time on one device**, while $W$ buys it with **more devices in parallel**. Module 9 implements the all-reduce, shows why averaging gradients is mathematically the same as one big batch, and covers its communication cost. In this module we always have $W = 1$.

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

<div class="callout warn"><p>The identity relies on every micro-batch holding the <strong>same number of real target tokens</strong>. Our pretraining <code>get_batch</code> always returns full <code>(b, T)</code> windows, so for us it is exact. When token counts differ, a plain <code>1/G</code> average is wrong — the fix is below.</p></div>

### When micro-batches have different token counts

Two everyday situations break the equal-count assumption:

- **A ragged tail.** If you iterate over a fixed dataset in order, the last micro-batch of an epoch can be short (e.g. 3 sequences instead of $b = 8$).
- **Padding / ignored targets.** In fine-tuning ([Module 14](lessons/module-14/lesson-01.md)) sequences have different lengths and are padded to a common $T$; padded positions (and often the prompt tokens) get target $-1$, which our `GPT.forward` passes as `ignore_index=-1`, so they contribute no loss. Two micro-batches of the same shape can then contain very different numbers of *real* tokens.

**Worked example.** Micro-batch 1 has 3 real tokens with losses $[2, 4, 6]$; micro-batch 2 has 1 real token with loss $[0]$ (the other positions are padding).

- What we want — the mean over all 4 real tokens: $\dfrac{2+4+6+0}{4} = 3.0$.
- Plain $1/G$ averaging of micro means: $\mathcal{L}_1 = 12/3 = 4.0$, $\mathcal{L}_2 = 0/1 = 0.0$, and $\tfrac12(4.0 + 0.0) = 2.0$. **Wrong** — the single token in micro-batch 2 got the same total weight as the three tokens in micro-batch 1, so each of its tokens counts 3× as much.

**The fix: sum, then divide by the total token count.** Let $N_g$ be the number of real target tokens in micro-batch $g$ and $N = \sum_g N_g$. Compute each micro-batch's loss as a **sum** $S_g$ over its tokens (not a mean) and backpropagate $S_g / N$:

$$
\mathcal{L}_{\text{full}} = \frac{1}{N}\sum_{g=1}^{G} S_g, \qquad
\nabla \mathcal{L}_{\text{full}} = \sum_{g=1}^{G} \nabla\!\left(\frac{S_g}{N}\right).
$$

Check: $S_1 = 12$, $S_2 = 0$, $N = 4$, so $\tfrac{12}{4} + \tfrac{0}{4} = 3.0$. ✓ When every $N_g$ is equal to $n$, $N = G\,n$ and $S_g/N = (S_g/n)/G = \mathcal{L}_g / G$ — the plain recipe is just the special case.

Because $N$ must be known before the first `backward()`, fetch all $G$ micro-batches first and count:

```python
import torch.nn.functional as F

batches = [get_batch("train") for _ in range(G)]           # G x ((b, T), (b, T))
n_total = sum((y != -1).sum().item() for _, y in batches)  # real target tokens N

optimizer.zero_grad()
for x, y in batches:
    logits, _ = model(x)                                   # (b, T, V)
    loss_sum = F.cross_entropy(                            # S_g: SUM over real tokens
        logits.view(-1, logits.size(-1)), y.view(-1),
        ignore_index=-1, reduction="sum")
    (loss_sum / n_total).backward()                        # accumulate S_g / N
clip_grad_norm_(params, cfg.grad_clip)
optimizer.step()
```

The cost is that all $G$ batches of token ids sit in memory at once — only `(b, T)` int64 tensors, tiny next to activations, which still exist for one micro-batch at a time. The alternative for the ragged-tail case is to **drop the short last batch** (PyTorch's `DataLoader(..., drop_last=True)` does exactly that), which is standard in pretraining where one missing batch out of millions does not matter; padding in fine-tuning cannot be dropped, so there the token-weighted version is required. With $W > 1$ workers the same rule applies globally: $N$ must be the token count summed across *all* workers, which takes one extra all-reduce of a single number — covered in [09.1](lessons/module-09/lesson-01.md).

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

Data-parallel workers $W$ are the third multiplier: each of $W$ devices runs this same loop on its own micro-batches, and an **all-reduce** averages their gradients before `step()` so all workers stay identical. We build that in [Module 9](lessons/module-09/lesson-01.md); until then, $W = 1$ and $B_{\text{global}} = b\cdot G$.

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

Continue to [07.3 · Checkpointing, resuming, seeds, reproducibility](lessons/module-07/lesson-03.md).
