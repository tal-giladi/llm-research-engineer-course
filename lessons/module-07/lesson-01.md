# 07.1 · The training loop: batching & loss

<div class="prereq">
<p><strong>Prerequisites:</strong> the assembled model and its <code>forward(idx, targets) -&gt; (logits, loss)</code> from <a href="../module-06/lesson-01.md">06.1 · Assembling GPT-2</a>; the from-scratch <a href="../module-03/lesson-02.md">AdamW optimizer</a> and the <a href="../module-03/lesson-03.md">cosine-warmup schedule &amp; gradient clipping</a>; the <a href="../module-04/lesson-03.md">data loader</a> (<code>get_batch</code>, <code>pack_documents</code>); backprop / <code>loss.backward()</code> from <a href="../module-02/lesson-01.md">Module 2</a>.</p>
<p><strong>You will learn:</strong> the one canonical training loop that trains every GPT, line by line — <code>get_batch → model(x, y) → zero_grad → backward → clip → step → set lr</code>; how the pieces you built across Modules 2–6 snap together into <code>llmre.training.loop.train</code>; and a fully runnable tiny example that trains a small GPT to <em>overfit</em> a short repeated sequence, with the real loss curve dropping from ≈<code>ln(vocab)</code> toward zero.</p>
<p><strong>Why this matters for ML:</strong> this loop <em>is</em> pretraining. Everything else in the field — data, scale, parallelism, fine-tuning, RLHF — is a variation on, or a wrapper around, these six lines. Once you can write the loop from memory and watch a loss curve fall, "training a language model" stops being magic and becomes a program you own.</p>
</div>

## 1. Intuition: the same six steps, forever

Training a neural network is a loop, and it is shorter than most people expect. Every step does the same six things:

1. **Get a batch** of examples — inputs `x` and the answers `y`.
2. **Forward**: run the model on `x` to get predictions, and compare them to `y` to get a single number, the **loss** (how wrong we were).
3. **Zero the gradients** left over from the previous step.
4. **Backward**: `loss.backward()` fills every parameter's `.grad` with "how the loss changes if I nudge this parameter".
5. **Clip** the gradients so no single freak batch can blow up the weights.
6. **Step**: the optimizer nudges every parameter a little way *downhill* in loss, and we set the learning rate for the next step from the schedule.

Then repeat, thousands to millions of times. For a language model the "answer" `y` is just the input shifted by one position — the next token — so the data is free: any text is its own supervision. That is the whole reason language models scale.

<div class="callout key"><p>The entire training loop, in the order the code runs it:</p>
<p><code>lr = schedule(step)</code> → <code>x, y = get_batch()</code> → <code>logits, loss = model(x, y)</code> → <code>zero_grad()</code> → <code>loss.backward()</code> → <code>clip_grad_norm_()</code> → <code>optimizer.step()</code>. Everything else is logging, checkpoints, and scale.</p></div>

## 2. Mathematics: what one step minimizes

The loss for a batch is the mean next-token cross-entropy you met in [01.4](../module-01/lesson-04.md) and [06.1](../module-06/lesson-01.md). For a batch of $B$ sequences of length $T$, with model parameters $\theta$,

$$
\mathcal{L}(\theta) = \frac{1}{B T}\sum_{b=1}^{B}\sum_{t=1}^{T} -\log p_\theta\big(y_{b,t} \mid x_{b,1}, \dots, x_{b,t}\big),
$$

where $p_\theta(\cdot)$ is the softmax over the model's logits and $y_{b,t} = x_{b,t+1}$ is the true next token. Backprop computes the gradient $\nabla_\theta \mathcal{L}$, and the optimizer takes a step

$$
\theta \leftarrow \theta - \eta_{\text{step}} \cdot \text{AdamW}\big(\nabla_\theta \mathcal{L}\big),
$$

with the learning rate $\eta_{\text{step}}$ set by the cosine-warmup schedule of [03.3](../module-03/lesson-03.md). Every symbol here is something you already built: the loss is `model.forward`, the gradient is `loss.backward`, the update is `AdamW.step`, and $\eta_{\text{step}}$ is `cosine_warmup_lr(step, ...)`.

### The one number to anchor on: the untrained loss

Before a single step, the model's weights are random, so its output distribution over the $V$-token vocabulary is roughly **uniform**. A uniform distribution assigns probability $1/V$ to the correct token, so the cross-entropy is

$$
\mathcal{L}_{\text{init}} \approx -\log\frac{1}{V} = \log V.
$$

That is your sanity check on step 0. If $V = 16$, an untrained model should report a loss near $\log 16 = 2.7726$. If your very first loss is wildly different from $\log V$, something is wrong (bad init, wrong vocab size, a label bug) — stop and fix it before training further.

<div class="callout key"><p>At step 0, expect <code>loss ≈ ln(vocab_size)</code>. For GPT-2's <code>V = 50257</code> that is ≈ 10.82. This is the cheapest, most reliable bug-catcher in all of training: check step 0 against <code>ln(V)</code> every single run.</p></div>

## 3. Numerical example: the loop by hand for one step

Take a toy with vocabulary $V = 16$ and one tiny batch. Step 0 forward gives, say, `loss = 2.8103` — right at $\log 16 = 2.7726$, exactly as predicted for a fresh model (the small excess is just the random init not being perfectly uniform). After `loss.backward()`, AdamW nudges $\theta$ downhill. Do this repeatedly and the loss falls. Here is the *actual* curve from the runnable example in section 5 (a 2-layer, 26,240-parameter GPT on a repeated 16-token pattern), printed every 20 steps:

```text
step    loss
   0    2.8103     <- ~ ln(16) = 2.7726, the untrained baseline
  20    0.9762
  40    0.0446
  60    0.1416
  80    0.0039
 100    0.0020
 120    0.0020
 140    0.0018
 160    0.0337
 180    0.0199
 199    0.0252
```

The loss falls from the $\log V$ baseline to essentially zero: the model has *memorized* the short pattern. That is exactly what "overfit a tiny dataset" is supposed to look like, and it is the second-cheapest bug-catcher after the step-0 check — **if a model cannot overfit a handful of examples, the training code is broken**, and there is no point launching a big run. (The little bumps at steps 60 and 160 are the optimizer briefly overshooting on a near-zero loss; the cosine decay of the learning rate settles it.)

## 4. Tensor shapes: what flows through one step

For the tiny example: batch $B = 4$, context $T = 8$, width $C = 32$, vocab $V = 16$.

```text
x            (4, 8)         int64    token ids from get_batch, on device
y            (4, 8)         int64    x shifted left by one (the targets)
logits       (4, 8, 16)     float32  one score per vocab token per position
loss         ()             float32  scalar mean cross-entropy
p.grad       same as p      float32  filled by loss.backward(), one per parameter
```

`x` and `y` come out of the data loader already shaped `(B, T)`. `model(x, y)` returns the `(B, T, V)` logits and the scalar `loss`. `loss.backward()` walks the graph and writes a `.grad` tensor — identical in shape, dtype and device to each parameter — into every leaf. The optimizer reads those `.grad`s and writes back into the parameters in place. No shape in the model ever changes across a step; only the *values* of the parameters move.

## 5. From-scratch PyTorch: a fully runnable overfit

Here is the complete, runnable example that produced the curve above. It uses only pieces from earlier modules plus the loop we are about to write. Paste it into a file and run it with `py`.

```python
import math, torch
from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT
from llmre.training.loop import TrainConfig, train

torch.manual_seed(0)

# A tiny GPT: 2 layers, 2 heads, width 32, vocab 16. ~26k params — CPU-instant.
cfg = GPTConfig(vocab_size=16, block_size=8, n_layer=2, n_head=2,
                n_embd=32, dropout=0.0, bias=True)
model = GPT(cfg)
print("params:", model.num_params())            # 26240
print("ln(vocab):", math.log(cfg.vocab_size))    # 2.7726 -> expected step-0 loss

# The "dataset": one short pattern, repeated. Trivial to memorize.
pattern = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,2]
stream  = torch.tensor(pattern * 64, dtype=torch.long)   # (1024,)

def get_batch(split):                 # the loop calls get_batch("train")
    b, T = 4, cfg.block_size
    ix = torch.randint(0, stream.numel() - T - 1, (b,))
    x = torch.stack([stream[i:i+T]     for i in ix])      # (4, 8)
    y = torch.stack([stream[i+1:i+1+T] for i in ix])      # (4, 8) = x shifted
    return x, y

tcfg = TrainConfig(max_steps=200, micro_batch_size=4, warmup_steps=20,
                   max_lr=1e-2, min_lr=1e-3, weight_decay=0.0,
                   grad_clip=1.0, seed=0, log_interval=20)

history = train(model, get_batch, tcfg)
for s, l in zip(history["step"], history["loss"]):
    print(f"step {s:3d}  loss {l:.4f}")
```

And here is the loop itself — `llmre/training/loop.py`, the heart of this module. Read it against the six steps of section 1.

```python
from llmre.optim.adamw import AdamW
from llmre.optim.clip import clip_grad_norm_
from llmre.optim.schedule import cosine_warmup_lr

def train(model, get_batch, cfg):
    if cfg.seed is not None:
        torch.manual_seed(cfg.seed)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=cfg.max_lr, betas=cfg.betas,
                      eps=cfg.eps, weight_decay=cfg.weight_decay)
    history = {"step": [], "loss": [], "lr": [], "grad_norm": [], ...}

    model.train()
    for step in range(cfg.max_steps):
        # (0) learning rate for this step, written into the optimizer
        lr = cosine_warmup_lr(step, cfg.warmup_steps, cfg.max_steps,
                              cfg.max_lr, cfg.min_lr)
        optimizer.lr = lr

        # (1-4) forward + backward. (grad_accum_steps handled in 07.2; = 1 here)
        optimizer.zero_grad()
        step_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = get_batch("train")             # (b, T), (b, T)
            _, loss = model(x, y)                  # forward -> scalar loss
            step_loss += loss.item() / cfg.grad_accum_steps
            (loss / cfg.grad_accum_steps).backward()   # accumulate grads

        # (5) clip the global gradient norm (returns the pre-clip norm)
        grad_norm = clip_grad_norm_(params, cfg.grad_clip)

        # (6) one AdamW update
        optimizer.step()
        # ... logging / periodic eval ...
    return history
```

Line by line:

- `optimizer.lr = cosine_warmup_lr(...)` — set the step's learning rate *before* stepping. Warmup ramps it up so the first, noisiest steps do not lurch; cosine decay eases it down so late steps fine-tune. (03.3.)
- `optimizer.zero_grad()` — `.grad` tensors **accumulate** on every `backward()`. If we do not clear them, this step's gradient is added to last step's, which is wrong. (This "add, don't overwrite" behavior is exactly what we *exploit* in 07.2 for gradient accumulation.)
- `x, y = get_batch("train")` — a fresh random batch from the loader.
- `_, loss = model(x, y)` — the forward pass of [06.1](../module-06/lesson-01.md); we ignore the logits and keep the scalar loss.
- `loss.backward()` — backprop (Module 2) fills every `.grad`.
- `clip_grad_norm_(params, cfg.grad_clip)` — rescale the whole gradient so its global L2 norm is at most `grad_clip`; this is the guard-rail against a single huge-gradient batch. (03.3.)
- `optimizer.step()` — AdamW reads `.grad`, updates its moment estimates, and nudges every parameter. (03.2.)

<div class="callout pt"><p>Order matters. <code>zero_grad</code> must come <em>before</em> <code>backward</code> (or right after <code>step</code>), and <code>clip</code> must come <em>after</em> <code>backward</code> but <em>before</em> <code>step</code> — clipping reads the gradients backward just produced and must run before the optimizer consumes them. Get the order wrong and the model still "trains", just wrongly, which is the worst kind of bug: silent.</p></div>

## 6. Under the hood: what each step actually costs

- **Forward** allocates and holds the activations of every layer, because backward needs them. For a batch $(B, T)$ the biggest single tensor is the logits, $(B, T, V)$ — for GPT-2 that $V = 50257$ makes the logits dwarf everything inside the stack (06.1). This is why activation memory, not parameter memory, usually limits how big a batch you can fit; we account for it in [07.4](lesson-04.md).
- **Backward** does roughly twice the arithmetic of forward (it computes gradients w.r.t. both activations and weights), which is the source of the "6N FLOPs per token" rule of [07.4](lesson-04.md).
- **`optimizer.step()`** for AdamW touches every parameter three times (the parameter, and its two moment buffers `m` and `v`), so Adam's optimizer *state* is 2× the size of the model itself — a memory fact that shapes every large run (07.4).
- **`loss.item()`** forces a GPU→CPU sync (it reads a scalar back to Python). Doing it every step is fine for study; in a real run you log it every $N$ steps to avoid stalling the GPU.

## Exercise

You write a loop and it "trains" — the loss goes down — but much more slowly than a colleague's identical model. You spot that your loop calls `optimizer.zero_grad()` **after** `optimizer.step()` at the *end* of the iteration, but you also, out of habit, left a second `loss.backward()` call earlier for logging the gradient norm. What is happening to your gradients, and why does the model still improve (just wrongly)?

<details><summary>Hint</summary>
What does <code>.grad</code> do when <code>backward()</code> is called twice with no <code>zero_grad()</code> in between?
</details>

<details><summary>Stronger hint</summary>
Gradients <em>accumulate</em>. Two <code>backward()</code> calls on the same batch with no zeroing between them leave <code>2 × ∇L</code> in <code>.grad</code>.
</details>

<details><summary>Solution</summary>

`backward()` *adds* into `.grad`; it does not overwrite. Calling it twice with no `zero_grad()` between leaves twice the true gradient in every `.grad`, so `optimizer.step()` effectively uses a gradient of $2\nabla\mathcal{L}$. AdamW is partly scale-invariant (it divides by $\sqrt{v}$, and $v$ scales with the gradient squared), so it does not simply double the step — but the bias-corrected moments are now computed from a mis-scaled gradient, the effective learning-rate dynamics are off, and the run is subtly wrong. It still improves because the gradient *direction* is unchanged (both `backward()` calls point the same way), so steps are still mostly downhill — which is exactly why this class of bug is so dangerous: the loss curve looks plausible. Fix: one `backward()` per accumulation micro-step, and `zero_grad()` exactly once per optimizer step, before the accumulation loop.

</details>

## Common mistakes

- **Forgetting `zero_grad`.** Gradients from every past step pile up; the loss diverges or thrashes. (Symptom: NaNs or a loss that explodes after a few steps.)
- **Not checking step 0 against `ln(V)`.** A step-0 loss far from $\log V$ means a label/vocab/init bug — catch it in one line before wasting GPU-hours.
- **Setting the LR after `step()`.** Then step $k$ uses step $k-1$'s learning rate; the warmup is off by one and the very first step uses the peak (or floor) LR. Set the LR *before* stepping.
- **Calling `loss.item()` inside a hot loop on GPU without need.** It syncs the device every step and throttles throughput.

## Check yourself

<details><summary>Your fresh model's first reported loss is 6.9 and your vocabulary has 1000 tokens. Is that expected?</summary>

Yes — $\ln(1000) = 6.9078$. A random-init model predicts roughly uniformly, so step-0 loss ≈ $\ln V$. A first loss near 6.9 is exactly the healthy signal.

</details>

<details><summary>Why must <code>clip_grad_norm_</code> run after <code>backward()</code> but before <code>optimizer.step()</code>?</summary>

Clipping rescales the `.grad` tensors, which only exist after `backward()` computes them. And the optimizer consumes `.grad` inside `step()`, so the clip must happen first or it clips nothing. The sandwich is: `backward → clip → step`.

</details>

<details><summary>If a model cannot drive the loss near zero on a tiny repeated dataset, what does that tell you — and what should you NOT do next?</summary>

The training code (or model) is broken — a working setup can always overfit a handful of examples. Do **not** launch a large, expensive run "to see if it works at scale". Fix the tiny case first; scale only multiplies whatever bug you have.

</details>

<details><summary>In the loop, why does <code>step_loss</code> use <code>loss.item()</code> divided by <code>grad_accum_steps</code>, while the backward call uses <code>(loss / grad_accum_steps).backward()</code>?</summary>

Both keep the *reported* and *back-propagated* quantities on the same scale as a single full batch (details in [07.2](lesson-02.md)). `loss.item()` pulls the scalar to Python for logging; `(loss / G).backward()` scales the gradient so that summing $G$ micro-batches equals one full-batch gradient. With `grad_accum_steps = 1` (this lesson) both are just the plain loss.

</details>

<div class="hw">
<p><strong>Hardware track — the tiny overfit.</strong></p>
<p><strong>Minimum:</strong> any laptop CPU. <strong>Recommended:</strong> same — no GPU needed. <strong>Runtime:</strong> ~1–2 seconds for all 200 steps. <strong>Memory:</strong> a few MB (26k params). <strong>GPU-hours:</strong> 0. <strong>CPU-only:</strong> yes, entirely.</p>
<p><strong>What a real pretraining run needs (for contrast):</strong> GPT-2 small (124M params) on a few billion tokens is days on a single modern GPU, or hours on 8× data-parallel; GPT-2 has $V = 50257$ so step-0 loss ≈ 10.82, and a good run reaches ≈ 3.0. Frontier models are thousands of GPUs for weeks. The <em>loop is identical</em> — only the model size, token count, and number of workers change. We size those up in <a href="lesson-02.md">07.2</a> and <a href="lesson-04.md">07.4</a>.</p>
</div>

## Next

You can now write and run the canonical loop and watch a model overfit. But real runs need an **effective batch far larger than fits in memory** — hundreds of thousands of tokens per step. The trick is gradient accumulation, and it forces us to define micro-batch, global batch, and workers precisely.

Continue to [07.2 · Micro-batch, global batch, gradient accumulation](lesson-02.md).
