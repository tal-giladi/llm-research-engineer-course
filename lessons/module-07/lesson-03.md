# 07.3 · Checkpointing, resuming, seeds, reproducibility

<div class="prereq">
<p><strong>Prerequisites:</strong> the training loop and its state — model, optimizer, step — from <a href="lesson-01.md">07.1 · The training loop</a>; the AdamW moment buffers <code>m</code>, <code>v</code> and step counter <code>t</code> from <a href="../module-03/lesson-02.md">03.2 · AdamW</a>; the data loader's use of an RNG to sample batches from <a href="../module-04/lesson-03.md">04.3</a>.</p>
<p><strong>You will learn:</strong> exactly what must be saved to resume a run <em>identically</em> — model <code>state_dict</code>, optimizer state, step number, and the torch/NumPy/Python RNG states; how to seed every RNG for determinism and the real caveats (nondeterministic CUDA kernels, <code>torch.use_deterministic_algorithms</code>); and how to implement <code>save_checkpoint</code> / <code>load_checkpoint</code> so a resumed run continues bit-for-bit.</p>
<p><strong>Why this matters for ML:</strong> real pretraining runs last days to weeks on hardware that fails, gets pre-empted, or is deliberately paused. A checkpoint you cannot resume <em>exactly</em> from is a checkpoint you cannot trust — a resume that silently diverges wastes the compute since the last save and corrupts any experiment comparing "before" and "after". Reproducibility is also how you debug: if you cannot re-run and get the same numbers, you cannot bisect a regression.</p>
</div>

## 1. Intuition: a resume must restore the whole machine, not just the weights

Beginners save `model.state_dict()` and think they have a checkpoint. They have a checkpoint for *inference* — enough to generate text. They do **not** have enough to *resume training*. Training has more moving parts than the weights:

- The **optimizer** carries memory. AdamW's per-parameter moment estimates $m$ and $v$ (03.2) are an exponential average of past gradients; drop them and Adam restarts cold, taking a few hundred wrong steps while the moments re-warm — a visible bump in the loss curve right at every resume.
- The **step counter** tells the learning-rate schedule where it is. Lose it and the LR jumps back to warmup, or to the wrong point on the cosine curve.
- The **RNG states** decide which batches the data loader samples next and which dropout masks fire. Lose them and the resumed run sees a *different* data order — still valid training, but not the *same* run.

<div class="callout key"><p>To resume <em>exactly</em>, save four things: <strong>(1)</strong> model <code>state_dict</code>, <strong>(2)</strong> optimizer state (AdamW's <code>t</code>, <code>m</code>, <code>v</code>), <strong>(3)</strong> the step number, <strong>(4)</strong> the RNG states (torch, NumPy, Python). Plus your config, so you know what you were training.</p></div>

## 2. What each piece is, and why losing it hurts

### Model `state_dict`
An ordered dict mapping every parameter and buffer name (e.g. `transformer.h.0.mlp.c_fc.weight`) to its tensor. This is the obvious part. Because our GPT groups modules under `transformer` (06.1), the names are stable and even match OpenAI's released GPT-2 checkpoints.

### Optimizer state
For AdamW (03.2): the step counter `t` (drives bias correction $1 - \beta^t$) and the two buffers `m`, `v` per parameter. These *are* the optimizer's memory of the run so far. A resume that reinitializes them to zero throws away that memory:

- `m` and `v` restart at 0, so the first post-resume step has heavy bias correction again (as if `t = 1`), producing an over-large effective step.
- The loss curve shows a characteristic spike-and-recovery at every resume — a dead giveaway that optimizer state was not restored.

### Step number
The single integer the LR schedule reads. `cosine_warmup_lr(step, ...)` (03.3) is a pure function of `step`; restore the wrong step and the entire schedule is misaligned.

### RNG states
Three separate generators can drive a run: PyTorch's (`torch.get_rng_state()`, used by dropout and `torch.randint`), NumPy's (`np.random`), and Python's (`random`). Whichever your data loader and model use must be restored, or the resumed run diverges from the original the moment it draws its next batch or dropout mask.

## 3. Numerical example: the resume that must match

The cleanest test of "did I save enough?" is: run $N$ steps straight through, versus run $N-1$ steps, checkpoint, reload into a *fresh* model and optimizer, and take the last step. If the checkpoint is complete, the final parameters are **identical**. If you forgot the optimizer state or the RNG, they differ.

This is exactly `test_resumed_optimizer_state_is_restored` in `code/tests/test_training.py`:

```text
run A (continuous):  step1 -> step2 -> step3            -> param P_A
run B (resumed):     step1 -> step2 -> [save/reload] -> step3 -> param P_B
assert P_A == P_B    (to 1e-6)
```

It passes because `save_checkpoint`/`load_checkpoint` restore AdamW's `m`, `v`, `t` (so step 3 uses the correct moments and bias correction) and the RNG (so step 3 draws the same batch/dropout). Delete the optimizer restore and `P_B` drifts from `P_A` immediately — the manufactured "resume diverges" bug of the debugging exercise below.

## 4. Tensor shapes / what is on disk

A checkpoint is a plain Python dict handed to `torch.save`:

```text
ckpt = {
  "model":     OrderedDict{name -> tensor}   # every param/buffer, same shapes as the model
  "optimizer": {"t": int,
                "m": [tensor, ...],           # one per parameter, shape = that parameter
                "v": [tensor, ...],           # one per parameter, shape = that parameter
                "lr": float, "betas": (f,f), "eps": float, "weight_decay": float}
  "step":      int
  "rng":       {"torch": ByteTensor, "numpy": tuple, "python": tuple}
  "extra":     {...}                          # your config, best loss, etc.
}
```

The moment buffers are saved as **CPU clones** so the file is device-independent — a checkpoint written on GPU reloads on a CPU-only laptop and vice versa. On load, each buffer is moved back onto the device and dtype of the matching live parameter.

## 5. From-scratch PyTorch: save and load

Because our `AdamW` is a plain object (not a `torch.optim.Optimizer` with its own `state_dict`), we serialize its fields directly. From `llmre/training/checkpoint.py`:

```python
import random
import numpy as np
import torch

def save_checkpoint(path, model, optimizer, step, extra=None):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": {
            "t": optimizer.t,
            "m": [b.detach().cpu().clone() for b in optimizer.m],   # CPU clones
            "v": [b.detach().cpu().clone() for b in optimizer.v],
            "lr": optimizer.lr, "betas": (optimizer.beta1, optimizer.beta2),
            "eps": optimizer.eps, "weight_decay": optimizer.weight_decay,
        },
        "step": step,
        "rng": {
            "torch":  torch.get_rng_state(),
            "numpy":  np.random.get_state(),
            "python": random.getstate(),
        },
        "extra": extra or {},
    }
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer=None, restore_rng=True, map_location=None):
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None:
        s = ckpt["optimizer"]
        optimizer.t = s["t"]; optimizer.lr = s["lr"]
        optimizer.beta1, optimizer.beta2 = s["betas"]
        optimizer.eps = s["eps"]; optimizer.weight_decay = s["weight_decay"]
        # move each buffer back onto its live parameter's device/dtype
        optimizer.m = [b.to(p.device, p.dtype) for b, p in zip(s["m"], optimizer.params)]
        optimizer.v = [b.to(p.device, p.dtype) for b, p in zip(s["v"], optimizer.params)]
    if restore_rng:
        r = ckpt["rng"]
        torch.set_rng_state(r["torch"])
        np.random.set_state(r["numpy"])
        random.setstate(r["python"])
    return ckpt["step"], ckpt["extra"]
```

Usage in a loop:

```python
# ... every eval_interval steps:
save_checkpoint("ckpt.pt", model, optimizer, step=step, extra={"cfg": cfg.__dict__})

# to resume:
model = GPT(cfg); opt = AdamW(list(model.parameters()), lr=cfg.max_lr)
start_step, extra = load_checkpoint("ckpt.pt", model, opt)
# continue the loop from start_step
```

<div class="callout pt"><p><code>torch.load(..., weights_only=False)</code> is needed here because the checkpoint holds more than tensors (the NumPy and Python RNG states are arbitrary Python objects). Only ever load checkpoints you trust — <code>weights_only=False</code> can execute arbitrary code during unpickling, so it is not safe for files from unknown sources. For inference-only loads of pure-tensor checkpoints, prefer <code>weights_only=True</code>.</p></div>

## 6. Determinism: seeding, and why it is not enough

To make a run reproducible, seed **every** RNG at the start:

```python
import random, numpy as np, torch
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)            # seeds CPU and (if present) all CUDA devices
```

Seeding makes the *stream* of random numbers deterministic: same weights at init, same batch order, same dropout masks. Our `train(..., cfg)` calls `torch.manual_seed(cfg.seed)` for exactly this reason.

But **seeding alone does not guarantee bit-identical results on GPU.** Two real caveats:

- **Nondeterministic CUDA kernels.** Some GPU operations (certain reductions, `scatter`/`gather`, some backward kernels) sum floating-point numbers in a nondeterministic order for speed. Floating-point addition is not associative, so $(a+b)+c \ne a+(b+c)$ in the last bits — two runs with the same seed can differ by ~$10^{-7}$, which compounds over thousands of steps. `torch.use_deterministic_algorithms(True)` forces PyTorch to pick deterministic kernels (or raise if none exists), at some speed cost; you also often need to set `CUBLAS_WORKSPACE_CONFIG` and `torch.backends.cudnn.deterministic = True`.
- **Hardware / library / parallelism changes.** The same seed on a different GPU model, CUDA version, or number of data-parallel workers gives different results — the reduction order and kernel selection change. Determinism is only promised *for a fixed environment*.

<div class="callout warn"><p>On CPU (this course's default), the same seed gives bit-identical results with no extra flags — which is why the checkpoint tests assert exact equality. On GPU, expect determinism only after <code>torch.use_deterministic_algorithms(True)</code> <em>and</em> a fixed hardware/library/worker configuration. "Same seed" is necessary, not sufficient.</p></div>

## Debugging exercise

A colleague reports: "Every time our job resumes from a checkpoint, the loss jumps up by ~0.3 for a few hundred steps, then recovers." Their `save_checkpoint` writes only `model.state_dict()` and `step`; their `load_checkpoint` does `model.load_state_dict(...)` and creates a **fresh** `AdamW(model.parameters(), lr=...)`. What is wrong, and what is the fix?

<details><summary>Hint</summary>
What state does a fresh <code>AdamW</code> start with, and what does that do to the first few steps after a resume?
</details>

<details><summary>Stronger hint</summary>
AdamW's moment buffers $m$, $v$ and step counter $t$ are its memory of past gradients. A fresh optimizer has $m = v = 0$ and $t = 0$.
</details>

<details><summary>Solution</summary>

The optimizer state is not saved, so every resume creates an AdamW with $m = v = 0$ and $t = 0$. Two things then go wrong for the first steps after each resume:

1. **Cold moments.** $v$ starts at 0, so the denominator $\sqrt{\hat v} + \epsilon$ is tiny for the first steps, producing over-large parameter updates until the second-moment average re-warms.
2. **Bias correction resets.** With $t$ back to 1, the bias-correction factors $1/(1-\beta_1^t)$ and $1/(1-\beta_2^t)$ are large again, further inflating early steps.

Together they knock the weights off the trajectory the continuous run was on — hence the loss spike, followed by recovery as the moments rebuild. (A second, subtler version of this bug: not restoring the data-loader RNG, so the resumed run re-samples batches it already trained on or skips others — the loss does not spike as sharply but the run is no longer the same run, which quietly ruins any before/after comparison.)

**Fix:** save the optimizer state (`t`, `m`, `v`) and the RNG states, and restore them on load — exactly what `save_checkpoint`/`load_checkpoint` above do. After the fix, the loss curve is continuous across resumes, and `test_resumed_optimizer_state_is_restored` passes to $10^{-6}$.

</details>

## Common mistakes

- **Saving only the weights.** Fine for inference, silently wrong for resuming — the optimizer-state spike above.
- **Re-tying weights after load, or forgetting to.** If you rebuild the head after `load_state_dict`, redo the weight tie (06.1) or you split the tied table.
- **Assuming "same seed" ⇒ "same result" on GPU.** Only true with deterministic algorithms forced and a fixed environment.
- **Checkpointing too rarely.** A crash loses everything since the last save; checkpoint often enough that lost work is cheap relative to the save cost.

## Check yourself

<details><summary>You resume a run and the loss spikes for ~200 steps every time. Which piece of state did you most likely forget?</summary>

The optimizer state (AdamW's `m`, `v`, `t`). Cold moment buffers and a reset bias-correction counter cause over-large steps right after the resume, seen as a loss spike that recovers as the moments re-warm.

</details>

<details><summary>Why save the RNG states, given the weights and optimizer are already saved?</summary>

The data loader and dropout draw from the RNGs. Without restoring them, the resumed run samples a *different* batch order and different dropout masks — valid training, but not the *same* run, which breaks exact reproducibility and any experiment comparing the resumed run to the original.

</details>

<details><summary>You set the same seed on two identical GPUs and get results differing in the 7th decimal that grow over training. Bug or expected?</summary>

Expected, unless you forced deterministic algorithms. Some CUDA kernels reduce in a nondeterministic order, and floating-point addition is not associative, so tiny differences appear and compound. `torch.use_deterministic_algorithms(True)` (plus cuDNN/cuBLAS settings) removes them at a speed cost.

</details>

<details><summary>Why are the optimizer moment buffers saved as CPU clones rather than in place?</summary>

So the checkpoint is device-independent: a file written on GPU can be reloaded on a CPU-only machine (and vice versa). `load_checkpoint` moves each buffer back onto the device and dtype of the matching live parameter.

</details>

<div class="hw">
<p><strong>Hardware track — checkpoint round-trip.</strong></p>
<p><strong>Minimum / recommended:</strong> any CPU. <strong>Runtime:</strong> milliseconds for the tiny model; a full GPT-2-small checkpoint is ~1.5 GB on disk (weights + optimizer state in fp32) and takes a second or two to write. <strong>Disk:</strong> budget for many checkpoints — optimizer state doubles the file size over weights alone (07.4). <strong>CPU-only:</strong> yes; on CPU the round-trip is bit-exact with no extra flags, which is why the tests assert exact equality.</p>
</div>

## Next

You can now train, size batches, and resume exactly. The last piece of infrastructure is knowing whether a run is *fast* — how many FLOPs it does, how many tokens/second it achieves, what fraction of the hardware it uses (MFU), and how much memory the parameters, gradients, optimizer state, and activations consume.

Continue to [07.4 · Throughput, FLOPs, MFU, memory accounting](lesson-04.md).
