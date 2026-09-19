# 14.3 · LoRA & QLoRA from scratch

<div class="prereq">
<p><strong>Prerequisites:</strong> loss masking from <a href="lesson-02.md">14.2</a>; <code>nn.Linear</code> and the weight matrix shape <code>(out, in)</code> from <a href="../module-05/lesson-04.md">05.4 · Linear layers</a>; the AdamW optimizer state (two moment tensors per parameter) from <a href="../module-03/lesson-03.md">Module 3</a>.</p>
<p><strong>You will learn:</strong> why full fine-tuning is expensive (optimizer state for every weight); what <strong>LoRA</strong> does — freeze the pretrained $W$ and train a low-rank update $\Delta W = \frac{\alpha}{r} B A$; why low-rank works and what the $\alpha$ scaling is for; the parameter/memory savings worked in numbers; how to implement <code>LoRALinear</code> from scratch and <code>merge()</code> it back into a plain layer; and what <strong>QLoRA</strong> adds (a 4-bit frozen base).</p>
<p><strong>Why this matters for ML:</strong> LoRA is how fine-tuning became something you can do on one GPU instead of a cluster. It cuts trainable parameters (and the optimizer memory that dominates fine-tuning) by one to two orders of magnitude, lets you keep dozens of task-specific adapters over one shared base, and — via <code>merge()</code> — costs nothing extra at inference. Every practical fine-tuning workflow today (PEFT, QLoRA) is built on it.</p>
</div>

## 1. Intuition: why full fine-tuning hurts

Full fine-tuning updates **every** weight in the model. The parameters themselves are only part of the cost. The AdamW optimizer (Module 3) keeps *two* extra tensors per trainable parameter — the first-moment (mean) and second-moment (variance) estimates — plus you need the gradients. So for each trainable parameter in fp32 you are carrying roughly: the weight, its gradient, and two optimizer moments — about **4 numbers**, not one.

For GPT-2 small (124M params) that is ~124M × 2 optimizer moments × 4 bytes ≈ 1 GB *just for the optimizer state*, on top of weights, gradients, and activations. For any real model this optimizer state is the thing that blows the memory budget. And if you fine-tune the same base model for ten different tasks, you store ten full copies of 124M weights.

LoRA attacks exactly this: **freeze the pretrained weights** (no gradient, no optimizer state for them) and train only a tiny add-on.

## 2. Mathematics: the low-rank update

Take one linear layer with frozen weight $W \in \mathbb{R}^{\text{out} \times \text{in}}$. Full fine-tuning would learn an update $\Delta W$ of the same shape and use $W + \Delta W$. LoRA instead *factorizes* that update as a product of two thin matrices:

$$
\Delta W = \frac{\alpha}{r} \, B A, \qquad A \in \mathbb{R}^{r \times \text{in}}, \quad B \in \mathbb{R}^{\text{out} \times r},
$$

where the **rank** $r$ is small ($r \ll \min(\text{in}, \text{out})$, e.g. $r = 8$). The forward pass becomes

$$
\mathbf{y} = W\mathbf{x} + \frac{\alpha}{r} B (A \mathbf{x}).
$$

Every symbol: $W$ the frozen pretrained weight (out×in); $A$ maps the input down to an $r$-dimensional bottleneck (r×in); $B$ maps that bottleneck up to the output (out×r); $r$ the rank (bottleneck width); $\alpha$ a scaling constant. Only $A$ and $B$ are trainable. The product $BA$ has shape out×in — the same as $\Delta W$ — but it is **rank-$r$**, because it is squeezed through the $r$-wide bottleneck.

Note the compute order in $B(A\mathbf{x})$: apply $A$ first (an $r\times\text{in}$ times a vector → an $r$-vector), then $B$. We never materialize the out×in matrix $BA$ during training — that would defeat the purpose.

<div class="callout key"><p><strong>Why low-rank works:</strong> the weight change a single downstream task needs is empirically close to low-rank — the "direction" fine-tuning moves in lives in a small subspace, not the full out×in space. Hu et al. (2021) measured this: adapting with a rank as small as 1–8 recovers most of the quality of full fine-tuning. So we do not need a full-rank $\Delta W$; a rank-$r$ factorization captures what the task actually requires.</p></div>

### The $\alpha$ scaling and the zero init

Two design choices make LoRA train stably:

- **$B$ is initialised to zero, $A$ to small random noise.** Then at step 0, $\Delta W = BA = 0$, so the wrapped layer outputs *exactly* $W\mathbf{x}$ — identical to the pretrained model. Training therefore starts from the pretrained model, not a randomly perturbed one, and the adapter learns a *correction* from there.
- **The $\alpha/r$ factor** decouples the effective update size from the rank. If you change $r$, the raw magnitude of $BA$ changes with it; dividing by $r$ (and multiplying by a fixed $\alpha$) keeps the effective scale roughly constant, so you can retune $r$ without re-tuning the learning rate. In practice people set $\alpha$ to a small multiple of $r$ (e.g. $\alpha = 16, r = 8$).

## 3. Numerical example (worked by hand, verified in Python)

Take a tiny frozen layer with $\text{in} = 3$, $\text{out} = 2$, no bias, rank $r = 1$, $\alpha = 2$ (so $\alpha/r = 2$):

$$
W = \begin{bmatrix} 1 & 0 & 2 \\ 0 & 1 & 1 \end{bmatrix}, \quad
A = \begin{bmatrix} 1 & 0 & 1 \end{bmatrix}\ (1\times 3), \quad
B = \begin{bmatrix} 1 \\ 2 \end{bmatrix}\ (2\times 1).
$$

Take input $\mathbf{x} = [1, 1, 1]$.

**Frozen path:** $W\mathbf{x} = [1{+}0{+}2,\ 0{+}1{+}1] = [3, 2]$.

**LoRA path, bottleneck first:** $A\mathbf{x} = 1\cdot1 + 0\cdot1 + 1\cdot1 = 2$ (a single number, since $r = 1$). Then $B(A\mathbf{x}) = [1, 2]\cdot 2 = [2, 4]$. Scale by $\alpha/r = 2$: $[4, 8]$.

**Sum:** $\mathbf{y} = [3, 2] + [4, 8] = [7, 10]$.

Now the equivalent **merged** weight $W' = W + \frac{\alpha}{r} BA$. Here $BA = \begin{bmatrix}1\\2\end{bmatrix}\begin{bmatrix}1&0&1\end{bmatrix} = \begin{bmatrix}1&0&1\\2&0&2\end{bmatrix}$, times $2$ is $\begin{bmatrix}2&0&2\\4&0&4\end{bmatrix}$, so

$$
W' = \begin{bmatrix} 3 & 0 & 4 \\ 4 & 1 & 5 \end{bmatrix}, \qquad W'\mathbf{x} = [3{+}0{+}4,\ 4{+}1{+}5] = [7, 10].
$$

Identical to the LoRA forward — that is what `merge()` guarantees. Verified in Python:

```python
import torch, torch.nn as nn
from llmre.sft.lora import LoRALinear

base = nn.Linear(3, 2, bias=False)
with torch.no_grad(): base.weight.copy_(torch.tensor([[1.,0.,2.],[0.,1.,1.]]))
lo = LoRALinear(base, r=1, alpha=2.0)
with torch.no_grad():
    lo.A.copy_(torch.tensor([[1., 0., 1.]]))   # (r, in) = (1, 3)
    lo.B.copy_(torch.tensor([[1.], [2.]]))     # (out, r) = (2, 1)

x = torch.tensor([1., 1., 1.])
print(lo(x))              # tensor([ 7., 10.])
print(lo.merge()(x))      # tensor([ 7., 10.])  <- identical
```

## The savings, in numbers

Take a realistic layer: one of GPT-2's projections, $\text{in} = \text{out} = 768$, with $r = 8$.

- **Full fine-tuning** trains $768 \times 768 = 589{,}824$ parameters for this layer.
- **LoRA** trains $A$ ($8 \times 768$) plus $B$ ($768 \times 8$) $= 12{,}288$ parameters.

$$
\frac{589{,}824}{12{,}288} = 48.0.
$$

**~48× fewer trainable parameters** for this layer — and since optimizer state scales with *trainable* parameters, the AdamW moment memory for this layer drops from ~4.7 MB (fp32) to ~0.1 MB. Multiply across every adapted layer and the fine-tuning memory footprint collapses. And to store a new task you save only the adapters (a few MB), not a fresh copy of the base model.

<div class="hw">
<p><strong>Hardware note — fine-tuning memory.</strong> Full fine-tuning of GPT-2 small (124M) in fp32 needs roughly: weights ~0.5 GB + gradients ~0.5 GB + AdamW state ~1 GB + activations — call it several GB, feasible on a 12–16 GB GPU but tight. LoRA (r=8 on the attention/MLP projections) trains &lt;1% of the parameters, so gradient + optimizer memory for the trainable set drops to tens of MB; the dominant cost becomes the frozen weights and activations. <strong>QLoRA</strong> pushes further by storing the frozen base in 4-bit (see below), bringing 7B-scale fine-tuning onto a single 24 GB consumer GPU. CPU-only works for the tiny toy layers in this lesson and the tests; real SFT wants a GPU.</p>
</div>

## 4. Tensor shapes / dtype / device

For `LoRALinear` wrapping `nn.Linear(in, out)`:

- `self.base.weight`: `(out, in)`, frozen (`requires_grad = False`).
- `self.A`: `(r, in)`, trainable, same dtype/device as the base weight; init Kaiming-uniform.
- `self.B`: `(out, r)`, trainable; init **zeros**.
- forward input `x`: `(..., in)` → `F.linear(x, A)` gives `(..., r)` → `F.linear(., B)` gives `(..., out)`; add to `base(x)` which is `(..., out)`.
- `merge()` returns a fresh `nn.Linear(in, out)` with weight `(out, in) = base.weight + (α/r)·B@A`.

## 5. From-scratch PyTorch

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float = 1.0):
        super().__init__()
        self.base = base
        self.r, self.alpha, self.scaling = r, float(alpha), alpha / r
        # Freeze the pretrained weight (and bias): no grad, no optimizer state.
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        w = base.weight
        self.A = nn.Parameter(torch.empty(r, base.in_features,  dtype=w.dtype, device=w.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, r, dtype=w.dtype, device=w.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # A random, B stays zero

    def forward(self, x):
        return self.base(x) + self.scaling * F.linear(F.linear(x, self.A), self.B)

    @torch.no_grad()
    def merge(self) -> nn.Linear:                 # fold BA into W for zero-overhead inference
        merged = nn.Linear(self.in_features, self.out_features,
                           bias=self.base.bias is not None,
                           device=self.base.weight.device, dtype=self.base.weight.dtype)
        merged.weight.copy_(self.base.weight + self.scaling * (self.B @ self.A))
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged
```

Two things to notice. `F.linear(F.linear(x, A), B)` is the "down then up" projection — it never builds the out×in matrix during training, so both memory and FLOPs stay proportional to $r$. And `merge()` produces a *plain* `nn.Linear`: once training is done you fold the adapter into the weight and ship a model that is byte-for-byte a normal GPT — LoRA adds **zero** inference cost after merging.

A helper freezes everything except the adapters when you wrap several layers in a full model:

```python
def mark_only_lora_trainable(model):
    for p in model.parameters():
        p.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.A.requires_grad_(True); m.B.requires_grad_(True)
```

Then hand only the trainable parameters to the optimizer: `torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=...)`.

## QLoRA: quantize the frozen base

The frozen base weights still sit in memory — for a 7B model that is ~14 GB in fp16, too big for a consumer GPU. **QLoRA** (Dettmers et al. 2023) observes that since the base is *frozen*, we never need its full-precision values for an optimizer update — only for the forward/backward *matmul*. So it stores the frozen base in **4-bit** (a special "NF4" format matched to the roughly-normal distribution of weights), dequantizes each weight block to compute-precision on the fly during the matmul, and keeps the small LoRA adapters (A, B) in higher precision, where all the gradients and optimizer state live.

The result: the memory-dominant frozen base shrinks ~4×, while the trainable part is already tiny thanks to LoRA. QLoRA is what makes fine-tuning a 65B model on a single 48 GB GPU — or a 7B on a 24 GB card — possible. The concept is exactly LoRA + a 4-bit frozen base; the engineering (NF4, double quantization, paged optimizers) is in the paper.

<div class="callout paper"><p><strong>LoRA</strong> — Hu et al. 2021, "LoRA: Low-Rank Adaptation of Large Language Models", <a href="https://arxiv.org/abs/2106.09685">arxiv 2106.09685</a>. (Not one of the 30 core papers; cited inline here.) <strong>QLoRA</strong> — Dettmers et al. 2023, <a href="https://arxiv.org/abs/2305.14314">arxiv 2305.14314</a>. Read the LoRA paper's rank-ablation section (how small $r$ can be) and its "which weights to adapt" section (attention $q,v$ projections give the most per parameter).</p></div>

## 6. Under the hood: the production library

`LoRALinear` above is the mechanism; **PEFT** (`peft`, from HuggingFace) is the production version. You call `get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["c_attn"]))` and it wraps the matching linears, marks only the adapters trainable, and handles saving/loading just the adapter weights. Under the hood it is doing precisely what we implemented: a frozen base linear, a rank-$r$ $A$/$B$ pair with an $\alpha/r$ scale, zero-init on $B$, and a merge step for deployment. **QLoRA** in PEFT is the same call with a 4-bit-quantized base model (via `bitsandbytes`).

## Common mistakes

- **Initialising $B$ nonzero.** Then $\Delta W \ne 0$ at step 0 and the wrapped layer no longer matches the pretrained model — you start from a perturbed model and often lose quality. Zero-init $B$.
- **Forgetting to freeze the base.** If the base weight keeps `requires_grad = True`, the optimizer trains it too and you have paid for LoRA while still doing (worse) full fine-tuning. Verify with `sum(p.requires_grad for p in ...)`.
- **Handing all parameters to the optimizer.** Even with the base frozen, pass only `p for p in model.parameters() if p.requires_grad`; otherwise AdamW allocates moment tensors for frozen params.
- **Materializing `B @ A` in the forward.** Computing the out×in product each step throws away the compute savings. Always go input → $A$ → $B$.

## Debugging exercise

This adapter is supposed to be a no-op at initialization but is not — the wrapped layer's output differs from the base layer's on a fresh `LoRALinear`. Find the bug.

```python
self.A = nn.Parameter(torch.zeros(r, in_features))
nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
self.B = nn.Parameter(torch.empty(out_features, r))
nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
```

<details><summary>Answer</summary>

`B` is initialised with random Kaiming noise instead of zeros. LoRA requires $B = 0$ at init so that $\Delta W = BA = 0$ and the fresh adapter reproduces the base layer exactly. Here $BA \ne 0$, so the output is perturbed from step 0. Fix: `self.B = nn.Parameter(torch.zeros(out_features, r))` and leave it zero (do not call `kaiming_uniform_` on it). ($A$ being random is fine and intended.)

</details>

## Check yourself

<details><summary>Why does freezing the base weights save so much more memory than the frozen weights themselves?</summary>

Because optimizer state and gradients scale with the number of *trainable* parameters. AdamW keeps two moment tensors per trainable param, plus gradients — roughly 3 extra tensors. Freezing the base removes all of that for the base, leaving optimizer/gradient memory proportional only to the tiny $A$/$B$. The frozen weights still occupy memory, but they carry no gradient or optimizer state.

</details>

<details><summary>For a 1024×1024 layer with r = 16, how many trainable parameters does LoRA use, and what is the reduction factor vs full?</summary>

$A$ is $16\times1024$ and $B$ is $1024\times16$, so $2 \times 16 \times 1024 = 32{,}768$ trainable params. Full is $1024 \times 1024 = 1{,}048{,}576$. Reduction factor $= 1{,}048{,}576 / 32{,}768 = 32\times$.

</details>

<details><summary>What is the purpose of the α/r scaling factor?</summary>

It decouples the effective magnitude of the update from the rank. Without it, changing $r$ changes the scale of $BA$ and forces you to re-tune the learning rate. Dividing by $r$ (times a fixed $\alpha$) keeps the effective update size roughly constant across ranks.

</details>

<details><summary>After training, why can you <code>merge()</code> the adapter and pay zero inference cost?</summary>

Because $W\mathbf{x} + \frac{\alpha}{r}B(A\mathbf{x}) = (W + \frac{\alpha}{r}BA)\mathbf{x}$. Folding $\frac{\alpha}{r}BA$ into $W$ once yields a single weight $W'$ of the original shape, so inference is a plain `nn.Linear` matmul — no extra $A$/$B$ multiplies. The adapter's cost was only during training.

</details>

<details><summary>What is the one core idea QLoRA adds to LoRA?</summary>

Store the *frozen* base weights in 4-bit precision (NF4), dequantizing on the fly only for the forward/backward matmul, while keeping the small LoRA adapters (and all gradients/optimizer state) in higher precision. Since the base is frozen, its full-precision values are never needed for an update, so the ~4× memory saving on the base is essentially free — enabling single-GPU fine-tuning of large models.

</details>

## Next

You can now fine-tune a pretrained model to follow instructions — format the data (14.1), mask the loss to the response (14.2), and train cheaply with LoRA (14.3). But SFT only imitates demonstrations; it cannot learn from *preferences* ("response A is better than B"). Module 15 adds that: reward modeling, RLHF/PPO, and DPO.

Continue to [15.1 · Bradley-Terry & the reward model](../module-15/lesson-01.md).
