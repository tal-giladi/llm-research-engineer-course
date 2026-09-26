# PyTorch & systems cheat sheet

A single-page lookup of the PyTorch idioms and systems facts used across the course. Snippets are
copy-pasteable and use the course conventions: `B` batch, `T` sequence length (context), `C =
n_embd` model width, `V` vocab size, `nh` heads, `hd = C // nh` head dimension. Follow the linked
lesson for the derivation and a worked example; this page is a reminder, not a tutorial.

Import assumptions for every snippet below:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

---

## Tensor creation, shape, dtype, device

```python
x = torch.zeros(2, 3)                 # (2,3) float32 on cpu
x = torch.ones(2, 3)
x = torch.arange(6)                   # tensor([0,1,2,3,4,5]) int64
x = torch.randn(4, 8)                 # standard normal, float32
x = torch.randint(0, 50257, (2, 16))  # (2,16) int64 token ids in [0, V)
x = torch.tensor([[1., 2.], [3., 4.]])

x.shape        # torch.Size([2, 2])  -> x.shape[0], x.size(0)
x.ndim         # number of dims
x.numel()      # total element count
x.dtype        # torch.float32
x.device       # device('cpu')

# dtype / device moves (return NEW tensors; assign the result)
x = x.to(torch.bfloat16)
x = x.to("cuda")                      # or .cuda(); .cpu()
x = x.to(device="cuda", dtype=torch.bfloat16)
x = x.float()                         # -> float32 ; .long() -> int64

# pick the device once, reuse it
device = "cuda" if torch.cuda.is_available() else "cpu"
```

Common dtypes: `float32` (default, 4 bytes), `bfloat16`/`float16` (2 bytes, training compute),
`int64`/`long` (token ids, indices), `bool` (masks). Token id tensors MUST be `long`.

<div class="callout warn"><p>Every op needs its tensors on the <em>same device</em> and usually
the same dtype. "Expected all tensors to be on the same device" almost always means a buffer
(mask, positions) was never moved with the model. Register it with <code>register_buffer</code> or
create it with <code>device=x.device</code>.</p></div>

## Reshape / view / permute / transpose / contiguous

```python
x = torch.arange(24)
x.view(2, 3, 4)            # (2,3,4) — NO copy; requires contiguous memory
x.reshape(2, 3, 4)         # like view, but copies if it must
x.view(2, -1)             # (2,12) — one dim can be -1 (inferred)

y = torch.randn(2, 3, 4)
y.transpose(1, 2).shape    # (2,4,3) — swap two dims (a view)
y.permute(2, 0, 1).shape   # (4,2,3) — arbitrary reorder (a view)
y.flatten(1).shape         # (2,12) — flatten dims 1.. onward
y.unsqueeze(0).shape       # (1,2,3,4) — add a size-1 dim
y.squeeze(0).shape         # drop size-1 dims

# The multi-head split/merge you will write many times:
# (B,T,C) -> (B,nh,T,hd)
B, T, C, nh = 2, 5, 8, 2
q = torch.randn(B, T, C)
q = q.view(B, T, nh, C // nh).transpose(1, 2)   # (B,nh,T,hd)
# ...attention... then merge back:
out = q.transpose(1, 2).contiguous().view(B, T, C)
```

`view`/`transpose`/`permute` return a *view* sharing storage — cheap, but the result may be
non-contiguous. `.contiguous()` forces a fresh contiguous copy; call it before a `view` that would
otherwise raise "view size is not compatible with input tensor's size and stride".

## Broadcasting rules

Align shapes from the *right*. Two dims are compatible if equal or one of them is `1`; a `1` is
stretched (no copy) to match. Missing leading dims are treated as `1`.

```python
a = torch.randn(B, T, C)     # (2,5,8)
b = torch.randn(C)           # (8,)      -> broadcasts to (2,5,8)
a + b                        # ok

m = torch.randn(T, 1)        # (5,1)
n = torch.randn(1, C)        # (1,8)
m + n                        # (5,8)

# add a bias per position vs per channel — watch which axis you're on:
bias_c = torch.randn(1, 1, C)   # per-channel
bias_t = torch.randn(1, T, 1)   # per-position
```

Shapes `(T,)` and `(T,1)` broadcast very differently — a frequent silent bug.

## Matmul & einsum

```python
A = torch.randn(B, T, C)
W = torch.randn(C, 4 * C)
A @ W                         # (B,T,4C) — batched matmul over last two dims
torch.matmul(A, W)            # same

# attention scores: (B,nh,T,hd) @ (B,nh,hd,T) -> (B,nh,T,T)
scores = q @ k.transpose(-2, -1)

# einsum names every axis — self-documenting, no manual transposes
scores = torch.einsum("bhqd,bhkd->bhqk", q, k)      # q·k over hd
ctx    = torch.einsum("bhqk,bhkd->bhqd", attn, v)   # weighted sum of v
y      = torch.einsum("btc,cf->btf", A, W)          # a linear layer
```

`@`/`matmul` batch over all leading dims and contract the last dim of the left with the
second-to-last of the right. `einsum` is the same math with explicit index letters; repeated
letters are summed, letters on the right are kept.

## Autograd

```python
w = torch.randn(3, requires_grad=True)   # leaf tensor that will collect .grad
x = torch.randn(3)
loss = (w * x).sum()
loss.backward()          # fills w.grad = d loss / d w
w.grad                   # (3,) — ACCUMULATES across backward calls

with torch.no_grad():    # no graph built — inference / manual param updates
    w -= 0.1 * w.grad

y = w.detach()           # same data, cut from the graph (no grad flows through y)
w.grad.zero_()           # clear before the next step (or optimizer.zero_grad())
w.requires_grad_(False)  # freeze a parameter
```

Key facts: only float tensors can require grad; `.grad` *accumulates*, so you must zero it each
step; `backward()` needs a scalar (call `.sum()`/`.mean()` first, or pass a `gradient=`);
`no_grad` disables graph construction (saves memory) while `detach` cuts one tensor out.

<div class="callout warn"><p>Forgetting <code>zero_grad()</code> silently sums this step's
gradient onto last step's — training destabilizes for no obvious reason. Zero every step.</p></div>

## The canonical training loop

```python
model = model.to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

model.train()
for step in range(max_steps):
    x, y = get_batch("train")            # x,(B,T) long ; y,(B,T) long
    x, y = x.to(device), y.to(device)

    logits, loss = model(x, y)           # forward + loss

    opt.zero_grad(set_to_none=True)      # 1. clear grads
    loss.backward()                      # 2. backward: fill .grad
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # 3. clip
    opt.step()                           # 4. update params
```

Order is fixed: `zero_grad -> backward -> (clip) -> step`. `set_to_none=True` is slightly faster
and the modern default. Call `model.eval()` + `torch.no_grad()` for validation. See
[Module 0](lessons/module-00/lesson-04.md) and [Module 7](lessons/module-07/lesson-01.md).

## nn.Module, Linear, Embedding, LayerNorm

```python
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()                        # ALWAYS first
        self.fc   = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
    def forward(self, x):                          # x (B,T,C) -> (B,T,C)
        return self.proj(F.gelu(self.fc(x)))

lin = nn.Linear(C, V, bias=False)   # weight (V,C); y = x @ W.T (+ b); (B,T,C)->(B,T,V)
emb = nn.Embedding(V, C)            # weight (V,C); LOOKUP by id; (B,T) long -> (B,T,C)
ln  = nn.LayerNorm(C)              # normalizes over the LAST dim (C); learns gamma,beta

model.parameters()                 # iterator of learnable tensors
sum(p.numel() for p in model.parameters())        # total param count
model.state_dict()                 # {name: tensor} for saving
```

`nn.Linear` stores its weight transposed, shape `(out, in)`. `nn.Embedding` is a *lookup table*,
not a matmul: it indexes rows by integer id. `LayerNorm(C)` normalizes across the width dimension,
independently per (batch, position). See [Module 5](lessons/module-05/lesson-04.md).

## F.cross_entropy and the (B,T,C) convention

`cross_entropy` wants logits shaped `(N, V)` and integer targets `(N,)` — so flatten the batch and
time axes together:

```python
# logits (B,T,V) float, targets (B,T) long
B, T, V = logits.shape
loss = F.cross_entropy(
    logits.view(B * T, V),     # (B*T, V)
    targets.view(B * T),       # (B*T,)
    ignore_index=-100,         # positions == -100 contribute NO loss (see masking)
)
```

`cross_entropy` = `log_softmax` + `nll_loss` fused (numerically stable — do NOT softmax first).
The `(B, T, C)` convention runs through the whole course: batch, then time/position, then channels/
width. Attention mixes along `T`; Linear/LayerNorm act along the last axis `C`. See
[Module 1](lessons/module-01/lesson-04.md).

## Mixed precision: autocast, bf16 vs fp16, GradScaler

```python
# bf16: preferred on Ampere+ (A100/H100/RTX 30xx+). No scaler needed.
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits, loss = model(x, y)
loss.backward()
opt.step()

# fp16: needs a GradScaler (tiny gradients underflow to 0 without it)
scaler = torch.cuda.amp.GradScaler()
with torch.autocast(device_type="cuda", dtype=torch.float16):
    logits, loss = model(x, y)
scaler.scale(loss).backward()
scaler.step(opt)
scaler.update()
```

`bf16` has the *same exponent range* as fp32 (no overflow, no scaler) but fewer mantissa bits;
`fp16` has more precision but a narrow range, so it needs loss scaling. Master weights and the
optimizer state stay fp32; only the forward/backward compute runs in low precision. See
[Module 8](lessons/module-08/lesson-03.md).

## Gradient accumulation

Simulate a large global batch on limited memory by summing gradients over `N` micro-batches before
one optimizer step. Divide the loss by `N` so the update matches a true large batch.

```python
accum = 8
opt.zero_grad(set_to_none=True)
for micro in range(accum):
    x, y = get_batch("train")
    x, y = x.to(device), y.to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
        loss = loss / accum          # so grads average, not sum
    loss.backward()                  # accumulates into .grad
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
opt.step()                           # ONE step per `accum` micro-batches
```

global_batch = micro_batch x accum x data_parallel_world_size. See
[Module 7](lessons/module-07/lesson-02.md).

## Checkpointing: save/load model + optim + step + rng

```python
# SAVE — everything needed to resume bit-for-bit
torch.save({
    "model": model.state_dict(),
    "optim": opt.state_dict(),
    "step": step,
    "config": cfg.__dict__,
    "rng": torch.get_rng_state(),
    "cuda_rng": torch.cuda.get_rng_state_all(),
}, "ckpt.pt")

# LOAD — build the model first, then fill it
ckpt = torch.load("ckpt.pt", map_location=device)
model.load_state_dict(ckpt["model"])
opt.load_state_dict(ckpt["optim"])
start_step = ckpt["step"] + 1
torch.set_rng_state(ckpt["rng"])
```

Saving only the model weights loses the optimizer's momentum/variance and the RNG position — a
resume then diverges from an uninterrupted run. Save `step` to resume the LR schedule. See
[Module 7](lessons/module-07/lesson-03.md).

## DDP / FSDP conceptual one-liners

```python
# DDP: every rank holds a FULL model copy; grads are averaged with all-reduce each step.
from torch.nn.parallel import DistributedDataParallel as DDP
model = DDP(model, device_ids=[local_rank])   # backward triggers grad all-reduce

# FSDP: parameters/grads/optim state are SHARDED across ranks; gathered just-in-time per layer.
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
model = FSDP(model)                            # trades communication for memory
```

DDP fits when the model fits on one GPU (scales *throughput*). FSDP / ZeRO shard state to fit
models that do *not* fit on one GPU (scales *capacity*). See
[Module 9](lessons/module-09/lesson-01.md) and [ZeRO/FSDP](lessons/module-09/lesson-02.md).

## torch.profiler basics

```python
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    logits, loss = model(x, y)
    loss.backward()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

# quick manual GPU timing (CUDA is async — you MUST synchronize):
torch.cuda.synchronize(); t0 = time.time()
run()
torch.cuda.synchronize(); dt = time.time() - t0
```

CUDA kernels launch asynchronously, so a bare `time.time()` around a GPU call measures only the
launch, not the work. Always `torch.cuda.synchronize()` before reading the clock. See
[Module 8](lessons/module-08/lesson-03.md).

## F.scaled_dot_product_attention

```python
# q,k,v : (B,nh,T,hd). is_causal=True applies the causal mask internally.
y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
# equivalent explicit form (what it fuses):
scale = 1.0 / (q.size(-1) ** 0.5)
att = (q @ k.transpose(-2, -1)) * scale
att = att.masked_fill(mask[:T, :T] == 0, float("-inf"))
att = F.softmax(att, dim=-1)
y = att @ v
```

`scaled_dot_product_attention` dispatches to a FlashAttention-style fused kernel when available:
it never materializes the full `(B,nh,T,T)` score matrix in HBM, saving memory and bandwidth.
Prefer `is_causal=True` over building your own mask. See
[Module 5](lessons/module-05/lesson-02.md) and [FlashAttention](lessons/module-08/lesson-04.md).

## A generation loop

```python
@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    # idx (B,T) long — the prompt; returns (B, T+max_new_tokens)
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.block_size:]      # crop to context window
        logits, _ = model(idx_cond)                    # (B,T,V)
        logits = logits[:, -1, :] / temperature        # (B,V) last position only
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)              # (B,V)
        next_id = torch.multinomial(probs, 1)          # (B,1) sample
        idx = torch.cat([idx, next_id], dim=1)         # append
    return idx
```

Only the *last* position's logits predict the next token. `temperature < 1` sharpens, `> 1`
flattens; `temperature -> 0` is greedy `argmax`. Crop to `block_size` or positional embeddings run
out of range. See [Module 6](lessons/module-06/lesson-03.md).

## Common shape-bug checklist

- **Token ids not `long`** — `Embedding`/`cross_entropy` demand int64. `x = x.long()`.
- **Targets off by one** — for next-token LM, `targets = x` shifted left by one position.
- **`cross_entropy` shape** — flatten to `(B*T, V)` logits and `(B*T,)` targets, not `(B,T,V)`.
- **Missing `transpose(-2,-1)`** — `q @ k` needs `k` transposed to `(...,hd,T)`; a wrong shape may
  still multiply silently if dims happen to match.
- **`view` after `transpose`** — non-contiguous; insert `.contiguous()`.
- **`(T,)` vs `(T,1)`** — they broadcast to different results; check the mask/bias axis.
- **Mask not on model device** — register as a buffer or build with `device=x.device`.
- **LayerNorm over the wrong axis** — it normalizes the *last* dim; make sure width is last.
- **Forgot `zero_grad`** — gradients accumulate across steps and training diverges.
- **`argmax`/`softmax` on the wrong `dim`** — over vocab it is `dim=-1`, not `dim=0`.
- **Squeezed a real batch of size 1** — `squeeze()` with no arg drops *every* size-1 dim,
  including `B=1`; pass the explicit dim.

---

See also: the [Math & ML cheat sheet](assets/math-cheatsheet.md) for the equations behind these ops, and
the [Glossary](assets/glossary.md) for term definitions.
