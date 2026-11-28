---
id: "09.4"
module: 9
minutes: 15
practice_minutes: 10
prerequisites: ["07.2", "09.1"]
objectives:
  - Tell apart a batch, a micro-batch, an accumulation window and a data-parallel shard in a training loop.
  - Compute the global batch from micro-batch size, accumulation steps and world size.
  - Place model.no_sync(), the optimizer step and gradient clipping correctly in a DDP accumulation loop.
volatility: concept
sources:
  - title: "PyTorch DistributedDataParallel (no_sync, comm hooks)"
    url: https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
  - title: "PyTorch performance tuning guide: skip unnecessary all-reduce with no_sync"
    url: https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
  - title: "PyTorch DistributedSampler"
    url: https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
last_verified: "2026-11-28"
---

# 09.4 · Summary: batches, accumulation, and DDP

Three separate mechanisms decide how many examples go into one weight update: the micro-batch size (what one forward/backward pass sees), gradient accumulation (how many passes each worker adds up before stepping), and data parallelism (how many workers contribute their sums). This page puts the three side by side, with one running example, so that a line like "use `model.no_sync()` for the first A−1 micro-batches" reads as obvious rather than as a new idea.

## The three questions

Every distributed training loop answers three independent questions. Keep them apart and everything else follows.

| Question | Knob | Symbol | Where it lives in code |
|---|---|---|---|
| How many examples in one forward/backward pass? | micro-batch size | $b$ | `DataLoader(batch_size=b)` |
| How many backward passes before one `optimizer.step()`? | accumulation steps | $A$ | `if (step + 1) % A == 0: optimizer.step()` |
| How many workers (GPUs, processes) each do the above at the same time? | world size | $W$ | `torchrun --nproc_per_node=W` |

The number of examples behind one optimizer update is the **global batch**:

$$
B_{\text{global}} = b \times A \times W.
$$

Example used on the rest of this page: $b = 32$, $A = 4$, $W = 4$, so $B_{\text{global}} = 32 \times 4 \times 4 = 512$. Each GPU sees $32 \times 4 = 128$ examples per update; four GPUs together see 512.

> [!IMPORTANT]
> Adding GPUs without changing $b$ or $A$ makes the global batch bigger. Going from 1 GPU to 4 with $b=32, A=4$ moves the global batch from 128 to 512. If you want to keep 128, set $A = 1$ on the 4-GPU run: $32 \times 1 \times 4 = 128$.

## Names: batch, mini-batch, micro-batch

The names are not fully standardized, and this is where most confusion starts.

- **Batch / mini-batch.** The examples behind **one optimizer update**. "Mini-batch gradient descent" just means you update after a group of examples rather than after one example (SGD in the strict sense) or after the whole dataset ("full-batch" gradient descent). With accumulation and DDP, this is the global batch, 512 in the example.
- **Micro-batch.** The examples in **one forward/backward pass on one GPU**, 32 in the example. It exists only because the full update batch does not fit in one GPU's memory at once. Micro-batches are **not** separate updates: their gradients are added up and only the sum is used.
- **Shard.** The part of the data a single worker sees. `DistributedSampler` gives each of the $W$ ranks a different, equally sized slice of the dataset every epoch.

> [!WARNING]
> Some people (and some older notes) say "mini-batch" for the small piece that fits on the GPU. In this course that piece is always the **micro-batch**, and "batch" or "global batch" is always the update batch. When you read a paper, check which one a number refers to: "batch size 4M tokens" is a global batch, never what one GPU sees in a pass.

In a PyTorch `DataLoader` with accumulation, `batch_size` is the **micro-batch** size, not the update batch. Nothing in the `DataLoader` knows about $A$ or $W$.

## Accumulation: adding gradients over time on one worker

Calling `loss.backward()` does not overwrite `param.grad`, it **adds** to it. Gradient accumulation is just calling `backward()` $A$ times before `optimizer.step()` and `optimizer.zero_grad()`.

Each micro-batch loss is a **mean** over its $b$ examples. To get the mean over all $bA$ examples, divide each micro-batch loss by $A$ before `backward()`. Rank $w$ then holds

$$
G_w = \sum_{a=1}^{A} \frac{g_{w,a}}{A},
$$

where $g_{w,a}$ is the gradient of rank $w$'s mean loss on micro-batch $a$.

Worked example, one scalar weight, $A = 2$: micro-batch gradients are 4 and 6. Without dividing, `.grad` ends at $4 + 6 = 10$, twice the mean. With dividing, it ends at $4/2 + 6/2 = 5$, the mean over both micro-batches, which is what one backward on the combined batch would give.

> [!NOTE]
> Dividing by $A$ gives the exact combined mean only when all micro-batches have the same size (same number of examples, or same number of loss tokens for a language model). With variable token counts, sum the per-token losses and divide by the total token count of the whole window instead.

## Data parallelism: averaging gradients across workers

DDP keeps one full copy of the model on each of the $W$ ranks. Each rank runs forward and backward on its own shard, then DDP **all-reduces** the gradients: it sums `param.grad` over the ranks and divides by $W$, so every rank ends with the same averaged gradient. Because every rank starts from the same weights and applies the same gradient with the same optimizer, the copies stay identical without ever sending the weights.

**All-reduce** is the operation ("everyone sends a tensor, everyone gets back the sum"). **Ring all-reduce** is one algorithm for doing it: the $W$ ranks form a ring, run a reduce-scatter phase ($W-1$ steps, each rank ends with the full sum of one $1/W$ chunk) and then an all-gather phase ($W-1$ more steps, the summed chunks travel round so everyone has all of them). Each rank sends about $2\frac{W-1}{W}$ times the gradient size, almost independent of $W$. NCCL picks ring or tree algorithms for you; DDP just calls it.

> [!IMPORTANT]
> Ring all-reduce, DDP and the sampler are not batching techniques. `DistributedSampler` decides **which** examples each rank sees, DDP decides **when** gradients are synchronized, and ring all-reduce is **how** the bytes move during that synchronization.

DDP does not wait for backward to finish. It groups parameters into buckets (about 25 MB each by default) and starts the all-reduce for a bucket as soon as all its gradients are ready, overlapping communication with the rest of backward.

## Putting the two together, and what `no_sync()` changes

With both mechanisms, the two averages multiply. Each rank divides by $A$ during accumulation, DDP divides by $W$ during the all-reduce:

$$
\bar{G} = \frac{1}{W}\sum_{w=1}^{W} G_w = \frac{1}{WA}\sum_{w=1}^{W}\sum_{a=1}^{A} g_{w,a}.
$$

That is the mean over all $W A$ micro-batches, the same gradient a single huge GPU would compute on the 512-example global batch.

By default DDP all-reduces on **every** `backward()`, including the intermediate micro-batches of an accumulation window. Those intermediate averages are never used, since nobody calls `optimizer.step()` until the window ends. `model.no_sync()` is a context manager that turns that communication off: inside it, backward only adds into this rank's own `.grad`. The first backward outside the block all-reduces as usual, and because `.grad` already holds the sum of the whole window, that one exchange averages everything.

Worked example, $W = 2$, $A = 2$, one scalar weight:

| | micro-batch 1 | micro-batch 2 |
|---|---|---|
| rank 0 gradient $g_{0,a}$ | 4 | 2 |
| rank 1 gradient $g_{1,a}$ | 6 | 0 |

Micro-batch 1 runs inside `no_sync()`: rank 0's `.grad` becomes $4/2 = 2$, rank 1's becomes $6/2 = 3$. They differ, which is fine because nobody steps yet. Micro-batch 2 runs normally: rank 0 reaches $2 + 2/2 = 3$, rank 1 reaches $3 + 0/2 = 3$, and the all-reduce gives $(3+3)/2 = 3$ on both. Check: the mean of all four gradients is $(4+2+6+0)/4 = 3$.

For $A = 4$ in the running example:

| Micro-batch in window | `backward()` adds to `.grad` | All-reduce across GPUs | `optimizer.step()` |
|---|---|---|---|
| 1 | yes | no (`no_sync`) | no |
| 2 | yes | no (`no_sync`) | no |
| 3 | yes | no (`no_sync`) | no |
| 4 | yes | **yes** | **yes**, then `zero_grad` |

Three facts to keep straight:

- **`no_sync()` is an optimization, not a correctness fix.** Averaging is linear, so if you forget it, each micro-batch gets averaged and the sums still add to the same gradient. What you lose is speed: $A$ full-gradient all-reduces per step instead of one. For a 1B-parameter model with fp32 gradients that is 4 GB of payload per micro-batch, so $A = 8$ moves 8× more data for the same update.
- **The forward pass must be inside `no_sync()` too.** DDP prepares its reducer during forward. If only `backward()` is inside the block, DDP still synchronizes.
- **Clip only after the sync.** Gradient clipping looks at the norm of the averaged gradient, so it goes on the last micro-batch, after its `backward()` and before `optimizer.step()`. Clipping inside the window would clip each rank's partial, unsynchronized sum.

## The full loop

The parts every rank runs, with the decision points marked. `MyDataset` and `MyModel` stand in for your own classes; the launch is `torchrun --standalone --nproc_per_node=4 train.py` on a 4-GPU machine (see [09.1](lesson-01.md) for the complete script and the Windows/CPU launch).

```python
import contextlib, os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

dist.init_process_group("nccl")                      # one process per GPU, started by torchrun
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

b, A, epochs = 32, 4, 10                             # micro-batch, accumulation steps
dataset = MyDataset()
sampler = DistributedSampler(dataset, shuffle=True, drop_last=True)  # a different shard per rank
loader = DataLoader(dataset, batch_size=b, sampler=sampler,           # batch_size = MICRO-batch
                    drop_last=True, num_workers=4, pin_memory=True)

model = DDP(MyModel().to(device), device_ids=[local_rank])
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

steps_per_epoch = len(loader) // A                   # drop an incomplete last window
for epoch in range(epochs):
    sampler.set_epoch(epoch)                         # reshuffle the shards each epoch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step, (x, y) in enumerate(loader):
        if step >= steps_per_epoch * A:
            break
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        is_last = (step + 1) % A == 0                # last micro-batch of this window?
        ctx = contextlib.nullcontext() if is_last else model.no_sync()
        with ctx:                                    # forward AND backward inside
            loss = torch.nn.functional.cross_entropy(model(x), y) / A
            loss.backward()                          # all-reduces only when is_last
        if is_last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # on the synced gradient
            optimizer.step()                         # same update on every rank
            optimizer.zero_grad(set_to_none=True)

dist.destroy_process_group()
```

Shapes, dtype, device for one micro-batch on rank `local_rank`: `x` is `[32, ...]` (your input shape), `y` is `[32]` `int64` class ids, both on `cuda:local_rank`; each `param.grad` has its parameter's shape and dtype (usually `float32`) on that same GPU. `no_sync()` changes no tensor and costs no extra memory, since accumulation happens in the `.grad` tensors that already exist.

Why the `break`: every rank must reach the synchronizing micro-batch at the same step, otherwise one rank waits forever in an all-reduce the others never join. `DistributedSampler` already gives every rank the same number of micro-batches; truncating to a multiple of $A$ makes sure no rank ends the epoch with a half-filled `.grad`.

## Practice · Check the three claims on CPU

Run [`accum_ddp_check.py`](../../code/experiments/accum_ddp_check.py). It starts two CPU processes with the Gloo backend (no GPU, works on Windows), each accumulating $A = 4$ micro-batches of 3 examples, and checks the claims on this page against one process computing the mean loss over all 24 examples. It takes under a minute.

```bash
cd code
py experiments/accum_ddp_check.py
```

You should see (checked 2026-11-28, PyTorch 2.14 CPU):

```text
global batch = W*A*b = 2*4*3 = 24
max |grad(no_sync) - grad(single process)| = 1.19e-07
max |grad(no_sync) - grad(sync every step)| = 5.96e-08
all-reduce calls per optimizer step: with no_sync = 1, without = 4
OK
```

The first line is the global-batch formula. The second shows that $W$ workers with $A$-step accumulation give the single-process gradient (up to float rounding). The third shows that `no_sync()` does not change the result. The fourth, counted by a communication hook, is the only thing `no_sync()` changes.

Now break it on purpose, one change at a time, and predict the output before you run it:

1. Remove `/ A` from the loss.
2. Move the forward pass (the line computing `loss`) out of the `with ctx:` block, leaving only `loss.backward()` inside.

<details><summary>Hint</summary>

For 1, ask what `.grad` holds after four undivided backwards. For 2, remember where DDP prepares its reducer.

</details>

<details><summary>Stronger hint and expected result</summary>

1. The gradient is $A = 4$ times the single-process gradient, so the first check fails. The all-reduce count stays 1.
2. The gradient is still correct (averaging is linear), but the all-reduce count with `no_sync` goes back to 4: DDP decided to synchronize during the forward, which ran outside the block. The script's last assert fails on the call count, not on the gradient.

</details>

## Common mistakes

- **Treating micro-batches as updates.** Calling `optimizer.step()` after every micro-batch turns accumulation off: you get $A$ small updates instead of one large one.
- **Forgetting `/ A`.** The update becomes $A$ times too large, which looks like a learning rate $A$ times too high.
- **Dividing by $W$ yourself.** DDP already averages over ranks. Dividing the loss by $W$ as well makes the gradient $W$ times too small.
- **Wrapping only `backward()` in `no_sync()`.** Still correct, still slow.
- **Clipping or logging grad-norm inside the window.** Before the last micro-batch, each rank holds a different, partial gradient.
- **Uneven last window.** If ranks disagree on which step is the last one, they hang in the all-reduce.

## Next

Data parallelism keeps a full copy of the weights, gradients and optimizer state on every GPU. The next module moves on to scaling laws; if you need the memory-sharding alternative to DDP first, go back to [09.2 · ZeRO & FSDP](lesson-02.md). Continue to [10.1 · FLOPs & parameter counting](../module-10/lesson-01.md).
