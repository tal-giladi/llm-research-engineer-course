# Glossary

Concise definitions of the terms used across the course, alphabetical by first letter. Each entry
is one or two sentences; follow the linked lesson for the full treatment, math, and a worked
example. Notation matches the lessons: $B$ batch, $T$ context length, $C=n_{\text{embd}}$ width,
$V$ vocab, $n_h$ heads, $d_h=C/n_h$ head dim, $L$ layers, $N$ params, $D$ training tokens.

---

## A

**Ablation** — an experiment that removes or changes one component to measure its contribution;
the standard way to justify a design choice. See [Module 19](../lessons/module-19/lesson-02.md).

**Adam** — an adaptive optimizer that keeps per-parameter running means of the gradient (first
moment) and squared gradient (second moment) and steps by their ratio. See [Module 3](../lessons/module-03/lesson-02.md).

**AdamW** — Adam with *decoupled* weight decay: the decay is applied directly to the weights, not
folded into the gradient, which is the correct form. The course default optimizer. See [Module 3](../lessons/module-03/lesson-02.md).

**Advantage** — in policy-gradient RL, how much better an action was than a baseline
($A = R - b$); using it in place of the raw return reduces gradient variance. See [Module 15](../lessons/module-15/lesson-02.md).

**Agent** — an LLM that runs in a loop of think → act (call a tool) → observe, pursuing a goal
over multiple steps rather than answering once. See [Module 19](../lessons/module-19/lesson-01.md).

**All-reduce** — a collective that sums (or averages) a tensor across all workers and returns the
result to every worker; how DDP averages gradients. See [Module 9](../lessons/module-09/lesson-01.md).

**Arithmetic intensity** — FLOPs performed per byte moved from memory; low intensity means an op
is memory-bandwidth-bound, high means compute-bound. See [Module 8](../lessons/module-08/lesson-02.md).

**Attention** — the operation letting each position read a weighted blend of the other positions'
values, weighted by query–key similarity: $\mathrm{softmax}(QK^\top/\sqrt{d_h}+M)V$. See [Module 5](../lessons/module-05/lesson-02.md).

**Autograd** — PyTorch's reverse-mode automatic differentiation: it records a graph of ops during
the forward pass and replays it backward to compute gradients. See [Module 2](../lessons/module-02/lesson-04.md).

## B

**Backpropagation** — computing loss gradients by applying the chain rule from the output back to
every parameter, reusing intermediate results. See [Module 2](../lessons/module-02/lesson-03.md).

**Batch** — the set of sequences processed together in one forward/backward pass; the leading
tensor dimension $B$. See [Module 7](../lessons/module-07/lesson-01.md).

**bf16 (bfloat16)** — a 16-bit float with fp32's exponent range but fewer mantissa bits; the
preferred training precision because it needs no loss scaling. See [Module 8](../lessons/module-08/lesson-03.md).

**Block** — one transformer layer: pre-norm attention plus a pre-norm MLP, each wrapped in a
residual connection. See [Module 5](../lessons/module-05/lesson-04.md).

**BPE (Byte-Pair Encoding)** — a tokenizer trained by repeatedly merging the most frequent
adjacent symbol pair, yielding a subword vocabulary. See [Module 4](../lessons/module-04/lesson-02.md).

**Bradley-Terry model** — a model of pairwise preference where the probability that A beats B is
$\sigma(r_A - r_B)$; the training objective for a reward model. See [Module 15](../lessons/module-15/lesson-01.md).

**Broadcasting** — the rule that lets ops combine tensors of different shapes by stretching size-1
dims, aligning from the right, without copying data. See [Module 0](../lessons/module-00/lesson-02.md).

## C

**Calibration** — how well a model's stated confidence matches its actual accuracy; a calibrated
model is right 70% of the time when it says 70%. See [Module 13](../lessons/module-13/lesson-02.md).

**Causal mask** — an additive mask that sets future-position scores to $-\infty$ before softmax so
position $t$ can attend only to positions $\le t$. See [Module 5](../lessons/module-05/lesson-02.md).

**Chat template** — the fixed formatting (role markers, special tokens) that turns a multi-turn
conversation into the single token stream an instruction-tuned model expects. See [Module 14](../lessons/module-14/lesson-01.md).

**Checkpoint** — a saved snapshot of model weights, optimizer state, step, and RNG that allows a
run to resume exactly where it stopped. See [Module 7](../lessons/module-07/lesson-03.md).

**Chinchilla** — the finding that for a fixed compute budget, params and tokens should scale
together (~20 tokens per param), correcting earlier undertraining. See [Module 10](../lessons/module-10/lesson-03.md).

**Chain-of-thought (CoT)** — prompting or training a model to emit intermediate reasoning steps
before its final answer, which improves accuracy on multi-step problems. See [Module 17](../lessons/module-17/lesson-01.md).

**Contamination** — test examples leaking into the training data, which inflates benchmark scores
without real capability gain. See [Module 11](../lessons/module-11/lesson-03.md).

**Context length** — the maximum number of tokens $T$ (a.k.a. `block_size`) the model can attend
over in one forward pass. See [Module 5](../lessons/module-05/lesson-01.md).

**Confidence interval** — a range that would contain the true metric value at a stated frequency
under repeated sampling; report one around every headline eval number. See [Module 19](../lessons/module-19/lesson-02.md).

**Contiguous** — a tensor whose elements are laid out in memory in row-major order with no gaps; a
`view` requires it, so `transpose`/`permute` results often need `.contiguous()`. See [Module 0](../lessons/module-00/lesson-02.md).

**Cosine schedule** — a learning-rate schedule that decays the LR from its peak to a floor along a
half-cosine curve after warmup. See [Module 3](../lessons/module-03/lesson-03.md).

**Cross-entropy** — the language-modeling loss: the mean negative log-probability the model assigns
to the true next token. See [Module 1](../lessons/module-01/lesson-03.md).

## D

**Data parallelism** — replicating the full model across workers, each processing a different
micro-batch, then averaging gradients with all-reduce. See [Module 9](../lessons/module-09/lesson-01.md).

**DDP (DistributedDataParallel)** — PyTorch's data-parallel wrapper that overlaps gradient
all-reduce with the backward pass. See [Module 9](../lessons/module-09/lesson-01.md).

**Deduplication** — removing duplicate or near-duplicate documents from a corpus, which improves
quality and reduces memorization. See [Module 11](../lessons/module-11/lesson-02.md).

**DeepSeek-R1** — a reasoning model trained largely with RL on verifiable rewards (GRPO), notable
for reasoning emerging without a large SFT stage. See [Module 17](../lessons/module-17/lesson-03.md).

**Detach** — cut a tensor out of the autograd graph so no gradient flows through it, while sharing
the same data. See [Module 0](../lessons/module-00/lesson-03.md).

**Distillation** — training a smaller/cheaper student model to imitate a larger teacher's outputs
(logits or generated data). See [Module 17](../lessons/module-17/lesson-03.md).

**DPO (Direct Preference Optimization)** — aligns a model directly on preference pairs with a
simple classification loss, skipping the explicit reward model and RL loop. See [Module 15](../lessons/module-15/lesson-03.md).

**Dropout** — regularization that randomly zeros a fraction of activations during training; set to
0 for most LM pretraining. See [Module 5](../lessons/module-05/lesson-04.md).

**dtype** — a tensor's element type (e.g. `float32`, `bfloat16`, `int64`); governs precision,
memory, and which ops are legal. See [Module 0](../lessons/module-00/lesson-01.md).

## E

**Embedding** — a learned lookup table mapping each token id to a $C$-dimensional vector;
`nn.Embedding` indexes rows, it is not a matmul. See [Module 5](../lessons/module-05/lesson-01.md).

**Entropy** — the expected surprise of a distribution, $H(p)=-\sum_i p_i\log p_i$; maximal for a
uniform distribution. See [Module 1](../lessons/module-01/lesson-03.md).

**Expert parallelism** — sharding a Mixture-of-Experts model by placing different experts on
different devices and routing tokens to them. See [Module 9](../lessons/module-09/lesson-03.md).

## F

**Few-shot** — prompting with a handful of worked examples before the real query, so the model
infers the task in-context without weight updates. See [Module 13](../lessons/module-13/lesson-01.md).

**FlashAttention** — an exact attention algorithm that tiles the computation and never writes the
full $T\times T$ score matrix to HBM, saving memory and bandwidth. See [Module 8](../lessons/module-08/lesson-04.md).

**FLOPs** — floating-point operations; the compute unit for cost estimates. A dense forward pass
is about $2N$ FLOPs per token, forward+backward about $6N$. See [Module 7](../lessons/module-07/lesson-04.md).

**fp16 (float16)** — a 16-bit float with more mantissa but a narrow exponent range; needs a
GradScaler to keep small gradients from underflowing. See [Module 8](../lessons/module-08/lesson-03.md).

**FSDP (Fully Sharded Data Parallel)** — shards parameters, gradients, and optimizer state across
workers, gathering each layer's params just in time; PyTorch's ZeRO. See [Module 9](../lessons/module-09/lesson-02.md).

**Function calling** — a model emitting a structured call (name + JSON arguments) matching a tool
schema, which the harness executes and feeds back. See [Module 18](../lessons/module-18/lesson-01.md).

## G

**GELU** — a smooth activation, $x\,\Phi(x)$ with $\Phi$ the standard-normal CDF; the classic
GPT-2 MLP nonlinearity. See [Module 5](../lessons/module-05/lesson-04.md).

**Global batch** — the effective number of sequences per optimizer step:
micro-batch × grad-accumulation × data-parallel workers. See [Module 7](../lessons/module-07/lesson-02.md).

**GQA (Grouped-Query Attention)** — a middle ground between MHA and MQA where groups of query
heads share one key/value head, shrinking the KV cache with little quality loss. See [Module 12](../lessons/module-12/lesson-03.md).

**Gradient** — the vector of partial derivatives of the loss with respect to parameters; points in
the direction of steepest increase, so we step against it. See [Module 2](../lessons/module-02/lesson-01.md).

**Gradient accumulation** — summing gradients over several micro-batches before one optimizer step,
to reach a large global batch under limited memory. See [Module 7](../lessons/module-07/lesson-02.md).

**Gradient clipping** — rescaling the gradient when its global norm exceeds a threshold, preventing
loss spikes from destabilizing training. See [Module 3](../lessons/module-03/lesson-03.md).

**GRPO (Group Relative Policy Optimization)** — a PPO variant that drops the value network and
estimates advantage from the mean reward of a group of sampled completions. See [Module 16](../lessons/module-16/lesson-01.md).

## H

**HBM (High-Bandwidth Memory)** — the large but comparatively slow off-chip GPU DRAM (tens of GB)
where tensors live; the target of most bandwidth optimization. See [Module 8](../lessons/module-08/lesson-01.md).

## I

**In-context learning** — a model adapting to a task from examples in its prompt alone, with no
gradient updates. See [Module 13](../lessons/module-13/lesson-01.md).

## J

**Jacobian** — the matrix of all partial derivatives of a vector-valued function; backprop never
forms it explicitly, using vector-Jacobian products instead. See [Module 2](../lessons/module-02/lesson-02.md).

## K

**Kaplan scaling laws** — the empirical power-law relationship between loss and compute, params,
and data from the 2020 OpenAI study. See [Module 10](../lessons/module-10/lesson-02.md).

**Key** — in attention, the vector each position advertises; a query is compared against all keys
to decide attention weights. See [Module 5](../lessons/module-05/lesson-02.md).

**KL divergence** — the extra coding cost of using distribution $q$ for samples from $p$,
$\mathrm{KL}(p\|q)=\sum_i p_i\log(p_i/q_i)\ge 0$; asymmetric. See [Module 1](../lessons/module-01/lesson-03.md).

**KL penalty** — a term in RLHF that keeps the trained policy close to the reference model,
preventing reward hacking and mode collapse. See [Module 15](../lessons/module-15/lesson-02.md).

**KV cache** — stored keys and values from past positions during generation, so each new token
costs one step instead of recomputing all of attention. See [Module 12](../lessons/module-12/lesson-03.md).

## L

**LayerNorm** — normalizes each vector across its width to zero mean and unit variance, then
applies a learned scale and shift; stabilizes training. See [Module 5](../lessons/module-05/lesson-04.md).

**Load balancing** — in MoE, an auxiliary loss or routing rule that spreads tokens evenly across
experts so none is starved or overloaded. See [Module 12](../lessons/module-12/lesson-04.md).

**Logits** — the raw, unnormalized scores the model outputs, shape $(B,T,V)$; softmax turns them
into next-token probabilities. See [Module 6](../lessons/module-06/lesson-02.md).

**Loss masking** — setting target ids to an ignore value (e.g. $-100$) on prompt/system tokens so
loss is computed only on the tokens the model should learn to produce. See [Module 14](../lessons/module-14/lesson-02.md).

**LoRA (Low-Rank Adaptation)** — freezes the base weights and trains a small low-rank update
$BA$ per matrix, cutting trainable params by orders of magnitude. See [Module 14](../lessons/module-14/lesson-03.md).

## M

**Matmul** — matrix multiplication; the dominant compute in a transformer, batched over all
leading dims by `@`/`torch.matmul`. See [Module 0](../lessons/module-00/lesson-02.md).

**MFU (Model FLOPs Utilization)** — achieved model FLOPs divided by the hardware's peak FLOPs; a
0–1 efficiency measure, with 40–55% typical for good large-scale training. See [Module 7](../lessons/module-07/lesson-04.md).

**Micro-batch** — the batch that fits on one device in one forward pass; several micro-batches
accumulate into a global batch. See [Module 7](../lessons/module-07/lesson-02.md).

**MinHash** — a hashing sketch that estimates the Jaccard similarity of two documents cheaply,
used for near-duplicate detection at corpus scale. See [Module 11](../lessons/module-11/lesson-02.md).

**Mixed precision** — running forward/backward compute in bf16/fp16 while keeping master weights
and optimizer state in fp32, for speed and memory with stability. See [Module 8](../lessons/module-08/lesson-03.md).

**MLP** — the per-position feed-forward network in a transformer block: an up-projection, a
nonlinearity, and a down-projection (usually width $4C$). See [Module 5](../lessons/module-05/lesson-04.md).

**MoE (Mixture of Experts)** — replaces the MLP with many expert MLPs and a router that sends each
token to a few, raising capacity while keeping per-token compute low. See [Module 12](../lessons/module-12/lesson-04.md).

**Momentum** — an optimizer term that accumulates an exponential average of past gradients,
smoothing the trajectory and accelerating along consistent directions. See [Module 3](../lessons/module-03/lesson-01.md).

**MQA (Multi-Query Attention)** — all query heads share a single key/value head, minimizing the KV
cache at some quality cost. See [Module 12](../lessons/module-12/lesson-03.md).

**Multi-head attention** — running attention in $n_h$ parallel subspaces of dimension $d_h=C/n_h$
and concatenating, so heads specialize on different relationships. See [Module 5](../lessons/module-05/lesson-03.md).

## N

**NCCL** — NVIDIA's collective communication library; the backend implementing all-reduce and
friends on GPU clusters. See [Module 9](../lessons/module-09/lesson-01.md).

**no_grad** — a context manager that disables graph construction for inference or manual parameter
updates, saving memory. See [Module 0](../lessons/module-00/lesson-03.md).

## O

**Outcome reward** — a reward given only for the final answer's correctness, regardless of the
reasoning path; contrast process reward. See [Module 17](../lessons/module-17/lesson-02.md).

## P

**pass@k** — the probability that at least one of $k$ sampled solutions is correct; a standard
metric for code and reasoning tasks. See [Module 13](../lessons/module-13/lesson-01.md).

**Perplexity** — the exponential of mean cross-entropy, $\exp(\mathcal{L})$; the model's effective
branching factor, lower is better. See [Module 1](../lessons/module-01/lesson-04.md).

**Pipeline parallelism** — splitting the model's layers across devices in stages and streaming
micro-batches through them to keep every stage busy. See [Module 9](../lessons/module-09/lesson-03.md).

**Positional encoding** — information about token order added to embeddings, since attention itself
is permutation-invariant; can be learned, sinusoidal, or rotary. See [Module 5](../lessons/module-05/lesson-01.md).

**PPO (Proximal Policy Optimization)** — a policy-gradient RL algorithm that clips the
policy-ratio update to stay near the old policy; the classic RLHF optimizer. See [Module 15](../lessons/module-15/lesson-02.md).

**Process reward (PRM)** — a reward signal scoring each intermediate reasoning step, not just the
final answer. See [Module 17](../lessons/module-17/lesson-02.md).

## Q

**QLoRA** — LoRA on top of a 4-bit quantized frozen base model, letting large models be fine-tuned
on a single GPU. See [Module 14](../lessons/module-14/lesson-03.md).

**Query** — in attention, the vector a position uses to ask what it is looking for; compared
against all keys. See [Module 5](../lessons/module-05/lesson-02.md).

## R

**ReAct** — an agent pattern interleaving reasoning traces with actions (tool calls), so thinking
and acting inform each other. See [Module 19](../lessons/module-19/lesson-01.md).

**Rejection sampling** — generating many candidate outputs and keeping only those a verifier
accepts, e.g. to build reasoning SFT data. See [Module 16](../lessons/module-16/lesson-03.md).

**Residual connection** — adding a sublayer's input to its output ($x + f(x)$), which gives
gradients a direct path and enables deep stacks. See [Module 5](../lessons/module-05/lesson-04.md).

**Reward model** — a model trained on preference pairs to score outputs; supplies the reward signal
in RLHF. See [Module 15](../lessons/module-15/lesson-01.md).

**RLHF** — Reinforcement Learning from Human Feedback: fit a reward model to human preferences,
then optimize the policy against it (e.g. with PPO). See [Module 15](../lessons/module-15/lesson-01.md).

**RLVR** — RL with Verifiable Rewards: use an automatic checker (math answer, unit tests) as the
reward instead of a learned reward model. See [Module 16](../lessons/module-16/lesson-02.md).

**RMSNorm** — a cheaper normalization that rescales by the root-mean-square of the vector (no mean
subtraction, no bias); standard in modern LLMs. See [Module 12](../lessons/module-12/lesson-02.md).

**Ring all-reduce** — a bandwidth-optimal all-reduce that passes chunks around a ring of workers
in reduce-scatter then all-gather phases. See [Module 9](../lessons/module-09/lesson-01.md).

**RoPE (Rotary Position Embedding)** — encodes position by rotating query/key pairs by an
angle proportional to their index, giving relative-position awareness. See [Module 12](../lessons/module-12/lesson-01.md).

**Roofline** — a plot bounding achievable performance by memory bandwidth and peak compute,
showing whether a kernel is memory- or compute-bound. See [Module 8](../lessons/module-08/lesson-02.md).

**Router** — the small gating network in MoE that scores experts per token and selects the top few.
See [Module 12](../lessons/module-12/lesson-04.md).

## S

**Scaling laws** — empirical power laws predicting loss from compute, params, and data, used to
plan runs and pick model size. See [Module 10](../lessons/module-10/lesson-02.md).

**Seed variance** — the spread in results from different random seeds; a difference smaller than it
is not a real effect. See [Module 19](../lessons/module-19/lesson-02.md).

**Self-consistency** — sampling several chains of thought and taking a majority vote over their
final answers, which beats a single greedy chain. See [Module 17](../lessons/module-17/lesson-01.md).

**SFT (Supervised Fine-Tuning)** — training a pretrained model on curated instruction–response
pairs with next-token loss on the response tokens. See [Module 14](../lessons/module-14/lesson-01.md).

**SGD (Stochastic Gradient Descent)** — updating parameters by a step against the gradient of a
mini-batch loss; the base of all the optimizers. See [Module 3](../lessons/module-03/lesson-01.md).

**Softmax** — turns a score vector into a probability distribution,
$\mathrm{softmax}(z)_i=e^{z_i}/\sum_j e^{z_j}$; compute it stably by subtracting the max. See [Module 1](../lessons/module-01/lesson-03.md).

**SRAM** — the small, very fast on-chip GPU memory (shared memory / registers) that fused kernels
like FlashAttention keep their working set in. See [Module 8](../lessons/module-08/lesson-01.md).

**STaR (Self-Taught Reasoner)** — bootstrapping reasoning by keeping model-generated chains that
reach the correct answer and fine-tuning on them. See [Module 17](../lessons/module-17/lesson-01.md).

## T

**Temperature** — a divisor on logits before softmax during sampling; $<1$ sharpens toward greedy,
$>1$ flattens toward uniform. See [Module 6](../lessons/module-06/lesson-03.md).

**Tensor** — a multi-dimensional array with a shape, dtype, and device; the universal data
structure of PyTorch. See [Module 0](../lessons/module-00/lesson-01.md).

**Tensor parallelism** — splitting individual weight matrices across devices so one layer's matmul
runs in parallel, with communication to stitch results. See [Module 9](../lessons/module-09/lesson-03.md).

**Tokenizer** — the reversible map between text and integer token ids that the model consumes;
usually BPE. See [Module 4](../lessons/module-04/lesson-01.md).

**Tool schema** — the machine-readable declaration of a tool's name, description, and JSON
parameters that a model reads to call it correctly. See [Module 18](../lessons/module-18/lesson-01.md).

**Top-k sampling** — restricting sampling to the $k$ highest-probability tokens (renormalized),
cutting off the unlikely tail. See [Module 6](../lessons/module-06/lesson-03.md).

**Top-p (nucleus) sampling** — sampling from the smallest set of tokens whose cumulative
probability exceeds $p$, an adaptive alternative to top-k. See [Module 6](../lessons/module-06/lesson-03.md).

## V

**Value (attention)** — in attention, the content a position hands over when attended to; the
attention weights blend values into the output. See [Module 5](../lessons/module-05/lesson-02.md).

**Value (RL)** — the expected future return from a state, estimated by a critic/value network in
PPO to form advantages. See [Module 15](../lessons/module-15/lesson-02.md).

**Verifier** — an automatic checker (exact-match answer, unit tests, a grader) that judges whether
an output is correct; supplies the reward in RLVR. See [Module 16](../lessons/module-16/lesson-02.md).

**View** — a tensor sharing another's storage with different shape/strides; cheap but constrained
to contiguous layouts. See [Module 0](../lessons/module-00/lesson-02.md).

**VJP (vector-Jacobian product)** — the core backprop operation: multiply an upstream gradient by a
layer's Jacobian without ever forming the Jacobian. See [Module 2](../lessons/module-02/lesson-02.md).

**Vocab size** — the number of distinct tokens $V$ the model can emit; sets the embedding table and
output-logit widths. See [Module 4](../lessons/module-04/lesson-01.md).

## W

**Warmup** — ramping the learning rate up from ~0 over the first steps before the main decay, to
avoid early instability. See [Module 3](../lessons/module-03/lesson-03.md).

**Weight decay** — an L2-style pull of weights toward zero each step for regularization; applied
decoupled in AdamW. See [Module 3](../lessons/module-03/lesson-02.md).

## Z

**Zero-shot** — evaluating a model on a task with only an instruction and no examples in the
prompt. See [Module 13](../lessons/module-13/lesson-01.md).

**ZeRO** — the DeepSpeed family of optimizations that shard optimizer state, gradients, and then
parameters across data-parallel workers to fit larger models; FSDP is PyTorch's implementation. See
[Module 9](../lessons/module-09/lesson-02.md).

---

## Numbered & symbolic

**6ND** — the rule of thumb that training a dense model of $N$ params on $D$ tokens costs about
$6ND$ FLOPs (≈$2ND$ forward + ≈$4ND$ backward). See [Module 10](../lessons/module-10/lesson-01.md).

**(B, T, C)** — the course's tensor layout convention: batch, then time/position, then
channels/width; attention mixes along $T$, Linear/Norm act along $C$. See [Module 5](../lessons/module-05/lesson-04.md).

---

See also: the [Math & ML cheat sheet](math-cheatsheet.md) and the
[PyTorch & systems cheat sheet](pytorch-cheatsheet.md).
