# 08.7 · Post-training quantization: int8/int4 weights, scales, GPTQ and AWQ

<div class="callout key"><p><strong>Added 2026-11-15 by the weekly curriculum review</strong> (<a href="#/research/CHANGELOG">changelog</a>). This extends Module 8; nothing earlier is replaced. The mixed precision of <a href="#/lessons/module-08/lesson-03">08.3</a> is still how models are <em>trained</em>: fp32 master weights, bf16 matmuls. This lesson is about what happens <em>after</em> training: storing a finished model's weights in 8, 4 or 3 bits so that the decode step of <a href="#/lessons/module-08/lesson-06">08.6</a>, which is limited by the bytes it reads, gets cheaper. The unquantized model stays the reference every quantized copy is measured against.</p></div>

<div class="prereq">
<p><strong>Prerequisites:</strong> floating-point formats, bf16 vs fp16 and what fp32 master weights do, from <a href="#/lessons/module-08/lesson-03">08.3 · Profiling &amp; mixed precision</a>; the decode roofline (time ≈ bytes read / bandwidth) and the 7B-on-A100 numbers from <a href="#/lessons/module-08/lesson-06">08.6 · Inference serving</a>; <code>nn.Linear</code> as $y = xW^\top$ with $W$ of shape (out, in) from <a href="#/lessons/module-05/lesson-04">05.4</a>; matrix inverses and the Cholesky factorization at the level of "$H = LL^\top$ for a symmetric positive-definite $H$" (explained where used). You met one quantized format already, QLoRA's NF4 base in <a href="#/lessons/module-14/lesson-03">14.3</a>, but you do not need it here.</p>
<p><strong>You will learn:</strong> the integer grid behind every format: <strong>scale</strong>, <strong>zero-point</strong>, symmetric vs asymmetric, and the granularity choice (per tensor, per output channel, per group of 128 columns); why one outlier ruins a per-tensor scale; why the right objective is the <strong>layer output error</strong> $\lVert XW^\top - X\hat W^\top\rVert^2$, which depends on the inputs through $H = X^\top X$, not the weight error; <strong>GPTQ</strong>, which quantizes one column at a time and pushes each rounding error onto the columns not yet quantized, worked by hand on a 2-weight example; <strong>AWQ</strong>, which rescales input channels by their activation size before rounding, also by hand; the <strong>MXFP4</strong> block format and the difference between post-training quantization and quantization-aware training; and the real cost: bytes per weight including scales, and when a 4-bit model is and is not faster. Code: <code>llmre.model.quantize</code>.</p>
<p><strong>Why this matters for ML:</strong> almost every model you will evaluate, serve, or use as an RL sampler or a distillation teacher is quantized somewhere: int4 checkpoints for local evaluation, FP8 rollouts in RL (<a href="#/lessons/module-16/lesson-04">16.4</a> lists precision as a mismatch source), 4-bit QLoRA bases. A research engineer has to be able to say what a quantized model is (which grid, which granularity, which calibration), predict its memory and speed from first principles, and measure what it changed. "We quantized to 4 bits and accuracy stayed the same" is a claim you should know how to check.</p>
</div>

<div class="hw">
<p><strong>Hardware track:</strong> CPU-only is enough. <code>tests/test_quantize.py</code> (23 tests) runs in about 2 seconds. <code>experiments/ptq_gpt.py</code> runs in under a minute on a laptop CPU (it trains a 4-layer GPT for 300 steps, then quantizes it nine ways), under 1 GB RAM, 0 GPU-hours. The 7B numbers in sections 3 and 6 are worked from the 08.6 roofline, not measured. Quantizing a real 7B model with GPTQ needs one GPU with about 24 GB and takes minutes to an hour depending on the tool. That is optional and not required by this lesson.</p>
</div>

## 1. Intuition: the decode step pays for bytes, so store fewer bytes

08.6 showed that a decode step at small batch streams every weight from HBM to produce one token per sequence, and does very little arithmetic per byte. For a 7B model in bf16 that is 14 GB per step. At 2 TB/s that is about 7 ms, whatever the GPU's FLOP rate.

So halve the bytes, and the step gets about twice as fast. Store each weight in 4 bits instead of 16, and it gets up to 4× faster, and the model fits in a quarter of the memory. That is **weight-only post-training quantization (PTQ)**. You take a finished model, map each weight to one of $2^b$ levels, store the level's index plus a little bookkeeping, and at inference a kernel turns the indices back into numbers just before the matmul.

The whole difficulty is that 16 levels (4 bits) is very few. Three questions decide whether the model survives:

1. **Where are the 16 levels?** A **scale** (and maybe a **zero-point**) maps the integer codes to real values. The levels are evenly spaced, so one huge weight stretches the grid and leaves almost no levels for the small weights. The fix is to use more scales: one per output row, or one per group of 128 weights.
2. **What error do we care about?** Not the weight error. The layer's output is $y = xW^\top$, and rounding a weight that only ever multiplies near-zero inputs costs nothing. The error that matters is measured on real inputs.
3. **Can rounding errors cancel?** Yes. If two inputs are correlated, rounding one weight up can be compensated by moving another weight down before it is rounded. That is **GPTQ**. Or, the weights that see large inputs can be given a finer grid at the expense of those that see small inputs. That is **AWQ**.

<div class="callout key"><p>Training mixed precision (08.3) keeps an fp32 master copy and changes the <em>compute</em> format. Post-training quantization has no master copy and no gradient: it changes the <em>stored</em> format of a finished model, once, using a few hundred calibration sequences. It trades a measurable change in the model's outputs for fewer bytes per decode step.</p></div>

## 2. Mathematics

### 2.1 The integer grid: scale and zero-point

Let $w$ be a group of real weights (a whole tensor, one row, or 128 consecutive entries of a row), $b$ the number of bits.

**Symmetric (absmax).** With $q_{\max} = 2^{b-1} - 1$:

$$
s = \frac{\max_j |w_j|}{q_{\max}}, \qquad q_j = \operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{w_j}{s}\right), -q_{\max}-1, q_{\max}\right), \qquad \hat w_j = s\, q_j .
$$

- $s$ — the **scale**, the real-valued step between neighbouring levels; stored once per group, usually as fp16;
- $q_j$ — the integer **code** that is stored, $b$ bits each;
- $\hat w_j$ — the **dequantized** value the matmul actually uses.

**Asymmetric (min/max).** With $q_{\max} = 2^b - 1$, $m = \min(\min_j w_j, 0)$, $M = \max(\max_j w_j, 0)$:

$$
s = \frac{M - m}{q_{\max}}, \qquad z = \operatorname{round}\!\left(-\frac{m}{s}\right), \qquad q_j = \operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{w_j}{s}\right) + z, 0, q_{\max}\right), \qquad \hat w_j = s\,(q_j - z).
$$

$z$ is the **zero-point**: the integer code that represents real 0. Including 0 in the range makes zero exactly representable. The asymmetric grid spends no levels on a sign that a lopsided group does not use, at the cost of storing $z$.

Rounding to the nearest level gives $|\hat w_j - w_j| \le s/2$ for every weight inside the range. That bound is the whole story of **granularity**: the error is proportional to $s$, and $s$ is set by the largest $|w|$ in the group. One weight 100× larger than its neighbours makes $s$ 100× larger for all of them.

**Granularity.** For a weight $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$:
- *per tensor*: one $s$ for all of $W$;
- *per output channel*: one $s$ per row (each row produces one output feature);
- *per group*: one $s$ per row per block of $g$ consecutive input columns, $g = 128$ being the common choice for int4.

### 2.2 The objective: output error on calibration inputs

Collect the inputs this layer actually sees on a few hundred calibration sequences, $X \in \mathbb{R}^{N \times d_{\text{in}}}$ ($N$ = number of token positions). For one output row $w$ (length $d_{\text{in}}$) and its quantized version $\hat w$, write $\delta = \hat w - w$. The squared output error is

$$
\lVert X\hat w - Xw \rVert^2 = \delta^\top X^\top X \,\delta = \delta^\top H\, \delta, \qquad H = X^\top X \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}.
$$

- $H$ — the (Gauss-Newton) **Hessian** of this layer's squared output error with respect to the row; it is the same for every row of $W$, because every row sees the same inputs;
- the diagonal $H_{jj} = \sum_n x_{nj}^2$ says how large input $j$ is; the off-diagonal $H_{jk}$ says how correlated inputs $j$ and $k$ are.

Round-to-nearest (**RTN**) minimizes $\lVert \delta \rVert$ per coordinate and ignores $H$ completely. If $H$ is diagonal, the error is $\sum_j H_{jj}\delta_j^2$ and nothing better can be done per coordinate. If $H$ has large off-diagonal entries, errors in correlated coordinates can cancel, and RTN leaves that on the table.

### 2.3 GPTQ: quantize one column, compensate the rest

GPTQ (Frantar et al. 2023) builds on Optimal Brain Quantization (OBQ). Quantize the coordinates of $w$ in a fixed order. Suppose coordinate $i$ has just been rounded, with error $e = w_i - \operatorname{quant}(w_i)$, and $F$ is the set of coordinates not yet quantized. Holding coordinate $i$ at its rounded value, the change to the remaining weights that minimizes $\delta^\top H\delta$ is

$$
w_F \leftarrow w_F - e \,\frac{[H_F^{-1}]_{F,i}}{[H_F^{-1}]_{ii}},
\qquad \text{and the error grows by } \frac{e^2}{[H_F^{-1}]_{ii}} .
$$

- $H_F^{-1}$ — the inverse of $H$ restricted to the coordinates still free (including $i$);
- $[H_F^{-1}]_{F,i}$ — its column $i$, restricted to the other free coordinates.

(This is a least-squares projection: minimize a quadratic in $\delta_F$ with one coordinate fixed; the formula is the standard Schur-complement solution.)

Recomputing $H_F^{-1}$ after every column would be too slow. GPTQ's observation: if you always quantize columns left to right, the rows you need are exactly the rows of the **upper Cholesky factor** $U$ of $H^{-1}$ ($H^{-1} = U^\top U$). The update becomes

$$
e = \frac{w_i - \operatorname{quant}(w_i)}{U_{ii}}, \qquad w_j \leftarrow w_j - e\, U_{ij} \quad (j > i),
$$

and it is applied to all $d_{\text{out}}$ rows at once, because they share $H$. Two more details from the paper: a small **damping** $\lambda = 0.01 \cdot \operatorname{mean}(\operatorname{diag} H)$ is added to the diagonal so $H$ is invertible, and, with groups, the scale of each group is fit on the weights *as they are when the group is reached*, after earlier compensation.

If $H$ is diagonal, $U$ is diagonal, $U_{ij} = 0$ for $j > i$, and GPTQ is exactly RTN. `test_gptq_with_diagonal_hessian_is_rtn` checks that.

### 2.4 AWQ: give the important channels a finer grid

AWQ (Lin et al. 2024) starts from the observation that weights are not equally important: input channels with large activations matter most (their $H_{jj}$ is large). Keeping those weights in fp16 would need mixed-precision kernels. Instead, AWQ scales the input channels. For any positive vector $s \in \mathbb{R}^{d_{\text{in}}}$,

$$
y = x W^\top = (x \oslash s)\,(W \operatorname{diag}(s))^\top ,
$$

exactly, in real arithmetic ($\oslash$ is element-wise division). Quantize $W \operatorname{diag}(s)$ instead of $W$. Column $j$ is multiplied by $s_j$ before rounding and its error is divided by $s_j$ afterwards, so a channel with $s_j > 1$ gets an effectively finer grid. The $\oslash s$ is folded into the previous operation (the preceding norm's gain, or the previous linear layer), so it is free at inference.

The choice of $s$: $s_j = \overline{|x_j|}^{\,\alpha}$, the mean absolute activation of channel $j$ on calibration data raised to a power $\alpha \in [0, 1)$, normalized so $\max s \cdot \min s = 1$. A grid search over about 20 values of $\alpha$ picks the one with the smallest output error. $\alpha = 0$ is RTN, so the search can never be worse than RTN on the calibration set. There is no gradient and no Hessian inverse.

### 2.5 Block floating point: MXFP4

The integer formats above put evenly spaced levels inside each group. **Microscaling (MX) formats** (OCP MX specification; Rouhani et al. 2023) instead store small floating-point elements that share one power-of-two scale per block of 32:

- **MXFP4**: elements in FP4 **E2M1** (1 sign, 2 exponent, 1 mantissa bit), which can hold $\pm\{0, 0.5, 1, 1.5, 2, 3, 4, 6\}$; scale $2^{e}$ stored as an 8-bit exponent (E8M0);
- **MXFP8**: the same with FP8 E4M3 or E5M2 elements.

The shared exponent is $e = \lfloor \log_2 \max_j |x_j| \rfloor - 2$, where 2 is the largest exponent E2M1 can hold ($6 = 1.5 \cdot 2^2$). Each element is $x_j / 2^e$ rounded to the nearest E2M1 value, saturating at $\pm 6$. Because the scale is a power of two, applying it is an exponent addition, which is why hardware supports it natively. The non-uniform levels (denser near zero) suit bell-shaped weight distributions better than an evenly spaced int4 grid.

## 3. Worked examples with tiny numbers

**Symmetric int4, one group.** $w = (0.8, -0.3, 0.05, -1.2)$, $b = 4$, $q_{\max} = 7$.
$s = 1.2 / 7 = 0.1714$. $w/s = (4.667, -1.75, 0.292, -7)$. Rounded: $q = (5, -2, 0, -7)$. Dequantized: $\hat w = (0.857, -0.343, 0, -1.2)$. Errors: $(0.057, -0.043, -0.05, 0)$, all within $s/2 = 0.086$. Notice the 8 negative-side levels the group barely uses: the grid is symmetric, the data is not.

**Asymmetric int4, same group.** $m = -1.2$, $M = 0.8$, $s = 2.0/15 = 0.1333$, $z = \operatorname{round}(1.2/0.1333) = 9$.
$q = \operatorname{round}(w/s) + 9 = (6, -2.25 \to -2, 0.375 \to 0, -9) + 9 = (15, 7, 9, 0)$.
$\hat w = s(q - 9) = (0.8, -0.267, 0, -1.2)$. Errors: $(0, 0.033, -0.05, 0)$. The step is $0.133$ instead of $0.171$ because no level is wasted. The price is one more stored number per group.

**One outlier, per tensor vs per channel.** A 16×64 weight with entries around $\pm 0.02$, except one row 100× larger (around $\pm 2$). Per-tensor int4 sets $s \approx 2.5/7 \approx 0.36$: every small weight rounds to 0, so the other 15 rows are wiped out entirely (mean error ≈ mean $|w|$ ≈ 0.016). Per-channel gives each of those rows its own $s \approx 0.007$, and the error drops about 10×. `test_outlier_hurts_per_tensor_more_than_per_channel` checks it. This is why nobody ships per-tensor int4, and why activation outliers (LLM.int8(), SmoothQuant) are the hard part of quantizing *activations*.

**GPTQ by hand.** One output row $w = (0.8, 0.3)$, two inputs with unit variance and correlation 0.9, so $H = \begin{pmatrix} 1 & 0.9 \\ 0.9 & 1 \end{pmatrix}$. To keep the numbers clean, use a fixed grid of multiples of 0.5 (a symmetric grid with $s = 0.5$).

*RTN:* $\hat w = (1.0, 0.5)$, $\delta = (0.2, 0.2)$. Error $\delta^\top H \delta = 0.04 + 0.04 + 2 \cdot 0.9 \cdot 0.04 = 0.152$. Both errors push the output the same way, and the inputs move together, so they add up.

*GPTQ:* $H^{-1} = \frac{1}{0.19}\begin{pmatrix} 1 & -0.9 \\ -0.9 & 1 \end{pmatrix} = \begin{pmatrix} 5.263 & -4.737 \\ -4.737 & 5.263 \end{pmatrix}$.
Column 1: $\operatorname{quant}(0.8) = 1.0$, $e = 0.8 - 1.0 = -0.2$. Compensate column 2: $w_2 \leftarrow 0.3 - (-0.2)\cdot(-4.737/5.263) = 0.3 - 0.18 = 0.12$. The first weight overshot by 0.2 on an input that is 90% correlated with the second, so the second weight is lowered by $0.9 \times 0.2$.
Column 2: $\operatorname{quant}(0.12) = 0$. Final $\hat w = (1.0, 0)$, $\delta = (0.2, -0.3)$.
Error: $0.04 + 0.09 + 2 \cdot 0.9 \cdot (0.2)(-0.3) = 0.13 - 0.108 = 0.022$.

Same grid, same number of bits, **7× lower output error**, even though the weight error $\lVert\delta\rVert^2 = 0.13$ is *larger* than RTN's 0.08. GPTQ is not trying to keep the weights close. It keeps the outputs close. (Check the step formula: the first step adds $e^2/[H^{-1}]_{11} = 0.04/5.263 = 0.0076$, the second adds $0.12^2 / 1 = 0.0144$; total $0.022$.) `test_gptq_lesson_hand_example` runs exactly this.

**AWQ by hand.** One row $w = (0.2, 0.9)$, int3 symmetric ($q_{\max} = 3$), one scale for the row. The first input channel is large, typically $|x_1| = 10$; the second small, $|x_2| = 1$. Take the input $x = (10, 1)$; the true output is $2 + 0.9 = 2.9$.

*RTN:* $s = 0.9/3 = 0.3$; $w/s = (0.667, 3) \to (1, 3)$; $\hat w = (0.3, 0.9)$; output $3 + 0.9 = 3.9$, error $1.0$. The small weight on the big channel was rounded by 0.1, and the big channel multiplies that by 10.

*AWQ with channel scales* $s = (2, 0.5)$ (the kind of vector the search produces: up-weight the big channel, down-weight the small one; $\max s \cdot \min s = 1$): $w \odot s = (0.4, 0.45)$; grid step $0.45/3 = 0.15$; codes $(2.67 \to 3, 3)$; values $(0.45, 0.45)$; divide by $s$: $\hat w = (0.225, 0.9)$. Output $2.25 + 0.9 = 3.15$, error $0.25$, **4× smaller**. The effective step on channel 1 became $0.15/2 = 0.075$ instead of $0.3$.

**MXFP4 by hand.** Block $(0.1, -0.7, 2.9, 0.26)$ (block size 4 for the example; the real format uses 32). $\max|x| = 2.9$, $\lfloor\log_2 2.9\rfloor = 1$, so $e = 1 - 2 = -1$, scale $0.5$. Divide: $(0.2, -1.4, 5.8, 0.52)$. Nearest E2M1 values: $(0, -1.5, 6, 0.5)$. Times the scale: $(0, -0.75, 3.0, 0.25)$. The small entry 0.1 vanished, the large one landed within 0.1. `test_mxfp4_lesson_example_and_grid` runs this example.

**Bytes, 7B model, int4, group 128.** Codes: $7 \times 10^9 \times 4 / 8 = 3.5$ GB. Scales: one fp16 per 128 weights, $7\times10^9/128 \times 2 = 0.11$ GB. Total 3.61 GB, an effective **4.125 bits per weight** (4.25 with an fp16 zero-point too). Against 14 GB in bf16, that is 3.9× fewer bytes. On the A100 roofline of 08.6 (2.039 TB/s), the batch-1 decode step falls from about 6.9 ms to about 1.8 ms, *if* the kernel reads only the packed bytes and its dequantization work hides behind the memory traffic.

## 4. Tensor shapes, dtype, device

| Tensor | Shape | dtype | Device |
|---|---|---|---|
| weight `W` of an `nn.Linear` | $(d_{\text{out}}, d_{\text{in}})$ | bf16/fp32 before; packed int4 (e.g. 8 codes per int32) after | GPU |
| scales, per channel / per group | $(d_{\text{out}}, 1)$ / $(d_{\text{out}}, d_{\text{in}}/g)$ | fp16 (int formats); uint8 exponent (MX) | GPU, next to the codes |
| zero-points (asymmetric) | same as scales | packed int4 or fp16, format-dependent | GPU |
| calibration inputs `X` of one layer | $(N, d_{\text{in}})$, $N$ = calibration sequences × length, e.g. $128 \times 2048$ | fp32 for the statistics | GPU during quantization only |
| Hessian `H = X^T X` | $(d_{\text{in}}, d_{\text{in}})$; 4096² fp32 = 67 MB | fp32 (our code: fp64) | GPU during quantization only |
| AWQ channel scales `s` | $(d_{\text{in}},)$ | fp16, folded into the previous op | gone after export |
| dequantized tile inside a W4A16 kernel | a tile of $W$ | fp16/bf16, in registers / shared memory | SM, never written to HBM |

`llmre.model.quantize` stores everything in float ("fake quantization"): the values in the weight are exactly the dequantized grid values, which is what a W4A16 kernel multiplies with. It does not pack codes into int32 words. Packing changes the bytes, not the numbers.

## 5. From scratch: `llmre.model.quantize`

The grid functions return codes and scales so you can inspect them:

```python
def quantize_symmetric(w, bits, dim=None):
    qmax = 2 ** (bits - 1) - 1
    amax = w.abs().max() if dim is None else w.abs().amax(dim=dim, keepdim=True)
    scale = amax / qmax
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))   # all-zero group
    q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return q, scale                         # dequantize: q * scale
```

`quantize_asymmetric`, `quantize_groupwise` (reshape $(d_{\text{out}}, d_{\text{in}})$ to $(d_{\text{out}}, d_{\text{in}}/g, g)$ and reduce over the last axis) and `rtn_quantize` follow the same pattern.

GPTQ is the update of section 2.3, column by column, for all rows at once:

```python
def gptq_quantize(w, x, bits=4, group_size=None, symmetric=True, damp=0.01):
    W = w.detach().double().clone()                  # (out, in), updated in place
    H = x.double().T @ x.double()                    # (in, in)
    H += damp * torch.diag(H).mean() * torch.eye(H.shape[0], dtype=H.dtype)
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    quant = _column_quantizer(W, bits, symmetric)    # per-row scales (refit per group)
    Q = torch.zeros_like(W)
    for i in range(W.shape[1]):
        q = quant(W[:, i])
        Q[:, i] = q
        err = (W[:, i] - q) / U[i, i]                            # (out,)
        W[:, i + 1:] -= err[:, None] * U[i, i + 1:][None, :]     # compensate the rest
    return Q.to(w.dtype)
```

`torch.cholesky_inverse(L)` computes $H^{-1}$ from the lower Cholesky factor $L$ of $H$, and the second `cholesky(..., upper=True)` gives the $U$ with $H^{-1} = U^\top U$. The file's version also handles dead inputs (columns with $H_{jj} = 0$ get weight 0), group scales and a fixed-scale mode for the hand example. The reference implementation adds *lazy batching*: it applies the updates to blocks of 128 columns at a time, which changes the speed (fewer, larger matmuls), not the result.

AWQ is a search over one scalar:

```python
def awq_quantize(w, x, bits=4, group_size=None, symmetric=True, n_grid=20):
    W, best = w.detach().double(), None
    for k in range(n_grid):
        s = awq_scales(x, alpha=k / n_grid)              # mean|x|^alpha, normalized
        w_eff = rtn_quantize(W * s[None, :], bits, group_size, symmetric) / s[None, :]
        err = layer_output_error(W, w_eff, x)
        if best is None or err < best[0]:
            best = (err, k / n_grid, w_eff)
    return best[2].to(w.dtype), best[1]
```

`quantize_gpt_weights(model, calib_idx, method, bits, group_size)` deep-copies the course GPT and quantizes every `nn.Linear` inside the blocks in forward order. Before each layer it re-collects that layer's inputs with a forward hook *from the partially quantized copy*, so later layers see, and compensate for, the errors of earlier ones. The token embedding and the tied LM head stay in full precision, as most weight-only recipes leave them.

### 5.1 Run it

```bash
cd code && PYTHONPATH=src python experiments/ptq_gpt.py
```

Part A is one $64 \times 128$ layer whose inputs are correlated and have one channel 20× larger than the rest. Output error $\lVert XW^\top - X\hat W^\top\rVert^2$:

```text
   4-bit
     RTN per-tensor       2269.328
     RTN per-channel      1189.848
     RTN group 32          677.973
     AWQ group 32          281.316
     GPTQ group 32         192.403
```

Part B trains a 4-layer, $d = 128$ GPT on a small byte corpus, quantizes its block Linears with per-channel scales, and compares each quantized copy with the original on held-out text: KL divergence between the next-token distributions, and how often the top-1 token agrees.

```text
   bits   kB     method  KL(fp||q) nats  top-1 agreement
   8    795.6   rtn         4.36e-06     99.7%
   4    402.4   rtn         1.55e-03     98.8%
   4    402.4   awq         1.51e-03     98.5%
   4    402.4   gptq        2.26e-04     99.3%
   3    304.1   rtn         8.76e-03     97.4%
   3    304.1   gptq        1.37e-03     99.0%
```

(Abridged; the exact numbers depend on the CPU's float arithmetic.)

**Expected observations.** Finer granularity always helps, and at 4 bits the gap from per-tensor to per-group is about 3×. GPTQ is the best at every bit width, by about 6–7× in KL on the model. AWQ helps a lot on the layer with a big outlier channel (part A) but barely on the tiny trained model (part B), which has no such channels. That matches AWQ's own premise: it pays off when activation magnitudes differ a lot across channels, as they do in large LLMs. 8 bits is essentially free. We measure KL to the original model instead of held-out loss because the change we want to see (about $10^{-3}$ nats) is smaller than the noise in a tiny corpus's loss.

## 6. What the production tools do, and what it costs

**The tools** (*publicly documented*). vLLM lists GPTQ (via GPTQModel), AWQ (AutoAWQ), bitsandbytes, GGUF, FP8 and the INT4 W4A16 / INT8 W8A8 schemes of LLM Compressor among its supported formats ([vLLM quantization docs](https://docs.vllm.ai/en/latest/features/quantization/)). The workflow is the one in section 5: load the model, run a few hundred calibration sequences through it, quantize layer by layer, save packed codes plus scales, and serve with a fused **dequantize-then-matmul** kernel. Typical settings for int4 weight-only: symmetric or asymmetric, group size 128, GPTQ or AWQ, embeddings and LM head left in 16 bits.

**What W4A16 does at inference.** The "A16" means the activations stay in 16 bits. The kernel loads packed int4 codes and fp16 scales from HBM, unpacks and rescales them in registers, and runs an ordinary fp16 matmul. So the *bytes* drop about 4×, but the *FLOPs* do not change at all.

- At **small batch** (memory-bound decode), the step time follows the bytes: up to about 3.9× faster for the 7B example in section 3, minus the unpack overhead.
- At **large batch** (compute-bound), the matmul is limited by FLOPs, which are the same, plus the unpacking work, so W4A16 can be *slower* than bf16 there. To save compute you need low-bit *activations* as well (W8A8 int8, FP8, FP4), which brings back the activation-outlier problem of LLM.int8() and SmoothQuant.

**Quantization cost.** GPTQ per layer: forming $H$ is $O(N d_{\text{in}}^2)$, the inverse and Cholesky $O(d_{\text{in}}^3)$, the column loop $O(d_{\text{out}} d_{\text{in}}^2)$. For a $4096 \times 4096$ layer that is about $7 \times 10^{10}$ FLOPs for the factorization, less than a second on a GPU. The GPTQ paper quantizes a 175B model in about four GPU-hours (authors' measurement). Memory: one $H$ per layer, 67 MB at $d_{\text{in}} = 4096$ in fp32. AWQ is cheaper still: 20 RTN passes per layer and no matrix inverse.

**PTQ vs quantization-aware training (QAT).** PTQ never changes the training run, so it cannot teach the model to tolerate its grid. QAT simulates the quantization in the forward pass during (some of) training and lets gradients flow through the rounding with a straight-through estimator, so the weights settle where rounding hurts least. QAT costs training compute. PTQ costs minutes.
*Real-world example (2026-11-15, company claim):* Moonshot's Kimi K3.1 Pro model card ([Hugging Face](https://huggingface.co/moonshotai/Kimi-K3.1-Pro)) lists "MXFP4 weights / MXFP8 activations (quantization-aware training)" for its open-weight release, so the shipped checkpoint was trained to tolerate the section 2.5 block format, and is quantized in both weights and activations, rather than rounded after the fact. The card does not publish the QAT recipe or a PTQ comparison.

**Documented vs inferred.** *Publicly documented:* the GPTQ, AWQ, OBC, LLM.int8() and SmoothQuant algorithms and their reported results (authors' measurements); the MX formats (OCP spec, Rouhani et al.); vLLM's format list; the Kimi K3.1 Pro card entry (company claim). *Reasonable industry practice:* int4 weight-only with group size 128 as the default for local and small-batch serving; keeping embeddings and the LM head in 16 bits; checking a quantized model with a KL or accuracy comparison against the original on your own data, not only on the quantizer's calibration set. *Inference:* the 7B speedup in section 3, which is a roofline bound; real kernels lose some of it to unpacking, and the gain shrinks as batch grows.

## 7. Side by side, and when to keep the original

| | bf16 (08.3, 08.6) | RTN int8 / int4 | GPTQ | AWQ | QAT (e.g. MXFP4) |
|---|---|---|---|---|---|
| needs data | no | no | ~100s of sequences | ~100s of sequences | a training run |
| objective | (none) | weight error | output error via $H$ | output error, scale search | the training loss itself |
| quantization cost | none | seconds | minutes to hours | minutes | training compute |
| bytes per weight | 2 | ~1 / ~0.52 | ~0.52 | ~0.52 | ~0.53 (MXFP4) |
| quality at 4 bits | reference | worst | best of the PTQ methods here | close to GPTQ on LLMs with outlier channels | best, if done |
| compute saved | none | only with low-bit activations | same | same | yes, if activations are quantized too |

**When you would still use the original.** Keep bf16 for training (08.3) and for any number you want to trust exactly: the reference in evaluations, the policy being updated in RL, and the teacher whose log-probs you distill from, unless you have measured that its quantized version agrees with it closely enough (16.4 shows how sampler and trainer disagreements bias an update). Keep bf16 at large batch where the step is compute-bound and W4A16 saves nothing. And keep RTN int8 when you need something quick: at 8 bits, as part B shows, the method hardly matters.

## Exercise: bytes and time for a 70B model

A 70B dense model is served at batch 1 on GPUs with 80 GB each and 3.35 TB/s of bandwidth (an H100 figure). Compute the weight bytes and the roofline decode-step time (weights only, ignore the KV cache) for (a) bf16, (b) int8 per-channel, (c) int4 with group size 128, symmetric. How many GPUs does each need just to hold the weights?

<details><summary>Hint</summary>

Bytes per weight: 2, then 1 plus one fp16 scale per row (negligible), then 0.5 plus 2 bytes per 128 weights. Time = bytes / bandwidth, and with $k$ GPUs in tensor parallel, each reads $1/k$ of the bytes at the same time.

</details>

<details><summary>Stronger hint</summary>

int4 g128: $4.125/8 = 0.516$ bytes per weight. 70B × 0.516 = 36.1 GB. One GPU reading 36.1 GB at 3.35 TB/s takes about 10.8 ms.

</details>

<details><summary>Solution</summary>

(a) 140 GB: does not fit on one 80 GB GPU, needs 2 (in practice more, to leave room for the KV cache). Per step, 140 GB / (2 × 3.35 TB/s) ≈ 20.9 ms. (b) ≈ 70 GB: fits on one GPU only just, with almost no room for the cache; 70 / 3.35 ≈ 20.9 ms on one GPU. (c) ≈ 36.1 GB: fits on one GPU with 44 GB to spare for the cache; ≈ 10.8 ms. So int4 on one GPU decodes about twice as fast as bf16 split over two. That is why 4-bit weight-only quantization is the default for single-GPU serving of large models, and also why at large batch, where compute dominates, the comparison changes.

</details>

## Implementation exercise: GPTQ with activation order

The GPTQ reference code has an option, `act_order` ("desc_act"), that quantizes columns in order of decreasing $H_{jj}$ instead of left to right: the columns with the largest inputs go first, while there are still many free columns left to absorb their errors. Write `gptq_quantize_act_order(w, x, bits)` in a new function (do not change `gptq_quantize`): permute the columns of $W$ and the rows and columns of $H$ by `torch.argsort(torch.diag(H), descending=True)`, run the same loop, and undo the permutation on the result. Test that with a diagonal $H$ it still equals RTN. Then compare its output error with plain per-channel GPTQ on the part A layer of `ptq_gpt.py`, at 4 and 3 bits.

<details><summary>Hint</summary>

Permuting coordinates does not change $\delta^\top H \delta$ if you permute $H$ the same way: $\delta_\pi^\top H_{\pi\pi} \delta_\pi = \delta^\top H \delta$. Only the order of the greedy updates changes.

</details>

<details><summary>Stronger hint</summary>

`perm = torch.argsort(torch.diag(H), descending=True)`; `inv = torch.argsort(perm)`; call the existing function with `hessian=H[perm][:, perm]` on `w[:, perm]`, then return `q[:, inv]`. Per-channel scales are per row, so they do not depend on the column order. Group scales do, which is why the reference code needs extra bookkeeping to combine `act_order` with groups.

</details>

<details><summary>Solution</summary>

```python
def gptq_quantize_act_order(w, x, bits=4, symmetric=True, damp=0.01):
    H = x.double().T @ x.double()
    perm = torch.argsort(torch.diag(H), descending=True)
    inv = torch.argsort(perm)
    q = gptq_quantize(w[:, perm], hessian=H[perm][:, perm], bits=bits,
                      symmetric=symmetric, damp=damp)
    return q[:, inv]
```

With a diagonal $H$ every $U_{ij}$ off the diagonal is still zero, so the result is RTN in any order. On the outlier layer, the outlier column goes first and its error is spread over all 127 others. Expect a modest further reduction (on that layer, about 2–5% lower output error for three different seeds), not a large one. The method is greedy, and the order is a heuristic, not an optimum.

</details>

## Debugging exercise

A colleague's GPTQ is never better than RTN, and on many layers it is worse:

```python
H = x.double().T @ x.double()
H += 0.01 * torch.diag(H).mean() * torch.eye(H.shape[0], dtype=H.dtype)
U = torch.linalg.cholesky(H, upper=True)          # factor of H
for i in range(W.shape[1]):
    q = quant(W[:, i]); Q[:, i] = q
    err = (W[:, i] - q) / U[i, i]
    W[:, i + 1:] -= err[:, None] * U[i, i + 1:][None, :]
```

What is wrong?

<details><summary>Answer</summary>

`U` is the Cholesky factor of $H$, not of $H^{-1}$. The compensation needs the correlations as seen through the *inverse* Hessian. In the hand example, $H_{12} = +0.9$ but $[H^{-1}]_{12} = -4.737$: the sign is opposite. So this code pushes the second weight *up* after the first overshoots, and the errors add instead of cancelling. Fix: `U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)`. `test_gptq_lesson_hand_example` catches this: with $U$ from $H$, the second weight moves from 0.3 up to 0.48 and rounds to 0.5, so the result is RTN's $(1.0, 0.5)$ instead of $(1.0, 0)$; `test_gptq_with_diagonal_hessian_is_rtn` would not, because with a diagonal $H$ both factors are diagonal.

</details>

## Common mistakes

- **Judging a quantized model by weight error.** GPTQ's weights can be further from the originals than RTN's while its outputs are closer (section 3). Measure the outputs: layer error, KL to the original model, task accuracy.
- **Calibrating on the wrong data.** $H$ and AWQ's activation statistics come from the calibration set. Calibrate a code model on prose and the channels that matter for code get the coarse grid. Use text like the deployment's. Then evaluate on *different* text, because the calibration set is also the training set of these methods.
- **Quoting "4 bits" without the overhead.** Scales and zero-points add 0.125–0.25 bits per weight at group 128, more at smaller groups. Embeddings and the LM head usually stay in 16 bits and can be a large share of a small model.
- **Expecting W4A16 to be faster at every batch size.** It saves bytes, not FLOPs. At large batch it can be slower than bf16.
- **Forgetting the zero-point when dequantizing** asymmetric codes: $s \cdot q$ instead of $s(q - z)$ shifts every weight by $s z$.
- **Using a quantized model as an RL sampler or distillation teacher without measuring the gap.** Its token probabilities differ from the bf16 model's. That is exactly the training–inference mismatch of 16.4, now introduced on purpose.
- **Confusing QAT with PTQ in a model card.** "Trained with MXFP4 QAT" and "we provide a GPTQ int4 checkpoint" are different claims with different quality expectations.

## What to read / what to skip

- **GPTQ** ([arXiv 2210.17323](https://arxiv.org/abs/2210.17323)): read section 3 (the three steps: arbitrary order, lazy batch updates, Cholesky reformulation) and Algorithm 1; skim the results tables for the bits-vs-perplexity trend; skip the 2-bit and ternary experiments on first read.
- **Optimal Brain Compression** ([arXiv 2208.11580](https://arxiv.org/abs/2208.11580)): read the OBQ derivation if you want the update formula of section 2.3 from first principles; skip the pruning half.
- **AWQ** ([arXiv 2306.00978](https://arxiv.org/abs/2306.00978)): read section 3 (salient channels, why scaling helps, the search); skip TinyChat unless you write kernels.
- **LLM.int8()** ([arXiv 2208.07339](https://arxiv.org/abs/2208.07339)) and **SmoothQuant** ([arXiv 2211.10438](https://arxiv.org/abs/2211.10438)): read the introductions and the outlier figures to see why *activation* quantization is the hard part; SmoothQuant's per-channel migration is AWQ's scaling trick applied for a different purpose.
- **Microscaling Data Formats** ([arXiv 2310.10537](https://arxiv.org/abs/2310.10537)): read Table 1 (the MX formats) and the conversion rule; skim the training results.
- **vLLM quantization docs** ([docs.vllm.ai](https://docs.vllm.ai/en/latest/features/quantization/)): read the hardware support table to see which format runs where.

## Code this after reading

Extend `experiments/ptq_gpt.py` with a column that applies `mxfp4_quantize` to every block weight (blocks of 32 along the input dimension) and reports the same KL and top-1 agreement. Where does MXFP4, as pure PTQ, land between int4 RTN and int4 GPTQ? Then explain in two sentences why a model released with MXFP4 QAT would land closer to bf16 than your PTQ number.

## Research connection

<div class="callout paper"><p><strong>GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers</strong> — Frantar, Ashkboos, Hoefler, Alistarh, ICLR 2023, <a href="https://arxiv.org/abs/2210.17323">arXiv 2210.17323</a>. <strong>Optimal Brain Compression</strong> — Frantar, Singh, Alistarh, NeurIPS 2022, <a href="https://arxiv.org/abs/2208.11580">arXiv 2208.11580</a>. <strong>AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration</strong> — Lin et al., MLSys 2024, <a href="https://arxiv.org/abs/2306.00978">arXiv 2306.00978</a>. <strong>LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale</strong> — Dettmers et al., NeurIPS 2022, <a href="https://arxiv.org/abs/2208.07339">arXiv 2208.07339</a>. <strong>SmoothQuant</strong> — Xiao et al., ICML 2023, <a href="https://arxiv.org/abs/2211.10438">arXiv 2211.10438</a>. <strong>Microscaling Data Formats for Deep Learning</strong> — Rouhani et al. 2023, <a href="https://arxiv.org/abs/2310.10537">arXiv 2310.10537</a>. Production: <a href="https://docs.vllm.ai/en/latest/features/quantization/">vLLM quantization</a>; QAT release example: <a href="https://huggingface.co/moonshotai/Kimi-K3.1-Pro">Kimi K3.1 Pro model card</a> (company claim).</p></div>

Open questions (2026): how far below 4 bits weight-only PTQ can go before QAT becomes necessary; how to choose calibration data for models used across many domains; and how a quantized sampler's mismatch interacts with RL training (16.4), where small probability differences are known to matter.

## Check yourself

<details><summary>Why does one large weight hurt every other weight in its group?</summary>

The scale is set by the largest $|w|$ in the group, and the rounding error of every member is up to $s/2$. A weight 100× larger than the rest makes $s$ 100× larger, so the small weights round to 0 or to one of very few levels. Smaller groups (per channel, per 128) confine the damage to the outlier's own group.

</details>

<details><summary>What does the zero-point buy, and what does it cost?</summary>

It shifts the grid so that a lopsided range (say $[-1.2, 0.8]$) uses all $2^b$ levels, giving a smaller step than a symmetric grid sized to $\max|w|$. It costs one stored number per group and a subtraction when dequantizing.

</details>

<details><summary>Write the output error of one row in terms of the weight error. What plays the role of the Hessian?</summary>

$\lVert X(\hat w - w)\rVert^2 = \delta^\top X^\top X\, \delta$ with $H = X^\top X$, built from the calibration inputs to that layer. Its diagonal measures input sizes, its off-diagonal measures correlations between inputs.

</details>

<details><summary>In one sentence, what does GPTQ do after rounding column $i$?</summary>

It adjusts all not-yet-quantized columns by $-e\,U_{ij}$ (with $e = (w_i - \hat w_i)/U_{ii}$ and $U$ the upper Cholesky factor of $H^{-1}$), the least-squares change that best cancels the new error on the calibration inputs.

</details>

<details><summary>When is GPTQ exactly round-to-nearest?</summary>

When $H$ is diagonal (uncorrelated inputs): then $U$ is diagonal and no compensation is applied. Also for the last column, which has nothing left to compensate.

</details>

<details><summary>Why can AWQ multiply weight columns by $s$ without changing the model?</summary>

Because $xW^\top = (x \oslash s)(W\operatorname{diag}(s))^\top$ exactly, and $\oslash s$ can be folded into the operation that produces $x$ (a norm's gain or the previous linear layer). Only the rounding sees the difference: columns with $s_j > 1$ get a finer effective grid.

</details>

<details><summary>A 4-bit W4A16 model is 3.5× faster than bf16 at batch 1 but no faster at batch 256. Why?</summary>

At batch 1 decode is memory-bound, so time follows the bytes read, which dropped about 4×. At batch 256 the step is compute-bound, and W4A16 does the same fp16 matmul plus unpacking work, so there is nothing to gain. Saving compute requires low-bit activations too (W8A8, FP8, FP4).

</details>

<details><summary>What is the difference between PTQ and QAT, and which one does a model card's "MXFP4 (QAT)" describe?</summary>

PTQ rounds a finished model using calibration data and no training. QAT simulates the rounding during training (straight-through gradients), so the weights adapt to the grid. "MXFP4 (QAT)" means the released weights were trained to be stored in MXFP4.

</details>

<details><summary>How many bits per weight does int4 with group size 64 and an fp16 scale and fp16 zero-point really cost?</summary>

$4 + (16 + 16)/64 = 4.5$ bits.

</details>

## Next

You can now say what a quantized checkpoint is and predict what it saves. The same bytes-versus-FLOPs reasoning continues across many GPUs in Module 9. In [16.4](lessons/module-16/lesson-04.md) you will see why a quantized RL sampler needs a correction on the trainer's side.

Continue to [09.1 · Data parallelism, all-reduce, DDP](lessons/module-09/lesson-01.md).
