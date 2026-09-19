# 12.1 · RoPE (Rotary Position Embedding)

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="../module-05/lesson-02.md">05.2 · Q/K/V &amp; scaled dot-product attention</a> (the score is a dot product $\mathbf{q}\cdot\mathbf{k}$); <a href="../module-05/lesson-01.md">05.1 · Embeddings &amp; positional encoding</a> (absolute learned position embeddings, the <code>wpe</code> table); the $(B, n_h, T, d_h)$ head layout from <a href="../module-05/lesson-03.md">05.3 · Multi-head attention</a>.</p>
<p><strong>You will learn:</strong> why absolute learned position embeddings do not generalize past their trained length and do not encode <em>relative</em> position; how RoPE injects position by <em>rotating</em> each query and key in 2-D coordinate pairs by an angle proportional to position; the one-line proof that the rotated dot product depends only on the offset $m-n$; a 2-D rotation worked by hand; and the LLaMA complex-number implementation line by line.</p>
<p><strong>Why this matters for ML:</strong> RoPE is the position encoding in nearly every current open LLM — LLaMA, Llama&nbsp;3, Mistral, Qwen, DeepSeek. It is the first change you make when turning a GPT-2 into a modern model, and it is what lets a model trained at 4K context be stretched to 32K+ with light adaptation.</p>
</div>

## 1. Intuition: why the GPT-2 scheme runs out of room

In Module 5 a token's input vector was the sum of two learned vectors: a **token embedding** (which word) and a **positional embedding** looked up from a table `wpe` of shape `(block_size, C)` (which slot). Row 0 is "position 0", row 1 is "position 1", and so on. This works, but it has two structural problems.

**It cannot extend.** The table has exactly `block_size` rows. If you trained with `block_size = 1024`, there is no row 1024 — position 1024 has never been seen and has no vector. The model simply cannot represent a context longer than it was trained on.

**It is absolute, not relative.** The model learns what "position 5" and "position 200" look like as fixed vectors, but the thing that usually matters linguistically is the *gap* between two tokens — "the adjective three words back", "the subject this verb agrees with". Nothing in the absolute scheme makes "5 apart" look the same whether it happens at positions (10, 15) or (300, 305). The model has to relearn every relationship at every absolute location.

**RoPE** (Rotary Position Embedding, Su et al. 2021) fixes both at once with a different idea: do not *add* a position vector at the input. Instead, at attention time, **rotate** each query and key vector by an angle that grows with its position. Because a dot product between two rotated vectors depends only on the *difference* of their rotation angles, the attention score automatically becomes a function of the relative offset — and rotation is defined for any position, so there is no table to run out of.

<div class="callout key"><p>Absolute embeddings add a per-position vector once, at the input, and live in a fixed-size table. RoPE rotates Q and K by a position-dependent angle at attention time, so the score $\mathbf{q}_m\cdot\mathbf{k}_n$ depends only on the offset $m-n$. Nothing is added to V or to the residual stream.</p></div>

## 2. Intuition: rotating a 2-D vector by its position

Take the simplest case: a vector of just two numbers, $\mathbf{v} = (x_0, x_1)$, drawn as an arrow in the plane. **Rotating** it by an angle $\theta$ (counter-clockwise) sends it to a new arrow of the *same length*, turned by $\theta$. The formula is the 2-D rotation matrix:

$$
R(\theta) \begin{bmatrix} x_0 \\ x_1 \end{bmatrix}
= \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
  \begin{bmatrix} x_0 \\ x_1 \end{bmatrix}
= \begin{bmatrix} x_0\cos\theta - x_1\sin\theta \\ x_0\sin\theta + x_1\cos\theta \end{bmatrix}.
$$

RoPE's rule for a 2-D vector at position $m$ is: **rotate it by $\theta = m\omega$**, where $\omega$ is a fixed frequency. Position 0 leaves it alone (angle 0), position 1 turns it by $\omega$, position 2 by $2\omega$, and so on — the further along the sequence, the more the arrow is turned.

Let us turn the unit vector $\mathbf{v}=(1,0)$ with frequency $\omega = 1$ radian per position:

- **position 0**, angle $0$: $R(0)(1,0) = (1.0000,\ 0.0000)$ — unchanged.
- **position 1**, angle $1$: $(\cos 1,\ \sin 1) = (0.5403,\ 0.8415)$.
- **position 2**, angle $2$: $(\cos 2,\ \sin 2) = (-0.4161,\ 0.9093)$.

Each output has length $1$ (a rotation never changes length), just pointing in a new direction that encodes "how far along" the token is. These are exactly the numbers PyTorch prints in §7.

## 3. Mathematics: why the dot product becomes relative

Here is the whole point of RoPE in one calculation. Take a query $\mathbf{q}$ at position $m$ and a key $\mathbf{k}$ at position $n$, both 2-D. RoPE rotates them to $R(m\omega)\mathbf{q}$ and $R(n\omega)\mathbf{k}$. Their attention score is the dot product

$$
\big(R(m\omega)\mathbf{q}\big)^\top \big(R(n\omega)\mathbf{k}\big)
= \mathbf{q}^\top R(m\omega)^\top R(n\omega)\, \mathbf{k}.
$$

Two facts about rotation matrices finish it. First, $R(a)^\top = R(-a)$: rotating backward by $a$ is the transpose of rotating forward by $a$. Second, rotations compose by adding angles: $R(-m\omega)R(n\omega) = R\big((n-m)\omega\big)$. Substituting,

$$
\big(R(m\omega)\mathbf{q}\big)^\top \big(R(n\omega)\mathbf{k}\big)
= \mathbf{q}^\top R\big((n-m)\omega\big)\, \mathbf{k}.
$$

The right-hand side contains $m$ and $n$ **only through the difference $n-m$**. Two tokens the same distance apart produce the same score contribution no matter *where* in the sequence they sit. That is relative position, obtained purely by rotating each side by its own absolute angle — no relative-position table, no cross terms.

<div class="callout key"><p>$\big(R(m\omega)\mathbf{q}\big)\cdot\big(R(n\omega)\mathbf{k}\big) = \mathbf{q}^\top R\big((n-m)\omega\big)\mathbf{k}$. Absolute rotations in, relative offset out. This identity <em>is</em> RoPE.</p></div>

### 3.1 More than two dimensions: pairs and per-pair frequencies

A real head vector has $d_h = 64$ or $128$ numbers, not two. RoPE splits the $d_h$ channels into $d_h/2$ **coordinate pairs** — $(x_0,x_1), (x_2,x_3), \dots$ — and rotates each pair independently, but each pair gets its **own frequency**:

$$
\omega_p = \text{base}^{-2p/d_h}, \qquad p = 0, 1, \dots, \tfrac{d_h}{2}-1,
$$

with $\text{base} = 10000$ (the original choice, inherited from the sinusoidal embeddings of the 2017 Transformer). Pair $0$ turns fastest ($\omega_0 = 1$ radian/position); later pairs turn ever more slowly (the last pair barely moves over the whole sequence). Fast pairs resolve fine, nearby offsets; slow pairs carry coarse, long-range position — a multi-scale "clock" with many hands running at different speeds. The relative-offset identity of §3 holds for every pair separately, so it holds for the full dot product, which is just the sum over pairs.

## 4. Tensor shapes: where RoPE sits in attention

RoPE acts on the queries and keys **after** the QKV projection and head-split, when they have shape $(B, n_h, T, d_h)$, and **before** the scores $QK^\top$ are formed:

$$
\underbrace{(B, n_h, T, d_h)}_{q,\ k}
\;\xrightarrow{\text{apply\_rope}}\;
\underbrace{(B, n_h, T, d_h)}_{\text{rotated }q,\ k}
\;\xrightarrow{\text{as before}}\; QK^\top \to \text{softmax} \to \cdot V.
$$

The shape is unchanged — RoPE only rotates within each length-$d_h$ vector. It is applied to **Q and K only**: the value $V$ and the residual stream are never rotated (position should steer *who attends to whom*, not corrupt the content that gets passed along). The rotation factors are precomputed once into a table `freqs_cis` of shape $(T_{\max}, d_h/2)$, dtype `complex64`, sliced to `[:T]` each batch and broadcast over the $(B, n_h)$ axes.

## 5. From-scratch PyTorch: the complex-number trick

A 2-D rotation by $\theta$ is exactly multiplication by the unit complex number $e^{i\theta} = \cos\theta + i\sin\theta$ (written $\operatorname{cis}\theta$). If we pack a pair $(x_0, x_1)$ into the complex number $x_0 + i x_1$, then

$$
(x_0 + i x_1)\,(\cos\theta + i\sin\theta) = (x_0\cos\theta - x_1\sin\theta) + i(x_0\sin\theta + x_1\cos\theta),
$$

whose real and imaginary parts are precisely the rotated pair from §2. So "rotate every pair" becomes one complex multiply — which is why the LLaMA implementation, and our [`llmre.attention.rope`](../../code/src/llmre/attention/rope.py), is built on `torch.view_as_complex`.

```python
import torch

def precompute_freqs_cis(dim, max_seq_len, base=10000.0):
    # One frequency per pair: omega_p = base^(-2p/dim).
    exponents = torch.arange(0, dim, 2).float() / dim        # (dim/2,): 2p/dim
    freqs = 1.0 / (base ** exponents)                        # (dim/2,): omega_p
    t = torch.arange(max_seq_len).float()                    # (max_seq_len,)
    angles = torch.outer(t, freqs)                           # (max_seq_len, dim/2): m*omega_p
    return torch.polar(torch.ones_like(angles), angles)      # e^{i*angle}, complex (T, dim/2)

def apply_rope(q, k, freqs_cis):
    B, nh, T, hd = q.shape
    # Read each length-hd vector as hd/2 complex numbers (adjacent pairs).
    q_c = torch.view_as_complex(q.float().reshape(B, nh, T, hd // 2, 2))   # (B,nh,T,hd/2)
    k_c = torch.view_as_complex(k.float().reshape(B, nh, T, hd // 2, 2))
    rot = freqs_cis[:T].view(1, 1, T, hd // 2)               # broadcast over B, nh
    q_rot = torch.view_as_real(q_c * rot).flatten(3)         # rotate, back to (B,nh,T,hd)
    k_rot = torch.view_as_real(k_c * rot).flatten(3)
    return q_rot.type_as(q), k_rot.type_as(k)
```

`torch.polar(magnitude, angle)` builds a complex tensor from magnitudes (all ones — we want unit rotations) and phases (the angles), so `freqs_cis[m, p] = e^{i·m·ω_p}`. `view_as_complex` reinterprets the last axis of size 2 as one complex number *without copying*; `view_as_real` reverses it. `flatten(3)` merges the pair axis back into $d_h$.

## 6. Numerical example: the relative property, verified

Let $\mathbf{q} = (1, 2)$ and $\mathbf{k} = (3, 1)$ (one pair, $\omega = 1$). Rotate $\mathbf{q}$ by position $m$ and $\mathbf{k}$ by position $n$, then dot. The claim is that the score depends only on $m-n$. Fixing the offset at $2$ and sliding the absolute positions:

| $m$ | $n$ | offset $m-n$ | score $\big(R(m)\mathbf{q}\big)\cdot\big(R(n)\mathbf{k}\big)$ |
|-----|-----|--------------|----------------------------------------|
| 3 | 1 | 2 | $-6.627221$ |
| 5 | 3 | 2 | $-6.627221$ |
| 4 | 2 | 2 | $-6.627221$ |

Identical to six decimals — exactly what §3 promised. Change the offset and the score changes; keep the offset and it does not. (These are the printed values from `code/`; run the snippet in §7 to reproduce them.)

## 7. Under the hood: verify it yourself

The 2-D rotations of §2 and the relative property of §6, straight from Python:

```python
import torch, math

# --- Section 2: rotate (1,0) by angle m for m = 0,1,2 (omega = 1) ---
for m in [0, 1, 2]:
    th = m * 1.0
    R = torch.tensor([[math.cos(th), -math.sin(th)],
                      [math.sin(th),  math.cos(th)]])
    print(m, (R @ torch.tensor([1., 0.])).tolist())
# 0 [1.0, 0.0]
# 1 [0.5403022766, 0.8414709568]
# 2 [-0.4161468446, 0.9092974066]

# --- Section 6: score depends only on m - n ---
def rot(v, m, omega=1.0):
    th = m * omega
    R = torch.tensor([[math.cos(th), -math.sin(th)],
                      [math.sin(th),  math.cos(th)]], dtype=torch.float64)
    return R @ v

q = torch.tensor([1., 2.], dtype=torch.float64)
k = torch.tensor([3., 1.], dtype=torch.float64)
for (m, n) in [(3, 1), (5, 3), (4, 2)]:
    print(m, n, float(rot(q, m) @ rot(k, n)))   # all -6.627221...
```

**Cost.** RoPE adds essentially nothing: the `freqs_cis` table is precomputed once, and applying it is one elementwise complex multiply over the $(B, n_h, T, d_h)$ queries and keys — $O(B\,n_h\,T\,d_h)$ work, negligible next to the $O(B\,n_h\,T^2 d_h)$ of the attention scores. There are **no learned parameters** (contrast the `wpe` table, which held `block_size × C` weights). And because rotation is defined at any position, you can evaluate at sequence lengths longer than training — the basis of long-context extension methods (position interpolation, YaRN), which rescale $\omega_p$ rather than adding table rows.

<div class="callout warn"><p><strong>Pairing convention matters — but only for consistency.</strong> This implementation pairs <em>adjacent</em> channels $(x_0,x_1),(x_2,x_3),\dots$ (the <code>view_as_complex</code> convention, used by LLaMA). Some codebases (e.g. GPT-NeoX / HF's default) pair the two <em>halves</em> $(x_0, x_{d_h/2}), (x_1, x_{d_h/2+1}),\dots$ instead. Both give identical math and the same relative property, but the rotation is applied to different channels, so a checkpoint trained with one convention will be scrambled if you load it with the other. When porting weights, match the convention.</p></div>

## Common mistakes

- **Rotating V or the residual stream.** RoPE is Q/K only. Rotating values would distort the content each token contributes, not just the attention pattern.
- **Applying RoPE before the head split.** It must act on the per-head $d_h$ vectors, because the pairing and frequencies are defined per head dimension, not across the full $C$.
- **Mismatched `freqs_cis` device/length.** `freqs_cis` must be on the same device as `q` and have at least `T` rows; slice `[:T]`.
- **Odd head dimension.** RoPE rotates in pairs, so $d_h$ must be even.

## Exercise

Show, using only the two rotation facts from §3, that RoPE gives a query attending to *itself* ($m=n$) a score independent of position.

<em>Hint:</em> what is $R\big((n-m)\omega\big)$ when $m=n$?

<em>Stronger hint:</em> $R(0)$ is the identity matrix.

<details><summary>Solution</summary>

With $m=n$, the offset is $0$, so the score is $\mathbf{q}^\top R(0)\,\mathbf{k} = \mathbf{q}^\top I\,\mathbf{k} = \mathbf{q}\cdot\mathbf{k}$ — the plain, unrotated dot product, the same at every position. So a token's self-score is untouched by RoPE, which makes sense: a token is always distance $0$ from itself.

</details>

## Debugging exercise

A student writes `apply_rope` but rotates the queries and keys by *different* tables — queries with `freqs_cis` and keys with `freqs_cis.conj()` (the complex conjugate, i.e. rotating keys by $-n\omega$ instead of $+n\omega$). What relative offset does the score now depend on, and why is that wrong?

<details><summary>Answer</summary>

Rotating keys by $-n\omega$ makes the key factor $R(-n\omega)$, so the score becomes $\mathbf{q}^\top R(-m\omega)^\top \cdot R(-n\omega)\mathbf{k}$... working it out, the combined rotation is $R\big((m+n)(-\omega)\big)$-flavoured — it depends on the **sum** $m+n$, not the difference $m-n$. The score is no longer translation-invariant: the same two tokens get a different score depending on where the pair sits in the sequence. Both Q and K must be rotated in the *same* direction for the $m-n$ cancellation to happen.

</details>

## Research connection

<div class="callout paper"><p><strong>RoFormer: Enhanced Transformer with Rotary Position Embedding</strong> (Su et al. 2021) introduced RoPE. See the reading guide in <a href="../../papers/index.md">the paper curriculum</a> (#8). The full paper derives RoPE in the complex plane for arbitrary dimension; the 2-D rotation intuition here is all you need to read it. RoPE is then a building block of <strong>LLaMA</strong> (#9) and <strong>Llama&nbsp;3</strong> (#10).</p></div>

## Check yourself

<details><summary>Why can't absolute learned position embeddings handle a context longer than training?</summary>

The `wpe` table has exactly `block_size` rows, one learned vector per position. Position `block_size` and beyond have no row — they were never seen during training, so there is nothing to look up. RoPE has no table: it computes a rotation angle $m\omega$ for any $m$, so it is defined at every position.

</details>

<details><summary>RoPE is applied to which tensors, and at what point in the attention computation?</summary>

To the queries and keys only, after the QKV projection and head-split (shape $(B, n_h, T, d_h)$) and before forming the scores $QK^\top$. The values $V$ and the residual stream are not rotated.

</details>

<details><summary>With base $10000$ and $d_h = 4$ (two pairs), what are the two frequencies $\omega_0, \omega_1$?</summary>

$\omega_p = 10000^{-2p/d_h}$. For $p=0$: $10000^{0} = 1$. For $p=1$: $10000^{-2/4} = 10000^{-1/2} = 1/\sqrt{10000} = 0.01$. So pair 0 turns at $1$ rad/position and pair 1 at $0.01$ rad/position — a fast hand and a slow hand.

</details>

<details><summary>Why does rotation preserve the norm of a query vector, and why is that desirable?</summary>

A rotation matrix is orthogonal ($R^\top R = I$), so $\|R\mathbf{q}\|^2 = \mathbf{q}^\top R^\top R \mathbf{q} = \mathbf{q}^\top\mathbf{q} = \|\mathbf{q}\|^2$. Keeping the norm fixed means RoPE changes only the *direction* (the relative-angle information), not the magnitude, so it does not rescale the attention logits or destabilize softmax.

</details>

## Next

RoPE replaced the position machinery. The next two changes replace the *normalization* and the *feed-forward network*. Continue to [12.2 · RMSNorm &amp; SwiGLU](lesson-02.md).
