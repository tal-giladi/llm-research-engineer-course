# 00.2 · Ops, broadcasting, views vs copies

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-00/lesson-01">00.1 · Tensors: shape, dtype, device</a> — you should be able to create a tensor and read its <code>.shape</code>, <code>.dtype</code>, and <code>.device</code>, and you should have met contiguous memory and strides.</p>
<p><strong>You will learn:</strong> elementwise operations and reductions (<code>sum</code>/<code>mean</code> over a chosen axis, with <code>keepdim</code>); matrix multiplication with <code>@</code>, worked by hand on tiny integers; the <strong>broadcasting</strong> rules that let differently-shaped tensors combine, shown number-by-number; and the difference between a <strong>view</strong> (a second label on the same memory) and a <strong>copy</strong>, including the classic <code>reshape</code>-after-<code>transpose</code> mistake.</p>
<p><strong>Why this matters for ML:</strong> a forward pass through a language model is a chain of exactly these operations — add a bias (broadcast), multiply by a weight matrix (<code>@</code>), average over a dimension (reduction), reshape activations into heads (views). Reading model code means reading these ops and tracking how each one changes the shape. Broadcasting and view/copy confusion cause a large fraction of real bugs, so we make both explicit.</p>
</div>

## 1. Elementwise operations

The simplest operations act on each element independently and return a tensor of the same shape. Arithmetic with a scalar applies to every element; arithmetic between two same-shape tensors pairs them up.

```python
import torch
a = torch.tensor([1., 2., 3., 4.])

a + 10        # tensor([11., 12., 13., 14.])   add 10 to each element
a * a         # tensor([ 1.,  4.,  9., 16.])   multiply element-by-element
```

Note that `a * a` is **elementwise** multiplication (each element squared), *not* matrix multiplication. This is a frequent source of confusion for people coming from a math background: in PyTorch, `*` is always elementwise (the Hadamard product), and matrix multiplication is a different operator, `@`, which we get to in section 3. Keep them straight: `*` never changes the shape; `@` follows the matrix-multiply shape rule.

## 2. Reductions: collapsing an axis

A **reduction** combines many values into fewer — `sum`, `mean`, `max`, and friends. The key question is always *which axis* you reduce over, because that decides the output shape. Take

```python
m = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])          # shape (2, 3)
```

With no axis argument, the reduction collapses **everything** to a scalar:

```python
m.sum()          # tensor(21.)   = 1+2+3+4+5+6
```

With a `dim`, it collapses **only that axis**, and that axis disappears from the shape:

```python
m.sum(dim=0)     # tensor([5., 7., 9.])    sum down the rows -> shape (3,)
m.sum(dim=1)     # tensor([ 6., 15.])      sum across the columns -> shape (2,)
```

The rule that keeps this straight: **`dim=k` removes axis `k`**. `m` has shape `(2, 3)`. `sum(dim=0)` removes axis 0 (the length-2 axis), leaving shape `(3,)` — one total per column: `[1+4, 2+5, 3+6] = [5, 7, 9]`. `sum(dim=1)` removes axis 1 (the length-3 axis), leaving `(2,)` — one total per row: `[1+2+3, 4+5+6] = [6, 15]`.

### keepdim: hold the collapsed axis open as length 1

Sometimes you reduce but want the result to keep the same number of axes, so it will broadcast cleanly back against the original (you will do exactly this in softmax and LayerNorm later). That is what `keepdim=True` is for:

```python
m.mean(dim=1, keepdim=True)
# tensor([[2.],
#         [5.]])
# shape (2, 1)  -- axis 1 kept as length 1, not removed
```

Without `keepdim` the result would be shape `(2,)`; with it, `(2, 1)`. The values are the row means, $\tfrac{1+2+3}{3}=2$ and $\tfrac{4+5+6}{3}=5$. Keeping the axis as length 1 is what lets you then write `m - m.mean(dim=1, keepdim=True)` and have it broadcast — which is the next topic.

## 3. Matrix multiplication with `@`

Matrix multiplication is the workhorse of neural networks (every `Linear` layer is one). The operator is `@`. The shape rule is:

$$
(n, k) \mathbin{@} (k, p) \rightarrow (n, p).
$$

The two **inner** dimensions must match (both `k` here) — they are what gets summed over — and they vanish; the two **outer** dimensions `n` and `p` survive as the result's shape. Entry $(i, j)$ of the result is the dot product of row $i$ of the left matrix with column $j$ of the right.

Let us do a concrete one by hand and check it. Take a $(2, 3)$ times a $(3, 4)$, which must give $(2, 4)$:

$$
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}, \qquad
B = \begin{bmatrix} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \\ 9 & 10 & 11 & 12 \end{bmatrix}.
$$

Inner dimensions: $A$ is $(2, \mathbf{3})$, $B$ is $(\mathbf{3}, 4)$ — the 3's match, so this is legal, and the result is $(2, 4)$. Compute the top-left entry, row 1 of $A$ dotted with column 1 of $B$:

$$
(A@B)_{11} = 1\cdot 1 + 2\cdot 5 + 3\cdot 9 = 1 + 10 + 27 = 38.
$$

One more, row 2 of $A$ dotted with column 4 of $B$:

$$
(A@B)_{24} = 4\cdot 4 + 5\cdot 8 + 6\cdot 12 = 16 + 40 + 72 = 128.
$$

Doing all eight entries gives

$$
A@B = \begin{bmatrix} 38 & 44 & 50 & 56 \\ 83 & 98 & 113 & 128 \end{bmatrix}.
$$

PyTorch agrees exactly:

```python
A = torch.tensor([[1., 2., 3.], [4., 5., 6.]])          # (2, 3)
B = torch.tensor([[1., 2., 3., 4.],
                  [5., 6., 7., 8.],
                  [9., 10., 11., 12.]])                 # (3, 4)
A @ B
# tensor([[ 38.,  44.,  50.,  56.],
#         [ 83.,  98., 113., 128.]])
(A @ B).shape   # torch.Size([2, 4])
```

<div class="callout warn"><p>The number-one shape error you will see is a matmul with mismatched inner dimensions:<br><code>RuntimeError: mat1 and mat2 shapes cannot be multiplied (2x3 and 4x5)</code>. Read it as "I tried $(2,\mathbf 3)@(\mathbf 4,5)$ and the inner dims 3 and 4 do not match." When a matmul raises, print both operands' <code>.shape</code> and line up the inner two numbers.</p></div>

## 4. Broadcasting

You constantly need to combine tensors of *different* shapes — add a length-$C$ bias vector to a whole batch of activations, subtract a per-row mean from a matrix. **Broadcasting** is the set of rules that lets PyTorch do this without you manually copying data.

The rule: line the two shapes up **from the right**. For each axis, the lengths are compatible if they are **equal**, or if **one of them is 1**. A length-1 axis is stretched (virtually repeated) to match the other. If neither is 1 and they differ, it is an error.

Worked example: a column vector plus a row vector.

$$
\underbrace{\begin{bmatrix} 10 \\ 20 \\ 30 \end{bmatrix}}_{(3,\,1)} \;+\; \underbrace{\begin{bmatrix} 1 & 2 & 3 & 4 \end{bmatrix}}_{(1,\,4)} \;\rightarrow\; (3,\,4).
$$

Align from the right: axis 1 is `1` vs `4` → one is 1, so it stretches to 4. Axis 0 is `3` vs `1` → one is 1, so it stretches to 3. Result shape `(3, 4)`. The left operand is virtually copied across the 4 columns; the right is virtually copied down the 3 rows; then they add elementwise. Number by number, entry $(i, j)$ is `left[i] + right[j]`:

$$
\begin{bmatrix}
10{+}1 & 10{+}2 & 10{+}3 & 10{+}4 \\
20{+}1 & 20{+}2 & 20{+}3 & 20{+}4 \\
30{+}1 & 30{+}2 & 30{+}3 & 30{+}4
\end{bmatrix}
=
\begin{bmatrix}
11 & 12 & 13 & 14 \\
21 & 22 & 23 & 24 \\
31 & 32 & 33 & 34
\end{bmatrix}.
$$

```python
aa = torch.tensor([[10.], [20.], [30.]])   # (3, 1)
bb = torch.tensor([[1., 2., 3., 4.]])      # (1, 4)
aa + bb
# tensor([[11., 12., 13., 14.],
#         [21., 22., 23., 24.],
#         [31., 32., 33., 34.]])
(aa + bb).shape   # torch.Size([3, 4])
```

Crucially, **no data is actually copied** — broadcasting is done by pretending a length-1 axis has stride 0 (stepping along it does not move in memory), so it costs no extra memory. That efficiency is why adding a bias to a `(B, T, C)` activation is written simply as `x + bias` with `bias` of shape `(C,)`: aligned from the right, `(B, T, C) + (C,)` stretches the bias across every batch and every position for free.

<div class="callout key"><p>Broadcasting rule: align shapes from the right; each axis pair must be equal or have a 1 (which stretches). Missing leading axes are treated as 1. It is how a small vector combines with a big tensor without a loop and without copying memory.</p></div>

## 5. Views vs copies

This is the subtlest idea in the lesson and a genuine bug factory, so go slowly. Recall from lesson 1 that a tensor's numbers live in one flat buffer, and the shape/stride are just metadata describing how to read it. Some operations give you a **new label on the same buffer** (a *view*); others allocate a **fresh buffer** (a *copy*). A view that you mutate changes the original, because they *are* the same memory.

### view and reshape

`view` gives you a differently-shaped label on the same data — no copy:

```python
t = torch.arange(6)          # tensor([0, 1, 2, 3, 4, 5])
v = t.view(2, 3)             # a (2,3) VIEW of the same 6 numbers
v[0, 0] = 99                 # mutate through the view...
t                            # tensor([99,  1,  2,  3,  4,  5])   <-- t changed too!
```

Writing `99` through `v` changed `t`, because `v` and `t` are two labels on one buffer. This is the behaviour to internalise: **a view shares storage with its parent.** It is a feature — it is what makes reshaping free — but if you did not expect the aliasing it looks like spooky action at a distance.

`reshape` does the same job but is more forgiving: it returns a view when it can (data already laid out compatibly) and silently makes a copy when it cannot. So `reshape` "always works" but you no longer know for sure whether you got a view or a copy. `view` "only works when it can be a genuine view" and errors otherwise — which is sometimes exactly the guarantee you want.

### permute / transpose: views that reorder axes

`transpose(i, j)` swaps two axes; `permute(...)` reorders all of them. Both return **views** — they do not move data, they just produce new strides:

```python
m = torch.arange(6).view(2, 3)   # [[0,1,2],[3,4,5]], stride (3,1), contiguous
mt = m.transpose(0, 1)           # a (3,2) view, stride (1,3)
mt.is_contiguous()               # False  <-- the flat order no longer matches the grid
```

After the transpose, reading `mt` left-to-right, top-to-bottom would visit the flat buffer as `0, 3, 1, 4, 2, 5` — not in flat order. The tensor is now **non-contiguous**: its logical layout and its physical layout disagree. That is legal and fine for most operations, but it sets up the classic mistake.

### The classic mistake: reshape/view after transpose

```python
mt = torch.arange(6).view(2, 3).transpose(0, 1)   # (3,2), non-contiguous
mt.view(6)
# RuntimeError: view size is not compatible with input tensor's size and
# stride (at least one dimension spans across two contiguous subspaces).
# Use .reshape(...) instead.
```

`view` refuses, because there is no way to relabel the *existing* non-contiguous buffer as a flat length-6 array without reordering the numbers. Two fixes:

```python
mt.reshape(6)                    # works: reshape makes a copy when it must
# tensor([0, 3, 1, 4, 2, 5])
mt.contiguous().view(6)          # works: .contiguous() first copies into flat order
# tensor([0, 3, 1, 4, 2, 5])
```

`.contiguous()` allocates a fresh buffer with the numbers laid out in the current logical order, giving you a contiguous tensor that `view` will happily reshape. You will see the pattern `x.transpose(1, 2).contiguous().view(B, T, C)` in the multi-head attention code of [module 5](lessons/module-05/lesson-03.md) for exactly this reason: after transposing the head axis back, the tensor is non-contiguous, so you make it contiguous before collapsing the heads back into the channel dimension.

<div class="callout warn"><p>If a <code>.view(...)</code> raises "view size is not compatible with input tensor's size and stride", the input is non-contiguous — almost always because you just <code>transpose</code>d or <code>permute</code>d it. Either call <code>.contiguous()</code> before <code>.view(...)</code>, or use <code>.reshape(...)</code> which copies when needed. Do not reach for <code>.reshape</code> blindly everywhere, though: if you rely on a view aliasing the original and <code>reshape</code> quietly hands you a copy, your in-place mutation will silently fail to propagate.</p></div>

## Common mistakes

- **Using `*` when you meant `@`** (or vice versa). `*` is elementwise and keeps the shape; `@` is matrix multiply and follows the inner-dimension rule. A model that "runs but learns nothing" sometimes has an elementwise product where a matmul belonged.
- **Reducing the wrong axis.** `sum(dim=0)` vs `sum(dim=1)` give different shapes and different meanings. Always ask which axis you want *gone*.
- **Forgetting `keepdim`** when you plan to broadcast the reduced tensor back — you get a shape that fails to line up (or, worse, lines up wrongly by accident).
- **Assuming a view is a copy.** Mutating a view mutates the parent. If you need an independent tensor, call `.clone()`.

## Exercise

You have a batch of activations `x` of shape `(B, T, C) = (2, 4, 6)` and want to compute, for each position, the mean over the `C` channels and subtract it (a baby version of the centring step inside LayerNorm). Write the one line that produces a `(2, 4, 6)` tensor where each length-6 channel vector has had its own mean removed.

<em>Optional hint:</em> you need to reduce over the last axis, and the result must line up with `x` for the subtraction.

<em>Stronger hint:</em> reduce with `dim=-1` (the channel axis) and think about what `keepdim` does to the shape so it broadcasts back against `(2, 4, 6)`.

<details><summary>Solution</summary>

```python
x_centered = x - x.mean(dim=-1, keepdim=True)
```

`x.mean(dim=-1, keepdim=True)` reduces the last axis to length 1, giving shape `(2, 4, 6) -> (2, 4, 1)`. Subtracting it from `x` broadcasts that `(2, 4, 1)` across the 6 channels (the length-1 axis stretches to 6), so every channel vector gets its own per-position mean removed, and the result is back to `(2, 4, 6)`. Without `keepdim=True` the mean would be `(2, 4)`, which does **not** align from the right with `(2, 4, 6)` and would raise or broadcast incorrectly.

</details>

## Check yourself

<details><summary>For <code>x</code> of shape (5, 3), what shapes do <code>x.sum(dim=0)</code>, <code>x.sum(dim=1)</code>, and <code>x.sum(dim=1, keepdim=True)</code> have?</summary>

`sum(dim=0)` removes axis 0 → shape `(3,)`. `sum(dim=1)` removes axis 1 → shape `(5,)`. `sum(dim=1, keepdim=True)` keeps axis 1 as length 1 → shape `(5, 1)`.

</details>

<details><summary>Can you compute <code>A @ B</code> for A of shape (4, 5) and B of shape (4, 5)? What about (4, 5) @ (5, 2)?</summary>

`(4, 5) @ (4, 5)` is illegal: the inner dimensions 5 and 4 do not match. `(4, 5) @ (5, 2)` is legal — inner dims 5 and 5 match — and gives `(4, 2)`.

</details>

<details><summary>Do shapes (3, 1) and (1, 4) broadcast? What is the result shape? What about (2, 3) and (3,)?</summary>

`(3, 1)` and `(1, 4)` broadcast to `(3, 4)` (each axis has a 1 that stretches). `(2, 3)` and `(3,)` align from the right: `3` vs `3` match, and the missing leading axis of the second is treated as 1, so it broadcasts to `(2, 3)`.

</details>

<details><summary>You run <code>y = x.view(2, 3)</code> then <code>y[0, 0] = 0</code>, and later find <code>x</code> also changed. Why? How would you have avoided it?</summary>

`view` returns a view that shares storage with `x`, so mutating `y` mutates the same underlying buffer that `x` labels. To get an independent tensor, use `x.view(2, 3).clone()` (or `x.clone().view(2, 3)`), which copies the data.

</details>

<details><summary>Why does <code>t.transpose(0, 1).view(-1)</code> sometimes raise, and what are two fixes?</summary>

`transpose` returns a non-contiguous view (logical order no longer matches physical/flat order), and `view` cannot relabel a non-contiguous buffer as flat without reordering. Fix by inserting `.contiguous()` before `.view(-1)`, or by using `.reshape(-1)`, which copies when it must.

</details>

## Next

You can now do the arithmetic, reductions, matmuls, broadcasts, and reshapes that make up a forward pass, and you know when two tensors secretly share memory. What we have not touched is how a network *learns*: how PyTorch automatically computes the derivative of a loss with respect to every parameter. That is **autograd**, and it is the reason PyTorch exists.

Continue to [00.3 · Autograd: backward and grad](lessons/module-00/lesson-03.md).
