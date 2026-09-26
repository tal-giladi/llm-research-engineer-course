# 02.2 · Jacobians, VJPs, computational graphs

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="#/lessons/module-02/lesson-01">02.1 · Derivatives, partials, chain rule, gradients</a> — the scalar chain rule and the gradient as the vector of partials. Matrix–vector multiplication and transpose (from <a href="#/lessons/module-00/lesson-01">Module 0</a>'s tensor work) are used freely.</p>
<p><strong>You will learn:</strong> the <strong>Jacobian</strong> of a vector→vector function (the matrix of all partial derivatives, and how to read its shape as (outputs, inputs)); the <strong>vector–Jacobian product</strong> (VJP) $\mathbf{v}^\top J$ that reverse-mode autodiff actually computes, and <em>why</em> it never forms the full Jacobian; and the <strong>computational graph</strong> as nodes (operations) and edges (tensors) that PyTorch records and then walks backward. A small VJP is worked by hand and checked in PyTorch.</p>
<p><strong>Why this matters for ML:</strong> a neural network is a chain of vector→vector functions, and its loss is a single scalar at the end. Backpropagation is the chain rule applied to that chain — but done as a sequence of vector–Jacobian products, right to left, so it never has to build or store a giant Jacobian matrix. Understanding VJPs is understanding why <code>loss.backward()</code> is cheap enough to train billion-parameter models, and why the loss must be a scalar.</p>
</div>

## 1. Intuition: layers are vector→vector, the loss is the one scalar

In lesson 02.1 every function ate one number and produced one number. Real networks are not like that. A layer eats a vector (say $n$ activations) and produces a vector (say $m$ activations). Stack layers and you have a composition of vector→vector functions. Only at the very end does a single scalar appear — the **loss** — because to train we need one number to push down.

So the object we differentiate is: a chain of vector→vector maps, capped by one scalar. The scalar chain rule from 02.1 still governs it, but "the derivative of one step" is no longer a single number — for a vector→vector step it is a whole *matrix* of partials, the Jacobian. The art of reverse-mode autodiff is applying the chain rule to this chain **without ever materializing those matrices**. That is the VJP.

## 2. Mathematics: the Jacobian and its shape

Let $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$ take an input vector $\mathbf{x} = (x_1, \dots, x_n)$ to an output vector $\mathbf{f}(\mathbf{x}) = (f_1, \dots, f_m)$. The **Jacobian** $J$ collects every partial "output $i$ with respect to input $j$":

$$
J = \begin{bmatrix}
\dfrac{\partial f_1}{\partial x_1} & \cdots & \dfrac{\partial f_1}{\partial x_n} \\[8pt]
\vdots & \ddots & \vdots \\[6pt]
\dfrac{\partial f_m}{\partial x_1} & \cdots & \dfrac{\partial f_m}{\partial x_n}
\end{bmatrix}, \qquad J_{ij} = \frac{\partial f_i}{\partial x_j}.
$$

Row $i$ is the gradient of the single output $f_i$. The shape is **(outputs, inputs) $= (m, n)$** — say it out loud every time, because getting the orientation right is most of the battle. A useful mnemonic: $J$ is the matrix that linearly maps a small input change to the resulting output change, $\Delta\mathbf{f} \approx J\,\Delta\mathbf{x}$, and for that multiply to type-check ($ (m,n)\times(n,1) = (m,1)$) the shape *must* be $(m, n)$.

When $m = 1$ (a scalar output) the Jacobian is a single row — which is exactly the gradient $\nabla f$ laid on its side. So the gradient is the special case of the Jacobian for scalar-valued functions. That connection is the hinge of the whole lesson.

## 3. Numerical example: a $3\times 2$ Jacobian

Take $\mathbf{f}: \mathbb{R}^2 \to \mathbb{R}^3$ defined by

$$
\mathbf{f}(a_1, a_2) = \begin{bmatrix} a_1^2 \\ a_1 a_2 \\ a_2^2 \end{bmatrix}.
$$

Two inputs, three outputs, so $J$ is $(3, 2)$. Differentiate each output by each input:

$$
J = \begin{bmatrix}
\frac{\partial (a_1^2)}{\partial a_1} & \frac{\partial (a_1^2)}{\partial a_2} \\[4pt]
\frac{\partial (a_1 a_2)}{\partial a_1} & \frac{\partial (a_1 a_2)}{\partial a_2} \\[4pt]
\frac{\partial (a_2^2)}{\partial a_1} & \frac{\partial (a_2^2)}{\partial a_2}
\end{bmatrix}
= \begin{bmatrix} 2a_1 & 0 \\ a_2 & a_1 \\ 0 & 2a_2 \end{bmatrix}.
$$

At $\mathbf{a} = (2, 3)$:

$$
J = \begin{bmatrix} 4 & 0 \\ 3 & 2 \\ 0 & 6 \end{bmatrix}.
$$

Sanity check row by row: output $1 = a_1^2$ ignores $a_2$, so its row is $[2a_1, 0] = [4, 0]$; output $2 = a_1 a_2$ depends on both, giving $[a_2, a_1] = [3, 2]$; output $3 = a_2^2$ ignores $a_1$, giving $[0, 2a_2] = [0, 6]$. This $6\times$ matrix has $m\cdot n = 6$ entries; keep that count in mind for section 5.

## 4. The chain rule as a product of Jacobians

For a composition $\mathbf{x} \to \mathbf{y} \to \mathbf{z}$ of vector→vector maps, the chain rule becomes a **matrix product of Jacobians**:

$$
J_{\mathbf{z}/\mathbf{x}} = J_{\mathbf{z}/\mathbf{y}}\; J_{\mathbf{y}/\mathbf{x}}.
$$

This is the exact analogue of $\frac{dz}{dx} = \frac{dz}{dy}\frac{dy}{dx}$ from 02.1 — the scalar products just became matrix products, and order now matters. For a deep network with layers $L_1, L_2, \dots, L_k$ ending in a scalar loss, the gradient of the loss with respect to the first layer's input is a long product:

$$
\underbrace{J_{L/L_k}}_{(1,\,d_k)}\; \underbrace{J_{L_k/L_{k-1}}}_{(d_k,\,d_{k-1})} \cdots \underbrace{J_{L_2/L_1}}_{(d_2,\,d_1)}\; \underbrace{J_{L_1/\mathbf{x}}}_{(d_1,\,d_0)}.
$$

The leftmost factor is $(1, d_k)$ because the loss is one scalar. Now comes the crucial observation about *how to evaluate this product*.

## 5. Why reverse mode computes VJPs, not Jacobians

You could evaluate that product left to right or right to left; matrix multiplication is associative, so the answer is identical. But the *cost* is wildly different.

**Multiply right to left (forward mode):** start with the first Jacobian $(d_1, d_0)$ and keep multiplying. You are propagating full matrices; the intermediate results are big, and you effectively build a Jacobian per input.

**Multiply left to right (reverse mode):** start with the leftmost factor, which is a $(1, d_k)$ **row vector** (because the loss is scalar). Multiply it by the next Jacobian: $(1, d_k)\times(d_k, d_{k-1}) = (1, d_{k-1})$ — still a row vector. Multiply by the next: still a row vector. **You are only ever carrying a vector, never a matrix.** At every step you compute a **vector–Jacobian product** (VJP): a row vector $\mathbf{v}^\top$ times a Jacobian $J$, giving another row vector $\mathbf{v}^\top J$.

$$
\mathbf{v}^\top J \;=\; \Big[\;\textstyle\sum_i v_i\,\frac{\partial f_i}{\partial x_1},\;\; \sum_i v_i\,\frac{\partial f_i}{\partial x_2},\;\;\dots\;\Big].
$$

This is the entire reason backprop is affordable. Because the final output is a single scalar, reverse mode starts from a $1$-row seed and **pushes a vector backward**, and each op only needs to know how to turn "gradient of the loss with respect to my output" into "gradient of the loss with respect to my input" — that map *is* the VJP. No full Jacobian is ever built.

<div class="callout key"><p>Reverse-mode autodiff never forms the Jacobian $J$. It only ever computes $\mathbf{v}^\top J$ — the effect of $J$ on one incoming gradient vector. Since the loss is scalar, the backward pass seeds $\mathbf{v} = [1]$ and threads a vector through the graph right to left. Full gradient, cost of ~one forward pass, memory of a vector not a matrix.</p></div>

### Why forming $J$ would be ruinous

Consider a single linear layer mapping $2048 \to 2048$ activations. Its Jacobian is $2048 \times 2048 \approx 4.2$ million numbers — for *one layer, one example*. A network has dozens of layers, and you would need a Jacobian per layer. Storing them is hopeless. But the VJP for that layer is just a $2048$-vector times the layer's weight structure, producing another $2048$-vector — kilobytes, not megabytes. The VJP gives the same gradient the Jacobian would, without the Jacobian ever existing in memory. This is not an optimization detail; it is what makes training possible.

## 6. Numerical example: a VJP by hand

Take the $J$ from section 3 (at $\mathbf{a} = (2,3)$) and push the vector $\mathbf{v} = [1, 1, 1]$ back through it. The VJP is $\mathbf{v}^\top J$, a $(1,3)\times(3,2) = (1,2)$ row vector — one gradient number per input:

$$
\mathbf{v}^\top J = \begin{bmatrix} 1 & 1 & 1 \end{bmatrix}
\begin{bmatrix} 4 & 0 \\ 3 & 2 \\ 0 & 6 \end{bmatrix}
= \big[\,1\cdot4 + 1\cdot3 + 1\cdot0,\;\; 1\cdot0 + 1\cdot2 + 1\cdot6\,\big]
= [\,7,\ 8\,].
$$

Interpretation: $\mathbf{v} = [1,1,1]$ is "the loss increases by $1$ for each unit of each output" — i.e. the loss is $f_1 + f_2 + f_3$. The VJP says that loss is sensitive to $a_1$ at rate $7$ and to $a_2$ at rate $8$. Cross-check directly: $L = a_1^2 + a_1 a_2 + a_2^2$, so $\frac{\partial L}{\partial a_1} = 2a_1 + a_2 = 7$ and $\frac{\partial L}{\partial a_2} = a_1 + 2a_2 = 8$. They match, and we never needed $J$ as an object — only its action on $\mathbf{v}$.

A different seed selects a different combination of outputs. With $\mathbf{v} = [1, 0, 2]$ (the loss $f_1 + 2f_3$): $\mathbf{v}^\top J = [\,4,\ 12\,]$.

## 7. The computational graph: nodes and edges

To apply the chain rule mechanically, an autodiff system needs a record of what was computed from what. That record is the **computational graph**: a directed acyclic graph (DAG) where

- **nodes are operations** (a `+`, a `*`, a `matmul`, a `tanh`, a `softmax`), and
- **edges are tensors** flowing from the output of one op into the input of the next.

Leaf nodes are the inputs and parameters (the tensors with `requires_grad=True`); the single root is the scalar loss. Building the graph *is* the forward pass. Walking it backward — from the loss root toward the leaves, applying each op's VJP — *is* the backward pass.

For the section 6 example, the forward graph for $L = a_1^2 + a_1a_2 + a_2^2$ looks like:

```
 a1 ─┬─► [square] ─► t1 ─┐
     │                   ├─► [add] ─► s1 ─┐
 a2 ─┼─► [mul(a1,a2)] ─► t2 ┘              ├─► [add] ─► L
     └─► [square] ─► t3 ─────────────────┘
```

Reverse mode starts at $L$ with gradient $1$, and each op node contributes its VJP to its inputs' gradients as the sweep moves left. Because $a_1$ feeds *three* ops, its gradient is the **sum** of the contributions arriving along all three edges ($2a_1$ from the square, $a_2$ from the product, $0$ from the third) — this fan-in summation is exactly why PyTorch *accumulates* into `.grad` rather than overwriting, and why you must zero gradients between steps (you saw this in [00.3](lessons/module-00/lesson-03.md)).

## 8. From-scratch PyTorch: computing a VJP directly

PyTorch exposes the VJP as the argument to `.backward()`. When the output is a *vector*, `.backward(v)` computes $\mathbf{v}^\top J$ — it does not need a scalar loss if you hand it the seed vector yourself:

```python
import torch

a = torch.tensor([2.0, 3.0], requires_grad=True)          # (2,) float32 cpu

def f(a):                                                  # R^2 -> R^3
    return torch.stack([a[0]**2, a[0]*a[1], a[1]**2])

out = f(a)                                                 # (3,) = [4., 6., 9.]
v = torch.tensor([1.0, 1.0, 1.0])                          # seed, same shape as out
out.backward(v)                                            # computes v^T J
print(a.grad.tolist())                                     # [7.0, 8.0]
```

`a.grad` is `[7.0, 8.0]` — exactly the hand VJP of section 6, and it has shape `(2,)`, matching the input. The seed `v` had to match the shape of `out` `(3,)`, because it is the "incoming gradient with respect to each output." For an actual loss, that seed is the implicit `[1.0]` PyTorch supplies when you call `loss.backward()` on a scalar. If you *do* want the full Jacobian (rare — only for small analyses, never inside training), PyTorch will build it explicitly, and you can see it costs one VJP per output row:

```python
J = torch.autograd.functional.jacobian(f, a)
print(J)
# tensor([[4., 0.],
#         [3., 2.],
#         [0., 6.]])
```

This matches the hand-derived $J$ from section 3. Notice `jacobian` had to call the backward machinery three times (once per output) to fill the three rows — concrete evidence that the Jacobian is $m$ VJPs stacked, and that a scalar loss ($m=1$) needs only one.

<div class="callout pt"><p><code>loss.backward()</code> is <code>loss.backward(torch.tensor(1.0))</code> — the seed is the scalar <code>1</code>. That single <code>1</code> is the $\mathbf{v}$ that reverse mode threads backward as a VJP through every op in the graph. This is the concrete meaning of "the loss must be a scalar": a scalar gives a length-1 seed, so the backward pass carries vectors, not matrices.</p></div>

## 9. Under the hood: cost, memory, and the forward-vs-reverse choice

**Time.** One backward pass costs a small constant (~2×) times one forward pass, *independent of the number of parameters*, because each op does a fixed amount of extra work (its VJP) and the sweep visits each node once.

**Memory.** The forward pass must **cache the intermediate tensors** each op needs for its VJP (a `mul` needs both inputs; a `matmul` needs the input activation; `tanh` needs its output). This cache — the saved activations — is why training memory grows with network depth and batch size, and it is the single biggest consumable in large-model training. Techniques like gradient (activation) checkpointing, which we meet in the systems modules, trade recomputation for smaller caches.

**Why reverse and not forward mode?** Forward-mode autodiff computes a *Jacobian–vector* product and is efficient when there are few inputs and many outputs. Training is the opposite regime: **many** inputs (millions of parameters) and **one** output (the scalar loss). Reverse mode's cost scales with the number of *outputs*, so one scalar output means one cheap sweep gives gradients for all inputs at once. That regime match is why every deep-learning framework defaults to reverse mode.

## Exercise

For $\mathbf{f}(x_1, x_2) = [\,x_1 + x_2,\; x_1 x_2\,]$ at $(x_1, x_2) = (3, 4)$, write the Jacobian, then compute the VJP for $\mathbf{v} = [1, 1]$ by hand and verify with `out.backward(v)`.

<details><summary>Optional hint</summary>

$J$ is $(2,2)$: outputs are $[x_1+x_2,\ x_1x_2]$, inputs are $[x_1, x_2]$. Fill in each $\partial f_i/\partial x_j$.

</details>

<details><summary>Stronger hint</summary>

$$J = \begin{bmatrix} \partial(x_1{+}x_2)/\partial x_1 & \partial(x_1{+}x_2)/\partial x_2 \\ \partial(x_1x_2)/\partial x_1 & \partial(x_1x_2)/\partial x_2 \end{bmatrix} = \begin{bmatrix} 1 & 1 \\ x_2 & x_1 \end{bmatrix}.$$
At $(3,4)$ that is $\begin{bmatrix} 1 & 1 \\ 4 & 3 \end{bmatrix}$. Now left-multiply by $\mathbf{v}^\top = [1,1]$.

</details>

<details><summary>Solution</summary>

$\mathbf{v}^\top J = [1,1]\begin{bmatrix}1&1\\4&3\end{bmatrix} = [\,1+4,\ 1+3\,] = [5, 4]$.

```python
import torch
x = torch.tensor([3.0, 4.0], requires_grad=True)
out = torch.stack([x[0]+x[1], x[0]*x[1]])
out.backward(torch.tensor([1.0, 1.0]))
print(x.grad.tolist())   # [5.0, 4.0]
```

Cross-check with the scalar loss $L = (x_1+x_2) + x_1x_2$: $\partial L/\partial x_1 = 1 + x_2 = 5$, $\partial L/\partial x_2 = 1 + x_1 = 4$. Matches.

</details>

## Debugging exercise

A student wants the gradient of a *vector* output and writes:

```python
import torch
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = x * 3          # y is a vector, shape (2,)
y.backward()       # RuntimeError!
```

This raises `grad can be implicitly created only for scalar outputs`. What is wrong, and what are the two correct fixes depending on intent?

<details><summary>Solution</summary>

`y` is a length-2 vector, so there is no implicit seed — PyTorch cannot guess which VJP you want. Two fixes:

- If you meant a specific VJP, pass the seed: `y.backward(torch.tensor([1.0, 1.0]))`, which computes $\mathbf{v}^\top J$ and gives `x.grad == [3.0, 3.0]`.
- If you actually have a loss, reduce to a scalar first: `y.sum().backward()` (or `.mean()`), then `x.grad` is well defined. `y.sum()` is exactly the case $\mathbf{v} = [1, 1]$.

The lesson: `.backward()` on a non-scalar is under-specified because it hides which vector to push back. Training always ends in a scalar loss, so this never bites you there — the seed is the implicit `1`.

</details>

## Check yourself

<details><summary>A layer maps $\mathbb{R}^{512} \to \mathbb{R}^{2048}$. What is the shape of its Jacobian, and how many numbers is that?</summary>

Shape is (outputs, inputs) $= (2048, 512)$, which is $1{,}048{,}576$ numbers — for one layer, one example. Reverse mode never builds this; it only computes its VJP, a $2048$-vector in, a $512$-vector out.

</details>

<details><summary>Why does the backward pass carry a vector rather than a matrix, and what fact about the loss makes that possible?</summary>

Because the loss is a single scalar, the leftmost Jacobian in the chain-rule product is a $(1, d)$ row vector. Multiplying it through the remaining Jacobians left to right keeps a row vector at every step (a VJP). If the output had $m$ components, you would carry an $m$-row object. Scalar loss $\Rightarrow$ length-1 seed $\Rightarrow$ vector, not matrix.

</details>

<details><summary>In the graph for $L = a_1^2 + a_1a_2 + a_2^2$, why is $a_1$'s gradient a sum of three terms?</summary>

Because $a_1$ feeds three operations (its square, the product, and — trivially — nothing in the third term). The chain rule sums contributions over every path from $a_1$ to $L$. This fan-in summation is why PyTorch accumulates into `.grad` and why you must zero gradients between optimizer steps. Here it sums to $2a_1 + a_2 = 7$.

</details>

<details><summary>You call <code>out.backward(v)</code> where <code>out</code> has shape <code>(3,)</code>. What must <code>v</code>'s shape be, and what does the call compute?</summary>

`v` must be shape `(3,)` — the same as `out` — because it is the incoming gradient with respect to each output component. The call computes the VJP $\mathbf{v}^\top J$, giving a gradient with the same shape as the input.

</details>

## Next

You now know the shape of the object autodiff differentiates (a chain of Jacobians capped by a scalar) and the trick that makes it cheap (push a vector backward via VJPs, never form a Jacobian). Next we apply this to the exact computation at the output of every language model — linear layer, softmax, cross-entropy — and derive by hand the single most important gradient in the course: $\frac{\partial L}{\partial \mathbf{z}} = \mathbf{p} - \text{onehot}(\text{target})$.

Continue to [02.3 · Backprop through linear + softmax + cross-entropy by hand](lessons/module-02/lesson-03.md).
