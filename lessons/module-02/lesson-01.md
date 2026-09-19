# 02.1 · Derivatives, partials, chain rule, gradients

<div class="prereq">
<p><strong>Prerequisites:</strong> <a href="../module-00/lesson-01.md">00.1 · Tensors: shape, dtype, device</a> — you should be comfortable with scalar (rank-0) tensors and with reading a tensor's <code>.shape</code>. A first look at <a href="../module-00/lesson-03.md">00.3 · Autograd: backward and grad</a> helps but is not required; this lesson supplies the calculus that lesson used.</p>
<p><strong>You will learn:</strong> the <strong>derivative</strong> as a slope, i.e. a measure of sensitivity ("if I nudge the input, how much does the output move?"); the <strong>partial derivative</strong> (the same idea for one input while all others are held fixed); the <strong>gradient</strong> $\nabla f$ as the vector of all partials, which points in the direction of steepest increase; and the <strong>scalar chain rule</strong> $\frac{dz}{dx} = \frac{dz}{dy}\cdot\frac{dy}{dx}$, worked by hand on $y = 3x+1,\; z = y^2$ at $x = 2$ and verified in PyTorch.</p>
<p><strong>Why this matters for ML:</strong> training a language model is nothing but repeatedly asking "which way, and how hard, should I nudge each of these millions of parameters to make the loss smaller?" The answer is the gradient. Every optimizer step in this course is a step downhill along $-\nabla(\text{loss})$. This lesson is the calculus those steps rest on; the next two lessons turn it into backpropagation, and Module 3 turns it into an optimizer.</p>
</div>

## 1. Intuition: a derivative is a sensitivity

Forget "slope of a tangent line" for a moment and think like an engineer profiling a function. You have a function $f$ that takes a number in and gives a number out. You are sitting at some input $x$, and you ask a purely operational question: **if I increase $x$ by a tiny amount, does $f(x)$ go up or down, and by how much per unit of $x$?** That ratio — output change divided by input change, in the limit of a tiny change — is the **derivative** $f'(x)$.

Two facts are all you need to carry forward:

- **Sign** tells you *direction*. If $f'(x) > 0$, pushing $x$ up pushes $f$ up. If $f'(x) < 0$, pushing $x$ up pushes $f$ down.
- **Magnitude** tells you *sensitivity*. $f'(x) = 100$ means $f$ reacts violently to $x$ here; $f'(x) = 0.001$ means $f$ barely notices.

In training we always want the loss to go *down*, so we read the derivative and step the input in the opposite direction of its sign. That single move, applied to every parameter at once, is gradient descent. Everything else is bookkeeping.

## 2. Mathematics: the definition and the three rules we actually use

Formally, the derivative of $f$ at $x$ is the limit of the average rate of change as the step $h$ shrinks to zero:

$$
f'(x) \;=\; \frac{df}{dx} \;=\; \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}.
$$

Here $f(x+h) - f(x)$ is the *output* change and $h$ is the *input* change; their ratio is a rate, and the limit pins down that rate exactly at $x$ rather than over a finite stretch. The notation $\frac{df}{dx}$ is deliberately a fraction-like symbol: it reads "$df$ per $dx$", output-change per input-change.

You will almost never compute that limit by hand. You need exactly three results, each of which you can treat as a lookup:

- **Power rule:** if $f(x) = x^n$ then $f'(x) = n\,x^{n-1}$. (So $x^2 \to 2x$, and a constant $\to 0$ because it never changes.)
- **Linearity:** the derivative of $a\,f(x) + b\,g(x)$ is $a\,f'(x) + b\,g'(x)$. Constants factor out; sums differentiate term by term.
- **Chain rule:** covered in section 5 — the one that makes backprop possible.

Take $f(x) = 3x + 1$. By linearity and the power rule, the derivative of $3x$ is $3$ and the derivative of the constant $1$ is $0$, so $f'(x) = 3$. It is constant because a straight line has the same slope everywhere: nudge $x$ by $h$ and the output always moves by $3h$.

## 3. Numerical example: a derivative as a measured slope

Let us *see* the sensitivity for $f(x) = x^2$ at $x = 3$. The power rule says $f'(3) = 2\cdot 3 = 6$. Check it with a tiny finite step $h = 0.001$:

$$
\frac{f(3.001) - f(3)}{0.001} = \frac{9.006001 - 9}{0.001} = \frac{0.006001}{0.001} = 6.001 \approx 6.
$$

The finite-step ratio is $6.001$; the exact derivative is the limit as $h \to 0$, which is $6$. The leftover $0.001$ is the error from using a finite $h$ instead of an infinitesimal one. This "wiggle the input, watch the output" check is worth keeping in your toolkit — it is exactly how you sanity-check a hand-derived gradient against reality (and how `torch.autograd.gradcheck` works internally).

## 4. Partial derivatives: one knob at a time

Real functions in ML take many inputs — a loss depends on millions of parameters. The **partial derivative** extends the derivative to that setting with one rule: *differentiate with respect to one variable while treating every other variable as a fixed constant.* The curly $\partial$ replaces $d$ to signal "there are other variables, and I am holding them still."

Take $f(x, y) = x^2 y + y^3$. To get $\frac{\partial f}{\partial x}$, treat $y$ as a constant:

$$
\frac{\partial f}{\partial x} = 2xy + 0 = 2xy,
$$

because $x^2 y$ differentiates (in $x$) to $2xy$ with the constant $y$ along for the ride, and $y^3$ has no $x$ in it so it is a constant and dies. To get $\frac{\partial f}{\partial y}$, treat $x$ as a constant:

$$
\frac{\partial f}{\partial y} = x^2 + 3y^2,
$$

since $x^2 y$ differentiates (in $y$) to $x^2$, and $y^3$ to $3y^2$. Each partial answers "how sensitive is $f$ to *this one* input, if I freeze all the others?"

## 5. The chain rule: differentiating a pipeline

Neural networks are compositions: the input flows through layer after layer, each feeding the next. To differentiate a composition we need the **chain rule**. If $z$ depends on $y$, and $y$ depends on $x$, then

$$
\frac{dz}{dx} \;=\; \frac{dz}{dy}\cdot\frac{dy}{dx}.
$$

Read it as sensitivities multiplying: how much $z$ moves per unit of $x$ equals how much $z$ moves per unit of $y$, times how much $y$ moves per unit of $x$. If wiggling $x$ moves $y$ twice as fast, and wiggling $y$ moves $z$ five times as fast, then wiggling $x$ moves $z$ ten times as fast. The intermediate sensitivities compose by multiplication. This is the single fact that backpropagation applies over and over, once per edge of the network.

<div class="callout key"><p>The chain rule is the whole engine of backprop: to get the gradient of the loss with respect to an early parameter, multiply the local sensitivities along every step of the path from that parameter to the loss. Backprop is just this rule applied systematically, from the loss backward.</p></div>

### Worked numeric example: $y = 3x + 1$, $z = y^2$, at $x = 2$

This is the example we will verify in PyTorch, so we do it slowly.

**Forward pass** (compute the values left to right):

$$
y = 3x + 1 = 3\cdot 2 + 1 = 7, \qquad z = y^2 = 7^2 = 49.
$$

**Local derivatives** (differentiate each step on its own):

$$
\frac{dy}{dx} = 3 \quad(\text{from } 3x+1), \qquad \frac{dz}{dy} = 2y = 2\cdot 7 = 14 \quad(\text{power rule, at } y = 7).
$$

**Chain them:**

$$
\frac{dz}{dx} = \frac{dz}{dy}\cdot\frac{dy}{dx} = 14 \cdot 3 = 42.
$$

So at $x = 2$, nudging $x$ up by a hair increases $z$ at a rate of $42$ per unit of $x$. Notice we needed the *value* $y = 7$ from the forward pass to evaluate $\frac{dz}{dy} = 2y$. That is why every autodiff system runs a forward pass first and **caches the intermediate values** — the backward pass needs them. We will see PyTorch do exactly this in lesson 02.4.

## 6. Tensor shapes: the gradient is a vector

Stack all the partials of a scalar function $f(\mathbf{x})$ into one vector and you have the **gradient**, written $\nabla f$ ("nabla f" or "grad f"):

$$
\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\[4pt] \frac{\partial f}{\partial x_2} \\[4pt] \vdots \\[4pt] \frac{\partial f}{\partial x_n} \end{bmatrix}.
$$

The gradient has **exactly the same shape as the input** it is taken with respect to. If $\mathbf{x}$ is a vector of $n$ numbers, $\nabla f$ is a vector of $n$ numbers; if a weight tensor is $(768, 768)$, its gradient is $(768, 768)$. This shape-matching is not a coincidence you will forget — it is the rule that makes the optimizer's `param -= lr * param.grad` line type-check: parameter and gradient are always the same shape, dtype, and device. PyTorch stores a tensor's gradient in its `.grad` attribute, and `param.grad.shape == param.shape` always.

For $f(x, y) = x^2 y + y^3$ we found $\frac{\partial f}{\partial x} = 2xy$ and $\frac{\partial f}{\partial y} = x^2 + 3y^2$, so

$$
\nabla f(x, y) = \begin{bmatrix} 2xy \\ x^2 + 3y^2 \end{bmatrix}, \qquad
\nabla f(2, 3) = \begin{bmatrix} 2\cdot2\cdot3 \\ 2^2 + 3\cdot3^2 \end{bmatrix} = \begin{bmatrix} 12 \\ 31 \end{bmatrix}.
$$

The single most important geometric fact about the gradient: **it points in the direction of steepest increase of $f$**, and its length is how steep that increase is. So $-\nabla f$ points in the direction of steepest *decrease*. That is why gradient descent steps against the gradient — it is walking directly downhill on the loss surface. We will use this constantly starting in [Module 3](../module-03/lesson-01.md).

## 7. From-scratch PyTorch: verifying $dz/dx = 42$

PyTorch computes these derivatives for you with autograd (which you met in [00.3](../module-00/lesson-03.md)). We mark the input with `requires_grad=True`, build the expression (this records a graph), call `.backward()` to run the chain rule, and read the result off `.grad`:

```python
import torch

x = torch.tensor(2.0, requires_grad=True)   # scalar, dtype float32, device cpu
y = 3 * x + 1                               # forward: y = 7.0, graph edge d y/d x = 3
z = y ** 2                                  # forward: z = 49.0, graph edge d z/d y = 2y

z.backward()                                # run the chain rule from z backward
print(y.item(), z.item(), x.grad.item())    # 7.0 49.0 42.0
```

The printed `x.grad` is `42.0` — exactly our hand computation. What PyTorch did: while executing the forward lines it recorded that `y` came from `x` via `3*x+1` (local derivative $3$) and `z` came from `y` via `y**2` (local derivative $2y$, and it stashed the value $y=7$). Then `z.backward()` seeded $\frac{dz}{dz}=1$ and multiplied the local derivatives back along the chain: $1 \cdot 2y \cdot 3 = 1\cdot 14\cdot 3 = 42$. It is doing section 5 by hand, mechanically. Lesson 02.4 opens up that machinery completely.

We can confirm the multivariable gradient too:

```python
v = torch.tensor([2.0, 3.0], requires_grad=True)   # x = 2, y = 3; shape (2,)
f = v[0]**2 * v[1] + v[1]**3                        # scalar
f.backward()
print(f.item(), v.grad.tolist())                    # 39.0 [12.0, 31.0]
```

`v.grad` is `[12.0, 31.0]` and has shape `(2,)` — the same shape as `v` — matching $\nabla f(2,3) = [12, 31]$ from section 6.

<div class="callout pt"><p><code>.backward()</code> only works when the thing you call it on is a <strong>scalar</strong> (a single number, like a loss). That is not an arbitrary restriction: the gradient <em>of a scalar</em> with respect to an input has the input's shape, which is exactly what an optimizer needs. Lesson 02.2 explains why a scalar output is what lets reverse-mode autodiff push a single <code>1</code> backward, and what to do when the output is a vector.</p></div>

## 8. Under the hood: why gradients, and not the function itself, are the whole game

A trained language model is a function with on the order of $10^9$–$10^{12}$ parameters. Nobody can reason about that surface directly. But at any single point, the gradient gives you a flat, local, linear picture: a vector of $10^9$ numbers, each saying "increase me → loss goes this way, at this rate." That local picture is cheap to compute (reverse-mode autodiff, lessons 02.2–02.4, gets the entire gradient in roughly the cost of *one* forward pass) and it is exactly enough to take one good downhill step. Training is millions of such steps.

The cost structure is worth internalizing now. A **forward pass** evaluates the function. The **backward pass** evaluates the gradient, and it costs roughly the same as the forward pass (about 2× in practice), *no matter how many parameters there are*. That astonishing fact — full gradient for the price of a forward pass — is what makes training billion-parameter models feasible at all, and it is the payoff of the chain rule organized as backpropagation. We build up to why in the next lesson.

## Exercise

Compute $\frac{dz}{dx}$ by hand for $y = x^3$, $z = \sin(y)$ at $x = 1$, then verify in PyTorch. (You may use $\frac{d}{dy}\sin(y) = \cos(y)$.)

<details><summary>Optional hint</summary>

Chain rule: $\frac{dz}{dx} = \frac{dz}{dy}\cdot\frac{dy}{dx}$. You need the value of $y$ at $x=1$ to evaluate $\cos(y)$.

</details>

<details><summary>Stronger hint</summary>

$\frac{dy}{dx} = 3x^2$ and $\frac{dz}{dy} = \cos(y)$. At $x = 1$: $y = 1^3 = 1$, so $\frac{dy}{dx} = 3$ and $\frac{dz}{dy} = \cos(1)$.

</details>

<details><summary>Solution</summary>

$\frac{dz}{dx} = \cos(y)\cdot 3x^2$. At $x = 1$, $y = 1$, this is $3\cos(1) \approx 3 \times 0.5403 = 1.6209$.

```python
import torch
x = torch.tensor(1.0, requires_grad=True)
z = torch.sin(x**3)
z.backward()
print(x.grad.item())   # 1.6209069
```

PyTorch prints `1.6209069`, matching $3\cos(1)$.

</details>

## Common mistakes

- **Confusing the derivative's value with the function's value.** $f(3) = 9$ and $f'(3) = 6$ are different questions. The forward pass gives the first; the backward pass gives the second.
- **Forgetting to hold others fixed in a partial.** When differentiating $x^2 y$ with respect to $x$, $y$ is a *constant multiplier* ($\to 2xy$), not something that also gets differentiated.
- **Expecting the gradient to have a different shape than the input.** $\nabla f$ always matches the shape of the variable you differentiated with respect to. A shape mismatch between a parameter and its `.grad` is always a bug.
- **Calling `.backward()` on a non-scalar.** PyTorch raises `grad can be implicitly created only for scalar outputs`. Reduce to a scalar first (e.g. a loss), or pass an explicit vector — see lesson 02.2.

## Check yourself

<details><summary>At $x=2$ for $y = 3x+1,\ z=y^2$, why did we need $y = 7$ during the backward pass, and where did it come from?</summary>

Because $\frac{dz}{dy} = 2y$ depends on the value of $y$, not just its symbol. That value ($7$) was produced during the forward pass and cached. Every autodiff system runs forward first precisely so the backward pass can reuse these intermediate values.

</details>

<details><summary>What is $\nabla f$ for $f(x,y,z) = xyz$, and what is its value at $(1,2,3)$?</summary>

$\nabla f = [\,yz,\ xz,\ xy\,]$ (each partial holds the other two fixed). At $(1,2,3)$ that is $[\,2\cdot3,\ 1\cdot3,\ 1\cdot2\,] = [6, 3, 2]$. It is a length-3 vector, matching the 3 inputs.

</details>

<details><summary>If $\frac{\partial L}{\partial w} = -4$ for some weight $w$, which way do you move $w$ to reduce $L$, and why?</summary>

Increase $w$. A negative partial means increasing $w$ decreases $L$. Gradient descent moves against the gradient: $w \leftarrow w - \eta\cdot(-4) = w + 4\eta$, i.e. $w$ goes up. The magnitude $4$ says $L$ is fairly sensitive to $w$ here.

</details>

<details><summary>Why does reverse-mode autodiff give the whole gradient for roughly the cost of one forward pass, rather than one forward pass per parameter?</summary>

Because it applies the chain rule *once*, sweeping backward from the single scalar loss and reusing shared intermediate results across all parameters, instead of re-perturbing each parameter separately. The next lesson makes this precise via vector–Jacobian products.

</details>

## Next

You can now read a derivative as a sensitivity, take partials one knob at a time, stack them into a gradient that points uphill, and chain sensitivities through a pipeline. Next we lift the chain rule from scalars to vectors and matrices — Jacobians — and see the specific, memory-saving way reverse-mode autodiff actually applies it: the vector–Jacobian product.

Continue to [02.2 · Jacobians, VJPs, computational graphs](lesson-02.md).
