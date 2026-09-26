# 00.1 · Tensors: shape, dtype, device

<div class="prereq">
<p><strong>Prerequisites:</strong> none — this is the entry point of the course. You need to be comfortable programming (arrays, loops, types) but no machine learning and no PyTorch is assumed.</p>
<p><strong>You will learn:</strong> what a tensor is (an n-dimensional array of numbers), how to create one (<code>torch.tensor</code>, <code>zeros</code>, <code>ones</code>, <code>arange</code>, <code>randn</code>), and the three properties you will read on <em>every</em> tensor for the rest of the course: its <code>.shape</code> (how many numbers and how they are arranged), its <code>.dtype</code> (what kind of number each element is, and why that decides memory and precision), and its <code>.device</code> (whether it lives in CPU RAM or GPU memory).</p>
<p><strong>Why this matters for ML:</strong> a neural network is nothing but tensors flowing through operations. Weights are tensors, inputs are tensors, gradients are tensors, the loss is a tensor. Ninety percent of the bugs you will hit training a language model are shape mismatches, dtype surprises, or a tensor sitting on the wrong device. Getting fluent with these three properties now is what makes every later module readable.</p>
</div>

## 1. Intuition: a tensor is a typed, n-dimensional array

If you come from C# or any typed language, you already know most of this — you just have not heard the word. A **tensor** is a rectangular grid of numbers, all of the same type, with a fixed number of axes (dimensions). The word "tensor" is borrowed from physics, but in PyTorch it means exactly "n-dimensional array", nothing more mystical than that.

The number of axes is the tensor's **rank** (PyTorch calls it the number of dimensions, `.ndim`). You already have names for the low ranks:

- **rank 0 — a scalar.** A single number: `3.0`. No axes.
- **rank 1 — a vector.** A list of numbers: `[1, 2, 3]`. One axis of length 3.
- **rank 2 — a matrix.** A grid: 2 rows, 3 columns. Two axes.
- **rank 3 and up — an ND tensor.** A stack of matrices, then a stack of those, and so on.

The reason we do not just use nested Python lists is that a tensor stores its numbers in one flat, contiguous block of memory of a single type, and every operation on it (add, multiply, matrix-multiply) is implemented as tight compiled code — and, when you move the tensor to a GPU, as thousands of parallel arithmetic units working at once. A Python list of lists can hold anything and is stored as scattered pointers; a tensor is the disciplined, fast version.

<div class="callout key"><p>A tensor is an n-dimensional array whose elements are all one numeric type, stored in one contiguous block of memory. Rank 0 = scalar, rank 1 = vector, rank 2 = matrix, higher = ND. Every value in a neural network — inputs, weights, activations, gradients — is a tensor.</p></div>

## 2. Creating tensors

There are a handful of constructors you will use constantly. Here they are, with what each is for.

```python
import torch

# From existing data (a Python list / nested list). dtype is inferred.
a = torch.tensor([[1., 2., 3.],
                  [4., 5., 6.]])          # a 2x3 matrix of floats

# Filled with a constant, given an explicit shape:
z = torch.zeros(2, 3)                     # 2x3 of 0.0
o = torch.ones(2, 3)                      # 2x3 of 1.0

# A range, like Python's range() but returning a tensor:
r = torch.arange(6)                       # tensor([0, 1, 2, 3, 4, 5])

# Random values from a standard normal distribution (mean 0, std 1):
torch.manual_seed(0)                      # make the randomness reproducible
n = torch.randn(2, 3)                     # 2x3 of gaussian noise
```

Two things to notice immediately, because they trip up newcomers:

- `torch.tensor([...])` takes the **actual data**. The shape comes from how the data is nested.
- `torch.zeros(2, 3)` / `torch.ones` / `torch.randn` take the **shape as separate arguments**, not a list. `torch.zeros(2, 3)` is a 2×3 grid; `torch.zeros([2, 3])` also works but the positional form is what you will read everywhere.

`torch.randn` is worth calling out because you will see it in almost every code example: it draws each element independently from the standard normal distribution (bell curve centred at 0). It is how we initialise weights and how we fabricate fake data for demos. `torch.manual_seed(0)` fixes the random number generator so the "random" values are the same every run — essential for reproducible examples and tests. With the seed set to 0, that `randn(2, 3)` prints exactly:

```
tensor([[ 1.5410, -0.2934, -2.1788],
        [ 0.5684, -1.0845, -1.3986]])
```

Run it yourself with the same seed and you will get the same six numbers.

## 3. Property one — `.shape`

The **shape** is a tuple giving the length of each axis, outermost first. It answers "how many numbers, arranged how?" For the `a` above:

```python
a.shape          # torch.Size([2, 3])
a.ndim           # 2   (two axes)
a.numel()        # 6   (total number of elements = 2 * 3)
```

`torch.Size([2, 3])` reads as "2 along the first axis, 3 along the second" — 2 rows of 3. The product of the shape entries is the total element count, `numel()`. You read shapes constantly: almost every debugging session starts with printing `.shape` and checking it is what you expected.

### Indexing and what an index gives back

Indexing works like nested arrays, one index per axis:

```python
a[0]       # first row:  tensor([1., 2., 3.])   -> shape (3,)
a[0, 2]    # row 0, col 2: tensor(3.)            -> shape ()  a scalar tensor
```

Note the second result is `tensor(3.)`, a **rank-0 tensor**, not the Python float `3.0`. It is still a tensor (it still has a dtype and a device). To pull the plain Python number out, call `.item()`: `a[0, 2].item()` gives `3.0`. This distinction matters later — a scalar tensor can carry a gradient; a Python float cannot.

<div class="callout key"><p>The shape convention you will see for the rest of the course is <strong>(B, T, C)</strong>: <strong>B</strong> = batch (how many sequences we process at once), <strong>T</strong> = time / sequence length (how many tokens in each), <strong>C</strong> = channels / features (the size of each token's vector). Every activation inside the language model is a rank-3 tensor of shape (B, T, C). We are not using it yet, but every time you read a shape from here on, ask "which axis is which?" — that habit is the whole game.</p></div>

## 4. Property two — `.dtype`

Every element of a tensor has the same numeric type, the **dtype**. It decides two things that matter enormously at scale: how many **bytes** each number takes, and how much **precision** it carries.

The dtypes you will actually meet in this course:

| dtype | bytes/element | what it is | where it shows up |
|---|---|---|---|
| `float32` | 4 | single-precision float, the default | almost everything by default |
| `float16` | 2 | half-precision float, narrow range | mixed-precision training (module 8) |
| `bfloat16` | 2 | "brain float", same range as float32 but fewer mantissa bits | the standard for training large models |
| `int64` (aka `long`) | 8 | 64-bit integer, the default for ints | **token IDs**, indices |
| `bool` | 1 | true/false | masks (e.g. the causal mask, module 5) |

Two defaults to memorise, because they cause silent surprises:

- `torch.tensor([1., 2.])` (with the decimal points) is **float32**.
- `torch.tensor([1, 2])` (no decimals) is **int64**, and integer tensors cannot carry gradients.
- `torch.zeros(...)`, `torch.ones(...)`, `torch.randn(...)` are **float32** unless you say otherwise.
- `torch.arange(6)` is **int64** — it is a range of integers.

You read and change dtype like this:

```python
a.dtype                      # torch.float32
h = a.to(torch.float16)      # a copy in half precision
h.dtype                      # torch.float16
```

### Why dtype decides memory and precision

A tensor's memory footprint is simply `numel * bytes_per_element`. Take our 2×3 float32 tensor:

$$
\text{bytes} = (\underbrace{2 \times 3}_{\text{numel}=6}) \times \underbrace{4}_{\text{bytes/elem}} = 24 \text{ bytes}.
$$

PyTorch confirms it:

```python
a.element_size()             # 4   (bytes per element for float32)
a.numel()                    # 6
a.element_size() * a.numel() # 24  bytes

h = a.to(torch.float16)
h.element_size()             # 2
h.element_size() * h.numel() # 12  bytes  -- half the memory
```

The float16 copy holds the *same six numbers* in **12 bytes instead of 24** — exactly half. That factor is the entire reason mixed precision exists. A GPT-2-scale model has ~124 million parameters; at float32 that is $124{,}000{,}000 \times 4 \approx 496$ MB just for the weights, and you need several more copies of that for gradients and optimizer state. Halving the bytes per element is the difference between fitting on a GPU and not. The cost is precision: float16 stores fewer significant digits and has a much narrower range, so values can overflow to infinity or underflow to zero — which is why `bfloat16` (same exponent range as float32, so it will not overflow, but coarser steps) has become the default training dtype. We handle the precision pitfalls carefully in [module 8](lessons/module-08/lesson-03.md); for now the takeaway is: **dtype is a memory-vs-precision dial, and you will spend real engineering effort on it.**

<div class="callout warn"><p>Token IDs are integers and must stay <code>int64</code> (<code>long</code>) — they are indices into the vocabulary, not measurements. A very common beginner error is turning token IDs into floats; the embedding lookup in <a href="#/lessons/module-05/lesson-01">module 5</a> needs a long tensor and will raise if you hand it floats. Conversely, anything the network does arithmetic and gradients on must be a float dtype, because integer tensors cannot have gradients.</p></div>

## 5. Property three — `.device`

A tensor lives somewhere: either in ordinary **CPU** RAM or in a **GPU**'s memory (`cuda`). The device is the third property you check.

```python
a.device                                  # device(type='cpu')

device = "cuda" if torch.cuda.is_available() else "cpu"
a_gpu = a.to(device)                       # copy the tensor to that device
```

`.to(device)` returns a tensor on the requested device (a copy, if it moves). The single most common runtime error in real training code is:

> `RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!`

An operation between two tensors requires them to be on the **same device** — you cannot add a CPU tensor to a GPU tensor, because their bytes are in physically different memories. The standard defence is to pick one `device` variable at the top of your program and `.to(device)` everything: the model, the inputs, the targets.

In this module everything runs on CPU (torch 2.14 CPU build), which is plenty for tensors this small. The `.to(device)` habit is what lets the exact same code later run on a GPU without change — you flip one variable.

<div class="callout pt"><p><code>.to(...)</code> is overloaded: <code>x.to(torch.float16)</code> changes dtype, <code>x.to("cuda")</code> changes device, and <code>x.to("cuda", torch.float16)</code> does both at once. It returns a new tensor when something actually changes, and the same tensor when nothing needs to move. It does <em>not</em> modify in place — you must assign the result: <code>x = x.to(device)</code>.</p></div>

## 6. Under the hood — contiguous memory and strides

You do not strictly need this to write correct code, but it explains a whole class of behaviour in the next lesson, so meet it now briefly.

A tensor's numbers are stored in **one flat 1-D array** in memory. The shape and a second tuple called the **stride** describe how to interpret that flat array as an n-D grid. The stride says: "to move one step along this axis, skip this many elements in the flat array." For our 2×3 tensor stored row-by-row:

```python
x = torch.arange(6).view(2, 3)   # [[0,1,2],[3,4,5]]
x.stride()                       # (3, 1)
x.is_contiguous()                # True
```

Stride `(3, 1)` means: to go down one row, jump 3 elements in the flat buffer (`0→3`); to go right one column, jump 1 element (`0→1`). This layout — where the flat order matches reading the grid left-to-right, top-to-bottom — is called **contiguous** (specifically, *row-major* or *C-contiguous*). It is the default, and many fast operations assume it.

The important consequence: because shape and stride are just metadata *describing* a shared flat buffer, PyTorch can hand you a different-shaped **view** of the same numbers without copying anything — it just makes a new (shape, stride) label pointing at the same memory. That is why reshaping is nearly free, and it is also the source of the "views share storage" surprises we take apart in the next lesson.

## Check yourself

<details><summary>What are the three properties you should be able to read off any tensor, and what does each tell you?</summary>

`.shape` — the length of each axis (how many numbers, arranged how). `.dtype` — the numeric type of each element (which fixes bytes-per-element and precision). `.device` — where the tensor's memory lives (CPU or a specific GPU). Together they answer "how big, what kind of number, and where".

</details>

<details><summary>How many bytes does a tensor of shape (4, 8) with dtype float32 occupy? What about bfloat16?</summary>

float32 is 4 bytes/element, and $4 \times 8 = 32$ elements, so $32 \times 4 = 128$ bytes. bfloat16 is 2 bytes/element, so $32 \times 2 = 64$ bytes — exactly half. In general: `numel * element_size`.

</details>

<details><summary>You write <code>ids = torch.tensor([5, 17, 42])</code> intending token IDs, then later the code adds them to a float tensor and complains, or an embedding layer accepts them fine. What is the dtype, and is it right for token IDs?</summary>

Because the literals have no decimal points, the dtype is `int64` (`long`). That is exactly right for token IDs — they are indices into the vocabulary and must stay integer/long for the embedding lookup. If you had written `[5., 17., 42.]` you would have gotten float32, which the embedding layer would reject.

</details>

<details><summary>What does <code>a[0, 2]</code> return for a 2-D tensor — a Python number or a tensor? How do you get the Python number?</summary>

It returns a rank-0 (scalar) tensor, e.g. `tensor(3.)`, which still has a dtype and device. Call `.item()` to extract the plain Python number: `a[0, 2].item()` gives `3.0`.

</details>

<details><summary>Why can PyTorch reshape a tensor without copying its data?</summary>

The numbers live in one flat contiguous buffer; the shape and stride are just metadata describing how to read that buffer as a grid. A reshape/view produces a new (shape, stride) label pointing at the same underlying memory, so no numbers are moved — it is nearly free. (This is also why a view and its parent share storage, as the next lesson explores.)

</details>

## Next

You can now create tensors and read their shape, dtype, and device. The next lesson puts tensors to work: elementwise arithmetic, reductions, matrix multiplication, the broadcasting rules that let differently-shaped tensors combine, and the crucial difference between a **view** (shares memory) and a **copy** — the source of some of the most confusing bugs in PyTorch.

Continue to [00.2 · Ops, broadcasting, views vs copies](lessons/module-00/lesson-02.md).
