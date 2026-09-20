# Math & ML cheat sheet

A single-page lookup of the key equations of the whole course. Each entry gives the formula, a
one-line "what it is", and every symbol named. Notation matches the lessons. This is a reference,
not a tutorial — follow the linked lesson for the derivation and a worked numerical example.

Conventions used throughout: $B$ batch, $T$ sequence length (context), $C = n_{\text{embd}}$ model
width, $V$ vocab size, $n_h$ number of heads, $d_h = C/n_h$ head dimension, $L = n_{\text{layer}}$
number of transformer blocks. $\sigma(z) = 1/(1+e^{-z})$ is the logistic sigmoid.

---

## Probability & information theory

**Softmax** — turns a vector of scores into a probability distribution.
$$\mathrm{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
$z \in \mathbb{R}^V$ logits; output is non-negative and sums to 1.

**Numerically stable softmax** — subtract the max before exponentiating; the result is identical
but never overflows.
$$\mathrm{softmax}(z)_i = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}, \qquad m = \max_j z_j$$

**Log-softmax (stable)** — the log of softmax, computed without forming the probabilities.
$$\log\mathrm{softmax}(z)_i = z_i - m - \log\!\sum_j e^{z_j - m}$$
The trailing term is the log-sum-exp; $m=\max_j z_j$ keeps it finite.

**Cross-entropy loss** — the LM training loss: negative log-probability the model assigns to the
true next token.
$$\mathcal{L}_{\text{CE}} = -\frac{1}{N}\sum_{t=1}^{N} \log p_\theta(y_t \mid y_{<t})$$
$y_t$ true token at position $t$; $p_\theta$ model probability; $N$ number of scored tokens. For
a single position with one-hot target $c$: $\mathcal{L} = -\log\mathrm{softmax}(z)_c$.

**Perplexity** — the exponential of mean cross-entropy; the model's effective branching factor
("how many equally-likely tokens it is choosing among").
$$\mathrm{PPL} = \exp(\mathcal{L}_{\text{CE}})$$
Lower is better; a uniform guess over $V$ tokens has $\mathrm{PPL}=V$.

**Entropy** — expected surprise of a distribution $p$ (nats when $\log$ is natural).
$$H(p) = -\sum_i p_i \log p_i$$

**KL divergence** — extra cost, in nats, of coding samples from $p$ using a code built for $q$;
$\ge 0$, and $0$ iff $p=q$. Not symmetric.
$$\mathrm{KL}(p\,\|\,q) = \sum_i p_i \log\frac{p_i}{q_i} = \mathbb{E}_{p}\!\left[\log\frac{p}{q}\right]$$

**Cross-entropy = entropy + KL** — why minimizing CE against a fixed target minimizes KL to it.
$$H(p,q) = -\sum_i p_i\log q_i = H(p) + \mathrm{KL}(p\,\|\,q)$$

**LM chain-rule factorization** — a language model is an exact factorization of the joint
sequence probability into next-token conditionals.
$$p_\theta(y_{1:T}) = \prod_{t=1}^{T} p_\theta(y_t \mid y_{<t})$$
$y_{<t}=(y_1,\dots,y_{t-1})$ the prefix. Taking $\log$ turns the product into the sum that CE
averages. See [Module 1](../lessons/module-01/lesson-04.md).

---

## Attention

**Scaled dot-product attention** — each query reads a weighted blend of values, weighted by
query–key similarity.
$$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}} + M\right)V$$
$Q,K,V$ shape $(B,n_h,T,d_h)$; $QK^\top$ shape $(B,n_h,T,T)$ raw scores; $\sqrt{d_h}$ keeps score
variance $\approx 1$ so softmax stays soft; $M$ causal mask. Output $(B,n_h,T,d_h)$. See
[Module 5](../lessons/module-05/lesson-02.md).

**Causal mask** — forbids attending to the future; added before softmax so masked weights are
exactly zero (since $e^{-\infty}=0$).
$$M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}$$

**Multi-head split** — reshape the width $C$ into $n_h$ independent heads of size $d_h$, attend per
head, then concatenate and project.
$$\mathrm{MHA}(x) = \mathrm{Concat}(\text{head}_1,\dots,\text{head}_{n_h})\,W_O, \qquad d_h = C/n_h$$
$x$ shape $(B,T,C)$ reshaped to $(B,n_h,T,d_h)$; $W_O$ shape $(C,C)$ output projection. See
[Module 5](../lessons/module-05/lesson-03.md).

**GQA / MQA** — share K/V heads across query heads to shrink the KV cache. $n_{kv}$ key/value
heads with $1 \le n_{kv} \le n_h$: MQA is $n_{kv}=1$, GQA is $1<n_{kv}<n_h$, full MHA is
$n_{kv}=n_h$. Each K/V head is reused by $n_h/n_{kv}$ query heads. See
[Module 12](../lessons/module-12/lesson-03.md).

---

## Normalization & activations

**LayerNorm** — standardize each token vector across its $C$ features, then learn a scale and
shift.
$$\mathrm{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}\odot\gamma + \beta, \quad
\mu = \frac{1}{C}\sum_i x_i, \; \sigma^2 = \frac{1}{C}\sum_i (x_i-\mu)^2$$
$\gamma,\beta \in \mathbb{R}^C$ learned; $\epsilon$ numerical floor; normalization is per token,
over the feature axis.

**RMSNorm** — LayerNorm without the mean subtraction; rescales by root-mean-square only (cheaper,
used in Llama/most modern LMs).
$$\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{C}\sum_i x_i^2 + \epsilon}}\odot\gamma$$
No re-centering ($\mu$) and no bias $\beta$. See [Module 12](../lessons/module-12/lesson-02.md).

**GELU** — smooth ("soft") ReLU used in GPT-2; gates the input by its Gaussian CDF $\Phi$.
$$\mathrm{GELU}(x) = x\,\Phi(x) \approx 0.5x\left(1 + \tanh\!\left[\sqrt{\tfrac{2}{\pi}}(x + 0.044715x^3)\right]\right)$$

**SwiGLU** — gated MLP activation (Llama/PaLM); the hidden projection is modulated by a SiLU gate.
$$\mathrm{SwiGLU}(x) = \big(\mathrm{SiLU}(xW_1)\odot xW_3\big)W_2, \qquad \mathrm{SiLU}(z)=z\,\sigma(z)$$
$W_1,W_3$ project up, $W_2$ projects down; the two up-projections make it "gated". Hidden size is
scaled by $2/3$ to match a plain MLP's parameter count. See
[Module 12](../lessons/module-12/lesson-02.md).

---

## Positional encoding (RoPE)

**RoPE rotation** — inject position by rotating each 2-D pair of query/key features by an angle
proportional to the position.
$$\tilde{q}_m = R_{\Theta,m}\,q_m, \qquad
R_{\Theta,m}^{(i)} = \begin{bmatrix}\cos m\theta_i & -\sin m\theta_i\\ \sin m\theta_i & \cos m\theta_i\end{bmatrix},
\quad \theta_i = 10000^{-2i/d_h}$$
$m$ absolute position; $q_m$ query at $m$; pair index $i=0,\dots,d_h/2-1$; $\theta_i$ per-pair
frequency (high $i$ = slow rotation).

**RoPE relative-position property** — after rotation the dot product depends only on the *offset*
$m-n$, so attention becomes relative-position aware for free.
$$(R_{\Theta,m}q)^\top (R_{\Theta,n}k) = q^\top R_{\Theta,\,n-m}\,k$$
See [Module 12](../lessons/module-12/lesson-01.md).

---

## Backprop building blocks

**Residual connection & its gradient** — the skip path lets gradient reach early layers
undiminished, enabling deep stacks.
$$y = x + f(x) \qquad\Rightarrow\qquad \frac{\partial y}{\partial x} = I + \frac{\partial f}{\partial x}$$
The identity term guarantees a gradient highway even when $\partial f/\partial x$ is tiny. See
[Module 5](../lessons/module-05/lesson-04.md).

**Chain rule / VJP** — backprop propagates the upstream gradient through each op by its transposed
Jacobian (vector–Jacobian product), never forming the full Jacobian.
$$\bar{x} = J^\top \bar{y}, \qquad J = \frac{\partial y}{\partial x}$$
$\bar{y}=\partial\mathcal{L}/\partial y$ upstream grad; $\bar{x}$ downstream grad. See
[Module 2](../lessons/module-02/lesson-02.md).

**Softmax+CE gradient** — the reason the LM backward pass is so clean: gradient at the logits is
just (prediction − target).
$$\frac{\partial \mathcal{L}_{\text{CE}}}{\partial z} = \mathrm{softmax}(z) - \mathbf{1}_{c}$$
$\mathbf{1}_c$ one-hot true class. See [Module 2](../lessons/module-02/lesson-03.md).

---

## Optimization

**SGD with momentum** — accumulate a velocity of past gradients to smooth and accelerate descent.
$$v_t = \mu v_{t-1} + g_t, \qquad \theta_t = \theta_{t-1} - \eta\, v_t$$
$g_t$ gradient; $\mu$ momentum ($\approx 0.9$); $\eta$ learning rate. See
[Module 3](../lessons/module-03/lesson-01.md).

**AdamW update** — per-parameter adaptive step (Adam) with *decoupled* weight decay applied
directly to the weights, not the gradient.
$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$
$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$
$$
\theta_t = \theta_{t-1} - \eta\left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon} + \lambda\,\theta_{t-1}\right)
$$
$m_t$ first moment (mean grad), $v_t$ second moment (mean sq grad); $\hat{m},\hat{v}$
bias-corrected (undo the zero-init bias, big early on); $\beta_1{=}0.9,\beta_2{=}0.95$ typical for
LMs; $\lambda$ weight decay; $\epsilon\approx 10^{-8}$. The $\lambda\theta$ term is *decoupled*
decay — the "W" in AdamW. See [Module 3](../lessons/module-03/lesson-02.md).

**Cosine schedule with linear warmup** — ramp the LR up over warmup steps, then decay it along a
half-cosine to a floor.
$$
\eta(t) = \begin{cases}
\dfrac{t}{T_w}\,\eta_{\max} & t < T_w \\[2mm]
\eta_{\min} + \tfrac{1}{2}(\eta_{\max}-\eta_{\min})\left(1 + \cos\dfrac{\pi(t-T_w)}{T_{\max}-T_w}\right) & t \ge T_w
\end{cases}
$$
$T_w$ warmup steps; $T_{\max}$ total steps; $\eta_{\max}$ peak, $\eta_{\min}$ floor. See
[Module 3](../lessons/module-03/lesson-03.md).

**Gradient clipping (by global norm)** — cap the total gradient norm to stop loss spikes from
blowing up a step.
$$g \leftarrow g\cdot\min\!\left(1, \frac{c}{\lVert g\rVert_2}\right), \qquad
\lVert g\rVert_2 = \sqrt{\textstyle\sum_i g_i^2}$$
$c$ the max-norm threshold (commonly $1.0$); the norm is over *all* parameters concatenated. See
[Module 3](../lessons/module-03/lesson-03.md).

---

## Scaling, FLOPs & parameter counts

**Training FLOPs (the 6ND rule)** — total compute to train a dense transformer.
$$C \approx 6\,N\,D$$
$C$ FLOPs; $N$ non-embedding parameters; $D$ training tokens. The 6 is $\approx 2$ (forward
multiply-add) $+\,4$ (backward); one forward pass alone is $\approx 2ND$. See
[Module 10](../lessons/module-10/lesson-01.md).

**Parameter count of a transformer** — dominated by the per-block attention + MLP matrices.
$$N \approx 12\,L\,C^2$$
$L$ layers, $C$ width. Per block: $4C^2$ attention (Q,K,V,O) $+\,8C^2$ MLP ($4C$ hidden) $=12C^2$;
excludes embeddings $VC$. See [Module 10](../lessons/module-10/lesson-01.md).

**Kaplan power law** — loss falls as a power law in each of $N$, $D$, $C$ (compute), over many
orders of magnitude.
$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}$$
$L$ loss; $N_c,\alpha_N$ fitted constants ($\alpha_N\approx 0.076$ in the original fit). See
[Module 10](../lessons/module-10/lesson-02.md).

**Chinchilla loss surface** — joint law with an irreducible floor plus separate model-size and
data terms.
$$L(N,D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$
$E$ irreducible loss (entropy of text); $A,B,\alpha,\beta$ fitted ($\alpha,\beta\approx 0.34$).
Compute-optimal training scales $N$ and $D$ together, giving the **$\approx 20$ tokens per
parameter** rule ($D \approx 20N$). See [Module 10](../lessons/module-10/lesson-03.md).

---

## Preference learning & RLHF

**Bradley-Terry model** — probability that completion $y_w$ beats $y_l$ is the sigmoid of their
reward gap.
$$P(y_w \succ y_l \mid x) = \sigma\big(r(x,y_w) - r(x,y_l)\big)$$
$x$ prompt; $y_w$ chosen (winner), $y_l$ rejected (loser); $r$ scalar reward. See
[Module 15](../lessons/module-15/lesson-01.md).

**Reward-model loss** — train $r_\phi$ by maximizing the Bradley-Terry likelihood of the human
preferences.
$$\mathcal{L}_{\text{RM}} = -\,\mathbb{E}_{(x,y_w,y_l)}\big[\log\sigma\big(r_\phi(x,y_w) - r_\phi(x,y_l)\big)\big]$$
$r_\phi$ the learned reward (LM + scalar head). See
[Module 15](../lessons/module-15/lesson-01.md).

**KL-constrained RLHF objective** — maximize reward while staying near the reference policy.
$$\max_\pi\; \mathbb{E}_{x,\,y\sim\pi}\big[r(x,y)\big] - \beta\,\mathrm{KL}\big(\pi(\cdot\mid x)\,\|\,\pi_{\text{ref}}(\cdot\mid x)\big)$$
$\pi$ trainable policy; $\pi_{\text{ref}}$ frozen reference (the SFT model); $\beta$ KL weight. See
[Module 15](../lessons/module-15/lesson-02.md).

**PPO clipped objective** — take the largest reward-improving step whose probability ratio stays
inside $[1-\epsilon, 1+\epsilon]$.
$$\mathcal{L}_{\text{PPO}} = \mathbb{E}_t\Big[\min\big(\rho_t A_t,\; \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\,A_t\big)\Big],
\qquad \rho_t = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}$$
$\rho_t$ new/old probability ratio; $A_t$ advantage (how much better action $a_t$ was than the
baseline); $\epsilon$ clip range ($\approx 0.2$). The $\min$+clip removes the incentive to move
$\rho_t$ far from 1. See [Module 15](../lessons/module-15/lesson-02.md).

**Per-token KL penalty** — the reward actually optimized in RLHF: task reward minus per-token drift
from the reference.
$$\tilde{r}_t = r(x,y) - \beta\,\log\frac{\pi_\theta(y_t\mid y_{<t},x)}{\pi_{\text{ref}}(y_t\mid y_{<t},x)}$$

**DPO loss** — RLHF's optimum rewritten as a supervised classification loss on preference pairs;
the reward model and partition function cancel.
$$\mathcal{L}_{\text{DPO}} = -\,\mathbb{E}_{(x,y_w,y_l)}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)} - \beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]$$
$\pi_\theta$ policy, $\pi_{\text{ref}}$ frozen reference; the bracket is the *implicit reward
margin*; $\beta$ plays the RLHF KL role. No reward model, no sampling. See
[Module 15](../lessons/module-15/lesson-03.md).

**Optimal RLHF policy (DPO's starting point)** — the closed form the DPO derivation inverts.
$$\pi^{*}(y\mid x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y\mid x)\exp\!\Big(\tfrac{1}{\beta}r(x,y)\Big)$$
$Z(x)=\sum_y \pi_{\text{ref}}(y\mid x)e^{r(x,y)/\beta}$ the intractable partition function, which
cancels in the DPO margin.

---

## RL for reasoning

**GRPO group-relative advantage** — replace PPO's value network with a baseline computed from a
*group* of sampled completions to the same prompt.
$$\hat{A}_i = \frac{r_i - \mathrm{mean}(\{r_1,\dots,r_G\})}{\mathrm{std}(\{r_1,\dots,r_G\})}$$
$G$ completions sampled per prompt; $r_i$ reward of completion $i$; the group mean is the
baseline, the group std normalizes scale. Same advantage is broadcast to every token of
completion $i$. No critic model. See [Module 16](../lessons/module-16/lesson-01.md).

**RLVR reward** — reinforcement learning from *verifiable* rewards: the reward is a programmatic
check, not a learned model.
$$r(x,y) = \mathbb{1}[\,\text{verify}(x,y)\,]$$
$1$ if the answer passes (unit tests green, final answer matches), else $0$. Removes reward-model
hacking. See [Module 16](../lessons/module-16/lesson-02.md).

---

## Parameter-efficient fine-tuning

**LoRA** — freeze the pretrained weight and learn a low-rank update, cutting trainable parameters
by orders of magnitude.
$$W' = W + \Delta W, \qquad \Delta W = \frac{\alpha}{r}\,BA$$
$W$ frozen $(d_{\text{out}}\times d_{\text{in}})$; $A$ is $(r\times d_{\text{in}})$, $B$ is
$(d_{\text{out}}\times r)$ with $r \ll \min(d_{\text{in}},d_{\text{out}})$; $\alpha$ scaling.
$A$ is randomly initialized, $B$ starts at zero so $\Delta W = 0$ at step 0. QLoRA adds a
4-bit-quantized frozen $W$. See [Module 14](../lessons/module-14/lesson-03.md).
