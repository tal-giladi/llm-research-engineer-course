# Staging — things worth considering

Temporary document for the current week. The daily run appends class **A** and **B**
candidates here (full records); the weekly review evaluates them, archives this content into
`weekly/YYYY-Www.md`, and resets this file. Nothing here is course material.

## Candidates

### C-20261117-01 · Numerically stable KL / JSD / entropy / K1-K2-K3 estimators between nearly equal distributions (follow-up to Softmax Collapse/Cancellation)

- **Class:** B
- **Date discovered:** 2026-11-17
- **Date published:** 2026-11-13
- **Source:** https://arxiv.org/abs/2611.18511 (verified; full text read at https://arxiv.org/html/2611.18511v1)
- **Organization/researchers:** Brandon Amos (Reflection AI, per the paper)
- **Category:** RL/post-training
- **What changed:** A second, independent author extends Barron et al.'s softmax/cross-entropy numerical analysis (deferred topic) to the divergences the course uses in distillation and RL: forward and reverse KL, entropy, generalized JSD, and the Monte Carlo KL estimators K1/K2/K3 (K3 is the GRPO/DeepSeekMath KL penalty).
- **Technical summary:** When the two distributions are nearly equal (exactly the regime of a KL penalty against a reference policy early in RL, or a converged distillation student), naive FP32 log-softmax subtraction and the cancellation inside the KL sum corrupt both value and gradient. Examples in the paper (authors): a sampled log-ratio with h = 1e-8 rounds to 0; forward KL with logit gap 2^-25 returns zero value and gradient although p != q; K3 at h = 1e-8 has zero value and zero input derivatives; entropy near uniform has an accurate value but a gradient that rounds to zero. Proposed fixes (eight): rebuild log-ratios from logit differences with expm1/log1p, compute the non-negative remainder E2(x) = e^x - 1 - x with a Taylor series near 0, a zero-sum gradient correction, centring logit differences for the reverse-KL gradient, a stable K3 remainder and derivatives, and h = -log1p(beta*expm1(-s)) for the JSD mixture log-ratio. JAX listings validated by property-based tests are in the paper. Stated limits: the reconstruction can overflow far from equality, and FP32 logit subtraction can still erase tiny differences (open problem).
- **Why it might matter:** The course implements KL (01.3), the K3 penalty in GRPO (16.x) and forward/reverse KL distillation (17.4) from scratch; this gives a concrete, testable failure mode and fix for those exact functions, and turns the deferred Barron topic from a single preprint into a two-author line of work with runnable code listings.
- **Evidence of adoption:** none found. The paper isolates arithmetic and does not measure training outcomes; it does not separately reproduce Barron et al.'s Softmax Collapse.
- **Major organizations using it:** none known.
- **Open-source implementation:** code listings inside the paper (no repository link). The paper links a Barron et al. implementation page (https://jonbarron.info/softmaxfix/), which could not be fetched from this session (host not allowed), so it is unverified.
- **Paper:** https://arxiv.org/abs/2611.18511 ; related https://arxiv.org/abs/2610.31642 (verified)
- **Code:** none (listings in paper)
- **Relationship to existing course material:** `research/deferred/softmax-cross-entropy-numerical-fixes.md` (WAIT, next review 2026-11-22; its "what would change the decision" asks for code or a check against the course's 01.3/01.4, 02.3 and 17.4 implementations); lessons/module-01/lesson-03.md, lessons/module-17/lesson-04.md, GRPO KL penalty in module 16.
- **Potential course lesson:** dated extension of 01.3 / 17.4: "KL between nearly equal distributions in FP32", with a from-scratch stable K3 and a property-based test in `code/tests/`.
- **Confidence:** medium
- **Recommendation:** attach to the 2026-11-22 re-review of `softmax-cross-entropy-numerical-fixes`; check the paper's listings against the course's own KL/K3/distill code before deciding. Still no training-outcome evidence.
