You are building a complete, self-contained **ML Research Engineer / LLM Training Engineer course** for an experienced software engineer who wants to understand how modern frontier LLMs are actually researched, trained, evaluated, fine-tuned, aligned, and deployed.

This must NOT be a beginner PyTorch course, an API course, or a "fine-tune Llama with Hugging Face" tutorial.

The target outcome is:

> After completing this curriculum, the student should understand the complete technical pipeline behind modern LLMs well enough to read current research papers, implement the important algorithms themselves, train meaningful models on GPUs, debug training failures, design experiments, understand distributed training, perform SFT/preference optimization/RL reasoning training, implement tool use, and be able to work as an ML Research Engineer or LLM Training Engineer.

Use **Stanford CS336: Language Modeling from Scratch (Spring 2026)** as the backbone of the curriculum. The official course explicitly covers the complete language-model development pipeline, including data collection/cleaning, Transformer construction, training, evaluation, GPU optimization, distributed training, scaling, SFT/RLHF, and reasoning RL. Use its structure as a foundation, but extend it substantially beyond CS336 into modern frontier-model research.

Official CS336:
https://cs336.stanford.edu/

The course must be implementation-heavy.

The student is an experienced C# software engineer, but should be treated as a beginner in ML mathematics and LLM internals unless a concept has already been explicitly established in the course.

Use Python + PyTorch as the implementation language.

Do NOT hide important mechanisms behind high-level libraries until the underlying mechanism has first been implemented manually.

==================================================
1. COURSE PHILOSOPHY
==================================================

The course must teach:

    understand → derive → implement → test → benchmark → experiment → read research → reproduce → modify

For every important concept:

1. Explain the intuition first.
2. Explain the mathematical formulation.
3. Explicitly explain tensor shapes.
4. Work through a numerical example with actual small numbers.
5. Implement it from scratch.
6. Unit-test it.
7. Compare it against PyTorch/reference implementations.
8. Explain its computational and memory cost.
9. Explain where it appears in real LLM systems.
10. Only then introduce the optimized/production implementation.

Never say:

"PyTorch does this for us."

Instead explain what PyTorch is doing.

The student specifically wants to understand what every tensor represents.

Whenever a variable appears, explain:

- its semantic meaning
- shape
- dtype
- device
- how the shape changes
- why the operation is being performed

==================================================
2. COURSE STRUCTURE
==================================================

Create approximately 15-20 major modules.

Each module must contain:

- learning objectives
- conceptual lesson
- mathematical lesson
- worked examples
- implementation lesson
- exercises
- coding assignment
- tests
- experiment
- paper reading
- "what to notice in the paper"
- "what you can safely skip"
- checkpoint quiz
- research-engineering notes
- common mistakes
- debugging exercises

The modules should approximately follow:

MODULE 0 — ML/PyTorch foundations required for LLM research

MODULE 1 — Probability, information theory and language modeling

MODULE 2 — Derivatives, gradients and backpropagation

MODULE 3 — Optimization

MODULE 4 — Tokenization and data representation

MODULE 5 — Transformer architecture from scratch

MODULE 6 — GPT-2-style language model from scratch

MODULE 7 — Pretraining infrastructure and training loops

MODULE 8 — GPU performance and memory

MODULE 9 — Distributed training

MODULE 10 — Scaling laws and compute-optimal training

MODULE 11 — LLM data engineering

MODULE 12 — Modern architectures: Llama, GQA/MQA, RoPE, RMSNorm, SwiGLU, MoE

MODULE 13 — Evaluation and experimentation

MODULE 14 — Supervised fine-tuning

MODULE 15 — Preference learning / RLHF / DPO

MODULE 16 — Reinforcement learning for reasoning

MODULE 17 — Modern reasoning models

MODULE 18 — Tool use and agents

MODULE 19 — Research engineering / reproducibility / experiment design

MODULE 20 — Capstone frontier-model training project

You may adjust the exact number of modules if doing so produces a better curriculum.

==================================================
3. REQUIRED PAPER CURRICULUM
==================================================

Build a carefully ordered reading curriculum of approximately 30 papers.

DO NOT merely list papers.

For every paper provide:

- number
- title
- authors
- year
- verified canonical URL
- prerequisite modules
- why it appears at this point
- what the student MUST understand
- what can be skipped
- important figures/tables to inspect
- implementation consequences
- "code this after reading"
- connections to previous papers
- connections to later papers

The paper sequence should approximately cover:

1. GPT-2 — Language Models are Unsupervised Multitask Learners
2. GPT-3 — Language Models are Few-Shot Learners
3. Scaling Laws for Neural Language Models
4. Chinchilla — Training Compute-Optimal Large Language Models
5. Megatron-LM
6. ZeRO
7. FlashAttention
8. RoFormer / RoPE
9. LLaMA
10. Llama 3
11. OLMo / OLMo 2
12. Switch Transformers
13. DeepSeekMoE
14. DeepSeek-V3
15. FLAN / instruction tuning
16. Self-Instruct
17. InstructGPT
18. Anthropic RLHF work
19. Constitutional AI
20. RLAIF / AI feedback
21. DPO
22. Chain-of-Thought
23. STaR
24. Process-level verification / Let's Verify Step by Step
25. DeepSeekMath
26. DeepSeek-R1
27. Toolformer
28. Gorilla
29. ReAct
30. SWE-agent or an equivalent modern agent/tool-use paper

You may replace individual papers if a newer or more technically relevant paper is clearly superior, but explain the replacement.

IMPORTANT:

Verify EVERY paper URL before putting it into the course.

Do not assume an arXiv identifier is correct.

For GPT-2 specifically, use the actual OpenAI GPT-2 paper:

"Language Models are Unsupervised Multitask Learners"

Official PDF:
https://cdn.openai.com/better-language-models/language-models.pdf

Also link the official OpenAI GPT-2 repository where useful:

https://github.com/openai/gpt-2

Never substitute an unrelated paper because an arXiv identifier happens to be associated with another document.

==================================================
4. DEPENDENCY GRAPH
==================================================

Create an explicit dependency graph.

For example:

Probability
    ↓
Cross entropy
    ↓
Language modeling
    ↓
Tokenization
    ↓
Attention
    ↓
Transformer
    ↓
GPT-2
    ↓
Pretraining
    ↓
Scaling
    ↓
Distributed training
    ↓
Llama
    ↓
MoE
    ↓
DeepSeek-V3
    ↓
SFT
    ↓
RLHF
    ↓
DPO
    ↓
RL
    ↓
Reasoning
    ↓
DeepSeek-R1
    ↓
Tool use
    ↓
Agents

Make the dependencies visible in the course UI.

A student should always be able to answer:

"What do I need to understand before starting this lesson?"

==================================================
5. MATHEMATICS CURRICULUM
==================================================

Include all mathematics required for serious LLM research.

Do NOT assume that knowing matrices means the student understands the mathematics required for modern ML.

Cover:

LINEAR ALGEBRA
- vectors
- matrices
- tensors
- matrix multiplication
- transpose
- dot products
- norms
- projections
- eigenvalues/eigenvectors
- SVD
- rank
- low-rank approximation
- attention as matrix operations

CALCULUS
- functions
- derivatives
- partial derivatives
- chain rule
- gradients
- Jacobians
- directional derivatives
- computational graphs
- backpropagation

PROBABILITY
- random variables
- distributions
- expectation
- variance
- conditional probability
- Bayes
- likelihood
- log likelihood
- entropy
- cross entropy
- KL divergence

OPTIMIZATION
- gradient descent
- SGD
- momentum
- Adam
- AdamW
- weight decay
- learning-rate schedules
- warmup
- cosine decay
- gradient clipping
- second-order intuition

MODERN ML MATH
- softmax
- log-softmax
- cross entropy
- numerical stability
- temperature
- sampling
- KL constraints
- policy gradients
- advantage
- importance sampling
- PPO
- GRPO
- preference objectives
- DPO derivation

Every important equation should be accompanied by a concrete numerical example.

==================================================
6. IMPLEMENTATION REQUIREMENTS
==================================================

The course should progressively build one coherent codebase.

Do NOT create unrelated toy scripts for every chapter.

Build a repository approximately like:

llm-research-course/

    src/
        tokenizer/
        model/
        attention/
        optim/
        training/
        distributed/
        data/
        evaluation/
        sft/
        preference/
        rl/
        reasoning/
        tools/
        agents/

    tests/

    experiments/

    configs/

    scripts/

    notebooks/

    reports/

    checkpoints/

Each major stage should extend the previous implementation.

==================================================
7. FROM-SCRATCH GPT PROJECT
==================================================

The student must implement a GPT-style Transformer from scratch.

Implement manually:

- tokenizer
- vocabulary
- embeddings
- positional encoding
- RoPE
- Q/K/V projections
- attention
- causal masking
- multi-head attention
- grouped-query attention
- MLP
- SwiGLU
- RMSNorm
- residual connections
- Transformer block
- final normalization
- LM head
- cross entropy
- AdamW
- learning-rate schedule
- gradient clipping
- mixed precision
- checkpointing
- generation
- temperature sampling
- top-k
- top-p

Do not introduce Hugging Face Trainer until after the complete training system has been implemented manually.

==================================================
8. NUMERICAL MICRO-EXAMPLES
==================================================

Every major algorithm must have a tiny hand-computable example.

For example:

Given:

X =
[[1,2],
 [3,4]]

and

W =
[[...],
 [...]]

show exactly:

XW

Then show the same operation in PyTorch.

Do this for:

- linear layers
- attention
- softmax
- cross entropy
- gradient calculation
- Adam
- AdamW
- RMSNorm
- RoPE
- residual connections
- LoRA
- DPO
- policy gradient
- PPO
- GRPO

The student should be able to trace a single token through the model.

==================================================
9. TRAINING SYSTEM
==================================================

Build an actual training loop.

Cover:

- batching
- sequence packing
- gradient accumulation
- micro-batches
- gradient clipping
- optimizer state
- checkpointing
- resuming
- validation
- logging
- experiment configuration
- random seeds
- reproducibility
- throughput
- tokens/sec
- FLOPs
- MFU
- memory usage

Explicitly teach the distinction between:

batch size
micro-batch size
gradient accumulation
tokens per batch
sequence length
global batch size

Include exercises where the student calculates these manually.

==================================================
10. GPU / SYSTEMS ENGINEERING
==================================================

This is REQUIRED.

Teach:

- GPU architecture
- CUDA basics
- memory hierarchy
- HBM
- SRAM/shared memory
- memory bandwidth
- FLOPS
- arithmetic intensity
- kernel launches
- synchronization
- CUDA graphs
- profiling
- PyTorch profiler
- Triton
- FlashAttention
- fused kernels

Have the student benchmark:

1. naive attention
2. optimized attention
3. PyTorch attention
4. FlashAttention

Explain WHY the performance differs.

==================================================
11. DISTRIBUTED TRAINING
==================================================

Teach from first principles:

- data parallelism
- DDP
- all-reduce
- NCCL
- FSDP
- ZeRO
- tensor parallelism
- pipeline parallelism
- sequence/context parallelism
- expert parallelism
- communication/computation overlap

Implement a minimal distributed trainer.

Then compare it with production systems.

Eventually introduce:

- DeepSpeed
- Megatron-LM
- Megatron Core
- PyTorch FSDP

The student must understand the systems before using the frameworks.

==================================================
12. DATA ENGINEERING
==================================================

Teach the actual LLM pretraining data pipeline.

Cover:

- Common Crawl
- web extraction
- language identification
- quality filtering
- toxicity filtering
- PII considerations
- deduplication
- exact deduplication
- fuzzy deduplication
- MinHash
- near-duplicate detection
- contamination
- benchmark leakage
- dataset mixing
- synthetic data
- data curriculum
- token counting

Build a small real preprocessing pipeline.

Do not treat "dataset = load_dataset(...)" as sufficient.

==================================================
13. SCALING LAWS
==================================================

Implement experiments demonstrating:

- parameter scaling
- dataset scaling
- compute scaling
- Chinchilla-style compute optimality

Fit scaling-law curves from experimental results.

Teach:

- FLOPs estimation
- parameter count
- activation memory
- optimizer memory
- training tokens
- GPU-hours
- cost estimation

Have the student predict the compute required for progressively larger models.

==================================================
14. MODERN ARCHITECTURES
==================================================

Implement and compare:

- GPT-style dense Transformer
- Llama-style Transformer
- MHA
- MQA
- GQA
- RoPE
- RMSNorm
- SwiGLU
- MoE
- top-k routing
- shared experts
- expert balancing

Then implement a small DeepSeek-style MoE.

Explain why modern companies use these architectural choices.

==================================================
15. EVALUATION
==================================================

Teach evaluation as a first-class research discipline.

Cover:

- train loss
- validation loss
- perplexity
- zero-shot evaluation
- few-shot evaluation
- benchmark contamination
- calibration
- instruction-following evaluation
- reasoning evaluation
- tool-use evaluation
- human evaluation
- preference evaluation
- regression testing

Build an evaluation harness.

Every experiment must produce reproducible metrics.

==================================================
16. SUPERVISED FINE-TUNING
==================================================

Implement SFT manually.

Cover:

- instruction datasets
- chat templates
- prompt formatting
- response masking
- loss masking
- packing
- sequence length
- full fine-tuning
- LoRA
- QLoRA

Implement LoRA from scratch before using PEFT.

Explain exactly why LoRA works mathematically.

==================================================
17. RLHF
==================================================

Teach the complete classical RLHF pipeline:

    pretrained model
        ↓
    SFT
        ↓
    preference dataset
        ↓
    reward model
        ↓
    PPO
        ↓
    aligned model

Implement simplified versions of:

- reward model
- Bradley-Terry preference modeling
- policy gradient
- advantage estimation
- PPO
- KL penalty

Use tiny models/datasets first.

The student must understand every tensor involved in PPO.

==================================================
18. DPO AND PREFERENCE OPTIMIZATION
==================================================

Derive DPO from the underlying preference objective.

Implement DPO manually.

Then compare:

- SFT
- reward-model + PPO
- DPO

Explain when each approach is useful.

==================================================
19. REASONING RL
==================================================

This is a major section.

Teach:

- chain-of-thought
- self-consistency
- process reward
- outcome reward
- verifier models
- rejection sampling
- synthetic reasoning data
- RLVR
- policy optimization
- GRPO

Implement a small reasoning RL system.

Use tasks with automatically verifiable answers such as:

- arithmetic
- algebra
- simple programming
- symbolic problems

The course must explain why verifiable rewards are particularly useful.

==================================================
20. DEEPSEEK-R1
==================================================

Use DeepSeek-R1 as a major case study.

Reconstruct its conceptual pipeline.

Compare:

DeepSeek-R1-Zero
vs
DeepSeek-R1

Explain:

- pure RL
- cold-start data
- SFT
- reasoning data
- RL
- rejection sampling
- distillation

The student should implement a miniature analogue.

Do not claim that this is necessarily the exact proprietary pipeline used by OpenAI or Anthropic. Clearly distinguish published evidence from inference.

==================================================
21. TOOL USE
==================================================

Teach:

- tool schemas
- function calling
- structured outputs
- API selection
- argument generation
- tool results
- retry
- tool errors
- planning
- observation/action loops

Implement a tool-use model/system.

Start with deterministic tools:

- calculator
- Python
- search
- database query

Then train/evaluate tool selection.

==================================================
22. AGENTS
==================================================

Teach the evolution:

    LLM
      ↓
    tool calling
      ↓
    ReAct
      ↓
    planning
      ↓
    observation
      ↓
    iterative action
      ↓
    agent

Build a simple software-engineering agent that can:

- inspect files
- modify code
- run tests
- observe failures
- fix code
- repeat

Use SWE-agent as a research reference.

==================================================
23. RESEARCH ENGINEERING
==================================================

This section is essential for the target career.

Teach:

- experiment design
- hypothesis formation
- ablation studies
- baselines
- controls
- statistical significance
- confidence intervals
- reproducibility
- seed variance
- checkpoint selection
- hyperparameter sweeps
- experiment tracking
- failure analysis
- paper reproduction
- negative results
- scientific writing

Require the student to write actual experiment reports.

Each report should contain:

QUESTION
HYPOTHESIS
METHOD
BASELINE
VARIABLES
RESULTS
ANALYSIS
LIMITATIONS
CONCLUSION
NEXT EXPERIMENT

==================================================
24. DEBUGGING
==================================================

Create deliberate broken implementations.

Examples:

- wrong causal mask
- incorrect tensor transpose
- wrong softmax dimension
- detached gradient
- incorrect loss normalization
- incorrect gradient accumulation
- wrong Adam bias correction
- incorrect weight decay
- NaNs
- exploding gradients
- vanishing gradients
- incorrect distributed reduction
- duplicated samples
- data leakage
- tokenizer mismatch
- checkpoint incompatibility

The student must diagnose each failure rather than simply receiving the corrected code.

==================================================
25. CAREER-LEVEL CAPSTONE
==================================================

Build one coherent final project:

"Mini Frontier LLM"

The student should:

1. Build tokenizer
2. Build GPT-style Transformer
3. Pretrain it
4. Build a real data pipeline
5. Train on GPU
6. Profile it
7. Optimize it
8. Add distributed training
9. Perform scaling experiments
10. Evaluate it
11. Perform SFT
12. Implement LoRA
13. Build preference data
14. Implement DPO
15. Implement reward modeling
16. Implement simplified PPO
17. Implement GRPO/RLVR
18. Train a reasoning model on a verifiable task
19. Add tool use
20. Build a simple agent
21. Perform ablation experiments
22. Produce a research report

The final project should have a reproducible configuration and README.

==================================================
26. HARDWARE TRACK
==================================================

The course must distinguish between:

LOCAL CPU
LOCAL GPU
A100
H100
H200
B200
multi-GPU
multi-node

For every assignment state:

- minimum hardware
- recommended hardware
- expected runtime
- approximate GPU memory requirement
- approximate GPU-hours
- whether CPU-only development is possible

Design experiments so the student can learn without requiring thousands of dollars.

For large-scale exercises, provide scaled-down versions.

==================================================
27. PRODUCTION FRAMEWORKS
==================================================

Only introduce production frameworks after explaining the underlying mechanism.

Cover where appropriate:

- PyTorch
- Hugging Face Transformers
- PEFT
- TRL
- FSDP
- DeepSpeed
- Megatron-LM
- Megatron Core
- Triton
- FlashAttention
- vLLM

For each framework explain:

"What problem does this solve?"

and:

"What would I have to implement myself if this framework did not exist?"

==================================================
28. COURSE WEBSITE / UX
==================================================

Build a clean technical course website.

Every lesson should have:

[Concept]

[Why it matters]

[Math]

[Concrete example]

[Tensor shapes]

[Implementation]

[Tests]

[Exercise]

[Research connection]

[Paper]

[What to read]

[What to skip]

[Code this]

[Checkpoint]

Do not dump enormous walls of text onto a page.

However, err on the side of including too much technical explanation rather than too little.

The student prefers learning in small sequential chunks.

For difficult material:

1. explain one concept
2. give a small example
3. ask a question/exercise
4. wait for confirmation
5. continue

==================================================
29. NO ANSWER SPOILERS
==================================================

For exercises, do NOT immediately reveal the answer.

Provide:

- problem
- relevant information
- optional hint
- stronger hint
- solution hidden/revealable

The student should be forced to reason.

==================================================
30. QUIZZES
==================================================

Every module must contain quizzes.

Include:

- conceptual questions
- tensor-shape questions
- numerical calculations
- code tracing
- debugging
- research interpretation

Generate Anki-compatible CSV exports for important concepts.

==================================================
31. PAPER READING MODE
==================================================

Create a special paper-reading template.

For every paper:

FIRST:

"What problem are they solving?"

THEN:

"What did people do before this?"

THEN:

"What is the key idea?"

THEN:

"What changed technically?"

THEN:

"What equations matter?"

THEN:

"What experiments prove the claim?"

THEN:

"What should you implement?"

THEN:

"What can you skip?"

THEN:

"How did this influence later LLMs?"

The student should never have to read a 40-page paper linearly without guidance.

==================================================
32. FRONTIER-LAB CONNECTION
==================================================

Throughout the course explicitly distinguish:

PUBLICLY DOCUMENTED
from
REASONABLE INDUSTRY PRACTICE
from
INFERENCE/SPECULATION

Never claim:

"OpenAI definitely does X"

unless publicly documented.

Instead write:

"OpenAI has publicly documented X."

or:

"This technique is widely used in large-scale open training systems."

Use OpenAI, Anthropic, DeepSeek, Meta, Google, NVIDIA and other primary sources whenever possible.

==================================================
33. PAPER LINK VERIFICATION
==================================================

Before generating the course content, verify every paper.

For each paper store:

title
authors
year
canonical URL
arXiv ID if applicable
official/project URL if available
GitHub implementation if available

Prefer:

official paper
arXiv
official project page
official GitHub

Do not use random paper aggregators.

The GPT-2 paper MUST point to:

https://cdn.openai.com/better-language-models/language-models.pdf

and not an unrelated arXiv result.

==================================================
34. COMPLETION CRITERIA
==================================================

Do not consider the course complete merely because all pages exist.

A module is complete only when the student can:

EXPLAIN it
DERIVE the important mathematics
IMPLEMENT it
TEST it
BENCHMARK it
DEBUG it
RELATE it to a research paper
and explain WHY modern LLM systems use it.

At the end, generate a "Research Engineer Readiness Matrix":

Skill                         Beginner / Intermediate / Advanced
---------------------------------------------------------------
PyTorch
Transformer internals
Backpropagation
Optimization
LLM pretraining
Data engineering
GPU optimization
Triton
Distributed training
FSDP
Megatron
Scaling laws
Evaluation
SFT
LoRA
DPO
RLHF
PPO
GRPO
RLVR
Reasoning training
MoE
Tool use
Agents
Experiment design
Paper reproduction
Research writing

The student's goal is not to merely finish the website.

The goal is to become capable of independently reading a new LLM training paper and answering:

"What are they doing, why are they doing it, how would I implement it, how would I test it, and what experiment would I run next?"

==================================================
35. FINAL PRINCIPLE
==================================================

Build this as a **professional research-engineering apprenticeship**, not as a collection of tutorials.

The student should gradually move through:

MATHEMATICS
    ↓
PYTORCH
    ↓
TRANSFORMERS
    ↓
GPT-2
    ↓
PRETRAINING
    ↓
GPU SYSTEMS
    ↓
DISTRIBUTED TRAINING
    ↓
SCALING
    ↓
DATA
    ↓
MODERN ARCHITECTURES
    ↓
SFT
    ↓
RLHF
    ↓
DPO
    ↓
REASONING RL
    ↓
DEEPSEEK-R1
    ↓
TOOL USE
    ↓
AGENTS
    ↓
RESEARCH

The course should feel like one continuous story.

Every new concept should answer a problem created by the previous stage.

Do not create isolated chapters.

For example:

"We built GPT-2. Now it doesn't follow instructions. Why?"

→ SFT.

"Now it follows instructions but doesn't reliably prefer better answers. Why?"

→ preference learning.

"Now we want reasoning. Why isn't ordinary SFT enough?"

→ reasoning data and RL.

"Now we need external information/actions."

→ tool use.

"Now we need multiple iterative actions."

→ agents.

That causal progression is one of the most important requirements of the course.