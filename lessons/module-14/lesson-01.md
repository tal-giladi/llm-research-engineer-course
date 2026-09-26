# 14.1 · Instruction data & chat templates

<div class="prereq">
<p><strong>Prerequisites:</strong> the assembled model from <a href="#/lessons/module-06/lesson-01">06.1 · Assembling GPT-2</a> and how it generates text; special/delimiter tokens and how a tokenizer treats them from <a href="#/lessons/module-04/lesson-03">04.3 · Special tokens</a>; the byte-level BPE <code>encode</code>/<code>decode</code> from <a href="#/lessons/module-04/lesson-02">04.2 · BPE</a>.</p>
<p><strong>You will learn:</strong> why a pretrained language model completes text but does not follow instructions; what an instruction dataset is (the <code>(instruction, [input], output)</code> triple); and what a <strong>chat template</strong> is — the exact, deterministic rule that flattens a list of role-tagged messages into one delimited token string, so the model can learn where the assistant turn starts and when to stop.</p>
<p><strong>Why this matters for ML:</strong> everything after pretraining — SFT, reward modeling, RLHF, DPO — operates on chat-formatted data. The chat template is the interface between "a conversation" and "a sequence of token ids." Get it inconsistent between training and inference and the model silently degrades: it never sees the delimiter it was trained to stop on, so it rambles. This is the first lesson of <strong>post-training</strong>.</p>
</div>

## The causal story so far

In Modules 6–7 we built GPT-2 from scratch and pretrained it. The training objective was exactly one thing: **given the tokens so far, predict the next token** (the cross-entropy loss of [01.4](lessons/module-01/lesson-04.md)). Trained on a large pile of internet text, the model becomes extremely good at *continuing* whatever text you give it.

That is not the same as *following an instruction*. Watch what a pure pretrained model tends to do:

```text
Prompt:   Write a haiku about the sea.
Continuation:  Write a haiku about the mountains. Write a haiku about the city.
               Write a haiku about...
```

It saw many web pages that are *lists of writing prompts*, so the most probable continuation of "Write a haiku about the sea." is another prompt in the list — not a haiku. The model is doing its job perfectly; its job is just not what we want.

<div class="callout key"><p>A pretrained LM models <em>text continuation</em> — "what tokens plausibly come next in a document like this." It does not model "answer the user's request." Nothing in pretraining ever told it that a prompt is a request to be fulfilled rather than text to be extended. <strong>Supervised fine-tuning (SFT)</strong> teaches it that behavior by continuing training on curated <em>(prompt, ideal response)</em> pairs.</p></div>

### 1. Intuition: SFT is just more next-token training, on different data

SFT does not change the model architecture or the loss function. It is the *same* next-token cross-entropy training from Module 7. Two things change:

1. **The data.** Instead of raw documents, we train on demonstrations of the behavior we want: a user asks something, and a high-quality *response* follows. After enough of these, "the plausible continuation of a user request" becomes "a helpful answer to it," because that is what the fine-tuning corpus looks like.
2. **Where we apply the loss.** We only score the *response* tokens, not the prompt — the topic of [14.2](lessons/module-14/lesson-02.md). This lesson sets up the data and formatting so 14.2 can do the masking.

The whole post-training pipeline (SFT → reward model → RLHF/DPO, Modules 15) is about shaping *behavior* on top of the *knowledge* pretraining already installed. SFT is the first and cheapest step.

## Instruction datasets: the `(instruction, [input], output)` triple

An instruction-tuning example is a small record. The classic FLAN / Alpaca shape is three fields:

- **instruction** — what to do: `"Translate the sentence to French."`
- **input** *(optional)* — the thing to do it to: `"The cat sat on the mat."`
- **output** — the demonstrated ideal response: `"Le chat s'est assis sur le tapis."`

Some tasks need no separate input (the instruction is self-contained, e.g. `"Write a haiku about the sea."` → the poem). Where there *is* an input, the instruction and input together form the prompt, and the output is what we train the model to produce.

Two influential ways such datasets are built:

- **FLAN** (Wei et al. 2021): take *existing* NLP datasets (translation, summarization, QA, sentiment...) and rewrite each as a natural-language instruction with many phrasing templates. Fine-tuning on this mixture of tasks makes the model generalize to *unseen* instructions zero-shot.
- **Self-Instruct** (Wang et al. 2022): bootstrap instruction data from the model itself — seed a few human-written examples, have a strong LM generate many more `(instruction, input, output)` triples, filter for quality, and train on the result. This is how you get large instruction sets without hand-writing every one (Alpaca used this recipe).

<div class="callout paper"><p><strong>Read:</strong> <a href="#/papers/index">FLAN</a> for "instructions as a training distribution" and <a href="#/papers/index">Self-Instruct</a> for generating that distribution cheaply. Both are core Module 14 readings. Skip the long per-task appendices on a first pass; read the data-construction sections closely.</p></div>

A single-turn triple is really just a two-message chat: a **user** message (instruction + input) and an **assistant** message (output). Multi-turn data adds more user/assistant messages, and often a **system** message that sets persistent behavior ("You are a terse assistant."). So the general object we must format is a *list of role-tagged messages*.

## Chat templates: from a message list to one token string

### 1. Intuition

The model eats a flat sequence of token ids, not a list of dicts. A **chat template** is the fixed rule that serializes

```python
[{"role": "system",    "content": "You are a terse assistant."},
 {"role": "user",      "content": "Capital of France?"},
 {"role": "assistant", "content": "Paris."}]
```

into one string, inserting **special delimiter tokens** that mark where each turn begins and ends. The delimiters are the load-bearing part: they are the only signal the model has for "a new turn started," "this is the assistant speaking," and — crucially — "the turn is over, stop generating."

### 2. The template we will use

We use four delimiter tokens. Three role headers open a turn, and one end-of-turn token closes it:

$$
\texttt{<|system|>}, \quad \texttt{<|user|>}, \quad \texttt{<|assistant|>}, \quad \texttt{<|end|>}.
$$

The rule for one message is: emit its role header, a newline, the content, then the end token and a newline. Concatenate over all messages. For the three-message chat above the template produces exactly this string (newlines shown as `\n`):

```text
<|system|>\nYou are a terse assistant.<|end|>\n<|user|>\nCapital of France?<|end|>\n<|assistant|>\nParis.<|end|>\n
```

<div class="callout warn"><p>These delimiters must be treated as <strong>atomic</strong> tokens by the tokenizer — one id each, never split into <code>&lt;</code>, <code>|</code>, <code>system</code>, … — and they must be <em>reserved</em> so no ordinary user text can forge them. That is exactly the special-token machinery of <a href="#/lessons/module-04/lesson-03">04.3</a>. Our from-scratch <code>BPETokenizer</code> does not reserve them, so in code below we encode <em>segment by segment</em> to keep the boundaries exact; a production tokenizer adds each delimiter to the vocabulary as a single id.</p></div>

### 3. Why a consistent template matters

The template is a *contract* the model learns during SFT. After fine-tuning, the model has learned:

- After `<|assistant|>\n`, my job is to produce a helpful response.
- When I have finished, I emit `<|end|>` and stop.

At inference we hand the model the same format but *stop before* the assistant content and let it generate — this is the **generation prompt**. We render the conversation up to and including a bare `<|assistant|>\n` and call `generate` (from [06.1](lessons/module-06/lesson-01.md)); the model fills in the response and emits `<|end|>`, at which point we stop decoding:

```text
<|system|>\nYou are a terse assistant.<|end|>\n<|user|>\nCapital of France?<|end|>\n<|assistant|>\n
```

If train-time and inference-time formatting disagree — a missing newline, a different header, no end token — the model is now off-distribution. The most common bug is a model that **never stops**: if it was trained with `<|end|>` as the stop signal but you forget to sample until it and treat that token as the end, it runs to the token limit every time.

### 5. From-scratch PyTorch

The module `llmre.sft.chat_template` implements exactly this. `render_chat` produces the string; `render_with_response_mask` also tells us which tokens are the assistant's (so 14.2 can mask the loss):

```python
from llmre.sft.chat_template import render_chat, render_with_response_mask
from llmre.tokenizer.bpe import BPETokenizer

messages = [
    {"role": "system",    "content": "You are a terse assistant."},
    {"role": "user",      "content": "Capital of France?"},
    {"role": "assistant", "content": "Paris."},
]

# 1) render to a string (training format: includes the assistant content)
print(render_chat(messages))
# <|system|>\nYou are a terse assistant.<|end|>\n<|user|>\n...<|assistant|>\nParis.<|end|>\n

# 2) render to a string for INFERENCE (stop right before the answer)
print(render_chat(messages[:2], add_generation_prompt=True))
# ...<|user|>\nCapital of France?<|end|>\n<|assistant|>\n

# 3) render to ids + a response mask, using any encoder
tok = BPETokenizer(); tok.train("some corpus ...", vocab_size=300)
ids, response_mask = render_with_response_mask(messages, tok.encode)
# ids[i] is a token; response_mask[i] == 1 iff it is an assistant-response token
```

The implementation walks the messages as `(text, is_response)` **segments** — header, content, closer — and encodes each segment independently, so the response mask is exact even though our BPE does not reserve the delimiters:

```python
def render_with_response_mask(messages, encode, add_generation_prompt=False):
    input_ids, response_mask = [], []
    for text, is_resp in _segments(messages, add_generation_prompt):
        seg_ids = encode(text)
        input_ids.extend(seg_ids)
        response_mask.extend([1 if is_resp else 0] * len(seg_ids))
    return input_ids, response_mask
```

The assistant's **content and its `<|end|>` closer** are marked as response (`is_resp = True`); everything else — system text, user text, all role headers — is marked prompt. Marking the closer as a response token is deliberate: it is how the model learns to *emit the stop token*.

### 4. Shapes / dtype / device

There are no tensors in this module. `messages` is a `list[dict[str, str]]`; `render_chat` returns a `str`; `render_with_response_mask` returns two equal-length `list[int]` (Python ints, not a tensor). The ids become a `torch.long` tensor only later, when 14.2 builds the labels and the training loop batches them onto the device.

<div class="callout pt"><p>HuggingFace's <code>tokenizer.apply_chat_template(messages, add_generation_prompt=...)</code> is this exact function; the template itself is a Jinja string shipped with each chat model. Different model families use different delimiters (<code>&lt;|im_start|&gt;</code>/<code>&lt;|im_end|&gt;</code>, <code>[INST]</code>/<code>[/INST]</code>, …) but the <em>job</em> is identical to <code>render_chat</code>: deterministic message-list → delimited string, with a generation-prompt switch for inference.</p></div>

## Common mistakes

- **Reusing the wrong template.** Formatting data for model A with model B's delimiters. The tokens exist in neither vocabulary as intended and the turn boundaries are gibberish.
- **Forgetting `add_generation_prompt` at inference.** Without the trailing `<|assistant|>\n`, the model is not cued that it is its turn and may continue the *user* message.
- **Not stopping on `<|end|>`.** The model dutifully emits the stop token; if your decode loop ignores it, you get a correct answer followed by hallucinated extra turns.
- **Training on the prompt tokens.** Covered in [14.2](lessons/module-14/lesson-02.md) — you must mask them out, or you teach the model to generate user turns.

## Debugging exercise

A colleague reports: "After SFT, the model answers correctly but then keeps going, inventing a fake follow-up question and answering that too, until it hits the length limit." The training data was formatted with `render_chat` (which appends `<|end|>\n` after every turn). What is the single most likely bug, and where is it — in training or in inference?

<details><summary>Answer</summary>

Almost certainly **inference**, not training. The training data does include `<|end|>`, so the model *did* learn to emit a stop token at the end of its turn. The bug is that the generation loop is not treating `<|end|>` as a stop condition — `generate` keeps sampling for the full `max_new_tokens`, so after emitting `<|end|>` the model just starts a new (hallucinated) turn. Fix: stop decoding as soon as the sampled token is the `<|end|>` id (or truncate the output at the first `<|end|>`). If instead the model *never* emitted `<|end|>`, the bug would be in training formatting.

</details>

## Check yourself

<details><summary>Why does a pretrained model continue "Write a haiku about the sea." with another writing prompt instead of a haiku?</summary>

Because it models text continuation, and in its training corpus the most frequent continuation of a line that looks like a writing prompt is *another writing prompt* (prompt lists, exercise sheets). It has never been trained that a prompt is a request to fulfill. SFT changes the training distribution so that the continuation of a request is a helpful answer.

</details>

<details><summary>What are the three fields of a classic instruction-tuning example, and which one is optional?</summary>

`instruction`, `input` (optional), and `output`. The instruction says what to do; the optional input is the data to do it to; the output is the demonstrated ideal response. Instruction + input form the prompt; output is what we train the model to produce.

</details>

<details><summary>What does <code>add_generation_prompt=True</code> change, and when do you use it?</summary>

It appends a trailing role header (`<|assistant|>\n`) with no content, cueing the model that it is now the assistant's turn to generate. You use it at **inference** — render the conversation so far, then call `generate` to fill in the assistant response. You do *not* use it when formatting training data, where the assistant content is already present.

</details>

<details><summary>Why must the delimiter tokens be atomic, reserved special tokens rather than ordinary text?</summary>

If <code>&lt;|end|&gt;</code> were tokenized as ordinary characters, (a) it could be split inconsistently across contexts, blurring the boundary the model must learn, and (b) a user could type the literal string and forge a turn boundary — a prompt-injection vector. Reserving each delimiter as one fixed id makes the boundary unambiguous and unforgeable. (See <a href="#/lessons/module-04/lesson-03">04.3</a>.)

</details>

## Next

We can now turn a conversation into a delimited token string and mark which tokens are the assistant's. Next we use that mask to train correctly: compute the loss on the response tokens **only**, and pack several short examples into one block without letting them leak into each other.

Continue to [14.2 · Loss masking & packing](lessons/module-14/lesson-02.md).
