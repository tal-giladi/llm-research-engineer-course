# Preference-format examples for lesson 15.1

These nine comparisons were written for the lesson. They are educational examples,
not a sample from a human-labeling study or enough data to train a useful grader.

`train.jsonl` has six comparisons; `validation.jsonl` has three different prompts.
Each record has `prompt`, `chosen`, and `rejected` string fields accepted by
`llmre.preference.train`. The same prompt and chat format are used for both answers.

The labels favor the requested tone, audience, constraints, actionable detail, and
appropriate uncertainty. Rejected answers need not be factually false. The gradient
accumulation example prefers a longer, more concrete explanation; the one-sentence
API example prefers a shorter answer. Length alone is not the labeling rule.

From `code/`, using your existing SFT bundle and original tokenizer:

```bash
python -m llmre.preference.train --sft sft_bundle.pt --train data/preferences-demo/train.jsonl --validation data/preferences-demo/validation.jsonl --output checkpoints/reward-demo --epochs 1 --pairs-per-batch 2 --grad-accum-steps 2
```

Your checkpoint's context window must fit each complete formatted prompt and answer.
The loader rejects overlong records rather than truncating them silently. Inspect
the actual token lengths with your tokenizer. Replace these illustrative rows with
substantial, reviewed preference data and keep a separate held-out test split for a
real experiment. The three-row validation file only checks the training workflow;
its accuracy is not evidence of generalization.
