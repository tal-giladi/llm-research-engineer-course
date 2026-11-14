"""Lesson 15.2: improve the SFT model with the frozen reward model from 15.1.

Run from code/: python -m llmre.rl.ppo_train --help

One iteration of the loop:

1. The policy (a copy of the SFT model) writes several answers per prompt.
2. The frozen reward model from 15.1 gives each whole answer one score.
3. A frozen copy of the original SFT model (the reference) says how far each
   answer has drifted from what the SFT model would have written: the KL
   penalty. Total reward = score - kl_coef * drift.
4. The value model (a trainable copy of the reward model) guesses, at each
   answer position, what total reward to expect. Advantage = total reward -
   that guess: how much better than expected the answer turned out.
5. PPO: several gradient steps on the same answers. Each answer token's
   log-probability is pushed up (positive advantage) or down (negative), and
   the clip stops any token from moving more than clip_eps (20%) per batch.

Single device, float32, no resume. The SFT model passed in is never modified.
Token ids and lengths are int64; logits, log-probabilities, values, and
rewards are float32.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT
from llmre.preference.train import load_reward_model, tokenizer_from_merges
from llmre.rl.ppo import kl_penalty, ppo_clip_objective
from llmre.sft.chat_template import END_TOKEN, render_with_response_mask


@dataclass
class PPOTrainConfig:
    iterations: int = 20            # rounds of: write answers -> score -> update
    prompts_per_iteration: int = 2
    answers_per_prompt: int = 4     # answers written for each prompt
    max_new_tokens: int = 64        # longest answer the policy may write
    ppo_epochs: int = 4             # passes over the same answers per iteration
    minibatch_answers: int = 4      # answers per gradient step
    lr: float = 1e-5                # policy learning rate
    value_lr: float = 1e-5          # value-model learning rate
    clip_eps: float = 0.2           # each token may move at most 20% per iteration
    kl_coef: float = 0.05           # beta: price of drifting from the SFT model
    grad_clip: float = 1.0
    seed: int = 0


def read_prompts(path):
    """Read JSONL rows with a "prompt" string; other fields are ignored."""
    with open(path, encoding="utf-8") as file:
        rows = [json.loads(line) for line in file if line.strip()]
    prompts = [row["prompt"] for row in rows]
    if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("every row needs a nonempty 'prompt' string")
    return prompts


def encode_prompt(prompt, tokenizer):
    """The same chat template as SFT, ending with the open assistant header."""
    ids, _ = render_with_response_mask([{"role": "user", "content": prompt}],
                                       tokenizer.encode, add_generation_prompt=True)
    return ids


@torch.no_grad()
def sample_answers(policy, prompt_ids, n, max_new_tokens, end_ids):
    """Write n answers to one prompt, sampling at temperature 1.

    All n rows share the same prompt, so they have the same length and need no
    padding while generating. A row stops once it has written the end-of-turn
    ids; the others keep going. Returns (answers, finished): n lists of ids and
    n booleans saying whether each answer closed its turn.
    """
    device = next(policy.parameters()).device
    room = policy.cfg.block_size - len(prompt_ids)
    if room < 1:
        raise ValueError("prompt leaves no room for an answer in the context")
    idx = torch.tensor([prompt_ids] * n, dtype=torch.long, device=device)  # (n, T0)
    answers = [[] for _ in range(n)]
    finished = [False] * n
    for _ in range(min(max_new_tokens, room)):
        logits, _ = policy(idx)                             # (n, T, V)
        probs = F.softmax(logits[:, -1, :].float(), dim=-1)  # (n, V)
        next_ids = torch.multinomial(probs, num_samples=1)  # (n, 1)
        idx = torch.cat([idx, next_ids], dim=1)
        for row in range(n):
            if not finished[row]:
                answers[row].append(int(next_ids[row]))
                finished[row] = answers[row][-len(end_ids):] == end_ids
        if all(finished):
            break
    return answers, finished


def collate(prompt_lengths, sequences, device):
    """Right-pad prompt+answer sequences and mark the answer decisions.

    Returns ids (N, T) int64, lengths (N,) int64, and answer_mask (N, T-1)
    float32. Column j of the mask stands for "the prediction made at position j
    of token j+1": it is 1 exactly when token j+1 belongs to the answer. Log-
    probabilities and values below use the same column convention.
    """
    lengths = torch.tensor([len(s) for s in sequences], device=device)
    ids = torch.zeros((len(sequences), int(lengths.max())), dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), ids.shape[1] - 1), device=device)
    for row, (start, seq) in enumerate(zip(prompt_lengths, sequences)):
        ids[row, :len(seq)] = torch.tensor(seq, device=device)
        mask[row, start - 1:len(seq) - 1] = 1.0      # predictions of answer tokens
    return ids, lengths, mask


def token_logprobs(model, ids):
    """log p(token j+1 | tokens 0..j) for every column j: (N, T-1) float32."""
    logits, _ = model(ids)                                   # (N, T, V)
    logp = F.log_softmax(logits[:, :-1, :].float(), dim=-1)  # (N, T-1, V)
    return logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def position_values(value_model, ids):
    """The value model's guess of the total reward, before each token: (N, T-1)."""
    hidden = value_model.backbone(ids)                  # (N, T, C)
    return value_model.score(hidden).squeeze(-1)[:, :-1].float()


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum()


def whiten(x, mask):
    """Shift and scale the answer positions to mean 0 and std 1."""
    mean = masked_mean(x, mask)
    std = masked_mean((x - mean) ** 2, mask).sqrt()
    return (x - mean) / (std + 1e-8) * mask


@torch.no_grad()
def rollout(policy, reference, reward_model, value_model, prompts, tokenizer, config):
    """Write answers, score them, and compute everything PPO needs.

    Nothing here needs gradients: these numbers are frozen labels for the
    update steps that follow.
    """
    end_ids = tokenizer.encode(END_TOKEN + "\n")
    device = next(policy.parameters()).device
    policy.eval()
    prompt_lengths, sequences, finished = [], [], []
    for prompt in prompts:
        prompt_ids = encode_prompt(prompt, tokenizer)
        answers, done = sample_answers(policy, prompt_ids, config.answers_per_prompt,
                                       config.max_new_tokens, end_ids)
        prompt_lengths += [len(prompt_ids)] * len(answers)
        sequences += [prompt_ids + answer for answer in answers]
        finished += done
    ids, lengths, mask = collate(prompt_lengths, sequences, device)

    scores = reward_model(ids, seq_lengths=lengths).float()      # (N,)
    old_logprobs = token_logprobs(policy, ids)                   # (N, T-1)
    ref_logprobs = token_logprobs(reference, ids)                # (N, T-1)
    drift = (kl_penalty(old_logprobs, ref_logprobs, reduction="none") * mask).sum(1)  # (N,)
    total_reward = scores - config.kl_coef * drift               # (N,)
    values = position_values(value_model, ids)                   # (N, T-1)
    advantages = whiten((total_reward[:, None] - values) * mask, mask)
    return {"ids": ids, "mask": mask, "prompt_lengths": prompt_lengths,
            "old_logprobs": old_logprobs, "advantages": advantages,
            "total_reward": total_reward, "scores": scores, "drift": drift,
            "finished": torch.tensor(finished, dtype=torch.float32)}


def ppo_update(policy, value_model, batch, policy_opt, value_opt, config):
    """Several clipped gradient steps on one batch of answers; returns stats."""
    policy.eval()        # no dropout: the same answer must get the same log-probs
    value_model.train()
    n = batch["ids"].shape[0]
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "clipped_fraction": 0.0, "steps": 0}
    for _ in range(config.ppo_epochs):
        order = torch.randperm(n).tolist()
        for start in range(0, n, config.minibatch_answers):
            rows = order[start:start + config.minibatch_answers]
            ids, mask = batch["ids"][rows], batch["mask"][rows]
            if mask.sum() == 0:
                continue
            new_logprobs = token_logprobs(policy, ids)           # (M, T-1), has grad
            objective = ppo_clip_objective(new_logprobs, batch["old_logprobs"][rows],
                                           batch["advantages"][rows], config.clip_eps,
                                           reduction="none")      # (M, T-1)
            policy_loss = -masked_mean(objective, mask)
            policy_opt.zero_grad(set_to_none=True)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.grad_clip)
            policy_opt.step()

            values = position_values(value_model, ids)           # (M, T-1), has grad
            target = batch["total_reward"][rows][:, None]        # (M, 1)
            value_loss = masked_mean((values - target) ** 2, mask)
            value_opt.zero_grad(set_to_none=True)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_model.parameters(), config.grad_clip)
            value_opt.step()

            with torch.no_grad():
                ratio = torch.exp(new_logprobs - batch["old_logprobs"][rows])
                clipped = ((ratio - 1).abs() > config.clip_eps).float()
            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["clipped_fraction"] += masked_mean(clipped, mask).item()
            stats["steps"] += 1
    steps = max(stats.pop("steps"), 1)
    return {key: value / steps for key, value in stats.items()}


def save_policy(policy, tokenizer, path):
    """Same bundle format as the SFT export in 15.1: config, model, tokenizer."""
    bundle = {"config": asdict(policy.cfg), "model": policy.state_dict(),
              "tokenizer_merges": [[a, b, i] for (a, b), i in tokenizer.merges.items()]}
    temporary = Path(str(path) + ".tmp")
    torch.save(bundle, temporary)
    temporary.replace(path)


def train_ppo(sft, tokenizer, reward_model, prompts, output_dir, config=None,
              device="cpu", value_model=None):
    """Return the improved policy and a list of per-iteration statistics.

    sft: the trained course GPT (left unchanged). reward_model: the frozen
    scorer from 15.1, built from the same SFT model and tokenizer. The value
    model starts as a trainable copy of the reward model unless one is given.
    """
    config = config or PPOTrainConfig()
    if min(config.iterations, config.prompts_per_iteration, config.answers_per_prompt,
           config.max_new_tokens, config.ppo_epochs, config.minibatch_answers) < 1:
        raise ValueError("counts in the config must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)

    policy = copy.deepcopy(sft).to(device=device, dtype=torch.float32)
    policy.requires_grad_(True)
    reference = copy.deepcopy(sft).to(device=device, dtype=torch.float32)
    reference.eval().requires_grad_(False)
    reward_model = reward_model.to(device).eval().requires_grad_(False)
    if value_model is None:
        value_model = copy.deepcopy(reward_model)
    value_model = value_model.to(device=device, dtype=torch.float32).requires_grad_(True)
    policy_opt = torch.optim.AdamW(policy.parameters(), lr=config.lr, weight_decay=0.0)
    value_opt = torch.optim.AdamW(value_model.parameters(), lr=config.value_lr,
                                  weight_decay=0.0)

    history = []
    for iteration in range(config.iterations):
        start = (iteration * config.prompts_per_iteration) % len(prompts)
        chosen = [prompts[(start + k) % len(prompts)]
                  for k in range(config.prompts_per_iteration)]
        batch = rollout(policy, reference, reward_model, value_model, chosen, tokenizer, config)
        update = ppo_update(policy, value_model, batch, policy_opt, value_opt, config)
        first = batch["ids"][0, batch["prompt_lengths"][0]:].tolist()
        record = {"iteration": iteration + 1,
                  "score": batch["scores"].mean().item(),
                  "drift": batch["drift"].mean().item(),
                  "total_reward": batch["total_reward"].mean().item(),
                  "finished": batch["finished"].mean().item(), **update,
                  "example": tokenizer.decode(first)}
        history.append(record)
        # json.dumps escapes non-ASCII text, so any console can print it.
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in record.items()}))
        save_policy(policy, tokenizer, output_dir / "policy_bundle.pt")
        (output_dir / "history.json").write_text(json.dumps(history, indent=1),
                                                 encoding="utf-8")
    return policy.eval(), history


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sft", required=True, help="sft_bundle.pt from 15.1")
    parser.add_argument("--reward", required=True, help="best.pt of the 15.1 reward run")
    parser.add_argument("--prompts", required=True, help="JSONL with a 'prompt' field")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    defaults = PPOTrainConfig()
    for name, value in asdict(defaults).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(value), default=value)
    args = parser.parse_args()
    saved = torch.load(args.sft, map_location="cpu", weights_only=True)
    sft = GPT(GPTConfig(**saved["config"]))
    sft.load_state_dict(saved["model"])
    tokenizer = tokenizer_from_merges(saved["tokenizer_merges"])
    reward_model, reward_tokenizer = load_reward_model(args.reward, args.device)
    if reward_tokenizer.merges != tokenizer.merges:
        raise ValueError("the reward model must use the SFT model's tokenizer")
    config = PPOTrainConfig(**{name: getattr(args, name) for name in asdict(defaults)})
    train_ppo(sft, tokenizer, reward_model, read_prompts(args.prompts), args.output,
              config, args.device)


if __name__ == "__main__":
    main()
