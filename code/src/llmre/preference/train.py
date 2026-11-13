"""Lesson 15.1: complete single-device reward-model training from an SFT GPT.

Run from code/: python -m llmre.preference.train --help
Uses the existing RewardModel and Bradley-Terry loss, not a demo replacement.
All model computation is float32. Token ids and lengths are int64. This is an
in-memory, single-device baseline; distributed/mixed-precision execution is a
separate scaling concern. Resume is supported at completed epoch boundaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT
from llmre.optim.schedule import cosine_warmup_lr
from llmre.preference.reward_model import RewardModel, bradley_terry_loss
from llmre.sft.chat_template import render_with_response_mask
from llmre.tokenizer.bpe import BPETokenizer


@dataclass
class RewardTrainConfig:
    epochs: int = 1
    pairs_per_batch: int = 4         # microbatch size in PAIRS, not sequences
    grad_accum_steps: int = 1
    lr: float = 1e-5
    warmup_steps: int = 0
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    seed: int = 0


def read_jsonl(path):
    """Read {prompt, chosen, rejected} string records; no tensor/device yet."""
    with open(path, encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def encode_pairs(rows, tokenizer, model_config):
    """Return [(chosen_ids, rejected_ids)], lists of ints on the Python side.

    Reuse exactly Module 14's segment-by-segment tokenization. Do not retrain
    the tokenizer or silently truncate answers and retain their old labels.
    """
    if not rows:
        raise ValueError("each data split needs at least one preference pair")
    pairs = []
    for row in rows:
        if any(not isinstance(row.get(k), str) or not row[k].strip()
               for k in ("prompt", "chosen", "rejected")):
            raise ValueError("prompt, chosen, rejected must be nonempty strings")
        sequences = []
        for answer in (row["chosen"], row["rejected"]):
            messages = [{"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": answer}]
            ids, _ = render_with_response_mask(messages, tokenizer.encode)
            if not 0 < len(ids) <= model_config.block_size:
                raise ValueError("complete preference sequence exceeds context; filter/review it")
            if min(ids) < 0 or max(ids) >= model_config.vocab_size:
                raise ValueError("tokenizer ids do not fit the SFT vocabulary")
            sequences.append(ids)
        if sequences[0] == sequences[1]:
            raise ValueError("identical answers cannot express a strict preference")
        pairs.append(tuple(sequences))
    return pairs


def collate_pairs(pairs, device):
    """Right-pad chosen-first, rejected-second rows on device.

    Return ids (2P, T) int64 and real lengths (2P,) int64. Pad id 0 is a
    valid embedding id, not a real part of the response. With this causal GPT,
    trailing padding cannot influence the selected earlier real-token states.
    """
    sequences = [p[0] for p in pairs] + [p[1] for p in pairs]
    lengths = torch.tensor([len(s) for s in sequences], device=device)
    ids = torch.zeros((len(sequences), int(lengths.max())),
                      dtype=torch.long, device=device)
    for row, sequence in enumerate(sequences):
        ids[row, :len(sequence)] = torch.tensor(sequence, device=device)
    return ids, lengths


@torch.no_grad()
def evaluate(model, pairs, pairs_per_batch, device):
    """Held-out pair loss/accuracy as Python floats; no gradients retained."""
    model.eval()
    loss_sum, correct = 0.0, 0
    for start in range(0, len(pairs), pairs_per_batch):
        batch = pairs[start:start + pairs_per_batch]
        ids, lengths = collate_pairs(batch, device)
        chosen, rejected = model(ids, seq_lengths=lengths).chunk(2)
        loss_sum += bradley_terry_loss(chosen, rejected, reduction="sum").item()
        correct += (chosen > rejected).sum().item()
    return {"loss": loss_sum / len(pairs), "accuracy": correct / len(pairs)}


def tokenizer_from_merges(merges):
    """Reconstruct the SAME course BPE tokenizer from saved integer triples."""
    tokenizer = BPETokenizer()
    for a, b, new_id in merges:
        if new_id != len(tokenizer.vocab):
            raise ValueError("tokenizer merges must be in training order")
        tokenizer.merges[a, b] = new_id
        tokenizer.vocab[new_id] = tokenizer.vocab[a] + tokenizer.vocab[b]
    return tokenizer


def load_reward_model(path, device="cpu"):
    """Load a frozen scorer and its tokenizer; no SFT checkpoint needed.

    The scorer accepts ids (B,T) and optional lengths (B,), int64 on device,
    and returns float32 rewards (B,). Its whole backbone is in the checkpoint.
    """
    saved = torch.load(path, map_location="cpu", weights_only=True)
    model = RewardModel.from_sft(GPT(GPTConfig(**saved["model_config"])))
    model.load_state_dict(saved["model"])
    model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    return model, tokenizer_from_merges(saved["tokenizer_merges"])


def fit_reward_model(sft, tokenizer, train_rows, val_rows, output_dir,
                     config=None, device="cpu", resume=None):
    """Train a separate scorer, save best/last checkpoints, return best frozen RM.

    sft is the already trained course GPT. tokenizer is its unchanged BPE
    tokenizer. Data rows are plain {prompt, chosen, rejected} strings.
    All tensors created here use device; parameters/activations use float32.
    The SFT model is not optimized. Resume restarts at the next saved epoch,
    using the same config and encoded data; interrupted partial epochs replay.
    """
    config = config or RewardTrainConfig()
    if min(config.epochs, config.pairs_per_batch, config.grad_accum_steps) < 1:
        raise ValueError("epochs, batch size, accumulation must be positive")
    if config.lr <= 0 or config.grad_clip <= 0 or config.weight_decay < 0:
        raise ValueError("invalid optimizer settings")
    train_pairs = encode_pairs(train_rows, tokenizer, sft.cfg)
    val_pairs = encode_pairs(val_rows, tokenizer, sft.cfg)
    if {r["prompt"].strip() for r in train_rows} & {r["prompt"].strip() for r in val_rows}:
        raise ValueError("split by prompt: train and validation prompts overlap")
    fingerprint = hashlib.sha256(json.dumps([train_pairs, val_pairs]).encode()).hexdigest()
    effective_pairs = config.pairs_per_batch * config.grad_accum_steps
    total_steps = config.epochs * math.ceil(len(train_pairs) / effective_pairs)
    if not 0 <= config.warmup_steps <= total_steps:
        raise ValueError("warmup_steps must be between zero and total updates")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if resume is None and any((output_dir / name).exists() for name in ("best.pt", "last.pt")):
        raise FileExistsError("choose a new output directory or resume the existing run")

    torch.manual_seed(config.seed)
    model = RewardModel.from_sft(sft).to(device=device, dtype=torch.float32)
    model.requires_grad_(True)                    # full backbone + head fine-tuning
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr,
                                  weight_decay=config.weight_decay)
    shuffle_rng = torch.Generator().manual_seed(config.seed)
    start_epoch, step, best_loss = 0, 0, float("inf")
    merges = [[a, b, i] for (a, b), i in tokenizer.merges.items()]
    if resume is not None:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if (saved["train_config"] != asdict(config) or
                saved["data_fingerprint"] != fingerprint or
                saved["model_config"] != asdict(sft.cfg) or
                saved["tokenizer_merges"] != merges):
            raise ValueError("resume requires the same config, tokenizer, and encoded data")
        if not (output_dir / "best.pt").exists():
            raise ValueError("resume in the original run directory containing best.pt")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, step, best_loss = saved["epoch"], saved["step"], saved["best_loss"]
        shuffle_rng.set_state(saved["shuffle_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available() and saved["cuda_rng"]:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])

    print("initial validation:", evaluate(model, val_pairs, config.pairs_per_batch, device))
    for epoch in range(start_epoch, config.epochs):
        model.train()                            # re-enable training-time dropout
        order = torch.randperm(len(train_pairs), generator=shuffle_rng).tolist()
        train_loss_sum, train_correct = 0.0, 0
        for start in range(0, len(order), effective_pairs):
            group = [train_pairs[i] for i in order[start:start + effective_pairs]]
            optimizer.zero_grad(set_to_none=True)
            for micro_start in range(0, len(group), config.pairs_per_batch):
                batch = group[micro_start:micro_start + config.pairs_per_batch]
                ids, lengths = collate_pairs(batch, device)
                rewards = model(ids, seq_lengths=lengths)   # (2P,)
                chosen, rejected = rewards.chunk(2)       # each (P,)
                pair_losses = bradley_terry_loss(chosen, rejected, reduction="none")
                if not torch.isfinite(pair_losses).all():
                    raise FloatingPointError("non-finite reward loss")
                # Sum/N gives each PAIR equal weight, even in a short last group.
                loss = pair_losses.sum() / len(group)
                loss.backward()                 # both branches; accumulate gradients
                train_loss_sum += pair_losses.detach().sum().item()
                train_correct += (chosen.detach() > rejected.detach()).sum().item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip,
                                           error_if_nonfinite=True)
            lr = cosine_warmup_lr(step, config.warmup_steps, total_steps,
                                  config.lr, config.lr * 0.1)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            optimizer.step()                    # ONE update per effective pair group
            step += 1

        val = evaluate(model, val_pairs, config.pairs_per_batch, device)
        if not math.isfinite(val["loss"]):
            raise FloatingPointError("non-finite validation loss")
        improved = val["loss"] < best_loss
        best_loss = min(best_loss, val["loss"])
        print({"epoch": epoch + 1, "step": step,
               "train_loss": train_loss_sum / len(train_pairs),
               "train_accuracy": train_correct / len(train_pairs), "validation": val})
        saved = {"model": model.state_dict(), "model_config": asdict(sft.cfg),
                 "tokenizer_merges": merges, "optimizer": optimizer.state_dict(),
                 "epoch": epoch + 1, "step": step, "best_loss": best_loss,
                 "train_config": asdict(config), "data_fingerprint": fingerprint,
                 "shuffle_rng": shuffle_rng.get_state(), "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
        # Write then rename: an interrupted save leaves the prior checkpoint intact.
        for name in (["best.pt", "last.pt"] if improved else ["last.pt"]):
            temporary = output_dir / (name + ".tmp")
            torch.save(saved, temporary)
            temporary.replace(output_dir / name)

    best = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(best["model"])
    return model.eval().requires_grad_(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sft", required=True, help="SFT bundle: config, model, tokenizer_merges")
    parser.add_argument("--train", required=True, help="training preferences JSONL")
    parser.add_argument("--validation", required=True, help="held-out preferences JSONL")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--pairs-per-batch", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--resume", help="last.pt from the original output directory")
    args = parser.parse_args()
    saved = torch.load(args.sft, map_location="cpu", weights_only=True)
    sft = GPT(GPTConfig(**saved["config"]))
    sft.load_state_dict(saved["model"])
    tokenizer = tokenizer_from_merges(saved["tokenizer_merges"])
    config = RewardTrainConfig(epochs=args.epochs, pairs_per_batch=args.pairs_per_batch,
                               grad_accum_steps=args.grad_accum_steps,
                               lr=args.lr, warmup_steps=args.warmup_steps)
    fit_reward_model(sft, tokenizer, read_jsonl(args.train), read_jsonl(args.validation),
                     args.output, config, args.device, args.resume)


if __name__ == "__main__":
    main()
