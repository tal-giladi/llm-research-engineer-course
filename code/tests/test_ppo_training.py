"""Lesson 15.2: the PPO loop that trains the SFT model with a frozen reward model."""

import math

import torch
import torch.nn as nn

from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT
from llmre.preference.reward_model import RewardModel
from llmre.rl import ppo_train as ppo
from llmre.sft.chat_template import END_TOKEN
from llmre.tokenizer.bpe import BPETokenizer


def tiny_setup(seed=0):
    tokenizer = BPETokenizer()
    tokenizer.train("Hi Maya, thanks! Yes no. " * 4, vocab_size=262)
    torch.manual_seed(seed)
    sft = GPT(GPTConfig(vocab_size=tokenizer.vocab_size, block_size=64,
                        n_layer=1, n_head=2, n_embd=16))
    return sft, tokenizer


class CountReward(nn.Module):
    """A stand-in grader: one point for every plain-ASCII token (id < 128)."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(()))

    def forward(self, ids, seq_lengths):
        positions = torch.arange(ids.shape[1])[None, :]
        real = positions < seq_lengths[:, None]
        return ((ids < 128) & real).sum(1).float()


def test_collate_mask_marks_exactly_the_answer_predictions():
    ids, lengths, mask = ppo.collate([3, 2], [[5, 6, 7, 8, 9], [1, 2, 3]], "cpu")
    assert ids.tolist() == [[5, 6, 7, 8, 9], [1, 2, 3, 0, 0]]
    assert lengths.tolist() == [5, 3]
    # row 0: answer tokens 8, 9 are predicted at columns 2, 3
    # row 1: answer token 3 is predicted at column 1; padding is excluded
    assert mask.tolist() == [[0, 0, 1, 1], [0, 1, 0, 0]]


def test_token_logprobs_match_a_manual_log_softmax():
    sft, _ = tiny_setup()
    ids = torch.tensor([[1, 2, 3, 4]])
    got = ppo.token_logprobs(sft, ids)
    logits, _ = sft(ids)
    manual = [torch.log_softmax(logits[0, j], -1)[ids[0, j + 1]].item() for j in range(3)]
    assert torch.allclose(got[0], torch.tensor(manual), atol=1e-6)


def test_sampling_stops_after_the_end_of_turn_ids():
    sft, tokenizer = tiny_setup()
    end_ids = tokenizer.encode(END_TOKEN + "\n")
    with torch.no_grad():                     # make the model always say the end ids
        sft.lm_head.weight.zero_()
        sft.transformer.wpe.weight.zero_()
    order = iter(end_ids * 10)

    def fake_multinomial(probs, num_samples):
        token = next(order)
        return torch.full((probs.shape[0], 1), token, dtype=torch.long)

    real = torch.multinomial
    torch.multinomial = fake_multinomial
    try:
        answers, finished = ppo.sample_answers(sft, [1, 2, 3], 2, 30, end_ids)
    finally:
        torch.multinomial = real
    assert finished == [True, True]
    assert answers == [end_ids, end_ids]


def test_first_rollout_has_zero_drift_and_ratio_one():
    sft, tokenizer = tiny_setup()
    reward = RewardModel.from_sft(sft).eval()
    config = ppo.PPOTrainConfig(answers_per_prompt=3, max_new_tokens=6)
    policy = sft
    batch = ppo.rollout(policy, sft, reward, RewardModel.from_sft(sft),
                        ["Say hi"], tokenizer, config)
    assert torch.allclose(batch["drift"], torch.zeros(3), atol=1e-6)
    assert torch.allclose(batch["total_reward"], batch["scores"], atol=1e-6)
    new = ppo.token_logprobs(policy, batch["ids"])
    assert torch.allclose(new * batch["mask"], batch["old_logprobs"] * batch["mask"], atol=1e-6)
    adv = batch["advantages"][batch["mask"] > 0]
    assert abs(adv.mean().item()) < 1e-5 and abs(adv.std(unbiased=False).item() - 1) < 1e-3


def test_clipped_tokens_get_no_gradient():
    old = torch.tensor([0.0, 0.0])
    new = torch.tensor([math.log(1.5), math.log(1.1)], requires_grad=True)
    adv = torch.tensor([2.0, 2.0])
    objective = ppo.ppo_clip_objective(new, old, adv, 0.2, reduction="sum")
    objective.backward()
    assert new.grad[0].item() == 0.0              # ratio 1.5 > 1.2: no more push
    assert abs(new.grad[1].item() - 1.1 * 2.0) < 1e-6


def test_training_raises_the_reward_and_leaves_the_sft_model_unchanged(tmp_path):
    sft, tokenizer = tiny_setup()
    before = {k: v.clone() for k, v in sft.state_dict().items()}
    config = ppo.PPOTrainConfig(iterations=20, prompts_per_iteration=1, answers_per_prompt=8,
                                max_new_tokens=8, ppo_epochs=4, minibatch_answers=8,
                                lr=1e-2, value_lr=1e-2, kl_coef=0.0)
    policy, history = ppo.train_ppo(sft, tokenizer, CountReward(), ["Say hi"],
                                    tmp_path, config,
                                    value_model=RewardModel.from_sft(sft))
    early = sum(h["score"] for h in history[:3]) / 3
    late = sum(h["score"] for h in history[-3:]) / 3
    assert late > early + 1.5
    assert all(torch.equal(v, before[k]) for k, v in sft.state_dict().items())
    saved = torch.load(tmp_path / "policy_bundle.pt", weights_only=True)
    reloaded = GPT(GPTConfig(**saved["config"]))
    reloaded.load_state_dict(saved["model"])
    assert torch.equal(reloaded.lm_head.weight, policy.lm_head.weight)
    assert (tmp_path / "history.json").exists()


def test_kl_penalty_holds_the_policy_closer_to_the_reference(tmp_path):
    drifts = {}
    for kl_coef in (0.0, 2.0):
        sft, tokenizer = tiny_setup()
        config = ppo.PPOTrainConfig(iterations=20, prompts_per_iteration=1,
                                    answers_per_prompt=8, max_new_tokens=8, ppo_epochs=4,
                                    minibatch_answers=8, lr=1e-2, value_lr=1e-2,
                                    kl_coef=kl_coef)
        _, history = ppo.train_ppo(sft, tokenizer, CountReward(), ["Say hi"],
                                   tmp_path / str(kl_coef), config,
                                   value_model=RewardModel.from_sft(sft))
        drifts[kl_coef] = sum(h["drift"] for h in history[-3:]) / 3
    assert drifts[2.0] < drifts[0.0]
