"""Full reward-training flow: SFT isolation, padding, data splits, and resume."""

from dataclasses import asdict

import pytest
import torch

from llmre.model.config import GPTConfig
from llmre.model.gpt import GPT
from llmre.preference import train as training
from llmre.preference.reward_model import RewardModel
from llmre.tokenizer.bpe import BPETokenizer


def fixture_data():
    tokenizer = BPETokenizer()
    tokenizer.train("Yes No Is fire hot? Is ice cold? " * 4, vocab_size=270)
    torch.manual_seed(4)
    sft = GPT(GPTConfig(vocab_size=tokenizer.vocab_size, block_size=128,
                        n_layer=1, n_head=2, n_embd=8, dropout=0.1))
    train = [{"prompt": f"training question {i}?", "chosen": "Yes",
              "rejected": "No, that is wrong."} for i in range(5)]
    validation = [{"prompt": "held out question?", "chosen": "Yes", "rejected": "No"}]
    return sft, tokenizer, train, validation


def test_sft_copy_and_real_token_readout_are_independent_of_padding():
    sft, tokenizer, rows, _ = fixture_data()
    model = RewardModel.from_sft(sft).eval()
    pairs = training.encode_pairs(rows[:2], tokenizer, sft.cfg)
    ids, lengths = training.collate_pairs(pairs, "cpu")
    with torch.no_grad():
        batched = model(ids, seq_lengths=lengths)
        singles = torch.stack([model(row[:length].unsqueeze(0))[0]
                               for row, length in zip(ids, lengths)])
    assert torch.allclose(batched, singles, atol=1e-6)
    assert model.backbone.transformer.wte.weight.data_ptr() != sft.transformer.wte.weight.data_ptr()
    assert not hasattr(model, "lm_head")
    assert torch.equal(model.backbone.transformer.wte.weight, sft.transformer.wte.weight)


def test_complete_training_saves_reloadable_scorer_and_keeps_sft_unchanged(tmp_path):
    sft, tokenizer, rows, validation = fixture_data()
    before = {name: value.clone() for name, value in sft.state_dict().items()}
    config = training.RewardTrainConfig(epochs=2, pairs_per_batch=2,
                                        grad_accum_steps=2, lr=0.01)
    model = training.fit_reward_model(sft, tokenizer, rows, validation, tmp_path, config)
    loaded, loaded_tokenizer = training.load_reward_model(tmp_path / "best.pt")
    assert all(torch.equal(value, before[name]) for name, value in sft.state_dict().items())
    assert all(torch.equal(value, loaded.state_dict()[name]) for name, value in model.state_dict().items())
    assert loaded_tokenizer.encode("new prompt") == tokenizer.encode("new prompt")
    assert not model.training and all(not p.requires_grad for p in model.parameters())
    last = torch.load(tmp_path / "last.pt", weights_only=True)
    assert last["step"] == 4             # 5 pairs, groups of 4 -> 2 updates/epoch
    assert last["epoch"] == 2
    assert not torch.equal(last["model"]["backbone.transformer.wte.weight"], before["transformer.wte.weight"])


def test_epoch_resume_matches_uninterrupted_training_with_dropout(tmp_path, monkeypatch):
    sft, tokenizer, rows, validation = fixture_data()
    config = training.RewardTrainConfig(epochs=2, pairs_per_batch=2,
                                        grad_accum_steps=2, lr=0.001)
    full = tmp_path / "full"
    resumed = tmp_path / "resumed"
    training.fit_reward_model(sft, tokenizer, rows, validation, full, config)
    real_save = torch.save

    class Interrupted(Exception):
        pass

    def stop_after_first_epoch(saved, path):
        real_save(saved, path)
        if path.name == "last.pt.tmp" and saved["epoch"] == 1:
            path.replace(path.with_name("last.pt"))
            raise Interrupted

    with monkeypatch.context() as patch:
        patch.setattr(torch, "save", stop_after_first_epoch)
        with pytest.raises(Interrupted):
            training.fit_reward_model(sft, tokenizer, rows, validation, resumed, config)
    training.fit_reward_model(sft, tokenizer, rows, validation, resumed, config,
                              resume=resumed / "last.pt")
    want = torch.load(full / "last.pt", weights_only=True)
    got = torch.load(resumed / "last.pt", weights_only=True)
    assert got["step"] == want["step"]
    assert all(torch.equal(value, got["model"][name]) for name, value in want["model"].items())
    assert torch.equal(got["shuffle_rng"], want["shuffle_rng"])


def test_bad_preferences_and_prompt_leakage_are_rejected(tmp_path):
    sft, tokenizer, rows, validation = fixture_data()
    with pytest.raises(ValueError, match="identical"):
        training.encode_pairs([dict(rows[0], rejected="Yes")], tokenizer, sft.cfg)
    with pytest.raises(ValueError, match="exceeds context"):
        training.encode_pairs([dict(rows[0], chosen="x" * 200)], tokenizer, sft.cfg)
    with pytest.raises(ValueError, match="overlap"):
        training.fit_reward_model(sft, tokenizer, rows, [rows[0]], tmp_path)


def test_accumulation_weights_short_microbatch_like_one_effective_batch(tmp_path):
    sft, tokenizer, rows, validation = fixture_data()
    # Disable dropout so both batching routes represent identical computation.
    for module in sft.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    config = training.RewardTrainConfig(epochs=1, pairs_per_batch=2,
                                        grad_accum_steps=2, lr=0.001)
    training.fit_reward_model(sft, tokenizer, rows[:3], validation, tmp_path / "micro", config)
    config = training.RewardTrainConfig(**{**asdict(config), "pairs_per_batch": 3, "grad_accum_steps": 1})
    training.fit_reward_model(sft, tokenizer, rows[:3], validation, tmp_path / "whole", config)
    micro = torch.load(tmp_path / "micro/last.pt", weights_only=True)["model"]
    whole = torch.load(tmp_path / "whole/last.pt", weights_only=True)["model"]
    assert all(torch.allclose(value, whole[name], atol=2e-6, rtol=1e-5)
               for name, value in micro.items())
