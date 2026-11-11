"""Tests for Module 14 (supervised fine-tuning): masking, chat templates, LoRA."""

import torch
import torch.nn as nn

from llmre.evaluation.metrics import cross_entropy
from llmre.sft.chat_template import (
    assistant_char_spans,
    render_chat,
    render_with_response_mask,
)
from llmre.sft.lora import LoRALinear, mark_only_lora_trainable
from llmre.sft.masking import build_labels, masked_cross_entropy


# --------------------------------------------------------------------------- #
# Loss masking
# --------------------------------------------------------------------------- #
def test_masked_loss_equals_ce_over_response_positions_only():
    # Masked cross-entropy over labels must equal plain cross-entropy computed on
    # ONLY the response rows (the whole point of loss masking in SFT).
    torch.manual_seed(0)
    T, V = 6, 10
    logits = torch.randn(T, V)
    input_ids = torch.randint(0, V, (T,))
    response_mask = torch.tensor([0, 0, 0, 1, 1, 1])  # last 3 are the response

    labels = build_labels(input_ids, response_mask)
    # Prompt positions ignored, response positions carry their token id.
    assert torch.equal(labels[:3], torch.full((3,), -100))
    assert torch.equal(labels[3:], input_ids[3:])

    masked = masked_cross_entropy(logits, labels)
    ref = cross_entropy(logits[3:], input_ids[3:])
    assert torch.allclose(masked, ref, atol=1e-6), (masked, ref)


def test_masked_loss_batched_matches_flat():
    torch.manual_seed(1)
    B, T, V = 2, 5, 7
    logits = torch.randn(B, T, V)
    input_ids = torch.randint(0, V, (B, T))
    response_mask = torch.zeros(B, T, dtype=torch.long)
    response_mask[:, 3:] = 1
    labels = build_labels(input_ids, response_mask)

    masked = masked_cross_entropy(logits, labels)
    keep_logits = logits[:, 3:, :].reshape(-1, V)
    keep_labels = input_ids[:, 3:].reshape(-1)
    ref = cross_entropy(keep_logits, keep_labels)
    assert torch.allclose(masked, ref, atol=1e-6)


# --------------------------------------------------------------------------- #
# Chat template + response mask
# --------------------------------------------------------------------------- #
def test_chat_template_masks_only_assistant_tokens():
    messages = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    text = render_chat(messages)
    assert "<|assistant|>" in text and "<|end|>" in text

    # A trivial byte-level encoder: every char -> its ordinal. Deterministic and
    # dependency-free, enough to check the mask lines up with the response text.
    def encode(s: str) -> list[int]:
        return [ord(c) for c in s]

    ids, mask = render_with_response_mask(messages, encode)
    assert len(ids) == len(mask)
    # The ids marked as response must decode back to the assistant content + closer.
    response_text = "".join(chr(i) for i, m in zip(ids, mask) if m)
    assert response_text == "hello<|end|>\n"

    # The char spans agree with what render_chat contains there.
    (start, end), = assistant_char_spans(messages)
    assert text[start:end] == "hello<|end|>\n"


def test_generation_prompt_appends_assistant_header_with_no_response():
    messages = [{"role": "user", "content": "q"}]
    text = render_chat(messages, add_generation_prompt=True)
    assert text.endswith("<|assistant|>\n")
    # No assistant content yet => no response span.
    assert assistant_char_spans(messages, add_generation_prompt=True) == []


# --------------------------------------------------------------------------- #
# LoRA
# --------------------------------------------------------------------------- #
def test_fresh_adapter_is_a_noop():
    # B is zero-initialised, so a freshly wrapped layer equals the base linear.
    torch.manual_seed(0)
    base = nn.Linear(8, 4)
    lora = LoRALinear(base, r=2, alpha=4.0)
    x = torch.randn(3, 8)
    assert torch.allclose(lora(x), base(x), atol=1e-6)


def test_merge_matches_lora_forward():
    torch.manual_seed(0)
    base = nn.Linear(8, 4)
    lora = LoRALinear(base, r=2, alpha=4.0)
    # Make the adapter nontrivial.
    nn.init.normal_(lora.A, std=0.5)
    nn.init.normal_(lora.B, std=0.5)

    x = torch.randn(5, 8)
    y = lora(x)
    merged = lora.merge()
    assert isinstance(merged, nn.Linear)
    assert torch.allclose(merged(x), y, atol=1e-6)


def test_only_lora_params_require_grad():
    base = nn.Linear(8, 4)
    lora = LoRALinear(base, r=2, alpha=4.0)
    trainable = {n for n, p in lora.named_parameters() if p.requires_grad}
    assert trainable == {"A", "B"}
    assert lora.base.weight.requires_grad is False
    assert lora.base.bias.requires_grad is False


def test_mark_only_lora_trainable_on_a_model():
    torch.manual_seed(0)
    model = nn.Sequential(
        LoRALinear(nn.Linear(8, 8), r=2, alpha=2.0),
        nn.ReLU(),
        nn.Linear(8, 4),  # a plain linear that must end up frozen
    )
    mark_only_lora_trainable(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"0.A", "0.B"}


def test_lora_has_far_fewer_trainable_params():
    base = nn.Linear(768, 768)
    lora = LoRALinear(base, r=8, alpha=16.0)
    trainable = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    full = base.weight.numel()  # what full fine-tuning would train
    assert trainable == 8 * 768 * 2
    assert full / trainable > 40  # ~48x fewer


# --------------------------------------------------------------------------- #
# Sequence packing
# --------------------------------------------------------------------------- #
from llmre.attention.attention import scaled_dot_product_attention  # noqa: E402
from llmre.sft.packing import (  # noqa: E402
    first_fit_decreasing,
    pack_examples,
    packed_attention_mask,
    packed_labels,
    packing_efficiency,
)


def _toy_examples():
    # (ids, response_mask): A = 3 prompt + 2 response, B = 2 + 2, C = 1 + 2
    A = ([11, 12, 13, 14, 15], [0, 0, 0, 1, 1])
    B = ([21, 22, 23, 24], [0, 0, 1, 1])
    C = ([31, 32, 33], [0, 1, 1])
    return [A, B, C]


def test_first_fit_decreasing_groups_longest_first():
    assert first_fit_decreasing([5, 4, 3], 8) == [[0, 2], [1]]
    assert first_fit_decreasing([20, 50, 30], 100) == [[1, 2, 0]]


def test_pack_examples_tensors():
    p = pack_examples(_toy_examples(), block_size=8, pad_id=0, rows=[[0, 2], [1]])
    assert p["input_ids"].tolist() == [[11, 12, 13, 14, 15, 31, 32, 33],
                                       [21, 22, 23, 24, 0, 0, 0, 0]]
    assert p["doc_ids"].tolist() == [[0, 0, 0, 0, 0, 1, 1, 1],
                                     [0, 0, 0, 0, -1, -1, -1, -1]]
    assert p["position_ids"].tolist() == [[0, 1, 2, 3, 4, 0, 1, 2],
                                          [0, 1, 2, 3, 0, 0, 0, 0]]
    assert abs(packing_efficiency(p["doc_ids"]) - 12 / 16) < 1e-9


def test_packed_labels_never_cross_a_boundary():
    p = pack_examples(_toy_examples(), block_size=8, rows=[[0, 2], [1]])
    labels = packed_labels(p["input_ids"], p["response_mask"], p["doc_ids"])
    I = -100
    assert labels.tolist() == [[I, I, 14, 15, I, 32, 33, I],
                               [I, 23, 24, I, I, I, I, I]]


def test_packed_attention_equals_separate_attention():
    torch.manual_seed(0)
    d, T = 8, 8
    emb, pos = torch.randn(64, d), torch.randn(T, d)
    Wq, Wk, Wv = torch.randn(d, d), torch.randn(d, d), torch.randn(d, d)

    def attend(ids, position_ids, mask):
        x = emb[ids] + pos[position_ids]
        return scaled_dot_product_attention(x @ Wq, x @ Wk, x @ Wv, mask)

    exs = _toy_examples()
    p = pack_examples(exs, block_size=T, rows=[[0, 2]])
    ids, pids, docs = p["input_ids"][0], p["position_ids"][0], p["doc_ids"][0]
    packed = attend(ids, pids, packed_attention_mask(docs)[0])

    causal = torch.full((T, T), float("-inf")).triu(1)
    for (ex_ids, _), (lo, hi) in zip([exs[0], exs[2]], [(0, 5), (5, 8)]):
        n = len(ex_ids)
        alone = attend(torch.tensor(ex_ids), torch.arange(n), causal[:n, :n])
        assert torch.allclose(packed[lo:hi], alone, atol=1e-5)

    # The bug: a plain causal mask lets example C see example A.
    leaky = attend(ids, torch.arange(T), causal)
    assert not torch.allclose(leaky[5:8], packed[5:8], atol=1e-3)
