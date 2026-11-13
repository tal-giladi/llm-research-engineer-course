"""Tests for the post-training data builders (lessons 14.4, 15.5, 16.5).

All offline: no downloads. The labs in each lesson run the same functions on
real Hugging Face datasets.
"""

from __future__ import annotations

import random

import pytest

from llmre.preference.data import (
    binarize,
    cohens_kappa,
    debiased_verdict,
    length_balanced_subset,
    length_bias,
    pair_from_judgment,
    percent_agreement,
    validate_pair,
)
from llmre.rl.data import (
    answer_is_verifiable,
    answers_match,
    check_constraints,
    difficulty_filter,
    extract_boxed,
    group_has_signal,
    normalize_answer,
    pass_rate,
)
from llmre.sft.chat_template import render_with_response_mask
from llmre.sft.data import (
    build_benchmark_index,
    contamination_score,
    decontaminate,
    dedupe_by_prompt,
    mix_to_budget,
    near_dedupe_prompts,
    response_issues,
    to_messages,
    validate_messages,
)


# --------------------------------------------------------------------------- #
# 14.4 SFT data
# --------------------------------------------------------------------------- #
def test_to_messages_three_formats_agree():
    alpaca = {"instruction": "Translate to French.", "input": "The cat.", "output": "Le chat."}
    sharegpt = {"conversations": [
        {"from": "human", "value": "Translate to French.\n\nThe cat."},
        {"from": "gpt", "value": "Le chat."},
    ]}
    msgs = {"messages": [
        {"role": "user", "content": "Translate to French.\n\nThe cat."},
        {"role": "assistant", "content": "Le chat."},
    ]}
    assert to_messages(alpaca) == to_messages(sharegpt) == to_messages(msgs)


def test_dolly_fields():
    row = {"instruction": "Summarize.", "context": "Long text.", "response": "Short."}
    assert to_messages(row)[0]["content"] == "Summarize.\n\nLong text."
    assert to_messages(row)[1]["content"] == "Short."


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        to_messages({"text": "raw"})


def test_validate_messages():
    good = [{"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."}]
    assert validate_messages(good) == []
    no_answer = good[:2]
    assert "does not end with an assistant turn" in validate_messages(no_answer)
    two_users = [{"role": "user", "content": "a"}, {"role": "user", "content": "b"},
                 {"role": "assistant", "content": "c"}]
    assert any("expected assistant" in p for p in validate_messages(two_users))
    empty = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "  "}]
    assert any("empty assistant" in p for p in validate_messages(empty))


def test_response_issues():
    boiler = [{"role": "user", "content": "Tell a joke"},
              {"role": "assistant", "content": "As an AI language model, I do not tell jokes."}]
    assert any("refusal" in i for i in response_issues(boiler))
    cut = [{"role": "user", "content": "Explain"},
           {"role": "assistant", "content": "word " * 60 + "and then the model"}]
    assert any("truncated" in i for i in response_issues(cut))
    fine = [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4."}]
    assert response_issues(fine) == []


def _chat(q, a="ok."):
    return [{"role": "user", "content": q}, {"role": "assistant", "content": a}]


def test_exact_and_near_dedupe():
    base = "Write a short cover letter for a junior data analyst job at a small retail company in Berlin"
    chats = [_chat(base),
             _chat(base.upper().replace(" ", "  ")),               # exact after normalize
             _chat(base + " please"),                               # near duplicate
             _chat("Write a poem about rain.")]
    kept, dropped = dedupe_by_prompt(chats)
    assert dropped == 1 and len(kept) == 3
    kept2, dropped2 = near_dedupe_prompts(kept, threshold=0.8)
    assert dropped2 == 1
    assert [first["content"] for first, _ in kept2] == [base, "Write a poem about rain."]


def test_decontaminate():
    bench = ["Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May."]
    index = build_benchmark_index(bench, n=8)
    leaked = _chat("Solve: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many?")
    clean = _chat("Tom has 3 apples and buys 4 more. How many apples does he have now?")
    assert contamination_score(leaked[0]["content"], index) > 0.5
    kept, removed = decontaminate([leaked, clean], index)
    assert kept == [clean] and removed == [leaked]


def test_mix_to_budget_respects_shares_and_never_upsamples():
    a = [_chat("q", "x" * 400) for _ in range(100)]     # ~100 tokens each
    b = [_chat("q", "y" * 400) for _ in range(3)]       # tiny source
    mix, taken = mix_to_budget({"a": a, "b": b}, {"a": 1, "b": 1}, total_tokens=2000)
    assert 1000 <= taken["a"] < 1000 + 110
    assert taken["b"] <= 3 * 101                         # all of b, nothing repeated
    assert len(mix) == len({id(c) for c in mix})


def test_cleaned_chat_renders_with_mask():
    chat = to_messages({"instruction": "Say hi.", "output": "Hi."})
    ids, mask = render_with_response_mask(chat, lambda s: list(s.encode("utf-8")))
    assert len(ids) == len(mask) and sum(mask) == len("Hi.<|end|>\n")


# --------------------------------------------------------------------------- #
# 15.5 preference data
# --------------------------------------------------------------------------- #
def test_binarize_strategies():
    comps, scores = ["a", "b", "c", "d"], [4.5, 2.0, 3.0, 4.5]
    worst = binarize("p", comps, scores, strategy="best_vs_worst")
    assert worst["chosen"] == "a" and worst["rejected"] == "b"
    rnd = binarize("p", comps, scores, rng=random.Random(1))
    assert rnd["chosen"] == "a" and rnd["rejected"] in {"b", "c"}   # never the tied "d"
    assert binarize("p", ["a", "b"], [3.0, 3.0]) is None
    assert binarize("p", comps, scores, strategy="best_vs_worst", min_margin=3.0) is None


def test_pair_from_judgment_and_position_debiasing():
    assert pair_from_judgment("p", "x", "y", "B") == {"prompt": "p", "chosen": "y", "rejected": "x"}
    assert pair_from_judgment("p", "x", "y", "tie") is None

    def always_first(prompt, first, second):
        return "A"

    def prefers_longer(prompt, first, second):
        return "A" if len(first) > len(second) else "B"

    assert debiased_verdict(always_first, "p", "x", "yy") == "tie"
    assert debiased_verdict(prefers_longer, "p", "x", "yy") == "B"


def test_agreement_and_kappa_by_hand():
    a = ["A", "A", "A", "B", "B", "A", "A", "B", "A", "A"]
    b = ["A", "A", "B", "B", "A", "A", "A", "B", "A", "A"]
    assert percent_agreement(a, b) == pytest.approx(0.8)
    # p1(A)=0.7, p2(A)=0.7 -> p_e = 0.49 + 0.09 = 0.58; kappa = 0.22/0.42
    assert cohens_kappa(a, b) == pytest.approx(0.22 / 0.42)


def test_length_bias_and_balancing():
    pairs = ([{"chosen": "x" * 10, "rejected": "x" * 5}] * 8
             + [{"chosen": "x" * 5, "rejected": "x" * 10}] * 2)
    stats = length_bias(pairs)
    assert stats["frac_chosen_longer"] == pytest.approx(0.8)
    bal = length_balanced_subset(pairs)
    assert length_bias(bal)["frac_chosen_longer"] == pytest.approx(0.5)


def test_validate_pair():
    assert validate_pair({"prompt": "p", "chosen": "a", "rejected": "b"}) == []
    assert validate_pair({"prompt": "p", "chosen": "a", "rejected": "a "}) == ["chosen == rejected"]
    assert validate_pair({"prompt": "", "chosen": "a", "rejected": "b"})


# --------------------------------------------------------------------------- #
# 16.5 RL prompt data
# --------------------------------------------------------------------------- #
def test_extract_boxed_nested():
    assert extract_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert extract_boxed(r"\boxed{3} then \boxed{4}") == "4"
    assert extract_boxed("no box") is None
    assert extract_boxed(r"\boxed{unclosed") is None


def test_normalize_and_match():
    assert normalize_answer(r"$\dfrac{3}{4}$") == "3/4"
    assert normalize_answer("x = 5.") == "5"
    assert normalize_answer("1,000") == "1000"
    assert normalize_answer(r"\$93,750") == "93750"
    assert normalize_answer(r"45^\circ") == "45"
    assert answers_match(r"\frac{1}{2}", "0.5") == 1.0
    assert answers_match(r"\text{12}", "12") == 1.0
    assert answers_match("13", "12") == 0.0
    assert answers_match(None, "12") == 0.0


def test_answer_is_verifiable():
    assert answer_is_verifiable("Compute 2+3.", "5") == []
    assert "proof problem" in answer_is_verifiable("Prove that sqrt 2 is irrational.", "QED")
    assert "multi-part answer" in answer_is_verifiable("Find both roots.", "2, 3")
    assert answer_is_verifiable("Find the point.", "(2, 3)") == []
    assert answer_is_verifiable("Find the interval.", "[0, 1)") == []
    assert "multiple-choice letter" in answer_is_verifiable("Which one?", "(C)")
    assert answer_is_verifiable("Q", None) == ["no final answer"]


def test_pass_rate_and_difficulty_filter():
    rng = random.Random(0)

    def policy_for(p_correct):
        return lambda problem: r"\boxed{7}" if rng.random() < p_correct else r"\boxed{0}"

    rows = [
        {"id": "easy", "pass_rate": pass_rate("q", "7", policy_for(1.0), k=8)},
        {"id": "hard", "pass_rate": pass_rate("q", "7", policy_for(0.0), k=8)},
        {"id": "mid", "pass_rate": pass_rate("q", "7", lambda q: r"\boxed{7}" if q else "", k=8) * 0.5},
    ]
    assert rows[0]["pass_rate"] == 1.0 and rows[1]["pass_rate"] == 0.0
    kept, counts = difficulty_filter(rows)
    assert [r["id"] for r in kept] == ["mid"]
    assert counts == {"too_easy": 1, "too_hard": 1, "kept": 1}
    assert not group_has_signal([1.0] * 8) and group_has_signal([1.0, 0.0])


def test_check_constraints():
    resp = "- budget is tight\n- hire later\n- cut travel\nAny other questions?"
    cons = [{"type": "num_bullets", "n": 3}, {"type": "include_keyword", "word": "budget"},
            {"type": "max_words", "n": 30}, {"type": "end_with", "phrase": "Any other questions?"}]
    assert check_constraints(resp, cons) == 1.0
    assert check_constraints(resp, cons + [{"type": "all_lowercase"}]) == 0.0
    with pytest.raises(ValueError):
        check_constraints(resp, [{"type": "rhymes"}])
