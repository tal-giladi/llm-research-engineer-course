"""Tests for Module 17 (reasoning models): CoT helpers + the mini-R1 pipeline.

Covers:
* self-consistency majority vote and its deterministic tie-break (cot.py),
* the answer-extraction helper parsing formatted completions (cot.py),
* the miniature DeepSeek-R1 pipeline running end to end and improving toy accuracy
  through the verifiable-reward stage (r1_pipeline.py).
"""

import math

from llmre.reasoning.cot import extract_answer, self_consistency
from llmre.reasoning.r1_pipeline import (
    make_toy_task,
    mini_r1,
    toy_verifier,
)


# --------------------------------------------------------------------------- #
# self_consistency (majority vote)
# --------------------------------------------------------------------------- #
def test_self_consistency_returns_the_majority_answer():
    ans, count = self_consistency(["4", "4", "7", "4", "9"])
    assert ans == "4"
    assert count == 3


def test_self_consistency_handles_non_string_answers():
    ans, count = self_consistency([42, 42, 7])
    assert ans == 42 and count == 2


def test_self_consistency_breaks_ties_deterministically_by_first_seen():
    # "x" and "y" both appear twice; the one seen first in input order wins,
    # regardless of hashing/dict order — so the result is reproducible.
    ans, count = self_consistency(["x", "y", "y", "x"])
    assert ans == "x" and count == 2
    # Reordering so "y" comes first flips the winner deterministically.
    ans2, count2 = self_consistency(["y", "x", "x", "y"])
    assert ans2 == "y" and count2 == 2


def test_self_consistency_empty_raises():
    try:
        self_consistency([])
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError on empty input")


# --------------------------------------------------------------------------- #
# extract_answer (parse the final answer out of a formatted completion)
# --------------------------------------------------------------------------- #
def test_extract_answer_parses_hash_marker():
    assert extract_answer("Let me think... #### 42") == "42"


def test_extract_answer_parses_phrase_marker_and_strips_period():
    assert extract_answer("2 plus 2 is 4. The answer is 4.") == "4"


def test_extract_answer_takes_the_last_marker():
    # A chain of thought may revise itself; the final stated answer wins.
    text = "Step 1: #### 5\nStep 2, reconsidered: #### 9"
    assert extract_answer(text) == "9"


def test_extract_answer_returns_none_when_no_marker():
    assert extract_answer("there is no answer marker in this text") is None


def test_extract_answer_custom_pattern():
    got = extract_answer("2+2 = 4. Final: 4", pattern=r"Final:\s*(.+)")
    assert got == "4"


# --------------------------------------------------------------------------- #
# toy verifier + task
# --------------------------------------------------------------------------- #
def test_toy_verifier_scores_correctness():
    _logits, correct = make_toy_task(n_problems=3, n_actions=4, seed=0)
    # Reward is 1.0 exactly on the gold answers, 0.0 otherwise.
    rewards = toy_verifier(correct, correct)
    assert rewards.tolist() == [1.0, 1.0, 1.0]
    wrong = (correct + 1) % 4
    assert toy_verifier(wrong, correct).tolist() == [0.0, 0.0, 0.0]


# --------------------------------------------------------------------------- #
# mini_r1 end-to-end
# --------------------------------------------------------------------------- #
def test_mini_r1_runs_and_returns_per_stage_metrics():
    m = mini_r1()
    # All three R1 stages are reported.
    assert set(m) == {"base", "sft", "rl"}
    assert {"accuracy", "expected_reward"} <= set(m["base"])
    assert {"demo_loss_before", "demo_loss_after"} <= set(m["sft"])
    assert {"accuracy_before", "accuracy_after"} <= set(m["rl"])
    # Metrics are plain floats in range.
    for stage in m.values():
        for v in stage.values():
            assert isinstance(v, float)
    for key in ("accuracy", "expected_reward"):
        assert 0.0 <= m["base"][key] <= 1.0


def test_mini_r1_cold_start_sft_reduces_demo_loss():
    m = mini_r1()
    assert m["sft"]["demo_loss_after"] < m["sft"]["demo_loss_before"]


def test_mini_r1_verifiable_reward_stage_does_not_decrease_accuracy():
    # The improvement (RL) stage must never make the toy policy worse. Group-relative
    # advantage stops updating a problem once it is solved, so greedy accuracy is
    # monotonic across the stage. Check over several seeds for robustness.
    for seed in range(8):
        m = mini_r1(seed=seed)
        assert m["rl"]["accuracy_after"] >= m["rl"]["accuracy_before"], (seed, m["rl"])
        assert (
            m["rl"]["expected_reward_after"] >= m["rl"]["expected_reward_before"]
        ), (seed, m["rl"])


def test_mini_r1_improves_over_base_by_the_end():
    m = mini_r1(seed=0)
    # By the end of the pipeline the policy is better than the uniform base.
    assert m["rl"]["accuracy_after"] > m["base"]["accuracy"]
    assert not math.isnan(m["rl"]["expected_reward_after"])
