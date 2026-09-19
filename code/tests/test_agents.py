"""Tests for llmre.agents (Module 19).

Decoupling note: these tests define their OWN tiny local tools (a ``calc``-style
adder/multiplier and a no-op). They deliberately do NOT import ``llmre.tools``
(Module 18), because the ReAct loop is tool-registry-agnostic — it takes an
injected ``name -> callable`` mapping. This keeps Module 19 buildable and
testable independently of Module 18.

Coverage:
* :func:`run_react` with a scripted, deterministic policy solves a toy
  arithmetic task in <= 3 steps and returns the correct answer.
* :func:`run_react` halts at ``max_steps`` when the policy never finalizes.
* the SWE-agent scaffold fixes a toy file via the injected file tools.
* :func:`mean_std` / :func:`confidence_interval` return correct values on a
  known list.
"""
import math

from llmre.agents.react import run_react
from llmre.agents.stats import confidence_interval, mean_std
from llmre.agents.swe_agent import Workspace, solve


# --- local tools (NOT llmre.tools) -----------------------------------------
def add(a, b):
    return a + b


def mul(a, b):
    return a * b


def noop():
    return "ok"


# --- ReAct: scripted policy solves a toy arithmetic task -------------------
def test_run_react_solves_arithmetic_in_three_steps():
    """Compute (3 + 4) * 2 = 14 with an injected calc toolset.

    The policy interleaves Thought/Action/Observation: add, then multiply the
    observed sum, then finalize the observed product.
    """
    tools = {"add": add, "mul": mul}

    def policy(history):
        last = history[-1]
        if len(history) == 1:  # only the task seed so far
            return {"thought": "add 3 + 4", "tool": "add", "args": {"a": 3, "b": 4}}
        if last.action.get("tool") == "add":
            return {
                "thought": "multiply the sum by 2",
                "tool": "mul",
                "args": {"a": last.observation, "b": 2},
            }
        if last.action.get("tool") == "mul":
            return {"thought": "return the product", "final": last.observation}
        raise AssertionError("policy reached an unexpected state")

    result = run_react(policy, tools, task="compute (3 + 4) * 2", max_steps=5)

    assert result.finished is True
    assert result.answer == 14
    assert result.steps == 3
    assert result.steps <= 3
    # The trace records the two tool observations plus the task seed + final.
    observations = [s.observation for s in result.trace]
    assert 7 in observations and 14 in observations


def test_run_react_halts_at_max_steps_without_final():
    """A policy that never finalizes must stop cleanly at the step budget."""
    tools = {"noop": noop}

    def never_finalize(history):
        return {"thought": "loop forever", "tool": "noop", "args": {}}

    result = run_react(never_finalize, tools, task="spin", max_steps=4)

    assert result.finished is False
    assert result.answer is None
    assert result.steps == 4


def test_run_react_reports_unknown_tool_as_observation():
    """An unknown tool becomes an observation, not a crash — the policy recovers."""
    calls = {"n": 0}

    def policy(history):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"tool": "does_not_exist", "args": {}}
        # second step: read the error observation and finalize
        return {"final": history[-1].observation}

    result = run_react(policy, tools={"add": add}, task="oops", max_steps=5)
    assert result.finished is True
    assert isinstance(result.answer, str) and "unknown tool" in result.answer


# --- SWE-agent scaffold: fix a toy file, observe tests pass ----------------
def test_swe_agent_scaffold_fixes_toy_bug():
    """The agent reads a broken constant, writes a fix, and reruns the tests."""
    ws = Workspace(files={"answer.txt": "41"})

    def test_runner(workspace):
        content = workspace.files.get("answer.txt", "")
        return (content.strip() == "42", f"answer.txt={content!r}")

    def policy(history):
        last = history[-1]
        if len(history) == 1:
            return {"thought": "inspect", "tool": "read_file",
                    "args": {"path": "answer.txt"}}
        if last.action.get("tool") == "read_file":
            return {"thought": "fix", "tool": "write_file",
                    "args": {"path": "answer.txt", "content": "42"}}
        if last.action.get("tool") == "write_file":
            return {"thought": "verify", "tool": "run_tests", "args": {}}
        if last.action.get("tool") == "run_tests":
            return {"final": last.observation}
        raise AssertionError("unexpected state")

    result = solve(policy, ws, task="make answer.txt equal 42",
                   test_runner=test_runner, max_steps=8)

    assert result.finished is True
    assert result.answer.startswith("PASS")
    assert ws.files["answer.txt"] == "42"


# --- stats: known-list values ----------------------------------------------
def test_mean_std_known_list():
    mean, std = mean_std([1, 2, 3, 4, 5])
    assert mean == 3.0
    # sample std of 1..5 is sqrt(2.5)
    assert math.isclose(std, math.sqrt(2.5), rel_tol=1e-12)


def test_mean_std_single_value():
    mean, std = mean_std([7.0])
    assert mean == 7.0
    assert std == 0.0


def test_confidence_interval_known_list():
    # 1..5: mean 3, sample std sqrt(2.5)=1.5811, n=5, df=4, t_.975,4 = 2.776.
    # margin = 2.776 * 1.5811 / sqrt(5) = 1.96293...
    lo, hi = confidence_interval([1, 2, 3, 4, 5], confidence=0.95)
    expected_margin = 2.776 * math.sqrt(2.5) / math.sqrt(5)
    assert math.isclose(lo, 3.0 - expected_margin, rel_tol=1e-9)
    assert math.isclose(hi, 3.0 + expected_margin, rel_tol=1e-9)


def test_confidence_interval_single_value_collapses():
    lo, hi = confidence_interval([5.0], confidence=0.95)
    assert lo == 5.0 and hi == 5.0
