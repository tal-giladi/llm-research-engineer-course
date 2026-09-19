"""Tests for llmre.tools (Module 18 — tool use)."""

from __future__ import annotations

import pytest

from llmre.tools.builtins import calculator, mock_search, python_eval, register_builtins
from llmre.tools.loop import run_tool_loop
from llmre.tools.parser import parse_tool_call
from llmre.tools.registry import Tool, ToolRegistry


# --------------------------------------------------------------------------- #
# registry + schemas                                                          #
# --------------------------------------------------------------------------- #

def _registry_with_builtins() -> ToolRegistry:
    return register_builtins(ToolRegistry())


def test_schemas_advertise_the_tools():
    reg = _registry_with_builtins()
    schemas = reg.schemas()
    names = {s["name"] for s in schemas}
    assert {"calculator", "python_eval", "mock_search"} <= names
    # Each schema is JSON-shaped: name/description/parameters present.
    for s in schemas:
        assert set(s) == {"name", "description", "parameters"}
        assert isinstance(s["parameters"], dict)
    calc = next(s for s in schemas if s["name"] == "calculator")
    assert "expression" in calc["parameters"]["properties"]


def test_registry_get_and_contains():
    reg = _registry_with_builtins()
    assert "calculator" in reg
    assert isinstance(reg.get("calculator"), Tool)
    with pytest.raises(KeyError):
        reg.get("nope")


# --------------------------------------------------------------------------- #
# calculator: correctness + safety                                            #
# --------------------------------------------------------------------------- #

def test_calculator_respects_precedence():
    assert calculator("2+3*4") == 14


def test_calculator_more_arithmetic():
    assert calculator("(2 + 3) * 4") == 20
    assert calculator("2 ** 10") == 1024
    assert calculator("7 // 2") == 3
    assert calculator("-5 + 2") == -3


def test_calculator_rejects_malicious_expression():
    # A code-injection attempt must NOT execute; it raises instead.
    with pytest.raises(ValueError):
        calculator("__import__('os').system('echo pwned')")
    with pytest.raises(ValueError):
        calculator("open('secret.txt').read()")


def test_calculator_division_by_zero_is_clean_error():
    with pytest.raises(ValueError):
        calculator("1/0")


def test_registry_call_catches_calculator_error():
    reg = _registry_with_builtins()
    res = reg.call("calculator", expression="__import__('os')")
    assert res["ok"] is False
    assert "error" in res and res["error_type"]  # structured, not a crash


# --------------------------------------------------------------------------- #
# python_eval + mock_search                                                   #
# --------------------------------------------------------------------------- #

def test_python_eval_restricted():
    assert python_eval("2 < 3 and 10 % 2 == 0") is True
    with pytest.raises(ValueError):
        python_eval("__import__('os').getcwd()")


def test_mock_search_canned_and_missing():
    assert "Paris" in mock_search("what is the capital of France?")
    assert "No results" in mock_search("something obscure and unknown")


# --------------------------------------------------------------------------- #
# parser                                                                       #
# --------------------------------------------------------------------------- #

def test_parser_extracts_from_fenced_block():
    text = 'Let me compute.\n```json\n{"name": "calculator", "arguments": {"expression": "2+2"}}\n```'
    call = parse_tool_call(text)
    assert call == {"name": "calculator", "arguments": {"expression": "2+2"}}


def test_parser_extracts_from_bare_object():
    text = 'I will search now {"name": "mock_search", "arguments": {"query": "chinchilla"}} ok'
    call = parse_tool_call(text)
    assert call == {"name": "mock_search", "arguments": {"query": "chinchilla"}}


def test_parser_returns_none_for_final_answer():
    assert parse_tool_call("The answer is 14.") is None


# --------------------------------------------------------------------------- #
# the observation/action loop                                                  #
# --------------------------------------------------------------------------- #

def test_run_tool_loop_calls_calculator_then_answers():
    reg = _registry_with_builtins()

    # A scripted, deterministic policy: first turn emit a calculator call, then
    # read the observation off the context and give a final answer.
    def policy(context):
        last = context[-1]
        if last["role"] == "user":
            return '```json\n{"name": "calculator", "arguments": {"expression": "2+3*4"}}\n```'
        # last observation is a tool message like "OBSERVATION: 14"
        value = last["content"].split()[-1]
        return f"The answer is {value}."

    out = run_tool_loop(policy, reg, "What is 2+3*4?", max_steps=5)
    assert out["stopped"] == "final_answer"
    assert out["answer"] == "The answer is 14."
    assert len(out["trace"]) == 1
    step = out["trace"][0]
    assert step["name"] == "calculator"
    assert step["result"] == {"ok": True, "result": 14}


def test_run_tool_loop_feeds_back_tool_error():
    reg = _registry_with_builtins()

    calls = {"n": 0}

    def policy(context):
        # First: a bad call (will raise inside the tool -> caught).
        # After seeing the error observation: answer, acknowledging it.
        if calls["n"] == 0:
            calls["n"] += 1
            return '{"name": "calculator", "arguments": {"expression": "1/0"}}'
        return "Recovered after the error."

    out = run_tool_loop(policy, reg, "compute 1/0", max_steps=5)
    assert out["answer"] == "Recovered after the error."
    assert len(out["trace"]) == 1
    err = out["trace"][0]["result"]
    assert err["ok"] is False  # caught error observation, not a crash
    # The error observation was appended to context for the model to read.
    tool_msgs = [m for m in out["context"] if m["role"] == "tool"]
    assert "ERROR" in tool_msgs[0]["content"]


def test_run_tool_loop_respects_max_steps():
    reg = _registry_with_builtins()

    def never_answers(context):
        return '{"name": "mock_search", "arguments": {"query": "loop"}}'

    out = run_tool_loop(never_answers, reg, "go", max_steps=3)
    assert out["stopped"] == "max_steps"
    assert out["answer"] is None
    assert len(out["trace"]) == 3
