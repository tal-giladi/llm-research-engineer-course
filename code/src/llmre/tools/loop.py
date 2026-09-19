"""The observation/action loop — the heart of a tool-using model.

The loop is small and deterministic on purpose:

1. Show the model the running ``context`` and ask its ``policy`` what to do.
2. Parse the reply. If it is a tool call, run the tool and turn the result into
   an **observation**; if it is prose, that is the final answer and we stop.
3. Append the observation to the context and repeat, up to ``max_steps``.

Errors are part of the loop, not exceptions to it: a bad call or a crashing tool
becomes an error observation fed back to the model so it can correct itself
(``ToolRegistry.call`` never raises). This is the reusable core that Module 19
wraps into a full agent.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from .parser import parse_tool_call
from .registry import ToolRegistry

# A policy maps the current context (list of message dicts) to the model's next
# text output — either a tool-call JSON or a final prose answer.
Policy = Callable[[list[dict[str, Any]]], str]


def _observation_text(result: dict[str, Any]) -> str:
    """Render a ``ToolRegistry.call`` result dict as observation text.

    Success -> the stringified result; failure -> an ``ERROR: ...`` line the
    model can read and recover from.
    """
    if result.get("ok"):
        return f"OBSERVATION: {result['result']}"
    return f"OBSERVATION: ERROR ({result.get('error_type')}): {result.get('error')}"


def run_tool_loop(
    policy: Policy,
    registry: ToolRegistry,
    prompt: str,
    max_steps: int = 10,
) -> dict[str, Any]:
    """Run the observation/action loop until a final answer or ``max_steps``.

    Args:
        policy: callable ``policy(context) -> str``. ``context`` is the list of
            message dicts so far; the returned string is either a tool-call JSON
            (fenced or bare) or a final prose answer. Deterministic policies keep
            the loop reproducible; this function itself does no I/O and no
            network.
        registry: the :class:`~llmre.tools.registry.ToolRegistry` used to execute
            parsed calls.
        prompt: the user's task, seeded as the first ``user`` message.
        max_steps: hard cap on model turns, so a policy that never answers cannot
            loop forever.

    Returns:
        A JSON-serializable dict:

        * ``"answer"``: the final prose answer (str), or ``None`` if the loop hit
          ``max_steps`` first.
        * ``"trace"``: list of executed calls, each
          ``{"step", "name", "arguments", "result"}`` where ``result`` is the
          raw registry result dict. ``len(trace)`` is the number of tool calls.
        * ``"context"``: the full message list (roles ``user`` / ``assistant`` /
          ``tool``) for inspection or rendering.
        * ``"stopped"``: ``"final_answer"`` or ``"max_steps"``.

    Message roles in ``context``: ``user`` (the prompt), ``assistant`` (each
    policy output), ``tool`` (each observation).
    """
    context: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    trace: list[dict[str, Any]] = []

    for step in range(max_steps):
        output = policy(context)
        context.append({"role": "assistant", "content": output})

        call = parse_tool_call(output)
        if call is None:
            # No tool call -> the model is done. This is the final answer.
            return {
                "answer": output,
                "trace": trace,
                "context": context,
                "stopped": "final_answer",
            }

        # Execute the tool. registry.call never raises: a failure comes back as
        # {"ok": False, ...} and becomes an error observation.
        result = registry.call(call["name"], **call["arguments"])
        trace.append({
            "step": step,
            "name": call["name"],
            "arguments": call["arguments"],
            "result": result,
        })
        context.append({
            "role": "tool",
            "name": call["name"],
            "content": _observation_text(result),
        })

    # Ran out of steps without a final answer.
    return {
        "answer": None,
        "trace": trace,
        "context": context,
        "stopped": "max_steps",
    }


def render_context(context: list[dict[str, Any]]) -> str:
    """Render a context message list as a readable transcript (str).

    Handy for logging/debugging a run: one ``ROLE: content`` line per message.
    """
    lines = []
    for msg in context:
        role = msg["role"].upper()
        content = msg["content"]
        if not isinstance(content, str):
            content = json.dumps(content)
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
