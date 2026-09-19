"""Parse a model's text output into a structured tool call.

Function calling means the model stops writing prose and instead emits a small
JSON object naming a tool and its arguments. Real deployments constrain the
decoder to emit that JSON (a "JSON mode" / structured-output constraint); here we
parse it after the fact, robustly, from ordinary generated text.

We accept two shapes a model commonly produces:

* a fenced block::

        ```json
        {"name": "calculator", "arguments": {"expression": "2+2"}}
        ```

* or a bare JSON object embedded in the text::

        Sure, let me compute that. {"name": "calculator", "arguments": {...}}

:func:`parse_tool_call` returns ``{"name": str, "arguments": dict}`` when it
finds a valid call, or ``None`` when the text is a plain final answer (no call).
"""

from __future__ import annotations

import json
import re
from typing import Any

# ```json ... ``` or ``` ... ``` (language tag optional). DOTALL so it spans lines.
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _coerce_call(obj: Any) -> dict[str, Any] | None:
    """Return a normalized ``{"name", "arguments"}`` dict if ``obj`` is a call.

    Accepts the canonical ``{"name": ..., "arguments": {...}}`` and the common
    variant ``{"tool"/"tool_name": ..., "args"/"parameters": {...}}``. Missing
    arguments default to ``{}``. Returns ``None`` if it is not a tool call.
    """
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool") or obj.get("tool_name")
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("arguments")
    if args is None:
        args = obj.get("args")
    if args is None:
        args = obj.get("parameters")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return None
    return {"name": name, "arguments": args}


def _find_json_objects(text: str) -> list[str]:
    """Yield candidate JSON-object substrings by brace matching.

    Scans left to right, tracking ``{`` / ``}`` depth while ignoring braces
    inside double-quoted strings (respecting backslash escapes), and returns each
    balanced top-level ``{...}`` span. This finds a bare object even when it is
    surrounded by prose.
    """
    spans: list[str] = []
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    spans.append(text[start:i + 1])
                    start = -1
    return spans


def parse_tool_call(text: str) -> dict[str, Any] | None:
    """Extract a tool call from model ``text``; return it or ``None``.

    Args:
        text: raw text the model produced.

    Returns:
        ``{"name": str, "arguments": dict}`` for the first valid call found, or
        ``None`` if the text contains no tool call (i.e. it is a final answer).

    Resolution order: try fenced ```` ```json ```` blocks first (the strongest
    signal of intent), then fall back to any balanced bare ``{...}`` object in
    the text. In each case the JSON must parse and look like a call
    (see :func:`_coerce_call`).
    """
    if not text:
        return None

    # 1) Fenced blocks, in order of appearance.
    for match in _FENCE_RE.finditer(text):
        candidate = match.group(1).strip()
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        call = _coerce_call(obj)
        if call is not None:
            return call

    # 2) Bare JSON objects embedded in prose.
    for span in _find_json_objects(text):
        try:
            obj = json.loads(span)
        except json.JSONDecodeError:
            continue
        call = _coerce_call(obj)
        if call is not None:
            return call

    return None
