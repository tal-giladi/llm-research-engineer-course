"""A few deterministic built-in tools plus a factory that registers them.

These are the tools a model calls when its own weights are not enough: it can
*talk* about arithmetic but cannot reliably *do* it, it cannot look anything up,
and it cannot run code. Each tool here is deterministic and offline so the whole
observation/action loop stays reproducible and testable.

Safety is the interesting part, and it is spelled out honestly per function:

* :func:`calculator` — evaluates an arithmetic expression by walking a parsed
  AST and allowing only number literals and ``+ - * / // % ** ()`` and unary
  ``+ -``. It never calls Python's :func:`eval` on the raw string, so there is
  no name lookup, no attribute access, no function call, no ``__import__``.
* :func:`python_eval` — a *restricted* evaluator for a single expression. It is
  hardened (no builtins, AST-vetted node types) but restricted execution in the
  real CPython process is not a security boundary; see its docstring.
* :func:`mock_search` — returns canned results so "search" is deterministic and
  needs no network.
"""

from __future__ import annotations

import ast
import operator
from typing import Any

# --------------------------------------------------------------------------- #
# calculator                                                                  #
# --------------------------------------------------------------------------- #

# Binary and unary operators we are willing to execute. Anything not in these
# tables is rejected, so the set of allowed behaviour is an allow-list, not a
# block-list (block-lists always miss something).
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Cap the exponent so a single call cannot pin the CPU with e.g. 9**9**9.
_MAX_POW_EXPONENT = 1000


def _eval_node(node: ast.AST) -> float:
    """Recursively evaluate one vetted AST node into a number."""
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError(f"only numbers allowed, got {node.value!r}")
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise ValueError(f"operator {op_type.__name__} not allowed")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if op_type is ast.Pow and abs(right) > _MAX_POW_EXPONENT:
            raise ValueError("exponent too large")
        return _BIN_OPS[op_type](left, right)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _UNARY_OPS:
            raise ValueError(f"unary operator {op_type.__name__} not allowed")
        return _UNARY_OPS[op_type](_eval_node(node.operand))
    # Names, calls, attributes, subscripts, comprehensions, etc. all land here.
    raise ValueError(f"disallowed expression element: {type(node).__name__}")


def calculator(expression: str) -> float:
    """Safely evaluate an arithmetic ``expression`` and return the number.

    Supports integer/float literals, ``+ - * / // % **``, unary ``+ -`` and
    parentheses. Rejects everything else — names, function calls, attribute
    access, imports — by walking the parsed AST against an allow-list of node
    types, so a malicious string such as ``__import__('os').system('rm -rf /')``
    raises ``ValueError`` instead of executing.

    Args:
        expression: an arithmetic expression, e.g. ``"2 + 3 * 4"``.

    Returns:
        The numeric result as ``float`` or ``int`` (Python's own numeric type).

    Raises:
        ValueError: if the string is not a pure arithmetic expression, or on a
            math error such as division by zero (re-wrapped for a clean message).

    Safety limits (honest): this is safe against *code execution* because it
    never runs the raw string — only vetted arithmetic AST nodes. It is not a
    sandbox against resource exhaustion beyond a capped ``**`` exponent; a giant
    literal expression could still be slow to parse.
    """
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"could not parse expression: {exc}") from exc
    try:
        return _eval_node(tree)
    except ZeroDivisionError as exc:
        raise ValueError("division by zero") from exc


# --------------------------------------------------------------------------- #
# python_eval                                                                 #
# --------------------------------------------------------------------------- #

# Extra node types python_eval tolerates on top of the calculator's arithmetic:
# comparisons, boolean/bitwise logic, and container/literal building. Still no
# Name, Call, Attribute, Import, or comprehension — nothing that reaches names.
_EXTRA_EVAL_NODES = (
    ast.Compare, ast.BoolOp, ast.And, ast.Or, ast.Not,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.List, ast.Tuple, ast.Set, ast.Dict,
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.Load,
)


def _vet_eval_tree(node: ast.AST) -> None:
    """Raise ``ValueError`` if any node is outside the python_eval allow-list."""
    allowed = (
        ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp,
        *_BIN_OPS, *_UNARY_OPS, *_EXTRA_EVAL_NODES,
    )
    for child in ast.walk(node):
        if not isinstance(child, allowed):
            raise ValueError(
                f"python_eval: disallowed element {type(child).__name__}"
            )


def python_eval(code: str) -> Any:
    """Evaluate a single restricted Python *expression* and return its value.

    On top of arithmetic this allows comparisons, boolean/bitwise logic and
    literal containers (lists/tuples/dicts/sets), so a model can compute e.g.
    ``"2 < 3 and 10 % 2 == 0"`` -> ``True`` or ``"[x for ...]"`` is *rejected*.

    It runs the compiled expression with ``{"__builtins__": {}}`` as globals, so
    there are no builtins (no ``open``, ``__import__``, ``eval``) and the AST is
    vetted first so there are no names, calls, attributes, or comprehensions.

    Args:
        code: a single Python expression (not statements).

    Returns:
        The value of the expression (any JSON-friendly Python value).

    Raises:
        ValueError: if parsing fails or a disallowed construct is present.

    Safety limits (honest and important): restricting ``eval`` inside the real
    CPython interpreter is **hardening, not a security sandbox**. The allow-list
    here blocks the obvious escapes, but for genuinely untrusted code you must
    isolate execution at the OS level (a separate process, container, seccomp,
    resource limits). Do not point this at adversarial input and call it safe.
    """
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"could not parse code: {exc}") from exc
    _vet_eval_tree(tree)
    compiled = compile(tree, filename="<python_eval>", mode="eval")
    return eval(compiled, {"__builtins__": {}}, {})  # noqa: S307 — vetted AST, no builtins


# --------------------------------------------------------------------------- #
# mock_search                                                                 #
# --------------------------------------------------------------------------- #

# A tiny canned "index" so search is deterministic and offline. Keys are matched
# case-insensitively as substrings of the query.
_CANNED_RESULTS: dict[str, str] = {
    "capital of france": "Paris is the capital of France.",
    "speed of light": "The speed of light in vacuum is 299,792,458 m/s.",
    "chinchilla": "Chinchilla (Hoffmann et al., 2022) found compute-optimal "
                  "training uses ~20 tokens per parameter.",
}


def mock_search(query: str) -> str:
    """Return a canned search result for ``query`` (deterministic, no network).

    Args:
        query: the search string.

    Returns:
        A canned answer string if any known key is a substring of the query
        (case-insensitive), otherwise a fixed "no results" message. Never raises
        for an unknown query — it just reports no results.

    Safety limits (honest): this performs no real network I/O by design, so it is
    safe and reproducible but obviously not a real search engine.
    """
    q = query.lower()
    for key, answer in _CANNED_RESULTS.items():
        if key in q:
            return answer
    return f"No results found for {query!r}."


# --------------------------------------------------------------------------- #
# registration helper                                                         #
# --------------------------------------------------------------------------- #

def register_builtins(registry: Any) -> Any:
    """Register calculator/python_eval/mock_search on a :class:`ToolRegistry`.

    Args:
        registry: a ``llmre.tools.registry.ToolRegistry`` (typed as ``Any`` to
            avoid a circular import).

    Returns:
        The same registry, so calls can be chained.
    """
    registry.register(
        calculator,
        name="calculator",
        description="Evaluate an arithmetic expression and return the number.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Arithmetic expression, e.g. '2 + 3 * 4'.",
                }
            },
            "required": ["expression"],
        },
    )
    registry.register(
        python_eval,
        name="python_eval",
        description="Evaluate a single restricted Python expression.",
        parameters={
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "A single Python expression (no statements).",
                }
            },
            "required": ["code"],
        },
    )
    registry.register(
        mock_search,
        name="mock_search",
        description="Look up a query in a small canned index (offline mock).",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for.",
                }
            },
            "required": ["query"],
        },
    )
    return registry
