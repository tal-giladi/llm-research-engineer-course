"""Tool registry — the runtime side of function calling.

A language model that wants to *use* a tool never runs code itself. It emits a
structured request ("call tool ``calculator`` with ``expression='2+3'``"), and a
runtime looks that name up, runs the real Python function, and hands the result
back. This module is that runtime.

Two objects:

* :class:`Tool` — one callable plus the metadata a model needs to call it: a
  ``name``, a ``description``, and a JSON-schema ``parameters`` block describing
  the arguments.
* :class:`ToolRegistry` — a name -> Tool table with :meth:`register`,
  :meth:`get`, :meth:`call` (which turns any exception into a structured error
  result instead of crashing the loop), and :meth:`schemas` (the JSON the model
  is shown so it knows what exists).

No tensors here: this is plain Python orchestration around the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    """A single callable tool the model may invoke.

    Attributes:
        name: unique identifier the model emits to select this tool (str).
        description: one-line natural-language summary the model reads to decide
            *when* to use the tool (str).
        func: the actual Python callable; invoked as ``func(**arguments)``.
        parameters: a JSON-schema object (a plain ``dict``, JSON-serializable)
            describing the argument shape, e.g.
            ``{"type": "object", "properties": {...}, "required": [...]}``.
    """

    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict[str, Any] = field(default_factory=dict)

    def schema(self) -> dict[str, Any]:
        """Return this tool's JSON-serializable schema (name/description/params).

        Shape: ``{"name": str, "description": str, "parameters": dict}``. This is
        the exact object a model is shown so it can produce a valid call.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolRegistry:
    """A name -> :class:`Tool` table plus safe invocation.

    The registry is the single place the observation/action loop (see
    :mod:`llmre.tools.loop`) goes to (a) advertise available tools via
    :meth:`schemas` and (b) execute a parsed call via :meth:`call`.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        func: Callable[..., Any] | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> Callable[..., Any]:
        """Register ``func`` as a tool; usable directly or as a decorator.

        Direct::

            reg.register(calculator, name="calculator",
                         description="...", parameters={...})

        Decorator::

            @reg.register(name="calc", description="...", parameters={...})
            def calculator(expression: str) -> float: ...

        ``name`` defaults to ``func.__name__`` and ``description`` to the first
        line of the docstring. Returns ``func`` unchanged so decoration is
        transparent.
        """
        # Called as @reg.register(...) with no function yet: return a decorator.
        if func is None:
            def _decorator(f: Callable[..., Any]) -> Callable[..., Any]:
                self.register(f, name=name, description=description,
                              parameters=parameters)
                return f
            return _decorator

        tool_name = name or func.__name__
        if description is not None:
            desc = description
        elif func.__doc__:
            desc = func.__doc__.strip().splitlines()[0]
        else:
            desc = ""
        tool = Tool(
            name=tool_name,
            description=desc,
            func=func,
            parameters=parameters or {},
        )
        self._tools[tool_name] = tool
        return func

    def get(self, name: str) -> Tool:
        """Return the registered :class:`Tool` named ``name`` or raise ``KeyError``."""
        return self._tools[name]

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        """Sorted list of registered tool names (list[str])."""
        return sorted(self._tools)

    def call(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """Execute tool ``name`` with keyword ``kwargs``, catching every failure.

        Returns a JSON-serializable result dict, never raising for a tool
        problem (a crashing tool must not crash the agent loop):

        * success -> ``{"ok": True, "result": <value>}``
        * unknown tool -> ``{"ok": False, "error": "...", "error_type": "KeyError"}``
        * tool raised -> ``{"ok": False, "error": str(exc), "error_type": <cls>}``

        The ``error`` string is meant to be fed straight back to the model as an
        observation so it can correct itself.
        """
        tool = self._tools.get(name)
        if tool is None:
            return {
                "ok": False,
                "error": f"unknown tool {name!r}; available: {self.names()}",
                "error_type": "KeyError",
            }
        try:
            result = tool.func(**kwargs)
        except Exception as exc:  # noqa: BLE001 — deliberate: never crash the loop
            return {
                "ok": False,
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        return {"ok": True, "result": result}

    def schemas(self) -> list[dict[str, Any]]:
        """Return schemas for all tools, sorted by name (list[dict], JSON-safe).

        This is what a model is shown so it knows which tools exist and the
        argument shape of each. See :meth:`Tool.schema`.
        """
        return [self._tools[n].schema() for n in self.names()]
