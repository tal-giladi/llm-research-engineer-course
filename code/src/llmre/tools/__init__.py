"""llmre.tools — tool use: schemas, function calling, and the action loop.

See course Module 18. Public surface:

* :class:`~llmre.tools.registry.Tool`, :class:`~llmre.tools.registry.ToolRegistry`
* built-in tools :func:`~llmre.tools.builtins.calculator`,
  :func:`~llmre.tools.builtins.python_eval`,
  :func:`~llmre.tools.builtins.mock_search`, and
  :func:`~llmre.tools.builtins.register_builtins`
* :func:`~llmre.tools.parser.parse_tool_call`
* :func:`~llmre.tools.loop.run_tool_loop`
"""

from .builtins import (
    calculator,
    mock_search,
    python_eval,
    register_builtins,
)
from .loop import render_context, run_tool_loop
from .parser import parse_tool_call
from .registry import Tool, ToolRegistry

__all__ = [
    "Tool",
    "ToolRegistry",
    "calculator",
    "python_eval",
    "mock_search",
    "register_builtins",
    "parse_tool_call",
    "run_tool_loop",
    "render_context",
]
