"""llmre.agents.swe_agent — a tiny SWE-agent scaffold (Module 19.1).

SWE-agent (Yang et al. 2024, https://arxiv.org/abs/2405.15793) makes one point
concrete: an agent that fixes code needs an **Agent-Computer Interface (ACI)** —
a small, well-behaved set of commands for *inspecting* files, *editing* code, and
*running tests*, plus the observations they return. The model does not touch a
raw shell; it acts through this narrow interface and reads the feedback.

This module is a **scaffold**, not a coding agent. It supplies file-oriented
tools over an in-memory workspace and wires them into the tool-agnostic
:func:`llmre.agents.react.run_react` loop. The *policy* is injected: here it is
any deterministic callable, but **in practice the policy would be a capable
LLM** emitting thoughts and tool calls. We are honest about that: without a real
model policy, this solves only whatever a scripted policy is written to solve.

No tensors appear here — a SWE-agent is control flow plus a file interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .react import ReActResult, Tools, run_react


@dataclass
class Workspace:
    """An in-memory file tree standing in for a checked-out repo.

    Attributes:
        files: mapping ``path:str -> contents:str``. Plain strings, no bytes,
               no real filesystem — the scaffold stays hermetic and testable.
    """

    files: dict[str, str] = field(default_factory=dict)

    def copy(self) -> "Workspace":
        """Return a deep-enough copy (dict of immutable strings)."""
        return Workspace(files=dict(self.files))


def make_tools(
    workspace: Workspace,
    test_runner: Callable[[Workspace], tuple[bool, str]],
) -> Tools:
    """Build the ACI: the injected ``name -> callable`` mapping for the loop.

    The three commands mirror SWE-agent's core interface, scaled to a scaffold:

        read_file(path)              -> file contents, or an error string
        write_file(path, content)    -> confirmation string (mutates workspace)
        run_tests()                  -> "PASS: ..." or "FAIL: ..." from the runner
        list_files()                 -> sorted list of paths in the workspace

    Args:
        workspace:   the :class:`Workspace` the tools read from and write to.
                     Captured by closure so the tools carry no global state.
        test_runner: callable ``(Workspace) -> (passed: bool, message: str)``.
                     Injected so tests define their own checker; the scaffold
                     never hard-codes what "passing" means.

    Returns:
        A tools dict suitable to pass straight to :func:`run_react`.
    """

    def read_file(path: str) -> str:
        if path not in workspace.files:
            return f"error: no such file {path!r}"
        return workspace.files[path]

    def write_file(path: str, content: str) -> str:
        workspace.files[path] = content
        return f"wrote {len(content)} chars to {path!r}"

    def run_tests() -> str:
        passed, message = test_runner(workspace)
        return f"{'PASS' if passed else 'FAIL'}: {message}"

    def list_files() -> list[str]:
        return sorted(workspace.files)

    return {
        "read_file": read_file,
        "write_file": write_file,
        "run_tests": run_tests,
        "list_files": list_files,
    }


def solve(
    policy: Callable[[list], dict],
    workspace: Workspace,
    task: Any,
    test_runner: Callable[[Workspace], tuple[bool, str]],
    max_steps: int = 12,
) -> ReActResult:
    """Run a mini SWE-agent: inspect -> edit -> run tests -> observe -> retry.

    This is just :func:`run_react` with the file-oriented ACI from
    :func:`make_tools`. The agent reads files, writes a fix, runs the tests, and
    sees PASS/FAIL as an observation it can act on — the inspect/act/observe/
    correct cycle that makes it an agent rather than a single tool call.

    Args:
        policy:      injected decision function ``history -> action``. An LLM in
                     practice; a scripted callable in tests.
        workspace:   the :class:`Workspace` to edit (mutated in place).
        task:        the task/issue description, seeded into the loop history.
        test_runner: ``(Workspace) -> (passed, message)`` correctness check.
        max_steps:   step budget handed to the ReAct loop.

    Returns:
        The :class:`ReActResult` from the underlying loop. Whether it actually
        fixed anything depends entirely on the injected policy.
    """
    tools = make_tools(workspace, test_runner)
    return run_react(policy=policy, tools=tools, task=task, max_steps=max_steps)
