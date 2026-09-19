"""llmre.agents.react — a minimal, deterministic ReAct loop (Module 19.1).

ReAct (Yao et al. 2022, https://arxiv.org/abs/2210.03629) interleaves three
things in one loop: a **Thought** (the policy reasons about what to do), an
**Action** (it calls a tool), and an **Observation** (the tool's result is fed
back). The loop repeats until the policy decides to emit a final answer or a step
budget is exhausted.

Design note — *tool-registry-agnostic by construction*. :func:`run_react` never
imports any concrete tool module. It receives a ``tools`` mapping
(name -> callable) as an argument, i.e. dependency injection. That keeps this
file decoupled from ``llmre.tools`` (Module 18) and makes the loop trivially
testable with tiny local tools.

There are **no tensors** in this module: an agent loop is plain control flow
around a policy and a set of callables. In a real system the policy would be a
capable LLM emitting thoughts and tool calls; here it is any Python callable, so
the loop is fully deterministic and unit-testable without a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

# An Action the policy returns is a plain dict, either:
#   {"tool": <name:str>, "args": {<kwargs>}}   -> call tools[name](**args)
#   {"final": <answer:Any>}                    -> stop and return the answer
# It may also carry an optional "thought": <str> for the trace/log.
Action = dict
Tools = dict[str, Callable[..., Any]]
# A policy maps the running history to the next Action.
Policy = Callable[[list["Step"]], Action]


@dataclass
class Step:
    """One Thought -> Action -> Observation record in the trace.

    Attributes (all plain Python, no tensors):
        thought:     the policy's reasoning string for this step (may be "").
        action:      the raw Action dict the policy returned.
        observation: the tool's return value, or an error string if the tool
                     raised / was unknown. ``None`` on the final step (no tool
                     is executed when the policy finalizes).
    """

    thought: str
    action: Action
    observation: Any = None


@dataclass
class ReActResult:
    """The outcome of a :func:`run_react` run.

    Attributes:
        answer:   the final answer the policy produced, or ``None`` if the loop
                  hit ``max_steps`` without the policy finalizing.
        trace:    the ordered list of :class:`Step` records (the full
                  Thought/Action/Observation history).
        finished: ``True`` iff the loop stopped because the policy emitted a
                  ``{"final": ...}`` action; ``False`` iff it halted at
                  ``max_steps``.
        steps:    number of policy calls actually made.
    """

    answer: Any
    trace: list[Step] = field(default_factory=list)
    finished: bool = False
    steps: int = 0


def run_react(
    policy: Policy,
    tools: Tools,
    task: Any,
    max_steps: int = 10,
) -> ReActResult:
    """Run the Thought/Action/Observation loop until a final answer or budget.

    The loop, each iteration:
      1. asks ``policy(history)`` for the next :data:`Action`;
      2. if the action is ``{"final": answer}``, records the step and returns;
      3. otherwise treats it as ``{"tool": name, "args": {...}}``, executes
         ``tools[name](**args)``, and appends the result as the Observation so
         the policy can see it on the next call;
      4. stops after ``max_steps`` policy calls even if nothing was finalized.

    Args:
        policy:    callable mapping the current ``list[Step]`` history to the
                   next Action dict. Deterministic here; an LLM in practice.
        tools:     injected mapping ``name -> callable``. Never imported; passed
                   in. Callables take keyword args and return any observation.
        task:      the task description; seeded as the first history entry so the
                   policy can read it via ``history[0]``.
        max_steps: maximum number of policy calls (>= 1). The loop halts here if
                   the policy never finalizes, leaving ``answer=None`` and
                   ``finished=False``.

    Returns:
        :class:`ReActResult` with the answer (or ``None``), the full trace,
        whether it finished, and the step count.

    Raises:
        ValueError: if ``max_steps < 1`` or an action is malformed (neither a
                    ``final`` nor a ``tool`` key).
    """
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")

    # Seed the history with the task as a synthetic step 0, so policies can read
    # the goal from history[0].observation without a separate argument channel.
    history: list[Step] = [
        Step(thought="task", action={"task": task}, observation=task)
    ]

    for _ in range(max_steps):
        action = policy(history)
        thought = str(action.get("thought", "")) if isinstance(action, dict) else ""

        if not isinstance(action, dict):
            raise ValueError(f"policy must return a dict action, got {type(action)!r}")

        if "final" in action:
            history.append(Step(thought=thought, action=action, observation=None))
            return ReActResult(
                answer=action["final"],
                trace=history,
                finished=True,
                # step 0 is the synthetic task seed; count only policy steps.
                steps=len(history) - 1,
            )

        if "tool" not in action:
            raise ValueError(
                "action must contain either 'final' or 'tool'; "
                f"got keys {sorted(action.keys())}"
            )

        name = action["tool"]
        args = action.get("args", {}) or {}
        if name not in tools:
            # An unknown tool is an Observation, not a crash: a real policy can
            # read the error and recover on the next step.
            observation: Any = f"error: unknown tool {name!r}"
        else:
            try:
                observation = tools[name](**args)
            except Exception as exc:  # surface tool failures as observations
                observation = f"error: {type(exc).__name__}: {exc}"

        history.append(Step(thought=thought, action=action, observation=observation))

    # Budget exhausted with no final answer.
    return ReActResult(
        answer=None,
        trace=history,
        finished=False,
        steps=len(history) - 1,
    )


def format_trace(result: ReActResult) -> str:
    """Render a :class:`ReActResult` as a readable Thought/Action/Observation log.

    Returns a plain ``str`` (one block per step). Useful for debugging an agent
    run by eye; not used by the loop itself.
    """
    lines: list[str] = []
    for i, step in enumerate(result.trace):
        if i == 0:
            lines.append(f"Task: {step.observation!r}")
            continue
        if step.thought:
            lines.append(f"Thought {i}: {step.thought}")
        if "final" in step.action:
            lines.append(f"Final {i}: {step.action['final']!r}")
        else:
            lines.append(
                f"Action {i}: {step.action.get('tool')}"
                f"({step.action.get('args', {})})"
            )
            lines.append(f"Observation {i}: {step.observation!r}")
    status = "finished" if result.finished else "halted (max_steps)"
    lines.append(f"[{status} after {result.steps} step(s); answer={result.answer!r}]")
    return "\n".join(lines)
