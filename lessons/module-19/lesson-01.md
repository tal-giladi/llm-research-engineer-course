# 19.1 · ReAct → a mini SWE-agent

<div class="prereq">
<p><strong>Prerequisites:</strong> tool use and function calling from <a href="#/lessons/module-18/lesson-01">18.1 · Tool schemas &amp; function calling</a> and the observation/action loop from <a href="#/lessons/module-18/lesson-02">18.2 · The observation/action loop</a>; generation (how a model produces text token by token) from <a href="#/lessons/module-06/lesson-03">06.3 · Generation</a>. Comfort reading plain Python control flow — this lesson has no tensors.</p>
<p><strong>You will learn:</strong> the ladder from a plain LLM to a tool-calling model to a <strong>ReAct</strong> agent (interleaved Thought → Action → Observation) to a planning, iterating agent; how to build a small, <em>deterministic</em>, tool-registry-agnostic ReAct loop you can unit-test without a model; and how the same loop, given file-oriented tools, becomes a <strong>mini SWE-agent</strong> that inspects code, edits it, runs tests, and retries.</p>
<p><strong>Why this matters for ML:</strong> agents are how the models you have spent this course training get <em>used</em> to do real work — and, increasingly, how they generate their own training data, run their own experiments, and fix their own code. As a research engineer you will both build agent scaffolds and be the human policy debugging them. The loop is simple; getting the interface and the feedback right is the craft.</p>
</div>

## 1. The causal story: one tool call is not enough

In <a href="#/lessons/module-18/lesson-01">Module 18</a> you gave a model a tool — a calculator, a search function, a `get_weather(city)` — and let it emit one call, splice the result back into the context, and continue. That answers *one* self-contained question: "what is 47 × 89?", "what is the weather in Oslo?".

Real tasks are not one question. "Fix the failing test in this repo" is a chain: look at the failure, find the file, read the relevant function, form a guess, edit it, rerun the tests, see a *new* failure, revise. Each step depends on what the previous step revealed. You cannot plan the whole thing up front because you do not yet know what you will see.

<div class="callout key"><p>A single tool call answers one question. Real tasks need many iterative actions with feedback — inspect, act, observe, correct. That loop, driven by a model that reasons about what to do next, <strong>is an agent</strong>.</p></div>

The ladder, rung by rung:

- **Plain LLM** — text in, text out. No access to the world.
- **Tool-calling LLM** (Module 18) — can emit *one* structured call and read *one* result.
- **ReAct** (this lesson) — interleaves reasoning and acting in a *loop*: think, act, observe, repeat.
- **Planning + iterative action** — the agent maintains a goal, breaks it into sub-steps, and adapts the plan as observations arrive.
- **Agent** — the whole thing wrapped with a task, a tool interface, and a stopping condition.

## 2. ReAct: Thought → Action → Observation

**ReAct** (Yao et al. 2022, "Reasoning + Acting") is the pattern under essentially every LLM agent. At each step the model emits two things and then reads a third:

1. **Thought** — a short natural-language reasoning step: *what do I know, what should I do next?* This is just generated text (Module 6), but it forces the model to plan in the open where the plan can condition the next action.
2. **Action** — a structured tool call, exactly the function-calling format of Module 18: a tool name plus arguments.
3. **Observation** — the tool's return value, fed back into the context so the next Thought can use it.

Then it loops. Reasoning helps choose good actions; actions ground the reasoning in real feedback instead of letting it hallucinate. Yao et al. showed this beats reasoning-only (the model makes things up) and acting-only (the model flails without a plan).

<div class="callout key"><p>ReAct = repeat <strong>Thought → Action → Observation</strong> until the model emits a final answer. The Thought is generated text; the Action is a tool call; the Observation is the tool result appended to the running history.</p></div>

### 2.1 Why a *deterministic* loop for a course

In production the "policy" that emits each Thought and Action is a capable LLM. But an LLM policy is non-deterministic and needs a GPU or an API key, which makes it useless for teaching the *mechanism* and impossible to unit-test. So we separate the two concerns cleanly:

- The **loop** is dumb, deterministic control flow: call the policy, run the tool it names, append the observation, repeat.
- The **policy** is any Python callable `policy(history) -> action`. In this lesson it is a small scripted function so every run is reproducible. **In practice it would be an LLM.** We are honest that our agent is a *scaffold*: without a capable policy it only does what the script says.

This split is exactly how real agent frameworks are structured, and it lets you understand and test the plumbing before worrying about the model.

## 3. A worked trace, by hand

Take the toy task *compute (3 + 4) × 2* with two injected tools, `add(a, b)` and `mul(a, b)`. A scripted policy that solves it:

- **Step 1** — sees only the task. Thought: "add 3 + 4". Action: `add(a=3, b=4)`. Observation: `7`.
- **Step 2** — sees the observation `7`. Thought: "multiply the sum by 2". Action: `mul(a=7, b=2)`. Observation: `14`.
- **Step 3** — sees the observation `14`. Thought: "return the product". Action: `{"final": 14}`. The loop stops.

Written out as the trace the loop actually records:

```text
Task: 'compute (3 + 4) * 2'
Thought 1: add 3 + 4
Action 1: add({'a': 3, 'b': 4})
Observation 1: 7
Thought 2: multiply the sum by 2
Action 2: mul({'a': 7, 'b': 2})
Observation 2: 14
Thought 3: return the product
Final 3: 14
[finished after 3 step(s); answer=14]
```

Notice the dependency chain: step 2's argument `a=7` came from step 1's observation. That is the whole point of a loop over a single call — the second action could not have been written in advance.

## 4. The loop, from scratch

Here is the core of `code/src/llmre/agents/react.py`. Read it as three moves: ask the policy, branch on whether it finalized, otherwise run the named tool and record the observation.

```python
def run_react(policy, tools, task, max_steps=10):
    if max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")

    # Seed history with the task as a synthetic step 0 so the policy can
    # read the goal from history[0].observation.
    history = [Step(thought="task", action={"task": task}, observation=task)]

    for _ in range(max_steps):
        action = policy(history)                      # 1) ask the policy
        thought = str(action.get("thought", ""))

        if "final" in action:                         # 2) policy is done
            history.append(Step(thought, action, observation=None))
            return ReActResult(answer=action["final"], trace=history,
                               finished=True, steps=len(history) - 1)

        name = action["tool"]                         # 3) run the named tool
        args = action.get("args", {}) or {}
        if name not in tools:
            observation = f"error: unknown tool {name!r}"
        else:
            try:
                observation = tools[name](**args)
            except Exception as exc:
                observation = f"error: {type(exc).__name__}: {exc}"
        history.append(Step(thought, action, observation))

    # Budget exhausted with no final answer.
    return ReActResult(answer=None, trace=history, finished=False,
                       steps=len(history) - 1)
```

Four design choices worth naming:

- **Dependency injection of tools.** `tools` is a plain `dict[str, callable]` passed *in*. `run_react` never imports any tool module, so it is completely decoupled from `llmre.tools` (Module 18). You can drive it with a calculator, a filesystem, a search API, or the tiny local tools our tests define — the loop does not care.
- **The task is seeded as `history[0]`.** The policy reads its goal from the history like any other step, so the loop needs no special "task" channel.
- **Errors become observations, not crashes.** An unknown tool or a tool that raises returns an error *string* the policy can read and recover from on the next step — exactly how a real agent handles a failed command. A crash would end the episode; an observation lets it retry.
- **`max_steps` is a hard budget.** Without it, a policy that never finalizes loops forever (and a real LLM policy sometimes does). Hitting the budget returns `answer=None, finished=False` — a clean, testable halt.

<div class="callout pt"><p>There is no PyTorch and no tensor in this file, and that is the lesson: an agent is control flow around a policy. What PyTorch <em>would</em> do — run the transformer forward pass that produces each Thought and Action — is hidden inside <code>policy(history)</code>. Swapping our scripted policy for a model changes nothing about the loop.</p></div>

## 5. From a loop to a mini SWE-agent

The same loop becomes a coding agent the moment its tools are *file-oriented*. This is the central idea of **SWE-agent** (Yang et al. 2024): the **Agent-Computer Interface (ACI)** — the specific set of commands you give the agent for viewing, editing, and running code — matters as much as the model. A well-designed, narrow interface (scoped edits, a `run_tests` command that returns a clean pass/fail) beats handing the model a raw shell, because every command returns a tidy observation the model can actually reason about.

Our scaffold in `code/src/llmre/agents/swe_agent.py` gives the loop four tools over an in-memory `Workspace` (a `dict` of `path -> contents`, so the scaffold is hermetic and testable):

```python
read_file(path)            # -> contents, or "error: no such file ..."
write_file(path, content)  # -> "wrote N chars to ..."   (mutates the workspace)
run_tests()                # -> "PASS: ..." or "FAIL: ..."  (from an injected checker)
list_files()               # -> sorted list of paths
```

`solve(policy, workspace, task, test_runner)` is *just* `run_react` with these tools:

```python
def solve(policy, workspace, task, test_runner, max_steps=12):
    tools = make_tools(workspace, test_runner)   # build the ACI
    return run_react(policy=policy, tools=tools, task=task, max_steps=max_steps)
```

The agent loop is then: `read_file` (inspect) → `write_file` (edit) → `run_tests` (observe PASS/FAIL) → decide → retry if it failed. That inspect-act-observe-correct cycle is what makes it an *agent* and not a one-shot code generator.

<div class="callout warn"><p>Be honest about what this is. Our <code>policy</code> is scripted, so <code>solve</code> only fixes the specific bug the script was written to fix. A <em>real</em> SWE-agent's policy is a capable LLM that decides, unprompted, which file to open and what edit to make. The scaffold teaches the interface and the loop; it does not teach the model the loop needs. Everything hard about SWE-agent lives in that missing policy.</p></div>

<div class="callout paper"><p><strong>Papers.</strong> <a href="#/papers/index">ReAct (Yao et al. 2022)</a> introduces the Thought/Action/Observation loop; <a href="#/papers/index">SWE-agent (Yang et al. 2024)</a> shows that the agent-computer interface — not just the model — determines how well a coding agent performs on real GitHub issues (SWE-bench). Read them together after this lesson; the code above is a miniature of both.</p></div>

<div class="hw">
<p><strong>Hardware track.</strong> None. Everything in this lesson runs on CPU in well under a second — there are no tensors, no model forward passes, no GPU. The only "compute" is Python control flow. A real agent's cost is entirely in the LLM policy calls (API latency or GPU inference), which this scaffold deliberately factors out.</p>
</div>

## 6. Running it

From `code/`:

```bash
py -m pytest tests/test_agents.py -q
```

The tests in `code/tests/test_agents.py` define their **own** tiny local tools (`add`, `mul`, `noop`) rather than importing `llmre.tools`, which is exactly the decoupling that lets Module 19 be built and tested independently of Module 18. They check that the scripted policy solves the arithmetic task in ≤ 3 steps, that the loop halts at `max_steps` when the policy never finalizes, and that the SWE-agent scaffold fixes a toy file.

## Exercise

Extend the arithmetic agent to compute *(10 − 3) × (2 + 5)* using injected `sub`, `add`, and `mul` tools, with a scripted policy, in as few steps as possible. Then answer: what is the minimum number of steps, and why can't the loop do it in one?

<p><em>Optional hint:</em> the two sub-expressions are independent, but each tool call returns exactly one number, and the final <code>mul</code> needs both results.</p>

<p><em>Stronger hint:</em> you need one call per binary operation, and the <code>mul</code> cannot run until both its inputs exist as observations. Count the binary operations in the expression.</p>

<details><summary>Solution</summary>

There are three binary operations — one subtraction, one addition, one multiplication — so the minimum is **three tool-call steps plus a final step**. The policy: step 1 `sub(a=10, b=3) -> 7`; step 2 `add(a=2, b=5) -> 7`; step 3 `mul(a=7, b=7) -> 49`; step 4 `{"final": 49}`. It cannot be done in one call because each tool returns a single value and `mul` depends on the outputs of the other two — the second and third actions can only be written *after* seeing the earlier observations. A sketch:

```python
def policy(history):
    last = history[-1]
    if len(history) == 1:
        return {"tool": "sub", "args": {"a": 10, "b": 3}}    # -> 7
    if last.action.get("tool") == "sub":
        return {"tool": "add", "args": {"a": 2, "b": 5}}      # -> 7
    if last.action.get("tool") == "add":
        left = history[1].observation                          # the sub result
        return {"tool": "mul", "args": {"a": left, "b": last.observation}}
    return {"final": last.observation}                         # 49
```

Note step 3 reaches *back* into the history (`history[1].observation`) for the subtraction result — the running trace is the agent's memory.

</details>

## Common mistakes

- **Letting a tool error crash the loop.** If `tools[name](**args)` is allowed to propagate, one bad action ends the whole episode. Catch it and return the error as an observation so the policy can recover — that is what real agents do.
- **No step budget.** An LLM policy will sometimes loop forever (call, observe, call the same thing again). Always cap iterations; treat "hit the budget" as a distinct, reported outcome, not success.
- **Importing the tool registry into the loop.** The moment `run_react` imports `llmre.tools`, it is coupled to a specific tool set and cannot be tested in isolation or reused with a different interface. Inject the tools.
- **Confusing the scaffold with an agent.** A deterministic policy is a test harness, not intelligence. The hard, unsolved-by-this-code part is the LLM that decides what to do.

## Debugging exercise

This policy is meant to add then finalize, but the loop always halts at `max_steps` with `answer=None`. Find the bug.

```python
def policy(history):
    last = history[-1]
    if len(history) == 1:
        return {"thought": "add", "tool": "add", "args": {"a": 3, "b": 4}}
    if last.action.get("tool") == "sum":       # <-- ?
        return {"final": last.observation}
    return {"thought": "wait", "tool": "add", "args": {"a": 0, "b": 0}}
```

<details><summary>Answer</summary>

The tool is named `add`, but the branch checks `last.action.get("tool") == "sum"`. That comparison is never true, so the policy never reaches the `final` branch; it falls through to the `add(0, 0)` no-op every step and the loop runs until `max_steps` and returns `answer=None, finished=False`. Fix: compare against `"add"`. A general lesson — the policy branches on the *history*, so an agent that never finalizes is almost always a policy that never recognizes the state in which it should stop.

</details>

## Check yourself

<details><summary>What are the three parts of a ReAct step, and which one is fed back into the model's context for the next step?</summary>

Thought (generated reasoning text), Action (a structured tool call), and Observation (the tool's result). The Observation is appended to the running history so the next Thought and Action can condition on it. In our implementation the whole `Step` — thought, action, and observation — is appended to `history`, and the policy receives the full history each iteration.

</details>

<details><summary>Why does <code>run_react</code> take <code>tools</code> as an argument instead of importing them?</summary>

Dependency injection keeps the loop tool-registry-agnostic: it is completely decoupled from any concrete tool module (like `llmre.tools` in Module 18), so it can be reused with any interface and, crucially, unit-tested with tiny local tools. It also avoids a build-order dependency between modules that are written in parallel.

</details>

<details><summary>The loop returns <code>answer=None, finished=False</code>. What happened, and is it a crash?</summary>

The policy ran for `max_steps` iterations without ever returning a `{"final": ...}` action, so the budget was exhausted. It is not a crash — it is a clean, deliberate halt with a distinct return value you can test for. A real agent hitting this usually means the policy got stuck in a loop or never recognized a stopping state.

</details>

<details><summary>In the SWE-agent scaffold, what does the injected <code>test_runner</code> decide, and why is it injected rather than hard-coded?</summary>

`test_runner(workspace) -> (passed: bool, message: str)` decides what "passing" means for the task at hand. It is injected so the scaffold never hard-codes a notion of correctness — each task (or test) supplies its own checker, exactly as different repos have different test suites. The `run_tests` tool just formats its result as a `PASS`/`FAIL` observation.

</details>

<details><summary>Our SWE-agent "solves" a toy bug. Why is it fair to call it only a scaffold?</summary>

Because the policy is scripted: it only makes the edits the script encodes, so it solves exactly one pre-planned bug. A real SWE-agent's policy is a capable LLM that autonomously decides which file to inspect and what fix to write from the failing-test observation. The scaffold provides the loop and the agent-computer interface; the intelligence — the LLM policy — is the part it deliberately leaves out.

</details>

## Next

You can now build the loop that turns a model into an agent, and you understand why the interface and the feedback — not just the model — decide whether it works. The final piece of the research-engineer toolkit is not code at all: it is knowing how to design an experiment, measure it honestly across seeds, and write up what you found so it is trustworthy and reproducible.

Continue to [19.2 · Experiment design & research writing](lessons/module-19/lesson-02.md).
