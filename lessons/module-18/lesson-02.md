# 18.2 · The observation/action loop

<div class="prereq">
<p><strong>Prerequisites:</strong> tool schemas, the tool registry, and <code>parse_tool_call</code> from <a href="lesson-01.md">18.1 · Tool schemas & function calling</a>. The chat/role message structure from <a href="../module-14/lesson-01.md">14.1 · Instruction data & chat templates</a>.</p>
<p><strong>You will learn:</strong> the core <strong>observation/action loop</strong> — model proposes an action (a tool call) → runtime executes it → returns an <strong>observation</strong> → the model continues with that observation in context → repeat until a final answer. How to handle <strong>tool errors and retries</strong> by catching failures and feeding them back so the model can correct. How to chain multiple tool calls. You will read a deterministic driver loop with a scripted policy that calls the calculator, then answers.</p>
<p><strong>Why this matters for ML:</strong> a single tool call (18.1) rarely finishes a task. Real work is a sequence: look something up, compute on the result, check it, answer. The loop that alternates model thinking and tool acting — feeding each observation back into context — is the pattern behind every LLM agent. Getting it simple, deterministic, and error-tolerant here is what makes the SWE-agent of Module 19 tractable.</p>
</div>

## 1. The causal story: one call is not enough

In 18.1 the model emitted one calculator call, we ran it, and we had the answer. That worked because the task was one step. Most tasks are not:

> "What is the capital of France, and how many letters are in its name?"

No single tool answers this. The model must (1) call `mock_search` to get "Paris", (2) read that observation, (3) call a tool to count the letters, (4) read *that*, and only then (5) answer "Paris, 5 letters." Each step depends on the result of the previous one, which the model could not know in advance.

So we need a **loop**: let the model act, show it what happened, let it act again, and stop when it produces a plain answer instead of a tool call. That loop is the whole of agentic behavior, stripped to its skeleton.

<div class="callout key"><p>The observation/action loop: <strong>the model proposes an action → the runtime executes it → the result becomes an observation appended to the context → the model runs again with that new information → repeat until the model returns a final answer (no tool call).</strong> The model supplies intelligence; the loop supplies memory (the growing context) and hands (the tools).</p></div>

## 2. Intuition: think, act, observe, repeat

Picture the model as a person solving a problem with a notepad and a set of instruments. The notepad is the **context**: everything said and seen so far. Each turn, the person reads the whole notepad, decides one thing to do next, and either:

- **acts** — writes down "use the calculator on `2+3*4`" (a tool call), or
- **answers** — writes down the final conclusion (prose, no call).

If they acted, someone runs the instrument and writes the result — the **observation** — onto the notepad. Then the person reads the notepad again, now richer by one fact, and decides the next thing. The process ends when they write an answer instead of an action.

Three properties make this work and make it safe to run in code:

1. **The context is the only memory.** The model is stateless between turns; everything it "remembers" is the message list we pass it. Appending observations is literally how the model learns what the tools returned.
2. **One action per turn.** Keeping it to one call per step makes the loop trivial to trace, test, and reason about. (Production systems sometimes allow parallel calls; we do not need that to understand the mechanism.)
3. **A step budget.** A model that keeps calling tools and never answers would loop forever. `max_steps` is the hard stop.

## 3. The loop, precisely — every symbol named

Let the loop take four things:

- **`policy`** — a callable `policy(context) -> str`. Given the current context (the list of messages so far), it returns the model's next text: either a tool-call JSON or a final prose answer. In production `policy` is "run the LLM's forward pass and decode"; in our tests it is a small scripted function, so the whole loop is deterministic and offline.
- **`registry`** — the `ToolRegistry` from 18.1 that executes calls and, crucially, never raises.
- **`prompt`** — the user's task, seeded as the first message.
- **`max_steps`** — the maximum number of model turns.

The **context** is a list of message dicts, each with a `role`:

- `user` — the original prompt.
- `assistant` — each thing the model emitted (a call or an answer).
- `tool` — each observation we appended after running a tool.

The loop, in pseudocode:

```
context = [user: prompt]
repeat up to max_steps:
    output   = policy(context)              # model's turn
    append assistant: output
    call     = parse_tool_call(output)
    if call is None:                        # prose, not a call
        return final answer = output
    result   = registry.call(call.name, **call.arguments)   # never raises
    append tool: observation(result)        # feed the result back
return no answer (hit max_steps)
```

Read the one branch that matters: `parse_tool_call` returning `None` is the signal "the model answered." Anything else is an action we must execute and feed back.

## 4. From-scratch implementation

The driver lives at `code/src/llmre/tools/loop.py`. Here is its core, lightly trimmed:

```python
def run_tool_loop(policy, registry, prompt, max_steps=10) -> dict:
    context = [{"role": "user", "content": prompt}]
    trace = []

    for step in range(max_steps):
        output = policy(context)                              # 1. model acts
        context.append({"role": "assistant", "content": output})

        call = parse_tool_call(output)
        if call is None:                                     # 2. prose => done
            return {"answer": output, "trace": trace,
                    "context": context, "stopped": "final_answer"}

        result = registry.call(call["name"], **call["arguments"])  # 3. execute
        trace.append({"step": step, "name": call["name"],
                      "arguments": call["arguments"], "result": result})
        context.append({"role": "tool", "name": call["name"],      # 4. observe
                        "content": _observation_text(result)})

    return {"answer": None, "trace": trace,
            "context": context, "stopped": "max_steps"}
```

The observation text is where a result dict becomes something the model can read:

```python
def _observation_text(result: dict) -> str:
    if result.get("ok"):
        return f"OBSERVATION: {result['result']}"
    return f"OBSERVATION: ERROR ({result.get('error_type')}): {result.get('error')}"
```

Notice the return value. It is a plain, JSON-serializable dict with four keys:

- `answer` — the final prose (or `None` if we hit `max_steps`).
- `trace` — the list of executed calls, one entry per tool call, each recording the step index, tool name, arguments, and the raw result dict. `len(trace)` is exactly the number of tool calls made.
- `context` — the full message list, for inspection or rendering.
- `stopped` — `"final_answer"` or `"max_steps"`, so the caller knows *why* the loop ended.

The `trace` is the audit log of the run: it is what you inspect to see what the agent actually did, and what tests assert against.

## 5. Numerical example: calculator then answer, worked step by step

Here is the smallest complete run. A scripted policy that, on the first turn (last message is the `user` prompt), emits a calculator call, and on the second turn (last message is the tool observation) reads the number off it and answers. Running against `code/src/llmre/tools/` gives the outputs shown, verbatim.

```python
from llmre.tools.registry import ToolRegistry
from llmre.tools.builtins import register_builtins
from llmre.tools.loop import run_tool_loop

reg = register_builtins(ToolRegistry())

def policy(context):
    last = context[-1]
    if last["role"] == "user":
        return '{"name":"calculator","arguments":{"expression":"2+3*4"}}'
    value = last["content"].split()[-1]           # "OBSERVATION: 14" -> "14"
    return f"The answer is {value}."

out = run_tool_loop(policy, reg, "What is 2+3*4?", max_steps=5)
print(out["answer"])     # The answer is 14.
print(out["stopped"])    # final_answer
print(len(out["trace"])) # 1
print(out["trace"][0])   # {'step': 0, 'name': 'calculator',
                         #  'arguments': {'expression': '2+3*4'},
                         #  'result': {'ok': True, 'result': 14}}
```

Trace the loop by hand:

1. **Step 0.** Context is `[user: "What is 2+3*4?"]`. `policy` sees the last role is `user` and returns the calculator call. We append it as `assistant`. `parse_tool_call` returns `{"name": "calculator", "arguments": {"expression": "2+3*4"}}` — not `None`, so it is an action. `registry.call` runs the safe evaluator: `14`. We append `tool: "OBSERVATION: 14"` and record the trace entry.
2. **Step 1.** Context now ends with the tool observation. `policy` sees the last role is `tool`, splits `"OBSERVATION: 14"` on spaces, takes `"14"`, and returns `"The answer is 14."`. `parse_tool_call` returns `None` (prose, no JSON call), so the loop returns this as the final answer with `stopped="final_answer"`.

One tool call, two model turns, correct answer. The model never multiplied anything — the loop and the tool did the exact work; the model supplied the plan and read the result.

## 6. Retries and tool errors — the loop must not crash

The single most important robustness property: **a broken tool call becomes an observation, not an exception.** Because `registry.call` catches everything (18.1), the loop keeps going and the model gets a chance to correct itself. Here is a run where the model first divides by zero, sees the error, and recovers:

```python
calls = {"n": 0}
def policy(context):
    if calls["n"] == 0:
        calls["n"] += 1
        return '{"name":"calculator","arguments":{"expression":"1/0"}}'
    return "Recovered after the error."

out = run_tool_loop(policy, reg, "compute 1/0", max_steps=5)
```

The full transcript (from `render_context(out["context"])`), verbatim:

```
USER: compute 1/0
ASSISTANT: {"name":"calculator","arguments":{"expression":"1/0"}}
TOOL: OBSERVATION: ERROR (ValueError): division by zero
ASSISTANT: Recovered after the error.
```

And the trace records the failed call, so the audit log tells you what went wrong:

```python
out["answer"]              # 'Recovered after the error.'
out["trace"][0]["result"]  # {'ok': False, 'error': 'division by zero',
                           #  'error_type': 'ValueError'}
```

The key line is `TOOL: OBSERVATION: ERROR (ValueError): division by zero`. That error text is appended to the context exactly like a successful result would be. A capable model reads it and adjusts — fixes the argument, picks a different tool, or explains the problem. This is **retry via feedback**: we do not silently retry in code; we hand the error to the intelligence and let it decide.

<div class="callout key"><p>Two kinds of failure both become error observations: <strong>bad arguments</strong> (wrong keys, wrong types → a <code>TypeError</code> the registry catches) and <strong>tool exceptions</strong> (division by zero, a failed lookup). In every case the loop appends an <code>OBSERVATION: ERROR ...</code> message and continues. The model corrects from the feedback; the driver never crashes.</p></div>

<div class="callout warn"><p>Give the loop a <strong>step budget</strong> and honor it. Without <code>max_steps</code>, a model that keeps emitting calls (or keeps repeating a failing one) loops forever. Our loop returns <code>{"answer": None, "stopped": "max_steps"}</code> when the budget is exhausted, so the caller can detect a non-terminating run instead of hanging.</p></div>

## 7. Multi-step tool use

Nothing in the loop limits it to one call. A policy can chain tools by branching on what the last observation was. Sketch of a two-tool run for *"capital of France, and letters in its name"*:

- Turn 0: last role `user` → emit `mock_search("capital of France")`. Observation: `"Paris is the capital of France."`
- Turn 1: last role `tool`, observation mentions Paris → emit `calculator`/`python_eval` on the letter count, or a `word_count`-style tool. Observation: `5`.
- Turn 2: last role `tool` with the count → answer `"Paris, which has 5 letters."`

The `trace` would then have length 2, one entry per tool call, in order. The context grows by two `assistant`+`tool` pairs. The loop code does not change at all — only the policy's branching does. That is the payoff of keeping the driver simple: arbitrarily long tool-use sequences are just more iterations of the same four steps.

## 8. Under the hood: what is the "policy" really?

In our tests the policy is a scripted Python function so runs are deterministic. In a real system, `policy(context)` does this:

1. Render the `context` message list into a prompt string using the **chat template** from [14.1](../module-14/lesson-01.md) — the tool schemas go in the system message, and each `tool` observation is formatted as its own turn.
2. Tokenize and run the model's forward pass (Module 6), decode a completion — ideally with **constrained decoding** (18.1 §4) so any tool call is valid JSON.
3. Return the decoded text.

Everything else — parsing, executing, appending, the step budget — is exactly the driver you read here. Swapping the scripted policy for a real model changes *what text is produced*, not *how the loop runs*. That is why we could build and fully test the entire loop with no model and no network: the loop's correctness is independent of the policy's intelligence.

The compute cost is dominated by the model turns: each iteration is one full generation over a context that grows every step. A 5-step run re-reads (and re-attends over) an ever-longer context — so long tool-use sessions are why context length and KV-cache efficiency matter for agents. The tools themselves (a calculator, a lookup) are negligible by comparison.

## Unit test

From `code/tests/test_tools.py`:

```python
def test_run_tool_loop_calls_calculator_then_answers():
    reg = register_builtins(ToolRegistry())
    def policy(context):
        last = context[-1]
        if last["role"] == "user":
            return '{"name":"calculator","arguments":{"expression":"2+3*4"}}'
        return f"The answer is {last['content'].split()[-1]}."
    out = run_tool_loop(policy, reg, "What is 2+3*4?", max_steps=5)
    assert out["answer"] == "The answer is 14."
    assert len(out["trace"]) == 1
    assert out["trace"][0]["result"] == {"ok": True, "result": 14}

def test_run_tool_loop_feeds_back_tool_error():
    reg = register_builtins(ToolRegistry())
    calls = {"n": 0}
    def policy(context):
        if calls["n"] == 0:
            calls["n"] += 1
            return '{"name":"calculator","arguments":{"expression":"1/0"}}'
        return "Recovered after the error."
    out = run_tool_loop(policy, reg, "compute 1/0", max_steps=5)
    assert out["answer"] == "Recovered after the error."
    assert out["trace"][0]["result"]["ok"] is False   # caught, not crashed

def test_run_tool_loop_respects_max_steps():
    reg = register_builtins(ToolRegistry())
    def never_answers(context):
        return '{"name":"mock_search","arguments":{"query":"loop"}}'
    out = run_tool_loop(never_answers, reg, "go", max_steps=3)
    assert out["stopped"] == "max_steps" and out["answer"] is None
    assert len(out["trace"]) == 3
```

## Common mistakes

- **Not appending the observation to context.** If the tool result is not fed back, the model on the next turn has no idea what happened and cannot use the result. The append is the whole point.
- **No step budget.** A policy that never answers loops forever. Always cap with `max_steps`.
- **Letting a tool exception escape the loop.** One crashing tool kills the whole run. Route every failure through the registry's try/except into an error observation.
- **Detecting "done" by looking for keywords** instead of "no tool call parsed." The clean signal is `parse_tool_call(output) is None`.
- **Silently retrying failed calls in code.** Better to feed the error to the model and let it decide — blind retries repeat the same mistake.

## Exercise

Extend the scripted policy so the loop makes **two** calls: first `mock_search("chinchilla")`, then answers using the returned string. Assert `len(trace) == 2`... or is it 1? Work out the expected trace length first.

*Hint:* the policy branches on `context[-1]["role"]` and on what the last observation contains.

*Stronger hint:* after the search observation, the policy should return a prose answer (no JSON), which ends the loop.

<details><summary>Solution</summary>

```python
def policy(context):
    last = context[-1]
    if last["role"] == "user":
        return '{"name":"mock_search","arguments":{"query":"chinchilla"}}'
    # last is the tool observation; answer in prose to end the loop
    return f"Here is what I found: {last['content'].removeprefix('OBSERVATION: ')}"

out = run_tool_loop(policy, reg, "tell me about chinchilla", max_steps=5)
assert len(out["trace"]) == 1          # ONE tool call, then a prose answer
assert out["stopped"] == "final_answer"
assert "20 tokens per parameter" in out["answer"]
```

The trace length is **1**: there is exactly one tool call (the search). The second model turn is a prose answer, which produces no trace entry and ends the loop. A common off-by-one is to expect 2 — remember the trace counts *tool calls*, not model turns. To get `len(trace) == 2` you would need the policy to emit a *second* JSON call before answering.

</details>

## Debugging exercise

This loop hangs / always stops at `max_steps` and never answers. Why?

```python
def policy(context):
    return f'{{"name": "calculator", "arguments": {{"expression": "2+2"}}}}'

out = run_tool_loop(policy, reg, "what is 2+2?", max_steps=8)
```

<details><summary>Answer</summary>

The policy returns a calculator **call on every turn**, unconditionally — it never branches to a prose answer. So `parse_tool_call` always finds a call, the loop always executes and appends, and it only stops when it exhausts `max_steps` (returning `answer=None`, `stopped="max_steps"`, `len(trace)==8`). **Fix:** the policy must inspect the context (e.g. `if context[-1]["role"] == "tool": return "The answer is 4."`) so that after seeing the observation it produces a prose answer and ends the loop. This is exactly why a step budget is mandatory: it turns an infinite loop into a detectable failure.

</details>

## Research connection

<div class="callout paper"><p>The observation/action loop is the substrate for <strong>ReAct</strong> (Yao et al., 2022), which interleaves reasoning traces ("thought") with actions and observations so the model plans and acts in the same stream — the direct basis for the agents in <a href="../module-19/lesson-01.md">Module 19</a>. See ReAct (#29) and SWE-agent (#30) in the <a href="../../papers/index.md">paper curriculum</a>. The tool-training side — teaching the model which calls to emit — is <strong>Toolformer</strong> and <strong>Gorilla</strong> from <a href="lesson-01.md">18.1</a>.</p></div>

## Check yourself

<details><summary>What exactly is an "observation" in this loop, and where does it come from?</summary>

An observation is the result of executing a tool, formatted as a `tool` message and appended to the context. It comes from `registry.call(...)` — either `OBSERVATION: <result>` on success or `OBSERVATION: ERROR (...): ...` on failure. It is how the model's next turn learns what the tool returned.

</details>

<details><summary>How does the loop decide the model is finished?</summary>

By parsing the model's output with `parse_tool_call`. If it returns a call dict, the output is an action to execute. If it returns `None` (the output is prose with no JSON call), the loop treats that text as the final answer and returns with `stopped="final_answer"`.

</details>

<details><summary>A tool raises an exception mid-run. Trace what happens to the loop and to the trace.</summary>

`registry.call` catches the exception and returns `{"ok": False, "error": ..., "error_type": ...}`. The loop appends an `OBSERVATION: ERROR ...` tool message to the context and records the failed result in the trace. The loop does **not** crash; it continues to the next turn, where the model can read the error and correct. The trace entry preserves the failure for auditing.

</details>

<details><summary>In the two-turn calculator run, why is <code>len(trace) == 1</code> when the model took two turns?</summary>

`trace` counts *tool calls*, not model turns. Turn 0 emitted a calculator call (one trace entry); turn 1 emitted a prose answer (no call, no trace entry, loop ends). Two model turns, one tool call → trace length 1.

</details>

<details><summary>Why can the entire loop be tested with no model and no network?</summary>

The loop's behavior depends only on the text the policy returns and the deterministic tools. By passing a scripted `policy` and using offline tools (a safe calculator, a canned `mock_search`), every run is reproducible. Swapping in a real model changes only *what text is produced*, not *how the loop executes* — so the driver's correctness is independent of the model.

</details>

## Next

You now have the complete tool-use core: describe tools with schemas, let the model emit structured calls, parse and execute them safely, and drive the whole thing in an error-tolerant observation/action loop with a step budget. Module 19 replaces the scripted policy with a real reasoning model and adds planning, memory, and richer tools to build a ReAct-style agent and a mini SWE-agent.

Continue to [19.1 · ReAct → a mini SWE-agent](../module-19/lesson-01.md).
