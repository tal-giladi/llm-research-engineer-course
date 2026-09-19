# 18.1 · Tool schemas & function calling

<div class="prereq">
<p><strong>Prerequisites:</strong> chat templates and the assistant/user role structure from <a href="../module-14/lesson-01.md">14.1 · Instruction data & chat templates</a>; the idea of constraining a model's output to a fixed format (structured outputs / JSON mode), which we build here from first principles. Comfort reading JSON and Python dataclasses.</p>
<p><strong>You will learn:</strong> what a <strong>tool schema</strong> is (name, description, JSON-schema parameters) and why a model needs one; what <strong>function calling</strong> means — the model emitting a structured call instead of prose, which a runtime parses and executes; how to build a tool registry and a couple of deterministic, <em>safe</em> tools (a calculator, a restricted Python evaluator, a mock search); and how to generate the schemas the model is shown.</p>
<p><strong>Why this matters for ML:</strong> the aligned model you built through Modules 14–17 can answer from its weights, but it cannot do arithmetic reliably, look anything up, or run code. Those abilities do not come from more parameters — they come from letting the model <em>call external tools</em>. Function calling is the interface that makes a frozen language model useful as the brain of a system that acts. It is the foundation for agents (Module 19).</p>
</div>

## 1. The causal story: a smart model that can't multiply

By the end of Module 17 you have a model that is genuinely good at language. Ask it to summarize a paragraph, follow an instruction, or reason step by step, and it does well. But three whole categories of task defeat it, and no amount of extra training fully fixes them:

- **Arithmetic.** Ask for `4831 * 7772` and the model produces a plausible-looking number that is usually wrong. It generates digits token by token from patterns, not by actually multiplying. It has no calculator inside.
- **Fresh or private facts.** "What is today's exchange rate?" or "What is in this file?" The answer is not in the weights — it was never in the training data, or it changed after the cutoff. The model will confidently hallucinate.
- **Running code / acting on the world.** "Sort this list", "query the database", "send this request." The model can *write* the code but cannot *execute* it, and it cannot touch anything outside its own text stream.

The fix is not a bigger model. It is to give the model **tools**: a calculator, a search function, a code executor, an API client. The model stays the reasoning engine; the tools do the exact, up-to-date, or side-effecting work. This lesson builds the machinery that lets a model call a tool. The next lesson turns single calls into a loop.

<div class="callout key"><p>A language model predicts text. A tool is any external function the model can invoke to get something its weights cannot provide: an exact computation, a lookup, or an effect on the world. <strong>Function calling</strong> is the protocol by which the model asks for a tool to be run and receives the result back as text.</p></div>

## 2. Intuition: how a model "calls" a function

The model only ever emits text. It cannot literally call a Python function — it has no interpreter, no memory beyond its context window, no hands. So "function calling" is a convention layered on top of text generation, with four actors:

1. **The schema.** Before generation, we put a description of the available tools into the model's context — each tool's name, what it does, and the shape of its arguments. Now the model *knows what exists*.
2. **The model.** Instead of answering in prose, the model emits a small, structured message: "call the tool named `calculator` with `expression = "4831*7772"`." In practice this is a JSON object.
3. **The runtime (us).** Code outside the model parses that JSON, finds the real `calculator` function, runs it with those arguments, and captures the result.
4. **The result goes back.** We paste the result into the model's context as a new message, and let the model continue — now with the true answer in front of it.

Steps 2–4 are the loop of the next lesson. This lesson is about steps 1 and 2: describing tools (the schema) and producing/parsing a single call.

The programmer's analogy: the model is like a caller who knows a REST API's documentation (the schema) and writes a request (the JSON call), but has no network stack. You are the network stack — you take the request, actually hit the endpoint, and hand back the response.

## 3. The tool schema — every symbol named

A **tool schema** is the contract you show the model. It has three parts:

- **`name`** — a unique string identifier the model emits to select the tool, e.g. `"calculator"`.
- **`description`** — one line of natural language telling the model *when* to use the tool. This is not decoration; it is what the model reads to decide between, say, `calculator` and `python_eval`. A vague description causes the model to pick the wrong tool.
- **`parameters`** — a **JSON Schema** object describing the arguments: their names, types, and which are required. JSON Schema is a small standard for describing the shape of a JSON value. The subset we need looks like this:

```json
{
  "type": "object",
  "properties": {
    "expression": {
      "type": "string",
      "description": "Arithmetic expression, e.g. '2 + 3 * 4'."
    }
  },
  "required": ["expression"]
}
```

Read it literally: the arguments are an *object* (a dict) with one *property* named `expression` of type *string*, and that property is *required*. The model uses this to know it must emit `{"expression": "..."}` and not, say, `{"expr": 42}`.

<div class="callout key"><p>The schema exists because the model must know two things it cannot guess: <strong>which tools exist</strong> (their names and purposes) and <strong>what arguments each takes</strong> (the parameter shape). No schema in context → the model invents tool names and argument shapes that your runtime cannot execute.</p></div>

Why JSON specifically? Because it is unambiguous to parse and the model has seen enormous amounts of it in training, so it produces well-formed JSON reliably. The same role could be played by XML or a custom syntax; JSON won because it is the cheapest to generate and parse correctly.

## 4. Function calling: the model emits a structured call

When the model decides to use a tool, instead of prose it emits a JSON object naming the tool and its arguments. In our convention:

```json
{"name": "calculator", "arguments": {"expression": "4831 * 7772"}}
```

Two fields: `name` (which tool) and `arguments` (a dict matching that tool's `parameters` schema). That is the entire "function call." It is just text the model generated — the intelligence is that it generated *this* text, in *this* shape, at the right moment, because the schema told it the shape and the description told it the moment.

**Structured output / JSON mode.** How do we make sure the model emits valid JSON and not "Sure! Let me calculate that for you..."? Two mechanisms, in increasing strength:

- **Prompting.** Tell the model in its system prompt: "To use a tool, respond with only a JSON object of the form `{...}`." Cheap, usually works, occasionally the model adds stray prose.
- **Constrained decoding (JSON mode).** At each generation step, the sampler is only *allowed* to pick tokens that keep the output valid against a grammar or JSON Schema. Invalid tokens have their probability forced to zero before sampling. This *guarantees* well-formed output. This is what "JSON mode" and "structured outputs" in production APIs do under the hood.

<div class="callout warn"><p>Constrained decoding guarantees the output is <em>syntactically</em> valid JSON matching the schema. It does <strong>not</strong> guarantee the arguments are <em>correct</em> — the model can still put a wrong expression in a well-formed call. Validity and correctness are different problems; the loop in 18.2 handles correctness via feedback.</p></div>

In this course we do not modify the sampler; instead we parse robustly *after* generation (Section 8), which is enough to build and test the whole mechanism deterministically. We flag where a production system would swap in constrained decoding.

## 5. The registry and the `Tool` object — from-scratch PyTorch-style code

We need one object that (a) holds each tool's callable plus its schema, and (b) can execute a tool safely and hand back the schemas. That is the **tool registry**. It lives at `code/src/llmre/tools/registry.py`.

A single tool is a small dataclass:

```python
from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass
class Tool:
    name: str                       # what the model emits to select this tool
    description: str                # one-line "when to use me" for the model
    func: Callable[..., Any]        # the real Python callable; called func(**arguments)
    parameters: dict[str, Any] = field(default_factory=dict)  # JSON-schema dict

    def schema(self) -> dict[str, Any]:
        return {"name": self.name,
                "description": self.description,
                "parameters": self.parameters}
```

`Tool.schema()` produces exactly the JSON-serializable object we show the model. The registry stores tools by name and adds three operations: `register`, `call`, and `schemas`.

```python
class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, func, *, name=None, description=None, parameters=None):
        tool = Tool(name=name or func.__name__,
                    description=description or (func.__doc__ or "").strip().splitlines()[0],
                    func=func, parameters=parameters or {})
        self._tools[tool.name] = tool
        return func

    def call(self, name: str, **kwargs) -> dict:
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "error": f"unknown tool {name!r}", "error_type": "KeyError"}
        try:
            return {"ok": True, "result": tool.func(**kwargs)}
        except Exception as exc:                       # never crash the loop
            return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}

    def schemas(self) -> list[dict]:
        return [t.schema() for t in self._tools.values()]
```

The one design decision that matters most: **`call` never raises.** A tool that blows up returns a structured `{"ok": False, "error": ...}` dict instead of propagating an exception. This is what lets the observation/action loop feed the error back to the model as an observation rather than crashing (18.2). The full implementation adds `get`, `__contains__`, sorted `names()`, and decorator support — see `code/src/llmre/tools/registry.py`.

## 6. Three deterministic tools — and honest safety

The built-ins live at `code/src/llmre/tools/builtins.py`. They are deterministic and offline so the whole system is reproducible and testable. The interesting engineering is **safety**: a tool runs real code with arguments the model chose, and a model can be wrong or adversarially steered.

### 6.1 A *safe* calculator

The naive calculator is `eval(expression)`. Never do this. `eval` on model-chosen text is a remote code execution hole: the "expression" `__import__('os').system('rm -rf /')` is valid Python and `eval` will run it. Instead we parse the expression into an **abstract syntax tree** and walk it, allowing only number literals and arithmetic operators — an *allow-list*, not a block-list:

```python
import ast, operator

_BIN_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
            ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod, ast.Pow: operator.pow}

def _eval_node(node):
    if isinstance(node, ast.Expression):        return _eval_node(node.body)
    if isinstance(node, ast.Constant):          # a number literal
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numbers allowed")
        return node.value
    if isinstance(node, ast.BinOp):             # left <op> right
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    # ... unary +/- handled similarly ...
    raise ValueError(f"disallowed element: {type(node).__name__}")

def calculator(expression: str) -> float:
    return _eval_node(ast.parse(expression, mode="eval"))
```

Any node that is not a number, an allowed operator, or parentheses — a name, a function call, an attribute access, an import — hits the final `raise`. The malicious string never runs; it raises `ValueError` because `__import__(...)` is an `ast.Call` node, which is not in the allow-list.

<div class="callout warn"><p><strong>Allow-list, never block-list.</strong> "Reject strings containing <code>import</code>" is a block-list and will always miss a case (<code>getattr</code>, dunder tricks, unicode). Enumerating exactly what is <em>permitted</em> — numbers and five operators — is the only approach that is safe by construction.</p></div>

### 6.2 A restricted `python_eval`, honestly labeled

`python_eval` allows a bit more (comparisons, boolean logic, literal lists/dicts) for a single expression, vets the AST the same way, and runs the compiled expression with `{"__builtins__": {}}` as globals so there is no `open`, `__import__`, or `eval` available. But we label its limits honestly in the docstring:

<div class="callout warn"><p>Restricting <code>eval</code> inside the real CPython process is <strong>hardening, not a sandbox</strong>. The allow-list blocks the obvious escapes, but for genuinely untrusted code you must isolate execution at the OS level — a separate process, a container, seccomp, CPU/memory limits. Do not point a restricted-eval tool at adversarial input and call it safe.</p></div>

This honesty is the point: a tool's schema advertises what it does; its documentation must state what it does *not* protect against. Frontier tool-use systems run code tools in real sandboxes (containers, microVMs) for exactly this reason.

### 6.3 A mock search

`mock_search(query)` matches the query against a tiny canned dict and returns a fixed string, or "No results found" — no network, fully deterministic, so tests never flake. A real search tool would call an API here; the *interface* (string in, string out) is identical, which is what lets us develop and test the loop offline.

## 7. Numerical example: generating a schema and "emitting" a call

Let us do the whole thing by hand with tiny pieces. Register the built-ins and print the calculator's schema. Running this against `code/src/llmre/tools/` produces, verbatim:

```python
from llmre.tools.registry import ToolRegistry
from llmre.tools.builtins import register_builtins
import json

reg = register_builtins(ToolRegistry())
print(json.dumps(reg.schemas()[0], indent=2))
```

```json
{
  "name": "calculator",
  "description": "Evaluate an arithmetic expression and return the number.",
  "parameters": {
    "type": "object",
    "properties": {
      "expression": {
        "type": "string",
        "description": "Arithmetic expression, e.g. '2 + 3 * 4'."
      }
    },
    "required": ["expression"]
  }
}
```

That JSON is what goes into the model's context. Now imagine the model, having seen it, "emits" a call for the prompt *"What is 2+3*4?"*:

```json
{"name": "calculator", "arguments": {"expression": "2+3*4"}}
```

We parse it (Section 8), then execute it through the registry:

```python
reg.call("calculator", expression="2+3*4")   # -> {'ok': True, 'result': 14}
```

The result is `14`, not `18` — the calculator honors operator precedence (`3*4` first). This is the value the model could not reliably produce itself; the tool did. And a malicious argument is caught, not run:

```python
reg.call("calculator", expression="__import__('os').system('x')")
# -> {'ok': False, 'error': 'disallowed expression element: Call', 'error_type': 'ValueError'}
```

Both outputs above are the exact strings the code prints.

## 8. Parsing the call out of model text

The model's raw output is text, and it may wrap the JSON in a fenced code block or drop it bare in the middle of prose. `code/src/llmre/tools/parser.py` handles both. `parse_tool_call(text)` returns `{"name", "arguments"}` for the first valid call it finds, or `None` if the text is a plain final answer.

```python
from llmre.tools.parser import parse_tool_call

parse_tool_call('ok\n```json\n{"name":"calculator","arguments":{"expression":"2+3*4"}}\n```')
# -> {'name': 'calculator', 'arguments': {'expression': '2+3*4'}}

parse_tool_call('let me {"name":"mock_search","arguments":{"query":"chinchilla"}} now')
# -> {'name': 'mock_search', 'arguments': {'query': 'chinchilla'}}

parse_tool_call('The answer is 14.')
# -> None
```

It tries fenced ```` ```json ```` blocks first (the strongest signal of intent), then falls back to brace-matching any balanced `{...}` object embedded in prose — carefully ignoring braces inside quoted strings. Returning `None` for prose is what the loop uses to detect "the model is done" versus "the model wants a tool."

<div class="callout pt"><p>In a production API, the model provider does this parsing for you and hands back a typed <code>tool_calls</code> field. We parse by hand so you see exactly what that layer does — and so the whole pipeline runs offline and deterministically for testing.</p></div>

## 9. Tensor shapes / where this sits relative to the model

There are no tensors in this lesson, and that is worth stating plainly. Function calling is **orchestration around a frozen model**, not a change to the model's forward pass. The model still maps token ids `(B, T)` to logits `(B, T, V)` exactly as in Module 6; the only difference is *what text is in the context* (schemas + observations) and *what we do with the generated text* (parse and execute instead of just displaying). The registry, parser, and tools are plain Python. This separation is deliberate: tool use is a systems problem layered on top of the language model, which is why a single trained model can be given new tools without retraining.

There is a second, deeper way to add tools — *training* the model to emit calls, as **Toolformer** does by inserting API calls into its training data (Section 11). That does touch the weights. But the runtime side — schema, parse, execute — is identical either way.

## Unit test

From `code/tests/test_tools.py` (run with `py -m pytest -q` from `code/`):

```python
def test_schemas_advertise_the_tools():
    reg = register_builtins(ToolRegistry())
    names = {s["name"] for s in reg.schemas()}
    assert {"calculator", "python_eval", "mock_search"} <= names

def test_calculator_respects_precedence():
    assert calculator("2+3*4") == 14

def test_calculator_rejects_malicious_expression():
    with pytest.raises(ValueError):
        calculator("__import__('os').system('echo pwned')")

def test_parser_extracts_from_fenced_block():
    text = '```json\n{"name": "calculator", "arguments": {"expression": "2+2"}}\n```'
    assert parse_tool_call(text) == {"name": "calculator", "arguments": {"expression": "2+2"}}
```

## Common mistakes

- **Using `eval()` for the calculator.** Instant RCE. Always parse to an AST and allow-list nodes.
- **A block-list instead of an allow-list.** Filtering out "dangerous" substrings always misses a case. Enumerate what is permitted.
- **A vague tool `description`.** The model chooses tools by their descriptions. "Does math" vs "Evaluate an arithmetic expression and return the number" is the difference between right and wrong tool selection.
- **Letting `call` raise.** If a crashing tool propagates its exception, the whole agent loop dies. Catch it and return a structured error the model can read.
- **Assuming valid JSON means correct arguments.** JSON mode guarantees shape, not truth. The model can still pass a wrong expression.

## Exercise

Add a `word_count(text: str) -> int` tool: register it with a correct schema, and confirm the model could call it via `parse_tool_call` + `registry.call`.

*Hint:* the `parameters` schema needs one required string property named `text`.

*Stronger hint:* mirror the `mock_search` registration block in `register_builtins`; the callable is just `lambda text: len(text.split())` (or a named function with a docstring).

<details><summary>Solution</summary>

```python
def word_count(text: str) -> int:
    """Count whitespace-separated words in text."""
    return len(text.split())

reg.register(
    word_count,
    name="word_count",
    description="Count the words in a piece of text.",
    parameters={
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to count words in."}},
        "required": ["text"],
    },
)

call = parse_tool_call('{"name":"word_count","arguments":{"text":"a b c"}}')
print(reg.call(call["name"], **call["arguments"]))   # {'ok': True, 'result': 3}
```

The tool now appears in `reg.schemas()`, so a model shown the schemas could select it. Note the schema's `properties` names must match the function's parameter names exactly, or `call(**arguments)` raises a `TypeError` — which, thanks to `call`'s try/except, comes back as a structured error rather than a crash.

</details>

## Debugging exercise

This registration looks fine but the model's calls always fail with an error observation. Why?

```python
def temperature(city: str) -> str:
    return f"It is 20C in {city}."

reg.register(temperature, name="temperature",
             description="Get the temperature in a city.",
             parameters={"type": "object",
                         "properties": {"location": {"type": "string"}},
                         "required": ["location"]})
```

<details><summary>Answer</summary>

The schema advertises a parameter named **`location`**, but the function's parameter is named **`city`**. The model, reading the schema, emits `{"arguments": {"location": "Paris"}}`. The registry then calls `temperature(location="Paris")`, which raises `TypeError: temperature() got an unexpected keyword argument 'location'`. Because `call` catches exceptions, you get `{"ok": False, "error_type": "TypeError", ...}` on every call instead of a crash — but the tool never works. **Fix:** the schema property names must exactly match the callable's parameter names (`"city"`).

</details>

## Research connection

<div class="callout paper"><p><strong>Toolformer</strong> (Schick et al., 2023) trains a model to <em>teach itself</em> when and how to call APIs (calculator, search, translation) by inserting candidate API calls into training text and keeping only those that reduce the loss on the following tokens — so tool use is learned self-supervised, not hand-annotated. <strong>Gorilla</strong> (Patil et al., 2023) fine-tunes a model to emit correct API calls against a large, changing catalog of real ML APIs, showing that a schema-aware model can pick the right call from thousands of options and adapt as APIs change. Both are in the <a href="../../papers/index.md">paper curriculum</a> (#27, #28), to read after this module.</p></div>

## Check yourself

<details><summary>Why does the model need a tool schema at all — can't it just call functions it knows?</summary>

The model only emits text and has no access to your process's functions. It cannot know which tools you have wired up, their exact names, or their argument shapes unless you tell it — that is what the schema in its context does. Without the schema it will invent tool names and argument keys that your runtime cannot execute.

</details>

<details><summary>The model emits <code>{"name": "calculator", "arguments": {"expression": "2+3*4"}}</code>. Trace what happens and what the final result is.</summary>

`parse_tool_call` extracts `{"name": "calculator", "arguments": {"expression": "2+3*4"}}`. The runtime calls `reg.call("calculator", expression="2+3*4")`, which runs the safe AST evaluator: `3*4 = 12`, then `2+12 = 14`. The result dict is `{"ok": True, "result": 14}`. The model itself never computed anything — it only asked.

</details>

<details><summary>Why is <code>calculator("__import__('os').system('x')")</code> safe here but <code>eval("__import__('os').system('x')")</code> catastrophic?</summary>

`eval` executes the string as real Python, so `__import__('os').system('x')` runs a shell command. Our `calculator` never executes the string — it parses it to an AST and walks it against an allow-list of numbers and arithmetic operators. `__import__(...)` is an `ast.Call` node, which is not allowed, so it raises `ValueError` before anything runs.

</details>

<details><summary>What is the difference between JSON mode guaranteeing "valid output" and the call being "correct"?</summary>

Constrained decoding (JSON mode) forces the generated tokens to form JSON matching the schema — the output is syntactically valid and has the right fields. It says nothing about whether the <em>values</em> are right: the model can emit a well-formed call with a wrong expression or the wrong tool. Correctness is fixed by executing and feeding results/errors back (18.2), not by the decoder.

</details>

## Next

You can now describe tools to a model, let it emit a structured call, and parse and execute that call safely. But one call is rarely enough: real tasks need the model to see the result, decide what to do next, maybe call another tool, and recover from errors. That repetition is the **observation/action loop**.

Continue to [18.2 · The observation/action loop](lesson-02.md).
