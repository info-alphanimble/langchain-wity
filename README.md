# langchain-wity

Use [Wity](https://wity.alphanimble.com), the typed-decision API, in LangChain. Ask Wity questions with fixed options and get a probability for every option, in a pipeline or inside an agent.

It has three parts:

- `WityClassifier`: a LangChain Runnable that asks Wity typed questions.
- `ModelRouterMiddleware`: Wity picks which LLM handles each agent run.
- `AutoModeMiddleware`: Wity checks risky tool calls before they run.

## Install

```sh
pip install langchain-wity
```

Or with uv: `uv add langchain-wity`. Needs Python 3.10 or newer. It's built on the [`wity-sdk`](https://pypi.org/project/wity-sdk/) package.

## Quick start

Set your key in the environment:

```sh
export WITY_API_KEY="wity_..."
```

Then ask questions:

```python
from langchain_wity import WityClassifier, choice, noul, score

classifier = WityClassifier()

res = classifier.invoke({
    "state": "I was charged twice for my subscription this month. Please refund one of them today.",
    "questions": {
        "team": choice("Which team should handle this ticket?", {
            "billing": "Payments, charges and refunds",
            "technical": "Bugs, errors and outages",
            "other": "Anything else",
        }),
        "urgent": noul(
            "Does the customer ask for something to happen soon?",
            yes="They ask for fast action",
            no="No time pressure",
        ),
        "severity": score("How much is this hurting the customer?", ["Not at all", "A little", "A lot"]),
    },
})

res.choices["team"].choice         # "billing"
res.choices["team"].probabilities  # {"billing": 0.99993..., "technical": 0.00003..., "other": 0.00003...}
res.nouls["urgent"].noul           # 0.99997... = probability of yes
res.scores["severity"].score       # 1.96507... = expected level, from 0 (Not at all) to 2 (A lot)
```

The answer is the `wity-sdk` response, unchanged. Answers come back under the names you gave the questions, in `res.choices`, `res.nouls` and `res.scores`. The [SDK README](https://pypi.org/project/wity-sdk/) has every field.

Probabilities sum to 1, but they aren't calibrated. Treat them as a ranking of how sure Wity is, not as exact odds.

Keep one classifier around and reuse it. It keeps its connections open, so later calls are faster. It works from any thread and any event loop. Call `close()` (or `await aclose()` in async code) when you're done with it.

## Questions in the constructor

Give the classifier its questions when you create it. Then each call takes only the state, so it fits in a pipeline after any step that outputs text:

```python
from langchain_core.runnables import RunnableLambda
from langchain_wity import WityClassifier, choice

team = WityClassifier(
    questions={
        "team": choice("Which team should handle this ticket?", {
            "billing": "Payments, charges and refunds",
            "technical": "Bugs, errors and outages",
        })
    },
    reasoning="off",
)

def send(res):
    answer = res.choices["team"]
    if answer.confidence < 0.7:
        return "Sent to a person: Wity isn't sure."
    return f"Sent to {answer.choice}."

pipeline = RunnableLambda(lambda ticket: ticket.strip()) | team | send

pipeline.invoke("  The app crashes when I upload a photo.  ")  # "Sent to technical."
pipeline.batch(["Please refund my last invoice.", "Login page shows error 500."])
# ["Sent to billing.", "Sent to technical."]
```

With questions in the constructor, the whole input is the state, even a dict that has a `"questions"` key. Without them, each call must be `{"state": ..., "questions": {...}}`.

## Messages as state

State can be text, a dict, a list, a Pydantic model, a dataclass, a LangChain message or a list of messages. Models and dataclasses become dicts, and dates, UUIDs and decimals become text. Messages are sent as `{role, content}`, and they can also sit inside dicts and lists:

```python
from langchain_core.messages import AIMessage, HumanMessage
from langchain_wity import WityClassifier, noul

upset = WityClassifier(questions={"upset": noul("Is the customer upset?")})

chat = [
    HumanMessage("Hi, my order hasn't arrived."),
    AIMessage("Sorry about that! It should arrive tomorrow."),
    HumanMessage("You said that last week. This is the third time."),
]

upset.invoke(chat).nouls["upset"].noul  # 0.99991...

res = await upset.ainvoke({"customer": {"plan": "pro", "orders": 14}, "chat": chat})
```

Images, audio and files inside messages are left out: Wity gets only their type, like `{"type": "image_url"}`. Sent as text, even a small image would be thousands of tokens Wity can't read.

`invoke`, `ainvoke`, `batch` and `abatch` all work.

## Model router

`ModelRouterMiddleware` sends easy tasks to a cheap model and hard ones to a strong model. At the start of each agent run, Wity reads the latest user message and picks a route. It also sees up to 4 earlier user messages, so a follow-up like "now write the migration for it" keeps its context.

This example uses OpenAI models, so it needs `langchain-openai` and an OpenAI key. Any LangChain chat model works.

```python
from langchain.agents import create_agent
from langchain_wity.middleware import ModelChoice, ModelRouterMiddleware

router = ModelRouterMiddleware(
    choices={
        "small": ModelChoice(model="openai:gpt-5-mini", criteria="Simple, well-scoped tasks."),
        "large": ModelChoice(model="openai:gpt-5", criteria="Complex tasks that need deeper reasoning."),
    },
    instructions="Choose the least costly model suited to the task.",
)
agent = create_agent("openai:gpt-5-mini", middleware=[router])

result = agent.invoke({"messages": [{"role": "user", "content": "Design a sharding plan for a 40 TB Postgres database."}]})
result["model_route"]["choice"]         # "large"
result["model_route"]["probabilities"]  # {"small": 0.0303..., "large": 0.9696...}
```

- You need at least two routes.
- `model` is a LangChain chat model, or a model string for `init_chat_model`.
- Every model call in the run goes to the picked model.
- Wity's full answer is saved in the agent state as `model_route`, so you can see why a model was picked. It's a plain dict, so checkpointers can save it.
- If Wity fails, the error stops the run, including when the message is longer than Wity accepts. The router never quietly picks a model for you.

## Tool-risk gate

`AutoModeMiddleware` checks the tools you list before they run. Wity sees the proposed call, the tool's description, every message the user wrote and the 30 most recent messages. It answers one yes/no question: should this call be stopped? It stops calls the user didn't ask for or said not to do yet, calls that do more than the user asked, and calls that expose private data. A call that does exactly what the user asked runs, even a lasting one like deleting the file they named.

Here `read_file`, `send_email` and `delete_file` stand for your own tools:

```python
from langchain.agents import create_agent
from langchain_wity.middleware import AutoModeMiddleware

gate = AutoModeMiddleware(tools=[send_email, delete_file])
agent = create_agent("openai:gpt-5-mini", tools=[read_file, send_email, delete_file], middleware=[gate])
```

- When the probability of risky is at or above `threshold` (0.5 by default), the tool doesn't run. The agent gets an error message instead, such as: "The tool call `send_email` was blocked because Wity rated it risky (probability 1.00). The tool was not run."
- Tools you don't list run without a check. If a name you list matches none of the agent's tools, for example because of a typo, you get a warning on the first model call.
- If Wity fails, the error is raised and the tool doesn't run. That includes a state longer than Wity accepts, which raises `WityContextOverflowError`.
- Wity treats everything in the state as data, not instructions. Only the user's own messages can approve an action. This keeps a web page or file from approving a call by pretending to give orders.
- The gate blocks; it doesn't ask a person. You can use it next to LangChain's `HumanInTheLoopMiddleware`, but the gate can't see a person's approval there. It still checks a call a person approved, and it can block it. To let a person have the last word on a tool, list that tool in `HumanInTheLoopMiddleware` only.

### Tuning the threshold

Since the probabilities aren't calibrated, try the gate on some of your own tool calls, safe ones and risky ones. Then set `threshold` between where they land:

```python
gate = AutoModeMiddleware(tools=[send_email], threshold=0.8)
```

You can also change the question and what risky and safe mean:

```python
from langchain_wity import NoulCriteria

gate = AutoModeMiddleware(
    tools=[delete_file],
    criteria=NoulCriteria(
        true="The call writes, deletes, publishes or changes access.",
        false="The call only reads public or user-provided data.",
    ),
)
```

## Reasoning and speed

Wity answers most questions in one fast pass. With `reasoning="auto"` it thinks first when a question needs it, for example when the top options are close. That usually takes 1–4 seconds.

| Setting | What it does |
|---|---|
| `"off"` | Never thinks. Fastest. |
| `"auto"` | Thinks only when the question needs it. The API default. |
| `"always"` | Always thinks. |

- `WityClassifier` uses the API default, `"auto"`. Set `reasoning=` to change it.
- The router uses `"auto"` too. Clear tasks get one fast pass; Wity thinks only when the routes are close.
- The risk gate uses `"auto"`, so close calls get a closer look.

To change a middleware's setting, pass your own classifier. It must not have questions of its own:

```python
router = ModelRouterMiddleware(choices=..., instructions=..., classifier=WityClassifier(reasoning="off"))
gate = AutoModeMiddleware(tools=[send_email], classifier=WityClassifier(reasoning="always"))
```

`max_latency_ms=` (a whole number from 200 to 120000) caps the time per request. Thinking stops early to meet it.

## Options

| Option | Default | What it does |
|---|---|---|
| `questions` | none | Fixed questions. Then each call takes only the state. |
| `reasoning` | API default (`"auto"`) | `"off"`, `"auto"` or `"always"` |
| `max_latency_ms` | none | Time limit per request, a whole number from 200 to 120000 ms |
| `api_key` | `WITY_API_KEY` | Your Wity key |
| `base_url` | `WITY_BASE_URL`, then the SDK's default | The API address |
| `timeout` | 60 | Seconds per attempt |
| `max_retries` | 2 | Retries on 408, 429, 5xx and network failures |
| `http_client`, `http_async_client` | none | Your own `httpx` clients, for example for a proxy |

## Errors, retries and tracing

**Errors.** Each error is both a `wity-sdk` error and the matching LangChain error. Catch whichever suits your code:

| Error | Also a `wity` error | Also a LangChain error |
|---|---|---|
| `WityBadRequestError` (400) | `BadRequestError` | `ModelInvalidRequestError` |
| `WityContextOverflowError` (400, state too long) | `BadRequestError` | `ContextOverflowError` |
| `WityAuthenticationError` (401) | `AuthenticationError` | `ModelAuthenticationError` |
| `WityInsufficientBalanceError` (402) | `InsufficientBalanceError` | `ModelAPIError` |
| `WityRateLimitError` (429) | `RateLimitError` | `ModelRateLimitError` |
| `WityInternalServerError` (5xx) | `InternalServerError` | `ModelAPIError` |
| `WityConnectionError` | `APIConnectionError` | `ModelConnectionError` |
| `WityTimeoutError` | `APITimeoutError` | `ModelTimeoutError` |

```python
from langchain_wity import WityRateLimitError

try:
    res = classifier.invoke(ticket)
except WityRateLimitError as err:
    print(err.status, err.headers.get("retry-after"))
```

Error messages and logs never include your state or your key.

**Retries.** The classifier already retries failed calls (`max_retries=2`). If you add LangChain's `.with_retry()`, set `max_retries=0` so calls aren't retried twice over:

```python
classifier = WityClassifier(questions=..., max_retries=0).with_retry(stop_after_attempt=3)
```

**Tracing.** Each Wity call shows up in LangSmith as a model call, with provider `wity`, model `wity-1` and the input token count. Your key never appears in traces, `repr()` or saved config. The middlewares keep tool arguments and messages out of their own trace step. The Wity call inside that step is traced like any other classifier call, with the state it sent.

LangChain's `get_usage_metadata_callback()` doesn't count Wity's tokens. Read them from `res.usage.input_tokens`, or in LangSmith.

**Pickling.** Classifiers and errors can be pickled, for example to send them to a process pool. A pickled classifier holds its options, not its connections. A key read from `WITY_API_KEY` isn't pickled: the other process reads its own.

## Examples

The `examples/` folder in the repository has a script for each section above. The router and risk-gate scripts use LangChain's fake chat model, so they run with only a Wity key:

```sh
export WITY_API_KEY="wity_..."
uv run python examples/quick_start.py
```

## Contributing

See `CONTRIBUTING.md` in the repository for how to run the checks and tests.

## License

Apache-2.0
