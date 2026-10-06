"""ModelRouterMiddleware in a real LangChain agent, with fake chat models and a fake Wity."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from langchain_wity import WityContextOverflowError, WityInternalServerError, noul
from langchain_wity.middleware import ModelChoice, ModelRouterMiddleware

from .conftest import FakeWity, choice_body, fake_model, tool_call

INSTRUCTIONS = "Choose the least costly model suited to the task."
ROUTE = {"fast": 0.1, "powerful": 0.9}


def router(fake: FakeWity, fast: Any, powerful: Any, **options: Any) -> ModelRouterMiddleware:
    options.setdefault("classifier", fake.classifier(reasoning="off"))
    return ModelRouterMiddleware(
        choices={
            "fast": ModelChoice(model=fast, criteria="Simple, well-scoped tasks."),
            "powerful": ModelChoice(model=powerful, criteria="Complex tasks that need deeper reasoning."),
        },
        instructions=INSTRUCTIONS,
        **options,
    )


def test_routes_the_run_to_the_picked_model(fake: FakeWity) -> None:
    fake.json(choice_body("model_route", "powerful", ROUTE))
    fast, powerful = fake_model(AIMessage("fast answer")), fake_model(AIMessage("powerful answer"))
    agent = create_agent(fast, middleware=[router(fake, fast, powerful)])

    result = agent.invoke({"messages": [HumanMessage("Plan a database migration.")]})

    assert result["messages"][-1].content == "powerful answer"
    assert (fast.calls, powerful.calls) == (0, 1)
    answer = result["model_route"]
    assert answer == {"type": "choice", "choice": "powerful", "probabilities": ROUTE, "confidence": 0.9}


def test_sends_the_latest_user_message_and_the_routes(fake: FakeWity) -> None:
    fake.json(choice_body("model_route", "fast", {"fast": 0.8, "powerful": 0.2}))
    fast = fake_model(AIMessage("ok"))
    agent = create_agent(fast, middleware=[router(fake, fast, fake_model())])
    agent.invoke({"messages": [HumanMessage("old question"), AIMessage("old answer"), HumanMessage("What's 2 + 2?")]})

    body = fake.bodies[0]
    assert body["state"] == "What's 2 + 2?"
    assert body["reasoning"] == "off"
    question = body["questions"]["model_route"]
    assert question["type"] == "choice"
    assert question["instructions"] == INSTRUCTIONS
    assert question["criteria"] == {
        "fast": "Simple, well-scoped tasks.",
        "powerful": "Complex tasks that need deeper reasoning.",
    }


def test_every_model_call_in_the_run_uses_the_picked_model(fake: FakeWity) -> None:
    @tool
    def lookup(city: str) -> str:
        """Look up the weather in a city."""
        return "Sunny"

    fake.json(choice_body("model_route", "powerful", ROUTE))
    fast = fake_model()
    powerful = fake_model(tool_call("lookup", {"city": "Bern"}), AIMessage("It's sunny in Bern."))
    agent = create_agent(fast, tools=[lookup], middleware=[router(fake, fast, powerful)])

    result = agent.invoke({"messages": [HumanMessage("Weather in Bern?")]})

    assert result["messages"][-1].content == "It's sunny in Bern."
    assert (fast.calls, powerful.calls) == (0, 2)
    assert len(fake.requests) == 1


def test_routes_again_on_each_run(fake: FakeWity) -> None:
    fake.json(choice_body("model_route", "fast", {"fast": 0.9, "powerful": 0.1}))
    fake.json(choice_body("model_route", "powerful", ROUTE))
    fast, powerful = fake_model(AIMessage("fast")), fake_model(AIMessage("powerful"))
    agent = create_agent(fast, middleware=[router(fake, fast, powerful)])

    first = agent.invoke({"messages": [HumanMessage("Hi")]})
    second = agent.invoke({"messages": [HumanMessage("Prove this theorem.")]})

    assert (first["model_route"]["choice"], second["model_route"]["choice"]) == ("fast", "powerful")
    assert (fast.calls, powerful.calls) == (1, 1)


def test_a_wity_error_stops_the_run(fake: FakeWity) -> None:
    fake.json({"error": "model error"}, status=500)
    fast, powerful = fake_model(AIMessage("fast")), fake_model(AIMessage("powerful"))
    agent = create_agent(fast, middleware=[router(fake, fast, powerful)])

    with pytest.raises(WityInternalServerError):
        agent.invoke({"messages": [HumanMessage("Hi")]})
    assert (fast.calls, powerful.calls) == (0, 0)


def test_no_user_message_is_a_clear_error(fake: FakeWity) -> None:
    fast = fake_model(AIMessage("fast"))
    agent = create_agent(fast, middleware=[router(fake, fast, fake_model())])

    with pytest.raises(ValueError, match="needs a user message"):
        agent.invoke({"messages": [AIMessage("Hello, how can I help?")]})
    assert fake.requests == []


def test_state_too_long_for_wity_stops_the_run(fake: FakeWity) -> None:
    fake.json({"error": "state must be a possibly empty string of at most 32000 characters"}, status=400)
    fast = fake_model(AIMessage("fast"))
    agent = create_agent(fast, middleware=[router(fake, fast, fake_model())])
    with pytest.raises(WityContextOverflowError):
        agent.invoke({"messages": [HumanMessage("x" * 300_000)]})
    assert fast.calls == 0


async def test_async_run(fake: FakeWity) -> None:
    fake.json(choice_body("model_route", "powerful", ROUTE))
    fast, powerful = fake_model(AIMessage("fast")), fake_model(AIMessage("powerful"))
    agent = create_agent(fast, middleware=[router(fake, fast, powerful)])

    result = await agent.ainvoke({"messages": [HumanMessage("Hard task")]})

    assert result["messages"][-1].content == "powerful"
    assert result["model_route"]["choice"] == "powerful"
    assert (fast.calls, powerful.calls) == (0, 1)


async def test_async_wity_error_stops_the_run(fake: FakeWity) -> None:
    fake.reply(httpx.Response(500, json={"error": "model error"}))
    fast = fake_model(AIMessage("fast"))
    agent = create_agent(fast, middleware=[router(fake, fast, fake_model())])
    with pytest.raises(WityInternalServerError):
        await agent.ainvoke({"messages": [HumanMessage("Hi")]})


def test_reasoning_is_off_by_default(fake: FakeWity, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WITY_API_KEY", "wity_from_env_0000000000")
    middleware = router(fake, fake_model(), fake_model(), classifier=None)
    assert middleware.classifier.reasoning == "off"


def test_your_own_classifier_is_used(fake: FakeWity) -> None:
    fake.json(choice_body("model_route", "fast", {"fast": 0.9, "powerful": 0.1}))
    fast = fake_model(AIMessage("ok"))
    agent = create_agent(fast, middleware=[router(fake, fast, fake_model(), classifier=fake.classifier())])
    agent.invoke({"messages": [HumanMessage("Hi")]})
    assert "reasoning" not in fake.bodies[0]


def test_a_classifier_with_its_own_questions_is_refused(fake: FakeWity) -> None:
    classifier = fake.classifier(questions={"urgent": noul("Urgent?")})
    with pytest.raises(ValueError, match="without questions"):
        router(fake, fake_model(), fake_model(), classifier=classifier)


def test_no_routes_is_refused(fake: FakeWity) -> None:
    with pytest.raises(ValueError, match="at least one route"):
        ModelRouterMiddleware(choices={}, instructions=INSTRUCTIONS, classifier=fake.classifier())


def test_works_with_a_checkpointer(fake: FakeWity, recwarn: pytest.WarningsRecorder) -> None:
    fake.json(choice_body("model_route", "powerful", ROUTE))
    fake.json(choice_body("model_route", "fast", {"fast": 0.9, "powerful": 0.1}))
    fast, powerful = fake_model(AIMessage("fast")), fake_model(AIMessage("powerful"))
    agent = create_agent(fast, middleware=[router(fake, fast, powerful)], checkpointer=InMemorySaver())
    config: Any = {"configurable": {"thread_id": "t1"}}

    agent.invoke({"messages": [HumanMessage("Hard task")]}, config)
    result = agent.invoke({"messages": [HumanMessage("Easy task")]}, config)

    assert result["model_route"]["choice"] == "fast"
    assert agent.get_state(config).values["model_route"]["choice"] == "fast"
    assert (fast.calls, powerful.calls) == (1, 1)
    assert [str(w.message) for w in recwarn] == []
