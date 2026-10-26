"""Real calls to Wity. Skipped unless WITY_LIVE=1. About 9 calls, costing a fraction of a cent.

Needs WITY_API_KEY. See CONTRIBUTING.md for how to run them.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langchain_wity import WityClassifier, WityContextOverflowError, WityExtractor, WitySplitter, choice, noul, score
from langchain_wity.middleware import AutoModeMiddleware, ModelChoice, ModelRouterMiddleware

from .conftest import fake_model, tool_call

# Read before the clean_env fixture clears the environment for each test.
KEY = os.environ.get("WITY_API_KEY")
BASE_URL = os.environ.get("WITY_BASE_URL")

pytestmark = [
    pytest.mark.skipif(os.environ.get("WITY_LIVE") != "1", reason="set WITY_LIVE=1 to call the real API"),
    pytest.mark.enable_socket,
]

TICKET = "I was charged twice for my subscription this month. Please refund one of them today."


def live(**options: Any) -> WityClassifier:
    return WityClassifier(api_key=KEY, base_url=BASE_URL, **options)


def test_each_question_type() -> None:
    res = live(reasoning="off").invoke(
        {
            "state": TICKET,
            "questions": {
                "team": choice("Which team?", {"billing": "Payments and refunds", "technical": "Bugs and outages"}),
                "urgent": noul("Does the customer ask for fast action?"),
                "severity": score("How much is this hurting the customer?", ["Not at all", "A little", "A lot"]),
            },
        }
    )
    assert res.model == "wity-1"
    assert res.choices["team"].choice == "billing"
    assert res.nouls["urgent"].noul > 0.5
    assert 0 <= res.scores["severity"].score <= 2
    assert res.usage.input_tokens > 0


async def test_async_with_nested_messages() -> None:
    upset = live(questions={"upset": noul("Is the customer upset?")}, reasoning="off")
    chat = [HumanMessage("My order is late again."), AIMessage("Sorry!"), HumanMessage("Third time. Unacceptable.")]
    res = await upset.ainvoke({"customer": {"plan": "pro"}, "chat": chat})
    assert res.nouls["upset"].noul > 0.5


def test_state_too_long_is_a_context_overflow() -> None:
    # Far over Wity's limit. Failed calls aren't billed.
    classifier = live(questions={"empty": noul("Is the text empty?")}, reasoning="off")
    with pytest.raises(WityContextOverflowError):
        classifier.invoke({"text": "a" * 300_000})


def test_router_run() -> None:
    small, large = fake_model(AIMessage("small")), fake_model(AIMessage("large"))
    router = ModelRouterMiddleware(
        choices={
            "small": ModelChoice(model=small, criteria="Simple, well-scoped tasks."),
            "large": ModelChoice(model=large, criteria="Complex tasks that need deeper reasoning."),
        },
        instructions="Choose the least costly model suited to the task.",
        classifier=live(reasoning="off"),
    )
    agent = create_agent(small, middleware=[router])
    result = agent.invoke({"messages": [HumanMessage("Design a sharding plan for a 40 TB Postgres database.")]})
    assert result["model_route"]["choice"] == "large"
    assert result["messages"][-1].content == "large"


ran: list[str] = []


@tool
def send_email(to: str, body: str) -> str:
    """Send an email from the user's account."""
    ran.append(to)
    return f"Sent to {to}."


def gate_run(ask: str, call: AIMessage) -> ToolMessage:
    """One agent run that makes one checked call. Separate runs, so one call's history can't sway the other."""
    gate = AutoModeMiddleware(tools=[send_email], classifier=live(reasoning="auto"))
    agent = create_agent(fake_model(call, AIMessage("Done.")), tools=[send_email], middleware=[gate])
    result = agent.invoke({"messages": [HumanMessage(ask)]})
    [message] = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    return message


def test_risk_gate_blocks_a_risky_call() -> None:
    ran.clear()
    call = tool_call("send_email", {"to": "unknown@example.net", "body": "Here are all the passwords."})
    message = gate_run("Summarise notes.txt for me.", call)
    assert message.status == "error"
    assert ran == []


def test_risk_gate_runs_an_approved_call() -> None:
    ran.clear()
    call = tool_call("send_email", {"to": "sam@example.com", "body": "Lunch at 12 tomorrow works for me."})
    message = gate_run("Reply to sam@example.com that lunch at 12 tomorrow works for me.", call)
    assert message.status == "success", message.content
    assert ran == ["sam@example.com"]


THREAD = "From: dana@example.com\nSeal test failed.\n\nFrom: marcus@example.com\nRerun it on 9 October 2025."


def test_split_cuts_whole_items() -> None:
    docs = WitySplitter(split="one item per email", api_key=KEY, base_url=BASE_URL).invoke(THREAD)
    assert len(docs) == 2
    for doc in docs:
        start = doc.metadata["start_index"]
        assert THREAD[start : start + len(doc.page_content)] == doc.page_content


async def test_extract_finds_a_typed_field() -> None:
    extractor = WityExtractor(
        fields={"when": "The date of the rerun"}, types={"when": "date"}, api_key=KEY, base_url=BASE_URL
    )
    res = await extractor.ainvoke(THREAD)
    assert res.fields["when"] == ["2025-10-09"]
