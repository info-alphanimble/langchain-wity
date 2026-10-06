"""Shared test helpers: a fake Wity server on a fake network."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from langchain_wity import WityClassifier

TEST_KEY = "wity_test_secret_1234567890"
"""A made-up key. Tests check that it never leaks into errors, logs or traces."""

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_body(name: str) -> dict[str, Any]:
    """The body of a real API response, saved in tests/fixtures/."""
    saved: dict[str, Any] = json.loads((FIXTURES / f"{name}.json").read_text())
    body: dict[str, Any] = saved["body"]
    return body


def choice_body(question: str, pick: str, probabilities: dict[str, float]) -> dict[str, Any]:
    """A choice answer in the API's shape."""
    body = fixture_body("choice")
    answer = {
        "type": "choice",
        "choice": pick,
        "probabilities": probabilities,
        "confidence": max(probabilities.values()),
    }
    return {**body, "answers": {question: answer}}


def noul_body(question: str, probability: float) -> dict[str, Any]:
    """A noul answer in the API's shape."""
    body = fixture_body("noul")
    return {**body, "answers": {question: {"type": "noul", "noul": probability}}}


Reply = httpx.Response | Callable[[httpx.Request], httpx.Response] | Exception


@dataclass
class FakeWity:
    """Answers every request with the next queued reply, or ``default``. Records what was sent."""

    default: Reply = field(default_factory=lambda: httpx.Response(200, json=fixture_body("noul")))
    queue: list[Reply] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)

    @property
    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]

    def reply(self, *replies: Reply) -> None:
        self.queue.extend(replies)

    def json(self, body: dict[str, Any], status: int = 200) -> None:
        self.reply(httpx.Response(status, json=body))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.queue.pop(0) if self.queue else self.default
        if isinstance(reply, Exception):
            raise reply
        if isinstance(reply, httpx.Response):
            return reply
        return reply(request)

    def classifier(self, **options: Any) -> WityClassifier:
        """A classifier wired to this fake server. No retries unless asked for."""
        transport = httpx.MockTransport(self._handle)
        options.setdefault("api_key", TEST_KEY)
        options.setdefault("max_retries", 0)
        return WityClassifier(
            http_client=httpx.Client(transport=transport),
            http_async_client=httpx.AsyncClient(transport=transport),
            **options,
        )


@pytest.fixture
def fake() -> FakeWity:
    return FakeWity()


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Tests never see the developer's real key or address, or LangSmith settings."""
    for name in ("WITY_API_KEY", "WITY_BASE_URL", "WITY_LOG_LEVEL", "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2"):
        monkeypatch.delenv(name, raising=False)
    yield


class FakeChatModel(GenericFakeChatModel):
    """LangChain's fake chat model, made usable by agents: binding tools changes nothing."""

    calls: int = 0

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self

    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        return super()._generate(*args, **kwargs)


def fake_model(*replies: AIMessage) -> FakeChatModel:
    """A chat model that answers with ``replies``, one per call."""
    return FakeChatModel(messages=iter(replies))


def tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> AIMessage:
    """An AI message that asks to run one tool."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class Recorder(BaseCallbackHandler):
    """Keeps every chain start, end and error."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def on_chain_start(self, serialized: dict[str, Any] | None, inputs: Any, **kwargs: Any) -> None:
        self.events.append(("start", {"serialized": serialized, "inputs": inputs, **kwargs}))

    def on_chain_end(self, outputs: Any, **kwargs: Any) -> None:
        self.events.append(("end", {"outputs": outputs, **kwargs}))

    def on_chain_error(self, error: BaseException, **kwargs: Any) -> None:
        self.events.append(("error", {"error": error, **kwargs}))
