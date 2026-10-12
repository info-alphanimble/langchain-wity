"""WityClassifier: input forms, state, options, errors, secrets and tracing."""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import datetime
import decimal
import logging
import pickle
import uuid
from typing import Any

import httpx
import pytest
import wity
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel

import langchain_wity.classifier
from langchain_wity import (
    SystemOneResponse,
    WityAPIError,
    WityAuthenticationError,
    WityBadRequestError,
    WityClassifier,
    WityConnectionError,
    WityContextOverflowError,
    WityInsufficientBalanceError,
    WityInternalServerError,
    WityRateLimitError,
    WityTimeoutError,
    choice,
    noul,
    score,
)

from .conftest import TEST_KEY, FakeWity, Recorder, fixture_body

URGENT = noul("Does the customer ask for fast action?", yes="They ask for fast action", no="No time pressure")
TEAM = choice("Which team?", {"billing": "Payments", "technical": "Bugs", "other": "Anything else"})
TICKET = "I was charged twice. Please refund one today."

# ---------------------------------------------------------------------------
# Input forms
# ---------------------------------------------------------------------------


def test_questions_in_constructor_take_the_whole_input_as_state(fake: FakeWity) -> None:
    res = fake.classifier(questions={"urgent": URGENT}).invoke(TICKET)
    assert isinstance(res, SystemOneResponse)
    assert res.nouls["urgent"].noul == pytest.approx(0.9998766053661012)
    body = fake.bodies[0]
    assert body["state"] == TICKET
    assert body["questions"]["urgent"]["type"] == "noul"
    assert body["questions"]["urgent"]["criteria"] == {"true": "They ask for fast action", "false": "No time pressure"}


def test_questions_in_constructor_treat_a_questions_key_as_state(fake: FakeWity) -> None:
    state = {"state": "hello", "questions": "this is just data"}
    fake.classifier(questions={"urgent": URGENT}).invoke(state)
    assert fake.bodies[0]["state"] == state
    assert list(fake.bodies[0]["questions"]) == ["urgent"]


def test_questions_with_each_call(fake: FakeWity) -> None:
    fake.json(fixture_body("multi-auto"))
    res = fake.classifier().invoke({"state": TICKET, "questions": {"team": TEAM, "urgent": URGENT}})
    assert res.choices["team"].choice == "billing"
    assert set(fake.bodies[0]["questions"]) == {"team", "urgent"}


def test_questions_can_be_plain_dicts(fake: FakeWity) -> None:
    question = {"type": "noul", "instructions": "Urgent?"}
    fake.classifier(questions={"urgent": question}).invoke(TICKET)
    assert fake.bodies[0]["questions"]["urgent"] == question


@pytest.mark.parametrize(
    "bad_input",
    [
        TICKET,
        {"state": TICKET},
        {"questions": {"urgent": URGENT}},
        {"state": TICKET, "questions": {"urgent": URGENT}, "reasoning": "off"},
        {"state": TICKET, "questions": {}},
        {"state": TICKET, "questions": "urgent"},
    ],
)
def test_without_constructor_questions_the_input_must_be_state_and_questions(fake: FakeWity, bad_input: Any) -> None:
    with pytest.raises(ValueError, match="questions"):
        fake.classifier().invoke(bad_input)
    assert fake.requests == []


def test_bad_input_error_never_shows_the_state(fake: FakeWity) -> None:
    with pytest.raises(ValueError) as caught:
        fake.classifier().invoke({"state": "secret customer text"})
    assert "secret customer text" not in str(caught.value)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def test_a_message_becomes_role_and_content(fake: FakeWity) -> None:
    fake.classifier(questions={"urgent": URGENT}).invoke(HumanMessage(TICKET, id="abc"))
    assert fake.bodies[0]["state"] == {"role": "user", "content": TICKET}


def test_a_list_of_messages(fake: FakeWity) -> None:
    messages = [SystemMessage("Be kind."), HumanMessage(TICKET), AIMessage("On it.")]
    fake.classifier(questions={"urgent": URGENT}).invoke(messages)
    assert fake.bodies[0]["state"] == [
        {"role": "system", "content": "Be kind."},
        {"role": "user", "content": TICKET},
        {"role": "assistant", "content": "On it."},
    ]


def test_messages_nested_in_dicts_and_lists(fake: FakeWity) -> None:
    state = {"customer": {"plan": "pro", "seats": 3}, "thread": [HumanMessage("Hi"), {"note": AIMessage("Hello")}]}
    fake.classifier().invoke({"state": state, "questions": {"urgent": URGENT}})
    assert fake.bodies[0]["state"] == {
        "customer": {"plan": "pro", "seats": 3},
        "thread": [{"role": "user", "content": "Hi"}, {"note": {"role": "assistant", "content": "Hello"}}],
    }


@pytest.mark.parametrize("bad_state", [42, None, {1: "number key"}, {"when": object()}])
def test_state_that_isnt_json_is_refused(fake: FakeWity, bad_state: Any) -> None:
    with pytest.raises(TypeError):
        fake.classifier(questions={"urgent": URGENT}).invoke(bad_state)
    assert fake.requests == []


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def test_reasoning_and_max_latency_are_sent(fake: FakeWity) -> None:
    fake.classifier(questions={"urgent": URGENT}, reasoning="off", max_latency_ms=800).invoke(TICKET)
    assert fake.bodies[0]["reasoning"] == "off"
    assert fake.bodies[0]["max_latency_ms"] == 800


def test_reasoning_is_left_to_the_api_by_default(fake: FakeWity) -> None:
    fake.classifier(questions={"urgent": URGENT}).invoke(TICKET)
    assert "reasoning" not in fake.bodies[0]
    assert "max_latency_ms" not in fake.bodies[0]


def test_key_and_base_url_are_used(fake: FakeWity) -> None:
    fake.classifier(questions={"urgent": URGENT}, base_url="http://localhost:8300/").invoke(TICKET)
    request = fake.requests[0]
    assert str(request.url) == "http://localhost:8300/v1/systemone"
    assert request.headers["authorization"] == f"Bearer {TEST_KEY}"


def test_key_and_base_url_come_from_the_environment(fake: FakeWity, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WITY_API_KEY", "wity_from_env_0000000000")
    monkeypatch.setenv("WITY_BASE_URL", "http://127.0.0.1:8300")
    fake.classifier(questions={"urgent": URGENT}, api_key=None).invoke(TICKET)
    assert fake.requests[0].headers["authorization"] == "Bearer wity_from_env_0000000000"
    assert str(fake.requests[0].url) == "http://127.0.0.1:8300/v1/systemone"


def test_no_key_is_a_clear_error() -> None:
    with pytest.raises(wity.WityError, match="WITY_API_KEY"):
        WityClassifier()


def test_max_retries_is_passed_to_the_sdk(fake: FakeWity) -> None:
    busy = httpx.Response(429, json={"error": "slow down"}, headers={"retry-after": "0"})
    fake.reply(busy, busy)
    res = fake.classifier(questions={"urgent": URGENT}, max_retries=2).invoke(TICKET)
    assert len(fake.requests) == 3
    assert res.nouls["urgent"].noul > 0.9


def test_timeout_is_passed_to_the_sdk(fake: FakeWity) -> None:
    def check_timeout(request: httpx.Request) -> httpx.Response:
        assert request.extensions["timeout"]["read"] == 7.5
        return httpx.Response(200, json=fixture_body("noul"))

    fake.reply(check_timeout)
    fake.classifier(questions={"urgent": URGENT}, timeout=7.5).invoke(TICKET)


def test_unknown_options_are_refused() -> None:
    with pytest.raises(ValueError, match="extra"):
        WityClassifier(api_key=TEST_KEY, reasonin="off")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Runnable behaviour
# ---------------------------------------------------------------------------


def test_works_in_a_pipeline(fake: FakeWity) -> None:
    fake.json(fixture_body("choice"))
    chain = RunnableLambda(lambda text: text.strip()) | fake.classifier(questions={"team": TEAM})
    res = chain.invoke(f"  {TICKET}  ")
    assert res.choices["team"].choice == "billing"
    assert fake.bodies[0]["state"] == TICKET


def test_batch(fake: FakeWity) -> None:
    results = fake.classifier(questions={"urgent": URGENT}).batch(["one", "two", "three"])
    assert len(results) == 3
    assert sorted(body["state"] for body in fake.bodies) == ["one", "three", "two"]


async def test_ainvoke_and_abatch(fake: FakeWity) -> None:
    classifier = fake.classifier(questions={"urgent": URGENT})
    res = await classifier.ainvoke([HumanMessage(TICKET)])
    assert res.nouls["urgent"].noul > 0.9
    assert fake.bodies[0]["state"] == [{"role": "user", "content": TICKET}]
    results = await classifier.abatch(["a", "b"])
    assert len(results) == 2


async def test_async_input_form_errors(fake: FakeWity) -> None:
    with pytest.raises(ValueError, match="questions"):
        await fake.classifier().ainvoke(TICKET)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

ERROR_CASES = [
    (
        400,
        {"error": "questions.urgent: unknown type"},
        WityBadRequestError,
        wity.BadRequestError,
        ModelInvalidRequestError,
    ),
    (401, {"error": "invalid API key"}, WityAuthenticationError, wity.AuthenticationError, ModelAuthenticationError),
    (
        402,
        {"error": "insufficient balance"},
        WityInsufficientBalanceError,
        wity.InsufficientBalanceError,
        ModelAPIError,
    ),
    (429, {"error": "slow down"}, WityRateLimitError, wity.RateLimitError, ModelRateLimitError),
    (500, {"error": "model error"}, WityInternalServerError, wity.InternalServerError, ModelAPIError),
    (404, {"error": "not found"}, WityAPIError, wity.APIError, ModelAPIError),
]


@pytest.mark.parametrize(("status", "body", "ours", "sdk", "langchain"), ERROR_CASES)
def test_api_errors_are_both_wity_and_langchain_errors(
    fake: FakeWity, status: int, body: dict[str, Any], ours: type, sdk: type, langchain: type
) -> None:
    fake.json(body, status=status)
    with pytest.raises(ours) as caught:
        fake.classifier(questions={"urgent": URGENT}).invoke("secret customer text")
    err: Any = caught.value
    assert err.status == status
    assert isinstance(err, sdk)
    assert isinstance(err, langchain)
    assert str(status) in str(err)
    assert "secret customer text" not in str(err)
    assert TEST_KEY not in str(err)


def test_state_too_long_is_a_context_overflow(fake: FakeWity) -> None:
    fake.json({"error": "state must be a string of at most 32000 characters"}, status=400)
    with pytest.raises(WityContextOverflowError) as caught:
        fake.classifier(questions={"urgent": URGENT}).invoke("x")
    assert isinstance(caught.value, ContextOverflowError)
    assert isinstance(caught.value, WityBadRequestError)
    assert isinstance(caught.value, ModelInvalidRequestError)


def test_no_balance_is_not_retryable() -> None:
    assert WityInsufficientBalanceError.is_retryable is False
    assert WityRateLimitError.is_retryable is True


def test_connection_failure(fake: FakeWity) -> None:
    fake.reply(httpx.ConnectError("refused"))
    with pytest.raises(WityConnectionError) as caught:
        fake.classifier(questions={"urgent": URGENT}).invoke(TICKET)
    assert isinstance(caught.value, wity.APIConnectionError)
    assert isinstance(caught.value, ModelConnectionError)


async def test_timeout_async(fake: FakeWity) -> None:
    fake.reply(httpx.ReadTimeout("slow"))
    with pytest.raises(WityTimeoutError) as caught:
        await fake.classifier(questions={"urgent": URGENT}).ainvoke(TICKET)
    assert isinstance(caught.value, wity.APITimeoutError)
    assert isinstance(caught.value, ModelTimeoutError)


async def test_async_api_error(fake: FakeWity) -> None:
    fake.json({"error": "slow down"}, status=429)
    with pytest.raises(WityRateLimitError):
        await fake.classifier(questions={"urgent": URGENT}).ainvoke(TICKET)


# ---------------------------------------------------------------------------
# The key stays secret
# ---------------------------------------------------------------------------


def test_key_never_in_repr_or_saved_config(fake: FakeWity) -> None:
    classifier = fake.classifier(questions={"urgent": URGENT})
    assert TEST_KEY not in repr(classifier)
    assert TEST_KEY not in str(classifier)
    assert TEST_KEY not in str(classifier.model_dump())
    assert TEST_KEY not in classifier.model_dump_json()
    assert TEST_KEY not in str(classifier.to_json())


def test_key_never_in_logs(fake: FakeWity, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.DEBUG)
    fake.json({"error": "invalid API key"}, status=401)
    with pytest.raises(WityAuthenticationError):
        fake.classifier(questions={"urgent": URGENT}).invoke(TICKET)
    fake.classifier(questions={"urgent": URGENT}).invoke(TICKET)
    assert caplog.records
    assert TEST_KEY not in caplog.text


# ---------------------------------------------------------------------------
# Tracing
# ---------------------------------------------------------------------------


def test_callbacks_show_a_wity_model_call(fake: FakeWity) -> None:
    recorder = Recorder()
    fake.classifier(questions={"urgent": URGENT}).invoke(TICKET, config={"callbacks": [recorder]})
    kinds = [kind for kind, _ in recorder.events]
    assert kinds == ["start", "end"]
    metadata = recorder.events[0][1]["metadata"]
    assert metadata["ls_provider"] == "wity"
    assert metadata["ls_model_name"] == "wity-1"
    assert TEST_KEY not in str(recorder.events)


def test_callbacks_keep_your_metadata(fake: FakeWity) -> None:
    recorder = Recorder()
    config: Any = {"callbacks": [recorder], "metadata": {"team": "support"}}
    fake.classifier(questions={"urgent": URGENT}).invoke(TICKET, config=config)
    metadata = recorder.events[0][1]["metadata"]
    assert metadata["team"] == "support"
    assert metadata["ls_provider"] == "wity"


async def test_async_callbacks_and_errors_are_traced(fake: FakeWity) -> None:
    recorder = Recorder()
    fake.json({"error": "slow down"}, status=429)
    with pytest.raises(WityRateLimitError):
        await fake.classifier(questions={"urgent": URGENT}).ainvoke(TICKET, config={"callbacks": [recorder]})
    assert [kind for kind, _ in recorder.events] == ["start", "error"]
    assert TEST_KEY not in str(recorder.events)


class FakeRun:
    def __init__(self) -> None:
        self.extra: dict[str, Any] = {}


def test_token_count_goes_on_the_trace(fake: FakeWity, monkeypatch: pytest.MonkeyPatch) -> None:
    run = FakeRun()
    monkeypatch.setattr(langchain_wity.classifier, "get_current_run_tree", lambda: run)
    fake.classifier(questions={"urgent": URGENT}).invoke(TICKET)
    assert run.extra["metadata"]["usage_metadata"] == {"input_tokens": 123, "output_tokens": 0, "total_tokens": 123}


def test_a_tracing_problem_never_fails_the_call(fake: FakeWity, monkeypatch: pytest.MonkeyPatch) -> None:
    def broken() -> None:
        raise RuntimeError("tracer down")

    monkeypatch.setattr(langchain_wity.classifier, "get_current_run_tree", broken)
    res = fake.classifier(questions={"urgent": URGENT}).invoke(TICKET)
    assert res.nouls["urgent"].noul > 0.9


def test_score_answers(fake: FakeWity) -> None:
    fake.json(fixture_body("score"))
    severity = score("How much is this hurting the customer?", ["Not at all", "A little", "A lot"])
    res = fake.classifier(questions={"severity": severity}).invoke(TICKET)
    assert set(res.scores["severity"].probabilities) == {"0", "1", "2"}


# ---------------------------------------------------------------------------
# Found by black-box testing (test/langchain in the root repo)
# ---------------------------------------------------------------------------


def test_max_latency_ms_is_sent_as_a_whole_number(fake: FakeWity) -> None:
    # The API refuses 800.0 with "max_latency_ms must be an integer".
    fake.classifier(questions={"urgent": URGENT}, max_latency_ms=800.0).invoke(TICKET)
    assert b'"max_latency_ms": 800' in fake.requests[0].content
    assert b"800.0" not in fake.requests[0].content


def test_max_latency_ms_with_a_fraction_is_refused(fake: FakeWity) -> None:
    with pytest.raises(ValueError, match="max_latency_ms"):
        fake.classifier(max_latency_ms=800.5)


class Ticket(BaseModel):
    subject: str
    opened: datetime.date


@dataclasses.dataclass
class Order:
    id: uuid.UUID
    total: decimal.Decimal


def test_models_dataclasses_and_dates_are_state(fake: FakeWity) -> None:
    order_id = uuid.UUID("12345678-1234-5678-1234-567812345678")
    fake.classifier(questions={"urgent": URGENT}).invoke(Ticket(subject="Refund", opened=datetime.date(2026, 10, 1)))
    fake.classifier(questions={"urgent": URGENT}).invoke(
        {"order": Order(id=order_id, total=decimal.Decimal("9.50")), "at": datetime.datetime(2026, 10, 1, 9, 30)}
    )
    assert fake.bodies[0]["state"] == {"subject": "Refund", "opened": "2026-10-01"}
    assert fake.bodies[1]["state"] == {
        "order": {"id": str(order_id), "total": "9.50"},
        "at": "2026-10-01T09:30:00",
    }


def test_media_in_messages_is_left_out(fake: FakeWity) -> None:
    message = HumanMessage(
        [
            {"type": "text", "text": "Is this receipt a duplicate?"},
            {"type": "image", "base64": "iVBORw0KGgo" * 1000, "mime_type": "image/png"},
            {"type": "image", "url": "https://example.com/receipt.png"},
        ]
    )
    fake.classifier(questions={"urgent": URGENT}).invoke([message])
    assert fake.bodies[0]["state"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Is this receipt a duplicate?"},
                {"type": "image_url"},
                {"type": "image_url"},
            ],
        }
    ]


def test_works_across_event_loops(fake: FakeWity) -> None:
    classifier = fake.classifier(questions={"urgent": URGENT})
    for _ in range(3):
        asyncio.run(classifier.ainvoke(TICKET))
    assert len(fake.requests) == 3


def test_drops_clients_of_closed_loops(fake: FakeWity) -> None:
    # A real connection keeps its loop alive, so the weak key never drops it. `loops` does the same here.
    classifier = fake.classifier(questions={"urgent": URGENT})
    loops = []
    for _ in range(3):
        loop = asyncio.new_event_loop()
        loops.append(loop)
        loop.run_until_complete(classifier.ainvoke(TICKET))
        loop.close()
    assert len(classifier._async_clients) == 1


def test_close_and_aclose(fake: FakeWity, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WITY_API_KEY", "wity_from_env_0000000000")
    classifier = WityClassifier(questions={"urgent": URGENT})
    classifier.close()

    async def use_and_close() -> None:
        await classifier.aclose()  # Nothing opened in this loop yet: does nothing.

    asyncio.run(use_and_close())


def test_pickle_and_deepcopy_make_fresh_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WITY_API_KEY", "wity_from_env_0000000000")
    classifier = WityClassifier(questions={"urgent": URGENT}, reasoning="off", max_latency_ms=900)
    for twin in (pickle.loads(pickle.dumps(classifier)), copy.deepcopy(classifier)):
        assert isinstance(twin, WityClassifier)
        assert (twin.questions, twin.reasoning, twin.max_latency_ms) == ({"urgent": URGENT}, "off", 900)
        assert twin._client is not classifier._client


def test_pickle_never_holds_a_key_read_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WITY_API_KEY", "wity_from_env_0000000000")
    assert b"wity_from_env" not in pickle.dumps(WityClassifier(questions={"urgent": URGENT}))


def test_errors_survive_pickling(fake: FakeWity) -> None:
    fake.reply(httpx.Response(429, json={"error": "slow down"}, headers={"retry-after": "3"}))
    with pytest.raises(WityRateLimitError) as caught:
        fake.classifier(questions={"urgent": URGENT}).invoke(TICKET)
    copy_ = pickle.loads(pickle.dumps(caught.value))
    assert type(copy_) is WityRateLimitError
    assert (copy_.status, copy_.headers.get("retry-after"), str(copy_)) == (429, "3", str(caught.value))
