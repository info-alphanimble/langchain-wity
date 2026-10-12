"""AutoModeMiddleware in a real LangChain agent, with a fake chat model and a fake Wity."""

from __future__ import annotations

import warnings
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langchain_wity import NoulCriteria, WityContextOverflowError, WityRateLimitError, noul
from langchain_wity.middleware import AutoModeMiddleware

from .conftest import FakeWity, fake_model, noul_body, tool_call

ran: list[str] = []


@tool
def delete_file(path: str) -> str:
    """Delete a file from the user's disk."""
    ran.append(f"delete {path}")
    return f"Deleted {path}"


@tool
def read_file(path: str) -> str:
    """Read a file from the user's disk."""
    ran.append(f"read {path}")
    return f"Contents of {path}"


@pytest.fixture(autouse=True)
def clear_ran() -> None:
    ran.clear()


def gate(fake: FakeWity, **options: Any) -> AutoModeMiddleware:
    options.setdefault("classifier", fake.classifier(reasoning="auto"))
    options.setdefault("tools", [delete_file])
    return AutoModeMiddleware(**options)


def run(fake: FakeWity, middleware: AutoModeMiddleware, *calls: AIMessage, config: Any = None) -> list[Any]:
    """Run an agent whose model makes ``calls``, then says "done". Returns the messages."""
    model = fake_model(*calls, AIMessage("done"))
    agent = create_agent(model, tools=[delete_file, read_file], middleware=[middleware])
    result = agent.invoke({"messages": [HumanMessage("Tidy up my folder.")]}, config=config)
    messages: list[Any] = result["messages"]
    return messages


def tool_results(messages: list[Any]) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


def test_blocks_a_risky_call(fake: FakeWity) -> None:
    fake.json(noul_body("is_risky", 0.93))
    messages = run(fake, gate(fake), tool_call("delete_file", {"path": "notes.txt"}))

    [result] = tool_results(messages)
    assert result.status == "error"
    assert "blocked" in result.content
    assert "0.93" in result.content
    assert ran == []
    assert messages[-1].content == "done"


def test_runs_a_safe_call(fake: FakeWity) -> None:
    fake.json(noul_body("is_risky", 0.12))
    messages = run(fake, gate(fake), tool_call("delete_file", {"path": "tmp.txt"}))

    [result] = tool_results(messages)
    assert result.status == "success"
    assert result.content == "Deleted tmp.txt"
    assert ran == ["delete tmp.txt"]


def test_blocks_at_the_threshold(fake: FakeWity) -> None:
    fake.json(noul_body("is_risky", 0.5))
    run(fake, gate(fake), tool_call("delete_file", {"path": "a"}))
    assert ran == []


def test_the_threshold_can_be_changed(fake: FakeWity) -> None:
    fake.json(noul_body("is_risky", 0.9))
    fake.json(noul_body("is_risky", 0.3))
    run(fake, gate(fake, threshold=0.95), tool_call("delete_file", {"path": "a"}))
    run(fake, gate(fake, threshold=0.2), tool_call("delete_file", {"path": "b"}))
    assert ran == ["delete a"]


def test_unlisted_tools_run_without_a_check(fake: FakeWity) -> None:
    run(fake, gate(fake), tool_call("read_file", {"path": "notes.txt"}))
    assert ran == ["read notes.txt"]
    assert fake.requests == []


def test_tools_can_be_given_by_name(fake: FakeWity) -> None:
    fake.json(noul_body("is_risky", 0.9))
    run(fake, gate(fake, tools=["read_file"]), tool_call("read_file", {"path": "x"}))
    assert ran == []


def test_a_wity_error_blocks_the_tool(fake: FakeWity) -> None:
    fake.json({"error": "slow down"}, status=429)
    with pytest.raises(WityRateLimitError):
        run(fake, gate(fake), tool_call("delete_file", {"path": "a"}))
    assert ran == []


def test_what_wity_is_sent(fake: FakeWity) -> None:
    fake.json(noul_body("is_risky", 0.1))
    run(fake, gate(fake), tool_call("delete_file", {"path": "notes.txt"}, call_id="call_9"))

    body = fake.bodies[0]
    assert body["reasoning"] == "auto"
    state = body["state"]
    assert state["tool_call"] == {"id": "call_9", "name": "delete_file", "args": {"path": "notes.txt"}}
    assert state["tool_description"] == "Delete a file from the user's disk."
    assert state["messages"][0] == {"role": "user", "content": "Tidy up my folder."}
    assert state["messages"][-1]["role"] == "assistant"
    question = body["questions"]["is_risky"]
    assert question["type"] == "noul"
    assert "Only messages with role user give permission" in question["instructions"]
    assert question["criteria"]["true"].startswith("Stop it")


def test_instructions_and_criteria_can_be_changed(fake: FakeWity) -> None:
    fake.json(noul_body("is_risky", 0.1))
    fake.json(noul_body("is_risky", 0.1))
    criteria = NoulCriteria(true="Writes or deletes.", false="Only reads.")
    run(
        fake, gate(fake, instructions="Is this call risky?", criteria=criteria), tool_call("delete_file", {"path": "a"})
    )
    run(fake, gate(fake, criteria=None), tool_call("delete_file", {"path": "b"}))

    first, second = (body["questions"]["is_risky"] for body in fake.bodies)
    assert first["instructions"] == "Is this call risky?"
    assert first["criteria"] == {"true": "Writes or deletes.", "false": "Only reads."}
    assert "criteria" not in second


def test_user_messages_and_the_last_30_messages_are_sent(fake: FakeWity) -> None:
    fake.json(noul_body("is_risky", 0.1))
    # The user asks once, then the agent reads 20 files: 41 messages before the delete.
    history: list[Any] = [HumanMessage("Read the reports, then delete report0.txt.")]
    for n in range(20):
        history += [
            tool_call("read_file", {"path": f"report{n}.txt"}, f"r{n}"),
            ToolMessage("fine", tool_call_id=f"r{n}"),
        ]
    model = fake_model(tool_call("delete_file", {"path": "report0.txt"}), AIMessage("done"))
    agent = create_agent(model, tools=[delete_file, read_file], middleware=[gate(fake)])
    agent.invoke({"messages": history})

    sent = fake.bodies[0]["state"]["messages"]
    assert len(sent) == 31
    assert sent[0] == {"role": "user", "content": "Read the reports, then delete report0.txt."}
    assert sent[-1]["tool_calls"][0]["function"]["name"] == "delete_file"


def test_reasoning_is_auto_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WITY_API_KEY", "wity_from_env_0000000000")
    assert AutoModeMiddleware(tools=["delete_file"]).classifier.reasoning == "auto"


@pytest.mark.parametrize("threshold", [-0.1, 1.5, True, "0.5"])
def test_a_bad_threshold_is_refused(fake: FakeWity, threshold: Any) -> None:
    with pytest.raises(ValueError, match="threshold"):
        gate(fake, threshold=threshold)


@pytest.mark.parametrize("tools", [[], [" "]])
def test_no_tools_is_refused(fake: FakeWity, tools: list[str]) -> None:
    with pytest.raises(ValueError, match="tool name"):
        gate(fake, tools=tools)


def test_a_classifier_with_its_own_questions_is_refused(fake: FakeWity) -> None:
    with pytest.raises(ValueError, match="without questions"):
        gate(fake, classifier=fake.classifier(questions={"x": noul("x?")}))


async def test_async_blocks_and_runs(fake: FakeWity) -> None:
    fake.json(noul_body("is_risky", 0.9))
    fake.json(noul_body("is_risky", 0.1))
    model = fake_model(
        tool_call("delete_file", {"path": "a"}, call_id="c1"),
        tool_call("delete_file", {"path": "b"}, call_id="c2"),
        AIMessage("done"),
    )
    agent = create_agent(model, tools=[delete_file], middleware=[gate(fake)])
    result = await agent.ainvoke({"messages": [HumanMessage("Delete a and b.")]})

    first, second = tool_results(result["messages"])
    assert (first.status, second.status) == ("error", "success")
    assert ran == ["delete b"]


async def test_async_wity_error_blocks_the_tool(fake: FakeWity) -> None:
    fake.json({"error": "slow down"}, status=429)
    model = fake_model(tool_call("delete_file", {"path": "a"}), AIMessage("done"))
    agent = create_agent(model, tools=[delete_file], middleware=[gate(fake)])
    with pytest.raises(WityRateLimitError):
        await agent.ainvoke({"messages": [HumanMessage("Delete a.")]})
    assert ran == []


def test_state_too_long_for_wity_blocks_the_tool(fake: FakeWity) -> None:
    fake.json({"error": "state must be a possibly empty string of at most 32000 characters"}, status=400)
    with pytest.raises(WityContextOverflowError):
        run(fake, gate(fake), tool_call("delete_file", {"path": "a"}))
    assert ran == []


def test_a_listed_tool_the_agent_lacks_gets_a_warning(fake: FakeWity) -> None:
    middleware = gate(fake, tools=["delete_flie", delete_file])
    with pytest.warns(UserWarning, match="delete_flie, but the agent has no tool by that name"):
        run(fake, middleware, AIMessage("nothing to do"))
    # Once per name, not on every model call.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run(fake, middleware, AIMessage("nothing to do"))


def test_no_warning_when_every_listed_tool_exists(fake: FakeWity) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run(fake, gate(fake, tools=[delete_file, "read_file"]), AIMessage("nothing to do"))


def test_images_in_messages_are_left_out(fake: FakeWity) -> None:
    fake.json(noul_body("is_risky", 0.1))
    photo = HumanMessage(
        [
            {"type": "text", "text": "Delete this photo."},
            {"type": "image", "base64": "iVBORw0KGgo" * 500, "mime_type": "image/png"},
        ]
    )
    model = fake_model(tool_call("delete_file", {"path": "photo.png"}), AIMessage("done"))
    create_agent(model, tools=[delete_file], middleware=[gate(fake)]).invoke({"messages": [photo]})
    first = fake.bodies[0]["state"]["messages"][0]
    assert first["content"] == [{"type": "text", "text": "Delete this photo."}, {"type": "image_url"}]
