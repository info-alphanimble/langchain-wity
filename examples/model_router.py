"""Wity picks the model for each agent run. Run: uv run python examples/model_router.py

The two chat models here are LangChain's fake ones, so this runs without an LLM key.
Use real ones in your app, like "openai:gpt-5-mini".
"""

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage

from langchain_wity.middleware import ModelChoice, ModelRouterMiddleware


class FakeChat(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


small = FakeChat(messages=iter([AIMessage("(the small model answered)")] * 2))
large = FakeChat(messages=iter([AIMessage("(the large model answered)")] * 2))

router = ModelRouterMiddleware(
    choices={
        "small": ModelChoice(model=small, criteria="Simple, well-scoped tasks."),
        "large": ModelChoice(model=large, criteria="Complex tasks that need deeper reasoning."),
    },
    instructions="Choose the least costly model suited to the task.",
)
agent = create_agent(small, middleware=[router])

for task in ["What's the capital of France?", "Design a sharding plan for a 40 TB Postgres database."]:
    result = agent.invoke({"messages": [HumanMessage(task)]})
    route = result["model_route"]
    print(f"{task!r} -> {route['choice']} {route['probabilities']}")
    print("  ", result["messages"][-1].content)
