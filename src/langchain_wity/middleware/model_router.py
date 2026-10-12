"""``ModelRouterMiddleware``: Wity picks which LLM handles each agent run."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from langchain.agents.middleware import Runtime
from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
    TracePolicy,
    omit_payload,
)
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from typing_extensions import NotRequired, override
from wity import Choice, SystemOneResponse

from ..classifier import WityClassifier
from ._shared import check_classifier

_QUESTION = "model_route"
_EARLIER_MESSAGES = 4
"""How many earlier user messages Wity sees, so a follow-up like "now do the same for prod" keeps its context."""


@dataclass(frozen=True)
class ModelChoice:
    """One model the router can pick, and a sentence saying when to pick it."""

    model: str | BaseChatModel
    """A LangChain chat model, or a model string for ``init_chat_model``, like ``"openai:gpt-5-mini"``."""
    criteria: str
    """When to use this model, for example ``"Simple, well-scoped tasks."``."""


class _RouterState(AgentState[ResponseT]):
    model_route: NotRequired[dict[str, Any]]
    """Wity's full answer as a dict: ``choice``, ``probabilities``, ``confidence`` and the reasoning info.

    A plain dict, so checkpointers can save it.
    """


class ModelRouterMiddleware(AgentMiddleware[_RouterState[ResponseT], ContextT, ResponseT]):
    """Pick the model for each agent run with a Wity ``choice``.

    Once at the start of each run, the latest user message goes to Wity, with up to 4 earlier user messages
    for context. The routes are the options. Wity's full answer is saved in the agent state as a dict under
    ``model_route``, and every model call in the run goes to the picked model. If Wity fails, the error stops
    the run: it never quietly picks a model.

    Reasoning is ``"auto"`` by default: clear tasks get one fast pass, and Wity thinks only when the routes
    are close. To change that, pass your own ``WityClassifier``.
    """

    state_schema = _RouterState  # type: ignore[assignment]
    trace_policy = TracePolicy(process_inputs=omit_payload)

    def __init__(
        self,
        *,
        choices: Mapping[str, ModelChoice],
        instructions: str,
        classifier: WityClassifier | None = None,
    ) -> None:
        """
        Args:
            choices: Routes by a name you choose. Each has a model and when to use it.
            instructions: How to pick, for example "Choose the least costly model suited to the task."
            classifier: Your own ``WityClassifier``, without questions of its own.
                Defaults to one with reasoning ``"auto"``.
        """
        super().__init__()
        if len(choices) < 2:
            raise ValueError("ModelRouterMiddleware needs at least two routes in choices.")
        self.question = Choice(
            instructions=instructions, criteria={name: route.criteria for name, route in choices.items()}
        )
        self.models: dict[str, BaseChatModel] = {
            name: init_chat_model(route.model) if isinstance(route.model, str) else route.model
            for name, route in choices.items()
        }
        self.classifier = check_classifier(classifier) if classifier is not None else WityClassifier(reasoning="auto")

    def _request(self, state: _RouterState[ResponseT]) -> dict[str, Any]:
        """The classifier input: the latest user message, and a few earlier ones for context."""
        asked = [m for m in state["messages"] if isinstance(m, HumanMessage)]
        if not asked:
            raise ValueError("ModelRouterMiddleware needs a user message to route on, and the agent state has none.")
        route_on: dict[str, Any] = {"latest_user_message": _text(asked[-1])}
        earlier = [_text(m) for m in asked[-1 - _EARLIER_MESSAGES : -1]]
        if earlier:
            route_on = {"earlier_user_messages": earlier, **route_on}
        return {"state": route_on, "questions": {_QUESTION: self.question}}

    @staticmethod
    def _update(response: SystemOneResponse) -> dict[str, Any]:
        return {"model_route": response.choices[_QUESTION].model_dump(mode="json", exclude_none=True)}

    @override
    def before_agent(self, state: _RouterState[ResponseT], runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        return self._update(self.classifier.invoke(self._request(state)))

    @override
    async def abefore_agent(self, state: _RouterState[ResponseT], runtime: Runtime[ContextT]) -> dict[str, Any] | None:
        return self._update(await self.classifier.ainvoke(self._request(state)))

    def _routed(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        answer: dict[str, Any] = request.state["model_route"]  # type: ignore[typeddict-item]
        return request.override(model=self.models[answer["choice"]])

    @override
    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        return handler(self._routed(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        return await handler(self._routed(request))


def _text(message: BaseMessage) -> str:
    """The message's text, with images and files shown by type, like "[image]", so they aren't lost."""
    parts = [str(message.text)] if message.text else []
    if isinstance(message.content, list):
        for block in message.content:
            if isinstance(block, dict) and block.get("type") != "text":
                parts.append(f"[{block.get('type', 'attachment')}]")
    return " ".join(parts)
