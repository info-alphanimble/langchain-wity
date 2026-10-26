"""``WityClassifier``: a LangChain Runnable that asks Wity typed questions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import wity
from langchain_core.runnables import RunnableConfig
from typing_extensions import override
from wity import Question, Reasoning, SystemOneResponse

from ._base import WityRunnable, record_usage
from ._state import to_wity_state
from .errors import convert_error

_NEEDS_STATE_AND_QUESTIONS = (
    "This WityClassifier has no questions of its own, so each call needs {'state': ..., 'questions': {...}}. "
    "Or pass questions= when you create the classifier."
)


class WityClassifier(WityRunnable[SystemOneResponse]):
    """Ask Wity typed questions and get typed answers with probabilities.

    With ``questions`` set here, each call takes only the state: the text or data to decide about.
    Without them, each call takes ``{"state": ..., "questions": {...}}``.

    State can be text, a dict, a list, a LangChain message or a list of messages.
    Messages can also sit inside dicts and lists. They're sent as ``{role, content}``.

    The answer is the ``wity-sdk`` response: read ``res.choices``, ``res.nouls`` and ``res.scores``.
    Keep one classifier around and reuse it, so connections stay open.
    """

    questions: dict[str, Question] | None = None
    """Questions by a name you choose. When set, the whole input of each call is the state."""
    reasoning: Reasoning | None = None
    """``"off"``, ``"auto"`` or ``"always"``. Leave out for the API default, ``"auto"``."""
    max_latency_ms: int | None = None
    """Time limit per request, 200 to 120000 ms. A whole number: the API refuses 200.5."""

    @override
    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> SystemOneResponse:
        return self._call_with_config(self._classify, input, self._traced_config(config), run_type="llm")

    @override
    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> SystemOneResponse:
        return await self._acall_with_config(self._aclassify, input, self._traced_config(config), run_type="llm")

    def _classify(self, input: Any) -> SystemOneResponse:
        state, questions = self._split(input)
        try:
            response = self._client.system_one(
                state=state, questions=questions, reasoning=self.reasoning, max_latency_ms=self.max_latency_ms
            )
        except wity.WityError as err:
            raise convert_error(err) from err
        return record_usage(response)

    async def _aclassify(self, input: Any) -> SystemOneResponse:
        state, questions = self._split(input)
        try:
            response = await self._async_client().system_one(
                state=state, questions=questions, reasoning=self.reasoning, max_latency_ms=self.max_latency_ms
            )
        except wity.WityError as err:
            raise convert_error(err) from err
        return record_usage(response)

    def _split(self, input: Any) -> tuple[str | dict[str, Any] | list[Any], Mapping[str, Any]]:
        """Return (state, questions) for one call."""
        if self.questions is not None:
            # Strict on purpose: with questions set here, the whole input is state, even a dict with "questions".
            return to_wity_state(input), self.questions
        if not isinstance(input, Mapping) or set(input) != {"state", "questions"}:
            raise ValueError(_NEEDS_STATE_AND_QUESTIONS)
        questions = input["questions"]
        if not isinstance(questions, Mapping) or not questions:
            raise ValueError("questions must be a non-empty dict of question name to question.")
        return to_wity_state(input["state"]), questions
