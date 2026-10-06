"""``WityClassifier``: a LangChain Runnable that asks Wity typed questions."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import httpx
import wity
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langsmith.run_helpers import get_current_run_tree
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr, field_validator
from typing_extensions import override
from wity import Question, Reasoning, SystemOneResponse

from ._state import to_wity_state
from .errors import convert_error

logger = logging.getLogger("langchain_wity")

_PROVIDER = "wity"
_MODEL = "wity-1"

_NEEDS_STATE_AND_QUESTIONS = (
    "This WityClassifier has no questions of its own, so each call needs {'state': ..., 'questions': {...}}. "
    "Or pass questions= when you create the classifier."
)


class WityClassifier(RunnableSerializable[Any, SystemOneResponse]):
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
    max_latency_ms: float | None = None
    """Time limit per request, 200 to 120000 ms."""
    api_key: SecretStr | str | None = Field(default=None, exclude=True, repr=False)
    """Defaults to the WITY_API_KEY environment variable."""
    base_url: str | None = None
    """Defaults to WITY_BASE_URL, then the SDK's default address."""
    timeout: float = wity.DEFAULT_TIMEOUT
    """Seconds per attempt."""
    max_retries: int = wity.DEFAULT_MAX_RETRIES
    """Retries on 408, 429, 5xx and network failures. Set 0 if you add ``.with_retry()``."""
    http_client: httpx.Client | None = Field(default=None, exclude=True, repr=False)
    """Your own ``httpx.Client`` for ``invoke`` and ``batch``, for example for a proxy."""
    http_async_client: httpx.AsyncClient | None = Field(default=None, exclude=True, repr=False)
    """Your own ``httpx.AsyncClient`` for ``ainvoke`` and ``abatch``."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _client: wity.WityClient = PrivateAttr()
    _async_client: wity.AsyncWityClient = PrivateAttr()

    @field_validator("api_key")
    @classmethod
    def _hide_key(cls, api_key: SecretStr | str | None) -> SecretStr | None:
        return SecretStr(api_key) if isinstance(api_key, str) else api_key

    @override
    def model_post_init(self, context: Any, /) -> None:
        key = self.api_key.get_secret_value() if isinstance(self.api_key, SecretStr) else None
        options: dict[str, Any] = {
            "api_key": key,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        self._client = wity.WityClient(**options, http_client=self.http_client)
        self._async_client = wity.AsyncWityClient(**options, http_client=self.http_async_client)

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
        return _record_usage(response)

    async def _aclassify(self, input: Any) -> SystemOneResponse:
        state, questions = self._split(input)
        try:
            response = await self._async_client.system_one(
                state=state, questions=questions, reasoning=self.reasoning, max_latency_ms=self.max_latency_ms
            )
        except wity.WityError as err:
            raise convert_error(err) from err
        return _record_usage(response)

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

    def _traced_config(self, config: RunnableConfig | None) -> RunnableConfig:
        """Mark the run as a Wity model call, so LangSmith shows provider and model."""
        config = ensure_config(config)
        config["metadata"] = {
            **(config.get("metadata") or {}),
            "ls_provider": _PROVIDER,
            "ls_model_name": _MODEL,
            "ls_model_type": "chat",
        }
        return config


def _record_usage(response: SystemOneResponse) -> SystemOneResponse:
    """Add the token count to the current trace, if there is one. Tracing problems never fail the call."""
    try:
        run = get_current_run_tree()
        if run is not None:
            tokens = response.usage.input_tokens
            run.extra.setdefault("metadata", {})["usage_metadata"] = {
                "input_tokens": tokens,
                "output_tokens": 0,
                "total_tokens": tokens,
            }
    except Exception:
        logger.debug("Could not add Wity usage to the trace.", exc_info=True)
    return response
