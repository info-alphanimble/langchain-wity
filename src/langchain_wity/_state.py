"""Turn state that holds LangChain messages into plain JSON for Wity."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import BaseMessage, convert_to_openai_messages


def to_wity_state(state: object) -> str | dict[str, Any] | list[Any]:
    """Return ``state`` with every LangChain message turned into ``{role, content}``.

    Messages can sit at the top, or anywhere inside dicts and lists.
    """
    if isinstance(state, BaseMessage):
        message: dict[str, Any] = convert_to_openai_messages(state)
        return message
    if isinstance(state, (str, Mapping)) or _is_list(state):
        value: str | dict[str, Any] | list[Any] = _convert(state)
        return value
    raise TypeError(f"Wity state must be text, a dict, a list or LangChain messages, not {type(state).__name__}.")


def _is_list(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _convert(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        return convert_to_openai_messages(value)
    if isinstance(value, Mapping):
        for key in value:
            if not isinstance(key, str):
                raise TypeError("Wity state dict keys must be strings.")
        return {key: _convert(item) for key, item in value.items()}
    if _is_list(value):
        return [_convert(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Wity state can't hold a {type(value).__name__}. Use text, numbers, dicts, lists or messages.")
