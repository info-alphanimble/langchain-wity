"""Turn state that holds LangChain messages into plain JSON for Wity."""

from __future__ import annotations

import dataclasses
import datetime
import decimal
import enum
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import BaseMessage, convert_to_openai_messages
from pydantic import BaseModel
from pydantic_core import to_jsonable_python

# Values pydantic turns into JSON the usual way: dates as ISO text, UUIDs and decimals as text, enums as their value.
_PLAIN_VALUES = (datetime.date, datetime.time, datetime.timedelta, uuid.UUID, decimal.Decimal, enum.Enum)


def to_wity_state(state: object) -> str | dict[str, Any] | list[Any]:
    """Return ``state`` as JSON Wity reads, with every LangChain message turned into ``{role, content}``.

    Messages can sit at the top, or anywhere inside dicts and lists. Images, audio and files in
    messages are left out: only their type is sent. Pydantic models and dataclasses become dicts.
    """
    if isinstance(state, (BaseMessage, BaseModel, str, Mapping)) or _is_dataclass(state) or _is_list(state):
        value: str | dict[str, Any] | list[Any] = _convert(state)
        return value
    raise TypeError(f"Wity state must be text, a dict, a list or LangChain messages, not {type(state).__name__}.")


def _is_list(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _is_dataclass(value: object) -> bool:
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def _convert(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        return _without_media(convert_to_openai_messages(value))
    if isinstance(value, BaseModel) or _is_dataclass(value):
        return _convert(to_jsonable_python(value))
    if isinstance(value, Mapping):
        for key in value:
            if not isinstance(key, str):
                raise TypeError("Wity state dict keys must be strings.")
        return {key: _convert(item) for key, item in value.items()}
    if _is_list(value):
        return [_convert(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, _PLAIN_VALUES):
        return to_jsonable_python(value)
    raise TypeError(f"Wity state can't hold a {type(value).__name__}. Use text, numbers, dicts, lists or messages.")


def _without_media(message: dict[str, Any]) -> dict[str, Any]:
    """Keep only the type of each image, audio or file block.

    Sent as text, a small image is thousands of tokens of base64 that Wity can't read.
    """
    content = message.get("content")
    if not isinstance(content, list):
        return message
    kept = [
        block if not isinstance(block, dict) or block.get("type") == "text" else {"type": block.get("type")}
        for block in content
    ]
    return {**message, "content": kept}
