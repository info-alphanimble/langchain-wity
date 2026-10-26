"""``WityExtractor``: pull named fields out of a document with Wity."""

from __future__ import annotations

from typing import Any

import wity
from langchain_core.runnables import RunnableConfig
from pydantic import field_validator
from typing_extensions import override
from wity import ExtractResponse, ExtractType

from ._base import WityRunnable, record_usage, text_and_metadata
from .errors import convert_error


class WityExtractor(WityRunnable[ExtractResponse]):
    """Pull named fields out of a document: names, dates, amounts, parties.

    Each call takes text or a ``Document`` and returns the ``wity-sdk`` response.
    ``res.fields`` has every field's values, as written. ``res.entities`` has the same values
    with their position in the text and a confidence. The confidence is not calibrated.
    """

    fields: dict[str, str]
    """1 to 16 fields: a name you choose, to what it asks for in plain words."""
    types: dict[str, ExtractType] | None = None
    """Field name to ``"date"``, ``"duration"`` or ``"number"``, for values that should come back normalised."""

    @field_validator("fields")
    @classmethod
    def _some_fields(cls, fields: dict[str, str]) -> dict[str, str]:
        if not fields:
            raise ValueError("fields is empty. Ask for at least one field.")
        return fields

    @override
    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> ExtractResponse:
        return self._call_with_config(self._extract, input, self._traced_config(config), run_type="llm")

    @override
    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> ExtractResponse:
        return await self._acall_with_config(self._aextract, input, self._traced_config(config), run_type="llm")

    def _extract(self, input: Any) -> ExtractResponse:
        text, _ = text_and_metadata(input)
        try:
            response = self._client.extract(text=text, fields=self.fields, types=self.types)
        except wity.WityError as err:
            raise convert_error(err) from err
        return record_usage(response)

    async def _aextract(self, input: Any) -> ExtractResponse:
        text, _ = text_and_metadata(input)
        try:
            response = await self._async_client().extract(text=text, fields=self.fields, types=self.types)
        except wity.WityError as err:
            raise convert_error(err) from err
        return record_usage(response)
