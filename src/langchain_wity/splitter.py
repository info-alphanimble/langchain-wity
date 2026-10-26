"""``WitySplitter``: cut documents into whole items (emails, clauses, messages) with Wity."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import wity
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.runnables import RunnableConfig
from pydantic import model_validator
from typing_extensions import Self, override
from wity import SplitResponse, SplitSpec

from ._base import WityRunnable, record_usage, text_and_metadata
from .errors import convert_error


class WitySplitter(WityRunnable[list[Document]], BaseDocumentTransformer):
    """Cut a document into whole items: emails in a thread, clauses in a contract, entries in a log.

    Each call takes text or a ``Document`` and returns one ``Document`` per item, in document order.
    Every item is an exact slice of the text. Its offset is in ``metadata["start_index"]``,
    next to the metadata of the document it came from.

    Use it like a LangChain text splitter: ``split_text``, ``split_documents``, or in a pipeline.
    """

    split: str | SplitSpec | dict[str, Any] | None = None
    """What one item is: plain words like ``"one item per email"``, or a ``SplitSpec``.
    Leave out for the API default, ``"one item per paragraph"``."""
    skill: str | None = None
    """Instead of ``split``: a split skill's id or name. It splits the way its examples do."""

    @model_validator(mode="after")
    def _split_or_skill(self) -> Self:
        if self.split is not None and self.skill is not None:
            raise ValueError("Give split or skill, not both. A split skill brings its own split.")
        return self

    @override
    def invoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> list[Document]:
        return self._call_with_config(self._split_one, input, self._traced_config(config), run_type="llm")

    @override
    async def ainvoke(self, input: Any, config: RunnableConfig | None = None, **kwargs: Any) -> list[Document]:
        return await self._acall_with_config(self._asplit_one, input, self._traced_config(config), run_type="llm")

    def _split_one(self, input: Any) -> list[Document]:
        text, metadata = text_and_metadata(input)
        try:
            response = self._client.split(text=text, split=self.split, skill=self.skill)
        except wity.WityError as err:
            raise convert_error(err) from err
        return _documents(record_usage(response), metadata)

    async def _asplit_one(self, input: Any) -> list[Document]:
        text, metadata = text_and_metadata(input)
        try:
            response = await self._async_client().split(text=text, split=self.split, skill=self.skill)
        except wity.WityError as err:
            raise convert_error(err) from err
        return _documents(record_usage(response), metadata)

    def split_text(self, text: str) -> list[str]:
        """The items of ``text``, as text."""
        return [doc.page_content for doc in self.invoke(text)]

    def create_documents(self, texts: list[str], metadatas: list[dict[str, Any]] | None = None) -> list[Document]:
        """Split each text. Each item gets the metadata at the same position in ``metadatas``."""
        if metadatas is not None and len(metadatas) != len(texts):
            raise ValueError("metadatas must have one entry per text.")
        metadatas = metadatas or [{} for _ in texts]
        return self.split_documents(
            [Document(page_content=text, metadata=meta) for text, meta in zip(texts, metadatas, strict=True)]
        )

    def split_documents(self, documents: Iterable[Document]) -> list[Document]:
        """Split each document, one request at a time. Use ``batch`` to split several at once."""
        return [item for doc in documents for item in self.invoke(doc)]

    @override
    def transform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        return self.split_documents(documents)

    @override
    async def atransform_documents(self, documents: Sequence[Document], **kwargs: Any) -> Sequence[Document]:
        return [item for doc in documents for item in await self.ainvoke(doc)]


def _documents(response: SplitResponse, metadata: dict[str, Any]) -> list[Document]:
    return [
        Document(page_content=item.text, metadata={**metadata, "start_index": item.start}) for item in response.items
    ]
