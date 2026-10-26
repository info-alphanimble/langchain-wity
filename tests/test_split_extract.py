"""WitySplitter and WityExtractor: input forms, documents, options, errors and tracing."""

from __future__ import annotations

import pickle
from typing import Any

import httpx
import pytest
from langchain_core.documents import Document
from langchain_core.exceptions import ModelInvalidRequestError
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from langchain_wity import (
    ExtractResponse,
    SplitSpec,
    WityAuthenticationError,
    WityBadRequestError,
    WityExtractor,
    WitySplitter,
)

from .conftest import TEST_KEY, FakeWity, Recorder

EMAIL = "From: dana@example.com\nSeal test failed.\n\nFrom: marcus@example.com\nRerun tomorrow."
SPLIT_REPLY: dict[str, Any] = {
    "items": [
        {"start": 0, "end": 40, "text": EMAIL[0:40]},
        {"start": 42, "end": len(EMAIL), "text": EMAIL[42:]},
    ],
    "how": "model",
    "timing_ms": {"total": 84},
    "usage": {"input_tokens": 412, "output_tokens": 0},
}
CONTRACT = "Between Kestrel Foods Ltd. and the supplier, effective 14 March 2025."
EXTRACT_REPLY: dict[str, Any] = {
    "fields": {"party": ["Kestrel Foods Ltd."], "effective date": ["2025-03-14"]},
    "entities": [
        {"field": "party", "text": "Kestrel Foods Ltd.", "start": 8, "end": 26, "confidence": 0.98},
        {"field": "effective date", "text": "2025-03-14", "start": 55, "end": 68, "confidence": 0.91},
    ],
    "timing_ms": 503,
    "usage": {"input_tokens": 301, "output_tokens": 0},
}
FIELDS = {"party": "Who signs the contract", "effective date": "When the contract starts"}


@pytest.fixture
def fake() -> FakeWity:
    return FakeWity(default=httpx.Response(200, json=SPLIT_REPLY))


# ---------------------------------------------------------------------------
# WitySplitter
# ---------------------------------------------------------------------------


def test_split_text_returns_the_items(fake: FakeWity) -> None:
    items = fake.splitter(split="one item per email").split_text(EMAIL)
    assert items == [EMAIL[0:40], EMAIL[42:]]
    assert fake.requests[0].url.path == "/v1/split"
    assert fake.bodies[0] == {"text": EMAIL, "split": "one item per email"}


def test_invoke_gives_documents_with_their_offset(fake: FakeWity) -> None:
    docs = fake.splitter().invoke(EMAIL)
    assert [doc.metadata["start_index"] for doc in docs] == [0, 42]
    for doc in docs:
        start = doc.metadata["start_index"]
        assert EMAIL[start : start + len(doc.page_content)] == doc.page_content


def test_split_documents_keeps_each_documents_metadata(fake: FakeWity) -> None:
    source = [Document(page_content=EMAIL, metadata={"source": "thread-1"})] * 2
    docs = fake.splitter().split_documents(source)
    assert len(docs) == 4
    assert all(doc.metadata["source"] == "thread-1" for doc in docs)
    assert len(fake.requests) == 2


def test_create_documents_pairs_texts_and_metadata(fake: FakeWity) -> None:
    docs = fake.splitter().create_documents([EMAIL, EMAIL], [{"n": 1}, {"n": 2}])
    assert [doc.metadata["n"] for doc in docs] == [1, 1, 2, 2]
    with pytest.raises(ValueError, match="one entry per text"):
        fake.splitter().create_documents([EMAIL], [{}, {}])


def test_the_api_default_split_is_left_to_the_api(fake: FakeWity) -> None:
    fake.splitter().invoke(EMAIL)
    assert fake.bodies[0] == {"text": EMAIL}


def test_a_split_spec_is_sent_as_json(fake: FakeWity) -> None:
    fake.splitter(split=SplitSpec(unit="email", starts_at=["From:"])).invoke(EMAIL)
    assert fake.bodies[0]["split"] == {"unit": "email", "starts_at": ["From:"]}


def test_a_split_skill(fake: FakeWity) -> None:
    fake.splitter(skill="emails").invoke(EMAIL)
    assert fake.bodies[0] == {"text": EMAIL, "skill": "emails"}


def test_split_and_skill_together_are_refused() -> None:
    with pytest.raises(ValidationError, match="not both"):
        WitySplitter(split="one item per email", skill="emails", api_key=TEST_KEY)


def test_input_that_isnt_text_or_a_document_is_refused(fake: FakeWity) -> None:
    with pytest.raises(TypeError, match="text or a LangChain Document"):
        fake.splitter().invoke({"text": EMAIL})
    assert fake.requests == []


async def test_async_split(fake: FakeWity) -> None:
    splitter = fake.splitter()
    docs = await splitter.ainvoke(Document(page_content=EMAIL, metadata={"source": "a"}))
    assert [doc.metadata for doc in docs] == [{"source": "a", "start_index": 0}, {"source": "a", "start_index": 42}]
    docs = list(await splitter.atransform_documents([Document(page_content=EMAIL)]))
    assert len(docs) == 2
    await splitter.aclose()


def test_transform_documents(fake: FakeWity) -> None:
    docs = fake.splitter().transform_documents([Document(page_content=EMAIL)])
    assert [doc.page_content for doc in docs] == [EMAIL[0:40], EMAIL[42:]]


def test_works_in_a_pipeline(fake: FakeWity) -> None:
    chain = fake.splitter() | RunnableLambda(lambda docs: len(docs))
    assert chain.invoke(EMAIL) == 2


def test_split_errors_are_both_wity_and_langchain_errors(fake: FakeWity) -> None:
    fake.json({"error": "text must be 1 to 60000 characters"}, status=400)
    with pytest.raises(WityBadRequestError) as caught:
        fake.splitter().invoke(EMAIL)
    assert isinstance(caught.value, ModelInvalidRequestError)


def test_splitter_pickles_without_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WITY_API_KEY", TEST_KEY)
    splitter = pickle.loads(pickle.dumps(WitySplitter(split="one item per email")))
    assert splitter.split == "one item per email"


def test_split_is_traced_as_a_wity_call(fake: FakeWity) -> None:
    recorder = Recorder()
    fake.splitter().invoke(EMAIL, config={"callbacks": [recorder]})
    start = next(data for kind, data in recorder.events if kind == "start")
    assert start["metadata"]["ls_provider"] == "wity"


# ---------------------------------------------------------------------------
# WityExtractor
# ---------------------------------------------------------------------------


def test_extract_returns_the_sdk_response(fake: FakeWity) -> None:
    fake.json(EXTRACT_REPLY)
    res = fake.extractor(fields=FIELDS, types={"effective date": "date"}).invoke(CONTRACT)
    assert isinstance(res, ExtractResponse)
    assert res.fields["party"] == ["Kestrel Foods Ltd."]
    assert res.entities[1].text == "2025-03-14"
    assert fake.requests[0].url.path == "/v1/extract"
    assert fake.bodies[0] == {"text": CONTRACT, "fields": FIELDS, "types": {"effective date": "date"}}


def test_extract_reads_a_document(fake: FakeWity) -> None:
    fake.json(EXTRACT_REPLY)
    fake.extractor(fields=FIELDS).invoke(Document(page_content=CONTRACT, metadata={"source": "c.pdf"}))
    assert fake.bodies[0] == {"text": CONTRACT, "fields": FIELDS}


def test_no_fields_is_refused() -> None:
    with pytest.raises(ValidationError, match="at least one field"):
        WityExtractor(fields={}, api_key=TEST_KEY)


async def test_async_extract(fake: FakeWity) -> None:
    fake.json(EXTRACT_REPLY)
    fake.json(EXTRACT_REPLY)
    extractor = fake.extractor(fields=FIELDS)
    results = await extractor.abatch([CONTRACT, CONTRACT])
    assert [res.fields["party"] for res in results] == [["Kestrel Foods Ltd."]] * 2
    await extractor.aclose()


def test_extract_errors_are_converted(fake: FakeWity) -> None:
    fake.json({"error": "bad key"}, status=401)
    with pytest.raises(WityAuthenticationError) as caught:
        fake.extractor(fields=FIELDS).invoke(CONTRACT)
    assert TEST_KEY not in str(caught.value)
