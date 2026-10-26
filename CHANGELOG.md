# Changelog

All notable changes to `langchain-wity`. Versions follow [semver](https://semver.org). Before 1.0, a minor version can add features.

## 0.2.0 (2026-10-26)

### Added

- **`WitySplitter`** cuts documents into whole items, like emails in a thread or clauses in a contract. It's a Runnable and a document transformer: use `split_text`, `split_documents`, or put it in a pipeline.
  - Every item is an exact slice of the text. Each new `Document` keeps its source's metadata, plus `start_index`.
  - `split` says what one item is, in plain words or as a `SplitSpec`. `skill=` splits the way a saved split skill does.
- **`WityExtractor`** pulls named fields out of text or a `Document`, and returns the `wity-sdk` response. `res.fields` has the values as written. `res.entities` adds where each value was found and a confidence. `types` normalises a field as a `"date"`, `"duration"` or `"number"`.
- **Update notice.** At a terminal, `langchain-wity` tells you on stderr when a newer version is on PyPI, at most once a day. The check runs in a background thread, so nothing waits for it. It stays quiet in CI, when output is piped, and on servers. `WITY_NO_UPDATE_CHECK=1` turns it off.
- `SplitSpec`, `SplitItem`, `SplitResponse`, `ExtractType`, `ExtractEntity` and `ExtractResponse`, exported from `langchain_wity`.

### Changed

- Needs `wity-sdk` 0.2.0 or newer.

## 0.1.0 (2026-10-12)

First release.

- `WityClassifier`: a Runnable that asks Wity typed questions, with a probability for every option. Works with `invoke`, `batch`, their async forms, and LangSmith tracing.
- `ModelRouterMiddleware`: Wity picks which LLM handles each agent run.
- `AutoModeMiddleware`: Wity checks risky tool calls before they run.
- Messages as state, `reasoning` modes and `max_latency_ms`.
- Errors as `Wity…Error` exceptions, like `WityRateLimitError` and `WityContextOverflowError`. Classifiers and errors can be pickled.
