"""The update notice: once-a-day checks and notices, and when it stays quiet. No network."""

from __future__ import annotations

import io
import sys
import types
from collections.abc import Callable
from typing import Any

import pytest

from langchain_wity import WityClassifier, WityExtractor, WitySplitter
from langchain_wity._update_notice import DAY, HOUR, UpdateDeps, default_deps, is_newer, update_notice

from .conftest import TEST_KEY

NOTICE = "\n  langchain-wity 0.3.0 is out. You have 0.2.0.\n  Update with: pip install -U langchain-wity\n\n"


class Fake:
    """A fake clock, cache file and PyPI. Checks run when ``finish()`` is called, like a thread that got time."""

    def __init__(self, latest: str | None, saved: dict[str, Any] | None = None) -> None:
        self.clock = 1_000_000.0
        self.latest = latest
        self.saved = saved
        self.fetches = 0
        self.pending: list[Callable[[], None]] = []

    def fetch(self) -> str | None:
        self.fetches += 1
        return self.latest

    def write(self, state: dict[str, Any]) -> None:
        self.saved = dict(state)

    @property
    def deps(self) -> UpdateDeps:
        return UpdateDeps(
            now=lambda: self.clock,
            read=lambda: self.saved,
            write=self.write,
            fetch_latest=self.fetch,
            start=self.pending.append,
        )

    def notice(self, current: str = "0.2.0") -> str | None:
        return update_notice("langchain-wity", current, "pip install -U langchain-wity", self.deps)

    def finish(self) -> None:
        for work in self.pending:
            work()
        self.pending.clear()


def test_first_run_checks_pypi_shows_nothing_and_saves_what_it_found() -> None:
    fake = Fake("0.3.0")
    assert fake.notice() is None
    fake.finish()
    assert fake.fetches == 1
    assert fake.saved == {"latest": "0.3.0", "checked_at": 1_000_000.0, "tried_at": 1_000_000.0}


def test_next_run_shows_the_notice_from_the_last_check() -> None:
    fake = Fake("0.3.0")
    fake.notice()
    fake.finish()
    fake.clock += 60
    assert fake.notice() == NOTICE


def test_checks_pypi_at_most_once_a_day() -> None:
    fake = Fake("0.3.0")
    fake.notice()
    fake.finish()
    fake.clock += DAY - 1
    fake.notice()
    fake.finish()
    assert fake.fetches == 1
    fake.clock += 1
    fake.notice()
    fake.finish()
    assert fake.fetches == 2


def test_a_check_cut_short_is_tried_again_after_an_hour_not_sooner() -> None:
    fake = Fake("0.3.0")
    fake.notice()
    fake.pending.clear()  # the program exited before the check finished
    fake.clock += HOUR - 1
    fake.notice()
    assert len(fake.pending) == 0
    fake.clock += 1
    fake.notice()
    assert len(fake.pending) == 1


def test_a_check_where_pypi_didnt_answer_counts_for_the_day() -> None:
    fake = Fake(None, {"latest": "0.2.0"})
    fake.notice()
    fake.finish()
    assert fake.saved is not None
    assert fake.saved["latest"] == "0.2.0"
    fake.clock += 2 * HOUR
    fake.notice()
    assert fake.pending == []


def test_a_check_that_raises_still_counts_and_never_escapes() -> None:
    fake = Fake("0.3.0")

    def broken() -> str | None:
        raise OSError("no network")

    fake.fetch = broken  # type: ignore[method-assign]
    fake.notice()
    fake.finish()
    assert fake.saved is not None
    assert "checked_at" in fake.saved


def test_shows_the_notice_at_most_once_a_day() -> None:
    fake = Fake("0.3.0", {"latest": "0.3.0", "checked_at": 1_000_000.0})
    assert fake.notice() == NOTICE
    assert fake.notice() is None
    fake.clock += DAY
    assert fake.notice() == NOTICE


def test_quiet_when_you_have_the_newest_version() -> None:
    fake = Fake("0.2.0", {"latest": "0.2.0", "checked_at": 1_000_000.0})
    assert fake.notice() is None


def test_never_raises_even_when_the_cache_cant_be_read() -> None:
    fake = Fake("0.3.0")

    def broken() -> dict[str, Any] | None:
        raise ValueError("broken")

    deps = fake.deps
    deps.read = broken
    assert update_notice("langchain-wity", "0.2.0", "pip install -U langchain-wity", deps) is None


@pytest.mark.parametrize(
    ("latest", "current", "expected"),
    [
        ("0.3.0", "0.2.0", True),
        ("0.2.1", "0.2.0", True),
        ("1.0.0", "0.9.9", True),
        ("0.10.0", "0.9.0", True),
        ("0.2.0", "0.2.0", False),
        ("0.1.9", "0.2.0", False),
        ("0.3.0b1", "0.2.0", False),
        ("0.2.0", "0.2.0rc1", True),
        ("nonsense", "0.2.0", False),
    ],
)
def test_is_newer(latest: str, current: str, expected: bool) -> None:
    assert is_newer(latest, current) is expected


class Terminal(io.StringIO):
    def isatty(self) -> bool:
        return True


def at_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    # Inside each test, not a fixture: pytest puts its own stdout back between a fixture and the test.
    monkeypatch.setattr("sys.stdout", Terminal())
    monkeypatch.setattr("sys.stderr", Terminal())
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("WITY_NO_UPDATE_CHECK", raising=False)


def test_runs_at_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    at_terminal(monkeypatch)
    assert default_deps("langchain-wity") is not None


def test_off_with_wity_no_update_check(monkeypatch: pytest.MonkeyPatch) -> None:
    at_terminal(monkeypatch)
    monkeypatch.setenv("WITY_NO_UPDATE_CHECK", "1")
    assert default_deps("langchain-wity") is None


def test_off_in_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    at_terminal(monkeypatch)
    monkeypatch.setenv("CI", "true")
    assert default_deps("langchain-wity") is None


def test_off_when_output_is_piped(monkeypatch: pytest.MonkeyPatch) -> None:
    at_terminal(monkeypatch)
    monkeypatch.setattr("sys.stdout", io.StringIO())
    assert default_deps("langchain-wity") is None


MAKERS = [
    lambda: WityClassifier(api_key=TEST_KEY),
    lambda: WitySplitter(api_key=TEST_KEY),
    lambda: WityExtractor(fields={"who": "Who signs"}, api_key=TEST_KEY),
]


@pytest.mark.parametrize("make", MAKERS)
def test_hides_the_sdk_notice(monkeypatch: pytest.MonkeyPatch, make: Callable[[], object]) -> None:
    # wity-sdk would say to upgrade it alone, past the version langchain-wity pins.
    sdk_notice = types.ModuleType("wity._update_notice")
    sdk_notice._shown = set()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "wity._update_notice", sdk_notice)
    make()
    assert sdk_notice._shown == {"wity-sdk"}


@pytest.mark.parametrize("make", MAKERS)
def test_works_with_an_sdk_that_has_no_notice(monkeypatch: pytest.MonkeyPatch, make: Callable[[], object]) -> None:
    # wity-sdk 0.2.0 has no update notice.
    monkeypatch.setitem(sys.modules, "wity._update_notice", None)
    make()
