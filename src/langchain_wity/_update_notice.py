"""Tell someone at a terminal when a newer version of this package is on PyPI.

- Only when both stdout and stderr are terminals, and not in CI. Servers, pipes and scripts never see it.
- Checks PyPI at most once a day, in a background thread. Nothing ever waits for the check.
  A check cut short because the program exited first is tried again after an hour.
- Shows the notice at most once a day, on stderr, from what an earlier check found.
- ``WITY_NO_UPDATE_CHECK=1`` turns it off. Every Wity package reads the same variable.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

HOUR = 60 * 60
DAY = 24 * HOUR
_CHECK_TIMEOUT = 5.0


@dataclass
class UpdateDeps:
    """Everything the notice touches, so tests can fake the clock, the file and PyPI."""

    now: Callable[[], float]
    read: Callable[[], dict[str, Any] | None]
    write: Callable[[dict[str, Any]], None]
    fetch_latest: Callable[[], str | None]
    """The newest version on PyPI, or None if PyPI couldn't be reached."""
    start: Callable[[Callable[[], None]], None]
    """Runs the check without waiting for it."""


def update_notice(name: str, current: str, update_command: str, deps: UpdateDeps) -> str | None:
    """Start a check if the last one was over a day ago, and return the notice to show, if one is due.

    Never raises and never waits: a broken cache file or a failed check just means no notice.
    State is kept in seconds: ``checked_at`` when a check last finished (whether or not PyPI answered),
    ``tried_at`` when one last started, ``shown_at`` when the notice was last shown, and ``latest``.
    """
    try:
        now = deps.now()
        state = deps.read() or {}
        notice = None
        latest = state.get("latest")
        if isinstance(latest, str) and is_newer(latest, current) and not _within(state.get("shown_at"), now, DAY):
            notice = f"\n  {name} {latest} is out. You have {current}.\n  Update with: {update_command}\n\n"
            state["shown_at"] = now

        due = not _within(state.get("checked_at"), now, DAY) and not _within(state.get("tried_at"), now, HOUR)
        if due:
            state["tried_at"] = now
        deps.write(state)

        if due:

            def check() -> None:
                try:
                    found = deps.fetch_latest()
                except Exception:
                    found = None
                # Finished, so it counts for the day even when PyPI didn't answer. Keep what an earlier check found.
                saved = deps.read() or state
                deps.write({**saved, "checked_at": deps.now(), **({"latest": found} if found else {})})

            deps.start(check)
        return notice
    except Exception:
        return None


def _within(then: object, now: float, seconds: float) -> bool:
    return isinstance(then, (int, float)) and 0 <= now - then < seconds


_VERSION = re.compile(r"^(\d+)\.(\d+)\.(\d+)(\S*)$")


def is_newer(latest: str, current: str) -> bool:
    """True when ``latest`` is a later release than ``current``. Pre-releases on PyPI never count as newer."""
    a = _VERSION.match(latest.strip())
    b = _VERSION.match(current.strip())
    if not a or not b or a.group(4):
        return False
    for i in (1, 2, 3):
        diff = int(a.group(i)) - int(b.group(i))
        if diff:
            return diff > 0
    # Same numbers: a release is newer than the pre-release you have.
    return bool(b.group(4))


def default_deps(name: str) -> UpdateDeps | None:
    """The real clock, cache file and PyPI. None when the notice shouldn't run: off, in CI, or not at a terminal."""
    env = os.environ
    if env.get("WITY_NO_UPDATE_CHECK", "").strip() or env.get("CI", "").strip():
        return None
    if not _is_terminal(sys.stdout) or not _is_terminal(sys.stderr):
        return None

    base = env.get("XDG_CACHE_HOME", "").strip()
    if not base and sys.platform == "win32":
        base = env.get("LOCALAPPDATA", "").strip()
    path = (Path(base) if base else Path.home() / ".cache") / "wity" / f"update-pypi-{name}.json"

    def read() -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def write(state: dict[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(state), "utf-8")
        except OSError:
            pass  # A read-only home folder just means no notice.

    def fetch_latest() -> str | None:
        url = f"https://pypi.org/pypi/{name}/json"
        with urllib.request.urlopen(url, timeout=_CHECK_TIMEOUT) as res:  # a fixed https address
            found = json.load(res).get("info", {}).get("version")
        return found if isinstance(found, str) else None

    def start(work: Callable[[], None]) -> None:
        # A daemon thread never keeps the program running. If it exits first, the check just didn't finish.
        threading.Thread(target=work, name=f"{name}-update-check", daemon=True).start()

    return UpdateDeps(now=time.time, read=read, write=write, fetch_latest=fetch_latest, start=start)


def _is_terminal(stream: object) -> bool:
    try:
        return bool(stream.isatty())  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        return False


_shown: set[str] = set()


def show_update_notice_once(name: str, update_command: str) -> None:
    """Check once per process for this package, and print a due notice to stderr. Safe to call on every client."""
    if name in _shown:
        return
    _shown.add(name)
    try:
        current = version(name)
    except PackageNotFoundError:
        return
    deps = default_deps(name)
    if deps is None:
        return
    notice = update_notice(name, current, update_command, deps)
    if notice:
        sys.stderr.write(notice)
