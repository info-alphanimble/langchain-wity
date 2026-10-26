"""What every Wity Runnable shares: the connection options, the clients, and tracing."""

from __future__ import annotations

import asyncio
import logging
import weakref
from typing import Any, Protocol, TypeVar

import httpx
import wity
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig, RunnableSerializable
from langchain_core.runnables.config import ensure_config
from langsmith.run_helpers import get_current_run_tree
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr, field_validator
from typing_extensions import override
from wity import Usage

from ._update_notice import show_update_notice_once

logger = logging.getLogger("langchain_wity")

_PROVIDER = "wity"
_MODEL = "wity-1"


class _HasUsage(Protocol):
    @property
    def usage(self) -> Usage: ...


Output = TypeVar("Output")
Response = TypeVar("Response", bound=_HasUsage)


class WityRunnable(RunnableSerializable[Any, Output]):
    """Connection options and clients for a Runnable that calls Wity."""

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
    _async_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, wity.AsyncWityClient] = PrivateAttr(
        default_factory=weakref.WeakKeyDictionary
    )
    """One async client per event loop. A connection opened in one loop can't be used from another."""

    @field_validator("api_key")
    @classmethod
    def _hide_key(cls, api_key: SecretStr | str | None) -> SecretStr | None:
        return SecretStr(api_key) if isinstance(api_key, str) else api_key

    @override
    def model_post_init(self, context: Any, /) -> None:
        hide_sdk_notice()
        # Made here, so a missing key or a bad address fails now, not at the first call.
        self._client = wity.WityClient(**self._options(), http_client=self.http_client)
        show_update_notice_once("langchain-wity", "pip install -U langchain-wity")

    def _options(self) -> dict[str, Any]:
        key = self.api_key.get_secret_value() if isinstance(self.api_key, SecretStr) else None
        return {"api_key": key, "base_url": self.base_url, "timeout": self.timeout, "max_retries": self.max_retries}

    def _async_client(self) -> wity.AsyncWityClient:
        """The async client for the running event loop, made on first use in that loop."""
        loop = asyncio.get_running_loop()
        client = self._async_clients.get(loop)
        if client is None:
            # A client holds its loop through its open connections, so the weak key alone never lets a
            # closed loop go. Drop those clients here, or each `asyncio.run` would leave a socket open.
            for old_loop in [old for old in self._async_clients if old.is_closed()]:
                del self._async_clients[old_loop]
            client = wity.AsyncWityClient(**self._options(), http_client=self.http_async_client)
            self._async_clients[loop] = client
        return client

    def close(self) -> None:
        """Close the connections ``invoke`` and ``batch`` use. Clients you passed in stay open."""
        self._client.close()

    async def aclose(self) -> None:
        """Close the connections ``ainvoke`` and ``abatch`` use in this event loop. Clients you passed in stay open."""
        client = self._async_clients.pop(asyncio.get_running_loop(), None)
        if client is not None:
            await client.close()

    def _settings(self) -> dict[str, Any]:
        """The options this Runnable was created with."""
        return {name: getattr(self, name) for name in self.model_fields_set}

    def __reduce__(self) -> tuple[Any, ...]:
        # Pickle the options, not the open connections. A process pool or Ray worker makes its own.
        return (_rebuild, (type(self), self._settings()))

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> Any:
        # A copy gets its own connections. Clients you passed in are shared, not copied.
        return type(self)(**self._settings())

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


def hide_sdk_notice() -> None:
    """Keep wity-sdk's own notice quiet. It says to upgrade wity-sdk alone, which could pass the version we pin."""
    try:
        from wity._update_notice import _shown  # type: ignore[import-not-found,unused-ignore]  # wity-sdk 0.2.1+
    except ImportError:
        return
    _shown.add("wity-sdk")


def _rebuild(cls: type[WityRunnable[Any]], settings: dict[str, Any]) -> WityRunnable[Any]:
    return cls(**settings)


def record_usage(response: Response) -> Response:
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


def text_and_metadata(input: Any) -> tuple[str, dict[str, Any]]:
    """The text of a Split or Extract input, and the metadata to keep. Input is text or a ``Document``."""
    if isinstance(input, str):
        return input, {}
    if isinstance(input, Document):
        return input.page_content, dict(input.metadata)
    raise TypeError(f"Input must be text or a LangChain Document, not {type(input).__name__}.")
