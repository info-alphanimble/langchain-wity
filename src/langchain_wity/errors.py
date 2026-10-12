"""Wity errors that are also LangChain's model errors.

Each class extends both the ``wity-sdk`` error and the matching ``langchain_core`` error.
So ``except wity.RateLimitError`` and ``except ModelRateLimitError`` both catch a 429.
"""

from __future__ import annotations

import wity
from langchain_core.exceptions import (
    ContextOverflowError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelConnectionError,
    ModelInvalidRequestError,
    ModelRateLimitError,
    ModelTimeoutError,
)


class _Picklable:
    """Lets the error be pickled, for example to cross a process pool. The SDK's errors need extra arguments."""

    args: tuple[object, ...]

    def __reduce__(self) -> tuple[object, ...]:
        return (_rebuild, (type(self), self.args, self.__dict__))


def _rebuild(cls: type[wity.WityError], args: tuple[object, ...], fields: dict[str, object]) -> wity.WityError:
    new = cls.__new__(cls)
    new.args = args
    new.__dict__.update(fields)
    return new


class WityBadRequestError(_Picklable, wity.BadRequestError, ModelInvalidRequestError):
    """400: the request is wrong. The message says which part."""


class WityContextOverflowError(WityBadRequestError, ContextOverflowError):
    """400: the state is longer than Wity accepts."""


class WityAuthenticationError(_Picklable, wity.AuthenticationError, ModelAuthenticationError):
    """401: the API key is missing or wrong."""


class WityInsufficientBalanceError(_Picklable, wity.InsufficientBalanceError, ModelAPIError):
    """402: the account has no balance left."""

    # Trying again won't help until someone tops up the account.
    is_retryable = False


class WityRateLimitError(_Picklable, wity.RateLimitError, ModelRateLimitError):
    """429: too many requests, after the SDK's retries ran out."""


class WityInternalServerError(_Picklable, wity.InternalServerError, ModelAPIError):
    """5xx: a problem on Wity's side, after the SDK's retries ran out."""


class WityAPIError(_Picklable, wity.APIError, ModelAPIError):
    """Any other error status."""


class WityConnectionError(_Picklable, wity.APIConnectionError, ModelConnectionError):
    """Wity couldn't be reached."""


class WityTimeoutError(_Picklable, wity.APITimeoutError, ModelTimeoutError):
    """Wity didn't answer within the timeout."""


def _is_state_too_long(err: wity.BadRequestError) -> bool:
    # The server says: "state must be a possibly empty string of at most <n> characters".
    message = str(err)
    return "state must be" in message and "at most" in message


def convert_error(err: wity.WityError) -> wity.WityError:
    """Return the matching class from this module, with the same message and fields.

    Errors that aren't about the API (a missing key, bad input) come back unchanged.
    """
    new_class: type[wity.WityError]
    if isinstance(err, wity.BadRequestError):
        new_class = WityContextOverflowError if _is_state_too_long(err) else WityBadRequestError
    elif isinstance(err, wity.AuthenticationError):
        new_class = WityAuthenticationError
    elif isinstance(err, wity.InsufficientBalanceError):
        new_class = WityInsufficientBalanceError
    elif isinstance(err, wity.RateLimitError):
        new_class = WityRateLimitError
    elif isinstance(err, wity.InternalServerError):
        new_class = WityInternalServerError
    elif isinstance(err, wity.APIError):
        new_class = WityAPIError
    elif isinstance(err, wity.APITimeoutError):
        new_class = WityTimeoutError
    elif isinstance(err, wity.APIConnectionError):
        new_class = WityConnectionError
    else:
        return err
    # Copy instead of calling __init__, so the message stays exactly the SDK's.
    return _rebuild(new_class, err.args, err.__dict__)
