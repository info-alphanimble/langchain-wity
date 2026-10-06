"""LangChain integration for Wity, the typed-decision API.

The middlewares are in ``langchain_wity.middleware``.
"""

from wity import (
    APIError,
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    NoulCriteria,
    Question,
    Reasoning,
    ReasoningInfo,
    Score,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
    WityError,
    choice,
    noul,
    score,
)

from .classifier import WityClassifier
from .errors import (
    WityAPIError,
    WityAuthenticationError,
    WityBadRequestError,
    WityConnectionError,
    WityContextOverflowError,
    WityInsufficientBalanceError,
    WityInternalServerError,
    WityRateLimitError,
    WityTimeoutError,
)

__version__ = "0.1.0"

__all__ = [
    "APIError",
    "Choice",
    "ChoiceAnswer",
    "Noul",
    "NoulAnswer",
    "NoulCriteria",
    "Question",
    "Reasoning",
    "ReasoningInfo",
    "Score",
    "ScoreAnswer",
    "SystemOneResponse",
    "Usage",
    "WityAPIError",
    "WityAuthenticationError",
    "WityBadRequestError",
    "WityClassifier",
    "WityConnectionError",
    "WityContextOverflowError",
    "WityError",
    "WityInsufficientBalanceError",
    "WityInternalServerError",
    "WityRateLimitError",
    "WityTimeoutError",
    "choice",
    "noul",
    "score",
]
