"""LangChain integration for Wity, the typed-decision API.

The middlewares are in ``langchain_wity.middleware``.
"""

from wity import (
    APIError,
    Choice,
    ChoiceAnswer,
    ExtractEntity,
    ExtractResponse,
    ExtractType,
    Noul,
    NoulAnswer,
    NoulCriteria,
    Question,
    Reasoning,
    ReasoningInfo,
    Score,
    ScoreAnswer,
    SplitItem,
    SplitResponse,
    SplitSpec,
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
from .extractor import WityExtractor
from .splitter import WitySplitter

__version__ = "0.2.0"

__all__ = [
    "APIError",
    "Choice",
    "ChoiceAnswer",
    "ExtractEntity",
    "ExtractResponse",
    "ExtractType",
    "Noul",
    "NoulAnswer",
    "NoulCriteria",
    "Question",
    "Reasoning",
    "ReasoningInfo",
    "Score",
    "ScoreAnswer",
    "SplitItem",
    "SplitResponse",
    "SplitSpec",
    "SystemOneResponse",
    "Usage",
    "WityAPIError",
    "WityAuthenticationError",
    "WityBadRequestError",
    "WityClassifier",
    "WityConnectionError",
    "WityContextOverflowError",
    "WityError",
    "WityExtractor",
    "WityInsufficientBalanceError",
    "WityInternalServerError",
    "WityRateLimitError",
    "WitySplitter",
    "WityTimeoutError",
    "choice",
    "noul",
    "score",
]
