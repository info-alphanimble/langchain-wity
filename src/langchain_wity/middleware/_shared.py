"""Checks both middlewares share."""

from __future__ import annotations

from ..classifier import WityClassifier


def check_classifier(classifier: WityClassifier) -> WityClassifier:
    """The middlewares send their own questions, so the classifier must not have any."""
    if not isinstance(classifier, WityClassifier):
        raise TypeError("classifier must be a WityClassifier.")
    if classifier.questions is not None:
        raise ValueError("Pass a WityClassifier without questions. The middleware sends its own.")
    return classifier
