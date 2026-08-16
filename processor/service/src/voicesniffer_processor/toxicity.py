from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ToxicityResult:
    category: str
    severity: int
    confidence: float
    matched_text: str
    language: str


ToxicityClassifier = Callable[[str, str], ToxicityResult | None]
