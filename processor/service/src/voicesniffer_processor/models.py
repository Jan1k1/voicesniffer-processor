from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
    text: str
    language: str


class UnsupportedTranscriptionLanguage(ValueError):
    """The speech model on this node was never trained on the language asked for.

    It lives here, beside ``TranscriptionResult``, because this is the module
    both sides of that call already import: the adapter raises it and the HTTP
    handler answers it, and neither should have to import the other.

    It exists because the two lists that decide what a node can do are set
    independently. ``models.toml`` says what the speech model was trained on and
    ``VOICESNIFFER_RULES_DIR`` says what rules are installed, and a code can be
    in the second without being in the first. The handler checks the rule packs
    and then hands the code to the transcriber, so that combination used to come
    back ``500 internal_error`` with nothing in the log: a deployment mistake
    reported as a crash. Named, it is answered the way the missing-pack case is,
    because from the caller's side it is the same fact -- this processor cannot
    moderate that language.

    A ``ValueError`` so that a caller which only knows the old contract, and
    every existing test written against it, still catches this.
    """

    def __init__(self, language: str) -> None:
        self.language = language
        super().__init__(f"unsupported transcription language: {language}")


class VerdictMatch(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: str
    category: str
    severity: int = Field(ge=1, le=3)
    matched_text: str
    context_required: bool


class ModerationVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str
    transcript: str
    language: str
    matches: tuple[VerdictMatch, ...]
    severity: int = Field(ge=0, le=3)
    confidence: float = Field(ge=0.0, le=1.0)
    processing_ms: int = Field(ge=0)


class ErrorResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    request_id: str | None
    code: str
    message: str
