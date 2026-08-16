from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from numpy.typing import NDArray
from voicesniffer_runtime.model_store import ModelSpec
from voicesniffer_runtime.transcribers import load_transcriber

from voicesniffer_processor.models import (
    TranscriptionResult,
    UnsupportedTranscriptionLanguage,
)

Loader = Callable[[ModelSpec, Path, str, int], Any]


@dataclass(frozen=True, slots=True)
class SherpaTranscriptionAdapter:
    model: ModelSpec
    transcribers: Mapping[str, Any]

    def __call__(self, samples: NDArray, requested_language: str) -> TranscriptionResult:
        transcriber = self._transcriber_for(requested_language)
        transcription = transcriber.transcribe(samples, 16_000)
        return self._result(transcription, requested_language)

    def batch(
        self,
        batch: Sequence[NDArray],
        requested_languages: Sequence[str],
    ) -> list[TranscriptionResult]:
        """Decode a micro-batch in as few model passes as possible.

        Multilingual single-recognizer families (Parakeet, Moonshine) collapse to
        exactly one pass regardless of the mix of requested languages, because
        every language maps to the same recognizer. Whisper keeps one recognizer
        per language, so it batches per group -- still a win, just a smaller one.
        """
        if len(batch) != len(requested_languages):
            raise ValueError("batch and language counts must match")
        groups: dict[int, list[int]] = {}
        transcribers: dict[int, Any] = {}
        for index, language in enumerate(requested_languages):
            transcriber = self._transcriber_for(language)
            key = id(transcriber)
            transcribers[key] = transcriber
            groups.setdefault(key, []).append(index)

        results: list[TranscriptionResult | None] = [None] * len(batch)
        for key, indexes in groups.items():
            transcriptions = transcribers[key].transcribe_batch(
                [batch[index] for index in indexes],
                16_000,
            )
            for index, transcription in zip(indexes, transcriptions, strict=True):
                results[index] = self._result(transcription, requested_languages[index])
        return [result for result in results if result is not None]

    def _transcriber_for(self, requested_language: str) -> Any:
        if requested_language == "auto":
            return self.transcribers[self.model.languages[0]]
        try:
            return self.transcribers[requested_language]
        except KeyError as exception:
            # Named rather than a plain ValueError so the HTTP handler can tell
            # this apart from a genuine fault and answer `language_unsupported`
            # instead of `internal_error`. See the exception's own docstring.
            raise UnsupportedTranscriptionLanguage(requested_language) from exception

    def _result(self, transcription: Any, requested_language: str) -> TranscriptionResult:
        if requested_language == "auto" and len(self.model.languages) == 1:
            language = self.model.languages[0]
        elif requested_language == "auto":
            language = transcription.language or "auto"
        else:
            language = requested_language
        if language != "auto" and language not in self.model.languages:
            raise ValueError("unsupported detected language")
        return TranscriptionResult(text=transcription.text.strip(), language=language)


def load_transcription_adapter(
    model_id: str,
    model_dir: Path,
    *,
    threads: int,
    registry: Mapping[str, ModelSpec],
    loader: Loader = load_transcriber,
) -> SherpaTranscriptionAdapter:
    try:
        model = registry[model_id]
    except KeyError as exception:
        raise ValueError(f"unknown model: {model_id}") from exception

    if model.family == "whisper":
        transcribers = {
            language: loader(model, model_dir, language, threads) for language in model.languages
        }
    else:
        transcriber = loader(model, model_dir, model.languages[0], threads)
        transcribers = dict.fromkeys(model.languages, transcriber)
    return SherpaTranscriptionAdapter(model, transcribers)
