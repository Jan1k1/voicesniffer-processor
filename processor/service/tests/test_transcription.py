from pathlib import Path

import numpy as np
import pytest
from voicesniffer_runtime.model_store import ModelSpec
from voicesniffer_runtime.transcribers import Transcription

from voicesniffer_processor.models import UnsupportedTranscriptionLanguage
from voicesniffer_processor.transcription import load_transcription_adapter


class FakeTranscriber:
    def __init__(self, language: str | None) -> None:
        self.language = language
        self.calls: list[tuple[np.ndarray, int]] = []

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> Transcription:
        self.calls.append((samples, sample_rate))
        return Transcription(" rozpoznaná řeč ", 0.01, self.language)


def test_loads_model_once_and_returns_detected_language(tmp_path: Path) -> None:
    model = ModelSpec(
        model_id="parakeet",
        family="nemo-transducer",
        archive_url="https://example.test/model.tar.bz2",
        archive_sha256="a" * 64,
        archive_root="package",
        languages=("en", "cs"),
        license_name="CC-BY-4.0",
        license_url="https://example.test/license",
        files={
            "encoder": "encoder.onnx",
            "decoder": "decoder.onnx",
            "joiner": "joiner.onnx",
            "tokens": "tokens.txt",
        },
    )
    loaded: list[tuple[ModelSpec, Path, str, int]] = []
    transcriber = FakeTranscriber("cs")

    def loader(
        spec: ModelSpec,
        model_dir: Path,
        language: str,
        threads: int,
    ) -> FakeTranscriber:
        loaded.append((spec, model_dir, language, threads))
        return transcriber

    adapter = load_transcription_adapter(
        "parakeet",
        tmp_path,
        threads=2,
        registry={"parakeet": model},
        loader=loader,
    )
    samples = np.zeros(1_600, dtype=np.float32)

    result = adapter(samples, "auto")

    assert result.text == "rozpoznaná řeč"
    assert result.language == "cs"
    assert loaded == [(model, tmp_path, "en", 2)]
    assert transcriber.calls == [(samples, 16_000)]


def test_uses_explicit_supported_language_for_rule_routing(tmp_path: Path) -> None:
    model = ModelSpec(
        model_id="english",
        family="moonshine-v2",
        archive_url="https://example.test/model.tar.bz2",
        archive_sha256="a" * 64,
        archive_root="package",
        languages=("en",),
        license_name="MIT",
        license_url="https://example.test/license",
        files={"encoder": "encoder.ort", "decoder": "decoder.ort", "tokens": "tokens.txt"},
    )
    transcriber = FakeTranscriber("en")
    adapter = load_transcription_adapter(
        "english",
        tmp_path,
        threads=1,
        registry={"english": model},
        loader=lambda *_args: transcriber,
    )

    result = adapter(np.zeros(320, dtype=np.float32), "en")

    assert result.language == "en"


def test_auto_uses_only_supported_language_when_runtime_cannot_detect(tmp_path: Path) -> None:
    model = ModelSpec(
        model_id="english",
        family="moonshine-v2",
        archive_url="https://example.test/model.tar.bz2",
        archive_sha256="a" * 64,
        archive_root="package",
        languages=("en",),
        license_name="MIT",
        license_url="https://example.test/license",
        files={"encoder": "encoder.ort", "decoder": "decoder.ort", "tokens": "tokens.txt"},
    )
    transcriber = FakeTranscriber(None)
    adapter = load_transcription_adapter(
        "english",
        tmp_path,
        threads=1,
        registry={"english": model},
        loader=lambda *_args: transcriber,
    )

    result = adapter(np.zeros(320, dtype=np.float32), "auto")

    assert result.language == "en"


def test_auto_preserves_unknown_language_for_multilingual_model(tmp_path: Path) -> None:
    model = ModelSpec(
        model_id="parakeet",
        family="nemo-transducer",
        archive_url="https://example.test/model.tar.bz2",
        archive_sha256="a" * 64,
        archive_root="package",
        languages=("en", "cs"),
        license_name="CC-BY-4.0",
        license_url="https://example.test/license",
        files={
            "encoder": "encoder.onnx",
            "decoder": "decoder.onnx",
            "joiner": "joiner.onnx",
            "tokens": "tokens.txt",
        },
    )
    transcriber = FakeTranscriber(None)
    adapter = load_transcription_adapter(
        "parakeet",
        tmp_path,
        threads=1,
        registry={"parakeet": model},
        loader=lambda *_args: transcriber,
    )

    result = adapter(np.zeros(320, dtype=np.float32), "auto")

    assert result.language == "auto"


class FakeBatchTranscriber(FakeTranscriber):
    def __init__(self, language: str | None) -> None:
        super().__init__(language)
        self.batches: list[int] = []

    def transcribe_batch(self, batch, sample_rate: int) -> list[Transcription]:
        self.batches.append(len(batch))
        return [
            Transcription(f" utterance {index} ", 0.01, self.language)
            for index in range(len(batch))
        ]


def _parakeet_spec() -> ModelSpec:
    return ModelSpec(
        model_id="parakeet",
        family="nemo-transducer",
        archive_url="https://example.test/model.tar.bz2",
        archive_sha256="a" * 64,
        archive_root="package",
        languages=("en", "cs"),
        license_name="CC-BY-4.0",
        license_url="https://example.test/license",
        files={
            "encoder": "encoder.onnx",
            "decoder": "decoder.onnx",
            "joiner": "joiner.onnx",
            "tokens": "tokens.txt",
        },
    )


def test_batch_uses_one_model_pass_for_a_multilingual_recognizer(tmp_path: Path) -> None:
    transcriber = FakeBatchTranscriber("cs")
    adapter = load_transcription_adapter(
        "parakeet",
        tmp_path,
        threads=1,
        registry={"parakeet": _parakeet_spec()},
        loader=lambda *_args: transcriber,
    )
    batch = [np.zeros(320, dtype=np.float32) for _ in range(4)]

    results = adapter.batch(batch, ["en", "cs", "auto", "en"])

    # One recognizer serves every language, so a mixed batch stays a single pass.
    assert transcriber.batches == [4]
    assert [result.text for result in results] == [
        "utterance 0",
        "utterance 1",
        "utterance 2",
        "utterance 3",
    ]
    assert [result.language for result in results] == ["en", "cs", "cs", "en"]


def test_batch_rejects_mismatched_language_count(tmp_path: Path) -> None:
    adapter = load_transcription_adapter(
        "parakeet",
        tmp_path,
        threads=1,
        registry={"parakeet": _parakeet_spec()},
        loader=lambda *_args: FakeBatchTranscriber("en"),
    )

    with pytest.raises(ValueError, match="must match"):
        adapter.batch([np.zeros(320, dtype=np.float32)], ["en", "cs"])


def test_batch_rejects_an_unlicensed_language(tmp_path: Path) -> None:
    adapter = load_transcription_adapter(
        "parakeet",
        tmp_path,
        threads=1,
        registry={"parakeet": _parakeet_spec()},
        loader=lambda *_args: FakeBatchTranscriber("en"),
    )

    with pytest.raises(ValueError, match="unsupported transcription language"):
        adapter.batch([np.zeros(320, dtype=np.float32)], ["de"])


def test_an_unsupported_language_is_named_rather_than_a_bare_value_error(tmp_path: Path) -> None:
    """The handler has to be able to tell this apart from a genuine fault.

    A plain ``ValueError`` here reached ``/v1/moderate``'s catch-all and came
    back ``500 internal_error`` with nothing logged, so a rules directory that
    outruns the deployed model looked like a crashing processor. It stays a
    ``ValueError`` so callers written against the old contract still catch it,
    and it carries the code so the log line can name which one it was.
    """
    adapter = load_transcription_adapter(
        "parakeet",
        tmp_path,
        threads=1,
        registry={"parakeet": _parakeet_spec()},
        loader=lambda *_args: FakeBatchTranscriber("en"),
    )

    with pytest.raises(UnsupportedTranscriptionLanguage) as raised:
        adapter(np.zeros(320, dtype=np.float32), "de")

    assert raised.value.language == "de"
    assert isinstance(raised.value, ValueError)


def test_batch_groups_per_recognizer_for_whisper(tmp_path: Path) -> None:
    model = ModelSpec(
        model_id="whisper",
        family="whisper",
        archive_url="https://example.test/model.tar.bz2",
        archive_sha256="a" * 64,
        archive_root="package",
        languages=("en", "cs"),
        license_name="MIT",
        license_url="https://example.test/license",
        files={"encoder": "encoder.onnx", "decoder": "decoder.onnx", "tokens": "tokens.txt"},
    )
    per_language: dict[str, FakeBatchTranscriber] = {}

    def loader(spec, model_dir, language, threads):
        per_language[language] = FakeBatchTranscriber(language)
        return per_language[language]

    adapter = load_transcription_adapter(
        "whisper",
        tmp_path,
        threads=1,
        registry={"whisper": model},
        loader=loader,
    )
    batch = [np.zeros(320, dtype=np.float32) for _ in range(3)]

    results = adapter.batch(batch, ["en", "cs", "en"])

    assert per_language["en"].batches == [2]
    assert per_language["cs"].batches == [1]
    assert [result.language for result in results] == ["en", "cs", "en"]
