import asyncio
import json
import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml
from fastapi import FastAPI
from voicesniffer_runtime.model_store import ModelSpec

import voicesniffer_processor.__main__ as service_main
from voicesniffer_processor.__main__ import build_application
from voicesniffer_processor.models import TranscriptionResult
from voicesniffer_processor.toxicity import ToxicityResult


def test_hypercorn_does_not_expire_idle_keep_alive_connections_with_read_timeout() -> None:
    config_path = Path(__file__).parents[1] / "hypercorn.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    # 300s: a voice-chat server is quiet whenever nobody speaks, and the plugin pools its
    # connection. A short server-side timeout closes the socket under the client and the next
    # utterance stalls into the dead connection (the 2026-07-22 timeout bug). Keep the server
    # window far above any realistic speech gap; the client recycles idle connections sooner.
    assert config["keep_alive_timeout"] == 300
    assert "read_timeout" not in config


def test_builds_ready_application_from_mounted_runtime_files(tmp_path: Path) -> None:
    runtime = write_runtime_files(tmp_path)

    loaded: list[tuple[str, Path, int, dict[str, ModelSpec]]] = []

    def load_adapter(
        model_id: str,
        model_dir: Path,
        *,
        threads: int,
        registry: dict[str, ModelSpec],
    ):
        loaded.append((model_id, model_dir, threads, registry))

        def transcribe(_samples: np.ndarray, _language: str) -> TranscriptionResult:
            return TranscriptionResult(text="clean greeting", language="en")

        return transcribe

    app = build_application(
        {
            "VOICESNIFFER_TOKENS_FILE": str(runtime.tokens_file),
            "VOICESNIFFER_MODEL_ID": "english",
            "VOICESNIFFER_MODELS_DIR": str(runtime.models_dir),
            "VOICESNIFFER_MODEL_REGISTRY": str(runtime.registry_file),
            "VOICESNIFFER_RULES_DIR": str(runtime.rules_dir),
            "VOICESNIFFER_MODEL_THREADS": "2",
        },
        adapter_loader=load_adapter,
    )

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://processor") as client:
            return await client.get("/healthz")

    response = asyncio.run(request())

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert len(loaded) == 1
    assert loaded[0][:3] == ("english", runtime.installation, 2)


def test_build_application_wires_tier_two_a_classifier(tmp_path: Path, monkeypatch) -> None:
    runtime = write_runtime_files(tmp_path)

    def load_adapter(
        _model_id: str,
        _model_dir: Path,
        *,
        threads: int,
        registry: dict[str, ModelSpec],
    ):
        def transcribe(_samples: np.ndarray, _language: str) -> TranscriptionResult:
            return TranscriptionResult(text="classifier only", language="en")

        return transcribe

    def classifier(text: str, language: str) -> ToxicityResult:
        return ToxicityResult(
            category="hate",
            severity=3,
            confidence=0.77,
            matched_text=text,
            language=language,
        )

    captured = {}

    def capture_app(*args, **kwargs) -> FastAPI:
        captured["toxicity_classifier"] = kwargs["toxicity_classifier"]
        return FastAPI()

    monkeypatch.setattr(service_main, "create_app", capture_app)

    app = build_application(
        {
            "VOICESNIFFER_TOKENS_FILE": str(runtime.tokens_file),
            "VOICESNIFFER_MODEL_ID": "english",
            "VOICESNIFFER_MODELS_DIR": str(runtime.models_dir),
            "VOICESNIFFER_MODEL_REGISTRY": str(runtime.registry_file),
            "VOICESNIFFER_RULES_DIR": str(runtime.rules_dir),
        },
        adapter_loader=load_adapter,
        toxicity_classifier=classifier,
    )

    assert isinstance(app, FastAPI)
    assert captured["toxicity_classifier"] is classifier


def test_startup_names_every_language_the_model_can_hear_but_no_pack_can_moderate(
    caplog,
) -> None:
    """The gap nothing used to compare.

    ``models.toml`` says what the speech model was trained on and the rules
    directory says what we have written rules for. The shipped combination is a
    25-language model and 23 packs, which is how ``fi`` and ``mt`` came to be
    offered in ``config.yml`` and the panel with no rule behind them. Startup is
    the one place both lists are in hand.
    """
    packs = {"en": None, "cs": None}

    with caplog.at_level(logging.ERROR, logger="voicesniffer.processor"):
        missing = service_main._report_unmoderatable_languages(("en", "cs", "fi", "mt"), packs)

    assert missing == ("fi", "mt")
    assert "languages_without_rules" in caplog.text
    assert "missing=fi,mt" in caplog.text


def test_startup_says_nothing_when_every_model_language_has_a_pack(caplog) -> None:
    with caplog.at_level(logging.ERROR, logger="voicesniffer.processor"):
        missing = service_main._report_unmoderatable_languages(("en",), {"en": None})

    assert missing == ()
    assert caplog.text == ""


def test_startup_names_every_pack_whose_language_the_model_cannot_hear(caplog) -> None:
    """The other direction, which nothing compared at all.

    A model language with no pack is refused cleanly at the front of the
    handler. A pack whose language the model was never trained on used to reach
    the recognizer, which raised, and the request came back 500 with an empty
    log. It appears the moment a smaller model meets the full rules directory:
    ``moonshine-*-en`` is English only, so every other pack lands here.
    """
    packs = {"cs": None, "de": None, "en": None}

    with caplog.at_level(logging.ERROR, logger="voicesniffer.processor"):
        missing = service_main._report_untranscribable_languages(("en",), packs)

    assert missing == ("cs", "de")
    assert "rules_without_a_model" in caplog.text
    assert "missing=cs,de" in caplog.text


def test_startup_says_nothing_when_the_model_can_hear_every_installed_pack(caplog) -> None:
    # The shipped combination: 23 packs against a 25-language model, so nothing
    # here fires and the line must stay quiet rather than warn on every start.
    with caplog.at_level(logging.ERROR, logger="voicesniffer.processor"):
        missing = service_main._report_untranscribable_languages(("en", "cs", "fi"), {"en": None})

    assert missing == ()
    assert caplog.text == ""


def test_a_missing_rule_pack_is_loud_but_does_not_stop_the_processor_starting(
    tmp_path: Path,
    caplog,
) -> None:
    """Deliberately not fatal, and the reason is worth a test rather than a comment.

    Refusing to start would crash-loop the deployed cloud node the moment it
    restarts, and a self-hosted operator who trims the rules directory to the
    languages they actually run is doing something reasonable. The hard refusal
    lives per request, in ``/v1/moderate``.
    """
    runtime = write_runtime_files(tmp_path)
    runtime.registry_file.write_text(
        runtime.registry_file.read_text(encoding="utf-8").replace(
            'languages = ["en"]', 'languages = ["en", "fi"]'
        ),
        encoding="utf-8",
    )

    def load_adapter(model_id, model_dir, *, threads, registry):
        return lambda _samples, _language: TranscriptionResult(text="hi", language="en")

    with caplog.at_level(logging.ERROR, logger="voicesniffer.processor"):
        app = build_application(
            {
                "VOICESNIFFER_TOKENS_FILE": str(runtime.tokens_file),
                "VOICESNIFFER_MODEL_ID": "english",
                "VOICESNIFFER_MODELS_DIR": str(runtime.models_dir),
                "VOICESNIFFER_MODEL_REGISTRY": str(runtime.registry_file),
                "VOICESNIFFER_RULES_DIR": str(runtime.rules_dir),
            },
            adapter_loader=load_adapter,
        )

    assert isinstance(app, FastAPI)
    assert "missing=fi" in caplog.text


@dataclass(frozen=True, slots=True)
class RuntimeFiles:
    tokens_file: Path
    rules_dir: Path
    registry_file: Path
    models_dir: Path
    installation: Path


def write_runtime_files(tmp_path: Path) -> RuntimeFiles:
    tokens_file = tmp_path / "tokens.json"
    tokens_file.write_text(json.dumps({"server-a": "secret-token"}), encoding="utf-8")
    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "en.yml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "language": "en",
                "rules": [
                    {
                        "id": "en.test.blocked",
                        "term": "blocked phrase",
                        "category": "harassment",
                        "severity": 2,
                        "match": "exact",
                        "variants": [],
                        "context_required": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    registry_file = tmp_path / "models.toml"
    registry_file.write_text(
        """
schema = 1

[[models]]
id = "english"
family = "moonshine-v2"
archive_url = "https://example.test/model.tar.bz2"
archive_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
archive_root = "package"
languages = ["en"]
license_name = "MIT"
license_url = "https://example.test/license"
files = { encoder = "encoder.ort", decoder = "decoder.ort", tokens = "tokens.txt" }
""".strip(),
        encoding="utf-8",
    )
    models_dir = tmp_path / "models"
    installation = models_dir / "english" / "aaaaaaaaaaaaaaaa"
    installation.mkdir(parents=True)
    for filename in ("encoder.ort", "decoder.ort", "tokens.txt"):
        (installation / filename).write_bytes(b"fixture")
    return RuntimeFiles(
        tokens_file=tokens_file,
        rules_dir=rules_dir,
        registry_file=registry_file,
        models_dir=models_dir,
        installation=installation,
    )


class _BatchAdapter:
    """Minimal stand-in for SherpaTranscriptionAdapter's batching surface."""

    def __init__(self) -> None:
        self.batches: list[int] = []

    def __call__(self, samples, language: str) -> TranscriptionResult:
        return TranscriptionResult(text="single", language="en")

    def batch(self, batch, languages) -> list[TranscriptionResult]:
        self.batches.append(len(batch))
        return [TranscriptionResult(text="batched", language="en") for _ in batch]


def _environment(runtime, **overrides: str) -> dict[str, str]:
    values = {
        "VOICESNIFFER_TOKENS_FILE": str(runtime.tokens_file),
        "VOICESNIFFER_MODEL_ID": "english",
        "VOICESNIFFER_MODELS_DIR": str(runtime.models_dir),
        "VOICESNIFFER_MODEL_REGISTRY": str(runtime.registry_file),
        "VOICESNIFFER_RULES_DIR": str(runtime.rules_dir),
    }
    values.update(overrides)
    return values


def test_micro_batching_is_off_unless_asked_for(tmp_path: Path) -> None:
    runtime = write_runtime_files(tmp_path)
    adapter = _BatchAdapter()

    app = build_application(
        _environment(runtime),
        adapter_loader=lambda *_args, **_kwargs: adapter,
    )

    # Default deployment behaviour must be byte-for-byte what it was: the adapter
    # itself is the transcribe callable, with no batching thread in the way.
    assert not hasattr(app.state, "micro_batcher")


def test_micro_batching_is_wired_when_enabled(tmp_path: Path) -> None:
    runtime = write_runtime_files(tmp_path)
    adapter = _BatchAdapter()

    app = build_application(
        _environment(
            runtime,
            VOICESNIFFER_MAX_BATCH_SIZE="8",
            VOICESNIFFER_BATCH_WINDOW_MS="10",
        ),
        adapter_loader=lambda *_args, **_kwargs: adapter,
    )

    batcher = app.state.micro_batcher
    try:
        assert batcher.max_batch_size == 8
        result = batcher(np.zeros(320, dtype=np.float32), "en")
        assert result.text == "batched"
        assert adapter.batches == [1]
    finally:
        batcher.close()


def test_micro_batching_needs_a_transcriber_that_can_batch(tmp_path: Path) -> None:
    runtime = write_runtime_files(tmp_path)

    with pytest.raises(ValueError, match="requires a batching transcriber"):
        build_application(
            _environment(runtime, VOICESNIFFER_MAX_BATCH_SIZE="4"),
            adapter_loader=lambda *_args, **_kwargs: (lambda samples, language: None),
        )


def test_batch_size_is_bounded(tmp_path: Path) -> None:
    runtime = write_runtime_files(tmp_path)

    with pytest.raises(ValueError, match="VOICESNIFFER_MAX_BATCH_SIZE"):
        build_application(
            _environment(runtime, VOICESNIFFER_MAX_BATCH_SIZE="65"),
            adapter_loader=lambda *_args, **_kwargs: _BatchAdapter(),
        )
