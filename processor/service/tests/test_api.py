import asyncio
import contextlib
import json
import logging
import math
import threading
import uuid
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml

from voicesniffer_processor.app import (
    RequestStageTrace,
    RequestTrace,
    _TenantAdmission,
    create_app,
)
from voicesniffer_processor.cloud_auth import IntrospectionUnavailable, _credential_from
from voicesniffer_processor.models import (
    TranscriptionResult,
    UnsupportedTranscriptionLanguage,
)
from voicesniffer_processor.opus import OpusPacketError, OpusPacketInfo
from voicesniffer_processor.rules import RulePack
from voicesniffer_processor.settings import ProcessorSettings
from voicesniffer_processor.toxicity import ToxicityResult


def test_health_is_unauthenticated_and_contains_no_secrets(tmp_path) -> None:
    app, _ = build_test_app(tmp_path)

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://processor") as client:
            return await client.get("/healthz")

    response = asyncio.run(request())

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert "secret-token" not in response.text


def test_bounds_concurrent_processing_to_configured_workers(tmp_path) -> None:
    settings = replace(load_settings(tmp_path), workers=1)
    active = 0
    maximum_active = 0
    lock = threading.Lock()
    started = threading.Event()
    release = threading.Event()

    def transcribe(_samples: np.ndarray, _language: str) -> TranscriptionResult:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            started.set()
        try:
            if not release.wait(2):
                raise TimeoutError("test inference release timed out")
            return TranscriptionResult(text="clean greeting", language="en")
        finally:
            with lock:
                active -= 1

    app = create_app(
        settings,
        transcribe=transcribe,
        rule_packs={"en": load_rules(tmp_path)},
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
        queue_wait_seconds=1.0,
    )

    async def request() -> tuple[httpx.Response, httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://processor") as client:
            first = asyncio.create_task(
                client.post(
                    "/v1/moderate",
                    headers=headers(tmp_path, str(uuid.uuid4())),
                    content=b"\x00\x01a",
                )
            )
            second = asyncio.create_task(
                client.post(
                    "/v1/moderate",
                    headers=headers(tmp_path, str(uuid.uuid4())),
                    content=b"\x00\x01a",
                )
            )
            assert await asyncio.to_thread(started.wait, 2)
            await asyncio.sleep(0.1)
            release.set()
            return await asyncio.gather(first, second)

    responses = asyncio.run(request())

    assert maximum_active == 1
    assert [response.status_code for response in responses] == [200, 200]


def test_cancelled_request_holds_worker_slot_until_inference_finishes(tmp_path) -> None:
    settings = replace(load_settings(tmp_path), workers=1)
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    lock = threading.Lock()

    def transcribe(_samples: np.ndarray, _language: str) -> TranscriptionResult:
        nonlocal calls
        with lock:
            calls += 1
            entered.set()
        release.wait(2)
        return TranscriptionResult(text="clean greeting", language="en")

    app = create_app(
        settings,
        transcribe=transcribe,
        rule_packs={"en": load_rules(tmp_path)},
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
    )

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://processor") as client:
            first = asyncio.create_task(
                client.post(
                    "/v1/moderate",
                    headers=headers(tmp_path, str(uuid.uuid4())),
                    content=b"\x00\x01a",
                )
            )
            assert await asyncio.to_thread(entered.wait, 2)
            first.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await first
            second = asyncio.create_task(
                client.post(
                    "/v1/moderate",
                    headers=headers(tmp_path, str(uuid.uuid4())),
                    content=b"\x00\x01a",
                )
            )
            await asyncio.sleep(0.05)
            assert calls == 1
            release.set()
            return await second

    response = asyncio.run(request())

    assert response.status_code == 200
    assert calls == 2


def test_cancelled_owner_keeps_same_request_id_coalesced(tmp_path) -> None:
    settings = replace(load_settings(tmp_path), workers=2)
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    lock = threading.Lock()
    request_id = str(uuid.uuid4())
    request_headers = headers(tmp_path, request_id)

    def transcribe(_samples: np.ndarray, _language: str) -> TranscriptionResult:
        nonlocal calls
        with lock:
            calls += 1
            entered.set()
        release.wait(2)
        return TranscriptionResult(text="clean greeting", language="en")

    app = create_app(
        settings,
        transcribe=transcribe,
        rule_packs={"en": load_rules(tmp_path)},
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
    )

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://processor") as client:
            first = asyncio.create_task(
                client.post(
                    "/v1/moderate",
                    headers=request_headers,
                    content=b"\x00\x01a",
                )
            )
            assert await asyncio.to_thread(entered.wait, 2)
            first.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await first
            duplicate = asyncio.create_task(
                client.post(
                    "/v1/moderate",
                    headers=request_headers,
                    content=b"\x00\x01a",
                )
            )
            await asyncio.sleep(0.05)
            assert calls == 1
            release.set()
            return await duplicate

    response = asyncio.run(request())

    assert response.status_code == 200
    assert calls == 1


def test_rejects_worker_queue_saturation(tmp_path) -> None:
    settings = replace(load_settings(tmp_path), workers=1)
    entered = threading.Event()
    release = threading.Event()

    def transcribe(_samples: np.ndarray, _language: str) -> TranscriptionResult:
        entered.set()
        release.wait(2)
        return TranscriptionResult(text="clean greeting", language="en")

    app = create_app(
        settings,
        transcribe=transcribe,
        rule_packs={"en": load_rules(tmp_path)},
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
        queue_wait_seconds=0.01,
    )

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://processor") as client:
            first = asyncio.create_task(
                client.post(
                    "/v1/moderate",
                    headers=headers(tmp_path, str(uuid.uuid4())),
                    content=b"\x00\x01a",
                )
            )
            assert await asyncio.to_thread(entered.wait, 2)
            second = await client.post(
                "/v1/moderate",
                headers=headers(tmp_path, str(uuid.uuid4())),
                content=b"\x00\x01a",
            )
            release.set()
            await first
            return second

    response = asyncio.run(request())

    assert response.status_code == 503
    assert response.json()["code"] == "processor_busy"


def test_returns_json_verdict_and_echoes_request_id(tmp_path) -> None:
    request_id = str(uuid.uuid4())
    traces: list[RequestTrace] = []
    stages: list[RequestStageTrace] = []
    app, calls = build_test_app(
        tmp_path,
        trace_sink=traces.append,
        stage_sink=stages.append,
    )

    response = post(app, headers(tmp_path, request_id), b"\x00\x01a")

    assert response.status_code == 200
    assert response.headers["x-request-id"] == request_id
    assert response.json() == {
        "request_id": request_id,
        "transcript": "blocked phrase",
        "language": "en",
        "matches": [
            {
                "rule_id": "en.test.blocked",
                "category": "harassment",
                "severity": 2,
                "matched_text": "blocked phrase",
                "context_required": False,
            }
        ],
        "severity": 2,
        "confidence": 1.0,
        "processing_ms": response.json()["processing_ms"],
    }
    assert isinstance(response.json()["processing_ms"], int)
    assert_timing_headers(response)
    assert calls == ["en"]
    assert len(traces) == 1
    trace = traces[0]
    assert trace.request_id == request_id
    assert trace.outcome == "success"
    assert trace.decode_ms >= 0
    assert trace.stt_ms >= 0
    assert trace.rules_ms >= 0
    assert trace.classifier_ms >= 0
    assert trace.serialize_ms >= 0
    assert trace.total_ms >= response.json()["processing_ms"] + trace.serialize_ms
    assert [(stage.stage, stage.state) for stage in stages] == [
        ("request", "received"),
        ("decode", "started"),
        ("decode", "completed"),
        ("stt", "started"),
        ("stt", "completed"),
        ("rules", "started"),
        ("rules", "completed"),
        ("classifier", "started"),
        ("classifier", "completed"),
        ("serialize", "started"),
        ("serialize", "completed"),
    ]
    assert all(stage.request_id == request_id for stage in stages)


def test_returns_exact_no_match_schema(tmp_path) -> None:
    request_id = str(uuid.uuid4())
    app, calls = build_test_app(tmp_path, transcription_text="clean greeting")

    response = post(app, headers(tmp_path, request_id), b"\x00\x01a")

    assert response.status_code == 200
    assert response.json() == {
        "request_id": request_id,
        "transcript": "clean greeting",
        "language": "en",
        "matches": [],
        "severity": 0,
        "confidence": 0.0,
        "processing_ms": response.json()["processing_ms"],
    }
    assert calls == ["en"]


def test_tier_two_a_classifier_can_raise_verdict_without_wordlist_match(tmp_path) -> None:
    request_id = str(uuid.uuid4())

    app = create_app(
        load_settings(tmp_path),
        transcribe=lambda _samples, _language: TranscriptionResult(
            text="misspelled hate phrase", language="en"
        ),
        rule_packs={"en": load_rules(tmp_path)},
        toxicity_classifier=lambda text, language: ToxicityResult(
            category="hate",
            severity=3,
            confidence=0.82,
            matched_text=text,
            language=language,
        ),
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
    )

    response = post(app, headers(tmp_path, request_id), b"\x00\x01a")

    assert response.status_code == 200
    assert response.json()["severity"] == 3
    assert response.json()["confidence"] == 0.82
    assert response.json()["matches"] == [
        {
            "rule_id": "tier2a.local-toxicity",
            "category": "hate",
            "severity": 3,
            "matched_text": "misspelled hate phrase",
            "context_required": False,
        }
    ]


def test_auto_language_checks_every_rule_pack(tmp_path) -> None:
    request_id = str(uuid.uuid4())
    rules_root = Path(__file__).parents[1] / "rules"
    app = create_app(
        load_settings(tmp_path),
        transcribe=lambda _samples, _language: TranscriptionResult(
            text="buzerant", language="auto"
        ),
        rule_packs={
            "en": RulePack.load(rules_root / "en.yml"),
            "cs": RulePack.load(rules_root / "cs.yml"),
        },
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
    )
    request_headers = headers(tmp_path, request_id) | {"X-VoiceSniffer-Language": "auto"}

    response = post(app, request_headers, b"\x00\x01a")

    assert response.status_code == 200
    assert response.json()["language"] == "auto"
    assert response.json()["severity"] == 3
    assert response.json()["matches"][0]["rule_id"] == "cs.hate.homophobic"


def test_logs_regex_timeouts_with_request_context(tmp_path, caplog) -> None:
    request_id = str(uuid.uuid4())
    rule_file = tmp_path / "timeout-en.yml"
    rule_file.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "language": "en",
                "rules": [
                    {
                        "id": "en.test.timeout",
                        "term": r"(a+)+$",
                        "category": "harassment",
                        "severity": 2,
                        "match": "regex",
                        "probe": "aaa",
                        "variants": [],
                        "context_required": False,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    app = create_app(
        load_settings(tmp_path),
        transcribe=lambda _samples, _language: TranscriptionResult(
            text="a" * 100_000 + "b", language="en"
        ),
        rule_packs={"en": RulePack.load(rule_file, regex_timeout_seconds=0.001)},
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
    )
    caplog.set_level(logging.WARNING, logger="voicesniffer.processor")

    response = post(app, headers(tmp_path, request_id), b"\x00\x01a")

    assert response.status_code == 200
    assert response.json()["matches"] == []
    assert [record.getMessage() for record in caplog.records] == [
        f"rule_timeout request_id={request_id} language=en count=1 rule_ids=en.test.timeout"
    ]


def test_logs_rule_budget_exhaustion_at_error_level(tmp_path, caplog) -> None:
    """A pack running out of its whole budget means part of the ruleset was never
    applied, so the verdict is known-incomplete rather than merely slow. That is
    an operator-visible degradation and is logged louder than a rule timeout."""
    request_id = str(uuid.uuid4())
    rule_file = tmp_path / "budget-en.yml"
    rule_file.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "language": "en",
                "rules": [
                    {
                        "id": "en.test.slow",
                        "term": r"(a+)+$",
                        "category": "harassment",
                        "severity": 2,
                        "match": "regex",
                        "probe": "aaa",
                        "variants": [],
                        "context_required": False,
                    },
                    {
                        "id": "en.test.never-reached",
                        "term": "blocked phrase",
                        "category": "harassment",
                        "severity": 3,
                        "match": "exact",
                        "variants": [],
                        "context_required": False,
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    app = create_app(
        load_settings(tmp_path),
        transcribe=lambda _samples, _language: TranscriptionResult(
            text="a" * 100_000 + "b blocked phrase", language="en"
        ),
        rule_packs={
            "en": RulePack.load(
                rule_file,
                regex_timeout_seconds=0.05,
                max_match_seconds=0.01,
            )
        },
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
    )
    caplog.set_level(logging.WARNING, logger="voicesniffer.processor")

    response = post(app, headers(tmp_path, request_id), b"\x00\x01a")

    assert response.status_code == 200
    # The severity-3 rule was in the pack and its term is in the transcript, but
    # the budget ran out before it was reached. Fail-open, and said so.
    assert response.json()["matches"] == []
    assert response.json()["severity"] == 0
    messages = [record.getMessage() for record in caplog.records]
    assert (
        f"rule_budget_exhausted request_id={request_id} language=en unevaluated=1 "
        f"rule_ids=en.test.never-reached" in messages
    ), messages
    assert [record.levelname for record in caplog.records if "budget" in record.getMessage()] == [
        "ERROR"
    ]


def test_rejects_invalid_bearer_before_reading_body(tmp_path) -> None:
    app, _ = build_test_app(tmp_path)

    class ExplodingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            raise AssertionError("unauthorized body was read")
            yield b""

    request_headers = headers(tmp_path, str(uuid.uuid4())) | {"Authorization": "Bearer wrong-token"}

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://processor") as client:
            return await client.post(
                "/v1/moderate",
                headers=request_headers,
                content=ExplodingStream(),
            )

    response = asyncio.run(request())

    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"


def test_rejects_wrong_content_type_without_processing(tmp_path) -> None:
    app, calls = build_test_app(tmp_path)
    request_headers = headers(tmp_path, str(uuid.uuid4())) | {"Content-Type": "audio/opus"}

    response = post(app, request_headers, b"\x00\x01a")

    assert response.status_code == 415
    assert response.json()["code"] == "unsupported_media_type"
    assert calls == []


def test_rejects_missing_json_accept_without_processing(tmp_path) -> None:
    app, calls = build_test_app(tmp_path)
    request_headers = headers(tmp_path, str(uuid.uuid4()))
    del request_headers["Accept"]

    response = post(app, request_headers, b"\x00\x01a")

    assert response.status_code == 406
    assert response.json()["code"] == "not_acceptable"
    assert calls == []


def test_rejects_body_above_configured_limit(tmp_path) -> None:
    settings = replace(load_settings(tmp_path), max_body_bytes=4)
    app, calls = build_test_app(tmp_path, settings=settings)

    response = post(app, headers(tmp_path, str(uuid.uuid4())), b"\x00\x03abc")

    assert response.status_code == 413
    assert response.json()["code"] == "body_too_large"
    assert calls == []


def test_rejects_negative_content_length(tmp_path) -> None:
    app, calls = build_test_app(tmp_path)
    request_headers = headers(tmp_path, str(uuid.uuid4())) | {"Content-Length": "-1"}

    response = post(app, request_headers, b"")

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_content_length"
    assert calls == []


def test_maps_malformed_envelope_to_stable_error(tmp_path) -> None:
    app, calls = build_test_app(tmp_path)

    response = post(app, headers(tmp_path, str(uuid.uuid4())), b"\x00")

    assert response.status_code == 400
    assert response.json()["code"] == "truncated_frame_length"
    assert calls == []


def test_maps_unavailable_opus_inspection_to_service_unavailable(tmp_path) -> None:
    def unavailable(_packet: bytes) -> OpusPacketInfo:
        raise OpusPacketError("opus_unavailable")

    app = create_app(
        load_settings(tmp_path),
        transcribe=lambda _samples, _language: TranscriptionResult(text="", language="en"),
        rule_packs={"en": load_rules(tmp_path)},
        inspect_packet=unavailable,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
    )

    response = post(app, headers(tmp_path, str(uuid.uuid4())), b"\x00\x01a")

    assert response.status_code == 503
    assert response.json()["code"] == "opus_unavailable"


def test_duplicate_request_runs_transcription_once(tmp_path) -> None:
    request_id = str(uuid.uuid4())
    app, calls = build_test_app(tmp_path)
    request_headers = headers(tmp_path, request_id)

    first = post(app, request_headers, b"\x00\x01a")
    second = post(app, request_headers, b"\x00\x01a")

    assert first.status_code == 200
    assert second.json() == first.json()
    assert calls == ["en"]


def test_request_id_reuse_with_changed_audio_returns_conflict(tmp_path) -> None:
    request_id = str(uuid.uuid4())
    app, calls = build_test_app(tmp_path)
    request_headers = headers(tmp_path, request_id)

    assert post(app, request_headers, b"\x00\x01a").status_code == 200
    changed = post(app, request_headers, b"\x00\x01b")

    assert changed.status_code == 409
    assert changed.json()["code"] == "request_id_reused"
    assert calls == ["en"]


def test_request_id_reuse_with_changed_partial_flag_returns_conflict(tmp_path) -> None:
    request_id = str(uuid.uuid4())
    app, calls = build_test_app(tmp_path)
    request_headers = headers(tmp_path, request_id)

    first = post(app, request_headers | {"X-VoiceSniffer-Partial": "0"}, b"\x00\x01a")
    assert first.status_code == 200
    changed = post(app, request_headers | {"X-VoiceSniffer-Partial": "1"}, b"\x00\x01a")

    assert changed.status_code == 409
    assert changed.json()["code"] == "request_id_reused"
    assert calls == ["en"]


def test_internal_failure_does_not_echo_exception_or_audio(tmp_path) -> None:
    settings = load_settings(tmp_path)

    def fail(_samples: np.ndarray, _language: str) -> TranscriptionResult:
        raise RuntimeError("secret-token private transcript")

    app = create_app(
        settings,
        transcribe=fail,
        rule_packs={"en": load_rules(tmp_path)},
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
    )

    response = post(app, headers(tmp_path, str(uuid.uuid4())), b"\x00\x06secret")

    encoded = response.content.decode("utf-8")
    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"
    assert "secret-token" not in encoded
    assert "private transcript" not in encoded
    assert "secret" not in encoded


def test_internal_failure_is_logged_with_enough_to_diagnose_it(tmp_path, caplog) -> None:
    """A 500 that logs nothing is a failure nobody can find again.

    The response carries a code and no detail on purpose, so the log line is the
    only record of what broke. There was none: an unexpected exception on this
    path produced ``500 internal_error`` on the wire, an ``outcome`` in the
    request trace, and no traceback anywhere. The one operational question --
    which of decode, speech-to-text, rules or the classifier raised, and with
    what -- had no answer at all.
    """
    settings = load_settings(tmp_path)
    request_id = str(uuid.uuid4())

    def fail(_samples: np.ndarray, _language: str) -> TranscriptionResult:
        raise RuntimeError("recognizer exploded")

    app = create_app(
        settings,
        transcribe=fail,
        rule_packs={"en": load_rules(tmp_path)},
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
    )

    with caplog.at_level(logging.ERROR, logger="voicesniffer.processor"):
        response = post(app, headers(tmp_path, request_id), b"\x00\x01a")

    assert response.status_code == 500
    assert response.json()["code"] == "internal_error"
    assert "request_failed" in caplog.text
    assert f"request_id={request_id}" in caplog.text
    assert "server=server-a" in caplog.text
    # The traceback, which is the half that names where it came from.
    assert "RuntimeError: recognizer exploded" in caplog.text
    assert "Traceback" in caplog.text


def test_a_language_the_model_cannot_transcribe_is_refused_not_a_crash(tmp_path, caplog) -> None:
    """A licensed pack the speech model was never trained on.

    ``models.toml`` and the rules directory are set independently, so a code can
    have a pack and no model behind it -- every pack but ``en`` on a
    ``moonshine-*-en`` node. The handler checks the packs, passes the code to the
    recognizer, and the recognizer raised: the caller got ``500 internal_error``
    for a deployment mistake, and nothing said which language did it. It is the
    same answer a missing pack gets, because it is the same fact: not here.
    """
    request_id = str(uuid.uuid4())

    def transcribe(_samples: np.ndarray, language: str) -> TranscriptionResult:
        if language != "en":
            raise UnsupportedTranscriptionLanguage(language)
        return TranscriptionResult(text="hi", language="en")

    app = create_app(
        load_settings(tmp_path),
        transcribe=transcribe,
        rule_packs={"en": load_rules(tmp_path), "cs": load_rules(tmp_path, language="cs")},
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
    )

    with caplog.at_level(logging.ERROR, logger="voicesniffer.processor"):
        response = post(
            app,
            headers(tmp_path, request_id) | {"X-VoiceSniffer-Language": "cs"},
            b"\x00\x01a",
        )

    assert response.status_code == 400
    assert response.json()["code"] == "language_unsupported"
    assert "language_untranscribable" in caplog.text
    assert "untranscribable=cs" in caplog.text
    assert "internal_error" not in caplog.text


def test_the_rate_limit_table_drops_servers_that_went_quiet(tmp_path) -> None:
    """One deque per (tenant, server) was kept for the life of the process.

    The window it holds only describes the last sixty seconds, and it was pruned
    on the way in, but nothing ever removed the key. Small per entry and
    unbounded in count, inside a container with a hard 3 GiB ceiling, which is
    the shape that eventually wins. The sweep is on a timer rather than on every
    call so it never costs the moderation path a pass over every tenant.
    """
    token_file = tmp_path / "tokens.json"
    token_file.write_text(
        json.dumps(
            {
                "server-a": {"token": "token-a", "rate-limit-per-minute": 10},
                "server-b": {"token": "token-b", "rate-limit-per-minute": 10},
            }
        ),
        encoding="utf-8",
    )
    settings = ProcessorSettings.from_environment({"VOICESNIFFER_TOKENS_FILE": str(token_file)})
    now = 1_000.0
    app = create_app(
        settings,
        transcribe=lambda _samples, _language: TranscriptionResult(text="hi", language="en"),
        rule_packs={"en": load_rules(tmp_path)},
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
        clock=lambda: now,
    )

    first = headers(tmp_path, str(uuid.uuid4())) | {"Authorization": "Bearer token-a"}
    assert post(app, first, b"\x00\x01a").status_code == 200
    assert set(app.state.rate_windows) == {(None, "server-a")}

    # server-a says nothing for two minutes; server-b arrives and triggers the sweep.
    now = 1_120.0
    second = headers(tmp_path, str(uuid.uuid4())) | {"Authorization": "Bearer token-b"}
    assert post(app, second, b"\x00\x01a").status_code == 200

    assert set(app.state.rate_windows) == {(None, "server-b")}


def test_the_rate_limit_sweep_never_drops_a_window_still_inside_its_minute(tmp_path) -> None:
    # The other half: a table that forgets a live window would hand a server a
    # fresh allowance every time somebody else made a request.
    token_file = tmp_path / "tokens.json"
    token_file.write_text(
        json.dumps(
            {
                "server-a": {"token": "token-a", "rate-limit-per-minute": 2},
                "server-b": {"token": "token-b", "rate-limit-per-minute": 2},
            }
        ),
        encoding="utf-8",
    )
    settings = ProcessorSettings.from_environment({"VOICESNIFFER_TOKENS_FILE": str(token_file)})
    now = 1_000.0
    app = create_app(
        settings,
        transcribe=lambda _samples, _language: TranscriptionResult(text="hi", language="en"),
        rule_packs={"en": load_rules(tmp_path)},
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
        clock=lambda: now,
    )

    for moment in (1_050.0, 1_055.0):
        now = moment
        request_headers = headers(tmp_path, str(uuid.uuid4())) | {"Authorization": "Bearer token-a"}
        assert post(app, request_headers, b"\x00\x01a").status_code == 200

    # Past the sweep interval, but server-a's window is still inside its minute.
    now = 1_070.0
    other = headers(tmp_path, str(uuid.uuid4())) | {"Authorization": "Bearer token-b"}
    assert post(app, other, b"\x00\x01a").status_code == 200
    assert (None, "server-a") in app.state.rate_windows

    third = headers(tmp_path, str(uuid.uuid4())) | {"Authorization": "Bearer token-a"}
    refused = post(app, third, b"\x00\x01a")
    assert refused.status_code == 429
    assert refused.json()["code"] == "rate_limited"


def test_rejects_invalid_request_metadata(tmp_path) -> None:
    app, calls = build_test_app(tmp_path)
    invalid = headers(tmp_path, "not-a-uuid") | {
        "X-VoiceSniffer-Player-Id": "not-a-player",
        "X-VoiceSniffer-Preroll-Samples": "one",
    }

    response = post(app, invalid, b"\x00\x01a")

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_request_id"
    assert response.json()["request_id"] is None
    assert "x-request-id" not in response.headers
    assert calls == []


def test_rejects_preroll_outside_protocol_range(tmp_path) -> None:
    app, calls = build_test_app(tmp_path)
    invalid = headers(tmp_path, str(uuid.uuid4())) | {"X-VoiceSniffer-Preroll-Samples": str(2**64)}

    response = post(app, invalid, b"\x00\x01a")

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_preroll"
    assert calls == []


def test_rejects_a_partial_header_that_is_neither_zero_nor_one(tmp_path) -> None:
    """A sender that means `true` has a bug, and guessing hides it.

    Guessing would be the expensive kind of forgiving here. Read the wrong way
    round, an unrecognised value silently becomes "this is a finished
    utterance", the windows go back to being filed as incidents and the flagged
    counter goes back to roughly double, with nothing anywhere saying why.
    """
    app, calls = build_test_app(tmp_path)
    invalid = headers(tmp_path, str(uuid.uuid4())) | {"X-VoiceSniffer-Partial": "true"}

    response = post(app, invalid, b"\x00\x01a")

    assert response.status_code == 400
    assert response.json()["code"] == "invalid_partial"
    assert calls == [], "a request refused for a bad header must not reach the recognizer"


def test_an_absent_partial_header_moderates_exactly_as_before(tmp_path) -> None:
    """Every plugin built before this header sends no header at all.

    Those must keep the counts they have today, so absent means `0` rather than
    unknown, and nothing about the response changes either.
    """
    app, _ = build_test_app(tmp_path)

    response = post(app, headers(tmp_path, str(uuid.uuid4())), b"\x00\x01a")

    assert response.status_code == 200
    assert response.json()["severity"] == 2


def test_a_window_is_counted_as_audio_and_not_as_an_incident(tmp_path) -> None:
    """The counting asymmetry, end to end through /v1/usage.

    One player says one thing once. It arrives twice, because the plugin sends
    an early window for speed and then the finished utterance for the phrase
    rules. Two request ids, two decodes, one incident.

    So: two requests, two seconds of audio, one flag. The audio is real work
    this box did and the cost of it was paid; the second severity is a preview
    of the first and counting it is what made the panel disagree with the
    console. `_ServerUsage` explains why these totals are not expected to divide
    into each other.
    """
    settings = load_settings(tmp_path)

    def transcribe(_samples: np.ndarray, _language: str) -> TranscriptionResult:
        return TranscriptionResult(text="Blocked, phrase!", language="en")

    app = create_app(
        settings,
        transcribe=transcribe,
        rule_packs={"en": load_rules(tmp_path)},
        inspect_packet=mono_packet,
        # One decoded second each, so `audio_seconds` is legible rather than
        # rounded away to zero.
        decode_audio=lambda _envelope: np.ones(16_000, dtype=np.float32),
    )
    window = headers(tmp_path, str(uuid.uuid4())) | {"X-VoiceSniffer-Partial": "1"}
    utterance = headers(tmp_path, str(uuid.uuid4())) | {"X-VoiceSniffer-Partial": "0"}

    windowed = post(app, window, b"\x00\x01a")
    finished = post(app, utterance, b"\x00\x01a")
    usage = post_usage(app)

    # The window still gets a real verdict back. It is not moderated any less;
    # it is counted differently.
    assert windowed.status_code == 200
    assert windowed.json()["severity"] == 2
    assert finished.json()["severity"] == 2
    counted = usage.json()["usage"]
    assert counted["requests"] == 2
    assert counted["audio_seconds"] == 2, "the decode happened and the cost of it is real"
    assert counted["flagged"] == 1, "one slur counted twice is the bug this header exists for"
    assert counted["severities"] == {"2": 1}


def test_one_tenant_may_hold_what_the_plugin_is_configured_to_send(tmp_path) -> None:
    """The two numbers were meant to correspond, so this asserts they do.

    ``plugin/src/main/resources/advanced.yml`` sets ``max-in-flight``, and the
    processor's admission pool is sized so one tenant can hold exactly that many
    at once. They were one apart for as long as they existed, because the pool
    was sized at ``workers * 8`` and ``_TenantAdmission`` keeps a slot back, so
    the plugin's last request met a limit neither number mentions.

    Read from the plugin's own file rather than hard-coded, so raising one side
    without the other fails here instead of in production.
    """
    settings = replace(load_settings(tmp_path), workers=4)
    app = create_app(
        settings,
        transcribe=lambda _samples, _language: TranscriptionResult(text="hello", language="en"),
        rule_packs={"en": load_rules(tmp_path)},
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
    )
    admission = _TenantAdmission(capacity=app.state.admission_capacity)

    granted = sum(1 for _ in range(200) if admission.acquire("lic_only") is None)

    assert granted == app.state.single_tenant_admissions
    assert granted == plugin_max_in_flight(), (
        "the plugin's max-in-flight and what one tenant may hold have drifted apart"
    )


def test_a_lone_tenant_pays_only_the_one_reserved_slot() -> None:
    """The whole cost of finding C's fix when there is one customer, as today.

    Seven of eight admissions rather than eight, and no narrowed share, because
    with nobody competing the share is the entire capacity. The eighth is the
    slot held back so a newcomer can ever get in. It costs nothing observable:
    ``processing_slots`` is 4 and is the resource that actually runs out, so the
    seventh admission is already queued behind three others long before the
    reservation is reached. A refusal here is still recorded as the box being
    full, not as contention that is not happening.
    """
    admission = _TenantAdmission(capacity=8)

    granted = sum(1 for _ in range(100) if admission.acquire("lic_only") is None)

    assert granted == 7
    assert admission.share("lic_only") == 8, "a lone tenant must not have its share narrowed"
    assert admission.acquire("lic_only") == "processor_busy", (
        "a lone tenant hitting a full box is not a contention signal"
    )


def test_one_tenant_cannot_take_every_admission_slot_from_another() -> None:
    """Finding C, as an invariant rather than as a race.

    ``max_active_entries`` was global, so one tenant sending that many requests
    at once held every admission and the next tenant was refused outright. Free
    with no rate cap means no unusual traffic is needed to do it.
    """
    admission = _TenantAdmission(capacity=8)
    admission.acquire("lic_a")
    admission.acquire("lic_b")  # b is now competing, so a's share narrows

    granted = sum(1 for _ in range(100) if admission.acquire("lic_a") is None)

    assert granted == 3, f"lic_a took {granted + 1} of 8 with a competitor present"
    assert admission.acquire("lic_b") is None, "lic_b was locked out by lic_a"


def test_a_tenant_that_saturated_an_empty_box_still_lets_a_newcomer_in() -> None:
    """The share on its own does not converge, and this is why.

    Nothing preempts, so a tenant that filled the box while it was alone keeps
    what it took. Refuse the newcomer and it never enters ``in_flight``; never
    entering, it never counts as competing; never competing, the incumbent's
    share never narrows and it holds the box for good. The reserved slot is what
    breaks the loop.
    """
    admission = _TenantAdmission(capacity=8)
    # Bounded, not `while acquire() is None`. An admission control with the cap
    # taken out makes that loop run forever, and a test that hangs on a
    # regression is a test nobody can read the result of.
    taken = sum(1 for _ in range(100) if admission.acquire("lic_incumbent") is None)

    assert taken == 7, f"the incumbent took {taken} of 8 with nothing held back"
    assert admission.acquire("lic_newcomer") is None, "the newcomer never got in at all"
    # And now that the newcomer is visible, the incumbent drains to its share
    # rather than retaking what it releases.
    admission.release("lic_incumbent")
    assert admission.acquire("lic_incumbent") == "processor_busy_tenant_share"


def test_a_tenant_that_goes_quiet_stops_shrinking_everybody_elses_share() -> None:
    # `competing` counts keys, so a released tenant has to leave the table. A
    # tenant remembered at zero would permanently narrow the survivors.
    admission = _TenantAdmission(capacity=8)
    admission.acquire("lic_a")
    admission.acquire("lic_b")
    admission.release("lic_b")

    assert admission.share("lic_a") == 8


def test_a_busy_tenant_does_not_get_a_second_tenant_refused(tmp_path) -> None:
    """The same finding end to end, through the handler that admits requests.

    One tenant fires ``max_active_entries`` requests at once. Before the share
    existed, all of them were admitted, the idempotency cache hit its global cap
    and the second tenant's first ever request came back 503 processor_busy
    without reaching a worker at all.

    The competitor's work still has to finish for the newcomer to be served,
    because nothing here preempts: the point is that the newcomer gets into the
    queue instead of being turned away at the door.
    """
    settings = replace(load_cloud_settings(tmp_path), workers=4)
    release = threading.Event()
    in_transcribe = threading.Semaphore(0)

    def transcribe(_samples: np.ndarray, _language: str) -> TranscriptionResult:
        in_transcribe.release()
        release.wait(10)
        return TranscriptionResult(text="clean greeting", language="en")

    app = create_app(
        settings,
        transcribe=transcribe,
        rule_packs={"en": load_rules(tmp_path)},
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
        queue_wait_seconds=5.0,
    )
    app.state.cloud_resolver._fetch = lambda token: _credential_from(  # type: ignore[method-assign]
        {"active": True, "server_id": "survival-1", "tenant_key": f"lic_{token}"},
        token,
    )

    async def run() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://processor") as client:

            def fire(token: str) -> asyncio.Task:
                request_headers = headers(tmp_path, str(uuid.uuid4())) | {
                    "Authorization": f"Bearer {token}"
                }
                return asyncio.create_task(
                    client.post("/v1/moderate", headers=request_headers, content=b"\x00\x01a")
                )

            noisy = [fire("noisy") for _ in range(8)]
            # All four workers busy, so every admission the noisy tenant is
            # going to get, it has.
            for _ in range(4):
                assert await asyncio.to_thread(in_transcribe.acquire, 10)
            await asyncio.sleep(0.2)
            quiet = fire("quiet")
            await asyncio.sleep(0.2)
            release.set()
            answer = await quiet
            await asyncio.gather(*noisy)
            return answer

    try:
        response = asyncio.run(run())
    finally:
        release.set()
        app.state.cloud_resolver.close()

    assert response.status_code == 200, (
        "a second tenant was refused while the first held every admission slot"
    )


def test_a_share_refusal_is_counted_apart_from_the_box_being_full(tmp_path) -> None:
    """Otherwise the owner cannot see this happening.

    `processor_busy` on the wire either way, because the plugin knows that code
    and the advice is the same. But "the box is full" and "one customer is
    crowding out the others" call for different responses, one a bigger machine
    and one a rate cap, and telling them apart afterwards is only possible if
    they were written down differently at the time.
    """
    settings = replace(load_cloud_settings(tmp_path), workers=4)
    release = threading.Event()
    in_transcribe = threading.Semaphore(0)

    def transcribe(_samples: np.ndarray, _language: str) -> TranscriptionResult:
        in_transcribe.release()
        release.wait(10)
        return TranscriptionResult(text="clean greeting", language="en")

    app = create_app(
        settings,
        transcribe=transcribe,
        rule_packs={"en": load_rules(tmp_path)},
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
        queue_wait_seconds=5.0,
    )
    app.state.cloud_resolver._fetch = lambda token: _credential_from(  # type: ignore[method-assign]
        {"active": True, "server_id": "survival-1", "tenant_key": f"lic_{token}"},
        token,
    )

    async def run() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://processor") as client:

            def fire(token: str) -> asyncio.Task:
                request_headers = headers(tmp_path, str(uuid.uuid4())) | {
                    "Authorization": f"Bearer {token}"
                }
                return asyncio.create_task(
                    client.post("/v1/moderate", headers=request_headers, content=b"\x00\x01a")
                )

            quiet = fire("quiet")
            assert await asyncio.to_thread(in_transcribe.acquire, 10)
            # One more than the noisy tenant's share, asked of the app rather
            # than hard-coded. This used to fire exactly 8, which was
            # `max_active_entries` when that was `max(8, workers * 2)`; when the
            # cap was resized the number stopped meaning anything and the test
            # would have passed while testing nothing.
            #
            # Ceiling, matching `_TenantAdmission.share`, because the capacity is
            # odd once the reserved slot is added to it and a floor here would
            # fire exactly the share and see no refusal at all.
            over_share = math.ceil(app.state.admission_capacity / 2) + 1
            noisy = [fire("noisy") for _ in range(over_share)]
            await asyncio.sleep(0.3)
            release.set()
            answers = await asyncio.gather(*noisy)
            await quiet
            usage = await client.get("/v1/usage", headers={"Authorization": "Bearer noisy"})
            return list(answers), usage.json()

    try:
        responses, usage = asyncio.run(run())
    finally:
        release.set()
        app.state.cloud_resolver.close()

    refused = [response for response in responses if response.status_code == 503]
    assert refused, "the noisy tenant was never capped"
    assert all(response.json()["code"] == "processor_busy" for response in refused), (
        "the wire code changed, which the plugin does not expect"
    )
    assert usage["usage"]["outcomes"].get("processor_busy_tenant_share") == len(refused), (
        "the share refusal is invisible to the owner"
    )


def build_test_app(
    tmp_path: Path,
    *,
    settings: ProcessorSettings | None = None,
    transcription_text: str = "Blocked, phrase!",
    trace_sink=None,
    stage_sink=None,
):
    transcribe_calls: list[str] = []

    def transcribe(_samples: np.ndarray, language: str) -> TranscriptionResult:
        transcribe_calls.append(language)
        return TranscriptionResult(text=transcription_text, language="en")

    app = create_app(
        load_settings(tmp_path) if settings is None else settings,
        transcribe=transcribe,
        rule_packs={"en": load_rules(tmp_path)},
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
        trace_sink=trace_sink,
        stage_sink=stage_sink,
    )
    return app, transcribe_calls


def load_settings(tmp_path: Path) -> ProcessorSettings:
    token_file = tmp_path / "tokens.json"
    token_file.write_text(json.dumps({"server-a": "secret-token"}), encoding="utf-8")
    return ProcessorSettings.from_environment({"VOICESNIFFER_TOKENS_FILE": str(token_file)})


def load_cloud_settings(tmp_path: Path) -> ProcessorSettings:
    """A cloud node, which is the only shape that has more than one tenant.

    tokens.json is one customer's file and every credential out of it has no
    tenant key at all, so telling two tenants apart needs the introspection
    path. The URL is never dialled: the tests replace ``_fetch``.
    """
    token_file = tmp_path / "tokens.json"
    token_file.write_text(json.dumps({"break-glass": "secret-token"}), encoding="utf-8")
    return ProcessorSettings.from_environment(
        {
            "VOICESNIFFER_TOKENS_FILE": str(token_file),
            "VOICESNIFFER_INTROSPECTION_URL": "http://127.0.0.1:18080/api/internal/introspect",
            "VOICESNIFFER_INTROSPECTION_TOKEN": "service-secret",
        }
    )


def load_rules(tmp_path: Path, language: str = "en") -> RulePack:
    rule_file = tmp_path / f"{language}.yml"
    rule_file.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "language": language,
                "rules": [
                    {
                        "id": f"{language}.test.blocked",
                        "term": "blocked phrase",
                        "category": "harassment",
                        "severity": 2,
                        "match": "exact",
                        "variants": [],
                        "context_required": False,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return RulePack.load(rule_file)


def headers(_tmp_path: Path, request_id: str) -> dict[str, str]:
    return {
        "Authorization": "Bearer secret-token",
        "Content-Type": "application/vnd.voicesniffer.opus.v1",
        "Accept": "application/json",
        "X-Request-Id": request_id,
        "X-VoiceSniffer-Player-Id": str(uuid.uuid4()),
        "X-VoiceSniffer-Language": "en",
        "X-VoiceSniffer-Preroll-Samples": "0",
    }


def mono_packet(_packet: bytes) -> OpusPacketInfo:
    return OpusPacketInfo(samples_48k=960, channels=1)


def plugin_max_in_flight() -> int:
    """The plugin's `processor-tuning.max-in-flight`. See CONTRIBUTING.md."""
    return 32


def post_usage(app) -> httpx.Response:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://processor") as client:
            return await client.get(
                "/v1/usage",
                headers={"Authorization": "Bearer secret-token"},
            )

    return asyncio.run(request())


def post(app, request_headers: dict[str, str], body: bytes) -> httpx.Response:
    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://processor") as client:
            return await client.post("/v1/moderate", headers=request_headers, content=body)

    return asyncio.run(request())


def assert_timing_headers(response: httpx.Response) -> None:
    for name in (
        "X-VoiceSniffer-Decode-Ms",
        "X-VoiceSniffer-Transcribe-Ms",
        "X-VoiceSniffer-Rules-Ms",
    ):
        assert int(response.headers[name]) >= 0


def _language_app(tmp_path, transcript: str = "buzerant"):
    """Both shipped packs loaded, so pack selection is observable in the verdict."""
    rules_root = Path(__file__).parents[1] / "rules"
    return create_app(
        load_settings(tmp_path),
        transcribe=lambda _samples, _language: TranscriptionResult(
            text=transcript, language="auto"
        ),
        rule_packs={
            "en": RulePack.load(rules_root / "en.yml"),
            "cs": RulePack.load(rules_root / "cs.yml"),
        },
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
    )


def test_a_language_list_selects_exactly_those_packs(tmp_path) -> None:
    # The whole reason the header takes a list. `buzerant` is Czech, so a server
    # that declares English only must not be told about it: at twenty-five packs
    # `auto` would run Polish, Slovak and Slovenian rules over Czech speech and
    # collide for real.
    request_id = str(uuid.uuid4())
    app = _language_app(tmp_path)

    english_only = post(
        app, headers(tmp_path, request_id) | {"X-VoiceSniffer-Language": "en"}, b"\x00\x01a"
    )
    assert english_only.status_code == 200
    assert english_only.json()["severity"] == 0

    both = post(
        app,
        headers(tmp_path, str(uuid.uuid4())) | {"X-VoiceSniffer-Language": "en,cs"},
        b"\x00\x01a",
    )
    assert both.status_code == 200
    assert both.json()["severity"] == 3
    assert both.json()["matches"][0]["rule_id"] == "cs.hate.homophobic"


@pytest.mark.parametrize(
    "header",
    [
        "en,,cs",  # empty member
        "en,cs,en",  # duplicate, which would double-count a match
        "en,EN",  # uppercase is not a code here
        "en cs",  # space rather than comma
        ",".join(f"l{index:02d}" for index in range(26)),  # over the cap
    ],
)
def test_a_malformed_language_list_is_refused(tmp_path, header: str) -> None:
    app = _language_app(tmp_path)
    response = post(
        app,
        headers(tmp_path, str(uuid.uuid4())) | {"X-VoiceSniffer-Language": header},
        b"\x00\x01a",
    )
    assert response.status_code == 400
    assert response.json()["code"] == "invalid_language"


def test_an_unknown_language_in_the_list_matches_nothing_rather_than_erroring(tmp_path) -> None:
    # `de` is a language the model supports but for which no pack is installed.
    # rule_packs.get returns None and the existing filter drops it. The request
    # still has to succeed, because the English half of the list is legitimate.
    request_id = str(uuid.uuid4())
    app = _language_app(tmp_path)

    response = post(
        app, headers(tmp_path, request_id) | {"X-VoiceSniffer-Language": "en,de"}, b"\x00\x01a"
    )

    assert response.status_code == 200
    assert response.json()["severity"] == 0


def test_a_language_with_no_installed_pack_is_refused_rather_than_moderated_by_nothing(
    tmp_path,
) -> None:
    """The `fi` and `mt` hole, at the door.

    ``models.toml`` declares 25 languages for the production model and 23 rule
    packs ship. A server pinned to one of the other two ran no rules at all and
    was answered ``200 {"severity": 0, "matches": []}``: no moderation event, no
    log line, and the audio seconds still billed. The operator's only evidence
    that anything was wrong would have been that nobody was ever muted.
    """
    app = _language_app(tmp_path, transcript="buzerant")

    response = post(
        app,
        headers(tmp_path, str(uuid.uuid4())) | {"X-VoiceSniffer-Language": "fi"},
        b"\x00\x01a",
    )

    assert response.status_code == 400
    assert response.json()["code"] == "language_unsupported"


def test_a_language_with_no_pack_is_refused_even_when_it_is_the_only_one_asked_for(
    tmp_path,
) -> None:
    # The partial case above is deliberately tolerant; this one cannot be. Every
    # requested code resolving to nothing leaves no legitimate half to serve.
    app = _language_app(tmp_path)

    response = post(
        app,
        headers(tmp_path, str(uuid.uuid4())) | {"X-VoiceSniffer-Language": "fi,mt"},
        b"\x00\x01a",
    )

    assert response.status_code == 400
    assert response.json()["code"] == "language_unsupported"


def test_a_licensing_api_outage_is_503_and_not_a_rejected_token(tmp_path) -> None:
    """The processor must not blame the customer for our own outage.

    ``resolve`` used to return ``None`` both for "no such token" and for "the
    licensing API could not be reached", and the handler turns ``None`` into
    ``401 unauthorized``. The plugin never retries a 401, surfaces it to the
    operator as ``http_401`` -- which reads as "your token is bad" -- and then
    keeps voice chat running unmoderated, because ``fail-closed`` is off by
    default. One redeploy of the licensing API was therefore enough to stop
    moderation on every paying server at once. 503 is in the plugin's retryable
    set, so a blip inside the request deadline now rides out instead.
    """
    app = create_app(
        load_cloud_settings(tmp_path),
        transcribe=lambda _samples, _language: TranscriptionResult(text="hi", language="en"),
        rule_packs={"en": load_rules(tmp_path)},
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
    )

    def unreachable(_token: str):
        raise IntrospectionUnavailable

    app.state.cloud_resolver._fetch = unreachable  # type: ignore[method-assign]
    try:
        response = post(
            app,
            headers(tmp_path, str(uuid.uuid4())) | {"Authorization": "Bearer vsc-live"},
            b"\x00\x01a",
        )
    finally:
        app.state.cloud_resolver.close()

    assert response.status_code == 503
    assert response.json()["code"] == "license_check_unavailable"


def test_a_token_the_licensing_api_rejects_is_still_401(tmp_path) -> None:
    # The other half of the same fix. Telling an outage apart from a refusal is
    # only worth anything if a real refusal still says `unauthorized`.
    app = create_app(
        load_cloud_settings(tmp_path),
        transcribe=lambda _samples, _language: TranscriptionResult(text="hi", language="en"),
        rule_packs={"en": load_rules(tmp_path)},
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
    )
    app.state.cloud_resolver._fetch = lambda _token: None  # type: ignore[method-assign]
    try:
        response = post(
            app,
            headers(tmp_path, str(uuid.uuid4())) | {"Authorization": "Bearer vsc-stolen"},
            b"\x00\x01a",
        )
    finally:
        app.state.cloud_resolver.close()

    assert response.status_code == 401
    assert response.json()["code"] == "unauthorized"


def test_the_usage_endpoint_separates_an_outage_from_a_rejected_token(tmp_path) -> None:
    app = create_app(
        load_cloud_settings(tmp_path),
        transcribe=lambda _samples, _language: TranscriptionResult(text="hi", language="en"),
        rule_packs={"en": load_rules(tmp_path)},
        inspect_packet=mono_packet,
        decode_audio=lambda _envelope: np.ones(320, dtype=np.float32),
    )

    def unreachable(_token: str):
        raise IntrospectionUnavailable

    app.state.cloud_resolver._fetch = unreachable  # type: ignore[method-assign]

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://processor") as client:
            return await client.get("/v1/usage", headers={"Authorization": "Bearer vsc-live"})

    try:
        response = asyncio.run(request())
    finally:
        app.state.cloud_resolver.close()

    assert response.status_code == 503
    assert response.json()["code"] == "license_check_unavailable"
