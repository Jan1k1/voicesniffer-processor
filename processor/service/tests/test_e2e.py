import asyncio
import base64
import json
import time
import uuid
from pathlib import Path

import httpx
import pytest

from voicesniffer_processor.app import create_app
from voicesniffer_processor.models import TranscriptionResult
from voicesniffer_processor.rules import RulePack
from voicesniffer_processor.settings import ProcessorSettings


@pytest.mark.native
def test_native_opus_to_tier_one_verdict_stays_inside_deadline(tmp_path: Path) -> None:
    try:
        __import__("opuslib_next")
    except Exception as exception:
        pytest.skip(f"libopus unavailable: {type(exception).__name__}")

    token_file = tmp_path / "tokens.json"
    token_file.write_text(json.dumps({"server-a": "secret-token"}), encoding="utf-8")
    fixture_root = Path(__file__).resolve().parents[3] / "integration" / "fixtures"
    frame = base64.b64decode((fixture_root / "neutral-opus-frame.b64").read_text(encoding="ascii"))
    body = len(frame).to_bytes(2, "big") + frame
    request_id = str(uuid.uuid4())
    app = create_app(
        ProcessorSettings.from_environment({"VOICESNIFFER_TOKENS_FILE": str(token_file)}),
        transcribe=lambda _samples, _language: TranscriptionResult(
            text="blocked phrase", language="en"
        ),
        rule_packs={"en": RulePack.load(fixture_root / "rules" / "en.yml")},
    )

    async def request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://processor") as client:
            return await client.post(
                "/v1/moderate",
                headers={
                    "Authorization": "Bearer secret-token",
                    "Content-Type": "application/vnd.voicesniffer.opus.v1",
                    "Accept": "application/json",
                    "X-Request-Id": request_id,
                    "X-VoiceSniffer-Player-Id": str(uuid.uuid4()),
                    "X-VoiceSniffer-Language": "en",
                    "X-VoiceSniffer-Preroll-Samples": "0",
                },
                content=body,
            )

    started = time.perf_counter()
    response = asyncio.run(request())
    elapsed = time.perf_counter() - started

    assert response.status_code == 200
    assert response.headers["x-request-id"] == request_id
    assert response.json()["severity"] == 2
    assert response.json()["matches"][0]["rule_id"] == "en.test.blocked"
    assert elapsed < 2.5
