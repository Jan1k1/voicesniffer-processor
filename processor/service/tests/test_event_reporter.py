"""Moderation history: the only feed in this service that carries a transcript.

The usage tests next door protect a payload that must never grow a text field.
These protect the opposite thing: a payload that carries text on purpose, and
therefore has to be exactly right about who it is for, when it is sent at all,
and what it is never allowed to do.

Four properties, and each one is a promise made in a legal document:

- a self-hosted processor has no reporter, no task and no socket;
- nothing is sent for speech that fired no rule;
- audio never appears, and neither does any field outside the DPA's list;
- the transcript reaches no log, at any level, on this side.
"""

import asyncio
import datetime
import json
import logging
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml

from voicesniffer_processor.app import create_app
from voicesniffer_processor.event_reporter import (
    MAX_MATCHED_TEXT_CHARS,
    MAX_TRANSCRIPT_CHARS,
    TASK_NAME,
    EventReporter,
    FlaggedEvent,
    events_url_from,
    flagged_event_from,
)
from voicesniffer_processor.internal_api import InternalApiError
from voicesniffer_processor.models import TranscriptionResult, VerdictMatch
from voicesniffer_processor.opus import OpusPacketInfo
from voicesniffer_processor.rules import RulePack
from voicesniffer_processor.settings import ProcessorSettings

INTROSPECTION_URL = "http://licensing.internal:8080/api/internal/introspect"
EVENTS_URL = "http://licensing.internal:8080/api/internal/events"
DECODED_SAMPLES = 32_000
AT = datetime.datetime(2026, 7, 31, 15, 0, tzinfo=datetime.UTC)


# ------------------------------------------------------- the payload itself


def test_an_event_carries_the_shape_the_contract_asks_for() -> None:
    """Section 4 of docs/cloud-contract.md, field for field.

    Asserted as one whole dict rather than key by key, on purpose and for the
    same reason the usage test is: the failure worth catching here is a field
    nobody meant to send, and a key-by-key test cannot see one. On this feed
    that field would be a player name, an IP address or a clip of audio.
    """
    made = reporter()

    assert made.payload([event()]) == {
        "reported_at": "2026-07-31T15:00:00.000Z",
        "events": [
            {
                "tenant_key": "vst_9f3",
                "server_id": "survival-1",
                "request_id": "5b1f0a3e-0000-4000-8000-000000000001",
                "player_uuid": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
                "occurred_at": "2026-07-31T14:59:58.000Z",
                "rule_id": "en.harassment.slur",
                "category": "harassment",
                "severity": 2,
                "rule_count": 1,
                "matched_text": "the phrase",
                "transcript": "he said the phrase out loud",
            }
        ],
    }


def test_the_payload_holds_no_audio_and_no_bytes() -> None:
    """Nothing on this feed is binary, at any depth.

    Audio is the one thing the DPA promises is never written anywhere, and the
    only object in this process that could carry it onto a socket is a payload.
    So the assertion is about types rather than about field names: a future
    field called anything at all cannot smuggle a clip past this.
    """
    body = json.dumps(reporter().payload([event()]))

    assert isinstance(body, str)
    for value in values_in(json.loads(body)):
        assert not isinstance(value, (bytes, bytearray))


def test_long_text_is_truncated_before_it_reaches_the_wire() -> None:
    """A decode gone wrong must not put an unbounded string on the network.

    The receiver has its own, larger, bound. This one exists so that hitting the
    receiver's bound means a bug here rather than a customer saying something
    long, because tripping it there costs a whole batch of other customers'
    events.
    """
    entry = reporter().payload([
        event(transcript="a" * 5_000, matched_text="b" * 5_000)
    ])["events"][0]

    assert len(entry["transcript"]) == MAX_TRANSCRIPT_CHARS
    assert len(entry["matched_text"]) == MAX_MATCHED_TEXT_CHARS


# ------------------------------------------------------- what counts as flagged


def test_nothing_is_built_for_an_utterance_that_matched_no_rule() -> None:
    """The gate, and the reason ordinary speech leaves no trace anywhere.

    It lives in the builder rather than at the call site so that "nothing is
    stored for speech that fired no rule" is a property of the only function
    that can produce one of these, not a check somebody could forget to copy
    into a second caller.
    """
    assert flagged_event_from(
        tenant_key="vst_9f3",
        server_id="survival-1",
        request_id="r",
        player_uuid="9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
        occurred_at=AT,
        transcript="an entirely unremarkable sentence",
        matches=[],
    ) is None


def test_the_match_named_is_the_one_that_set_the_severity() -> None:
    """Several rules can fire on one sentence; the record names the worst.

    That is the one an operator would name if asked why this row is in his list,
    and `rule_count` is what says the others existed. The transcript is carried
    once rather than once per match, which is why the answer has to be a choice
    rather than a list.
    """
    built = flagged_event_from(
        tenant_key="vst_9f3",
        server_id="survival-1",
        request_id="r",
        player_uuid="9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
        occurred_at=AT,
        transcript="two things at once",
        matches=[
            VerdictMatch(
                rule_id="en.spam.mild",
                category="spam",
                severity=1,
                matched_text="two things",
                context_required=False,
            ),
            VerdictMatch(
                rule_id="en.threat.bad",
                category="threat",
                severity=3,
                matched_text="at once",
                context_required=False,
            ),
        ],
    )

    assert (built.rule_id, built.severity, built.rule_count) == ("en.threat.bad", 3, 2)


def test_a_context_required_match_still_produces_an_event() -> None:
    """Deliberately looser than the plugin's enforcement rule.

    The plugin ignores a context-required match when deciding whether to mute,
    because acting on one automatically is what a false positive looks like. The
    history is the other half of that: an owner reading his own server's log is
    exactly the context the rule was waiting for, and hiding these would hide the
    flags most worth a human's eye.
    """
    built = flagged_event_from(
        tenant_key="vst_9f3",
        server_id="survival-1",
        request_id="r",
        player_uuid="9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
        occurred_at=AT,
        transcript="ambiguous",
        matches=[
            VerdictMatch(
                rule_id="en.hate.maybe",
                category="hate",
                severity=1,
                matched_text="ambiguous",
                context_required=True,
            )
        ],
    )

    assert built is not None and built.rule_id == "en.hate.maybe"


# ------------------------------------------------------- the self-hosted rule


def test_a_self_hosted_processor_never_reports_an_event(tmp_path: Path, monkeypatch) -> None:
    """The hard rule, restated for the feed that carries text.

    Not "reports nothing useful" and not "reports to nowhere": no reporter
    object, no task, no socket. On a customer's own hardware what their players
    say has never left the box and this module must not be what changes that.
    The utterance below really does flag, so the absence proves the gate rather
    than an absence of anything to send.
    """
    opened: list[str] = []
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout=None: opened.append(request.full_url),
    )
    app = build_app(tmp_path, environment={})

    async def scenario() -> tuple[list[str], int]:
        async with app.router.lifespan_context(app):
            response = await moderate(app, token="operator-token")
            return [task.get_name() for task in asyncio.all_tasks()], response.status_code

    task_names, status = asyncio.run(scenario())

    assert app.state.event_reporter is None
    assert status == 200, "moderation still has to work without a reporter"
    assert TASK_NAME not in task_names
    assert opened == [], f"a self-hosted processor sent a transcript: {opened}"


def test_an_events_url_on_its_own_does_not_make_a_processor_report(tmp_path: Path) -> None:
    """Half a cloud compose file, copied onto a customer's own box.

    The switch is the introspection settings, because those are what make a node
    multi-tenant. An events URL without them is a self-hosted processor with a
    stray environment variable, and the rule is not "sends to wherever it was
    told" but "does not send".
    """
    app = build_app(tmp_path, environment={"VOICESNIFFER_EVENTS_URL": EVENTS_URL})

    assert app.state.event_reporter is None


def test_a_cloud_node_that_cannot_address_events_still_moderates(tmp_path: Path) -> None:
    """History is a convenience; the verdict is what the customer is paying for."""
    app = build_cloud_app(tmp_path, introspection_url="http://api:8080/api/internal/tokens")

    assert app.state.event_reporter is None
    assert asyncio.run(moderate(app, token="operator-token")).status_code == 200


@pytest.mark.parametrize(
    "introspection_url",
    [
        "http://api:8080/api/internal/tokens",
        "http://api:8080/api/internal/introspect/",
        "",
    ],
)
def test_events_url_is_only_derived_from_a_real_introspection_url(introspection_url: str) -> None:
    """A payload carrying a transcript must land where it was meant to.

    Guessing a sibling path from anything that is not exactly `/introspect` could
    aim this feed at whatever endpoint happened to be there. Of everything this
    process sends, this is the one where that matters most.
    """
    assert events_url_from(introspection_url) == ""


def test_the_events_url_is_derived_from_the_introspection_one(tmp_path: Path) -> None:
    app = build_cloud_app(tmp_path)

    assert app.state.event_reporter.events_url == EVENTS_URL


def test_an_explicit_events_url_overrides_the_derived_one(tmp_path: Path) -> None:
    app = build_cloud_app(tmp_path, events_url="http://elsewhere:9000/events")

    assert app.state.event_reporter.events_url == "http://elsewhere:9000/events"


# ------------------------------------------------------- the log

def test_no_transcript_reaches_any_log_at_any_level(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    """The whole run, at DEBUG, with a failing API and a phrase to look for.

    A log line is a second copy of a transcript with no retention deadline on
    it, and the table this feed writes to exists precisely because its rows have
    one. So the assertion is not "the reporter does not log the transcript", it
    is that the string never appears anywhere in the captured output of a
    complete moderation plus a failed delivery plus a shutdown flush.
    """
    api = FakeLicensingAPI(events_error=urllib.error.URLError("down"))
    monkeypatch.setattr(urllib.request, "urlopen", api.urlopen)
    app = build_cloud_app(tmp_path)

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            await moderate(app)
            await app.state.event_reporter.flush_once()

    with caplog.at_level(logging.DEBUG):
        asyncio.run(scenario())

    assert "blocked phrase" not in caplog.text
    assert "blocked phrase" not in "".join(
        str(record.args) for record in caplog.records if record.args
    )


# ------------------------------------------------------- delivery


def test_a_flagged_utterance_reaches_the_licensing_api(tmp_path: Path, monkeypatch) -> None:
    """End to end on a cloud node: speak, flag, deliver, attributed to a tenant."""
    api = FakeLicensingAPI()
    monkeypatch.setattr(urllib.request, "urlopen", api.urlopen)
    app = build_cloud_app(tmp_path)

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            await moderate(app)
            await app.state.event_reporter.flush_once()

    asyncio.run(scenario())

    assert len(api.event_reports) == 1
    delivered = api.event_reports[0]["events"]
    assert len(delivered) == 1
    assert delivered[0]["tenant_key"] == "vst_9f3"
    assert delivered[0]["server_id"] == "survival-1"
    assert delivered[0]["transcript"] == "blocked phrase"
    assert delivered[0]["rule_id"] == "en.test.blocked"
    assert api.event_authorizations == ["Bearer service-secret"]


def test_a_clean_utterance_is_never_delivered(tmp_path: Path, monkeypatch) -> None:
    """The ordinary case, which has to leave no record at all.

    This is the difference between a moderation log and a recording of a voice
    channel, and it is the single most important assertion in this file.
    """
    api = FakeLicensingAPI()
    monkeypatch.setattr(urllib.request, "urlopen", api.urlopen)
    app = build_cloud_app(tmp_path, transcript="nothing worth repeating")

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            await moderate(app)
            await app.state.event_reporter.flush_once()

    asyncio.run(scenario())

    assert api.event_reports == []


def test_a_credential_with_no_tenant_key_files_nothing(tmp_path: Path, monkeypatch) -> None:
    """The operator's own break-glass token on a cloud box.

    It comes out of tokens.json and was never told a tenant key, so there is
    nobody to file the row under, nobody who could read it back and nobody whose
    retention window it belongs in. Dropped rather than stored unattributed.
    """
    api = FakeLicensingAPI()
    monkeypatch.setattr(urllib.request, "urlopen", api.urlopen)
    app = build_cloud_app(tmp_path)

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            await moderate(app, token="operator-token")
            await app.state.event_reporter.flush_once()

    asyncio.run(scenario())

    assert api.event_reports == []


def test_an_early_window_files_nothing_and_the_utterance_files_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """One slur, two requests, one row in the moderation history.

    The plugin sends an early window while the sentence is still running and
    then the finished utterance. Different request ids and different audio, so
    neither the idempotency cache nor the receiver's unique index can tell they
    are the same incident: only the sender knows, and it says so with
    ``X-VoiceSniffer-Partial``.

    Filed under the utterance's request id, not the window's, which is the one
    that carried the whole sentence.
    """
    api = FakeLicensingAPI()
    monkeypatch.setattr(urllib.request, "urlopen", api.urlopen)
    app = build_cloud_app(tmp_path)
    window = headers("cloud-token") | {"X-VoiceSniffer-Partial": "1"}
    utterance = headers("cloud-token") | {"X-VoiceSniffer-Partial": "0"}

    async def scenario() -> list[httpx.Response]:
        async with app.router.lifespan_context(app):
            answers = [
                await moderate(app, request_headers=window),
                await moderate(app, request_headers=utterance),
            ]
            await app.state.event_reporter.flush_once()
            return answers

    answers = asyncio.run(scenario())

    # The window is moderated as fully as anything else. What changes is what it
    # is counted as, not whether it is looked at.
    assert [answer.json()["severity"] for answer in answers] == [2, 2]
    assert [len(report["events"]) for report in api.event_reports] == [1]
    assert api.event_reports[0]["events"][0]["request_id"] == utterance["X-Request-Id"]


def test_a_retried_utterance_is_filed_once(tmp_path: Path, monkeypatch) -> None:
    """The plugin retries inside its deadline; the owner must see one incident.

    Capture sits inside the idempotency cache's own body, so the second delivery
    of the same request id is answered from the cache and never reaches it.
    """
    api = FakeLicensingAPI()
    monkeypatch.setattr(urllib.request, "urlopen", api.urlopen)
    app = build_cloud_app(tmp_path)
    request_headers = headers("cloud-token")

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            await moderate(app, request_headers=request_headers)
            await moderate(app, request_headers=request_headers)
            await app.state.event_reporter.flush_once()

    asyncio.run(scenario())

    assert [len(report["events"]) for report in api.event_reports] == [1]


def test_a_failed_delivery_keeps_the_events_for_the_next_attempt() -> None:
    """A licensing API that is briefly down must not erase a moderation record.

    The batch goes back on the front of the queue, so the order an owner reads
    them in survives the retry too.
    """
    made = reporter(post_error=urllib.error.URLError("down"))
    made.record(event())

    assert asyncio.run(made.flush_once()) is False
    assert len(made._queue) == 1

    made.posted_error = None
    assert asyncio.run(made.flush_once()) is True
    assert len(made._queue) == 0


def test_the_queue_is_bounded_and_drops_the_oldest() -> None:
    """A flag storm during an outage may not grow without limit.

    Losing the oldest entries is the deliberate trade: a moderation request may
    never wait on this feed, and the flags somebody is asking about are the
    recent ones.
    """
    made = reporter(max_queue=2)
    for index in range(4):
        made.record(event(request_id=f"r{index}"))

    assert [held.request_id for held in made._queue] == ["r2", "r3"]


def test_recording_never_raises_into_moderation() -> None:
    """`record` does one thing: append. No I/O, no validation that can throw."""
    made = reporter(max_queue=1)

    assert made.record(event()) is None
    assert made.record(event()) is None


def test_a_batch_is_split_rather_than_sent_whole() -> None:
    """The receiver caps a batch at 500; this stays under it by construction."""
    made = reporter(max_batch=2)
    for index in range(5):
        made.record(event(request_id=f"r{index}"))

    assert asyncio.run(made.flush_once()) is True
    assert len(made._queue) == 3


def test_shutdown_flushes_what_is_left(tmp_path: Path, monkeypatch) -> None:
    """Unlike usage, which is cumulative and loses nothing to a missed report.

    An event is a fact that happened once. If the process is stopping, the queue
    is the only copy, and one post on the way out is the difference between a
    moderation record with a gap at every deploy and one without.
    """
    api = FakeLicensingAPI()
    monkeypatch.setattr(urllib.request, "urlopen", api.urlopen)
    app = build_cloud_app(tmp_path)

    async def scenario() -> None:
        async with app.router.lifespan_context(app):
            await moderate(app)

    asyncio.run(scenario())

    assert len(api.event_reports) == 1
    assert api.event_reports[0]["events"][0]["transcript"] == "blocked phrase"


def test_starting_twice_leaves_one_task(tmp_path: Path, monkeypatch) -> None:
    api = FakeLicensingAPI()
    monkeypatch.setattr(urllib.request, "urlopen", api.urlopen)
    app = build_cloud_app(tmp_path)
    made = app.state.event_reporter

    async def scenario() -> int:
        made.start()
        made.start()
        running = len([task for task in asyncio.all_tasks() if task.get_name() == TASK_NAME])
        await made.stop()
        return running

    assert asyncio.run(scenario()) == 1


def test_stopping_a_reporter_that_never_started_is_not_an_error() -> None:
    assert asyncio.run(reporter().stop()) is None


# ------------------------------------------------- refused, retried, or lost


def test_a_refused_batch_is_discarded_rather_than_retried_forever(caplog) -> None:
    """A 400 is the one failure that must not go back on the queue.

    The same bytes are refused every time, so a requeue means the head of the
    queue never moves again: every event recorded afterwards is dropped by the
    bound while one poisoned batch is retried for the life of the process. One
    lost batch, said out loud, beats a feed that is silently dead.
    """
    made = reporter(post_error=refusal(400, "Invalid event payload."))
    made.record(event())

    with caplog.at_level(logging.DEBUG):
        assert asyncio.run(made.flush_once()) is False

    assert len(made._queue) == 0
    assert "event_report_rejected" in caplog.text
    assert "status=400" in caplog.text
    assert "reason=payload_refused" in caplog.text
    assert "discarded=1" in caplog.text


def test_a_licensing_api_that_is_down_keeps_the_batch(caplog) -> None:
    """The other half of the same decision. A 500 will answer properly later."""
    made = reporter(post_error=refusal(500, "Internal Server Error"))
    made.record(event())

    with caplog.at_level(logging.DEBUG):
        assert asyncio.run(made.flush_once()) is False

    assert len(made._queue) == 1
    assert "event_report_failed" in caplog.text
    assert "status=500" in caplog.text
    assert "reason=api_error" in caplog.text


def test_a_wrong_service_token_is_named_rather_than_retried_silently(caplog) -> None:
    # The deployment failure this feed is most likely to have, and the one a
    # bare exception class name says nothing about.
    made = reporter(post_error=refusal(401, "Unauthorized"))
    made.record(event())

    with caplog.at_level(logging.DEBUG):
        asyncio.run(made.flush_once())

    assert "reason=service_token_refused" in caplog.text


def test_the_failure_line_carries_no_transcript_and_no_answer_from_the_wire(caplog) -> None:
    """The module's rule holds on the failure path too.

    The API's own message is not logged here even though it is available: this
    process had just put a transcript on that socket, and what comes back is not
    something this feed copies into a file with no retention deadline on it.
    """
    made = reporter(post_error=refusal(400, "he said the phrase out loud"))
    made.record(event(transcript="he said the phrase out loud"))

    with caplog.at_level(logging.DEBUG):
        asyncio.run(made.flush_once())

    assert "phrase" not in caplog.text


def test_the_delay_grows_while_the_api_is_down_and_resets_when_it_returns() -> None:
    """A feed that keeps posting every fifteen seconds through a long outage is
    load on an API that is already unwell. The ceiling is low anyway, because
    the queue is bounded and a longer gap loses more of a flag storm."""
    made = reporter(post_error=urllib.error.URLError("down"), interval_seconds=15.0)
    made.record(event())
    first = made.delay_seconds()

    asyncio.run(made.flush_once())
    after_one = made.delay_seconds()
    asyncio.run(made.flush_once())
    after_two = made.delay_seconds()
    for _ in range(20):
        asyncio.run(made.flush_once())
    at_ceiling = made.delay_seconds()

    made.posted_error = None
    asyncio.run(made.flush_once())

    assert (first, after_one, after_two) == (15.0, 30.0, 60.0)
    assert at_ceiling == 120.0
    assert made.delay_seconds() == 15.0


def test_events_lost_to_the_bound_are_reported_during_the_outage(caplog) -> None:
    """The line used to sit after a successful delivery, which is the one moment
    it cannot happen: the queue only overflows while the API is unreachable, and
    that path returned before reaching it. A flag storm during an outage lost
    events and said nothing, which is the failure this module is written
    against."""
    made = reporter(post_error=urllib.error.URLError("down"), max_queue=2, max_batch=2)
    for index in range(5):
        made.record(event(request_id=f"r{index}"))

    with caplog.at_level(logging.DEBUG):
        asyncio.run(made.flush_once())
        asyncio.run(made.flush_once())

    assert "event_reports_dropped" in caplog.text
    assert "reason=queue_full" in caplog.text


def test_an_event_the_api_files_under_nobody_is_a_warning(caplog) -> None:
    """202 with `ignored` is a flagged utterance with no owner and no reader.

    It looks like success from this side, which is exactly why it is said out
    loud: a dashboard reading zero while the log reads fine is how this feed
    failed the first time.
    """
    made = reporter()
    made._post = lambda payload: {"accepted": 0, "ignored": 1}
    made.record(event())

    with caplog.at_level(logging.DEBUG):
        assert asyncio.run(made.flush_once()) is True

    assert "event_report_ignored" in caplog.text
    assert "reason=unknown_tenant_key" in caplog.text


# ---------------------------------------------------------------- helpers


def refusal(status: int, message: str) -> InternalApiError:
    """What `internal_api.post_json` raises for a status, without a socket."""
    return InternalApiError(
        kind="HTTPError",
        status=status,
        detail=message,
        retryable=status >= 500 or status == 429,
    )


def values_in(value) -> list:
    if isinstance(value, dict):
        return [found for item in value.values() for found in values_in(item)]
    if isinstance(value, list):
        return [found for item in value for found in values_in(item)]
    return [value]


def event(**overrides) -> FlaggedEvent:
    fields = {
        "tenant_key": "vst_9f3",
        "server_id": "survival-1",
        "request_id": "5b1f0a3e-0000-4000-8000-000000000001",
        "player_uuid": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
        "occurred_at": datetime.datetime(2026, 7, 31, 14, 59, 58, tzinfo=datetime.UTC),
        "rule_id": "en.harassment.slur",
        "category": "harassment",
        "severity": 2,
        "rule_count": 1,
        "matched_text": "the phrase",
        "transcript": "he said the phrase out loud",
    }
    fields.update(overrides)
    return FlaggedEvent(**fields)


def reporter(*, post_error: Exception | None = None, **overrides) -> EventReporter:
    made = EventReporter(
        events_url=EVENTS_URL,
        service_token="service-secret",
        utc_now=lambda: AT,
        **overrides,
    )
    # Replaces the socket rather than the method under test, so `flush_once`
    # keeps its own requeue and counting behaviour.
    made.posted_error = post_error
    made._post = lambda payload: _raise_if(made.posted_error)
    return made


def _raise_if(error: Exception | None) -> None:
    if error is not None:
        raise error


class FakeResponse:
    def __init__(self, body):
        self._body = json.dumps(body).encode("utf-8")

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class FakeLicensingAPI:
    """The three internal endpoints behind one fake, dispatching on the path.

    One object because that is the deployment: one host, one service token,
    three paths. It also means a test cannot answer introspection while leaving
    the events feed pointed at the real network.
    """

    def __init__(self, *, events_error: Exception | None = None) -> None:
        self.tokens = {"cloud-token": {"server_id": "survival-1", "tenant_key": "vst_9f3"}}
        self.events_error = events_error
        self.event_reports: list[dict] = []
        self.event_authorizations: list[str] = []
        self.usage_reports: list[dict] = []

    def urlopen(self, request, timeout=None):
        body = json.loads(request.data.decode("utf-8"))
        if request.full_url.endswith("/introspect"):
            answer = self.tokens.get(body["token"])
            if answer is None:
                return FakeResponse({"active": False})
            return FakeResponse({"active": True, "plan": "cloud", **answer})
        if request.full_url.endswith("/events"):
            if self.events_error is not None:
                raise self.events_error
            self.event_reports.append(body)
            self.event_authorizations.append(request.get_header("Authorization"))
            return FakeResponse({"accepted": len(body["events"]), "ignored": 0})
        self.usage_reports.append(body)
        return FakeResponse({"ok": True})


def build_cloud_app(
    tmp_path: Path,
    *,
    introspection_url: str = INTROSPECTION_URL,
    events_url: str = "",
    transcript: str = "blocked phrase",
):
    return build_app(
        tmp_path,
        environment={
            "VOICESNIFFER_INTROSPECTION_URL": introspection_url,
            "VOICESNIFFER_INTROSPECTION_TOKEN": "service-secret",
            "VOICESNIFFER_EVENTS_URL": events_url,
        },
        transcript=transcript,
    )


def build_app(tmp_path: Path, *, environment: dict, transcript: str = "blocked phrase"):
    tokens_file = tmp_path / "tokens.json"
    tokens_file.write_text(
        json.dumps({"break-glass": {"token": "operator-token"}}),
        encoding="utf-8",
    )
    settings = ProcessorSettings.from_environment(
        {"VOICESNIFFER_TOKENS_FILE": str(tokens_file), **environment}
    )

    def transcribe(_samples: np.ndarray, _language: str) -> TranscriptionResult:
        return TranscriptionResult(text=transcript, language="en")

    return create_app(
        settings,
        transcribe=transcribe,
        rule_packs={"en": rule_pack(tmp_path)},
        inspect_packet=lambda _packet: OpusPacketInfo(samples_48k=960, channels=1),
        decode_audio=lambda _envelope: np.ones(DECODED_SAMPLES, dtype=np.float32),
    )


def rule_pack(tmp_path: Path) -> RulePack:
    rule_file = tmp_path / "en.yml"
    rule_file.write_text(
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
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return RulePack.load(rule_file)


def headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/vnd.voicesniffer.opus.v1",
        "Accept": "application/json",
        "X-Request-Id": str(uuid.uuid4()),
        "X-VoiceSniffer-Player-Id": str(uuid.uuid4()),
        "X-VoiceSniffer-Language": "en",
        "X-VoiceSniffer-Preroll-Samples": "0",
    }


async def moderate(
    app,
    *,
    token: str = "cloud-token",
    request_headers: dict[str, str] | None = None,
) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://processor") as client:
        return await client.post(
            "/v1/moderate",
            headers=headers(token) if request_headers is None else request_headers,
            content=b"\x00\x01a",
        )
