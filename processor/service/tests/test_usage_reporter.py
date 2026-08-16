"""The usage feed, which is the only durable record that a tenant used anything.

Two things are being protected here, and they pull in opposite directions.

The first is that the numbers have to arrive. Live counters die with the
process, so without this feed a redeploy erases the answer to "how much has this
customer used this month", and the free-while-new decision cannot be revisited
from evidence later.

The second is that a usage feed is the easiest place in the whole service to
leak what players said. It runs unattended, nobody reads its payloads, and one
plausible-looking field -- a sample transcript, the phrase that matched, the id
of the rule that fired -- would turn it into a recording. So the tests below
pin the shape of the payload as tightly as they pin its contents, and a
self-hosted processor is held to sending nothing at all.
"""

import asyncio
import datetime
import json
import logging
import threading
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml

from voicesniffer_processor.app import create_app
from voicesniffer_processor.internal_api import InternalApiError
from voicesniffer_processor.models import TranscriptionResult
from voicesniffer_processor.opus import OpusPacketInfo
from voicesniffer_processor.rules import RulePack
from voicesniffer_processor.settings import ProcessorSettings
from voicesniffer_processor.usage_reporter import (
    TASK_NAME,
    ServerUsageSnapshot,
    UsageReporter,
    usage_url_from,
)

INTROSPECTION_URL = "http://licensing.internal:8080/api/internal/introspect"
USAGE_URL = "http://licensing.internal:8080/api/internal/usage"
# Two seconds at the rate the decoder emits, so audio_seconds has an answer a
# reader can check in their head.
DECODED_SAMPLES = 32_000


# ------------------------------------------------------- the payload itself


def test_a_report_carries_the_shape_the_contract_asks_for() -> None:
    """Section 3 of docs/cloud-contract.md, field for field.

    Asserted as one whole dict rather than key by key, because the failure this
    prevents is an extra field nobody meant to send, and a key-by-key test
    cannot see one.
    """
    made, clock = reporter_for([snapshot()])
    clock.advance(3_600)

    assert made.payload() == {
        "reported_at": "2026-07-30T15:00:00Z",
        "uptime_seconds": 3_600,
        "servers": [
            {
                "server_id": "survival-1",
                "tenant_key": "lic_9f3",
                "requests": 12_045,
                "outcomes": {"processor_busy": 55, "success": 11_990},
                "severities": {"0": 11_800, "1": 120, "2": 50, "3": 20},
                "audio_seconds": 40_150,
            }
        ],
    }


def test_counters_are_cumulative_rather_than_deltas() -> None:
    """The receiver subtracts consecutive reports and treats a decrease as a
    restart. A reporter that sent deltas would double count every one of them
    once, silently, and only on the second report."""
    counters = [snapshot(requests=10), snapshot(requests=25)]
    made, _ = reporter_for(counters)

    first = made.payload()["servers"][0]["requests"]
    counters[:] = [snapshot(requests=25)]
    second = made.payload()["servers"][0]["requests"]

    assert (first, second) == (10, 25)


def test_a_tenant_key_that_is_absent_is_omitted_rather_than_sent_as_null() -> None:
    # The break-glass token on a cloud node comes from tokens.json and belongs to
    # us, not to a tenant. The receiving API keys usage on tenant_key and has
    # nothing to attribute a null to, so it should never be asked to.
    made, _ = reporter_for([snapshot(tenant_key=None)])

    assert "tenant_key" not in made.payload()["servers"][0]


def test_an_empty_processor_still_reports_its_uptime() -> None:
    # A node with no traffic must still be visibly alive, otherwise "no usage"
    # and "the reporter died three weeks ago" look identical from the API side.
    made, clock = reporter_for([])
    clock.advance(90)

    assert made.payload() == {
        "reported_at": "2026-07-30T15:00:00Z",
        "uptime_seconds": 90,
        "servers": [],
    }


# ------------------------------------------------------- where it is posted


def test_the_usage_url_is_derived_from_the_introspection_url() -> None:
    # The two are siblings by contract, so the deployed compose file needs no
    # new environment variable to start reporting.
    assert usage_url_from(INTROSPECTION_URL) == USAGE_URL


@pytest.mark.parametrize(
    "introspection_url",
    [
        "",
        "http://api:8080/api/internal/",
        "http://api:8080/api/internal/introspect/",
        "http://api:8080/api/internal/tokens",
    ],
)
def test_a_url_that_is_not_the_introspection_endpoint_is_never_guessed_at(
    introspection_url: str,
) -> None:
    """Guessing wrong means POSTing a usage payload at whatever that URL is.

    If it is the introspection endpoint, that is a body arriving at the one
    place in the system whose job is answering questions about tokens.
    """
    assert usage_url_from(introspection_url) == ""


def test_an_explicit_usage_url_overrides_the_derived_one(tmp_path: Path) -> None:
    app = build_cloud_app(tmp_path, usage_url="http://elsewhere:9000/usage")

    assert app.state.usage_reporter.usage_url == "http://elsewhere:9000/usage"


def test_a_cloud_node_that_cannot_address_usage_still_moderates(tmp_path: Path) -> None:
    """Usage is evidence for a future rate cap, not something moderation needs.

    A node that cannot work out where to post it logs and carries on rather than
    crash-looping and taking every tenant's moderation down with it.
    """
    app = build_cloud_app(tmp_path, introspection_url="http://api:8080/api/internal/tokens")

    assert app.state.usage_reporter is None
    assert asyncio.run(moderate(app, token="operator-token")).status_code == 200


# ------------------------------------------------------- the self-hosted rule


def test_a_self_hosted_processor_never_reports(tmp_path: Path, monkeypatch) -> None:
    """The hard rule of the whole module.

    Not "reports nothing useful" and not "reports to nowhere": no reporter
    object, no task, no socket. A container a customer runs on their own
    hardware that opens an outbound connection once a minute is a different
    product from the one they were sold.
    """
    opened: list[str] = []
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout=None: opened.append(request.full_url),
    )
    app = build_self_hosted_app(tmp_path)

    async def scenario() -> tuple[list[str], int]:
        async with app.router.lifespan_context(app):
            response = await moderate(app, token="operator-token")
            return [task.get_name() for task in asyncio.all_tasks()], response.status_code

    task_names, status = asyncio.run(scenario())

    assert app.state.usage_reporter is None
    assert status == 200, "moderation still has to work without a reporter"
    assert TASK_NAME not in task_names
    assert opened == [], f"a self-hosted processor phoned home: {opened}"


def test_a_usage_url_on_its_own_does_not_make_a_processor_report(tmp_path: Path) -> None:
    """Half a cloud compose file, copied onto a customer's own box.

    The switch is the introspection settings, because those are what make a node
    multi-tenant in the first place. A usage URL without them is a self-hosted
    processor with a stray environment variable, and the rule is not "reports to
    wherever it was told" but "does not report".
    """
    app = build_app(tmp_path, environment={"VOICESNIFFER_USAGE_URL": USAGE_URL})

    assert app.state.usage_reporter is None


def test_an_introspection_url_without_its_token_does_not_report(tmp_path: Path) -> None:
    # Half-configured is not configured. This node cannot authenticate a tenant,
    # so it has no tenant usage to report and no credential to report it with.
    app = build_app(
        tmp_path,
        environment={"VOICESNIFFER_INTROSPECTION_URL": INTROSPECTION_URL},
    )

    assert app.state.usage_reporter is None


# ------------------------------------------------------- end to end, on a cloud node


def test_tenant_key_flows_from_introspection_into_the_report(tmp_path: Path, monkeypatch) -> None:
    """The attribution fix, end to end.

    server_id is chosen by the operator and defaults to the Minecraft server's
    own name, so it identifies nobody. Without the tenant key beside it the
    licensing API has a row it cannot file and drops it.
    """
    api = FakeLicensingAPI()
    monkeypatch.setattr(urllib.request, "urlopen", api.urlopen)
    app = build_cloud_app(tmp_path)

    async def scenario() -> bool:
        assert (await moderate(app, token="cloud-token")).status_code == 200
        return await app.state.usage_reporter.report_once()

    assert asyncio.run(scenario()) is True
    assert len(api.usage_reports) == 1
    entry = api.usage_reports[0]["servers"][0]
    assert entry["server_id"] == "survival-1"
    assert entry["tenant_key"] == "lic_9f3"
    assert api.usage_authorizations == ["Bearer service-secret"], (
        "the usage endpoint is loopback, and loopback is not a short list of callers"
    )


def test_two_customers_with_the_same_server_name_are_two_entries(
    tmp_path: Path, monkeypatch
) -> None:
    """The reason the tenant key exists at all.

    Both of these are called survival-1. Keying the in-memory counters on the
    name alone would add their traffic together and then report the total under
    whichever tenant happened to arrive first, which is worse than reporting
    nothing: it is a wrong number that looks right.
    """
    api = FakeLicensingAPI(
        tokens={
            "token-a": {"server_id": "survival-1", "tenant_key": "lic_a"},
            "token-b": {"server_id": "survival-1", "tenant_key": "lic_b"},
        }
    )
    monkeypatch.setattr(urllib.request, "urlopen", api.urlopen)
    app = build_cloud_app(tmp_path)

    async def scenario() -> dict:
        assert (await moderate(app, token="token-a")).status_code == 200
        assert (await moderate(app, token="token-b")).status_code == 200
        assert (await moderate(app, token="token-b")).status_code == 200
        return app.state.usage_reporter.payload()

    entries = asyncio.run(scenario())["servers"]

    assert len(entries) == 2, f"one entry means one customer's usage was lost: {entries}"
    by_tenant = {entry["tenant_key"]: entry["requests"] for entry in entries}
    assert by_tenant == {"lic_a": 1, "lic_b": 2}


def test_one_customer_cannot_spend_another_customers_rate_budget(
    tmp_path: Path, monkeypatch
) -> None:
    """The same collision as the counters, one line above them.

    Dormant today, because rate_limit_per_minute is null for every cloud licence
    by decision. It matters because the contract calls imposing a cap later "a
    data change, not a code change", and with a name-keyed window that data
    change would quietly throttle two strangers against each other.
    """
    def capped(tenant_key: str) -> dict:
        return {
            "server_id": "survival-1",
            "tenant_key": tenant_key,
            "rate_limit_per_minute": 1,
        }

    api = FakeLicensingAPI(tokens={"token-a": capped("lic_a"), "token-b": capped("lic_b")})
    monkeypatch.setattr(urllib.request, "urlopen", api.urlopen)
    app = build_cloud_app(tmp_path)

    async def scenario() -> list[int]:
        return [
            (await moderate(app, token="token-a")).status_code,
            (await moderate(app, token="token-b")).status_code,
            (await moderate(app, token="token-a")).status_code,
        ]

    first_a, first_b, second_a = asyncio.run(scenario())

    assert (first_a, first_b) == (200, 200), "one tenant used up the other's minute"
    assert second_a == 429, "the tenant's own limit still has to bite"


def test_a_break_glass_token_on_a_cloud_node_omits_the_tenant_key(
    tmp_path: Path, monkeypatch
) -> None:
    # tokens.json is still consulted first on a cloud node: it holds the
    # operator's own token so the box can be exercised while the licensing API
    # is down. That traffic belongs to no tenant.
    api = FakeLicensingAPI()
    monkeypatch.setattr(urllib.request, "urlopen", api.urlopen)
    app = build_cloud_app(tmp_path)

    async def scenario() -> dict:
        assert (await moderate(app, token="operator-token")).status_code == 200
        return app.state.usage_reporter.payload()

    entry = asyncio.run(scenario())["servers"][0]

    assert entry["server_id"] == "break-glass"
    assert "tenant_key" not in entry


def test_the_report_matches_what_the_processor_actually_did(
    tmp_path: Path, monkeypatch
) -> None:
    api = FakeLicensingAPI()
    monkeypatch.setattr(urllib.request, "urlopen", api.urlopen)
    app = build_cloud_app(tmp_path)

    async def scenario() -> dict:
        assert (await moderate(app, token="cloud-token")).status_code == 200
        assert (await moderate(app, token="cloud-token")).status_code == 200
        assert (await moderate(app, token="unknown-token")).status_code == 401
        return app.state.usage_reporter.payload()

    entry = asyncio.run(scenario())["servers"][0]

    assert entry["requests"] == 2, "an unauthenticated caller is nobody's usage"
    assert entry["outcomes"] == {"success": 2}
    # The rule pack matches the transcript, so both verdicts are severity 2.
    assert entry["severities"] == {"2": 2}
    assert entry["audio_seconds"] == 4


def test_a_retried_request_is_not_counted_as_more_audio(tmp_path: Path, monkeypatch) -> None:
    """The plugin retries a timed-out request with the same request id.

    The idempotency cache answers the retry from the stored verdict without
    decoding anything, so audio_seconds must not move. The request count does:
    the retry really was a request.
    """
    api = FakeLicensingAPI()
    monkeypatch.setattr(urllib.request, "urlopen", api.urlopen)
    app = build_cloud_app(tmp_path)
    retried = headers("cloud-token")

    async def scenario() -> dict:
        assert (await moderate(app, request_headers=retried)).status_code == 200
        assert (await moderate(app, request_headers=retried)).status_code == 200
        return app.state.usage_reporter.payload()

    entry = asyncio.run(scenario())["servers"][0]

    assert entry["requests"] == 2
    assert entry["audio_seconds"] == 2, "the same utterance was counted twice"


# ------------------------------------------------------- what must never be on the wire


def test_the_payload_carries_no_text_beyond_the_two_identifiers(
    tmp_path: Path, monkeypatch
) -> None:
    """The property the service is sold on.

    The processor holds the transcript and the matched phrase in memory while it
    builds this payload, so the only thing keeping them off the wire is that
    nothing here has a field to put them in. This walks every value in the
    payload and refuses to find any string that is not one of the identifiers or
    the timestamp, which is the check that survives somebody adding a field
    later for a good reason.
    """
    api = FakeLicensingAPI()
    monkeypatch.setattr(urllib.request, "urlopen", api.urlopen)
    app = build_cloud_app(tmp_path)

    async def scenario() -> tuple[dict, dict]:
        verdict = await moderate(app, token="cloud-token")
        assert await app.state.usage_reporter.report_once() is True
        return verdict.json(), api.usage_reports[0]

    verdict, posted = asyncio.run(scenario())

    # Not a vacuous test: the text existed, and the verdict carried it.
    assert verdict["transcript"] == "blocked phrase"
    assert verdict["matches"][0]["matched_text"] == "blocked phrase"

    assert set(strings_in(posted)) == {posted["reported_at"], "survival-1", "lic_9f3"}
    assert "blocked" not in json.dumps(posted)
    assert set(posted["servers"][0]) == {
        "server_id",
        "tenant_key",
        "requests",
        "outcomes",
        "severities",
        "audio_seconds",
    }


# ------------------------------------------------------- failure and shutdown


def test_a_failed_report_neither_stops_the_task_nor_touches_moderation(
    tmp_path: Path, monkeypatch
) -> None:
    """The licensing API being down is a logged line and nothing else.

    Three failures in a row, then a moderation request, then a check that the
    task is still alive. If a report ever raised into the request path, or ended
    its own loop, an outage would take moderation down with it or stop the feed
    permanently until someone noticed and redeployed.
    """
    api = FakeLicensingAPI(usage_error=urllib.error.URLError("licensing api is down"))
    monkeypatch.setattr(urllib.request, "urlopen", api.urlopen)
    app = build_cloud_app(tmp_path)
    reporter = app.state.usage_reporter
    reporter.interval_seconds = 0.01

    async def scenario() -> tuple[int, bool, int]:
        async with app.router.lifespan_context(app):
            # Set from a worker thread, so a threading primitive rather than an
            # asyncio one.
            await asyncio.to_thread(api.attempts_reached.wait, 5.0)
            response = await moderate(app, token="cloud-token")
            still_running = reporter._task is not None and not reporter._task.done()
            return response.status_code, still_running, len(api.usage_reports)

    status, still_running, attempts = asyncio.run(scenario())

    assert status == 200
    assert still_running, "one outage ended the usage feed for the life of the process"
    assert attempts >= 3


def test_the_task_stops_when_the_application_does(tmp_path: Path, monkeypatch) -> None:
    api = FakeLicensingAPI()
    monkeypatch.setattr(urllib.request, "urlopen", api.urlopen)
    app = build_cloud_app(tmp_path)
    reporter = app.state.usage_reporter

    async def scenario() -> tuple[asyncio.Task, list[str]]:
        async with app.router.lifespan_context(app):
            running = reporter._task
            assert running is not None and not running.done()
        return running, [task.get_name() for task in asyncio.all_tasks()]

    stopped, task_names = asyncio.run(scenario())

    assert stopped.done(), "shutdown returned while the reporter was still running"
    assert reporter._task is None
    assert TASK_NAME not in task_names


def test_starting_twice_does_not_leave_two_tasks_reporting(
    tmp_path: Path, monkeypatch
) -> None:
    # Two tasks on the same counters would report the same cumulative numbers
    # twice a minute, which the receiver reads as two of everything.
    api = FakeLicensingAPI()
    monkeypatch.setattr(urllib.request, "urlopen", api.urlopen)
    app = build_cloud_app(tmp_path)
    reporter = app.state.usage_reporter

    async def scenario() -> int:
        reporter.start()
        reporter.start()
        running = len([task for task in asyncio.all_tasks() if task.get_name() == TASK_NAME])
        await reporter.stop()
        return running

    assert asyncio.run(scenario()) == 1


def test_stopping_a_reporter_that_never_started_is_not_an_error() -> None:
    # The shutdown hook runs whether or not startup got far enough to run.
    made, _ = reporter_for([])

    assert asyncio.run(made.stop()) is None


# ------------------------------------------------------- saying what went wrong


def test_a_failed_report_names_the_status_the_cause_and_the_url(caplog) -> None:
    """`error=HTTPError` on its own is the same word for four different problems.

    A refused payload, a wrong service token, a missing route and an API having
    a bad afternoon all arrive as one exception class, and which of them it is
    decides whether the fix is a redeploy or an environment variable. This feed
    has already failed once by reporting nothing and explaining nothing.
    """
    made, _ = reporter_for([snapshot()])
    made._post = raising(
        InternalApiError(
            kind="HTTPError", status=400, detail="Invalid usage payload.", retryable=False
        )
    )

    with caplog.at_level(logging.DEBUG):
        assert asyncio.run(made.report_once()) is False

    assert "usage_report_failed" in caplog.text
    assert "status=400" in caplog.text
    assert "reason=payload_refused" in caplog.text
    assert 'detail="Invalid usage payload."' in caplog.text
    assert USAGE_URL in caplog.text


def test_a_report_the_api_files_under_nobody_is_a_warning(caplog) -> None:
    """The failure that arrives as a success, and the exact state that had the
    dashboard reading zero while the processor logged nothing wrong: the API
    accepted the report and could not resolve the tenant key it was keyed on."""
    made, _ = reporter_for([snapshot()])
    made._post = lambda payload: {"accepted": 0, "ignored": 1}

    with caplog.at_level(logging.DEBUG):
        assert asyncio.run(made.report_once()) is True

    assert "usage_report_ignored" in caplog.text
    assert "reason=unknown_tenant_key" in caplog.text


def test_the_break_glass_token_being_ignored_is_not_an_alarm(caplog) -> None:
    """An entry with no tenant key is the operator's own token out of tokens.json.

    The API has nothing to file it under by design, so warning about it would
    put a line in the log every sixty seconds for the life of the container the
    first time anybody used that token. What is compared is what the API ignored
    against what this side already knew it could not attribute.
    """
    made, _ = reporter_for([snapshot(tenant_key=None)])
    made._post = lambda payload: {"accepted": 0, "ignored": 1}

    with caplog.at_level(logging.DEBUG):
        assert asyncio.run(made.report_once()) is True

    assert "usage_report_ignored" not in caplog.text


def test_a_tenant_keyed_entry_being_ignored_is_still_an_alarm(caplog) -> None:
    # One of each: the break-glass entry is expected to be ignored, the paired
    # server is not, and one ignore too many is what the warning is counting.
    made, _ = reporter_for([snapshot(tenant_key=None), snapshot(server_id="survival-2")])
    made._post = lambda payload: {"accepted": 0, "ignored": 2}

    with caplog.at_level(logging.DEBUG):
        asyncio.run(made.report_once())

    assert "usage_report_ignored servers=2 ignored=2 expected=1" in caplog.text


def test_a_delivered_report_says_so_once_at_info(caplog) -> None:
    # Once, so a deployment can be confirmed from the log without turning debug
    # on, and then quiet: this runs every minute for the life of the process.
    made, _ = reporter_for([snapshot()])
    made._post = lambda payload: {"accepted": 1, "ignored": 0}

    with caplog.at_level(logging.INFO):
        asyncio.run(made.report_once())
        asyncio.run(made.report_once())

    assert caplog.text.count("usage_reporting_ok") == 1


def test_the_delay_grows_while_the_api_is_down_and_resets_when_it_returns() -> None:
    """Nothing is lost by waiting: the counters are cumulative, so the next
    report that lands carries everything the skipped ones would have. What is
    traded is freshness, against knocking once a minute for an hour on an API
    that shares a box with the website."""
    made, _ = reporter_for([snapshot()])
    made._post = raising(urllib.error.URLError("down"))
    first = made.delay_seconds()

    asyncio.run(made.report_once())
    after_one = made.delay_seconds()
    asyncio.run(made.report_once())
    after_two = made.delay_seconds()
    for _ in range(20):
        asyncio.run(made.report_once())
    at_ceiling = made.delay_seconds()

    made._post = lambda payload: {"accepted": 1, "ignored": 0}
    asyncio.run(made.report_once())

    assert (first, after_one, after_two) == (60.0, 120.0, 240.0)
    assert at_ceiling == 300.0
    assert made.delay_seconds() == 60.0


def test_the_feed_says_when_it_comes_back(caplog) -> None:
    # Otherwise the log has a beginning and no end, and nobody reading it later
    # can tell whether the outage lasted a minute or the whole afternoon.
    made, _ = reporter_for([snapshot()])
    made._post = raising(urllib.error.URLError("down"))
    asyncio.run(made.report_once())
    asyncio.run(made.report_once())
    made._post = lambda payload: {"accepted": 1, "ignored": 0}

    with caplog.at_level(logging.INFO):
        asyncio.run(made.report_once())

    assert "usage_reporting_recovered after_failed_attempts=2" in caplog.text


# ---------------------------------------------------------------- helpers


def raising(error: Exception):
    def post(payload):
        raise error

    return post


class FakeClock:
    def __init__(self) -> None:
        self.value = 1_000.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


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
    """Both internal endpoints behind one fake, dispatching on the path.

    One object because that is how it is deployed: one host, one service token,
    two paths. It also means a test cannot accidentally answer introspection
    while leaving usage pointed at the real network.
    """

    def __init__(self, *, tokens: dict | None = None, usage_error: Exception | None = None) -> None:
        self.tokens = {"cloud-token": {"server_id": "survival-1", "tenant_key": "lic_9f3"}}
        if tokens is not None:
            self.tokens = tokens
        self.usage_error = usage_error
        self.introspections: list[dict] = []
        self.usage_reports: list[dict] = []
        self.usage_authorizations: list[str] = []
        self.attempts_reached = threading.Event()

    def urlopen(self, request, timeout=None):
        body = json.loads(request.data.decode("utf-8"))
        if request.full_url.endswith("/introspect"):
            self.introspections.append(body)
            answer = self.tokens.get(body["token"])
            if answer is None:
                return FakeResponse({"active": False})
            return FakeResponse({"active": True, "plan": "cloud", **answer})
        self.usage_reports.append(body)
        self.usage_authorizations.append(request.get_header("Authorization"))
        if len(self.usage_reports) >= 3:
            self.attempts_reached.set()
        if self.usage_error is not None:
            raise self.usage_error
        return FakeResponse({"ok": True})


def snapshot(**overrides) -> ServerUsageSnapshot:
    fields = {
        "server_id": "survival-1",
        "tenant_key": "lic_9f3",
        "requests": 12_045,
        "outcomes": {"success": 11_990, "processor_busy": 55},
        "severities": {0: 11_800, 1: 120, 2: 50, 3: 20},
        "audio_seconds": 40_150,
    }
    fields.update(overrides)
    return ServerUsageSnapshot(**fields)


def reporter_for(snapshots) -> tuple[UsageReporter, FakeClock]:
    clock = FakeClock()
    made = UsageReporter(
        usage_url=USAGE_URL,
        service_token="service-secret",
        collect=lambda: snapshots,
        now=clock,
        utc_now=lambda: datetime.datetime(2026, 7, 30, 15, 0, tzinfo=datetime.UTC),
    )
    return made, clock


def strings_in(value) -> list[str]:
    """Every string anywhere in the payload, keys excluded."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [found for item in value.values() for found in strings_in(item)]
    if isinstance(value, list):
        return [found for item in value for found in strings_in(item)]
    return []


def build_self_hosted_app(tmp_path: Path):
    return build_app(tmp_path, environment={})


def build_cloud_app(
    tmp_path: Path,
    *,
    introspection_url: str = INTROSPECTION_URL,
    usage_url: str = "",
):
    return build_app(
        tmp_path,
        environment={
            "VOICESNIFFER_INTROSPECTION_URL": introspection_url,
            "VOICESNIFFER_INTROSPECTION_TOKEN": "service-secret",
            "VOICESNIFFER_USAGE_URL": usage_url,
        },
    )


def build_app(tmp_path: Path, *, environment: dict):
    tokens_file = tmp_path / "tokens.json"
    tokens_file.write_text(
        json.dumps({"break-glass": {"token": "operator-token"}}),
        encoding="utf-8",
    )
    settings = ProcessorSettings.from_environment(
        {"VOICESNIFFER_TOKENS_FILE": str(tokens_file), **environment}
    )

    def transcribe(_samples: np.ndarray, _language: str) -> TranscriptionResult:
        return TranscriptionResult(text="blocked phrase", language="en")

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
