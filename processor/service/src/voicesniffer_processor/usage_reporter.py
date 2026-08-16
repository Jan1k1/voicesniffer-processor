"""The usage feed: cumulative counters, pushed to the licensing API every minute.

The processor knows what it did. Nothing else does. The live counters behind
``/v1/usage`` are per process and vanish on the next deploy, which makes them
useless for the one question the admin view has to answer: how much has this
customer used this month. So the numbers are pushed somewhere durable, and this
module is the push.

Four properties, each a choice rather than an accident:

**A self-hosted processor never runs this.** The reporter is built only when the
introspection settings are present, which is the same switch that makes a node a
cloud node. A container a customer runs on their own hardware must not open a
socket to us at all, and "must not" here means there is no object to start
rather than a flag somebody could misread. ``create_app`` does the check and
``tests/test_usage_reporter.py`` pins it.

**Counts only.** Identifiers, request and outcome tallies, a severity histogram
and a duration. No transcript, no audio, no matched text, no rule id. The whole
pitch of the service is that it does not keep what players said, and a usage
feed carrying one text field would quietly make that untrue in the one place
nobody would look. Everything that reaches the wire passes through
``ServerUsageSnapshot``, whose only string fields are the two identifiers, so
putting text on this feed takes a deliberate edit here rather than an accident
somewhere upstream.

**Cumulative since process start.** Not deltas. The receiver subtracts
consecutive reports and reads a decrease as a restart, which means a report that
never arrives costs nothing: the next one that gets through carries everything
the lost ones would have. That is also why a failed report is a logged line and
nothing more.

**It cannot touch a moderation request.** It is a separate task, the HTTP call
happens in a worker thread, and every exception below the loop is swallowed. The
licensing API being down must not delay moderation, must not raise into it, and
must not end the reporting task.

**When it fails it says why.** Not the exception class on its own, which is the
same word for a refused payload, a wrong service token and a box that is down.
The status, a cause name and the API's own sentence go in the line, because the
failure mode this feed actually had was silence: no rows anywhere and no log
line saying anything was wrong. The one that reads as a success gets the same
treatment: a report the API accepts and counts as belonging to nobody is
reported as ``ignored``, which is what a dashboard reading zero looks like from
this side.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from voicesniffer_processor.internal_api import describe_failure, post_json

LOGGER = logging.getLogger("voicesniffer.processor")

# Section 3 of docs/cloud-contract.md says 60 seconds. It is the report interval
# and therefore also the worst case loss when a process is killed, which is why
# it is not minutes.
DEFAULT_INTERVAL_SECONDS = 60.0
# Longer than the introspection timeout on purpose. Nobody is waiting on this
# call, so it can afford to survive a slow moment rather than give up and leave
# a gap in the ledger.
DEFAULT_TIMEOUT_SECONDS = 5.0
# The ceiling on the backoff below. Five minutes, because the counters are
# cumulative and a report skipped during an outage costs nothing, while the
# licensing API shares a box with the website and does not need this process
# knocking once a minute for an hour while it is already unwell. It also bounds
# how stale `lastSeenAt` gets after a recovery, which is what the console reads.
DEFAULT_MAX_BACKOFF_SECONDS = 300.0
INTROSPECT_SUFFIX = "/introspect"
USAGE_SUFFIX = "/usage"
TASK_NAME = "voicesniffer-usage-reporter"


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.UTC)


def _iso_utc(moment: datetime.datetime) -> str:
    return moment.isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class ServerUsageSnapshot:
    """One server's counters at one instant, cumulative since process start.

    Deliberately a type rather than a dict. Everything that reaches the wire
    passes through these fields, so "the usage feed carries no text" is a
    property somebody would have to change on purpose, not a convention.

    ``tenant_key`` is ``None`` for a credential that came out of ``tokens.json``,
    which on a cloud node is the operator's own break-glass token. That entry
    omits the field rather than sending null, because the receiving API keys
    usage on it and has nothing to key such an entry to.
    """

    server_id: str
    tenant_key: str | None
    requests: int
    outcomes: Mapping[str, int]
    severities: Mapping[int, int]
    audio_seconds: int


Collect = Callable[[], Sequence[ServerUsageSnapshot]]


def usage_url_from(introspection_url: str) -> str:
    """Derive the usage endpoint from the introspection one, or return ``""``.

    The two are siblings under ``/api/internal/`` by contract, and a deployment
    that already names one gains nothing by naming the other except one more
    environment variable to get wrong in a compose file that gets copied between
    hosts. So the common case is derived, and the deployed cloud compose file
    needs no edit at all to start reporting.

    Derivation happens only from a URL that really ends in ``/introspect``.
    Guessing from anything else risks POSTing a usage payload at the
    introspection endpoint, and that endpoint exists to answer questions about
    tokens: a body it did not expect arriving there should never be this
    module's doing. When the two are not siblings, ``VOICESNIFFER_USAGE_URL``
    says so explicitly instead.
    """
    trimmed = introspection_url.strip()
    if not trimmed.endswith(INTROSPECT_SUFFIX):
        return ""
    return trimmed[: -len(INTROSPECT_SUFFIX)] + USAGE_SUFFIX


@dataclass
class UsageReporter:
    """Posts a cumulative usage report on a timer, and never lets that matter."""

    usage_url: str
    service_token: str = field(repr=False)
    collect: Collect = field(repr=False)
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS
    now: Callable[[], float] = time.monotonic
    utc_now: Callable[[], datetime.datetime] = _utc_now
    _started_at: float = field(init=False, default=0.0, repr=False)
    _task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    # How many attempts in a row have failed. Drives the backoff and makes the
    # recovery line able to say how long the gap was.
    _failures: int = field(init=False, default=0, repr=False)
    _reported_ok: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self._started_at = self.now()

    def start(self) -> None:
        """Begin reporting. Registered as an application startup hook.

        Idempotent: a second call while a task is alive is ignored rather than
        leaving an orphan task reporting the same counters twice.
        """
        if self._task is not None and not self._task.done():
            return
        self._started_at = self.now()
        self._task = asyncio.create_task(self.run(), name=TASK_NAME)
        LOGGER.info(
            "usage_reporting_started url=%s interval_seconds=%s",
            self.usage_url,
            self.interval_seconds,
        )

    async def stop(self) -> None:
        """Cancel and await the task. Registered as an application shutdown hook.

        No final flush. A network call on the shutdown path can hang a container
        stop, and the counters are cumulative, so the most a clean shutdown
        loses is the part of one interval nobody had reported yet. That is a
        better trade than a stop that occasionally takes the whole timeout.
        """
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        LOGGER.info("usage_reporting_stopped")

    async def run(self) -> None:
        """Sleep, report, repeat, forever, without ever raising.

        Sleep first: at startup there is nothing to report, and an empty payload
        every time a container restarts is noise in the ledger.

        The interval sits between reports rather than on a fixed schedule, so a
        slow call pushes the next one out instead of stacking calls up. The
        receiver reads ``reported_at`` and cumulative counters, so drift costs
        nothing.
        """
        while True:
            await asyncio.sleep(self.delay_seconds())
            await self.report_once()

    def delay_seconds(self) -> float:
        """How long before the next attempt: the interval, then backoff.

        Doubling from the interval up to the ceiling. Nothing is lost by waiting
        here, because the counters are cumulative and the next report that gets
        through carries everything the skipped ones would have, so the only
        thing being traded is freshness against load on an API that is already
        failing. The exponent is capped before the multiplication so a process
        left running through a week-long outage cannot compute an overflow.
        """
        if self._failures == 0:
            return self.interval_seconds
        return min(
            self.interval_seconds * (2 ** min(self._failures, 8)),
            self.max_backoff_seconds,
        )

    async def report_once(self) -> bool:
        """One report. Returns whether it landed; raises only on cancellation.

        The except clause is as broad as it can be without swallowing
        cancellation, on purpose. Everything reachable from here -- building the
        payload, DNS, TCP, TLS, a 500 from the API, a counter in a shape nobody
        expected -- is a thing that must not end the task and must not surface
        anywhere near a moderation request. ``CancelledError`` derives from
        ``BaseException``, so shutdown still works.
        """
        try:
            payload = self.payload()
            answer = await asyncio.to_thread(self._post, payload) or {}
        except Exception as error:
            self._failures += 1
            status, reason, detail = describe_failure(error)
            LOGGER.warning(
                'usage_report_failed error=%s status=%s reason=%s detail="%s" '
                "url=%s attempt=%s next_attempt_seconds=%s",
                type(error).__name__,
                status,
                reason,
                detail,
                self.usage_url,
                self._failures,
                int(self.delay_seconds()),
            )
            return False
        if self._failures:
            LOGGER.info("usage_reporting_recovered after_failed_attempts=%s", self._failures)
            self._failures = 0
        self._describe(payload, answer)
        return True

    def _describe(self, payload: Mapping[str, object], answer: Mapping[str, object]) -> None:
        """Say what the API did with it, loudly the first time and when it is wrong.

        ``ignored`` is the failure that arrives as a success: the API took the
        report and could not resolve a tenant key, so nothing is stored and
        nothing complains. That is precisely the state where a dashboard reads
        zero while this process believes it is reporting, so it is a warning
        rather than a debug line.

        **Except for the ignores this side already expects.** An entry with no
        tenant key is a credential out of ``tokens.json``, which on a cloud node
        is the operator's own break-glass token, and the API has nothing to file
        it under by design. Warning about those would put a line in the log every
        sixty seconds for the life of the container the first time anybody used
        that token, and an alarm that is always on is an alarm nobody reads. So
        what is compared is the count the API ignored against the count this
        process knew it could not attribute, and only the difference is a
        problem.
        """
        servers = payload["servers"]
        counted = len(servers) if isinstance(servers, Sequence) else 0
        accepted = answer.get("accepted")
        ignored = answer.get("ignored")
        unattributable = (
            sum(1 for entry in servers if "tenant_key" not in entry)
            if isinstance(servers, Sequence)
            else 0
        )
        if isinstance(ignored, int) and ignored > unattributable:
            LOGGER.warning(
                "usage_report_ignored servers=%s ignored=%s expected=%s "
                "reason=unknown_tenant_key url=%s",
                counted,
                ignored,
                unattributable,
                self.usage_url,
            )
            return
        # Once at INFO, so a deployment can be confirmed from the log without
        # turning debug on, and then quiet: this runs every minute forever.
        if not self._reported_ok:
            self._reported_ok = True
            LOGGER.info(
                "usage_reporting_ok servers=%s accepted=%s url=%s",
                counted,
                accepted,
                self.usage_url,
            )
            return
        LOGGER.debug("usage_reported servers=%s accepted=%s", counted, accepted)

    def payload(self) -> dict[str, object]:
        """The exact body of contract section 3, built one field at a time.

        ``collect`` runs here, synchronously, on the event loop, which is also
        the only place the counters are ever written. No await sits between
        reading them and building the payload, so no request can land halfway
        through and produce an entry whose own totals disagree.
        """
        servers = [_server_entry(snapshot) for snapshot in self.collect()]
        return {
            "reported_at": _iso_utc(self.utc_now()),
            "uptime_seconds": max(0, int(self.now() - self._started_at)),
            "servers": servers,
        }

    def _post(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        """The socket, and the seam a test replaces. Raises ``InternalApiError``.

        The same service token the introspection call uses, for the same reason:
        this endpoint is loopback, and loopback is not a short list of callers on
        a box that also runs a web stack. See ``internal_api``.
        """
        return post_json(
            self.usage_url,
            service_token=self.service_token,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )


def _server_entry(snapshot: ServerUsageSnapshot) -> dict[str, object]:
    """One element of ``servers``. Identifiers and counts, nothing else.

    Keys are sorted so two consecutive reports differ only where the numbers
    differ, which is what makes a captured payload readable in a log.
    """
    entry: dict[str, object] = {"server_id": snapshot.server_id}
    if snapshot.tenant_key is not None:
        entry["tenant_key"] = snapshot.tenant_key
    entry["requests"] = int(snapshot.requests)
    # Only outcomes that actually happened. A wall of zeroes tells the receiver
    # nothing an absent key does not, and the outcome vocabulary is open: it
    # grows every time a new refusal code does.
    entry["outcomes"] = {name: int(count) for name, count in sorted(snapshot.outcomes.items())}
    # String keys, because JSON object keys are strings anyway and the receiver
    # should not have to work out whether "0" and 0 are the same bucket.
    entry["severities"] = {
        str(level): int(count) for level, count in sorted(snapshot.severities.items())
    }
    entry["audio_seconds"] = int(snapshot.audio_seconds)
    return entry
