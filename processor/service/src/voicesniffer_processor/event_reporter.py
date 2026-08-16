"""The moderation history feed: flagged utterances, pushed to the licensing API.

The usage feed next door answers "how much did this customer use". This one
answers the question an owner actually asks after a mute lands in his ticket
queue: who said what, when, and which rule caught it. That means it carries the
transcript, which nothing else in this system stores, and every decision here
follows from that one fact.

**Only utterances that fired a rule.** The gate is a non-empty match list, not a
severity: a rule that matched but needs context still fired, and the owner
reviewing his own server is the context. Speech that matched nothing produces no
event of any kind, so the ordinary case, somebody saying something unremarkable,
leaves no record anywhere. That is not a filter applied on the way out, it is the
only condition under which ``record`` is ever called.

**A self-hosted processor never runs this**, for the same structural reason it
never runs the usage reporter, and a stronger one. The reporter is built only
when the introspection settings are present, which is the switch that makes a
node a cloud node, so on a customer's own hardware there is no object to start
rather than a flag somebody could misread. What a self-hosted customer's players
say never leaves their box, and this module must not be the thing that changes
that. ``create_app`` does the check and ``tests/test_event_reporter.py`` pins it.

**Never audio.** The processor holds decoded PCM for the length of one request
and this module is never given it. What crosses the wire is a subset of the text
the verdict already returned to the caller's own server.

**Nothing here is logged.** Not the transcript, not the matched text, not at
debug. Failures are counted and named by exception type, the way the usage
reporter names them, because a log line is a second copy with no retention
deadline on it and the whole point of the receiving table is that its rows have
one.

**Bounded, and lossy on purpose.** The queue has a ceiling and drops the oldest
entry when it is full. A moderation request may not wait on this and may not fail
because of it: an owner losing the tail of a flag storm from a licensing API
outage is a worse record, and a server whose voice chat stalls because a
history feed is behind is a broken product.

**Delivered at most once, and idempotent when it is not.** Each event carries the
request id the verdict already has, and the receiver's unique index is on
``(tenant_key, request_id)``, so a batch redelivered after a timeout that
actually landed writes nothing the second time. The capture point upstream sits
inside the idempotency cache's own body, so a plugin retrying an utterance
produces one event rather than two before this module ever sees it.

**A refused batch is discarded rather than retried forever.** The two failures
are not the same thing. A licensing API that is down or answering 5xx will
answer properly later, so that batch waits and the delay grows. A 400 means
these exact bytes are refused, and putting them back at the front of the queue
means the head of the queue never moves again: every event recorded afterwards
is dropped by the bound while one poisoned batch is retried every fifteen
seconds for the life of the process. So a refusal ends that batch, and says so
at ``error`` level with the count it cost.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime
import logging
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from voicesniffer_processor.internal_api import describe_failure, is_retryable, post_json

LOGGER = logging.getLogger("voicesniffer.processor")

# Shorter than the usage interval. Usage is a ledger nobody reads live; this is a
# moderation log an owner refreshes while somebody is arguing with him in
# Discord, so a minute of lag is worse here than there.
DEFAULT_INTERVAL_SECONDS = 15.0
DEFAULT_TIMEOUT_SECONDS = 5.0
# The ceiling on the backoff after consecutive failures. Two minutes, an eighth
# of the usage feed's, because what is waiting here is a moderation log somebody
# refreshes while an argument is going on, and because the queue is bounded: the
# longer the gap between attempts, the more of a flag storm falls off the end.
DEFAULT_MAX_BACKOFF_SECONDS = 120.0
# One post carries at most this many. The receiving schema's own ceiling is 500,
# and staying under it means a burst is split into several posts rather than
# rejected whole.
DEFAULT_MAX_BATCH = 200
# The ceiling on what is held while the licensing API is unreachable. Sized on
# what it costs rather than on what is likely: 2,000 events at the truncation
# limits below is a few megabytes against a container with gigabytes, and it is
# roughly an hour of a very loud server flagging constantly.
DEFAULT_MAX_QUEUE = 2_000
# Truncation, applied here rather than trusted from upstream. An utterance is
# capped at eight seconds of audio so a transcript is naturally short, and these
# bounds exist so that a decode that goes wrong cannot put an unbounded string on
# the wire. Both sit well under the receiver's own limits.
MAX_TRANSCRIPT_CHARS = 1_000
MAX_MATCHED_TEXT_CHARS = 200
INTROSPECT_SUFFIX = "/introspect"
EVENTS_SUFFIX = "/events"
TASK_NAME = "voicesniffer-event-reporter"


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.UTC)


def _iso_utc(moment: datetime.datetime) -> str:
    return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")


class RuleMatch(Protocol):
    """The four fields of a verdict match this module reads.

    Structural rather than an import of ``models.VerdictMatch``, so nothing here
    depends on the pydantic model and a test can pass a plain object.
    """

    rule_id: str
    category: str
    severity: int
    matched_text: str


@dataclass(frozen=True, slots=True)
class FlaggedEvent:
    """One utterance that fired at least one rule.

    A type rather than a dict, for the reason ``ServerUsageSnapshot`` is: this is
    the complete list of what may leave the box about a flagged utterance, so
    adding to it is a deliberate edit in one file rather than something that
    happens by accident at a call site.

    ``rule_id``, ``category``, ``severity`` and ``matched_text`` describe the one
    match that set the verdict's severity. ``rule_count`` says how many fired in
    total. The transcript is the utterance's, so it is carried once rather than
    once per match.
    """

    tenant_key: str
    server_id: str
    request_id: str
    player_uuid: str
    occurred_at: datetime.datetime
    rule_id: str
    category: str
    severity: int
    rule_count: int
    matched_text: str
    transcript: str


def events_url_from(introspection_url: str) -> str:
    """Derive the events endpoint from the introspection one, or return ``""``.

    Same construction and same caution as ``usage_reporter.usage_url_from``. The
    three internal routes are siblings under ``/api/internal/`` by contract, so
    a deployment that names one gains nothing by naming the others except more
    environment variables to get wrong in a compose file that gets copied
    between hosts.

    Derivation happens only from a URL that really ends in ``/introspect``.
    Guessing from anything else could aim a payload carrying a transcript at
    whatever endpoint happened to be there, and of everything this process
    sends, that is the one that must land where it was meant to.
    """
    trimmed = introspection_url.strip()
    if not trimmed.endswith(INTROSPECT_SUFFIX):
        return ""
    return trimmed[: -len(INTROSPECT_SUFFIX)] + EVENTS_SUFFIX


@dataclass
class EventReporter:
    """Buffers flagged utterances and posts them in batches, never blocking."""

    events_url: str
    service_token: str = field(repr=False)
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_batch: int = DEFAULT_MAX_BATCH
    max_queue: int = DEFAULT_MAX_QUEUE
    max_backoff_seconds: float = DEFAULT_MAX_BACKOFF_SECONDS
    utc_now: Callable[[], datetime.datetime] = _utc_now
    # Not a dataclass field anybody passes. The deque is bounded, which is what
    # makes "drop the oldest" the structure rather than a rule somebody has to
    # remember to apply.
    _queue: deque[FlaggedEvent] = field(init=False, repr=False, default_factory=deque)
    _dropped: int = field(init=False, default=0, repr=False)
    _task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    # Consecutive failed deliveries. Drives the backoff, and lets the recovery
    # line say how long the moderation log was behind.
    _failures: int = field(init=False, default=0, repr=False)

    def __post_init__(self) -> None:
        self._queue = deque(maxlen=self.max_queue)

    def record(self, event: FlaggedEvent) -> None:
        """Queue one event. Synchronous, allocation-only, and never raises.

        Called from the moderation path, so it does exactly one thing: append.
        No I/O, no await, no validation that could throw. A bounded deque discards
        its oldest entry silently when full, which is counted here so the drop is
        visible in a log line that carries a number and nothing else.
        """
        if len(self._queue) == self._queue.maxlen:
            self._dropped += 1
        self._queue.append(event)

    def start(self) -> None:
        """Begin delivering. Registered as an application startup hook."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self.run(), name=TASK_NAME)
        LOGGER.info(
            "event_reporting_started url=%s interval_seconds=%s",
            self.events_url,
            self.interval_seconds,
        )

    async def stop(self) -> None:
        """Cancel and await the task. Registered as an application shutdown hook.

        One last flush, unlike the usage reporter, and bounded by the same
        timeout as any other post. The counters that reporter carries are
        cumulative, so a report it never sends is folded into the next one and
        losing it costs nothing. An event is a fact that happened once: if this
        process is stopping, the queue is the only copy, and a single post on the
        way out is the difference between a moderation record with a gap in it at
        every deploy and one without.
        """
        task = self._task
        self._task = None
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        with contextlib.suppress(Exception):
            await self.flush_once()
        LOGGER.info("event_reporting_stopped queued=%s dropped=%s", len(self._queue), self._dropped)

    async def run(self) -> None:
        """Sleep, deliver, repeat, forever, without ever raising."""
        while True:
            await asyncio.sleep(self.delay_seconds())
            await self.flush_once()

    def delay_seconds(self) -> float:
        """How long before the next attempt: the interval, then backoff.

        Unlike the usage feed, waiting here costs something: the queue is
        bounded, so a long gap between attempts means more of a flag storm falls
        off the end. That is why the ceiling is two minutes rather than five, and
        why it is a ceiling at all rather than an ever-doubling delay.
        """
        if self._failures == 0:
            return self.interval_seconds
        return min(
            self.interval_seconds * (2 ** min(self._failures, 8)),
            self.max_backoff_seconds,
        )

    async def flush_once(self) -> bool:
        """Deliver up to ``max_batch`` events. Returns whether they landed.

        The batch is taken off the queue before the post. What happens to it
        afterwards depends on why the post failed, and the difference is the
        whole reason this is not one ``except``:

        - **The API could not answer** -- down, slow, 5xx, rate limited. The
          batch goes back on the front of the queue, so nothing is lost to a
          licensing API that is briefly down and the order a moderator reads
          them in survives the retry. Putting them back can overflow the bound,
          which drops the oldest, which is the same trade ``record`` makes and
          the right one: the newest flags are the ones somebody is asking about.
        - **The API refused these bytes** -- a 400, a 401, a route that is not
          there. Retrying is an infinite loop that also blocks the queue behind
          it forever, so the batch ends here and the line says so at ``error``.

        The except clause is as broad as it can be without swallowing
        cancellation, exactly as ``usage_reporter.report_once`` explains: nothing
        reachable from here may end the task or surface near a moderation
        request. Nothing about the events themselves reaches the log, which is
        why the failure line carries a status and a cause name derived from it
        and never the API's own answer, whose bytes came back off a wire this
        process had just put transcripts on.
        """
        self._report_drops()
        if not self._queue:
            return True
        batch = [self._queue.popleft() for _ in range(min(self.max_batch, len(self._queue)))]
        try:
            payload = self.payload(batch)
            answer = await asyncio.to_thread(self._post, payload) or {}
        except Exception as error:
            self._failed(error, batch)
            return False
        if self._failures:
            LOGGER.info("event_reporting_recovered after_failed_attempts=%s", self._failures)
            self._failures = 0
        ignored = answer.get("ignored")
        if isinstance(ignored, int) and ignored > 0:
            # Accepted by the API and stored for nobody. On this feed that is a
            # flagged utterance with no owner and no reader, which is worse than
            # a failed post: it looks like success from here.
            LOGGER.warning(
                "event_report_ignored count=%s of=%s reason=unknown_tenant_key",
                ignored,
                len(batch),
            )
        LOGGER.debug("events_reported count=%s", len(batch))
        return True

    def _failed(self, error: Exception, batch: Sequence[FlaggedEvent]) -> None:
        """Requeue or discard, and name the cause either way.

        Only a retryable failure counts towards the backoff. A refusal is not an
        outage: the API answered, promptly, that these bytes are wrong, and the
        next batch is different bytes. Backing off after one would delay
        perfectly deliverable events on the strength of a batch that is already
        gone.
        """
        status, reason, _detail = describe_failure(error)
        if is_retryable(error):
            self._failures += 1
            self._requeue(batch)
            LOGGER.warning(
                "event_report_failed error=%s status=%s reason=%s pending=%s "
                "attempt=%s next_attempt_seconds=%s",
                type(error).__name__,
                status,
                reason,
                len(self._queue),
                self._failures,
                int(self.delay_seconds()),
            )
            return
        LOGGER.error(
            "event_report_rejected error=%s status=%s reason=%s discarded=%s pending=%s",
            type(error).__name__,
            status,
            reason,
            len(batch),
            len(self._queue),
        )

    def _report_drops(self) -> None:
        """Say what the bound cost, on every attempt rather than only after a good one.

        This used to be logged after a successful delivery, which is the one
        moment it cannot happen: the queue only overflows while the API is
        unreachable, and back then that path returned before reaching the line.
        A flag storm during an outage therefore lost events and said nothing,
        which is the failure this whole module is written against.
        """
        if not self._dropped:
            return
        LOGGER.warning("event_reports_dropped count=%s reason=queue_full", self._dropped)
        self._dropped = 0

    def payload(self, batch: Sequence[FlaggedEvent]) -> dict[str, object]:
        """The exact body of contract section 4, built one field at a time."""
        return {
            "reported_at": _iso_utc(self.utc_now()),
            "events": [_event_entry(event) for event in batch],
        }

    def _requeue(self, batch: Sequence[FlaggedEvent]) -> None:
        for event in reversed(batch):
            self._queue.appendleft(event)

    def _post(self, payload: Mapping[str, object]) -> Mapping[str, object]:
        """The socket, and the seam a test replaces. Raises ``InternalApiError``.

        The same service token the other two internal calls use. This endpoint is
        loopback, and loopback is not a short list of callers on a box that also
        runs a web stack. See ``internal_api``.
        """
        return post_json(
            self.events_url,
            service_token=self.service_token,
            payload=payload,
            timeout_seconds=self.timeout_seconds,
        )


def _event_entry(event: FlaggedEvent) -> dict[str, object]:
    """One element of ``events``. The DPA's list, and nothing outside it."""
    return {
        "tenant_key": event.tenant_key,
        "server_id": event.server_id,
        "request_id": event.request_id,
        "player_uuid": event.player_uuid,
        "occurred_at": _iso_utc(event.occurred_at),
        "rule_id": event.rule_id,
        "category": event.category,
        "severity": int(event.severity),
        "rule_count": int(event.rule_count),
        "matched_text": event.matched_text[:MAX_MATCHED_TEXT_CHARS],
        "transcript": event.transcript[:MAX_TRANSCRIPT_CHARS],
    }


def flagged_event_from(
    *,
    tenant_key: str,
    server_id: str,
    request_id: str,
    player_uuid: str,
    occurred_at: datetime.datetime,
    transcript: str,
    matches: Sequence[RuleMatch],
) -> FlaggedEvent | None:
    """Build the event for a verdict, or ``None`` when the verdict flagged nothing.

    The ``None`` is the gate, and it lives here rather than at the call site so
    that "nothing is stored for speech that fired no rule" is a property of the
    only function that can build one of these.

    The match chosen is the highest severity, ties going to the first, which is
    the one an operator would name if asked why this utterance is in the list.
    ``context_required`` matches are eligible: they fired, and a human reviewing
    his own server is exactly the context the flag was waiting for. That is
    deliberately looser than the plugin's enforcement rule, which ignores them
    when deciding whether to mute, because being told about something and acting
    on it are different questions.
    """
    if not matches:
        return None
    # `max` keeps the first of equal keys, which is what makes "ties go to the
    # first" a property rather than a coincidence of iteration order.
    top = max(matches, key=lambda match: match.severity)
    return FlaggedEvent(
        tenant_key=tenant_key,
        server_id=server_id,
        request_id=request_id,
        player_uuid=player_uuid,
        occurred_at=occurred_at,
        rule_id=top.rule_id,
        category=top.category,
        severity=top.severity,
        rule_count=len(matches),
        matched_text=top.matched_text,
        transcript=transcript,
    )
