from __future__ import annotations

import asyncio
import datetime
import hashlib
import logging
import math
import re
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from numpy.typing import NDArray

from voicesniffer_processor.cloud_auth import (
    CloudCredentialResolver,
    IntrospectionUnavailable,
)
from voicesniffer_processor.event_reporter import (
    EventReporter,
    events_url_from,
    flagged_event_from,
)
from voicesniffer_processor.idempotency import (
    IdempotencyCache,
    IdempotencyCapacityError,
    RequestIdReuseError,
)
from voicesniffer_processor.models import (
    ErrorResponse,
    ModerationVerdict,
    TranscriptionResult,
    UnsupportedTranscriptionLanguage,
    VerdictMatch,
)
from voicesniffer_processor.opus import (
    DECODED_SAMPLE_RATE,
    OpusDecodeError,
    OpusPacketInfo,
    decode_opus,
    inspect_opus_packet,
)
from voicesniffer_processor.protocol import (
    OpusEnvelope,
    ProtocolError,
    ProtocolLimits,
    parse_opus_envelope,
)
from voicesniffer_processor.rules import (
    CATEGORIES,
    LANGUAGE_PATTERN,
    RulePack,
    normalize_transcript,
)
from voicesniffer_processor.settings import ProcessorSettings, ServerCredential
from voicesniffer_processor.toxicity import ToxicityClassifier
from voicesniffer_processor.usage_reporter import (
    ServerUsageSnapshot,
    UsageReporter,
    usage_url_from,
)

CONTENT_TYPE = "application/vnd.voicesniffer.opus.v1"
# `1` for an early window, `0` or absent for a finished utterance.
#
# The plugin sends a word twice on purpose. An early window closed while the
# player is still talking buys latency, and the finished utterance that follows
# carries the whole sentence the phrase rules need. Both requests reach this
# service with their own request id and their own body, so nothing here could
# collapse them: two ids carrying one slur used to become two rows in the
# moderation history and two ticks of the flagged counter, and the panel then
# disagreed with the server console by roughly a factor of two.
#
# So the plugin says which one this is, and a window is counted as work rather
# than as an incident. Absent means `0`, which is what every plugin built before
# this header sends, and those keep behaving exactly as they did.
PARTIAL_HEADER = "X-VoiceSniffer-Partial"
PARTIAL_VALUES = {"0": False, "1": True}
DECODE_TIMING_HEADER = "X-VoiceSniffer-Decode-Ms"
TRANSCRIBE_TIMING_HEADER = "X-VoiceSniffer-Transcribe-Ms"
RULES_TIMING_HEADER = "X-VoiceSniffer-Rules-Ms"
AUTHORIZATION_PATTERN = re.compile(r"Bearer ([^\s]+)\Z", re.IGNORECASE)
PAYLOAD_ERROR_CODES = frozenset(
    {"body_too_large", "duration_too_long", "frame_too_large", "too_many_frames"}
)
MAX_LOGGED_RULE_TIMEOUTS = 20
# How many admissions `_TenantAdmission` holds back from any tenant that already
# has work in flight, so a newcomer can always get its first request in. Named
# rather than written as `- 1` in two places, because the admission pool is
# sized around it: see ``docs/processor-admission-sizing.md``.
ADMISSION_RESERVED_SLOTS = 1
# The model advertises 25, so this is a bound on nonsense rather than a policy:
# a header naming every language costs the same as `auto` and a server that
# wants that should say `auto` and mean it. Note this is the *advertised* count
# and not the number of packs installed, which is 23 -- `fi` and `mt` have no
# rules, which is why a request that resolves to no pack at all is refused
# rather than answered with an empty verdict.
MAX_REQUEST_LANGUAGES = 25
# The rate limiter's window, and how often the idle windows are swept out of the
# table that holds them. Sweeping on the same period as the window means an entry
# survives at most one sweep past the point it stopped meaning anything.
RATE_WINDOW_SECONDS = 60.0
RATE_WINDOW_SWEEP_SECONDS = 60.0
InspectPacket = Callable[[bytes], OpusPacketInfo]
DecodeAudio = Callable[[OpusEnvelope], NDArray[np.float32]]
Transcribe = Callable[[NDArray[np.float32], str], TranscriptionResult]
LOGGER = logging.getLogger("voicesniffer.processor")


@dataclass(frozen=True, slots=True)
class RequestTrace:
    request_id: str
    outcome: str
    decode_ms: int | None
    stt_ms: int | None
    rules_ms: int | None
    classifier_ms: int | None
    serialize_ms: int | None
    total_ms: int


@dataclass(frozen=True, slots=True)
class RequestStageTrace:
    request_id: str
    stage: str
    state: str
    duration_ms: int | None = None


TraceSink = Callable[[RequestTrace], None]
StageSink = Callable[[RequestStageTrace], None]


class ProcessorBusyError(RuntimeError):
    pass


@dataclass(slots=True)
class _TenantAdmission:
    """A tenant's share of concurrent admissions, and one slot kept for strangers.

    ``processing_slots`` and the idempotency cache's ``max_active_entries`` are
    both global, so one tenant sending ``max_active_entries`` requests at once
    took every admission and the next tenant to arrive was refused outright,
    without ever reaching a worker. On a free tier with no rate cap that is one
    customer switching cloud off for all the others, and it takes no unusual
    traffic to do it.

    Two rules, and the second one is the one that took a test failure to find.

    **A share that narrows only under contention.** ``capacity / tenants
    currently trying``, so with a single tenant it is the whole capacity and
    this class does nothing. The division rounds up: shares that sum to slightly
    over capacity keep the box busy when a competitor is present but idle, and
    the reservation below is still the hard stop.

    **One admission slot no busy tenant may take.** The share alone does not
    work, which is worth stating because it looks like it should. Nothing here
    preempts, so a tenant that saturated the box while it was alone keeps what
    it took; the newcomer is refused; being refused it never enters
    ``in_flight``; never entering it never counts as competing; and the
    incumbent's share therefore never narrows. The system does not converge, it
    deadlocks in the incumbent's favour. Holding one slot back for a tenant with
    nothing in flight breaks that: the newcomer always gets its first request
    in, that request makes it visibly competing, and the incumbent then drains
    to its share as its own work completes.

    What that reservation costs with one tenant, which is the situation today:
    that tenant runs ``capacity - ADMISSION_RESERVED_SLOTS`` concurrent
    admissions instead of ``capacity``. ``create_app`` sizes the pool with that
    subtraction already applied, so a lone tenant gets 32 at the production
    sizing, which is the plugin's default ``max-in-flight``. It costs nothing
    observable, because admissions are not the scarce resource.
    ``processing_slots`` is, and it is 4, so an admission well below the
    reservation is already queued behind three others.

    The claim above used to read "7 instead of 8", and that was the whole
    problem rather than a rounding detail. At a capacity of 8 the reservation
    was not comfortably clear of the real limit, it *was* the real limit:
    measured under the load the plugin now sends, a single tenant was refused at
    seven in flight while three quarters of the CPU sat idle. See
    ``docs/processor-admission-sizing.md``.

    Keyed on the tenant, not on ``(tenant, server_id)`` like the usage counters
    and the idempotency scope. Those answer "whose traffic is this" and want the
    server. This answers "whose turn is it", and what is being shared out is a
    box shared between paying customers, so a customer running five servers is
    still one customer with one share.

    A share is held for exactly as long as the idempotency entry it protects,
    which is why the release is wired to the cache task and not to the request
    handler: a client that disconnects mid-request leaves the work running, and
    freeing its share at that moment would let one tenant hold any number of
    admissions by hanging up over and over.
    """

    capacity: int
    in_flight: dict[str, int] = field(default_factory=dict)

    def competing(self, tenant: str) -> int:
        return len(self.in_flight) + (0 if tenant in self.in_flight else 1)

    def share(self, tenant: str) -> int:
        return math.ceil(self.capacity / self.competing(tenant))

    def held_total(self) -> int:
        # At most `capacity` keys can be present, since every key in here holds
        # at least one admission, so this sums over eight entries at the worst.
        return sum(self.in_flight.values())

    def acquire(self, tenant: str) -> str | None:
        """``None`` when admitted, otherwise the outcome to record.

        The two refusals are not the same event and must not be counted as one.
        Hitting the reservation means the box is full, which is the answer a
        lone tenant gets and is what ``processor_busy`` has always meant. Hitting
        the share means somebody else is waiting, which is a different problem
        with a different fix.
        """
        held = self.in_flight.get(tenant, 0)
        # Contention first. Both rules refuse the same requests whichever way
        # round they are asked, so the order decides only which reason gets
        # recorded, and a tenant that is over its share while somebody else is
        # waiting is a contention story even when the box happens to be full
        # as well. Reported the other way round it would read as "buy a bigger
        # machine", which is the wrong answer to that state.
        #
        # The `competing > 1` guard is said out loud rather than left to fall
        # out of the arithmetic: with nobody competing the share is the whole
        # capacity, and this signal must never fire at a customer that has no
        # competition.
        if self.competing(tenant) > 1 and held >= self.share(tenant):
            return "processor_busy_tenant_share"
        if held and self.held_total() >= self.capacity - ADMISSION_RESERVED_SLOTS:
            return "processor_busy"
        self.in_flight[tenant] = held + 1
        return None

    def release(self, tenant: str) -> None:
        held = self.in_flight.get(tenant, 0) - 1
        if held > 0:
            self.in_flight[tenant] = held
        else:
            # Removed, not left at zero. `competing` counts the keys, so a tenant
            # that has gone quiet must stop shrinking everybody else's share.
            self.in_flight.pop(tenant, None)


@dataclass(frozen=True, slots=True)
class _ProcessedVerdict:
    verdict: ModerationVerdict
    decode_ms: int
    transcribe_ms: int
    rules_ms: int
    classifier_ms: int
    # Decoded, post-preroll-trim, at DECODED_SAMPLE_RATE. Carried out of the
    # decode stage rather than estimated from the request body, because a body
    # size is a compression ratio away from the answer and this is a number a
    # customer could one day be billed on.
    audio_samples: int


@dataclass(slots=True)
class _RateLimitWindows:
    """Per-(tenant, server) sliding windows for ``rate_limit_per_minute``.

    Keyed by tenant and server for the same reason the usage counters are. It
    has no effect today, because rate_limit_per_minute is null for every cloud
    licence by decision, and that is exactly why it is worth fixing now: the
    contract says imposing a cap later is "a data change, not a code change",
    and with a name-only key that change would silently make two customers who
    both picked ``survival-1`` share one bucket.

    A window only describes the last sixty seconds, and nothing used to remove
    the entry when a server went quiet: the deque was pruned on the way in and
    the key itself stayed for the life of the process. One entry per (tenant,
    server) that has ever sent a request under a rate cap, kept forever, in a
    container with a hard 3 GiB ceiling. Small per entry and unbounded in
    count, which is the shape that eventually wins.

    Swept on a timer rather than on every call. Checking every key per request
    would put a pass over every tenant on the moderation path, which is the
    wrong trade at a few hundred milliseconds of budget; once a minute costs
    one pass a minute and holds the table to the servers that were actually
    active inside it. Nothing is lost by dropping a stale entry: every
    timestamp in it is already older than the window and would have been
    discarded by the first line of ``wait`` anyway.
    """

    clock: Callable[[], float]
    windows: dict[tuple[str | None, str], deque[float]] = field(default_factory=dict)
    last_sweep: float = field(init=False)

    def __post_init__(self) -> None:
        self.last_sweep = self.clock()

    def sweep(self, now: float) -> None:
        if now - self.last_sweep < RATE_WINDOW_SWEEP_SECONDS:
            return
        self.last_sweep = now
        stale = [
            key
            for key, window in self.windows.items()
            if not window or now - window[-1] >= RATE_WINDOW_SECONDS
        ]
        for key in stale:
            del self.windows[key]

    def wait(self, credential: ServerCredential) -> float | None:
        """None = admitted; otherwise seconds until a slot frees up."""
        limit = credential.rate_limit_per_minute
        if limit is None:
            return None
        now = self.clock()
        self.sweep(now)
        window = self.windows.setdefault(
            (credential.tenant_key, credential.server_id),
            deque(),
        )
        while window and now - window[0] >= RATE_WINDOW_SECONDS:
            window.popleft()
        if len(window) >= limit:
            return max(1.0, RATE_WINDOW_SECONDS - (now - window[0]))
        window.append(now)
        return None


@dataclass(slots=True)
class _UsageAccounting:
    """In-process usage keyed by tenant and server, not server name alone.

    Operators name their own servers and two customers both running a
    ``survival-1`` is ordinary, so a dict keyed on the name would add their
    traffic together and then report the total under whichever tenant happened
    to be first.
    """

    by_server: dict[tuple[str | None, str], _ServerUsage] = field(default_factory=dict)

    def record(
        self,
        credential: ServerCredential,
        outcome: str,
        severity: int | None = None,
        audio_samples: int = 0,
    ) -> None:
        key = (credential.tenant_key, credential.server_id)
        usage = self.by_server.setdefault(key, _ServerUsage())
        usage.record(outcome, severity, audio_samples)

    def for_credential(self, credential: ServerCredential) -> _ServerUsage:
        return self.by_server.get(
            (credential.tenant_key, credential.server_id),
            _ServerUsage(),
        )

    def snapshots(self) -> list[ServerUsageSnapshot]:
        """What the reporter sends. Runs on the event loop, like every write."""
        return [
            ServerUsageSnapshot(
                server_id=server_id,
                tenant_key=tenant_key,
                requests=usage.requests,
                outcomes=dict(usage.outcomes),
                severities=dict(usage.severities),
                audio_seconds=usage.audio_seconds(),
            )
            for (tenant_key, server_id), usage in self.by_server.items()
        ]


def create_app(
    settings: ProcessorSettings,
    *,
    transcribe: Transcribe,
    rule_packs: Mapping[str, RulePack],
    toxicity_classifier: ToxicityClassifier | None = None,
    inspect_packet: InspectPacket = inspect_opus_packet,
    decode_audio: DecodeAudio = decode_opus,
    clock: Callable[[], float] = time.monotonic,
    trace_sink: TraceSink | None = None,
    stage_sink: StageSink | None = None,
    queue_wait_seconds: float = 0.8,
    max_active_entries: int | None = None,
) -> FastAPI:
    if queue_wait_seconds <= 0:
        raise ValueError("queue_wait_seconds must be positive")
    if max_active_entries is not None and max_active_entries < 1:
        raise ValueError("max_active_entries must be positive")
    app = FastAPI()
    # Admission pool: stay out of the way of queue_wait_seconds, and line up
    # with the plugin's max-in-flight after the reserved slot. Measurements and
    # the arithmetic are in docs/processor-admission-sizing.md.
    if max_active_entries is None:
        max_active_entries = max(16, settings.workers * 8) + ADMISSION_RESERVED_SLOTS
    cache = IdempotencyCache(
        max_entries=10_000,
        max_active_entries=max_active_entries,
        ttl_seconds=120,
        clock=clock,
    )
    # Sized on the same number, because this is the resource being shared out:
    # the admission count is what a flooding tenant exhausts, and exhausting it
    # is what refused everybody else before any of them reached a worker.
    tenant_admission = _TenantAdmission(capacity=max_active_entries)
    # Published so an operator can see what the box is actually admitting, and so
    # a test can ask rather than re-derive the formula and quietly go stale when
    # it changes. This one already did.
    app.state.admission_capacity = max_active_entries
    # What one tenant may actually hold, which is the number that has to line up
    # with the plugin's `max-in-flight` and the only one of the two a customer
    # ever experiences. Published separately because reading the capacity as the
    # per-tenant limit is wrong by the reserved slot, and that mistake is what
    # left the two sides of the connection one apart for as long as they were.
    app.state.single_tenant_admissions = max_active_entries - ADMISSION_RESERVED_SLOTS
    # Absent unless this node is configured as a cloud tenant host. A self-hosted
    # processor therefore never opens a socket to authenticate anybody.
    cloud_resolver = (
        CloudCredentialResolver(
            introspection_url=settings.introspection_url,
            service_token=settings.introspection_token,
        )
        if settings.introspection_url and settings.introspection_token
        else None
    )
    processing_slots = asyncio.Semaphore(settings.workers)
    limits = ProtocolLimits(
        max_body_bytes=settings.max_body_bytes,
        max_frame_bytes=settings.max_frame_bytes,
        max_frames=settings.max_frames,
        max_samples_48k=settings.max_samples_48k,
    )
    emit_trace = _log_request_trace if trace_sink is None else trace_sink
    emit_stage = _log_request_stage if stage_sink is None else stage_sink
    rate_limits = _RateLimitWindows(clock=clock)
    # Published so a test can ask rather than re-derive the sweep, and so an
    # operator can see the table the bound is supposed to hold.
    app.state.rate_windows = rate_limits.windows
    usage_accounting = _UsageAccounting()

    # Same switch as the resolver above, and for a stronger reason. A self-hosted
    # processor may not phone home at all: the customer's audio, counters and
    # server names are theirs, and a container that quietly opens an outbound
    # connection once a minute would be a different product from the one they
    # were sold. So when there is no cloud_resolver there is no reporter object
    # to start, rather than a reporter with nowhere to post.
    app.state.usage_reporter = None
    # Same switch again, and the one where it matters most. This reporter is the
    # only thing in the process that can put a word somebody said onto a socket,
    # so on a self-hosted node it is not a disabled feature, it is an object that
    # was never built. `tests/test_event_reporter.py` pins that, alongside the
    # older assertion that a self-hosted processor makes no outbound call at all.
    app.state.event_reporter = None
    app.state.cloud_resolver = cloud_resolver
    if cloud_resolver is not None:
        # Introspection runs on its own thread pool so a slow licensing API
        # cannot starve decode and speech-to-text. Those threads outlive the
        # request that made them, so they get shut down with the application.
        app.router.on_shutdown.append(cloud_resolver.close)
        usage_url = settings.usage_url or usage_url_from(settings.introspection_url)
        if usage_url:
            app.state.usage_reporter = UsageReporter(
                usage_url=usage_url,
                service_token=settings.introspection_token,
                collect=usage_accounting.snapshots,
            )
            # Startup rather than here: create_app is synchronous and there is no
            # running loop yet. `__main__` uses the same router hooks for the
            # micro-batcher, Starlette 1.x having dropped add_event_handler.
            app.router.on_startup.append(app.state.usage_reporter.start)
            app.router.on_shutdown.append(app.state.usage_reporter.stop)
        else:
            # Loud, and not fatal. Usage is evidence for a future rate cap, not
            # something moderation depends on, so a cloud node that cannot work
            # out where to post it should still moderate rather than crash-loop.
            LOGGER.error(
                "usage_reporting_disabled reason=no_usage_url introspection_url=%s "
                "hint=set VOICESNIFFER_USAGE_URL",
                settings.introspection_url,
            )

        events_url = settings.events_url or events_url_from(settings.introspection_url)
        if events_url:
            app.state.event_reporter = EventReporter(
                events_url=events_url,
                service_token=settings.introspection_token,
            )
            app.router.on_startup.append(app.state.event_reporter.start)
            app.router.on_shutdown.append(app.state.event_reporter.stop)
        else:
            # Loud, and not fatal, for the same reason usage is. A cloud node
            # that cannot work out where to file moderation history should keep
            # moderating: the customer's server still gets its verdicts, and a
            # history with a gap in it is a better outcome than a crash loop.
            LOGGER.error(
                "event_reporting_disabled reason=no_events_url introspection_url=%s "
                "hint=set VOICESNIFFER_EVENTS_URL",
                settings.introspection_url,
            )

    @app.get("/healthz", include_in_schema=False)
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/usage")
    async def usage(request: Request) -> JSONResponse:
        try:
            credential = await _authenticate(request, settings, cloud_resolver)
        except IntrospectionUnavailable:
            return _error(None, "license_check_unavailable", 503)
        if credential is None:
            return _error(None, "unauthorized", 401)
        allowed = credential.languages
        recorded = usage_accounting.for_credential(credential)
        return JSONResponse(
            {
                "server_id": credential.server_id,
                "plan": credential.plan,
                "languages": "all" if allowed is None else sorted(allowed),
                # What this processor can actually apply rules in, for this
                # credential. `languages` above is what the licence permits, and
                # the two are not the same number: a language with no installed
                # rule pack is licensed and unmoderatable at once. That gap used
                # to be visible nowhere at all, so an operator could buy a
                # language, configure it, and be told everything was fine.
                "moderated_languages": sorted(_entitled(rule_packs, allowed)),
                "rate_limit_per_minute": credential.rate_limit_per_minute,
                "usage": recorded.snapshot(),
            }
        )

    @app.post("/v1/moderate")
    async def moderate(request: Request) -> JSONResponse:
        try:
            credential = await _authenticate(request, settings, cloud_resolver)
        except IntrospectionUnavailable:
            # Not 401. The licensing API could not be asked, which is our
            # problem and not a statement about this customer's token; saying
            # `unauthorized` here told every cloud server its credentials had
            # been rejected every time that API restarted, and the plugin does
            # not retry a 401. 503 is retryable there, so a blip inside the
            # request deadline now rides out, and a longer outage at least names
            # itself. `cloud_auth` explains why an outage still admits nobody.
            return _error(None, "license_check_unavailable", 503)
        if credential is None:
            return _error(None, "unauthorized", 401)
        server_id = credential.server_id
        if credential.expires_at is not None and _utc_now() >= credential.expires_at:
            usage_accounting.record(credential, "license_expired")
            return _error(None, "license_expired", 403)
        retry_after = rate_limits.wait(credential)
        if retry_after is not None:
            usage_accounting.record(credential, "rate_limited")
            return _error(
                None,
                "rate_limited",
                429,
                headers={"Retry-After": str(math.ceil(retry_after))},
            )
        request_id = _parse_request_id(request.headers.get("X-Request-Id", ""))
        if request_id is None:
            return _error(None, "invalid_request_id", 400)
        request_started = clock()
        emit_stage(RequestStageTrace(request_id, "request", "received"))
        if request.headers.get("Content-Type", "").split(";", 1)[0].strip() != CONTENT_TYPE:
            return _error(request_id, "unsupported_media_type", 415)
        if request.headers.get("Accept", "").strip() != "application/json":
            return _error(request_id, "not_acceptable", 406)

        metadata = _parse_metadata(request, request_id, settings.max_samples_48k)
        if isinstance(metadata, JSONResponse):
            return metadata
        player_id, language, preroll_samples = metadata

        partial = _parse_partial(request, request_id)
        if isinstance(partial, JSONResponse):
            return partial

        # Plan enforcement: a pinned language outside the licence is refused outright;
        # auto-detect is narrowed to the licensed packs so an en-only plan can never
        # produce (or bill for) moderation in a language it did not buy.
        requested_languages = () if language == "auto" else tuple(language.split(","))
        allowed_languages = credential.languages
        if allowed_languages is not None and any(
            code not in allowed_languages for code in requested_languages
        ):
            usage_accounting.record(credential, "language_not_licensed")
            return _error(request_id, "language_not_licensed", 403)
        entitled_packs = _entitled(rule_packs, allowed_languages)

        # Which packs this request will actually run. A pinned code with no
        # installed pack contributes nothing: `_process` selects with
        # `rule_packs.get(code)` and filters the `None`s out.
        #
        # Dropping *some* of a list is deliberate and stays -- a server that
        # says `en,de` on a box with no de.yml should still have its English
        # moderated. Dropping *all* of them is a different thing entirely. That
        # request would come back 200 with `severity: 0` and no matches, file no
        # moderation event, log nothing, and bill for the audio, which is a
        # server that believes it is protected and is running no rules at all.
        # The model advertises 25 languages and 23 packs ship, so `fi` and `mt`
        # are exactly this case today: see `_report_unmoderatable_languages` in
        # `__main__`, which names the gap at startup.
        moderated = (
            tuple(code for code in requested_languages if code in entitled_packs)
            if requested_languages
            else tuple(entitled_packs)
        )
        if not moderated:
            LOGGER.error(
                "language_unsupported request_id=%s server=%s requested=%s installed=%s",
                request_id,
                server_id,
                language,
                ",".join(sorted(rule_packs)) or "none",
            )
            usage_accounting.record(credential, "language_unsupported")
            return _error(request_id, "language_unsupported", 400)

        content_length = request.headers.get("Content-Length")
        if content_length is not None:
            try:
                parsed_content_length = int(content_length)
                if parsed_content_length < 0:
                    return _error(request_id, "invalid_content_length", 400)
                if parsed_content_length > settings.max_body_bytes:
                    return _error(request_id, "body_too_large", 413)
            except ValueError:
                return _error(request_id, "invalid_content_length", 400)

        body = bytearray()
        async for chunk in request.stream():
            if len(chunk) > settings.max_body_bytes - len(body):
                return _error(request_id, "body_too_large", 413)
            body.extend(chunk)
        audio = bytes(body)
        digest = _request_digest(player_id, language, preroll_samples, partial, audio)

        decoded_audio_samples = 0

        async def process() -> _ProcessedVerdict:
            nonlocal decoded_audio_samples
            result = await _run_with_processing_slot(
                processing_slots,
                queue_wait_seconds,
                lambda: _in_thread(
                    request_id,
                    language,
                    preroll_samples,
                    audio,
                    limits,
                    transcribe,
                    entitled_packs,
                    toxicity_classifier,
                    inspect_packet,
                    decode_audio,
                    emit_stage,
                ),
            )
            # Counted here rather than from the verdict the handler ends up with,
            # because the idempotency cache only calls this for the request that
            # actually decoded. A plugin retrying after a timeout gets the stored
            # verdict back and must not have the same audio counted twice.
            decoded_audio_samples = result.audio_samples
            # Moderation history, captured in the same place and for the same
            # reason. This body runs once per utterance that really got decoded;
            # a retry inside the idempotency window is answered from the cache
            # and never reaches here, so the transcript is filed exactly once
            # rather than once per attempt. The receiver's unique index on
            # (tenant_key, request_id) is the second line of that defence, for
            # the case where a retry arrives after the cache entry has aged out.
            #
            # Neither of those defences can see the duplicate this header is
            # about: a window and the utterance that follows it are different
            # request ids carrying different audio, so they are two genuine
            # requests by every test available here. Only the sender knows one
            # of them is a preview of the other, so only the sender can say so.
            if not partial:
                _capture_flagged_event(
                    app.state.event_reporter,
                    credential,
                    request_id,
                    player_id,
                    result.verdict,
                )
            return result

        processed: _ProcessedVerdict | None = None
        serialize_ms: int | None = None
        # Scoped to the tenant, not the server name. `server_id` is chosen by
        # the customer and the contract says outright that two of them may both
        # pick "survival-1". The rate limiter and the usage counters a few lines
        # above were keyed this way for exactly that reason and this one was
        # missed in the same pass, which mattered more here than there: what
        # crosses an idempotency key is a ModerationVerdict carrying transcript
        # and matched_text. "No customer's transcript can reach another
        # customer" is now a property of the key rather than an argument about
        # how unlikely a UUID collision is.
        cache_scope = f"{credential.tenant_key or 'selfhost'}|{server_id}"
        tenant_scope = credential.tenant_key or "selfhost"
        # Last thing before the work starts, so a malformed request never costs
        # anybody a share, and first thing before the idempotency cache, because
        # its global admission cap is what one tenant was exhausting.
        refusal = tenant_admission.acquire(tenant_scope)
        if refusal is not None:
            # Counted apart from `processor_busy`, sent as 503 with
            # `processor_busy` in the body.
            #
            # What the plugin does with that: it reads the status and nothing
            # else. The code in the body has never reached it, so neither name
            # would change its behaviour, and the reason to keep the wire code
            # stable is the contract and the operator reading a response by
            # hand, not the client. Anything that has to change what the plugin
            # does has to change the status.
            #
            # Operationally the two refusals are not the same: one says the box
            # is full, the other says the box is contended and names who is
            # doing it, which is the difference between needing a bigger machine
            # and needing the rate cap the contract says is a data change. That
            # distinction exists only if it is written down, so it goes to the
            # log and to /v1/usage.
            LOGGER.warning(
                "admission_refused reason=%s tenant=%s held=%s share=%s competing=%s",
                refusal,
                tenant_scope,
                tenant_admission.in_flight.get(tenant_scope, 0),
                tenant_admission.share(tenant_scope),
                tenant_admission.competing(tenant_scope),
            )
            usage_accounting.record(credential, refusal)
            emit_trace(_request_trace(request_id, refusal, None, None, request_started, clock))
            return _error(request_id, "processor_busy", 503)
        cache_task = asyncio.create_task(cache.execute(cache_scope, request_id, digest, process))
        # On the task, not in a `finally` on this handler. The handler can be
        # cancelled while the task keeps running, and the entry it holds is the
        # thing being rationed, so the share has to last exactly as long as it.
        cache_task.add_done_callback(lambda _task: tenant_admission.release(tenant_scope))
        try:
            processed = await asyncio.shield(cache_task)
        except (ProcessorBusyError, IdempotencyCapacityError):
            outcome = "processor_busy"
            response = _error(request_id, outcome, 503)
        except RequestIdReuseError:
            outcome = "request_id_reused"
            response = _error(request_id, outcome, 409)
        except ProtocolError as exception:
            if exception.code == "opus_unavailable":
                status = 503
            else:
                status = 413 if exception.code in PAYLOAD_ERROR_CODES else 400
            outcome = exception.code
            response = _error(request_id, outcome, status)
        except OpusDecodeError as exception:
            status = 503 if exception.code == "opus_unavailable" else 400
            outcome = exception.code
            response = _error(request_id, outcome, status)
        except UnsupportedTranscriptionLanguage as exception:
            # A licensed code with an installed rule pack that the speech model
            # on this node cannot transcribe. It used to reach the clause below
            # and come back 500 with an empty log, so a deployment whose model
            # and rules directory disagree looked like a crashing processor. The
            # answer is the one a missing pack already gets, because the caller
            # is being told the same thing: not here. `__main__` names the whole
            # gap at startup so this does not have to be discovered per request.
            outcome = "language_unsupported"
            LOGGER.error(
                "language_untranscribable request_id=%s server=%s requested=%s untranscribable=%s "
                "hint=the model in models.toml was not trained on this language; "
                "either deploy a model that was or remove the rule pack",
                request_id,
                server_id,
                language,
                exception.language,
            )
            response = _error(request_id, outcome, 400)
        except asyncio.CancelledError:
            cache_task.add_done_callback(_consume_background_result)
            emit_trace(_request_trace(request_id, "cancelled", None, None, request_started, clock))
            raise
        except Exception:
            outcome = "internal_error"
            # A 500 that logs nothing is a failure nobody can diagnose. The
            # response carries a code and no detail on purpose, so if the
            # traceback is not written down here the only record of what broke
            # is gone by the time anybody looks. What goes in the log is the
            # call path and the exception, never the request body and never the
            # transcript: neither is a local of any frame on this stack, and
            # `logging` formats source lines rather than variables.
            LOGGER.exception(
                "request_failed request_id=%s server=%s language=%s",
                request_id,
                server_id,
                language,
            )
            response = _error(request_id, outcome, 500)
        else:
            outcome = "success"
            emit_stage(RequestStageTrace(request_id, "serialize", "started"))
            serialize_started = clock()
            response = JSONResponse(
                processed.verdict.model_dump(mode="json"),
                headers={
                    "X-Request-Id": request_id,
                    DECODE_TIMING_HEADER: str(processed.decode_ms),
                    TRANSCRIBE_TIMING_HEADER: str(processed.transcribe_ms),
                    RULES_TIMING_HEADER: str(processed.rules_ms),
                },
            )
            serialize_ms = _elapsed_ms(serialize_started, clock)
            emit_stage(RequestStageTrace(request_id, "serialize", "completed", serialize_ms))
        emit_trace(
            _request_trace(request_id, outcome, processed, serialize_ms, request_started, clock)
        )
        # A window contributes its decoded audio and nothing else. The audio is
        # real work on real seconds and the cost of it was paid; the verdict is a
        # preview of one the finished utterance will produce again, and counting
        # a severity for both is what inflated the flagged number. See
        # `_ServerUsage` for why the two totals are not expected to agree.
        usage_accounting.record(
            credential,
            outcome,
            processed.verdict.severity if processed is not None and not partial else None,
            decoded_audio_samples,
        )
        return response

    return app


def _entitled(
    rule_packs: Mapping[str, RulePack],
    allowed_languages: frozenset[str] | None,
) -> Mapping[str, RulePack]:
    """The installed packs this credential is licensed for.

    ``None`` means every installed pack, which is what an unrestricted licence
    and every self-hosted credential carry.
    """
    if allowed_languages is None:
        return rule_packs
    return {code: pack for code, pack in rule_packs.items() if code in allowed_languages}


def _capture_flagged_event(
    reporter: EventReporter | None,
    credential: ServerCredential,
    request_id: str,
    player_id: uuid.UUID,
    verdict: ModerationVerdict,
) -> None:
    """Queue this utterance for the moderation history, if it is one that counts.

    Four conditions, and all of them are refusals rather than filters applied
    later, because each one is a promise somebody made in writing:

    - **No reporter.** Self-hosted. Nothing about this customer's players leaves
      their machine, and the object that could send it was never built.
    - **No tenant key.** A cloud node's break-glass token out of ``tokens.json``,
      which belongs to the operator rather than to a paying tenant. There is
      nobody to file it under and nobody who could read it back.
    - **No matches.** Speech that fired no rule. This is the ordinary case and it
      leaves no record anywhere, which is the difference between a moderation log
      and a recording of a voice channel.
    - **Anything at all going wrong.** Swallowed. History is a convenience for
      the owner; the verdict is what the customer is paying for, and nothing on
      this path may fail a moderation request or delay one. Nothing about the
      utterance reaches the log line either, which is why it carries a type name.
    """
    if reporter is None or credential.tenant_key is None:
        return
    try:
        event = flagged_event_from(
            tenant_key=credential.tenant_key,
            server_id=credential.server_id,
            request_id=request_id,
            player_uuid=str(player_id),
            occurred_at=_utc_now(),
            transcript=verdict.transcript,
            matches=verdict.matches,
        )
        if event is not None:
            reporter.record(event)
    except Exception as error:
        LOGGER.warning("event_capture_failed error=%s", type(error).__name__)


class _ServerUsage:
    """In-memory per-server counters since process start. Deliberately simple: the
    durable ledger lives with the licensing infrastructure; this is the processor's
    own view for operators, the /v1/usage endpoint and the usage reporter.

    Counts only. Whatever is added here ends up on the wire once a minute, so a
    field holding a transcript, a matched phrase or a rule id would turn a usage
    feed into a record of what players said. Nothing here may hold text.

    **These counters are not all counting the same thing, and that is on
    purpose.** `flagged` and `severities` are per *incident*: one player saying
    one thing once, however many requests it took to establish that.
    `audio_seconds` is per *decode*: every second of audio this box actually
    turned into samples, whoever asked and whatever came of it. `requests` and
    `outcomes` are per request, which is a third thing again.

    The gap between them is the early windows. The plugin sends a word twice,
    once in a window while the sentence is still running and once in the
    finished utterance (see `PARTIAL_HEADER`), so one incident can arrive as two
    or more decodes. A window's audio is counted and its verdict is not.

    So do not divide these into each other and do not expect a ratio to hold.
    "Flagged per audio second" is not a rate this service reports, and reading
    it as one will make a busy server look like a clean one, because windows add
    to the denominator and never to the numerator. If a future plan bills on
    audio it bills on `audio_seconds`; if it caps incidents it caps on
    `flagged`; nothing converts between them."""

    __slots__ = ("audio_samples", "flagged", "outcomes", "requests", "severities", "started_at")

    def __init__(self) -> None:
        self.started_at = time.time()
        self.requests = 0
        self.flagged = 0
        self.outcomes: dict[str, int] = {}
        self.severities: dict[int, int] = {}
        self.audio_samples = 0

    def record(self, outcome: str, severity: int | None = None, audio_samples: int = 0) -> None:
        self.requests += 1
        self.outcomes[outcome] = self.outcomes.get(outcome, 0) + 1
        if severity is not None:
            self.severities[severity] = self.severities.get(severity, 0) + 1
            if severity > 0:
                self.flagged += 1
        self.audio_samples += audio_samples

    def audio_seconds(self) -> int:
        # Truncated, not rounded. The receiver stores the difference between
        # consecutive cumulative reports, and flooring a non-decreasing sample
        # count keeps that difference non-negative while never overstating what
        # a tenant sent.
        return self.audio_samples // DECODED_SAMPLE_RATE

    def snapshot(self) -> dict[str, object]:
        return {
            "since": datetime.datetime.fromtimestamp(
                self.started_at, tz=datetime.UTC
            ).isoformat(timespec="seconds"),
            "requests": self.requests,
            "flagged": self.flagged,
            "outcomes": dict(sorted(self.outcomes.items())),
            "severities": {str(level): count for level, count in sorted(self.severities.items())},
            "audio_seconds": self.audio_seconds(),
        }


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(tz=datetime.UTC)


async def _run_with_processing_slot(
    processing_slots: asyncio.Semaphore,
    queue_wait_seconds: float,
    operation: Callable[[], Awaitable[_ProcessedVerdict]],
) -> _ProcessedVerdict:
    try:
        await asyncio.wait_for(processing_slots.acquire(), timeout=queue_wait_seconds)
    except TimeoutError as exception:
        raise ProcessorBusyError from exception

    worker = asyncio.create_task(operation())
    try:
        return await asyncio.shield(worker)
    finally:
        if worker.done():
            processing_slots.release()
        else:
            worker.add_done_callback(
                lambda completed: _release_processing_slot(processing_slots, completed)
            )


def _release_processing_slot(
    processing_slots: asyncio.Semaphore,
    worker: asyncio.Task[_ProcessedVerdict],
) -> None:
    processing_slots.release()
    if not worker.cancelled():
        worker.exception()


def _consume_background_result(worker: asyncio.Task[_ProcessedVerdict]) -> None:
    if not worker.cancelled():
        worker.exception()


async def _in_thread(
    request_id: str,
    language: str,
    preroll_samples: int,
    body: bytes,
    limits: ProtocolLimits,
    transcribe: Transcribe,
    rule_packs: Mapping[str, RulePack],
    toxicity_classifier: ToxicityClassifier | None,
    inspect_packet: InspectPacket,
    decode_audio: DecodeAudio,
    emit_stage: StageSink,
) -> _ProcessedVerdict:
    return await asyncio.to_thread(
        _process,
        request_id,
        language,
        preroll_samples,
        body,
        limits,
        transcribe,
        rule_packs,
        toxicity_classifier,
        inspect_packet,
        decode_audio,
        emit_stage,
    )


def _process(
    request_id: str,
    language: str,
    preroll_samples: int,
    body: bytes,
    limits: ProtocolLimits,
    transcribe: Transcribe,
    rule_packs: Mapping[str, RulePack],
    toxicity_classifier: ToxicityClassifier | None,
    inspect_packet: InspectPacket,
    decode_audio: DecodeAudio,
    emit_stage: StageSink,
) -> _ProcessedVerdict:
    started = time.perf_counter()
    emit_stage(RequestStageTrace(request_id, "decode", "started"))
    decode_started = time.perf_counter()
    envelope = parse_opus_envelope(body, preroll_samples, limits, inspect_packet)
    samples = decode_audio(envelope)
    decode_ms = max(0, round((time.perf_counter() - decode_started) * 1_000))
    emit_stage(RequestStageTrace(request_id, "decode", "completed", decode_ms))
    emit_stage(RequestStageTrace(request_id, "stt", "started"))
    transcribe_started = time.perf_counter()
    # The transducer takes no language argument at all -- every code maps to the
    # same recognizer -- so which of the requested codes is handed over does not
    # affect the transcript. It still has to be one the adapter knows.
    requested = () if language == "auto" else tuple(language.split(","))
    transcription = transcribe(samples, requested[0] if requested else "auto")
    transcribe_ms = max(0, round((time.perf_counter() - transcribe_started) * 1_000))
    emit_stage(RequestStageTrace(request_id, "stt", "completed", transcribe_ms))
    emit_stage(RequestStageTrace(request_id, "rules", "started"))
    rules_started = time.perf_counter()
    transcript = normalize_transcript(transcription.text)
    result_language = transcription.language
    # Selection follows what the server asked for, not what the model claims to
    # have heard, because for this model it claims nothing. `auto` still means
    # every entitled pack, and now says out loud that it is the expensive option.
    if requested:
        packs = tuple(rule_packs.get(code) for code in requested)
    elif result_language == "auto":
        packs = tuple(rule_packs.values())
    else:
        packs = (rule_packs.get(result_language),)
    # `transcript` is already normalised above; say so rather than paying for it
    # again inside every pack.
    rule_verdicts = tuple(
        pack.match(transcript, normalized=True) for pack in packs if pack is not None
    )
    timed_out_rule_ids = tuple(
        rule_id for rule_verdict in rule_verdicts for rule_id in rule_verdict.timed_out_rule_ids
    )
    if timed_out_rule_ids:
        LOGGER.warning(
            "rule_timeout request_id=%s language=%s count=%s rule_ids=%s",
            request_id,
            result_language,
            len(timed_out_rule_ids),
            ",".join(timed_out_rule_ids[:MAX_LOGGED_RULE_TIMEOUTS]),
        )
    skipped_rule_ids = tuple(
        rule_id for rule_verdict in rule_verdicts for rule_id in rule_verdict.skipped_rule_ids
    )
    if skipped_rule_ids:
        # Louder than a rule timeout: this means the pack ran out of its whole
        # budget and part of the ruleset was never applied to this utterance, so
        # the verdict is known-incomplete rather than merely slow.
        LOGGER.error(
            "rule_budget_exhausted request_id=%s language=%s unevaluated=%s rule_ids=%s",
            request_id,
            result_language,
            len(skipped_rule_ids),
            ",".join(skipped_rule_ids[:MAX_LOGGED_RULE_TIMEOUTS]),
        )
    matches = tuple(
        VerdictMatch(
            rule_id=match.rule_id,
            category=match.category,
            severity=match.severity,
            matched_text=match.matched_text,
            context_required=match.context_required,
        )
        for rule_verdict in rule_verdicts
        for match in rule_verdict.matches
    )
    rules_ms = max(0, round((time.perf_counter() - rules_started) * 1_000))
    emit_stage(RequestStageTrace(request_id, "rules", "completed", rules_ms))
    emit_stage(RequestStageTrace(request_id, "classifier", "started"))
    classifier_started = time.perf_counter()
    toxicity_match, toxicity_confidence = _toxicity_match(
        toxicity_classifier,
        transcript,
        result_language,
    )
    if toxicity_match is not None:
        matches = (*matches, toxicity_match)
    classifier_ms = max(0, round((time.perf_counter() - classifier_started) * 1_000))
    emit_stage(RequestStageTrace(request_id, "classifier", "completed", classifier_ms))
    severity = max((match.severity for match in matches if not match.context_required), default=0)
    rule_confidence = (
        1.0 if any(match.rule_id != "tier2a.local-toxicity" for match in matches) else 0.0
    )
    confidence = max(rule_confidence, toxicity_confidence)
    elapsed_ms = max(0, round((time.perf_counter() - started) * 1_000))
    return _ProcessedVerdict(
        verdict=ModerationVerdict(
            request_id=request_id,
            transcript=transcript,
            language=result_language,
            matches=matches,
            severity=severity,
            confidence=confidence,
            processing_ms=elapsed_ms,
        ),
        decode_ms=decode_ms,
        transcribe_ms=transcribe_ms,
        rules_ms=rules_ms,
        classifier_ms=classifier_ms,
        # Known exactly here and nowhere else, which is why the contract's
        # audio_seconds is measured rather than estimated: this is the count the
        # recognizer was actually handed, after the preroll trim.
        audio_samples=int(samples.size),
    )


def _request_trace(
    request_id: str,
    outcome: str,
    processed: _ProcessedVerdict | None,
    serialize_ms: int | None,
    started: float,
    clock: Callable[[], float],
) -> RequestTrace:
    total_ms = _elapsed_ms(started, clock)
    if processed is not None:
        total_ms = max(total_ms, processed.verdict.processing_ms + (serialize_ms or 0))
    return RequestTrace(
        request_id=request_id,
        outcome=outcome,
        decode_ms=None if processed is None else processed.decode_ms,
        stt_ms=None if processed is None else processed.transcribe_ms,
        rules_ms=None if processed is None else processed.rules_ms,
        classifier_ms=None if processed is None else processed.classifier_ms,
        serialize_ms=serialize_ms,
        total_ms=total_ms,
    )


def _elapsed_ms(started: float, clock: Callable[[], float]) -> int:
    return max(0, round((clock() - started) * 1_000))


def _log_request_trace(trace: RequestTrace) -> None:
    LOGGER.info(
        "request_trace request_id=%s outcome=%s total_ms=%s decode_ms=%s stt_ms=%s "
        "rules_ms=%s classifier_ms=%s serialize_ms=%s",
        trace.request_id,
        trace.outcome,
        trace.total_ms,
        trace.decode_ms,
        trace.stt_ms,
        trace.rules_ms,
        trace.classifier_ms,
        trace.serialize_ms,
    )


def _log_request_stage(trace: RequestStageTrace) -> None:
    LOGGER.info(
        "request_stage request_id=%s stage=%s state=%s duration_ms=%s",
        trace.request_id,
        trace.stage,
        trace.state,
        trace.duration_ms,
    )


def _toxicity_match(
    toxicity_classifier: ToxicityClassifier | None,
    transcript: str,
    language: str,
) -> tuple[VerdictMatch | None, float]:
    if toxicity_classifier is None:
        return None, 0.0
    result = toxicity_classifier(transcript, language)
    if result is None:
        return None, 0.0
    if (
        result.category not in CATEGORIES
        or result.severity not in (1, 2, 3)
        or result.confidence < 0.0
        or result.confidence > 1.0
    ):
        raise ValueError("invalid toxicity classifier result")
    return VerdictMatch(
        rule_id="tier2a.local-toxicity",
        category=result.category,
        severity=result.severity,
        matched_text=result.matched_text,
        context_required=False,
    ), result.confidence


async def _authenticate(
    request: Request,
    settings: ProcessorSettings,
    resolver: CloudCredentialResolver | None = None,
) -> ServerCredential | None:
    """The static file first, then the licensing API if this is a cloud node.

    Order matters. tokens.json is local, free and cannot be influenced from
    outside, so a self-hosted processor never makes a network call and a cloud
    processor still answers instantly for anything it has already seen.

    ``None`` means "no credential for this token". It does **not** mean "the
    licensing API could not be asked": that raises ``IntrospectionUnavailable``
    and every caller has to answer it differently, because one is the customer's
    problem and the other is ours.
    """
    authorization = request.headers.get("Authorization", "")
    match = AUTHORIZATION_PATTERN.fullmatch(authorization)
    if match is None:
        return None
    presented = match.group(1)
    credential = settings.credential_for_token(presented)
    if credential is not None or resolver is None:
        return credential
    return await resolver.resolve(presented)


def _parse_metadata(
    request: Request,
    request_id: str,
    max_samples_48k: int,
) -> tuple[uuid.UUID, str, int] | JSONResponse:
    player_id = request.headers.get("X-VoiceSniffer-Player-Id", "")
    try:
        parsed_player_id = uuid.UUID(player_id)
    except ValueError:
        return _error(request_id, "invalid_player_id", 400)

    # One code, a comma-separated list of them, or "auto". The list is the point:
    # the model never reports which language it heard, so "auto" runs the
    # transcript through every installed pack. At two packs that was tolerable.
    # At twenty-five it means a Czech utterance is matched against Polish,
    # Slovak, Croatian and Slovenian rules, which are close enough to collide for
    # real. A server knows what its own community speaks, so it says so.
    language = request.headers.get("X-VoiceSniffer-Language", "")
    if language != "auto":
        codes = language.split(",")
        if not codes or len(codes) > MAX_REQUEST_LANGUAGES:
            return _error(request_id, "invalid_language", 400)
        if any(LANGUAGE_PATTERN.fullmatch(code) is None for code in codes):
            return _error(request_id, "invalid_language", 400)
        if len(set(codes)) != len(codes):
            return _error(request_id, "invalid_language", 400)
    try:
        preroll_samples = int(request.headers.get("X-VoiceSniffer-Preroll-Samples", ""))
    except ValueError:
        return _error(request_id, "invalid_preroll", 400)
    if preroll_samples < 0 or preroll_samples > max_samples_48k or preroll_samples % 3 != 0:
        return _error(request_id, "invalid_preroll", 400)
    return parsed_player_id, language, preroll_samples


def _parse_partial(request: Request, request_id: str) -> bool | JSONResponse:
    """Is this an early window rather than the finished utterance?

    Absent means no, because that is what every plugin released before this
    header says, and one of those must keep getting exactly the counts it gets
    today. Present means it has to be `0` or `1`: a sender that means to mark
    windows and writes `true` or `yes` has a bug, and guessing at it would hide
    the bug behind counters that quietly go back to counting one slur twice.
    Validated like the language and preroll headers next door, for that reason.
    """
    value = request.headers.get(PARTIAL_HEADER)
    if value is None:
        return False
    partial = PARTIAL_VALUES.get(value.strip())
    if partial is None:
        return _error(request_id, "invalid_partial", 400)
    return partial


def _parse_request_id(value: str) -> str | None:
    try:
        parsed = uuid.UUID(value)
    except ValueError:
        return None
    canonical = str(parsed)
    return canonical if canonical == value else None


def _request_digest(
    player_id: uuid.UUID,
    language: str,
    preroll_samples: int,
    partial: bool,
    body: bytes,
) -> bytes:
    digest = hashlib.sha256()
    digest.update(player_id.bytes)
    digest.update(language.encode("ascii"))
    digest.update(preroll_samples.to_bytes(8, byteorder="big"))
    digest.update(b"\x01" if partial else b"\x00")
    digest.update(body)
    return digest.digest()


def _error(
    request_id: str | None,
    code: str,
    status: int,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    response = ErrorResponse(request_id=request_id, code=code, message=code)
    combined = {} if request_id is None else {"X-Request-Id": request_id}
    if headers:
        combined.update(headers)
    return JSONResponse(
        response.model_dump(mode="json"),
        status_code=status,
        headers=combined,
    )
