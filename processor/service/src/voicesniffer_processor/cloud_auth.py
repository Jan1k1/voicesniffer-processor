"""Credentials for cloud tenants, resolved against the licensing API.

Self-hosted processors read ``tokens.json`` at startup. Cloud processors ask
the licensing API about tokens they have not seen. The design, covering revocation
versus offline verification, split positive/negative caches, outage versus
refusal, and why introspection has its own executor, is in
``docs/cloud-credential-resolver.md``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.error
import urllib.request
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC

from .settings import ServerCredential

LOGGER = logging.getLogger("voicesniffer.processor")

# A hit is cheap to re-check and a revocation should bite quickly.
DEFAULT_TTL_SECONDS = 30.0
# A miss is almost always a scan, so it is worth remembering for longer. This is
# also the ceiling on how fast a newly minted token starts working, which is why
# it is not minutes: a customer running /vs connect should not wait.
DEFAULT_NEGATIVE_TTL_SECONDS = 60.0
DEFAULT_TIMEOUT_SECONDS = 2.0
# Known-good credentials, and nothing an unauthenticated caller can reach. The
# real population is the number of paired cloud servers with a live token, which
# is orders of magnitude below this, so in practice a paying customer is never
# evicted at all. It stays at the old whole-cache figure because evicting one of
# these costs a paying customer a round trip and there is no reason to make that
# more likely than it already was.
MAX_POSITIVE_CACHE_ENTRIES = 4_096
# Refusals. This is the only budget a stranger can drive, which is why it is a
# budget: it caps what a flood can cost in memory, and holding more junk than
# this buys nothing, because a caller who never repeats a token misses whatever
# the size. What it does buy is the case negative caching exists for, a bad
# token presented over and over, and that needs room for the working set rather
# than for the scan. Kept at the old whole-cache figure so absorbing repeated
# refusals is no worse than before.
MAX_NEGATIVE_CACHE_ENTRIES = 4_096
# Introspection is a loopback POST that blocks its thread on I/O, so this is a
# concurrency knob and not a share of the CPU. Eight is enough that legitimate
# re-validation keeps flowing even when the licensing API is sick and every call
# runs to the full timeout: at two seconds each that still clears four lookups a
# second, well above what the TTL asks for from any plausible number of servers.
INTROSPECTION_THREADS = 8


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    credential: ServerCredential | None
    expires_at: float


@dataclass(slots=True)
class _Inflight:
    """One lookup's lock, and a count of the callers that still need it.

    The count is the whole point. ``_locks`` used to be pruned only by ``_store``,
    and the ``IntrospectionUnavailable`` path leaves before ``_store`` runs, so
    during a licensing API outage every distinct token presented left a lock
    object behind for good: 5,000 tokens, 5,000 locks, against a 3 GB container.
    An outage is already a degraded state and is not the moment to start leaking.

    Counting instead of pruning also fixes a smaller bug in the old shape. It
    evicted the lock belonging to whichever entry the cache dropped, which could
    be a lock another coroutine was at that moment waiting on; the next caller
    for that token created a second lock and the single-flight guarantee quietly
    stopped holding. Here the last caller out removes the entry, so a burst
    still shares exactly one lock, and the table is empty again afterwards.
    """

    lock: asyncio.Lock
    waiters: int = 0


@dataclass
class CloudCredentialResolver:
    """Looks a bearer token up with the licensing API, and remembers the answer."""

    introspection_url: str
    service_token: str
    ttl_seconds: float = DEFAULT_TTL_SECONDS
    negative_ttl_seconds: float = DEFAULT_NEGATIVE_TTL_SECONDS
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    now: Callable[[], float] = time.monotonic
    # Two tables, two budgets. See docs/cloud-credential-resolver.md: the split
    # is what stops unauthenticated traffic from evicting paying customers.
    _positive: OrderedDict[bytes, _CacheEntry] = field(default_factory=OrderedDict, repr=False)
    _negative: OrderedDict[bytes, _CacheEntry] = field(default_factory=OrderedDict, repr=False)
    _locks: dict[bytes, _Inflight] = field(default_factory=dict, repr=False)
    _executor: ThreadPoolExecutor | None = field(default=None, repr=False)

    async def resolve(self, presented_token: str) -> ServerCredential | None:
        """The credential for this token, ``None`` if the API says there is none.

        Raises ``IntrospectionUnavailable`` when the licensing API could not be
        asked at all. That is a third outcome, not a refusal, and the caller has
        to keep it separate: a refusal is the customer's problem and an outage is
        ours.
        """
        from .settings import _token_digest

        digest = _token_digest(presented_token)
        cached = self._cached(digest)
        if cached is not None:
            return cached.credential

        # One in-flight lookup per token. Without this a server starting up with
        # forty threads asks the licensing API forty times for the same answer.
        inflight = self._locks.get(digest)
        if inflight is None:
            inflight = _Inflight(asyncio.Lock())
            self._locks[digest] = inflight
        # Claimed before the await, dropped in the finally, so every exit path
        # tidies up: cache hit, refusal, outage and cancellation alike.
        inflight.waiters += 1
        try:
            async with inflight.lock:
                cached = self._cached(digest)
                if cached is not None:
                    return cached.credential
                # `_introspect` raises `IntrospectionUnavailable` on an outage,
                # and that propagates on purpose. It is deliberately *not*
                # cached either, so a paying customer whose lookup landed during
                # a restart works again on the next request rather than in a
                # minute. The `finally` below still runs, so the lock is
                # released on this path exactly as on the others.
                credential = await self._introspect(presented_token)
                self._store(digest, credential)
                return credential
        finally:
            inflight.waiters -= 1
            if inflight.waiters == 0 and self._locks.get(digest) is inflight:
                del self._locks[digest]

    def _cached(self, digest: bytes) -> _CacheEntry | None:
        """The live entry for this token, from whichever table holds it.

        Returns the entry rather than the credential because a cached refusal is
        a real answer and ``None`` would be indistinguishable from a miss.
        """
        now = self.now()
        for table in (self._positive, self._negative):
            entry = table.get(digest)
            if entry is None:
                continue
            if entry.expires_at > now:
                return entry
            # Expired, so drop it here rather than waiting for the eviction path
            # to reach it. A table full of stale entries is a table with no room
            # for live ones.
            del table[digest]
        return None

    def _store(self, digest: bytes, credential: ServerCredential | None) -> None:
        if credential is not None:
            table, other = self._positive, self._negative
            budget, ttl = MAX_POSITIVE_CACHE_ENTRIES, self.ttl_seconds
        else:
            table, other = self._negative, self._positive
            budget, ttl = MAX_NEGATIVE_CACHE_ENTRIES, self.negative_ttl_seconds
        # A token whose answer changed, revoked or newly minted, must not leave
        # an entry of the other kind behind for `_cached` to find first.
        other.pop(digest, None)
        # Re-inserting rather than overwriting, so insertion order stays expiry
        # order and a refreshed entry goes to the back where it belongs.
        table.pop(digest, None)
        while len(table) >= budget:
            # Oldest first. Within one table every entry has the same TTL, so
            # insertion order *is* expiry order and this evicts exactly what the
            # old `min(expires_at)` scan did, in constant time instead of one
            # pass over 4,096 entries per insert. That scan was its own small
            # amplifier: once the cache was full, every junk token in a flood
            # paid for it.
            table.popitem(last=False)
        table[digest] = _CacheEntry(credential, self.now() + ttl)

    async def _introspect(self, presented_token: str) -> ServerCredential | None:
        """``_fetch`` on the introspection executor, never the default one.

        ``asyncio.to_thread`` would put a two second blocking socket read into
        the same pool that decode and speech-to-text use. See the module
        docstring for the measurement.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._pool(), self._fetch, presented_token)

    def _pool(self) -> ThreadPoolExecutor:
        # Built on first use, so a self-hosted processor and every test that
        # never resolves a cloud token pay nothing for it.
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=INTROSPECTION_THREADS,
                thread_name_prefix="introspect",
            )
        return self._executor

    def close(self) -> None:
        """Release the introspection threads. Wired to application shutdown."""
        executor, self._executor = self._executor, None
        if executor is not None:
            # Not waited on: shutdown must not block on a lookup that is itself
            # blocked on an unresponsive licensing API.
            executor.shutdown(wait=False, cancel_futures=True)

    def _fetch(self, presented_token: str) -> ServerCredential | None:
        payload = json.dumps({"token": presented_token}).encode("utf-8")
        request = urllib.request.Request(
            self.introspection_url,
            data=payload,
            method="POST",
            headers={
                "content-type": "application/json",
                "accept": "application/json",
                # Proves the caller is the processor, not anything else that can
                # reach the loopback port. Without it this endpoint is an oracle
                # for testing stolen tokens.
                "authorization": f"Bearer {self.service_token}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code in (401, 403, 404):
                return None
            LOGGER.error("introspection_failed status=%s", error.code)
            raise IntrospectionUnavailable from error
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            LOGGER.error("introspection_unreachable error=%s", type(error).__name__)
            raise IntrospectionUnavailable from error

        if not isinstance(body, dict) or not body.get("active"):
            return None
        return _credential_from(body, presented_token)


class IntrospectionUnavailable(Exception):
    """The licensing API could not answer. Distinct from answering "no".

    Public because the distinction has to survive as far as the HTTP handler:
    "no" is ``401 unauthorized`` and belongs to the customer, "could not ask" is
    ``503 license_check_unavailable`` and belongs to us. Collapsing the two is
    what made a licensing API restart look like a rejected bearer token on every
    cloud server at once.
    """


def _credential_from(body: dict, presented_token: str) -> ServerCredential | None:
    """Build a credential from the API's answer, refusing anything malformed.

    Every field is validated rather than trusted. The licensing API is ours, but
    a processor that accepts whatever shape arrives is one API bug away from
    granting unlimited rate or every language.
    """
    from datetime import datetime

    # Checked here rather than only at the call site. The function that builds a
    # credential should be the one that refuses to build one, otherwise a second
    # caller added later inherits a gap nobody sees.
    if not body.get("active"):
        return None

    server_id = body.get("server_id")
    if not isinstance(server_id, str) or not server_id:
        LOGGER.error("introspection_malformed field=server_id")
        return None

    # Optional, and validated the way every other optional field here is:
    # absent means "not told", present and malformed means the whole credential
    # is refused. Absent has to be tolerated because the API grew this field
    # after the processor did, and a processor deployed first must not refuse
    # every paying customer until the other side catches up. What it costs is a
    # usage entry the API cannot attribute, which is the situation before this
    # field existed at all, not a regression.
    tenant_key = body.get("tenant_key")
    if tenant_key is not None and (not isinstance(tenant_key, str) or not tenant_key):
        LOGGER.error("introspection_malformed field=tenant_key")
        return None

    plan = body.get("plan", "cloud")
    if not isinstance(plan, str) or not plan:
        return None

    languages = None
    raw_languages = body.get("languages")
    if raw_languages is not None:
        if not isinstance(raw_languages, list) or not all(
            isinstance(code, str) for code in raw_languages
        ):
            LOGGER.error("introspection_malformed field=languages")
            return None
        languages = frozenset(raw_languages) if raw_languages else None

    rate_limit = body.get("rate_limit_per_minute")
    if rate_limit is not None and (not isinstance(rate_limit, int) or rate_limit <= 0):
        LOGGER.error("introspection_malformed field=rate_limit_per_minute")
        return None

    expires_at = None
    raw_expires = body.get("expires")
    if raw_expires is not None:
        if not isinstance(raw_expires, str):
            return None
        try:
            expires_at = datetime.fromisoformat(raw_expires.replace("Z", "+00:00"))
        except ValueError:
            LOGGER.error("introspection_malformed field=expires")
            return None
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=UTC)

    return ServerCredential(
        server_id=server_id,
        token=presented_token,
        tenant_key=tenant_key,
        plan=plan,
        languages=languages,
        rate_limit_per_minute=rate_limit,
        expires_at=expires_at,
    )
