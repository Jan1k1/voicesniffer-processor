"""The cloud credential path, which is the boundary between a paying customer
and everybody else.

Self-hosted processors never reach this code: tokens.json is the whole truth and
``CloudCredentialResolver`` is not even constructed. Everything here is about the
cloud node, where a token is minted per licence and has to stop working the
moment that licence does.
"""

import asyncio
import datetime
import json
import threading
import urllib.error
from concurrent.futures import ThreadPoolExecutor

import pytest

from voicesniffer_processor.cloud_auth import (
    MAX_NEGATIVE_CACHE_ENTRIES,
    MAX_POSITIVE_CACHE_ENTRIES,
    CloudCredentialResolver,
    IntrospectionUnavailable,
    _credential_from,
)
from voicesniffer_processor.settings import ProcessorSettings, _token_digest


class FakeClock:
    def __init__(self) -> None:
        self.value = 1_000.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def resolver(clock: FakeClock, answers: list) -> tuple[CloudCredentialResolver, list]:
    """A resolver whose network call is a scripted list, and a call log."""
    calls: list[str] = []

    made = CloudCredentialResolver(
        introspection_url="http://127.0.0.1:18080/api/internal/introspect",
        service_token="service-secret",
        now=clock,
    )

    def fetch(presented_token: str):
        calls.append(presented_token)
        answer = answers.pop(0) if answers else None
        if isinstance(answer, Exception):
            raise answer
        return answer

    made._fetch = fetch  # type: ignore[method-assign]
    return made, calls


def active(**overrides):
    body = {
        "active": True,
        "server_id": "srv-1",
        "plan": "cloud",
        "languages": ["en"],
        "rate_limit_per_minute": 600,
    }
    body.update(overrides)
    return _credential_from(body, "tok-live")


def test_a_valid_token_resolves_and_is_then_served_from_cache() -> None:
    clock = FakeClock()
    made, calls = resolver(clock, [active()])

    first = asyncio.run(made.resolve("tok-live"))
    second = asyncio.run(made.resolve("tok-live"))

    assert first is not None and first.server_id == "srv-1"
    assert second is not None and second.server_id == "srv-1"
    assert calls == ["tok-live"], "the second request must not hit the licensing API"


def test_a_revoked_token_stops_working_within_the_cache_ttl() -> None:
    """The whole reason this is a lookup rather than a signature check.

    A signed token is valid until it expires; this one dies as soon as the cache
    entry does, which is why the TTL is seconds rather than hours.
    """
    clock = FakeClock()
    made, calls = resolver(clock, [active(), None])

    assert asyncio.run(made.resolve("tok-live")) is not None
    clock.advance(made.ttl_seconds + 0.1)
    assert asyncio.run(made.resolve("tok-live")) is None
    assert len(calls) == 2


def test_an_unknown_token_is_refused_and_the_refusal_is_cached() -> None:
    # Without negative caching, one guessed token is an unlimited load generator
    # pointed at the licensing API.
    clock = FakeClock()
    made, calls = resolver(clock, [None])

    for _ in range(5):
        assert asyncio.run(made.resolve("tok-guess")) is None

    assert calls == ["tok-guess"]


def test_an_outage_refuses_without_poisoning_the_cache() -> None:
    """A failure to answer is not an answer.

    It must not grant access, and it must not be remembered as a refusal either:
    a paying customer whose lookup happened during a restart should work again on
    the next request, not in a minute.
    """
    clock = FakeClock()
    made, calls = resolver(clock, [IntrospectionUnavailable(), active()])

    with pytest.raises(IntrospectionUnavailable):
        asyncio.run(made.resolve("tok-live"))
    assert asyncio.run(made.resolve("tok-live")) is not None
    assert len(calls) == 2


def test_an_outage_is_raised_rather_than_returned_as_no_such_token() -> None:
    """The distinction the endpoints turn into 503 instead of 401.

    Collapsing "could not ask" into the same ``None`` as "no such token" meant a
    licensing API restart told every paying cloud server that its bearer token
    had been rejected. The plugin does not retry a 401, reports it to the
    operator as ``http_401``, and then runs voice chat unmoderated. A refusal
    still returns ``None``; only the outage raises.
    """
    clock = FakeClock()
    refuses, _ = resolver(clock, [None])
    assert asyncio.run(refuses.resolve("tok-guess")) is None

    outage, _ = resolver(clock, [IntrospectionUnavailable()])
    with pytest.raises(IntrospectionUnavailable):
        asyncio.run(outage.resolve("tok-live"))


def test_an_outage_still_admits_nobody() -> None:
    """The fail-open question, pinned.

    Reporting an outage honestly is not the same as riding one out. A token the
    resolver has never validated gets nothing during an outage, and a token
    whose positive entry has expired gets nothing either: the entry is gone and
    the lookup that would refresh it cannot run. Changing that is a decision
    about revocation latency, not a bug fix, so it is written down here.
    """
    clock = FakeClock()
    made, _ = resolver(clock, [active(), IntrospectionUnavailable()])

    assert asyncio.run(made.resolve("tok-live")) is not None
    clock.advance(31.0)  # past DEFAULT_TTL_SECONDS
    with pytest.raises(IntrospectionUnavailable):
        asyncio.run(made.resolve("tok-live"))
    assert made._positive == {}, "an expired entry must not survive the outage"


def test_concurrent_first_requests_make_one_lookup() -> None:
    # A server starting up presents the same token from every worker at once.
    clock = FakeClock()
    made, calls = resolver(clock, [active()])

    async def storm():
        return await asyncio.gather(*(made.resolve("tok-live") for _ in range(20)))

    results = asyncio.run(storm())

    assert all(result is not None for result in results)
    assert calls == ["tok-live"], f"{len(calls)} lookups for one token"


def test_the_cache_is_bounded() -> None:
    clock = FakeClock()
    made, _ = resolver(clock, [])
    for index in range(6_000):
        made._store(f"negative-{index}".encode(), None)
        made._store(f"positive-{index}".encode(), active())

    assert len(made._negative) <= MAX_NEGATIVE_CACHE_ENTRIES
    assert len(made._positive) <= MAX_POSITIVE_CACHE_ENTRIES


def test_a_flood_of_unknown_tokens_does_not_evict_a_paying_customer() -> None:
    """The reason positives and negatives have separate budgets.

    ``/v1/moderate`` resolves before it authorises anything, so an
    unauthenticated caller decides what this cache is asked to hold. One shared
    budget evicted by soonest expiry made a positive the eviction candidate
    every single time, because positives live 30 seconds and negatives 60, and
    4,096 requests carrying made-up bearer tokens flushed every paying tenant
    out of it. Each of their utterances then cost a POST to the licensing API
    and an indexed read, sustained, against the box that also runs the website,
    the dashboard and licence validation. Cloud is free with no rate cap by
    decision, so there is nothing else standing between that and the door.

    Five times the old break point here, on purpose: the point is not that the
    number moved, it is that no number works any more.

    Asserting on the lookup count and not on the returned credential, because
    eviction does not change the answer. The evicted customer still resolves,
    by paying for the round trip this cache exists to avoid, and that round trip
    is the whole attack.
    """
    clock = FakeClock()
    made, _ = resolver(clock, [])
    lookups: list[str] = []

    def fetch(presented_token: str):
        lookups.append(presented_token)
        return active() if presented_token == "tok-paying" else None

    made._fetch = fetch  # type: ignore[method-assign]

    async def flood():
        assert await made.resolve("tok-paying") is not None
        for index in range(20_000):
            await made.resolve(f"garbage-{index}")
        return await made.resolve("tok-paying")

    again = asyncio.run(flood())

    assert again is not None
    assert lookups.count("tok-paying") == 1, (
        f"the paying customer was introspected {lookups.count('tok-paying')} times: "
        "20,000 unknown tokens flushed it out of the cache"
    )
    assert len(made._negative) <= MAX_NEGATIVE_CACHE_ENTRIES, "the flood is still bounded"


def test_an_outage_does_not_leak_a_lock_for_every_token_presented() -> None:
    """``_locks`` used to be pruned only by ``_store``.

    The ``IntrospectionUnavailable`` path leaves before ``_store`` runs, so
    every distinct token presented during a licensing API outage left a lock
    object behind for good: 5,000 tokens, 5,000 locks, with the cache itself
    still correctly bounded. An outage is when the processor is already degraded
    and is the worst possible moment to start growing without limit against a
    3 GB container.
    """
    clock = FakeClock()
    made, _ = resolver(clock, [])
    made._fetch = lambda _token: (_ for _ in ()).throw(IntrospectionUnavailable())  # type: ignore[method-assign]

    async def outage():
        for index in range(5_000):
            with pytest.raises(IntrospectionUnavailable):
                await made.resolve(f"tok-{index}")

    asyncio.run(outage())

    assert made._locks == {}, f"{len(made._locks)} locks retained after the outage"


def test_a_lock_is_released_when_the_answer_is_a_refusal_too() -> None:
    # The outage path was the unbounded one, but nothing should be left behind
    # on any path. `_locks` holds in-flight lookups, not a history of tokens.
    clock = FakeClock()
    made, _ = resolver(clock, [active(), None])

    asyncio.run(made.resolve("tok-live"))
    asyncio.run(made.resolve("tok-guess"))

    assert made._locks == {}


def test_a_revoked_token_leaves_no_stale_entry_in_the_other_table() -> None:
    """Two tables, one answer per token.

    Splitting the cache means a digest could sit in both halves at once, and a
    stale positive found first would keep a revoked token working for the whole
    negative TTL. That is the failure mode the split could have introduced, so
    it is the one worth a test.
    """
    clock = FakeClock()
    made, _ = resolver(clock, [active(), None])

    assert asyncio.run(made.resolve("tok-live")) is not None
    clock.advance(made.ttl_seconds + 0.1)
    assert asyncio.run(made.resolve("tok-live")) is None

    digest = _token_digest("tok-live")
    assert digest not in made._positive
    assert digest in made._negative
    # And it stays refused for the negative TTL rather than resurfacing.
    assert asyncio.run(made.resolve("tok-live")) is None


def test_a_slow_introspection_does_not_starve_the_decode_and_stt_pool() -> None:
    """``_fetch`` blocks a thread for up to ``timeout_seconds``, which is 2.

    Under ``asyncio.to_thread`` that block landed in the event loop's single
    default executor, which is the same pool ``_in_thread`` uses for decode and
    speech-to-text. Nothing authenticates before ``resolve`` runs, so the number
    of concurrent lookups is chosen by whoever is calling. Measured against a
    pool of eight threads, the size a 4 vCPU box gets, 128 concurrent
    introspections running to the full timeout pushed the time for a
    transcription to so much as start from 0.7 ms to 31.8 seconds.

    A pool of one here is that same exhaustion with the arithmetic taken out.
    """
    clock = FakeClock()
    made, _ = resolver(clock, [])
    holding = threading.Event()
    release = threading.Event()

    def slow_fetch(_presented_token: str):
        holding.set()
        release.wait(10)
        return None

    made._fetch = slow_fetch  # type: ignore[method-assign]

    async def run() -> str:
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        lookup = asyncio.create_task(made.resolve("tok-slow"))
        try:
            for _ in range(1_000):
                if holding.is_set():
                    break
                await asyncio.sleep(0.01)
            assert holding.is_set(), "the introspection never reached a thread"
            # The work a paying customer is waiting on. It must not be queued
            # behind an introspection that nobody had to authenticate to start.
            return await asyncio.wait_for(asyncio.to_thread(lambda: "transcribed"), timeout=3.0)
        finally:
            release.set()
            await lookup

    try:
        assert asyncio.run(run()) == "transcribed"
    finally:
        made.close()


def test_the_introspection_executor_is_not_built_until_it_is_needed() -> None:
    # A self-hosted processor never constructs a resolver at all, and most of
    # this suite never resolves. Neither should pay for eight idle threads.
    made = CloudCredentialResolver(introspection_url="http://x/y", service_token="s")

    assert made._executor is None


@pytest.mark.parametrize(
    "body",
    [
        {"active": False, "server_id": "srv-1"},
        {"active": True},
        {"active": True, "server_id": ""},
        {"active": True, "server_id": "srv-1", "languages": "en"},
        {"active": True, "server_id": "srv-1", "languages": [1, 2]},
        {"active": True, "server_id": "srv-1", "rate_limit_per_minute": 0},
        {"active": True, "server_id": "srv-1", "rate_limit_per_minute": -5},
        {"active": True, "server_id": "srv-1", "rate_limit_per_minute": "lots"},
        {"active": True, "server_id": "srv-1", "expires": "not-a-date"},
        {"active": True, "server_id": "srv-1", "plan": ""},
        {"active": True, "server_id": "srv-1", "tenant_key": ""},
        {"active": True, "server_id": "srv-1", "tenant_key": 42},
        {"active": True, "server_id": "srv-1", "tenant_key": ["lic_1"]},
    ],
)
def test_a_malformed_answer_grants_nothing(body: dict) -> None:
    """The licensing API is ours, which is not a reason to trust its output.

    A processor that accepts whatever arrives is one API bug away from handing
    out unlimited rate or every language.
    """
    assert _credential_from(body, "tok") is None


def test_entitlements_come_across_intact() -> None:
    credential = _credential_from(
        {
            "active": True,
            "server_id": "srv-9",
            "plan": "cloud-pro",
            "languages": ["en", "cs"],
            "rate_limit_per_minute": 1_200,
            "expires": "2027-01-01T00:00:00Z",
        },
        "tok",
    )

    assert credential is not None
    assert credential.server_id == "srv-9"
    assert credential.plan == "cloud-pro"
    assert credential.languages == frozenset({"en", "cs"})
    assert credential.rate_limit_per_minute == 1_200
    assert credential.expires_at == datetime.datetime(
        2027, 1, 1, tzinfo=datetime.UTC
    )


def test_the_tenant_key_comes_across_so_usage_can_be_attributed() -> None:
    """server_id identifies a server to its operator, and nobody to us.

    Operators pick it and it defaults to the Minecraft server's own name, so two
    unrelated customers both running a `survival-1` is the ordinary case. The
    licensing API keys usage on the pair, and this is where the other half of
    the pair enters the processor.
    """
    credential = _credential_from(
        {"active": True, "server_id": "survival-1", "tenant_key": "lic_9f3"},
        "tok",
    )

    assert credential is not None
    assert credential.tenant_key == "lic_9f3"


def test_a_missing_tenant_key_is_tolerated_rather_than_refused() -> None:
    """Deliberate, and the one place this field is not treated as required.

    The API grew this field after the processor did. If an absent one refused
    the credential, deploying the two in the wrong order would take every paying
    cloud customer offline. What it costs instead is a usage entry the API
    cannot attribute, which is exactly the situation before the field existed.
    """
    credential = _credential_from({"active": True, "server_id": "survival-1"}, "tok")

    assert credential is not None
    assert credential.tenant_key is None


def test_a_self_hosted_credential_has_no_tenant_key(tmp_path) -> None:
    # tokens.json belongs to one customer, so there is no second tenant to tell
    # it apart from, and `tenant_key` is not a key that file accepts.
    tokens_file = tmp_path / "tokens.json"
    tokens_file.write_text(json.dumps({"survival-1": "local-token"}), encoding="utf-8")
    settings = ProcessorSettings.from_environment({"VOICESNIFFER_TOKENS_FILE": str(tokens_file)})

    credential = settings.credential_for_token("local-token")

    assert credential is not None
    assert credential.tenant_key is None


def test_an_empty_language_list_means_every_pack_not_no_packs() -> None:
    # `None` is "all languages" everywhere else in this codebase, and an empty
    # list arriving from the API must not silently mean "moderate nothing".
    credential = _credential_from(
        {"active": True, "server_id": "srv-1", "languages": []}, "tok"
    )

    assert credential is not None
    assert credential.languages is None


def test_the_service_token_is_sent_so_the_endpoint_is_not_an_oracle(monkeypatch) -> None:
    """Anything that can reach the loopback port could otherwise test stolen
    tokens against the licensing API one at a time."""
    seen: dict = {}

    class FakeResponse:
        def read(self):
            return json.dumps({"active": True, "server_id": "srv-1"}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_urlopen(request, timeout=None):
        seen["authorization"] = request.get_header("Authorization")
        seen["body"] = json.loads(request.data.decode())
        return FakeResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    made = CloudCredentialResolver(
        introspection_url="http://127.0.0.1:18080/x", service_token="service-secret"
    )

    assert made._fetch("tok-live") is not None
    assert seen["authorization"] == "Bearer service-secret"
    assert seen["body"] == {"token": "tok-live"}


@pytest.mark.parametrize("status", [401, 403, 404])
def test_a_definite_no_from_the_api_is_a_refusal(monkeypatch, status: int) -> None:
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, status, "no", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    made = CloudCredentialResolver(introspection_url="http://x/y", service_token="s")

    assert made._fetch("tok") is None


@pytest.mark.parametrize("status", [500, 502, 503])
def test_a_server_error_is_an_outage_not_a_refusal(monkeypatch, status: int) -> None:
    # The difference matters: a refusal is cached, an outage is not.
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, status, "boom", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    made = CloudCredentialResolver(introspection_url="http://x/y", service_token="s")

    with pytest.raises(IntrospectionUnavailable):
        made._fetch("tok")
