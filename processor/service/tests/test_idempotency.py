import asyncio
from collections import OrderedDict

import pytest

from voicesniffer_processor.idempotency import (
    IdempotencyCache,
    IdempotencyCapacityError,
    RequestIdReuseError,
)


def test_reuses_completed_value_for_same_digest() -> None:
    async def scenario() -> None:
        cache = IdempotencyCache(max_entries=10, ttl_seconds=120)
        calls = 0

        async def produce() -> str:
            nonlocal calls
            calls += 1
            return "verdict"

        first = await cache.execute("server", "request", b"digest", produce)
        second = await cache.execute("server", "request", b"digest", produce)

        assert first == "verdict"
        assert second == "verdict"
        assert calls == 1

    asyncio.run(scenario())


def test_rejects_same_request_id_with_changed_digest() -> None:
    async def scenario() -> None:
        cache = IdempotencyCache(max_entries=10, ttl_seconds=120)

        async def produce() -> str:
            return "verdict"

        await cache.execute("server", "request", b"first", produce)
        with pytest.raises(RequestIdReuseError):
            await cache.execute("server", "request", b"second", produce)

    asyncio.run(scenario())


def test_coalesces_concurrent_duplicates() -> None:
    async def scenario() -> None:
        cache = IdempotencyCache(max_entries=10, ttl_seconds=120)
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = 0

        async def produce() -> str:
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return "verdict"

        first = asyncio.create_task(cache.execute("server", "request", b"digest", produce))
        await entered.wait()
        second = asyncio.create_task(cache.execute("server", "request", b"digest", produce))
        await asyncio.sleep(0)
        release.set()

        assert await asyncio.gather(first, second) == ["verdict", "verdict"]
        assert calls == 1

    asyncio.run(scenario())


def test_expired_value_runs_operation_again() -> None:
    async def scenario() -> None:
        now = [10.0]
        cache = IdempotencyCache(
            max_entries=10,
            ttl_seconds=5,
            clock=lambda: now[0],
        )
        calls = 0

        async def produce() -> int:
            nonlocal calls
            calls += 1
            return calls

        assert await cache.execute("server", "request", b"digest", produce) == 1
        now[0] = 16.0
        assert await cache.execute("server", "request", b"digest", produce) == 2

    asyncio.run(scenario())


def test_capacity_evicts_oldest_completed_value() -> None:
    async def scenario() -> None:
        cache = IdempotencyCache(max_entries=2, ttl_seconds=120)
        calls: dict[str, int] = {}

        async def execute(request_id: str) -> int:
            async def produce() -> int:
                calls[request_id] = calls.get(request_id, 0) + 1
                return calls[request_id]

            return await cache.execute("server", request_id, request_id.encode(), produce)

        assert await execute("one") == 1
        assert await execute("two") == 1
        assert await execute("three") == 1
        assert await execute("one") == 2

    asyncio.run(scenario())


def test_failed_operation_is_not_cached() -> None:
    async def scenario() -> None:
        cache = IdempotencyCache(max_entries=10, ttl_seconds=120)
        calls = 0

        async def produce() -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("failed")
            return "recovered"

        with pytest.raises(RuntimeError, match="failed"):
            await cache.execute("server", "request", b"digest", produce)
        assert await cache.execute("server", "request", b"digest", produce) == "recovered"

    asyncio.run(scenario())


def test_active_capacity_rejects_new_owner_but_allows_duplicate() -> None:
    async def scenario() -> None:
        cache = IdempotencyCache(max_entries=10, max_active_entries=1, ttl_seconds=120)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def produce() -> str:
            entered.set()
            await release.wait()
            return "verdict"

        first = asyncio.create_task(cache.execute("server", "one", b"digest", produce))
        await entered.wait()
        duplicate = asyncio.create_task(cache.execute("server", "one", b"digest", produce))
        with pytest.raises(IdempotencyCapacityError):
            await cache.execute("server", "two", b"other", produce)
        release.set()

        assert await asyncio.gather(first, duplicate) == ["verdict", "verdict"]

    asyncio.run(scenario())


def test_cache_maintenance_does_not_scan_all_entries() -> None:
    class NoScanOrderedDict(OrderedDict):
        def items(self):
            raise AssertionError("cache maintenance must not scan all entries")

        def values(self):
            raise AssertionError("cache maintenance must not scan all entries")

    async def scenario() -> None:
        cache = IdempotencyCache(max_entries=10, ttl_seconds=120)
        cache._entries = NoScanOrderedDict()

        async def produce() -> str:
            return "verdict"

        assert await cache.execute("server", "request", b"digest", produce) == "verdict"

    asyncio.run(scenario())


def test_two_tenants_sharing_a_server_name_never_share_a_verdict() -> None:
    """The cache key used to be `(server_id, request_id)`, and `server_id` is a
    name the customer types. Two customers both choosing "survival-1" is not an
    edge case, it is written into the cloud contract as expected.

    What crosses an idempotency key is a ModerationVerdict carrying `transcript`
    and `matched_text`, so a shared key is one customer's player's speech
    reaching another customer's staff. The rate limiter and the usage counters
    were scoped to the tenant for the same reason; this was missed in that pass
    and found by an adversarial review that reproduced it.
    """
    import asyncio

    async def scenario() -> tuple[str, str]:
        cache = IdempotencyCache(max_entries=10, max_active_entries=4, ttl_seconds=60)
        request_id = "shared-request-id"
        digest = b"identical-audio"

        async def speech_of(tenant: str):
            async def produce() -> str:
                return f"{tenant} player said this"
            return produce

        first = await cache.execute(
            "tenant-a|survival-1", request_id, digest, await speech_of("tenant A")
        )
        second = await cache.execute(
            "tenant-b|survival-1", request_id, digest, await speech_of("tenant B")
        )
        return first, second

    first, second = asyncio.run(scenario())

    assert first == "tenant A player said this"
    assert second == "tenant B player said this", (
        "tenant B was served tenant A's verdict: the cache key is not tenant scoped"
    )
