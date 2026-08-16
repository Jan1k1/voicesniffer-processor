from __future__ import annotations

import asyncio
import heapq
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar, cast

T = TypeVar("T")
CacheKey = tuple[str, str]


class RequestIdReuseError(ValueError):
    pass


class IdempotencyCapacityError(RuntimeError):
    pass


@dataclass
class _Entry[T]:
    digest: bytes
    event: asyncio.Event
    result: T | None = None
    error: BaseException | None = None
    expires_at: float | None = None


class IdempotencyCache:
    def __init__(
        self,
        *,
        max_entries: int,
        max_active_entries: int | None = None,
        ttl_seconds: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        if max_active_entries is not None and max_active_entries < 1:
            raise ValueError("max_active_entries must be positive")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._max_entries = max_entries
        self._max_active_entries = max_entries if max_active_entries is None else max_active_entries
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries: OrderedDict[CacheKey, _Entry[object]] = OrderedDict()
        self._expiry_queue: list[tuple[float, CacheKey]] = []
        self._active_entries = 0
        self._completed_entries = 0
        self._lock = asyncio.Lock()

    async def execute(
        self,
        scope: str,
        request_id: str,
        digest: bytes,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        # `scope` identifies the tenant AND the server, never the
        # customer-chosen name alone. See the call site in app.py.
        key = (scope, request_id)
        async with self._lock:
            self._remove_expired()
            entry = self._entries.get(key)
            if entry is not None:
                if entry.digest != digest:
                    raise RequestIdReuseError
                owner = False
            else:
                if self._active_entries >= self._max_active_entries:
                    raise IdempotencyCapacityError
                entry = _Entry(digest=digest, event=asyncio.Event())
                self._entries[key] = entry
                self._active_entries += 1
                owner = True

        if not owner:
            await entry.event.wait()
            if entry.error is not None:
                raise entry.error
            return cast(T, entry.result)

        try:
            result = await operation()
        except BaseException as exception:
            async with self._lock:
                entry.error = exception
                entry.event.set()
                if self._entries.get(key) is entry:
                    del self._entries[key]
                    self._active_entries -= 1
            raise

        async with self._lock:
            entry.result = result
            entry.expires_at = self._clock() + self._ttl_seconds
            self._active_entries -= 1
            self._completed_entries += 1
            heapq.heappush(self._expiry_queue, (entry.expires_at, key))
            entry.event.set()
            self._evict_completed()
        return result

    def _remove_expired(self) -> None:
        now = self._clock()
        while self._expiry_queue and self._expiry_queue[0][0] <= now:
            expires_at, key = heapq.heappop(self._expiry_queue)
            entry = self._entries.get(key)
            if entry is None or entry.expires_at != expires_at:
                continue
            del self._entries[key]
            self._completed_entries -= 1

    def _evict_completed(self) -> None:
        while self._completed_entries > self._max_entries:
            expires_at, key = heapq.heappop(self._expiry_queue)
            entry = self._entries.get(key)
            if entry is None or entry.expires_at != expires_at:
                continue
            del self._entries[key]
            self._completed_entries -= 1
