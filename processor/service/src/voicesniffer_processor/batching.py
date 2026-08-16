"""Micro-batching in front of the transcriber.

The processor answers one HTTP request per utterance, and every request lands in
its own worker thread via ``asyncio.to_thread``. Handed to the model one at a
time, a GPU spends most of its life idle between 1-4 s clips. This collects the
utterances that are in flight at the same instant and decodes them as one batch.

Batching is *opportunistic*, never unconditional:

* whatever is already queued is taken for free, with no wait at all;
* the collection window is only paid when the drain found evidence of
  concurrency (more than one utterance waiting). A lone request on an idle
  processor is handed straight to the model and pays nothing.

**The window defaults to zero, on measurement.** The intuition that a 20-50 ms
collection window is needed to form batches turned out to be wrong for this
workload: requests arrive fast enough that the free drain already fills batches,
and the wait is pure added latency on the critical path. Measured end-to-end on
VM 104 (Parakeet fp32 on CUDA, 16 workers, concurrency 8):

    window 25 ms   19.5 utt/s   p95 504 ms
    window  0 ms   22.3 utt/s   p95 399 ms

so the window cost 12% of throughput and 100 ms of tail to buy nothing. It is
kept configurable because a slower device, or a model whose per-utterance cost
is large enough that requests genuinely trickle, could still want one.

Free-drain batching is worth having on its own merits: against no batching at
all it measured +14% throughput and **-33% p95** under load, because one batched
GPU submission does not contend the way N concurrent single-stream inferences do.

**Batches are grouped by duration**, because ``decode_streams`` pads every
stream in a batch out to the longest one. The whole batch therefore costs what
its longest member costs, and one long utterance is charged to everybody in it.
Measured on the P2000 with Parakeet fp32 at batch 8:

    eight short clips                    28.0 utt/s   batch 269 ms
    seven short clips + one 4 s clip     19.7 utt/s   batch 407 ms
    seven short clips + one 8 s clip     12.1 utt/s   batch 659 ms

A single 8 s monologue -- which the wire contract allows -- more than halves the
throughput of the seven people it is batched with. The fix is *not* to truncate
anyone's speech, which would drop moderation content by construction; it is to
put utterances of similar length in the same batch. On the evaluation corpus
that is worth **+13.5% throughput for nothing**: 19.2 utt/s in arrival order
against 21.8 grouped by length, same clips, same total audio.

The public surface is deliberately identical to the plain transcribe callable --
``(samples, language) -> TranscriptionResult`` -- so ``create_app`` and the wire
contract are untouched.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from numpy.typing import NDArray

from voicesniffer_processor.models import TranscriptionResult

LOGGER = logging.getLogger("voicesniffer.processor.batching")

BatchTranscribe = Callable[
    [Sequence[NDArray], Sequence[str]],
    Sequence[TranscriptionResult],
]

DEFAULT_MAX_BATCH_SIZE = 8
DEFAULT_WINDOW_SECONDS = 0.0
DEFAULT_RESULT_TIMEOUT_SECONDS = 20.0


class BatcherClosedError(RuntimeError):
    pass


@dataclass(slots=True)
class _Pending:
    samples: NDArray
    language: str
    done: threading.Event = field(default_factory=threading.Event)
    result: TranscriptionResult | None = None
    error: BaseException | None = None

    def settle(self, result: TranscriptionResult | None, error: BaseException | None) -> None:
        self.result = result
        self.error = error
        self.done.set()


@dataclass(frozen=True, slots=True)
class BatchStats:
    batches: int
    utterances: int
    largest_batch: int

    @property
    def mean_batch_size(self) -> float:
        return self.utterances / self.batches if self.batches else 0.0


class MicroBatcher:
    def __init__(
        self,
        transcribe_batch: BatchTranscribe,
        *,
        max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        result_timeout_seconds: float = DEFAULT_RESULT_TIMEOUT_SECONDS,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if window_seconds < 0:
            raise ValueError("window_seconds must not be negative")
        if result_timeout_seconds <= 0:
            raise ValueError("result_timeout_seconds must be positive")
        self._transcribe_batch = transcribe_batch
        self._max_batch_size = max_batch_size
        self._window_seconds = window_seconds
        self._result_timeout_seconds = result_timeout_seconds
        self._queue: queue.Queue[_Pending | None] = queue.Queue()
        self._closed = threading.Event()
        self._lock = threading.Lock()
        self._batches = 0
        self._utterances = 0
        self._largest_batch = 0
        self._worker = threading.Thread(
            target=self._run,
            name="voicesniffer-microbatch",
            daemon=True,
        )
        self._worker.start()

    @property
    def max_batch_size(self) -> int:
        return self._max_batch_size

    def stats(self) -> BatchStats:
        with self._lock:
            return BatchStats(self._batches, self._utterances, self._largest_batch)

    def __call__(self, samples: NDArray, language: str) -> TranscriptionResult:
        if self._closed.is_set():
            raise BatcherClosedError("micro-batcher is closed")
        pending = _Pending(samples, language)
        self._queue.put(pending)
        if not pending.done.wait(self._result_timeout_seconds):
            raise TimeoutError("transcription batch did not complete in time")
        if pending.error is not None:
            raise pending.error
        assert pending.result is not None
        return pending.result

    def close(self, timeout: float = 5.0) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(None)
        self._worker.join(timeout)

    def _run(self) -> None:
        # Anything drained but not dispatched stays here in arrival order, so it
        # is reconsidered next round alongside whatever has arrived since.
        waiting: list[_Pending] = []
        while True:
            if not waiting:
                first = self._queue.get()
                if first is None:
                    self._drain_on_close()
                    return
                waiting.append(first)
            closing = self._collect(waiting) is None
            batch, waiting = self._select(waiting)
            self._dispatch(batch)
            if closing:
                # Whatever was already drained still gets decoded -- a shutdown
                # should not turn requests that were in flight into errors -- and
                # only then is the queue failed.
                while waiting:
                    batch, waiting = self._select(waiting)
                    self._dispatch(batch)
                self._drain_on_close()
                return

    def _collect(self, waiting: list[_Pending]) -> list[_Pending] | None:
        """Fill ``waiting`` from the queue. ``None`` means shutdown was seen."""
        # Free drain: anything already queued costs nothing to include, and
        # more candidates means better length grouping, so drain past one batch.
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                return None
            waiting.append(item)
        # Only pay the window when the drain proved there is concurrent load.
        if 1 < len(waiting) < self._max_batch_size and self._window_seconds > 0:
            deadline = time.monotonic() + self._window_seconds
            while len(waiting) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except queue.Empty:
                    break
                if item is None:
                    return None
                waiting.append(item)
        return waiting

    def _select(self, waiting: list[_Pending]) -> tuple[list[_Pending], list[_Pending]]:
        """Take a batch of similar-length utterances that includes the oldest.

        Sorting alone would starve a long utterance whenever short ones keep
        arriving, so the window is anchored on ``waiting[0]`` -- the request that
        has waited longest. That keeps the head of the line moving while still
        putting each utterance next to its nearest neighbours in duration.
        """
        if len(waiting) <= self._max_batch_size:
            return waiting, []
        order = sorted(range(len(waiting)), key=lambda index: waiting[index].samples.size)
        oldest = order.index(0)
        start = min(
            max(0, oldest - self._max_batch_size // 2),
            len(waiting) - self._max_batch_size,
        )
        chosen = set(order[start : start + self._max_batch_size])
        batch = [item for index, item in enumerate(waiting) if index in chosen]
        rest = [item for index, item in enumerate(waiting) if index not in chosen]
        return batch, rest

    def _dispatch(self, batch: list[_Pending]) -> None:
        with self._lock:
            self._batches += 1
            self._utterances += len(batch)
            self._largest_batch = max(self._largest_batch, len(batch))
        try:
            results = self._transcribe_batch(
                [item.samples for item in batch],
                [item.language for item in batch],
            )
        except BaseException as exception:
            for item in batch:
                item.settle(None, exception)
            return
        if len(results) != len(batch):
            error = RuntimeError("transcriber returned the wrong number of results")
            for item in batch:
                item.settle(None, error)
            return
        for item, result in zip(batch, results, strict=True):
            item.settle(result, None)

    def _drain_on_close(self) -> None:
        error = BatcherClosedError("micro-batcher is closed")
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is not None:
                item.settle(None, error)
