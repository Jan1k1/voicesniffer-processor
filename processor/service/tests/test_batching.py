from __future__ import annotations

import threading
import time
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray

from voicesniffer_processor.batching import BatcherClosedError, MicroBatcher
from voicesniffer_processor.models import TranscriptionResult


def _samples(value: float) -> NDArray:
    return np.full(160, value, dtype=np.float32)


class RecordingBatcher:
    """Stands in for the model. Records the size of every batch it is handed."""

    def __init__(self, delay: float = 0.0) -> None:
        self.batch_sizes: list[int] = []
        self.delay = delay
        self.lock = threading.Lock()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def __call__(
        self,
        batch: Sequence[NDArray],
        languages: Sequence[str],
    ) -> list[TranscriptionResult]:
        with self.lock:
            self.batch_sizes.append(len(batch))
        self.entered.set()
        self.release.wait(5.0)
        if self.delay:
            time.sleep(self.delay)
        return [
            TranscriptionResult(text=f"{float(item[0]):.0f}-{language}", language="en")
            for item, language in zip(batch, languages, strict=True)
        ]


def test_single_request_is_not_delayed_by_the_window() -> None:
    model = RecordingBatcher()
    batcher = MicroBatcher(model, max_batch_size=8, window_seconds=0.5)
    try:
        started = time.monotonic()
        result = batcher(_samples(1.0), "en")
        elapsed = time.monotonic() - started
    finally:
        batcher.close()

    assert result.text == "1-en"
    assert model.batch_sizes == [1]
    # The window is 500 ms; an idle single request must not pay any of it.
    assert elapsed < 0.25


def test_concurrent_requests_are_merged_into_one_batch() -> None:
    model = RecordingBatcher()
    model.release.clear()
    batcher = MicroBatcher(model, max_batch_size=8, window_seconds=0.2)
    results: dict[int, TranscriptionResult] = {}

    def submit(index: int) -> None:
        results[index] = batcher(_samples(float(index)), "en")

    # Hold the first batch inside the model so the rest queue up behind it.
    first = threading.Thread(target=submit, args=(0,))
    first.start()
    assert model.entered.wait(5.0)

    followers = [threading.Thread(target=submit, args=(index,)) for index in range(1, 6)]
    for thread in followers:
        thread.start()
    time.sleep(0.05)
    model.release.set()
    first.join(10)
    for thread in followers:
        thread.join(10)
    batcher.close()

    assert len(results) == 6
    assert model.batch_sizes[0] == 1
    assert max(model.batch_sizes) > 1, "queued utterances must be merged"
    assert sum(model.batch_sizes) == 6
    stats = batcher.stats()
    assert stats.utterances == 6
    assert stats.largest_batch == max(model.batch_sizes)


def test_batch_never_exceeds_the_configured_maximum() -> None:
    model = RecordingBatcher()
    model.release.clear()
    batcher = MicroBatcher(model, max_batch_size=2, window_seconds=0.2)
    threads = [
        threading.Thread(target=lambda index=index: batcher(_samples(float(index)), "en"))
        for index in range(7)
    ]
    for thread in threads:
        thread.start()
    assert model.entered.wait(5.0)
    time.sleep(0.05)
    model.release.set()
    for thread in threads:
        thread.join(10)
    batcher.close()

    assert max(model.batch_sizes) <= 2
    assert sum(model.batch_sizes) == 7


def test_results_are_matched_to_their_own_caller() -> None:
    model = RecordingBatcher()
    model.release.clear()
    batcher = MicroBatcher(model, max_batch_size=8, window_seconds=0.2)
    collected: dict[int, str] = {}
    lock = threading.Lock()

    def submit(index: int) -> None:
        result = batcher(_samples(float(index)), "cs" if index % 2 else "en")
        with lock:
            collected[index] = result.text

    threads = [threading.Thread(target=submit, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    assert model.entered.wait(5.0)
    time.sleep(0.05)
    model.release.set()
    for thread in threads:
        thread.join(10)
    batcher.close()

    for index in range(8):
        assert collected[index] == f"{index}-{'cs' if index % 2 else 'en'}"


def test_model_failure_is_raised_to_every_caller() -> None:
    def explode(batch: Sequence[NDArray], languages: Sequence[str]) -> list[TranscriptionResult]:
        raise ValueError("model exploded")

    batcher = MicroBatcher(explode, max_batch_size=4, window_seconds=0.0)
    try:
        with pytest.raises(ValueError, match="model exploded"):
            batcher(_samples(1.0), "en")
        with pytest.raises(ValueError, match="model exploded"):
            batcher(_samples(2.0), "en")
    finally:
        batcher.close()


def test_mismatched_result_count_is_an_error() -> None:
    def wrong(batch: Sequence[NDArray], languages: Sequence[str]) -> list[TranscriptionResult]:
        return []

    batcher = MicroBatcher(wrong, max_batch_size=4, window_seconds=0.0)
    try:
        with pytest.raises(RuntimeError, match="wrong number of results"):
            batcher(_samples(1.0), "en")
    finally:
        batcher.close()


def test_calls_after_close_are_refused() -> None:
    batcher = MicroBatcher(RecordingBatcher(), max_batch_size=4, window_seconds=0.0)
    batcher.close()
    with pytest.raises(BatcherClosedError):
        batcher(_samples(1.0), "en")


def test_result_timeout_does_not_hang_the_caller() -> None:
    stop = threading.Event()

    def slow(batch: Sequence[NDArray], languages: Sequence[str]) -> list[TranscriptionResult]:
        stop.wait(30)
        return [TranscriptionResult(text="", language="en") for _ in batch]

    batcher = MicroBatcher(
        slow,
        max_batch_size=4,
        window_seconds=0.0,
        result_timeout_seconds=0.2,
    )
    try:
        with pytest.raises(TimeoutError):
            batcher(_samples(1.0), "en")
    finally:
        stop.set()
        batcher.close()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_batch_size": 0}, "max_batch_size"),
        ({"window_seconds": -1.0}, "window_seconds"),
        ({"result_timeout_seconds": 0.0}, "result_timeout_seconds"),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MicroBatcher(RecordingBatcher(), **kwargs)


class LengthRecordingBatcher:
    """Records the *durations* in each batch, which is what padding charges for."""

    def __init__(self) -> None:
        self.batches: list[list[int]] = []
        self.lock = threading.Lock()
        self.gate = threading.Event()
        self.entered = threading.Event()

    def __call__(
        self,
        batch: Sequence[NDArray],
        languages: Sequence[str],
    ) -> list[TranscriptionResult]:
        self.entered.set()
        self.gate.wait(5.0)
        with self.lock:
            self.batches.append([int(item.size) for item in batch])
        return [TranscriptionResult(text=str(item.size), language="en") for item in batch]


def hold_the_decode_thread(
    batcher: MicroBatcher,
    model: LengthRecordingBatcher,
) -> threading.Thread:
    """Park the decode thread inside the model so a pool can build up behind it.

    Without this the first request is dispatched the instant it arrives, and the
    batcher never has more than one batch worth of candidates to choose from --
    which is the situation length grouping cannot help with anyway.
    """
    primer = threading.Thread(target=batcher, args=(_sized(160), "en"), daemon=True)
    primer.start()
    assert model.entered.wait(5.0)
    return primer


def _sized(samples: int) -> NDArray:
    return np.full(samples, 0.1, dtype=np.float32)


def test_batches_group_utterances_of_similar_length() -> None:
    # decode_streams pads to the longest member, so a batch costs what its
    # longest utterance costs. Mixing a monologue with seven short clips charges
    # all eight for the monologue.
    model = LengthRecordingBatcher()
    batcher = MicroBatcher(model, max_batch_size=4, window_seconds=0.0)
    sizes = [16_000, 128_000, 16_320, 129_000, 16_640, 130_000, 16_960, 131_000]
    try:
        primer = hold_the_decode_thread(batcher, model)
        threads = [
            threading.Thread(target=batcher, args=(_sized(size), "en"), daemon=True)
            for size in sizes
        ]
        for thread in threads:
            thread.start()
        time.sleep(0.2)
        model.gate.set()
        for thread in (primer, *threads):
            thread.join(5.0)
    finally:
        batcher.close()

    grouped = [batch for batch in model.batches if len(batch) == 4]
    assert len(grouped) == 2, model.batches
    for batch in grouped:
        assert max(batch) / min(batch) < 2, model.batches


def test_the_longest_waiting_request_is_never_starved() -> None:
    # Sorting by length alone would defer a long utterance for as long as short
    # ones keep arriving. The oldest request must always be in the batch.
    model = LengthRecordingBatcher()
    batcher = MicroBatcher(model, max_batch_size=2, window_seconds=0.0)
    try:
        primer = hold_the_decode_thread(batcher, model)
        # The long one arrives first, so it is the oldest and must go out first
        # even though every other candidate is a better length match.
        sizes = (400_000, 16_000, 16_100, 16_200, 16_300, 16_400)
        threads = [
            threading.Thread(target=batcher, args=(_sized(size), "en"), daemon=True)
            for size in sizes
        ]
        for thread in threads:
            thread.start()
            time.sleep(0.02)
        model.gate.set()
        for thread in (primer, *threads):
            thread.join(5.0)
    finally:
        batcher.close()

    dispatched = [batch for batch in model.batches if len(batch) == 2]
    assert dispatched, model.batches
    assert 400_000 in dispatched[0], model.batches


def test_shutdown_settles_requests_already_drained_from_the_queue() -> None:
    model = LengthRecordingBatcher()
    batcher = MicroBatcher(model, max_batch_size=8, window_seconds=0.0)
    errors: list[BaseException] = []

    def call() -> None:
        try:
            batcher(_sized(16_000), "en")
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=call, daemon=True) for _ in range(3)]
    for thread in threads:
        thread.start()
    time.sleep(0.1)
    batcher.close()
    model.gate.set()
    for thread in threads:
        thread.join(5.0)

    assert not [error for error in errors if isinstance(error, TimeoutError)]
