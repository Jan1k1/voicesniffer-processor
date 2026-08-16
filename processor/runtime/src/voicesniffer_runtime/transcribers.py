from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import numpy as np
import sherpa_onnx

from voicesniffer_runtime.model_store import ModelSpec

PROVIDERS = frozenset({"cpu", "cuda"})

DEFAULT_INPUT_RMS = 0.06
# Below this the clip is silence or room tone. Lifting it to speech level does
# not recover speech, it just hands the decoder amplified noise to hallucinate
# from, so leave it alone.
#
# 0.0005, not the 0.002 first shipped. Three of the 260 English evaluation clips
# sit between the two -- and they are quiet *speech*, exactly the speaker this
# whole pass exists to rescue, not room tone. Refusing to amplify them cost
# 0.25 pp of English WER (6.15% -> 5.90%), one more empty transcript, and 0.4 pp
# of trigger recall (88.4% -> 88.8%), for nothing: 0 false flags either way, and
# dropping the floor further to 0.0001 or to zero changes no result at all, so
# 0.0005 takes the whole benefit while still refusing digital silence.
SILENCE_RMS_FLOOR = 0.0005
MAX_INPUT_GAIN = 8.0

# Both default to off, on measurement. See ``trim_silence`` -- removing silence
# from either end of an utterance costs this model accuracy in proportion to the
# duration removed, and no setting was found that bought throughput for free.
# The knobs stay because a different model may not share the behaviour; the
# defaults stay off because this one does.
DEFAULT_TRIM_HEAD_PAD_MS = -1
DEFAULT_TRIM_TAIL_PAD_MS = -1
TRIM_FRAME_MS = 20
# The threshold is a fraction of the clip's own 90th-percentile frame energy,
# never an absolute level: speakers differ by 20 dB and an absolute floor would
# be either deaf to the quiet ones or useless on the loud ones. The 90th
# percentile rather than the peak so a single keyboard click or plosive cannot
# drag the threshold above the speech it sits next to.
TRIM_REFERENCE_PERCENTILE = 90.0
TRIM_RATIO = 0.15
# Nothing in the clip reaches speech energy at all -- do not guess where it
# starts, hand the whole thing over unchanged.
TRIM_PEAK_FLOOR = 0.004
# A transducer handed a sliver behaves badly, and 320 ms is longer than the
# shortest utterance the plugin will even send (minimum-utterance-ms is 250).
# Never trim below this.
MIN_TRIMMED_MS = 320
TRIM_DISABLED = -1

# Filterbank features are computed on whichever thread calls accept_waveform,
# which in the batching deployment is the one decode thread -- so eight
# utterances' features are computed one after another on a single core while the
# GPU has nothing to do at all. Measured at batch 8 on the P2000 that is 42.7 ms
# of a 424 ms batch (10.1%); four threads take it to 22.6 ms and the batch from
# 19.0 to 19.8 utt/s, p50 and p95 both about 4% lower.
#
# Default 1 -- byte-for-byte the previous behaviour -- because on the CPU
# deployment those cores already belong to the four workers, and there the batch
# path is not used at all.
DEFAULT_FEATURE_THREADS = 1
MAX_FEATURE_THREADS = 8

# How the transducer turns encoder output into tokens. ``modified_beam_search``
# is the default because it is the only lever measured in four rounds that
# improves the *decision* rather than trading one speed figure for another:
# against greedy on the same 520-utterance corpus, eight of fourteen accuracy
# figures improved, six were identical, **none regressed** -- EN WER
# 5.83% -> 5.71%, CS 9.97% -> 9.75%, EN trigger recall 88.8% -> 89.7%, CS
# holding at 88.9%, zero false flags either way, and VRAM unchanged at 3399 MiB.
#
# It costs about 23% of the node's throughput. That is affordable because the
# node it replaces (CPU int8) ships 12.0 utt/s and this still clears it, and
# because a moderation product is sold on decision recall, not on utterances
# per second. ``greedy_search`` remains one environment variable away.
DECODING_METHODS = frozenset({"greedy_search", "modified_beam_search"})
DEFAULT_DECODING_METHOD = "modified_beam_search"
# Two, not four. Four paths takes the same English gain, buys 0.14 pp of Czech
# WER on top, costs 46% of throughput instead of 23%, and -- the part that
# decides it -- wants 4433 MiB of a 5120 MiB card, which is the same headroom
# problem that ruled out batch 16. Two paths does not move VRAM at all.
DEFAULT_MAX_ACTIVE_PATHS = 2
MAX_ACTIVE_PATHS_CEILING = 8


@dataclass(frozen=True, slots=True)
class Transcription:
    text: str
    elapsed_seconds: float
    language: str | None


def resolve_provider(environment: dict[str, str] | None = None) -> str:
    """ONNX Runtime execution provider. Defaults to ``cpu`` so an unconfigured
    deployment behaves exactly as before; ``cuda`` is opt-in and only pays off
    for a model whose weights are not dynamically quantised."""
    values = os.environ if environment is None else environment
    provider = values.get("VOICESNIFFER_PROVIDER", "cpu").strip().lower() or "cpu"
    if provider not in PROVIDERS:
        raise ValueError(f"VOICESNIFFER_PROVIDER must be one of {sorted(PROVIDERS)}")
    return provider


def resolve_input_rms(environment: dict[str, str] | None = None) -> float:
    values = os.environ if environment is None else environment
    raw = values.get("VOICESNIFFER_INPUT_RMS", str(DEFAULT_INPUT_RMS)).strip()
    try:
        target = float(raw or DEFAULT_INPUT_RMS)
    except ValueError as exception:
        raise ValueError("VOICESNIFFER_INPUT_RMS must be a number") from exception
    if target < 0 or target > 0.5:
        raise ValueError("VOICESNIFFER_INPUT_RMS must be between 0 and 0.5")
    return target


def resolve_feature_threads(environment: dict[str, str] | None = None) -> int:
    """How many threads compute filterbank features for one batch. ``1`` is the
    shipped behaviour; higher only pays where a batch is decoded on a device
    that leaves the CPU idle."""
    values = os.environ if environment is None else environment
    raw = values.get("VOICESNIFFER_FEATURE_THREADS", str(DEFAULT_FEATURE_THREADS)).strip()
    try:
        threads = int(raw or DEFAULT_FEATURE_THREADS)
    except ValueError as exception:
        raise ValueError("VOICESNIFFER_FEATURE_THREADS must be an integer") from exception
    if threads < 1 or threads > MAX_FEATURE_THREADS:
        raise ValueError(
            f"VOICESNIFFER_FEATURE_THREADS must be between 1 and {MAX_FEATURE_THREADS}"
        )
    return threads


def resolve_decoding_method(environment: dict[str, str] | None = None) -> str:
    """Transducer search strategy. Defaults to ``modified_beam_search`` -- see
    ``DEFAULT_DECODING_METHOD``. Only the transducer families read this; Whisper
    and Moonshine recognizers have no equivalent knob and ignore it."""
    values = os.environ if environment is None else environment
    raw = values.get("VOICESNIFFER_DECODING_METHOD", DEFAULT_DECODING_METHOD).strip()
    method = raw or DEFAULT_DECODING_METHOD
    if method not in DECODING_METHODS:
        raise ValueError(f"VOICESNIFFER_DECODING_METHOD must be one of {sorted(DECODING_METHODS)}")
    return method


def resolve_max_active_paths(environment: dict[str, str] | None = None) -> int:
    """Beam width for ``modified_beam_search``; ignored by ``greedy_search``."""
    values = os.environ if environment is None else environment
    raw = values.get("VOICESNIFFER_MAX_ACTIVE_PATHS", str(DEFAULT_MAX_ACTIVE_PATHS)).strip()
    try:
        paths = int(raw or DEFAULT_MAX_ACTIVE_PATHS)
    except ValueError as exception:
        raise ValueError("VOICESNIFFER_MAX_ACTIVE_PATHS must be an integer") from exception
    if paths < 1 or paths > MAX_ACTIVE_PATHS_CEILING:
        raise ValueError(
            f"VOICESNIFFER_MAX_ACTIVE_PATHS must be between 1 and {MAX_ACTIVE_PATHS_CEILING}"
        )
    return paths


def _resolve_pad_ms(values: dict[str, str], name: str, default: int) -> int:
    raw = values.get(name, str(default)).strip()
    try:
        pad_ms = int(raw or default)
    except ValueError as exception:
        raise ValueError(f"{name} must be an integer") from exception
    if pad_ms < TRIM_DISABLED or pad_ms > 2_000:
        raise ValueError(f"{name} must be between -1 and 2000")
    return pad_ms


def resolve_trim_pads(environment: dict[str, str] | None = None) -> tuple[int, int]:
    """Guard band kept before the first and after the last voiced frame, in
    milliseconds. ``-1`` leaves that end of the clip alone; ``0`` trims hard
    against the voiced frame itself."""
    values = os.environ if environment is None else environment
    return (
        _resolve_pad_ms(values, "VOICESNIFFER_TRIM_HEAD_PAD_MS", DEFAULT_TRIM_HEAD_PAD_MS),
        _resolve_pad_ms(values, "VOICESNIFFER_TRIM_TAIL_PAD_MS", DEFAULT_TRIM_TAIL_PAD_MS),
    )


def trim_silence(
    samples: np.ndarray,
    sample_rate: int,
    head_pad_ms: int,
    tail_pad_ms: int,
    *,
    ratio: float = TRIM_RATIO,
) -> np.ndarray:
    """Drop the silence at the ends before the model is paid to listen to it.

    **Off by default, and that is the finding.** The reasoning going in was
    sound: transcription cost is proportional to audio *duration*, not content,
    and the plugin's segmenter guarantees silence in every payload because it
    closes an utterance only after ``hangover-ms`` (600) of continuous
    non-speech. On the evaluation corpus the two ends are 46% (EN) / 44% (CS)
    of the audio. It should have been free throughput.

    It is not. Measured on the 520-utterance corpus with this exact function,
    accuracy tracks the *duration removed* and nothing else:

        both ends, 120 ms guard   65% of duration   EN 7.53%  CS 16.16%
        tail only, 240 ms guard   85% of duration   EN 6.21%  CS 11.14%
        tail only, 600 ms guard   93% of duration   EN 6.21%  CS 10.63%
        untouched                100% of duration   EN 6.15%  CS 10.19%

    Two candidate explanations were tested and eliminated. It is not the
    detector cutting into speech: a gentler threshold that removes the same
    duration lands on the same WER, and a 600 ms guard band cannot be clipping
    phonemes. It is not the input gain either: normalising *before* trimming --
    which leaves the surviving speech samples bit-identical to the untouched
    run -- reproduces the loss (EN 7.53%, CS 15.57%). The model wants the
    silence around the speech, most plausibly through NeMo's per-utterance
    feature normalisation, and Czech wants it more.

    So the knobs exist, they are honest, and they are off. Turn them on only
    for a model measured to tolerate it. ``head_pad_ms`` / ``tail_pad_ms``
    below zero leave that end alone entirely.
    """
    if head_pad_ms < TRIM_DISABLED or tail_pad_ms < TRIM_DISABLED:
        raise ValueError("pads must be -1 (off) or a non-negative millisecond count")
    if head_pad_ms < 0 and tail_pad_ms < 0:
        return samples
    frame = sample_rate * TRIM_FRAME_MS // 1000
    minimum = MIN_TRIMMED_MS * sample_rate // 1000
    if frame <= 0 or samples.size <= minimum or samples.size < frame * 3:
        return samples
    usable = (samples.size // frame) * frame
    blocks = samples[:usable].astype(np.float64, copy=False).reshape(-1, frame)
    frame_rms = np.sqrt(np.mean(np.square(blocks), axis=1))
    reference = float(np.percentile(frame_rms, TRIM_REFERENCE_PERCENTILE))
    if reference < TRIM_PEAK_FLOOR:
        return samples
    voiced = np.flatnonzero(frame_rms >= reference * ratio)
    if voiced.size == 0:
        return samples
    start = 0
    if head_pad_ms >= 0:
        start = max(0, int(voiced[0]) * frame - head_pad_ms * sample_rate // 1000)
    stop = samples.size
    if tail_pad_ms >= 0:
        stop = min(
            samples.size,
            (int(voiced[-1]) + 1) * frame + tail_pad_ms * sample_rate // 1000,
        )
    if stop - start < minimum:
        # Grow the window rather than surrender the whole clip: centre the
        # speech we found inside the shortest slice we are willing to decode.
        start = max(0, start - (minimum - (stop - start)) // 2)
        stop = min(samples.size, start + minimum)
        start = max(0, stop - minimum)
    if stop - start >= samples.size:
        return samples
    return np.ascontiguousarray(samples[start:stop])


def normalize_level(samples: np.ndarray, target_rms: float) -> np.ndarray:
    """Bring an utterance to a fixed loudness before it reaches the model.

    Quiet speakers were being dropped outright. Measured on 260 English Common
    Voice utterances pushed through the real Opus path, the shipped int8 model
    returned an *empty transcript* for 52 of them -- a moderation event that
    silently never happens -- and the clips it lost had a median RMS of 0.036
    against 0.056 for the rest. Normalising the level first takes that model
    from 28.11% to 11.04% WER (52 empties down to 7); the fp32 build improves
    too, 7.59% to 5.83% (6 empties down to 1).

    A player who whispers is exactly the player a moderation system must not
    lose, so this is on by default.
    """
    if target_rms <= 0:
        return samples
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    if rms < SILENCE_RMS_FLOOR:
        return samples
    gain = min(target_rms / rms, MAX_INPUT_GAIN)
    scaled = samples * np.float32(gain)
    peak = float(np.max(np.abs(scaled)))
    if peak > 0.99:
        scaled = scaled * np.float32(0.99 / peak)
    return scaled.astype(np.float32, copy=False)


class SherpaTranscriber:
    def __init__(
        self,
        recognizer: Any,
        spec: ModelSpec,
        language: str,
        threads: int,
        load_seconds: float,
        provider: str = "cpu",
        input_rms: float = 0.0,
        trim_pads: tuple[int, int] = (TRIM_DISABLED, TRIM_DISABLED),
        feature_threads: int = DEFAULT_FEATURE_THREADS,
        decoding_method: str = DEFAULT_DECODING_METHOD,
        max_active_paths: int = DEFAULT_MAX_ACTIVE_PATHS,
    ) -> None:
        self.recognizer = recognizer
        self.spec = spec
        self.language = language
        self.threads = threads
        self.load_seconds = load_seconds
        self.provider = provider
        self.input_rms = input_rms
        self.trim_pads = trim_pads
        self.feature_threads = feature_threads
        self.decoding_method = decoding_method
        self.max_active_paths = max_active_paths
        self._feature_pool = (
            ThreadPoolExecutor(
                max_workers=feature_threads, thread_name_prefix="voicesniffer-features"
            )
            if feature_threads > 1
            else None
        )

    def transcribe(self, samples: np.ndarray, sample_rate: int) -> Transcription:
        prepared = self._prepare(samples, sample_rate)
        started = perf_counter_ns()
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, prepared)
        self.recognizer.decode_stream(stream)
        elapsed_seconds = (perf_counter_ns() - started) / 1_000_000_000
        return self._result(stream, elapsed_seconds)

    def transcribe_batch(
        self,
        batch: Sequence[np.ndarray],
        sample_rate: int,
    ) -> list[Transcription]:
        """Decode several utterances in one pass. On CUDA this is the difference
        between a card idling between single 1-4 s clips and one kept fed; on CPU
        it is neutral, so callers can batch unconditionally."""
        if not batch:
            return []
        if len(batch) == 1:
            return [self.transcribe(batch[0], sample_rate)]
        prepared = [self._prepare(samples, sample_rate) for samples in batch]
        started = perf_counter_ns()
        streams = self._build_streams(prepared, sample_rate)
        self.recognizer.decode_streams(streams)
        elapsed_seconds = (perf_counter_ns() - started) / 1_000_000_000
        # Every member waited for the whole batch, so each is charged the whole
        # batch: the figure stays an honest per-request cost.
        return [self._result(stream, elapsed_seconds) for stream in streams]

    def _build_streams(self, prepared: list[np.ndarray], sample_rate: int) -> list[Any]:
        """One stream per utterance, features computed into it.

        This is the only CPU work left in a CUDA batch and it is not small:
        42.7 ms of a 424 ms batch of eight on the P2000, spent on one core with
        the card idle. ``accept_waveform`` releases the GIL, so a small pool
        actually overlaps -- measured 1.72x on four threads of a 4-vCPU box.
        Order is preserved: ``map`` yields in submission order, and a transducer
        batch is positional."""

        def build(samples: np.ndarray) -> Any:
            stream = self.recognizer.create_stream()
            stream.accept_waveform(sample_rate, samples)
            return stream

        if self._feature_pool is None:
            return [build(samples) for samples in prepared]
        return list(self._feature_pool.map(build, prepared))

    def _prepare(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        prepared = np.asarray(samples, dtype=np.float32)
        if sample_rate != 16_000:
            raise ValueError("transcriber input must be 16 kHz")
        if prepared.ndim != 1 or prepared.size == 0 or not np.all(np.isfinite(prepared)):
            raise ValueError("transcriber input must be finite mono audio")
        head_pad_ms, tail_pad_ms = self.trim_pads
        if head_pad_ms >= 0 or tail_pad_ms >= 0:
            # Trim before gain, not after: the target RMS then describes the
            # speech rather than an average of speech and the silence around it,
            # so two clips from the same speaker with different amounts of
            # trailing hangover no longer land at different loudness.
            prepared = trim_silence(prepared, sample_rate, head_pad_ms, tail_pad_ms)
        return normalize_level(prepared, self.input_rms)

    @staticmethod
    def _result(stream: Any, elapsed_seconds: float) -> Transcription:
        return Transcription(
            text=stream.result.text.strip(),
            elapsed_seconds=elapsed_seconds,
            language=getattr(stream.result, "lang", "") or None,
        )


def load_transcriber(
    spec: ModelSpec,
    model_dir: Path,
    language: str,
    threads: int,
    recognizer_type: Any = sherpa_onnx.OfflineRecognizer,
    provider: str | None = None,
    input_rms: float | None = None,
    trim_pads: tuple[int, int] | None = None,
    feature_threads: int | None = None,
    decoding_method: str | None = None,
    max_active_paths: int | None = None,
) -> SherpaTranscriber:
    if not spec.supports_language(language):
        raise ValueError(f"{spec.model_id} does not support {language}")
    if isinstance(threads, bool) or threads <= 0:
        raise ValueError("threads must be positive")
    if provider is None:
        provider = resolve_provider()
    elif provider not in PROVIDERS:
        raise ValueError(f"provider must be one of {sorted(PROVIDERS)}")
    if input_rms is None:
        input_rms = resolve_input_rms()
    elif input_rms < 0 or input_rms > 0.5:
        raise ValueError("input_rms must be between 0 and 0.5")
    if trim_pads is None:
        trim_pads = resolve_trim_pads()
    elif any(pad < TRIM_DISABLED or pad > 2_000 for pad in trim_pads):
        raise ValueError("trim pads must be between -1 and 2000")
    if feature_threads is None:
        feature_threads = resolve_feature_threads()
    elif isinstance(feature_threads, bool) or not 1 <= feature_threads <= MAX_FEATURE_THREADS:
        raise ValueError(f"feature_threads must be between 1 and {MAX_FEATURE_THREADS}")
    if decoding_method is None:
        decoding_method = resolve_decoding_method()
    elif decoding_method not in DECODING_METHODS:
        raise ValueError(f"decoding_method must be one of {sorted(DECODING_METHODS)}")
    if max_active_paths is None:
        max_active_paths = resolve_max_active_paths()
    elif (
        isinstance(max_active_paths, bool) or not 1 <= max_active_paths <= MAX_ACTIVE_PATHS_CEILING
    ):
        raise ValueError(f"max_active_paths must be between 1 and {MAX_ACTIVE_PATHS_CEILING}")
    paths = {role: _model_file(model_dir, relative) for role, relative in spec.files.items()}
    started = perf_counter_ns()
    if spec.family == "moonshine-v2":
        recognizer = recognizer_type.from_moonshine_v2(
            encoder=str(paths["encoder"]),
            decoder=str(paths["decoder"]),
            tokens=str(paths["tokens"]),
            num_threads=threads,
            provider=provider,
        )
    elif spec.family == "nemo-transducer":
        recognizer = recognizer_type.from_transducer(
            encoder=str(paths["encoder"]),
            decoder=str(paths["decoder"]),
            joiner=str(paths["joiner"]),
            tokens=str(paths["tokens"]),
            num_threads=threads,
            model_type="nemo_transducer",
            provider=provider,
            decoding_method=decoding_method,
            max_active_paths=max_active_paths,
        )
    elif spec.family == "whisper":
        recognizer = recognizer_type.from_whisper(
            encoder=str(paths["encoder"]),
            decoder=str(paths["decoder"]),
            tokens=str(paths["tokens"]),
            language=language,
            task="transcribe",
            num_threads=threads,
            provider=provider,
        )
    else:
        raise ValueError(f"unsupported model family: {spec.family}")
    load_seconds = (perf_counter_ns() - started) / 1_000_000_000
    return SherpaTranscriber(
        recognizer,
        spec,
        language,
        threads,
        load_seconds,
        provider,
        input_rms,
        trim_pads,
        feature_threads,
        decoding_method,
        max_active_paths,
    )


def _model_file(model_dir: Path, relative_path: str) -> Path:
    root = model_dir.resolve()
    path = (root / Path(*relative_path.split("/"))).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"missing required model file: {relative_path}")
    return path
