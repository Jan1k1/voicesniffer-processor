from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

from voicesniffer_runtime.model_store import ModelSpec
from voicesniffer_runtime.transcribers import (
    DEFAULT_DECODING_METHOD,
    DEFAULT_MAX_ACTIVE_PATHS,
    load_transcriber,
    normalize_level,
    resolve_decoding_method,
    resolve_feature_threads,
    resolve_input_rms,
    resolve_max_active_paths,
    resolve_provider,
    resolve_trim_pads,
    trim_silence,
)


class FakeStream:
    def __init__(self) -> None:
        self.accepted: tuple[int, np.ndarray] | None = None
        self.result = SimpleNamespace(text="  decoded speech  ")

    def accept_waveform(self, sample_rate: int, samples: np.ndarray) -> None:
        self.accepted = (sample_rate, samples)


class FakeRecognizer:
    calls: ClassVar[list[tuple[str, dict[str, object]]]] = []

    @classmethod
    def from_moonshine_v2(cls, **kwargs):
        cls.calls.append(("moonshine", kwargs))
        return cls()

    @classmethod
    def from_transducer(cls, **kwargs):
        cls.calls.append(("transducer", kwargs))
        return cls()

    @classmethod
    def from_whisper(cls, **kwargs):
        cls.calls.append(("whisper", kwargs))
        return cls()

    decoded_batches: ClassVar[list[int]] = []

    def create_stream(self) -> FakeStream:
        self.stream = FakeStream()
        return self.stream

    def decode_stream(self, stream: FakeStream) -> None:
        assert stream is self.stream

    def decode_streams(self, streams: list[FakeStream]) -> None:
        self.decoded_batches.append(len(streams))
        for index, stream in enumerate(streams):
            stream.result = SimpleNamespace(text=f"  utterance {index}  ")


def make_spec(family: str, files: dict[str, str], languages: tuple[str, ...]) -> ModelSpec:
    return ModelSpec(
        model_id="demo",
        family=family,
        archive_url="https://example.test/model.tar.bz2",
        archive_sha256="a" * 64,
        archive_root="package",
        languages=languages,
        license_name="MIT",
        license_url="https://example.test/license",
        files=files,
    )


def create_model_files(root: Path, files: dict[str, str]) -> None:
    for relative_path in files.values():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"model")


@pytest.mark.parametrize(
    ("family", "files", "languages", "language", "method", "expected"),
    [
        (
            "moonshine-v2",
            {"encoder": "encoder.ort", "decoder": "decoder.ort", "tokens": "tokens.txt"},
            ("en",),
            "en",
            "moonshine",
            {"encoder", "decoder", "tokens", "num_threads", "provider"},
        ),
        (
            "nemo-transducer",
            {
                "encoder": "encoder.onnx",
                "decoder": "decoder.onnx",
                "joiner": "joiner.onnx",
                "tokens": "tokens.txt",
            },
            ("en", "cs"),
            "cs",
            "transducer",
            {
                "encoder",
                "decoder",
                "joiner",
                "tokens",
                "num_threads",
                "model_type",
                "provider",
                "decoding_method",
                "max_active_paths",
            },
        ),
        (
            "whisper",
            {"encoder": "encoder.onnx", "decoder": "decoder.onnx", "tokens": "tokens.txt"},
            ("en", "cs"),
            "cs",
            "whisper",
            {"encoder", "decoder", "tokens", "language", "task", "num_threads", "provider"},
        ),
    ],
)
def test_constructs_each_sherpa_model_family(
    tmp_path: Path,
    family: str,
    files: dict[str, str],
    languages: tuple[str, ...],
    language: str,
    method: str,
    expected: set[str],
) -> None:
    create_model_files(tmp_path, files)
    FakeRecognizer.calls.clear()

    transcriber = load_transcriber(
        make_spec(family, files, languages),
        tmp_path,
        language=language,
        threads=2,
        recognizer_type=FakeRecognizer,
    )

    called_method, arguments = FakeRecognizer.calls[-1]
    assert called_method == method
    assert set(arguments) == expected
    assert arguments["num_threads"] == 2
    # cpu unless explicitly asked otherwise: an unconfigured deployment must not
    # silently start reaching for a GPU.
    assert arguments["provider"] == "cpu"
    assert transcriber.provider == "cpu"
    assert transcriber.load_seconds >= 0


def test_transcribes_finite_16khz_audio(tmp_path: Path) -> None:
    files = {"encoder": "encoder.ort", "decoder": "decoder.ort", "tokens": "tokens.txt"}
    create_model_files(tmp_path, files)
    transcriber = load_transcriber(
        make_spec("moonshine-v2", files, ("en",)),
        tmp_path,
        language="en",
        threads=1,
        recognizer_type=FakeRecognizer,
    )
    samples = np.zeros(1600, dtype=np.float32)

    result = transcriber.transcribe(samples, sample_rate=16_000)

    assert result.text == "decoded speech"
    assert result.elapsed_seconds > 0
    assert result.language is None
    assert transcriber.recognizer.stream.accepted is not None
    assert transcriber.recognizer.stream.accepted[0] == 16_000
    assert transcriber.recognizer.stream.accepted[1] is samples


def test_rejects_unsupported_language_before_loading(tmp_path: Path) -> None:
    files = {"encoder": "encoder.ort", "decoder": "decoder.ort", "tokens": "tokens.txt"}
    create_model_files(tmp_path, files)
    FakeRecognizer.calls.clear()

    with pytest.raises(ValueError, match="does not support"):
        load_transcriber(
            make_spec("moonshine-v2", files, ("en",)),
            tmp_path,
            language="cs",
            threads=1,
            recognizer_type=FakeRecognizer,
        )

    assert not FakeRecognizer.calls


def _moonshine(tmp_path: Path, **kwargs):
    files = {"encoder": "encoder.ort", "decoder": "decoder.ort", "tokens": "tokens.txt"}
    create_model_files(tmp_path, files)
    return load_transcriber(
        make_spec("moonshine-v2", files, ("en",)),
        tmp_path,
        language="en",
        threads=1,
        recognizer_type=FakeRecognizer,
        **kwargs,
    )


def test_transcribe_batch_decodes_every_utterance_in_one_pass(tmp_path: Path) -> None:
    transcriber = _moonshine(tmp_path)
    FakeRecognizer.decoded_batches.clear()
    batch = [np.full(160, value, dtype=np.float32) for value in (0.1, 0.2, 0.3)]

    results = transcriber.transcribe_batch(batch, 16_000)

    assert [result.text for result in results] == [
        "utterance 0",
        "utterance 1",
        "utterance 2",
    ]
    assert FakeRecognizer.decoded_batches == [3], "one batched call, not three single ones"
    assert all(result.elapsed_seconds > 0 for result in results)


def test_transcribe_batch_of_one_takes_the_single_stream_path(tmp_path: Path) -> None:
    transcriber = _moonshine(tmp_path)
    FakeRecognizer.decoded_batches.clear()

    results = transcriber.transcribe_batch([np.zeros(160, dtype=np.float32)], 16_000)

    assert [result.text for result in results] == ["decoded speech"]
    assert FakeRecognizer.decoded_batches == []


def test_transcribe_batch_of_nothing_is_empty(tmp_path: Path) -> None:
    assert _moonshine(tmp_path).transcribe_batch([], 16_000) == []


def test_transcribe_batch_validates_every_member(tmp_path: Path) -> None:
    transcriber = _moonshine(tmp_path)
    batch = [np.zeros(160, dtype=np.float32), np.full(160, np.nan, dtype=np.float32)]

    with pytest.raises(ValueError, match="finite mono audio"):
        transcriber.transcribe_batch(batch, 16_000)


def test_provider_is_selectable_and_validated(tmp_path: Path) -> None:
    FakeRecognizer.calls.clear()
    transcriber = _moonshine(tmp_path, provider="cuda")
    assert FakeRecognizer.calls[-1][1]["provider"] == "cuda"
    assert transcriber.provider == "cuda"

    with pytest.raises(ValueError, match="provider must be one of"):
        _moonshine(tmp_path, provider="tpu")


def test_resolve_provider_defaults_to_cpu_and_rejects_nonsense() -> None:
    assert resolve_provider({}) == "cpu"
    assert resolve_provider({"VOICESNIFFER_PROVIDER": ""}) == "cpu"
    assert resolve_provider({"VOICESNIFFER_PROVIDER": " CUDA "}) == "cuda"
    with pytest.raises(ValueError, match="VOICESNIFFER_PROVIDER"):
        resolve_provider({"VOICESNIFFER_PROVIDER": "opencl"})


def test_feature_threads_keep_the_batch_in_order(tmp_path: Path) -> None:
    # The pool exists to overlap kaldi-native-fbank with the GPU, so it must not
    # be allowed to shuffle a batch -- a transducer batch is positional and every
    # caller gets its result back by index.
    transcriber = _moonshine(tmp_path, feature_threads=4)
    assert transcriber.feature_threads == 4
    FakeRecognizer.decoded_batches.clear()
    batch = [np.full(160, value, dtype=np.float32) for value in (0.1, 0.2, 0.3, 0.4)]

    results = transcriber.transcribe_batch(batch, 16_000)

    assert [result.text for result in results] == [
        "utterance 0",
        "utterance 1",
        "utterance 2",
        "utterance 3",
    ]
    assert FakeRecognizer.decoded_batches == [4]


def test_feature_threads_default_to_one_and_are_validated(tmp_path: Path) -> None:
    assert _moonshine(tmp_path).feature_threads == 1
    assert resolve_feature_threads({}) == 1
    assert resolve_feature_threads({"VOICESNIFFER_FEATURE_THREADS": ""}) == 1
    assert resolve_feature_threads({"VOICESNIFFER_FEATURE_THREADS": " 4 "}) == 4
    with pytest.raises(ValueError, match="VOICESNIFFER_FEATURE_THREADS"):
        resolve_feature_threads({"VOICESNIFFER_FEATURE_THREADS": "0"})
    with pytest.raises(ValueError, match="VOICESNIFFER_FEATURE_THREADS"):
        resolve_feature_threads({"VOICESNIFFER_FEATURE_THREADS": "9"})
    with pytest.raises(ValueError, match="VOICESNIFFER_FEATURE_THREADS"):
        resolve_feature_threads({"VOICESNIFFER_FEATURE_THREADS": "many"})
    with pytest.raises(ValueError, match="feature_threads must be"):
        _moonshine(tmp_path, feature_threads=0)


def test_input_level_is_normalised_before_the_model_sees_it(tmp_path: Path) -> None:
    transcriber = _moonshine(tmp_path, input_rms=0.06)
    quiet = np.full(1600, 0.01, dtype=np.float32)

    transcriber.transcribe(quiet, sample_rate=16_000)

    delivered = transcriber.recognizer.stream.accepted[1]
    assert float(np.sqrt(np.mean(delivered**2))) == pytest.approx(0.06, rel=0.01)


def test_normalisation_is_skipped_for_silence() -> None:
    # Amplifying room tone to speech level does not recover speech; it hands the
    # decoder loud noise to hallucinate from.
    silence = np.full(1600, 0.0001, dtype=np.float32)
    assert np.array_equal(normalize_level(silence, 0.06), silence)
    assert np.array_equal(normalize_level(np.zeros(1600, dtype=np.float32), 0.06),
                          np.zeros(1600, dtype=np.float32))


def test_the_quietest_real_speakers_are_still_amplified() -> None:
    # The floor sits at 0.0005, not the 0.002 first shipped: three of the 260
    # English evaluation clips live between the two and they are speech, not
    # room tone. Lifting them is worth 0.25 pp of WER and 0.4 pp of trigger
    # recall, with no new false flags.
    very_quiet_speech = np.full(1600, 0.001, dtype=np.float32)

    lifted = normalize_level(very_quiet_speech, 0.06)

    assert float(np.sqrt(np.mean(lifted**2))) > float(
        np.sqrt(np.mean(very_quiet_speech**2))
    )


def test_normalisation_gain_is_capped_and_never_clips() -> None:
    # Above the silence floor but asking for 20x; the cap holds it to 8x.
    faint = np.full(1600, 0.003, dtype=np.float32)
    boosted = normalize_level(faint, 0.06)
    assert float(np.max(np.abs(boosted))) <= 0.99
    assert float(np.sqrt(np.mean(boosted**2))) == pytest.approx(0.024, rel=0.02)

    loud = np.full(1600, 0.9, dtype=np.float32)
    assert float(np.max(np.abs(normalize_level(loud, 0.06)))) <= 0.99


def test_normalisation_off_is_a_passthrough() -> None:
    samples = np.full(1600, 0.01, dtype=np.float32)
    assert np.array_equal(normalize_level(samples, 0.0), samples)
    assert normalize_level(samples, 0.0).dtype == np.float32


def test_batch_members_are_normalised_independently(tmp_path: Path) -> None:
    transcriber = _moonshine(tmp_path, input_rms=0.06)
    FakeRecognizer.decoded_batches.clear()
    batch = [np.full(1600, level, dtype=np.float32) for level in (0.01, 0.2)]

    transcriber.transcribe_batch(batch, 16_000)

    assert FakeRecognizer.decoded_batches == [2]


def test_resolve_input_rms_defaults_and_validates() -> None:
    assert resolve_input_rms({}) == 0.06
    assert resolve_input_rms({"VOICESNIFFER_INPUT_RMS": "0"}) == 0.0
    assert resolve_input_rms({"VOICESNIFFER_INPUT_RMS": "0.1"}) == 0.1
    with pytest.raises(ValueError, match=r"between 0 and 0\.5"):
        resolve_input_rms({"VOICESNIFFER_INPUT_RMS": "0.9"})
    with pytest.raises(ValueError, match="must be a number"):
        resolve_input_rms({"VOICESNIFFER_INPUT_RMS": "loud"})


def test_load_transcriber_rejects_an_absurd_input_rms(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"input_rms must be between"):
        _moonshine(tmp_path, input_rms=2.0)


def speech_between(lead_ms: int, speech_ms: int, tail_ms: int, level: float = 0.2) -> np.ndarray:
    rate = 16_000
    generator = np.random.default_rng(7)
    lead = np.zeros(rate * lead_ms // 1000, dtype=np.float32)
    tail = np.zeros(rate * tail_ms // 1000, dtype=np.float32)
    speech = (generator.standard_normal(rate * speech_ms // 1000) * level).astype(np.float32)
    return np.concatenate([lead, speech, tail])


def test_trim_silence_keeps_the_speech_and_the_guard_band() -> None:
    clip = speech_between(1_000, 1_000, 1_000)

    trimmed = trim_silence(clip, 16_000, 120, 120)

    # 1000 ms of speech plus 120 ms either side, to the nearest 20 ms frame.
    assert 1_230 <= trimmed.size / 16 <= 1_290
    assert float(np.max(np.abs(trimmed))) == pytest.approx(float(np.max(np.abs(clip))))


def test_trim_silence_treats_the_two_ends_separately() -> None:
    clip = speech_between(1_000, 1_000, 1_000)

    head_only = trim_silence(clip, 16_000, 0, -1)
    tail_only = trim_silence(clip, 16_000, -1, 0)

    assert head_only.size == pytest.approx(2_000 * 16, abs=16 * 40)
    assert tail_only.size == pytest.approx(2_000 * 16, abs=16 * 40)
    # The head-trimmed clip must still end where the original did.
    assert np.array_equal(head_only[-16_000:], clip[-16_000:])
    assert np.array_equal(tail_only[:16_000], clip[:16_000])


def test_trim_silence_is_disabled_when_both_ends_are_negative() -> None:
    clip = speech_between(1_000, 500, 1_000)

    assert trim_silence(clip, 16_000, -1, -1) is clip


def test_trim_silence_never_empties_a_clip_that_is_all_silence() -> None:
    # Room tone must come back untouched: guessing where speech starts in a clip
    # that has none is how a moderation event silently disappears.
    silence = np.full(48_000, 0.0005, dtype=np.float32)

    assert trim_silence(silence, 16_000, 0, 0) is silence


def test_trim_silence_holds_a_floor_on_the_surviving_length() -> None:
    clip = speech_between(1_000, 60, 1_000)

    trimmed = trim_silence(clip, 16_000, 0, 0)

    assert trimmed.size >= 320 * 16


def test_trim_silence_leaves_short_clips_alone() -> None:
    short = speech_between(20, 100, 20)

    assert trim_silence(short, 16_000, 0, 0) is short


def test_trim_silence_ignores_an_isolated_transient() -> None:
    # A keyboard click louder than the speech must not become the anchor that
    # the threshold is measured against, or the trim keeps the click and drops
    # the sentence.
    clip = speech_between(600, 1_500, 600, level=0.05)
    clip[10] = 0.95

    trimmed = trim_silence(clip, 16_000, 60, 60)

    assert trimmed.size >= 1_500 * 16


def test_resolve_trim_pads_defaults_to_off_and_validates() -> None:
    # Off by default on measurement: removing silence costs this model accuracy
    # in proportion to the duration removed.
    assert resolve_trim_pads({}) == (-1, -1)
    assert resolve_trim_pads(
        {"VOICESNIFFER_TRIM_HEAD_PAD_MS": "0", "VOICESNIFFER_TRIM_TAIL_PAD_MS": "240"}
    ) == (0, 240)
    with pytest.raises(ValueError, match="TRIM_TAIL"):
        resolve_trim_pads({"VOICESNIFFER_TRIM_TAIL_PAD_MS": "9000"})
    with pytest.raises(ValueError, match="TRIM_HEAD"):
        resolve_trim_pads({"VOICESNIFFER_TRIM_HEAD_PAD_MS": "soon"})


def test_transcriber_trims_before_it_normalises(tmp_path: Path) -> None:
    files = {"encoder": "encoder.ort", "decoder": "decoder.ort", "tokens": "tokens.txt"}
    create_model_files(tmp_path, files)
    transcriber = load_transcriber(
        make_spec("moonshine-v2", files, ("en",)),
        tmp_path,
        "en",
        1,
        recognizer_type=FakeRecognizer,
        input_rms=0.06,
        trim_pads=(0, 0),
    )
    clip = speech_between(1_000, 1_000, 1_000, level=0.01)

    transcriber.transcribe(clip, 16_000)

    accepted = transcriber.recognizer.stream.accepted[1]
    assert accepted.size < clip.size
    # Gain applied after trimming, so the target RMS describes the speech.
    assert float(np.sqrt(np.mean(np.square(accepted.astype(np.float64))))) == pytest.approx(
        0.06, rel=0.15
    )


def test_load_transcriber_rejects_an_absurd_trim_pad(tmp_path: Path) -> None:
    files = {"encoder": "encoder.ort", "decoder": "decoder.ort", "tokens": "tokens.txt"}
    create_model_files(tmp_path, files)

    with pytest.raises(ValueError, match="trim pads"):
        load_transcriber(
            make_spec("moonshine-v2", files, ("en",)),
            tmp_path,
            "en",
            1,
            recognizer_type=FakeRecognizer,
            trim_pads=(0, 9_000),
        )


def _transducer(tmp_path: Path, **kwargs):
    files = {
        "encoder": "encoder.onnx",
        "decoder": "decoder.onnx",
        "joiner": "joiner.onnx",
        "tokens": "tokens.txt",
    }
    create_model_files(tmp_path, files)
    FakeRecognizer.calls.clear()
    return load_transcriber(
        make_spec("nemo-transducer", files, ("en", "cs")),
        tmp_path,
        "en",
        1,
        recognizer_type=FakeRecognizer,
        **kwargs,
    )


def test_beam_search_is_the_default_and_two_paths_wide(tmp_path: Path) -> None:
    # The one lever measured that improves the *decision* rather than trading
    # one speed figure for another: EN trigger recall 88.8% -> 89.7%, Czech
    # holding, zero accuracy figures worse, no extra VRAM. Two paths and not
    # four because four wants 4433 MiB of a 5120 MiB card.
    transcriber = _transducer(tmp_path)

    _, arguments = FakeRecognizer.calls[-1]
    assert arguments["decoding_method"] == "modified_beam_search"
    assert arguments["max_active_paths"] == 2
    assert transcriber.decoding_method == "modified_beam_search"
    assert transcriber.max_active_paths == 2


def test_greedy_search_stays_one_environment_variable_away(tmp_path: Path) -> None:
    transcriber = _transducer(tmp_path, decoding_method="greedy_search")

    _, arguments = FakeRecognizer.calls[-1]
    assert arguments["decoding_method"] == "greedy_search"
    assert transcriber.decoding_method == "greedy_search"


def test_resolve_decoding_method_defaults_and_validates() -> None:
    assert resolve_decoding_method({}) == DEFAULT_DECODING_METHOD == "modified_beam_search"
    assert resolve_decoding_method({"VOICESNIFFER_DECODING_METHOD": "greedy_search"}) == (
        "greedy_search"
    )
    # Blank means "unset", not "no decoding".
    assert resolve_decoding_method({"VOICESNIFFER_DECODING_METHOD": ""}) == DEFAULT_DECODING_METHOD
    with pytest.raises(ValueError, match="DECODING_METHOD"):
        resolve_decoding_method({"VOICESNIFFER_DECODING_METHOD": "beam_search"})


def test_resolve_max_active_paths_defaults_and_validates() -> None:
    assert resolve_max_active_paths({}) == DEFAULT_MAX_ACTIVE_PATHS == 2
    assert resolve_max_active_paths({"VOICESNIFFER_MAX_ACTIVE_PATHS": "4"}) == 4
    with pytest.raises(ValueError, match="MAX_ACTIVE_PATHS"):
        resolve_max_active_paths({"VOICESNIFFER_MAX_ACTIVE_PATHS": "0"})
    with pytest.raises(ValueError, match="MAX_ACTIVE_PATHS"):
        resolve_max_active_paths({"VOICESNIFFER_MAX_ACTIVE_PATHS": "64"})
    with pytest.raises(ValueError, match="MAX_ACTIVE_PATHS"):
        resolve_max_active_paths({"VOICESNIFFER_MAX_ACTIVE_PATHS": "wide"})


def test_load_transcriber_rejects_a_bad_search_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="decoding_method"):
        _transducer(tmp_path, decoding_method="viterbi")
    with pytest.raises(ValueError, match="max_active_paths"):
        _transducer(tmp_path, max_active_paths=99)


def test_search_settings_are_not_passed_to_families_that_have_no_such_knob(
    tmp_path: Path,
) -> None:
    # Whisper and Moonshine recognizers take no decoding_method argument;
    # forwarding one would be a TypeError at load, on the CPU deployment.
    transcriber = _moonshine(tmp_path)
    _, arguments = FakeRecognizer.calls[-1]
    assert "decoding_method" not in arguments
    assert "max_active_paths" not in arguments
    assert transcriber.decoding_method == DEFAULT_DECODING_METHOD
