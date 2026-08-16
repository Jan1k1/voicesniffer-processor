import numpy as np
import pytest

from voicesniffer_processor.opus import (
    OpusDecodeError,
    OpusPacketError,
    OpusPacketInfo,
    decode_opus,
    inspect_opus_packet,
)
from voicesniffer_processor.protocol import OpusEnvelope


def test_inspects_packet_duration_and_channels() -> None:
    packet = b"opus"

    packet_info = inspect_opus_packet(
        packet,
        samples_per_frame=lambda received, sample_rate: (
            960 if received == packet and sample_rate == 48_000 else 0
        ),
        frame_count=lambda received: 2 if received == packet else 0,
        channel_count=lambda received: 1 if received == packet else 0,
    )

    assert packet_info == OpusPacketInfo(samples_48k=1_920, channels=1)


@pytest.mark.parametrize(
    ("samples_per_frame", "frame_count", "channel_count"),
    [
        (0, 1, 1),
        (960, 0, 1),
        (960, 1, 0),
        (2_880, 3, 1),
    ],
)
def test_rejects_invalid_packet_metadata(
    samples_per_frame: int,
    frame_count: int,
    channel_count: int,
) -> None:
    with pytest.raises(OpusPacketError, match="invalid_opus_packet"):
        inspect_opus_packet(
            b"packet",
            samples_per_frame=lambda _packet, _rate: samples_per_frame,
            frame_count=lambda _packet: frame_count,
            channel_count=lambda _packet: channel_count,
        )


def test_maps_native_inspection_error() -> None:
    def fail(_packet: bytes, _sample_rate: int) -> int:
        raise RuntimeError("native details must not leak")

    with pytest.raises(OpusPacketError) as raised:
        inspect_opus_packet(
            b"private-packet",
            samples_per_frame=fail,
            frame_count=lambda _packet: 1,
            channel_count=lambda _packet: 1,
        )

    assert raised.value.code == "invalid_opus_packet"
    assert "native details" not in str(raised.value)
    assert "private-packet" not in str(raised.value)


def test_decodes_frames_in_order_and_trims_converted_preroll() -> None:
    decoder = FakeDecoder(
        {
            b"first": np.array([1_000, 2_000], dtype="<i2").tobytes(),
            b"second": np.array([3_000], dtype="<i2").tobytes(),
        }
    )
    envelope = OpusEnvelope(
        frames=(b"first", b"second"),
        total_samples_48k=9,
        preroll_samples_48k=3,
    )

    samples = decode_opus(envelope, decoder_factory=lambda: decoder)

    np.testing.assert_allclose(samples, np.array([2_000, 3_000]) / 32_768.0)
    assert samples.dtype == np.float32
    assert decoder.decoded_frames == [b"first", b"second"]
    assert decoder.closed is True


def test_rejects_preroll_that_cannot_convert_exactly_to_16khz() -> None:
    envelope = OpusEnvelope(
        frames=(b"frame",),
        total_samples_48k=960,
        preroll_samples_48k=1,
    )

    with pytest.raises(OpusDecodeError, match="invalid_preroll"):
        decode_opus(envelope, decoder_factory=lambda: FakeDecoder({}))


def test_rejects_audio_emptied_by_preroll() -> None:
    decoder = FakeDecoder({b"frame": np.array([1_000], dtype="<i2").tobytes()})
    envelope = OpusEnvelope(
        frames=(b"frame",),
        total_samples_48k=3,
        preroll_samples_48k=3,
    )

    with pytest.raises(OpusDecodeError, match="empty_audio"):
        decode_opus(envelope, decoder_factory=lambda: decoder)

    assert decoder.closed is True


def test_closes_decoder_and_redacts_native_failure() -> None:
    decoder = FakeDecoder({b"first": np.array([1_000], dtype="<i2").tobytes()})
    decoder.failure = RuntimeError("private native detail")
    envelope = OpusEnvelope(
        frames=(b"private-frame",),
        total_samples_48k=960,
        preroll_samples_48k=0,
    )

    with pytest.raises(OpusDecodeError) as raised:
        decode_opus(envelope, decoder_factory=lambda: decoder)

    assert raised.value.code == "opus_decode_failed"
    assert "private" not in str(raised.value)
    assert decoder.closed is True


def test_rejects_odd_length_pcm() -> None:
    decoder = FakeDecoder({b"frame": b"\x01"})
    envelope = OpusEnvelope(
        frames=(b"frame",),
        total_samples_48k=960,
        preroll_samples_48k=0,
    )

    with pytest.raises(OpusDecodeError, match="invalid_pcm"):
        decode_opus(envelope, decoder_factory=lambda: decoder)


class FakeDecoder:
    def __init__(self, decoded: dict[bytes, bytes]) -> None:
        self.decoded = decoded
        self.decoded_frames: list[bytes] = []
        self.failure: Exception | None = None
        self.closed = False

    def decode(self, packet: bytes, frame_size: int) -> bytes:
        assert frame_size == 1_920
        self.decoded_frames.append(packet)
        if self.failure is not None:
            raise self.failure
        return self.decoded[packet]

    def close(self) -> None:
        self.closed = True
