import pytest

from voicesniffer_processor.opus import OpusPacketError, OpusPacketInfo
from voicesniffer_processor.protocol import ProtocolError, ProtocolLimits, parse_opus_envelope


def test_parses_ordered_variable_length_frames() -> None:
    inspected: list[bytes] = []

    def inspect_packet(packet: bytes) -> OpusPacketInfo:
        inspected.append(packet)
        return OpusPacketInfo(samples_48k=960, channels=1)

    envelope = parse_opus_envelope(
        b"\x00\x03abc\x00\x02de",
        preroll_samples_48k=960,
        limits=limits(),
        inspect_packet=inspect_packet,
    )

    assert envelope.frames == (b"abc", b"de")
    assert envelope.total_samples_48k == 1_920
    assert envelope.preroll_samples_48k == 960
    assert inspected == [b"abc", b"de"]


@pytest.mark.parametrize(
    ("body", "expected_code"),
    [
        (b"", "empty_body"),
        (b"\x00", "truncated_frame_length"),
        (b"\x00\x00", "empty_frame"),
        (b"\x00\x03ab", "truncated_frame"),
    ],
)
def test_rejects_malformed_framing(body: bytes, expected_code: str) -> None:
    with pytest.raises(ProtocolError) as raised:
        parse_opus_envelope(body, 0, limits(), mono_packet)

    assert raised.value.code == expected_code
    if body:
        assert body.hex() not in str(raised.value)


def test_rejects_body_above_limit_before_inspection() -> None:
    inspected = False

    def inspect_packet(packet: bytes) -> OpusPacketInfo:
        nonlocal inspected
        inspected = True
        return mono_packet(packet)

    with pytest.raises(ProtocolError) as raised:
        parse_opus_envelope(
            b"\x00\x03abc",
            0,
            limits(max_body_bytes=4),
            inspect_packet,
        )

    assert raised.value.code == "body_too_large"
    assert inspected is False


def test_rejects_frame_above_limit() -> None:
    with pytest.raises(ProtocolError) as raised:
        parse_opus_envelope(
            b"\x00\x03abc",
            0,
            limits(max_frame_bytes=2),
            mono_packet,
        )

    assert raised.value.code == "frame_too_large"


def test_rejects_frame_count_above_limit() -> None:
    body = b"\x00\x01a\x00\x01b\x00\x01c"

    with pytest.raises(ProtocolError) as raised:
        parse_opus_envelope(body, 0, limits(max_frames=2), mono_packet)

    assert raised.value.code == "too_many_frames"


def test_maps_invalid_opus_without_exposing_packet() -> None:
    def reject_packet(packet: bytes) -> OpusPacketInfo:
        raise OpusPacketError("invalid_opus_packet")

    with pytest.raises(ProtocolError) as raised:
        parse_opus_envelope(b"\x00\x06secret", 0, limits(), reject_packet)

    assert raised.value.code == "invalid_opus_packet"
    assert "secret" not in str(raised.value)


def test_rejects_stereo_packet() -> None:
    with pytest.raises(ProtocolError) as raised:
        parse_opus_envelope(
            b"\x00\x01a",
            0,
            limits(),
            lambda _: OpusPacketInfo(samples_48k=960, channels=2),
        )

    assert raised.value.code == "stereo_not_supported"


def test_rejects_total_duration_above_limit() -> None:
    body = b"\x00\x01a\x00\x01b"

    with pytest.raises(ProtocolError) as raised:
        parse_opus_envelope(
            body,
            0,
            limits(max_samples_48k=1_000),
            mono_packet,
        )

    assert raised.value.code == "duration_too_long"


@pytest.mark.parametrize("preroll_samples", [-1, 1, 961])
def test_rejects_preroll_outside_first_packet(preroll_samples: int) -> None:
    with pytest.raises(ProtocolError) as raised:
        parse_opus_envelope(b"\x00\x01a", preroll_samples, limits(), mono_packet)

    assert raised.value.code == "invalid_preroll"


def mono_packet(_: bytes) -> OpusPacketInfo:
    return OpusPacketInfo(samples_48k=960, channels=1)


def limits(
    *,
    max_body_bytes: int = 1_048_576,
    max_frame_bytes: int = 4_000,
    max_frames: int = 3_200,
    max_samples_48k: int = 384_000,
) -> ProtocolLimits:
    return ProtocolLimits(
        max_body_bytes=max_body_bytes,
        max_frame_bytes=max_frame_bytes,
        max_frames=max_frames,
        max_samples_48k=max_samples_48k,
    )
