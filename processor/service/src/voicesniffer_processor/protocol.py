from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from voicesniffer_processor.opus import (
    OpusPacketError,
    OpusPacketInfo,
    inspect_opus_packet,
)


class ProtocolError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class ProtocolLimits:
    max_body_bytes: int
    max_frame_bytes: int
    max_frames: int
    max_samples_48k: int


@dataclass(frozen=True)
class OpusEnvelope:
    frames: tuple[bytes, ...]
    total_samples_48k: int
    preroll_samples_48k: int


def parse_opus_envelope(
    body: bytes,
    preroll_samples_48k: int,
    limits: ProtocolLimits,
    inspect_packet: Callable[[bytes], OpusPacketInfo] = inspect_opus_packet,
) -> OpusEnvelope:
    if not body:
        raise ProtocolError("empty_body")
    if len(body) > limits.max_body_bytes:
        raise ProtocolError("body_too_large")
    if preroll_samples_48k < 0 or preroll_samples_48k % 3 != 0:
        raise ProtocolError("invalid_preroll")

    frames: list[bytes] = []
    frame_samples: list[int] = []
    total_samples = 0
    offset = 0
    while offset < len(body):
        if len(body) - offset < 2:
            raise ProtocolError("truncated_frame_length")
        frame_length = int.from_bytes(body[offset : offset + 2], byteorder="big")
        offset += 2
        if frame_length == 0:
            raise ProtocolError("empty_frame")
        if frame_length > limits.max_frame_bytes:
            raise ProtocolError("frame_too_large")
        if len(frames) >= limits.max_frames:
            raise ProtocolError("too_many_frames")
        frame_end = offset + frame_length
        if frame_end > len(body):
            raise ProtocolError("truncated_frame")

        frame = body[offset:frame_end]
        offset = frame_end
        try:
            packet_info = inspect_packet(frame)
        except OpusPacketError as exception:
            raise ProtocolError(exception.code) from exception
        if packet_info.channels != 1:
            raise ProtocolError("stereo_not_supported")

        total_samples += packet_info.samples_48k
        if total_samples > limits.max_samples_48k:
            raise ProtocolError("duration_too_long")
        frames.append(frame)
        frame_samples.append(packet_info.samples_48k)

    if preroll_samples_48k > frame_samples[0]:
        raise ProtocolError("invalid_preroll")
    return OpusEnvelope(
        frames=tuple(frames),
        total_samples_48k=total_samples,
        preroll_samples_48k=preroll_samples_48k,
    )
