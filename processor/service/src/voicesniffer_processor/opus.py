from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from voicesniffer_processor.protocol import OpusEnvelope

SamplesPerFrame = Callable[[bytes, int], int]
FrameCount = Callable[[bytes], int]
ChannelCount = Callable[[bytes], int]
# What `decode_opus` returns, mono. Opus carries 48 kHz on the wire and the
# decoder resamples on the way out, because the recognizer wants 16 kHz and
# doing it here is one pass instead of two. Named because it is also the divisor
# that turns a decoded sample count into the seconds of audio a tenant sent.
DECODED_SAMPLE_RATE = 16_000


class OpusPacketError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class OpusDecodeError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class OpusPacketInfo:
    samples_48k: int
    channels: int


def inspect_opus_packet(
    packet: bytes,
    *,
    samples_per_frame: SamplesPerFrame | None = None,
    frame_count: FrameCount | None = None,
    channel_count: ChannelCount | None = None,
) -> OpusPacketInfo:
    if samples_per_frame is None or frame_count is None or channel_count is None:
        samples_per_frame, frame_count, channel_count = _native_packet_functions()

    try:
        samples = samples_per_frame(packet, 48_000) * frame_count(packet)
        channels = channel_count(packet)
    except Exception as exception:
        raise OpusPacketError("invalid_opus_packet") from exception

    if samples < 1 or samples > 5_760 or channels not in (1, 2):
        raise OpusPacketError("invalid_opus_packet")
    return OpusPacketInfo(samples_48k=samples, channels=channels)


def decode_opus(
    envelope: OpusEnvelope,
    *,
    decoder_factory: Callable[[], Any] | None = None,
) -> NDArray[np.float32]:
    if envelope.preroll_samples_48k % 3 != 0:
        raise OpusDecodeError("invalid_preroll")

    factory = _NativeDecoder if decoder_factory is None else decoder_factory
    decoder = None
    try:
        decoder = factory()
        decoded_frames: list[NDArray[np.int16]] = []
        for frame in envelope.frames:
            pcm = decoder.decode(frame, frame_size=1_920)
            if len(pcm) % 2 != 0:
                raise OpusDecodeError("invalid_pcm")
            decoded_frames.append(np.frombuffer(pcm, dtype="<i2"))
        if not decoded_frames:
            raise OpusDecodeError("empty_audio")
        decoded = np.concatenate(decoded_frames)
        trimmed = decoded[envelope.preroll_samples_48k // 3 :]
        if trimmed.size == 0:
            raise OpusDecodeError("empty_audio")
        return (trimmed.astype(np.float32) / np.float32(32_768.0)).astype(
            np.float32,
            copy=False,
        )
    except OpusDecodeError:
        raise
    except Exception as exception:
        raise OpusDecodeError("opus_decode_failed") from exception
    finally:
        if decoder is not None:
            decoder.close()


def _native_packet_functions() -> tuple[SamplesPerFrame, FrameCount, ChannelCount]:
    try:
        from opuslib_next.api.decoder import (
            packet_get_nb_channels,
            packet_get_nb_frames,
            packet_get_samples_per_frame,
        )
    except Exception as exception:
        raise OpusPacketError("opus_unavailable") from exception
    return packet_get_samples_per_frame, packet_get_nb_frames, packet_get_nb_channels


class _NativeDecoder:
    def __init__(self) -> None:
        try:
            from opuslib_next.api import decoder as decoder_api
        except Exception as exception:
            raise OpusDecodeError("opus_unavailable") from exception
        self._api = decoder_api
        self._state = decoder_api.create_state(DECODED_SAMPLE_RATE, 1)

    def decode(self, packet: bytes, frame_size: int) -> bytes:
        return self._api.decode(
            self._state,
            packet,
            len(packet),
            frame_size,
            False,
            channels=1,
        )

    def close(self) -> None:
        if self._state is not None:
            self._api.destroy(self._state)
            self._state = None
