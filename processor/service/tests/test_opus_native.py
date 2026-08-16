import numpy as np
import pytest

from voicesniffer_processor.opus import decode_opus
from voicesniffer_processor.protocol import ProtocolLimits, parse_opus_envelope


@pytest.mark.native
def test_native_libopus_encode_inspect_decode_round_trip() -> None:
    try:
        import opuslib_next
    except Exception as exception:
        pytest.skip(f"libopus unavailable: {type(exception).__name__}")

    pcm = np.full(960, 2_000, dtype="<i2")
    encoder = opuslib_next.Encoder(48_000, 1, "voip")
    packet = encoder.encode(pcm.tobytes(), 960)
    body = len(packet).to_bytes(2, "big") + packet
    envelope = parse_opus_envelope(
        body,
        preroll_samples_48k=0,
        limits=ProtocolLimits(
            max_body_bytes=1_048_576,
            max_frame_bytes=4_000,
            max_frames=3_200,
            max_samples_48k=384_000,
        ),
    )

    decoded = decode_opus(envelope)

    assert decoded.dtype == np.float32
    assert decoded.shape == (320,)
    assert np.any(decoded != 0)
