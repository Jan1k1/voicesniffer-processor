"""The transport under both reporting feeds, and what it says when it fails.

Both feeds used to catch everything and log the exception class. That is the
same word -- ``HTTPError`` -- for a refused payload, a wrong service token and a
licensing API that is having a bad afternoon, and the difference between those
is the whole of what somebody debugging this needs. Worse, it is also the
difference between "put the batch back" and "putting the batch back retries it
forever and blocks the queue behind it".

So these tests pin two things: that a failure carries a status and a cause name,
and that the retryable/not decision follows the status rather than the caller.
"""

import io
import json
import urllib.error
import urllib.request

import pytest

from voicesniffer_processor.internal_api import (
    MAX_DETAIL_CHARS,
    InternalApiError,
    describe_failure,
    is_retryable,
    post_json,
)

URL = "http://licensing.internal:8080/api/internal/usage"


def test_a_delivered_report_returns_what_the_api_said(monkeypatch) -> None:
    """``ignored`` is the failure that arrives as a success, so it comes back."""
    monkeypatch.setattr(urllib.request, "urlopen", answering({"accepted": 2, "ignored": 1}))

    assert post_json(URL, service_token="s", payload={}, timeout_seconds=1) == {
        "accepted": 2,
        "ignored": 1,
    }


def test_a_body_that_is_not_json_is_not_a_failure(monkeypatch) -> None:
    # A 202 with an empty body is a perfectly good delivery. Nothing here
    # depends on the answer being parseable.
    monkeypatch.setattr(urllib.request, "urlopen", answering_raw(b""))

    assert post_json(URL, service_token="s", payload={}, timeout_seconds=1) == {}


def test_the_service_token_and_content_type_are_on_every_request(monkeypatch) -> None:
    sent: list[urllib.request.Request] = []
    monkeypatch.setattr(urllib.request, "urlopen", answering({}, capture=sent))

    post_json(URL, service_token="service-secret", payload={"a": 1}, timeout_seconds=1)

    assert sent[0].get_header("Authorization") == "Bearer service-secret"
    assert sent[0].get_header("Content-type") == "application/json"
    assert json.loads(sent[0].data.decode("utf-8")) == {"a": 1}


@pytest.mark.parametrize(
    ("status", "reason", "retryable"),
    [
        (400, "payload_refused", False),
        (401, "service_token_refused", False),
        (403, "service_token_refused", False),
        (404, "endpoint_not_found", False),
        (413, "batch_too_large", False),
        (422, "payload_refused", False),
        (429, "rate_limited", True),
        (500, "api_error", True),
        (502, "api_error", True),
        (503, "api_error", True),
    ],
)
def test_a_status_becomes_a_cause_and_a_retry_decision(
    monkeypatch, status: int, reason: str, retryable: bool
) -> None:
    """The one table this module exists for.

    A 5xx will answer properly later, so the batch waits. A 400 will refuse the
    same bytes forever, so retrying it is an infinite loop that also stops
    everything queued behind it from ever being delivered.
    """
    monkeypatch.setattr(urllib.request, "urlopen", refusing(status, {"message": "Nope."}))

    with pytest.raises(InternalApiError) as raised:
        post_json(URL, service_token="s", payload={}, timeout_seconds=1)

    assert (raised.value.status, raised.value.reason) == (status, reason)
    assert raised.value.retryable is retryable
    assert is_retryable(raised.value) is retryable


def test_the_api_s_own_sentence_is_kept_as_the_detail(monkeypatch) -> None:
    """It is the useful half of a 400, and the internal routes never echo the
    payload, which is what makes keeping it safe on the usage feed."""
    monkeypatch.setattr(
        urllib.request, "urlopen", refusing(400, {"message": "Invalid usage payload."})
    )

    with pytest.raises(InternalApiError) as raised:
        post_json(URL, service_token="s", payload={}, timeout_seconds=1)

    assert raised.value.detail == "Invalid usage payload."


def test_a_detail_is_bounded_and_fits_on_one_line(monkeypatch) -> None:
    # Nothing unexpected on the far end may fill a log file or wrap a line.
    monkeypatch.setattr(urllib.request, "urlopen", refusing(400, {"message": "x\ny " * 400}))

    with pytest.raises(InternalApiError) as raised:
        post_json(URL, service_token="s", payload={}, timeout_seconds=1)

    assert len(raised.value.detail) <= MAX_DETAIL_CHARS + 3
    assert "\n" not in raised.value.detail


def test_a_refusal_names_the_cause_without_the_body(monkeypatch) -> None:
    """``reason`` is derived from the status alone.

    The event feed logs this and never ``detail``, because its payload was a
    transcript and the answer came back off the same wire.
    """
    monkeypatch.setattr(urllib.request, "urlopen", refusing(400, {"message": "he said the phrase"}))

    with pytest.raises(InternalApiError) as raised:
        post_json(URL, service_token="s", payload={}, timeout_seconds=1)

    assert raised.value.reason == "payload_refused"
    assert "phrase" not in raised.value.reason


def test_an_unreachable_api_is_retryable_and_named(monkeypatch) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", raising(urllib.error.URLError("refused")))

    with pytest.raises(InternalApiError) as raised:
        post_json(URL, service_token="s", payload={}, timeout_seconds=1)

    assert (raised.value.status, raised.value.reason, raised.value.retryable) == (
        None,
        "unreachable",
        True,
    )


def test_a_timeout_is_retryable_and_says_how_long_it_waited(monkeypatch) -> None:
    monkeypatch.setattr(urllib.request, "urlopen", raising(TimeoutError()))

    with pytest.raises(InternalApiError) as raised:
        post_json(URL, service_token="s", payload={}, timeout_seconds=2.5)

    assert raised.value.reason == "timeout"
    assert raised.value.detail == "no answer in 2.5s"
    assert raised.value.retryable is True


def test_an_error_body_that_cannot_be_read_still_names_the_status(monkeypatch) -> None:
    # A connection that died between the header and the body. The status is
    # already known and is the more useful half.
    monkeypatch.setattr(urllib.request, "urlopen", refusing_unreadable(503))

    with pytest.raises(InternalApiError) as raised:
        post_json(URL, service_token="s", payload={}, timeout_seconds=1)

    assert (raised.value.status, raised.value.reason) == (503, "api_error")


def test_anything_unclassified_is_described_rather_than_dropped() -> None:
    """The failure path has to survive exceptions this module never raised.

    A test replacing the socket, a payload that will not serialise, a bug here:
    each gets an honest name instead of a missing log line.
    """
    assert describe_failure(ValueError("circular reference"))[:2] == (None, "unexpected")
    assert describe_failure(urllib.error.URLError("down"))[:2] == (None, "unreachable")
    assert describe_failure(ConnectionResetError("peer went away"))[:2] == (None, "unreachable")


def test_an_unclassified_failure_is_retried_rather_than_discarded() -> None:
    # Guessing "permanent" would throw away a moderation record on the strength
    # of an exception nobody has seen yet.
    assert is_retryable(ValueError("who knows")) is True


# ---------------------------------------------------------------- helpers


class FakeResponse:
    def __init__(self, raw: bytes) -> None:
        self._raw = raw

    def read(self) -> bytes:
        return self._raw

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def answering(body, *, capture: list | None = None):
    def urlopen(request, timeout=None):
        if capture is not None:
            capture.append(request)
        return FakeResponse(json.dumps(body).encode("utf-8"))

    return urlopen


def answering_raw(raw: bytes):
    def urlopen(request, timeout=None):
        return FakeResponse(raw)

    return urlopen


def refusing(status: int, body):
    def urlopen(request, timeout=None):
        raise urllib.error.HTTPError(
            URL,
            status,
            "Refused",
            {},
            io.BytesIO(json.dumps(body).encode("utf-8")),
        )

    return urlopen


def refusing_unreadable(status: int):
    class Broken(io.BytesIO):
        def read(self, *args):
            raise OSError("connection reset while reading the body")

    def urlopen(request, timeout=None):
        raise urllib.error.HTTPError(URL, status, "Refused", {}, Broken())

    return urlopen


def raising(error: Exception):
    def urlopen(request, timeout=None):
        raise error

    return urlopen
