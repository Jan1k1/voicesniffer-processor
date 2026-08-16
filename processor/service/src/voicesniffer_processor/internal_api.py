"""One POST to one of the licensing API's internal endpoints, and a name for it
when it fails.

Both reporters next door used to catch every exception and log
``error=<ClassName>``. That is true and almost useless: a refused payload, a
wrong service token, a route that is not there and a licensing API that is down
all arrive as ``HTTPError`` or ``URLError``, and the whole difference between
"redeploy the API" and "fix an environment variable" is the part that was thrown
away. This product has lost days to a feed that reported nothing and said
nothing about why, so the transport now classifies what happened and hands back
something the caller can put in a log line.

Two decisions worth stating.

**Retryable is a property of the failure, not of the caller.** A 5xx, a timeout
and a refused connection all mean "the API could not answer, ask again". A 400
means "these exact bytes are wrong", and asking again with the same bytes is an
infinite loop that also blocks everything queued behind them. The event feed
requeues the first kind and discards the second precisely because of that
difference, so the difference is computed once, here.

**The API's own message is available but is not for everybody.** The internal
routes answer refusals with a fixed sentence and never echo the payload, so
``detail`` is safe on the usage feed. The event feed carries transcripts and its
rule is that nothing it holds reaches a log at any level, so it logs ``reason``,
which is derived from the status code alone and cannot contain a byte that came
back from the wire.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping

# Long enough for the API's own sentence, short enough that nothing unexpected
# fills a log file. The internal routes answer with a `message` of a few words.
MAX_DETAIL_CHARS = 160

# Statuses worth asking about again. Everything else in the 4xx range is a
# refusal of the request itself, which a retry reproduces exactly.
RETRYABLE_STATUSES = frozenset({408, 425, 429})

_REASON_BY_STATUS = {
    400: "payload_refused",
    401: "service_token_refused",
    403: "service_token_refused",
    404: "endpoint_not_found",
    413: "batch_too_large",
    422: "payload_refused",
    429: "rate_limited",
}


class InternalApiError(Exception):
    """A failed call to an internal endpoint, described rather than just typed.

    ``kind`` is the exception class name, kept because it is what the existing
    log lines and their tests say. ``status`` is the HTTP status when there was
    a response at all. ``reason`` is a short name derived from the status, safe
    to log anywhere. ``detail`` is the API's own message, safe to log only where
    the payload was not a transcript.
    """

    def __init__(
        self,
        *,
        kind: str,
        status: int | None,
        detail: str,
        retryable: bool,
    ) -> None:
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind
        self.status = status
        self.detail = detail
        self.retryable = retryable

    @property
    def reason(self) -> str:
        """A cause name built from the status alone, never from the response body."""
        if self.status is None:
            return "unreachable" if self.kind != "TimeoutError" else "timeout"
        named = _REASON_BY_STATUS.get(self.status)
        if named is not None:
            return named
        if self.status >= 500:
            return "api_error"
        return f"http_{self.status}"


def post_json(
    url: str,
    *,
    service_token: str,
    payload: Mapping[str, object],
    timeout_seconds: float,
) -> Mapping[str, object]:
    """POST ``payload`` and return the parsed answer, or raise ``InternalApiError``.

    The answer matters: both internal feeds reply ``{"accepted": n,
    "ignored": n}``, and ``ignored`` is the one failure mode that arrives as a
    success. A report the API accepts and then counts as belonging to nobody is
    exactly the state that leaves a dashboard reading zero while the processor's
    log says everything is fine, so it is returned rather than dropped.

    A body that is not JSON is not an error. Nothing here depends on the answer
    being parseable, and a 202 with an empty body is a perfectly good delivery.
    """
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "content-type": "application/json",
            "accept": "application/json",
            # The same service token the introspection call uses, for the same
            # reason: these endpoints are loopback, and loopback is not a short
            # list of callers on a box that also runs a web stack.
            "authorization": f"Bearer {service_token}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return _parsed(response.read())
    except urllib.error.HTTPError as error:
        raise _from_http_error(error) from error
    except urllib.error.URLError as error:
        raise InternalApiError(
            kind=type(error).__name__,
            status=None,
            detail=_trimmed(str(error.reason)),
            retryable=True,
        ) from error
    except TimeoutError as error:
        raise InternalApiError(
            kind="TimeoutError",
            status=None,
            detail=f"no answer in {timeout_seconds}s",
            retryable=True,
        ) from error
    except OSError as error:
        # A socket that died mid-read rather than mid-connect. The API is having
        # a bad time, which is the retryable kind.
        raise InternalApiError(
            kind=type(error).__name__,
            status=None,
            detail=_trimmed(str(error)),
            retryable=True,
        ) from error


def describe_failure(error: BaseException) -> tuple[int | None, str, str]:
    """``(status, reason, detail)`` for any exception a report can fail with.

    ``post_json`` raises ``InternalApiError`` for everything it can name, but a
    reporter's failure path also has to survive whatever else reaches it -- a
    payload that will not serialise, a test replacing the socket with something
    that raises a plain ``URLError``, a bug in this file. Those get an honest
    "unexpected" rather than a missing log line, which is the whole point of
    this module.
    """
    if isinstance(error, InternalApiError):
        return error.status, error.reason, error.detail
    if isinstance(error, urllib.error.URLError):
        return None, "unreachable", _trimmed(str(error.reason))
    if isinstance(error, TimeoutError):
        return None, "timeout", _trimmed(str(error))
    if isinstance(error, OSError):
        return None, "unreachable", _trimmed(str(error))
    return None, "unexpected", _trimmed(str(error))


def is_retryable(error: BaseException) -> bool:
    """Whether asking again with the same bytes could ever work.

    Anything this module did not classify is treated as retryable. Guessing
    "permanent" would throw away a moderation record on the strength of an
    exception nobody has seen yet; guessing "transient" costs one more attempt.
    """
    return error.retryable if isinstance(error, InternalApiError) else True


def _from_http_error(error: urllib.error.HTTPError) -> InternalApiError:
    status = int(error.code)
    return InternalApiError(
        kind="HTTPError",
        status=status,
        detail=_message_in(error),
        retryable=status >= 500 or status in RETRYABLE_STATUSES,
    )


def _message_in(error: urllib.error.HTTPError) -> str:
    """The API's own sentence, or its status line when there is not one.

    Reading the body of an error response can itself fail, on a connection that
    died between the header and the body. That is not worth a second failure
    path: the status is already known and is the more useful half.
    """
    try:
        parsed = _parsed(error.read())
    except Exception:
        return _trimmed(str(error.reason))
    message = parsed.get("message")
    if isinstance(message, str) and message:
        return _trimmed(message)
    return _trimmed(str(error.reason))


def _parsed(raw: bytes) -> Mapping[str, object]:
    try:
        answer = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return answer if isinstance(answer, dict) else {}


def _trimmed(text: str) -> str:
    """One line, bounded, and quotable in a key=value log line."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= MAX_DETAIL_CHARS:
        return collapsed
    return collapsed[:MAX_DETAIL_CHARS] + "..."
