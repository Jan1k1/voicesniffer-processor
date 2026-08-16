from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

SERVER_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]{1,64}\Z")
LANGUAGE_CODE_PATTERN = re.compile(r"[a-z]{2,8}\Z")
PLAN_PATTERN = re.compile(r"[a-z0-9-]{1,32}\Z")


@dataclass(frozen=True)
class ServerCredential:
    """One paired server, and where its limits come from.

    Two paths produce one of these, and the difference matters.

    **Self-hosted.** The customer writes ``tokens.json`` themselves and the
    limits in it are their own. Nothing mints anything. An earlier version of
    this docstring claimed the token was "minted by our licensing infrastructure
    from the customer's license", which was never true for this path and had
    reached the published documentation before anyone checked.

    **Cloud.** The token is minted per licence by the licensing API, and
    ``plan``/``languages``/limits mirror that licence, so enforcement here really
    is enforcement of the licence. See ``cloud_auth``.

    Either way a client cannot widen its own access: the credential is built on
    this side from a source the client does not control.

    ``tenant_key`` exists because ``server_id`` does not identify anybody.
    Operators pick it, it defaults to the Minecraft server's own name, and two
    unrelated customers both calling one ``survival-1`` is the expected case
    rather than a collision worth warning about. The licensing API therefore
    keys usage on ``(tenant_key, server_id)`` and refuses to guess without it.
    Only the cloud path has one: it comes back from introspection. A self-hosted
    credential leaves it ``None``, which is honest, because ``tokens.json``
    belongs to one customer and there is no second tenant to tell it apart
    from."""

    server_id: str
    token: str = field(repr=False)
    tenant_key: str | None = None  # None = self-hosted, nothing to attribute to
    plan: str = "selfhost"
    languages: frozenset[str] | None = None  # None = every installed rule pack
    rate_limit_per_minute: int | None = None  # None = unlimited
    expires_at: datetime.datetime | None = None  # None = never


@dataclass(frozen=True)
class ProcessorSettings:
    credentials_by_digest: Mapping[bytes, ServerCredential] = field(repr=False)
    tokens_file: Path
    bind_host: str
    bind_port: int
    workers: int
    max_body_bytes: int
    max_frame_bytes: int
    max_frames: int
    max_samples_48k: int
    # Set only on a cloud processor. Empty means self-hosted: tokens.json is the
    # whole truth and an unknown token is simply unauthorised.
    introspection_url: str = ""
    introspection_token: str = field(default="", repr=False)
    # Override for the usage endpoint. Normally left empty and derived from the
    # introspection URL, which the two being siblings makes safe. See
    # ``usage_reporter.usage_url_from``. It reuses ``introspection_token``: one
    # service token, two internal endpoints, as the contract has it.
    usage_url: str = ""
    # Same again for the flagged-utterance feed, derived the same way from the
    # same sibling path. It is a separate setting from ``usage_url`` and never a
    # fallback for it: the two feeds carry different things and a deployment that
    # aimed one at the other's endpoint would be posting transcripts at a route
    # whose schema refuses them, or worse, the reverse.
    events_url: str = ""

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> ProcessorSettings:
        values = os.environ if environment is None else environment
        tokens_file_value = values.get("VOICESNIFFER_TOKENS_FILE", "").strip()
        if not tokens_file_value:
            raise ValueError("VOICESNIFFER_TOKENS_FILE is required")

        tokens_file = Path(tokens_file_value)
        credentials_by_digest = _load_credentials(tokens_file)
        bind_host = values.get("VOICESNIFFER_BIND_HOST", "127.0.0.1").strip()
        if not bind_host:
            raise ValueError("VOICESNIFFER_BIND_HOST must not be blank")

        duration_ms = _bounded_integer(
            values,
            "VOICESNIFFER_MAX_DURATION_MS",
            default=8_000,
            minimum=500,
            maximum=8_000,
        )
        return cls(
            credentials_by_digest=credentials_by_digest,
            tokens_file=tokens_file,
            bind_host=bind_host,
            bind_port=_bounded_integer(
                values,
                "VOICESNIFFER_BIND_PORT",
                default=8_080,
                minimum=1,
                maximum=65_535,
            ),
            workers=_bounded_integer(
                values,
                "VOICESNIFFER_WORKERS",
                default=1,
                minimum=1,
                maximum=64,
            ),
            max_body_bytes=_bounded_integer(
                values,
                "VOICESNIFFER_MAX_BODY_BYTES",
                default=1_048_576,
                minimum=1_024,
                maximum=1_048_576,
            ),
            max_frame_bytes=_bounded_integer(
                values,
                "VOICESNIFFER_MAX_FRAME_BYTES",
                default=4_000,
                minimum=1,
                maximum=4_000,
            ),
            max_frames=_bounded_integer(
                values,
                "VOICESNIFFER_MAX_FRAMES",
                default=3_200,
                minimum=1,
                maximum=3_200,
            ),
            max_samples_48k=duration_ms * 48,
            introspection_url=values.get("VOICESNIFFER_INTROSPECTION_URL", "").strip(),
            introspection_token=_secret(values, "VOICESNIFFER_INTROSPECTION_TOKEN"),
            usage_url=values.get("VOICESNIFFER_USAGE_URL", "").strip(),
            events_url=values.get("VOICESNIFFER_EVENTS_URL", "").strip(),
        )

    def server_for_token(self, presented_token: str) -> str | None:
        credential = self.credential_for_token(presented_token)
        return None if credential is None else credential.server_id

    def credential_for_token(self, presented_token: str) -> ServerCredential | None:
        credential = self.credentials_by_digest.get(_token_digest(presented_token))
        if credential is None:
            return None
        if not secrets.compare_digest(credential.token, presented_token):
            return None
        return credential


def _secret(values: Mapping[str, str], name: str) -> str:
    """A secret from ``NAME_FILE`` if present, else from ``NAME``.

    The file form is what Docker secrets give you, and it keeps the value out of
    the container's environment where ``docker inspect`` and any crash dump would
    show it. The plain form stays for local runs and tests.
    """
    path = values.get(f"{name}_FILE", "").strip()
    if path:
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except OSError as error:
            raise ValueError(f"{name}_FILE is set but unreadable: {error}") from error
    return values.get(name, "").strip()

def _load_credentials(tokens_file: Path) -> Mapping[bytes, ServerCredential]:
    # Each failure gets its own message. The single "must contain a readable JSON
    # object" wrapped all of them, and the common case in practice is a
    # permission error -- a root-owned 0600 tokens file crash-loops the container
    # while reporting what looks like a syntax problem. That cost a debugging
    # session and got written up as a deployment gotcha instead of being fixed.
    try:
        raw = tokens_file.read_text(encoding="utf-8")
    except PermissionError as exception:
        raise ValueError(
            f"VOICESNIFFER_TOKENS_FILE is not readable by uid {os.getuid()}: {tokens_file}. "
            f"The container runs as 10001; chown or relax the mode on the host file."
            if hasattr(os, "getuid")
            else f"VOICESNIFFER_TOKENS_FILE is not readable: {tokens_file}"
        ) from exception
    except FileNotFoundError as exception:
        raise ValueError(f"VOICESNIFFER_TOKENS_FILE does not exist: {tokens_file}") from exception
    except IsADirectoryError as exception:
        raise ValueError(f"VOICESNIFFER_TOKENS_FILE is a directory: {tokens_file}") from exception
    except OSError as exception:
        raise ValueError(
            f"VOICESNIFFER_TOKENS_FILE could not be read: {tokens_file} ({exception})"
        ) from exception
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exception:
        raise ValueError(
            f"VOICESNIFFER_TOKENS_FILE is not valid JSON: {tokens_file} "
            f"(line {exception.lineno}, column {exception.colno})"
        ) from exception

    if not isinstance(parsed, dict) or not parsed:
        raise ValueError("VOICESNIFFER_TOKENS_FILE must contain at least one server token")

    credentials_by_digest: dict[bytes, ServerCredential] = {}
    for server_id, entry in parsed.items():
        if not isinstance(server_id, str) or SERVER_ID_PATTERN.fullmatch(server_id) is None:
            raise ValueError("VOICESNIFFER_TOKENS_FILE contains an invalid server ID")
        credential = _parse_credential(server_id, entry)
        token_digest = _token_digest(credential.token)
        if token_digest in credentials_by_digest:
            raise ValueError("VOICESNIFFER_TOKENS_FILE contains duplicate tokens")
        credentials_by_digest[token_digest] = credential

    return MappingProxyType(credentials_by_digest)


def _parse_credential(server_id: str, entry: object) -> ServerCredential:
    # Legacy shape: "server-id": "token". Full access, no limits.
    if isinstance(entry, str):
        if not entry:
            raise ValueError("VOICESNIFFER_TOKENS_FILE contains a blank or invalid token")
        return ServerCredential(server_id, entry)

    if not isinstance(entry, dict):
        raise ValueError("VOICESNIFFER_TOKENS_FILE entries must be a token or an object")
    unknown = set(entry) - {"token", "plan", "languages", "rate-limit-per-minute", "expires"}
    if unknown:
        raise ValueError(f"VOICESNIFFER_TOKENS_FILE entry has unknown keys: {sorted(unknown)}")

    token = entry.get("token")
    if not isinstance(token, str) or not token:
        raise ValueError("VOICESNIFFER_TOKENS_FILE contains a blank or invalid token")

    plan = entry.get("plan", "selfhost")
    if not isinstance(plan, str) or PLAN_PATTERN.fullmatch(plan) is None:
        raise ValueError("VOICESNIFFER_TOKENS_FILE contains an invalid plan")

    languages: frozenset[str] | None = None
    raw_languages = entry.get("languages")
    if raw_languages is not None:
        if (
            not isinstance(raw_languages, list)
            or not raw_languages
            or not all(
                isinstance(code, str) and LANGUAGE_CODE_PATTERN.fullmatch(code)
                for code in raw_languages
            )
        ):
            raise ValueError("VOICESNIFFER_TOKENS_FILE contains an invalid languages list")
        languages = frozenset(raw_languages)

    rate_limit = entry.get("rate-limit-per-minute")
    if rate_limit is not None and (
        not isinstance(rate_limit, int)
        or isinstance(rate_limit, bool)
        or rate_limit < 1
        or rate_limit > 100_000
    ):
        raise ValueError("VOICESNIFFER_TOKENS_FILE contains an invalid rate limit")

    expires_at: datetime.datetime | None = None
    raw_expires = entry.get("expires")
    if raw_expires is not None:
        if not isinstance(raw_expires, str):
            raise ValueError("VOICESNIFFER_TOKENS_FILE contains an invalid expiry")
        try:
            expires_at = datetime.datetime.fromisoformat(raw_expires)
        except ValueError as exception:
            raise ValueError("VOICESNIFFER_TOKENS_FILE contains an invalid expiry") from exception
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=datetime.UTC)

    return ServerCredential(
        server_id,
        token,
        plan=plan,
        languages=languages,
        rate_limit_per_minute=rate_limit,
        expires_at=expires_at,
    )


def _token_digest(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def _bounded_integer(
    values: Mapping[str, str],
    name: str,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    raw_value = values.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exception:
        raise ValueError(f"{name} must be an integer") from exception
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value
