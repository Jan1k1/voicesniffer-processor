# Wire protocol

Source of truth: `protocol.py` for the audio envelope, `app.py` for the handler,
`models.py` for the response shapes.

| Method | Path | Auth |
|---|---|---|
| `GET` | `/healthz` | none |
| `POST` | `/v1/moderate` | bearer |
| `GET` | `/v1/usage` | bearer |

That is the whole API.

## Authentication

```
Authorization: Bearer <token>
```

One space, no trailing whitespace. `Bearer` is case-insensitive, the token is
not. Comparison is constant-time against a SHA-256 keyed lookup. An unknown
token is `401 unauthorized`.

The token is the entire authentication. Read [SECURITY.md](../SECURITY.md)
before exposing the port.

## GET /healthz

```
200 {"status": "ok"}
```

Unauthenticated. It answers nothing about authorization.

## POST /v1/moderate

### Request headers

| Header | Required | Value |
|---|---|---|
| `Authorization` | yes | `Bearer <token>` |
| `Content-Type` | yes | `application/vnd.voicesniffer.opus.v1`; parameters after `;` ignored |
| `Accept` | yes | exactly `application/json`; `*/*` is refused |
| `X-Request-Id` | yes | UUID, canonical lowercase hyphenated form only |
| `X-VoiceSniffer-Player-Id` | yes | the speaker's UUID |
| `X-VoiceSniffer-Language` | yes | `auto`, or a comma-separated list of codes |
| `X-VoiceSniffer-Preroll-Samples` | yes | 48 kHz samples to discard from the front |
| `X-VoiceSniffer-Partial` | no | `1` for an early window, `0` or absent otherwise |

`X-Request-Id` must satisfy `str(uuid.UUID(value)) == value`. Uppercase hex,
braces and the unhyphenated form are rejected; idempotency keys on this string.

`X-VoiceSniffer-Language` takes up to 25 codes matching
`[a-z]{2,3}(-[A-Za-z0-9]{2,8})*`, no duplicates. Pin them. The shipped
transducer never reports which language it heard, so `auto` runs the transcript
through every installed pack, and Czech speech then gets matched against Polish,
Slovak, Croatian and Slovenian rules that are close enough to collide.

A pinned code with no installed pack is dropped and the rest still run, so
`en,de` on a node without `de.yml` still moderates English. A request where every
code drops out is `400 language_unsupported`.

`X-VoiceSniffer-Preroll-Samples` must be a multiple of 3 (the decoder resamples
48 kHz to 16 kHz and trims on the 16 kHz stream), no larger than the first
frame's sample count, and no larger than the duration limit. `0` keeps
everything.

`X-VoiceSniffer-Partial: 1` marks an early window sent while the speaker is
still talking, before the finished utterance arrives with its own request id and
body. The window still gets a verdict and its audio is still counted; its
severity is not counted as an incident. Absent means `0`. Present must be `0` or
`1`; anything else is `400 invalid_partial`.

### Request body

Length-prefixed Opus packets, concatenated, nothing before or after:

```
+---------+--------------+---------+--------------+-----
| len u16 | packet bytes | len u16 | packet bytes | ...
+---------+--------------+---------+--------------+-----
```

`len` is unsigned 16-bit big-endian, the byte length of the packet that follows.
Every packet must be mono and must decode to 1 to 5760 samples at 48 kHz. Stereo
is refused, not downmixed.

The format version is in the content type, so a client that only knows v1 fails
at the media type rather than misparsing bytes.

| Limit | Default | Error |
|---|---|---|
| Total body | 1 MiB | `body_too_large` (413) |
| Bytes per frame | 4000 | `frame_too_large` (413) |
| Frame count | 3200 | `too_many_frames` (413) |
| Decoded duration | 8000 ms | `duration_too_long` (413) |

Duration is checked frame by frame during parsing, so an oversized clip is
refused before anything decodes.

### 200 response

Headers: `X-Request-Id`, `X-VoiceSniffer-Decode-Ms`,
`X-VoiceSniffer-Transcribe-Ms`, `X-VoiceSniffer-Rules-Ms`.

```json
{
  "request_id": "0d0f6c1e-6a1d-4a2f-9b71-2f4a5f9c1234",
  "transcript": "join my server",
  "language": "en",
  "matches": [
    {
      "rule_id": "en.spam.server-ad",
      "category": "spam",
      "severity": 1,
      "matched_text": "join my server",
      "context_required": false
    }
  ],
  "severity": 1,
  "confidence": 1.0,
  "processing_ms": 214
}
```

| Field | Type | Notes |
|---|---|---|
| `transcript` | string | Normalised: NFKC, casefolded, non-alphanumerics collapsed to single spaces. This is the text the rules ran against. |
| `language` | string | The code the verdict was produced under, or `auto`. |
| `matches` | array | Every rule that fired, review tier included. May be empty. |
| `matches[].category` | string | `harassment`, `hate`, `sexual`, `threat`, `spam`. |
| `matches[].severity` | int | 1, 2 or 3. |
| `severity` | int 0-3 | Highest severity among matches with `context_required: false`. |
| `confidence` | float 0-1 | `1.0` when a pack rule matched, `0.0` when nothing did. Carries more only with a toxicity classifier wired in, which the shipped image does not do. |
| `processing_ms` | int | Decode plus transcribe plus rules plus classifier. |

**Act on the top-level `severity`, not on `matches[].severity`.** A match with
`context_required: true` is review tier and is deliberately excluded from the
top-level number. That is where every pack's self-harm disclosure rule lives. A
client that acts on per-match severity will mute people for saying they are
struggling.

`severity: 0` with a non-empty `matches` array is normal and means exactly that.

### Error response

Same shape at every status:

```json
{"request_id": "0d0f...", "code": "body_too_large", "message": "body_too_large"}
```

`message` currently equals `code`. Match on `code`.

`request_id` is `null` when the request was refused before the id was parsed,
which covers every authentication, licence and rate-limit refusal.

### Status codes

Checks run in this order; the first failure is what you get.

| Status | `code` | Cause |
|---|---|---|
| 401 | `unauthorized` | Missing or malformed `Authorization`, or an unknown token. |
| 503 | `license_check_unavailable` | Cloud nodes only: the licensing API could not be asked. Retryable, and not a statement about your token. |
| 403 | `license_expired` | Credential `expires` is in the past. |
| 429 | `rate_limited` | Over `rate-limit-per-minute`. Carries `Retry-After` in seconds. |
| 400 | `invalid_request_id` | Missing, or not a canonical UUID. |
| 415 | `unsupported_media_type` | Wrong `Content-Type`. |
| 406 | `not_acceptable` | `Accept` is not exactly `application/json`. |
| 400 | `invalid_player_id` | Not a UUID. |
| 400 | `invalid_language` | Empty, over 25 codes, malformed code, or a duplicate. |
| 400 | `invalid_preroll` | Not an integer, negative, over the duration limit, not a multiple of 3, or longer than the first frame. |
| 400 | `invalid_partial` | Present and not `0` or `1`. |
| 403 | `language_not_licensed` | A pinned code outside the credential's `languages`. |
| 400 | `language_unsupported` | No installed pack for any requested code, or the speech model here cannot transcribe it. |
| 400 | `invalid_content_length` | `Content-Length` present and not a non-negative integer. |
| 413 | `body_too_large` | Over the limit, by header or by what arrived. |
| 503 | `processor_busy` | No capacity. Checked before any work starts. |
| 409 | `request_id_reused` | This id was seen inside the idempotency window with a different body. |
| 400 | `empty_body` | Zero bytes. |
| 400 | `truncated_frame_length` | Fewer than 2 bytes where a length prefix should be. |
| 400 | `empty_frame` | Length prefix of `0`. |
| 400 | `truncated_frame` | Length prefix runs past the end of the body. |
| 413 | `frame_too_large` | One packet over the per-frame limit. |
| 413 | `too_many_frames` | More packets than the frame limit. |
| 413 | `duration_too_long` | Decoded audio over the duration limit. |
| 400 | `stereo_not_supported` | Packet declares two channels. |
| 400 | `invalid_opus_packet` | libopus could not read the packet header. |
| 400 | `invalid_pcm` | Decoder returned an odd byte count for 16-bit samples. |
| 400 | `empty_audio` | Nothing left after the preroll trim. |
| 400 | `opus_decode_failed` | libopus refused the packet body. |
| 503 | `opus_unavailable` | libopus is not installed on the host. |
| 500 | `internal_error` | A fault. The traceback is in the log; the response carries no detail. |

### Idempotency

The processor remembers `(tenant, server, request_id)` for 120 seconds with a
SHA-256 of the body and metadata.

- Same id, same body: the stored verdict. Decoded once, counted once.
- Same id, different body: `409 request_id_reused`.
- Same id, still running: the second caller waits and gets the same answer.

Fresh UUID per utterance; reuse an id only to retry the identical request.

### processor_busy

`503 processor_busy` covers two states: the box is full, or another tenant is
waiting and you are over your share of the admission pool. They are counted
separately in the log and on `/v1/usage` (`processor_busy` against
`processor_busy_tenant_share`) because the operator's fix differs. The wire code
is the same for both. Back off and retry.

## GET /v1/usage

```json
{
  "server_id": "my-server",
  "plan": "selfhost",
  "languages": "all",
  "moderated_languages": ["bg", "cs", "da", "de", "el", "en"],
  "rate_limit_per_minute": null,
  "usage": {
    "since": "2026-08-16T11:20:31+00:00",
    "requests": 0,
    "flagged": 0,
    "outcomes": {},
    "severities": {},
    "audio_seconds": 0
  }
}
```

`languages` is what the credential permits: `"all"` or a sorted array.
`moderated_languages` is that intersected with the installed rule packs.

`moderated_languages` does not account for the speech model. An English-only
model with the full rules directory still lists twenty-three codes here while
`/v1/moderate` refuses twenty-two of them. The startup log line
`rules_without_a_model` is the authority when they disagree.

`outcomes` counts requests by the `code` they ended with, plus `success`.
`severities` counts finished utterances by verdict severity.

**These counters do not divide into each other.** `requests` and `outcomes` are
per request. `flagged` and `severities` are per incident. `audio_seconds` is per
decode. Early windows land in the denominator and never in the numerator, so
"flagged per audio second" makes a busy server look clean.

Errors: `401 unauthorized`, and `503 license_check_unavailable` on cloud nodes.

## Worked request

`integration/fixtures/neutral-opus-frame.b64.json` holds one real 20 ms Opus
frame, base64 encoded.

```python
import base64, json, uuid, urllib.request

frame = base64.b64decode(
    json.load(open("integration/fixtures/neutral-opus-frame.b64.json"))["frames"][0]
)
body = len(frame).to_bytes(2, "big") + frame

request = urllib.request.Request(
    "http://127.0.0.1:8090/v1/moderate",
    data=body,
    method="POST",
    headers={
        "Authorization": "Bearer <your token>",
        "Content-Type": "application/vnd.voicesniffer.opus.v1",
        "Accept": "application/json",
        "X-Request-Id": str(uuid.uuid4()),
        "X-VoiceSniffer-Player-Id": str(uuid.uuid4()),
        "X-VoiceSniffer-Language": "en",
        "X-VoiceSniffer-Preroll-Samples": "0",
    },
)
with urllib.request.urlopen(request) as response:
    print(response.read().decode())
```

The frame is silence, so the verdict is an empty transcript and `severity: 0`.
