# Security

## Reporting

Use GitHub's private vulnerability reporting on this repository: **Security**
tab, then **Report a vulnerability**. Do not open a public issue for a
vulnerability; if private reporting is unavailable to you, open an issue saying
only that you have a report and asking for a contact.

Include the version or image digest, the exact request and response, and a
minimal reproduction. There is no bounty.

Fixes go into a new release of the container image. There is no supported older
branch.

## The bearer token is the entire authentication

No second factor, no client certificate, no body signature, no IP allowlist.
Anyone holding a valid token can submit audio and read that credential's counters
on `/v1/usage`. To rotate, edit the tokens file and restart. There is no
revocation list on a self-hosted node; the tokens file is the whole truth.

1. Generate tokens with a CSPRNG: `openssl rand -hex 32`.
2. Never send one over plain HTTP outside a private network. It is replayable by
   anyone who reads it once, on a connection also carrying voice audio.
3. Keep the tokens file secret. It is gitignored, and the compose file mounts it
   as a secret rather than an environment variable so `docker inspect` cannot
   print it.

A token cannot widen its own access. The credential, including licensed
languages and any rate limit, is built server-side from the tokens file or the
licensing API; nothing the client sends contributes to it. Comparison is
constant-time against a SHA-256 keyed lookup.

## Attack surface

A bearer token, a body of Opus packets handed to a native library, and a few
headers. No admin endpoint, no upload path, no template rendering, no database,
no shell.

### Untrusted audio into libopus

Bounds applied before decoding: total body, per-frame size, frame count and
decoded duration, all checked during parsing; mono only; 1 to 5760 samples per
packet at 48 kHz. All four ceilings can only be lowered by configuration.

`invalid_opus_packet` and `opus_decode_failed` are fixed codes. Nothing the
native layer said reaches the caller or the log, and `tests/test_opus.py` asserts
it.

If a parser bug gets through, the container limits what it is worth: read-only
root filesystem, all capabilities dropped, `no-new-privileges`, uid 10001, 128
process limit, base image pinned by digest.

### Attacker-influenced text into the rule matcher

The transcript is run against every pattern in the entitled packs. Two budgets,
both fixed in code: 5 ms per regex search, and 250 ms per pack across all its
rules. The per-rule budget alone is not enough, since one rule may hold 65
patterns and 65 near-timeout searches cost about 325 ms without any single one
raising.

A rule that exceeds its own budget is logged `rule_timeout`.
`rule_budget_exhausted` is logged at error level and means a verdict was returned
with part of the ruleset never applied.

### Denial of service by a valid client

The admission pool gives each tenant a share that narrows only under contention
and holds one slot back that no busy tenant may take, so a flooding tenant cannot
lock out a newcomer. Refusals are `503 processor_busy`, counted apart in the log
by cause.

`rate-limit-per-minute` is off by default. Turn it on if the processor is
reachable by anything you do not run.

### Model supply chain

Every file is pinned by SHA-256 and verified before install; a mismatch is a hard
failure. Sources must be `https` or a local `file:` mirror, and the default
model's URL names an immutable upstream revision. Archive extraction rejects
absolute paths, `..` components, symlinks and anything resolving outside the
destination.

The loose-file form used by the default model is the stronger of the two: the
archive form can only pin files named as loader roles, leaving 2.4 GB of ONNX
external data unverified.

### Logs and outbound data

No word anybody said reaches a log at any level. Usage counters hold integers
only. The `500` path logs the traceback, but neither the body nor the transcript
is a local variable on that stack. The moderation-history path swallows every exception
and logs only a type name.

Nothing is written to disk. Verdicts, which carry transcripts, live in the
idempotency cache in memory for 120 seconds, keyed by tenant as well as request
id.

Self-hosted, nothing about an utterance leaves the process: the object that could
send it is not built without a licensing endpoint.

### Cloud mode

A licensing API outage answers `503 license_check_unavailable`, not `401`, and
admits nobody.

## In scope

- Getting a verdict, reading usage, or affecting another caller without a valid
  token.
- Container escape, code execution, or reading a file it should not, including
  through the Opus path.
- A transcript, matched phrase or token appearing in a log, in a usage report, or
  in a response to the wrong credential.
- One credential's data reaching another.
- A request that hangs or crashes the process, or costs unbounded memory or CPU
  past the documented limits.
- A model download installing bytes the registry did not pin.
- Anything in `docker/m2/*.yml` that undoes the hardening it claims.

## Out of scope

- **The VoiceSniffer plugin.** Closed source, separate product. Use the product's
  support channel.
- **The hosted service and licensing API.** Not in this repository.
- **The speech models.** Third-party weights under their own licences.
- **Moderation accuracy.** A missed slur or a false positive is a rule pack bug
  worth reporting, but not a vulnerability. Open an issue; see
  [CONTRIBUTING.md](CONTRIBUTING.md).
- **Consequences of the documented design.** With a valid token you can send
  audio and consume CPU. Rate limiting exists in the tokens file and is off by
  default.
- **Hardening you can apply yourself**: TLS, firewall rules, a reverse proxy. The
  default is loopback-only so that exposing it is a decision somebody makes.
- **Scanner output with no reproduction.**

## Not done

No third-party audit. No fuzzing campaign against the Opus envelope parser beyond
the unit tests in `tests/test_protocol.py` and `tests/test_opus.py`.
