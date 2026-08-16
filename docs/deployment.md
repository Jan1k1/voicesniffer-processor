# Deployment

`docker/m2/docker-compose.yml` is production, not a simplified example. Read it
alongside this page.

## Beside the Minecraft server

The default. The plugin talks to `http://127.0.0.1:8090`, nothing is published
to the network, no audio leaves the machine. Nothing to configure beyond the
quickstart.

Budget about 2.5 GiB and four cores that Minecraft is not using. On a shared
4-core box, lower `VOICESNIFFER_WORKERS` and `VOICESNIFFER_CPUS` rather than
letting tick rate suffer.

## On a separate box, same network

```sh
VOICESNIFFER_BIND_ADDRESS=10.0.0.5 docker compose up -d
```

The plugin accepts plain `http` to `127.x`, `10.x`, `172.16-31.x`, `192.168.x`
and `169.254.x`.

`VOICESNIFFER_BIND_HOST` is what the process listens on inside the container and
is already `0.0.0.0`. `VOICESNIFFER_BIND_ADDRESS` is the host interface Docker
publishes to. Changing the wrong one gives you a container that looks configured
and is unreachable, or one that is reachable and was not meant to be.

Firewall the port to the game server's address. The bearer token is the only
thing between an open port and anyone feeding this service whatever audio they
like.

## Anywhere routable

Terminate TLS in front and give the plugin the `https` address. The plugin
enforces this by refusing a non-private `http://` endpoint: plain HTTP puts voice
audio and a replayable bearer token on the wire.

Two things the reverse proxy must not break:

- Request bodies up to 1 MiB.
- The `X-Request-Id` and `X-VoiceSniffer-*` request headers. Stripping them gives
  `400 invalid_request_id` or `400 invalid_player_id` on every request.

Keep the processor bound to loopback or to a Docker network the proxy reaches.

## Sizing

| Setting | Shipped | Basis |
|---|---|---|
| `VOICESNIFFER_WORKERS` | 4 | 38 utterances per second against 26 at two workers, on a 4-vCPU host. |
| `VOICESNIFFER_MODEL_THREADS` | 1 | Many short clips favour worker parallelism over intra-op threading. |
| `VOICESNIFFER_CPUS` | 4 | One core per worker. |
| `VOICESNIFFER_MEMORY` | 3g | fp32 Parakeet plus per-worker arenas. Runs at about 2.5 GiB. |

Raise memory before workers. Eight workers on a 3 GiB limit is an OOM kill, not
eight workers.

`mem_limit` and `memswap_limit` are set equal, which means no swap. Docker's
default allows twice the memory in swap, and an ASR process paging model arenas
does not fail: it goes slow, drags the rest of the box down and keeps answering
the healthcheck. Both limits read `VOICESNIFFER_MEMORY`, so they move together.

Plan against utterances per second, which depends on how much people talk rather
than on player count. Watch `audio_seconds` on `/v1/usage` for a week before
sizing a second box.

## First start

2.4 GB download before the port binds. The compose file sets a 900 second
healthcheck start period; the image's own is 60 seconds, and Docker takes the
compose block wholesale when one is present, so a hand-written compose file needs
every field repeated.

Models cache in a named volume keyed by a content address of the pinned files, so
changing `VOICESNIFFER_MODEL_ID` installs beside the old one. Downloads resume
per file.

## What to watch

**OOM restart loops.** `restart: unless-stopped` restarts an OOM-killed
container, and Docker's backoff resets after ten seconds of uptime. Startup takes
longer than that, so the loop never backs off and never stops while the
healthcheck flickers. Watch the restart count or the `OOMKilled` flag in
`docker inspect` from outside the compose file.

**Two startup log lines**, each logged once. Neither is fatal, because trimming
the rules directory to the languages you run is legitimate. Neither is safe to
ignore.

```
languages_without_rules ... missing=fi,mt
```
The model can transcribe these; no rule pack exists. Requests pinned to one are
`400 language_unsupported`. Expected today: twenty-three packs against a model
advertising twenty-five languages.

```
rules_without_a_model ... missing=bg,cs,da,...
```
Rule packs exist; the model here was not trained on them. This is what an
English-only model with the full rules directory looks like. `/v1/usage` will not
warn you: its `moderated_languages` counts packs and licence, not the model.

**Log lines worth an alert:**

| Line | Meaning |
|---|---|
| `rule_budget_exhausted` | A pack ran out of its time budget; part of the ruleset never ran. Those verdicts are incomplete. |
| `rule_timeout` | One rule exceeded its regex budget. A pattern of them is not fine. |
| `admission_refused reason=processor_busy` | Box is full. |
| `admission_refused reason=processor_busy_tenant_share` | Box is contended, and the line names who. |
| `language_unsupported` | A server is sending a language this node cannot moderate. Each one is an utterance nobody checked. |
| `request_failed` | A 500, with the traceback on the same line. |

**Normal traffic** is one `request_trace` line per utterance:

```
request_trace request_id=... outcome=success total_ms=214 decode_ms=3 stt_ms=198 rules_ms=1 classifier_ms=0 serialize_ms=1
```

The compose file caps the json-file driver at 20 MB across 5 files for that
reason.

## Hardening already in the compose file

Listed so you know what you would drop by writing your own.

- Read-only root filesystem, 64 MB tmpfs at `/tmp`. Only the models volume is
  writable.
- All capabilities dropped, `no-new-privileges`.
- Non-root uid 10001. The image creates `/models` owned by that uid, so a fresh
  named volume is seeded writable with no host-side `chown`.
- `pids_limit: 128`.
- Base image pinned by digest.
- Tokens file as a compose secret mounted `0444`, not an environment variable.
- `stop_grace_period: 30s`. The default ten seconds kills a cloud node mid-flush.

## Your own rule packs

```sh
VOICESNIFFER_RULES_DIR_HOST=/srv/voicesniffer/rules \
  docker compose -f docker-compose.yml -f docker-compose.rules.yml up -d
```

The directory must contain at least one valid `*.yml` pack or the container will
not start. Format and gates: [CONTRIBUTING.md](../CONTRIBUTING.md).

## Building instead of pulling

```sh
cd docker/m2
docker compose -f docker-compose.yml -f docker-compose.build.yml up -d --build
```

Compose tags the local build with the base service's `image:` value, so it
shadows the published tag under the same name. `docker compose pull` restores it.
