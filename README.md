# VoiceSniffer processor

Speech moderation service for VoiceSniffer. A Minecraft server sends short Opus
clips from voice chat; this transcribes them, matches the transcript against
language rule packs, and returns a verdict.

The **plugin** that captures the audio and acts on the verdict is a separate
paid, closed-source product. It is not in this repository and nothing here
replaces it. This is the half you can self-host, under AGPL-3.0.

A self-hosted processor makes no outbound connection once the model is
downloaded. With no licensing endpoint configured, `create_app` builds no usage
reporter and no event reporter, so the object that could put a transcript on a
socket does not exist.

## Quickstart

amd64 Linux with Docker. You need `docker/m2/docker-compose.yml` and nothing
else: the image carries its own rule packs.

```sh
cd docker/m2
printf '{"my-server":"%s"}\n' "$(openssl rand -hex 32)" > tokens.local.json
docker compose up -d
```

First start downloads a 2.4 GB model before it binds a port. Expect several
minutes; the compose file sets a 900 second healthcheck start period to cover
it. Watch with `docker compose logs -f`.

```sh
curl -fsS http://127.0.0.1:8090/healthz
# {"status":"ok"}
```

`/healthz` answering means the model is loaded, because `main()` builds the
application before it binds.

Check the token, and see which languages this node can actually moderate:

```sh
TOKEN=$(python3 -c 'import json; print(json.load(open("tokens.local.json"))["my-server"])')
curl -fsS -H "Authorization: Bearer $TOKEN" http://127.0.0.1:8090/v1/usage
```

Keep `tokens.local.json` out of any repository and off any share. That token is
the entire authentication in front of the service.

## Host requirements

| | |
|---|---|
| Architecture | amd64 only |
| Memory | 3 GiB for the container; it runs at about 2.5 GiB |
| CPU | 4 vCPU at the shipped sizing |
| Disk | 347 MB image, plus 2.4 GB for the default model |
| Network | Outbound HTTPS on first start only, to fetch the model |

## Pointing the plugin at it

Give the plugin the base URL and the same token.

The compose file publishes on `127.0.0.1:8090`. To run the processor on another
box, set `VOICESNIFFER_BIND_ADDRESS` to that host's LAN address. The plugin
accepts plain `http` to `127.x`, `10.x`, `172.16-31.x`, `192.168.x` and
`169.254.x`. Anything routable off site needs TLS in front; the plugin refuses a
non-private `http://` endpoint, and it is carrying voice audio and a replayable
bearer token.

The plugin's `processor-tuning.max-in-flight` (32) and what one tenant may hold
in the processor's admission pool are meant to match. The shipped sizing lines
them up. Change one, change the other.

## Running from source

Python 3.12 and [uv](https://docs.astral.sh/uv/). Runtime first: the service
resolves it through a relative path source.

```sh
cd processor/runtime && uv sync --frozen && uv run pytest -q
cd ../service        && uv sync --frozen && uv run pytest -q
```

```sh
cd processor/service
printf '{"my-server":"dev-token"}\n' > tokens.local.json
VOICESNIFFER_TOKENS_FILE=./tokens.local.json \
VOICESNIFFER_MODEL_ID=moonshine-tiny-v2-en \
VOICESNIFFER_MODELS_DIR=./.models \
VOICESNIFFER_RULES_DIR=./rules \
  uv run python -m voicesniffer_processor
```

`moonshine-tiny-v2-en` is used here because it downloads in seconds. It is
English only and has no accuracy measurement recorded in this repository. Use
`parakeet-tdt-0.6b-v3-fp32` for anything real.

Opus decoding needs `libopus` on the host (`apt-get install libopus0`). Without
it, two tests skip and `/v1/moderate` answers `503 opus_unavailable`.

## Limitations and things to know

**amd64 only.** The runtime installs prebuilt sherpa-onnx wheels. No arm64 image
is published.

**2.4 GB on first start.** The default model is fetched as loose files, each
pinned by SHA-256 against an immutable upstream revision, and cached in a named
volume per model pin.

**Accuracy is measured for two languages, not twenty-three.**
`parakeet-tdt-0.6b-v3-fp32` measures 5.8% word error rate for English and 10.2%
for Czech, on a 520-utterance corpus pushed through the real Opus path. The other
twenty-one packs are marked `unmeasured` in `processor/runtime/languages.toml`,
and those two numbers do not transfer to them. A rule pack existing is not
evidence that speech recognition in that language is good enough for the pack to
matter.

**The int8 Parakeet export is a bad trade for its size.** Same checkpoint, 28.1%
English word error rate against the fp32 export's 5.8%, and an empty transcript
for 52 of 260 English utterances. The Moonshine models are English only and have
no measurement recorded here.

**Twenty-three rule packs against a model advertising twenty-five languages.**
`fi` and `mt` can be transcribed and not moderated. Run a smaller model against
the full rules directory and the gap opens the other way. Either way, a request
that resolves to no rule pack at all is refused with `400 language_unsupported`
rather than answered `200` with `severity: 0`. Both gaps are named in the log at
startup.

**Rule packs are term lists with a fuzzy matcher, not a model.** They catch what
they list. `create_app` accepts a toxicity classifier; the shipped image passes
none.

**Eight seconds of decoded audio per request**, 1 MiB per body, 3200 frames.

**Nothing is written to disk.** Verdicts live in memory for 120 seconds in the
idempotency cache. The transcript goes back in the response and, on a self-hosted
node, nowhere else.

## Documentation

- [docs/protocol.md](docs/protocol.md): the wire contract, enough to write
  another client.
- [docs/configuration.md](docs/configuration.md): every environment variable.
- [docs/deployment.md](docs/deployment.md): exposure, TLS, sizing, what to watch.
- [SECURITY.md](SECURITY.md), [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

AGPL-3.0. See [LICENSE](LICENSE).

Speech models are separate works under their own licences, recorded per model in
`processor/runtime/models.toml`. Parakeet is CC-BY-4.0; the Moonshine and Whisper
builds are MIT.
