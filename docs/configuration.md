# Configuration

Everything is an environment variable. Out-of-range values fail at startup and
name the variable; nothing is clamped.

## Required

The container image defaults three of these. It deliberately does not default
`VOICESNIFFER_TOKENS_FILE`, because a processor that started without one would
accept unauthenticated audio. Both CI workflows assert that absence.

| Variable | Container default | Meaning |
|---|---|---|
| `VOICESNIFFER_TOKENS_FILE` | none | Path to the JSON file of server tokens. |
| `VOICESNIFFER_MODEL_ID` | `parakeet-tdt-0.6b-v3-fp32` | Which registry model to load. |
| `VOICESNIFFER_MODELS_DIR` | `/models` | Model download cache. Must be writable. |
| `VOICESNIFFER_RULES_DIR` | `/app/rules` | Directory of `*.yml` rule packs. Must contain at least one. |

Running from source, none of the four are defaulted.

## Tokens file

JSON, server id to credential. Ids match `[A-Za-z0-9._-]{1,64}`. Duplicate token
values anywhere in the file are a startup error.

```json
{
  "survival-1": "a-long-random-string",
  "creative-1": {
    "token": "a-different-long-random-string",
    "plan": "selfhost",
    "languages": ["en", "cs"],
    "rate-limit-per-minute": 600,
    "expires": "2027-01-01T00:00:00Z"
  }
}
```

The string form grants full access with no limits. In the object form:

| Key | Default | Meaning |
|---|---|---|
| `token` | required | The bearer token. |
| `plan` | `selfhost` | Label matching `[a-z0-9-]{1,32}`, reported by `/v1/usage`. |
| `languages` | every installed pack | A pinned code outside the list is `403 language_not_licensed`; `auto` narrows to the list. |
| `rate-limit-per-minute` | unlimited | Sliding 60-second window. Over it is `429` with `Retry-After`. |
| `expires` | never | ISO 8601. Naive timestamps are read as UTC. |

Generate tokens with `openssl rand -hex 32`.

The file must be readable by uid 10001. Compose mounts file secrets `0444`, so
the shipped compose needs no `chown`. A permission failure is reported by name
rather than as a JSON error.

## Network

| Variable | Default | Range | Meaning |
|---|---|---|---|
| `VOICESNIFFER_BIND_HOST` | `127.0.0.1` | non-empty | Address the process binds. |
| `VOICESNIFFER_BIND_PORT` | `8080` | 1-65535 | Port the process binds. |
| `VOICESNIFFER_HYPERCORN_CONFIG` | `hypercorn.toml` | path | Keep-alive, graceful timeout, log level. |

The shipped compose sets `BIND_HOST=0.0.0.0` and controls exposure with the
published port instead. See [deployment.md](deployment.md).

Hypercorn runs one worker and the application forces it, because a second
process means a second copy of the model in memory. Concurrency is
`VOICESNIFFER_WORKERS`.

## Throughput

| Variable | Default | Range | Meaning |
|---|---|---|---|
| `VOICESNIFFER_WORKERS` | `1` | 1-64 | Concurrent utterances in speech-to-text. |
| `VOICESNIFFER_MODEL_THREADS` | `1` | 1-64 | ONNX intra-op threads per inference. |

Voice chat is many short utterances, so worker parallelism beats wider
per-inference threading. On the 4-vCPU host the compose file is tuned for, four
workers with one model thread each measured 38 utterances per second against 26
at two workers. Raise memory before raising workers.

`VOICESNIFFER_WORKERS` also sizes the admission pool at `max(16, workers * 8) + 1`,
one slot of which is reserved so a tenant with nothing in flight can always get
in. At four workers that is 32 admissions for a single tenant, matching the
plugin's `processor-tuning.max-in-flight`.

## Request limits

| Variable | Default | Range |
|---|---|---|
| `VOICESNIFFER_MAX_BODY_BYTES` | `1048576` | 1024-1048576 |
| `VOICESNIFFER_MAX_FRAME_BYTES` | `4000` | 1-4000 |
| `VOICESNIFFER_MAX_FRAMES` | `3200` | 1-3200 |
| `VOICESNIFFER_MAX_DURATION_MS` | `8000` | 500-8000 |

Each ceiling equals its default, so these can only be lowered. Raising one needs
a code change.

## Model selection and tuning

| Variable | Default | Range | Meaning |
|---|---|---|---|
| `VOICESNIFFER_MODEL_REGISTRY` | packaged `models.toml` | path | Which registry to read. |
| `VOICESNIFFER_PROVIDER` | `cpu` | `cpu`, `cuda` | ONNX Runtime execution provider. |
| `VOICESNIFFER_DECODING_METHOD` | `modified_beam_search` | `greedy_search`, `modified_beam_search` | Transducer search. Whisper and Moonshine ignore it. |
| `VOICESNIFFER_MAX_ACTIVE_PATHS` | `2` | 1-8 | Beam width. Ignored by `greedy_search`. |
| `VOICESNIFFER_FEATURE_THREADS` | `1` | 1-8 | Threads computing filterbank features per batch. |
| `VOICESNIFFER_INPUT_RMS` | `0.06` | 0-0.5 | Target RMS for input normalisation. |
| `VOICESNIFFER_TRIM_HEAD_PAD_MS` | `-1` | -1 to 2000 | Guard band before the first voiced frame. `-1` disables trimming. |
| `VOICESNIFFER_TRIM_TAIL_PAD_MS` | `-1` | -1 to 2000 | Same, after the last voiced frame. |
| `VOICESNIFFER_MAX_BATCH_SIZE` | `1` | 1-64 | Micro-batch size. `1` disables batching. |
| `VOICESNIFFER_BATCH_WINDOW_MS` | `0` | 0-500 | How long a partial batch waits. |

Beam search is the default despite costing about 23% of throughput: against
greedy on the same 520-utterance corpus, eight of fourteen accuracy figures
improved, six were identical, none regressed.

The silence trimmers default to off because trimming loses accuracy in
proportion to the duration removed, even with a 600 ms guard band and even with
normalisation done before the trim. The knobs stay for models that behave
differently.

Micro-batching is a GPU lever. It funnels every utterance through one decode
thread, which on CPU measured 8.4 utterances per second against 11.8 with the
same cores spent on worker parallelism. Raise `MAX_BATCH_SIZE` only where one
device does the arithmetic and idles between clips.

### Registry models

| `VOICESNIFFER_MODEL_ID` | Languages | Notes |
|---|---|---|
| `parakeet-tdt-0.6b-v3-fp32` | 25 | Default. 5.8% English / 10.2% Czech WER. About 2.4 GB. |
| `parakeet-tdt-0.6b-v3-int8` | 25 | Same checkpoint quantised. 28.1% English / 25.7% Czech, and 52 of 260 English utterances come back empty. |
| `whisper-small-multilingual-int8` | 25 declared | 13.5% English / 45.2% Czech. The only shipped model taking an explicit language argument, so the only route to a language outside the other twenty-five. |
| `moonshine-base-v2-en` | English | No measurement recorded here. |
| `moonshine-tiny-v2-en` | English | Smallest. What CI boots to prove the container starts. |

## Cloud mode

Four variables turn a processor into a node of the hosted service. Self-hosted
deployments set none of them.

| Variable | Default | Meaning |
|---|---|---|
| `VOICESNIFFER_INTROSPECTION_URL` | empty | Licensing API endpoint resolving unknown tokens. |
| `VOICESNIFFER_INTROSPECTION_TOKEN` | empty | Service token for the internal endpoints. |
| `VOICESNIFFER_USAGE_URL` | derived | Where usage counters are posted. |
| `VOICESNIFFER_EVENTS_URL` | derived | Where flagged utterances are filed. |

`VOICESNIFFER_INTROSPECTION_TOKEN_FILE` is read in preference to the plain
variable, which keeps the value out of `docker inspect`.

The two URLs are derived from an introspection URL ending in `/introspect`.
Neither is a fallback for the other: the usage feed carries counters, the event
feed carries transcripts.

With no introspection URL there is no credential resolver, no usage reporter and
no event reporter. The event reporter is the only thing in the process that can
put a spoken word on a socket, so self-hosted it is absent rather than disabled.

## Compose-only variables

Read by `docker/m2/docker-compose.yml`, not by the application.

| Variable | Default | Meaning |
|---|---|---|
| `VOICESNIFFER_IMAGE_TAG` | `1.0.0` | Image tag. |
| `VOICESNIFFER_BIND_ADDRESS` | `127.0.0.1` | Host interface the port is published on. |
| `VOICESNIFFER_PROCESSOR_PORT` | `8090` | Host port. |
| `VOICESNIFFER_MEMORY` | `3g` | Sets `mem_limit` and `memswap_limit` together. |
| `VOICESNIFFER_CPUS` | `4` | CPU quota. |
| `VOICESNIFFER_TOKENS_FILE_HOST` | `./tokens.local.json` | Host path of the tokens secret. |
| `VOICESNIFFER_RULES_DIR_HOST` | required | Host rules directory, for `docker-compose.rules.yml`. |

`BIND_HOST` is what the process listens on inside the container.
`BIND_ADDRESS` is the host interface Docker publishes to, and is the one that
decides who can reach the service.
