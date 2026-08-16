# Contributing

This repository is the processor: HTTP service, Opus decode, speech-to-text,
rule matching. The plugin that calls it is a separate closed-source product and
is not here, so changes that only make sense alongside a plugin change belong in
an issue first.

Bug reports and rule pack corrections are the most useful things to send.

## Setup

Python 3.12 and [uv](https://docs.astral.sh/uv/). Runtime first: the service
depends on it through a relative path source.

```sh
cd processor/runtime && uv sync --frozen
cd ../service        && uv sync --frozen
```

Both packages pin dependencies in a committed `uv.lock`, so `--frozen` is the
right flag.

Install `libopus` (`apt-get install libopus0`, or `brew install opus`) or two
tests silently skip.

## Checks

Exactly what CI runs, in both packages:

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

Ruff config is in each `pyproject.toml`: line length 100, target py312,
`E,F,I,UP,B,SIM,RUF`, LF endings. `ruff format --check` is a gate.

### The native tests

`tests/test_opus_native.py` and `tests/test_e2e.py` are marked `native` and need
libopus. Without it they skip, so a green local run does not prove they passed:

```sh
uv run pytest -m native -q -rs
```

`-rs` prints the reason. `libopus unavailable` means they did not run. CI
installs `libopus0`.

### Inside the image

```sh
cd docker/m2
docker compose -f docker-compose.yml -f docker-compose.build.yml \
  --profile test run --rm test
```

CI builds and runs that stage too.

## What CI checks

`.github/workflows/ci.yml`. Every job is runnable from a fresh clone.

`lint + tests` installs libopus0, then per package runs `uv sync --frozen`,
`ruff check`, `ruff format --check`, `pytest -q`.

`image builds, starts, and is self-sufficient` also asserts:

- Both compose files parse, base and base plus build override.
- The image carries at least 23 rule packs at `/app/rules`. Counted, not checked
  for existence, because an empty directory fails the same way a missing one
  does.
- `VOICESNIFFER_MODEL_ID`, `VOICESNIFFER_MODELS_DIR` and
  `VOICESNIFFER_RULES_DIR` are defaulted in the image and
  `VOICESNIFFER_TOKENS_FILE` is not.
- The container reaches `healthy` run the way somebody who pulled the image runs
  it: a token file, no bind mounts, no source tree.
- An unauthenticated `POST /v1/moderate` returns 401 or 403.

## Tests that are not here

The private monorepo runs a wider gate, because it can check the processor
against the plugin. **Three test modules that do that are not in this
repository**, and `ci.yml` does not pretend to run them.

That leaves one number that can go stale silently. `tests/test_api.py` has
`plugin_max_in_flight()`, which returns `32`. In the monorepo it reads the
plugin's `advanced.yml`, so the two sides cannot drift without a red build. Here
it is written down.

It is a sizing assumption, not a correctness one: the processor works with any
value. If you run a modified plugin with a different
`processor-tuning.max-in-flight`, change that function too, or
`test_one_tenant_may_hold_what_the_plugin_is_configured_to_send` fails for a
reason that is not a bug.

The token-level false-positive sweeps described in
`tests/test_rule_pack_content.py` use scripts under `processor/bench/`, which is
also not in this checkout. They are a required gate before a fuzzy pattern
changes. If you cannot run them, say so in the pull request and describe what you
checked instead.

## Rule packs

One YAML file per language in `processor/service/rules/`, named by language code.
The processor globs that directory, so adding a file adds a language, and two
files declaring the same `language` is a startup error.

```yaml
version: 1
language: en
allow:
  - kill yourself to respawn
rules:
  - id: en.spam.server-ad
    term: join my server
    category: spam
    severity: 1
    match: exact
    variants:
      - visit my server
      - come to my server
    context_required: false
```

| Field | Required | Notes |
|---|---|---|
| `id` | yes | `[a-z0-9][a-z0-9._-]{0,127}`, unique, prefixed with the pack's language code. |
| `term` | yes | Primary pattern, up to 256 characters. |
| `variants` | no | Up to 64 more patterns for the same rule. |
| `category` | yes | `harassment`, `hate`, `sexual`, `threat`, `spam`. |
| `severity` | yes | 1, 2 or 3. |
| `match` | yes | `exact`, `contains`, `regex`, `fuzzy`. |
| `context_required` | no | Defaults to `false`. |
| `probe` | regex only | A string the pattern must actually match once normalised. |

Up to 1000 rules per pack, 512 `allow` entries.

### Match types

Transcripts are normalised before matching: NFKC, casefolded, non-alphanumerics
to single spaces. Source patterns for `exact`, `contains` and `fuzzy` go through
the same normalisation at load, so `JOIN MY SERVER!` loads as `join my server`.

- **`exact`**: whole tokens, bounded by whitespace or the ends of the transcript.
- **`contains`**: anywhere, including inside a word.
- **`regex`**: compiled with `IGNORECASE | FULLCASE` against the already-normalised
  transcript. Regex source is not normalised, so a literal apostrophe, hyphen or
  escaped dot can never match. Capitals are fine.
- **`fuzzy`**: compares the sequence of maximal same-letter runs. A token matches
  with the same number of runs in the same order and at most `budget` differing,
  where a differing run may change length (`nigga` to `niiigga`), swap one vowel
  for another (`nigga` to `niggo`), or use a digit for the exact vowel it depicts
  (`1` for `i`, never for `a`). A different consonant ends the comparison. Budget
  is 1 below six characters, 2 at six or above. Patterns must be a single token
  of at least four characters; both are refused at load.

`regex` rules must carry a `probe`. The loader normalises it and refuses the pack
unless a pattern matches it, because reachability is not decidable statically
here and an unreachable regex looks identical to a working one until an incident.

`context_required: true` is the review tier: reported, and excluded from the
severity that drives an automatic action. Every pack's self-harm disclosure rule
lives there.

`allow` suppresses any match falling inside it, for cases a matcher cannot
resolve: `niger` the country against the slur, `kill yourself to respawn` against
`kill yourself`. An entry identical to a listed pattern is refused at load, since
it would make the rule stop existing while still looking present. Containment is
the mechanism; equality is always a mistake. Keep these lists close to empty.

## Proposing a rule pack change

A missed slur is a moderation gap. A false positive is an hour-long mute on
somebody who said `chunk`, and enough of those end the product. So changes arrive
with evidence.

1. **Give the utterance, not the word.** `test_rule_pack_content.py` is a corpus
   of confirmed cases. Its first version was built from words somebody thought
   might be risky, and passed while the packs hour-muted `nagas` (a Twilight
   Forest boss), `nagar`, `nagger`, `ritard`, `bezny` and `I'll shoot you with my
   bow`.
2. **Add it to that corpus.** Must-flag phrases go in the realistic-utterance
   lists; confirmed false positives go in the collision lists, which assert
   nothing acts on them. Both run against every pack found in the directory.
3. **Run both rule suites.** `test_rules_regression.py` pins the matcher and is
   vocabulary-free. `test_rule_pack_content.py` pins coverage. Neither subsumes
   the other, and a benign corpus alone also passes if the fuzzy matcher returns
   nothing for everything, which is what `test_fuzzy_matching_is_alive` guards.
4. **Touching a `fuzzy` pattern means running the token sweeps.** They are not
   sufficient on their own: run against the pre-fix patterns they found
   `fagot`/`fagots` in both English dictionaries and `cogan`/`cygan`/`cákaný` in
   the Czech one, and missed `nagas`, `nagar`, `nagger` and `ritard`, because a
   game boss, a place name, a derived form and a music abbreviation are in no word
   dictionary.

The suite already enforces: every rule fires on its own terms; every declared
category and severity is reachable; every `allow` entry suppresses something;
rule ids are namespaced to their pack; a term with a marked consonant ships a
de-accented twin, because the transcriber drops the mark; contractions ship both
spellings.

`kill`, `chunk`, `spawn`, `raid`, `wither`, `nether`, `ghast`, `naked` (no
armour), `ip` (server address) and `shoot` are ordinary speech on a Minecraft
server. A rule that cannot tell PvP talk from a threat mutes players for playing.

## Pull requests

One thing per pull request. Say why in the description; the diff says what. Green
CI including `ruff format --check`. New behaviour needs a test; a fix needs a test
that fails without it. If your change makes a comment in the tree wrong, fix the
comment in the same pull request. House style for prose in this repository uses
no em dashes.

## Licence

Contributions are licensed under AGPL-3.0, the same as the rest of the
repository.
