from pathlib import Path

import pytest

from voicesniffer_processor.rules import (
    DEFAULT_MAX_MATCH_SECONDS,
    RulePack,
    RulePackError,
    normalize_transcript,
)


def test_normalizes_unicode_case_punctuation_and_whitespace() -> None:
    assert normalize_transcript("  ČAU,\u00a0Straße! \uff21\uff22\uff23  ") == "čau strasse abc"


def test_exact_match_uses_token_boundaries(tmp_path) -> None:
    rules = load_pack(
        tmp_path,
        [rule(term="assassin", match="exact")],
    )

    assert [match.matched_text for match in rules.match("that assassin won").matches] == [
        "assassin"
    ]
    assert rules.match("the assassination failed").matches == ()


def test_contains_match_and_variants(tmp_path) -> None:
    rules = load_pack(
        tmp_path,
        [
            rule(
                term="blocked phrase",
                match="contains",
                variants=["prohibited phrase"],
            )
        ],
    )

    verdict = rules.match("a prohibited phrase was spoken")

    assert verdict.matches[0].rule_id == "en.test.blocked"
    assert verdict.matches[0].matched_text == "prohibited phrase"
    assert verdict.severity == 2


def test_regex_match_runs_against_normalized_transcript(tmp_path) -> None:
    rules = load_pack(
        tmp_path,
        [rule(term=r"block(?:ed|ing) phrase", match="regex", probe="blocked phrase")],
    )

    verdict = rules.match("BLOCKING, phrase")

    assert verdict.matches[0].matched_text == "blocking phrase"


def test_context_required_match_cannot_raise_effective_severity(tmp_path) -> None:
    rules = load_pack(
        tmp_path,
        [rule(term="ambiguous phrase", severity=3, context_required=True)],
    )

    verdict = rules.match("an ambiguous phrase")

    assert verdict.matches[0].context_required is True
    assert verdict.severity == 0


def test_multiple_actionable_matches_use_highest_severity(tmp_path) -> None:
    rules = load_pack(
        tmp_path,
        [
            rule(rule_id="en.test.low", term="low phrase", severity=1),
            rule(rule_id="en.test.high", term="high phrase", severity=3),
        ],
    )

    verdict = rules.match("low phrase and high phrase")

    assert verdict.severity == 3
    assert [match.rule_id for match in verdict.matches] == ["en.test.low", "en.test.high"]


@pytest.mark.parametrize(
    "changed",
    [
        {"category": "other"},
        {"severity": 0},
        {"severity": 4},
        {"match": "other"},
        {"term": ""},
        {"term": "!!!"},
        {"term": "x" * 257},
        {"variants": "not-a-list"},
        {"context_required": "yes"},
    ],
)
def test_rejects_invalid_rule_fields(tmp_path, changed) -> None:
    invalid_rule = rule() | changed

    with pytest.raises(RulePackError):
        load_pack(tmp_path, [invalid_rule])


def test_rejects_duplicate_rule_ids(tmp_path) -> None:
    duplicate = rule()

    with pytest.raises(RulePackError, match="duplicate_rule_id"):
        load_pack(tmp_path, [duplicate, duplicate])


def test_rejects_invalid_regex(tmp_path) -> None:
    with pytest.raises(RulePackError, match="invalid_regex"):
        load_pack(tmp_path, [rule(term="(", match="regex")])


def test_rejects_regex_that_matches_empty_text(tmp_path) -> None:
    with pytest.raises(RulePackError, match="empty_regex"):
        load_pack(tmp_path, [rule(term="a*", match="regex")])


def test_regex_timeout_is_not_a_match(tmp_path) -> None:
    rules = load_pack(
        tmp_path,
        [rule(term=r"(a+)+$", match="regex", probe="aaa")],
        regex_timeout_seconds=0.001,
    )

    verdict = rules.match("a" * 100_000 + "b")

    assert verdict.matches == ()
    assert verdict.timed_out_rule_ids == ("en.test.blocked",)


# ---------------------------------------------------------------------------
# Loader guards: patterns that load clean and can never match
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("changed", "code"),
    (
        ({"term": "kill yourself"}, "fuzzy_pattern_not_single_token"),
        ({"variants": ["kill yourself"]}, "fuzzy_pattern_not_single_token"),
        ({"term": "abc"}, "fuzzy_pattern_too_short"),
        ({"variants": ["abc"]}, "fuzzy_pattern_too_short"),
        # Normalisation splits on punctuation, so a hyphenated term becomes two
        # tokens and is refused for the same reason a spaced one is.
        ({"term": "kill-yourself"}, "fuzzy_pattern_not_single_token"),
    ),
)
def test_rejects_fuzzy_patterns_that_can_never_match(tmp_path, changed, code) -> None:
    """`_match_fuzzy` compares one transcript token at a time and skips tokens
    under four characters, so both of these are dead on arrival. Before the guard
    they loaded, passed validation, and silently matched nothing forever."""
    invalid = rule(term="faggot", match="fuzzy") | changed

    with pytest.raises(RulePackError, match=code):
        load_pack(tmp_path, [invalid])


def test_accepts_a_fuzzy_pattern_at_exactly_the_minimum_length(tmp_path) -> None:
    rules = load_pack(tmp_path, [rule(term="abcd", match="fuzzy")])

    assert rules.match("abcd").matches[0].matched_text == "abcd"


def test_rejects_a_regex_rule_without_a_probe(tmp_path) -> None:
    """Regex source is the one pattern type the loader does not normalise, so an
    unreachable pattern is indistinguishable from a working one. The rule has to
    carry the proof."""
    with pytest.raises(RulePackError, match="regex_rule_needs_probe"):
        load_pack(tmp_path, [rule(term="blocked phrase", match="regex")])


@pytest.mark.parametrize(
    ("term", "probe", "code"),
    (
        # The classic: a literal hyphen cannot survive normalisation, so this
        # pattern could never have fired.
        (r"kill-yourself", "kill yourself", "regex_probe_does_not_match"),
        (r"don't", "dont", "regex_probe_does_not_match"),
        (r"blocked\.", "blocked", "regex_probe_does_not_match"),
        (r"blocked phrase", "BLOCKED PHRASE!", None),
        # Capitals are *not* a reachability problem: `_compile_pattern` uses
        # IGNORECASE | FULLCASE, so an uppercase literal matches normalised text
        # fine. Pinned because the opposite was asserted in review and in an
        # earlier version of this pack's header, and it was wrong.
        (r"Blocked Phrase", "blocked phrase", None),
        (r"blocked", "!!!", "regex_probe_normalises_to_nothing"),
    ),
)
def test_regex_probe_must_actually_match_the_normalised_form(tmp_path, term, probe, code) -> None:
    candidate = [rule(term=term, match="regex", probe=probe)]

    if code is None:
        assert load_pack(tmp_path, candidate).match("blocked phrase").matches != ()
        return
    with pytest.raises(RulePackError, match=code):
        load_pack(tmp_path, candidate)


def test_rejects_an_allow_entry_that_shadows_a_listed_pattern(tmp_path) -> None:
    """Equality means the rule can never fire on any input while still looking
    present in the file. This was the hole the old shadow *test* could not see,
    because a shadowed rule produces exactly the no-match the test asserted."""
    with pytest.raises(RulePackError, match="allow_shadows_rule"):
        load_pack(tmp_path, [rule(term="blocked phrase")], allow=["blocked phrase"])


def test_allows_an_allow_entry_that_merely_contains_a_listed_pattern(tmp_path) -> None:
    """Containment is the mechanism working as designed -- `kill yourself to
    respawn` carving out `kill yourself` -- and must not be refused."""
    rules = load_pack(
        tmp_path,
        [rule(term="blocked phrase")],
        allow=["blocked phrase is fine here"],
    )

    assert rules.match("blocked phrase is fine here").matches == ()
    assert rules.match("that blocked phrase").severity == 2


# ---------------------------------------------------------------------------
# The pack-wide budget
# ---------------------------------------------------------------------------
def test_a_pack_wide_budget_stops_matching_and_reports_what_it_skipped(tmp_path) -> None:
    """The per-rule regex timeout bounds one rule, not the pack: at the schema
    limits one rule may hold 65 patterns, so 65 near-timeout searches cost
    ~325 ms without ever raising. Here a single slow rule exhausts a 10 ms pack
    budget and the rule after it is never evaluated -- and says so."""
    rules = load_pack(
        tmp_path,
        [
            rule(rule_id="en.test.slow", term=r"(a+)+$", match="regex", probe="aaa"),
            rule(rule_id="en.test.later", term="blocked phrase"),
        ],
        regex_timeout_seconds=0.05,
        max_match_seconds=0.01,
    )

    verdict = rules.match("a" * 100_000 + "b blocked phrase")

    assert verdict.timed_out_rule_ids == ("en.test.slow",)
    assert verdict.skipped_rule_ids == ("en.test.later",)
    assert verdict.matches == ()
    assert verdict.severity == 0


def test_an_ordinary_transcript_skips_nothing(tmp_path) -> None:
    rules = load_pack(tmp_path, [rule(term="blocked phrase")])

    verdict = rules.match("a blocked phrase was spoken")

    assert verdict.skipped_rule_ids == ()
    assert verdict.timed_out_rule_ids == ()


def test_rejects_an_invalid_pack_budget(tmp_path) -> None:
    with pytest.raises(RulePackError, match="invalid_max_match_seconds"):
        load_pack(tmp_path, [rule()], max_match_seconds=0)


def test_the_shipped_packs_stay_far_inside_the_budget() -> None:
    """The budget is 250 ms and both packs together cost ~1.3 ms on a long
    transcript. Asserted so that a pack growing by an order of magnitude is a
    test failure rather than a production `rule_budget_exhausted`."""
    import time

    transcript = " ".join(["load the chunk and kill the wither at spawn"] * 20)
    packs = [load_production_pack("en"), load_production_pack("cs")]
    started = time.perf_counter()
    for pack in packs:
        assert pack.match(transcript).skipped_rule_ids == ()
    elapsed = time.perf_counter() - started

    assert elapsed < 0.05, f"both packs took {elapsed * 1000:.1f} ms on one transcript"


# ---------------------------------------------------------------------------
# The normalised fast path
# ---------------------------------------------------------------------------
def test_normalized_true_promises_the_caller_already_normalised(tmp_path) -> None:
    """The processor normalises once because it needs the text for the verdict,
    then passed it to every pack to be normalised again. `normalized=True` is
    that caller's promise -- and it defaults to False so a direct caller cannot
    get a silently wrong answer by forgetting."""
    rules = load_pack(tmp_path, [rule(term="blocked phrase")])
    raw = "BLOCKED, PHRASE"

    assert rules.match(raw).matches != ()
    assert rules.match(raw, normalized=True).matches == ()
    assert rules.match(normalize_transcript(raw), normalized=True) == rules.match(raw)


def test_loads_czech_pack_with_diacritics(tmp_path) -> None:
    rules = load_pack(
        tmp_path,
        [rule(rule_id="cs.test", term="zakázaná fráze")],
        language="cs",
    )

    assert rules.match("tohle je ZAKÁZANÁ FRÁZE").matches[0].rule_id == "cs.test"


def test_fuzzy_match_catches_slur_spelling_drift(tmp_path) -> None:
    rules = load_pack(
        tmp_path,
        [
            rule(
                rule_id="en.hate.nword",
                term="nigger",
                category="hate",
                severity=3,
                match="fuzzy",
                variants=["nigga"],
            )
        ],
    )

    verdict = rules.match("that sounded like niggor")

    assert verdict.matches[0].rule_id == "en.hate.nword"
    assert verdict.matches[0].matched_text == "niggor"
    assert verdict.severity == 3


def test_fuzzy_match_keeps_common_near_words_clean(tmp_path) -> None:
    rules = load_pack(
        tmp_path,
        [
            rule(
                rule_id="en.hate.nword",
                term="nigger",
                category="hate",
                severity=3,
                match="fuzzy",
                variants=["nigga"],
            )
        ],
    )

    assert rules.match("bigger night trigger vinegar").matches == ()


def test_fuzzy_match_reuses_precomputed_rule_run_encodings(tmp_path, monkeypatch) -> None:
    import voicesniffer_processor.rules as rules_module

    rules = load_pack(
        tmp_path,
        [
            rule(
                rule_id="en.hate.nword",
                term="nigger",
                category="hate",
                severity=3,
                match="fuzzy",
                variants=["nigga"],
            )
        ],
    )
    original_runs = rules_module._runs
    calls: list[str] = []

    def recording_runs(value: str):
        calls.append(value)
        return original_runs(value)

    monkeypatch.setattr(rules_module, "_runs", recording_runs)

    assert rules.match("nixxxx").matches == ()
    assert calls == ["nixxxx"]


def test_production_english_pack_catches_nword_variants() -> None:
    rules = load_production_pack("en")

    for transcript in (
        "nigger",
        "niggers",
        "nigga",
        "niggas",
        "niggah",
        "nigguh",
        "niggaz",
        "niggor",
        "niggar",
        "niggur",
        "ape niggers",
    ):
        verdict = rules.match(transcript)
        assert verdict.severity == 3
        # Two rules now: `en.hate.nword` keeps the three spellings that are safe
        # to match fuzzily, `en.hate.nword-listed` holds the ones whose fuzzy
        # accept sets reached ordinary words (`niggas` accepted **nagas**, a
        # Twilight Forest boss; `niggar` accepted **Nagar**). Same category, same
        # severity, no obfuscation tolerance on the listed half.
        assert verdict.matches[0].rule_id in {"en.hate.nword", "en.hate.nword-listed"}


@pytest.mark.parametrize(
    "transcript",
    (
        "kill yourself",
        "you should kill yourself now",
        "kys",
        "neck yourself",
    ),
)
def test_production_english_pack_catches_harassment_phrases(transcript: str) -> None:
    verdict = load_production_pack("en").match(transcript)

    assert verdict.severity == 3
    assert verdict.matches[0].rule_id == "en.harassment.self-harm"


def test_production_english_harassment_rule_does_not_match_gameplay() -> None:
    rules = load_production_pack("en")

    assert rules.match("kill the ender dragon yourself").matches == ()


@pytest.mark.parametrize(
    ("transcript", "rule_id"),
    (
        ("buzerant", "cs.hate.homophobic"),
        ("buzeranti", "cs.hate.homophobic"),
        # `buzna`/`buzny` moved to the exact-listed rule: as fuzzy patterns they
        # accepted `bezna`/`bezny`, which is `běžná`/`běžný` as the transcriber
        # writes it once the marks are dropped.
        ("buzna", "cs.hate.homophobic-listed"),
        ("cigoš", "cs.hate.roma"),
        ("cigoši", "cs.hate.roma"),
        ("cigos", "cs.hate.roma"),
    ),
)
def test_production_czech_pack_catches_slurs(transcript: str, rule_id: str) -> None:
    verdict = load_production_pack("cs").match(transcript)

    assert verdict.severity == 3
    assert verdict.matches[0].rule_id == rule_id


@pytest.mark.parametrize(
    "transcript",
    (
        "zabij se",
        "jdi se zabít",
        "jdi se zabit",
        "oběs se",
        "obes se",
        "skoč z mostu",
        "skoc z mostu",
    ),
)
def test_production_czech_pack_catches_harassment_phrases(transcript: str) -> None:
    verdict = load_production_pack("cs").match(transcript)

    assert verdict.severity == 3
    assert verdict.matches[0].rule_id == "cs.harassment.self-harm"


def test_production_czech_harassment_rule_does_not_match_gameplay() -> None:
    rules = load_production_pack("cs")

    assert rules.match("jdi zabít ender draka").matches == ()


def load_production_pack(language: str) -> RulePack:
    rules_path = Path(__file__).resolve().parents[1] / "rules" / f"{language}.yml"
    return RulePack.load(rules_path)


def load_pack(
    directory: Path,
    rules: list[dict],
    *,
    language: str = "en",
    regex_timeout_seconds: float = 0.005,
    max_match_seconds: float = DEFAULT_MAX_MATCH_SECONDS,
    allow: list[str] | None = None,
) -> RulePack:
    path = directory / f"{language}.yml"
    path.write_text(
        yaml_pack(language, rules, allow),
        encoding="utf-8",
    )
    return RulePack.load(
        path,
        regex_timeout_seconds=regex_timeout_seconds,
        max_match_seconds=max_match_seconds,
    )


def yaml_pack(language: str, rules: list[dict], allow: list[str] | None = None) -> str:
    import yaml

    document: dict[str, object] = {"version": 1, "language": language}
    if allow is not None:
        document["allow"] = allow
    document["rules"] = rules
    return yaml.safe_dump(document, allow_unicode=True, sort_keys=False)


def rule(
    *,
    rule_id: str = "en.test.blocked",
    term: str = "blocked phrase",
    category: str = "harassment",
    severity: int = 2,
    match: str = "exact",
    variants: list[str] | None = None,
    context_required: bool = False,
    probe: str | None = None,
) -> dict:
    built = {
        "id": rule_id,
        "term": term,
        "category": category,
        "severity": severity,
        "match": match,
        "variants": [] if variants is None else variants,
        "context_required": context_required,
    }
    # A `regex` rule must carry a probe the loader can verify actually matches
    # once normalised -- regex source is the one pattern type the loader does not
    # normalise, so an unreachable pattern is otherwise indistinguishable from a
    # working one. Tests that build a deliberately invalid regex never reach the
    # check (compilation fails first) and so need no probe.
    if probe is not None:
        built["probe"] = probe
    return built
