"""The false-positive regression suite for the shipped rule packs.

Three layers, deliberately:

**Enumerated.** ``processor/bench/domain-corpus/fp_surface.py`` walks a system
dictionary through the real matcher and prints every ordinary word the packs
accept. Its July output -- ten English words and eight Czech ones, every one of
them a severity-3 hate flag -- is frozen below in carrier sentences. That list
is data, not invention: it is what a dictionary said, not what somebody thought
to try.

**Generated.** A frozen word list can only ever cover the words somebody has
already found. The generated layer takes each rule's own terms and mutates them
-- a consonant swapped, a consonant inserted, a consonant dropped, a vowel
swapped, a letter held down -- and asserts which of those the matcher must and
must not accept. It names no vocabulary of its own, it grows automatically when
a pack grows, and it is what actually pins the *mechanism* rather than a list.

**Recall.** Every term and every variant in both shipped packs must still fire
at its declared severity, and the obfuscations the packs are supposed to catch
must still be caught. A precision fix that quietly costs recall is not a fix, so
the two halves live in one file and run together.

Nothing here spells a slur. The benign strings are place names and ordinary
vocabulary; everything on the must-flag side is read out of the packs at run
time. Full numbers: ``docs/stt-model-selection.md`` section 13.
"""

from pathlib import Path

import pytest

from voicesniffer_processor.rules import (
    FUZZY_BUDGET_LENGTH,
    VOWELS,
    RulePack,
    RulePackError,
    _fuzzy_token_matches,
    _runs,
    normalize_transcript,
)

RULES_DIRECTORY = Path(__file__).resolve().parents[1] / "rules"

# Words the July dictionary sweep found inside the fuzzy rules' accept set, in
# the kind of sentence a player would actually say them in, plus same-shape
# words that did *not* fire -- so the set cannot be satisfied by a matcher that
# has simply stopped matching.
BENIGN_ENGLISH = (
    "we drove up to niagara falls last summer",
    "he moved to nigeria last year",
    "the horse went neigh and bolted",
    "the flight lands in nagoya at six",
    "the naga boss hits really hard",
    "my history teacher is called mister nagy",
    "the ridge is nigher than it looks on the map",
    "the river niger runs through five countries",
    "a nigerian server admin joined the call",
    "that creeper blast was bigger than the last one",
    "pull the trigger on the crossbow already",
    "you need vinegar for the potion recipe",
    "it gets dark at night so light some torches",
    "we should negotiate a trade for the diamonds",
    "the navigator kept us on course",
    "a nightingale was singing outside the base",
)

BENIGN_CZECH = (
    "v tom módu je bizon fakt silný",
    "toho bizona jsem nikdy neviděl",
    "bizoni chodí ve stádu",
    "dej toho bizone pryč",
    "skoč do toho bazénu",
    "kdo mi zaplaví ten bazen",
    "jeli jsme přes bezno na sever",
    "buza je normální slovo",
    "je to bujná fantazie",
    "je to bujna fantazie",
    "bukna není nic zvláštního",
    "cigorka mi nechutná vůbec",
    "cigorky nekupuju",
    "koupil jsem si cigaretu na nádraží",
    "vezmi si buzolu na výlet",
    "je to úplně běžný postup",
    "ten build je fakt bizarní",
    "bazoni tam nikdy nebyli",
    "cagaš se tomu neříká",
)

CONSONANT_SAMPLE = "bdfgklmnprstvz"


def pack(language: str) -> RulePack:
    return RulePack.load(RULES_DIRECTORY / f"{language}.yml")


def fuzzy_patterns(language: str) -> list[str]:
    return [
        pattern
        for rule in pack(language).rules
        if rule.match_type == "fuzzy"
        for pattern in rule.patterns
    ]


def listed(language: str) -> set[str]:
    return {pattern for rule in pack(language).rules for pattern in rule.patterns}


# ---------------------------------------------------------------------------
# Enumerated: the words a dictionary found, which must not flag
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("transcript", BENIGN_ENGLISH)
def test_english_benign_near_misses_do_not_flag(transcript: str) -> None:
    verdict = pack("en").match(transcript)

    assert verdict.severity == 0, [match.rule_id for match in verdict.matches]
    assert verdict.matches == ()


@pytest.mark.parametrize("transcript", BENIGN_CZECH)
def test_czech_benign_near_misses_do_not_flag(transcript: str) -> None:
    verdict = pack("cs").match(transcript)

    assert verdict.severity == 0, [match.rule_id for match in verdict.matches]
    assert verdict.matches == ()


@pytest.mark.parametrize("language", ("en", "cs"))
def test_benign_near_misses_do_not_flag_the_other_language_pack(language: str) -> None:
    """A pack is loaded per language, but a mislabelled utterance must not turn
    into a hate flag either."""
    other = BENIGN_CZECH if language == "en" else BENIGN_ENGLISH

    for transcript in other:
        assert pack(language).match(transcript).severity == 0, transcript


# ---------------------------------------------------------------------------
# Generated: the mechanism, derived from whatever the packs happen to contain
# ---------------------------------------------------------------------------
def accepts(pattern: str, token: str) -> bool:
    """Ask the matcher about **this** pattern rather than the whole pack. A
    mutation of one variant frequently lands inside another variant's legitimate
    accept set -- ``niggor`` -> ``niggos`` is a consonant swap away from one term
    and a vowel swap away from another -- and a pack-level assertion could not
    tell the two apart. The claim being pinned here is about the predicate."""
    return _fuzzy_token_matches(pattern, _runs(pattern), token, _runs(token))


@pytest.mark.parametrize("language", ("en", "cs"))
def test_a_swapped_consonant_is_never_accepted(language: str) -> None:
    """The bug was a matcher that tolerated *any* single edit. A consonant is
    the part of a word that says which word it is, so changing one must end the
    comparison -- that single property is what keeps `Niagara`, `bizon`,
    `nigher` and `bujna` out."""
    for pattern in fuzzy_patterns(language):
        for index, character in enumerate(pattern):
            if character in VOWELS:
                continue
            for replacement in CONSONANT_SAMPLE:
                if replacement == character:
                    continue
                mutated = pattern[:index] + replacement + pattern[index + 1 :]
                assert not accepts(pattern, mutated), (pattern, mutated)


@pytest.mark.parametrize("language", ("en", "cs"))
def test_an_inserted_consonant_is_never_accepted(language: str) -> None:
    for pattern in fuzzy_patterns(language):
        for index in range(len(pattern) + 1):
            for inserted in CONSONANT_SAMPLE:
                # Doubling a letter that is already there is elongation, not a
                # new consonant, and elongation is an obfuscation the rules are
                # meant to catch.
                if inserted in (pattern[index - 1 : index], pattern[index : index + 1]):
                    continue
                mutated = pattern[:index] + inserted + pattern[index:]
                assert not accepts(pattern, mutated), (pattern, mutated)


@pytest.mark.parametrize("language", ("en", "cs"))
def test_a_dropped_consonant_is_never_accepted(language: str) -> None:
    for pattern in fuzzy_patterns(language):
        for index, character in enumerate(pattern):
            if character in VOWELS:
                continue
            # Dropping one of a doubled pair shortens a run rather than removing
            # a consonant; that case is the Niger collision and is covered by
            # the allow-list test below.
            if character in (pattern[index - 1 : index], pattern[index + 1 : index + 2]):
                continue
            mutated = pattern[:index] + pattern[index + 1 :]
            assert not accepts(pattern, mutated), (pattern, mutated)


@pytest.mark.parametrize("language", ("en", "cs"))
def test_a_swapped_vowel_is_still_accepted(language: str) -> None:
    """The other half of the same property: swapping a vowel is exactly the
    drift a slur rule exists to absorb, and tightening the matcher must not have
    taken it away."""
    loaded = pack(language)
    for rule in loaded.rules:
        if rule.match_type != "fuzzy":
            continue
        for pattern in rule.patterns:
            for index, character in enumerate(pattern):
                if character not in VOWELS:
                    continue
                for replacement in VOWELS - {character}:
                    mutated = pattern[:index] + replacement + pattern[index + 1 :]
                    # An allow entry is the documented exception to this rule:
                    # `nagger` is one vowel from a listed term and is an ordinary
                    # English word, so the pack carves it out explicitly rather
                    # than flagging it. The claim being pinned is that vowel
                    # drift is absorbed *unless the pack says otherwise*, which
                    # is what the allow layer exists to say.
                    if mutated in loaded.allow:
                        continue
                    assert loaded.match(mutated).severity >= rule.severity, mutated


@pytest.mark.parametrize("language", ("en", "cs"))
def test_a_held_down_letter_is_still_accepted(language: str) -> None:
    loaded = pack(language)
    for rule in loaded.rules:
        if rule.match_type != "fuzzy":
            continue
        for pattern in rule.patterns:
            for index in range(len(pattern)):
                mutated = pattern[:index] + pattern[index] * 3 + pattern[index + 1 :]
                assert loaded.match(mutated).severity >= rule.severity, mutated


@pytest.mark.parametrize("language", ("en", "cs"))
def test_two_vowels_only_drift_on_terms_long_enough_to_afford_it(language: str) -> None:
    """The budget is the one number in the matcher that is a judgement call, so
    it gets asserted rather than left implicit: five-letter terms tolerate one
    differing run, six and above tolerate two. At two for five-letter terms the
    ordinary Czech words `bezno` and `cagaš` come back."""
    for pattern in fuzzy_patterns(language):
        vowels = [index for index, character in enumerate(pattern) if character in VOWELS]
        if len(vowels) < 2:
            continue
        mutated = list(pattern)
        for index in vowels[:2]:
            mutated[index] = "a" if pattern[index] != "a" else "u"
        candidate = "".join(mutated)
        assert accepts(pattern, candidate) is (len(pattern) >= FUZZY_BUDGET_LENGTH), (
            pattern,
            candidate,
        )


# ---------------------------------------------------------------------------
# Recall: nothing the packs already promised may have been lost
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("language", ("en", "cs"))
def test_every_listed_term_and_variant_still_flags(language: str) -> None:
    """Read out of the pack rather than written down here, so the suite covers
    whatever the packs contain today and whatever is added tomorrow."""
    loaded = pack(language)

    for rule in loaded.rules:
        for pattern in rule.patterns:
            if rule.match_type == "regex":
                continue
            verdict = loaded.match(pattern)
            assert rule.rule_id in {match.rule_id for match in verdict.matches}, pattern
            if not rule.context_required:
                assert verdict.severity >= rule.severity, pattern


@pytest.mark.parametrize("language", ("en", "cs"))
def test_listed_terms_still_flag_inside_ordinary_sentences(language: str) -> None:
    carrier = "okay so {} right" if language == "en" else "tak jo {} no"
    loaded = pack(language)

    for rule in loaded.rules:
        if rule.match_type == "regex" or rule.context_required:
            continue
        for pattern in rule.patterns:
            assert loaded.match(carrier.format(pattern)).severity >= rule.severity, pattern


@pytest.mark.parametrize("language", ("en", "cs"))
def test_the_pack_never_flags_its_own_language_name_or_empty_input(language: str) -> None:
    loaded = pack(language)

    assert loaded.match("").matches == ()
    assert loaded.match("   ").matches == ()
    assert loaded.match(language).matches == ()


# ---------------------------------------------------------------------------
# The allow layer
# ---------------------------------------------------------------------------
def test_allow_suppresses_a_match_that_the_matcher_cannot_refuse() -> None:
    """`Niger` differs from a listed term by one held-down letter, which is the
    one drift the matcher must keep accepting. The pack says so explicitly."""
    assert "niger" in pack("en").allow
    assert pack("en").match("the river niger runs through five countries").matches == ()


def test_allow_does_not_suppress_a_second_hit_elsewhere_in_the_transcript(tmp_path) -> None:
    path = tmp_path / "en.yml"
    path.write_text(
        "version: 1\nlanguage: en\nallow: [safe harbour]\n"
        "rules:\n"
        "  - id: en.test.blocked\n"
        "    term: harbour\n"
        "    category: harassment\n"
        "    severity: 2\n"
        "    match: contains\n"
        "    variants: []\n"
        "    context_required: false\n",
        encoding="utf-8",
    )
    loaded = RulePack.load(path)

    assert loaded.match("we reached safe harbour").matches == ()
    assert loaded.match("safe harbour and then the harbour master").severity == 2


def test_the_regex_timeout_still_applies_on_the_allow_path(tmp_path) -> None:
    """A non-empty allow list swaps `search` for `finditer` so suppressed hits do
    not hide a real one. The 5 ms per-rule budget has to survive that swap."""
    path = tmp_path / "en.yml"
    path.write_text(
        "\n".join(
            (
                "version: 1",
                "language: en",
                "allow: [safe]",
                "rules:",
                "  - id: en.test.blocked",
                "    term: (a+)+$",
                "    category: harassment",
                "    severity: 2",
                "    match: regex",
                "    probe: aaa",
                "    variants: []",
                "    context_required: false",
            )
        ),
        encoding="utf-8",
    )
    verdict = RulePack.load(path, regex_timeout_seconds=0.001).match("a" * 100_000 + "b")

    assert verdict.matches == ()
    assert verdict.timed_out_rule_ids == ("en.test.blocked",)


def test_allow_is_per_language_and_does_not_leak_between_packs() -> None:
    """Both packs carry allow entries now -- Czech gained the `zabij se a
    respawni` carve-out -- so emptiness is no longer the claim. The claim is that
    neither pack can see the other's, which is what the earlier `cs` list being
    empty was really standing in for."""
    assert "niger" not in pack("cs").allow
    assert "nagger" not in pack("cs").allow
    assert not set(pack("cs").allow) & set(pack("en").allow)


@pytest.mark.parametrize(
    "allow",
    ("not-a-list", [""], [123], ["!!!"], ["x" * 257], [None]),
)
def test_rejects_invalid_allow(tmp_path, allow) -> None:
    import yaml

    path = tmp_path / "en.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "language": "en",
                "allow": allow,
                "rules": [
                    {
                        "id": "en.test.blocked",
                        "term": "blocked phrase",
                        "category": "harassment",
                        "severity": 2,
                        "match": "exact",
                        "variants": [],
                        "context_required": False,
                    }
                ],
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RulePackError, match="invalid_allow"):
        RulePack.load(path)


# ---------------------------------------------------------------------------
# Known gaps, asserted so they cannot change silently
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("language", ("en", "cs"))
def test_spacing_and_infixes_still_defeat_exact_phrase_rules(language: str) -> None:
    """Not a bug being introduced here -- a gap section 11.2 measured and this
    change does not touch. An `exact` rule is a fixed token sequence, so one
    filler word between the tokens defeats it. Asserted rather than left
    undiscovered, so whoever fixes phrase matching sees this fail."""
    loaded = pack(language)
    for rule in loaded.rules:
        if rule.match_type != "exact":
            continue
        for pattern in rule.patterns:
            tokens = pattern.split()
            if len(tokens) < 2:
                continue
            with_filler = " ".join([tokens[0], "uh", *tokens[1:]])
            assert loaded.match(with_filler).matches == (), with_filler
            return


# ---------------------------------------------------------------------------
# Evasion the matcher could not see, and the words it still must not reach
# ---------------------------------------------------------------------------
# A digit standing in for the vowel it depicts. Both of these matched nothing in
# any pack: the run comparison refused a digit against a vowel outright.
LEET_MUST_CATCH = (
    ("en", "n1gger", 3),
    ("en", "n1ggers", 3),
    ("en", "n1gg3r", 3),
    ("en", "f4ggot", 3),
    ("en", "f4gg0t", 3),
)

# The mapping is one digit to one vowel, so a digit standing for the *wrong*
# vowel is still a different letter and still refused. This is what stops the
# change from turning every digit into a wildcard vowel, which would put
# ordinary tokens back inside the accept set.
LEET_MUST_REFUSE = (
    "n0gger",
    "n3gger",
    "f1ggot",
    "the seed is 4 4 4 and the coords are 100 64 200",
    "we mined 3 stacks in chunk 12",
)

# Spelled out loud, which is native to voice chat rather than to text: the
# speaker says "kay why ess" and the transcriber writes single letters. Every
# token is one character, so every one of them was skipped.
SPELLED_MUST_CATCH = (
    ("en", "n i g g e r", 3),
    ("en", "k y s", 3),
    ("en", "he told me to k y s in chat", 3),
    ("cs", "k y s", 3),
)

# Ordinary utterances that contain a run of single letters and must stay silent.
SPELLED_MUST_STAY_SILENT = (
    "x y z coordinates please",
    "a b c is how you spell it",
    "the sign says a b and then c",
    "i saw a l a n join the server",
    "g g w p everyone",
)


@pytest.mark.parametrize(("language", "token", "severity"), LEET_MUST_CATCH)
def test_a_digit_standing_for_its_own_vowel_is_caught(
    language: str, token: str, severity: int
) -> None:
    assert pack(language).match(token).severity == severity, token


@pytest.mark.parametrize("transcript", LEET_MUST_REFUSE)
@pytest.mark.parametrize("language", ("en", "cs"))
def test_a_digit_standing_for_a_different_vowel_is_still_refused(
    language: str, transcript: str
) -> None:
    verdict = pack(language).match(transcript)

    assert verdict.matches == (), [match.rule_id for match in verdict.matches]


@pytest.mark.parametrize(("language", "transcript", "severity"), SPELLED_MUST_CATCH)
def test_a_word_spelled_out_loud_reaches_the_same_rule(
    language: str, transcript: str, severity: int
) -> None:
    assert pack(language).match(transcript).severity == severity, transcript


@pytest.mark.parametrize("transcript", SPELLED_MUST_STAY_SILENT)
@pytest.mark.parametrize("language", ("en", "cs"))
def test_joining_a_run_of_single_letters_does_not_invent_matches(
    language: str, transcript: str
) -> None:
    verdict = pack(language).match(transcript)

    assert verdict.matches == (), [match.rule_id for match in verdict.matches]


def test_the_de_spelled_pass_reports_each_rule_once() -> None:
    """Both passes see ``kys``, and a staff member must not read the same rule
    twice for one utterance."""
    verdict = pack("en").match("kys k y s")

    rule_ids = [match.rule_id for match in verdict.matches]
    assert rule_ids.count("en.harassment.self-harm") == 1, rule_ids


def test_the_de_spelled_pass_costs_nothing_when_nobody_spelled_anything() -> None:
    """The second pass is skipped unless joining actually changed the text, which
    is every ordinary utterance."""
    from voicesniffer_processor.rules import _join_spelled_out

    for transcript in (*BENIGN_ENGLISH, *BENIGN_CZECH):
        normalised = normalize_transcript(transcript)
        assert _join_spelled_out(normalised) == normalised, transcript


@pytest.mark.parametrize("language", ("en", "cs"))
def test_pack_terms_are_written_in_normalised_form(language: str) -> None:
    """The matcher compares against ``normalize_transcript`` output, so a term
    written with punctuation or capitals silently never fires. Cheap to check,
    and it is the failure mode a pack author would not notice."""
    for rule in pack(language).rules:
        if rule.match_type == "regex":
            continue
        for pattern in rule.patterns:
            assert normalize_transcript(pattern) == pattern
