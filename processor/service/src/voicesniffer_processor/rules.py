from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import regex
import yaml

CATEGORIES = frozenset({"harassment", "hate", "sexual", "threat", "spam"})
MATCH_TYPES = frozenset({"exact", "contains", "regex", "fuzzy"})
LANGUAGE_PATTERN = re.compile(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*\Z")
RULE_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
MAX_RULES = 1_000
MAX_VARIANTS = 64
MAX_PATTERN_LENGTH = 256
MAX_ALLOW = 512
# Ceiling on the time one pack may spend on one transcript. Both shipped packs
# together cost 1.3 ms on an 879-character transcript, so this is ~190x measured
# headroom -- it exists for the pathological case the per-rule regex budget does
# not bound (see RulePack.match), not to trim normal work.
DEFAULT_MAX_MATCH_SECONDS = 0.25

# ---------------------------------------------------------------------------
# Fuzzy matching
#
# A ``fuzzy`` rule has to accept the ways a term drifts -- a vowel swapped for
# effect, a letter held down, a transcriber hearing the wrong vowel -- without
# accepting *different words*. The first version compared a vowel-stripped
# "phonetic key", which made ``nigger``, ``Niagara``, ``Nigeria``, ``negro``
# and ``neigh`` all collapse to ``ngr``/``ng``: ten ordinary English words and
# eight Czech ones fired a severity-3 hate rule (``docs/stt-model-selection.md``
# section 13). Deleting the vowels deleted the evidence that distinguishes the
# words.
#
# What replaces it compares the *run structure* of the two strings: the
# sequence of maximal same-letter runs. A token matches when it has the same
# number of runs in the same order and at most ``budget`` of those runs differ,
# where a differing run may only differ in **how long it is** (``nigga`` ->
# ``niiigga``) or in **which vowel it is** (``nigga`` -> ``niggo``). A
# consonant that is simply a different consonant ends the comparison, which is
# what keeps ``Niagara`` and ``bizon`` out: their consonant frames do not line
# up with anything the packs list.
#
# The budget scales with the term because a longer term makes an accidental
# collision far less likely: five-letter terms tolerate one differing run, six
# and above tolerate two. That threshold is measured, not guessed -- at two for
# five-letter terms the Czech words ``bezno`` and ``cagaš`` come back.
FUZZY_MIN_LENGTH = 4
FUZZY_BUDGET_LENGTH = 6
FUZZY_MAX_PADDING = 6
VOWELS = frozenset("aeiouy")

# The one obfuscation the run comparison above refused for a reason that does not
# survive inspection: a digit standing in for the vowel it is drawn to look like.
# ``n1gger`` and ``f4ggot`` matched nothing anywhere, because a digit is not a
# vowel and a differing run may only swap one vowel for another.
#
# The mapping is deliberately *tight*. A digit is accepted only against the exact
# vowel it depicts -- ``1`` for ``i``, never for ``a`` -- so this does not widen
# the accept set the way treating every digit as a wildcard vowel would. The
# token still has to carry the same number of runs in the same order with the
# same consonants, which is what keeps ``Nigeria`` and ``chunk`` out, and the
# only tokens it newly reaches are tokens containing a digit inside a word.
LEET_VOWELS = {"0": "o", "1": "i", "3": "e", "4": "a"}

# Spelling a word out loud is the other evasion the matcher could not see, and
# unlike leetspeak it is native to voice chat: a player says "kay why ess" and
# the transcriber writes ``k y s``. Every token is one character, so ``_match_fuzzy``
# skips all of them (they are under ``FUZZY_MIN_LENGTH``) and no exact pattern is
# one character long either. The whole utterance matched nothing.
#
# ``_join_spelled_out`` collapses a run of single-letter tokens back into one
# token and the pack is matched a second time against that text. Three is the
# threshold because ``kys`` is three letters; below three the risk is real, since
# ``i`` and ``a`` are ordinary English words and Slavic prepositions are single
# letters, and at three or more a run of them is somebody spelling.
SPELLED_OUT_MIN_LETTERS = 3


class RulePackError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class RuleMatch:
    rule_id: str
    category: str
    severity: int
    matched_text: str
    context_required: bool


@dataclass(frozen=True, slots=True)
class RuleVerdict:
    matches: tuple[RuleMatch, ...]
    severity: int
    timed_out_rule_ids: tuple[str, ...]
    # Rules never evaluated because the pack-wide budget ran out. Distinct from
    # ``timed_out_rule_ids``, which is one rule exceeding its own regex budget.
    skipped_rule_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Rule:
    rule_id: str
    category: str
    severity: int
    match_type: str
    patterns: tuple[str, ...]
    pattern_runs: tuple[tuple[tuple[str, int], ...], ...]
    compiled_patterns: tuple[regex.Pattern, ...]
    context_required: bool


@dataclass(frozen=True, slots=True)
class RulePack:
    language: str
    rules: tuple[_Rule, ...]
    regex_timeout_seconds: float
    allow: tuple[str, ...] = ()
    max_match_seconds: float = DEFAULT_MAX_MATCH_SECONDS

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        regex_timeout_seconds: float = 0.005,
        max_match_seconds: float = DEFAULT_MAX_MATCH_SECONDS,
    ) -> RulePack:
        if regex_timeout_seconds <= 0:
            raise RulePackError("invalid_regex_timeout")
        if max_match_seconds <= 0:
            raise RulePackError("invalid_max_match_seconds")
        try:
            document = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exception:
            raise RulePackError("invalid_yaml") from exception
        if not isinstance(document, dict) or document.get("version") != 1:
            raise RulePackError("invalid_pack")

        language = document.get("language")
        if not isinstance(language, str) or LANGUAGE_PATTERN.fullmatch(language) is None:
            raise RulePackError("invalid_language")
        raw_rules = document.get("rules")
        if not isinstance(raw_rules, list) or not raw_rules or len(raw_rules) > MAX_RULES:
            raise RulePackError("invalid_rules")

        rules: list[_Rule] = []
        rule_ids: set[str] = set()
        for raw_rule in raw_rules:
            parsed_rule = _parse_rule(raw_rule)
            if parsed_rule.rule_id in rule_ids:
                raise RulePackError("duplicate_rule_id")
            rule_ids.add(parsed_rule.rule_id)
            rules.append(parsed_rule)

        allow = _parse_allow(document)
        # An allow entry identical to a listed pattern suppresses every
        # occurrence of it, so the rule stops existing while still looking
        # present in the file. Containment is fine and is what the mechanism is
        # for -- `kill yourself to respawn` carving out `kill yourself` -- but
        # equality is always a mistake.
        listed = {pattern for rule in rules for pattern in rule.patterns}
        if shadowed := listed & set(allow):
            raise RulePackError(f"allow_shadows_rule:{sorted(shadowed)[0]}")
        return cls(language, tuple(rules), regex_timeout_seconds, allow, max_match_seconds)

    def match(self, transcript: str, *, normalized: bool = False) -> RuleVerdict:
        """``normalized=True`` promises the caller already ran
        ``normalize_transcript``, which the processor does before it needs the
        text for the verdict. Default stays False so a direct caller cannot get
        a silently wrong answer by forgetting."""
        normalized_text = transcript if normalized else normalize_transcript(transcript)
        if not normalized_text:
            return RuleVerdict((), 0, ())
        verdict = self._match_text(normalized_text)
        despelled = _join_spelled_out(normalized_text)
        if despelled == normalized_text:
            return verdict
        # Somebody spelled something out. Match the joined form too and merge, so
        # that ``k y s`` reaches the same rule ``kys`` does without changing what
        # any rule means. One rule reports once, from whichever pass saw it first.
        return _merge(verdict, self._match_text(despelled))

    def _match_text(self, normalized_text: str) -> RuleVerdict:
        allowed = _allowed_spans(normalized_text, self.allow)

        matches: list[RuleMatch] = []
        timed_out_rule_ids: list[str] = []
        skipped_rule_ids: list[str] = []
        # The per-rule regex budget bounds one rule, not the pack: at the schema
        # limits a single rule may hold 65 patterns, so 65 near-timeout searches
        # cost ~325 ms without ever raising, and eight such rules exceed the
        # plugin's whole 2.5 s request budget. This is the bound on the total.
        deadline = time.perf_counter() + self.max_match_seconds
        for index, rule in enumerate(self.rules):
            if time.perf_counter() >= deadline:
                skipped_rule_ids.extend(later.rule_id for later in self.rules[index:])
                break
            try:
                matched_text = _match_rule(
                    rule, normalized_text, self.regex_timeout_seconds, allowed
                )
            except TimeoutError:
                timed_out_rule_ids.append(rule.rule_id)
                continue
            if matched_text is None:
                continue
            matches.append(
                RuleMatch(
                    rule_id=rule.rule_id,
                    category=rule.category,
                    severity=rule.severity,
                    matched_text=matched_text,
                    context_required=rule.context_required,
                )
            )

        severity = max(
            (match.severity for match in matches if not match.context_required),
            default=0,
        )
        return RuleVerdict(
            tuple(matches),
            severity,
            tuple(timed_out_rule_ids),
            tuple(skipped_rule_ids),
        )


def _merge(first: RuleVerdict, second: RuleVerdict) -> RuleVerdict:
    seen = {match.rule_id for match in first.matches}
    matches = (*first.matches, *(m for m in second.matches if m.rule_id not in seen))
    severity = max(
        (match.severity for match in matches if not match.context_required),
        default=0,
    )
    return RuleVerdict(
        matches,
        severity,
        tuple(dict.fromkeys((*first.timed_out_rule_ids, *second.timed_out_rule_ids))),
        tuple(dict.fromkeys((*first.skipped_rule_ids, *second.skipped_rule_ids))),
    )


def _join_spelled_out(transcript: str) -> str:
    """Collapse every run of ``SPELLED_OUT_MIN_LETTERS`` or more single-letter
    tokens into one token. Digits are excluded: a coordinate is not a spelling."""
    joined: list[str] = []
    run: list[str] = []
    for token in transcript.split(" "):
        if len(token) == 1 and token.isalpha():
            run.append(token)
            continue
        if len(run) >= SPELLED_OUT_MIN_LETTERS:
            joined.append("".join(run))
        else:
            joined.extend(run)
        run.clear()
        joined.append(token)
    if len(run) >= SPELLED_OUT_MIN_LETTERS:
        joined.append("".join(run))
    else:
        joined.extend(run)
    return " ".join(joined)


def normalize_transcript(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    alphanumeric = "".join(character if character.isalnum() else " " for character in normalized)
    return " ".join(alphanumeric.split())


def _parse_allow(document: dict) -> tuple[str, ...]:
    """Per-language benign terms that suppress any rule match falling inside
    them. This is the escape hatch for the cases a matcher cannot be expected to
    resolve on its own -- a real word that genuinely *is* a listed term with a
    letter held down, ``Niger`` against the slur being the canonical one. It
    replaces a hard-coded ``FUZZY_DENY_TOKENS`` set that lived in this module,
    applied to English only, and was invisible to whoever wrote the Czech pack.

    It is deliberately a last resort. A word list can never be complete, so
    every entry here is an admission that the matching rule above could not tell
    two words apart; the packs should stay close to empty."""
    raw_allow = document.get("allow", [])
    if not isinstance(raw_allow, list) or len(raw_allow) > MAX_ALLOW:
        raise RulePackError("invalid_allow")
    allow: list[str] = []
    for entry in raw_allow:
        if not isinstance(entry, str) or not entry or len(entry) > MAX_PATTERN_LENGTH:
            raise RulePackError("invalid_allow")
        normalized = normalize_transcript(entry)
        if not normalized:
            raise RulePackError("invalid_allow")
        allow.append(normalized)
    return tuple(allow)


def _allowed_spans(transcript: str, allow: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    """Token-aligned occurrences of every allowed term, as character spans."""
    if not allow:
        return ()
    spans: list[tuple[int, int]] = []
    for term in allow:
        start = 0
        while (index := transcript.find(term, start)) >= 0:
            end = index + len(term)
            if (index == 0 or transcript[index - 1] == " ") and (
                end == len(transcript) or transcript[end] == " "
            ):
                spans.append((index, end))
            start = index + 1
    return tuple(spans)


def _is_allowed(start: int, end: int, allowed: tuple[tuple[int, int], ...]) -> bool:
    return any(
        allowed_start <= start and end <= allowed_end for allowed_start, allowed_end in allowed
    )


def _parse_rule(raw_rule: Any) -> _Rule:
    if not isinstance(raw_rule, dict):
        raise RulePackError("invalid_rule")
    rule_id = raw_rule.get("id")
    category = raw_rule.get("category")
    severity = raw_rule.get("severity")
    match_type = raw_rule.get("match")
    term = raw_rule.get("term")
    variants = raw_rule.get("variants", [])
    context_required = raw_rule.get("context_required", False)
    if not isinstance(rule_id, str) or RULE_ID_PATTERN.fullmatch(rule_id) is None:
        raise RulePackError("invalid_rule_id")
    if category not in CATEGORIES:
        raise RulePackError("invalid_category")
    if isinstance(severity, bool) or severity not in (1, 2, 3):
        raise RulePackError("invalid_severity")
    if match_type not in MATCH_TYPES:
        raise RulePackError("invalid_match_type")
    if not isinstance(term, str) or not term or len(term) > MAX_PATTERN_LENGTH:
        raise RulePackError("invalid_term")
    if not isinstance(variants, list) or len(variants) > MAX_VARIANTS:
        raise RulePackError("invalid_variants")
    if not all(
        isinstance(variant, str) and variant and len(variant) <= MAX_PATTERN_LENGTH
        for variant in variants
    ):
        raise RulePackError("invalid_variants")
    if not isinstance(context_required, bool):
        raise RulePackError("invalid_context_required")

    source_patterns = (term, *variants)
    if match_type == "regex":
        compiled_patterns = tuple(_compile_pattern(pattern) for pattern in source_patterns)
        patterns = source_patterns
        _check_regex_reachable(raw_rule, compiled_patterns)
    else:
        patterns = tuple(normalize_transcript(pattern) for pattern in source_patterns)
        if any(not pattern for pattern in patterns):
            raise RulePackError("invalid_term")
        if match_type == "fuzzy":
            _check_fuzzy_patterns(patterns)
        compiled_patterns = ()
    pattern_runs = tuple(_runs(pattern) for pattern in patterns) if match_type == "fuzzy" else ()
    return _Rule(
        rule_id=rule_id,
        category=category,
        severity=severity,
        match_type=match_type,
        patterns=patterns,
        pattern_runs=pattern_runs,
        compiled_patterns=compiled_patterns,
        context_required=context_required,
    )


def _check_fuzzy_patterns(patterns: tuple[str, ...]) -> None:
    """Refuse the two fuzzy patterns that can never match anything.

    ``_match_fuzzy`` walks ``transcript.split(" ")`` and skips any token shorter
    than ``FUZZY_MIN_LENGTH`` before comparing, so a pattern containing a space
    is never asked about and a pattern under four characters is only ever
    compared against tokens that were already skipped. Both load clean, both look
    like working rules, and neither can produce a match. Loudly at load time is
    the only place to catch it that does not depend on somebody running a test."""
    for pattern in patterns:
        if " " in pattern:
            raise RulePackError(f"fuzzy_pattern_not_single_token:{pattern}")
        if len(pattern) < FUZZY_MIN_LENGTH:
            raise RulePackError(f"fuzzy_pattern_too_short:{pattern}")


def _check_regex_reachable(raw_rule: dict, compiled_patterns: tuple[regex.Pattern, ...]) -> None:
    """A ``regex`` rule is matched against text that has already been through
    ``normalize_transcript``: alphanumeric, single-spaced. A pattern whose
    literals include punctuation -- an apostrophe, a hyphen, an escaped dot --
    therefore cannot match anything, and unlike the other match types the loader
    does *not* normalise regex source, so nothing else notices. Capitals are
    *not* a problem: ``_compile_pattern`` uses IGNORECASE | FULLCASE.

    Static analysis cannot decide reachability here (``\\W`` and ``\\S`` need
    their capitals, ``\\-`` is a legitimate escape), so the rule carries the
    proof instead: a
    ``probe`` string that must actually match once normalised. It is required
    rather than optional because an unreachable regex is indistinguishable from a
    working one until an incident, and both shipped packs currently use no regex
    rules at all -- so the cost of requiring it is zero today and the cost of
    adding it later is a schema migration."""
    probe = raw_rule.get("probe")
    if not isinstance(probe, str) or not probe:
        raise RulePackError("regex_rule_needs_probe")
    normalized_probe = normalize_transcript(probe)
    if not normalized_probe:
        raise RulePackError("regex_probe_normalises_to_nothing")
    if not any(compiled.search(normalized_probe) for compiled in compiled_patterns):
        raise RulePackError(f"regex_probe_does_not_match:{normalized_probe}")


def _compile_pattern(pattern: str) -> regex.Pattern:
    try:
        compiled = regex.compile(pattern, regex.VERSION1 | regex.IGNORECASE | regex.FULLCASE)
    except regex.error as exception:
        raise RulePackError("invalid_regex") from exception
    if compiled.search("") is not None:
        raise RulePackError("empty_regex")
    return compiled


def _match_rule(
    rule: _Rule,
    transcript: str,
    timeout: float,
    allowed: tuple[tuple[int, int], ...],
) -> str | None:
    if rule.match_type == "regex":
        for compiled in rule.compiled_patterns:
            if not allowed:
                match = compiled.search(transcript, timeout=timeout)
                if match is not None:
                    return match.group()
                continue
            for match in compiled.finditer(transcript, timeout=timeout):
                if not _is_allowed(match.start(), match.end(), allowed):
                    return match.group()
        return None
    if rule.match_type == "contains":
        for pattern in rule.patterns:
            start = 0
            while (index := transcript.find(pattern, start)) >= 0:
                end = index + len(pattern)
                if not _is_allowed(index, end, allowed):
                    return transcript[index:end]
                start = index + 1
        return None
    if rule.match_type == "fuzzy":
        return _match_fuzzy(rule.patterns, rule.pattern_runs, transcript, allowed)
    for pattern in rule.patterns:
        start = 0
        while (index := transcript.find(pattern, start)) >= 0:
            end = index + len(pattern)
            left_boundary = index == 0 or transcript[index - 1].isspace()
            right_boundary = end == len(transcript) or transcript[end].isspace()
            if left_boundary and right_boundary and not _is_allowed(index, end, allowed):
                return transcript[index:end]
            start = index + 1
    return None


def _match_fuzzy(
    patterns: tuple[str, ...],
    pattern_runs: tuple[tuple[tuple[str, int], ...], ...],
    transcript: str,
    allowed: tuple[tuple[int, int], ...],
) -> str | None:
    # ``normalize_transcript`` guarantees single-space separation, so token
    # offsets come for free and the run encoding is built once per token rather
    # than once per token per pattern.
    offset = 0
    for token in transcript.split(" "):
        start = offset
        offset += len(token) + 1
        if len(token) < FUZZY_MIN_LENGTH:
            continue
        if _is_allowed(start, start + len(token), allowed):
            continue
        token_runs = _runs(token)
        for pattern, runs in zip(patterns, pattern_runs, strict=True):
            if _fuzzy_token_matches(pattern, runs, token, token_runs):
                return token
    return None


def _fuzzy_token_matches(
    pattern: str,
    pattern_runs: tuple[tuple[str, int], ...],
    token: str,
    token_runs: tuple[tuple[str, int], ...],
) -> bool:
    """The token is the pattern with at most ``budget`` runs altered, and an
    altered run may only change its length or swap one vowel for another."""
    if token == pattern:
        return True
    if len(pattern) < FUZZY_MIN_LENGTH or len(token) < FUZZY_MIN_LENGTH:
        return False
    if len(token) > len(pattern) + FUZZY_MAX_PADDING:
        return False
    if len(pattern_runs) != len(token_runs):
        return False
    budget = 2 if len(pattern) >= FUZZY_BUDGET_LENGTH else 1
    changes = 0
    for (pattern_character, pattern_count), (token_character, token_count) in zip(
        pattern_runs, token_runs, strict=True
    ):
        if pattern_character == token_character:
            if pattern_count == token_count:
                continue
        elif not (_is_vowel(pattern_character) and _is_vowel(token_character)) and not _is_leet_for(
            token_character, pattern_character
        ):
            return False
        changes += 1
        if changes > budget:
            return False
    return True


def _runs(value: str) -> tuple[tuple[str, int], ...]:
    runs: list[tuple[str, int]] = []
    previous = ""
    count = 0
    for character in value:
        if character == previous:
            count += 1
            continue
        if previous:
            runs.append((previous, count))
        previous = character
        count = 1
    if previous:
        runs.append((previous, count))
    return tuple(runs)


@lru_cache(maxsize=4096)
def _is_leet_for(token_character: str, pattern_character: str) -> bool:
    """Is the token's character a digit drawn to look like *this* vowel?

    Tight on purpose: ``1`` stands for ``i`` and for nothing else. A wildcard
    digit would let one token satisfy any vowel position and widen an accept set
    whose narrowness is the only thing keeping ordinary words out."""
    depicted = LEET_VOWELS.get(token_character)
    if depicted is None:
        return False
    return depicted == unicodedata.normalize("NFKD", pattern_character)[0]


@lru_cache(maxsize=4096)
def _is_vowel(character: str) -> bool:
    """Czech writes the same vowel several ways -- ``i``/``í``, ``e``/``ě`` --
    and the transcriber drops the mark. Ask about the base letter so a dropped
    accent counts as one vowel change rather than a different letter. Consonant
    marks are deliberately *not* folded: ``ž`` and ``z`` stay distinct, because
    folding them puts the ordinary Czech word ``běžný`` on top of a pack term."""
    return unicodedata.normalize("NFKD", character)[0] in VOWELS
