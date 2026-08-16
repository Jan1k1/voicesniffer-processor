"""Load the canonical language catalog from languages.toml.

Moderation support is still the rule-pack directory. This module is the
processor-importable view of *status*: Cloud selectability, evidence tier, and
which model a code is associated with. It does not load packs and it does not
treat `auto` as a language.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

ALLOWED_RULE_PACK = frozenset({"none", "draft", "reviewed", "verified"})
ALLOWED_EVIDENCE = frozenset({"unmeasured", "pilot", "verified"})
_CODE_RE = re.compile(r"^[a-z]{2}$")
_ENTRY_KEYS = frozenset(
    {
        "code",
        "name",
        "cloud_base",
        "cloud_selectable",
        "models",
        "rule_pack",
        "evidence",
    }
)


def default_catalog_path() -> Path:
    packaged = Path(__file__).with_name("languages.toml")
    if packaged.is_file():
        return packaged.resolve()
    return (Path(__file__).parents[2] / "languages.toml").resolve()


@dataclass(frozen=True, slots=True)
class Language:
    code: str
    name: str
    cloud_base: bool
    cloud_selectable: bool
    models: tuple[str, ...]
    rule_pack: str
    evidence: str

    def as_json(self) -> dict[str, object]:
        return {
            "code": self.code,
            "name": self.name,
            "cloud_base": self.cloud_base,
            "cloud_selectable": self.cloud_selectable,
            "rule_pack": self.rule_pack,
            "evidence": self.evidence,
            "models": list(self.models),
        }


@dataclass(frozen=True, slots=True)
class LanguageCatalog:
    schema: int
    languages: tuple[Language, ...]

    def by_code(self) -> Mapping[str, Language]:
        return MappingProxyType({entry.code: entry for entry in self.languages})

    def moderation_codes(self) -> frozenset[str]:
        """Codes with a pack status other than none. Matches rules/*.yml."""
        return frozenset(entry.code for entry in self.languages if entry.rule_pack != "none")

    def cloud_selectable_codes(self) -> frozenset[str]:
        return frozenset(entry.code for entry in self.languages if entry.cloud_selectable)

    def cloud_base_code(self) -> str:
        bases = [entry.code for entry in self.languages if entry.cloud_base]
        if len(bases) != 1:
            raise ValueError(f"catalog must have exactly one cloud_base language, not {bases!r}")
        return bases[0]


def load_language_catalog(path: Path | None = None) -> LanguageCatalog:
    catalog_path = path if path is not None else default_catalog_path()
    with catalog_path.open("rb") as catalog_file:
        document = tomllib.load(catalog_file)
    return _parse_catalog(document)


def _parse_catalog(document: object) -> LanguageCatalog:
    if not isinstance(document, dict):
        raise ValueError("language catalog must be a TOML table")
    if set(document) != {"schema", "languages"}:
        raise ValueError("language catalog must contain only schema and languages")
    schema = document["schema"]
    if schema != 1:
        raise ValueError(f"unsupported language catalog schema: {schema!r}")
    raw_languages = document["languages"]
    if not isinstance(raw_languages, list) or not raw_languages:
        raise ValueError("language catalog must contain languages")

    languages: list[Language] = []
    seen: set[str] = set()
    for raw in raw_languages:
        entry = _parse_language(raw)
        if entry.code in seen:
            raise ValueError(f"duplicate language code: {entry.code}")
        if entry.code == "auto":
            raise ValueError("auto is not a language and must not appear in the catalog")
        seen.add(entry.code)
        languages.append(entry)

    bases = [entry.code for entry in languages if entry.cloud_base]
    if bases != ["en"]:
        raise ValueError(f"cloud_base must be exactly English, not {bases!r}")
    for entry in languages:
        if entry.cloud_base and not entry.cloud_selectable:
            raise ValueError(f"{entry.code}: cloud_base language must be cloud_selectable")
        if entry.cloud_selectable and entry.rule_pack != "verified":
            raise ValueError(
                f"{entry.code}: cloud_selectable requires rule_pack=verified in this release"
            )

    languages.sort(key=lambda entry: entry.code)
    return LanguageCatalog(schema=1, languages=tuple(languages))


def _parse_language(raw: object) -> Language:
    if not isinstance(raw, dict):
        raise ValueError("each [[languages]] entry must be a table")
    if set(raw) != _ENTRY_KEYS:
        raise ValueError(f"language entry keys must be {sorted(_ENTRY_KEYS)}, not {sorted(raw)}")
    code = raw["code"]
    name = raw["name"]
    if not isinstance(code, str):
        raise ValueError(f"invalid language code: {code!r}")
    if code == "auto":
        raise ValueError("auto is not a language and must not appear in the catalog")
    if not _CODE_RE.fullmatch(code):
        raise ValueError(f"invalid language code: {code!r}")
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"language {code} needs a non-empty name")
    if not isinstance(raw["cloud_base"], bool) or not isinstance(raw["cloud_selectable"], bool):
        raise ValueError(f"language {code} cloud_base/cloud_selectable must be booleans")
    models = raw["models"]
    if (
        not isinstance(models, list)
        or not models
        or any(not isinstance(item, str) or not item for item in models)
    ):
        raise ValueError(f"language {code} models must be a non-empty list of strings")
    if len(set(models)) != len(models):
        raise ValueError(f"language {code} models must be unique")
    rule_pack = raw["rule_pack"]
    evidence = raw["evidence"]
    if rule_pack not in ALLOWED_RULE_PACK:
        raise ValueError(f"language {code} has invalid rule_pack: {rule_pack!r}")
    if evidence not in ALLOWED_EVIDENCE:
        raise ValueError(f"language {code} has invalid evidence: {evidence!r}")
    return Language(
        code=code,
        name=name.strip(),
        cloud_base=raw["cloud_base"],
        cloud_selectable=raw["cloud_selectable"],
        models=tuple(models),
        rule_pack=rule_pack,
        evidence=evidence,
    )
