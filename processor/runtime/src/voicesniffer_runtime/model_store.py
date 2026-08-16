from __future__ import annotations

import errno
import hashlib
import os
import re
import shutil
import tarfile
import time
import tomllib
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any
from urllib.request import Request, urlopen

_FAMILY_FILES = {
    "moonshine-v2": frozenset({"encoder", "decoder", "tokens"}),
    "nemo-transducer": frozenset({"encoder", "decoder", "joiner", "tokens"}),
    "whisper": frozenset({"encoder", "decoder", "tokens"}),
}
# The 24 official EU languages except Irish, plus Russian and Ukrainian: exactly
# what parakeet-tdt-0.6b-v3 was trained on, per NVIDIA's model card. Irish is in
# the EU set but not the model, so it is not here.
#
# Turkish is deliberately absent even though the model's tokens.txt contains a
# `<|tr|>` tag. That file carries the whole ISO 639-1 tag block inherited from
# NeMo's unified vocabulary -- 183 tags including Afar and Abkhaz -- and reflects
# the tokenizer, not the training set. Reading it as evidence of Turkish support
# is the obvious mistake to make here.
_LANGUAGES = frozenset({
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu",
    "it", "lt", "lv", "mt", "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
})
# 1: archive-only. 2: adds the base_url + downloads form, which pins every file
# individually. Both shapes load under either number; the bump exists so an old
# reader fails loudly on a registry it cannot fully verify.
_SCHEMAS = (1, 2)
_MODEL_ID = re.compile(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?")
_SHA256 = re.compile(r"[0-9a-fA-F]{64}")


def default_registry_path() -> Path:
    packaged = Path(__file__).with_name("models.toml")
    if packaged.is_file():
        return packaged.resolve()
    return (Path(__file__).parents[2] / "models.toml").resolve()


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One pinned model, fetched in one of two shapes.

    **Archive** (``archive_url`` + ``archive_sha256`` + ``archive_root``) is the
    original shape: one tarball, one digest.

    **Loose files** (``base_url`` + ``downloads``) exists because not every
    upstream publishes a tarball. `parakeet-tdt-0.6b-v3` fp32 does not, and it
    cannot be re-published as one either -- the archive is 2.21 GiB, over
    GitHub's 2 GiB per-asset ceiling, and xz only takes it to ~2.19 GiB. The
    loose form pins **every** file individually against a URL that names an
    immutable revision, which is strictly stronger than the archive form: the
    archive shape can only pin files named in ``files``, so ONNX external data
    (`encoder.weights`, 2.4 GB of the actual model) rides along unverified.
    """

    model_id: str
    family: str
    languages: tuple[str, ...]
    license_name: str
    license_url: str
    files: Mapping[str, str]
    archive_url: str = ""
    archive_sha256: str = ""
    archive_root: str = ""
    base_url: str = ""
    downloads: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not _MODEL_ID.fullmatch(self.model_id):
            raise ValueError(f"invalid model id: {self.model_id!r}")
        required_files = _FAMILY_FILES.get(self.family)
        if required_files is None:
            raise ValueError(f"unsupported model family: {self.family!r}")
        if not self.languages or len(set(self.languages)) != len(self.languages):
            raise ValueError("languages must be unique and non-empty")
        if unsupported := set(self.languages) - _LANGUAGES:
            raise ValueError(f"unsupported languages: {', '.join(sorted(unsupported))}")
        if set(self.files) != required_files:
            raise ValueError(f"{self.family} requires files: {', '.join(sorted(required_files))}")
        for role, relative_path in self.files.items():
            _relative_path(relative_path, f"files.{role}")
        object.__setattr__(self, "files", MappingProxyType(dict(self.files)))
        if not self.license_name or not self.license_url:
            raise ValueError("model license fields must be non-empty")
        if bool(self.archive_url) == bool(self.base_url):
            raise ValueError("a model needs exactly one of archive_url or base_url")
        if self.archive_url:
            self._check_archive()
        else:
            self._check_downloads()

    def _check_archive(self) -> None:
        if self.downloads:
            raise ValueError("downloads belongs to the base_url form")
        if not _SHA256.fullmatch(self.archive_sha256):
            raise ValueError("archive_sha256 must be a 64-character hexadecimal digest")
        object.__setattr__(self, "archive_sha256", self.archive_sha256.lower())
        _relative_path(self.archive_root, "archive_root")

    def _check_downloads(self) -> None:
        if self.archive_sha256 or self.archive_root:
            raise ValueError("archive_sha256 and archive_root belong to the archive form")
        if not self.downloads:
            raise ValueError("base_url requires a non-empty downloads table")
        # The digest is the integrity guarantee, not the scheme -- but there is
        # no reason for a shipped registry to name plain http, and refusing it
        # keeps a downgrade out of the file. ``file:`` stays allowed so a
        # mirror, or a test, can be a directory.
        if not self.base_url.startswith(("https://", "file:")):
            raise ValueError("base_url must be https (or a file: mirror)")
        seen: set[str] = set()
        normalised: list[tuple[str, str]] = []
        for relative_path, digest in self.downloads:
            _relative_path(relative_path, "downloads.path")
            if relative_path in seen:
                raise ValueError(f"duplicate download path: {relative_path}")
            seen.add(relative_path)
            if not _SHA256.fullmatch(digest):
                raise ValueError(f"downloads.{relative_path} needs a SHA-256 digest")
            normalised.append((relative_path, digest.lower()))
        # Every file the loader will open must be one of the files we verified.
        if missing := set(self.files.values()) - seen:
            raise ValueError(f"downloads must cover {', '.join(sorted(missing))}")
        object.__setattr__(self, "downloads", tuple(normalised))

    @property
    def installation_key(self) -> str:
        """Content address of the whole pinned set, so a changed pin installs
        beside the old one rather than over it."""
        if self.archive_url:
            return self.archive_sha256[:16]
        digest = hashlib.sha256()
        for relative_path, file_digest in sorted(self.downloads):
            digest.update(f"{relative_path}\0{file_digest}\n".encode())
        return digest.hexdigest()[:16]

    def supports_language(self, language: str) -> bool:
        return language in self.languages


def load_registry(path: Path) -> dict[str, ModelSpec]:
    with path.open("rb") as registry_file:
        document = tomllib.load(registry_file)
    if set(document) != {"schema", "models"} or document["schema"] not in _SCHEMAS:
        raise ValueError(f"model registry must use schema {' or '.join(map(str, _SCHEMAS))}")
    raw_models = document["models"]
    if not isinstance(raw_models, list) or not raw_models:
        raise ValueError("model registry must contain models")
    registry: dict[str, ModelSpec] = {}
    for raw_model in raw_models:
        spec = _parse_model(raw_model)
        if spec.model_id in registry:
            raise ValueError(f"duplicate model id: {spec.model_id}")
        registry[spec.model_id] = spec
    return registry


class ModelStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def fetch(self, spec: ModelSpec) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        key = spec.installation_key
        installation = self.root / spec.model_id / key
        if installation.exists():
            _validate_installation(installation, spec)
            return installation
        lock = self.root / ".locks" / f"{spec.model_id}-{key}.lock"
        with _exclusive_file_lock(lock):
            if installation.exists():
                _validate_installation(installation, spec)
                return installation
            staging = self.root / f".extract-{spec.model_id}-{uuid.uuid4().hex}"
            staging.mkdir()
            try:
                if spec.archive_url:
                    archive = self._fetch_archive(spec)
                    _extract_archive(archive, staging)
                    prepared = staging / _platform_path(spec.archive_root)
                else:
                    self._fetch_files(spec, staging)
                    prepared = staging
                _validate_installation(prepared, spec)
                if installation.exists():
                    _validate_installation(installation, spec)
                    return installation
                installation.parent.mkdir(parents=True, exist_ok=True)
                os.replace(prepared, installation)
                return installation
            finally:
                shutil.rmtree(staging, ignore_errors=True)

    def _fetch_files(self, spec: ModelSpec, destination: Path) -> None:
        """Download each pinned file straight into the staging directory.

        Reuses whatever a previous attempt already verified, so a 2.4 GB weights
        file is not re-fetched because a 90 KB token list failed."""
        cache = self.root / ".files" / spec.model_id
        cache.mkdir(parents=True, exist_ok=True)
        for relative_path, digest in spec.downloads:
            cached = cache / f"{digest}-{PurePosixPath(relative_path).name}"
            if not cached.is_file() or _file_sha256(cached) != digest:
                cached.unlink(missing_ok=True)
                self._download_verified(
                    f"{spec.base_url.rstrip('/')}/{relative_path}", digest, cached
                )
            target = destination / _platform_path(relative_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            # Hard link rather than copy: the resume cache would otherwise cost a
            # second 2.4 GB on disk for the encoder weights alone. Model files are
            # never written after install, so sharing the inode is safe. Falls
            # back to a copy when the store spans devices or the filesystem has
            # no links.
            try:
                os.link(cached, target)
            except OSError:
                shutil.copyfile(cached, target)

    @staticmethod
    def _download_verified(url: str, digest: str, destination: Path) -> None:
        partial = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.partial")
        running = hashlib.sha256()
        request = Request(url, headers={"User-Agent": "VoiceSniffer/0.1"})
        try:
            with urlopen(request, timeout=60) as response, partial.open("wb") as sink:
                while chunk := response.read(1024 * 1024):
                    sink.write(chunk)
                    running.update(chunk)
            if running.hexdigest() != digest:
                raise ValueError(f"SHA-256 mismatch for {url}")
            os.replace(partial, destination)
        finally:
            partial.unlink(missing_ok=True)

    def _fetch_archive(self, spec: ModelSpec) -> Path:
        archives = self.root / ".archives"
        archives.mkdir(exist_ok=True)
        archive = archives / f"{spec.model_id}-{spec.archive_sha256[:12]}.tar.bz2"
        if archive.is_file() and _file_sha256(archive) == spec.archive_sha256:
            return archive
        archive.unlink(missing_ok=True)
        partial = archives / f"{archive.name}.{uuid.uuid4().hex}.partial"
        digest = hashlib.sha256()
        request = Request(spec.archive_url, headers={"User-Agent": "VoiceSniffer/0.1"})
        try:
            with urlopen(request, timeout=60) as response, partial.open("wb") as destination:
                while chunk := response.read(1024 * 1024):
                    destination.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != spec.archive_sha256:
                raise ValueError(f"SHA-256 mismatch for {spec.model_id}")
            os.replace(partial, archive)
            return archive
        finally:
            partial.unlink(missing_ok=True)


@contextmanager
def _exclusive_file_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as lock_file:
        lock_file.seek(0, os.SEEK_END)
        if lock_file.tell() == 0:
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as error:
                    if error.errno not in {errno.EACCES, errno.EAGAIN}:
                        raise
                    time.sleep(0.05)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


_CORE_FIELDS = frozenset({"id", "family", "languages", "license_name", "license_url", "files"})
_ARCHIVE_FIELDS = frozenset({"archive_url", "archive_sha256", "archive_root"})
_DOWNLOAD_FIELDS = frozenset({"base_url", "downloads"})


def _parse_model(raw_model: Any) -> ModelSpec:
    if not isinstance(raw_model, dict):
        raise ValueError("each registry model must be a table")
    present = set(raw_model)
    if not present >= _CORE_FIELDS:
        raise ValueError(f"registry model is missing {', '.join(sorted(_CORE_FIELDS - present))}")
    source = present - _CORE_FIELDS
    if source not in (_ARCHIVE_FIELDS, _DOWNLOAD_FIELDS):
        raise ValueError(
            "each registry model needs exactly one source form: "
            "archive_url + archive_sha256 + archive_root, or base_url + downloads"
        )
    languages = raw_model["languages"]
    files = raw_model["files"]
    if not isinstance(languages, list) or not all(isinstance(item, str) for item in languages):
        raise ValueError("languages must be a string array")
    if not isinstance(files, dict) or not all(
        isinstance(role, str) and isinstance(value, str) for role, value in files.items()
    ):
        raise ValueError("files must be a string table")
    string_fields = (present - {"languages", "files", "downloads"})
    if not all(isinstance(raw_model[field], str) for field in string_fields):
        raise ValueError("model fields must use the registry schema types")
    return ModelSpec(
        model_id=raw_model["id"],
        family=raw_model["family"],
        languages=tuple(languages),
        license_name=raw_model["license_name"],
        license_url=raw_model["license_url"],
        files=dict(files),
        archive_url=raw_model.get("archive_url", ""),
        archive_sha256=raw_model.get("archive_sha256", ""),
        archive_root=raw_model.get("archive_root", ""),
        base_url=raw_model.get("base_url", ""),
        downloads=_parse_downloads(raw_model.get("downloads")),
    )


def _parse_downloads(raw_downloads: Any) -> tuple[tuple[str, str], ...]:
    if raw_downloads is None:
        return ()
    if not isinstance(raw_downloads, list) or not raw_downloads:
        raise ValueError("downloads must be a non-empty array of tables")
    parsed: list[tuple[str, str]] = []
    for entry in raw_downloads:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256"}
            or not all(isinstance(value, str) for value in entry.values())
        ):
            raise ValueError("each download must be { path = \"...\", sha256 = \"...\" }")
        parsed.append((entry["path"], entry["sha256"]))
    return tuple(parsed)


def _relative_path(value: str, field: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or "\\" in value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path")
    return path


def _platform_path(value: str) -> Path:
    return Path(*PurePosixPath(value).parts)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_archive(archive_path: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:*") as archive:
        for member in archive.getmembers():
            relative = _safe_archive_member(member)
            target = (destination / relative).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"unsafe archive member: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"cannot read archive member: {member.name}")
            with source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def _safe_archive_member(member: tarfile.TarInfo) -> Path:
    try:
        relative = _relative_path(member.name, "archive member")
    except ValueError as error:
        raise ValueError(f"unsafe archive member: {member.name}") from error
    if not member.isdir() and not member.isfile():
        raise ValueError(f"unsafe archive member: {member.name}")
    return Path(*relative.parts)


def _validate_installation(root: Path, spec: ModelSpec) -> None:
    resolved_root = root.resolve()
    if not resolved_root.is_dir():
        raise ValueError(f"missing model archive root: {spec.archive_root or spec.model_id}")
    # For the loose-file form this covers ONNX external data too -- `files` only
    # names the four loader roles, so an archive-form model never checked the
    # 2.4 GB `encoder.weights` that sits beside `encoder.onnx`.
    required = set(spec.files.values()) | {path for path, _ in spec.downloads}
    for relative_path in sorted(required):
        model_file = (resolved_root / _platform_path(relative_path)).resolve()
        if not model_file.is_relative_to(resolved_root) or not model_file.is_file():
            raise ValueError(f"missing required model file: {relative_path}")
