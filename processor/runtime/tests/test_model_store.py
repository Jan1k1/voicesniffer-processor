from __future__ import annotations

import hashlib
import io
import multiprocessing
import tarfile
import time
from pathlib import Path

import pytest

from voicesniffer_runtime.model_store import (
    ModelSpec,
    ModelStore,
    default_registry_path,
    load_registry,
)

CHECKED_IN_REGISTRY = Path(__file__).parents[1] / "models.toml"


def test_default_registry_resolves_in_editable_install() -> None:
    assert default_registry_path() == CHECKED_IN_REGISTRY.resolve()


def create_archive(path: Path, members: dict[str, bytes]) -> str:
    with tarfile.open(path, "w:bz2") as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_spec(archive: Path, digest: str) -> ModelSpec:
    return ModelSpec(
        model_id="demo",
        family="moonshine-v2",
        archive_url=archive.as_uri(),
        archive_sha256=digest,
        archive_root="package",
        languages=("en",),
        license_name="MIT",
        license_url="https://example.test/license",
        files={"encoder": "encoder.ort", "decoder": "decoder.ort", "tokens": "tokens.txt"},
    )


def fetch_concurrently(
    store_root: Path,
    archive: Path,
    digest: str,
    barrier,
    results,
) -> None:
    import voicesniffer_runtime.model_store as model_store

    extract_archive = model_store._extract_archive

    def delayed_extract(archive_path: Path, destination: Path) -> None:
        time.sleep(0.5)
        extract_archive(archive_path, destination)

    model_store._extract_archive = delayed_extract
    try:
        barrier.wait(timeout=10)
        installation = ModelStore(store_root).fetch(model_spec(archive, digest))
    except Exception as error:
        results.put(("error", f"{type(error).__name__}: {error}"))
    else:
        results.put(("ok", str(installation)))


def test_loads_strict_registry_and_language_declarations(tmp_path: Path) -> None:
    registry_path = tmp_path / "models.toml"
    registry_path.write_text(
        """
schema = 1

[[models]]
id = "demo"
family = "moonshine-v2"
archive_url = "https://example.test/model.tar.bz2"
archive_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
archive_root = "package"
languages = ["en"]
license_name = "MIT"
license_url = "https://example.test/license"
files = { encoder = "encoder.ort", decoder = "decoder.ort", tokens = "tokens.txt" }
""".strip(),
        encoding="utf-8",
    )

    registry = load_registry(registry_path)

    assert tuple(registry) == ("demo",)
    assert registry["demo"].supports_language("en")
    assert not registry["demo"].supports_language("cs")


def test_checked_in_registry_pins_the_m1b_candidates() -> None:
    registry = load_registry(CHECKED_IN_REGISTRY)

    assert tuple(registry) == (
        "moonshine-tiny-v2-en",
        "moonshine-base-v2-en",
        "parakeet-tdt-0.6b-v3-int8",
        "parakeet-tdt-0.6b-v3-fp32",
        "whisper-small-multilingual-int8",
    )
    assert registry["moonshine-tiny-v2-en"].languages == ("en",)
    assert registry["parakeet-tdt-0.6b-v3-int8"].supports_language("cs")
    assert registry["whisper-small-multilingual-int8"].supports_language("cs")


def test_model_spec_files_cannot_change_after_validation(tmp_path: Path) -> None:
    archive = tmp_path / "model.tar.bz2"
    digest = create_archive(archive, {"package/encoder.ort": b"encoder"})
    spec = model_spec(archive, digest)

    with pytest.raises(TypeError):
        spec.files["encoder"] = "changed.ort"


def test_fetches_verified_archive_and_requires_model_files(tmp_path: Path) -> None:
    archive = tmp_path / "model.tar.bz2"
    digest = create_archive(
        archive,
        {
            "package/encoder.ort": b"encoder",
            "package/decoder.ort": b"decoder",
            "package/tokens.txt": b"tokens",
        },
    )
    store = ModelStore(tmp_path / "models")

    installation = store.fetch(model_spec(archive, digest))

    assert installation == tmp_path / "models" / "demo" / digest[:16]
    assert (installation / "encoder.ort").read_bytes() == b"encoder"
    assert store.fetch(model_spec(archive, digest)) == installation


def test_existing_installation_does_not_require_a_writable_lock_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import voicesniffer_runtime.model_store as model_store

    archive = tmp_path / "model.tar.bz2"
    digest = create_archive(
        archive,
        {
            "package/encoder.ort": b"encoder",
            "package/decoder.ort": b"decoder",
            "package/tokens.txt": b"tokens",
        },
    )
    store = ModelStore(tmp_path / "models")
    installation = store.fetch(model_spec(archive, digest))

    def fail_lock(_path: Path) -> None:
        raise AssertionError("an installed model must not acquire a writable lock")

    monkeypatch.setattr(model_store, "_exclusive_file_lock", fail_lock)

    assert store.fetch(model_spec(archive, digest)) == installation


def test_concurrent_processes_fetch_the_same_model_once(tmp_path: Path) -> None:
    archive = tmp_path / "model.tar.bz2"
    digest = create_archive(
        archive,
        {
            "package/encoder.ort": b"encoder",
            "package/decoder.ort": b"decoder",
            "package/tokens.txt": b"tokens",
        },
    )
    store_root = tmp_path / "models"
    expected = store_root / "demo" / digest[:16]
    context = multiprocessing.get_context("spawn")
    process_count = 4
    barrier = context.Barrier(process_count)
    results = context.Queue()
    processes = [
        context.Process(
            target=fetch_concurrently,
            args=(store_root, archive, digest, barrier, results),
        )
        for _ in range(process_count)
    ]

    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=20)

        assert all(process.exitcode == 0 for process in processes)
        outcomes = [results.get(timeout=5) for _ in processes]
        assert outcomes == [("ok", str(expected))] * process_count
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        results.close()
        results.join_thread()


def test_separates_installations_when_a_pinned_digest_changes(tmp_path: Path) -> None:
    first_archive = tmp_path / "first.tar.bz2"
    second_archive = tmp_path / "second.tar.bz2"
    first_digest = create_archive(
        first_archive,
        {
            "package/encoder.ort": b"first",
            "package/decoder.ort": b"decoder",
            "package/tokens.txt": b"tokens",
        },
    )
    second_digest = create_archive(
        second_archive,
        {
            "package/encoder.ort": b"second",
            "package/decoder.ort": b"decoder",
            "package/tokens.txt": b"tokens",
        },
    )
    store = ModelStore(tmp_path / "models")

    first = store.fetch(model_spec(first_archive, first_digest))
    second = store.fetch(model_spec(second_archive, second_digest))

    assert first != second
    assert (first / "encoder.ort").read_bytes() == b"first"
    assert (second / "encoder.ort").read_bytes() == b"second"


def test_rejects_checksum_mismatch_without_partial_install(tmp_path: Path) -> None:
    archive = tmp_path / "model.tar.bz2"
    create_archive(archive, {"package/encoder.ort": b"encoder"})
    store_root = tmp_path / "models"

    with pytest.raises(ValueError, match="SHA-256"):
        ModelStore(store_root).fetch(model_spec(archive, "0" * 64))

    assert not (store_root / "demo").exists()
    assert not list(store_root.rglob("*.partial"))


def test_rejects_missing_required_model_file(tmp_path: Path) -> None:
    archive = tmp_path / "model.tar.bz2"
    digest = create_archive(archive, {"package/encoder.ort": b"encoder"})

    with pytest.raises(ValueError, match="required model file"):
        ModelStore(tmp_path / "models").fetch(model_spec(archive, digest))


def test_rejects_archive_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "model.tar.bz2"
    digest = create_archive(
        archive,
        {
            "../escaped": b"bad",
            "package/encoder.ort": b"encoder",
            "package/decoder.ort": b"decoder",
            "package/tokens.txt": b"tokens",
        },
    )

    with pytest.raises(ValueError, match="unsafe archive member"):
        ModelStore(tmp_path / "models").fetch(model_spec(archive, digest))

    assert not (tmp_path / "escaped").exists()


def loose_files(directory: Path, contents: dict[str, bytes]) -> tuple[str, list[tuple[str, str]]]:
    """Lay out an upstream that publishes loose files instead of a tarball."""
    directory.mkdir(parents=True, exist_ok=True)
    downloads = []
    for name, payload in contents.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        downloads.append((name, hashlib.sha256(payload).hexdigest()))
    return directory.as_uri(), downloads


def loose_spec(base_url: str, downloads: list[tuple[str, str]]) -> ModelSpec:
    return ModelSpec(
        model_id="demo",
        family="nemo-transducer",
        base_url=base_url,
        downloads=tuple(downloads),
        languages=("en", "cs"),
        license_name="CC-BY-4.0",
        license_url="https://example.test/license",
        files={
            "encoder": "encoder.onnx",
            "decoder": "decoder.onnx",
            "joiner": "joiner.onnx",
            "tokens": "tokens.txt",
        },
    )


LOOSE_CONTENTS = {
    "encoder.onnx": b"encoder graph",
    # ONNX external data: named by no loader role, so the archive form could
    # never have verified it. Here it is pinned like everything else.
    "encoder.weights": b"two point four gigabytes, pretend",
    "decoder.onnx": b"decoder",
    "joiner.onnx": b"joiner",
    "tokens.txt": b"tokens",
}


def test_fetches_and_verifies_every_loose_file(tmp_path: Path) -> None:
    base_url, downloads = loose_files(tmp_path / "upstream", LOOSE_CONTENTS)

    installation = ModelStore(tmp_path / "models").fetch(loose_spec(base_url, downloads))

    for name, payload in LOOSE_CONTENTS.items():
        assert (installation / name).read_bytes() == payload


def test_loose_form_rejects_a_tampered_file_without_installing(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    base_url, downloads = loose_files(upstream, LOOSE_CONTENTS)
    # The 2.4 GB of external weights is exactly what an attacker would swap:
    # the graph still parses, the model says something else.
    (upstream / "encoder.weights").write_bytes(b"tampered")
    store_root = tmp_path / "models"

    with pytest.raises(ValueError, match="SHA-256"):
        ModelStore(store_root).fetch(loose_spec(base_url, downloads))

    assert not (store_root / "demo").exists()
    assert not list(store_root.rglob("*.partial"))


def test_loose_form_reuses_verified_files_across_attempts(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream"
    base_url, downloads = loose_files(upstream, LOOSE_CONTENTS)
    broken = [(path, "0" * 64 if path == "tokens.txt" else digest) for path, digest in downloads]
    store_root = tmp_path / "models"

    with pytest.raises(ValueError, match="SHA-256"):
        ModelStore(store_root).fetch(loose_spec(base_url, broken))

    cached = sorted(p.name for p in (store_root / ".files" / "demo").iterdir())
    assert any(name.endswith("encoder.weights") for name in cached)

    upstream_weights = upstream / "encoder.weights"
    upstream_weights.unlink()  # a second attempt must not need it again
    installation = ModelStore(store_root).fetch(loose_spec(base_url, downloads))
    assert (installation / "encoder.weights").read_bytes() == LOOSE_CONTENTS["encoder.weights"]


def test_loose_form_installation_key_tracks_every_digest(tmp_path: Path) -> None:
    base_url, downloads = loose_files(tmp_path / "upstream", LOOSE_CONTENTS)
    original = loose_spec(base_url, downloads)
    moved = [(path, digest) for path, digest in downloads if path != "joiner.onnx"]
    moved.append(("joiner.onnx", "b" * 64))

    assert loose_spec(base_url, moved).installation_key != original.installation_key


def test_registry_accepts_the_loose_file_form(tmp_path: Path) -> None:
    registry_path = tmp_path / "models.toml"
    roles = ", ".join(
        f'{role} = "{name}"'
        for role, name in (
            ("encoder", "encoder.onnx"),
            ("decoder", "decoder.onnx"),
            ("joiner", "joiner.onnx"),
            ("tokens", "tokens.txt"),
        )
    )
    pins = "\n".join(
        f'  {{ path = "{name}", sha256 = "{letter * 64}" }},'
        for name, letter in (
            ("encoder.onnx", "a"),
            ("encoder.weights", "b"),
            ("decoder.onnx", "c"),
            ("joiner.onnx", "d"),
            ("tokens.txt", "e"),
        )
    )
    registry_path.write_text(
        "\n".join(
            [
                "schema = 2",
                "",
                "[[models]]",
                'id = "demo"',
                'family = "nemo-transducer"',
                'base_url = "https://example.test/repo/resolve/abc123"',
                'languages = ["en", "cs"]',
                'license_name = "CC-BY-4.0"',
                'license_url = "https://example.test/license"',
                f"files = {{ {roles} }}",
                "downloads = [",
                pins,
                "]",
            ]
        ),
        encoding="utf-8",
    )

    spec = load_registry(registry_path)["demo"]

    assert spec.base_url.endswith("abc123")
    assert dict(spec.downloads)["encoder.weights"] == "b" * 64
    assert not spec.archive_url


def test_registry_rejects_mixing_the_two_source_forms(tmp_path: Path) -> None:
    registry_path = tmp_path / "models.toml"
    registry_path.write_text(
        """
schema = 2

[[models]]
id = "demo"
family = "moonshine-v2"
archive_url = "https://example.test/model.tar.bz2"
archive_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
archive_root = "package"
base_url = "https://example.test/repo"
languages = ["en"]
license_name = "MIT"
license_url = "https://example.test/license"
files = { encoder = "encoder.ort", decoder = "decoder.ort", tokens = "tokens.txt" }
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one source form"):
        load_registry(registry_path)


def test_loose_spec_requires_downloads_to_cover_every_loader_file() -> None:
    with pytest.raises(ValueError, match="downloads must cover"):
        loose_spec("https://example.test/repo", [("encoder.onnx", "a" * 64)])


def test_loose_spec_refuses_plain_http() -> None:
    with pytest.raises(ValueError, match="https"):
        loose_spec(
            "http://example.test/repo",
            [(name, "a" * 64) for name in LOOSE_CONTENTS],
        )


def test_checked_in_registry_pins_the_fp32_export_by_revision() -> None:
    spec = load_registry(CHECKED_IN_REGISTRY)["parakeet-tdt-0.6b-v3-fp32"]

    # A mutable "main"/"latest" coordinate is what the plugin JAR pinning broke
    # on; this one names a revision.
    assert "/resolve/1a468a35cbba69418f126de829e75261dea4a4e4" in spec.base_url
    assert spec.supports_language("en") and spec.supports_language("cs")
    assert dict(spec.downloads).keys() == {
        "encoder.onnx",
        "encoder.weights",
        "decoder.onnx",
        "joiner.onnx",
        "tokens.txt",
    }
