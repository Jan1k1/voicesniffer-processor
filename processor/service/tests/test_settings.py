import json
import os

import pytest

from voicesniffer_processor.settings import ProcessorSettings


def test_loads_server_tokens_without_exposing_them(tmp_path) -> None:
    token_file = tmp_path / "tokens.json"
    token_file.write_text(
        json.dumps({"server-a": "secret-a", "server-b": "secret-b"}),
        encoding="utf-8",
    )

    settings = ProcessorSettings.from_environment({"VOICESNIFFER_TOKENS_FILE": str(token_file)})

    assert settings.server_for_token("secret-a") == "server-a"
    assert settings.server_for_token("wrong") is None
    assert settings.bind_host == "127.0.0.1"
    assert settings.bind_port == 8080
    assert "secret-a" not in repr(settings)
    assert "secret-b" not in repr(settings)


@pytest.mark.parametrize(
    "tokens",
    [
        {},
        {"server-a": ""},
        {"server-a": "same-token", "server-b": "same-token"},
        {"invalid server": "secret"},
    ],
)
def test_rejects_invalid_token_mappings_without_echoing_tokens(tmp_path, tokens) -> None:
    token_file = tmp_path / "tokens.json"
    token_file.write_text(json.dumps(tokens), encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        ProcessorSettings.from_environment({"VOICESNIFFER_TOKENS_FILE": str(token_file)})

    assert all(token not in str(raised.value) for token in tokens.values() if token)


def test_requires_a_token_file() -> None:
    with pytest.raises(ValueError, match="VOICESNIFFER_TOKENS_FILE is required"):
        ProcessorSettings.from_environment({})


def test_a_missing_token_file_says_so(tmp_path) -> None:
    """These four cases all reported "must contain a readable JSON object", which
    is only true of one of them. The permission case is the one that actually
    happens -- a root-owned 0600 file crash-loops the container while the message
    points at JSON syntax -- and it was written up as a deployment gotcha rather
    than fixed."""
    with pytest.raises(ValueError, match="does not exist"):
        ProcessorSettings.from_environment(
            {"VOICESNIFFER_TOKENS_FILE": str(tmp_path / "absent.json")}
        )


def test_a_token_file_that_is_a_directory_says_so(tmp_path) -> None:
    directory = tmp_path / "tokens.json"
    directory.mkdir()

    # Reading a directory raises IsADirectoryError on Linux and PermissionError on
    # Windows; either message names the path and neither claims bad JSON.
    with pytest.raises(ValueError, match=r"is a directory|not readable|could not be read"):
        ProcessorSettings.from_environment({"VOICESNIFFER_TOKENS_FILE": str(directory)})


def test_malformed_json_reports_where(tmp_path) -> None:
    token_file = tmp_path / "tokens.json"
    token_file.write_text('{"server-a": "secret",\n  "oops"\n}', encoding="utf-8")

    with pytest.raises(ValueError) as raised:
        ProcessorSettings.from_environment({"VOICESNIFFER_TOKENS_FILE": str(token_file)})

    message = str(raised.value)
    assert "not valid JSON" in message
    assert "line 3" in message, message
    assert "secret" not in message


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX permission semantics")
def test_an_unreadable_token_file_names_the_uid(tmp_path) -> None:
    token_file = tmp_path / "tokens.json"
    token_file.write_text(json.dumps({"server-a": "secret"}), encoding="utf-8")
    token_file.chmod(0o000)
    if os.getuid() == 0:
        pytest.skip("root ignores file modes")

    with pytest.raises(ValueError) as raised:
        ProcessorSettings.from_environment({"VOICESNIFFER_TOKENS_FILE": str(token_file)})

    message = str(raised.value)
    assert "not readable by uid" in message
    assert "10001" in message, "the message should say which uid the container runs as"
    assert "secret" not in message


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("VOICESNIFFER_BIND_PORT", "0"),
        ("VOICESNIFFER_WORKERS", "65"),
        ("VOICESNIFFER_MAX_BODY_BYTES", "1048577"),
        ("VOICESNIFFER_MAX_FRAME_BYTES", "4001"),
        ("VOICESNIFFER_MAX_FRAMES", "3201"),
        ("VOICESNIFFER_MAX_DURATION_MS", "8001"),
    ],
)
def test_rejects_limits_above_hard_bounds(tmp_path, name, value) -> None:
    token_file = tmp_path / "tokens.json"
    token_file.write_text(json.dumps({"server-a": "secret"}), encoding="utf-8")

    with pytest.raises(ValueError, match=name):
        ProcessorSettings.from_environment(
            {"VOICESNIFFER_TOKENS_FILE": str(token_file), name: value}
        )


def test_reads_bounded_runtime_limits(tmp_path) -> None:
    token_file = tmp_path / "tokens.json"
    token_file.write_text(json.dumps({"server-a": "secret"}), encoding="utf-8")

    settings = ProcessorSettings.from_environment(
        {
            "VOICESNIFFER_TOKENS_FILE": str(token_file),
            "VOICESNIFFER_BIND_HOST": "0.0.0.0",
            "VOICESNIFFER_BIND_PORT": "9000",
            "VOICESNIFFER_WORKERS": "4",
            "VOICESNIFFER_MAX_BODY_BYTES": "500000",
            "VOICESNIFFER_MAX_FRAME_BYTES": "2000",
            "VOICESNIFFER_MAX_FRAMES": "1000",
            "VOICESNIFFER_MAX_DURATION_MS": "6000",
        }
    )

    assert settings.bind_host == "0.0.0.0"
    assert settings.bind_port == 9000
    assert settings.workers == 4
    assert settings.max_body_bytes == 500_000
    assert settings.max_frame_bytes == 2_000
    assert settings.max_frames == 1_000
    assert settings.max_samples_48k == 288_000


def test_token_lookup_verifies_only_the_digest_match(tmp_path, monkeypatch) -> None:
    token_file = tmp_path / "tokens.json"
    token_file.write_text(
        json.dumps({f"server-{index}": f"secret-{index}" for index in range(1_000)}),
        encoding="utf-8",
    )
    settings = ProcessorSettings.from_environment({"VOICESNIFFER_TOKENS_FILE": str(token_file)})
    comparisons: list[tuple[str, str]] = []

    def compare_digest(stored: str, presented: str) -> bool:
        comparisons.append((stored, presented))
        return stored == presented

    monkeypatch.setattr("voicesniffer_processor.settings.secrets.compare_digest", compare_digest)

    assert settings.server_for_token("secret-999") == "server-999"
    assert comparisons == [("secret-999", "secret-999")]


def test_unknown_token_requires_no_constant_time_comparison(tmp_path, monkeypatch) -> None:
    token_file = tmp_path / "tokens.json"
    token_file.write_text(json.dumps({"server-a": "secret-a"}), encoding="utf-8")
    settings = ProcessorSettings.from_environment({"VOICESNIFFER_TOKENS_FILE": str(token_file)})
    comparisons = 0

    def compare_digest(_stored: str, _presented: str) -> bool:
        nonlocal comparisons
        comparisons += 1
        return False

    monkeypatch.setattr("voicesniffer_processor.settings.secrets.compare_digest", compare_digest)

    assert settings.server_for_token("unknown") is None
    assert comparisons == 0
