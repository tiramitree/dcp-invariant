from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_scanner():
    path = Path(__file__).parents[1] / "scripts" / "privacy_scan.py"
    spec = importlib.util.spec_from_file_location("privacy_scan", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_clean_tree_passes(tmp_path: Path) -> None:
    scanner = load_scanner()
    (tmp_path / "README.md").write_text("pseudonymous evidence\n", encoding="utf-8")
    assert scanner.scan(tmp_path, denylist=[]) == []


def test_directory_walk_error_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanner = load_scanner()

    def unreadable_walk(
        root: Path,
        *,
        topdown: bool,
        onerror,
        followlinks: bool,
    ):
        assert root == tmp_path
        assert topdown is True
        assert followlinks is False
        onerror(PermissionError("registered unreadable-directory canary"))
        return []

    monkeypatch.setattr(scanner.os, "walk", unreadable_walk)
    with pytest.raises(PermissionError, match="unreadable-directory"):
        scanner.scan(tmp_path, denylist=[])


@pytest.mark.parametrize(
    ("payload", "kind"),
    [
        ("contact: person" + chr(64) + "example.net\n", "email-address"),
        ("key = ghp_" + ("a" * 30), "github-token"),
        ("-----BEGIN " + "PRIVATE KEY-----\n", "private-key-marker"),
        (
            "pass" + 'word = "' + ("a" * 12) + '"\n',
            "credential-assignment",
        ),
        (
            "path=C:" + chr(92) + "Us" + "ers" + chr(92) + "candidate\n",
            "absolute-windows-user-path",
        ),
        ("path=/" + "home" + "/candidate/data\n", "absolute-posix-user-path"),
    ],
)
def test_generic_canaries_are_detected(
    tmp_path: Path,
    payload: str,
    kind: str,
) -> None:
    scanner = load_scanner()
    (tmp_path / "canary.txt").write_text(payload, encoding="utf-8")
    assert scanner.scan(tmp_path, denylist=[])[0]["kind"] == kind


def test_external_denylist_match_does_not_echo_literal(tmp_path: Path) -> None:
    scanner = load_scanner()
    canary_literal = "private-literal-canary"
    (tmp_path / "content.txt").write_text(canary_literal, encoding="utf-8")
    findings = scanner.scan(tmp_path, denylist=[canary_literal])
    assert findings == [{"kind": "denylist-content", "path": "content.txt"}]
    assert canary_literal not in repr(findings)


def test_non_utf8_fails_closed(tmp_path: Path) -> None:
    scanner = load_scanner()
    (tmp_path / "binary.bin").write_bytes(b"\xff\xfe")
    assert scanner.scan(tmp_path, denylist=[])[0]["kind"] == "non-utf8-file"


@pytest.mark.parametrize(
    "name",
    [
        ".metadata",
        "checkpoint-receipt.json",
        "__0_0.distcp",
    ],
)
def test_native_checkpoint_payload_is_rejected(
    tmp_path: Path,
    name: str,
) -> None:
    scanner = load_scanner()
    (tmp_path / name).write_text("plain-text canary\n", encoding="utf-8")
    assert scanner.scan(tmp_path, denylist=[]) == [
        {"kind": "native-checkpoint-payload", "path": name}
    ]


def test_file_change_between_reads_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanner = load_scanner()
    target = tmp_path / "changing.txt"
    target.write_text("first\n", encoding="utf-8")
    original = scanner.Path.read_bytes
    calls = 0

    def changing_read(path: Path) -> bytes:
        nonlocal calls
        if path == target:
            calls += 1
            return b"first\n" if calls == 1 else b"other\n"
        return original(path)

    monkeypatch.setattr(scanner.Path, "read_bytes", changing_read)
    assert scanner.scan(tmp_path, denylist=[]) == [
        {"kind": "changed-during-read", "path": "changing.txt"}
    ]


def test_denylisted_filename_is_redacted_for_every_finding(tmp_path: Path) -> None:
    scanner = load_scanner()
    canary_literal = "sensitive-name-canary"
    (tmp_path / f"{canary_literal}.bin").write_bytes(b"\xff\xfe")
    findings = scanner.scan(tmp_path, denylist=[canary_literal])
    assert {item["kind"] for item in findings} == {
        "denylist-path",
        "non-utf8-file",
    }
    assert all(item["path"] == "<redacted-path>" for item in findings)
    assert canary_literal not in repr(findings)


def test_nested_directory_symlink_fails_closed_when_supported(tmp_path: Path) -> None:
    scanner = load_scanner()
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    link = tmp_path / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        outside.rmdir()
        pytest.skip("ordinary user cannot create directory symlinks")
    findings = scanner.scan(tmp_path, denylist=[])
    assert findings == [{"kind": "non-ordinary-directory", "path": "linked"}]
    outside.rmdir()


def test_symlink_scan_root_is_rejected_when_supported(tmp_path: Path) -> None:
    scanner = load_scanner()
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "root-link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("ordinary user cannot create directory symlinks")
    with pytest.raises(ValueError, match="ordinary directory"):
        scanner.scan(link, denylist=[])
