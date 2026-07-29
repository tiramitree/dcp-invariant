from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


def load_scanner():
    path = Path(__file__).parents[1] / "scripts" / "privacy_scan.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("privacy_scan", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


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
            "absolute-windows-path",
        ),
        (
            "path=/" + "home" + "/" + "candidate" + "/" + "data\n",
            "absolute-posix-path",
        ),
        ("path=/" + "tmp\n", "absolute-posix-path"),
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


def test_public_finding_summary_never_echoes_repository_paths() -> None:
    scanner = load_scanner()
    private_path_canary = "private-name-canary/credential.txt"
    summarized = scanner.summarize_findings(
        [
            {"kind": "email-address", "path": private_path_canary},
            {"kind": "github-token", "path": "<git-object:blob>"},
        ]
    )

    assert summarized == [
        {"kind": "email-address", "scope": "source-tree"},
        {"kind": "github-token", "scope": "git-history"},
    ]
    assert private_path_canary not in repr(summarized)


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
            if calls == 4:
                return b"other\n"
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


def test_generic_sensitive_filename_is_redacted(tmp_path: Path) -> None:
    scanner = load_scanner()
    canary = "private-contact" + chr(64) + "example.invalid"
    (tmp_path / f"{canary}.txt").write_text("plain\n", encoding="utf-8")

    findings = scanner.scan(tmp_path, denylist=[])

    assert findings == [{"kind": "email-address", "path": "<redacted-path>"}]
    assert canary not in repr(findings)


def test_generic_sensitive_directory_and_child_paths_are_redacted(
    tmp_path: Path,
) -> None:
    scanner = load_scanner()
    token_canary = "ghp_" + ("a" * 30)
    directory = tmp_path / token_canary
    directory.mkdir()
    (directory / "child.txt").write_text("plain\n", encoding="utf-8")

    findings = scanner.scan(tmp_path, denylist=[])

    assert findings
    assert {item["kind"] for item in findings} == {"github-token"}
    assert all(item["path"] == "<redacted-path>" for item in findings)
    assert token_canary not in repr(findings)


def test_source_snapshot_rejects_concurrent_new_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanner = load_scanner()
    target = tmp_path / "stable.txt"
    target.write_text("stable\n", encoding="utf-8")
    original = scanner.Path.read_bytes
    injected = False

    def add_during_read(path: Path) -> bytes:
        nonlocal injected
        raw = original(path)
        if path == target and not injected:
            injected = True
            (tmp_path / "concurrent.txt").write_text("new\n", encoding="utf-8")
        return raw

    monkeypatch.setattr(scanner.Path, "read_bytes", add_during_read)
    with pytest.raises(ValueError, match="source tree changed"):
        scanner.scan(tmp_path, denylist=[])


def test_relative_unicode_and_punctuation_paths_are_not_absolute_paths(
    tmp_path: Path,
) -> None:
    scanner = load_scanner()
    nested = tmp_path / "unicode-\u7236!" / "sub"
    nested.mkdir(parents=True)
    (nested / "file.txt").write_text("plain\n", encoding="utf-8")

    assert scanner.scan(tmp_path, denylist=[]) == []


def test_source_snapshot_hash_rejects_same_size_rewrite_with_restored_mtime(
    tmp_path: Path,
) -> None:
    scanner = load_scanner()
    target = tmp_path / "stable.txt"
    target.write_text("first\n", encoding="utf-8")
    before = target.stat()
    snapshot = scanner.capture_source_snapshot(tmp_path)

    target.write_text("other\n", encoding="utf-8")
    os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))

    with pytest.raises(ValueError, match="source tree changed"):
        scanner.assert_source_snapshot(tmp_path, snapshot)


def test_source_path_encoding_and_controls_fail_without_echo() -> None:
    scanner = load_scanner()
    for codepoint in (1, 127, 0x80, 0x2028, 0x202E, 0x2066):
        with pytest.raises(ValueError, match="unsupported path") as control:
            scanner._relative_bytes("safe/" + chr(codepoint) + "canary")
        assert "canary" not in str(control.value)
    with pytest.raises(ValueError, match="unsupported path") as surrogate:
        scanner._relative_bytes("safe/" + chr(0xD800) + "canary")

    assert "canary" not in str(surrogate.value)


def test_denylist_utf8_bom_is_rejected(tmp_path: Path) -> None:
    scanner = load_scanner()
    denylist = tmp_path / "denylist.txt"
    denylist.write_bytes(b"\xef\xbb\xbfprivate-canary\n")

    with pytest.raises(ValueError, match="BOM"):
        scanner.load_denylist(denylist, required=True)


def test_pattern_sources_do_not_trigger_their_own_patterns() -> None:
    scanner = load_scanner()
    root = Path(__file__).parents[1]

    assert (
        scanner._generic_match_kinds(
            (root / "scripts" / "privacy_scan.py").read_bytes()
        )
        == []
    )
    assert (
        scanner._generic_match_kinds(
            (root / "scripts" / "verify_distribution.py").read_bytes()
        )
        == []
    )


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


def run_git(root: Path, *arguments: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )


def git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def git_hash_object(root: Path, object_type: str, raw: bytes) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "hash-object", "-t", object_type, "-w", "--stdin"],
        check=True,
        capture_output=True,
        input=raw,
    )
    return completed.stdout.decode("ascii").strip()


def make_git_repository(root: Path) -> None:
    root.mkdir()
    run_git(root, "init")
    run_git(root, "config", "commit.gpgsign", "false")
    run_git(root, "config", "tag.gpgsign", "false")
    run_git(root, "config", "user.name", "tiramitree")
    run_git(
        root,
        "config",
        "user.email",
        "89479100+tiramitree" + chr(64) + "users.noreply.github.com",
    )
    (root / "README.md").write_text("history fixture\n", encoding="utf-8")
    run_git(root, "add", "README.md")
    run_git(root, "commit", "-m", "Add history fixture")


def test_clean_fetched_reachable_git_history_passes(tmp_path: Path) -> None:
    scanner = load_scanner()
    root = tmp_path / "repository"
    make_git_repository(root)

    result = scanner.scan_git_history(root, denylist=[])

    assert result["findings"] == []
    assert result["object_count"] >= 3
    assert result["ref_count"] == 1


def test_privacy_history_uses_shared_signed_object_rejection(
    tmp_path: Path,
) -> None:
    scanner = load_scanner()
    root = tmp_path / "repository"
    make_git_repository(root)
    tree = git_output(root, "rev-parse", "HEAD^{tree}")
    public_identity = (
        "tiramitree <"
        + "89479100+tiramitree"
        + chr(64)
        + "users.noreply.github.com> 0 +0000"
    ).encode("utf-8")
    payload_canary = b"synthetic-signature-payload-canary"
    raw = (
        b"tree "
        + tree.encode("ascii")
        + b"\nauthor "
        + public_identity
        + b"\ncommitter "
        + public_identity
        + b"\ngpgsig synthetic-header\n "
        + payload_canary
        + b"\n\nSynthetic signed commit\n"
    )
    oid = git_hash_object(root, "commit", raw)
    run_git(root, "update-ref", "refs/archive/signed-commit", oid)

    with pytest.raises(scanner.GitSnapshotError, match="signature") as captured:
        scanner.scan_git_history(root, denylist=[])

    assert payload_canary.decode("ascii") not in str(captured.value)


def test_reachable_deleted_blob_matches_private_denylist_without_echo(
    tmp_path: Path,
) -> None:
    scanner = load_scanner()
    root = tmp_path / "repository"
    make_git_repository(root)
    canary = "historical-private-canary"
    (root / "private.txt").write_text(canary, encoding="utf-8")
    run_git(root, "add", "private.txt")
    run_git(root, "commit", "-m", "Add temporary fixture")
    run_git(root, "rm", "private.txt")
    run_git(root, "commit", "-m", "Remove temporary fixture")

    result = scanner.scan_git_history(root, denylist=[canary])

    assert {"kind": "denylist-content", "path": "<git-object:blob>"} in result[
        "findings"
    ]
    assert canary not in repr(result)


def test_reachable_commit_and_annotated_tag_messages_are_scanned(
    tmp_path: Path,
) -> None:
    scanner = load_scanner()
    root = tmp_path / "repository"
    make_git_repository(root)
    commit_canary = "commit-message-private-canary"
    tag_canary = "tag-message-private-canary"
    (root / "second.txt").write_text("second\n", encoding="utf-8")
    run_git(root, "add", "second.txt")
    run_git(root, "commit", "-m", commit_canary)
    run_git(root, "tag", "-a", "v1", "-m", tag_canary)

    result = scanner.scan_git_history(
        root,
        denylist=[commit_canary, tag_canary],
    )

    assert {"kind": "denylist-content", "path": "<git-object:commit>"} in result[
        "findings"
    ]
    assert {"kind": "denylist-content", "path": "<git-object:tag>"} in result[
        "findings"
    ]
    assert commit_canary not in repr(result)
    assert tag_canary not in repr(result)


def test_reachable_deleted_blob_gets_generic_contact_scan(tmp_path: Path) -> None:
    scanner = load_scanner()
    root = tmp_path / "repository"
    make_git_repository(root)
    contact = "private-contact" + chr(64) + "example.invalid"
    (root / "contact.txt").write_text(contact, encoding="utf-8")
    run_git(root, "add", "contact.txt")
    run_git(root, "commit", "-m", "Add historical contact fixture")
    run_git(root, "rm", "contact.txt")
    run_git(root, "commit", "-m", "Remove historical contact fixture")

    result = scanner.scan_git_history(root, denylist=[])

    assert {"kind": "email-address", "path": "<git-object:blob>"} in result["findings"]


def test_reachable_full_historical_path_is_scanned(tmp_path: Path) -> None:
    scanner = load_scanner()
    root = tmp_path / "repository"
    make_git_repository(root)
    first = "private-parent-canary"
    second = "private-child-canary"
    nested = root / first
    nested.mkdir()
    (nested / second).write_text("plain\n", encoding="utf-8")
    run_git(root, "add", first)
    run_git(root, "commit", "-m", "Add nested path fixture")
    run_git(root, "rm", "-r", first)
    run_git(root, "commit", "-m", "Remove nested path fixture")

    result = scanner.scan_git_history(
        root,
        denylist=[f"{first}/{second}"],
    )

    assert {"kind": "denylist-content", "path": "<git-path>"} in result["findings"]
    assert first not in repr(result)
    assert second not in repr(result)


def test_reachable_opaque_blob_fails_closed(tmp_path: Path) -> None:
    scanner = load_scanner()
    root = tmp_path / "repository"
    make_git_repository(root)
    (root / "opaque.bin").write_bytes(b"\xff\xfe\x00")
    run_git(root, "add", "opaque.bin")
    run_git(root, "commit", "-m", "Add opaque fixture")

    result = scanner.scan_git_history(root, denylist=[])

    assert {"kind": "opaque-git-object", "path": "<git-object:blob>"} in result[
        "findings"
    ]


def test_reachable_lfs_pointer_fails_closed(tmp_path: Path) -> None:
    scanner = load_scanner()
    root = tmp_path / "repository"
    make_git_repository(root)
    pointer = (
        b"version https://git-lfs.github.com/spec/v1\n"
        b"oid sha256:" + (b"a" * 64) + b"\nsize 1\n"
    )
    (root / "pointer.txt").write_bytes(pointer)
    run_git(root, "add", "pointer.txt")
    run_git(root, "commit", "-m", "Add pointer fixture")

    result = scanner.scan_git_history(root, denylist=[])

    assert {"kind": "opaque-git-blob", "path": "<git-object:blob>"} in result[
        "findings"
    ]


def test_replace_ref_is_rejected(tmp_path: Path) -> None:
    scanner = load_scanner()
    root = tmp_path / "repository"
    make_git_repository(root)
    original = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    (root / "README.md").write_text("replacement\n", encoding="utf-8")
    run_git(root, "add", "README.md")
    run_git(root, "commit", "-m", "Add replacement fixture")
    replacement = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()
    run_git(root, "replace", original, replacement)

    with pytest.raises(scanner.GitSnapshotError, match="replace"):
        scanner.scan_git_history(root, denylist=[])


def test_promisor_configuration_is_rejected(tmp_path: Path) -> None:
    scanner = load_scanner()
    root = tmp_path / "repository"
    make_git_repository(root)
    run_git(root, "config", "remote.origin.promisor", "true")

    with pytest.raises(scanner.GitSnapshotError, match="partial"):
        scanner.scan_git_history(root, denylist=[])


def test_owner_release_requires_external_denylist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scanner = load_scanner()
    monkeypatch.setattr(
        sys,
        "argv",
        ["privacy_scan.py", str(tmp_path), "--owner-release"],
    )

    assert scanner.main() == 2
