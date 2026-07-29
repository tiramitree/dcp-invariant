from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest


def load_gate():
    path = Path(__file__).parents[1] / "scripts" / "verify_git_identity.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location("verify_git_identity", path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


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


def make_repository(
    root: Path,
    *,
    author_name: str,
    author_email: str,
) -> None:
    root.mkdir()
    run_git(root, "init")
    run_git(root, "config", "commit.gpgsign", "false")
    run_git(root, "config", "user.name", author_name)
    run_git(root, "config", "user.email", author_email)
    (root / "README.md").write_text("identity fixture\n", encoding="utf-8")
    run_git(root, "add", "README.md")
    run_git(root, "commit", "-m", "Add identity fixture")


def test_registered_pseudonymous_history_passes(tmp_path: Path) -> None:
    gate = load_gate()
    root = tmp_path / "repository"
    make_repository(
        root,
        author_name=gate.OWNER_NAME,
        author_email=gate.OWNER_EMAIL,
    )

    audit = gate.audit_repository(root)

    assert audit["commit_count"] == 1
    assert audit["reachable_annotated_tag_count"] == 0
    assert audit["tag_ref_count"] == 0


def test_unregistered_author_fails_without_echoing_identity(
    tmp_path: Path,
) -> None:
    gate = load_gate()
    root = tmp_path / "repository"
    make_repository(
        root,
        author_name="Real" + " Person",
        author_email="person" + chr(64) + "example.invalid",
    )

    with pytest.raises(gate.GitIdentityError, match="author"):
        gate.audit_repository(root)


def test_unregistered_committer_fails(tmp_path: Path) -> None:
    gate = load_gate()
    root = tmp_path / "repository"
    make_repository(
        root,
        author_name=gate.OWNER_NAME,
        author_email=gate.OWNER_EMAIL,
    )
    run_git(root, "config", "user.name", "Other" + " Person")
    run_git(
        root,
        "config",
        "user.email",
        "other" + chr(64) + "example.invalid",
    )
    (root / "README.md").write_text("second fixture\n", encoding="utf-8")
    run_git(root, "add", "README.md")
    run_git(
        root,
        "commit",
        f"--author={gate.OWNER_NAME} <{gate.OWNER_EMAIL}>",
        "-m",
        "Add second fixture",
    )

    with pytest.raises(gate.GitIdentityError, match="committer"):
        gate.audit_repository(root)


def test_registered_annotated_tag_tagger_passes(tmp_path: Path) -> None:
    gate = load_gate()
    root = tmp_path / "repository"
    make_repository(
        root,
        author_name=gate.OWNER_NAME,
        author_email=gate.OWNER_EMAIL,
    )
    run_git(root, "tag", "-a", "v1", "-m", "Safe annotated tag")

    audit = gate.audit_repository(root)

    assert audit["commit_count"] == 1
    assert audit["reachable_annotated_tag_count"] == 1
    assert audit["tag_ref_count"] == 1


def test_unregistered_annotated_tag_tagger_fails_without_echo(
    tmp_path: Path,
) -> None:
    gate = load_gate()
    root = tmp_path / "repository"
    make_repository(
        root,
        author_name=gate.OWNER_NAME,
        author_email=gate.OWNER_EMAIL,
    )
    unregistered_name = "Other" + " Tagger"
    unregistered_email = "tagger" + chr(64) + "example.invalid"
    run_git(root, "config", "user.name", unregistered_name)
    run_git(root, "config", "user.email", unregistered_email)
    run_git(root, "tag", "-a", "v1", "-m", "Unsafe tagger fixture")

    with pytest.raises(gate.GitIdentityError, match="tagger") as captured:
        gate.audit_repository(root)

    rendered = str(captured.value)
    assert unregistered_name not in rendered
    assert unregistered_email not in rendered


def test_nested_reachable_tag_checks_deleted_inner_ref(tmp_path: Path) -> None:
    gate = load_gate()
    root = tmp_path / "repository"
    make_repository(
        root,
        author_name=gate.OWNER_NAME,
        author_email=gate.OWNER_EMAIL,
    )
    unregistered_name = "Nested" + " Tagger"
    unregistered_email = "nested" + chr(64) + "example.invalid"
    run_git(root, "config", "user.name", unregistered_name)
    run_git(root, "config", "user.email", unregistered_email)
    run_git(root, "tag", "-a", "inner", "-m", "Inner tag")
    run_git(root, "config", "user.name", gate.OWNER_NAME)
    run_git(root, "config", "user.email", gate.OWNER_EMAIL)
    run_git(root, "tag", "-a", "outer", "inner", "-m", "Outer tag")
    run_git(root, "tag", "-d", "inner")

    with pytest.raises(gate.GitIdentityError, match="tagger"):
        gate.audit_repository(root)


def test_non_tag_ref_carrying_tag_object_is_audited(tmp_path: Path) -> None:
    gate = load_gate()
    root = tmp_path / "repository"
    make_repository(
        root,
        author_name=gate.OWNER_NAME,
        author_email=gate.OWNER_EMAIL,
    )
    run_git(root, "config", "user.name", "Other" + " Tagger")
    run_git(
        root,
        "config",
        "user.email",
        "other-tagger" + chr(64) + "example.invalid",
    )
    run_git(root, "tag", "-a", "temporary", "-m", "Temporary tag")
    tag_oid = git_output(root, "rev-parse", "refs/tags/temporary")
    run_git(root, "update-ref", "refs/archive/retained", tag_oid)
    run_git(root, "tag", "-d", "temporary")

    with pytest.raises(gate.GitIdentityError, match="tagger"):
        gate.audit_repository(root)


def test_same_tag_object_through_multiple_refs_is_counted_once(
    tmp_path: Path,
) -> None:
    gate = load_gate()
    root = tmp_path / "repository"
    make_repository(
        root,
        author_name=gate.OWNER_NAME,
        author_email=gate.OWNER_EMAIL,
    )
    run_git(root, "tag", "-a", "v1", "-m", "Shared tag")
    tag_oid = git_output(root, "rev-parse", "refs/tags/v1")
    run_git(root, "update-ref", "refs/archive/shared", tag_oid)

    audit = gate.audit_repository(root)

    assert audit["reachable_annotated_tag_count"] == 1
    assert audit["tag_ref_count"] == 1


def test_lightweight_tag_has_no_tagger_requirement(tmp_path: Path) -> None:
    gate = load_gate()
    root = tmp_path / "repository"
    make_repository(
        root,
        author_name=gate.OWNER_NAME,
        author_email=gate.OWNER_EMAIL,
    )
    run_git(root, "tag", "v1")

    audit = gate.audit_repository(root)

    assert audit["reachable_annotated_tag_count"] == 0
    assert audit["tag_ref_count"] == 1


@pytest.mark.parametrize(
    "signature_header",
    [
        b"gpgsig synthetic-header",
        b"gpgsig-sha256 synthetic-header",
        b"gpgsig-newhash synthetic-header",
        b"mergetag synthetic-header",
    ],
)
def test_reachable_signed_commit_headers_fail_without_echo(
    tmp_path: Path,
    signature_header: bytes,
) -> None:
    gate = load_gate()
    root = tmp_path / "repository"
    make_repository(
        root,
        author_name=gate.OWNER_NAME,
        author_email=gate.OWNER_EMAIL,
    )
    tree = git_output(root, "rev-parse", "HEAD^{tree}")
    identity = (f"{gate.OWNER_NAME} <{gate.OWNER_EMAIL}> 0 +0000").encode()
    payload_canary = b"synthetic-signature-payload-canary"
    raw = (
        b"tree "
        + tree.encode("ascii")
        + b"\nauthor "
        + identity
        + b"\ncommitter "
        + identity
        + b"\n"
        + signature_header
        + b"\n "
        + payload_canary
        + b"\n\nSynthetic signed commit\n"
    )
    oid = git_hash_object(root, "commit", raw)
    run_git(root, "update-ref", "refs/archive/signed-commit", oid)

    with pytest.raises(gate.GitSnapshotError, match="signature") as captured:
        gate.audit_repository(root)

    assert payload_canary.decode("ascii") not in str(captured.value)


@pytest.mark.parametrize(
    "armor_label",
    [
        b"PGP SIGNATURE",
        b"PGP MESSAGE",
        b"SSH SIGNATURE",
        b"SIGNED MESSAGE",
    ],
)
def test_reachable_signed_tag_armor_fails_without_echo(
    tmp_path: Path,
    armor_label: bytes,
) -> None:
    gate = load_gate()
    root = tmp_path / "repository"
    make_repository(
        root,
        author_name=gate.OWNER_NAME,
        author_email=gate.OWNER_EMAIL,
    )
    commit = git_output(root, "rev-parse", "HEAD")
    tagger = (f"{gate.OWNER_NAME} <{gate.OWNER_EMAIL}> 0 +0000").encode()
    payload_canary = b"synthetic-signature-payload-canary"
    raw = (
        b"object "
        + commit.encode("ascii")
        + b"\ntype commit\ntag signed-fixture\ntagger "
        + tagger
        + b"\n\nSynthetic signed tag\n-----BEGIN "
        + armor_label
        + b"-----\n"
        + payload_canary
        + b"\n-----END "
        + armor_label
        + b"-----\n"
    )
    oid = git_hash_object(root, "tag", raw)
    run_git(root, "update-ref", "refs/tags/signed-fixture", oid)

    with pytest.raises(gate.GitSnapshotError, match="signature") as captured:
        gate.audit_repository(root)

    assert payload_canary.decode("ascii") not in str(captured.value)
