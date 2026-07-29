from __future__ import annotations

import base64
import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load_gate():
    path = Path(__file__).parents[1] / "scripts" / "owner_release_privacy.py"
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(
            "owner_release_privacy",
            path,
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def history_result(*, findings: list[dict[str, str]]) -> dict[str, object]:
    return {
        "commit_path_record_count": 3,
        "findings": findings,
        "inventory_sha256": "a" * 64,
        "object_count": 4,
        "object_type_counts": {
            "blob": 1,
            "commit": 1,
            "tag": 1,
            "tree": 1,
        },
        "ref_count": 2,
        "snapshot_sha256": "b" * 64,
        "total_object_bytes": 10,
    }


def identity_result() -> dict[str, object]:
    return {
        "commit_count": 1,
        "fetched_ref_count": 2,
        "inventory_sha256": "a" * 64,
        "reachable_annotated_tag_count": 1,
        "reachable_object_count": 4,
        "tag_ref_count": 1,
    }


def run_git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def initialize_git_repository(root: Path) -> str:
    run_git(root, "init", "-b", "main")
    run_git(root, "config", "user.name", "tiramitree")
    run_git(
        root,
        "config",
        "user.email",
        "89479100+tiramitree" + chr(64) + "users.noreply.github.com",
    )
    run_git(root, "config", "commit.gpgsign", "false")
    run_git(root, "config", "tag.gpgSign", "false")
    run_git(root, "add", "pyproject.toml")
    run_git(root, "commit", "-m", "Initial fixture")
    return run_git(root, "rev-parse", "HEAD")


def make_source_root(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    (root / "pyproject.toml").write_bytes(
        b'[project]\nname = "dcp-invariant"\nversion = "0.4.0"\n',
    )
    return root


def make_release_inputs(
    tmp_path: Path,
) -> tuple[list[Path], Path, Path]:
    asset_bytes = {
        "dcp-invariant-evidence-v0.4.0.tar.gz": b"evidence-fixture",
        "dcp_invariant-0.4.0-py3-none-any.whl": b"wheel-fixture",
        "dcp_invariant-0.4.0.tar.gz": b"sdist-fixture",
    }
    assets = []
    for name, raw in asset_bytes.items():
        path = tmp_path / name
        path.write_bytes(raw)
        assets.append(path)
    checksum = tmp_path / "SHA256SUMS"
    checksum.write_bytes(
        "".join(
            f"{hashlib.sha256(asset_bytes[name]).hexdigest()}  {name}\n"
            for name in sorted(asset_bytes)
        ).encode("ascii"),
    )
    assets.append(checksum)
    notes = tmp_path / "dcp-invariant-v0.4.0-release-notes.md"
    notes.write_bytes(
        b"# DCPInvariant v0.4.0\n\n"
        b"Fixture-scoped generation-lineage evidence with explicit limits.\n",
    )
    denylist = tmp_path / "denylist.txt"
    denylist.write_text("private-canary\n", encoding="utf-8")
    return assets, notes, denylist


def mock_clean_boundaries(
    gate,
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gate, "_SCRIPT_ROOT", root.resolve())
    monkeypatch.setattr(
        gate.privacy_scan,
        "scan",
        lambda scanned_root, *, denylist: [],
    )
    monkeypatch.setattr(
        gate.privacy_scan,
        "scan_git_history",
        lambda scanned_root, *, denylist: history_result(findings=[]),
    )
    monkeypatch.setattr(
        gate.verify_git_identity,
        "audit_repository",
        lambda scanned_root: identity_result(),
    )
    monkeypatch.setattr(
        gate,
        "_release_git_binding",
        lambda scanned_root, identity, source_snapshot: gate.ReleaseGitBinding(
            source_revision="f" * 40,
            source_tree_sha256="e" * 64,
            source_file_count=1,
            inventory_sha256="a" * 64,
            snapshot_sha256="b" * 64,
            object_count=4,
            ref_count=2,
        ),
    )

    def verify(
        path: Path,
        *,
        identity,
        source_snapshot,
        denylist: list[str],
        expected_sha256: str,
    ) -> dict[str, object]:
        assert identity.version == "0.4.0"
        assert source_snapshot.file_sha256
        assert denylist == ["private-canary"]
        assert expected_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        return {
            "archive_sha256": expected_sha256,
            "member_count": 1,
            "status": "PASS",
            "total_unpacked_bytes": 1,
        }

    monkeypatch.setattr(
        gate,
        "_verify_source_bound_distribution",
        verify,
    )

    def verify_evidence(
        path: Path,
        *,
        identity,
        denylist: list[str],
        expected_sha256: str,
        expected_source_revision: str,
    ) -> dict[str, object]:
        assert identity.version == "0.4.0"
        assert denylist == ["private-canary"]
        assert expected_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
        assert expected_source_revision == "f" * 40
        return {
            "archive_sha256": expected_sha256,
            "artifact_manifest_sha256": "c" * 64,
            "artifact_scenario_count": 13,
            "member_count": 33,
            "status": "PASS",
            "total_unpacked_bytes": 1,
        }

    monkeypatch.setattr(gate, "_verify_evidence_archive", verify_evidence)


def test_owner_gate_binds_source_history_identity_four_assets_and_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = load_gate()
    root = make_source_root(tmp_path)
    assets, notes, denylist = make_release_inputs(tmp_path)
    mock_clean_boundaries(gate, root, monkeypatch)

    result = gate.run_gate(
        root,
        assets,
        denylist_path=denylist,
        release_notes_path=notes,
    )

    assert result["status"] == "PASS"
    assert result["asset_count"] == 4
    assert result["checksum_asset_count"] == 3
    assert result["release_tag"] == "v0.4.0"
    assert result["release_title"] == "DCPInvariant v0.4.0"
    assert result["git_commit_count"] == 1
    assert result["git_tag_object_count"] == 1
    assert result["release_source_revision"] == "f" * 40
    assert result["release_source_tree_sha256"] == "e" * 64
    assert result["release_source_file_count"] == 1
    assert {item["name"] for item in result["assets"]} == {path.name for path in assets}


def test_release_git_binding_requires_annotated_tag_at_head(tmp_path: Path) -> None:
    gate = load_gate()
    root = make_source_root(tmp_path)
    head = initialize_git_repository(root)
    run_git(root, "tag", "-a", "v0.4.0", "-m", "DCPInvariant v0.4.0")

    binding = gate._release_git_binding(
        root,
        gate._release_identity(root),
        gate.privacy_scan.capture_source_snapshot(root),
    )

    assert binding.source_revision == head
    assert binding.source_file_count == 1
    assert len(binding.source_tree_sha256) == 64
    assert binding.object_count > 0
    assert binding.ref_count == 2


def test_release_git_binding_rejects_missing_or_lightweight_tag(
    tmp_path: Path,
) -> None:
    gate = load_gate()
    root = make_source_root(tmp_path)
    initialize_git_repository(root)
    identity = gate._release_identity(root)
    snapshot = gate.privacy_scan.capture_source_snapshot(root)

    with pytest.raises(ValueError, match="missing or not annotated"):
        gate._release_git_binding(root, identity, snapshot)

    run_git(root, "tag", "v0.4.0")
    with pytest.raises(ValueError, match="missing or not annotated"):
        gate._release_git_binding(root, identity, snapshot)


def test_release_git_binding_rejects_tag_not_at_head(tmp_path: Path) -> None:
    gate = load_gate()
    root = make_source_root(tmp_path)
    initialize_git_repository(root)
    run_git(root, "tag", "-a", "v0.4.0", "-m", "DCPInvariant v0.4.0")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "dcp-invariant"\nversion = "0.4.0"\n# next\n',
        encoding="utf-8",
    )
    run_git(root, "add", "pyproject.toml")
    run_git(root, "commit", "-m", "Advance fixture")

    with pytest.raises(ValueError, match="does not target current HEAD"):
        gate._release_git_binding(
            root,
            gate._release_identity(root),
            gate.privacy_scan.capture_source_snapshot(root),
        )


def test_release_git_binding_rejects_tracked_worktree_change(tmp_path: Path) -> None:
    gate = load_gate()
    root = make_source_root(tmp_path)
    initialize_git_repository(root)
    run_git(root, "tag", "-a", "v0.4.0", "-m", "DCPInvariant v0.4.0")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "dcp-invariant"\nversion = "0.4.0"\n# dirty\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="working tree does not match"):
        gate._release_git_binding(
            root,
            gate._release_identity(root),
            gate.privacy_scan.capture_source_snapshot(root),
        )


def test_release_git_binding_rejects_untracked_worktree_file(tmp_path: Path) -> None:
    gate = load_gate()
    root = make_source_root(tmp_path)
    initialize_git_repository(root)
    run_git(root, "tag", "-a", "v0.4.0", "-m", "DCPInvariant v0.4.0")
    (root / "untracked.txt").write_text("safe\n", encoding="utf-8")

    with pytest.raises(ValueError, match="working tree does not match"):
        gate._release_git_binding(
            root,
            gate._release_identity(root),
            gate.privacy_scan.capture_source_snapshot(root),
        )


def test_owner_gate_stops_before_assets_on_source_or_history_finding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = load_gate()
    root = make_source_root(tmp_path)
    assets, notes, denylist = make_release_inputs(tmp_path)
    mock_clean_boundaries(gate, root, monkeypatch)
    monkeypatch.setattr(
        gate.privacy_scan,
        "scan_git_history",
        lambda scanned_root, *, denylist: history_result(
            findings=[{"kind": "denylist-content", "path": "<git-object:blob>"}]
        ),
    )

    result = gate.run_gate(
        root,
        assets,
        denylist_path=denylist,
        release_notes_path=notes,
    )

    assert result["status"] == "FAIL"
    assert result["asset_count"] == 0


def test_owner_gate_rejects_denylist_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = load_gate()
    root = make_source_root(tmp_path)
    assets, notes, denylist = make_release_inputs(tmp_path)
    mock_clean_boundaries(gate, root, monkeypatch)
    values = iter((["private-canary"], ["changed-canary"]))
    monkeypatch.setattr(
        gate.privacy_scan,
        "load_denylist",
        lambda path, *, required: next(values),
    )

    with pytest.raises(ValueError, match="changed"):
        gate.run_gate(
            root,
            assets,
            denylist_path=denylist,
            release_notes_path=notes,
        )


def test_owner_gate_rejects_asset_inside_source_root(tmp_path: Path) -> None:
    gate = load_gate()
    root = make_source_root(tmp_path)
    gate._SCRIPT_ROOT = root.resolve()
    assets, notes, denylist = make_release_inputs(tmp_path)
    asset = root / assets[0].name
    asset.write_bytes(assets[0].read_bytes())
    assets[0] = asset

    with pytest.raises(ValueError, match="outside"):
        gate.run_gate(
            root,
            assets,
            denylist_path=denylist,
            release_notes_path=notes,
        )


def test_owner_gate_requires_fixed_four_asset_inventory(tmp_path: Path) -> None:
    gate = load_gate()
    root = make_source_root(tmp_path)
    gate._SCRIPT_ROOT = root.resolve()
    assets, notes, denylist = make_release_inputs(tmp_path)

    with pytest.raises(ValueError, match="four-file"):
        gate.run_gate(
            root,
            assets[:-1],
            denylist_path=denylist,
            release_notes_path=notes,
        )


def test_owner_gate_rejects_unregistered_wheel_filename(tmp_path: Path) -> None:
    gate = load_gate()
    root = make_source_root(tmp_path)
    gate._SCRIPT_ROOT = root.resolve()
    assets, notes, denylist = make_release_inputs(tmp_path)
    wheel = next(path for path in assets if path.suffix == ".whl")
    replacement = tmp_path / "dcp_invariant-0.4.0-unregistered-canary.whl"
    replacement.write_bytes(wheel.read_bytes())
    assets[assets.index(wheel)] = replacement

    with pytest.raises(ValueError, match="four-file"):
        gate.run_gate(
            root,
            assets,
            denylist_path=denylist,
            release_notes_path=notes,
        )


def test_owner_gate_rejects_git_identity_inventory_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = load_gate()
    root = make_source_root(tmp_path)
    assets, notes, denylist = make_release_inputs(tmp_path)
    mock_clean_boundaries(gate, root, monkeypatch)
    mismatched = identity_result()
    mismatched["inventory_sha256"] = "d" * 64
    monkeypatch.setattr(
        gate.verify_git_identity,
        "audit_repository",
        lambda scanned_root: mismatched,
    )

    with pytest.raises(ValueError, match="inventories differ"):
        gate.run_gate(
            root,
            assets,
            denylist_path=denylist,
            release_notes_path=notes,
        )


def test_owner_gate_rejects_sensitive_release_notes_without_echo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = load_gate()
    root = make_source_root(tmp_path)
    assets, notes, denylist = make_release_inputs(tmp_path)
    mock_clean_boundaries(gate, root, monkeypatch)
    canary = "private-contact" + chr(64) + "example.invalid"
    notes.write_bytes(f"# Release\n\n{canary}\n".encode())

    with pytest.raises(ValueError, match="privacy class") as captured:
        gate.run_gate(
            root,
            assets,
            denylist_path=denylist,
            release_notes_path=notes,
        )

    assert canary not in str(captured.value)


def test_owner_gate_rejects_noncanonical_checksum_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = load_gate()
    root = make_source_root(tmp_path)
    assets, notes, denylist = make_release_inputs(tmp_path)
    mock_clean_boundaries(gate, root, monkeypatch)
    checksum = next(path for path in assets if path.name == "SHA256SUMS")
    lines = checksum.read_text(encoding="ascii").splitlines()
    checksum.write_bytes(("\n".join(reversed(lines)) + "\n").encode("ascii"))

    with pytest.raises(ValueError, match="order"):
        gate.run_gate(
            root,
            assets,
            denylist_path=denylist,
            release_notes_path=notes,
        )


def test_owner_gate_rejects_source_change_across_full_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = load_gate()
    root = make_source_root(tmp_path)
    assets, notes, denylist = make_release_inputs(tmp_path)
    mock_clean_boundaries(gate, root, monkeypatch)
    original = gate.privacy_scan.assert_source_snapshot
    calls = 0

    def change_before_assert(path: Path, expected) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            (root / "concurrent.txt").write_text("new\n", encoding="utf-8")
        original(path, expected)

    monkeypatch.setattr(
        gate.privacy_scan,
        "assert_source_snapshot",
        change_before_assert,
    )

    with pytest.raises(ValueError, match="source tree changed"):
        gate.run_gate(
            root,
            assets,
            denylist_path=denylist,
            release_notes_path=notes,
        )


def test_evidence_archive_requires_exact_v4_inventory_without_echo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = load_gate()
    root = make_source_root(tmp_path)
    identity = gate._release_identity(root)
    archive = tmp_path / "dcp-invariant-evidence-v0.4.0.tar.gz"
    archive.write_bytes(b"fixture")
    canary = "unexpected-private-member-canary"
    member = gate.verify_distribution.ArchiveMember(
        canary,
        b"plain\n",
        is_directory=False,
    )
    monkeypatch.setattr(
        gate.verify_distribution,
        "snapshot_verified_distribution",
        lambda path, *, denylist: (
            {
                "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "member_count": 1,
                "status": "PASS",
                "total_unpacked_bytes": 6,
            },
            (member,),
        ),
    )

    with pytest.raises(ValueError, match="inventory") as captured:
        gate._verify_evidence_archive(
            archive,
            identity=identity,
            denylist=[],
            expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            expected_source_revision="f" * 40,
        )

    assert canary not in str(captured.value)


def test_evidence_archive_requires_release_source_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = load_gate()
    root = make_source_root(tmp_path)
    identity = gate._release_identity(root)
    archive = tmp_path / "dcp-invariant-evidence-v0.4.0.tar.gz"
    archive.write_bytes(b"fixture")
    inventory = gate._expected_evidence_members(identity)
    members = tuple(
        gate.verify_distribution.ArchiveMember(
            name,
            b"" if is_directory else b"{}\n",
            is_directory=is_directory,
        )
        for name, is_directory in inventory.items()
    )
    monkeypatch.setattr(
        gate.verify_distribution,
        "snapshot_verified_distribution",
        lambda path, *, denylist: (
            {
                "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "member_count": len(members),
                "status": "PASS",
                "total_unpacked_bytes": sum(len(member.data) for member in members),
            },
            members,
        ),
    )
    monkeypatch.setattr(
        gate,
        "verify_evidence_artifact",
        lambda path: SimpleNamespace(
            manifest_sha256="c" * 64,
            provenance={"source_revision": "1" * 40},
            summary={"scenario_count": 13},
        ),
    )

    with pytest.raises(ValueError, match="release source revision"):
        gate._verify_evidence_archive(
            archive,
            identity=identity,
            denylist=[],
            expected_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            expected_source_revision="2" * 40,
        )


def test_wheel_source_binding_rejects_stale_package_bytes(tmp_path: Path) -> None:
    gate = load_gate()
    root = make_source_root(tmp_path)
    identity = gate._release_identity(root)
    package = b'__version__ = "0.4.0"\n'
    license_bytes = b"license\n"
    notices = b"notices\n"
    source_hashes = {
        "src/dcp_invariant/__init__.py": hashlib.sha256(package).hexdigest(),
        "LICENSE": hashlib.sha256(license_bytes).hexdigest(),
        "THIRD_PARTY_NOTICES.md": hashlib.sha256(notices).hexdigest(),
    }
    dist_info = "dcp_invariant-0.4.0.dist-info"
    members = (
        gate.verify_distribution.ArchiveMember(
            "dcp_invariant/__init__.py",
            b'__version__ = "0.3.0"\n',
        ),
        gate.verify_distribution.ArchiveMember(
            f"{dist_info}/METADATA",
            b"Metadata-Version: 2.4\nName: dcp-invariant\nVersion: 0.4.0\n",
        ),
        gate.verify_distribution.ArchiveMember(f"{dist_info}/RECORD", b""),
        gate.verify_distribution.ArchiveMember(
            f"{dist_info}/WHEEL",
            b"Wheel-Version: 1.0\n"
            b"Generator: hatchling 1.31.0\n"
            b"Root-Is-Purelib: true\n"
            b"Tag: py3-none-any\n",
        ),
        gate.verify_distribution.ArchiveMember(
            f"{dist_info}/entry_points.txt",
            b"[console_scripts]\ndcp-invariant = dcp_invariant.cli:main\n",
        ),
        gate.verify_distribution.ArchiveMember(
            f"{dist_info}/" + "licenses/LICENSE",
            license_bytes,
        ),
        gate.verify_distribution.ArchiveMember(
            f"{dist_info}/" + "licenses/THIRD_PARTY_NOTICES.md",
            notices,
        ),
    )

    with pytest.raises(ValueError, match="package bytes"):
        gate._verify_wheel_source_binding(
            members,
            identity=identity,
            source_hashes=source_hashes,
        )


def test_sdist_source_binding_rejects_stale_source_bytes(tmp_path: Path) -> None:
    gate = load_gate()
    root = make_source_root(tmp_path)
    identity = gate._release_identity(root)
    source = b"current\n"
    source_hashes = {
        "src/dcp_invariant/value.py": hashlib.sha256(source).hexdigest(),
    }
    prefix = "dcp_invariant-0.4.0"
    members = (
        gate.verify_distribution.ArchiveMember(
            f"{prefix}/" + "src/dcp_invariant/value.py",
            b"stale\n",
        ),
        gate.verify_distribution.ArchiveMember(
            f"{prefix}/PKG-INFO",
            b"Metadata-Version: 2.4\nName: dcp-invariant\nVersion: 0.4.0\n",
        ),
    )

    with pytest.raises(ValueError, match="source distribution bytes"):
        gate._verify_sdist_source_binding(
            members,
            identity=identity,
            source_hashes=source_hashes,
        )


def test_wheel_record_binds_every_member() -> None:
    gate = load_gate()
    record_name = "package-0.4.0.dist-info/RECORD"
    payload = b"safe\n"
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    member_map = {
        "package/value.py": payload,
        record_name: (
            f"package/value.py,sha256={digest},{len(payload)}\n{record_name},,\n"
        ).encode("ascii"),
    }

    gate._verify_wheel_record(member_map, record_name=record_name)

    member_map["package/value.py"] = b"stale\n"
    with pytest.raises(ValueError, match="digest or size"):
        gate._verify_wheel_record(member_map, record_name=record_name)


def test_expected_v4_evidence_inventory_is_fixed(tmp_path: Path) -> None:
    gate = load_gate()
    root = make_source_root(tmp_path)
    identity = gate._release_identity(root)

    inventory = gate._expected_evidence_members(identity)

    assert len(inventory) == 33
    assert sum(inventory.values()) == 3
    assert len(inventory) - sum(inventory.values()) == 30
