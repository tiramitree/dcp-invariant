"""Owner-only release privacy gate using one private denylist snapshot."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import stat
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import NamedTuple

sys.dont_write_bytecode = True
_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_IMPORT_ROOT = str(_SCRIPT_ROOT / "src")
sys.path.insert(0, _SOURCE_IMPORT_ROOT)

import privacy_scan  # noqa: E402
import verify_distribution  # noqa: E402
import verify_git_identity  # noqa: E402
from git_snapshot import (  # noqa: E402
    GitSnapshotError,
    freeze_closure,
    list_commit_entries,
    read_object,
)

from dcp_invariant.artifact import (  # noqa: E402
    REGISTERED_SCENARIOS,
    EvidenceArtifactError,
    verify_evidence_artifact,
)

sys.path.remove(_SOURCE_IMPORT_ROOT)
MAX_RELEASE_NOTES_BYTES = 64 * 1024
MAX_CHECKSUM_BYTES = 4 * 1024


class FileSnapshot(NamedTuple):
    path: Path
    identity: tuple[int, int, int, int, int]
    raw: bytes
    sha256: str
    maximum: int


class ReleaseIdentity(NamedTuple):
    project_name: str
    normalized_name: str
    version: str
    tag: str
    title: str
    evidence_prefix: str
    release_notes_name: str
    asset_names: tuple[str, ...]


class ReleaseGitBinding(NamedTuple):
    source_revision: str
    source_tree_sha256: str
    source_file_count: int
    inventory_sha256: str
    snapshot_sha256: str
    object_count: int
    ref_count: int


def _release_identity(root: Path) -> ReleaseIdentity:
    try:
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"
        ]
        name = project["name"]
        version = project["version"]
    except (KeyError, OSError, UnicodeError, tomllib.TOMLDecodeError, TypeError):
        raise ValueError("project release metadata is invalid") from None
    if (
        not isinstance(name, str)
        or not isinstance(version, str)
        or re.fullmatch(r"[a-z0-9][a-z0-9.-]*", name) is None
        or re.fullmatch(r"[0-9][0-9A-Za-z.+-]*", version) is None
    ):
        raise ValueError("project release identity is invalid")
    normalized = re.sub(r"[-_.]+", "_", name).lower()
    tag = f"v{version}"
    evidence_prefix = f"dcp-invariant-evidence-{tag}"
    asset_names = tuple(
        sorted(
            {
                f"{evidence_prefix}.tar.gz",
                f"{normalized}-{version}-py3-none-any.whl",
                f"{normalized}-{version}.tar.gz",
                "SHA256SUMS",
            }
        )
    )
    return ReleaseIdentity(
        project_name=name,
        normalized_name=normalized,
        version=version,
        tag=tag,
        title=f"DCPInvariant {tag}",
        evidence_prefix=evidence_prefix,
        release_notes_name=f"dcp-invariant-{tag}-release-notes.md",
        asset_names=asset_names,
    )


def _validate_release_inventory(
    identity: ReleaseIdentity,
    assets: list[Path],
) -> dict[str, Path]:
    names = [path.name for path in assets]
    expected = set(identity.asset_names)
    if len(names) != 4 or len(names) != len(set(names)) or set(names) != expected:
        raise ValueError("release asset inventory is not the fixed four-file set")
    return {path.name: path for path in assets}


def _distribution_names(identity: ReleaseIdentity) -> set[str]:
    return {
        f"{identity.normalized_name}-{identity.version}-py3-none-any.whl",
        f"{identity.normalized_name}-{identity.version}.tar.gz",
    }


def _release_git_binding(
    root: Path,
    identity: ReleaseIdentity,
    source_snapshot: privacy_scan.SourceSnapshot,
) -> ReleaseGitBinding:
    closure = freeze_closure(root)
    tag_ref = f"refs/tags/{identity.tag}"
    tag_oid = dict(closure.snapshot.refs).get(tag_ref)
    objects = {value.oid: value for value in closure.objects}
    tag_object = objects.get(tag_oid) if tag_oid is not None else None
    if tag_object is None or tag_object.object_type != "tag":
        raise ValueError("registered release tag is missing or not annotated")
    raw = read_object(root, tag_object)
    header, separator, _ = raw.partition(b"\n\n")
    if not separator:
        raise ValueError("registered release tag object is malformed")
    lines = [line for line in header.splitlines() if not line.startswith(b" ")]
    object_lines = [line for line in lines if line.startswith(b"object ")]
    type_lines = [line for line in lines if line.startswith(b"type ")]
    name_lines = [line for line in lines if line.startswith(b"tag ")]
    if (
        len(object_lines) != 1
        or len(type_lines) != 1
        or len(name_lines) != 1
        or type_lines[0] != b"type commit"
        or name_lines[0] != b"tag " + identity.tag.encode("ascii")
    ):
        raise ValueError("registered release tag object is malformed")
    try:
        target_oid = object_lines[0].split(b" ", 1)[1].decode("ascii")
    except (IndexError, UnicodeDecodeError):
        raise ValueError("registered release tag target is malformed") from None
    target = objects.get(target_oid)
    if (
        target is None
        or target.object_type != "commit"
        or target_oid != closure.snapshot.head_oid
    ):
        raise ValueError("registered release tag does not target current HEAD")
    source_files = dict(source_snapshot.file_sha256)
    head_files: dict[bytes, str] = {}
    tree_digest = hashlib.sha256()
    tree_digest.update(b"dcp-release-source-tree-v1\0")
    for entry in list_commit_entries(root, target_oid):
        if (
            entry.object_type != "blob"
            or entry.mode not in {"100644", "100755"}
            or entry.path in head_files
        ):
            raise ValueError("release commit source tree is not registered")
        blob = objects.get(entry.oid)
        if blob is None or blob.object_type != "blob":
            raise ValueError("release commit source tree is incomplete")
        raw_blob = read_object(root, blob)
        digest = hashlib.sha256(raw_blob).hexdigest()
        head_files[entry.path] = digest
        tree_digest.update(entry.mode.encode("ascii"))
        tree_digest.update(b"\0")
        tree_digest.update(entry.path)
        tree_digest.update(b"\0")
        tree_digest.update(digest.encode("ascii"))
        tree_digest.update(b"\0")
    if not head_files or head_files != source_files:
        raise ValueError("frozen working tree does not match release HEAD")
    return ReleaseGitBinding(
        source_revision=target_oid,
        source_tree_sha256=tree_digest.hexdigest(),
        source_file_count=len(head_files),
        inventory_sha256=closure.inventory_sha256,
        snapshot_sha256=closure.snapshot.digest,
        object_count=len(closure.objects),
        ref_count=len(closure.snapshot.refs),
    )


def _snapshot_file(path: Path, *, maximum: int, label: str) -> FileSnapshot:
    resolved = path.resolve(strict=True)
    before = resolved.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or resolved.is_symlink()
        or privacy_scan.is_reparse(before)
        or before.st_nlink != 1
        or before.st_size < 0
        or before.st_size > maximum
    ):
        raise ValueError(f"{label} is not a bounded ordinary file")
    raw = resolved.read_bytes()
    middle = resolved.lstat()
    repeated = resolved.read_bytes()
    after = resolved.lstat()
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_nlink,
    )
    if (
        identity
        != (
            middle.st_dev,
            middle.st_ino,
            middle.st_size,
            middle.st_mtime_ns,
            middle.st_nlink,
        )
        or identity
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_nlink,
        )
        or len(raw) != before.st_size
        or repeated != raw
    ):
        raise ValueError(f"{label} changed during snapshot")
    return FileSnapshot(
        path=resolved,
        identity=identity,
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
        maximum=maximum,
    )


def _assert_file_snapshot(expected: FileSnapshot, *, label: str) -> None:
    current = _snapshot_file(expected.path, maximum=expected.maximum, label=label)
    if (
        current.identity != expected.identity
        or current.raw != expected.raw
        or current.sha256 != expected.sha256
    ):
        raise ValueError(f"{label} changed during owner release gate")


def _assert_outside_root(root: Path, path: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError:
        return
    raise ValueError(f"{label} must be outside the source root")


def _validate_public_text(
    raw: bytes,
    *,
    denylist: list[str],
    label: str,
) -> str:
    if raw.startswith(b"\xef\xbb\xbf") or b"\0" in raw:
        raise ValueError(f"{label} encoding is invalid")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"{label} is not UTF-8") from None
    if not text.endswith("\n") or "\r" in text or not text.strip():
        raise ValueError(f"{label} is not canonical LF text")
    lowered = text.casefold()
    if any(value in lowered for value in denylist):
        raise ValueError(f"{label} matched the external denylist")
    if privacy_scan._generic_match_kinds(raw):
        raise ValueError(f"{label} matched a forbidden privacy class")
    return text


def _metadata_identity(
    raw: bytes,
    *,
    identity: ReleaseIdentity,
    label: str,
) -> None:
    if b"\r" in raw or b"\0" in raw:
        raise ValueError(f"{label} metadata encoding is invalid")
    lines = raw.split(b"\n")
    expected = {
        b"Name": identity.project_name.encode("ascii"),
        b"Version": identity.version.encode("ascii"),
    }
    for field, value in expected.items():
        matches = [line for line in lines if line.startswith(field + b":")]
        if matches != [field + b": " + value]:
            raise ValueError(f"{label} release identity is invalid")


def _verify_wheel_record(
    member_map: dict[str, bytes],
    *,
    record_name: str,
) -> None:
    raw = member_map[record_name]
    if b"\r" in raw or not raw.endswith(b"\n"):
        raise ValueError("wheel RECORD is not canonical LF text")
    try:
        lines = raw[:-1].decode("ascii").split("\n")
    except UnicodeDecodeError:
        raise ValueError("wheel RECORD is not ASCII") from None
    if len(lines) != len(member_map):
        raise ValueError("wheel RECORD inventory is invalid")
    parsed_names = []
    for line in lines:
        fields = line.split(",")
        if len(fields) != 3:
            raise ValueError("wheel RECORD row is invalid")
        name, digest, size = fields
        if name not in member_map:
            raise ValueError("wheel RECORD inventory is invalid")
        parsed_names.append(name)
        if name == record_name:
            if digest or size:
                raise ValueError("wheel RECORD self row is invalid")
            continue
        raw_digest = hashlib.sha256(member_map[name]).digest()
        encoded = base64.urlsafe_b64encode(raw_digest).rstrip(b"=").decode("ascii")
        if digest != f"sha256={encoded}" or size != str(len(member_map[name])):
            raise ValueError("wheel RECORD digest or size is invalid")
    if parsed_names != list(member_map) or len(parsed_names) != len(set(parsed_names)):
        raise ValueError("wheel RECORD order or uniqueness is invalid")


def _source_file_hashes(
    source_snapshot: privacy_scan.SourceSnapshot,
) -> dict[str, str]:
    try:
        values = {
            raw_path.decode("utf-8"): digest
            for raw_path, digest in source_snapshot.file_sha256
        }
    except UnicodeDecodeError:
        raise ValueError("source snapshot path encoding is invalid") from None
    if not values:
        raise ValueError("source snapshot file inventory is empty")
    return values


def _verify_wheel_source_binding(
    members: tuple[verify_distribution.ArchiveMember, ...],
    *,
    identity: ReleaseIdentity,
    source_hashes: dict[str, str],
) -> None:
    if any(member.is_directory for member in members):
        raise ValueError("wheel contains an unregistered directory member")
    member_map = {member.name: member.data for member in members}
    expected_package = {
        relative.removeprefix("src/"): digest
        for relative, digest in source_hashes.items()
        if relative.startswith("src/dcp_invariant/")
    }
    if not expected_package:
        raise ValueError("wheel source inventory is empty")
    dist_info = f"{identity.normalized_name}-{identity.version}.dist-info"
    expected_metadata = {
        f"{dist_info}/METADATA",
        f"{dist_info}/RECORD",
        f"{dist_info}/WHEEL",
        f"{dist_info}/entry_points.txt",
        f"{dist_info}/" + "licenses/LICENSE",
        f"{dist_info}/" + "licenses/THIRD_PARTY_NOTICES.md",
    }
    if set(member_map) != {*expected_package, *expected_metadata}:
        raise ValueError("wheel member inventory is not source-bound")
    for name, expected_sha256 in expected_package.items():
        if hashlib.sha256(member_map[name]).hexdigest() != expected_sha256:
            raise ValueError("wheel package bytes do not match the source snapshot")
    for source_name in ("LICENSE", "THIRD_PARTY_NOTICES.md"):
        packaged = f"{dist_info}/" + f"licenses/{source_name}"
        if (
            source_name not in source_hashes
            or hashlib.sha256(member_map[packaged]).hexdigest()
            != source_hashes[source_name]
        ):
            raise ValueError("wheel license bytes do not match the source snapshot")
    _metadata_identity(
        member_map[f"{dist_info}/METADATA"],
        identity=identity,
        label="wheel",
    )
    if member_map[f"{dist_info}/WHEEL"] != (
        b"Wheel-Version: 1.0\n"
        b"Generator: hatchling 1.31.0\n"
        b"Root-Is-Purelib: true\n"
        b"Tag: py3-none-any\n"
    ):
        raise ValueError("wheel build metadata is not registered")
    if member_map[f"{dist_info}/entry_points.txt"] != (
        b"[console_scripts]\ndcp-invariant = dcp_invariant.cli:main\n"
    ):
        raise ValueError("wheel entry point is not registered")
    _verify_wheel_record(
        member_map,
        record_name=f"{dist_info}/RECORD",
    )


def _verify_sdist_source_binding(
    members: tuple[verify_distribution.ArchiveMember, ...],
    *,
    identity: ReleaseIdentity,
    source_hashes: dict[str, str],
) -> None:
    if any(member.is_directory for member in members):
        raise ValueError("source distribution contains a directory member")
    prefix = f"{identity.normalized_name}-{identity.version}"
    member_map = {member.name: member.data for member in members}
    expected_sources = {
        f"{prefix}/{relative}": digest for relative, digest in source_hashes.items()
    }
    metadata_name = f"{prefix}/PKG-INFO"
    if set(member_map) != {*expected_sources, metadata_name}:
        raise ValueError("source distribution inventory is not source-bound")
    for name, expected_sha256 in expected_sources.items():
        if hashlib.sha256(member_map[name]).hexdigest() != expected_sha256:
            raise ValueError(
                "source distribution bytes do not match the source snapshot"
            )
    _metadata_identity(
        member_map[metadata_name],
        identity=identity,
        label="source distribution",
    )


def _verify_source_bound_distribution(
    path: Path,
    *,
    identity: ReleaseIdentity,
    source_snapshot: privacy_scan.SourceSnapshot,
    denylist: list[str],
    expected_sha256: str,
) -> dict[str, object]:
    result, members = verify_distribution.snapshot_verified_distribution(
        path,
        denylist=tuple(denylist),
    )
    if result["archive_sha256"] != expected_sha256:
        raise ValueError("distribution changed before verification")
    source_hashes = _source_file_hashes(source_snapshot)
    if path.name.endswith(".whl"):
        _verify_wheel_source_binding(
            members,
            identity=identity,
            source_hashes=source_hashes,
        )
    else:
        _verify_sdist_source_binding(
            members,
            identity=identity,
            source_hashes=source_hashes,
        )
    return result


def _parse_checksums(
    snapshot: FileSnapshot,
    *,
    expected_assets: dict[str, FileSnapshot],
    denylist: list[str],
) -> dict[str, str]:
    text = _validate_public_text(
        snapshot.raw,
        denylist=denylist,
        label="checksum manifest",
    )
    lines = text[:-1].split("\n")
    expected_names = tuple(sorted(expected_assets))
    if len(lines) != len(expected_names):
        raise ValueError("checksum manifest row count is invalid")
    parsed: dict[str, str] = {}
    line_pattern = re.compile(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)")
    for line in lines:
        matched = line_pattern.fullmatch(line)
        if matched is None:
            raise ValueError("checksum manifest line is invalid")
        digest, name = matched.groups()
        if name in parsed:
            raise ValueError("checksum manifest contains a duplicate asset")
        parsed[name] = digest
    if tuple(parsed) != expected_names:
        raise ValueError("checksum manifest inventory or order is invalid")
    for name, expected in expected_assets.items():
        if parsed[name] != expected.sha256:
            raise ValueError("checksum manifest digest does not match an asset")
    return parsed


def _expected_evidence_members(identity: ReleaseIdentity) -> dict[str, bool]:
    prefix = identity.evidence_prefix
    expected = {
        prefix: True,
        f"{prefix}/observations": True,
        f"{prefix}/results": True,
        f"{prefix}/junit.xml": False,
        f"{prefix}/manifest.sha256": False,
        f"{prefix}/provenance.json": False,
        f"{prefix}/summary.json": False,
    }
    expected.update(
        {
            f"{prefix}/observations/{scenario}.json": False
            for scenario in REGISTERED_SCENARIOS
        }
    )
    expected.update(
        {
            f"{prefix}/results/{scenario}.json": False
            for scenario in REGISTERED_SCENARIOS
        }
    )
    return expected


def _verify_evidence_archive(
    path: Path,
    *,
    identity: ReleaseIdentity,
    denylist: list[str],
    expected_sha256: str,
    expected_source_revision: str,
) -> dict[str, object]:
    distribution, members = verify_distribution.snapshot_verified_distribution(
        path,
        denylist=tuple(denylist),
    )
    if distribution["archive_sha256"] != expected_sha256:
        raise ValueError("evidence archive changed before verification")
    expected = _expected_evidence_members(identity)
    inventory = {member.name: member.is_directory for member in members}
    if len(inventory) != len(members) or inventory != expected:
        raise ValueError("evidence archive inventory is invalid")
    with tempfile.TemporaryDirectory(prefix="dcp-evidence-verify-") as temporary:
        extraction_root = Path(temporary)
        for member in sorted(
            members,
            key=lambda value: (len(Path(value.name).parts), value.name),
        ):
            target = extraction_root.joinpath(*member.name.split("/"))
            if member.is_directory:
                target.mkdir()
            else:
                target.write_bytes(member.data)
        verified = verify_evidence_artifact(extraction_root / identity.evidence_prefix)
    if verified.summary["scenario_count"] != len(REGISTERED_SCENARIOS):
        raise ValueError("evidence archive scenario count is invalid")
    if verified.provenance["source_revision"] != expected_source_revision:
        raise ValueError("evidence archive does not bind the release source revision")
    return {
        **distribution,
        "artifact_manifest_sha256": verified.manifest_sha256,
        "artifact_scenario_count": verified.summary["scenario_count"],
    }


def run_gate(
    root: Path,
    assets: list[Path],
    *,
    denylist_path: Path,
    release_notes_path: Path,
) -> dict[str, object]:
    resolved_root = root.resolve(strict=True)
    if resolved_root != _SCRIPT_ROOT:
        raise ValueError("owner release gate must audit its own source root")
    source_snapshot = privacy_scan.capture_source_snapshot(resolved_root)
    identity = _release_identity(resolved_root)
    if not assets or len(assets) != len(set(assets)):
        raise ValueError("release asset inventory is empty or duplicated")
    resolved_assets = []
    for path in assets:
        resolved = path.resolve(strict=True)
        _assert_outside_root(resolved_root, resolved, label="release asset")
        resolved_assets.append(resolved)
    if len(resolved_assets) != len(set(resolved_assets)):
        raise ValueError("release asset inventory resolves to duplicates")
    asset_paths = _validate_release_inventory(identity, resolved_assets)
    resolved_notes = release_notes_path.resolve(strict=True)
    _assert_outside_root(
        resolved_root,
        resolved_notes,
        label="release notes",
    )
    if resolved_notes.name != identity.release_notes_name:
        raise ValueError("release notes filename is not registered")
    resolved_denylist = denylist_path.resolve(strict=True)
    _assert_outside_root(
        resolved_root,
        resolved_denylist,
        label="external denylist",
    )
    release_git = _release_git_binding(
        resolved_root,
        identity,
        source_snapshot,
    )
    asset_snapshots = {
        name: _snapshot_file(
            path,
            maximum=(
                MAX_CHECKSUM_BYTES
                if name == "SHA256SUMS"
                else verify_distribution.MAX_ARCHIVE_BYTES
            ),
            label="release asset",
        )
        for name, path in asset_paths.items()
    }
    notes_snapshot = _snapshot_file(
        resolved_notes,
        maximum=MAX_RELEASE_NOTES_BYTES,
        label="release notes",
    )
    denylist = privacy_scan.load_denylist(denylist_path, required=True)
    metadata_raw = (identity.tag + "\n" + identity.title + "\n").encode("utf-8")
    if any(value in metadata_raw.decode("utf-8").casefold() for value in denylist):
        raise ValueError("release metadata matched the external denylist")
    if privacy_scan._generic_match_kinds(metadata_raw):
        raise ValueError("release metadata matched a forbidden privacy class")
    release_notes = _validate_public_text(
        notes_snapshot.raw,
        denylist=denylist,
        label="release notes",
    )
    source_findings = privacy_scan.scan(resolved_root, denylist=denylist)
    history = privacy_scan.scan_git_history(resolved_root, denylist=denylist)
    git_identity = verify_git_identity.audit_repository(resolved_root)
    if (
        release_git.inventory_sha256 != history["inventory_sha256"]
        or release_git.snapshot_sha256 != history["snapshot_sha256"]
        or release_git.object_count != history["object_count"]
        or release_git.ref_count != history["ref_count"]
        or git_identity["inventory_sha256"] != history["inventory_sha256"]
        or git_identity["reachable_object_count"] != history["object_count"]
        or git_identity["fetched_ref_count"] != history["ref_count"]
    ):
        raise ValueError("Git privacy and identity inventories differ")
    findings = [*source_findings, *history["findings"]]
    if findings:
        repeated = privacy_scan.load_denylist(denylist_path, required=True)
        if repeated != denylist:
            raise ValueError("external denylist changed during owner release gate")
        privacy_scan.assert_source_snapshot(resolved_root, source_snapshot)
        _assert_file_snapshot(notes_snapshot, label="release notes")
        for snapshot in asset_snapshots.values():
            _assert_file_snapshot(snapshot, label="release asset")
        return {
            "asset_count": 0,
            "denylist_enforced": True,
            "finding_count": len(findings),
            "git_inventory_sha256": history["inventory_sha256"],
            "reachable_git_object_count": history["object_count"],
            "status": "FAIL",
        }
    results: dict[str, dict[str, object]] = {}
    for name in sorted(_distribution_names(identity)):
        snapshot = asset_snapshots[name]
        result = _verify_source_bound_distribution(
            snapshot.path,
            identity=identity,
            source_snapshot=source_snapshot,
            denylist=denylist,
            expected_sha256=snapshot.sha256,
        )
        results[name] = result
    evidence_name = f"{identity.evidence_prefix}.tar.gz"
    evidence_snapshot = asset_snapshots[evidence_name]
    results[evidence_name] = _verify_evidence_archive(
        evidence_snapshot.path,
        identity=identity,
        denylist=denylist,
        expected_sha256=evidence_snapshot.sha256,
        expected_source_revision=release_git.source_revision,
    )
    checksums = _parse_checksums(
        asset_snapshots["SHA256SUMS"],
        expected_assets={
            name: snapshot
            for name, snapshot in asset_snapshots.items()
            if name != "SHA256SUMS"
        },
        denylist=denylist,
    )
    repeated = privacy_scan.load_denylist(denylist_path, required=True)
    if repeated != denylist:
        raise ValueError("external denylist changed during owner release gate")
    final_history = privacy_scan.scan_git_history(resolved_root, denylist=denylist)
    if (
        final_history["inventory_sha256"] != history["inventory_sha256"]
        or final_history["snapshot_sha256"] != history["snapshot_sha256"]
        or final_history["object_count"] != history["object_count"]
        or final_history["ref_count"] != history["ref_count"]
        or final_history["findings"]
    ):
        raise ValueError("reachable Git closure changed during owner release gate")
    privacy_scan.assert_source_snapshot(resolved_root, source_snapshot)
    _assert_file_snapshot(notes_snapshot, label="release notes")
    for snapshot in asset_snapshots.values():
        _assert_file_snapshot(snapshot, label="release asset")
    public_assets = [
        {
            "bytes": len(asset_snapshots[name].raw),
            "name": name,
            "sha256": asset_snapshots[name].sha256,
            **(
                {
                    "artifact_manifest_sha256": results[name][
                        "artifact_manifest_sha256"
                    ],
                    "artifact_scenario_count": results[name]["artifact_scenario_count"],
                    "member_count": results[name]["member_count"],
                }
                if name == evidence_name
                else {
                    "member_count": results[name]["member_count"],
                }
                if name in results
                else {}
            ),
        }
        for name in identity.asset_names
    ]
    return {
        "asset_count": len(public_assets),
        "assets": public_assets,
        "checksum_asset_count": len(checksums),
        "commit_path_record_count": history["commit_path_record_count"],
        "denylist_enforced": True,
        "finding_count": 0,
        "git_commit_count": git_identity["commit_count"],
        "git_inventory_sha256": history["inventory_sha256"],
        "git_object_type_counts": history["object_type_counts"],
        "git_snapshot_sha256": history["snapshot_sha256"],
        "git_tag_object_count": git_identity["reachable_annotated_tag_count"],
        "reachable_git_object_count": history["object_count"],
        "reachable_git_ref_count": history["ref_count"],
        "reachable_git_total_bytes": history["total_object_bytes"],
        "release_notes_bytes": len(notes_snapshot.raw),
        "release_notes_line_count": len(release_notes.splitlines()),
        "release_notes_sha256": notes_snapshot.sha256,
        "release_source_file_count": release_git.source_file_count,
        "release_source_revision": release_git.source_revision,
        "release_source_tree_sha256": release_git.source_tree_sha256,
        "release_tag": identity.tag,
        "release_title": identity.title,
        "status": "PASS",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("assets", nargs="+", type=Path)
    parser.add_argument("--denylist-file", type=Path, required=True)
    parser.add_argument("--release-notes-file", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        result = run_gate(
            arguments.root,
            arguments.assets,
            denylist_path=arguments.denylist_file,
            release_notes_path=arguments.release_notes_file,
        )
    except (
        EvidenceArtifactError,
        GitSnapshotError,
        OSError,
        UnicodeError,
        ValueError,
        verify_distribution.DistributionBoundaryError,
        verify_git_identity.GitIdentityError,
    ) as error:
        print(
            json.dumps(
                {
                    "error": type(error).__name__,
                    "status": "ERROR",
                },
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
