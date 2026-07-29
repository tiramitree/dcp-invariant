"""Fail-closed repository and evidence privacy scanner."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import unicodedata
from pathlib import Path

sys.dont_write_bytecode = True

from git_snapshot import (  # noqa: E402
    GitSnapshotError,
    assert_snapshot_unchanged,
    freeze_closure,
    list_commit_entries,
    read_object,
)

MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_HISTORY_PATH_RECORDS = 500_000
MAX_HISTORY_PATH_BYTES = 64 * 1024 * 1024
MAX_SOURCE_ENTRIES = 100_000
MAX_SOURCE_PATH_BYTES = 32 * 1024 * 1024
MAX_SOURCE_SNAPSHOT_BYTES = 128 * 1024 * 1024
_USER_DIRECTORY = b"us" + b"ers"
_PRIVATE_KEY = b"PRIVATE " + b"KEY"
_AGE_SECRET_KEY = b"AGE-SECRET-" + b"KEY-"
_PUTTY_KEY_FILE = b"PuTTY-User-" + b"Key-File-"
_WXID = b"wx" + b"id_"
_XWECHAT = b"xwe" + b"chat"
_PATTERNS = {
    "absolute-windows-path": re.compile(
        rb"(?i)(?<![a-z0-9._-])(?:[a-z]:[\\/](?![\\/])"
        rb"|\\\\[a-z0-9._-]+\\[a-z0-9$._-]+)"
    ),
    "absolute-posix-path": re.compile(
        rb"(?i)(?<![:/a-z0-9._-])/(?!/)(?:(?:[a-z0-9._-]+/)+"
        rb"[a-z0-9._-]+|(?:home|users|tmp|var|mnt|etc|root|opt|data))"
        rb"(?:/|(?=$)|(?=[\s\"'<>),;]))"
    ),
    "email-address": re.compile(
        rb"(?i)(?<![a-z0-9._%+-])[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}"
    ),
    "private-key-marker": re.compile(
        rb"(?i)(?:-----BEGIN (?:[A-Z0-9 ]*"
        + _PRIVATE_KEY
        + rb"|PGP "
        + _PRIVATE_KEY
        + rb" BLOCK)-----"
        + rb"|"
        + _AGE_SECRET_KEY
        + rb"|"
        + _PUTTY_KEY_FILE
        + rb"[0-9]+:)"
    ),
    "github-token": re.compile(
        rb"\b(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9_]{20,})\b"
    ),
    "openai-token": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "cloud-or-service-token": re.compile(
        rb"\b(?:AKIA[A-Z0-9]{16}|xox[baprs]-[A-Za-z0-9-]{10,}"
        rb"|hf_[A-Za-z0-9]{20,})\b"
    ),
    "url-embedded-credential": re.compile(rb"(?i)https?://[^/\s:@]+:[^/\s@]+@"),
    "credential-assignment": re.compile(
        rb"(?i)\b(?:[a-z0-9]+[_-])*(?:api[_-]?key|access[_-]?(?:key|token)"
        rb"|auth[_-]?token|pass(?:word|wd)|secret(?:[_-]?(?:access[_-]?key|key))?"
        rb"|token)\s*[:=]\s*(?:[\"'][^\"'\r\n]{8,}[\"']"
        rb"|[a-z0-9_./+=:@-]{12,})"
    ),
    "labelled-phone-or-messaging-contact": re.compile(
        rb"(?i)\b(?:phone|tel(?:ephone)?|mobile|wechat|whatsapp|signal|telegram)"
        rb"\s*[:=]\s*\+?[\d(). -]{7,}\d"
    ),
    "messaging-identifier": re.compile(
        rb"(?i)\b(?:" + _WXID + rb"|" + _XWECHAT + rb"[_/])"
    ),
    "private-network-address": re.compile(
        rb"(?<![\d.])(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}"
        rb"|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}"
        rb"|100\.(?:6[4-9]|[789]\d|1[01]\d|12[0-7])(?:\.\d{1,3}){2}"
        rb"|169\.254(?:\.\d{1,3}){2})(?![\d.])"
    ),
}
_FORBIDDEN_NATIVE_NAMES = {
    ".metadata",
    "checkpoint-receipt.json",
}
_FORBIDDEN_NATIVE_SUFFIXES = (".distcp",)
_PUBLIC_GIT_EMAILS = tuple(
    value.encode("utf-8")
    for value in (
        "89479100+tiramitree" + chr(64) + "users.noreply.github.com",
        "noreply" + chr(64) + "github.com",
        "web-flow" + chr(64) + "users.noreply.github.com",
    )
)
_OPAQUE_BLOB_PREFIXES = (
    b"PK\x03\x04",
    b"7z\xbc\xaf\x27\x1c",
    b"Rar!\x1a\x07",
    b"\x1f\x8b",
)
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _generic_match_kinds(
    raw: bytes,
    *,
    exempt_public_git_emails: bool = False,
    relative_path: bool = False,
) -> list[str]:
    kinds: list[str] = []
    for label, pattern in _PATTERNS.items():
        if relative_path and label == "absolute-posix-path":
            continue
        matches = tuple(pattern.finditer(raw))
        if (
            label == "email-address"
            and exempt_public_git_emails
            and matches
            and all(
                matched.group(0).lower() in _PUBLIC_GIT_EMAILS for matched in matches
            )
        ):
            continue
        if matches:
            kinds.append(label)
    return kinds


def is_reparse(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & marker)


def raise_walk_error(error: OSError) -> None:
    """Turn any incomplete directory walk into a scan error."""

    raise error


def safe_relative(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    if relative.startswith("/") or ".." in Path(relative).parts:
        raise ValueError("unsafe relative path")
    return relative


def _relative_bytes(relative: str) -> bytes:
    if any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs", "Zl", "Zp"}
        for character in relative
    ):
        raise ValueError("source tree contains an unsupported path")
    try:
        raw = relative.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError("source tree contains an unsupported path") from None
    if any(value < 32 or value == 127 for value in raw):
        raise ValueError("source tree contains an unsupported path")
    return raw


def _remove_root_git_metadata(
    root: Path,
    current_path: Path,
    directories: list[str],
    files: list[str],
) -> None:
    if current_path != root:
        return
    if ".git" in directories:
        git_path = root / ".git"
        value = git_path.lstat()
        if (
            not stat.S_ISDIR(value.st_mode)
            or git_path.is_symlink()
            or is_reparse(value)
        ):
            raise ValueError(".git is not an ordinary root-level directory")
        directories.remove(".git")
    if ".git" in files:
        git_path = root / ".git"
        value = git_path.lstat()
        if (
            not stat.S_ISREG(value.st_mode)
            or git_path.is_symlink()
            or is_reparse(value)
            or value.st_nlink != 1
        ):
            raise ValueError(".git is not an ordinary root-level file")
        files.remove(".git")


class SourceSnapshot:
    __slots__ = ("_entries", "digest")

    def __init__(self, entries: tuple[tuple[object, ...], ...]) -> None:
        self._entries = entries
        self.digest = hashlib.sha256(repr(entries).encode("ascii")).hexdigest()

    @property
    def file_sha256(self) -> tuple[tuple[bytes, str], ...]:
        """Return the bounded ordinary-file closure captured by this snapshot."""

        return tuple(
            (entry[0], entry[-1])
            for entry in self._entries
            if entry[0] and isinstance(entry[-1], str)
        )


def capture_source_snapshot(root: Path) -> SourceSnapshot:
    root = Path(os.path.abspath(root))
    root_value = root.lstat()
    entries: list[tuple[object, ...]] = [
        (
            b"",
            root_value.st_mode,
            root_value.st_dev,
            root_value.st_ino,
            root_value.st_size,
            root_value.st_mtime_ns,
            root_value.st_ctime_ns,
            root_value.st_nlink,
            getattr(root_value, "st_file_attributes", 0),
            None,
        )
    ]
    path_bytes = 0
    snapshot_bytes = 0
    for current, directories, files in os.walk(
        root,
        topdown=True,
        onerror=raise_walk_error,
        followlinks=False,
    ):
        current_path = Path(current)
        _remove_root_git_metadata(root, current_path, directories, files)
        directories.sort()
        files.sort()
        unsafe_directories: list[str] = []
        for name in [*directories, *files]:
            path = current_path / name
            raw_path = _relative_bytes(safe_relative(root, path))
            value = path.lstat()
            content_sha256 = None
            if (
                stat.S_ISREG(value.st_mode)
                and not path.is_symlink()
                and not is_reparse(value)
                and value.st_nlink == 1
                and 0 <= value.st_size <= MAX_TEXT_BYTES
            ):
                raw = path.read_bytes()
                middle = path.lstat()
                repeated = path.read_bytes()
                after = path.lstat()
                identity = (
                    value.st_dev,
                    value.st_ino,
                    value.st_size,
                    value.st_mtime_ns,
                    value.st_ctime_ns,
                    value.st_nlink,
                )
                if (
                    identity
                    != (
                        middle.st_dev,
                        middle.st_ino,
                        middle.st_size,
                        middle.st_mtime_ns,
                        middle.st_ctime_ns,
                        middle.st_nlink,
                    )
                    or identity
                    != (
                        after.st_dev,
                        after.st_ino,
                        after.st_size,
                        after.st_mtime_ns,
                        after.st_ctime_ns,
                        after.st_nlink,
                    )
                    or len(raw) != value.st_size
                    or repeated != raw
                ):
                    raise ValueError("source tree changed during snapshot")
                snapshot_bytes += len(raw)
                if snapshot_bytes > MAX_SOURCE_SNAPSHOT_BYTES:
                    raise ValueError("source tree snapshot exceeds byte bound")
                content_sha256 = hashlib.sha256(raw).hexdigest()
            path_bytes += len(raw_path)
            entries.append(
                (
                    raw_path,
                    value.st_mode,
                    value.st_dev,
                    value.st_ino,
                    value.st_size,
                    value.st_mtime_ns,
                    value.st_ctime_ns,
                    value.st_nlink,
                    getattr(value, "st_file_attributes", 0),
                    content_sha256,
                )
            )
            if name in directories and (
                not stat.S_ISDIR(value.st_mode)
                or path.is_symlink()
                or is_reparse(value)
            ):
                unsafe_directories.append(name)
            if len(entries) > MAX_SOURCE_ENTRIES or path_bytes > MAX_SOURCE_PATH_BYTES:
                raise ValueError("source tree inventory exceeds bound")
        for name in unsafe_directories:
            directories.remove(name)
    entries.sort(key=lambda item: item[0])
    return SourceSnapshot(tuple(entries))


def assert_source_snapshot(root: Path, expected: SourceSnapshot) -> None:
    if capture_source_snapshot(root)._entries != expected._entries:
        raise ValueError("source tree changed during privacy gate")


def load_denylist(path: Path | None, *, required: bool) -> list[str]:
    if path is None:
        if required:
            raise ValueError("an external denylist is required")
        return []
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or path.is_symlink()
        or is_reparse(before)
        or before.st_nlink != 1
        or before.st_size < 0
        or before.st_size > MAX_TEXT_BYTES
    ):
        raise ValueError("denylist is not an ordinary file")
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ValueError("denylist must not contain a UTF-8 BOM")
    middle = path.lstat()
    repeated = path.read_bytes()
    after = path.lstat()
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
        raise ValueError("denylist changed during read")
    text = raw.decode("utf-8")
    values = [
        line.strip().casefold()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if required and not values:
        raise ValueError("external denylist is empty")
    return values


def scan(
    root: Path,
    *,
    denylist: list[str],
) -> list[dict[str, str]]:
    root = Path(os.path.abspath(root))
    root_stat = root.lstat()
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or root.is_symlink()
        or is_reparse(root_stat)
    ):
        raise ValueError("scan root is not an ordinary directory")

    source_snapshot = capture_source_snapshot(root)
    findings: list[dict[str, str]] = []
    redacted_paths: set[str] = set()

    def inspect_path(relative: str) -> None:
        raw = _relative_bytes(relative)
        denylist_matched = any(value in relative.casefold() for value in denylist)
        kinds = _generic_match_kinds(raw, relative_path=True)
        if denylist_matched or kinds:
            redacted_paths.add(relative)
        if denylist_matched:
            findings.append({"kind": "denylist-path", "path": "<redacted-path>"})
        findings.extend({"kind": kind, "path": "<redacted-path>"} for kind in kinds)

    def finding_path(relative: str) -> str:
        if relative in redacted_paths:
            return "<redacted-path>"
        return relative

    for current, directories, files in os.walk(
        root,
        topdown=True,
        onerror=raise_walk_error,
        followlinks=False,
    ):
        current_path = Path(current)
        _remove_root_git_metadata(root, current_path, directories, files)
        directories.sort()
        files.sort()

        unsafe_directories: list[str] = []
        for name in directories:
            path = current_path / name
            relative = safe_relative(root, path)
            inspect_path(relative)
            value = path.lstat()
            if (
                not stat.S_ISDIR(value.st_mode)
                or path.is_symlink()
                or is_reparse(value)
            ):
                findings.append(
                    {
                        "kind": "non-ordinary-directory",
                        "path": finding_path(relative),
                    }
                )
                unsafe_directories.append(name)
        for name in unsafe_directories:
            directories.remove(name)

        for name in files:
            path = current_path / name
            relative = safe_relative(root, path)
            inspect_path(relative)

        for name in files:
            path = current_path / name
            relative = safe_relative(root, path)
            reported = finding_path(relative)
            value = path.lstat()
            if (
                not stat.S_ISREG(value.st_mode)
                or path.is_symlink()
                or is_reparse(value)
                or value.st_nlink != 1
            ):
                findings.append({"kind": "non-ordinary-file", "path": reported})
                continue
            lowered_name = name.casefold()
            if lowered_name in _FORBIDDEN_NATIVE_NAMES or lowered_name.endswith(
                _FORBIDDEN_NATIVE_SUFFIXES
            ):
                findings.append(
                    {
                        "kind": "native-checkpoint-payload",
                        "path": reported,
                    }
                )
                continue
            if value.st_size > MAX_TEXT_BYTES:
                findings.append({"kind": "oversized-file", "path": reported})
                continue
            raw = path.read_bytes()
            middle = path.lstat()
            repeated = path.read_bytes()
            after = path.lstat()
            identity = (
                value.st_dev,
                value.st_ino,
                value.st_size,
                value.st_mtime_ns,
                value.st_nlink,
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
                or len(raw) != value.st_size
                or repeated != raw
            ):
                findings.append({"kind": "changed-during-read", "path": reported})
                continue
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                findings.append({"kind": "non-utf8-file", "path": reported})
                continue
            lowered = text.casefold()
            if any(value in lowered for value in denylist):
                findings.append({"kind": "denylist-content", "path": reported})
            findings.extend(
                {"kind": kind, "path": reported} for kind in _generic_match_kinds(raw)
            )
    assert_source_snapshot(root, source_snapshot)
    return findings


def _history_matches(
    raw: bytes,
    *,
    denylist: list[str],
    path: str,
    exempt_public_git_emails: bool = False,
    relative_path: bool = False,
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    lowered = raw.decode("utf-8").casefold()
    if any(value in lowered for value in denylist):
        findings.append({"kind": "denylist-content", "path": path})
    findings.extend(
        {"kind": kind, "path": path}
        for kind in _generic_match_kinds(
            raw,
            exempt_public_git_emails=exempt_public_git_emails,
            relative_path=relative_path,
        )
    )
    return findings


def scan_git_history(
    root: Path,
    *,
    denylist: list[str],
) -> dict[str, object]:
    """Scan a frozen closure reachable from fetched refs and HEAD.

    Findings expose only a content class and coarse object type. They never
    echo a matched denylist literal, ref name, object identifier, or payload.
    """

    root = Path(os.path.abspath(root))
    closure = freeze_closure(root)
    findings: list[dict[str, str]] = []
    for refname, _ in closure.snapshot.refs:
        findings.extend(
            _history_matches(
                refname.encode("utf-8"),
                denylist=denylist,
                path="<git-ref>",
            )
        )
    type_counts = {"blob": 0, "commit": 0, "tag": 0, "tree": 0}
    path_count = 0
    path_bytes = 0
    for value in closure.objects:
        type_counts[value.object_type] += 1
        raw = read_object(root, value)
        if value.object_type == "tree":
            continue
        reported = f"<git-object:{value.object_type}>"
        opaque_kind = None
        if b"\0" in raw:
            opaque_kind = "opaque-git-object"
        else:
            try:
                raw.decode("utf-8")
            except UnicodeDecodeError:
                opaque_kind = "opaque-git-object"
        if value.object_type == "blob" and (
            raw.startswith(_OPAQUE_BLOB_PREFIXES) or raw.startswith(_LFS_POINTER_PREFIX)
        ):
            opaque_kind = "opaque-git-blob"
        if opaque_kind is not None:
            findings.append({"kind": opaque_kind, "path": reported})
        else:
            findings.extend(
                _history_matches(
                    raw,
                    denylist=denylist,
                    path=reported,
                    exempt_public_git_emails=value.object_type in {"commit", "tag"},
                )
            )
        if value.object_type != "commit":
            continue
        for entry in list_commit_entries(root, value.oid):
            path_count += 1
            path_bytes += len(entry.path)
            if (
                path_count > MAX_HISTORY_PATH_RECORDS
                or path_bytes > MAX_HISTORY_PATH_BYTES
            ):
                raise GitSnapshotError("Git path inventory exceeds total bound")
            if entry.object_type == "commit" or entry.mode == "160000":
                findings.append({"kind": "git-submodule", "path": "<git-path>"})
                continue
            try:
                entry.path.decode("utf-8")
            except UnicodeDecodeError:
                findings.append({"kind": "opaque-git-path", "path": "<git-path>"})
                continue
            findings.extend(
                _history_matches(
                    entry.path,
                    denylist=denylist,
                    path="<git-path>",
                    relative_path=True,
                )
            )
    assert_snapshot_unchanged(root, closure.snapshot)
    return {
        "commit_path_record_count": path_count,
        "findings": findings,
        "inventory_sha256": closure.inventory_sha256,
        "object_count": len(closure.objects),
        "object_type_counts": type_counts,
        "ref_count": len(closure.snapshot.refs),
        "snapshot_sha256": closure.snapshot.digest,
        "total_object_bytes": closure.total_object_bytes,
    }


def summarize_findings(findings: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return a public-safe finding summary without repository path values."""

    summarized = []
    for finding in findings:
        raw_path = finding.get("path", "")
        scope = (
            "git-history"
            if raw_path.startswith("<git-") or raw_path.startswith("<git-object:")
            else "source-tree"
        )
        summarized.append({"kind": finding["kind"], "scope": scope})
    return summarized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--denylist-file", type=Path)
    parser.add_argument("--require-denylist", action="store_true")
    parser.add_argument("--include-git-history", action="store_true")
    parser.add_argument("--owner-release", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        denylist = load_denylist(
            args.denylist_file,
            required=args.require_denylist or args.owner_release,
        )
        findings = scan(args.root, denylist=denylist)
        history = None
        if args.include_git_history or args.owner_release:
            history = scan_git_history(args.root, denylist=denylist)
            findings.extend(history["findings"])
    except (
        GitSnapshotError,
        OSError,
        UnicodeError,
        ValueError,
    ) as error:
        print(json.dumps({"error": type(error).__name__, "status": "ERROR"}))
        return 2
    result = {
        "denylist_enforced": bool(denylist),
        "finding_count": len(findings),
        "findings": summarize_findings(findings),
        "git_history_scanned": history is not None,
        "status": "PASS" if not findings else "FAIL",
    }
    if history is not None:
        result.update(
            {
                "commit_path_record_count": history["commit_path_record_count"],
                "git_inventory_sha256": history["inventory_sha256"],
                "git_object_type_counts": history["object_type_counts"],
                "reachable_git_object_count": history["object_count"],
                "reachable_git_ref_count": history["ref_count"],
                "git_snapshot_sha256": history["snapshot_sha256"],
                "reachable_git_total_bytes": history["total_object_bytes"],
            }
        )
    print(json.dumps(result, sort_keys=True))
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
