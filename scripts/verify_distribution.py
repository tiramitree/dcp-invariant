"""Fail-closed wheel and source-distribution boundary verifier."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import NamedTuple

MAX_MEMBER_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_UNPACKED_BYTES = 16 * 1024 * 1024
MAX_DENYLIST_BYTES = 2 * 1024 * 1024
_USER_DIRECTORY = b"us" + b"ers"
_SENSITIVE_PATTERNS = {
    "absolute-windows-user-path": re.compile(
        rb"(?i)(?:[a-z]:[\\/]+"
        + _USER_DIRECTORY
        + rb"[\\/]+|\\\\"
        + _USER_DIRECTORY
        + rb"\\\\)"
    ),
    "absolute-posix-user-path": re.compile(
        rb"(?i)/(?:home|users|private/var)/[a-z0-9._-]+/"
    ),
    "email-address": re.compile(
        rb"(?i)(?<![a-z0-9._%+-])[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}"
    ),
    "private-key-marker": re.compile(rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    "github-token": re.compile(rb"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "openai-token": re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "credential-assignment": re.compile(
        rb"(?i)\b(?:"
        + b"api"
        + rb"[_-]?key|access[_-]?token|pass"
        + b"word"
        + rb"|secret)\s*[:=]\s*[\"'][^\"'\r\n]{8,}[\"']"
    ),
}
_FORBIDDEN_SUFFIXES = (
    ".distcp",
    ".pyc",
    ".pyo",
)
_FORBIDDEN_NAMES = {
    ".metadata",
    "checkpoint-receipt.json",
}


class DistributionBoundaryError(ValueError):
    """A distribution crossed the registered source-only boundary."""


class ArchiveMember(NamedTuple):
    name: str
    data: bytes


def _safe_member_name(name: str) -> str:
    if not name or "\\" in name:
        raise DistributionBoundaryError("archive member name is invalid")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise DistributionBoundaryError("archive member path is unsafe")
    return path.as_posix()


def _is_reparse(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & marker)


def _zip_members(raw_archive: bytes) -> list[ArchiveMember]:
    members: list[ArchiveMember] = []
    total = 0
    with zipfile.ZipFile(io.BytesIO(raw_archive)) as archive:
        for info in archive.infolist():
            name = _safe_member_name(info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            if info.is_dir():
                continue
            if stat.S_ISLNK(mode):
                raise DistributionBoundaryError("wheel contains a symbolic link")
            if info.file_size < 0 or info.file_size > MAX_MEMBER_BYTES:
                raise DistributionBoundaryError("archive member exceeds size bound")
            total += info.file_size
            if total > MAX_TOTAL_UNPACKED_BYTES:
                raise DistributionBoundaryError(
                    "archive unpacked bytes exceed size bound"
                )
            data = archive.read(info)
            if len(data) != info.file_size:
                raise DistributionBoundaryError("archive member size changed")
            members.append(ArchiveMember(name, data))
    return members


def _tar_members(raw_archive: bytes) -> list[ArchiveMember]:
    members: list[ArchiveMember] = []
    total = 0
    with tarfile.open(fileobj=io.BytesIO(raw_archive), mode="r:gz") as archive:
        for info in archive.getmembers():
            name = _safe_member_name(info.name)
            if info.isdir():
                continue
            if not info.isfile() or info.issym() or info.islnk():
                raise DistributionBoundaryError(
                    "source distribution contains a non-ordinary member"
                )
            if info.size < 0 or info.size > MAX_MEMBER_BYTES:
                raise DistributionBoundaryError("archive member exceeds size bound")
            total += info.size
            if total > MAX_TOTAL_UNPACKED_BYTES:
                raise DistributionBoundaryError(
                    "archive unpacked bytes exceed size bound"
                )
            source = archive.extractfile(info)
            if source is None:
                raise DistributionBoundaryError("archive member cannot be read")
            data = source.read(MAX_MEMBER_BYTES + 1)
            if len(data) != info.size:
                raise DistributionBoundaryError("archive member size changed")
            members.append(ArchiveMember(name, data))
    return members


def _snapshot_archive(path: Path) -> bytes:
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or path.is_symlink()
        or _is_reparse(before)
        or before.st_nlink != 1
        or before.st_size < 0
        or before.st_size > MAX_ARCHIVE_BYTES
    ):
        raise DistributionBoundaryError("distribution is not a bounded ordinary file")
    raw = path.read_bytes()
    after = path.lstat()
    if len(raw) != before.st_size or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_nlink,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_nlink,
    ):
        raise DistributionBoundaryError("distribution changed during snapshot")
    return raw


def _read_archive(path: Path, raw_archive: bytes) -> list[ArchiveMember]:
    name = path.name.lower()
    if name.endswith(".whl"):
        return _zip_members(raw_archive)
    if name.endswith(".tar.gz"):
        return _tar_members(raw_archive)
    raise DistributionBoundaryError("distribution type is not registered")


def verify_distribution(
    path: Path,
    *,
    denylist: tuple[str, ...] = (),
) -> dict[str, object]:
    raw_archive = _snapshot_archive(path)
    members = _read_archive(path, raw_archive)
    names = [member.name for member in members]
    if len(names) != len(set(names)):
        raise DistributionBoundaryError("archive contains duplicate member names")
    total = sum(len(member.data) for member in members)
    if total > MAX_TOTAL_UNPACKED_BYTES:
        raise DistributionBoundaryError("archive unpacked bytes exceed size bound")
    for member in members:
        pure = PurePosixPath(member.name)
        lowered_parts = [part.lower() for part in pure.parts]
        lowered_name = pure.name.lower()
        if (
            "__pycache__" in lowered_parts
            or "torch" in lowered_parts
            or lowered_name in _FORBIDDEN_NAMES
            or lowered_name.endswith(_FORBIDDEN_SUFFIXES)
        ):
            raise DistributionBoundaryError("archive contains a forbidden payload")
        lowered_path = member.name.casefold()
        lowered_content = member.data.decode("utf-8", errors="ignore").casefold()
        if any(value in lowered_path or value in lowered_content for value in denylist):
            raise DistributionBoundaryError("archive matched external denylist")
        for label, pattern in _SENSITIVE_PATTERNS.items():
            if pattern.search(member.data):
                raise DistributionBoundaryError(
                    f"archive content matched forbidden class {label}"
                )
    return {
        "archive_sha256": hashlib.sha256(raw_archive).hexdigest(),
        "member_count": len(members),
        "status": "PASS",
        "total_unpacked_bytes": total,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("archives", nargs="+", type=Path)
    parser.add_argument("--denylist-file", type=Path)
    parser.add_argument("--require-denylist", action="store_true")
    return parser


def _load_denylist(path: Path | None, *, required: bool) -> tuple[str, ...]:
    if path is None:
        if required:
            raise DistributionBoundaryError("an external denylist is required")
        return ()
    before = path.lstat()
    if (
        not stat.S_ISREG(before.st_mode)
        or path.is_symlink()
        or _is_reparse(before)
        or before.st_nlink != 1
        or before.st_size < 0
        or before.st_size > MAX_DENYLIST_BYTES
    ):
        raise DistributionBoundaryError("denylist is not a bounded ordinary file")
    raw = path.read_bytes()
    after = path.lstat()
    if len(raw) != before.st_size or (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_nlink,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_nlink,
    ):
        raise DistributionBoundaryError("denylist changed during snapshot")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise DistributionBoundaryError("denylist is not UTF-8") from error
    values = tuple(
        line.strip().casefold()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if required and not values:
        raise DistributionBoundaryError("external denylist is empty")
    return values


def _expand_inputs(paths: list[Path]) -> list[Path]:
    archives: list[Path] = []
    for path in paths:
        value = path.lstat()
        if (
            stat.S_ISDIR(value.st_mode)
            and not path.is_symlink()
            and not _is_reparse(value)
        ):
            entries = sorted(
                (
                    entry
                    for entry in path.iterdir()
                    if entry.name.endswith((".whl", ".tar.gz"))
                ),
                key=lambda entry: entry.name,
            )
            if not entries:
                raise DistributionBoundaryError(
                    "distribution directory contains no registered archives"
                )
            archives.extend(entries)
        elif stat.S_ISDIR(value.st_mode) or path.is_symlink() or _is_reparse(value):
            raise DistributionBoundaryError(
                "distribution input is not an ordinary file or directory"
            )
        else:
            archives.append(path)
    if len(archives) != len(set(archives)):
        raise DistributionBoundaryError("distribution input contains duplicates")
    return archives


def main() -> int:
    arguments = _parser().parse_args()
    results = []
    try:
        denylist = _load_denylist(
            arguments.denylist_file,
            required=arguments.require_denylist,
        )
        for path in _expand_inputs(arguments.archives):
            result = verify_distribution(path, denylist=denylist)
            results.append({"archive": path.name, **result})
    except (OSError, tarfile.TarError, zipfile.BadZipFile, DistributionBoundaryError):
        print(json.dumps({"status": "ERROR"}, sort_keys=True))
        return 2
    print(json.dumps({"archives": results, "status": "PASS"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
