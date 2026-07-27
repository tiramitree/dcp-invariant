"""Fail-closed repository and evidence privacy scanner."""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from pathlib import Path

MAX_TEXT_BYTES = 2 * 1024 * 1024
_USER_DIRECTORY = b"us" + b"ers"
_PATTERNS = {
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
    "url-embedded-credential": re.compile(rb"(?i)https?://[^/\s:@]+:[^/\s@]+@"),
    "credential-assignment": re.compile(
        rb"(?i)\b(?:"
        + b"api"
        + rb"[_-]?key|access[_-]?token|pass"
        + b"word"
        + rb"|secret)\s*[:=]\s*[\"'][^\"'\r\n]{8,}[\"']"
    ),
}
_FORBIDDEN_NATIVE_NAMES = {
    ".metadata",
    "checkpoint-receipt.json",
}
_FORBIDDEN_NATIVE_SUFFIXES = (".distcp",)


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

    def finding_path(relative: str) -> str:
        if any(value in relative.casefold() for value in denylist):
            return "<redacted-path>"
        return relative

    findings: list[dict[str, str]] = []
    for current, directories, files in os.walk(
        root,
        topdown=True,
        onerror=raise_walk_error,
        followlinks=False,
    ):
        current_path = Path(current)
        if current_path == root and ".git" in directories:
            git_path = root / ".git"
            git_stat = git_path.lstat()
            if (
                not stat.S_ISDIR(git_stat.st_mode)
                or git_path.is_symlink()
                or is_reparse(git_stat)
            ):
                raise ValueError(".git is not an ordinary root-level directory")
            directories.remove(".git")
        directories.sort()
        files.sort()

        unsafe_directories: list[str] = []
        for name in directories:
            path = current_path / name
            relative = safe_relative(root, path)
            if any(value in relative.casefold() for value in denylist):
                findings.append({"kind": "denylist-path", "path": "<redacted-path>"})
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
            if any(value in relative.casefold() for value in denylist):
                findings.append({"kind": "denylist-path", "path": "<redacted-path>"})

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
            for label, pattern in _PATTERNS.items():
                if pattern.search(raw):
                    findings.append({"kind": label, "path": reported})
    return findings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--denylist-file", type=Path)
    parser.add_argument("--require-denylist", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        denylist = load_denylist(
            args.denylist_file,
            required=args.require_denylist,
        )
        findings = scan(args.root, denylist=denylist)
    except (OSError, UnicodeError, ValueError) as error:
        print(json.dumps({"error": type(error).__name__, "status": "ERROR"}))
        return 2
    print(
        json.dumps(
            {
                "denylist_enforced": bool(denylist),
                "finding_count": len(findings),
                "findings": findings,
                "status": "PASS" if not findings else "FAIL",
            },
            sort_keys=True,
        )
    )
    return 0 if not findings else 1


if __name__ == "__main__":
    sys.exit(main())
