"""Fail-closed wheel and source-distribution boundary verifier."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import struct
import tarfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath
from typing import NamedTuple

MAX_MEMBER_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_UNPACKED_BYTES = 16 * 1024 * 1024
MAX_DENYLIST_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 4096
MAX_TAR_CONTAINER_BYTES = 32 * 1024 * 1024
_USER_DIRECTORY = b"us" + b"ers"
_PRIVATE_KEY = b"PRIVATE " + b"KEY"
_AGE_SECRET_KEY = b"AGE-SECRET-" + b"KEY-"
_PUTTY_KEY_FILE = b"PuTTY-User-" + b"Key-File-"
_WXID = b"wx" + b"id_"
_XWECHAT = b"xwe" + b"chat"
_SENSITIVE_PATTERNS = {
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
_OPAQUE_MEMBER_PREFIXES = (
    b"PK\x03\x04",
    b"7z\xbc\xaf\x27\x1c",
    b"Rar!\x1a\x07",
    b"\x1f\x8b",
)
_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"
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
    is_directory: bool = False


def _safe_member_name(name: str) -> str:
    if (
        not name
        or not name.isascii()
        or "\\" in name
        or "\0" in name
        or any(ord(character) < 32 or ord(character) == 127 for character in name)
    ):
        raise DistributionBoundaryError("archive member name is invalid")
    name.encode("utf-8")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise DistributionBoundaryError("archive member path is unsafe")
    return path.as_posix()


def _is_reparse(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & marker)


def _validate_gzip_header(
    raw_archive: bytes,
    *,
    expected_filename: str,
    denylist: tuple[str, ...],
) -> None:
    if len(raw_archive) < 10 or raw_archive[:3] != b"\x1f\x8b\x08":
        raise DistributionBoundaryError(
            "source distribution has unregistered gzip metadata"
        )
    flags = raw_archive[3]
    if flags not in {0, 0x08}:
        raise DistributionBoundaryError(
            "source distribution has unregistered gzip metadata"
        )
    if flags == 0:
        return
    end = raw_archive.find(b"\0", 10, min(len(raw_archive), 10 + 4096))
    if end < 0:
        raise DistributionBoundaryError(
            "source distribution has invalid gzip filename metadata"
        )
    raw_filename = raw_archive[10:end]
    try:
        filename = raw_filename.decode("utf-8")
    except UnicodeDecodeError:
        raise DistributionBoundaryError(
            "source distribution has non-UTF-8 gzip metadata"
        ) from None
    if filename != expected_filename:
        raise DistributionBoundaryError(
            "source distribution has unregistered gzip metadata"
        )
    lowered = filename.casefold()
    if any(value in lowered for value in denylist):
        raise DistributionBoundaryError("archive matched external denylist")
    for label, pattern in _SENSITIVE_PATTERNS.items():
        if pattern.search(raw_filename):
            raise DistributionBoundaryError(
                f"archive metadata matched forbidden class {label}"
            )


_ZIP_LOCAL = struct.Struct("<4s5H3L2H")
_ZIP_CENTRAL = struct.Struct("<4s6H3L5H2L")
_ZIP_EOCD = struct.Struct("<4s4H2LH")


def _decode_zip_name(raw_name: bytes, flags: int) -> str:
    encoding = "utf-8" if flags & 0x800 else "cp437"
    try:
        return raw_name.decode(encoding)
    except UnicodeDecodeError:
        raise DistributionBoundaryError(
            "wheel member name encoding is invalid"
        ) from None


def _inflate_exact(compressed: bytes, expected_size: int) -> bytes:
    decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
    try:
        data = decompressor.decompress(compressed, expected_size + 1)
    except zlib.error:
        raise DistributionBoundaryError("wheel member decompression failed") from None
    if (
        len(data) != expected_size
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
        or decompressor.flush()
    ):
        raise DistributionBoundaryError("wheel member compressed span is not exact")
    return data


def _strict_zip_members(
    raw_archive: bytes,
    infos: list[zipfile.ZipInfo],
) -> list[ArchiveMember]:
    if (
        not infos
        or len(infos) > MAX_ARCHIVE_MEMBERS
        or len(raw_archive) < _ZIP_EOCD.size
    ):
        raise DistributionBoundaryError("wheel member inventory is invalid")
    eocd_offset = len(raw_archive) - _ZIP_EOCD.size
    try:
        (
            signature,
            disk_number,
            central_disk,
            disk_entries,
            total_entries,
            central_size,
            central_offset,
            comment_size,
        ) = _ZIP_EOCD.unpack_from(raw_archive, eocd_offset)
    except struct.error:
        raise DistributionBoundaryError("wheel EOCD is invalid") from None
    if (
        signature != b"PK\x05\x06"
        or disk_number != 0
        or central_disk != 0
        or disk_entries != total_entries
        or total_entries != len(infos)
        or total_entries == 0xFFFF
        or central_size == 0xFFFFFFFF
        or central_offset == 0xFFFFFFFF
        or comment_size != 0
        or central_offset + central_size != eocd_offset
        or raw_archive[:4] != b"PK\x03\x04"
    ):
        raise DistributionBoundaryError("wheel container span is not exact")

    central_cursor = central_offset
    for info in infos:
        try:
            central = _ZIP_CENTRAL.unpack_from(raw_archive, central_cursor)
        except struct.error:
            raise DistributionBoundaryError(
                "wheel central directory is invalid"
            ) from None
        (
            central_signature,
            _made_by,
            _needed,
            flags,
            method,
            _time,
            _date,
            crc,
            compressed_size,
            file_size,
            name_size,
            extra_size,
            member_comment_size,
            start_disk,
            _internal_attributes,
            _external_attributes,
            local_offset,
        ) = central
        name_start = central_cursor + _ZIP_CENTRAL.size
        name_end = name_start + name_size
        record_end = name_end + extra_size + member_comment_size
        if (
            central_signature != b"PK\x01\x02"
            or record_end > eocd_offset
            or extra_size != 0
            or member_comment_size != 0
            or start_disk != 0
            or flags != info.flag_bits
            or method != info.compress_type
            or crc != info.CRC
            or compressed_size != info.compress_size
            or file_size != info.file_size
            or local_offset != info.header_offset
            or _decode_zip_name(raw_archive[name_start:name_end], flags)
            != info.filename
        ):
            raise DistributionBoundaryError("wheel central directory is inconsistent")
        central_cursor = record_end
    if central_cursor != eocd_offset:
        raise DistributionBoundaryError(
            "wheel central directory has unregistered bytes"
        )

    members: list[ArchiveMember] = []
    total = 0
    local_cursor = 0
    for info in sorted(infos, key=lambda value: value.header_offset):
        if info.header_offset != local_cursor:
            raise DistributionBoundaryError("wheel local records contain a gap")
        try:
            local = _ZIP_LOCAL.unpack_from(raw_archive, local_cursor)
        except struct.error:
            raise DistributionBoundaryError("wheel local header is invalid") from None
        (
            local_signature,
            _needed,
            flags,
            method,
            _time,
            _date,
            crc,
            compressed_size,
            file_size,
            name_size,
            extra_size,
        ) = local
        name_start = local_cursor + _ZIP_LOCAL.size
        name_end = name_start + name_size
        data_start = name_end + extra_size
        data_end = data_start + compressed_size
        if (
            local_signature != b"PK\x03\x04"
            or data_end > central_offset
            or extra_size != 0
            or flags != info.flag_bits
            or flags & ~0x800
            or method != info.compress_type
            or method not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}
            or crc != info.CRC
            or compressed_size != info.compress_size
            or file_size != info.file_size
            or _decode_zip_name(raw_archive[name_start:name_end], flags)
            != info.filename
        ):
            raise DistributionBoundaryError("wheel local header is inconsistent")
        name = _safe_member_name(info.filename)
        mode = (info.external_attr >> 16) & 0xFFFF
        if stat.S_ISLNK(mode):
            raise DistributionBoundaryError("wheel contains a symbolic link")
        if info.file_size < 0 or info.file_size > MAX_MEMBER_BYTES:
            raise DistributionBoundaryError("archive member exceeds size bound")
        compressed = raw_archive[data_start:data_end]
        data = (
            compressed
            if method == zipfile.ZIP_STORED
            else _inflate_exact(compressed, info.file_size)
        )
        if (
            len(data) != info.file_size
            or zlib.crc32(data) & 0xFFFFFFFF != info.CRC
            or (info.is_dir() and data)
        ):
            raise DistributionBoundaryError("wheel member integrity is invalid")
        total += len(data)
        if total > MAX_TOTAL_UNPACKED_BYTES:
            raise DistributionBoundaryError("archive unpacked bytes exceed size bound")
        members.append(ArchiveMember(name, data, is_directory=info.is_dir()))
        local_cursor = data_end
    if local_cursor != central_offset:
        raise DistributionBoundaryError("wheel local records have unregistered bytes")
    return members


def _zip_members(raw_archive: bytes) -> list[ArchiveMember]:
    try:
        with zipfile.ZipFile(io.BytesIO(raw_archive)) as archive:
            if archive.comment:
                raise DistributionBoundaryError("wheel has unregistered ZIP metadata")
            infos = archive.infolist()
            for info in infos:
                if info.orig_filename != info.filename or info.comment or info.extra:
                    raise DistributionBoundaryError(
                        "wheel member has unregistered ZIP metadata"
                    )
            return _strict_zip_members(raw_archive, infos)
    except DistributionBoundaryError:
        raise
    except (RuntimeError, UnicodeError, ValueError, zipfile.BadZipFile, zlib.error):
        raise DistributionBoundaryError("wheel parsing failed") from None


def _validate_tar_member_metadata(info: tarfile.TarInfo) -> None:
    if (
        info.uid != 0
        or info.gid != 0
        or info.uname not in {"", "root"}
        or info.gname not in {"", "root"}
        or info.linkname
        or info.pax_headers
        or info.devmajor != 0
        or info.devminor != 0
    ):
        raise DistributionBoundaryError(
            "source distribution has unregistered TAR metadata"
        )


def _single_gzip_payload(raw_archive: bytes) -> bytes:
    decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)
    try:
        payload = decompressor.decompress(
            raw_archive,
            MAX_TAR_CONTAINER_BYTES + 1,
        )
    except zlib.error:
        raise DistributionBoundaryError(
            "source distribution decompression failed"
        ) from None
    if (
        len(payload) > MAX_TAR_CONTAINER_BYTES
        or not decompressor.eof
        or decompressor.unused_data
        or decompressor.unconsumed_tail
    ):
        raise DistributionBoundaryError(
            "source distribution gzip member span is not exact"
        )
    return payload


def _scan_raw_tar_metadata(raw_tar: bytes, denylist: tuple[str, ...]) -> None:
    lowered = raw_tar.decode("utf-8", errors="ignore").casefold()
    if any(value in lowered for value in denylist):
        raise DistributionBoundaryError("archive matched external denylist")
    for label, pattern in _SENSITIVE_PATTERNS.items():
        if pattern.search(raw_tar):
            raise DistributionBoundaryError(
                f"archive container matched forbidden class {label}"
            )


def _tar_members(
    raw_archive: bytes,
    *,
    expected_gzip_filename: str,
    denylist: tuple[str, ...],
) -> list[ArchiveMember]:
    _validate_gzip_header(
        raw_archive, expected_filename=expected_gzip_filename, denylist=denylist
    )
    members: list[ArchiveMember] = []
    total = 0
    raw_tar = _single_gzip_payload(raw_archive)
    _scan_raw_tar_metadata(raw_tar, denylist)
    try:
        with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r:") as archive:
            if archive.pax_headers:
                raise DistributionBoundaryError(
                    "source distribution has unregistered global TAR metadata"
                )
            infos = archive.getmembers()
            if not infos or len(infos) > MAX_ARCHIVE_MEMBERS:
                raise DistributionBoundaryError(
                    "source distribution member inventory is invalid"
                )
            cursor = 0
            for info in infos:
                _validate_tar_member_metadata(info)
                name = _safe_member_name(info.name)
                if (
                    info.offset != cursor
                    or info.offset_data != cursor + 512
                    or info.offset_data > len(raw_tar)
                    or raw_tar[cursor + 500 : cursor + 512] != b"\0" * 12
                ):
                    raise DistributionBoundaryError(
                        "source distribution TAR records are not exact"
                    )
                data_end = info.offset_data + info.size
                padded_end = (data_end + 511) & ~511
                if (
                    data_end > len(raw_tar)
                    or padded_end > len(raw_tar)
                    or any(raw_tar[data_end:padded_end])
                ):
                    raise DistributionBoundaryError(
                        "source distribution member padding is not zero"
                    )
                if info.isdir():
                    if info.size != 0:
                        raise DistributionBoundaryError(
                            "source distribution directory has payload bytes"
                        )
                    data = b""
                elif not info.isfile() or info.issym() or info.islnk():
                    raise DistributionBoundaryError(
                        "source distribution contains a non-ordinary member"
                    )
                else:
                    if info.size < 0 or info.size > MAX_MEMBER_BYTES:
                        raise DistributionBoundaryError(
                            "archive member exceeds size bound"
                        )
                    data = raw_tar[info.offset_data : data_end]
                total += len(data)
                if total > MAX_TOTAL_UNPACKED_BYTES:
                    raise DistributionBoundaryError(
                        "archive unpacked bytes exceed size bound"
                    )
                members.append(ArchiveMember(name, data, is_directory=info.isdir()))
                cursor = padded_end
            if (
                archive.offset != cursor
                or len(raw_tar) - cursor < 1024
                or len(raw_tar) % 512 != 0
                or any(raw_tar[cursor:])
            ):
                raise DistributionBoundaryError(
                    "source distribution EOF padding is not exact"
                )
            return members
    except DistributionBoundaryError:
        raise
    except (UnicodeError, ValueError, tarfile.TarError):
        raise DistributionBoundaryError("source distribution parsing failed") from None


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


def _read_archive(
    path: Path,
    raw_archive: bytes,
    *,
    denylist: tuple[str, ...],
) -> list[ArchiveMember]:
    name = path.name.lower()
    if name.endswith(".whl"):
        return _zip_members(raw_archive)
    if name.endswith(".tar.gz"):
        return _tar_members(
            raw_archive,
            expected_gzip_filename=path.name[:-3],
            denylist=denylist,
        )
    raise DistributionBoundaryError("distribution type is not registered")


def _snapshot_verified_distribution_inner(
    path: Path,
    *,
    denylist: tuple[str, ...] = (),
) -> tuple[dict[str, object], tuple[ArchiveMember, ...]]:
    raw_archive = _snapshot_archive(path)
    members = _read_archive(path, raw_archive, denylist=denylist)
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
            or "numpy" in lowered_parts
            or lowered_name in _FORBIDDEN_NAMES
            or lowered_name.endswith(_FORBIDDEN_SUFFIXES)
        ):
            raise DistributionBoundaryError("archive contains a forbidden payload")
        lowered_path = member.name.casefold()
        if (
            b"\0" in member.data
            or member.data.startswith(_OPAQUE_MEMBER_PREFIXES)
            or member.data.startswith(_LFS_POINTER_PREFIX)
        ):
            raise DistributionBoundaryError("archive contains an opaque payload")
        try:
            lowered_content = member.data.decode("utf-8").casefold()
        except UnicodeDecodeError:
            raise DistributionBoundaryError(
                "archive contains a non-UTF-8 payload"
            ) from None
        if any(value in lowered_path or value in lowered_content for value in denylist):
            raise DistributionBoundaryError("archive matched external denylist")
        encoded_name = member.name.encode("utf-8")
        for label, pattern in _SENSITIVE_PATTERNS.items():
            if pattern.search(encoded_name) or pattern.search(member.data):
                raise DistributionBoundaryError(
                    f"archive content matched forbidden class {label}"
                )
    return (
        {
            "archive_sha256": hashlib.sha256(raw_archive).hexdigest(),
            "member_count": len(members),
            "status": "PASS",
            "total_unpacked_bytes": total,
        },
        tuple(members),
    )


def snapshot_verified_distribution(
    path: Path,
    *,
    denylist: tuple[str, ...] = (),
) -> tuple[dict[str, object], tuple[ArchiveMember, ...]]:
    try:
        return _snapshot_verified_distribution_inner(path, denylist=denylist)
    except DistributionBoundaryError:
        raise
    except Exception:
        raise DistributionBoundaryError(
            "distribution boundary validation failed"
        ) from None


def verify_distribution(
    path: Path,
    *,
    denylist: tuple[str, ...] = (),
) -> dict[str, object]:
    return snapshot_verified_distribution(path, denylist=denylist)[0]


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
    if raw.startswith(b"\xef\xbb\xbf"):
        raise DistributionBoundaryError("denylist must not contain a UTF-8 BOM")
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
