"""Fail-closed receipts for one registered PyTorch DCP filesystem layout."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import canonical_json, exact_json_equal, strict_json_loads

RECEIPT_NAME = "checkpoint-receipt.json"
RECEIPT_SCHEMA = "dcp-invariant-checkpoint-receipt-v1"
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_RECEIPT_BYTES = 128 * 1024
_SHARD_NAME = re.compile(r"__([0-9]+)_([0-9]+)\.distcp\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class CheckpointReceiptError(ValueError):
    """The checkpoint inventory or its receipt failed closed."""


@dataclass(frozen=True)
class FileSnapshot:
    name: str
    size: int
    sha256: str

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "sha256": self.sha256, "size": self.size}


def _is_reparse(stat_result: os.stat_result) -> bool:
    attributes = getattr(stat_result, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & marker)


def _ordinary_lstat(path: Path) -> os.stat_result:
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode) or path.is_symlink() or _is_reparse(value):
        raise CheckpointReceiptError("checkpoint entry is not an ordinary file")
    if value.st_nlink != 1:
        raise CheckpointReceiptError("checkpoint entry must not have hard-link aliases")
    if value.st_size < 0 or value.st_size > MAX_FILE_BYTES:
        raise CheckpointReceiptError(
            "checkpoint entry exceeds the registered size bound"
        )
    return value


def _hash_regular_file(path: Path) -> FileSnapshot:
    before = _ordinary_lstat(path)
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as source:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_FILE_BYTES:
                raise CheckpointReceiptError(
                    "checkpoint entry exceeds the registered size bound"
                )
            digest.update(chunk)
    after = _ordinary_lstat(path)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or total != before.st_size:
        raise CheckpointReceiptError("checkpoint entry changed during snapshot")
    return FileSnapshot(path.name, total, digest.hexdigest())


def _checkpoint_entries(root: Path) -> list[Path]:
    root_stat = root.lstat()
    if (
        not stat.S_ISDIR(root_stat.st_mode)
        or root.is_symlink()
        or _is_reparse(root_stat)
    ):
        raise CheckpointReceiptError("checkpoint root is not an ordinary directory")
    entries = sorted(root.iterdir(), key=lambda path: path.name)
    for entry in entries:
        if entry.name in {".", ".."}:
            raise CheckpointReceiptError("unsafe checkpoint entry")
    return entries


def _validate_native_entries(
    entries: list[Path],
    *,
    receipt_expected: bool,
) -> list[Path]:
    names = [entry.name for entry in entries]
    receipt_count = names.count(RECEIPT_NAME)
    if receipt_expected and receipt_count != 1:
        raise CheckpointReceiptError("checkpoint receipt inventory is invalid")
    if not receipt_expected and receipt_count:
        raise CheckpointReceiptError("receipt already exists")
    native = [entry for entry in entries if entry.name != RECEIPT_NAME]
    native_names = [entry.name for entry in native]
    if native_names.count(".metadata") != 1:
        raise CheckpointReceiptError("checkpoint must contain exactly one .metadata")
    shard_names = [name for name in native_names if _SHARD_NAME.fullmatch(name)]
    if not shard_names or sorted([".metadata", *shard_names]) != native_names:
        raise CheckpointReceiptError(
            "checkpoint inventory is outside the registered shape"
        )
    rank_indices = [
        tuple(map(int, _SHARD_NAME.fullmatch(name).groups())) for name in shard_names
    ]  # type: ignore[union-attr]
    if len(rank_indices) != len(set(rank_indices)):
        raise CheckpointReceiptError("duplicate shard coordinates")
    identities = []
    for entry in native:
        value = _ordinary_lstat(entry)
        identities.append((value.st_dev, value.st_ino))
    if len(identities) != len(set(identities)):
        raise CheckpointReceiptError("checkpoint entries share a filesystem identity")
    return native


def snapshot_native_checkpoint(root: Path) -> list[FileSnapshot]:
    """Snapshot native DCP files before any receipt is present."""

    entries = _checkpoint_entries(root)
    native = _validate_native_entries(entries, receipt_expected=False)
    return [_hash_regular_file(entry) for entry in native]


def build_receipt(
    root: Path,
    *,
    logical_checkpoint_id: str,
    torch_version: str,
    state_contract_sha256: str,
) -> dict[str, Any]:
    if logical_checkpoint_id not in {"checkpoint-one", "checkpoint-two"}:
        raise CheckpointReceiptError("unregistered logical checkpoint identifier")
    if not re.fullmatch(r"2\.11\.0(?:\+cpu)?", torch_version):
        raise CheckpointReceiptError("unregistered PyTorch version")
    if not _SHA256.fullmatch(state_contract_sha256):
        raise CheckpointReceiptError("invalid state contract digest")
    files = snapshot_native_checkpoint(root)
    return {
        "files": [item.to_json() for item in files],
        "logical_checkpoint_id": logical_checkpoint_id,
        "receipt_schema": RECEIPT_SCHEMA,
        "state_contract_sha256": state_contract_sha256,
        "torch_version": torch_version,
    }


def write_receipt(root: Path, receipt: dict[str, Any]) -> None:
    target = root / RECEIPT_NAME
    if target.exists() or target.is_symlink():
        raise CheckpointReceiptError("receipt target must start absent")
    temporary = root / f".{RECEIPT_NAME}.pending"
    if temporary.exists() or temporary.is_symlink():
        raise CheckpointReceiptError("receipt staging target must start absent")
    validated = _validate_receipt_shape(receipt)
    expected = build_receipt(
        root,
        logical_checkpoint_id=validated["logical_checkpoint_id"],
        torch_version=validated["torch_version"],
        state_contract_sha256=validated["state_contract_sha256"],
    )
    if not exact_json_equal(validated, expected):
        raise CheckpointReceiptError(
            "receipt does not match the current checkpoint bytes"
        )
    encoded = (canonical_json(receipt) + "\n").encode("utf-8")
    if len(encoded) > MAX_RECEIPT_BYTES:
        raise CheckpointReceiptError("receipt exceeds size bound")
    try:
        with temporary.open("xb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, target)
        _fsync_directory(root)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_receipt(root: Path) -> dict[str, Any]:
    path = root / RECEIPT_NAME
    before = _ordinary_lstat(path)
    if before.st_size > MAX_RECEIPT_BYTES:
        raise CheckpointReceiptError("receipt exceeds size bound")
    raw = path.read_bytes()
    after = _ordinary_lstat(path)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise CheckpointReceiptError("receipt changed during read")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CheckpointReceiptError("receipt is not UTF-8") from error
    if not text.endswith("\n") or "\r" in text:
        raise CheckpointReceiptError("receipt must use one canonical LF record")
    try:
        parsed = strict_json_loads(text[:-1])
    except (TypeError, ValueError) as error:
        raise CheckpointReceiptError("receipt JSON is invalid") from error
    if not isinstance(parsed, dict):
        raise CheckpointReceiptError("receipt root must be an object")
    if canonical_json(parsed) + "\n" != text:
        raise CheckpointReceiptError("receipt JSON is not canonical")
    return parsed


def _validate_receipt_shape(receipt: object) -> dict[str, Any]:
    expected_keys = {
        "files",
        "logical_checkpoint_id",
        "receipt_schema",
        "state_contract_sha256",
        "torch_version",
    }
    if type(receipt) is not dict or receipt.keys() != expected_keys:
        raise CheckpointReceiptError("receipt field set is invalid")
    if receipt["receipt_schema"] != RECEIPT_SCHEMA:
        raise CheckpointReceiptError("receipt schema is invalid")
    if receipt["logical_checkpoint_id"] not in {
        "checkpoint-one",
        "checkpoint-two",
    }:
        raise CheckpointReceiptError("logical checkpoint identifier is invalid")
    if not isinstance(receipt["torch_version"], str) or not re.fullmatch(
        r"2\.11\.0(?:\+cpu)?", receipt["torch_version"]
    ):
        raise CheckpointReceiptError("PyTorch version is invalid")
    if not isinstance(receipt["state_contract_sha256"], str) or not _SHA256.fullmatch(
        receipt["state_contract_sha256"]
    ):
        raise CheckpointReceiptError("state contract digest is invalid")
    if not isinstance(receipt["files"], list) or not receipt["files"]:
        raise CheckpointReceiptError("receipt file list is invalid")
    return receipt


def _directory_identity(root: Path) -> tuple[int, int, int]:
    value = root.lstat()
    if not stat.S_ISDIR(value.st_mode) or root.is_symlink() or _is_reparse(value):
        raise CheckpointReceiptError("checkpoint root is not an ordinary directory")
    return value.st_dev, value.st_ino, value.st_mtime_ns


def _fsync_directory(root: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(root, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def verify_checkpoint(root: Path) -> dict[str, Any]:
    """Verify every native byte before a caller is allowed to invoke DCP load."""

    root_before = _directory_identity(root)
    receipt = _parse_receipt(root)
    validated = _validate_receipt_shape(receipt)
    first = [item.to_json() for item in snapshot_native_checkpoint_with_receipt(root)]
    second = [item.to_json() for item in snapshot_native_checkpoint_with_receipt(root)]
    receipt_after = _parse_receipt(root)
    root_after = _directory_identity(root)
    if (
        not exact_json_equal(validated["files"], first)
        or not exact_json_equal(first, second)
        or not exact_json_equal(validated, receipt_after)
        or root_before != root_after
    ):
        raise CheckpointReceiptError("checkpoint bytes do not match the receipt")
    return validated


def snapshot_native_checkpoint_with_receipt(root: Path) -> list[FileSnapshot]:
    entries = _checkpoint_entries(root)
    native = _validate_native_entries(entries, receipt_expected=True)
    return [_hash_regular_file(entry) for entry in native]
