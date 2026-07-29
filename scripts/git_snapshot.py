"""Bounded, replacement-free snapshot of fetched Git refs and HEAD."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

MAX_REF_COUNT = 4096
MAX_OBJECT_COUNT = 100_000
MAX_OBJECT_BYTES = 16 * 1024 * 1024
MAX_TOTAL_OBJECT_BYTES = 128 * 1024 * 1024
MAX_TREE_RECORDS = 250_000
MAX_TREE_OUTPUT_BYTES = 64 * 1024 * 1024
MAX_PATH_BYTES = 4096
_OBJECT_TYPES = frozenset({"blob", "commit", "tag", "tree"})
_OBJECT_FORMAT_LENGTHS = {"sha1": 40, "sha256": 64}
_SIGNATURE_ARMOR_MARKERS = (
    b"-----BEGIN PGP SIGNATURE-----",
    b"-----END PGP SIGNATURE-----",
    b"-----BEGIN PGP MESSAGE-----",
    b"-----END PGP MESSAGE-----",
    b"-----BEGIN SSH SIGNATURE-----",
    b"-----END SSH SIGNATURE-----",
    b"-----BEGIN SIGNED MESSAGE-----",
    b"-----END SIGNED MESSAGE-----",
    b"-----BEGIN PKCS7-----",
    b"-----END PKCS7-----",
    b"-----BEGIN CERTIFICATE-----",
    b"-----END CERTIFICATE-----",
)


class GitSnapshotError(ValueError):
    """The local Git checkout could not provide a bounded complete closure."""


@dataclass(frozen=True)
class RefHeadSnapshot:
    refs: tuple[tuple[str, str], ...]
    head_symbolic: str | None
    head_oid: str
    oid_length: int

    @property
    def roots(self) -> tuple[str, ...]:
        return tuple(sorted({self.head_oid, *(oid for _, oid in self.refs)}))

    @property
    def digest(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"git-ref-head-snapshot-v1\0")
        digest.update(str(self.oid_length).encode("ascii"))
        digest.update(b"\0")
        digest.update((self.head_symbolic or "<detached>").encode("utf-8"))
        digest.update(b"\0")
        digest.update(self.head_oid.encode("ascii"))
        for refname, oid in self.refs:
            digest.update(b"\0")
            digest.update(refname.encode("utf-8"))
            digest.update(b"\0")
            digest.update(oid.encode("ascii"))
        return digest.hexdigest()


@dataclass(frozen=True)
class GitObject:
    oid: str
    object_type: str
    size: int


@dataclass(frozen=True)
class GitClosure:
    snapshot: RefHeadSnapshot
    objects: tuple[GitObject, ...]
    inventory_sha256: str
    total_object_bytes: int


@dataclass(frozen=True)
class GitTreeEntry:
    mode: str
    object_type: str
    oid: str
    path: bytes


def _git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_NO_LAZY_FETCH"] = "1"
    environment["GIT_GRAFT_FILE"] = os.devnull
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    environment["LC_ALL"] = "C"
    return environment


def _run_git(
    root: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
    allowed_returncodes: frozenset[int] = frozenset({0}),
) -> subprocess.CompletedProcess[bytes]:
    try:
        completed = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "--no-lazy-fetch",
                "--no-optional-locks",
                "-c",
                "protocol.allow=never",
                "-C",
                str(root),
                *arguments,
            ],
            check=False,
            capture_output=True,
            input=input_bytes,
            env=_git_environment(),
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise GitSnapshotError("Git command could not start") from None
    if completed.returncode not in allowed_returncodes:
        raise GitSnapshotError("Git command failed")
    return completed


def _git_bytes(
    root: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
) -> bytes:
    return _run_git(root, *arguments, input_bytes=input_bytes).stdout


def _git_text(
    root: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
) -> str:
    raw = _git_bytes(root, *arguments, input_bytes=input_bytes)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        raise GitSnapshotError("Git command output is not UTF-8") from None


def _optional_git_text(root: Path, *arguments: str) -> tuple[int, str]:
    completed = _run_git(
        root,
        *arguments,
        allowed_returncodes=frozenset({0, 1}),
    )
    try:
        text = completed.stdout.decode("utf-8")
    except UnicodeDecodeError:
        raise GitSnapshotError("Git command output is not UTF-8") from None
    return completed.returncode, text


def _reject_history_override_files(root: Path) -> None:
    raw = _git_text(
        root,
        "rev-parse",
        "--path-format=absolute",
        "--git-common-dir",
    ).strip()
    if not raw:
        raise GitSnapshotError("Git common directory is empty")
    common = Path(raw)
    if not common.is_dir() or common.is_symlink():
        raise GitSnapshotError("Git common directory is not ordinary")
    for relative in (
        Path("info/grafts"),
        Path("shallow"),
        Path("objects/info/alternates"),
    ):
        path = common / relative
        if path.exists() or path.is_symlink():
            raise GitSnapshotError("unsupported Git history override is present")
    pack = common / "objects/pack"
    if not pack.is_dir() or pack.is_symlink():
        raise GitSnapshotError("Git pack directory is not ordinary")
    if any(entry.name.casefold().endswith(".promisor") for entry in pack.iterdir()):
        raise GitSnapshotError("partial Git object pack is not accepted")


def _validate_repository(root: Path) -> int:
    if _git_text(root, "rev-parse", "--is-inside-work-tree").strip() != "true":
        raise GitSnapshotError("scan root is not a Git worktree")
    if _git_text(root, "rev-parse", "--is-shallow-repository").strip() != "false":
        raise GitSnapshotError("shallow Git history is not accepted")
    object_format = _git_text(
        root,
        "rev-parse",
        "--show-object-format",
    ).strip()
    try:
        oid_length = _OBJECT_FORMAT_LENGTHS[object_format]
    except KeyError:
        raise GitSnapshotError("Git object format is not registered") from None
    config = _git_bytes(
        root,
        "config",
        "--local",
        "--null",
        "--name-only",
        "--list",
    )
    if config and not config.endswith(b"\0"):
        raise GitSnapshotError("local Git config inventory is malformed")
    try:
        config_keys = [
            value.decode("utf-8").casefold() for value in config.split(b"\0") if value
        ]
    except UnicodeDecodeError:
        raise GitSnapshotError("local Git config keys are not UTF-8") from None
    if any(
        key == "extensions.partialclone"
        or (key.startswith("remote.") and key.endswith(".promisor"))
        for key in config_keys
    ):
        raise GitSnapshotError("partial Git history is not accepted")
    _reject_history_override_files(root)
    return oid_length


def _valid_oid(value: str, *, length: int) -> bool:
    return len(value) == length and re.fullmatch(r"[0-9a-f]+", value) is not None


def read_snapshot(root: Path) -> RefHeadSnapshot:
    root = root.resolve(strict=True)
    oid_length = _validate_repository(root)
    raw_refs = _git_text(
        root,
        "for-each-ref",
        f"--count={MAX_REF_COUNT + 1}",
        "--sort=refname",
        "--format=%(refname)%00%(objectname)",
    )
    refs: list[tuple[str, str]] = []
    for record in raw_refs.splitlines():
        fields = record.split("\0")
        if (
            len(fields) != 2
            or not fields[0].startswith("refs/")
            or _valid_oid(fields[1], length=oid_length) is False
        ):
            raise GitSnapshotError("Git ref inventory is malformed")
        if fields[0].startswith("refs/replace/"):
            raise GitSnapshotError("Git replace refs are not accepted")
        refs.append((fields[0], fields[1]))
    if not refs or len(refs) > MAX_REF_COUNT or len(refs) != len(set(refs)):
        raise GitSnapshotError("Git ref inventory is outside bounds")

    symbolic_code, symbolic_text = _optional_git_text(
        root,
        "symbolic-ref",
        "-q",
        "HEAD",
    )
    head_symbolic = symbolic_text.strip() if symbolic_code == 0 else None
    if head_symbolic is not None and not head_symbolic.startswith("refs/"):
        raise GitSnapshotError("symbolic HEAD is malformed")
    head_oid = _git_text(root, "rev-parse", "--verify", "HEAD").strip()
    if not _valid_oid(head_oid, length=oid_length):
        raise GitSnapshotError("HEAD object identifier is malformed")
    refs_tuple = tuple(sorted(refs))
    if head_symbolic is not None and dict(refs_tuple).get(head_symbolic) != head_oid:
        raise GitSnapshotError("symbolic HEAD does not match its ref")
    snapshot = RefHeadSnapshot(
        refs=refs_tuple,
        head_symbolic=head_symbolic,
        head_oid=head_oid,
        oid_length=oid_length,
    )
    for root_oid in snapshot.roots:
        peeled = _git_text(
            root,
            "rev-parse",
            "--verify",
            f"{root_oid}^{{commit}}",
        ).strip()
        if not _valid_oid(peeled, length=oid_length):
            raise GitSnapshotError("fetched ref does not peel to a commit")
    return snapshot


def freeze_closure(root: Path) -> GitClosure:
    root = root.resolve(strict=True)
    snapshot = read_snapshot(root)
    repeated = read_snapshot(root)
    if repeated != snapshot:
        raise GitSnapshotError("Git refs or HEAD changed while freezing")
    roots = ("\n".join(snapshot.roots) + "\n").encode("ascii")
    raw_inventory = _git_bytes(
        root,
        "rev-list",
        "--objects",
        "--no-object-names",
        "--missing=error",
        "--stdin",
        input_bytes=roots,
    )
    try:
        object_names = {
            line for line in raw_inventory.decode("ascii").splitlines() if line
        }
    except UnicodeDecodeError:
        raise GitSnapshotError("reachable object inventory is malformed") from None
    object_names.update(snapshot.roots)
    if (
        not object_names
        or len(object_names) > MAX_OBJECT_COUNT
        or any(
            not _valid_oid(value, length=snapshot.oid_length) for value in object_names
        )
    ):
        raise GitSnapshotError("reachable object inventory is outside bounds")

    objects: list[GitObject] = []
    total_size = 0
    for oid in sorted(object_names):
        object_type = _git_text(root, "cat-file", "-t", oid).strip()
        size_text = _git_text(root, "cat-file", "-s", oid).strip()
        if object_type not in _OBJECT_TYPES or not size_text.isdecimal():
            raise GitSnapshotError("reachable Git object metadata is malformed")
        size = int(size_text)
        if size < 0 or size > MAX_OBJECT_BYTES:
            raise GitSnapshotError("reachable Git object exceeds size bound")
        total_size += size
        if total_size > MAX_TOTAL_OBJECT_BYTES:
            raise GitSnapshotError("reachable Git objects exceed total size bound")
        objects.append(GitObject(oid=oid, object_type=object_type, size=size))

    digest = hashlib.sha256()
    digest.update(b"git-reachable-object-inventory-v1\0")
    digest.update(snapshot.digest.encode("ascii"))
    for value in objects:
        digest.update(b"\0")
        digest.update(value.oid.encode("ascii"))
        digest.update(b"\0")
        digest.update(value.object_type.encode("ascii"))
        digest.update(b"\0")
        digest.update(str(value.size).encode("ascii"))
    return GitClosure(
        snapshot=snapshot,
        objects=tuple(objects),
        inventory_sha256=digest.hexdigest(),
        total_object_bytes=total_size,
    )


def _reject_signed_object(object_type: str, raw: bytes) -> None:
    if object_type not in {"commit", "tag"}:
        return
    header = raw.partition(b"\n\n")[0]
    for line in header.splitlines():
        if not line or line.startswith(b" "):
            continue
        field = line.partition(b" ")[0].lower()
        if field == b"mergetag" or field == b"gpgsig" or field.startswith(b"gpgsig-"):
            raise GitSnapshotError("reachable Git signature metadata is not permitted")
    if any(marker in raw for marker in _SIGNATURE_ARMOR_MARKERS):
        raise GitSnapshotError("reachable Git signature payload is not permitted")


def read_object(root: Path, value: GitObject) -> bytes:
    raw = _git_bytes(
        root.resolve(strict=True),
        "cat-file",
        value.object_type,
        value.oid,
    )
    if len(raw) != value.size:
        raise GitSnapshotError("reachable Git object changed during read")
    algorithm = "sha1" if len(value.oid) == 40 else "sha256"
    digest = hashlib.new(algorithm)
    digest.update(f"{value.object_type} {value.size}".encode("ascii"))
    digest.update(b"\0")
    digest.update(raw)
    if digest.hexdigest() != value.oid:
        raise GitSnapshotError("reachable Git object hash is invalid")
    _reject_signed_object(value.object_type, raw)
    return raw


def list_commit_entries(root: Path, commit_oid: str) -> tuple[GitTreeEntry, ...]:
    raw = _git_bytes(
        root.resolve(strict=True),
        "ls-tree",
        "-r",
        "-z",
        "--full-tree",
        "--format=%(objectmode)%x00%(objecttype)%x00%(objectname)%x00%(path)",
        commit_oid,
    )
    if len(raw) > MAX_TREE_OUTPUT_BYTES:
        raise GitSnapshotError("Git path inventory exceeds byte bound")
    if not raw:
        return ()
    fields = raw.split(b"\0")
    if fields[-1] != b"" or (len(fields) - 1) % 4 != 0:
        raise GitSnapshotError("Git path inventory is malformed")
    entries: list[GitTreeEntry] = []
    for offset in range(0, len(fields) - 1, 4):
        mode_raw, object_type_raw, oid_raw, path = fields[offset : offset + 4]
        try:
            mode = mode_raw.decode("ascii")
            object_type = object_type_raw.decode("ascii")
            oid = oid_raw.decode("ascii")
        except UnicodeDecodeError:
            raise GitSnapshotError("Git path metadata is malformed") from None
        if (
            not mode.isdecimal()
            or object_type not in {"blob", "commit"}
            or not _valid_oid(oid, length=len(commit_oid))
            or not path
            or len(path) > MAX_PATH_BYTES
            or path.startswith((b"/", b"\\"))
            or b"\0" in path
        ):
            raise GitSnapshotError("Git path entry is outside bounds")
        entries.append(
            GitTreeEntry(
                mode=mode,
                object_type=object_type,
                oid=oid,
                path=path,
            )
        )
        if len(entries) > MAX_TREE_RECORDS:
            raise GitSnapshotError("Git path inventory exceeds record bound")
    return tuple(entries)


def assert_snapshot_unchanged(root: Path, expected: RefHeadSnapshot) -> None:
    if read_snapshot(root.resolve(strict=True)) != expected:
        raise GitSnapshotError("Git refs or HEAD changed during audit")
