"""Cross-platform local worker supervision and lineage-bound promotion."""

from __future__ import annotations

import errno
import hashlib
import os
import re
import secrets
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from .canonical import canonical_json, strict_json_loads
from .checkpoint_receipt import RECEIPT_NAME

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
LEGACY_LATEST_SCHEMA = "dcp-invariant-latest-v1"
LATEST_SCHEMA = "dcp-invariant-latest-v2"
LINEAGE_SCHEMA = "dcp-invariant-generation-lineage-v1"
LINEAGE_RECORD_NAME = "generation-lineage.json"
POINTER_LOCK_NAME = ".LATEST.commit.lock"
REGISTERED_CHECKPOINT_IDS = {"checkpoint-async", "checkpoint-one", "checkpoint-two"}


class SupervisorError(RuntimeError):
    """A registered worker or promotion invariant failed."""


@dataclass(frozen=True)
class ParentVersion:
    pointer_sha256: str | None
    sequence: int


@dataclass(frozen=True)
class CommittedGeneration:
    target: Path
    logical_checkpoint_id: str
    generation: str
    lineage_sha256: str
    parent_pointer_sha256: str | None
    sequence: int
    reused: bool


class StaleParentError(SupervisorError):
    """A committed generation no longer descends from the selected parent."""

    def __init__(self, committed: CommittedGeneration) -> None:
        self.committed = committed
        super().__init__("stale_parent")


@dataclass(frozen=True)
class WorkerResult:
    exit_codes: tuple[int, ...]
    timed_out: bool


class WorkerOutcomeError(SupervisorError):
    """A worker group did not produce the exact registered exit vector."""

    def __init__(
        self,
        result: WorkerResult,
        expected_exit_codes: tuple[int, ...],
    ) -> None:
        self.result = result
        self.expected_exit_codes = expected_exit_codes
        super().__init__(
            "registered worker outcome failed"
            f" (timed_out={result.timed_out}, exit_codes={result.exit_codes})"
        )


def _is_reparse(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & marker)


def _ordinary_directory(path: Path) -> None:
    try:
        value = path.lstat()
    except OSError as error:
        raise SupervisorError("worker directory cannot be inspected") from error
    if not stat.S_ISDIR(value.st_mode) or path.is_symlink() or _is_reparse(value):
        raise SupervisorError("worker directory is not an ordinary directory")


def _entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _ordinary_file(path: Path, *, maximum_bytes: int) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as error:
        raise SupervisorError("promotion file cannot be inspected") from error
    if not stat.S_ISREG(value.st_mode) or path.is_symlink() or _is_reparse(value):
        raise SupervisorError("promotion file is not an ordinary file")
    if value.st_size < 0 or value.st_size > maximum_bytes:
        raise SupervisorError("promotion file exceeds its registered size bound")
    return value


def _hash_ordinary_file(path: Path, *, maximum_bytes: int) -> str:
    before = _ordinary_file(path, maximum_bytes=maximum_bytes)
    digest = hashlib.sha256()
    total = 0
    try:
        with path.open("rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > maximum_bytes:
                    raise SupervisorError(
                        "promotion file exceeds its registered size bound"
                    )
                digest.update(chunk)
    except OSError as error:
        raise SupervisorError("promotion file cannot be read") from error
    after = _ordinary_file(path, maximum_bytes=maximum_bytes)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or total != before.st_size:
        raise SupervisorError("promotion file changed during hashing")
    return digest.hexdigest()


def _direct_child(parent: Path, child: Path) -> None:
    try:
        parent_resolved = parent.resolve(strict=True)
        child_resolved = child.resolve(strict=True)
    except OSError as error:
        raise SupervisorError(
            "registered parent or child cannot be resolved"
        ) from error
    if child_resolved.parent != parent_resolved:
        raise SupervisorError("path is outside its registered parent")


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def minimal_worker_environment(
    isolated_home: Path | None = None,
) -> dict[str, str]:
    """Return a small launch environment without credentials or network proxies."""

    allow = {
        "COMSPEC",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "WINDIR",
    }
    result = {key: value for key, value in os.environ.items() if key in allow}
    result.update(
        {
            "OMP_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
            "USE_LIBUV": "0",
        }
    )
    if isolated_home is not None:
        _ordinary_directory(isolated_home)
        isolated = str(isolated_home.resolve(strict=True))
        result["HOME"] = isolated
        result["LNAME"] = "dcp-invariant"
        result["LOGNAME"] = "dcp-invariant"
        result["TEMP"] = isolated
        result["TMP"] = isolated
        result["USER"] = "dcp-invariant"
        result["USERNAME"] = "dcp-invariant"
        if os.name == "nt":
            result["USERPROFILE"] = isolated
    return result


def run_workers(
    *,
    module: str,
    common_arguments: Sequence[str],
    world_size: int,
    cwd: Path,
    isolated_home: Path | None = None,
    timeout_seconds: float = 30.0,
    expected_exit_codes: Sequence[int] | None = None,
) -> WorkerResult:
    """Run one registered local process group without exposing raw worker logs."""

    if not re.fullmatch(r"[a-z_][a-z0-9_.]*", module):
        raise SupervisorError("worker module name is invalid")
    if world_size not in {1, 2}:
        raise SupervisorError("only one or two local workers are registered")
    if not (0.1 <= timeout_seconds <= 300.0):
        raise SupervisorError("worker timeout is outside the registered bound")
    _ordinary_directory(cwd)
    if isolated_home is None:
        isolated_home = cwd
    _ordinary_directory(isolated_home)
    port = free_loopback_port()
    processes: list[subprocess.Popen[bytes]] = []
    creationflags = (
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    )
    try:
        for rank in range(world_size):
            command = [
                sys.executable,
                "-m",
                module,
                *common_arguments,
                "--rank",
                str(rank),
                "--world-size",
                str(world_size),
                "--master-port",
                str(port),
            ]
            processes.append(
                subprocess.Popen(
                    command,
                    cwd=cwd,
                    env=minimal_worker_environment(isolated_home),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                    start_new_session=os.name != "nt",
                )
            )

        deadline = time.monotonic() + timeout_seconds
        timed_out = False
        exit_codes: list[int] = []
        for process in processes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            try:
                exit_codes.append(process.wait(timeout=remaining))
            except subprocess.TimeoutExpired:
                timed_out = True
                break
        if timed_out:
            for process in processes:
                if process.poll() is None:
                    process.kill()
            exit_codes = [process.wait(timeout=10) for process in processes]
        else:
            exit_codes.extend(
                process.wait(timeout=10) for process in processes[len(exit_codes) :]
            )
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)

    result = WorkerResult(tuple(exit_codes), timed_out)
    if expected_exit_codes is None:
        expected_exit_codes = [0] * world_size
    if len(expected_exit_codes) != world_size:
        raise SupervisorError("expected exit-code vector has the wrong size")
    expected = tuple(expected_exit_codes)
    if result.timed_out or result.exit_codes != expected:
        raise WorkerOutcomeError(result, expected)
    return result


def _stable_file_bytes(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    before = _ordinary_file(path, maximum_bytes=maximum_bytes)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SupervisorError(f"{label} cannot be read") from error
    after = _ordinary_file(path, maximum_bytes=maximum_bytes)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or len(raw) != before.st_size:
        raise SupervisorError(f"{label} changed during read")
    return raw


def _canonical_object(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        text = raw.decode("utf-8")
        parsed = strict_json_loads(text[:-1])
    except (UnicodeError, TypeError, ValueError) as error:
        raise SupervisorError(f"{label} is invalid") from error
    if (
        not text.endswith("\n")
        or "\r" in text
        or canonical_json(parsed) + "\n" != text
        or type(parsed) is not dict
    ):
        raise SupervisorError(f"{label} is invalid")
    return parsed


def _validate_parent_sha256(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise SupervisorError(f"{label} is invalid")
    return value


def _validate_sequence(value: object, *, label: str) -> int:
    if type(value) is not int or not (0 <= value <= (2**63 - 1)):
        raise SupervisorError(f"{label} is invalid")
    return value


def _read_pointer(
    path: Path,
) -> tuple[dict[str, object] | None, ParentVersion]:
    if not _entry_exists(path):
        return None, ParentVersion(pointer_sha256=None, sequence=-1)
    raw = _stable_file_bytes(path, maximum_bytes=2048, label="existing pointer")
    parsed = _canonical_object(raw, label="existing pointer")
    schema = parsed.get("pointer_schema")
    if schema == LEGACY_LATEST_SCHEMA:
        if (
            set(parsed) != {"generation", "pointer_schema"}
            or type(parsed["generation"]) is not str
            or _SHA256.fullmatch(parsed["generation"]) is None
        ):
            raise SupervisorError("existing pointer is invalid")
        sequence = -1
    elif schema == LATEST_SCHEMA:
        if set(parsed) != {
            "generation",
            "lineage_sha256",
            "parent_pointer_sha256",
            "pointer_schema",
            "sequence",
        }:
            raise SupervisorError("existing pointer is invalid")
        if (
            type(parsed["generation"]) is not str
            or _SHA256.fullmatch(parsed["generation"]) is None
            or type(parsed["lineage_sha256"]) is not str
            or _SHA256.fullmatch(parsed["lineage_sha256"]) is None
        ):
            raise SupervisorError("existing pointer is invalid")
        parent = _validate_parent_sha256(
            parsed["parent_pointer_sha256"],
            label="existing pointer parent",
        )
        sequence = _validate_sequence(
            parsed["sequence"],
            label="existing pointer sequence",
        )
        if sequence > 0 and parent is None:
            raise SupervisorError("existing pointer is invalid")
    else:
        raise SupervisorError("existing pointer is invalid")
    if schema == LATEST_SCHEMA:
        generation = str(parsed["generation"])
        committed_root = path.parent / "committed"
        _ordinary_directory(committed_root)
        target = committed_root / generation
        _direct_child(committed_root, target)
        observed = _read_committed_generation(target)
        if (
            observed.generation != generation
            or observed.lineage_sha256 != parsed["lineage_sha256"]
            or observed.parent_pointer_sha256 != parsed["parent_pointer_sha256"]
            or observed.sequence != sequence
        ):
            raise SupervisorError(
                "existing pointer is not bound to its committed lineage"
            )
        if (
            _hash_ordinary_file(
                observed.target / observed.logical_checkpoint_id / RECEIPT_NAME,
                maximum_bytes=128 * 1024,
            )
            != generation
        ):
            raise SupervisorError(
                "existing pointer is not bound to its committed receipt"
            )
    return parsed, ParentVersion(
        pointer_sha256=hashlib.sha256(raw).hexdigest(),
        sequence=sequence,
    )


def read_parent_version(root: Path) -> ParentVersion:
    """Capture the exact selected parent before expensive candidate work."""

    _ordinary_directory(root)
    _, version = _read_pointer(root / "LATEST.json")
    return version


def _write_exclusive(path: Path, value: bytes, *, label: str) -> None:
    try:
        with path.open("xb") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise SupervisorError(f"{label} cannot be created") from error


def _write_atomic(path: Path, value: bytes) -> None:
    temporary: Path | None = None
    for _ in range(16):
        candidate = path.with_name(f".{path.name}.{secrets.token_hex(12)}.pending")
        try:
            with candidate.open("xb") as output:
                temporary = candidate
                output.write(value)
                output.flush()
                os.fsync(output.fileno())
            break
        except FileExistsError:
            continue
        except OSError as error:
            raise SupervisorError("pointer staging target cannot be created") from error
    if temporary is None:
        raise SupervisorError("pointer staging target could not be reserved")
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if _entry_exists(temporary):
            try:
                temporary.unlink()
            except OSError as error:
                raise SupervisorError(
                    "pointer staging target cannot be removed"
                ) from error


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _try_pointer_lock(descriptor: int) -> bool:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError as error:
            if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                return False
            raise
        return True
    import fcntl

    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _unlock_pointer(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)


def _validate_open_lock(path: Path, descriptor: int) -> None:
    try:
        opened = os.fstat(descriptor)
        linked = path.lstat()
    except OSError as error:
        raise SupervisorError("pointer commit lock cannot be inspected") from error
    if (
        not stat.S_ISREG(opened.st_mode)
        or not stat.S_ISREG(linked.st_mode)
        or path.is_symlink()
        or _is_reparse(linked)
        or opened.st_nlink != 1
        or linked.st_nlink != 1
        or (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino)
    ):
        raise SupervisorError("pointer commit lock is not one ordinary file")
    if opened.st_size == 0:
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.write(descriptor, b"\0") != 1:
            raise SupervisorError("pointer commit lock initialization was incomplete")
        os.fsync(descriptor)
    elif opened.st_size == 1:
        os.lseek(descriptor, 0, os.SEEK_SET)
        if os.read(descriptor, 1) != b"\0":
            raise SupervisorError("pointer commit lock has invalid content")
    else:
        raise SupervisorError("pointer commit lock has invalid size")
    opened_after = os.fstat(descriptor)
    linked_after = path.lstat()
    if (
        opened_after.st_size != 1
        or linked_after.st_size != 1
        or (opened_after.st_dev, opened_after.st_ino)
        != (linked_after.st_dev, linked_after.st_ino)
    ):
        raise SupervisorError("pointer commit lock changed during validation")


@contextmanager
def _pointer_commit_lock(
    root: Path,
    *,
    timeout_seconds: float,
) -> Iterator[None]:
    if not (0.1 <= timeout_seconds <= 30.0):
        raise SupervisorError("pointer commit lock timeout is outside its bound")
    path = root / POINTER_LOCK_NAME
    flags = os.O_RDWR | os.O_CREAT
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOINHERIT", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError as error:
        raise SupervisorError("pointer commit lock cannot be opened") from error
    acquired = False
    try:
        deadline = time.monotonic() + timeout_seconds
        while not acquired:
            try:
                acquired = _try_pointer_lock(descriptor)
            except OSError as error:
                raise SupervisorError("pointer commit lock failed") from error
            if acquired:
                break
            if time.monotonic() >= deadline:
                raise SupervisorError("pointer commit lock timed out")
            time.sleep(0.01)
        _validate_open_lock(path, descriptor)
        yield
    finally:
        try:
            if acquired:
                _unlock_pointer(descriptor)
        finally:
            os.close(descriptor)


def _lineage_payload(
    *,
    generation: str,
    logical_checkpoint_id: str,
    parent: ParentVersion,
) -> dict[str, object]:
    sequence = parent.sequence + 1
    if not (0 <= sequence <= (2**63 - 1)):
        raise SupervisorError("generation lineage sequence is outside its bound")
    return {
        "generation": generation,
        "lineage_schema": LINEAGE_SCHEMA,
        "logical_checkpoint_id": logical_checkpoint_id,
        "parent_pointer_sha256": parent.pointer_sha256,
        "sequence": sequence,
    }


def _read_committed_generation(target: Path) -> CommittedGeneration:
    _ordinary_directory(target)
    if type(target.name) is not str or _SHA256.fullmatch(target.name) is None:
        raise SupervisorError("committed generation name is invalid")
    record_path = target / LINEAGE_RECORD_NAME
    raw = _stable_file_bytes(
        record_path,
        maximum_bytes=2048,
        label="generation lineage record",
    )
    record = _canonical_object(raw, label="generation lineage record")
    if set(record) != {
        "generation",
        "lineage_schema",
        "logical_checkpoint_id",
        "parent_pointer_sha256",
        "sequence",
    }:
        raise SupervisorError("generation lineage record is invalid")
    if (
        record["lineage_schema"] != LINEAGE_SCHEMA
        or record["generation"] != target.name
        or type(record["logical_checkpoint_id"]) is not str
        or record["logical_checkpoint_id"] not in REGISTERED_CHECKPOINT_IDS
    ):
        raise SupervisorError("generation lineage record is invalid")
    parent = _validate_parent_sha256(
        record["parent_pointer_sha256"],
        label="generation lineage parent",
    )
    sequence = _validate_sequence(
        record["sequence"],
        label="generation lineage sequence",
    )
    if sequence > 0 and parent is None:
        raise SupervisorError("generation lineage record is invalid")
    checkpoint_id = str(record["logical_checkpoint_id"])
    try:
        entries = sorted(entry.name for entry in target.iterdir())
    except OSError as error:
        raise SupervisorError("committed generation cannot be enumerated") from error
    if entries != sorted([LINEAGE_RECORD_NAME, checkpoint_id]):
        raise SupervisorError("committed generation field set is invalid")
    checkpoint = target / checkpoint_id
    _ordinary_directory(checkpoint)
    _direct_child(target, checkpoint)
    return CommittedGeneration(
        target=target,
        logical_checkpoint_id=checkpoint_id,
        generation=target.name,
        lineage_sha256=hashlib.sha256(raw).hexdigest(),
        parent_pointer_sha256=parent,
        sequence=sequence,
        reused=False,
    )


def _same_committed_generation(
    observed: CommittedGeneration,
    expected: CommittedGeneration,
) -> bool:
    return (
        observed.target == expected.target
        and observed.logical_checkpoint_id == expected.logical_checkpoint_id
        and observed.generation == expected.generation
        and observed.lineage_sha256 == expected.lineage_sha256
        and observed.parent_pointer_sha256 == expected.parent_pointer_sha256
        and observed.sequence == expected.sequence
    )


def commit_candidate(
    *,
    root: Path,
    candidate: Path,
    logical_checkpoint_id: str,
    verify: Callable[[Path], bool],
    parent: ParentVersion,
) -> CommittedGeneration:
    """Commit one receipt-bound generation without selecting it."""

    if (
        type(logical_checkpoint_id) is not str
        or logical_checkpoint_id not in REGISTERED_CHECKPOINT_IDS
    ):
        raise SupervisorError("logical checkpoint identifier is not registered")
    if type(parent) is not ParentVersion:
        raise SupervisorError("parent version is invalid")
    if (
        (
            parent.pointer_sha256 is not None
            and (
                type(parent.pointer_sha256) is not str
                or _SHA256.fullmatch(parent.pointer_sha256) is None
            )
        )
        or type(parent.sequence) is not int
        or not (-1 <= parent.sequence <= (2**63 - 2))
    ):
        raise SupervisorError("parent version is invalid")
    if read_parent_version(root) != parent:
        raise SupervisorError("parent_version_invalid")
    candidates = root / "candidates"
    committed = root / "committed"
    _ordinary_directory(candidates)
    _ordinary_directory(committed)
    _ordinary_directory(candidate)
    _direct_child(candidates, candidate)
    try:
        candidate_entries = {entry.name for entry in candidate.iterdir()}
    except OSError as error:
        raise SupervisorError("candidate wrapper cannot be enumerated") from error
    if candidate_entries not in (
        {logical_checkpoint_id},
        {logical_checkpoint_id, LINEAGE_RECORD_NAME},
    ):
        raise SupervisorError("candidate wrapper contains an unregistered entry")
    checkpoint = candidate / logical_checkpoint_id
    _ordinary_directory(checkpoint)
    _direct_child(candidate, checkpoint)
    if verify(checkpoint) is not True:
        raise SupervisorError("candidate verifier did not return exact success")
    receipt_sha256 = _hash_ordinary_file(
        checkpoint / RECEIPT_NAME,
        maximum_bytes=128 * 1024,
    )
    target = committed / receipt_sha256
    payload = _lineage_payload(
        generation=receipt_sha256,
        logical_checkpoint_id=logical_checkpoint_id,
        parent=parent,
    )
    encoded = (canonical_json(payload) + "\n").encode("utf-8")
    expected = CommittedGeneration(
        target=target,
        logical_checkpoint_id=logical_checkpoint_id,
        generation=receipt_sha256,
        lineage_sha256=hashlib.sha256(encoded).hexdigest(),
        parent_pointer_sha256=parent.pointer_sha256,
        sequence=parent.sequence + 1,
        reused=False,
    )

    record_path = candidate / LINEAGE_RECORD_NAME
    if _entry_exists(record_path) and (
        _stable_file_bytes(
            record_path,
            maximum_bytes=2048,
            label="generation lineage record",
        )
        != encoded
    ):
        raise SupervisorError("generation_lineage_conflict")

    if _entry_exists(target):
        observed = _read_committed_generation(target)
        if not _same_committed_generation(observed, expected):
            raise SupervisorError("generation_lineage_conflict")
        if verify(target / logical_checkpoint_id) is not True:
            raise SupervisorError("existing committed verifier did not return success")
        if (
            _hash_ordinary_file(
                target / logical_checkpoint_id / RECEIPT_NAME,
                maximum_bytes=128 * 1024,
            )
            != receipt_sha256
        ):
            raise SupervisorError("existing committed receipt digest changed")
        return replace(observed, reused=True)

    created_lineage = False
    if not _entry_exists(record_path):
        _write_exclusive(record_path, encoded, label="generation lineage record")
        created_lineage = True
    try:
        candidate.rename(target)
    except OSError as error:
        collision = error.errno in {errno.EEXIST, errno.ENOTEMPTY}
        if not collision or not _entry_exists(target):
            if created_lineage:
                try:
                    record_path.unlink()
                except OSError as cleanup_error:
                    raise SupervisorError(
                        "candidate lineage cleanup failed after commit error"
                    ) from cleanup_error
            raise SupervisorError("candidate generation cannot be committed") from error
        if created_lineage:
            try:
                record_path.unlink()
            except OSError as cleanup_error:
                raise SupervisorError(
                    "candidate lineage cleanup failed after commit collision"
                ) from cleanup_error
        observed = _read_committed_generation(target)
        if not _same_committed_generation(observed, expected):
            raise SupervisorError("generation_lineage_conflict") from None
        if verify(target / logical_checkpoint_id) is not True:
            raise SupervisorError(
                "existing committed verifier did not return success"
            ) from None
        if (
            _hash_ordinary_file(
                target / logical_checkpoint_id / RECEIPT_NAME,
                maximum_bytes=128 * 1024,
            )
            != receipt_sha256
        ):
            raise SupervisorError("existing committed receipt digest changed") from None
        return replace(observed, reused=True)
    _fsync_directory(candidates)
    _fsync_directory(committed)

    observed = _read_committed_generation(target)
    if not _same_committed_generation(observed, expected):
        raise SupervisorError("committed generation lineage changed")
    committed_checkpoint = target / logical_checkpoint_id
    if verify(committed_checkpoint) is not True:
        raise SupervisorError(
            "committed generation verifier did not return exact success"
        )
    committed_receipt_sha256 = _hash_ordinary_file(
        committed_checkpoint / RECEIPT_NAME,
        maximum_bytes=128 * 1024,
    )
    if committed_receipt_sha256 != receipt_sha256:
        raise SupervisorError("committed receipt digest changed during promotion")
    return observed


def load_committed_generation(
    *,
    root: Path,
    generation: str,
    logical_checkpoint_id: str,
    verify: Callable[[Path], bool],
) -> CommittedGeneration:
    """Reconstruct a verified descriptor after the commit/publish crash window."""

    _ordinary_directory(root)
    if (
        type(generation) is not str
        or _SHA256.fullmatch(generation) is None
        or type(logical_checkpoint_id) is not str
        or logical_checkpoint_id not in REGISTERED_CHECKPOINT_IDS
    ):
        raise SupervisorError("committed generation recovery input is invalid")
    committed_root = root / "committed"
    _ordinary_directory(committed_root)
    target = committed_root / generation
    _direct_child(committed_root, target)
    observed = _read_committed_generation(target)
    if observed.logical_checkpoint_id != logical_checkpoint_id:
        raise SupervisorError("committed generation checkpoint identifier changed")
    checkpoint = target / logical_checkpoint_id
    if verify(checkpoint) is not True:
        raise SupervisorError(
            "recovered committed verifier did not return exact success"
        )
    if (
        _hash_ordinary_file(
            checkpoint / RECEIPT_NAME,
            maximum_bytes=128 * 1024,
        )
        != generation
    ):
        raise SupervisorError("recovered committed receipt digest changed")
    return observed


def _pointer_for_generation(
    committed: CommittedGeneration,
) -> dict[str, object]:
    return {
        "generation": committed.generation,
        "lineage_sha256": committed.lineage_sha256,
        "parent_pointer_sha256": committed.parent_pointer_sha256,
        "pointer_schema": LATEST_SCHEMA,
        "sequence": committed.sequence,
    }


def publish_committed_generation(
    *,
    root: Path,
    committed: CommittedGeneration,
    verify: Callable[[Path], bool],
    lock_timeout_seconds: float = 10.0,
    _publication_return_hook: Callable[[], None] | None = None,
) -> str:
    """Conditionally select one committed child under a cooperative local lock."""

    if _publication_return_hook is not None and not callable(_publication_return_hook):
        raise SupervisorError("publication return hook is invalid")
    _ordinary_directory(root)
    committed_root = root / "committed"
    _ordinary_directory(committed_root)
    _direct_child(committed_root, committed.target)
    observed = _read_committed_generation(committed.target)
    if not _same_committed_generation(observed, committed):
        raise SupervisorError("committed generation descriptor changed")
    checkpoint = observed.target / observed.logical_checkpoint_id
    if verify(checkpoint) is not True:
        raise SupervisorError(
            "committed generation verifier did not return exact success"
        )
    if (
        _hash_ordinary_file(
            checkpoint / RECEIPT_NAME,
            maximum_bytes=128 * 1024,
        )
        != observed.generation
    ):
        raise SupervisorError("committed receipt digest changed before publication")

    latest = root / "LATEST.json"
    desired = _pointer_for_generation(observed)
    encoded = (canonical_json(desired) + "\n").encode("utf-8")
    desired_sha256 = hashlib.sha256(encoded).hexdigest()
    with _pointer_commit_lock(root, timeout_seconds=lock_timeout_seconds):
        current, parent = _read_pointer(latest)
        if (
            parent.pointer_sha256 == desired_sha256
            and current is not None
            and canonical_json(current) == canonical_json(desired)
        ):
            return "already_published"
        expected = ParentVersion(
            pointer_sha256=observed.parent_pointer_sha256,
            sequence=observed.sequence - 1,
        )
        if parent != expected:
            raise StaleParentError(observed)
        _write_atomic(latest, encoded)
        selected, selected_version = _read_pointer(latest)
        if (
            selected_version.pointer_sha256 != desired_sha256
            or selected is None
            or canonical_json(selected) != canonical_json(desired)
        ):
            raise SupervisorError("published pointer did not remain exact")
    if _publication_return_hook is not None:
        _publication_return_hook()
    return "published"


def promote_candidate(
    *,
    root: Path,
    candidate: Path,
    logical_checkpoint_id: str,
    verify: Callable[[Path], bool],
    lock_timeout_seconds: float = 10.0,
) -> Path:
    """Commit and conditionally select one receipt-bound generation."""

    parent = read_parent_version(root)
    committed = commit_candidate(
        root=root,
        candidate=candidate,
        logical_checkpoint_id=logical_checkpoint_id,
        verify=verify,
        parent=parent,
    )
    publish_committed_generation(
        root=root,
        committed=committed,
        verify=verify,
        lock_timeout_seconds=lock_timeout_seconds,
    )
    return committed.target
