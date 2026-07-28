"""Cross-platform local worker supervision and single-writer promotion."""

from __future__ import annotations

import hashlib
import os
import re
import socket
import stat
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .canonical import canonical_json, strict_json_loads
from .checkpoint_receipt import RECEIPT_NAME

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
LATEST_SCHEMA = "dcp-invariant-latest-v1"
REGISTERED_CHECKPOINT_IDS = {"checkpoint-async", "checkpoint-one", "checkpoint-two"}


class SupervisorError(RuntimeError):
    """A registered worker or promotion invariant failed."""


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
    value = path.lstat()
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
    parent_resolved = parent.resolve(strict=True)
    child_resolved = child.resolve(strict=True)
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


def _validate_existing_pointer(path: Path) -> None:
    if not _entry_exists(path):
        return
    before = _ordinary_file(path, maximum_bytes=1024)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SupervisorError("existing pointer cannot be read") from error
    after = _ordinary_file(path, maximum_bytes=1024)
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
        raise SupervisorError("existing pointer changed during read")
    try:
        text = raw.decode("utf-8")
        parsed = strict_json_loads(text[:-1])
    except (UnicodeError, TypeError, ValueError) as error:
        raise SupervisorError("existing pointer is invalid") from error
    if (
        not text.endswith("\n")
        or "\r" in text
        or canonical_json(parsed) + "\n" != text
        or type(parsed) is not dict
        or set(parsed) != {"generation", "pointer_schema"}
        or parsed["pointer_schema"] != LATEST_SCHEMA
        or type(parsed["generation"]) is not str
        or not _SHA256.fullmatch(parsed["generation"])
    ):
        raise SupervisorError("existing pointer is invalid")


def _write_atomic(path: Path, value: bytes) -> None:
    temporary = path.with_name(f".{path.name}.pending")
    if _entry_exists(temporary):
        raise SupervisorError("pointer staging target must start absent")
    try:
        with temporary.open("xb") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def promote_candidate(
    *,
    root: Path,
    candidate: Path,
    logical_checkpoint_id: str,
    verify: Callable[[Path], bool],
) -> Path:
    """Promote one verified generation, then atomically update LATEST.json."""

    if logical_checkpoint_id not in REGISTERED_CHECKPOINT_IDS:
        raise SupervisorError("logical checkpoint identifier is not registered")
    _ordinary_directory(root)
    candidates = root / "candidates"
    committed = root / "committed"
    _ordinary_directory(candidates)
    _ordinary_directory(committed)
    _ordinary_directory(candidate)
    _direct_child(candidates, candidate)
    try:
        candidate_entries = list(candidate.iterdir())
    except OSError as error:
        raise SupervisorError("candidate wrapper cannot be enumerated") from error
    if [entry.name for entry in candidate_entries] != [logical_checkpoint_id]:
        raise SupervisorError(
            "candidate wrapper must contain only its logical checkpoint"
        )
    checkpoint = candidate / logical_checkpoint_id
    _ordinary_directory(checkpoint)
    _direct_child(candidate, checkpoint)

    latest = root / "LATEST.json"
    _validate_existing_pointer(latest)
    if verify(checkpoint) is not True:
        raise SupervisorError("candidate verifier did not return exact success")
    receipt_sha256 = _hash_ordinary_file(
        checkpoint / RECEIPT_NAME,
        maximum_bytes=128 * 1024,
    )
    target = committed / receipt_sha256
    if _entry_exists(target):
        raise SupervisorError("committed generation already exists")

    pointer = {
        "generation": receipt_sha256,
        "pointer_schema": LATEST_SCHEMA,
    }
    encoded = (canonical_json(pointer) + "\n").encode("utf-8")
    candidate.rename(target)
    _fsync_directory(candidates)
    _fsync_directory(committed)
    try:
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
        _write_atomic(latest, encoded)
    except BaseException:
        # The committed generation may remain as an unselected orphan, but the
        # old pointer is never rewritten before the candidate is verified.
        raise
    return target
