"""Private deterministic workers for the registered lineage scenario."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import time
from pathlib import Path
from typing import Any

from .canonical import canonical_json
from .checkpoint_receipt import RECEIPT_NAME
from .supervisor import (
    StaleParentError,
    _pointer_for_generation,
    _write_atomic,
    commit_candidate,
    publish_committed_generation,
    read_parent_version,
)

REPORT_SCHEMA = "dcp-invariant-lineage-worker-report-v1"
FIXTURE_SCHEMA = "dcp-invariant-lineage-fixture-v1"
CHECKPOINT_ID = "checkpoint-one"
CRASH_AFTER_COMMIT_EXIT = 73
CRASH_AFTER_PUBLISH_EXIT = 74


class LineageWorkerError(RuntimeError):
    """The private deterministic lineage worker contract failed."""


def _entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _ordinary_directory(path: Path, label: str) -> None:
    try:
        value = path.lstat()
    except OSError as error:
        raise LineageWorkerError(f"{label} cannot be inspected") from error
    if not stat.S_ISDIR(value.st_mode) or path.is_symlink():
        raise LineageWorkerError(f"{label} is not an ordinary directory")


def _write_record(path: Path, value: dict[str, Any]) -> None:
    raw = (canonical_json(value) + "\n").encode("utf-8")
    try:
        with path.open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise LineageWorkerError(
            "private coordination record cannot be created"
        ) from error


def _wait_for(paths: tuple[Path, ...], timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not all(_entry_exists(path) for path in paths):
        if time.monotonic() >= deadline:
            raise LineageWorkerError("private coordination barrier timed out")
        time.sleep(0.01)


def _tree_digest(root: Path) -> str:
    _ordinary_directory(root, "committed generation")
    records: list[dict[str, str]] = []
    try:
        entries = sorted(root.rglob("*"), key=lambda value: value.as_posix())
    except OSError as error:
        raise LineageWorkerError("committed generation cannot be enumerated") from error
    for entry in entries:
        try:
            value = entry.lstat()
        except OSError as error:
            raise LineageWorkerError("committed entry cannot be inspected") from error
        relative = entry.relative_to(root).as_posix()
        if entry.is_symlink():
            raise LineageWorkerError("committed generation contains a link")
        if stat.S_ISDIR(value.st_mode):
            records.append({"kind": "directory", "name": relative})
        elif stat.S_ISREG(value.st_mode):
            try:
                digest = hashlib.sha256(entry.read_bytes()).hexdigest()
            except OSError as error:
                raise LineageWorkerError(
                    "committed generation file cannot be read"
                ) from error
            records.append(
                {
                    "kind": "file",
                    "name": relative,
                    "sha256": digest,
                }
            )
        else:
            raise LineageWorkerError("committed generation contains a special entry")
    return hashlib.sha256(canonical_json(records).encode("utf-8")).hexdigest()


def _make_candidate(root: Path, rank: int) -> tuple[Path, bytes]:
    wrapper = root / "candidates" / f"candidate-{rank}"
    checkpoint = wrapper / CHECKPOINT_ID
    receipt = (
        canonical_json(
            {
                "fixture_schema": FIXTURE_SCHEMA,
                "ordinal": rank,
            }
        )
        + "\n"
    ).encode("utf-8")
    try:
        checkpoint.mkdir(parents=True)
        with (checkpoint / RECEIPT_NAME).open("xb") as output:
            output.write(receipt)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise LineageWorkerError("private candidate cannot be created") from error
    return wrapper, receipt


def _verifier(receipt: bytes):
    def verify(checkpoint: Path) -> bool:
        try:
            return (
                checkpoint.name == CHECKPOINT_ID
                and (checkpoint / RECEIPT_NAME).read_bytes() == receipt
            )
        except OSError:
            return False

    return verify


def _commit(
    *,
    root: Path,
    rank: int,
    coordination: Path,
    barrier_timeout_seconds: float,
    use_barriers: bool,
):
    parent = read_parent_version(root)
    wrapper, receipt = _make_candidate(root, rank)
    verifier = _verifier(receipt)
    if use_barriers:
        _write_record(
            coordination / f"parent-{rank}.json",
            {
                "parent_pointer_sha256": parent.pointer_sha256,
                "sequence": parent.sequence,
            },
        )
        _wait_for(
            (
                coordination / "parent-0.json",
                coordination / "parent-1.json",
            ),
            barrier_timeout_seconds,
        )
    committed = commit_candidate(
        root=root,
        candidate=wrapper,
        logical_checkpoint_id=CHECKPOINT_ID,
        verify=verifier,
        parent=parent,
    )
    tree_sha256 = _tree_digest(committed.target)
    if use_barriers:
        _write_record(
            coordination / f"committed-{rank}.json",
            {
                "generation": committed.generation,
                "generation_tree_sha256": tree_sha256,
                "lineage_sha256": committed.lineage_sha256,
                "parent_pointer_sha256": committed.parent_pointer_sha256,
                "sequence": committed.sequence,
            },
        )
        _wait_for(
            (
                coordination / "committed-0.json",
                coordination / "committed-1.json",
            ),
            barrier_timeout_seconds,
        )
    return committed, verifier, tree_sha256


def _unfenced_reference_publish(root: Path, committed) -> str:
    pointer = _pointer_for_generation(committed)
    _write_atomic(
        root / "LATEST.json",
        (canonical_json(pointer) + "\n").encode("utf-8"),
    )
    return "published_unfenced"


def _race_mode(
    *,
    mode: str,
    root: Path,
    coordination: Path,
    reports: Path,
    rank: int,
    barrier_timeout_seconds: float,
) -> int:
    committed, verifier, tree_sha256 = _commit(
        root=root,
        rank=rank,
        coordination=coordination,
        barrier_timeout_seconds=barrier_timeout_seconds,
        use_barriers=True,
    )
    if rank == 1:
        _wait_for((coordination / "published-0.json",), barrier_timeout_seconds)
    if mode == "control":
        outcome = _unfenced_reference_publish(root, committed)
    else:
        try:
            outcome = publish_committed_generation(
                root=root,
                committed=committed,
                verify=verifier,
            )
        except StaleParentError:
            outcome = "stale_parent"
    selected_after_action = read_parent_version(root)
    _write_record(
        reports / f"rank-{rank}.json",
        {
            "commit_barrier_passed": True,
            "generation": committed.generation,
            "generation_tree_sha256_before": tree_sha256,
            "lineage_sha256": committed.lineage_sha256,
            "mode": mode,
            "outcome": outcome,
            "parent_pointer_sha256": committed.parent_pointer_sha256,
            "publish_ordinal": rank,
            "pointer_sequence_after_action": selected_after_action.sequence,
            "pointer_sha256_after_action": selected_after_action.pointer_sha256,
            "rank": rank,
            "report_schema": REPORT_SCHEMA,
            "sequence": committed.sequence,
        },
    )
    if rank == 0:
        _write_record(coordination / "published-0.json", {"outcome": outcome})
    return 0


def _crash_mode(
    *,
    mode: str,
    root: Path,
    coordination: Path,
) -> int:
    committed, verifier, tree_sha256 = _commit(
        root=root,
        rank=0,
        coordination=coordination,
        barrier_timeout_seconds=1.0,
        use_barriers=False,
    )
    if mode == "crash-after-commit":
        _write_record(
            coordination / "recovery.json",
            {
                "generation": committed.generation,
                "generation_tree_sha256_before": tree_sha256,
                "outcome_before_exit": "committed",
            },
        )
        os._exit(CRASH_AFTER_COMMIT_EXIT)

    _write_record(
        coordination / "recovery.json",
        {
            "generation": committed.generation,
            "generation_tree_sha256_before": tree_sha256,
            "outcome_before_exit": "publication_return_lost",
        },
    )

    def terminate_before_return() -> None:
        os._exit(CRASH_AFTER_PUBLISH_EXIT)

    publish_committed_generation(
        root=root,
        committed=committed,
        verify=verifier,
        _publication_return_hook=terminate_before_return,
    )
    raise LineageWorkerError("publication return-loss hook did not terminate")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a private lineage worker.")
    parser.add_argument(
        "--mode",
        choices=(
            "control",
            "protected",
            "crash-after-commit",
            "crash-after-publish",
        ),
        required=True,
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--coordination", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--barrier-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--master-port", type=int, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    _ordinary_directory(arguments.root, "lineage root")
    _ordinary_directory(arguments.coordination, "coordination directory")
    _ordinary_directory(arguments.report_dir, "report directory")
    if not (0.1 <= arguments.barrier_timeout_seconds <= 60.0):
        raise LineageWorkerError("barrier timeout is outside its bound")
    if not (1 <= arguments.master_port <= 65535):
        raise LineageWorkerError("private launch port is invalid")
    if arguments.mode in {"control", "protected"}:
        if arguments.world_size != 2 or arguments.rank not in {0, 1}:
            raise LineageWorkerError("race mode requires exactly two ranks")
        return _race_mode(
            mode=arguments.mode,
            root=arguments.root,
            coordination=arguments.coordination,
            reports=arguments.report_dir,
            rank=arguments.rank,
            barrier_timeout_seconds=arguments.barrier_timeout_seconds,
        )
    if arguments.world_size != 1 or arguments.rank != 0:
        raise LineageWorkerError("crash mode requires exactly one rank")
    return _crash_mode(
        mode=arguments.mode,
        root=arguments.root,
        coordination=arguments.coordination,
    )


if __name__ == "__main__":
    raise SystemExit(main())
