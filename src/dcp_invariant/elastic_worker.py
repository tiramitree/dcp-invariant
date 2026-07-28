"""Fixed two-rank worker for one bounded PyTorch elastic restart scenario."""

from __future__ import annotations

import argparse
import hashlib
import os
import stat
import time
from collections.abc import Mapping
from pathlib import Path

from .canonical import canonical_json, strict_json_loads
from .elastic_contract import (
    CONTROL_REPORT_DIRECTORY_NAME,
    ELASTIC_REPORT_SCHEMA,
    FAILURE_MARKER_NAME,
    LOAD_REPORT_DIRECTORY_NAME,
    REGISTERED_FAILURE_EXIT_CODE,
    ElasticContractError,
    ElasticEnvironment,
    failure_marker_payload,
    parse_elastic_environment,
)

_MARKER_MAX_BYTES = 4096


class ElasticWorkerContractError(ElasticContractError):
    """The launcher environment or private elastic evidence is invalid."""


def _is_reparse(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & marker)


def _ordinary_directory(path: Path, label: str) -> None:
    try:
        value = path.lstat()
    except OSError as error:
        raise ElasticWorkerContractError(f"{label} cannot be inspected") from error
    if not stat.S_ISDIR(value.st_mode) or path.is_symlink() or _is_reparse(value):
        raise ElasticWorkerContractError(f"{label} is not an ordinary directory")


def _ordinary_file(path: Path, label: str) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as error:
        raise ElasticWorkerContractError(f"{label} cannot be inspected") from error
    if not stat.S_ISREG(value.st_mode) or path.is_symlink() or _is_reparse(value):
        raise ElasticWorkerContractError(f"{label} is not an ordinary file")
    if value.st_size < 1 or value.st_size > _MARKER_MAX_BYTES:
        raise ElasticWorkerContractError(f"{label} size is invalid")
    return value


def _marker_bytes() -> bytes:
    return (canonical_json(failure_marker_payload()) + "\n").encode("utf-8")


def write_failure_marker(path: Path) -> str:
    if path.name != FAILURE_MARKER_NAME:
        raise ElasticWorkerContractError("failure marker name is not registered")
    _ordinary_directory(path.parent, "failure marker parent")
    raw = _marker_bytes()
    try:
        with path.open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise ElasticWorkerContractError(
            "failure marker must start absent and writable"
        ) from error
    observed, digest = read_failure_marker(path)
    if observed != failure_marker_payload():
        raise ElasticWorkerContractError("failure marker changed after creation")
    return digest


def read_failure_marker(path: Path) -> tuple[dict[str, object], str]:
    if path.name != FAILURE_MARKER_NAME:
        raise ElasticWorkerContractError("failure marker name is not registered")
    before = _ordinary_file(path, "failure marker")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise ElasticWorkerContractError("failure marker cannot be read") from error
    after = _ordinary_file(path, "failure marker")
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
        raise ElasticWorkerContractError("failure marker changed during read")
    try:
        text = raw.decode("utf-8")
        value = strict_json_loads(text[:-1])
    except (UnicodeError, TypeError, ValueError) as error:
        raise ElasticWorkerContractError("failure marker is not strict JSON") from error
    expected = failure_marker_payload()
    if (
        not text.endswith("\n")
        or "\r" in text
        or type(value) is not dict
        or value != expected
        or canonical_json(value) + "\n" != text
    ):
        raise ElasticWorkerContractError("failure marker is not canonical")
    return value, hashlib.sha256(raw).hexdigest()


def _wait_for_failure_marker(path: Path, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            read_failure_marker(path)
        except ElasticWorkerContractError:
            time.sleep(0.01)
            continue
        return
    raise ElasticWorkerContractError("rank zero did not observe the failure marker")


def _write_control_report(
    root: Path,
    *,
    environment: ElasticEnvironment,
    failure_marker_sha256: str,
) -> Path:
    if root.name != CONTROL_REPORT_DIRECTORY_NAME:
        raise ElasticWorkerContractError("elastic control report directory is invalid")
    _ordinary_directory(root, "elastic control report directory")
    report = {
        "elastic_report_schema": ELASTIC_REPORT_SCHEMA,
        "failure_marker_sha256": failure_marker_sha256,
        "loopback_rendezvous": True,
        "max_restarts": environment.max_restarts,
        "rank": environment.rank,
        "restart_count": environment.restart_count,
        "shared_rendezvous_tcpstore_disabled": True,
        "world_size": environment.world_size,
    }
    target = root / f"rank-{environment.rank}.json"
    pending = root / f".rank-{environment.rank}.json.pending"
    if (
        target.exists()
        or target.is_symlink()
        or pending.exists()
        or pending.is_symlink()
    ):
        raise ElasticWorkerContractError(
            "elastic control report target must start absent"
        )
    raw = (canonical_json(report) + "\n").encode("utf-8")
    try:
        with pending.open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        os.replace(pending, target)
    finally:
        if pending.exists():
            pending.unlink()
    return target


def execute_elastic_worker(
    *,
    checkpoint_id: str,
    load_report_dir: Path,
    control_report_dir: Path,
    failure_marker: Path,
    environment: Mapping[str, str],
) -> int:
    if checkpoint_id != "checkpoint-one":
        raise ElasticWorkerContractError("elastic checkpoint is not registered")
    if load_report_dir.name != LOAD_REPORT_DIRECTORY_NAME:
        raise ElasticWorkerContractError("elastic load report directory is invalid")
    coordinates = parse_elastic_environment(environment)

    if coordinates.restart_count == 0:
        if coordinates.rank == 1:
            write_failure_marker(failure_marker)
            return REGISTERED_FAILURE_EXIT_CODE
        _wait_for_failure_marker(failure_marker)
        return 0

    _, marker_sha256 = read_failure_marker(failure_marker)
    from datetime import timedelta

    import torch
    import torch.distributed as dist

    from .worker import (
        WORKER_REPORT_SCHEMA,
        run_registered_action,
        write_report,
    )

    if dist.is_initialized():
        raise ElasticWorkerContractError("process group must start uninitialized")
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
        timeout=timedelta(seconds=60),
    )
    try:
        if (
            dist.get_rank() != coordinates.rank
            or dist.get_world_size() != coordinates.world_size
        ):
            raise ElasticWorkerContractError(
                "elastic process group coordinates changed"
            )
        report = run_registered_action(
            action="training-load-next",
            checkpoint_id=checkpoint_id,
            rank=coordinates.rank,
            world_size=coordinates.world_size,
        )
        if report.get("report_schema") != WORKER_REPORT_SCHEMA:
            raise ElasticWorkerContractError("elastic load report schema is invalid")
        write_report(load_report_dir, rank=coordinates.rank, report=report)
        _write_control_report(
            control_report_dir,
            environment=coordinates,
            failure_marker_sha256=marker_sha256,
        )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dcp_invariant.elastic_worker")
    parser.add_argument(
        "--checkpoint-id",
        required=True,
        choices=["checkpoint-one"],
    )
    parser.add_argument("--load-report-dir", required=True, type=Path)
    parser.add_argument("--control-report-dir", required=True, type=Path)
    parser.add_argument("--failure-marker", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    return execute_elastic_worker(
        checkpoint_id=arguments.checkpoint_id,
        load_report_dir=arguments.load_report_dir,
        control_report_dir=arguments.control_report_dir,
        failure_marker=arguments.failure_marker,
        environment=os.environ,
    )


if __name__ == "__main__":
    raise SystemExit(main())
