"""Bounded launcher for one real single-host PyTorch elastic restart."""

from __future__ import annotations

import contextlib
import os
import signal
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .elastic_contract import (
    BOOTSTRAP_ATTESTATION_ENV,
    BOOTSTRAP_ATTESTATION_NAME,
    BOOTSTRAP_SHARED_STORE_ENV,
    CONTROL_REPORT_DIRECTORY_NAME,
    FAILURE_MARKER_NAME,
    LOAD_REPORT_DIRECTORY_NAME,
)
from .supervisor import minimal_worker_environment


class ElasticSupervisorError(RuntimeError):
    """The fixed elastic launcher failed its process or cleanup contract."""


@dataclass(frozen=True)
class ElasticResult:
    exit_code: int
    timed_out: bool
    tree_cleanup: str


class ElasticOutcomeError(ElasticSupervisorError):
    """The fixed elastic launch did not complete successfully."""

    def __init__(self, result: ElasticResult) -> None:
        self.result = result
        super().__init__(
            "registered elastic outcome failed"
            f" (timed_out={result.timed_out}, exit_code={result.exit_code})"
        )


def _is_reparse(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & marker)


def _ordinary_directory(path: Path, label: str) -> None:
    try:
        value = path.lstat()
    except OSError as error:
        raise ElasticSupervisorError(f"{label} cannot be inspected") from error
    if not stat.S_ISDIR(value.st_mode) or path.is_symlink() or _is_reparse(value):
        raise ElasticSupervisorError(f"{label} is not an ordinary directory")


def _ordinary_file(path: Path, label: str) -> None:
    try:
        value = path.lstat()
    except OSError as error:
        raise ElasticSupervisorError(f"{label} cannot be inspected") from error
    if not stat.S_ISREG(value.st_mode) or path.is_symlink() or _is_reparse(value):
        raise ElasticSupervisorError(f"{label} is not an ordinary file")
    if value.st_size < 1 or value.st_size > 64 * 1024:
        raise ElasticSupervisorError(f"{label} size is invalid")


def _torchrun_environment(
    isolated_home: Path,
    bootstrap_attestation: Path,
) -> dict[str, str]:
    if bootstrap_attestation.name != BOOTSTRAP_ATTESTATION_NAME:
        raise ElasticSupervisorError("torchrun bootstrap attestation name is invalid")
    bootstrap_directory = Path(__file__).with_name("_torchrun_bootstrap")
    _ordinary_directory(bootstrap_directory, "torchrun bootstrap directory")
    _ordinary_file(
        bootstrap_directory / "sitecustomize.py",
        "torchrun sitecustomize bootstrap",
    )
    environment = minimal_worker_environment(isolated_home)
    if "PYTHONPATH" in environment:
        raise ElasticSupervisorError("untrusted Python path survived environment reset")
    environment["PYTHONPATH"] = str(bootstrap_directory.resolve(strict=True))
    environment[BOOTSTRAP_SHARED_STORE_ENV] = "1"
    environment[BOOTSTRAP_ATTESTATION_ENV] = str(
        bootstrap_attestation.parent.resolve(strict=True) / bootstrap_attestation.name
    )
    return environment


def elastic_command(
    *,
    load_report_dir: Path,
    control_report_dir: Path,
    failure_marker: Path,
) -> tuple[str, ...]:
    """Return the exact shell-free command for the registered elastic launch."""

    if load_report_dir.name != LOAD_REPORT_DIRECTORY_NAME:
        raise ElasticSupervisorError("elastic load report directory is invalid")
    if control_report_dir.name != CONTROL_REPORT_DIRECTORY_NAME:
        raise ElasticSupervisorError("elastic control report directory is invalid")
    if failure_marker.name != FAILURE_MARKER_NAME:
        raise ElasticSupervisorError("elastic failure marker is invalid")
    return (
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--local-addr=127.0.0.1",
        "--nnodes=1",
        "--nproc-per-node=2",
        "--max-restarts=1",
        "--monitor-interval=0.1",
        "--module",
        "dcp_invariant.elastic_worker",
        "--checkpoint-id",
        "checkpoint-one",
        "--load-report-dir",
        str(load_report_dir),
        "--control-report-dir",
        str(control_report_dir),
        "--failure-marker",
        str(failure_marker),
    )


def _terminate_process_tree(process: subprocess.Popen[bytes]) -> str:
    """Best-effort bounded tree termination; only normal exits are published."""

    if process.poll() is not None:
        return "normal-agent-exit"
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        return "posix-process-group-kill"
    if os.name == "nt":
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            completed = subprocess.run(
                [
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
                creationflags=flags,
            )
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            return "windows-agent-kill-fallback"
        if completed.returncode != 0 and process.poll() is None:
            process.kill()
            return "windows-agent-kill-fallback"
        return "windows-taskkill-tree"
    process.kill()
    return "agent-kill-unknown-platform"


def run_elastic_workers(
    *,
    cwd: Path,
    isolated_home: Path,
    load_report_dir: Path,
    control_report_dir: Path,
    failure_marker: Path,
    timeout_seconds: float,
) -> ElasticResult:
    """Run one exact elastic job and reject any nonzero or timed-out outcome."""

    if not (1.0 <= timeout_seconds <= 300.0):
        raise ElasticSupervisorError("elastic timeout is outside the registered bound")
    for path, label in (
        (cwd, "elastic working directory"),
        (isolated_home, "elastic isolated home"),
        (load_report_dir, "elastic load report directory"),
        (control_report_dir, "elastic control report directory"),
        (failure_marker.parent, "elastic marker parent"),
    ):
        _ordinary_directory(path, label)
    if failure_marker.exists() or failure_marker.is_symlink():
        raise ElasticSupervisorError("elastic failure marker must start absent")
    bootstrap_attestation = failure_marker.parent / BOOTSTRAP_ATTESTATION_NAME
    if bootstrap_attestation.exists() or bootstrap_attestation.is_symlink():
        raise ElasticSupervisorError("torchrun bootstrap attestation must start absent")

    command = elastic_command(
        load_report_dir=load_report_dir,
        control_report_dir=control_report_dir,
        failure_marker=failure_marker,
    )
    creationflags = (
        getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    )
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=_torchrun_environment(isolated_home, bootstrap_attestation),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        start_new_session=os.name != "nt",
    )
    timed_out = False
    tree_cleanup = "normal-agent-exit"
    try:
        try:
            exit_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            tree_cleanup = _terminate_process_tree(process)
            exit_code = process.wait(timeout=15)
    finally:
        if process.poll() is None:
            tree_cleanup = _terminate_process_tree(process)
            process.wait(timeout=15)

    result = ElasticResult(
        exit_code=exit_code,
        timed_out=timed_out,
        tree_cleanup=tree_cleanup,
    )
    if result.timed_out or result.exit_code != 0:
        raise ElasticOutcomeError(result)
    _ordinary_file(
        bootstrap_attestation,
        "torchrun bootstrap attestation",
    )
    return result
