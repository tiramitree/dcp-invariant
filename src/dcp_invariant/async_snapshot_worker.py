"""Two-rank ResNet18 worker for one staged asynchronous DCP snapshot."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import os
import stat
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.nn.parallel import DistributedDataParallel

from .async_snapshot_contract import (
    ASYNC_CHECKPOINT_ID,
    ASYNC_GATE_DIRECTORY_NAME,
    ASYNC_REPORT_DIRECTORY_NAME,
    ASYNC_SNAPSHOT_ACTION,
    ASYNC_SNAPSHOT_REPORT_SCHEMA,
    ASYNC_WORLD_SIZE,
    GLOBAL_INPUT_SHAPE,
    RESNET18_PARAMETER_COUNT,
    is_registered_torchvision_version_pair,
    workload_contract_digest,
)
from .canonical import canonical_json, strict_json_loads
from .checkpoint_receipt import build_receipt, verify_checkpoint, write_receipt
from .state import normalized_digest
from .worker import initialize_process_group

_MARKER_SCHEMA = "dcp-invariant-async-gate-marker-v1"
_MARKER_MAX_BYTES = 4096
_WAIT_SECONDS = 90.0


class AsyncSnapshotWorkerError(RuntimeError):
    """The fixed workload, gate, or checkpoint outcome is invalid."""


def _is_reparse(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & marker)


def _ordinary_directory(path: Path, label: str) -> None:
    try:
        value = path.lstat()
    except OSError as error:
        raise AsyncSnapshotWorkerError(f"{label} cannot be inspected") from error
    if not stat.S_ISDIR(value.st_mode) or path.is_symlink() or _is_reparse(value):
        raise AsyncSnapshotWorkerError(f"{label} is not an ordinary directory")


def _marker_payload(kind: str, rank: int | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": kind,
        "marker_schema": _MARKER_SCHEMA,
    }
    if rank is not None:
        payload["rank"] = rank
    return payload


def _marker_bytes(kind: str, rank: int | None = None) -> bytes:
    return (canonical_json(_marker_payload(kind, rank)) + "\n").encode("utf-8")


def _write_marker(path: Path, *, kind: str, rank: int | None = None) -> None:
    expected = _marker_bytes(kind, rank)
    try:
        with path.open("xb") as output:
            output.write(expected)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise AsyncSnapshotWorkerError(
            f"{kind} marker must start absent and writable"
        ) from error
    if _read_marker(path) != _marker_payload(kind, rank):
        raise AsyncSnapshotWorkerError(f"{kind} marker changed after creation")


def _read_marker(path: Path) -> dict[str, object]:
    try:
        before = path.lstat()
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise AsyncSnapshotWorkerError("gate marker cannot be read") from error
    if (
        not stat.S_ISREG(before.st_mode)
        or path.is_symlink()
        or _is_reparse(before)
        or before.st_nlink != 1
        or before.st_size < 1
        or before.st_size > _MARKER_MAX_BYTES
        or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        or len(raw) != before.st_size
    ):
        raise AsyncSnapshotWorkerError("gate marker identity is invalid")
    try:
        text = raw.decode("utf-8")
        value = strict_json_loads(text[:-1])
    except (UnicodeError, TypeError, ValueError) as error:
        raise AsyncSnapshotWorkerError("gate marker is not strict JSON") from error
    if (
        not text.endswith("\n")
        or "\r" in text
        or type(value) is not dict
        or canonical_json(value) + "\n" != text
    ):
        raise AsyncSnapshotWorkerError("gate marker is not canonical")
    return value


def _wait_markers(
    gate_root: Path,
    *,
    kind: str,
    ranks: tuple[int, ...] = (0, 1),
) -> None:
    deadline = time.monotonic() + _WAIT_SECONDS
    while time.monotonic() < deadline:
        ready = True
        for rank in ranks:
            path = gate_root / f"{kind}-rank-{rank}.json"
            try:
                value = _read_marker(path)
            except AsyncSnapshotWorkerError:
                ready = False
                break
            if value != _marker_payload(kind, rank):
                raise AsyncSnapshotWorkerError(f"{kind} marker payload is invalid")
        if ready:
            return
        time.sleep(0.01)
    raise AsyncSnapshotWorkerError(f"{kind} marker set timed out")


def _wait_release(gate_root: Path) -> None:
    deadline = time.monotonic() + _WAIT_SECONDS
    path = gate_root / "release.json"
    while time.monotonic() < deadline:
        try:
            value = _read_marker(path)
        except AsyncSnapshotWorkerError:
            time.sleep(0.01)
            continue
        if value != _marker_payload("release"):
            raise AsyncSnapshotWorkerError("release marker payload is invalid")
        return
    raise AsyncSnapshotWorkerError("release marker timed out")


class _GateFileSystemWriter(FileSystemWriter):
    """Block the public write-data hook after DCP staging has completed."""

    def __init__(self, checkpoint_id: str, gate_root: Path, rank: int) -> None:
        super().__init__(checkpoint_id, overwrite=False)
        self._gate_root = gate_root
        self._rank = rank
        self.gate_entered = False
        self.gate_released = False
        self.stage_calls = 0
        self.stage_completed = False
        self.staged_model_sha256: str | None = None
        self.staged_optimizer_sha256: str | None = None
        self.staged_state_sha256: str | None = None

    def stage(self, state_dict):
        staged = super().stage(state_dict)
        self.stage_calls += 1
        self.staged_model_sha256 = normalized_digest(staged["model"])
        self.staged_optimizer_sha256 = normalized_digest(staged["optimizer"])
        self.staged_state_sha256 = normalized_digest(staged)
        self.stage_completed = True
        _write_marker(
            self._gate_root / f"staged-rank-{self._rank}.json",
            kind="staged",
            rank=self._rank,
        )
        return staged

    def write_data(self, plan, planner):
        _write_marker(
            self._gate_root / f"entered-rank-{self._rank}.json",
            kind="entered",
            rank=self._rank,
        )
        self.gate_entered = True
        _wait_release(self._gate_root)
        self.gate_released = True
        return super().write_data(plan, planner)


def _versions() -> tuple[str, str, str, str]:
    torch_version = str(torch.__version__)
    try:
        distribution_version = importlib.metadata.version("torchvision")
        pillow_version = importlib.metadata.version("Pillow")
        import PIL
        import torchvision
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise AsyncSnapshotWorkerError(
            "registered vision runtime is unavailable"
        ) from error
    runtime_version = str(torchvision.__version__)
    if not is_registered_torchvision_version_pair(
        distribution_version,
        runtime_version,
    ):
        raise AsyncSnapshotWorkerError(
            "torchvision distribution/runtime pair is not registered"
        )
    if pillow_version != "12.3.0" or str(PIL.__version__) != pillow_version:
        raise AsyncSnapshotWorkerError(
            "Pillow distribution/runtime pair is not registered"
        )
    return torch_version, distribution_version, runtime_version, pillow_version


def _build_model() -> torch.nn.Module:
    from torchvision.models import resnet18

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260729)
        model = resnet18(weights=None)
    if sum(parameter.numel() for parameter in model.parameters()) != (
        RESNET18_PARAMETER_COUNT
    ):
        raise AsyncSnapshotWorkerError("ResNet18 parameter count is invalid")
    return model


def _build_optimizer(parameters) -> torch.optim.SGD:
    return torch.optim.SGD(
        parameters,
        lr=2**-8,
        momentum=0.5,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
        maximize=False,
        foreach=False,
        differentiable=False,
    )


def _global_batch(step: int) -> tuple[torch.Tensor, torch.Tensor]:
    if step != 0:
        raise AsyncSnapshotWorkerError("training step is not registered")
    count = 1
    for size in GLOBAL_INPUT_SHAPE:
        count *= size
    inputs = (
        torch.arange(count, dtype=torch.float32)
        .reshape(GLOBAL_INPUT_SHAPE)
        .remainder(257)
        .div(256)
    )
    targets = torch.tensor((0, 1, 2, 3), dtype=torch.int64)
    return inputs, targets


def _train_step(
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    rank: int,
) -> None:
    inputs, targets = _global_batch(step)
    local_rows = GLOBAL_INPUT_SHAPE[0] // ASYNC_WORLD_SIZE
    start = rank * local_rows
    stop = start + local_rows
    optimizer.zero_grad(set_to_none=True)
    output = model(inputs[start:stop])
    loss = torch.nn.functional.cross_entropy(output, targets[start:stop])
    loss.backward()
    optimizer.step()


def _synchronize_buffers_from_rank_zero(
    model: DistributedDataParallel,
) -> None:
    for buffer in model.module.buffers():
        dist.broadcast(buffer, src=0)


def _assert_rank_consensus(value: str, *, label: str) -> None:
    values: list[str | None] = [None] * ASYNC_WORLD_SIZE
    dist.all_gather_object(values, value)
    if values != [value] * ASYNC_WORLD_SIZE:
        raise AsyncSnapshotWorkerError(f"{label} differs across ranks")


def _mutate_after_staging(model: DistributedDataParallel) -> None:
    with torch.no_grad():
        model.module.conv1.weight.add_(2**-10)


def _write_async_report(
    report_dir: Path,
    *,
    rank: int,
    report: dict[str, Any],
) -> Path:
    if (
        report.get("rank") != rank
        or report.get("world_size") != ASYNC_WORLD_SIZE
        or report.get("report_schema") != ASYNC_SNAPSHOT_REPORT_SCHEMA
    ):
        raise AsyncSnapshotWorkerError("async report identity is invalid")
    target = report_dir / f"rank-{rank}.json"
    pending = report_dir / f".rank-{rank}.json.pending"
    if (
        target.exists()
        or target.is_symlink()
        or pending.exists()
        or pending.is_symlink()
    ):
        raise AsyncSnapshotWorkerError("async report target must start absent")
    encoded = (canonical_json(report) + "\n").encode("utf-8")
    if len(encoded) > 64 * 1024:
        raise AsyncSnapshotWorkerError("async report exceeds its size bound")
    try:
        with pending.open("xb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(pending, target)
    finally:
        if pending.exists():
            pending.unlink()
    return target


def _dcp_state(
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    cursor: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model_state, optimizer_state = get_state_dict(model, optimizer)
    return (
        {
            "application": {
                "cursor": torch.tensor([cursor], dtype=torch.int64),
            },
            "model": model_state,
            "optimizer": optimizer_state,
        },
        model_state,
        optimizer_state,
    )


def _state_digests(
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    cursor: int,
) -> dict[str, str]:
    state, model_state, optimizer_state = _dcp_state(model, optimizer, cursor)
    return {
        "cursor_sha256": normalized_digest(cursor),
        "model_sha256": normalized_digest(model_state),
        "optimizer_sha256": normalized_digest(optimizer_state),
        "state_sha256": normalized_digest(state),
    }


def _receipt_file_sha256(receipt: dict[str, Any]) -> str:
    encoded = (canonical_json(receipt) + "\n").encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _seal_checkpoint(
    checkpoint_id: str,
    *,
    rank: int,
    torch_version: str,
) -> tuple[dict[str, Any], str]:
    dist.barrier()
    if rank == 0:
        receipt = build_receipt(
            Path(checkpoint_id),
            logical_checkpoint_id=checkpoint_id,
            torch_version=torch_version,
            state_contract_sha256=workload_contract_digest(),
        )
        write_receipt(Path(checkpoint_id), receipt)
    dist.barrier()
    local_ok = True
    try:
        receipt = verify_checkpoint(Path(checkpoint_id))
        local_ok = (
            receipt["logical_checkpoint_id"] == checkpoint_id
            and receipt["torch_version"] == torch_version
            and receipt["state_contract_sha256"] == workload_contract_digest()
        )
    except (OSError, KeyError, TypeError, ValueError):
        local_ok = False
        receipt = {}
    flag = torch.tensor([1 if local_ok else 0], dtype=torch.int64)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not bool(flag.item()) or type(receipt) is not dict:
        raise AsyncSnapshotWorkerError("async checkpoint receipt is invalid")
    return receipt, _receipt_file_sha256(receipt)


def _load_digests(
    checkpoint_id: str,
    *,
    rank: int,
) -> tuple[dict[str, str], str, str]:
    model = DistributedDataParallel(_build_model())
    optimizer = _build_optimizer(model.parameters())
    load_state, model_state, optimizer_state = _dcp_state(model, optimizer, 0)
    load_target_before_sha256 = normalized_digest(model_state)
    dcp.load(load_state, checkpoint_id=checkpoint_id)
    direct_loaded_model_sha256 = normalized_digest(model_state)
    set_state_dict(
        model,
        optimizer,
        model_state_dict=model_state,
        optim_state_dict=optimizer_state,
    )
    cursor_value = load_state["application"]["cursor"]
    if (
        not isinstance(cursor_value, torch.Tensor)
        or cursor_value.shape != (1,)
        or cursor_value.dtype is not torch.int64
        or int(cursor_value.item()) != 1
    ):
        raise AsyncSnapshotWorkerError("loaded cursor is outside the contract")
    digests = _state_digests(model, optimizer, 1)
    dist.barrier()
    del rank
    return digests, direct_loaded_model_sha256, load_target_before_sha256


def execute_async_snapshot(
    *,
    checkpoint_id: str,
    report_dir: Path,
    gate_dir: Path,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    if checkpoint_id != ASYNC_CHECKPOINT_ID:
        raise AsyncSnapshotWorkerError("checkpoint identifier is invalid")
    if report_dir.name != ASYNC_REPORT_DIRECTORY_NAME:
        raise AsyncSnapshotWorkerError("report directory name is invalid")
    if gate_dir.name != ASYNC_GATE_DIRECTORY_NAME:
        raise AsyncSnapshotWorkerError("gate directory name is invalid")
    if world_size != ASYNC_WORLD_SIZE or rank not in {0, 1}:
        raise AsyncSnapshotWorkerError("rank coordinates are invalid")
    _ordinary_directory(report_dir, "async report directory")
    _ordinary_directory(gate_dir, "async gate directory")
    (
        torch_version,
        distribution_version,
        runtime_version,
        pillow_version,
    ) = _versions()
    if not dist.is_initialized():
        raise AsyncSnapshotWorkerError("process group is not initialized")
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise AsyncSnapshotWorkerError("process group coordinates changed")

    torch.set_num_threads(1)
    model = DistributedDataParallel(_build_model())
    optimizer = _build_optimizer(model.parameters())
    _train_step(model, optimizer, step=0, rank=rank)
    _synchronize_buffers_from_rank_zero(model)
    cursor = 1
    save_state, _, _ = _dcp_state(model, optimizer, cursor)
    pre = _state_digests(model, optimizer, cursor)
    _assert_rank_consensus(pre["state_sha256"], label="pre-snapshot state")

    writer = _GateFileSystemWriter(checkpoint_id, gate_dir, rank)
    future = dcp.async_save(
        save_state,
        storage_writer=writer,
        async_stager=writer,
    )
    release_path = gate_dir / "release.json"
    try:
        _wait_markers(gate_dir, kind="staged")
        _wait_markers(gate_dir, kind="entered")
        future_pending_at_mutation = not future.done()
        _mutate_after_staging(model)
        post = _state_digests(model, optimizer, cursor)
        _write_marker(
            gate_dir / f"mutated-rank-{rank}.json",
            kind="mutated",
            rank=rank,
        )
        if rank == 0:
            _wait_markers(gate_dir, kind="mutated")
            _write_marker(release_path, kind="release")
        future.result(timeout=_WAIT_SECONDS)
    finally:
        if rank == 0 and not release_path.exists():
            _write_marker(release_path, kind="release")
        if not future.done():
            future.result(timeout=_WAIT_SECONDS)
        writer.close()
    if not writer.gate_entered or not writer.gate_released:
        raise AsyncSnapshotWorkerError("writer gate did not complete")

    receipt, receipt_sha256 = _seal_checkpoint(
        checkpoint_id,
        rank=rank,
        torch_version=torch_version,
    )
    (
        loaded,
        direct_loaded_model_sha256,
        load_target_before_sha256,
    ) = _load_digests(checkpoint_id, rank=rank)
    loaded_equals_pre = loaded == pre
    post_load_receipt = verify_checkpoint(Path(checkpoint_id))
    if _receipt_file_sha256(post_load_receipt) != receipt_sha256:
        raise AsyncSnapshotWorkerError("checkpoint receipt changed across the load")
    loaded_equals_post = loaded == post
    post_differs_from_pre = post != pre
    if (
        not future_pending_at_mutation
        or not writer.stage_completed
        or writer.stage_calls != 1
        or writer.staged_model_sha256 != pre["model_sha256"]
        or writer.staged_optimizer_sha256 != pre["optimizer_sha256"]
        or writer.staged_state_sha256 != pre["state_sha256"]
        or not loaded_equals_pre
        or loaded_equals_post
        or not post_differs_from_pre
        or post["cursor_sha256"] != pre["cursor_sha256"]
        or post["optimizer_sha256"] != pre["optimizer_sha256"]
        or post["model_sha256"] == pre["model_sha256"]
        or direct_loaded_model_sha256 != pre["model_sha256"]
        or load_target_before_sha256 == pre["model_sha256"]
    ):
        raise AsyncSnapshotWorkerError(
            "async snapshot state relation is invalid"
            f" (loaded_equals_pre={loaded_equals_pre},"
            f" loaded_equals_post={loaded_equals_post},"
            f" post_differs_from_pre={post_differs_from_pre})"
            f" cursor={loaded['cursor_sha256'] == pre['cursor_sha256']},"
            f" model={loaded['model_sha256'] == pre['model_sha256']},"
            f" optimizer="
            f"{loaded['optimizer_sha256'] == pre['optimizer_sha256']},"
            f" aggregate="
            f"{loaded['state_sha256'] == pre['state_sha256']}"
            f" staged_model="
            f"{writer.staged_model_sha256 == pre['model_sha256']},"
            f" staged_optimizer="
            f"{writer.staged_optimizer_sha256 == pre['optimizer_sha256']},"
            f" staged_state="
            f"{writer.staged_state_sha256 == pre['state_sha256']},"
            f" direct_loaded_model="
            f"{direct_loaded_model_sha256 == pre['model_sha256']},"
            f" load_target_differs="
            f"{load_target_before_sha256 != pre['model_sha256']}"
        )
    report = {
        "action": ASYNC_SNAPSHOT_ACTION,
        "async_checkpointer": "thread",
        "direct_loaded_model_sha256": direct_loaded_model_sha256,
        "future_pending_at_mutation": future_pending_at_mutation,
        "loaded_equals_post": loaded_equals_post,
        "loaded_equals_pre": loaded_equals_pre,
        "loaded_cursor_sha256": loaded["cursor_sha256"],
        "loaded_model_sha256": loaded["model_sha256"],
        "loaded_optimizer_sha256": loaded["optimizer_sha256"],
        "loaded_state_sha256": loaded["state_sha256"],
        "load_target_before_model_sha256": load_target_before_sha256,
        "pillow_version": pillow_version,
        "post_cursor_sha256": post["cursor_sha256"],
        "post_differs_from_pre": post_differs_from_pre,
        "post_model_sha256": post["model_sha256"],
        "post_optimizer_sha256": post["optimizer_sha256"],
        "post_state_sha256": post["state_sha256"],
        "pre_cursor_sha256": pre["cursor_sha256"],
        "pre_model_sha256": pre["model_sha256"],
        "pre_optimizer_sha256": pre["optimizer_sha256"],
        "pre_state_sha256": pre["state_sha256"],
        "rank": rank,
        "receipt_sha256": receipt_sha256,
        "receipt_verified_after_load": True,
        "receipt_verified_after_save": True,
        "report_schema": ASYNC_SNAPSHOT_REPORT_SCHEMA,
        "stage_completed_before_mutation": writer.stage_completed,
        "stage_call_count": writer.stage_calls,
        "staged_model_sha256": writer.staged_model_sha256,
        "staged_optimizer_sha256": writer.staged_optimizer_sha256,
        "staged_state_sha256": writer.staged_state_sha256,
        "torch_version": torch_version,
        "torchvision_distribution_version": distribution_version,
        "torchvision_runtime_version": runtime_version,
        "weights_downloaded": False,
        "workload_contract_sha256": workload_contract_digest(),
        "world_size": world_size,
        "writer_gate_entered": writer.gate_entered,
        "writer_gate_released": writer.gate_released,
    }
    _write_async_report(report_dir, rank=rank, report=report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dcp_invariant.async_snapshot_worker"
    )
    parser.add_argument(
        "--checkpoint-id",
        required=True,
        choices=[ASYNC_CHECKPOINT_ID],
    )
    parser.add_argument("--gate-dir", required=True, type=Path)
    parser.add_argument("--master-port", required=True, type=int)
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--world-size", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    initialize_process_group(
        rank=arguments.rank,
        world_size=arguments.world_size,
        master_port=arguments.master_port,
    )
    try:
        execute_async_snapshot(
            checkpoint_id=arguments.checkpoint_id,
            report_dir=arguments.report_dir,
            gate_dir=arguments.gate_dir,
            rank=arguments.rank,
            world_size=arguments.world_size,
        )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
