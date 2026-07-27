"""Exact PyTorch 2.11 CPU/Gloo worker for registered restart scenarios."""

from __future__ import annotations

import argparse
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from torch.nn.parallel import DistributedDataParallel

from .canonical import canonical_json
from .checkpoint_receipt import (
    CheckpointReceiptError,
    build_receipt,
    verify_checkpoint,
    write_receipt,
)
from .state import (
    BIAS_SHAPE,
    GLOBAL_BATCH_SHAPE,
    GLOBAL_TARGET_SHAPE,
    MODEL_SHAPE,
    build_generator,
    build_model,
    build_optimizer,
    canonical_global_batches,
    normalized_digest,
    state_contract_digest,
    state_digests,
)

WORKER_REPORT_SCHEMA = "dcp-invariant-worker-report-v1"
REGISTERED_ACTIONS = {
    "training-save-baseline",
    "training-load-next",
    "dtensor-save",
    "dtensor-load",
}
REGISTERED_CHECKPOINT_IDS = {"checkpoint-one", "checkpoint-two"}
REGISTERED_WORLD_SIZES = {1, 2}
DTENSOR_GLOBAL_SHAPE = (4, 4)


class WorkerContractError(ValueError):
    """Worker arguments or state are outside the registered scenario."""


def validate_rank_world_size(rank: int, world_size: int) -> None:
    if type(world_size) is not int or world_size not in REGISTERED_WORLD_SIZES:
        raise WorkerContractError("world size is not registered")
    if type(rank) is not int or rank < 0 or rank >= world_size:
        raise WorkerContractError("rank is invalid")


def registered_init_method(master_port: int) -> str:
    if type(master_port) is not int or master_port < 1 or master_port > 65535:
        raise WorkerContractError("master port is invalid")
    return f"tcp://127.0.0.1:{master_port}?use_libuv=0"


def validate_checkpoint_id(checkpoint_id: str) -> str:
    if type(checkpoint_id) is not str or checkpoint_id not in (
        REGISTERED_CHECKPOINT_IDS
    ):
        raise WorkerContractError("checkpoint identifier is not registered")
    path = Path(checkpoint_id)
    if path.is_absolute() or path.parent != Path(".") or path.name != checkpoint_id:
        raise WorkerContractError("checkpoint identifier must be one relative name")
    return checkpoint_id


def initialize_process_group(*, rank: int, world_size: int, master_port: int) -> None:
    validate_rank_world_size(rank, world_size)
    if dist.is_initialized():
        raise WorkerContractError("process group must start uninitialized")
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=registered_init_method(master_port),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )


def _torch_version() -> str:
    version = str(torch.__version__)
    if version not in {"2.11.0", "2.11.0+cpu"}:
        raise WorkerContractError("worker requires the registered PyTorch version")
    return version


def _slice_global_batch(
    value: torch.Tensor, *, rank: int, world_size: int
) -> torch.Tensor:
    if value.shape[0] % world_size:
        raise WorkerContractError("global batch does not divide across ranks")
    local_size = value.shape[0] // world_size
    start = rank * local_size
    return value.narrow(0, start, local_size)


def train_one_global_batch(
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    *,
    cursor: int,
    rank: int,
    world_size: int,
) -> int:
    batches = canonical_global_batches()
    if type(cursor) is not int or cursor < 0 or cursor >= len(batches):
        raise WorkerContractError("no registered batch exists at this cursor")
    inputs, targets = batches[cursor]
    jitter_bits = torch.randint(
        0,
        2,
        GLOBAL_TARGET_SHAPE,
        dtype=torch.int64,
        device="cpu",
        generator=generator,
    )
    registered_targets = targets + jitter_bits.to(torch.float64).mul_(0.0625)
    local_inputs = _slice_global_batch(inputs, rank=rank, world_size=world_size)
    local_targets = _slice_global_batch(
        registered_targets, rank=rank, world_size=world_size
    )
    optimizer.zero_grad(set_to_none=True)
    prediction = model(local_inputs)
    loss = torch.nn.functional.mse_loss(prediction, local_targets, reduction="mean")
    loss.backward()
    optimizer.step()
    return cursor + 1


def _training_dcp_state(
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    cursor: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    model_state, optimizer_state = get_state_dict(model, optimizer)
    application_state = {
        "cursor": torch.tensor([cursor], dtype=torch.int64),
        "generator_state": generator.get_state(),
    }
    return (
        {
            "application": application_state,
            "model": model_state,
            "optimizer": optimizer_state,
        },
        model_state,
        optimizer_state,
    )


def _current_training_digests(
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    cursor: int,
) -> dict[str, str]:
    _, model_state, optimizer_state = _training_dcp_state(
        model, optimizer, generator, cursor
    )
    return state_digests(
        model_state=model_state,
        optimizer_state=optimizer_state,
        generator_state=generator.get_state(),
        cursor=cursor,
    )


def _add_digest_fields(
    report: dict[str, Any], *, prefix: str, digests: dict[str, str]
) -> None:
    for name, value in digests.items():
        report[f"{prefix}_{name}"] = value


def _collective_verify_receipt(checkpoint_id: str) -> bool:
    local_ok = True
    try:
        receipt = verify_checkpoint(Path(checkpoint_id))
        local_ok = (
            receipt["logical_checkpoint_id"] == checkpoint_id
            and receipt["torch_version"] == _torch_version()
            and receipt["state_contract_sha256"] == state_contract_digest()
        )
    except (CheckpointReceiptError, OSError, KeyError, TypeError):
        local_ok = False
    flag = torch.tensor([1 if local_ok else 0], dtype=torch.int64)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _seal_checkpoint(checkpoint_id: str, rank: int) -> bool:
    dist.barrier()
    if rank == 0:
        receipt = build_receipt(
            Path(checkpoint_id),
            logical_checkpoint_id=checkpoint_id,
            torch_version=_torch_version(),
            state_contract_sha256=state_contract_digest(),
        )
        write_receipt(Path(checkpoint_id), receipt)
    dist.barrier()
    return _collective_verify_receipt(checkpoint_id)


def _base_report(*, action: str, rank: int, world_size: int) -> dict[str, Any]:
    return {
        "action": action,
        "bias_shape": list(BIAS_SHAPE),
        "global_batch_shape": list(GLOBAL_BATCH_SHAPE),
        "global_target_shape": list(GLOBAL_TARGET_SHAPE),
        "model_shape": list(MODEL_SHAPE),
        "rank": rank,
        "report_schema": WORKER_REPORT_SCHEMA,
        "state_contract_sha256": state_contract_digest(),
        "world_size": world_size,
    }


def _training_save_baseline(
    *, checkpoint_id: str, rank: int, world_size: int
) -> dict[str, Any]:
    model = DistributedDataParallel(build_model())
    optimizer = build_optimizer(model.parameters())
    generator = build_generator()
    cursor = train_one_global_batch(
        model,
        optimizer,
        generator,
        cursor=0,
        rank=rank,
        world_size=world_size,
    )
    save_state, _, _ = _training_dcp_state(model, optimizer, generator, cursor)
    checkpoint_digests = _current_training_digests(model, optimizer, generator, cursor)
    dcp.save(save_state, checkpoint_id=checkpoint_id)
    receipt_verified = _seal_checkpoint(checkpoint_id, rank)
    if not receipt_verified:
        raise WorkerContractError("saved checkpoint failed receipt verification")
    cursor = train_one_global_batch(
        model,
        optimizer,
        generator,
        cursor=cursor,
        rank=rank,
        world_size=world_size,
    )
    next_digests = _current_training_digests(model, optimizer, generator, cursor)
    report = _base_report(
        action="training-save-baseline", rank=rank, world_size=world_size
    )
    _add_digest_fields(report, prefix="checkpoint", digests=checkpoint_digests)
    _add_digest_fields(report, prefix="next", digests=next_digests)
    report["receipt_verified_after_save"] = receipt_verified
    return report


def _training_load_next(
    *, checkpoint_id: str, rank: int, world_size: int
) -> dict[str, Any]:
    if not _collective_verify_receipt(checkpoint_id):
        raise WorkerContractError("checkpoint receipt failed before trusted load")
    model = DistributedDataParallel(build_model())
    optimizer = build_optimizer(model.parameters())
    generator = build_generator()
    cursor = 0
    load_state, model_state, optimizer_state = _training_dcp_state(
        model, optimizer, generator, cursor
    )
    dcp.load(load_state, checkpoint_id=checkpoint_id)
    if not _collective_verify_receipt(checkpoint_id):
        raise WorkerContractError("checkpoint receipt changed during trusted load")
    set_state_dict(
        model,
        optimizer,
        model_state_dict=model_state,
        optim_state_dict=optimizer_state,
    )
    application_state = load_state["application"]
    generator.set_state(application_state["generator_state"].cpu())
    cursor_tensor = application_state["cursor"]
    if cursor_tensor.shape != (1,) or cursor_tensor.dtype is not torch.int64:
        raise WorkerContractError("loaded cursor tensor is outside the contract")
    cursor = int(cursor_tensor.item())
    if cursor != 1:
        raise WorkerContractError("training checkpoint cursor must equal one")
    loaded_digests = _current_training_digests(model, optimizer, generator, cursor)
    cursor = train_one_global_batch(
        model,
        optimizer,
        generator,
        cursor=cursor,
        rank=rank,
        world_size=world_size,
    )
    next_digests = _current_training_digests(model, optimizer, generator, cursor)
    report = _base_report(action="training-load-next", rank=rank, world_size=world_size)
    _add_digest_fields(report, prefix="loaded", digests=loaded_digests)
    _add_digest_fields(report, prefix="next", digests=next_digests)
    report["receipt_verified_after_load"] = True
    report["receipt_verified_before_load"] = True
    return report


def _dtensor_global_value() -> torch.Tensor:
    return torch.arange(16, dtype=torch.float64).reshape(DTENSOR_GLOBAL_SHAPE).div(8)


def _dtensor_for_save(rank: int, world_size: int) -> DTensor:
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("data",))
    global_value = _dtensor_global_value()
    local_value = _slice_global_batch(
        global_value, rank=rank, world_size=world_size
    ).clone()
    return DTensor.from_local(
        local_value,
        mesh,
        [Shard(0)],
        run_check=False,
        shape=torch.Size(DTENSOR_GLOBAL_SHAPE),
        stride=(DTENSOR_GLOBAL_SHAPE[1], 1),
    )


def _empty_dtensor(rank: int, world_size: int) -> DTensor:
    mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("data",))
    local_rows = DTENSOR_GLOBAL_SHAPE[0] // world_size
    local_value = torch.empty(
        (local_rows, DTENSOR_GLOBAL_SHAPE[1]), dtype=torch.float64
    )
    return DTensor.from_local(
        local_value,
        mesh,
        [Shard(0)],
        run_check=False,
        shape=torch.Size(DTENSOR_GLOBAL_SHAPE),
        stride=(DTENSOR_GLOBAL_SHAPE[1], 1),
    )


def _dtensor_report(
    *,
    action: str,
    tensor: DTensor,
    rank: int,
    world_size: int,
    receipt_verified: bool,
) -> dict[str, Any]:
    full = tensor.full_tensor()
    report = _base_report(action=action, rank=rank, world_size=world_size)
    report["dtensor_global_sha256"] = normalized_digest(full)
    report["dtensor_global_shape"] = list(full.shape)
    report["dtensor_local_shape"] = list(tensor.to_local().shape)
    if action == "dtensor-save":
        report["receipt_verified_after_save"] = receipt_verified
    else:
        report["receipt_verified_after_load"] = receipt_verified
        report["receipt_verified_before_load"] = receipt_verified
    return report


def _dtensor_save(*, checkpoint_id: str, rank: int, world_size: int) -> dict[str, Any]:
    tensor = _dtensor_for_save(rank, world_size)
    dcp.save({"tensor": tensor}, checkpoint_id=checkpoint_id)
    receipt_verified = _seal_checkpoint(checkpoint_id, rank)
    if not receipt_verified:
        raise WorkerContractError("saved DTensor failed receipt verification")
    return _dtensor_report(
        action="dtensor-save",
        tensor=tensor,
        rank=rank,
        world_size=world_size,
        receipt_verified=receipt_verified,
    )


def _dtensor_load(*, checkpoint_id: str, rank: int, world_size: int) -> dict[str, Any]:
    if not _collective_verify_receipt(checkpoint_id):
        raise WorkerContractError("checkpoint receipt failed before trusted load")
    tensor = _empty_dtensor(rank, world_size)
    state = {"tensor": tensor}
    dcp.load(state, checkpoint_id=checkpoint_id)
    if not _collective_verify_receipt(checkpoint_id):
        raise WorkerContractError("checkpoint receipt changed during trusted load")
    return _dtensor_report(
        action="dtensor-load",
        tensor=state["tensor"],
        rank=rank,
        world_size=world_size,
        receipt_verified=True,
    )


def run_registered_action(
    *,
    action: str,
    checkpoint_id: str,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    if action not in REGISTERED_ACTIONS:
        raise WorkerContractError("worker action is not registered")
    validate_rank_world_size(rank, world_size)
    validate_checkpoint_id(checkpoint_id)
    _torch_version()
    if not dist.is_initialized():
        raise WorkerContractError("process group is not initialized")
    if dist.get_rank() != rank or dist.get_world_size() != world_size:
        raise WorkerContractError("process group does not match worker arguments")
    if action == "training-save-baseline":
        return _training_save_baseline(
            checkpoint_id=checkpoint_id, rank=rank, world_size=world_size
        )
    if action == "training-load-next":
        return _training_load_next(
            checkpoint_id=checkpoint_id, rank=rank, world_size=world_size
        )
    if action == "dtensor-save":
        return _dtensor_save(
            checkpoint_id=checkpoint_id, rank=rank, world_size=world_size
        )
    return _dtensor_load(checkpoint_id=checkpoint_id, rank=rank, world_size=world_size)


def write_report(report_dir: Path, *, rank: int, report: dict[str, Any]) -> Path:
    validate_rank_world_size(rank, int(report.get("world_size", 0)))
    if report.get("rank") != rank:
        raise WorkerContractError("report rank does not match writer rank")
    if report.get("report_schema") != WORKER_REPORT_SCHEMA:
        raise WorkerContractError("report schema is invalid")
    report_dir.mkdir(parents=True, exist_ok=True)
    target = report_dir / f"rank-{rank}.json"
    pending = report_dir / f".rank-{rank}.json.pending"
    if (
        target.exists()
        or target.is_symlink()
        or pending.exists()
        or pending.is_symlink()
    ):
        raise WorkerContractError("worker report target must start absent")
    encoded = (canonical_json(report) + "\n").encode("utf-8")
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


def execute_worker(
    *,
    action: str,
    checkpoint_id: str,
    rank: int,
    world_size: int,
    master_port: int,
    report_dir: Path,
) -> dict[str, Any]:
    initialize_process_group(rank=rank, world_size=world_size, master_port=master_port)
    try:
        report = run_registered_action(
            action=action,
            checkpoint_id=checkpoint_id,
            rank=rank,
            world_size=world_size,
        )
        write_report(report_dir, rank=rank, report=report)
        return report
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dcp_invariant.worker")
    parser.add_argument("--action", required=True, choices=sorted(REGISTERED_ACTIONS))
    parser.add_argument(
        "--checkpoint-id",
        required=True,
        choices=sorted(REGISTERED_CHECKPOINT_IDS),
    )
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--master-port", required=True, type=int)
    parser.add_argument("--report-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    execute_worker(
        action=arguments.action,
        checkpoint_id=arguments.checkpoint_id,
        rank=arguments.rank,
        world_size=arguments.world_size,
        master_port=arguments.master_port,
        report_dir=arguments.report_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
