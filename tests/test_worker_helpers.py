from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from dcp_invariant.worker import (
    WORKER_REPORT_SCHEMA,
    WorkerContractError,
    execute_worker,
    registered_init_method,
    validate_checkpoint_id,
    validate_rank_world_size,
    write_report,
)


def free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


@pytest.mark.parametrize("checkpoint_id", ["checkpoint-one", "checkpoint-two"])
def test_registered_checkpoint_id_is_one_relative_name(checkpoint_id: str) -> None:
    assert validate_checkpoint_id(checkpoint_id) == checkpoint_id


@pytest.mark.parametrize(
    "checkpoint_id",
    [
        "",
        ".",
        "checkpoint-three",
        "../checkpoint-one",
        "nested/checkpoint-one",
        "C:\\checkpoint-one",
    ],
)
def test_unregistered_checkpoint_id_is_rejected(checkpoint_id: str) -> None:
    with pytest.raises(WorkerContractError):
        validate_checkpoint_id(checkpoint_id)


def test_process_group_arguments_are_explicit_and_bounded() -> None:
    validate_rank_world_size(0, 1)
    validate_rank_world_size(1, 2)
    assert registered_init_method(29501) == "tcp://127.0.0.1:29501?use_libuv=0"
    with pytest.raises(WorkerContractError):
        validate_rank_world_size(2, 2)
    with pytest.raises(WorkerContractError):
        registered_init_method(True)


def test_report_is_canonical_and_does_not_overwrite(tmp_path: Path) -> None:
    report = {
        "action": "training-save-baseline",
        "rank": 0,
        "report_schema": WORKER_REPORT_SCHEMA,
        "state_sha256": "a" * 64,
        "world_size": 1,
    }
    target = write_report(tmp_path, rank=0, report=report)
    raw = target.read_text(encoding="utf-8")
    assert raw.endswith("\n")
    assert "\r" not in raw
    assert json.loads(raw) == report
    with pytest.raises(WorkerContractError, match="must start absent"):
        write_report(tmp_path, rank=0, report=report)


def test_world_size_one_training_save_and_trusted_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    save_report = execute_worker(
        action="training-save-baseline",
        checkpoint_id="checkpoint-one",
        rank=0,
        world_size=1,
        master_port=free_local_port(),
        report_dir=tmp_path / "save-reports",
    )
    load_report = execute_worker(
        action="training-load-next",
        checkpoint_id="checkpoint-one",
        rank=0,
        world_size=1,
        master_port=free_local_port(),
        report_dir=tmp_path / "load-reports",
    )
    assert save_report["receipt_verified_after_save"] is True
    assert load_report["receipt_verified_before_load"] is True
    assert load_report["receipt_verified_after_load"] is True
    assert save_report["checkpoint_state_sha256"] == load_report["loaded_state_sha256"]
    assert save_report["next_state_sha256"] == load_report["next_state_sha256"]


def test_world_size_one_dtensor_save_and_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    save_report = execute_worker(
        action="dtensor-save",
        checkpoint_id="checkpoint-two",
        rank=0,
        world_size=1,
        master_port=free_local_port(),
        report_dir=tmp_path / "dtensor-save-reports",
    )
    load_report = execute_worker(
        action="dtensor-load",
        checkpoint_id="checkpoint-two",
        rank=0,
        world_size=1,
        master_port=free_local_port(),
        report_dir=tmp_path / "dtensor-load-reports",
    )
    assert save_report["receipt_verified_after_save"] is True
    assert load_report["receipt_verified_before_load"] is True
    assert load_report["receipt_verified_after_load"] is True
    assert save_report["dtensor_global_sha256"] == load_report["dtensor_global_sha256"]
    assert save_report["dtensor_global_shape"] == [4, 4]
