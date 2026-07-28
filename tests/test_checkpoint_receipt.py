from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from dcp_invariant.checkpoint_receipt import (
    CheckpointReceiptError,
    build_receipt,
    verify_checkpoint,
    write_receipt,
)

CONTRACT = "a" * 64


def native_checkpoint(root: Path, *, shards: int = 2) -> Path:
    checkpoint = root / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / ".metadata").write_bytes(b"registered relative metadata")
    for rank in range(shards):
        (checkpoint / f"__{rank}_0.distcp").write_bytes(
            f"rank={rank}:synthetic".encode()
        )
    return checkpoint


def seal(checkpoint: Path) -> dict[str, object]:
    receipt = build_receipt(
        checkpoint,
        logical_checkpoint_id="checkpoint-two",
        torch_version="2.11.0+cpu",
        state_contract_sha256=CONTRACT,
    )
    write_receipt(checkpoint, receipt)
    return receipt


def test_receipt_round_trip(tmp_path: Path) -> None:
    checkpoint = native_checkpoint(tmp_path)
    expected = seal(checkpoint)
    assert verify_checkpoint(checkpoint) == expected


def test_async_checkpoint_identifier_is_receipt_bound(tmp_path: Path) -> None:
    checkpoint = native_checkpoint(tmp_path)
    receipt = build_receipt(
        checkpoint,
        logical_checkpoint_id="checkpoint-async",
        torch_version="2.11.0+cpu",
        state_contract_sha256=CONTRACT,
    )
    write_receipt(checkpoint, receipt)
    assert verify_checkpoint(checkpoint)["logical_checkpoint_id"] == "checkpoint-async"


def test_changed_shard_is_rejected_before_load(tmp_path: Path) -> None:
    checkpoint = native_checkpoint(tmp_path)
    seal(checkpoint)
    (checkpoint / "__0_0.distcp").write_bytes(b"changed")
    with pytest.raises(CheckpointReceiptError, match="do not match"):
        verify_checkpoint(checkpoint)


def test_missing_metadata_is_rejected(tmp_path: Path) -> None:
    checkpoint = native_checkpoint(tmp_path)
    (checkpoint / ".metadata").unlink()
    with pytest.raises(CheckpointReceiptError, match="metadata"):
        build_receipt(
            checkpoint,
            logical_checkpoint_id="checkpoint-two",
            torch_version="2.11.0",
            state_contract_sha256=CONTRACT,
        )


def test_extra_file_is_rejected(tmp_path: Path) -> None:
    checkpoint = native_checkpoint(tmp_path)
    seal(checkpoint)
    (checkpoint / "unexpected.txt").write_text("no", encoding="utf-8")
    with pytest.raises(CheckpointReceiptError, match="outside"):
        verify_checkpoint(checkpoint)


def test_noncanonical_receipt_is_rejected(tmp_path: Path) -> None:
    checkpoint = native_checkpoint(tmp_path)
    receipt = seal(checkpoint)
    (checkpoint / "checkpoint-receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(CheckpointReceiptError, match="canonical"):
        verify_checkpoint(checkpoint)


def test_receipt_is_not_overwritten(tmp_path: Path) -> None:
    checkpoint = native_checkpoint(tmp_path)
    receipt = seal(checkpoint)
    with pytest.raises(CheckpointReceiptError, match="must start absent"):
        write_receipt(checkpoint, receipt)


def test_symlink_entry_is_rejected_when_supported(tmp_path: Path) -> None:
    checkpoint = native_checkpoint(tmp_path)
    target = checkpoint / "__0_0.distcp"
    link = checkpoint / "__2_0.distcp"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("ordinary user cannot create symlinks on this platform")
    with pytest.raises(CheckpointReceiptError, match="ordinary"):
        build_receipt(
            checkpoint,
            logical_checkpoint_id="checkpoint-two",
            torch_version="2.11.0",
            state_contract_sha256=CONTRACT,
        )


def test_duplicate_numeric_shard_coordinate_is_rejected(tmp_path: Path) -> None:
    checkpoint = native_checkpoint(tmp_path, shards=1)
    (checkpoint / "__00_0.distcp").write_bytes(b"duplicate coordinate")
    with pytest.raises(CheckpointReceiptError, match="duplicate shard"):
        build_receipt(
            checkpoint,
            logical_checkpoint_id="checkpoint-two",
            torch_version="2.11.0",
            state_contract_sha256=CONTRACT,
        )


def test_arbitrary_receipt_cannot_be_written(tmp_path: Path) -> None:
    checkpoint = native_checkpoint(tmp_path)
    receipt = build_receipt(
        checkpoint,
        logical_checkpoint_id="checkpoint-two",
        torch_version="2.11.0",
        state_contract_sha256=CONTRACT,
    )
    receipt["files"][0]["sha256"] = "b" * 64
    with pytest.raises(CheckpointReceiptError, match="current checkpoint"):
        write_receipt(checkpoint, receipt)
    assert not (checkpoint / "checkpoint-receipt.json").exists()


def test_hard_link_alias_is_rejected_when_supported(tmp_path: Path) -> None:
    checkpoint = native_checkpoint(tmp_path, shards=1)
    try:
        os.link(
            checkpoint / "__0_0.distcp",
            checkpoint / "__1_0.distcp",
        )
    except OSError:
        pytest.skip("hard links are unavailable on this filesystem")
    with pytest.raises(CheckpointReceiptError, match="hard-link|filesystem identity"):
        build_receipt(
            checkpoint,
            logical_checkpoint_id="checkpoint-two",
            torch_version="2.11.0",
            state_contract_sha256=CONTRACT,
        )
