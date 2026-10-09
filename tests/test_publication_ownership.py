from __future__ import annotations

import errno
import hashlib
import os
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

import dcp_invariant.checkpoint_receipt as receipts
import dcp_invariant.supervisor as supervisor
from dcp_invariant.checkpoint_receipt import CheckpointReceiptError
from dcp_invariant.supervisor import SupervisorError


def native_checkpoint(root: Path) -> tuple[Path, dict]:
    checkpoint = root / "checkpoint-one"
    checkpoint.mkdir()
    (checkpoint / ".metadata").write_bytes(b"synthetic metadata")
    (checkpoint / "__0_0.distcp").write_bytes(b"synthetic state")
    receipt = receipts.build_receipt(
        checkpoint,
        logical_checkpoint_id="checkpoint-one",
        torch_version="2.11.0+cpu",
        state_contract_sha256="a" * 64,
    )
    return checkpoint, receipt


def test_competing_staging_file_is_not_deleted(tmp_path: Path) -> None:
    checkpoint, receipt = native_checkpoint(tmp_path)
    temporary = checkpoint / f".{receipts.RECEIPT_NAME}.pending"
    marker = b"another writer's staged receipt\n"
    real_open = Path.open

    def competing_open(path, mode="r", *args, **kwargs):
        if path == temporary and mode == "xb":
            with real_open(temporary, "xb") as other:
                other.write(marker)
            raise FileExistsError(errno.EEXIST, "synthetic staging collision")
        return real_open(path, mode, *args, **kwargs)

    with (
        mock.patch.object(Path, "open", competing_open),
        pytest.raises((CheckpointReceiptError, OSError)),
    ):
        receipts.write_receipt(checkpoint, receipt)
    assert temporary.is_file(), "another writer's staging file was deleted"
    assert temporary.read_bytes() == marker
    assert not (checkpoint / receipts.RECEIPT_NAME).exists()


def test_receipt_that_appears_during_staging_is_not_overwritten(tmp_path: Path) -> None:
    checkpoint, receipt = native_checkpoint(tmp_path)
    target = checkpoint / receipts.RECEIPT_NAME
    marker = b"another writer's published receipt\n"
    real_sync = os.fsync
    injected = False

    def competing_publish(descriptor):
        nonlocal injected
        if not injected and stat.S_ISREG(os.fstat(descriptor).st_mode):
            injected = True
            target.write_bytes(marker)
        return real_sync(descriptor)

    with (
        mock.patch.object(receipts.os, "fsync", side_effect=competing_publish),
        pytest.raises((CheckpointReceiptError, OSError)),
    ):
        receipts.write_receipt(checkpoint, receipt)
    assert injected
    assert target.read_bytes() == marker


def test_replaced_staging_file_is_not_deleted_after_flush_failure(
    tmp_path: Path,
) -> None:
    checkpoint, receipt = native_checkpoint(tmp_path)
    temporary = checkpoint / f".{receipts.RECEIPT_NAME}.pending"
    displaced = tmp_path / "owned-staging-displaced"
    marker = b"another writer's replacement staging\n"
    real_open = Path.open

    class FailingClose:
        def __enter__(self):
            self.output = real_open(temporary, "xb")
            return self.output

        def __exit__(self, *exception):
            self.output.close()
            temporary.rename(displaced)
            temporary.write_bytes(marker)
            raise OSError(errno.EIO, "synthetic staging close failure")

    def replace_after_close(path, mode="r", *args, **kwargs):
        if path == temporary and mode == "xb":
            return FailingClose()
        return real_open(path, mode, *args, **kwargs)

    with (
        mock.patch.object(Path, "open", replace_after_close),
        pytest.raises((CheckpointReceiptError, OSError)),
    ):
        receipts.write_receipt(checkpoint, receipt)
    assert temporary.is_file(), "a replacement staging file was deleted"
    assert temporary.read_bytes() == marker
    assert displaced.is_file()
    assert not (checkpoint / receipts.RECEIPT_NAME).exists()


def test_preexisting_staging_file_is_preserved(tmp_path: Path) -> None:
    checkpoint, receipt = native_checkpoint(tmp_path)
    temporary = checkpoint / f".{receipts.RECEIPT_NAME}.pending"
    temporary.write_bytes(b"preexisting staging")
    with pytest.raises(CheckpointReceiptError):
        receipts.write_receipt(checkpoint, receipt)
    assert temporary.read_bytes() == b"preexisting staging"


def test_receipt_publication_round_trip_has_one_file_identity(tmp_path: Path) -> None:
    checkpoint, receipt = native_checkpoint(tmp_path)
    receipts.write_receipt(checkpoint, receipt)
    assert receipts.verify_checkpoint(checkpoint) == receipt
    assert (checkpoint / receipts.RECEIPT_NAME).stat().st_nlink == 1
    assert not (checkpoint / f".{receipts.RECEIPT_NAME}.pending").exists()


def test_generation_directory_appearing_during_commit_is_preserved(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    (root / "candidates").mkdir(parents=True)
    (root / "committed").mkdir()
    candidate = root / "candidates" / "candidate-one"
    candidate.mkdir()
    checkpoint, receipt = native_checkpoint(candidate)
    receipts.write_receipt(checkpoint, receipt)
    real_write = supervisor._write_exclusive
    competing_target = None
    competing_identity = None

    def create_competing_directory(path, value, *, label):
        nonlocal competing_target, competing_identity
        real_write(path, value, label=label)
        competing_target = (
            root
            / "committed"
            / hashlib.sha256(
                (checkpoint / receipts.RECEIPT_NAME).read_bytes()
            ).hexdigest()
        )
        competing_target.mkdir()
        metadata = competing_target.stat()
        competing_identity = (metadata.st_dev, metadata.st_ino)

    def verify(path):
        return receipts.verify_checkpoint(path) == receipt

    with (
        mock.patch.object(supervisor, "_write_exclusive", create_competing_directory),
        pytest.raises(SupervisorError),
    ):
        supervisor.commit_candidate(
            root=root,
            candidate=candidate,
            logical_checkpoint_id="checkpoint-one",
            verify=verify,
            parent=supervisor.read_parent_version(root),
        )
    assert competing_target is not None and competing_target.is_dir()
    observed = competing_target.stat()
    assert (observed.st_dev, observed.st_ino) == competing_identity
    assert list(competing_target.iterdir()) == []
    assert candidate.is_dir()
    assert not (root / "LATEST.json").exists()


def test_owned_staging_is_retained_when_flush_fails(tmp_path: Path) -> None:
    checkpoint, receipt = native_checkpoint(tmp_path)
    with (
        mock.patch.object(
            receipts.os, "fsync", side_effect=OSError(errno.EIO, "synthetic")
        ),
        pytest.raises(CheckpointReceiptError),
    ):
        receipts.write_receipt(checkpoint, receipt)
    temporary = checkpoint / f".{receipts.RECEIPT_NAME}.pending"
    assert temporary.is_file()
    assert not (checkpoint / receipts.RECEIPT_NAME).exists()
    assert (checkpoint / "__0_0.distcp").read_bytes() == b"synthetic state"


@pytest.mark.skipif(os.name != "posix", reason="POSIX native primitive availability")
def test_missing_native_primitive_fails_without_publishing(tmp_path: Path) -> None:
    checkpoint, receipt = native_checkpoint(tmp_path)
    with (
        mock.patch("ctypes.CDLL", return_value=SimpleNamespace()),
        pytest.raises(CheckpointReceiptError),
    ):
        receipts.write_receipt(checkpoint, receipt)
    assert not (checkpoint / receipts.RECEIPT_NAME).exists()
    assert (checkpoint / f".{receipts.RECEIPT_NAME}.pending").is_file()
