from __future__ import annotations

import errno
import json
import os
import stat
from pathlib import Path
from unittest import mock

import pytest

from dcp_invariant import elastic_worker
from dcp_invariant.elastic_contract import ElasticEnvironment


@pytest.fixture(params=["worker", "elastic", "async"])
def report_writer(request, tmp_path: Path):
    if request.param == "elastic":
        root = tmp_path / "elastic-control"
        root.mkdir()

        def write():
            return elastic_worker._write_control_report(
                root,
                environment=ElasticEnvironment(0, 0, 2, 2, 1, 1),
                failure_marker_sha256="a" * 64,
            )

        error_type = elastic_worker.ElasticWorkerContractError
    elif request.param == "async":
        pytest.importorskip("torch")
        from dcp_invariant import async_snapshot_worker

        root = tmp_path

        def write():
            return async_snapshot_worker._write_async_report(
                root,
                rank=0,
                report={
                    "rank": 0,
                    "world_size": 2,
                    "report_schema": async_snapshot_worker.ASYNC_SNAPSHOT_REPORT_SCHEMA,
                },
            )

        error_type = async_snapshot_worker.AsyncSnapshotWorkerError
    else:
        pytest.importorskip("torch")
        from dcp_invariant import worker

        root = tmp_path

        def write():
            return worker.write_report(
                root,
                rank=0,
                report={
                    "rank": 0,
                    "world_size": 1,
                    "report_schema": worker.WORKER_REPORT_SCHEMA,
                },
            )

        error_type = worker.WorkerContractError
    return root, write, error_type


def test_competing_report_staging_is_preserved(report_writer) -> None:
    root, write, error_type = report_writer
    pending = root / ".rank-0.json.pending"
    marker = b"another writer's staging\n"
    real_open = Path.open

    def competing_open(path, mode="r", *args, **kwargs):
        if path == pending and mode == "xb":
            with real_open(pending, "xb") as other:
                other.write(marker)
            raise FileExistsError(errno.EEXIST, "synthetic staging collision")
        return real_open(path, mode, *args, **kwargs)

    with (
        mock.patch.object(Path, "open", competing_open),
        pytest.raises((error_type, OSError)),
    ):
        write()
    assert pending.read_bytes() == marker
    assert not (root / "rank-0.json").exists()


def test_report_appearing_during_staging_is_preserved(report_writer) -> None:
    root, write, error_type = report_writer
    target = root / "rank-0.json"
    marker = b"another writer's completed report\n"
    real_sync = os.fsync
    injected = False

    def competing_publish(descriptor):
        nonlocal injected
        if not injected and stat.S_ISREG(os.fstat(descriptor).st_mode):
            injected = True
            target.write_bytes(marker)
        return real_sync(descriptor)

    with (
        mock.patch.object(os, "fsync", side_effect=competing_publish),
        pytest.raises((error_type, OSError)),
    ):
        write()
    assert injected
    assert target.read_bytes() == marker
    assert (root / ".rank-0.json.pending").is_file()


def test_report_flush_failure_retains_staging(report_writer) -> None:
    root, write, error_type = report_writer
    with (
        mock.patch.object(
            os, "fsync", side_effect=OSError(errno.EIO, "synthetic flush failure")
        ),
        pytest.raises((error_type, OSError)),
    ):
        write()
    assert (root / ".rank-0.json.pending").is_file()
    assert not (root / "rank-0.json").exists()


def test_replaced_report_staging_survives_close_failure(report_writer) -> None:
    root, write, error_type = report_writer
    pending = root / ".rank-0.json.pending"
    displaced = root / "displaced-staging"
    marker = b"another writer's replacement staging\n"
    real_open = Path.open

    class FailingClose:
        def __enter__(self):
            self.output = real_open(pending, "xb")
            return self.output

        def __exit__(self, *exception):
            self.output.close()
            pending.rename(displaced)
            pending.write_bytes(marker)
            raise OSError(errno.EIO, "synthetic close failure")

    def replace_after_close(path, mode="r", *args, **kwargs):
        if path == pending and mode == "xb":
            return FailingClose()
        return real_open(path, mode, *args, **kwargs)

    with (
        mock.patch.object(Path, "open", replace_after_close),
        pytest.raises((error_type, OSError)),
    ):
        write()
    assert pending.read_bytes() == marker
    assert displaced.is_file()
    assert not (root / "rank-0.json").exists()


def test_preexisting_report_staging_is_preserved(report_writer) -> None:
    root, write, error_type = report_writer
    pending = root / ".rank-0.json.pending"
    pending.write_bytes(b"preexisting staging")
    with pytest.raises(error_type):
        write()
    assert pending.read_bytes() == b"preexisting staging"


def test_successful_report_has_single_identity_and_no_pending(report_writer) -> None:
    root, write, _ = report_writer
    target = write()
    assert target == root / "rank-0.json"
    assert target.stat().st_nlink == 1
    assert json.loads(target.read_bytes())["rank"] == 0
    assert not (root / ".rank-0.json.pending").exists()
