from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from dcp_invariant.canonical import canonical_json
from dcp_invariant.elastic_contract import failure_marker_payload
from dcp_invariant.elastic_worker import (
    ElasticWorkerContractError,
    execute_elastic_worker,
    read_failure_marker,
    write_failure_marker,
)


def environment(*, rank: int, restart_count: int) -> dict[str, str]:
    return {
        "LOCAL_RANK": str(rank),
        "LOCAL_WORLD_SIZE": "2",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29501",
        "RANK": str(rank),
        "TORCH_DISABLE_SHARE_RDZV_TCP_STORE": "1",
        "TORCHELASTIC_MAX_RESTARTS": "1",
        "TORCHELASTIC_RESTART_COUNT": str(restart_count),
        "WORLD_SIZE": "2",
    }


def test_worker_module_import_is_pytorch_free(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "src"
    script = """
import sys
import dcp_invariant.elastic_worker
raise SystemExit(1 if "torch" in sys.modules else 0)
"""
    environment = {**os.environ, "PYTHONPATH": str(source)}
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0


def test_failure_marker_is_exclusive_canonical_and_hashed(tmp_path: Path) -> None:
    marker = tmp_path / ".elastic-failure.json"
    digest = write_failure_marker(marker)
    value, observed = read_failure_marker(marker)
    expected_raw = (canonical_json(failure_marker_payload()) + "\n").encode()
    assert value == failure_marker_payload()
    assert marker.read_bytes() == expected_raw
    assert digest == observed == hashlib.sha256(expected_raw).hexdigest()
    with pytest.raises(ElasticWorkerContractError, match="start absent"):
        write_failure_marker(marker)


def test_restart_zero_rank_one_injects_91_and_rank_zero_observes_marker(
    tmp_path: Path,
) -> None:
    load = tmp_path / "elastic-load"
    control = tmp_path / "elastic-control"
    marker = tmp_path / ".elastic-failure.json"
    assert (
        execute_elastic_worker(
            checkpoint_id="checkpoint-one",
            load_report_dir=load,
            control_report_dir=control,
            failure_marker=marker,
            environment=environment(rank=1, restart_count=0),
        )
        == 91
    )
    assert (
        execute_elastic_worker(
            checkpoint_id="checkpoint-one",
            load_report_dir=load,
            control_report_dir=control,
            failure_marker=marker,
            environment=environment(rank=0, restart_count=0),
        )
        == 0
    )


def test_unregistered_checkpoint_is_rejected_before_torch_import(
    tmp_path: Path,
) -> None:
    with pytest.raises(ElasticWorkerContractError, match="checkpoint"):
        execute_elastic_worker(
            checkpoint_id="checkpoint-two",
            load_report_dir=tmp_path / "elastic-load",
            control_report_dir=tmp_path / "elastic-control",
            failure_marker=tmp_path / ".elastic-failure.json",
            environment=environment(rank=1, restart_count=0),
        )
