from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from dcp_invariant.elastic_contract import (
    BOOTSTRAP_ATTESTATION_NAME,
    BOOTSTRAP_SHARED_STORE_ENV,
)
from dcp_invariant.elastic_supervisor import (
    ElasticSupervisorError,
    _torchrun_environment,
    elastic_command,
)


def test_supervisor_import_does_not_import_torch(tmp_path: Path) -> None:
    source = Path(__file__).parents[1] / "src"
    script = """
import sys
import dcp_invariant.elastic_supervisor
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


def test_inherited_torchrun_worker_bootstrap_is_pytorch_free_noop(
    tmp_path: Path,
) -> None:
    attestation = tmp_path / BOOTSTRAP_ATTESTATION_NAME
    attestation.write_text("private-agent-attestation", encoding="utf-8")
    environment = _torchrun_environment(tmp_path, attestation)
    assert environment[BOOTSTRAP_SHARED_STORE_ENV] == "1"
    environment.update(
        {
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": "2",
            "RANK": "0",
            "WORLD_SIZE": "2",
        }
    )
    script = """
import sys
raise SystemExit(1 if "torch" in sys.modules else 0)
"""
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
    assert attestation.read_text(encoding="utf-8") == "private-agent-attestation"


def test_elastic_command_is_exact_shell_free_module_launch(tmp_path: Path) -> None:
    load = tmp_path / "elastic-load"
    control = tmp_path / "elastic-control"
    marker = tmp_path / ".elastic-failure.json"
    command = elastic_command(
        load_report_dir=load,
        control_report_dir=control,
        failure_marker=marker,
    )
    assert command[:3] == (sys.executable, "-m", "torch.distributed.run")
    assert command[3:10] == (
        "--standalone",
        "--local-addr=127.0.0.1",
        "--nnodes=1",
        "--nproc-per-node=2",
        "--max-restarts=1",
        "--monitor-interval=0.1",
        "--module",
    )
    assert command[10:12] == (
        "dcp_invariant.elastic_worker",
        "--checkpoint-id",
    )
    assert "shell" not in command


def test_elastic_command_rejects_unregistered_private_names(tmp_path: Path) -> None:
    with pytest.raises(ElasticSupervisorError, match="load report"):
        elastic_command(
            load_report_dir=tmp_path / "other",
            control_report_dir=tmp_path / "elastic-control",
            failure_marker=tmp_path / ".elastic-failure.json",
        )
