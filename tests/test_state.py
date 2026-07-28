from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from dcp_invariant.canonical import canonical_json, strict_json_loads
from dcp_invariant.elastic_contract import (
    BOOTSTRAP_ATTESTATION_NAME,
    bootstrap_attestation_payload,
)
from dcp_invariant.elastic_supervisor import _torchrun_environment
from dcp_invariant.state import (
    BIAS_SHAPE,
    GLOBAL_BATCH_SHAPE,
    GLOBAL_TARGET_SHAPE,
    MODEL_SHAPE,
    build_generator,
    build_model,
    canonical_global_batches,
    normalize_for_digest,
    normalized_digest,
    state_contract_digest,
    tensor_record,
)


def test_registered_fixture_shapes_and_dtypes() -> None:
    model = build_model()
    assert tuple(model.weight.shape) == MODEL_SHAPE
    assert tuple(model.bias.shape) == BIAS_SHAPE
    assert model.weight.dtype is torch.float64
    batches = canonical_global_batches()
    assert len(batches) == 2
    assert all(tuple(inputs.shape) == GLOBAL_BATCH_SHAPE for inputs, _ in batches)
    assert all(tuple(targets.shape) == GLOBAL_TARGET_SHAPE for _, targets in batches)
    assert all(inputs.dtype is torch.float64 for inputs, _ in batches)
    assert all(targets.dtype is torch.float64 for _, targets in batches)


def test_fixture_builders_return_independent_identical_state() -> None:
    first_model = build_model()
    second_model = build_model()
    assert torch.equal(first_model.weight, second_model.weight)
    assert torch.equal(first_model.bias, second_model.bias)
    first_generator = build_generator()
    second_generator = build_generator()
    assert torch.equal(first_generator.get_state(), second_generator.get_state())
    first_batches = canonical_global_batches()
    first_batches[0][0][0, 0] = 99.0
    assert canonical_global_batches()[0][0][0, 0].item() == 1.0


def test_normalization_distinguishes_bool_and_integer() -> None:
    assert normalize_for_digest(True) != normalize_for_digest(1)
    assert normalized_digest(True) != normalized_digest(1)


def test_tensor_digest_binds_dtype_shape_and_bytes() -> None:
    base = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    reshaped = base.reshape(2, 2)
    as_float = base.to(torch.float64)
    changed = torch.tensor([1, 2, 3, 5], dtype=torch.int64)
    records = [tensor_record(value) for value in (base, reshaped, as_float, changed)]
    assert records[0]["data_sha256"] == records[1]["data_sha256"]
    assert records[0]["shape"] != records[1]["shape"]
    assert records[0]["dtype"] != records[2]["dtype"]
    assert records[0]["data_sha256"] != records[3]["data_sha256"]
    digests = {
        normalized_digest(value) for value in (base, reshaped, as_float, changed)
    }
    assert len(digests) == 4


def test_state_contract_digest_is_stable_sha256() -> None:
    digest = state_contract_digest()
    assert len(digest) == 64
    assert digest == state_contract_digest()


def test_torchrun_bootstrap_agent_creates_exact_attestation(tmp_path: Path) -> None:
    attestation = tmp_path / BOOTSTRAP_ATTESTATION_NAME
    environment = _torchrun_environment(tmp_path, attestation)
    script = """
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous import c10d_rendezvous_backend

params = RendezvousParameters(
    backend="c10d",
    endpoint="127.0.0.1:0",
    run_id="registered-bootstrap-test",
    min_nodes=1,
    max_nodes=1,
    is_host=True,
    read_timeout=5,
)
store = c10d_rendezvous_backend._create_tcp_store(params)
assert store is not None
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
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    raw = attestation.read_text(encoding="utf-8")
    expected = bootstrap_attestation_payload(str(torch.__version__))
    assert raw == canonical_json(expected) + "\n"
    assert strict_json_loads(raw[:-1]) == expected


def test_torchrun_bootstrap_rejects_direct_tcpstore_attestation(
    tmp_path: Path,
) -> None:
    attestation = tmp_path / BOOTSTRAP_ATTESTATION_NAME
    environment = _torchrun_environment(tmp_path, attestation)
    script = """
from datetime import timedelta
from torch.distributed.elastic.rendezvous import c10d_rendezvous_backend

c10d_rendezvous_backend.TCPStore(
    "127.0.0.1",
    0,
    is_master=True,
    multi_tenant=True,
    timeout=timedelta(seconds=5),
)
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
    assert result.returncode != 0
    assert (
        "TCPStore call did not originate from _create_tcp_store"
        in result.stderr.decode(errors="replace")
    )
    assert not attestation.exists()
