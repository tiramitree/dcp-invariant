from __future__ import annotations

from dcp_invariant.async_snapshot_contract import (
    ASYNC_SNAPSHOT_WORKLOAD_SCHEMA,
    is_registered_torchvision_version_pair,
    workload_contract,
    workload_contract_digest,
)
from dcp_invariant.canonical import sha256_json


def test_workload_contract_is_fixed_truthful_and_digest_bound() -> None:
    workload = workload_contract()

    assert workload["workload_schema"] == ASYNC_SNAPSHOT_WORKLOAD_SCHEMA
    assert workload["world_size"] == 2
    assert workload["model_constructor"] == "torchvision.models.resnet18"
    assert workload["model_weights"] == "none"
    assert workload["weights_downloaded"] is False
    assert workload["pre_snapshot_training_steps"] == 1
    assert workload["application_cursor_mutated"] is False
    assert workload["post_stage_mutation"] == {
        "operation": "add-fixed-scalar",
        "scalar_hex": (2**-10).hex(),
        "target": "model.conv1.weight",
    }
    assert workload["torchvision_release"] == "0.26.0"
    assert workload["pillow_release"] == "12.3.0"
    assert workload_contract_digest() == sha256_json(workload)


def test_only_registered_torchvision_distribution_runtime_pairs_pass() -> None:
    assert is_registered_torchvision_version_pair("0.26.0", "0.26.0+cpu")
    assert is_registered_torchvision_version_pair("0.26.0+cpu", "0.26.0+cpu")
    assert not is_registered_torchvision_version_pair("0.26.0", "0.26.0")
    assert not is_registered_torchvision_version_pair("0.26.1", "0.26.1+cpu")
    assert not is_registered_torchvision_version_pair(None, "0.26.0+cpu")
