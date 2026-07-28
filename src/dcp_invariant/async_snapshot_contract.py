"""PyTorch-free contract for the fixed asynchronous snapshot scenario."""

from __future__ import annotations

from .canonical import sha256_json

ASYNC_SNAPSHOT_SCENARIO = "async_snapshot_resnet18_2r"
ASYNC_SNAPSHOT_OBSERVATION_SCHEMA = "dcp-invariant-async-snapshot-observation-v1"
ASYNC_SNAPSHOT_REPORT_SCHEMA = "dcp-invariant-async-snapshot-report-v1"
ASYNC_SNAPSHOT_WORKLOAD_SCHEMA = "dcp-invariant-async-snapshot-workload-v1"
ASYNC_SNAPSHOT_ACTION = "async-snapshot-save-load"
ASYNC_CHECKPOINT_ID = "checkpoint-async"
ASYNC_GATE_DIRECTORY_NAME = "async-gate"
ASYNC_REPORT_DIRECTORY_NAME = "async-reports"
ASYNC_WORLD_SIZE = 2
RESNET18_PARAMETER_COUNT = 11_689_512
GLOBAL_INPUT_SHAPE = (4, 3, 32, 32)
GLOBAL_TARGET_SHAPE = (4,)
REGISTERED_TORCHVISION_VERSION_PAIRS = frozenset(
    {
        ("0.26.0", "0.26.0+cpu"),
        ("0.26.0+cpu", "0.26.0+cpu"),
    }
)


def workload_contract() -> dict[str, object]:
    """Return the public, deterministic workload and gate declaration."""

    return {
        "async_api": "torch.distributed.checkpoint.async_save",
        "async_checkpointer": "thread",
        "async_stager": "torch.distributed.checkpoint.FileSystemWriter.stage",
        "application_cursor_mutated": False,
        "checkpoint_writer_gate": "StorageWriter.write_data-before-delegate",
        "distributed_backend": "gloo",
        "global_input_shape": list(GLOBAL_INPUT_SHAPE),
        "global_target_shape": list(GLOBAL_TARGET_SHAPE),
        "input_source": "synthetic-arange",
        "model_constructor": "torchvision.models.resnet18",
        "model_parameter_count": RESNET18_PARAMETER_COUNT,
        "model_weights": "none",
        "optimizer": {
            "learning_rate_hex": float(2**-8).hex(),
            "momentum_hex": (0.5).hex(),
            "type": "torch.optim.SGD",
        },
        "pillow_release": "12.3.0",
        "post_stage_mutation": {
            "operation": "add-fixed-scalar",
            "scalar_hex": float(2**-10).hex(),
            "target": "model.conv1.weight",
        },
        "pre_snapshot_training_steps": 1,
        "torchvision_release": "0.26.0",
        "weights_downloaded": False,
        "workload_schema": ASYNC_SNAPSHOT_WORKLOAD_SCHEMA,
        "world_size": ASYNC_WORLD_SIZE,
    }


def workload_contract_digest() -> str:
    return sha256_json(workload_contract())


def is_registered_torchvision_version_pair(
    distribution_version: object,
    runtime_version: object,
) -> bool:
    return (
        type(distribution_version) is str
        and type(runtime_version) is str
        and (distribution_version, runtime_version)
        in REGISTERED_TORCHVISION_VERSION_PAIRS
    )
