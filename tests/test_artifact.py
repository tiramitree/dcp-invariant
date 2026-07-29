from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

import dcp_invariant.artifact as artifact_module
from dcp_invariant.artifact import (
    DTENSOR_OBSERVATION_SCHEMA,
    ELASTIC_OBSERVATION_SCHEMA,
    FAULT_OBSERVATION_SCHEMA,
    LATEST_SCHEMA,
    MANIFEST_NAME,
    REGISTERED_SCENARIOS,
    TRAINING_OBSERVATION_SCHEMA,
    EvidenceArtifactError,
    build_evidence_artifact,
    normalize_observation,
    verify_evidence_artifact,
)
from dcp_invariant.async_snapshot_contract import (
    ASYNC_CHECKPOINT_ID,
    ASYNC_SNAPSHOT_ACTION,
    ASYNC_SNAPSHOT_OBSERVATION_SCHEMA,
    ASYNC_SNAPSHOT_REPORT_SCHEMA,
    ASYNC_SNAPSHOT_SCENARIO,
    workload_contract,
    workload_contract_digest,
)
from dcp_invariant.canonical import canonical_json
from dcp_invariant.elastic_contract import (
    BOOTSTRAP_ID,
    ELASTIC_REPORT_SCHEMA,
    bootstrap_attestation_payload,
    failure_marker_payload,
)

SOURCE_REVISION = "a" * 40
PYTHON_VERSION = "3.12.10"
TORCH_DISTRIBUTION_VERSION = "2.11.0"
TORCH_VERSION = "2.11.0+cpu"
NUMPY_VERSION = "2.4.6"
STATE_CONTRACT = hashlib.sha256(b"state-contract").hexdigest()
COMPONENTS = ("cursor", "model", "optimizer", "rng", "state")


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def pointer(
    generation: str,
    checkpoint_id: str = "checkpoint-one",
    *,
    parent_pointer_sha256: str | None = None,
    sequence: int = 0,
) -> dict[str, object]:
    lineage = {
        "generation": generation,
        "lineage_schema": "dcp-invariant-generation-lineage-v1",
        "logical_checkpoint_id": checkpoint_id,
        "parent_pointer_sha256": parent_pointer_sha256,
        "sequence": sequence,
    }
    value = {
        "generation": generation,
        "lineage_sha256": hashlib.sha256(
            (canonical_json(lineage) + "\n").encode()
        ).hexdigest(),
        "parent_pointer_sha256": parent_pointer_sha256,
        "pointer_schema": LATEST_SCHEMA,
        "sequence": sequence,
    }
    return {
        **value,
        "pointer_sha256": hashlib.sha256(
            (canonical_json(value) + "\n").encode()
        ).hexdigest(),
    }


def common_report(action: str, rank: int, world_size: int) -> dict[str, object]:
    return {
        "action": action,
        "bias_shape": [2],
        "global_batch_shape": [4, 3],
        "global_target_shape": [4, 2],
        "model_shape": [2, 3],
        "rank": rank,
        "report_schema": "dcp-invariant-worker-report-v1",
        "state_contract_sha256": STATE_CONTRACT,
        "world_size": world_size,
    }


def training_reports(
    *,
    scenario: str,
    action: str,
    world_size: int,
) -> list[dict[str, object]]:
    first_prefix = "checkpoint" if action == "training-save-baseline" else "loaded"
    reports = []
    for rank in range(world_size):
        report = common_report(action, rank, world_size)
        for component in COMPONENTS:
            report[f"{first_prefix}_{component}_sha256"] = digest(
                f"{scenario}-checkpoint-{component}"
            )
            report[f"next_{component}_sha256"] = digest(f"{scenario}-next-{component}")
        if action == "training-save-baseline":
            report["receipt_verified_after_save"] = True
        else:
            report["receipt_verified_before_load"] = True
            report["receipt_verified_after_load"] = True
        reports.append(report)
    return reports


def worker_outcome(world_size: int) -> dict[str, object]:
    return {"exit_codes": [0] * world_size, "timed_out": False}


def async_observation() -> dict[str, object]:
    receipt = digest("async-receipt")
    pre = {
        "cursor": digest("async-pre-cursor"),
        "model": digest("async-pre-model"),
        "optimizer": digest("async-pre-optimizer"),
        "state": digest("async-pre-state"),
    }
    post = {
        "cursor": pre["cursor"],
        "model": digest("async-post-model"),
        "optimizer": pre["optimizer"],
        "state": digest("async-post-state"),
    }
    reports = []
    for rank in range(2):
        reports.append(
            {
                "action": ASYNC_SNAPSHOT_ACTION,
                "async_checkpointer": "thread",
                "direct_loaded_model_sha256": pre["model"],
                "future_pending_at_mutation": True,
                "loaded_cursor_sha256": pre["cursor"],
                "loaded_equals_post": False,
                "loaded_equals_pre": True,
                "loaded_model_sha256": pre["model"],
                "loaded_optimizer_sha256": pre["optimizer"],
                "loaded_state_sha256": pre["state"],
                "load_target_before_model_sha256": digest("async-load-target"),
                "pillow_version": "12.3.0",
                "post_cursor_sha256": post["cursor"],
                "post_differs_from_pre": True,
                "post_model_sha256": post["model"],
                "post_optimizer_sha256": post["optimizer"],
                "post_state_sha256": post["state"],
                "pre_cursor_sha256": pre["cursor"],
                "pre_model_sha256": pre["model"],
                "pre_optimizer_sha256": pre["optimizer"],
                "pre_state_sha256": pre["state"],
                "rank": rank,
                "receipt_sha256": receipt,
                "receipt_verified_after_load": True,
                "receipt_verified_after_save": True,
                "report_schema": ASYNC_SNAPSHOT_REPORT_SCHEMA,
                "stage_call_count": 1,
                "stage_completed_before_mutation": True,
                "staged_model_sha256": pre["model"],
                "staged_optimizer_sha256": pre["optimizer"],
                "staged_state_sha256": pre["state"],
                "torch_version": TORCH_VERSION,
                "torchvision_distribution_version": "0.26.0+cpu",
                "torchvision_runtime_version": "0.26.0+cpu",
                "weights_downloaded": False,
                "workload_contract_sha256": workload_contract_digest(),
                "world_size": 2,
                "writer_gate_entered": True,
                "writer_gate_released": True,
            }
        )
    return {
        "checkpoint_id": ASYNC_CHECKPOINT_ID,
        "observation_schema": ASYNC_SNAPSHOT_OBSERVATION_SCHEMA,
        "promotion_pointer": pointer(receipt, ASYNC_CHECKPOINT_ID),
        "rank_reports": reports,
        "receipt_sha256": receipt,
        "receipt_verified_after_load": True,
        "receipt_verified_after_promotion": True,
        "scenario": ASYNC_SNAPSHOT_SCENARIO,
        "source_world_size": 2,
        "target_world_size": 2,
        "worker": worker_outcome(2),
        "workload": workload_contract(),
    }


def positive_common(
    *,
    scenario: str,
    source_world_size: int,
    target_world_size: int,
    schema: str,
    checkpoint_id: str,
) -> dict[str, object]:
    receipt = digest(f"{scenario}-receipt")
    return {
        "checkpoint_id": checkpoint_id,
        "load_worker": worker_outcome(target_world_size),
        "observation_schema": schema,
        "promotion_pointer": pointer(receipt, checkpoint_id),
        "receipt_sha256": receipt,
        "receipt_verified_after_load": True,
        "receipt_verified_after_promotion": True,
        "save_worker": worker_outcome(source_world_size),
        "scenario": scenario,
        "source_world_size": source_world_size,
        "target_world_size": target_world_size,
    }


def training_observation(
    scenario: str,
    source_world_size: int,
    target_world_size: int,
) -> dict[str, object]:
    return {
        **positive_common(
            scenario=scenario,
            source_world_size=source_world_size,
            target_world_size=target_world_size,
            schema=TRAINING_OBSERVATION_SCHEMA,
            checkpoint_id="checkpoint-one",
        ),
        "load_reports": training_reports(
            scenario=scenario,
            action="training-load-next",
            world_size=target_world_size,
        ),
        "save_reports": training_reports(
            scenario=scenario,
            action="training-save-baseline",
            world_size=source_world_size,
        ),
    }


def dtensor_reports(
    *,
    scenario: str,
    action: str,
    world_size: int,
) -> list[dict[str, object]]:
    reports = []
    for rank in range(world_size):
        report = common_report(action, rank, world_size)
        report.update(
            {
                "dtensor_global_sha256": digest(f"{scenario}-global"),
                "dtensor_global_shape": [4, 4],
                "dtensor_local_shape": [4 // world_size, 4],
            }
        )
        if action == "dtensor-save":
            report["receipt_verified_after_save"] = True
        else:
            report["receipt_verified_before_load"] = True
            report["receipt_verified_after_load"] = True
        reports.append(report)
    return reports


def dtensor_observation(
    scenario: str,
    source_world_size: int,
    target_world_size: int,
) -> dict[str, object]:
    return {
        **positive_common(
            scenario=scenario,
            source_world_size=source_world_size,
            target_world_size=target_world_size,
            schema=DTENSOR_OBSERVATION_SCHEMA,
            checkpoint_id="checkpoint-two",
        ),
        "load_reports": dtensor_reports(
            scenario=scenario,
            action="dtensor-load",
            world_size=target_world_size,
        ),
        "save_reports": dtensor_reports(
            scenario=scenario,
            action="dtensor-save",
            world_size=source_world_size,
        ),
    }


def elastic_observation() -> dict[str, object]:
    scenario = "elastic_restart_2_to_2"
    receipt = digest(f"{scenario}-receipt")
    promotion = pointer(receipt)
    marker = failure_marker_payload()
    marker_sha256 = hashlib.sha256((canonical_json(marker) + "\n").encode()).hexdigest()
    bootstrap = bootstrap_attestation_payload(
        torch_distribution_version=TORCH_DISTRIBUTION_VERSION,
        torch_version=TORCH_VERSION,
    )
    bootstrap_sha256 = hashlib.sha256(
        (canonical_json(bootstrap) + "\n").encode()
    ).hexdigest()
    elastic_reports = [
        {
            "elastic_report_schema": ELASTIC_REPORT_SCHEMA,
            "failure_marker_sha256": marker_sha256,
            "loopback_rendezvous": True,
            "max_restarts": 1,
            "rank": rank,
            "restart_count": 1,
            "shared_rendezvous_tcpstore_disabled": True,
            "world_size": 2,
        }
        for rank in range(2)
    ]
    return {
        "bootstrap": {**bootstrap, "attestation_sha256": bootstrap_sha256},
        "checkpoint_id": "checkpoint-one",
        "elastic_reports": elastic_reports,
        "failure": {**marker, "marker_sha256": marker_sha256},
        "launcher": {"exit_code": 0, "timed_out": False},
        "load_reports": training_reports(
            scenario=scenario,
            action="training-load-next",
            world_size=2,
        ),
        "max_restarts": 1,
        "observation_schema": ELASTIC_OBSERVATION_SCHEMA,
        "promotion_pointer_after": promotion,
        "promotion_pointer_before": promotion,
        "receipt_sha256_after_restart": receipt,
        "receipt_sha256_before_restart": receipt,
        "receipt_verified_after_load": True,
        "receipt_verified_after_promotion": True,
        "restart_count": 1,
        "save_reports": training_reports(
            scenario=scenario,
            action="training-save-baseline",
            world_size=2,
        ),
        "save_worker": worker_outcome(2),
        "scenario": scenario,
        "source_world_size": 2,
        "target_world_size": 2,
    }


def rank_exit_observation() -> dict[str, object]:
    prior = pointer(digest("prior-generation"))
    return {
        "candidate_preserved": True,
        "candidate_receipt_present": False,
        "checkpoint_id": "checkpoint-one",
        "exit_codes": [0, 91],
        "fault_code": "rank-exit",
        "load_attempted": False,
        "observation_schema": FAULT_OBSERVATION_SCHEMA,
        "promotion_attempted": False,
        "promotion_pointer_after": prior,
        "promotion_pointer_before": prior,
        "rank_reports": [],
        "receipt_rejected": False,
        "rejection_stage": "worker-supervision",
        "scenario": "rank_exit_no_promotion",
        "source_world_size": 2,
        "target_world_size": 2,
        "timed_out": False,
    }


def receipt_fault_observation(scenario: str) -> dict[str, object]:
    prior = pointer(digest("prior-generation"))
    missing = scenario != "corrupt_shard"
    subject = "metadata" if scenario == "missing_metadata" else "shard"
    operation = "remove" if missing else "flip-first-byte"
    before = digest(f"{scenario}-before")
    return {
        "candidate_preserved": True,
        "checkpoint_id": "checkpoint-one",
        "exit_codes": [0, 0],
        "fault_code": scenario.replace("_", "-"),
        "load_attempted": False,
        "mutation": {
            "after_present": not missing,
            "after_sha256": None if missing else digest(f"{scenario}-after"),
            "before_sha256": before,
            "operation": operation,
            "subject_kind": subject,
        },
        "observation_schema": FAULT_OBSERVATION_SCHEMA,
        "promotion_attempted": True,
        "promotion_pointer_after": prior,
        "promotion_pointer_before": prior,
        "receipt_rejected": True,
        "receipt_sha256": digest(f"{scenario}-receipt"),
        "rejection_stage": "receipt-before-load",
        "save_reports": training_reports(
            scenario=scenario,
            action="training-save-baseline",
            world_size=2,
        ),
        "scenario": scenario,
        "source_world_size": 2,
        "target_world_size": 2,
        "timed_out": False,
    }


def lineage_observation() -> dict[str, object]:
    seed = digest("lineage-seed")
    first = digest("lineage-first")
    second = digest("lineage-second")
    first_tree = digest("lineage-first-tree")
    second_tree = digest("lineage-second-tree")
    starting_pointer = pointer(seed)
    parent_pointer_sha256 = starting_pointer["pointer_sha256"]
    assert isinstance(parent_pointer_sha256, str)
    first_pointer = pointer(
        first,
        parent_pointer_sha256=parent_pointer_sha256,
        sequence=1,
    )
    second_pointer = pointer(
        second,
        parent_pointer_sha256=parent_pointer_sha256,
        sequence=1,
    )
    control = {
        "both_committed_before_publication": True,
        "committed_generation_count": 3,
        "final_generation_sha256": second,
        "final_pointer": second_pointer,
        "first_generation_sha256": first,
        "first_generation_tree_sha256_after": first_tree,
        "first_generation_tree_sha256_before": first_tree,
        "first_lineage_sha256": first_pointer["lineage_sha256"],
        "first_outcome": "published_unfenced",
        "first_pointer_sha256_after_publish": first_pointer["pointer_sha256"],
        "generation_bytes_unchanged": True,
        "publish_order": [0, 1],
        "reference_overwrite_observed": True,
        "second_generation_sha256": second,
        "second_generation_tree_sha256_after": second_tree,
        "second_generation_tree_sha256_before": second_tree,
        "second_lineage_sha256": second_pointer["lineage_sha256"],
        "second_outcome": "published_unfenced",
        "selected_ordinal": 1,
        "starting_pointer": starting_pointer,
        "stale_orphan_preserved": False,
        "stale_writer_rejected": False,
        "worker": worker_outcome(2),
    }
    protected = {
        "both_committed_before_publication": True,
        "committed_generation_count": 3,
        "final_generation_sha256": first,
        "final_pointer": first_pointer,
        "first_generation_sha256": first,
        "first_generation_tree_sha256_after": first_tree,
        "first_generation_tree_sha256_before": first_tree,
        "first_lineage_sha256": first_pointer["lineage_sha256"],
        "first_outcome": "published",
        "first_pointer_sha256_after_publish": first_pointer["pointer_sha256"],
        "generation_bytes_unchanged": True,
        "publish_order": [0, 1],
        "reference_overwrite_observed": False,
        "second_generation_sha256": second,
        "second_generation_tree_sha256_after": second_tree,
        "second_generation_tree_sha256_before": second_tree,
        "second_lineage_sha256": second_pointer["lineage_sha256"],
        "second_outcome": "stale_parent",
        "selected_ordinal": 0,
        "starting_pointer": starting_pointer,
        "stale_orphan_preserved": True,
        "stale_writer_rejected": True,
        "worker": worker_outcome(2),
    }
    recovery_after_commit = {
        "exit_code": 73,
        "generation_bytes_unchanged": True,
        "generation_sha256": first,
        "generation_tree_sha256_after": first_tree,
        "generation_tree_sha256_before": first_tree,
        "outcome_before_exit": "committed",
        "pointer_after_retry": first_pointer,
        "pointer_unchanged_on_retry": False,
        "recovery_outcome": "published",
        "starting_pointer": starting_pointer,
        "worker": {"exit_codes": [73], "timed_out": False},
    }
    recovery_after_publish = {
        "exit_code": 74,
        "generation_bytes_unchanged": True,
        "generation_sha256": first,
        "generation_tree_sha256_after": first_tree,
        "generation_tree_sha256_before": first_tree,
        "outcome_before_exit": "publication_return_lost",
        "pointer_after_retry": first_pointer,
        "pointer_unchanged_on_retry": True,
        "recovery_outcome": "already_published",
        "starting_pointer": starting_pointer,
        "worker": {"exit_codes": [74], "timed_out": False},
    }
    return {
        "control": control,
        "observation_schema": ("dcp-invariant-generation-lineage-observation-v1"),
        "protected": protected,
        "recovery_after_commit": recovery_after_commit,
        "recovery_after_publish": recovery_after_publish,
        "rejections": {
            "candidates_preserved": True,
            "forged_parent": "parent_version_invalid",
            "lineage_conflict": "generation_lineage_conflict",
            "pointers_unchanged": True,
            "sequence_mismatch": "parent_version_invalid",
        },
        "scenario": "generation_lineage_stale_writer_2p",
        "publisher_process_count": 2,
        "selected_head_count": 1,
    }


def complete_observations() -> dict[str, dict[str, object]]:
    return {
        "dtensor_1_to_2": dtensor_observation("dtensor_1_to_2", 1, 2),
        ASYNC_SNAPSHOT_SCENARIO: async_observation(),
        "dtensor_2_to_1": dtensor_observation("dtensor_2_to_1", 2, 1),
        "training_1_to_1": training_observation("training_1_to_1", 1, 1),
        "training_1_to_2": training_observation("training_1_to_2", 1, 2),
        "training_2_to_1": training_observation("training_2_to_1", 2, 1),
        "training_2_to_2": training_observation("training_2_to_2", 2, 2),
        "elastic_restart_2_to_2": elastic_observation(),
        "generation_lineage_stale_writer_2p": lineage_observation(),
        "rank_exit_no_promotion": rank_exit_observation(),
        "missing_metadata": receipt_fault_observation("missing_metadata"),
        "missing_shard": receipt_fault_observation("missing_shard"),
        "corrupt_shard": receipt_fault_observation("corrupt_shard"),
    }


def build(root: Path):
    return build_evidence_artifact(
        root,
        source_revision=SOURCE_REVISION,
        python_version=PYTHON_VERSION,
        torch_version=TORCH_VERSION,
        numpy_version=NUMPY_VERSION,
        observations=complete_observations(),
    )


def payload_paths() -> tuple[str, ...]:
    return tuple(
        sorted(
            [
                "junit.xml",
                "provenance.json",
                "summary.json",
                *[f"observations/{scenario}.json" for scenario in REGISTERED_SCENARIOS],
                *[f"results/{scenario}.json" for scenario in REGISTERED_SCENARIOS],
            ]
        )
    )


def reseal(root: Path) -> None:
    lines = []
    for relative in payload_paths():
        raw = root.joinpath(*relative.split("/")).read_bytes()
        lines.append(f"{hashlib.sha256(raw).hexdigest()}  {relative}")
    (root / MANIFEST_NAME).write_text(
        "\n".join(lines) + "\n",
        encoding="ascii",
        newline="\n",
    )


def rewrite_json(path: Path, value: object) -> None:
    path.write_text(
        canonical_json(value) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def test_round_trip_binds_observations_results_and_fixed_inventory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    created = build(root)
    verified = verify_evidence_artifact(root)
    assert created == verified
    assert set(verified.observations) == set(REGISTERED_SCENARIOS)
    assert verified.summary["async_snapshot_equalities"] == 1
    assert set(verified.results) == set(REGISTERED_SCENARIOS)
    assert verified.summary["passed_scenarios"] == 13
    assert verified.summary["state_equalities"] == 4
    assert verified.summary["elastic_recoveries"] == 1
    assert verified.summary["global_tensor_equalities"] == 2
    assert verified.summary["fault_rejections"] == 4
    assert verified.summary["promotion_allowed_scenarios"] == 9
    assert verified.summary["stale_writer_rejections"] == 1
    assert set(path.name for path in root.iterdir()) == {
        "junit.xml",
        MANIFEST_NAME,
        "observations",
        "provenance.json",
        "results",
        "summary.json",
    }


@pytest.mark.parametrize(
    ("field_path", "replacement"),
    [
        (("control", "selected_ordinal"), 0),
        (("control", "both_committed_before_publication"), False),
        (("control", "publish_order"), [1, 0]),
        (("control", "first_pointer_sha256_after_publish"), digest("forged")),
        (("protected", "second_outcome"), "published"),
        (("protected", "stale_orphan_preserved"), False),
        (
            ("protected", "second_generation_tree_sha256_after"),
            digest("changed-tree"),
        ),
        (("recovery_after_publish", "recovery_outcome"), "published"),
        (("rejections", "lineage_conflict"), "parent_version_invalid"),
    ],
)
def test_generation_lineage_semantic_tampering_is_rejected(
    field_path: tuple[str, str],
    replacement: object,
) -> None:
    observation = lineage_observation()
    section = observation[field_path[0]]
    assert isinstance(section, dict)
    section[field_path[1]] = replacement

    with pytest.raises(EvidenceArtifactError):
        normalize_observation(observation)


def test_generation_lineage_matched_tree_tampering_is_rejected() -> None:
    observation = lineage_observation()
    control = observation["control"]
    assert isinstance(control, dict)
    changed = digest("matched-tree-tamper")
    control["first_generation_tree_sha256_before"] = changed
    control["first_generation_tree_sha256_after"] = changed

    with pytest.raises(EvidenceArtifactError, match="matched generation trees"):
        normalize_observation(observation)


def test_generation_lineage_recovery_tree_tampering_is_rejected() -> None:
    observation = lineage_observation()
    recovery = observation["recovery_after_publish"]
    assert isinstance(recovery, dict)
    changed = digest("recovery-tree-tamper")
    recovery["generation_tree_sha256_before"] = changed
    recovery["generation_tree_sha256_after"] = changed

    with pytest.raises(EvidenceArtifactError, match="recovery tree"):
        normalize_observation(observation)


def test_results_are_derived_from_observations_not_caller_hashes(
    tmp_path: Path,
) -> None:
    signature = inspect.signature(build_evidence_artifact)
    assert "observations" in signature.parameters
    assert "results" not in signature.parameters
    verified = build(tmp_path / "evidence")
    result = verified.results["training_1_to_2"]
    assert result["checkpoint_state_sha256"] == result["loaded_state_sha256"]
    assert result["reference_state_sha256"] == result["resumed_state_sha256"]
    assert (
        result["observation_sha256"]
        == hashlib.sha256(
            canonical_json(verified.observations["training_1_to_2"]).encode()
        ).hexdigest()
    )
    elastic = verified.results["elastic_restart_2_to_2"]
    assert elastic["promotion_allowed"] is True
    assert elastic["committed_generation_reused"] is True
    assert elastic["post_failure_promotion_attempted"] is False
    assert elastic["torchrun_bootstrap_id"] == BOOTSTRAP_ID
    assert (
        elastic["bootstrap_attestation_sha256"]
        == (
            verified.observations["elastic_restart_2_to_2"]["bootstrap"][
                "attestation_sha256"
            ]
        )
    )


def test_async_result_is_derived_from_staged_loaded_and_mutated_states(
    tmp_path: Path,
) -> None:
    verified = build(tmp_path / "evidence")
    result = verified.results[ASYNC_SNAPSHOT_SCENARIO]

    assert result["staged_state_sha256"] == result["pre_snapshot_state_sha256"]
    assert result["loaded_state_sha256"] == result["pre_snapshot_state_sha256"]
    assert result["post_mutation_state_sha256"] != result["pre_snapshot_state_sha256"]
    assert result["promotion_allowed"] is True
    assert result["workload_contract_sha256"] == workload_contract_digest()


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("future_pending_at_mutation", False, "pending future"),
        ("stage_call_count", 0, "stage call"),
        ("staged_model_sha256", "b" * 64, "staged state"),
        ("loaded_model_sha256", "b" * 64, "loaded state"),
        ("post_optimizer_sha256", "b" * 64, "targeted mutation"),
        ("pillow_version", "12.2.0", "Pillow"),
        (
            "torchvision_runtime_version",
            "0.26.1+cpu",
            "torchvision",
        ),
        ("workload_contract_sha256", "b" * 64, "workload contract"),
    ],
)
def test_async_observation_rejects_false_gate_runtime_or_state_evidence(
    field: str,
    replacement: object,
    message: str,
) -> None:
    observation = async_observation()
    for report in observation["rank_reports"]:
        report[field] = replacement
    with pytest.raises(EvidenceArtifactError, match=message):
        normalize_observation(observation)


def test_async_observation_requires_targeted_model_mutation() -> None:
    observation = async_observation()
    for report in observation["rank_reports"]:
        report["post_model_sha256"] = report["pre_model_sha256"]
    with pytest.raises(EvidenceArtifactError, match="targeted mutation"):
        normalize_observation(observation)


def test_async_observation_rejects_rank_disagreement_and_extra_timing() -> None:
    observation = async_observation()
    observation["rank_reports"][1]["post_state_sha256"] = digest("other-post-state")
    with pytest.raises(EvidenceArtifactError, match="disagree"):
        normalize_observation(observation)

    observation = async_observation()
    observation["rank_reports"][0]["staging_elapsed_ns"] = 1
    with pytest.raises(EvidenceArtifactError, match="field set"):
        normalize_observation(observation)


def test_dtensor_result_names_global_tensor_not_shards(tmp_path: Path) -> None:
    result = build(tmp_path / "evidence").results["dtensor_1_to_2"]
    assert (
        result["reference_global_tensor_sha256"]
        == result["restored_global_tensor_sha256"]
    )
    assert not any("shard" in key for key in result)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["save_reports"].pop(),
            "report set",
        ),
        (
            lambda value: value["load_reports"][0].__setitem__(
                "receipt_verified_before_load", False
            ),
            "before load",
        ),
        (
            lambda value: value["load_reports"][0].__setitem__("world_size", 2),
            "world size",
        ),
        (
            lambda value: value["load_reports"][0].__setitem__(
                "next_state_sha256", "b" * 64
            ),
            "next state",
        ),
    ],
)
def test_training_observation_rejects_incomplete_or_unequal_rank_evidence(
    mutation,
    message: str,
) -> None:
    value = training_observation("training_1_to_1", 1, 1)
    mutation(value)
    with pytest.raises(EvidenceArtifactError, match=message):
        normalize_observation(value)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value.__setitem__("restart_count", 0),
            "restart count",
        ),
        (
            lambda value: value["elastic_reports"][0].__setitem__("restart_count", 0),
            "restart count",
        ),
        (
            lambda value: value["failure"].__setitem__("marker_sha256", "b" * 64),
            "marker digest",
        ),
        (
            lambda value: value["bootstrap"].__setitem__(
                "attestation_sha256", "b" * 64
            ),
            "bootstrap digest",
        ),
        (
            lambda value: value["bootstrap"].__setitem__(
                "shared_rendezvous_tcpstore_disabled", False
            ),
            "shared_rendezvous_tcpstore_disabled",
        ),
        (
            lambda value: value["elastic_reports"][0].__setitem__(
                "shared_rendezvous_tcpstore_disabled", False
            ),
            "shared rendezvous TCPStore opt-out",
        ),
        (
            lambda value: value["elastic_reports"][0].__setitem__(
                "loopback_rendezvous", False
            ),
            "loopback rendezvous",
        ),
        (
            lambda value: value.__setitem__(
                "promotion_pointer_after", pointer(digest("changed"))
            ),
            "pointer changed",
        ),
    ],
)
def test_elastic_observation_rejects_fabricated_restart_evidence(
    mutation,
    message: str,
) -> None:
    value = elastic_observation()
    mutation(value)
    with pytest.raises(EvidenceArtifactError, match=message):
        normalize_observation(value)


def test_elastic_observation_requires_exact_resumed_state() -> None:
    value = elastic_observation()
    for report in value["load_reports"]:
        report["next_state_sha256"] = "b" * 64
    with pytest.raises(EvidenceArtifactError, match="resumed next state"):
        normalize_observation(value)


def test_elastic_observation_requires_stable_receipt() -> None:
    value = elastic_observation()
    value["receipt_sha256_after_restart"] = "b" * 64
    with pytest.raises(EvidenceArtifactError, match="receipt changed"):
        normalize_observation(value)


def test_dtensor_observation_rejects_global_inequality() -> None:
    value = dtensor_observation("dtensor_1_to_2", 1, 2)
    value["load_reports"][0]["dtensor_global_sha256"] = "b" * 64
    value["load_reports"][1]["dtensor_global_sha256"] = "b" * 64
    with pytest.raises(EvidenceArtifactError, match="global tensors"):
        normalize_observation(value)


def test_promotion_pointer_must_bind_receipt_and_canonical_bytes() -> None:
    value = training_observation("training_1_to_2", 1, 2)
    value["promotion_pointer"]["generation"] = "b" * 64
    with pytest.raises(EvidenceArtifactError, match="lineage|digest"):
        normalize_observation(value)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("exit_codes", [1, 91], "exit vector"),
        ("timed_out", True, "timeout"),
        ("promotion_attempted", True, "promotion attempt"),
        ("candidate_preserved", False, "candidate preservation"),
    ],
)
def test_rank_exit_requires_exact_supervisor_observation(
    field: str,
    value: object,
    message: str,
) -> None:
    observation = rank_exit_observation()
    observation[field] = value
    with pytest.raises(EvidenceArtifactError, match=message):
        normalize_observation(observation)


def test_fault_rejection_requires_no_load_and_unchanged_pointer() -> None:
    observation = receipt_fault_observation("missing_shard")
    observation["load_attempted"] = True
    with pytest.raises(EvidenceArtifactError, match="load attempt"):
        normalize_observation(observation)

    observation = receipt_fault_observation("missing_shard")
    observation["promotion_pointer_after"] = pointer(digest("changed"))
    with pytest.raises(EvidenceArtifactError, match="pointer changed"):
        normalize_observation(observation)


def test_corruption_must_change_the_recorded_bytes() -> None:
    observation = receipt_fault_observation("corrupt_shard")
    observation["mutation"]["after_sha256"] = observation["mutation"]["before_sha256"]
    with pytest.raises(EvidenceArtifactError, match="did not change"):
        normalize_observation(observation)


def test_resealed_result_cannot_disagree_with_observation(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    build(root)
    target = root / "results" / "training_2_to_2.json"
    result = json.loads(target.read_text(encoding="utf-8"))
    result["reference_state_sha256"] = "b" * 64
    rewrite_json(target, result)
    reseal(root)
    with pytest.raises(EvidenceArtifactError, match="do not match"):
        verify_evidence_artifact(root)


def test_resealed_observation_semantic_tampering_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    build(root)
    target = root / "observations" / "training_2_to_1.json"
    observation = json.loads(target.read_text(encoding="utf-8"))
    observation["load_reports"][0]["next_model_sha256"] = "b" * 64
    rewrite_json(target, observation)
    reseal(root)
    with pytest.raises(EvidenceArtifactError, match="next model"):
        verify_evidence_artifact(root)


def test_builder_rejects_bootstrap_provenance_runtime_mismatch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    with pytest.raises(EvidenceArtifactError, match="PyTorch versions differ"):
        build_evidence_artifact(
            root,
            source_revision=SOURCE_REVISION,
            python_version=PYTHON_VERSION,
            torch_version="2.11.0",
            numpy_version=NUMPY_VERSION,
            observations=complete_observations(),
        )
    assert not root.exists()


def test_resealed_coordinated_unregistered_torch_pair_is_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "evidence"
    build(root)
    observation_path = root / "observations" / "elastic_restart_2_to_2.json"
    provenance_path = root / "provenance.json"
    observation = json.loads(observation_path.read_text(encoding="utf-8"))
    bootstrap = dict(observation["bootstrap"])
    bootstrap.pop("attestation_sha256")
    bootstrap["torch_distribution_version"] = "2.11.0"
    bootstrap["torch_version"] = "2.11.0"
    observation["bootstrap"] = {
        **bootstrap,
        "attestation_sha256": hashlib.sha256(
            (canonical_json(bootstrap) + "\n").encode()
        ).hexdigest(),
    }
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["runtime"]["torch_version"] = "2.11.0"
    rewrite_json(observation_path, observation)
    rewrite_json(provenance_path, provenance)
    reseal(root)
    with pytest.raises(
        EvidenceArtifactError,
        match="distribution/runtime pair is invalid",
    ):
        verify_evidence_artifact(root)


def test_payload_tampering_without_reseal_breaks_hash_closure(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    build(root)
    target = root / "summary.json"
    target.write_bytes(target.read_bytes().replace(b'"pass"', b'"fail"'))
    with pytest.raises(EvidenceArtifactError, match="digest mismatch"):
        verify_evidence_artifact(root)


def test_verifier_reads_each_payload_once_into_one_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "evidence"
    build(root)
    original = artifact_module._read_regular_bytes
    calls: dict[Path, int] = {}

    def counted(path: Path, maximum: int, label: str) -> bytes:
        calls[path] = calls.get(path, 0) + 1
        return original(path, maximum, label)

    monkeypatch.setattr(artifact_module, "_read_regular_bytes", counted)
    verify_evidence_artifact(root)
    assert all(count == 1 for count in calls.values())
    assert set(calls) == {
        root / MANIFEST_NAME,
        *[root.joinpath(*relative.split("/")) for relative in payload_paths()],
    }


def test_provenance_is_minimal_and_manifest_is_explicitly_unsigned(
    tmp_path: Path,
) -> None:
    verified = build(tmp_path / "evidence")
    provenance = verified.provenance
    assert provenance["runtime"] == {
        "implementation": "CPython",
        "numpy_version": NUMPY_VERSION,
        "python_version": PYTHON_VERSION,
        "torch_version": TORCH_VERSION,
    }
    assert provenance["manifest"]["authenticated"] is False
    text = canonical_json(provenance).lower()
    for forbidden in (
        "hostname",
        "username",
        "absolute_path",
        '"pid"',
        '"port"',
        "raw_tensor",
        "worker_log",
    ):
        assert forbidden not in text


def test_extra_native_checkpoint_file_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    build(root)
    (root / ".metadata").write_bytes(b"must never be published")
    with pytest.raises(EvidenceArtifactError, match="top-level inventory"):
        verify_evidence_artifact(root)


def test_absolute_path_cannot_enter_registered_observation() -> None:
    observation = training_observation("training_1_to_1", 1, 1)
    observation["checkpoint_id"] = "C:" + chr(92) + "Users" + chr(92) + "person"
    with pytest.raises(EvidenceArtifactError, match="checkpoint identifier"):
        normalize_observation(observation)


def test_builder_never_overwrites_existing_root(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    marker = root / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(EvidenceArtifactError, match="start absent"):
        build(root)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_symlinked_observation_is_rejected_when_supported(tmp_path: Path) -> None:
    root = tmp_path / "evidence"
    build(root)
    target = root / "observations" / "training_1_to_1.json"
    saved = tmp_path / "saved.json"
    target.replace(saved)
    try:
        target.symlink_to(saved)
    except OSError:
        saved.replace(target)
        pytest.skip("ordinary user cannot create symlinks on this platform")
    with pytest.raises(EvidenceArtifactError, match="ordinary file"):
        verify_evidence_artifact(root)
