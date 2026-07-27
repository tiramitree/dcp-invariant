from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

import dcp_invariant.artifact as artifact_module
from dcp_invariant.artifact import (
    DTENSOR_OBSERVATION_SCHEMA,
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
from dcp_invariant.canonical import canonical_json

SOURCE_REVISION = "a" * 40
PYTHON_VERSION = "3.12.10"
TORCH_VERSION = "2.11.0+cpu"
STATE_CONTRACT = hashlib.sha256(b"state-contract").hexdigest()
COMPONENTS = ("cursor", "model", "optimizer", "rng", "state")


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def pointer(generation: str) -> dict[str, object]:
    value = {"generation": generation, "pointer_schema": LATEST_SCHEMA}
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
        "promotion_pointer": pointer(receipt),
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


def complete_observations() -> dict[str, dict[str, object]]:
    return {
        "dtensor_1_to_2": dtensor_observation("dtensor_1_to_2", 1, 2),
        "dtensor_2_to_1": dtensor_observation("dtensor_2_to_1", 2, 1),
        "training_1_to_1": training_observation("training_1_to_1", 1, 1),
        "training_1_to_2": training_observation("training_1_to_2", 1, 2),
        "training_2_to_1": training_observation("training_2_to_1", 2, 1),
        "training_2_to_2": training_observation("training_2_to_2", 2, 2),
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
    assert set(verified.results) == set(REGISTERED_SCENARIOS)
    assert verified.summary["passed_scenarios"] == 10
    assert verified.summary["state_equalities"] == 4
    assert verified.summary["global_tensor_equalities"] == 2
    assert verified.summary["fault_rejections"] == 4
    assert set(path.name for path in root.iterdir()) == {
        "junit.xml",
        MANIFEST_NAME,
        "observations",
        "provenance.json",
        "results",
        "summary.json",
    }


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


def test_dtensor_observation_rejects_global_inequality() -> None:
    value = dtensor_observation("dtensor_1_to_2", 1, 2)
    value["load_reports"][0]["dtensor_global_sha256"] = "b" * 64
    value["load_reports"][1]["dtensor_global_sha256"] = "b" * 64
    with pytest.raises(EvidenceArtifactError, match="global tensors"):
        normalize_observation(value)


def test_promotion_pointer_must_bind_receipt_and_canonical_bytes() -> None:
    value = training_observation("training_1_to_2", 1, 2)
    value["promotion_pointer"]["generation"] = "b" * 64
    with pytest.raises(EvidenceArtifactError, match="digest"):
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
