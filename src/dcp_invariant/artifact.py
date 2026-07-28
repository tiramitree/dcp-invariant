"""Normalized, fail-closed evidence artifacts for DCPInvariant.

The verifier consumes one immutable in-memory snapshot of every payload. The
manifest is intentionally unsigned: it closes the internal byte inventory but
does not authenticate a publisher or establish an external timestamp.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import canonical_json, exact_json_equal, sha256_json, strict_json_loads
from .elastic_contract import (
    BOOTSTRAP_ATTESTATION_SCHEMA,
    BOOTSTRAP_ID,
    ELASTIC_REPORT_SCHEMA,
    FAILURE_MARKER_SCHEMA,
    REGISTERED_MAX_RESTARTS,
    REGISTERED_WORLD_SIZE,
    bootstrap_attestation_payload,
    failure_marker_payload,
    is_registered_torch_version_pair,
)

ARTIFACT_SCHEMA = "dcp-invariant-evidence-v2"
RESULT_SCHEMA = "dcp-invariant-scenario-result-v2"
SUMMARY_SCHEMA = "dcp-invariant-summary-v2"
TRAINING_OBSERVATION_SCHEMA = "dcp-invariant-training-observation-v1"
DTENSOR_OBSERVATION_SCHEMA = "dcp-invariant-dtensor-observation-v1"
FAULT_OBSERVATION_SCHEMA = "dcp-invariant-fault-observation-v1"
ELASTIC_OBSERVATION_SCHEMA = "dcp-invariant-elastic-observation-v2"
WORKER_REPORT_SCHEMA = "dcp-invariant-worker-report-v1"
LATEST_SCHEMA = "dcp-invariant-latest-v1"
MANIFEST_NAME = "manifest.sha256"
MANIFEST_AUTHENTICATED = False

MAX_JSON_BYTES = 256 * 1024
MAX_JUNIT_BYTES = 64 * 1024
MAX_MANIFEST_BYTES = 32 * 1024

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_PYTHON_VERSION = re.compile(r"3\.(?:11|12|13)\.[0-9]+\Z")
_TORCH_VERSION = re.compile(r"2\.11\.0(?:\+cpu)?\Z")
_NUMPY_VERSION = re.compile(r"2\.4\.6\Z")
_MANIFEST_LINE = re.compile(r"([0-9a-f]{64})  ([a-z0-9_.\-/]+)\Z")
_WINDOWS_ABSOLUTE = re.compile(r"(?:\A|[\s\"'=])(?:[A-Za-z]:[\\/]|\\\\)")
_COMMON_POSIX_ABSOLUTE = re.compile(
    r"(?:\A|[\s\"'=])/(?:home|users|tmp|var|mnt|etc|root)(?:/|\Z)",
    re.IGNORECASE,
)
_FORBIDDEN_FIELD_PARTS = frozenset(
    {
        "contact",
        "email",
        "hostname",
        "password",
        "path",
        "phone",
        "pid",
        "port",
        "secret",
        "task",
        "token",
        "username",
    }
)
_FORBIDDEN_FIELDS = frozenset(
    {
        "host_name",
        "process_id",
        "raw_tensor",
        "raw_tensors",
        "tensor_values",
        "user_name",
        "worker_log",
    }
)
_STATE_COMPONENTS = ("cursor", "model", "optimizer", "rng", "state")
_COMMON_REPORT_FIELDS = frozenset(
    {
        "action",
        "bias_shape",
        "global_batch_shape",
        "global_target_shape",
        "model_shape",
        "rank",
        "report_schema",
        "state_contract_sha256",
        "world_size",
    }
)


class EvidenceArtifactError(ValueError):
    """An observation or artifact failed the registered evidence contract."""


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    category: str
    source_world_size: int
    target_world_size: int
    fault_code: str | None = None
    rejection_stage: str | None = None

    def registry_json(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "name": self.name,
            "source_world_size": self.source_world_size,
            "target_world_size": self.target_world_size,
        }


SCENARIO_SPECS = (
    ScenarioSpec("dtensor_1_to_2", "dtensor-exact-global-tensor", 1, 2),
    ScenarioSpec("dtensor_2_to_1", "dtensor-exact-global-tensor", 2, 1),
    ScenarioSpec("training_1_to_1", "training-exact-state", 1, 1),
    ScenarioSpec("training_1_to_2", "training-exact-state", 1, 2),
    ScenarioSpec("training_2_to_1", "training-exact-state", 2, 1),
    ScenarioSpec("training_2_to_2", "training-exact-state", 2, 2),
    ScenarioSpec(
        "elastic_restart_2_to_2",
        "elastic-restart-exact-state",
        2,
        2,
    ),
    ScenarioSpec(
        "rank_exit_no_promotion",
        "fault-rejection",
        2,
        2,
        "rank-exit",
        "worker-supervision",
    ),
    ScenarioSpec(
        "missing_metadata",
        "fault-rejection",
        2,
        2,
        "missing-metadata",
        "receipt-before-load",
    ),
    ScenarioSpec(
        "missing_shard",
        "fault-rejection",
        2,
        2,
        "missing-shard",
        "receipt-before-load",
    ),
    ScenarioSpec(
        "corrupt_shard",
        "fault-rejection",
        2,
        2,
        "corrupt-shard",
        "receipt-before-load",
    ),
)
REGISTERED_SCENARIOS = tuple(spec.name for spec in SCENARIO_SPECS)
_SPECS_BY_NAME = {spec.name: spec for spec in SCENARIO_SPECS}
_TRAINING_SCENARIOS = frozenset(
    spec.name for spec in SCENARIO_SPECS if spec.category == "training-exact-state"
)
_DTENSOR_SCENARIOS = frozenset(
    spec.name
    for spec in SCENARIO_SPECS
    if spec.category == "dtensor-exact-global-tensor"
)
_ELASTIC_SCENARIOS = frozenset(
    spec.name
    for spec in SCENARIO_SPECS
    if spec.category == "elastic-restart-exact-state"
)
_FAULT_SCENARIOS = frozenset(
    spec.name for spec in SCENARIO_SPECS if spec.category == "fault-rejection"
)


@dataclass(frozen=True)
class VerifiedArtifact:
    """Parsed data returned only after byte and semantic verification."""

    provenance: dict[str, Any]
    summary: dict[str, Any]
    observations: dict[str, dict[str, Any]]
    results: dict[str, dict[str, Any]]
    manifest_sha256: str


def _expect_exact_fields(
    value: object,
    expected: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise EvidenceArtifactError(f"{label} field set is invalid")
    return value


def _expect_exact(value: object, expected: object, field: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise EvidenceArtifactError(f"{field} is invalid")


def _require_sha256(value: object, field: str) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise EvidenceArtifactError(f"{field} must be one lowercase SHA-256 digest")
    return value


def _registered_spec(name: object) -> ScenarioSpec:
    if type(name) is not str or name not in _SPECS_BY_NAME:
        raise EvidenceArtifactError("scenario name is not registered")
    return _SPECS_BY_NAME[name]


def _validate_worker_outcome(
    value: object,
    *,
    world_size: int,
    label: str,
) -> dict[str, Any]:
    outcome = _expect_exact_fields(
        value,
        frozenset({"exit_codes", "timed_out"}),
        label,
    )
    _expect_exact(outcome["timed_out"], False, f"{label} timeout")
    expected_codes = [0] * world_size
    if (
        type(outcome["exit_codes"]) is not list
        or any(type(code) is not int for code in outcome["exit_codes"])
        or outcome["exit_codes"] != expected_codes
    ):
        raise EvidenceArtifactError(f"{label} exit vector is invalid")
    return outcome


def _common_report(
    value: object,
    *,
    action: str,
    rank: int,
    world_size: int,
    extra_fields: frozenset[str],
) -> dict[str, Any]:
    report = _expect_exact_fields(
        value,
        _COMMON_REPORT_FIELDS | extra_fields,
        f"{action} rank {rank} report",
    )
    _expect_exact(report["action"], action, "worker action")
    _expect_exact(report["bias_shape"], [2], "bias shape")
    _expect_exact(report["global_batch_shape"], [4, 3], "global batch shape")
    _expect_exact(report["global_target_shape"], [4, 2], "global target shape")
    _expect_exact(report["model_shape"], [2, 3], "model shape")
    _expect_exact(report["rank"], rank, "worker rank")
    _expect_exact(report["report_schema"], WORKER_REPORT_SCHEMA, "worker report schema")
    _require_sha256(report["state_contract_sha256"], "state contract")
    _expect_exact(report["world_size"], world_size, "worker world size")
    return report


def _state_digest_fields(prefix: str) -> frozenset[str]:
    return frozenset(f"{prefix}_{component}_sha256" for component in _STATE_COMPONENTS)


def _training_reports(
    value: object,
    *,
    action: str,
    world_size: int,
    digest_prefixes: tuple[str, str],
) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != world_size:
        raise EvidenceArtifactError(f"{action} report set is incomplete")
    receipt_fields = (
        frozenset({"receipt_verified_after_save"})
        if action == "training-save-baseline"
        else frozenset(
            {
                "receipt_verified_after_load",
                "receipt_verified_before_load",
            }
        )
    )
    extra = receipt_fields
    for prefix in digest_prefixes:
        extra |= _state_digest_fields(prefix)
    reports = [
        _common_report(
            report,
            action=action,
            rank=rank,
            world_size=world_size,
            extra_fields=extra,
        )
        for rank, report in enumerate(value)
    ]
    for report in reports:
        for prefix in digest_prefixes:
            for component in _STATE_COMPONENTS:
                _require_sha256(
                    report[f"{prefix}_{component}_sha256"],
                    f"{prefix} {component}",
                )
        if action == "training-save-baseline":
            _expect_exact(
                report["receipt_verified_after_save"],
                True,
                "receipt verification after save",
            )
        else:
            _expect_exact(
                report["receipt_verified_before_load"],
                True,
                "receipt verification before load",
            )
            _expect_exact(
                report["receipt_verified_after_load"],
                True,
                "receipt verification after load",
            )
    _require_report_consensus(reports, {"state_contract_sha256", *extra})
    return reports


def _dtensor_reports(
    value: object,
    *,
    action: str,
    world_size: int,
) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != world_size:
        raise EvidenceArtifactError(f"{action} report set is incomplete")
    receipt_fields = (
        frozenset({"receipt_verified_after_save"})
        if action == "dtensor-save"
        else frozenset(
            {
                "receipt_verified_after_load",
                "receipt_verified_before_load",
            }
        )
    )
    extra = (
        frozenset(
            {
                "dtensor_global_sha256",
                "dtensor_global_shape",
                "dtensor_local_shape",
            }
        )
        | receipt_fields
    )
    reports = [
        _common_report(
            report,
            action=action,
            rank=rank,
            world_size=world_size,
            extra_fields=extra,
        )
        for rank, report in enumerate(value)
    ]
    for report in reports:
        _require_sha256(report["dtensor_global_sha256"], "DTensor global tensor")
        _expect_exact(report["dtensor_global_shape"], [4, 4], "DTensor global shape")
        _expect_exact(
            report["dtensor_local_shape"],
            [4 // world_size, 4],
            "DTensor local shape",
        )
        if action == "dtensor-save":
            _expect_exact(
                report["receipt_verified_after_save"],
                True,
                "receipt verification after save",
            )
        else:
            _expect_exact(
                report["receipt_verified_before_load"],
                True,
                "receipt verification before load",
            )
            _expect_exact(
                report["receipt_verified_after_load"],
                True,
                "receipt verification after load",
            )
    _require_report_consensus(
        reports,
        {"state_contract_sha256", "dtensor_global_sha256", *receipt_fields},
    )
    return reports


def _require_report_consensus(
    reports: Sequence[Mapping[str, Any]],
    fields: set[str],
) -> None:
    for field in fields:
        values = [report[field] for report in reports]
        if not all(exact_json_equal(values[0], value) for value in values[1:]):
            raise EvidenceArtifactError(f"rank reports disagree on {field}")


def _pointer_digest(generation: str) -> str:
    return hashlib.sha256(
        (
            canonical_json(
                {
                    "generation": generation,
                    "pointer_schema": LATEST_SCHEMA,
                }
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()


def _validate_pointer(value: object, label: str) -> dict[str, Any]:
    pointer = _expect_exact_fields(
        value,
        frozenset({"generation", "pointer_schema", "pointer_sha256"}),
        label,
    )
    generation = _require_sha256(pointer["generation"], f"{label} generation")
    _expect_exact(pointer["pointer_schema"], LATEST_SCHEMA, f"{label} schema")
    pointer_sha256 = _require_sha256(pointer["pointer_sha256"], f"{label} digest")
    if pointer_sha256 != _pointer_digest(generation):
        raise EvidenceArtifactError(f"{label} digest does not match its fields")
    return pointer


def _validate_positive_common(
    observation: dict[str, Any],
    spec: ScenarioSpec,
    *,
    expected_schema: str,
    checkpoint_id: str,
) -> tuple[dict[str, Any], str]:
    _expect_exact(
        observation["observation_schema"],
        expected_schema,
        "observation schema",
    )
    _expect_exact(observation["scenario"], spec.name, "scenario")
    _expect_exact(
        observation["source_world_size"],
        spec.source_world_size,
        "source world size",
    )
    _expect_exact(
        observation["target_world_size"],
        spec.target_world_size,
        "target world size",
    )
    _expect_exact(observation["checkpoint_id"], checkpoint_id, "checkpoint identifier")
    receipt = _require_sha256(observation["receipt_sha256"], "checkpoint receipt")
    pointer = _validate_pointer(observation["promotion_pointer"], "promotion pointer")
    if pointer["generation"] != receipt:
        raise EvidenceArtifactError("promotion generation does not match receipt")
    _expect_exact(
        observation["receipt_verified_after_promotion"],
        True,
        "receipt verification after promotion",
    )
    _expect_exact(
        observation["receipt_verified_after_load"],
        True,
        "receipt verification after load",
    )
    return pointer, receipt


def _validate_training_observation(
    value: object,
    spec: ScenarioSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fields = frozenset(
        {
            "checkpoint_id",
            "load_reports",
            "load_worker",
            "observation_schema",
            "promotion_pointer",
            "receipt_sha256",
            "receipt_verified_after_load",
            "receipt_verified_after_promotion",
            "save_reports",
            "save_worker",
            "scenario",
            "source_world_size",
            "target_world_size",
        }
    )
    observation = _expect_exact_fields(value, fields, spec.name)
    _validate_positive_common(
        observation,
        spec,
        expected_schema=TRAINING_OBSERVATION_SCHEMA,
        checkpoint_id="checkpoint-one",
    )
    _validate_worker_outcome(
        observation["save_worker"],
        world_size=spec.source_world_size,
        label="save worker",
    )
    _validate_worker_outcome(
        observation["load_worker"],
        world_size=spec.target_world_size,
        label="load worker",
    )
    save_reports = _training_reports(
        observation["save_reports"],
        action="training-save-baseline",
        world_size=spec.source_world_size,
        digest_prefixes=("checkpoint", "next"),
    )
    load_reports = _training_reports(
        observation["load_reports"],
        action="training-load-next",
        world_size=spec.target_world_size,
        digest_prefixes=("loaded", "next"),
    )
    if (
        save_reports[0]["state_contract_sha256"]
        != load_reports[0]["state_contract_sha256"]
    ):
        raise EvidenceArtifactError("save and load state contracts differ")
    for component in _STATE_COMPONENTS:
        if (
            save_reports[0][f"checkpoint_{component}_sha256"]
            != load_reports[0][f"loaded_{component}_sha256"]
        ):
            raise EvidenceArtifactError(
                f"checkpoint and loaded {component} states differ"
            )
        if (
            save_reports[0][f"next_{component}_sha256"]
            != load_reports[0][f"next_{component}_sha256"]
        ):
            raise EvidenceArtifactError(
                f"baseline and resumed next {component} states differ"
            )
    normalized = strict_json_loads(canonical_json(observation))
    result = {
        "checkpoint_state_sha256": save_reports[0]["checkpoint_state_sha256"],
        "contract_status": "pass",
        "loaded_state_sha256": load_reports[0]["loaded_state_sha256"],
        "observation_sha256": sha256_json(normalized),
        "promotion_allowed": True,
        "reference_state_sha256": save_reports[0]["next_state_sha256"],
        "result_schema": RESULT_SCHEMA,
        "resumed_state_sha256": load_reports[0]["next_state_sha256"],
        "scenario": spec.name,
        "source_world_size": spec.source_world_size,
        "target_world_size": spec.target_world_size,
    }
    return normalized, result


def _validate_dtensor_observation(
    value: object,
    spec: ScenarioSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fields = frozenset(
        {
            "checkpoint_id",
            "load_reports",
            "load_worker",
            "observation_schema",
            "promotion_pointer",
            "receipt_sha256",
            "receipt_verified_after_load",
            "receipt_verified_after_promotion",
            "save_reports",
            "save_worker",
            "scenario",
            "source_world_size",
            "target_world_size",
        }
    )
    observation = _expect_exact_fields(value, fields, spec.name)
    _validate_positive_common(
        observation,
        spec,
        expected_schema=DTENSOR_OBSERVATION_SCHEMA,
        checkpoint_id="checkpoint-two",
    )
    _validate_worker_outcome(
        observation["save_worker"],
        world_size=spec.source_world_size,
        label="save worker",
    )
    _validate_worker_outcome(
        observation["load_worker"],
        world_size=spec.target_world_size,
        label="load worker",
    )
    save_reports = _dtensor_reports(
        observation["save_reports"],
        action="dtensor-save",
        world_size=spec.source_world_size,
    )
    load_reports = _dtensor_reports(
        observation["load_reports"],
        action="dtensor-load",
        world_size=spec.target_world_size,
    )
    reference = save_reports[0]["dtensor_global_sha256"]
    restored = load_reports[0]["dtensor_global_sha256"]
    if reference != restored:
        raise EvidenceArtifactError("DTensor global tensors are not exactly equal")
    if (
        save_reports[0]["state_contract_sha256"]
        != load_reports[0]["state_contract_sha256"]
    ):
        raise EvidenceArtifactError("save and load state contracts differ")
    normalized = strict_json_loads(canonical_json(observation))
    result = {
        "contract_status": "pass",
        "observation_sha256": sha256_json(normalized),
        "promotion_allowed": True,
        "reference_global_tensor_sha256": reference,
        "restored_global_tensor_sha256": restored,
        "result_schema": RESULT_SCHEMA,
        "scenario": spec.name,
        "source_world_size": spec.source_world_size,
        "target_world_size": spec.target_world_size,
    }
    return normalized, result


def _validate_elastic_bootstrap(value: object) -> tuple[dict[str, Any], str]:
    fields = frozenset(
        {
            "attestation_schema",
            "attestation_sha256",
            "backend_module",
            "bootstrap_id",
            "create_tcp_store_call_verified",
            "forced_use_libuv",
            "shared_rendezvous_tcpstore_disabled",
            "source_sha256",
            "tcpstore_created",
            "torch_distribution_version",
            "torch_version",
        }
    )
    bootstrap = _expect_exact_fields(value, fields, "torchrun bootstrap")
    torch_distribution_version = bootstrap["torch_distribution_version"]
    torch_version = bootstrap["torch_version"]
    if (
        type(torch_distribution_version) is not str
        or _TORCH_VERSION.fullmatch(torch_distribution_version) is None
    ):
        raise EvidenceArtifactError(
            "torchrun bootstrap distribution version is invalid"
        )
    if (
        type(torch_version) is not str
        or _TORCH_VERSION.fullmatch(torch_version) is None
    ):
        raise EvidenceArtifactError("torchrun bootstrap runtime is invalid")
    if not is_registered_torch_version_pair(
        torch_distribution_version,
        torch_version,
    ):
        raise EvidenceArtifactError(
            "torchrun bootstrap distribution/runtime pair is invalid"
        )
    expected = bootstrap_attestation_payload(
        torch_distribution_version=torch_distribution_version,
        torch_version=torch_version,
    )
    for field, expected_value in expected.items():
        _expect_exact(
            bootstrap[field],
            expected_value,
            f"torchrun bootstrap {field}",
        )
    _expect_exact(
        bootstrap["attestation_schema"],
        BOOTSTRAP_ATTESTATION_SCHEMA,
        "torchrun bootstrap schema",
    )
    digest = _require_sha256(
        bootstrap["attestation_sha256"],
        "torchrun bootstrap attestation",
    )
    raw = (canonical_json(expected) + "\n").encode("utf-8")
    if digest != hashlib.sha256(raw).hexdigest():
        raise EvidenceArtifactError("torchrun bootstrap digest is invalid")
    return bootstrap, digest


def _validate_elastic_failure(value: object) -> tuple[dict[str, Any], str]:
    fields = frozenset(
        {
            "injected_exit_code",
            "marker_schema",
            "marker_sha256",
            "injected_rank",
            "injection_restart_count",
            "world_size",
        }
    )
    failure = _expect_exact_fields(value, fields, "elastic failure")
    expected = failure_marker_payload()
    for field, expected_value in expected.items():
        _expect_exact(failure[field], expected_value, f"elastic failure {field}")
    _expect_exact(
        failure["marker_schema"],
        FAILURE_MARKER_SCHEMA,
        "elastic failure marker schema",
    )
    marker_sha256 = _require_sha256(
        failure["marker_sha256"],
        "elastic failure marker",
    )
    marker_raw = (canonical_json(expected) + "\n").encode("utf-8")
    if marker_sha256 != hashlib.sha256(marker_raw).hexdigest():
        raise EvidenceArtifactError("elastic failure marker digest is invalid")
    return failure, marker_sha256


def _validate_elastic_reports(
    value: object,
    *,
    failure_marker_sha256: str,
) -> list[dict[str, Any]]:
    if type(value) is not list or len(value) != REGISTERED_WORLD_SIZE:
        raise EvidenceArtifactError("elastic control report set is incomplete")
    fields = frozenset(
        {
            "elastic_report_schema",
            "failure_marker_sha256",
            "loopback_rendezvous",
            "max_restarts",
            "rank",
            "restart_count",
            "shared_rendezvous_tcpstore_disabled",
            "world_size",
        }
    )
    reports: list[dict[str, Any]] = []
    for rank, value_report in enumerate(value):
        report = _expect_exact_fields(
            value_report,
            fields,
            f"elastic rank {rank} report",
        )
        _expect_exact(
            report["elastic_report_schema"],
            ELASTIC_REPORT_SCHEMA,
            "elastic report schema",
        )
        _expect_exact(report["rank"], rank, "elastic rank")
        _expect_exact(
            report["loopback_rendezvous"],
            True,
            "elastic loopback rendezvous",
        )
        _expect_exact(
            report["world_size"],
            REGISTERED_WORLD_SIZE,
            "elastic world size",
        )
        _expect_exact(report["restart_count"], 1, "elastic restart count")
        _expect_exact(
            report["shared_rendezvous_tcpstore_disabled"],
            True,
            "elastic shared rendezvous TCPStore opt-out",
        )
        _expect_exact(
            report["max_restarts"],
            REGISTERED_MAX_RESTARTS,
            "elastic max restarts",
        )
        _expect_exact(
            report["failure_marker_sha256"],
            failure_marker_sha256,
            "elastic failure marker",
        )
        reports.append(report)
    _require_report_consensus(
        reports,
        {
            "elastic_report_schema",
            "failure_marker_sha256",
            "loopback_rendezvous",
            "max_restarts",
            "restart_count",
            "shared_rendezvous_tcpstore_disabled",
            "world_size",
        },
    )
    return reports


def _validate_elastic_observation(
    value: object,
    spec: ScenarioSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    fields = frozenset(
        {
            "bootstrap",
            "checkpoint_id",
            "elastic_reports",
            "failure",
            "launcher",
            "load_reports",
            "max_restarts",
            "observation_schema",
            "promotion_pointer_after",
            "promotion_pointer_before",
            "receipt_sha256_after_restart",
            "receipt_sha256_before_restart",
            "receipt_verified_after_load",
            "receipt_verified_after_promotion",
            "restart_count",
            "save_reports",
            "save_worker",
            "scenario",
            "source_world_size",
            "target_world_size",
        }
    )
    observation = _expect_exact_fields(value, fields, spec.name)
    _expect_exact(
        observation["observation_schema"],
        ELASTIC_OBSERVATION_SCHEMA,
        "elastic observation schema",
    )
    _expect_exact(observation["scenario"], spec.name, "scenario")
    _expect_exact(
        observation["source_world_size"],
        REGISTERED_WORLD_SIZE,
        "elastic source world size",
    )
    _expect_exact(
        observation["target_world_size"],
        REGISTERED_WORLD_SIZE,
        "elastic target world size",
    )
    _expect_exact(
        observation["checkpoint_id"],
        "checkpoint-one",
        "elastic checkpoint identifier",
    )
    _expect_exact(
        observation["max_restarts"],
        REGISTERED_MAX_RESTARTS,
        "elastic max restarts",
    )
    _expect_exact(observation["restart_count"], 1, "elastic restart count")
    launcher = _expect_exact_fields(
        observation["launcher"],
        frozenset({"exit_code", "timed_out"}),
        "elastic launcher",
    )
    _expect_exact(launcher["exit_code"], 0, "elastic launcher exit code")
    _expect_exact(launcher["timed_out"], False, "elastic launcher timeout")
    _validate_worker_outcome(
        observation["save_worker"],
        world_size=REGISTERED_WORLD_SIZE,
        label="elastic save worker",
    )

    receipt_before = _require_sha256(
        observation["receipt_sha256_before_restart"],
        "checkpoint receipt before restart",
    )
    receipt_after = _require_sha256(
        observation["receipt_sha256_after_restart"],
        "checkpoint receipt after restart",
    )
    if receipt_before != receipt_after:
        raise EvidenceArtifactError("checkpoint receipt changed during elastic restart")
    receipt = receipt_before
    pointer_before = _validate_pointer(
        observation["promotion_pointer_before"],
        "promotion pointer before elastic restart",
    )
    pointer_after = _validate_pointer(
        observation["promotion_pointer_after"],
        "promotion pointer after elastic restart",
    )
    if not exact_json_equal(pointer_before, pointer_after):
        raise EvidenceArtifactError("promotion pointer changed during elastic restart")
    if pointer_before["generation"] != receipt:
        raise EvidenceArtifactError(
            "elastic promotion generation does not match receipt"
        )
    _expect_exact(
        observation["receipt_verified_after_promotion"],
        True,
        "receipt verification after promotion",
    )
    _expect_exact(
        observation["receipt_verified_after_load"],
        True,
        "receipt verification after elastic load",
    )

    _, bootstrap_sha256 = _validate_elastic_bootstrap(observation["bootstrap"])
    _, marker_sha256 = _validate_elastic_failure(observation["failure"])
    _validate_elastic_reports(
        observation["elastic_reports"],
        failure_marker_sha256=marker_sha256,
    )
    save_reports = _training_reports(
        observation["save_reports"],
        action="training-save-baseline",
        world_size=REGISTERED_WORLD_SIZE,
        digest_prefixes=("checkpoint", "next"),
    )
    load_reports = _training_reports(
        observation["load_reports"],
        action="training-load-next",
        world_size=REGISTERED_WORLD_SIZE,
        digest_prefixes=("loaded", "next"),
    )
    if (
        save_reports[0]["state_contract_sha256"]
        != load_reports[0]["state_contract_sha256"]
    ):
        raise EvidenceArtifactError("elastic save and load state contracts differ")
    for component in _STATE_COMPONENTS:
        if (
            save_reports[0][f"checkpoint_{component}_sha256"]
            != load_reports[0][f"loaded_{component}_sha256"]
        ):
            raise EvidenceArtifactError(
                f"elastic checkpoint and loaded {component} states differ"
            )
        if (
            save_reports[0][f"next_{component}_sha256"]
            != load_reports[0][f"next_{component}_sha256"]
        ):
            raise EvidenceArtifactError(
                f"elastic baseline and resumed next {component} states differ"
            )

    normalized = strict_json_loads(canonical_json(observation))
    result = {
        "bootstrap_attestation_sha256": bootstrap_sha256,
        "checkpoint_state_sha256": save_reports[0]["checkpoint_state_sha256"],
        "committed_generation_reused": True,
        "contract_status": "pass",
        "failure_marker_sha256": marker_sha256,
        "loaded_state_sha256": load_reports[0]["loaded_state_sha256"],
        "max_restarts": REGISTERED_MAX_RESTARTS,
        "observation_sha256": sha256_json(normalized),
        "post_failure_promotion_attempted": False,
        "promotion_allowed": True,
        "receipt_sha256": receipt,
        "reference_state_sha256": save_reports[0]["next_state_sha256"],
        "restart_count": 1,
        "result_schema": RESULT_SCHEMA,
        "resumed_state_sha256": load_reports[0]["next_state_sha256"],
        "scenario": spec.name,
        "source_world_size": spec.source_world_size,
        "target_world_size": spec.target_world_size,
        "torchrun_bootstrap_id": BOOTSTRAP_ID,
    }
    return normalized, result


def _validate_fault_pointer_pair(observation: Mapping[str, Any]) -> None:
    before = _validate_pointer(
        observation["promotion_pointer_before"],
        "promotion pointer before fault",
    )
    after = _validate_pointer(
        observation["promotion_pointer_after"],
        "promotion pointer after fault",
    )
    if not exact_json_equal(before, after):
        raise EvidenceArtifactError("promotion pointer changed after rejected fault")


def _validate_mutation(value: object, spec: ScenarioSpec) -> dict[str, Any]:
    mutation = _expect_exact_fields(
        value,
        frozenset(
            {
                "after_present",
                "after_sha256",
                "before_sha256",
                "operation",
                "subject_kind",
            }
        ),
        "fault mutation",
    )
    before = _require_sha256(mutation["before_sha256"], "mutation before")
    expected_subject = "metadata" if spec.name == "missing_metadata" else "shard"
    expected_operation = "flip-first-byte" if spec.name == "corrupt_shard" else "remove"
    _expect_exact(mutation["subject_kind"], expected_subject, "mutation subject")
    _expect_exact(mutation["operation"], expected_operation, "mutation operation")
    if expected_operation == "remove":
        _expect_exact(mutation["after_present"], False, "mutation presence")
        _expect_exact(mutation["after_sha256"], None, "mutation after digest")
    else:
        _expect_exact(mutation["after_present"], True, "mutation presence")
        after = _require_sha256(mutation["after_sha256"], "mutation after")
        if after == before:
            raise EvidenceArtifactError("corrupt-shard mutation did not change bytes")
    return mutation


def _validate_fault_observation(
    value: object,
    spec: ScenarioSpec,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rank_exit_fields = frozenset(
        {
            "candidate_preserved",
            "candidate_receipt_present",
            "checkpoint_id",
            "exit_codes",
            "fault_code",
            "load_attempted",
            "observation_schema",
            "promotion_attempted",
            "promotion_pointer_after",
            "promotion_pointer_before",
            "rank_reports",
            "receipt_rejected",
            "rejection_stage",
            "scenario",
            "source_world_size",
            "target_world_size",
            "timed_out",
        }
    )
    receipt_fault_fields = frozenset(
        {
            "candidate_preserved",
            "checkpoint_id",
            "exit_codes",
            "fault_code",
            "load_attempted",
            "mutation",
            "observation_schema",
            "promotion_attempted",
            "promotion_pointer_after",
            "promotion_pointer_before",
            "receipt_rejected",
            "receipt_sha256",
            "rejection_stage",
            "save_reports",
            "scenario",
            "source_world_size",
            "target_world_size",
            "timed_out",
        }
    )
    fields = (
        rank_exit_fields
        if spec.name == "rank_exit_no_promotion"
        else receipt_fault_fields
    )
    observation = _expect_exact_fields(value, fields, spec.name)
    _expect_exact(
        observation["observation_schema"],
        FAULT_OBSERVATION_SCHEMA,
        "fault observation schema",
    )
    _expect_exact(observation["scenario"], spec.name, "scenario")
    _expect_exact(
        observation["source_world_size"],
        spec.source_world_size,
        "source world size",
    )
    _expect_exact(
        observation["target_world_size"],
        spec.target_world_size,
        "target world size",
    )
    _expect_exact(
        observation["checkpoint_id"], "checkpoint-one", "checkpoint identifier"
    )
    _expect_exact(observation["fault_code"], spec.fault_code, "fault code")
    _expect_exact(
        observation["rejection_stage"], spec.rejection_stage, "rejection stage"
    )
    _expect_exact(observation["timed_out"], False, "fault timeout")
    _expect_exact(observation["candidate_preserved"], True, "candidate preservation")
    _expect_exact(observation["load_attempted"], False, "load attempt")
    _validate_fault_pointer_pair(observation)

    if spec.name == "rank_exit_no_promotion":
        _expect_exact(observation["exit_codes"], [0, 91], "rank-exit vector")
        _expect_exact(observation["rank_reports"], [], "rank-exit reports")
        _expect_exact(
            observation["candidate_receipt_present"],
            False,
            "rank-exit receipt presence",
        )
        _expect_exact(observation["receipt_rejected"], False, "receipt rejection")
        _expect_exact(observation["promotion_attempted"], False, "promotion attempt")
    else:
        _expect_exact(observation["exit_codes"], [0, 0], "save exit vector")
        _expect_exact(observation["receipt_rejected"], True, "receipt rejection")
        _expect_exact(observation["promotion_attempted"], True, "promotion attempt")
        _require_sha256(observation["receipt_sha256"], "checkpoint receipt")
        reports = _training_reports(
            observation["save_reports"],
            action="training-save-baseline",
            world_size=2,
            digest_prefixes=("checkpoint", "next"),
        )
        if len(reports) != 2:
            raise EvidenceArtifactError("fault save report set is incomplete")
        _validate_mutation(observation["mutation"], spec)

    normalized = strict_json_loads(canonical_json(observation))
    result = {
        "contract_status": "pass",
        "fault_code": spec.fault_code,
        "observation_sha256": sha256_json(normalized),
        "observed_outcome": "rejected",
        "promotion_allowed": False,
        "rejection_evidence_sha256": sha256_json(normalized),
        "rejection_stage": spec.rejection_stage,
        "result_schema": RESULT_SCHEMA,
        "scenario": spec.name,
        "source_world_size": spec.source_world_size,
        "target_world_size": spec.target_world_size,
    }
    return normalized, result


def normalize_observation(
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate one normalized observation and derive its result."""

    if type(value) is not dict:
        raise EvidenceArtifactError("observation must be one JSON object")
    spec = _registered_spec(value.get("scenario"))
    if spec.name in _TRAINING_SCENARIOS:
        observation, result = _validate_training_observation(value, spec)
    elif spec.name in _DTENSOR_SCENARIOS:
        observation, result = _validate_dtensor_observation(value, spec)
    elif spec.name in _ELASTIC_SCENARIOS:
        observation, result = _validate_elastic_observation(value, spec)
    else:
        observation, result = _validate_fault_observation(value, spec)
    _assert_no_sensitive_shape(observation)
    _assert_no_sensitive_shape(result)
    return observation, result


def _validate_observations(
    value: object,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if type(value) is not dict or set(value) != set(REGISTERED_SCENARIOS):
        raise EvidenceArtifactError(
            "observation registry is incomplete or contains extras"
        )
    observations: dict[str, dict[str, Any]] = {}
    results: dict[str, dict[str, Any]] = {}
    for spec in SCENARIO_SPECS:
        observation, result = normalize_observation(value[spec.name])
        if observation["scenario"] != spec.name:
            raise EvidenceArtifactError(
                "observation registry key does not match scenario"
            )
        observations[spec.name] = observation
        results[spec.name] = result
    return observations, results


def _build_provenance(
    *,
    source_revision: str,
    python_version: str,
    torch_version: str,
    numpy_version: str,
) -> dict[str, Any]:
    if type(source_revision) is not str or not _SOURCE_REVISION.fullmatch(
        source_revision
    ):
        raise EvidenceArtifactError(
            "source revision must be one lowercase 40-hex identifier"
        )
    if type(python_version) is not str or not _PYTHON_VERSION.fullmatch(python_version):
        raise EvidenceArtifactError("Python version is outside the registered range")
    if type(torch_version) is not str or not _TORCH_VERSION.fullmatch(torch_version):
        raise EvidenceArtifactError("PyTorch version is outside the registered range")
    if type(numpy_version) is not str or not _NUMPY_VERSION.fullmatch(numpy_version):
        raise EvidenceArtifactError("NumPy version is outside the registered range")
    return {
        "artifact_schema": ARTIFACT_SCHEMA,
        "evidence_boundary": {
            "native_checkpoint_included": False,
            "normalized_observations_only": True,
        },
        "manifest": {
            "algorithm": "sha256",
            "authenticated": MANIFEST_AUTHENTICATED,
            "scope": "all-payload-files",
        },
        "runtime": {
            "implementation": "CPython",
            "numpy_version": numpy_version,
            "python_version": python_version,
            "torch_version": torch_version,
        },
        "scenario_registry": [spec.registry_json() for spec in SCENARIO_SPECS],
        "source_revision": source_revision,
    }


def _validate_provenance(value: object) -> dict[str, Any]:
    expected_fields = frozenset(
        {
            "artifact_schema",
            "evidence_boundary",
            "manifest",
            "runtime",
            "scenario_registry",
            "source_revision",
        }
    )
    provenance = _expect_exact_fields(value, expected_fields, "provenance")
    _expect_exact(provenance["artifact_schema"], ARTIFACT_SCHEMA, "artifact schema")
    boundary = _expect_exact_fields(
        provenance["evidence_boundary"],
        frozenset({"native_checkpoint_included", "normalized_observations_only"}),
        "evidence boundary",
    )
    _expect_exact(
        boundary["native_checkpoint_included"],
        False,
        "native checkpoint boundary",
    )
    _expect_exact(
        boundary["normalized_observations_only"],
        True,
        "normalized observation boundary",
    )
    manifest = _expect_exact_fields(
        provenance["manifest"],
        frozenset({"algorithm", "authenticated", "scope"}),
        "manifest declaration",
    )
    _expect_exact(manifest["algorithm"], "sha256", "manifest algorithm")
    _expect_exact(
        manifest["authenticated"],
        MANIFEST_AUTHENTICATED,
        "manifest authentication",
    )
    _expect_exact(manifest["scope"], "all-payload-files", "manifest scope")
    if type(provenance["source_revision"]) is not str or not _SOURCE_REVISION.fullmatch(
        provenance["source_revision"]
    ):
        raise EvidenceArtifactError("source revision is invalid")
    runtime = _expect_exact_fields(
        provenance["runtime"],
        frozenset(
            {
                "implementation",
                "numpy_version",
                "python_version",
                "torch_version",
            }
        ),
        "runtime",
    )
    _expect_exact(runtime["implementation"], "CPython", "runtime implementation")
    if type(runtime["python_version"]) is not str or not _PYTHON_VERSION.fullmatch(
        runtime["python_version"]
    ):
        raise EvidenceArtifactError("Python version is invalid")
    if type(runtime["torch_version"]) is not str or not _TORCH_VERSION.fullmatch(
        runtime["torch_version"]
    ):
        raise EvidenceArtifactError("PyTorch version is invalid")
    if type(runtime["numpy_version"]) is not str or not _NUMPY_VERSION.fullmatch(
        runtime["numpy_version"]
    ):
        raise EvidenceArtifactError("NumPy version is invalid")
    expected_registry = [spec.registry_json() for spec in SCENARIO_SPECS]
    if not exact_json_equal(provenance["scenario_registry"], expected_registry):
        raise EvidenceArtifactError("scenario registry is invalid")
    _assert_no_sensitive_shape(provenance)
    return provenance


def _validate_bootstrap_runtime_alignment(
    observations: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
) -> None:
    bootstrap_version = observations["elastic_restart_2_to_2"]["bootstrap"][
        "torch_version"
    ]
    provenance_version = provenance["runtime"]["torch_version"]
    if bootstrap_version != provenance_version:
        raise EvidenceArtifactError(
            "torchrun bootstrap and provenance PyTorch versions differ"
        )


def _build_summary(
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "artifact_schema": SUMMARY_SCHEMA,
        "elastic_recoveries": len(_ELASTIC_SCENARIOS),
        "fault_rejections": len(_FAULT_SCENARIOS),
        "global_tensor_equalities": len(_DTENSOR_SCENARIOS),
        "overall_status": "pass",
        "passed_scenarios": len(REGISTERED_SCENARIOS),
        "promotion_allowed_scenarios": sum(
            result["promotion_allowed"] is True for result in results.values()
        ),
        "scenario_count": len(REGISTERED_SCENARIOS),
        "scenarios": [
            {
                "contract_status": results[spec.name]["contract_status"],
                "promotion_allowed": results[spec.name]["promotion_allowed"],
                "scenario": spec.name,
            }
            for spec in SCENARIO_SPECS
        ],
        "state_equalities": len(_TRAINING_SCENARIOS),
    }


def _validate_summary(
    value: object,
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected = _build_summary(results)
    if not exact_json_equal(value, expected):
        raise EvidenceArtifactError("summary does not match registered results")
    _assert_no_sensitive_shape(value)
    return value


def _junit_bytes() -> bytes:
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            '<testsuite name="dcp-invariant-evidence" '
            f'tests="{len(REGISTERED_SCENARIOS)}" failures="0" errors="0" '
            'skipped="0">'
        ),
        *[
            f'  <testcase classname="dcp_invariant.evidence" name="{scenario}"/>'
            for scenario in REGISTERED_SCENARIOS
        ],
        "</testsuite>",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _assert_no_sensitive_shape(value: object) -> None:
    if type(value) is dict:
        for key, child in value.items():
            if type(key) is not str:
                raise EvidenceArtifactError("JSON object key must be text")
            lowered = key.lower()
            parts = set(lowered.split("_"))
            if lowered in _FORBIDDEN_FIELDS or parts.intersection(
                _FORBIDDEN_FIELD_PARTS
            ):
                raise EvidenceArtifactError("sensitive field is forbidden")
            _assert_no_sensitive_shape(child)
        return
    if type(value) is list:
        for child in value:
            _assert_no_sensitive_shape(child)
        return
    if type(value) is str:
        lowered = value.lower()
        if (
            "file://" in lowered
            or _WINDOWS_ABSOLUTE.search(value)
            or _COMMON_POSIX_ABSOLUTE.search(value)
        ):
            raise EvidenceArtifactError("absolute path is forbidden")


def _is_reparse(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & marker)


def _ordinary_directory(path: Path, label: str) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as error:
        raise EvidenceArtifactError(f"{label} cannot be inspected") from error
    if not stat.S_ISDIR(value.st_mode) or path.is_symlink() or _is_reparse(value):
        raise EvidenceArtifactError(f"{label} is not an ordinary directory")
    return value


def _ordinary_file(path: Path, maximum: int, label: str) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as error:
        raise EvidenceArtifactError(f"{label} cannot be inspected") from error
    if not stat.S_ISREG(value.st_mode) or path.is_symlink() or _is_reparse(value):
        raise EvidenceArtifactError(f"{label} is not an ordinary file")
    if value.st_size < 0 or value.st_size > maximum:
        raise EvidenceArtifactError(f"{label} exceeds its registered size bound")
    return value


def _read_regular_bytes(path: Path, maximum: int, label: str) -> bytes:
    before = _ordinary_file(path, maximum, label)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise EvidenceArtifactError(f"{label} cannot be read") from error
    after = _ordinary_file(path, maximum, label)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity or len(raw) != before.st_size:
        raise EvidenceArtifactError(f"{label} changed during read")
    return raw


def _parse_canonical_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise EvidenceArtifactError(f"{label} is not UTF-8") from error
    if not text.endswith("\n") or "\r" in text:
        raise EvidenceArtifactError(f"{label} must be one canonical LF record")
    try:
        parsed = strict_json_loads(text[:-1])
    except (TypeError, ValueError) as error:
        raise EvidenceArtifactError(f"{label} JSON is invalid") from error
    if type(parsed) is not dict:
        raise EvidenceArtifactError(f"{label} root must be an object")
    if canonical_json(parsed) + "\n" != text:
        raise EvidenceArtifactError(f"{label} JSON is not canonical")
    return parsed


def _payload_paths() -> tuple[str, ...]:
    paths = [
        "junit.xml",
        "provenance.json",
        "summary.json",
        *[f"observations/{scenario}.json" for scenario in REGISTERED_SCENARIOS],
        *[f"results/{scenario}.json" for scenario in REGISTERED_SCENARIOS],
    ]
    return tuple(sorted(paths))


def _path_from_relative(root: Path, relative: str) -> Path:
    return root.joinpath(*relative.split("/"))


def _maximum_for(relative: str) -> int:
    return MAX_JUNIT_BYTES if relative == "junit.xml" else MAX_JSON_BYTES


def _manifest_bytes(payloads: Mapping[str, bytes]) -> bytes:
    lines = [
        f"{hashlib.sha256(payloads[relative]).hexdigest()}  {relative}"
        for relative in _payload_paths()
    ]
    return ("\n".join(lines) + "\n").encode("ascii")


def _parse_manifest_bytes(
    raw: bytes,
    payloads: Mapping[str, bytes],
) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as error:
        raise EvidenceArtifactError("manifest is not ASCII") from error
    if not text.endswith("\n") or "\r" in text:
        raise EvidenceArtifactError("manifest must use canonical LF records")
    lines = text[:-1].split("\n")
    expected_paths = _payload_paths()
    if len(lines) != len(expected_paths):
        raise EvidenceArtifactError("manifest entry count is invalid")
    parsed: dict[str, str] = {}
    for line in lines:
        match = _MANIFEST_LINE.fullmatch(line)
        if match is None:
            raise EvidenceArtifactError("manifest line is invalid")
        digest, relative = match.groups()
        if relative in parsed:
            raise EvidenceArtifactError("manifest contains a duplicate path")
        parsed[relative] = digest
    if tuple(parsed) != expected_paths:
        raise EvidenceArtifactError("manifest inventory or order is invalid")
    if set(payloads) != set(expected_paths):
        raise EvidenceArtifactError("payload snapshot inventory is invalid")
    for relative, expected_digest in parsed.items():
        if hashlib.sha256(payloads[relative]).hexdigest() != expected_digest:
            raise EvidenceArtifactError("manifest payload digest mismatch")
    return parsed


def _validate_inventory(root: Path) -> None:
    _ordinary_directory(root, "artifact root")
    try:
        top_entries = {entry.name: entry for entry in root.iterdir()}
    except OSError as error:
        raise EvidenceArtifactError("artifact root cannot be enumerated") from error
    expected_top = {
        "junit.xml",
        MANIFEST_NAME,
        "observations",
        "provenance.json",
        "results",
        "summary.json",
    }
    if set(top_entries) != expected_top:
        raise EvidenceArtifactError("artifact top-level inventory is invalid")
    for directory_name in ("observations", "results"):
        _ordinary_directory(top_entries[directory_name], f"{directory_name} directory")
    for name, entry in top_entries.items():
        if name not in {"observations", "results"}:
            maximum = (
                MAX_JUNIT_BYTES
                if name == "junit.xml"
                else MAX_MANIFEST_BYTES
                if name == MANIFEST_NAME
                else MAX_JSON_BYTES
            )
            _ordinary_file(entry, maximum, name)
    expected_scenarios = {f"{scenario}.json" for scenario in REGISTERED_SCENARIOS}
    for directory_name in ("observations", "results"):
        directory = top_entries[directory_name]
        try:
            entries = {entry.name: entry for entry in directory.iterdir()}
        except OSError as error:
            raise EvidenceArtifactError(
                f"{directory_name} directory cannot be enumerated"
            ) from error
        if set(entries) != expected_scenarios:
            raise EvidenceArtifactError(f"{directory_name} file inventory is invalid")
        for name, entry in entries.items():
            _ordinary_file(entry, MAX_JSON_BYTES, f"{directory_name}/{name}")


def _snapshot_payloads(root: Path) -> tuple[dict[str, bytes], bytes]:
    payloads = {
        relative: _read_regular_bytes(
            _path_from_relative(root, relative),
            _maximum_for(relative),
            relative,
        )
        for relative in _payload_paths()
    }
    manifest = _read_regular_bytes(
        root / MANIFEST_NAME,
        MAX_MANIFEST_BYTES,
        MANIFEST_NAME,
    )
    return payloads, manifest


def _write_exclusive(path: Path, raw: bytes) -> None:
    try:
        with path.open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise EvidenceArtifactError(
            f"cannot create artifact file {path.name}"
        ) from error


def _json_bytes(value: object) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def build_evidence_artifact(
    root: Path,
    *,
    source_revision: str,
    python_version: str,
    torch_version: str,
    numpy_version: str,
    observations: Mapping[str, Mapping[str, Any]],
) -> VerifiedArtifact:
    """Build the exact v2 artifact from validated execution observations."""

    if not isinstance(root, Path):
        raise EvidenceArtifactError("artifact root must be a pathlib.Path")
    normalized_observations, results = _validate_observations(dict(observations))
    provenance = _build_provenance(
        source_revision=source_revision,
        python_version=python_version,
        torch_version=torch_version,
        numpy_version=numpy_version,
    )
    _validate_provenance(provenance)
    _validate_bootstrap_runtime_alignment(normalized_observations, provenance)
    summary = _build_summary(results)
    _validate_summary(summary, results)
    try:
        root.mkdir()
        observations_root = root / "observations"
        results_root = root / "results"
        observations_root.mkdir()
        results_root.mkdir()
    except OSError as error:
        raise EvidenceArtifactError(
            "artifact root must start absent and creatable"
        ) from error
    _write_exclusive(root / "provenance.json", _json_bytes(provenance))
    _write_exclusive(root / "summary.json", _json_bytes(summary))
    for scenario in REGISTERED_SCENARIOS:
        _write_exclusive(
            observations_root / f"{scenario}.json",
            _json_bytes(normalized_observations[scenario]),
        )
        _write_exclusive(
            results_root / f"{scenario}.json",
            _json_bytes(results[scenario]),
        )
    _write_exclusive(root / "junit.xml", _junit_bytes())
    payloads = {
        relative: _read_regular_bytes(
            _path_from_relative(root, relative),
            _maximum_for(relative),
            relative,
        )
        for relative in _payload_paths()
    }
    _write_exclusive(root / MANIFEST_NAME, _manifest_bytes(payloads))
    return verify_evidence_artifact(root)


def verify_evidence_artifact(root: Path) -> VerifiedArtifact:
    """Verify one fixed byte snapshot, then parse and validate that same snapshot."""

    if not isinstance(root, Path):
        raise EvidenceArtifactError("artifact root must be a pathlib.Path")
    _validate_inventory(root)
    payloads, manifest_raw = _snapshot_payloads(root)
    _parse_manifest_bytes(manifest_raw, payloads)
    provenance = _validate_provenance(
        _parse_canonical_json_bytes(payloads["provenance.json"], "provenance.json")
    )
    parsed_observations = {
        scenario: _parse_canonical_json_bytes(
            payloads[f"observations/{scenario}.json"],
            f"observations/{scenario}.json",
        )
        for scenario in REGISTERED_SCENARIOS
    }
    observations, derived_results = _validate_observations(parsed_observations)
    _validate_bootstrap_runtime_alignment(observations, provenance)
    parsed_results = {
        scenario: _parse_canonical_json_bytes(
            payloads[f"results/{scenario}.json"],
            f"results/{scenario}.json",
        )
        for scenario in REGISTERED_SCENARIOS
    }
    if not exact_json_equal(parsed_results, derived_results):
        raise EvidenceArtifactError(
            "published results do not match normalized observations"
        )
    summary = _validate_summary(
        _parse_canonical_json_bytes(payloads["summary.json"], "summary.json"),
        derived_results,
    )
    if payloads["junit.xml"] != _junit_bytes():
        raise EvidenceArtifactError(
            "JUnit cases do not exactly match registered passing results"
        )
    return VerifiedArtifact(
        provenance=provenance,
        summary=summary,
        observations=observations,
        results=derived_results,
        manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
    )
