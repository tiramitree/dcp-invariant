"""Run the registered DCP invariants without publishing native checkpoints.

The suite is deliberately split from :mod:`dcp_invariant.artifact`.  This
module may launch the pinned PyTorch worker, but it returns only normalized
observations.  Native DCP files and machine-specific launch details stay in an
isolated temporary directory that is removed before the evidence artifact is
created.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import re
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifact import (
    ELASTIC_OBSERVATION_SCHEMA,
    REGISTERED_SCENARIOS,
    VerifiedArtifact,
    build_evidence_artifact,
    normalize_observation,
    verify_evidence_artifact,
)
from .async_snapshot_contract import (
    ASYNC_CHECKPOINT_ID,
    ASYNC_GATE_DIRECTORY_NAME,
    ASYNC_REPORT_DIRECTORY_NAME,
    ASYNC_SNAPSHOT_ACTION,
    ASYNC_SNAPSHOT_OBSERVATION_SCHEMA,
    ASYNC_SNAPSHOT_REPORT_SCHEMA,
    ASYNC_SNAPSHOT_SCENARIO,
    ASYNC_WORLD_SIZE,
    is_registered_torchvision_version_pair,
    workload_contract,
    workload_contract_digest,
)
from .canonical import canonical_json, exact_json_equal, strict_json_loads
from .checkpoint_receipt import (
    RECEIPT_NAME,
    CheckpointReceiptError,
    verify_checkpoint,
)
from .elastic_contract import (
    BOOTSTRAP_ATTESTATION_NAME,
    CONTROL_REPORT_DIRECTORY_NAME,
    FAILURE_MARKER_NAME,
    LOAD_REPORT_DIRECTORY_NAME,
    REGISTERED_MAX_RESTARTS,
    REGISTERED_WORLD_SIZE,
    bootstrap_attestation_payload,
    failure_marker_payload,
    is_registered_torch_version_pair,
)
from .elastic_supervisor import ElasticResult, run_elastic_workers
from .supervisor import (
    LATEST_SCHEMA,
    SupervisorError,
    WorkerResult,
    promote_candidate,
    run_workers,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_WORKER_REPORT_SCHEMA = "dcp-invariant-worker-report-v1"
_REPORT_MAX_BYTES = 64 * 1024
_POINTER_MAX_BYTES = 1024
_CHECKPOINT_ONE = "checkpoint-one"
_CHECKPOINT_TWO = "checkpoint-two"
_ASYNC_SNAPSHOT_MODULE = "dcp_invariant.async_snapshot_worker"
_WORKER_MODULE = "dcp_invariant.worker"
_RANK_EXIT_MODULE = "dcp_invariant.rank_exit_worker"
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

_TRAINING_SAVE_FIELDS = _COMMON_REPORT_FIELDS | frozenset(
    {
        *(f"checkpoint_{name}_sha256" for name in _STATE_COMPONENTS),
        *(f"next_{name}_sha256" for name in _STATE_COMPONENTS),
        "receipt_verified_after_save",
    }
)
_TRAINING_LOAD_FIELDS = _COMMON_REPORT_FIELDS | frozenset(
    {
        *(f"loaded_{name}_sha256" for name in _STATE_COMPONENTS),
        *(f"next_{name}_sha256" for name in _STATE_COMPONENTS),
        "receipt_verified_after_load",
        "receipt_verified_before_load",
    }
)
_DTENSOR_SAVE_FIELDS = _COMMON_REPORT_FIELDS | frozenset(
    {
        "dtensor_global_sha256",
        "dtensor_global_shape",
        "dtensor_local_shape",
        "receipt_verified_after_save",
    }
)
_DTENSOR_LOAD_FIELDS = _COMMON_REPORT_FIELDS | frozenset(
    {
        "dtensor_global_sha256",
        "dtensor_global_shape",
        "dtensor_local_shape",
        "receipt_verified_after_load",
        "receipt_verified_before_load",
    }
)

RankExitRunner = Callable[[Path, Path, float], WorkerResult]
ElasticRunner = Callable[[Path, Path, Path, Path, Path, float], ElasticResult]


class SuiteError(RuntimeError):
    """A live scenario or normalized-observation invariant failed."""


@dataclass(frozen=True)
class SuiteRun:
    """A verified public artifact and its in-memory normalized observations."""

    artifact: VerifiedArtifact
    observations: dict[str, dict[str, Any]]
    native_work_cleaned: bool


@dataclass(frozen=True)
class _PromotionLayout:
    root: Path
    wrapper: Path
    reports: Path
    worker_home: Path


def _entry_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _is_reparse(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(attributes & marker)


def _ordinary_directory(path: Path, label: str) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as error:
        raise SuiteError(f"{label} cannot be inspected") from error
    if not stat.S_ISDIR(value.st_mode) or path.is_symlink() or _is_reparse(value):
        raise SuiteError(f"{label} is not an ordinary directory")
    return value


def _ensure_ordinary_output_parent(output_root: Path) -> None:
    """Create missing parents without following an existing link or reparse point."""

    absolute = Path(os.path.abspath(output_root))
    missing: list[Path] = []
    cursor = absolute.parent
    while not _entry_exists(cursor):
        missing.append(cursor)
        parent = cursor.parent
        if parent == cursor:
            raise SuiteError("output parent has no inspectable ancestor")
        cursor = parent
    _ordinary_directory(cursor, "existing output ancestor")
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except OSError as error:
            raise SuiteError("output parent cannot be created") from error
        _ordinary_directory(directory, "created output parent")


def _ordinary_file(path: Path, maximum: int, label: str) -> os.stat_result:
    try:
        value = path.lstat()
    except OSError as error:
        raise SuiteError(f"{label} cannot be inspected") from error
    if not stat.S_ISREG(value.st_mode) or path.is_symlink() or _is_reparse(value):
        raise SuiteError(f"{label} is not an ordinary file")
    if value.st_size < 0 or value.st_size > maximum:
        raise SuiteError(f"{label} exceeds its registered size bound")
    return value


def _read_regular_bytes(path: Path, maximum: int, label: str) -> bytes:
    before = _ordinary_file(path, maximum, label)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SuiteError(f"{label} cannot be read") from error
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
        raise SuiteError(f"{label} changed during read")
    return raw


def _read_canonical_object(path: Path, maximum: int, label: str) -> dict[str, Any]:
    raw = _read_regular_bytes(path, maximum, label)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SuiteError(f"{label} is not UTF-8") from error
    if not text.endswith("\n") or "\r" in text:
        raise SuiteError(f"{label} is not one canonical LF record")
    try:
        value = strict_json_loads(text[:-1])
    except (TypeError, ValueError) as error:
        raise SuiteError(f"{label} is not strict JSON") from error
    if type(value) is not dict or canonical_json(value) + "\n" != text:
        raise SuiteError(f"{label} is not one canonical object")
    return value


def _hash_regular_file(path: Path, maximum: int, label: str) -> str:
    return hashlib.sha256(_read_regular_bytes(path, maximum, label)).hexdigest()


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise SuiteError(f"{label} is not one lowercase SHA-256 digest")
    return value


def _require_exact_int(value: object, expected: int, label: str) -> None:
    if type(value) is not int or value != expected:
        raise SuiteError(f"{label} is invalid")


def _require_exact(value: object, expected: object, label: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise SuiteError(f"{label} is invalid")


def _report_fields(action: str) -> frozenset[str]:
    fields = {
        "training-save-baseline": _TRAINING_SAVE_FIELDS,
        "training-load-next": _TRAINING_LOAD_FIELDS,
        "dtensor-save": _DTENSOR_SAVE_FIELDS,
        "dtensor-load": _DTENSOR_LOAD_FIELDS,
    }
    try:
        return fields[action]
    except KeyError as error:
        raise SuiteError("worker action is not registered") from error


def _validate_report(
    value: dict[str, Any],
    *,
    action: str,
    rank: int,
    world_size: int,
) -> None:
    if set(value) != set(_report_fields(action)):
        raise SuiteError("worker report field set is invalid")
    _require_exact(value["action"], action, "worker action")
    _require_exact_int(value["rank"], rank, "worker rank")
    _require_exact_int(value["world_size"], world_size, "worker world size")
    _require_exact(
        value["report_schema"],
        _WORKER_REPORT_SCHEMA,
        "worker report schema",
    )
    _require_exact(value["bias_shape"], [2], "bias shape")
    _require_exact(value["model_shape"], [2, 3], "model shape")
    _require_exact(value["global_batch_shape"], [4, 3], "global batch shape")
    _require_exact(value["global_target_shape"], [4, 2], "global target shape")
    _require_sha256(value["state_contract_sha256"], "state contract")

    digest_fields = [
        name
        for name in value
        if name.endswith("_sha256") and name != "state_contract_sha256"
    ]
    for name in digest_fields:
        _require_sha256(value[name], name)

    if action.endswith("save-baseline") or action == "dtensor-save":
        _require_exact(
            value["receipt_verified_after_save"],
            True,
            "receipt verification after save",
        )
    else:
        _require_exact(
            value["receipt_verified_before_load"],
            True,
            "receipt verification before load",
        )
        _require_exact(
            value["receipt_verified_after_load"],
            True,
            "receipt verification after load",
        )

    if action.startswith("dtensor-"):
        _require_exact(value["dtensor_global_shape"], [4, 4], "DTensor shape")
        expected_local = [4 // world_size, 4]
        _require_exact(value["dtensor_local_shape"], expected_local, "local shape")


def _validate_rank_consensus(reports: Sequence[Mapping[str, Any]]) -> None:
    if not reports:
        raise SuiteError("worker report set is empty")
    reference = dict(reports[0])
    reference.pop("rank")
    for report in reports[1:]:
        candidate = dict(report)
        candidate.pop("rank")
        if not exact_json_equal(candidate, reference):
            raise SuiteError("worker ranks do not report one normalized state")


def _read_reports(
    root: Path,
    *,
    action: str,
    world_size: int,
) -> list[dict[str, Any]]:
    _ordinary_directory(root, "worker report directory")
    try:
        entries = {entry.name: entry for entry in root.iterdir()}
    except OSError as error:
        raise SuiteError("worker report directory cannot be enumerated") from error
    expected_names = {f"rank-{rank}.json" for rank in range(world_size)}
    if set(entries) != expected_names:
        raise SuiteError("worker report rank set is incomplete or contains extras")
    reports: list[dict[str, Any]] = []
    for rank in range(world_size):
        name = f"rank-{rank}.json"
        report = _read_canonical_object(
            entries[name],
            _REPORT_MAX_BYTES,
            "worker report",
        )
        _validate_report(
            report,
            action=action,
            rank=rank,
            world_size=world_size,
        )
        reports.append(report)
    _validate_rank_consensus(reports)
    return reports


_ASYNC_REPORT_FIELDS = frozenset(
    {
        "action",
        "async_checkpointer",
        "direct_loaded_model_sha256",
        "future_pending_at_mutation",
        "loaded_cursor_sha256",
        "loaded_equals_post",
        "loaded_equals_pre",
        "loaded_model_sha256",
        "loaded_optimizer_sha256",
        "loaded_state_sha256",
        "load_target_before_model_sha256",
        "pillow_version",
        "post_cursor_sha256",
        "post_differs_from_pre",
        "post_model_sha256",
        "post_optimizer_sha256",
        "post_state_sha256",
        "pre_cursor_sha256",
        "pre_model_sha256",
        "pre_optimizer_sha256",
        "pre_state_sha256",
        "rank",
        "receipt_sha256",
        "receipt_verified_after_load",
        "receipt_verified_after_save",
        "report_schema",
        "stage_call_count",
        "stage_completed_before_mutation",
        "staged_model_sha256",
        "staged_optimizer_sha256",
        "staged_state_sha256",
        "torch_version",
        "torchvision_distribution_version",
        "torchvision_runtime_version",
        "weights_downloaded",
        "workload_contract_sha256",
        "world_size",
        "writer_gate_entered",
        "writer_gate_released",
    }
)


def _validate_async_report(value: dict[str, Any], *, rank: int) -> None:
    if set(value) != set(_ASYNC_REPORT_FIELDS):
        raise SuiteError("async report field set is invalid")
    _require_exact(value["action"], ASYNC_SNAPSHOT_ACTION, "async action")
    _require_exact(value["async_checkpointer"], "thread", "async checkpointer")
    _require_exact_int(value["rank"], rank, "async rank")
    _require_exact_int(value["world_size"], ASYNC_WORLD_SIZE, "async world size")
    _require_exact(
        value["report_schema"],
        ASYNC_SNAPSHOT_REPORT_SCHEMA,
        "async report schema",
    )
    for field in _ASYNC_REPORT_FIELDS:
        if field.endswith("_sha256"):
            _require_sha256(value[field], field)
    for field in (
        "future_pending_at_mutation",
        "loaded_equals_pre",
        "post_differs_from_pre",
        "receipt_verified_after_load",
        "receipt_verified_after_save",
        "stage_completed_before_mutation",
        "writer_gate_entered",
        "writer_gate_released",
    ):
        _require_exact(value[field], True, field)
    _require_exact(value["loaded_equals_post"], False, "loaded post inequality")
    _require_exact(value["stage_call_count"], 1, "async stage call count")
    _require_exact(value["weights_downloaded"], False, "weight download boundary")
    _require_exact(value["pillow_version"], "12.3.0", "Pillow version")
    if type(value["torch_version"]) is not str or not re.fullmatch(
        r"2\.11\.0(?:\+cpu)?", value["torch_version"]
    ):
        raise SuiteError("async PyTorch version is invalid")
    if not is_registered_torchvision_version_pair(
        value["torchvision_distribution_version"],
        value["torchvision_runtime_version"],
    ):
        raise SuiteError("async torchvision pair is invalid")
    if value["workload_contract_sha256"] != workload_contract_digest():
        raise SuiteError("async workload contract differs")
    if (
        value["staged_model_sha256"] != value["pre_model_sha256"]
        or value["staged_optimizer_sha256"] != value["pre_optimizer_sha256"]
        or value["staged_state_sha256"] != value["pre_state_sha256"]
        or value["loaded_cursor_sha256"] != value["pre_cursor_sha256"]
        or value["loaded_model_sha256"] != value["pre_model_sha256"]
        or value["loaded_optimizer_sha256"] != value["pre_optimizer_sha256"]
        or value["loaded_state_sha256"] != value["pre_state_sha256"]
        or value["direct_loaded_model_sha256"] != value["pre_model_sha256"]
        or value["load_target_before_model_sha256"] == value["pre_model_sha256"]
        or value["post_cursor_sha256"] != value["pre_cursor_sha256"]
        or value["post_optimizer_sha256"] != value["pre_optimizer_sha256"]
        or value["post_model_sha256"] == value["pre_model_sha256"]
        or value["post_state_sha256"] == value["pre_state_sha256"]
    ):
        raise SuiteError("async state relation is invalid")


def _read_async_reports(root: Path) -> list[dict[str, Any]]:
    _ordinary_directory(root, "async report directory")
    try:
        entries = {entry.name: entry for entry in root.iterdir()}
    except OSError as error:
        raise SuiteError("async report directory cannot be enumerated") from error
    expected_names = {f"rank-{rank}.json" for rank in range(ASYNC_WORLD_SIZE)}
    if set(entries) != expected_names:
        raise SuiteError("async report rank set is incomplete or contains extras")
    reports: list[dict[str, Any]] = []
    for rank in range(ASYNC_WORLD_SIZE):
        report = _read_canonical_object(
            entries[f"rank-{rank}.json"],
            _REPORT_MAX_BYTES,
            "async worker report",
        )
        _validate_async_report(report, rank=rank)
        reports.append(report)
    reference = dict(reports[0])
    reference.pop("rank")
    for report in reports[1:]:
        candidate = dict(report)
        candidate.pop("rank")
        if not exact_json_equal(candidate, reference):
            raise SuiteError("async ranks do not report one normalized state")
    return reports


def _worker_observation(result: WorkerResult) -> dict[str, Any]:
    return {
        "exit_codes": list(result.exit_codes),
        "timed_out": result.timed_out,
    }


def _make_promotion_layout(root: Path) -> _PromotionLayout:
    root.mkdir()
    candidates = root / "candidates"
    committed = root / "committed"
    reports = root / "reports"
    worker_home = root / "worker-home"
    candidates.mkdir()
    committed.mkdir()
    reports.mkdir()
    worker_home.mkdir()
    wrapper = candidates / "candidate"
    wrapper.mkdir()
    return _PromotionLayout(
        root=root,
        wrapper=wrapper,
        reports=reports,
        worker_home=worker_home,
    )


def _run_action(
    *,
    action: str,
    checkpoint_id: str,
    world_size: int,
    cwd: Path,
    isolated_home: Path,
    report_root: Path,
    timeout_seconds: float,
) -> tuple[WorkerResult, list[dict[str, Any]]]:
    result = run_workers(
        module=_WORKER_MODULE,
        common_arguments=[
            "--action",
            action,
            "--checkpoint-id",
            checkpoint_id,
            "--report-dir",
            str(report_root),
        ],
        world_size=world_size,
        cwd=cwd,
        isolated_home=isolated_home,
        timeout_seconds=timeout_seconds,
    )
    reports = _read_reports(
        report_root,
        action=action,
        world_size=world_size,
    )
    return result, reports


def _verified_receipt(
    checkpoint: Path,
    *,
    checkpoint_id: str,
    expected_state_contract: str,
    expected_torch_version: str | None = None,
) -> tuple[dict[str, Any], str]:
    try:
        receipt = verify_checkpoint(checkpoint)
    except (CheckpointReceiptError, OSError) as error:
        raise SuiteError("checkpoint receipt verification failed") from error
    if receipt.get("logical_checkpoint_id") != checkpoint_id:
        raise SuiteError("checkpoint receipt logical identifier is invalid")
    if receipt.get("state_contract_sha256") != expected_state_contract:
        raise SuiteError("checkpoint receipt state contract is inconsistent")
    torch_version = receipt.get("torch_version")
    if type(torch_version) is not str:
        raise SuiteError("checkpoint receipt runtime version is invalid")
    if expected_torch_version is not None and torch_version != expected_torch_version:
        raise SuiteError("checkpoint receipt runtime version changed")
    receipt_sha256 = _hash_regular_file(
        checkpoint / RECEIPT_NAME,
        128 * 1024,
        "checkpoint receipt",
    )
    return receipt, receipt_sha256


def _promotion_verifier(
    *,
    checkpoint_id: str,
    state_contract_sha256: str,
    torch_version: str,
    receipt_sha256: str,
) -> Callable[[Path], bool]:
    def verify(checkpoint: Path) -> bool:
        try:
            receipt, observed_sha256 = _verified_receipt(
                checkpoint,
                checkpoint_id=checkpoint_id,
                expected_state_contract=state_contract_sha256,
                expected_torch_version=torch_version,
            )
        except SuiteError:
            return False
        return (
            observed_sha256 == receipt_sha256
            and receipt["state_contract_sha256"] == state_contract_sha256
        )

    return verify


def _read_pointer(root: Path) -> tuple[dict[str, Any], str]:
    path = root / "LATEST.json"
    value = _read_canonical_object(path, _POINTER_MAX_BYTES, "promotion pointer")
    if set(value) != {"generation", "pointer_schema"}:
        raise SuiteError("promotion pointer field set is invalid")
    _require_sha256(value["generation"], "promotion generation")
    _require_exact(value["pointer_schema"], LATEST_SCHEMA, "pointer schema")
    digest = _hash_regular_file(path, _POINTER_MAX_BYTES, "promotion pointer")
    return value, digest


def _read_normalized_pointer(root: Path) -> dict[str, Any]:
    pointer, digest = _read_pointer(root)
    return {**pointer, "pointer_sha256": digest}


def _write_seed_pointer(root: Path) -> dict[str, Any]:
    pointer = {
        "generation": "0" * 64,
        "pointer_schema": LATEST_SCHEMA,
    }
    raw = (canonical_json(pointer) + "\n").encode("utf-8")
    path = root / "LATEST.json"
    try:
        with path.open("xb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    except OSError as error:
        raise SuiteError("cannot create the registered old pointer") from error
    observed = _read_normalized_pointer(root)
    digest = observed.pop("pointer_sha256")
    if not exact_json_equal(observed, pointer):
        raise SuiteError("registered old pointer changed during creation")
    return {**pointer, "pointer_sha256": digest}


def _compare_training_reports(
    save_reports: Sequence[Mapping[str, Any]],
    load_reports: Sequence[Mapping[str, Any]],
) -> None:
    save = save_reports[0]
    load = load_reports[0]
    if save["state_contract_sha256"] != load["state_contract_sha256"]:
        raise SuiteError("training state contract changed across restart")
    for component in _STATE_COMPONENTS:
        if save[f"checkpoint_{component}_sha256"] != load[f"loaded_{component}_sha256"]:
            raise SuiteError("loaded checkpoint state is not exact")
        if save[f"next_{component}_sha256"] != load[f"next_{component}_sha256"]:
            raise SuiteError("resumed next training state is not exact")


def _compare_dtensor_reports(
    save_reports: Sequence[Mapping[str, Any]],
    load_reports: Sequence[Mapping[str, Any]],
) -> None:
    save = save_reports[0]
    load = load_reports[0]
    if save["state_contract_sha256"] != load["state_contract_sha256"]:
        raise SuiteError("DTensor state contract changed across restart")
    if save["dtensor_global_sha256"] != load["dtensor_global_sha256"]:
        raise SuiteError("restored global DTensor is not exact")
    if save["dtensor_global_shape"] != load["dtensor_global_shape"]:
        raise SuiteError("restored global DTensor shape changed")


def _run_positive_scenario(
    root: Path,
    *,
    scenario: str,
    source_world_size: int,
    target_world_size: int,
    checkpoint_id: str,
    save_action: str,
    load_action: str,
    observation_schema: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], str]:
    layout = _make_promotion_layout(root)
    save_result, save_reports = _run_action(
        action=save_action,
        checkpoint_id=checkpoint_id,
        world_size=source_world_size,
        cwd=layout.wrapper,
        isolated_home=layout.worker_home,
        report_root=layout.reports / "save",
        timeout_seconds=timeout_seconds,
    )
    state_contract = _require_sha256(
        save_reports[0]["state_contract_sha256"],
        "state contract",
    )
    candidate_checkpoint = layout.wrapper / checkpoint_id
    receipt, receipt_sha256 = _verified_receipt(
        candidate_checkpoint,
        checkpoint_id=checkpoint_id,
        expected_state_contract=state_contract,
    )
    torch_version = receipt["torch_version"]
    verifier = _promotion_verifier(
        checkpoint_id=checkpoint_id,
        state_contract_sha256=state_contract,
        torch_version=torch_version,
        receipt_sha256=receipt_sha256,
    )
    promoted_wrapper = promote_candidate(
        root=layout.root,
        candidate=layout.wrapper,
        logical_checkpoint_id=checkpoint_id,
        verify=verifier,
    )
    pointer, pointer_sha256 = _read_pointer(layout.root)
    if pointer["generation"] != receipt_sha256:
        raise SuiteError("promotion pointer is not bound to the receipt")
    normalized_pointer = {
        **pointer,
        "pointer_sha256": pointer_sha256,
    }

    promoted_checkpoint = promoted_wrapper / checkpoint_id
    post_promotion_receipt, post_promotion_sha256 = _verified_receipt(
        promoted_checkpoint,
        checkpoint_id=checkpoint_id,
        expected_state_contract=state_contract,
        expected_torch_version=torch_version,
    )
    if post_promotion_sha256 != receipt_sha256 or not exact_json_equal(
        post_promotion_receipt, receipt
    ):
        raise SuiteError("checkpoint changed during promotion")

    load_result, load_reports = _run_action(
        action=load_action,
        checkpoint_id=checkpoint_id,
        world_size=target_world_size,
        cwd=promoted_wrapper,
        isolated_home=layout.worker_home,
        report_root=layout.reports / "load",
        timeout_seconds=timeout_seconds,
    )
    post_load_receipt, post_load_sha256 = _verified_receipt(
        promoted_checkpoint,
        checkpoint_id=checkpoint_id,
        expected_state_contract=state_contract,
        expected_torch_version=torch_version,
    )
    if post_load_sha256 != receipt_sha256 or not exact_json_equal(
        post_load_receipt, receipt
    ):
        raise SuiteError("checkpoint changed during trusted load")

    if scenario.startswith("training_"):
        _compare_training_reports(save_reports, load_reports)
    else:
        _compare_dtensor_reports(save_reports, load_reports)

    observation = {
        "checkpoint_id": checkpoint_id,
        "load_reports": load_reports,
        "load_worker": _worker_observation(load_result),
        "observation_schema": observation_schema,
        "promotion_pointer": normalized_pointer,
        "receipt_sha256": receipt_sha256,
        "receipt_verified_after_load": True,
        "receipt_verified_after_promotion": True,
        "save_reports": save_reports,
        "save_worker": _worker_observation(save_result),
        "scenario": scenario,
        "source_world_size": source_world_size,
        "target_world_size": target_world_size,
    }
    return observation, torch_version


def _receipt_file_record(
    receipt: Mapping[str, Any],
    *,
    subject_kind: str,
) -> Mapping[str, Any]:
    files = receipt.get("files")
    if type(files) is not list:
        raise SuiteError("checkpoint receipt file list is invalid")
    if subject_kind == "metadata":
        matches = [item for item in files if item.get("name") == ".metadata"]
    else:
        matches = [
            item
            for item in files
            if type(item.get("name")) is str and item["name"].endswith(".distcp")
        ]
    if not matches:
        raise SuiteError("checkpoint receipt lacks the registered fault subject")
    record = matches[0]
    if type(record) is not dict:
        raise SuiteError("checkpoint receipt file record is invalid")
    _require_sha256(record.get("sha256"), "fault subject")
    return record


def _mutate_checkpoint(
    checkpoint: Path,
    *,
    receipt: Mapping[str, Any],
    fault_code: str,
) -> dict[str, Any]:
    if fault_code == "missing-metadata":
        subject_kind = "metadata"
        operation = "remove"
    elif fault_code == "missing-shard":
        subject_kind = "shard"
        operation = "remove"
    elif fault_code == "corrupt-shard":
        subject_kind = "shard"
        operation = "flip-first-byte"
    else:
        raise SuiteError("receipt fault is not registered")
    record = _receipt_file_record(receipt, subject_kind=subject_kind)
    name = record["name"]
    if type(name) is not str or Path(name).name != name:
        raise SuiteError("fault subject name is unsafe")
    target = checkpoint / name
    before_sha256 = _hash_regular_file(
        target,
        64 * 1024 * 1024,
        "fault subject",
    )
    if before_sha256 != record["sha256"]:
        raise SuiteError("fault subject changed before injection")

    if operation == "remove":
        try:
            target.unlink()
        except OSError as error:
            raise SuiteError(
                "cannot inject the registered missing-file fault"
            ) from error
        if _entry_exists(target):
            raise SuiteError("missing-file fault did not remove its subject")
        after_present = False
        after_sha256 = None
    else:
        try:
            with target.open("r+b") as output:
                first = output.read(1)
                if len(first) != 1:
                    raise SuiteError("registered shard is unexpectedly empty")
                output.seek(0)
                output.write(bytes([first[0] ^ 0x01]))
                output.flush()
                os.fsync(output.fileno())
        except OSError as error:
            raise SuiteError(
                "cannot inject the registered corrupt-shard fault"
            ) from error
        after_present = True
        after_sha256 = _hash_regular_file(
            target,
            64 * 1024 * 1024,
            "corrupted fault subject",
        )
        if after_sha256 == before_sha256:
            raise SuiteError("corrupt-shard fault did not change its subject")

    return {
        "after_present": after_present,
        "after_sha256": after_sha256,
        "before_sha256": before_sha256,
        "operation": operation,
        "subject_kind": subject_kind,
    }


def _run_receipt_fault(
    root: Path,
    *,
    scenario: str,
    fault_code: str,
    timeout_seconds: float,
) -> tuple[dict[str, Any], str]:
    layout = _make_promotion_layout(root)
    pointer_before = _write_seed_pointer(layout.root)
    save_result, save_reports = _run_action(
        action="training-save-baseline",
        checkpoint_id=_CHECKPOINT_ONE,
        world_size=2,
        cwd=layout.wrapper,
        isolated_home=layout.worker_home,
        report_root=layout.reports / "save",
        timeout_seconds=timeout_seconds,
    )
    state_contract = _require_sha256(
        save_reports[0]["state_contract_sha256"],
        "state contract",
    )
    checkpoint = layout.wrapper / _CHECKPOINT_ONE
    receipt, receipt_sha256 = _verified_receipt(
        checkpoint,
        checkpoint_id=_CHECKPOINT_ONE,
        expected_state_contract=state_contract,
    )
    torch_version = receipt["torch_version"]
    mutation = _mutate_checkpoint(
        checkpoint,
        receipt=receipt,
        fault_code=fault_code,
    )

    receipt_rejected = False
    try:
        verify_checkpoint(checkpoint)
    except CheckpointReceiptError:
        receipt_rejected = True
    if not receipt_rejected:
        raise SuiteError("faulted checkpoint did not fail receipt verification")

    verification_attempts = 0

    def reject_faulted_checkpoint(candidate_checkpoint: Path) -> bool:
        nonlocal verification_attempts
        verification_attempts += 1
        try:
            verify_checkpoint(candidate_checkpoint)
        except (CheckpointReceiptError, OSError):
            return False
        return True

    try:
        promote_candidate(
            root=layout.root,
            candidate=layout.wrapper,
            logical_checkpoint_id=_CHECKPOINT_ONE,
            verify=reject_faulted_checkpoint,
        )
    except SupervisorError:
        pass
    else:
        raise SuiteError("faulted checkpoint was promoted")
    if verification_attempts != 1:
        raise SuiteError("fault promotion did not stop at candidate verification")

    _ordinary_directory(layout.wrapper, "rejected candidate wrapper")
    _ordinary_directory(
        layout.wrapper / _CHECKPOINT_ONE,
        "rejected candidate checkpoint",
    )
    try:
        committed_entries = list((layout.root / "committed").iterdir())
    except OSError as error:
        raise SuiteError("committed generation directory cannot be read") from error
    if committed_entries:
        raise SuiteError("fault rejection created a committed generation")
    pointer_after = _read_normalized_pointer(layout.root)
    if not exact_json_equal(pointer_after, pointer_before):
        raise SuiteError("fault rejection changed the old promotion pointer")

    observation = {
        "candidate_preserved": True,
        "checkpoint_id": _CHECKPOINT_ONE,
        "exit_codes": list(save_result.exit_codes),
        "fault_code": fault_code,
        "load_attempted": False,
        "mutation": mutation,
        "observation_schema": "dcp-invariant-fault-observation-v1",
        "promotion_attempted": True,
        "promotion_pointer_after": pointer_after,
        "promotion_pointer_before": pointer_before,
        "receipt_rejected": True,
        "receipt_sha256": receipt_sha256,
        "rejection_stage": "receipt-before-load",
        "save_reports": save_reports,
        "scenario": scenario,
        "source_world_size": 2,
        "target_world_size": 2,
        "timed_out": save_result.timed_out,
    }
    return observation, torch_version


def default_rank_exit_runner(
    wrapper: Path,
    report_root: Path,
    timeout_seconds: float,
) -> WorkerResult:
    """Run the separately registered rank-exit worker.

    The module is intentionally separate from the successful DCP worker.  A
    caller may inject the same three-argument contract for focused tests.
    """

    return run_workers(
        module=_RANK_EXIT_MODULE,
        common_arguments=[],
        world_size=2,
        cwd=wrapper,
        isolated_home=report_root.parent.parent / "worker-home",
        timeout_seconds=timeout_seconds,
        expected_exit_codes=(0, 91),
    )


def _empty_rank_report_inventory(root: Path) -> list[dict[str, Any]]:
    if not _entry_exists(root):
        return []
    _ordinary_directory(root, "rank-exit report directory")
    try:
        entries = list(root.iterdir())
    except OSError as error:
        raise SuiteError("rank-exit report directory cannot be enumerated") from error
    if entries:
        raise SuiteError("rank-exit scenario published an incomplete rank report")
    return []


def _run_rank_exit_fault(
    root: Path,
    *,
    timeout_seconds: float,
    runner: RankExitRunner,
) -> dict[str, Any]:
    layout = _make_promotion_layout(root)
    pointer_before = _write_seed_pointer(layout.root)
    report_root = layout.reports / "rank-exit"
    result = runner(layout.wrapper, report_root, timeout_seconds)
    if result.timed_out or result.exit_codes != (0, 91):
        raise SuiteError("rank-exit worker outcome is outside the registered vector")
    rank_reports = _empty_rank_report_inventory(report_root)

    _ordinary_directory(layout.wrapper, "rank-exit candidate wrapper")
    checkpoint = layout.wrapper / _CHECKPOINT_ONE
    receipt_path = checkpoint / RECEIPT_NAME
    candidate_receipt_present = _entry_exists(receipt_path)
    if candidate_receipt_present:
        raise SuiteError("rank-exit candidate unexpectedly contains a receipt")

    pointer_after = _read_normalized_pointer(layout.root)
    if not exact_json_equal(pointer_after, pointer_before):
        raise SuiteError("rank-exit worker changed the old promotion pointer")

    return {
        "candidate_preserved": True,
        "candidate_receipt_present": False,
        "checkpoint_id": _CHECKPOINT_ONE,
        "exit_codes": list(result.exit_codes),
        "fault_code": "rank-exit",
        "load_attempted": False,
        "observation_schema": "dcp-invariant-fault-observation-v1",
        "promotion_attempted": False,
        "promotion_pointer_after": pointer_after,
        "promotion_pointer_before": pointer_before,
        "rank_reports": rank_reports,
        "receipt_rejected": False,
        "rejection_stage": "worker-supervision",
        "scenario": "rank_exit_no_promotion",
        "source_world_size": 2,
        "target_world_size": 2,
        "timed_out": result.timed_out,
    }


def default_elastic_runner(
    cwd: Path,
    isolated_home: Path,
    load_report_dir: Path,
    control_report_dir: Path,
    failure_marker: Path,
    timeout_seconds: float,
) -> ElasticResult:
    return run_elastic_workers(
        cwd=cwd,
        isolated_home=isolated_home,
        load_report_dir=load_report_dir,
        control_report_dir=control_report_dir,
        failure_marker=failure_marker,
        timeout_seconds=timeout_seconds,
    )


def _read_elastic_control_reports(
    root: Path,
    *,
    failure_marker_sha256: str,
) -> list[dict[str, Any]]:
    _ordinary_directory(root, "elastic control report directory")
    try:
        entries = {entry.name: entry for entry in root.iterdir()}
    except OSError as error:
        raise SuiteError("elastic control report directory cannot be read") from error
    expected = {f"rank-{rank}.json" for rank in range(REGISTERED_WORLD_SIZE)}
    if set(entries) != expected:
        raise SuiteError("elastic control rank set is incomplete or contains extras")
    reports: list[dict[str, Any]] = []
    fields = {
        "elastic_report_schema",
        "failure_marker_sha256",
        "loopback_rendezvous",
        "max_restarts",
        "rank",
        "restart_count",
        "shared_rendezvous_tcpstore_disabled",
        "world_size",
    }
    for rank in range(REGISTERED_WORLD_SIZE):
        report = _read_canonical_object(
            entries[f"rank-{rank}.json"],
            _REPORT_MAX_BYTES,
            "elastic control report",
        )
        if set(report) != fields:
            raise SuiteError("elastic control report field set is invalid")
        _require_exact(
            report["elastic_report_schema"],
            "dcp-invariant-elastic-report-v2",
            "elastic report schema",
        )
        _require_exact_int(report["rank"], rank, "elastic rank")
        _require_exact(
            report["loopback_rendezvous"],
            True,
            "elastic loopback rendezvous",
        )
        _require_exact_int(
            report["world_size"],
            REGISTERED_WORLD_SIZE,
            "elastic world size",
        )
        _require_exact_int(
            report["restart_count"],
            1,
            "elastic restart count",
        )
        _require_exact(
            report["shared_rendezvous_tcpstore_disabled"],
            True,
            "elastic shared rendezvous TCPStore opt-out",
        )
        _require_exact_int(
            report["max_restarts"],
            REGISTERED_MAX_RESTARTS,
            "elastic max restarts",
        )
        _require_exact(
            report["failure_marker_sha256"],
            failure_marker_sha256,
            "elastic failure marker",
        )
        reports.append(report)
    reference = dict(reports[0])
    reference.pop("rank")
    candidate = dict(reports[1])
    candidate.pop("rank")
    if not exact_json_equal(reference, candidate):
        raise SuiteError("elastic ranks do not report one restart outcome")
    return reports


def _read_failure_marker(path: Path) -> tuple[dict[str, Any], str]:
    if path.name != FAILURE_MARKER_NAME:
        raise SuiteError("elastic failure marker name is invalid")
    marker = _read_canonical_object(
        path,
        _REPORT_MAX_BYTES,
        "elastic failure marker",
    )
    expected = failure_marker_payload()
    if not exact_json_equal(marker, expected):
        raise SuiteError("elastic failure marker is invalid")
    digest = _hash_regular_file(
        path,
        _REPORT_MAX_BYTES,
        "elastic failure marker",
    )
    return marker, digest


def _read_bootstrap_attestation(
    path: Path,
    *,
    torch_distribution_version: str,
    torch_version: str,
) -> dict[str, Any]:
    if path.name != BOOTSTRAP_ATTESTATION_NAME:
        raise SuiteError("torchrun bootstrap attestation name is invalid")
    attestation = _read_canonical_object(
        path,
        _REPORT_MAX_BYTES,
        "torchrun bootstrap attestation",
    )
    expected = bootstrap_attestation_payload(
        torch_distribution_version=torch_distribution_version,
        torch_version=torch_version,
    )
    if not exact_json_equal(attestation, expected):
        raise SuiteError("torchrun bootstrap attestation is invalid")
    digest = _hash_regular_file(
        path,
        _REPORT_MAX_BYTES,
        "torchrun bootstrap attestation",
    )
    return {**attestation, "attestation_sha256": digest}


def _run_elastic_scenario(
    root: Path,
    *,
    torch_distribution_version: str,
    timeout_seconds: float,
    runner: ElasticRunner,
) -> tuple[dict[str, Any], str]:
    scenario = "elastic_restart_2_to_2"
    layout = _make_promotion_layout(root)
    save_result, save_reports = _run_action(
        action="training-save-baseline",
        checkpoint_id=_CHECKPOINT_ONE,
        world_size=REGISTERED_WORLD_SIZE,
        cwd=layout.wrapper,
        isolated_home=layout.worker_home,
        report_root=layout.reports / "save",
        timeout_seconds=timeout_seconds,
    )
    state_contract = _require_sha256(
        save_reports[0]["state_contract_sha256"],
        "state contract",
    )
    candidate_checkpoint = layout.wrapper / _CHECKPOINT_ONE
    receipt, receipt_sha256 = _verified_receipt(
        candidate_checkpoint,
        checkpoint_id=_CHECKPOINT_ONE,
        expected_state_contract=state_contract,
    )
    torch_version = receipt["torch_version"]
    verifier = _promotion_verifier(
        checkpoint_id=_CHECKPOINT_ONE,
        state_contract_sha256=state_contract,
        torch_version=torch_version,
        receipt_sha256=receipt_sha256,
    )
    promoted_wrapper = promote_candidate(
        root=layout.root,
        candidate=layout.wrapper,
        logical_checkpoint_id=_CHECKPOINT_ONE,
        verify=verifier,
    )
    pointer_before = _read_normalized_pointer(layout.root)
    if pointer_before["generation"] != receipt_sha256:
        raise SuiteError("elastic promotion pointer is not bound to the receipt")

    load_report_dir = layout.reports / LOAD_REPORT_DIRECTORY_NAME
    control_report_dir = layout.reports / CONTROL_REPORT_DIRECTORY_NAME
    load_report_dir.mkdir()
    control_report_dir.mkdir()
    failure_marker = layout.root / FAILURE_MARKER_NAME
    elastic_result = runner(
        promoted_wrapper,
        layout.worker_home,
        load_report_dir,
        control_report_dir,
        failure_marker,
        timeout_seconds,
    )
    if (
        elastic_result.timed_out
        or elastic_result.exit_code != 0
        or elastic_result.tree_cleanup != "normal-agent-exit"
    ):
        raise SuiteError("elastic launcher did not complete by normal agent exit")

    bootstrap = _read_bootstrap_attestation(
        layout.root / BOOTSTRAP_ATTESTATION_NAME,
        torch_distribution_version=torch_distribution_version,
        torch_version=torch_version,
    )
    marker, marker_sha256 = _read_failure_marker(failure_marker)
    load_reports = _read_reports(
        load_report_dir,
        action="training-load-next",
        world_size=REGISTERED_WORLD_SIZE,
    )
    elastic_reports = _read_elastic_control_reports(
        control_report_dir,
        failure_marker_sha256=marker_sha256,
    )
    _compare_training_reports(save_reports, load_reports)

    promoted_checkpoint = promoted_wrapper / _CHECKPOINT_ONE
    post_load_receipt, post_load_sha256 = _verified_receipt(
        promoted_checkpoint,
        checkpoint_id=_CHECKPOINT_ONE,
        expected_state_contract=state_contract,
        expected_torch_version=torch_version,
    )
    if post_load_sha256 != receipt_sha256 or not exact_json_equal(
        post_load_receipt, receipt
    ):
        raise SuiteError("checkpoint changed during elastic trusted load")
    pointer_after = _read_normalized_pointer(layout.root)
    if not exact_json_equal(pointer_before, pointer_after):
        raise SuiteError("promotion pointer changed during elastic restart")

    observation = {
        "bootstrap": bootstrap,
        "checkpoint_id": _CHECKPOINT_ONE,
        "elastic_reports": elastic_reports,
        "failure": {**marker, "marker_sha256": marker_sha256},
        "launcher": {
            "exit_code": elastic_result.exit_code,
            "timed_out": elastic_result.timed_out,
        },
        "load_reports": load_reports,
        "max_restarts": REGISTERED_MAX_RESTARTS,
        "observation_schema": ELASTIC_OBSERVATION_SCHEMA,
        "promotion_pointer_after": pointer_after,
        "promotion_pointer_before": pointer_before,
        "receipt_sha256_after_restart": post_load_sha256,
        "receipt_sha256_before_restart": receipt_sha256,
        "receipt_verified_after_load": True,
        "receipt_verified_after_promotion": True,
        "restart_count": 1,
        "save_reports": save_reports,
        "save_worker": _worker_observation(save_result),
        "scenario": scenario,
        "source_world_size": REGISTERED_WORLD_SIZE,
        "target_world_size": REGISTERED_WORLD_SIZE,
    }
    return observation, torch_version


def _run_async_snapshot_scenario(
    root: Path,
    *,
    timeout_seconds: float,
) -> tuple[dict[str, Any], str, str, str]:
    try:
        root.mkdir()
        candidates = root / "candidates"
        committed = root / "committed"
        gate = root / ASYNC_GATE_DIRECTORY_NAME
        reports_root = root / ASYNC_REPORT_DIRECTORY_NAME
        worker_home = root / "worker-home"
        candidates.mkdir()
        committed.mkdir()
        gate.mkdir()
        reports_root.mkdir()
        worker_home.mkdir()
        candidate = candidates / "candidate"
        candidate.mkdir()
    except OSError as error:
        raise SuiteError("cannot create async scenario layout") from error

    worker_result = run_workers(
        module=_ASYNC_SNAPSHOT_MODULE,
        common_arguments=[
            "--checkpoint-id",
            ASYNC_CHECKPOINT_ID,
            "--gate-dir",
            str(gate),
            "--report-dir",
            str(reports_root),
        ],
        world_size=ASYNC_WORLD_SIZE,
        cwd=candidate,
        isolated_home=worker_home,
        timeout_seconds=timeout_seconds,
    )
    reports = _read_async_reports(reports_root)
    torch_version = reports[0]["torch_version"]
    torchvision_distribution_version = reports[0]["torchvision_distribution_version"]
    torchvision_runtime_version = reports[0]["torchvision_runtime_version"]
    checkpoint = candidate / ASYNC_CHECKPOINT_ID
    receipt, receipt_sha256 = _verified_receipt(
        checkpoint,
        checkpoint_id=ASYNC_CHECKPOINT_ID,
        expected_state_contract=workload_contract_digest(),
        expected_torch_version=torch_version,
    )
    if (
        reports[0]["receipt_sha256"] != receipt_sha256
        or receipt["state_contract_sha256"] != workload_contract_digest()
    ):
        raise SuiteError("async receipt differs from worker evidence")

    committed_generation = promote_candidate(
        root=root,
        candidate=candidate,
        logical_checkpoint_id=ASYNC_CHECKPOINT_ID,
        verify=_promotion_verifier(
            checkpoint_id=ASYNC_CHECKPOINT_ID,
            state_contract_sha256=workload_contract_digest(),
            torch_version=torch_version,
            receipt_sha256=receipt_sha256,
        ),
    )
    committed_checkpoint = committed_generation / ASYNC_CHECKPOINT_ID
    _, committed_receipt_sha256 = _verified_receipt(
        committed_checkpoint,
        checkpoint_id=ASYNC_CHECKPOINT_ID,
        expected_state_contract=workload_contract_digest(),
        expected_torch_version=torch_version,
    )
    if committed_receipt_sha256 != receipt_sha256:
        raise SuiteError("async receipt changed during promotion")
    pointer = _read_normalized_pointer(root)
    if pointer["generation"] != receipt_sha256:
        raise SuiteError("async promotion pointer differs from receipt")

    observation = {
        "checkpoint_id": ASYNC_CHECKPOINT_ID,
        "observation_schema": ASYNC_SNAPSHOT_OBSERVATION_SCHEMA,
        "promotion_pointer": pointer,
        "rank_reports": reports,
        "receipt_sha256": receipt_sha256,
        "receipt_verified_after_load": True,
        "receipt_verified_after_promotion": True,
        "scenario": ASYNC_SNAPSHOT_SCENARIO,
        "source_world_size": ASYNC_WORLD_SIZE,
        "target_world_size": ASYNC_WORLD_SIZE,
        "worker": _worker_observation(worker_result),
        "workload": workload_contract(),
    }
    try:
        normalized, _ = normalize_observation(observation)
    except (TypeError, ValueError) as error:
        raise SuiteError("async observation failed normalization") from error
    if not exact_json_equal(normalized, observation):
        raise SuiteError("async observation changed during normalization")
    return (
        observation,
        torch_version,
        torchvision_distribution_version,
        torchvision_runtime_version,
    )


def _validate_runtime_versions(
    versions: set[str],
    *,
    torch_distribution_version: str,
) -> str:
    if len(versions) != 1:
        raise SuiteError("scenarios did not use one exact PyTorch runtime")
    torch_version = next(iter(versions))
    if not is_registered_torch_version_pair(
        torch_distribution_version,
        torch_version,
    ):
        raise SuiteError("worker PyTorch distribution/runtime pair is not registered")
    return torch_version


def _validated_torch_distribution_version() -> str:
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError as error:
        raise SuiteError("registered PyTorch distribution is unavailable") from error
    if type(version) is not str or version not in {"2.11.0", "2.11.0+cpu"}:
        raise SuiteError("PyTorch distribution is outside the registered versions")
    return version


def _validated_torchvision_distribution_version() -> str:
    try:
        version = importlib.metadata.version("torchvision")
    except importlib.metadata.PackageNotFoundError as error:
        raise SuiteError(
            "registered torchvision distribution is unavailable"
        ) from error
    if version not in {"0.26.0", "0.26.0+cpu"}:
        raise SuiteError("torchvision distribution is outside the registered version")
    return version


def _validated_pillow_version() -> str:
    try:
        version = importlib.metadata.version("Pillow")
        import PIL
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise SuiteError("registered Pillow runtime is unavailable") from error
    if version != "12.3.0" or str(PIL.__version__) != version:
        raise SuiteError(
            "Pillow distribution/runtime pair is outside the registered version"
        )
    return version


def _validated_numpy_version() -> str:
    try:
        version = importlib.metadata.version("numpy")
    except importlib.metadata.PackageNotFoundError as error:
        raise SuiteError("registered NumPy runtime is unavailable") from error
    if version != "2.4.6":
        raise SuiteError("NumPy runtime is outside the registered version")
    return version


def _validate_all_state_contracts(
    observations: Mapping[str, Mapping[str, Any]],
) -> None:
    contracts: set[str] = set()
    for observation in observations.values():
        for key in ("save_reports", "load_reports", "rank_reports"):
            reports = observation.get(key, [])
            if type(reports) is not list:
                raise SuiteError("normalized report set is invalid")
            for report in reports:
                if type(report) is not dict:
                    raise SuiteError("normalized worker report is invalid")
                contract = report.get("state_contract_sha256")
                if contract is not None:
                    contracts.add(_require_sha256(contract, "state contract"))
    if len(contracts) != 1:
        raise SuiteError("scenarios did not use one exact state contract")


def _validate_observation_registry(
    observations: Mapping[str, Mapping[str, Any]],
) -> None:
    if set(observations) != set(REGISTERED_SCENARIOS):
        raise SuiteError("normalized observation registry is incomplete")
    for scenario in REGISTERED_SCENARIOS:
        try:
            normalized, _ = normalize_observation(observations[scenario])
        except (TypeError, ValueError) as error:
            raise SuiteError(
                "normalized observation failed the artifact contract"
            ) from error
        if not exact_json_equal(normalized, observations[scenario]):
            raise SuiteError("normalized observation changed during validation")


def run_suite(
    output_root: Path,
    *,
    source_revision: str,
    timeout_seconds: float = 180.0,
    rank_exit_runner: RankExitRunner | None = None,
    elastic_runner: ElasticRunner | None = None,
) -> SuiteRun:
    """Run all twelve registered scenarios and create one offline artifact.

    ``output_root`` must start absent.  No public file is created until the
    temporary native-checkpoint tree has been removed successfully.
    """

    if not isinstance(output_root, Path):
        raise SuiteError("output root must be a pathlib.Path")
    if _entry_exists(output_root):
        raise SuiteError("output root must start absent")
    if type(source_revision) is not str or not _SOURCE_REVISION.fullmatch(
        source_revision
    ):
        raise SuiteError("source revision must be one lowercase 40-hex value")
    if not (1.0 <= timeout_seconds <= 300.0):
        raise SuiteError("suite timeout is outside the registered bound")
    if rank_exit_runner is None:
        rank_exit_runner = default_rank_exit_runner
    if elastic_runner is None:
        elastic_runner = default_elastic_runner
    _ensure_ordinary_output_parent(output_root)

    observations: dict[str, dict[str, Any]] = {}
    torch_versions: set[str] = set()
    torch_distribution_version = _validated_torch_distribution_version()
    native_root: Path | None = None
    torchvision_distribution_version = _validated_torchvision_distribution_version()
    _validated_pillow_version()

    with tempfile.TemporaryDirectory(prefix="dcp-invariant-") as temporary:
        native_root = Path(temporary)
        _ordinary_directory(native_root, "native temporary root")

        (
            async_observation,
            async_torch_version,
            async_torchvision_distribution_version,
            async_torchvision_runtime_version,
        ) = _run_async_snapshot_scenario(
            native_root / ASYNC_SNAPSHOT_SCENARIO,
            timeout_seconds=timeout_seconds,
        )
        if (
            async_torchvision_distribution_version != torchvision_distribution_version
            or not is_registered_torchvision_version_pair(
                async_torchvision_distribution_version,
                async_torchvision_runtime_version,
            )
        ):
            raise SuiteError("async torchvision runtime changed across launch")
        observations[ASYNC_SNAPSHOT_SCENARIO] = async_observation
        torch_versions.add(async_torch_version)

        for source_world_size, target_world_size in (
            (1, 1),
            (1, 2),
            (2, 1),
            (2, 2),
        ):
            scenario = f"training_{source_world_size}_to_{target_world_size}"
            observation, torch_version = _run_positive_scenario(
                native_root / scenario,
                scenario=scenario,
                source_world_size=source_world_size,
                target_world_size=target_world_size,
                checkpoint_id=_CHECKPOINT_ONE,
                save_action="training-save-baseline",
                load_action="training-load-next",
                observation_schema="dcp-invariant-training-observation-v1",
                timeout_seconds=timeout_seconds,
            )
            observations[scenario] = observation
            torch_versions.add(torch_version)

        elastic_observation, elastic_torch_version = _run_elastic_scenario(
            native_root / "elastic_restart_2_to_2",
            torch_distribution_version=torch_distribution_version,
            timeout_seconds=timeout_seconds,
            runner=elastic_runner,
        )
        observations["elastic_restart_2_to_2"] = elastic_observation
        torch_versions.add(elastic_torch_version)

        for source_world_size, target_world_size in ((1, 2), (2, 1)):
            scenario = f"dtensor_{source_world_size}_to_{target_world_size}"
            observation, torch_version = _run_positive_scenario(
                native_root / scenario,
                scenario=scenario,
                source_world_size=source_world_size,
                target_world_size=target_world_size,
                checkpoint_id=_CHECKPOINT_TWO,
                save_action="dtensor-save",
                load_action="dtensor-load",
                observation_schema="dcp-invariant-dtensor-observation-v1",
                timeout_seconds=timeout_seconds,
            )
            observations[scenario] = observation
            torch_versions.add(torch_version)

        observations["rank_exit_no_promotion"] = _run_rank_exit_fault(
            native_root / "rank_exit_no_promotion",
            timeout_seconds=timeout_seconds,
            runner=rank_exit_runner,
        )

        for scenario, fault_code in (
            ("missing_metadata", "missing-metadata"),
            ("missing_shard", "missing-shard"),
            ("corrupt_shard", "corrupt-shard"),
        ):
            observation, torch_version = _run_receipt_fault(
                native_root / scenario,
                scenario=scenario,
                fault_code=fault_code,
                timeout_seconds=timeout_seconds,
            )
            observations[scenario] = observation
            torch_versions.add(torch_version)

        _validate_observation_registry(observations)
        _validate_all_state_contracts(observations)
        torch_version = _validate_runtime_versions(
            torch_versions,
            torch_distribution_version=torch_distribution_version,
        )
        numpy_version = _validated_numpy_version()

    native_work_cleaned = (
        native_root is not None
        and not native_root.exists()
        and not native_root.is_symlink()
    )
    if not native_work_cleaned:
        raise SuiteError("native temporary work was not removed")

    build_evidence_artifact(
        output_root,
        source_revision=source_revision,
        python_version=platform.python_version(),
        torch_version=torch_version,
        numpy_version=numpy_version,
        observations=observations,
    )
    artifact = verify_evidence_artifact(output_root)
    return SuiteRun(
        artifact=artifact,
        observations=observations,
        native_work_cleaned=True,
    )
