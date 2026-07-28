from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from dcp_invariant.artifact import normalize_observation
from dcp_invariant.async_snapshot_contract import (
    ASYNC_SNAPSHOT_ACTION,
    ASYNC_SNAPSHOT_REPORT_SCHEMA,
    workload_contract_digest,
)
from dcp_invariant.canonical import canonical_json
from dcp_invariant.suite import (
    SuiteError,
    _ensure_ordinary_output_parent,
    _read_async_reports,
    _read_reports,
    _run_rank_exit_fault,
    _validate_async_report,
    _validate_rank_consensus,
    _validate_report,
    _validate_runtime_versions,
    _write_seed_pointer,
    run_suite,
)
from dcp_invariant.supervisor import LATEST_SCHEMA, WorkerResult

SHA256 = "a" * 64


@pytest.mark.parametrize(
    "torch_distribution_version",
    ["2.11.0", "2.11.0+cpu"],
)
def test_suite_accepts_only_registered_torch_pairs(
    torch_distribution_version: str,
) -> None:
    assert (
        _validate_runtime_versions(
            {"2.11.0+cpu"},
            torch_distribution_version=torch_distribution_version,
        )
        == "2.11.0+cpu"
    )


@pytest.mark.parametrize(
    ("torch_distribution_version", "torch_version"),
    [
        ("2.11.0", "2.11.0"),
        ("2.11.0+cpu", "2.11.0"),
        ("2.11.1+cpu", "2.11.1+cpu"),
    ],
)
def test_suite_rejects_inferred_or_unknown_torch_pairs(
    torch_distribution_version: str,
    torch_version: str,
) -> None:
    with pytest.raises(SuiteError, match="pair"):
        _validate_runtime_versions(
            {torch_version},
            torch_distribution_version=torch_distribution_version,
        )


def training_save_report(rank: int, world_size: int) -> dict[str, object]:
    report: dict[str, object] = {
        "action": "training-save-baseline",
        "bias_shape": [2],
        "global_batch_shape": [4, 3],
        "global_target_shape": [4, 2],
        "model_shape": [2, 3],
        "rank": rank,
        "receipt_verified_after_save": True,
        "report_schema": "dcp-invariant-worker-report-v1",
        "state_contract_sha256": SHA256,
        "world_size": world_size,
    }
    for prefix in ("checkpoint", "next"):
        for component in ("cursor", "model", "optimizer", "rng", "state"):
            report[f"{prefix}_{component}_sha256"] = SHA256
    return report


def write_report(path: Path, report: dict[str, object]) -> None:
    path.write_text(
        canonical_json(report) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def async_report(rank: int) -> dict[str, object]:
    pre_cursor = hashlib.sha256(b"pre-cursor").hexdigest()
    pre_model = hashlib.sha256(b"pre-model").hexdigest()
    pre_optimizer = hashlib.sha256(b"pre-optimizer").hexdigest()
    pre_state = hashlib.sha256(b"pre-state").hexdigest()
    return {
        "action": ASYNC_SNAPSHOT_ACTION,
        "async_checkpointer": "thread",
        "direct_loaded_model_sha256": pre_model,
        "future_pending_at_mutation": True,
        "loaded_cursor_sha256": pre_cursor,
        "loaded_equals_post": False,
        "loaded_equals_pre": True,
        "loaded_model_sha256": pre_model,
        "loaded_optimizer_sha256": pre_optimizer,
        "loaded_state_sha256": pre_state,
        "load_target_before_model_sha256": hashlib.sha256(b"load-target").hexdigest(),
        "pillow_version": "12.3.0",
        "post_cursor_sha256": pre_cursor,
        "post_differs_from_pre": True,
        "post_model_sha256": hashlib.sha256(b"post-model").hexdigest(),
        "post_optimizer_sha256": pre_optimizer,
        "post_state_sha256": hashlib.sha256(b"post-state").hexdigest(),
        "pre_cursor_sha256": pre_cursor,
        "pre_model_sha256": pre_model,
        "pre_optimizer_sha256": pre_optimizer,
        "pre_state_sha256": pre_state,
        "rank": rank,
        "receipt_sha256": hashlib.sha256(b"receipt").hexdigest(),
        "receipt_verified_after_load": True,
        "receipt_verified_after_save": True,
        "report_schema": ASYNC_SNAPSHOT_REPORT_SCHEMA,
        "stage_call_count": 1,
        "stage_completed_before_mutation": True,
        "staged_model_sha256": pre_model,
        "staged_optimizer_sha256": pre_optimizer,
        "staged_state_sha256": pre_state,
        "torch_version": "2.11.0+cpu",
        "torchvision_distribution_version": "0.26.0+cpu",
        "torchvision_runtime_version": "0.26.0+cpu",
        "weights_downloaded": False,
        "workload_contract_sha256": workload_contract_digest(),
        "world_size": 2,
        "writer_gate_entered": True,
        "writer_gate_released": True,
    }


def test_training_report_requires_exact_fields_and_receipt_gate() -> None:
    report = training_save_report(0, 1)
    _validate_report(
        report,
        action="training-save-baseline",
        rank=0,
        world_size=1,
    )

    report["receipt_verified_after_save"] = False
    with pytest.raises(SuiteError, match="receipt verification"):
        _validate_report(
            report,
            action="training-save-baseline",
            rank=0,
            world_size=1,
        )


def test_async_report_requires_stage_pending_gate_and_exact_relations() -> None:
    report = async_report(0)
    _validate_async_report(report, rank=0)

    report["future_pending_at_mutation"] = False
    with pytest.raises(SuiteError, match="pending"):
        _validate_async_report(report, rank=0)


def test_async_report_reader_requires_complete_consensus(tmp_path: Path) -> None:
    reports = tmp_path / "async-reports"
    reports.mkdir()
    write_report(reports / "rank-0.json", async_report(0))
    with pytest.raises(SuiteError, match="rank set"):
        _read_async_reports(reports)

    write_report(reports / "rank-1.json", async_report(1))
    parsed = _read_async_reports(reports)
    assert [report["rank"] for report in parsed] == [0, 1]

    (reports / "rank-1.json").unlink()
    changed = async_report(1)
    changed["post_model_sha256"] = hashlib.sha256(b"other-post").hexdigest()
    write_report(reports / "rank-1.json", changed)
    with pytest.raises(SuiteError, match="one normalized state"):
        _read_async_reports(reports)


def test_rank_consensus_ignores_only_rank() -> None:
    rank_zero = training_save_report(0, 2)
    rank_one = training_save_report(1, 2)
    _validate_rank_consensus([rank_zero, rank_one])

    rank_one["next_state_sha256"] = "b" * 64
    with pytest.raises(SuiteError, match="one normalized state"):
        _validate_rank_consensus([rank_zero, rank_one])


def test_report_reader_requires_complete_canonical_rank_set(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    reports.mkdir()
    write_report(reports / "rank-0.json", training_save_report(0, 2))
    with pytest.raises(SuiteError, match="rank set"):
        _read_reports(
            reports,
            action="training-save-baseline",
            world_size=2,
        )

    write_report(reports / "rank-1.json", training_save_report(1, 2))
    parsed = _read_reports(
        reports,
        action="training-save-baseline",
        world_size=2,
    )
    assert [report["rank"] for report in parsed] == [0, 1]


def test_seed_pointer_binds_canonical_pointer_bytes(tmp_path: Path) -> None:
    root = tmp_path / "promotion"
    root.mkdir()
    pointer = _write_seed_pointer(root)
    expected = {
        "generation": "0" * 64,
        "pointer_schema": LATEST_SCHEMA,
    }
    raw = (canonical_json(expected) + "\n").encode("utf-8")

    assert pointer == {
        **expected,
        "pointer_sha256": hashlib.sha256(raw).hexdigest(),
    }
    assert (root / "LATEST.json").read_bytes() == raw


def test_injected_rank_exit_preserves_candidate_and_pointer(
    tmp_path: Path,
) -> None:
    def runner(
        wrapper: Path,
        report_root: Path,
        timeout_seconds: float,
    ) -> WorkerResult:
        assert wrapper.is_dir()
        assert not report_root.exists()
        assert timeout_seconds == 10.0
        return WorkerResult((0, 91), False)

    observation = _run_rank_exit_fault(
        tmp_path / "rank-exit",
        timeout_seconds=10.0,
        runner=runner,
    )
    normalized, result = normalize_observation(observation)

    assert normalized == observation
    assert result["contract_status"] == "pass"
    assert observation["candidate_preserved"] is True
    assert observation["candidate_receipt_present"] is False
    assert (
        observation["promotion_pointer_before"]
        == observation["promotion_pointer_after"]
    )


def test_suite_rejects_subsecond_timeout_at_entry(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match="timeout"):
        run_suite(
            tmp_path / "evidence",
            source_revision="a" * 40,
            timeout_seconds=0.5,
        )


def test_output_root_must_start_absent(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(SuiteError, match="start absent"):
        run_suite(output, source_revision="a" * 40)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_missing_output_parents_are_created_without_creating_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "one" / "two" / "artifact"
    _ensure_ordinary_output_parent(output)

    assert output.parent.is_dir()
    assert not output.exists()


@pytest.mark.skipif(
    os.environ.get("DCP_INVARIANT_LIVE") != "1",
    reason="set DCP_INVARIANT_LIVE=1 in the pinned PyTorch integration job",
)
def test_live_registered_suite_has_no_native_public_files(tmp_path: Path) -> None:
    output = tmp_path / "evidence"
    result = run_suite(
        output,
        source_revision="a" * 40,
        timeout_seconds=180.0,
    )

    assert result.native_work_cleaned is True
    assert result.artifact.summary["passed_scenarios"] == 12
    assert len(result.observations) == 12
    assert not [
        path
        for path in output.rglob("*")
        if path.is_file()
        and (
            path.name in {".elastic-failure.json", ".metadata"}
            or path.suffix == ".distcp"
        )
    ]
