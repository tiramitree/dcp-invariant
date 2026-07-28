from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from dcp_invariant.artifact import normalize_observation
from dcp_invariant.canonical import canonical_json
from dcp_invariant.suite import (
    SuiteError,
    _ensure_ordinary_output_parent,
    _read_reports,
    _run_rank_exit_fault,
    _validate_rank_consensus,
    _validate_report,
    _write_seed_pointer,
    run_suite,
)
from dcp_invariant.supervisor import LATEST_SCHEMA, WorkerResult

SHA256 = "a" * 64


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
    assert result.artifact.summary["passed_scenarios"] == 11
    assert len(result.observations) == 11
    assert not [
        path
        for path in output.rglob("*")
        if path.is_file()
        and (
            path.name in {".elastic-failure.json", ".metadata"}
            or path.suffix == ".distcp"
        )
    ]
