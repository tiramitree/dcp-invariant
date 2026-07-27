from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from dcp_invariant.supervisor import (
    LATEST_SCHEMA,
    SupervisorError,
    WorkerOutcomeError,
    minimal_worker_environment,
    promote_candidate,
    run_workers,
)


def promotion_root(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "root"
    candidates = root / "candidates"
    committed = root / "committed"
    candidate = candidates / "candidate-1"
    checkpoint = candidate / "checkpoint-one"
    checkpoint.mkdir(parents=True)
    committed.mkdir()
    (checkpoint / "value.txt").write_text("verified", encoding="utf-8")
    (checkpoint / "checkpoint-receipt.json").write_text(
        '{"receipt":"fixture"}\n',
        encoding="utf-8",
        newline="\n",
    )
    return root, candidate


def test_minimal_environment_excludes_common_credentials() -> None:
    environment = minimal_worker_environment()
    assert "HOME" not in environment
    assert "USERPROFILE" not in environment
    assert not any("TOKEN" in key or "KEY" in key for key in environment)
    assert environment["USE_LIBUV"] == "0"


def test_worker_environment_uses_only_isolated_home(tmp_path: Path) -> None:
    environment = minimal_worker_environment(tmp_path)
    isolated = str(tmp_path.resolve())
    assert environment["HOME"] == isolated
    assert environment["TEMP"] == isolated
    assert environment["TMP"] == isolated
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert {
        environment["LNAME"],
        environment["LOGNAME"],
        environment["USER"],
        environment["USERNAME"],
    } == {"dcp-invariant"}
    if sys.platform == "win32":
        assert environment["USERPROFILE"] == isolated


def test_worker_success(tmp_path: Path) -> None:
    (tmp_path / "success_helper.py").write_text(
        "raise SystemExit(0)\n", encoding="utf-8"
    )
    result = run_workers(
        module="success_helper",
        common_arguments=[],
        world_size=1,
        cwd=tmp_path,
        timeout_seconds=5,
    )
    assert result.exit_codes == (0,)
    assert not result.timed_out


def test_worker_failure_is_normalized(tmp_path: Path) -> None:
    with pytest.raises(WorkerOutcomeError, match="exit_codes") as caught:
        run_workers(
            module="does_not_exist",
            common_arguments=[],
            world_size=1,
            cwd=tmp_path,
            timeout_seconds=5,
        )
    assert caught.value.result.exit_codes != (0,)
    assert caught.value.expected_exit_codes == (0,)


def test_worker_timeout_is_normalized(tmp_path: Path) -> None:
    helper = tmp_path / "sleep_helper.py"
    helper.write_text("import time; time.sleep(10)\n", encoding="utf-8")
    with pytest.raises(WorkerOutcomeError, match="timed_out=True") as caught:
        run_workers(
            module="sleep_helper",
            common_arguments=[],
            world_size=1,
            cwd=tmp_path,
            timeout_seconds=0.2,
        )
    assert caught.value.result.timed_out is True


def test_successful_promotion_updates_pointer(tmp_path: Path) -> None:
    root, candidate = promotion_root(tmp_path)

    def verify(checkpoint: Path) -> bool:
        assert (checkpoint / "value.txt").read_text(encoding="utf-8") == "verified"
        return True

    target = promote_candidate(
        root=root,
        candidate=candidate,
        logical_checkpoint_id="checkpoint-one",
        verify=verify,
    )
    import hashlib

    digest = hashlib.sha256(
        (target / "checkpoint-one" / "checkpoint-receipt.json").read_bytes()
    ).hexdigest()
    assert target == root / "committed" / digest
    assert target.is_dir()
    assert not candidate.exists()
    assert json.loads((root / "LATEST.json").read_text(encoding="utf-8")) == {
        "generation": digest,
        "pointer_schema": LATEST_SCHEMA,
    }


def test_failed_verification_preserves_old_pointer(tmp_path: Path) -> None:
    root, candidate = promotion_root(tmp_path)
    previous = (
        '{"generation":"' + ("b" * 64) + '","pointer_schema":"' + LATEST_SCHEMA + '"}\n'
    )
    (root / "LATEST.json").write_text(previous, encoding="utf-8", newline="\n")

    def reject(path: Path) -> bool:
        raise ValueError(path.name)

    with pytest.raises(ValueError):
        promote_candidate(
            root=root,
            candidate=candidate,
            logical_checkpoint_id="checkpoint-one",
            verify=reject,
        )
    assert (root / "LATEST.json").read_text(encoding="utf-8") == previous
    assert candidate.is_dir()
    assert not (root / "committed" / ("a" * 64)).exists()


def test_existing_generation_is_not_overwritten(tmp_path: Path) -> None:
    root, candidate = promotion_root(tmp_path)
    import hashlib

    digest = hashlib.sha256(
        (candidate / "checkpoint-one" / "checkpoint-receipt.json").read_bytes()
    ).hexdigest()
    (root / "committed" / digest).mkdir()
    with pytest.raises(SupervisorError, match="already exists"):
        promote_candidate(
            root=root,
            candidate=candidate,
            logical_checkpoint_id="checkpoint-one",
            verify=lambda path: True,
        )


@pytest.mark.skipif(sys.platform == "win32", reason="symlink privilege varies")
def test_symlink_candidate_is_rejected(tmp_path: Path) -> None:
    root, candidate = promotion_root(tmp_path)
    target = root / "real"
    target.mkdir()
    checkpoint = candidate / "checkpoint-one"
    (checkpoint / "value.txt").unlink()
    (checkpoint / "checkpoint-receipt.json").unlink()
    checkpoint.rmdir()
    candidate.rmdir()
    candidate.symlink_to(target, target_is_directory=True)
    with pytest.raises(SupervisorError, match="ordinary"):
        promote_candidate(
            root=root,
            candidate=candidate,
            logical_checkpoint_id="checkpoint-one",
            verify=lambda path: True,
        )


def test_false_verifier_result_cannot_promote(tmp_path: Path) -> None:
    root, candidate = promotion_root(tmp_path)
    with pytest.raises(SupervisorError, match="exact success"):
        promote_candidate(
            root=root,
            candidate=candidate,
            logical_checkpoint_id="checkpoint-one",
            verify=lambda path: False,
        )
    assert candidate.is_dir()
    assert not (root / "LATEST.json").exists()


def test_receipt_digest_names_generation(tmp_path: Path) -> None:
    import hashlib

    root, candidate = promotion_root(tmp_path)
    expected = hashlib.sha256(
        (candidate / "checkpoint-one" / "checkpoint-receipt.json").read_bytes()
    ).hexdigest()
    target = promote_candidate(
        root=root,
        candidate=candidate,
        logical_checkpoint_id="checkpoint-one",
        verify=lambda path: True,
    )
    assert target.name == expected
    pointer = json.loads((root / "LATEST.json").read_text(encoding="utf-8"))
    assert pointer["generation"] == expected


def test_invalid_existing_pointer_is_not_overwritten(tmp_path: Path) -> None:
    root, candidate = promotion_root(tmp_path)
    (root / "LATEST.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SupervisorError, match="existing pointer"):
        promote_candidate(
            root=root,
            candidate=candidate,
            logical_checkpoint_id="checkpoint-one",
            verify=lambda path: True,
        )
    assert (root / "LATEST.json").read_text(encoding="utf-8") == "{}\n"
    assert candidate.is_dir()


def test_candidate_wrapper_rejects_unregistered_sibling(tmp_path: Path) -> None:
    root, candidate = promotion_root(tmp_path)
    (candidate / "worker-report.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SupervisorError, match="contain only"):
        promote_candidate(
            root=root,
            candidate=candidate,
            logical_checkpoint_id="checkpoint-one",
            verify=lambda path: True,
        )
    assert candidate.is_dir()


def test_registered_rank_exit_vector_is_structured(tmp_path: Path) -> None:
    with pytest.raises(WorkerOutcomeError) as caught:
        run_workers(
            module="dcp_invariant.rank_exit_worker",
            common_arguments=[],
            world_size=2,
            cwd=tmp_path,
            timeout_seconds=5,
        )
    assert caught.value.result.timed_out is False
    assert caught.value.result.exit_codes == (0, 91)
    assert caught.value.expected_exit_codes == (0, 0)
