from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from dcp_invariant.canonical import canonical_json
from dcp_invariant.supervisor import (
    LEGACY_LATEST_SCHEMA,
    LINEAGE_RECORD_NAME,
    POINTER_LOCK_NAME,
    CommittedGeneration,
    ParentVersion,
    StaleParentError,
    SupervisorError,
    commit_candidate,
    load_committed_generation,
    publish_committed_generation,
    read_parent_version,
)

CHECKPOINT_ID = "checkpoint-one"
RECEIPT_NAME = "checkpoint-receipt.json"


def make_root(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "candidates").mkdir(parents=True)
    (root / "committed").mkdir()
    return root


def make_candidate(root: Path, name: str, receipt: str) -> Path:
    wrapper = root / "candidates" / name
    checkpoint = wrapper / CHECKPOINT_ID
    checkpoint.mkdir(parents=True)
    (checkpoint / "value.txt").write_text(receipt, encoding="utf-8", newline="\n")
    (checkpoint / RECEIPT_NAME).write_text(
        canonical_json({"receipt": receipt}) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return wrapper


def verify(checkpoint: Path) -> bool:
    return (
        checkpoint.name == CHECKPOINT_ID
        and (checkpoint / "value.txt").is_file()
        and (checkpoint / RECEIPT_NAME).is_file()
    )


def commit(
    root: Path,
    candidate: Path,
    parent: ParentVersion,
) -> CommittedGeneration:
    return commit_candidate(
        root=root,
        candidate=candidate,
        logical_checkpoint_id=CHECKPOINT_ID,
        verify=verify,
        parent=parent,
    )


def read_pointer(root: Path) -> tuple[dict[str, object], bytes]:
    raw = (root / "LATEST.json").read_bytes()
    return json.loads(raw), raw


def test_two_children_of_one_parent_select_exactly_one_and_preserve_orphan(
    tmp_path: Path,
) -> None:
    root = make_root(tmp_path)
    parent = read_parent_version(root)
    first = commit(root, make_candidate(root, "candidate-a", "a"), parent)
    second = commit(root, make_candidate(root, "candidate-b", "b"), parent)

    assert (
        publish_committed_generation(
            root=root,
            committed=first,
            verify=verify,
        )
        == "published"
    )
    selected, selected_raw = read_pointer(root)

    with pytest.raises(StaleParentError, match="stale_parent") as caught:
        publish_committed_generation(
            root=root,
            committed=second,
            verify=verify,
        )

    assert caught.value.committed.generation == second.generation
    assert (root / "LATEST.json").read_bytes() == selected_raw
    assert selected["generation"] == first.generation
    assert selected["sequence"] == 0
    assert selected["parent_pointer_sha256"] is None
    assert first.target.is_dir()
    assert second.target.is_dir()
    assert second.target != first.target


def test_commit_publish_crash_window_can_resume(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    parent = read_parent_version(root)
    generation = commit(
        root,
        make_candidate(root, "candidate-a", "resume-before-publish"),
        parent,
    )

    assert generation.target.is_dir()
    assert not (root / "LATEST.json").exists()
    assert (
        publish_committed_generation(
            root=root,
            committed=generation,
            verify=verify,
        )
        == "published"
    )
    assert read_pointer(root)[0]["generation"] == generation.generation


def test_publish_return_crash_window_is_idempotent(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    generation = commit(
        root,
        make_candidate(root, "candidate-a", "resume-after-publish"),
        read_parent_version(root),
    )
    assert (
        publish_committed_generation(
            root=root,
            committed=generation,
            verify=verify,
        )
        == "published"
    )
    selected_raw = read_pointer(root)[1]

    assert (
        publish_committed_generation(
            root=root,
            committed=generation,
            verify=verify,
        )
        == "already_published"
    )
    assert (root / "LATEST.json").read_bytes() == selected_raw
    assert len(list((root / "committed").iterdir())) == 1


def test_publication_return_hook_runs_after_pointer_is_selected(
    tmp_path: Path,
) -> None:
    root = make_root(tmp_path)
    generation = commit(
        root,
        make_candidate(root, "candidate", "receipt"),
        read_parent_version(root),
    )
    observed: list[ParentVersion] = []

    def observe_selected_pointer() -> None:
        observed.append(read_parent_version(root))

    assert (
        publish_committed_generation(
            root=root,
            committed=generation,
            verify=verify,
            _publication_return_hook=observe_selected_pointer,
        )
        == "published"
    )
    assert observed == [read_parent_version(root)]
    assert observed[0].pointer_sha256 is not None


def test_same_receipt_and_lineage_reuses_immutable_generation(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    parent = read_parent_version(root)
    first = commit(root, make_candidate(root, "candidate-a", "same"), parent)
    retry_candidate = make_candidate(root, "candidate-retry", "same")

    retry = commit(root, retry_candidate, parent)

    assert retry.reused is True
    assert retry.target == first.target
    assert retry.lineage_sha256 == first.lineage_sha256
    assert retry_candidate.is_dir()
    assert len(list((root / "committed").iterdir())) == 1


def test_same_receipt_with_different_lineage_fails_closed(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    first = commit(
        root,
        make_candidate(root, "candidate-a", "same"),
        read_parent_version(root),
    )
    assert (
        publish_committed_generation(
            root=root,
            committed=first,
            verify=verify,
        )
        == "published"
    )
    pointer_before = (root / "LATEST.json").read_bytes()
    conflicting = make_candidate(root, "candidate-conflict", "same")

    with pytest.raises(SupervisorError, match="generation_lineage_conflict"):
        commit(root, conflicting, read_parent_version(root))

    assert conflicting.is_dir()
    assert (root / "LATEST.json").read_bytes() == pointer_before
    assert len(list((root / "committed").iterdir())) == 1


@pytest.mark.parametrize(
    "forged",
    [
        ParentVersion("a" * 64, -1),
        ParentVersion(None, 0),
        ParentVersion(None, True),
        ParentVersion("not-a-digest", -1),
    ],
)
def test_forged_parent_version_fails_before_candidate_mutation(
    tmp_path: Path,
    forged: ParentVersion,
) -> None:
    root = make_root(tmp_path)
    candidate = make_candidate(root, "candidate-a", "forged-parent")

    with pytest.raises(SupervisorError, match="parent"):
        commit(root, candidate, forged)

    assert candidate.is_dir()
    assert list((root / "committed").iterdir()) == []
    assert not (root / "LATEST.json").exists()


def test_legacy_pointer_is_an_exact_migration_anchor(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    legacy = (
        canonical_json(
            {
                "generation": "b" * 64,
                "pointer_schema": LEGACY_LATEST_SCHEMA,
            }
        )
        + "\n"
    ).encode("utf-8")
    (root / "LATEST.json").write_bytes(legacy)
    parent = read_parent_version(root)

    assert parent == ParentVersion(
        pointer_sha256=hashlib.sha256(legacy).hexdigest(),
        sequence=-1,
    )
    generation = commit(
        root,
        make_candidate(root, "candidate-a", "legacy-child"),
        parent,
    )
    assert generation.sequence == 0
    assert generation.parent_pointer_sha256 == parent.pointer_sha256
    assert (
        publish_committed_generation(
            root=root,
            committed=generation,
            verify=verify,
        )
        == "published"
    )
    pointer, _ = read_pointer(root)
    assert pointer["sequence"] == 0
    assert pointer["parent_pointer_sha256"] == parent.pointer_sha256


def test_malformed_lineage_fails_without_pointer_change(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    generation = commit(
        root,
        make_candidate(root, "candidate-a", "malformed-lineage"),
        read_parent_version(root),
    )
    (generation.target / LINEAGE_RECORD_NAME).write_text(
        '{"logical_checkpoint_id":[]}\n',
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(SupervisorError, match="lineage"):
        publish_committed_generation(
            root=root,
            committed=generation,
            verify=verify,
        )

    assert not (root / "LATEST.json").exists()


def test_invalid_persistent_lock_content_fails_closed(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    generation = commit(
        root,
        make_candidate(root, "candidate-a", "invalid-lock"),
        read_parent_version(root),
    )
    (root / POINTER_LOCK_NAME).write_bytes(b"x")

    with pytest.raises(SupervisorError, match="lock"):
        publish_committed_generation(
            root=root,
            committed=generation,
            verify=verify,
        )

    assert not (root / "LATEST.json").exists()
    assert (root / POINTER_LOCK_NAME).read_bytes() == b"x"


def test_unknown_old_pending_name_is_not_deleted_or_reused(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    unknown = root / ".LATEST.json.pending"
    unknown.write_bytes(b"do-not-touch")
    generation = commit(
        root,
        make_candidate(root, "candidate-a", "unique-staging"),
        read_parent_version(root),
    )

    assert (
        publish_committed_generation(
            root=root,
            committed=generation,
            verify=verify,
        )
        == "published"
    )
    assert unknown.read_bytes() == b"do-not-touch"


def test_lock_file_is_one_persistent_zero_byte_record(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    generation = commit(
        root,
        make_candidate(root, "candidate-a", "lock-record"),
        read_parent_version(root),
    )
    publish_committed_generation(
        root=root,
        committed=generation,
        verify=verify,
    )

    lock = root / POINTER_LOCK_NAME
    assert lock.read_bytes() == b"\0"
    assert lock.is_file()
    if sys.platform != "win32":
        assert lock.stat().st_nlink == 1


def test_exact_precreated_lineage_record_resumes_commit(tmp_path: Path) -> None:
    root = make_root(tmp_path)
    parent = read_parent_version(root)
    candidate = make_candidate(root, "candidate-a", "resume-before-rename")
    receipt = candidate / CHECKPOINT_ID / RECEIPT_NAME
    generation = hashlib.sha256(receipt.read_bytes()).hexdigest()
    payload = {
        "generation": generation,
        "lineage_schema": "dcp-invariant-generation-lineage-v1",
        "logical_checkpoint_id": CHECKPOINT_ID,
        "parent_pointer_sha256": None,
        "sequence": 0,
    }
    (candidate / LINEAGE_RECORD_NAME).write_text(
        canonical_json(payload) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    committed = commit(root, candidate, parent)

    assert committed.generation == generation
    assert committed.target.is_dir()
    assert not candidate.exists()


def test_committed_descriptor_can_be_recovered_after_process_loss(
    tmp_path: Path,
) -> None:
    root = make_root(tmp_path)
    generation = commit(
        root,
        make_candidate(root, "candidate-a", "recover-descriptor"),
        read_parent_version(root),
    ).generation

    recovered = load_committed_generation(
        root=root,
        generation=generation,
        logical_checkpoint_id=CHECKPOINT_ID,
        verify=verify,
    )

    assert recovered.generation == generation
    assert recovered.reused is False
    assert (
        publish_committed_generation(
            root=root,
            committed=recovered,
            verify=verify,
        )
        == "published"
    )


def test_pointer_lineage_digest_mismatch_is_not_a_valid_parent(
    tmp_path: Path,
) -> None:
    root = make_root(tmp_path)
    generation = commit(
        root,
        make_candidate(root, "candidate-a", "pointer-binding"),
        read_parent_version(root),
    )
    publish_committed_generation(
        root=root,
        committed=generation,
        verify=verify,
    )
    pointer, _ = read_pointer(root)
    pointer["lineage_sha256"] = "f" * 64
    forged = (canonical_json(pointer) + "\n").encode("utf-8")
    (root / "LATEST.json").write_bytes(forged)

    with pytest.raises(SupervisorError, match="bound"):
        read_parent_version(root)

    assert (root / "LATEST.json").read_bytes() == forged


def test_pointer_missing_committed_generation_is_not_a_valid_parent(
    tmp_path: Path,
) -> None:
    root = make_root(tmp_path)
    pointer = {
        "generation": "e" * 64,
        "lineage_sha256": "d" * 64,
        "parent_pointer_sha256": None,
        "pointer_schema": "dcp-invariant-latest-v2",
        "sequence": 0,
    }
    raw = (canonical_json(pointer) + "\n").encode("utf-8")
    (root / "LATEST.json").write_bytes(raw)

    with pytest.raises(SupervisorError):
        read_parent_version(root)

    assert (root / "LATEST.json").read_bytes() == raw
