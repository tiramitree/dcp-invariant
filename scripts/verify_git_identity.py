"""Fail closed when reachable Git history contains an unregistered identity."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from git_snapshot import (  # noqa: E402
    GitSnapshotError,
    assert_snapshot_unchanged,
    freeze_closure,
    read_object,
)

OWNER_NAME = "tiramitree"
OWNER_EMAIL = "89479100+tiramitree" + chr(64) + "users.noreply.github.com"
_GITHUB_IDENTITIES = {
    ("GitHub", "noreply" + chr(64) + "github.com"),
    ("GitHub", "web-flow" + chr(64) + "users.noreply.github.com"),
}
ALLOWED_IDENTITIES = frozenset({(OWNER_NAME, OWNER_EMAIL), *_GITHUB_IDENTITIES})
_ALLOWED_IDENTITY_BYTES = frozenset(
    (name.encode("utf-8"), email.encode("utf-8")) for name, email in ALLOWED_IDENTITIES
)
_IDENTITY = re.compile(
    rb"^(author|committer|tagger) ([^<>\r\n]+) <([^<>\r\n]+)> "
    rb"([0-9]+) ([+-][0-9]{4})$"
)


class GitIdentityError(ValueError):
    """Reachable history did not satisfy the pseudonymous identity policy."""


def _header_lines(raw: bytes) -> list[bytes]:
    if b"\0" in raw:
        raise GitIdentityError("Git identity object contains NUL")
    header, separator, _ = raw.partition(b"\n\n")
    if not separator or not header:
        raise GitIdentityError("Git identity object headers are malformed")
    return [line for line in header.split(b"\n") if not line.startswith(b" ")]


def _identity(raw: bytes, *, role: bytes) -> tuple[bytes, bytes]:
    lines = [line for line in _header_lines(raw) if line.startswith(role + b" ")]
    if len(lines) != 1:
        raise GitIdentityError("Git identity header cardinality is invalid")
    matched = _IDENTITY.fullmatch(lines[0])
    if matched is None or matched.group(1) != role:
        raise GitIdentityError("Git identity header is malformed")
    return matched.group(2), matched.group(3)


def _check_identity(raw: bytes, *, role: bytes) -> None:
    if _identity(raw, role=role) not in _ALLOWED_IDENTITY_BYTES:
        role_name = role.decode("ascii")
        raise GitIdentityError(f"Git {role_name} identity is not registered")


def audit_repository(root: Path) -> dict[str, int | str]:
    """Audit raw identities in a frozen fetched-ref and HEAD closure.

    GitHub pull-request refs, unreachable server objects, and external forks
    are not represented as inspected.
    """

    root = root.resolve(strict=True)
    closure = freeze_closure(root)
    commit_count = 0
    reachable_tag_count = 0
    for value in closure.objects:
        if value.object_type == "commit":
            raw = read_object(root, value)
            _check_identity(raw, role=b"author")
            _check_identity(raw, role=b"committer")
            commit_count += 1
        elif value.object_type == "tag":
            raw = read_object(root, value)
            _check_identity(raw, role=b"tagger")
            reachable_tag_count += 1
    if commit_count == 0:
        raise GitIdentityError("reachable Git commit inventory is empty")
    assert_snapshot_unchanged(root, closure.snapshot)
    return {
        "commit_count": commit_count,
        "fetched_ref_count": len(closure.snapshot.refs),
        "inventory_sha256": closure.inventory_sha256,
        "reachable_annotated_tag_count": reachable_tag_count,
        "reachable_object_count": len(closure.objects),
        "tag_ref_count": sum(
            refname.startswith("refs/tags/") for refname, _ in closure.snapshot.refs
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        audit = audit_repository(arguments.root)
    except (
        GitSnapshotError,
        GitIdentityError,
        OSError,
        UnicodeError,
    ) as error:
        print(
            json.dumps(
                {
                    "error": type(error).__name__,
                    "status": "ERROR",
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                **audit,
                "status": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
