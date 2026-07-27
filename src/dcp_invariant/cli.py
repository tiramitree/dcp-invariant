"""Command-line entry point with a PyTorch-free offline verifier path."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

from .artifact import EvidenceArtifactError, verify_evidence_artifact
from .canonical import canonical_json

_SOURCE_REVISION = re.compile(r"[0-9a-f]{40}\Z")


def _revision(value: str) -> str:
    if not _SOURCE_REVISION.fullmatch(value):
        raise argparse.ArgumentTypeError(
            "source revision must be one lowercase 40-hex value"
        )
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dcp-invariant")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run all registered DCP scenarios")
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--source-revision", required=True, type=_revision)
    run.add_argument("--timeout-seconds", type=float, default=180.0)

    verify = commands.add_parser(
        "verify",
        help="verify an existing normalized evidence artifact offline",
    )
    verify.add_argument("--artifact-dir", required=True, type=Path)
    return parser


def _success_payload(artifact: Any) -> dict[str, Any]:
    return {
        "manifest_sha256": artifact.manifest_sha256,
        "overall_status": artifact.summary["overall_status"],
        "passed_scenarios": artifact.summary["passed_scenarios"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "verify":
            artifact = verify_evidence_artifact(arguments.artifact_dir)
        else:
            # Keep the offline verifier import graph free of torch and the live
            # worker module.  The suite itself also launches torch out of process.
            from .suite import run_suite

            artifact = run_suite(
                arguments.output_dir,
                source_revision=arguments.source_revision,
                timeout_seconds=arguments.timeout_seconds,
            ).artifact
    except (EvidenceArtifactError, OSError, RuntimeError):
        parser.exit(1, "dcp-invariant: command failed\n")
    print(canonical_json(_success_payload(artifact)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
