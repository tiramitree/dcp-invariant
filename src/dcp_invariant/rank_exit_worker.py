"""Deterministic child-exit fixture for the supervisor promotion gate."""

from __future__ import annotations

import argparse

REGISTERED_EXIT_CODE = 91


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dcp_invariant.rank_exit_worker")
    parser.add_argument("--rank", required=True, type=int)
    parser.add_argument("--world-size", required=True, type=int)
    parser.add_argument("--master-port", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.world_size != 2:
        raise ValueError("rank-exit fixture requires exactly two workers")
    if arguments.rank not in {0, 1}:
        raise ValueError("rank-exit fixture rank is invalid")
    if not 1 <= arguments.master_port <= 65535:
        raise ValueError("rank-exit fixture port is invalid")
    return REGISTERED_EXIT_CODE if arguments.rank == 1 else 0


if __name__ == "__main__":
    raise SystemExit(main())
