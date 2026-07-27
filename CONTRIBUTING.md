# Contributing

DCPInvariant is pre-release and owner-operated. Reproductions and narrowly
scoped bug reports are welcome through this repository's GitHub issue tracker.

Before proposing a change:

1. preserve the exact claim boundary;
2. add a failing test for any correctness or privacy defect;
3. run `pytest`, `ruff check`, and `ruff format --check`;
4. run the complete DCP suite when changing worker, receipt, supervisor, or
   observation semantics;
5. never commit native `.metadata` or `.distcp` files, raw worker logs, secrets,
   machine paths, or personal contact details.

Performance or production claims require a separately preregistered public
workload and protocol. A synthetic result must not be generalized beyond the
specific engineering invariant it tests.
