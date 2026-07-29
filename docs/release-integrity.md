# Release integrity and audit scope

## Current release gate

The v0.4 owner release gate accepts one fixed release-notes file and exactly
four fixed-name assets held outside the source checkout:

- the Python wheel;
- the source distribution;
- the schema-v4 evidence archive with its fixed 13-scenario inventory; and
- `SHA256SUMS`, which binds exactly the other three assets.

It scans the frozen source tree and fetched Git closure, checks every reachable
commit and annotated-tag identity against the registered pseudonymous
identities, and requires the unsigned annotated release tag to point directly
to current `HEAD`. The frozen working tree must match that commit's complete
ordinary-file blob inventory, rejecting both tracked changes and untracked
files. It parses exact ZIP, gzip, and TAR bytes with ASCII-only member names,
validates logical archive members, verifies that the evidence artifact binds
the same source revision, and validates canonical release notes and checksums.
The wheel package and license bytes and the source distribution's complete
source inventory must equal the frozen source snapshot. A private owner
denylist supplements generic credential, contact, identity, network, and
user-path rules without entering the repository. The gate rechecks every
input and the source/history snapshots after the full audit.

This is a publication boundary, not an authenticity scheme. Release assets and
their checksum file remain unsigned.

## Retrospective v0.1.0-v0.3.0 audit

On 2026-07-29, the owner re-downloaded the public GitHub Release metadata and
all attached assets for v0.1.0, v0.2.0, and v0.3.0. The inspected snapshot
contained 12 assets: three wheels, three source distributions, three evidence
archives, and three `SHA256SUMS` files.

Within that snapshot:

- every downloaded asset's byte length and SHA-256 digest matched the values
  reported by the GitHub Release API;
- each `SHA256SUMS` file was strict UTF-8, contained exactly three canonical
  rows, covered exactly the other three assets in its Release, and matched
  their downloaded bytes;
- all nine archives passed exact container checks for prefixes, suffixes,
  gaps, duplicate or extra metadata, encryption, data descriptors,
  compression termination, CRCs, TAR headers, padding, and end-of-file
  structure; and
- release titles, release notes, asset names, raw archive bytes, member names,
  member contents, and checksum manifests produced no finding under the
  current generic privacy rules plus the private owner denylist.

No in-scope finding justified deleting those Releases. Historical evidence
schemas remain version-specific: use v0.1 for schema v1, v0.2 for schema v2,
and v0.3 for schema v3.

## Limits

This was an owner-operated retrospective audit, not independent reproduction,
third-party review, security certification, or proof of external use. It
covers only the Release metadata and asset bytes retrieved for the stated
snapshot. It does not prove erasure from cached pages, search indexes,
unfetched GitHub-managed refs, unreachable or reflog-only server objects, LFS
storage, submodule repositories, prior downloads, mirrors, or external forks.
Those surfaces can also change after the snapshot.
