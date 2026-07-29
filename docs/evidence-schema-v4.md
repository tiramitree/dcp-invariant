# Evidence artifact schema v4

The v4 artifact is a normalized, fixed-inventory report for thirteen
registered scenarios. It excludes native PyTorch checkpoint files, tensor
values, raw worker or launcher logs, raw lineage coordination records,
standalone asynchronous gate-marker and rank-report files, the raw elastic
marker and bootstrap attestation, absolute paths, real user or host identity,
process identifiers, ports, rendezvous values, environment values, timings,
and byte profiles.

Validated fixed worker fields are embedded only where required by a registered
contract. They remain subject to exact field-set, canonical-JSON, and
cross-record checks.

## Compatibility

Schema v4 is not byte- or inventory-compatible with v3, v2, or v1. The v0.4
verifier accepts v4 only. Historical twelve-scenario v3, eleven-scenario v2,
and ten-scenario v1 artifacts require their matching v0.3, v0.2, or v0.1
verifier.

## Inventory

The only accepted entries are:

- `provenance.json`
- `summary.json`
- `observations/<scenario>.json` for all thirteen registered scenarios
- `results/<scenario>.json` for all thirteen registered scenarios
- `junit.xml`
- `manifest.sha256`

The scenarios are:

- `async_snapshot_resnet18_2r`
- `dtensor_1_to_2`
- `dtensor_2_to_1`
- `training_1_to_1`
- `training_1_to_2`
- `training_2_to_1`
- `training_2_to_2`
- `elastic_restart_2_to_2`
- `generation_lineage_stale_writer_2p`
- `rank_exit_no_promotion`
- `missing_metadata`
- `missing_shard`
- `corrupt_shard`

All JSON is one compact, key-sorted UTF-8 record terminated by LF. JUnit has
one passing case per registered scenario and contains no durations or output
streams.

`provenance.json` binds the exact CPython, PyTorch, and NumPy versions used by
the run together with the referenced 40-hex source revision. The asynchronous
rank reports separately bind the registered torchvision distribution/runtime
pair and Pillow release.

`summary.json` requires thirteen passing scenarios, nine promotion-allowed
scenarios, four training-state equalities, two reconstructed global-tensor
equalities, one elastic recovery, one asynchronous staged-snapshot equality,
one stale-writer rejection, and four fault rejections.

## Pointer and lineage binding

Every normalized v2 pointer has exactly:

- `generation`
- `lineage_sha256`
- `parent_pointer_sha256`
- `pointer_schema`
- `pointer_sha256`
- `sequence`

`generation` is the SHA-256 of the canonical checkpoint receipt.
`lineage_sha256` is recomputed from the canonical lineage record containing
the generation, logical checkpoint identifier, selected-parent pointer digest,
sequence, and `dcp-invariant-generation-lineage-v1` schema. `pointer_sha256`
is recomputed from the five persisted pointer fields. A non-root sequence
requires a parent digest. The live supervisor additionally requires the
pointer to be backed by an ordinary committed target whose receipt and lineage
record match those fields.

The v4 artifact does not include the native committed tree or lineage record.
It carries their verified digests and normalized pointer fields. This supports
internal binding, not publisher authentication or an external timestamp.

## Existing state, snapshot, elastic, and fault scenarios

Results are derived by the verifier; callers cannot provide result hashes
separately.

Training validation requires complete ordered rank sets, rank consensus on
model, optimizer, RNG, cursor, and aggregate hashes, checkpoint equality at
load, uninterrupted/resumed equality after the next registered step, and
successful receipt checks around save, publication, and load.

DTensor validation requires rank consensus and exact equality of the
reconstructed global tensor. Local shard shapes are checked against each world
size, but local shards are not claimed to be identical across topologies.

The asynchronous observation retains the v3 fixed ResNet18 workload,
public-staging-hook, write-gate, targeted mutation, rank-consensus, direct
loaded-state, and receipt requirements. The elastic observation retains the
v3 exact torchrun, registered restart, loopback, bootstrap-source guard,
pre/post receipt equality, selected-generation reuse, and no-post-failure-
publication requirements. Their promotion pointers use the v2 lineage-bound
format.

Fault validation binds the exact child-exit vector or mutation description,
prohibits DCP load, requires candidate preservation, and requires a real
backed seed-generation pointer to remain byte-identical. Missing or corrupt
checkpoint faults must be rejected by receipt verification during the
publication attempt.

## Generation-lineage stale-writer scenario

`generation_lineage_stale_writer_2p` has one matched unfenced control arm, one
protected arm, two process-exit recovery records, and three fail-closed
rejection checks.


Its category-specific top-level fields are `publisher_process_count:2` and
`selected_head_count:1`; they are not source and target process topologies.
Each arm begins from the same canonical receipt- and lineage-backed v2 seed
pointer at sequence zero.
For both two-process arms:

- two real subprocesses capture the same backed seed pointer before either
  commits;
- both commit distinct immutable generations before either publishes;
- the complete committed inventory contains the seed plus exactly those two
  child generations;
- both worker exit codes are zero with no timeout;
- each generation-tree SHA-256 is equal before and after publication; and
- the final pointer's generation, lineage digest, parent digest, sequence, and
  canonical pointer digest are revalidated.

The public arm also binds `both_committed_before_publication:true`,
`publish_order:[0,1]`, the complete starting pointer, both child lineage
digests, and A's pointer digest immediately after A's action. The control and
protected arms must match on the starting pointer, child generation digests,
child lineage digests, and before/after generation-tree digests.

The private negative-control arm performs fixed A-then-B unfenced writes.
Both outcomes are `published_unfenced`, the final pointer selects B, and the
observation records `reference_overwrite_observed:true`. A's recorded
intermediate pointer must differ from B's final pointer.

The protected arm uses the production conditional-publication protocol. A
returns `published`; B returns `stale_parent`; the final pointer selects A;
B remains present as an unselected committed generation; and the observation
records exactly one stale-writer rejection. The control and protected arms
must use matched generation digests.

The recovery records require:

- exit code 73 after commit but before publication, followed by descriptor
  reload by the surviving supervisor, `published`, the matching pointer, and
  unchanged committed bytes;
- exit code 74 from a test-only hook after successful pointer re-read and lock
  release but before the publication function returns, followed by descriptor
  reload by the surviving supervisor, `already_published`, a byte-identical
  pointer, and unchanged committed bytes;
- the same backed starting pointer, generation digest, lineage, and
  generation-tree digest as the protected arm's first child.

The rejection record requires:

- a forged parent digest to return `parent_version_invalid`;
- a parent sequence mismatch to return `parent_version_invalid`;
- same-receipt reuse under a different lineage to return
  `generation_lineage_conflict`;
- all three candidates to remain present; and
- all affected pointers to remain unchanged.

This scenario supports only the registered single-host, ordinary-local-
filesystem, cooperating-process interleaving. It does not establish a
filesystem compare-and-swap, multi-host consensus, network-filesystem
semantics, hostile-writer exclusion, power-loss durability, or recovery from
failure during the lineage-record write itself.

## Snapshot and manifest boundary

The verifier reads each payload once into one bounded in-memory snapshot. It
uses those same bytes for manifest hashing, JSON parsing, semantic validation,
derived results, and the returned object.

`manifest.sha256` covers every payload except itself in fixed order. It is
**not signed**: it does not authenticate a publisher, establish an external
timestamp, or prevent coordinated replacement of payloads and manifest. Its
purpose is internal byte closure, and provenance fixes `authenticated` to
`false`.

Normalized observations remain producer-generated evidence. Their strength
comes from the open runner, strict schema, and any independently inspectable CI
or reproduction that actually ran them. A configured but unrun workflow does
not add evidence, and the unsigned artifact alone is not independent
attestation.
