# Evidence artifact schema v1

The v1 artifact is a normalized, fixed-inventory report. It excludes native
PyTorch checkpoint files, tensor values, raw logs, absolute paths, real user or
host identity, process identifiers, ports, and environment values.

## Inventory

The only accepted entries are:

- `provenance.json`
- `summary.json`
- `observations/<scenario>.json` for all ten registered scenarios
- `results/<scenario>.json` for all ten registered scenarios
- `junit.xml`
- `manifest.sha256`

The scenarios are:

- `dtensor_1_to_2`
- `dtensor_2_to_1`
- `training_1_to_1`
- `training_1_to_2`
- `training_2_to_1`
- `training_2_to_2`
- `rank_exit_no_promotion`
- `missing_metadata`
- `missing_shard`
- `corrupt_shard`

All JSON is one compact, key-sorted UTF-8 record terminated by LF. JUnit has
one passing case per registered scenario and contains no durations or output
streams.

## Observation-to-result binding

Results are derived by the verifier; callers cannot provide result hashes
separately.

Each positive observation includes the complete ordered rank-report set,
exact exit vectors, registered action and world sizes, the state-contract
digest, receipt checks, receipt digest, and the canonical promotion pointer.

Training validation requires:

- every expected rank and no extra rank;
- rank consensus on all model, optimizer, RNG, cursor, and aggregate hashes;
- saved checkpoint components equal loaded components;
- uninterrupted next-step components equal resumed next-step components;
- successful receipt checks after save, after promotion, before load, and
  after load.

DTensor validation requires rank consensus and exact equality of the
reconstructed global tensor. Local shard shapes are checked against each world
size, but local shards are not claimed to be identical across topologies.

Fault validation binds the exact child-exit vector or mutation description,
prohibits DCP load, requires candidate preservation, and requires the canonical
generation pointer to remain byte-identical. Missing/corrupt checkpoint faults
must be rejected by receipt verification during the promotion attempt.

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
comes from the open runner, strict schema, public CI, and reproducibility; the
unsigned artifact alone is not independent attestation.
