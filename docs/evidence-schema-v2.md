# Evidence artifact schema v2

The v2 artifact is a normalized, fixed-inventory report. It excludes native
PyTorch checkpoint files, tensor values, raw worker or launcher logs, the raw
elastic marker and its path, absolute paths, real user or host identity,
process identifiers, ports, rendezvous values, environment values, and the raw
torchrun-bootstrap attestation or its path.

## Compatibility

Schema v2 is not inventory-compatible with v1. The v0.2 verifier accepts v2
only. The v1 protocol remains in `evidence-schema-v1.md` and requires the v0.1
verifier.

## Inventory

The only accepted entries are:

- `provenance.json`
- `summary.json`
- `observations/<scenario>.json` for all eleven registered scenarios
- `results/<scenario>.json` for all eleven registered scenarios
- `junit.xml`
- `manifest.sha256`

The scenarios are:

- `dtensor_1_to_2`
- `dtensor_2_to_1`
- `training_1_to_1`
- `training_1_to_2`
- `training_2_to_1`
- `training_2_to_2`
- `elastic_restart_2_to_2`
- `rank_exit_no_promotion`
- `missing_metadata`
- `missing_shard`
- `corrupt_shard`

All JSON is one compact, key-sorted UTF-8 record terminated by LF. JUnit has
one passing case per registered scenario and contains no durations or output
streams.

`provenance.json` binds the exact CPython, PyTorch, and NumPy versions used by
the run together with the referenced 40-hex source revision.

## Observation-to-result binding

Results are derived by the verifier; callers cannot provide result hashes
separately.

Each positive observation includes the complete ordered rank-report set,
registered action and world sizes, state-contract digest, receipt checks,
receipt digest, and canonical promotion pointer.

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

The elastic observation additionally requires:

- outer launcher exit zero without timeout;
- an exact normalized derivative and canonical-file digest for the private
  torchrun-bootstrap attestation. It proves that the guarded exact PyTorch
  2.11.0 c10d module-local `TCPStore` constructor returned with
  `use_libuv=False` from an invocation whose immediate caller frame and current
  module binding both matched the registered `_create_tcp_store` function. The
  exact shared-rendezvous-store opt-out was fixed to true and repeated by both
  final worker control reports. The attested torch version must equal the
  provenance runtime version;
- a fixed `loopback_rendezvous:true` in both final control reports, derived
  from worker-side numeric-IP parsing and loopback validation of the actual
  rendezvous environment without publishing its address or port;
- a fixed normalized derivative of the private marker naming the registered
  injected rank, attempt, and exit code, plus its canonical-file digest;
- two final elastic reports with rank set `{0,1}`, restart count one, maximum
  restarts one, and the same marker digest;
- exact checkpoint and uninterrupted-next-step equality after the restarted
  load;
- equal pre-restart and post-restart receipt digests;
- byte-identical generation pointers before and after restart;
- reuse of the pre-failure committed generation and no post-failure promotion.

The marker establishes what rank one registered and injected before exiting.
The successful final restart count establishes that the elastic agent restarted
the group. The outer launcher does not independently expose an attempt-zero
rank exit vector.

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
comes from the open runner, strict schema, and any independently inspectable CI
or reproduction that actually ran them. A configured but unrun workflow does
not add evidence, and the unsigned artifact alone is not independent
attestation.
