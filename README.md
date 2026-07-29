# DCPInvariant

DCPInvariant is a CPU-only evidence harness for exact state, staged-snapshot,
and generation-lineage invariants around PyTorch Distributed Checkpoint (DCP).

It asks four bounded questions:

1. for a deterministic registered training fixture, do model parameters,
   optimizer momentum, an explicit generator state, and a data cursor produce
   the same next-step state after a checkpoint is loaded under a different
   local process count?
2. does one fixed two-worker PyTorch elastic job reload the same committed DCP
   generation and reproduce that next-step state after its only registered
   restart?
3. after the public DCP staging hook has completed for one fixed two-rank
   ResNet18 workload, does an asynchronous save load the staged pre-mutation
   state rather than a later, deliberately mutated model state?
4. when two cooperating local publishers commit distinct generations from the
   same selected parent, does conditional publication reject the stale second
   writer without changing either committed generation?

The v0.4 target is PyTorch 2.11.0, torchvision 0.26.0, Pillow 12.3.0, and
NumPy 2.4.6 over single-host CPU/Gloo with one or two processes. The suite also
checks DTensor global-tensor equality after 1-to-2 and 2-to-1 resharding, a
deterministic two-publisher lineage race, two process-exit recovery windows,
a separate worker-exit promotion gate, missing native files, and controlled
shard corruption.

## What the suite proves

One run must pass thirteen fixed scenarios:

- asynchronous staged snapshot: fixed ResNet18 at two processes;
- training restart: 1-to-1, 1-to-2, 2-to-1, and 2-to-2 processes;
- elastic restart: one 2-to-2 job launched by
  `python -m torch.distributed.run --standalone --local-addr=127.0.0.1
  --nnodes=1 --nproc-per-node=2 --max-restarts=1`;
- DTensor restore: 1-to-2 and 2-to-1 processes;
- generation lineage: a matched unfenced reference arm and a conditionally
  published two-process arm, plus two process-exit recovery windows;
- expected rejection: child exit, missing metadata, missing shard, and corrupt
  shard.

The lineage scenario first publishes one receipt- and lineage-backed v2 seed
pointer. Two real subprocesses then capture that same selected parent and
commit two distinct immutable children before either publishes. In its private
negative-control arm, fixed A-then-B unfenced pointer writes reproduce the
last-writer overwrite. The public observation binds the starting pointer, both
child lineage digests, the completed commit barrier, fixed publish order, A's
intermediate pointer digest, and the final B pointer. In the matched protected
arm, publication holds a persistent advisory byte lock, re-reads the selected
parent inside the critical section, publishes A, and returns `stale_parent`
for B while preserving B as an unselected committed generation.

One worker also exits abruptly after commit. Another uses a test-only hook
after publication verification and lock release but before
`publish_committed_generation` returns, then exits abruptly. The surviving
supervisor reloads each committed descriptor and verifies completion or
idempotent retry. Forged parents, inconsistent sequence numbers, and
same-receipt/different-lineage reuse are rejected. This is a single-host
ordinary-local-filesystem witness for cooperating processes, not a distributed
compare-and-swap, network-filesystem, hostile-writer, or power-loss result.

The elastic scenario first commits a two-rank checkpoint. On elastic attempt
zero, rank one records the registered injected exit code 91 in a private
canonical marker and exits with that code. The successful restarted group must
report `TORCHELASTIC_RESTART_COUNT=1` and
`TORCHELASTIC_MAX_RESTARTS=1`, reload the same committed generation, and match
the uninterrupted next-step state on both ranks. The launcher observes only
the final agent exit and timeout; the final worker control reports record the
restart count. The launcher does not independently expose
the attempt-zero rank exit vector.

For the exact PyTorch 2.11.0 runtime, the supervisor injects a package-owned
startup directory only into this torchrun agent. Its fail-closed compatibility
guard accepts only the registered distribution/runtime pairs
`2.11.0`/`2.11.0+cpu` and `2.11.0+cpu`/`2.11.0+cpu`, verifies the registered
`_create_tcp_store` source digest and pristine module-local `TCPStore`
reference, then forces only that c10d reference to `use_libuv=False`. The
replacement rejects calls unless the module still binds the exact guarded
`_create_tcp_store` function and the immediate caller frame is that function.
Only after the underlying TCPStore constructor returns does it write a private
canonical attestation. The same controlled environment fixes
`TORCH_DISABLE_SHARE_RDZV_TCP_STORE=1`, so the dynamic rendezvous does not
create a second default-libuv shared store. This supports Windows CPU builds
without libuv while keeping the required world size, restart budget, and
failure mechanism unchanged. The fixed `--local-addr=127.0.0.1` parameter and
worker-side numeric-IP validation keep the worker rendezvous inside the
registered loopback contract. Torchrun workers detect their fixed rank
coordinates and skip the bootstrap. The public observation contains only the
attestation's fixed fields and digest, including the guarded torch
distribution version; its torch runtime version must equal the provenance
runtime version.

Training evidence binds every rank report and every model, optimizer, RNG, and
cursor component digest. It requires both the checkpoint state and the next
training state to match. DTensor evidence measures the reconstructed global
tensor; it does not claim that local shard layouts are identical.

The training, DTensor, elastic, and asynchronous positive checkpoints are
sealed by an ordinary-file inventory and SHA-256 receipt. Promotion first
installs an immutable generation containing a canonical lineage record, then
conditionally publishes a v2 pointer bound to the generation, lineage-record
digest, selected-parent pointer digest, and sequence. The selected parent is
checked again while the local advisory publication lock is held. Positive
loaders use only that committed generation and verify it again after load.
The elastic observation carries separately normalized pre-restart and
post-restart receipt digests and requires them to be equal. The canonical
generation pointer must also remain byte-identical during the failure and
restart. No new
post-failure promotion is attempted. The asynchronous scenario instead seals
and loads its candidate before the suite performs receipt-bound promotion; both
the candidate load and the later committed generation are independently
receipt-verified.

The asynchronous scenario constructs the official
`torchvision.models.resnet18(weights=None)` model without downloading weights
or data. Two CPU/Gloo ranks perform one fixed synthetic SGD step and establish
rank-consensus state digests. A package-owned `FileSystemWriter` subclass calls
the public `FileSystemWriter.stage` hook, records the staged model, optimizer,
and aggregate-state digests, and blocks the public `StorageWriter.write_data`
hook before delegating to native checkpoint I/O. Once both ranks have completed
staging, entered that write gate, and shown that the asynchronous future is
still pending, the main thread changes only `model.conv1.weight` by one fixed
scalar. It does not advance the application cursor, change the optimizer, or
make an explicit process-group collective call while native writing is blocked.

After the gate is released, the native checkpoint is sealed and loaded into a
deliberately different target. A pass requires the staged and loaded model,
optimizer, cursor, and aggregate-state digests to equal the pre-mutation
digests; the post-mutation cursor and optimizer must remain equal to pre, while
the post-mutation model and aggregate state must differ. The receipt is
verified after save, after load, and again around receipt-bound promotion. This
is a fixed correctness witness, not a timing or throughput benchmark.

## Run and verify

Create and activate a virtual environment. Use the command for your shell:

```text
python -m venv .venv
```

```text
# POSIX shell
. .venv/bin/activate
```

```text
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

Then install the project and the exact official CPU runtime:

```text
python -m pip install -e ".[dev,runtime]"
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
```

Run all scenarios into a path that does not already exist:

```text
dcp-invariant run --output-dir artifacts/local --source-revision <40-hex-git-revision>
dcp-invariant verify --artifact-dir artifacts/local
```

The offline `verify` command imports neither PyTorch nor the live worker
modules.

The public artifact contains fixed-schema normalized observations, derived
results, JUnit, provenance, and an unsigned SHA-256 manifest. Native `.metadata`
and `.distcp` files, standalone asynchronous gate-marker and rank-report files,
the raw elastic marker, the raw torchrun-bootstrap attestation, their
filesystem locations, launcher logs, rendezvous values, and environment values
are removed with the private temporary tree before the artifact directory is
created. The validated fixed fields from asynchronous rank reports are embedded
in the normalized observation; the artifact otherwise retains only normalized
fixed fields and canonical-file digests needed by its registered contracts.

Artifact schema v4 has thirteen scenarios and is not byte- or
inventory-compatible with v3, v2, or v1. The v0.4 verifier accepts v4 only.
Use the v0.3 release to verify a v3 artifact, v0.2 for v2, and v0.1 for v1.

## Evidence and privacy boundary

This is an owner-operated pre-alpha project. It has no verified external
users, adoption, independent reproduction, third-party review, production
deployment, or recruiting outcome.

The training and DTensor fixtures use small float64 tensors and binary-exact
values. The asynchronous fixture uses the fixed FP32 ResNet18 construction,
synthetic input, SGD step, public staging hook, write gate, and targeted
mutation described above. A pass does not establish bitwise determinism or
snapshot behavior for arbitrary models, optimizers, datasets, writers, kernels,
failure points, or process topologies.

DCPInvariant does **not** establish multi-node recovery, membership changes,
GPU/NCCL or FSDP correctness, network-filesystem or power-loss durability,
performance, recovery time, throughput, high availability, hostile-checkpoint
safety, production reliability, model quality, framework superiority, or
official PyTorch certification.

PyTorch DCP records its checkpoint identifier in `.metadata`. Workers therefore
run with isolated temporary HOME, user, and temp values and receive only
`checkpoint-one`, `checkpoint-two`, or `checkpoint-async` as relative
identifiers. Public evidence contains no native checkpoint, tensor values,
absolute path, real username, hostname, process ID, port, environment, timing,
byte profile, or worker log.

Public CI scans the clean checkout and a frozen closure rooted at refs and HEAD
present in that checkout for generic secret, contact, and user-path patterns.
The closure includes commit and tag messages, deleted-but-reachable blobs, and
full paths reconstructed for every reachable commit. Opaque blobs, Git LFS
pointers, gitlinks, shallow or partial history, replacements, grafts,
alternates, missing objects, changing refs or HEAD, and unregistered object
formats or inventory bounds fail closed.

A separate raw-object audit checks every reachable commit author/committer and
every reachable annotated-tag tagger, including nested tags and tag objects
carried by a non-tag ref, against the registered pseudonymous GitHub
identities. Exact private identity literals are intentionally not committed as
CI configuration.

Before a release, the owner runs `scripts/owner_release_privacy.py` once over
the current tree, frozen fetched closure, release notes, and exactly four
assets held outside the checkout: one wheel, one source distribution, one
schema-v4 evidence archive, and one `SHA256SUMS` file that binds the other
three. The gate requires an unsigned annotated v0.4 tag that points directly
to current `HEAD` and is already inside the frozen ref closure. The frozen
ordinary-file working tree must match that commit's complete blob path and
byte inventory, so tracked modifications, deletions, and untracked files fail
the gate. It requires the evidence archive's fixed 13-scenario inventory and
binds its `source_revision` to that same commit. The wheel's package and
license bytes must equal the frozen source snapshot; the source distribution
must contain that complete source snapshot byte-for-byte plus its generated
metadata.

Exact ZIP, gzip, and TAR bytes and ASCII-only member names are scanned before
logical members, canonical release notes, and checksums are validated. The
gate reasserts the source-file bytes and metadata, refs, `HEAD`, reachable
objects, identities, assets, notes, and external denylist after the full
audit. The private denylist is derived from protected local identity sources
and is never committed. Only bounded counts and cryptographic inventory
digests enter the local release record.

These gates do not claim inspection or erasure of GitHub-managed pull-request
refs not fetched into the checkout, unreachable server objects, cached views,
reflog-only objects, existing release assets, LFS object storage, submodule
repositories, or external forks.

The owner also performed a bounded retrospective audit of the v0.1.0 through
v0.3.0 Release surfaces. See
[release integrity and audit scope](docs/release-integrity.md) for the result
and its limits.

The lineage fixture adds no claims about external adoption, production
deployment, multi-host coordination, or failure-free behavior outside its
registered interleaving and recovery windows.

See [the evidence schema](docs/evidence-schema-v4.md),
[claim boundaries](docs/claim-boundaries.md), and
[security model](docs/security-model.md).

## License

Original code is licensed under Apache-2.0. PyTorch, torchvision, NumPy, and
Pillow remain under their own licenses and are not redistributed by this
project.
