# Security and failure model

## Trusted inputs

The live suite loads only checkpoints it created during the same run. PyTorch
DCP metadata uses pickle-compatible structures, so the receipt verifier is not
a malicious-checkpoint sandbox.

The offline artifact verifier does not import PyTorch. It accepts only a fixed
small inventory of ordinary files with bounded sizes, canonical JSON, strict
field sets, and no links or reparse points.

## Checkpoint integrity

Before promotion, the runner:

1. requires one registered relative checkpoint name;
2. rejects links, hard-link aliases, unexpected entries, duplicate shard
   coordinates, oversized files, and a missing metadata or shard set;
3. hashes every ordinary native file twice and binds the inventory to a
   canonical receipt;
4. requires exact verifier success and derives the generation name from the
   receipt bytes;
5. writes a canonical lineage record binding that generation to the logical
   checkpoint identifier, selected-parent pointer digest, and sequence, then
   renames the candidate wrapper into `committed/<receipt-sha256>` and verifies
   the committed receipt and lineage again;
6. acquires the persistent local `.LATEST.commit.lock`, re-reads and validates
   the selected pointer while holding the lock, rejects a stale parent, writes
   a unique pointer temporary, atomically replaces `LATEST.json`, and re-reads
   the published pointer before releasing the lock.

Commit and publication are separate operations. A committed generation can
therefore remain as an unselected orphan after a stale publication attempt.
This is deliberate: it preserves immutable work and makes the stale outcome
auditable. A caller can reload a committed descriptor by generation digest
after losing its in-memory return value, provided it retained or can
reconstruct the registered receipt verifier and logical checkpoint identifier.

An exact same-receipt/same-lineage retry reuses the existing committed
generation and leaves the retry candidate wrapper in place for audit. The
lower-level `commit_candidate` result exposes `reused`; the compatibility
`promote_candidate` wrapper continues to return only the committed target
path. Same-receipt reuse under a different lineage fails closed.

The v2 pointer has exactly five canonical fields:
`generation`, `lineage_sha256`, `parent_pointer_sha256`, `pointer_schema`, and
`sequence`. Its canonical byte digest is the parent version captured by a
child. A v2 pointer is accepted only if its target is an ordinary direct child
of `committed/`, the target receipt hashes to `generation`, and the embedded
lineage record hashes to `lineage_sha256` and repeats the pointer fields. A v1
pointer can be read only as a legacy parent anchor; new publication emits v2.

The publication lock is an advisory one-byte OS lock on a persistent ordinary
file. POSIX uses `flock`; Windows uses `msvcrt.locking` at offset zero. The
file is initialized with one NUL byte and is never unlinked during normal
operation. Acquisition is bounded and retries fail closed. The protected
contract assumes one host, an ordinary local filesystem, and all writers using
this protocol. It does not claim network-filesystem locking, distributed
consensus, hostile-writer exclusion, or end-to-end power-loss durability.

The registered lineage scenario publishes a real backed seed pointer, then
uses two real subprocesses. Both capture that same parent and commit before
fixed A-then-B publication. A private unfenced reference arm deterministically
records A as the intermediate pointer and ends on B. The protected arm ends on
A and returns `stale_parent` for B while preserving both committed trees
unchanged. One worker exits abruptly after commit. A second uses a private
test-only callback that runs after the new pointer was re-read successfully and
the publication lock was released, but before the publication function
returns; it exits abruptly there. The surviving supervisor reloads the
committed descriptor and verifies completion or idempotent retry. The scenario
does not inject a failure during the lineage-record write itself.

The training, DTensor, and elastic positive loaders verify the committed
checkpoint before and after DCP load. The elastic scenario publishes both
normalized receipt digests and requires exact equality, while the generation
pointer is read before launch and after the successful restarted load and must
remain byte-identical. A concurrent change therefore cannot produce passing
evidence, although this is not a defense against an adversary capable of
perfectly racing and restoring bytes.

The asynchronous scenario calls the public `FileSystemWriter.stage` hook,
records staged state digests, and then blocks the public
`StorageWriter.write_data` hook before it delegates to native I/O. Both ranks
must record stage completion and gate entry exactly once, and the future must
still be pending, before the main thread applies the one registered model-only
mutation. The main thread makes no explicit process-group collective call while
writing is blocked. A `finally` path releases the gate and joins the future so
a local validation failure does not intentionally strand the writer thread.
This is a deterministic test gate around public APIs, not a claim about
scheduling or durability in an arbitrary application. Its native candidate is
sealed, loaded, and verified again after load before the suite performs
receipt-bound promotion and verifies the committed generation.

On POSIX, this project fsyncs the receipt and pointer files it writes and their
affected directories. It does not control how PyTorch persists native shard or
metadata files. On Windows it claims process-visible atomic replacement only.
No operating system has a claimed end-to-end power-loss durability result.

## Process and privacy isolation

Workers receive a minimal environment. HOME, temporary directories, and user
variables are replaced with isolated or fixed pseudonymous values. Credentials
and proxy variables are not inherited. Ordinary worker stdout and stderr are
discarded; only fixed-schema reports are consumed.

The non-elastic scenarios use a loopback TCP rendezvous. A local port is
selected immediately before launch, so a local port-race failure remains
possible; such a failure stops the suite without evidence publication.

The elastic scenario uses the real `torch.distributed.run` local elastic agent,
which creates a bounded agent-to-worker process tree. It does not use
torchrun's redirect feature, which is unavailable on Windows; the outer
supervisor discards launcher output instead. A passing artifact requires a
normal agent exit after two final rank reports. The private attempt-zero marker
contains only fixed injected-failure fields, and the public artifact includes
only their normalized derivative and digest.


The public source gate detects generic credential, contact, private-key, and
user-path patterns in both the clean checkout and a frozen closure rooted at
fetched refs plus HEAD. It scans commit and tag payloads, each unique reachable
blob, and full paths for every reachable commit. It rejects opaque blobs, Git
LFS pointers, gitlinks, shallow or partial history, replace refs, grafts,
alternates, missing objects, unsupported object formats, changing refs or
HEAD, and bounded-inventory violations. Replacement, graft, lazy-fetch, and
network protocol resolution are disabled in its Git subprocesses.

A separate raw-object identity gate permits only the registered pseudonymous
GitHub identities in every reachable commit author/committer header and every
reachable annotated-tag tagger header. This includes nested tag objects and
tag objects carried by non-tag refs. Their command outputs report only bounded
counts, digests, coarse scopes, and finding classes: never source paths,
identity values, ref names, object identifiers, archive names, or matched
payloads.

Exact real-name, contact, and local-home literals are held in an external
private denylist, never committed to the repository. The owner-only combined
release gate requires one unchanged denylist, one fixed release-notes file,
and exactly four fixed-name assets outside the source root: wheel, source
distribution, schema-v4 evidence archive, and `SHA256SUMS`. It freezes the
source tree's file bytes and metadata, fetched closure, raw pseudonymous
identity inventory, asset bytes, notes, and denylist, then reasserts those
boundaries after every archive and evidence check. An unsigned annotated
release tag must already be in that closure and point directly to current
`HEAD`. The frozen working tree's complete ordinary-file path and byte
inventory must equal that commit's blob tree; tracked modifications,
deletions, and untracked files are rejected. The evidence asset must have the
exact 13-scenario v4 inventory and bind that commit as its `source_revision`.
The wheel's package and license bytes and the source distribution's complete
source inventory must match the frozen source snapshot. Archive member names
are ASCII-only, and the checksum file must bind exactly the other three
assets. An in-tree input, stale build, stale evidence artifact, or concurrent
change fails closed. Public CI cannot reconstruct the private literals and is
not represented as doing so.

Neither gate claims inspection or erasure of unfetched GitHub pull-request
refs, unreachable server or reflog-only objects, cached views, existing
release assets, LFS object storage, submodule repositories, or external forks.
The separate [release integrity record](release-integrity.md) documents the
owner-side retrospective check of the v0.1.0 through v0.3.0 Release surfaces
without treating it as external review.

PyTorch 2.11.0 Windows CPU builds may lack libuv while the standalone c10d
rendezvous path still defaults to it. The elastic supervisor does not inherit a
user `PYTHONPATH`; it supplies one package-owned bootstrap directory only to
the exact torchrun subprocess. Before changing anything, that bootstrap
accepts only the registered distribution/runtime pairs
`2.11.0`/`2.11.0+cpu` and `2.11.0+cpu`/`2.11.0+cpu`. It also requires the exact
registered source digest for `_create_tcp_store` and the pristine c10d-module
`TCPStore` reference, then replaces only that module reference and forces
`use_libuv=False`. The
replacement rejects every call unless the module still binds the guarded exact
`_create_tcp_store` function and the immediate caller frame is that function.
The private attestation is written only after the underlying TCPStore
constructor returns. The supervisor also fixes
`TORCH_DISABLE_SHARE_RDZV_TCP_STORE=1`, an exact PyTorch switch that prevents
dynamic rendezvous from creating a second default-libuv shared TCPStore. Both
the attestation and final worker control reports bind that switch as enabled.
The exact command also fixes `--local-addr=127.0.0.1`; worker-side environment
parsing requires a numeric IP and verifies its loopback property. Public
control reports expose only `loopback_rendezvous:true`, never the address or
port. Inherited workers identify the complete fixed rank-coordinate
environment and no-op before importing torch. The suite requires the
exclusive canonical private attestation's exact content and digest, and the
attested torch distribution version is preserved as a normalized public field,
and the attested runtime version must equal the public provenance runtime
version. The public artifact carries only normalized fixed fields and the
digest. A
missing, stale, altered, wrong-version, wrong-source, or wrong-caller
attestation stops publication.

If the elastic launch times out, no artifact is created. On POSIX the
supervisor kills its new process group. On Windows it requests `taskkill /T /F`
for the fixed agent tree and falls back to killing the agent if that command
fails. These failure-path actions are best-effort cleanup, not proof of general
Windows process-tree containment or detached-descendant cleanup.

## Public artifact

Native checkpoints, standalone asynchronous gate-marker and rank-report files,
raw lineage coordination and worker records, the raw elastic marker, the raw
bootstrap attestation, torchrun temporary data, and machine-specific launch
details live only under an automatically removed temporary root. The
normalized public artifact is created only after removal is confirmed. Public
observations embed only validated fixed fields, registered runtime versions,
fixed booleans, state and tree digests, workload declarations, normalized
worker outcomes, and receipt/promotion evidence; they contain no raw tensor,
timing, checkpoint byte size, path, host, process identifier, port,
environment, or log.

The manifest is an unsigned integrity closure, not authentication. The v4
verifier intentionally rejects the v3, v2, and v1 fixed inventories; use their
matching v0.3, v0.2, or v0.1 verifier for historical artifacts.

The receipt-fault scenarios seed a real synthetic committed generation and
require its backed v2 pointer to remain byte-identical. This establishes only
the exact seed's availability within the fixture, not arbitrary historical
retention. The lineage scenario separately verifies preservation of its stale
committed child and both registered recovery windows.
