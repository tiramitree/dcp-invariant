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
5. renames the candidate wrapper into `committed/<receipt-sha256>` and verifies
   it again before atomically replacing `LATEST.json`.

The loader verifies the committed checkpoint before and after DCP load. The
elastic scenario publishes both normalized receipt digests and requires exact
equality, while the generation pointer is read before launch and after the
successful restarted load and must remain byte-identical. A concurrent change
therefore cannot produce passing evidence, although this is not a defense
against an adversary capable of perfectly racing and restoring bytes.

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

PyTorch 2.11.0 Windows CPU builds may lack libuv while the standalone c10d
rendezvous path still defaults to it. The elastic supervisor does not inherit a
user `PYTHONPATH`; it supplies one package-owned bootstrap directory only to
the exact torchrun subprocess. Before changing anything, that bootstrap
requires exact torch 2.11.0 metadata/runtime, an exact registered source digest
for `_create_tcp_store`, and the pristine c10d-module `TCPStore` reference. It
then replaces only that module reference and forces `use_libuv=False`. The
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
attested torch version must equal the public provenance runtime version. The
public artifact carries only normalized fixed fields and the digest. A
missing, stale, altered, wrong-version, wrong-source, or wrong-caller
attestation stops publication.

If the elastic launch times out, no artifact is created. On POSIX the
supervisor kills its new process group. On Windows it requests `taskkill /T /F`
for the fixed agent tree and falls back to killing the agent if that command
fails. These failure-path actions are best-effort cleanup, not proof of general
Windows process-tree containment or detached-descendant cleanup.

## Public artifact

Native checkpoints, the raw elastic marker, the raw bootstrap attestation,
torchrun temporary data, and machine-specific launch details live only under
an automatically removed temporary root. The normalized public artifact is created only after removal is
confirmed.

The manifest is an unsigned integrity closure, not authentication. The v2
verifier intentionally rejects the v1 fixed inventory; use the v0.1 verifier
for v1 artifacts.

The negative-control scenarios seed a canonical pointer record and require its
bytes to remain unchanged. They do not claim that this sentinel points to an
available older checkpoint; usable committed-generation recovery is exercised
only by the positive restart scenarios.
