# DCPInvariant

DCPInvariant is a CPU-only evidence harness for exact restart invariants around
PyTorch Distributed Checkpoint (DCP).

It asks two bounded questions:

1. for a deterministic registered training fixture, do model parameters,
   optimizer momentum, an explicit generator state, and a data cursor produce
   the same next-step state after a checkpoint is loaded under a different
   local process count?
2. does one fixed two-worker PyTorch elastic job reload the same committed DCP
   generation and reproduce that next-step state after its only registered
   restart?

The v0.2 target is PyTorch 2.11.0 with NumPy 2.4.6 over single-host CPU/Gloo
with one or two processes. The suite also checks DTensor global-tensor equality
after 1-to-2 and 2-to-1 resharding, a separate worker-exit promotion gate,
missing native files, and controlled shard corruption.

## What the suite proves

One run must pass eleven fixed scenarios:

- training restart: 1-to-1, 1-to-2, 2-to-1, and 2-to-2 processes;
- elastic restart: one 2-to-2 job launched by
  `python -m torch.distributed.run --standalone --local-addr=127.0.0.1
  --nnodes=1 --nproc-per-node=2 --max-restarts=1`;
- DTensor restore: 1-to-2 and 2-to-1 processes;
- expected rejection: child exit, missing metadata, missing shard, and corrupt
  shard.

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

A checkpoint is sealed by an ordinary-file inventory and SHA-256 receipt,
promoted under the receipt digest, loaded only from that committed generation,
and verified again after load. The elastic observation carries separately
normalized pre-restart and post-restart receipt digests and requires them to be
equal. The canonical generation pointer must also remain byte-identical during
the failure and restart. No new post-failure promotion is attempted.

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
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
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
and `.distcp` files, the raw elastic marker, the raw torchrun-bootstrap
attestation, their filesystem locations, launcher logs, rendezvous values, and
environment values are removed with the private temporary tree before the
artifact directory is created. The artifact contains normalized fixed fields
and canonical-file digests for both private records.

Artifact schema v2 has eleven scenarios and is not byte- or inventory-compatible
with v1. The v0.2 verifier accepts v2 only. Use the v0.1 release to verify a v1
artifact.

## Evidence and privacy boundary

This is an owner-operated pre-alpha project. It has no verified external
users, adoption, independent reproduction, third-party review, production
deployment, or recruiting outcome.

The fixture uses small float64 tensors and binary-exact values. A pass does not
establish bitwise determinism for arbitrary models, optimizers, datasets,
kernels, failure points, or process topologies.

DCPInvariant does **not** establish multi-node recovery, membership changes,
GPU/NCCL or FSDP correctness, network-filesystem or power-loss durability,
performance, recovery time, throughput, high availability, hostile-checkpoint
safety, production reliability, model quality, framework superiority, or
official PyTorch certification.

PyTorch DCP records its checkpoint identifier in `.metadata`. Workers therefore
run with isolated temporary HOME, user, and temp values and receive only
`checkpoint-one` or `checkpoint-two` as relative identifiers. Public evidence
contains no native checkpoint, tensor values, absolute path, real username,
hostname, process ID, port, environment, or worker log.

See [the evidence schema](docs/evidence-schema-v2.md),
[claim boundaries](docs/claim-boundaries.md), and
[security model](docs/security-model.md).

## License

Original code is licensed under Apache-2.0. PyTorch remains under its own
BSD-3-Clause license and is not redistributed by this project.
