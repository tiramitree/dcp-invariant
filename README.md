# DCPInvariant

DCPInvariant is a CPU-only evidence harness for exact restart invariants around
PyTorch Distributed Checkpoint (DCP).

It asks one bounded question:

> For a deterministic registered training fixture, do model parameters,
> optimizer momentum, an explicit generator state, and a data cursor produce
> the same next-step state after a checkpoint is loaded under a different
> local process count?

The v0.1 target is PyTorch 2.11.0 over single-host CPU/Gloo with one or two
processes. The suite also checks DTensor global-tensor equality after 1-to-2
and 2-to-1 resharding, worker-exit promotion gating, missing native files, and
controlled shard corruption.

## What the suite proves

One run must pass ten fixed scenarios:

- training restart: 1-to-1, 1-to-2, 2-to-1, and 2-to-2 processes;
- DTensor restore: 1-to-2 and 2-to-1 processes;
- expected rejection: child exit, missing metadata, missing shard, and corrupt
  shard.

Training evidence binds every rank report and every model, optimizer, RNG, and
cursor component digest. It requires both the checkpoint state and the next
training state to match. DTensor evidence measures the reconstructed global
tensor; it does not claim that local shard layouts are identical.

A checkpoint is sealed by an ordinary-file inventory and SHA-256 receipt,
promoted under the receipt digest, loaded only from that committed generation,
and verified again after load. A failed worker or receipt check cannot update
the generation pointer.

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
python -m pip install -e ".[dev]"
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

Run all scenarios into a path that does not already exist:

```text
dcp-invariant run --output-dir artifacts/local --source-revision <40-hex-git-revision>
dcp-invariant verify --artifact-dir artifacts/local
```

The offline `verify` command imports neither PyTorch nor the live worker
module.

The public artifact contains fixed-schema normalized observations, derived
results, JUnit, provenance, and an unsigned SHA-256 manifest. Native `.metadata`
and `.distcp` files are removed before the artifact directory is created.

## Evidence and privacy boundary

This is an owner-operated pre-release project. It has no verified external
users, adoption, independent reproduction, third-party review, production
deployment, or recruiting outcome.

The fixture uses small float64 tensors and binary-exact values. A pass does not
establish bitwise determinism for arbitrary models, optimizers, datasets,
kernels, or process topologies.

DCPInvariant does **not** establish multi-node recovery, GPU/NCCL or FSDP
correctness, network-filesystem or power-loss durability, performance,
throughput, high availability, hostile-checkpoint safety, production
reliability, model quality, framework superiority, or official PyTorch
certification.

PyTorch DCP records its checkpoint identifier in `.metadata`. Workers therefore
run with isolated temporary HOME, user, and temp values and receive only
`checkpoint-one` or `checkpoint-two` as relative identifiers. Public evidence
contains no native checkpoint, tensor values, absolute path, real username,
hostname, process ID, port, environment, or worker log.

See [the evidence schema](docs/evidence-schema-v1.md),
[claim boundaries](docs/claim-boundaries.md), and
[security model](docs/security-model.md).

## License

Original code is licensed under Apache-2.0. PyTorch remains under its own
BSD-3-Clause license and is not redistributed by this project.
