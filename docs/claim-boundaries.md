# Claim boundaries

## Supported by a passing v2 artifact

- PyTorch 2.11.0 CPU/Gloo executed the fixed single-host fixture.
- DDP training state was saved at one or two processes and loaded at one or
  two processes.
- Model, SGD momentum, explicit CPU generator, and data cursor matched at the
  checkpoint and after the next registered training step.
- One fixed two-worker job was launched through
  `python -m torch.distributed.run` with `--standalone`,
  `--local-addr=127.0.0.1`, `--nproc-per-node=2`, and `--max-restarts=1`.
- The exact PyTorch 2.11.0 c10d source guard passed, its module-local
  `TCPStore` constructor returned with `use_libuv=False` from a call whose
  immediate caller frame and current module binding both matched the guarded
  `_create_tcp_store` function. The shared-rendezvous TCPStore opt-out was
  fixed and repeated by both final worker reports. Both reports recorded only
  the fixed `loopback_rendezvous:true` derivative after validating the actual
  worker environment as a numeric loopback IP. The private canonical bootstrap
  attestation matched its normalized public fields and digest, including one
  of the two registered torch distribution/runtime pairs; its runtime version
  equaled the provenance runtime version.
- Rank one recorded the registered attempt-zero injected exit code 91 in the
  private marker; the final two reports both recorded restart count one and
  maximum restarts one. The launcher did not independently expose an
  attempt-zero child exit vector.
- The restarted group loaded the pre-failure committed generation and matched
  the uninterrupted next-step state. The normalized receipt digests before and
  after restart were equal, and the generation pointer remained byte-identical.
- No post-failure promotion was attempted in the elastic scenario.
- The reconstructed DTensor global tensor matched after 1-to-2 and 2-to-1
  resharding.
- The separate registered child exit prevented promotion.
- Missing metadata, a missing shard, and a one-byte shard corruption were
  rejected before DCP load and left the pre-existing sentinel pointer record
  byte-identical.
- The public artifact contained no native checkpoint, raw marker or bootstrap
  attestation, private-record path, launcher log, rendezvous value, or
  environment value.

These claims apply only to the exact fixture, versions, process counts,
single injected failure point, and protocol bound by the artifact and its
referenced source revision.

## Not supported

- arbitrary-model or arbitrary-optimizer determinism;
- equality of local DTensor shards across different topologies;
- more than two processes, multiple hosts, elastic membership changes, scale
  up, or scale down;
- failures at arbitrary training or checkpoint phases;
- GPU, NCCL, FSDP, or device-failure recovery;
- latency, recovery time, throughput, scalability, or resource-efficiency
  conclusions;
- network filesystems, cloud object stores, machine loss, or power-loss
  durability;
- high availability, production reliability, or general detached-process-tree
  containment;
- availability or validity of an older generation during the negative-control
  cases; those cases establish only that the pre-existing canonical pointer
  record remains byte-identical;
- recovery from hostile or untrusted pickle-compatible metadata;
- equivalence to an unmodified libuv rendezvous path or compatibility beyond
  the exact guarded PyTorch 2.11.0 c10d source;
- model quality, framework superiority, or official PyTorch certification;
- external adoption, independent reproduction, or third-party review.

Synthetic tensors and controlled failures support only these state-integrity,
elastic-launch, and control-plane boundaries. They are not evidence of
industry-workload performance.

## Schema compatibility

The v0.2 verifier accepts evidence schema v2 only. It does not validate the
ten-scenario v1 inventory. The v1 protocol remains documented separately and
requires the v0.1 verifier.
