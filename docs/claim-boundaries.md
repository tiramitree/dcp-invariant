# Claim boundaries

## Supported by a passing v3 artifact

- PyTorch 2.11.0 CPU/Gloo executed the fixed single-host fixtures.
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
- The fixed two-rank asynchronous workload constructed
  `torchvision.models.resnet18(weights=None)` with the registered torchvision
  0.26.0 and Pillow 12.3.0 runtime, synthetic input, and no weight download.
- Both asynchronous ranks completed the public `FileSystemWriter.stage` hook
  exactly once, entered the public `StorageWriter.write_data` gate, and
  observed the asynchronous future still pending before the registered
  mutation.
- Staged model, optimizer, and aggregate-state digests equaled the
  pre-mutation digests. The main thread then changed only
  `model.conv1.weight` by the fixed registered scalar while leaving the
  application cursor and optimizer state unchanged.
- Loading into a deliberately different target reproduced the pre-mutation
  cursor, model, optimizer, and aggregate state, not the post-mutation model
  state. Direct loaded-model evidence was checked before applying the loaded
  state to the target model.
- The asynchronous checkpoint receipt was verified after save, after load, and
  around receipt-bound promotion.
- The separate registered child exit prevented promotion.
- Missing metadata, a missing shard, and a one-byte shard corruption were
  rejected before DCP load and left the pre-existing sentinel pointer record
  byte-identical.
- The public artifact contained no native checkpoint, standalone asynchronous
  gate-marker or rank-report file, raw elastic marker, bootstrap attestation,
  private-record path, launcher log, rendezvous value, environment value,
  timing, or byte profile; it embedded only the validated fixed fields from
  asynchronous rank reports.

These claims apply only to the exact fixture, versions, process counts,
single injected failure point, and protocol bound by the artifact and its
referenced source revision.

## Not supported

- arbitrary-model or arbitrary-optimizer determinism;
- equality of local DTensor shards across different topologies;
- more than two processes, multiple hosts, elastic membership changes, scale
  up, or scale down;
- failures at arbitrary training or checkpoint phases;
- asynchronous-snapshot behavior for an arbitrary model, optimizer, stager,
  writer, mutation, or training loop;
- GPU, NCCL, FSDP, or device-failure recovery;
- latency, save overlap, recovery time, throughput, scalability, memory use,
  checkpoint size, or resource-efficiency conclusions;
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
staged-snapshot, elastic-launch, and control-plane boundaries. ResNet18 is a
recognized fixed model construction here, but this synthetic execution is not
evidence of training quality, production behavior, or industry-workload
performance.

## Schema compatibility

The v0.3 verifier accepts evidence schema v3 only. It does not validate the
eleven-scenario v2 or ten-scenario v1 inventory. Those protocols remain
documented separately and require their matching v0.2 or v0.1 verifier.
