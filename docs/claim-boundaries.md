# Claim boundaries

## Supported by a passing artifact

- PyTorch 2.11.0 CPU/Gloo executed the fixed single-host fixture.
- DDP training state was saved at one or two processes and loaded at one or
  two processes.
- Model, SGD momentum, explicit CPU generator, and data cursor matched at the
  checkpoint and after the next registered training step.
- The reconstructed DTensor global tensor matched after 1-to-2 and 2-to-1
  resharding.
- The registered child exit prevented promotion.
- Missing metadata, a missing shard, and a one-byte shard corruption were
  rejected before DCP load and left the pre-existing sentinel pointer record
  byte-identical.
- The public artifact contained no native checkpoint payload.

These claims apply only to the exact fixture, versions, process counts, and
protocol bound by the artifact and its referenced source revision.

## Not supported

- arbitrary-model or arbitrary-optimizer determinism;
- equality of local DTensor shards across different topologies;
- more than two processes, multiple hosts, GPU, NCCL, FSDP, or elastic jobs;
- latency, throughput, scalability, or resource-efficiency conclusions;
- network filesystems, cloud object stores, or power-loss durability;
- availability or validity of an older generation during the negative-control
  cases; those cases establish only that the pre-existing canonical pointer
  record remains byte-identical;
- recovery from hostile or untrusted pickle-compatible metadata;
- model quality, production reliability, framework superiority, or official
  PyTorch certification;
- external adoption, independent reproduction, or third-party review.

Synthetic tensors and controlled failures support only these state-integrity
and control-plane boundaries. They are not evidence of industry workload
performance.
