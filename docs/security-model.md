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

The loader verifies the committed checkpoint before and after DCP load. A
concurrent change therefore cannot produce passing evidence, although this is
not a defense against an adversary capable of perfectly racing and restoring
bytes.

On POSIX, this project fsyncs the receipt and pointer files it writes and their
affected directories. It does not control how PyTorch persists native shard or
metadata files. On Windows it claims process-visible atomic replacement only.
No operating system has a claimed end-to-end power-loss durability result.

## Process and privacy isolation

Workers receive a minimal environment. HOME, temporary directories, and user
variables are replaced with isolated or fixed pseudonymous values. Credentials
and proxy variables are not inherited. Worker stdout and stderr are discarded;
only fixed-schema rank reports are consumed.

The suite uses a loopback TCP rendezvous. A local port is selected immediately
before launch, so a local port-race failure remains possible; such a failure
causes the suite to stop without evidence publication.

Registered workers do not spawn child trees. Timeout cleanup terminates the
direct worker processes; the implementation does not claim general process-tree
containment for arbitrary worker modules.

## Public artifact

Native checkpoints live only under an automatically removed temporary root.
The normalized public artifact is created only after removal is confirmed.
The manifest is an unsigned integrity closure, not authentication.

The negative-control scenarios seed a canonical pointer record and require its
bytes to remain unchanged. They do not claim that this sentinel points to an
available older checkpoint; usable committed-generation recovery is exercised
only by the positive restart scenarios.
