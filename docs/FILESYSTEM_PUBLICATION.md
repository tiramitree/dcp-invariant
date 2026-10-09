# Local checkpoint publication

Receipt, worker report, and generation publication require an absent destination. The final
rename uses a native no-replace operation on Windows and Linux; it refuses a
destination that appears after preflight, including an empty generation
directory. Missing native support or a filesystem error stops publication.
The Darwin branch is present but was not executed in this review.

Receipt and worker report staging files are not automatically deleted after a failed create,
write, close, rename, or flush. Such a path may belong to another writer or be
evidence of an interrupted publication. Keep it for review and use a new absent
candidate wrapper for a fresh attempt. A checkpoint with an extra pending file
is rejected by the ordinary checkpoint inventory verifier.

Regular worker, elastic control, and asynchronous snapshot reports use the same
no-replace publication primitive. A report left pending after an error is not a
completed report. Review the failed private run and use a fresh run directory;
do not remove a path that might belong to a competing writer in order to retry.

A directory flush can fail after its rename has taken effect. An exception is
therefore not proof that no file or generation was committed. Recovery must
reload the existing receipt/lineage and use the existing commit/publish recovery
protocol. Do not overwrite an uncertain result merely to retry.

These checks cover local publication conflicts and registered recovery paths.
They are not authorization, protection from a privileged concurrent writer,
proof of physical power-loss durability, or validation of a training run on a
different runtime or device.
