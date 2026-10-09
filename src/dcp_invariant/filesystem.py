"""Atomic start-absent publication within one local filesystem."""

from __future__ import annotations

import errno
import os
import sys
from pathlib import Path


def rename_no_replace(source: Path, destination: Path) -> None:
    """Rename a file or directory without replacing any destination entry."""

    if os.name == "nt":
        os.rename(source, destination)
        return

    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    encoded_source = os.fsencode(source)
    encoded_destination = os.fsencode(destination)
    if sys.platform.startswith("linux"):
        try:
            rename = libc.renameat2
        except AttributeError as error:
            raise OSError(
                errno.ENOSYS, "atomic no-replace publication unavailable"
            ) from error
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        result = rename(-100, encoded_source, -100, encoded_destination, 1)
    elif sys.platform == "darwin":
        try:
            rename = libc.renamex_np
        except AttributeError as error:
            raise OSError(
                errno.ENOSYS, "atomic no-replace publication unavailable"
            ) from error
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        result = rename(encoded_source, encoded_destination, 0x00000004)
    else:
        raise OSError(errno.ENOSYS, "atomic no-replace publication unavailable")
    if result != 0:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number), str(destination))
