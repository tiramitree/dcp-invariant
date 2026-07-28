"""Exact PyTorch 2.11 c10d startup guard for builds without libuv."""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
import stat
import sys
from pathlib import Path

from dcp_invariant.elastic_contract import (
    BOOTSTRAP_ATTESTATION_ENV,
    BOOTSTRAP_ATTESTATION_NAME,
    BOOTSTRAP_SHARED_STORE_ENV,
    EXPECTED_C10D_CREATE_TCP_STORE_SHA256,
    bootstrap_attestation_payload,
    is_registered_torch_version_pair,
)


def _ordinary_directory(path: Path) -> None:
    value = path.lstat()
    attributes = getattr(value, "st_file_attributes", 0)
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        not stat.S_ISDIR(value.st_mode)
        or path.is_symlink()
        or bool(attributes & reparse)
    ):
        raise RuntimeError("torchrun bootstrap parent is not an ordinary directory")


def _attestation_path() -> Path:
    raw = os.environ.get(BOOTSTRAP_ATTESTATION_ENV)
    if type(raw) is not str or not raw:
        raise RuntimeError("torchrun bootstrap attestation path is absent")
    path = Path(raw)
    if not path.is_absolute() or path.name != BOOTSTRAP_ATTESTATION_NAME:
        raise RuntimeError("torchrun bootstrap attestation path is invalid")
    _ordinary_directory(path.parent)
    if path.exists() or path.is_symlink():
        raise RuntimeError("torchrun bootstrap attestation must start absent")
    return path


def _guard_exact_source(torch, c10d_rendezvous_backend) -> tuple[str, str]:
    if os.environ.get(BOOTSTRAP_SHARED_STORE_ENV) != "1":
        raise RuntimeError("torchrun shared rendezvous store opt-out is absent")
    torch_distribution_version = importlib.metadata.version("torch")
    torch_version = str(torch.__version__)
    if not is_registered_torch_version_pair(
        torch_distribution_version,
        torch_version,
    ):
        raise RuntimeError(
            "torchrun bootstrap requires a registered torch distribution/runtime pair"
        )
    if c10d_rendezvous_backend.TCPStore is not torch.distributed.TCPStore:
        raise RuntimeError("torchrun c10d TCPStore reference is not pristine")
    create_tcp_store = c10d_rendezvous_backend._create_tcp_store
    if (
        create_tcp_store.__module__ != c10d_rendezvous_backend.__name__
        or create_tcp_store.__name__ != "_create_tcp_store"
    ):
        raise RuntimeError("torchrun c10d store creator is not pristine")
    source = inspect.getsource(create_tcp_store).encode("utf-8")
    if (
        len(source) != 2195
        or hashlib.sha256(source).hexdigest() != EXPECTED_C10D_CREATE_TCP_STORE_SHA256
    ):
        raise RuntimeError("torchrun c10d TCPStore source is not registered")
    return torch_distribution_version, torch_version


def _tcp_store_without_libuv(*args, **kwargs):
    caller = sys._getframe(1)
    if (
        _C10D_RENDEZVOUS_BACKEND._create_tcp_store is not _ORIGINAL_CREATE_TCP_STORE
        or caller.f_code is not _ORIGINAL_CREATE_TCP_STORE.__code__
    ):
        raise RuntimeError(
            "torchrun c10d TCPStore call did not originate from _create_tcp_store"
        )
    if "use_libuv" in kwargs and kwargs["use_libuv"] is not False:
        raise RuntimeError("torchrun c10d requested an unregistered libuv mode")
    kwargs["use_libuv"] = False
    store = _ORIGINAL_TCP_STORE(*args, **kwargs)
    del caller
    payload = bootstrap_attestation_payload(
        torch_distribution_version=_TORCH_DISTRIBUTION_VERSION,
        torch_version=_TORCH_VERSION,
    )
    raw = (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    with _ATTESTATION_PATH.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    return store


def _is_torchrun_worker() -> bool:
    keys = frozenset({"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE"})
    present = keys.intersection(os.environ)
    if present and present != keys:
        raise RuntimeError("torchrun worker coordinates are incomplete")
    return present == keys


def _install_agent_bootstrap() -> None:
    global _ATTESTATION_PATH
    global _C10D_RENDEZVOUS_BACKEND
    global _ORIGINAL_CREATE_TCP_STORE
    global _ORIGINAL_TCP_STORE
    global _TORCH_DISTRIBUTION_VERSION
    global _TORCH_VERSION

    import torch
    from torch.distributed.elastic.rendezvous import c10d_rendezvous_backend

    _ATTESTATION_PATH = _attestation_path()
    (
        _TORCH_DISTRIBUTION_VERSION,
        _TORCH_VERSION,
    ) = _guard_exact_source(torch, c10d_rendezvous_backend)
    _ORIGINAL_TCP_STORE = c10d_rendezvous_backend.TCPStore
    _ORIGINAL_CREATE_TCP_STORE = c10d_rendezvous_backend._create_tcp_store
    _C10D_RENDEZVOUS_BACKEND = c10d_rendezvous_backend
    c10d_rendezvous_backend.TCPStore = _tcp_store_without_libuv


if not _is_torchrun_worker():
    _install_agent_bootstrap()
