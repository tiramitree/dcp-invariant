"""PyTorch-free constants and environment validation for elastic evidence."""

from __future__ import annotations

import ipaddress
from collections.abc import Mapping
from dataclasses import dataclass

ELASTIC_REPORT_SCHEMA = "dcp-invariant-elastic-report-v2"
FAILURE_MARKER_SCHEMA = "dcp-invariant-elastic-failure-v2"
REGISTERED_FAILURE_EXIT_CODE = 91
REGISTERED_WORLD_SIZE = 2
REGISTERED_MAX_RESTARTS = 1
FAILURE_MARKER_NAME = ".elastic-failure.json"
LOAD_REPORT_DIRECTORY_NAME = "elastic-load"
CONTROL_REPORT_DIRECTORY_NAME = "elastic-control"
BOOTSTRAP_ATTESTATION_NAME = ".torchrun-bootstrap-attestation.json"
BOOTSTRAP_ATTESTATION_ENV = "DCP_INVARIANT_TORCHRUN_BOOTSTRAP_ATTESTATION"
BOOTSTRAP_SHARED_STORE_ENV = "TORCH_DISABLE_SHARE_RDZV_TCP_STORE"
BOOTSTRAP_ATTESTATION_SCHEMA = "dcp-invariant-torchrun-bootstrap-attestation-v3"
BOOTSTRAP_ID = "torch-2.11-c10d-call-verified-no-libuv-v3"
EXPECTED_C10D_CREATE_TCP_STORE_SHA256 = (
    "488aee8200995402157248d051fc337c9dd02b77dc460ddb9abd2b5bf22bc19f"
)
REGISTERED_TORCH_VERSION_PAIRS = frozenset(
    {
        ("2.11.0", "2.11.0+cpu"),
        ("2.11.0+cpu", "2.11.0+cpu"),
    }
)


class ElasticContractError(ValueError):
    """The fixed elastic coordinates are outside the public contract."""


@dataclass(frozen=True)
class ElasticEnvironment:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    restart_count: int
    max_restarts: int


def _exact_decimal(environment: Mapping[str, str], name: str) -> int:
    value = environment.get(name)
    if type(value) is not str or not value.isascii() or not value.isdecimal():
        raise ElasticContractError(f"{name} is not one decimal integer")
    return int(value)


def parse_elastic_environment(
    environment: Mapping[str, str],
) -> ElasticEnvironment:
    """Parse only fixed, non-secret elastic coordinates from the worker environment."""

    parsed = ElasticEnvironment(
        rank=_exact_decimal(environment, "RANK"),
        local_rank=_exact_decimal(environment, "LOCAL_RANK"),
        world_size=_exact_decimal(environment, "WORLD_SIZE"),
        local_world_size=_exact_decimal(environment, "LOCAL_WORLD_SIZE"),
        restart_count=_exact_decimal(environment, "TORCHELASTIC_RESTART_COUNT"),
        max_restarts=_exact_decimal(environment, "TORCHELASTIC_MAX_RESTARTS"),
    )
    if parsed.world_size != REGISTERED_WORLD_SIZE:
        raise ElasticContractError("elastic world size is not registered")
    if parsed.local_world_size != REGISTERED_WORLD_SIZE:
        raise ElasticContractError("elastic local world size is not registered")
    if parsed.rank not in {0, 1} or parsed.local_rank != parsed.rank:
        raise ElasticContractError("elastic rank coordinates are invalid")
    if parsed.max_restarts != REGISTERED_MAX_RESTARTS:
        raise ElasticContractError("elastic restart budget is not registered")
    if parsed.restart_count not in {0, 1}:
        raise ElasticContractError("elastic restart count is not registered")
    if environment.get(BOOTSTRAP_SHARED_STORE_ENV) != "1":
        raise ElasticContractError(
            "elastic shared rendezvous TCPStore opt-out is not registered"
        )

    master_address = environment.get("MASTER_ADDR")
    if type(master_address) is not str:
        raise ElasticContractError("elastic rendezvous is not numeric")
    try:
        parsed_address = ipaddress.ip_address(master_address)
    except ValueError as error:
        raise ElasticContractError("elastic rendezvous is not numeric") from error
    if not parsed_address.is_loopback:
        raise ElasticContractError("elastic rendezvous is not loopback")
    master_port = _exact_decimal(environment, "MASTER_PORT")
    if not 1 <= master_port <= 65535:
        raise ElasticContractError("elastic rendezvous port is invalid")
    return parsed


def failure_marker_payload() -> dict[str, object]:
    return {
        "injected_exit_code": REGISTERED_FAILURE_EXIT_CODE,
        "injected_rank": 1,
        "injection_restart_count": 0,
        "marker_schema": FAILURE_MARKER_SCHEMA,
        "world_size": REGISTERED_WORLD_SIZE,
    }


def is_registered_torch_version_pair(
    torch_distribution_version: object,
    torch_version: object,
) -> bool:
    return (
        type(torch_distribution_version) is str
        and type(torch_version) is str
        and (torch_distribution_version, torch_version)
        in REGISTERED_TORCH_VERSION_PAIRS
    )


def bootstrap_attestation_payload(
    *,
    torch_distribution_version: str,
    torch_version: str,
) -> dict[str, object]:
    if not is_registered_torch_version_pair(
        torch_distribution_version,
        torch_version,
    ):
        raise ElasticContractError(
            "torchrun bootstrap distribution/runtime pair is not registered"
        )
    return {
        "attestation_schema": BOOTSTRAP_ATTESTATION_SCHEMA,
        "backend_module": (
            "torch.distributed.elastic.rendezvous.c10d_rendezvous_backend"
        ),
        "bootstrap_id": BOOTSTRAP_ID,
        "create_tcp_store_call_verified": True,
        "forced_use_libuv": False,
        "shared_rendezvous_tcpstore_disabled": True,
        "source_sha256": EXPECTED_C10D_CREATE_TCP_STORE_SHA256,
        "tcpstore_created": True,
        "torch_distribution_version": torch_distribution_version,
        "torch_version": torch_version,
    }
