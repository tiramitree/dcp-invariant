from __future__ import annotations

import pytest

from dcp_invariant.elastic_contract import (
    BOOTSTRAP_SHARED_STORE_ENV,
    REGISTERED_MAX_RESTARTS,
    REGISTERED_WORLD_SIZE,
    ElasticContractError,
    failure_marker_payload,
    parse_elastic_environment,
)


def elastic_environment(*, rank: int = 0, restart_count: int = 0) -> dict[str, str]:
    return {
        "LOCAL_RANK": str(rank),
        "LOCAL_WORLD_SIZE": str(REGISTERED_WORLD_SIZE),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29501",
        "RANK": str(rank),
        BOOTSTRAP_SHARED_STORE_ENV: "1",
        "TORCHELASTIC_MAX_RESTARTS": str(REGISTERED_MAX_RESTARTS),
        "TORCHELASTIC_RESTART_COUNT": str(restart_count),
        "WORLD_SIZE": str(REGISTERED_WORLD_SIZE),
    }


@pytest.mark.parametrize(("rank", "restart_count"), [(0, 0), (1, 0), (0, 1), (1, 1)])
def test_fixed_elastic_coordinates_are_accepted(
    rank: int,
    restart_count: int,
) -> None:
    parsed = parse_elastic_environment(
        elastic_environment(rank=rank, restart_count=restart_count)
    )
    assert parsed.rank == rank
    assert parsed.local_rank == rank
    assert parsed.world_size == 2
    assert parsed.max_restarts == 1
    assert parsed.restart_count == restart_count


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("WORLD_SIZE", "1"),
        ("LOCAL_WORLD_SIZE", "3"),
        ("RANK", "2"),
        ("LOCAL_RANK", "1"),
        (BOOTSTRAP_SHARED_STORE_ENV, "0"),
        ("TORCHELASTIC_MAX_RESTARTS", "2"),
        ("TORCHELASTIC_RESTART_COUNT", "2"),
        ("MASTER_ADDR", "example.invalid"),
        ("MASTER_ADDR", "localhost"),
        ("MASTER_ADDR", "2130706433"),
        ("MASTER_ADDR", "192.0.2.1"),
        ("MASTER_PORT", "0"),
        ("MASTER_PORT", "not-decimal"),
    ],
)
def test_elastic_environment_is_fail_closed(field: str, value: str) -> None:
    environment = elastic_environment()
    environment[field] = value
    with pytest.raises(ElasticContractError):
        parse_elastic_environment(environment)


def test_numeric_ipv6_loopback_is_accepted() -> None:
    environment = elastic_environment()
    environment["MASTER_ADDR"] = "::1"
    parsed = parse_elastic_environment(environment)
    assert parsed.rank == 0


def test_non_string_master_address_is_rejected() -> None:
    environment = elastic_environment()
    environment["MASTER_ADDR"] = 2130706433
    with pytest.raises(ElasticContractError, match="not numeric"):
        parse_elastic_environment(environment)


def test_missing_elastic_environment_field_is_rejected() -> None:
    environment = elastic_environment()
    environment.pop("TORCHELASTIC_RESTART_COUNT")
    with pytest.raises(ElasticContractError, match="RESTART_COUNT"):
        parse_elastic_environment(environment)


def test_failure_payload_is_fixed_and_contains_no_machine_coordinate() -> None:
    assert failure_marker_payload() == {
        "injected_exit_code": 91,
        "injected_rank": 1,
        "injection_restart_count": 0,
        "marker_schema": "dcp-invariant-elastic-failure-v2",
        "world_size": 2,
    }
    assert not {
        "environment",
        "hostname",
        "path",
        "pid",
        "port",
        "username",
    }.intersection(failure_marker_payload())
