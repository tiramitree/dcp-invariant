"""Registered deterministic training state and typed digest helpers."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from .canonical import canonical_json, sha256_json

STATE_CONTRACT_SCHEMA = "dcp-invariant-state-contract-v1"
NORMALIZED_VALUE_SCHEMA = "dcp-invariant-normalized-value-v1"
MODEL_SHAPE = (2, 3)
BIAS_SHAPE = (2,)
GLOBAL_BATCH_SHAPE = (4, 3)
GLOBAL_TARGET_SHAPE = (4, 2)
GENERATOR_SEED = 20260728
LEARNING_RATE = 0.125
MOMENTUM = 0.5

_INITIAL_WEIGHT = (
    (0.25, -0.5, 0.75),
    (-0.125, 0.375, 0.625),
)
_INITIAL_BIAS = (0.0625, -0.1875)
_GLOBAL_BATCHES = (
    (
        (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
        ),
        (
            (0.5, -0.5),
            (0.25, 0.75),
            (-0.25, 0.125),
            (0.375, -0.375),
        ),
    ),
    (
        (
            (2.0, 0.0, 0.0),
            (0.0, 2.0, 0.0),
            (0.0, 0.0, 2.0),
            (1.0, 1.0, 0.0),
        ),
        (
            (0.75, -0.25),
            (-0.5, 0.5),
            (0.125, 0.625),
            (-0.375, 0.25),
        ),
    ),
)


class StateContractError(ValueError):
    """A value is outside the registered deterministic state contract."""


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().cpu().contiguous()
    return tensor.numpy().tobytes(order="C")


def tensor_record(value: torch.Tensor) -> dict[str, Any]:
    """Describe tensor identity without publishing its raw values."""

    if value.layout is not torch.strided:
        raise StateContractError("only strided tensors are registered")
    raw = _tensor_bytes(value)
    return {
        "byte_length": len(raw),
        "data_sha256": hashlib.sha256(raw).hexdigest(),
        "dtype": str(value.dtype).removeprefix("torch."),
        "shape": list(value.shape),
    }


def normalize_for_digest(value: Any) -> dict[str, Any]:
    """Normalize values with explicit type tags before canonical hashing."""

    if value is None:
        return {"kind": "none"}
    if type(value) is bool:
        return {"kind": "bool", "value": value}
    if type(value) is int:
        return {"kind": "int", "value": str(value)}
    if type(value) is float:
        if not math.isfinite(value):
            raise StateContractError("non-finite floats are not registered")
        return {"kind": "float", "value": value.hex()}
    if type(value) is str:
        return {"kind": "string", "value": value}
    if isinstance(value, torch.Tensor):
        return {"kind": "tensor", "value": tensor_record(value)}
    if isinstance(value, Mapping):
        entries = [
            {
                "key": normalize_for_digest(key),
                "value": normalize_for_digest(item),
            }
            for key, item in value.items()
        ]
        entries.sort(key=lambda entry: canonical_json(entry["key"]))
        return {"entries": entries, "kind": "mapping"}
    if isinstance(value, tuple):
        return {
            "items": [normalize_for_digest(item) for item in value],
            "kind": "tuple",
        }
    if isinstance(value, list):
        return {
            "items": [normalize_for_digest(item) for item in value],
            "kind": "list",
        }
    raise StateContractError(f"unregistered state value type: {type(value).__name__}")


def normalized_digest(value: Any) -> str:
    return sha256_json(
        {
            "normalization_schema": NORMALIZED_VALUE_SCHEMA,
            "value": normalize_for_digest(value),
        }
    )


def state_digests(
    *,
    model_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    generator_state: torch.Tensor,
    cursor: int,
) -> dict[str, str]:
    """Return component and aggregate hashes for the complete restart state."""

    if type(cursor) is not int or cursor < 0 or cursor > len(_GLOBAL_BATCHES):
        raise StateContractError("cursor is outside the registered batch sequence")
    normalized = {
        "cursor": normalize_for_digest(cursor),
        "generator": normalize_for_digest(generator_state),
        "model": normalize_for_digest(model_state),
        "optimizer": normalize_for_digest(optimizer_state),
    }
    return {
        "cursor_sha256": normalized_digest(cursor),
        "model_sha256": normalized_digest(model_state),
        "optimizer_sha256": normalized_digest(optimizer_state),
        "rng_sha256": normalized_digest(generator_state),
        "state_sha256": sha256_json(
            {
                "normalization_schema": NORMALIZED_VALUE_SCHEMA,
                "state": normalized,
            }
        ),
    }


def build_model() -> torch.nn.Linear:
    """Create the registered float64 Linear(3 -> 2) fixture."""

    with torch.random.fork_rng(devices=[]):
        model = torch.nn.Linear(3, 2, bias=True, device="cpu", dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(_INITIAL_WEIGHT, dtype=torch.float64))
        model.bias.copy_(torch.tensor(_INITIAL_BIAS, dtype=torch.float64))
    return model


def build_optimizer(
    parameters: Sequence[torch.nn.Parameter] | Any,
) -> torch.optim.SGD:
    return torch.optim.SGD(
        parameters,
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
        maximize=False,
        foreach=False,
        differentiable=False,
    )


def build_generator() -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(GENERATOR_SEED)
    return generator


def canonical_global_batches() -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    """Return fresh tensors for the two registered four-sample batches."""

    batches = []
    for inputs, targets in _GLOBAL_BATCHES:
        batches.append(
            (
                torch.tensor(inputs, dtype=torch.float64),
                torch.tensor(targets, dtype=torch.float64),
            )
        )
    return tuple(batches)  # type: ignore[return-value]


def state_contract() -> dict[str, Any]:
    batches = canonical_global_batches()
    model = build_model()
    return {
        "batch_inputs": [tensor_record(inputs) for inputs, _ in batches],
        "batch_targets": [tensor_record(targets) for _, targets in batches],
        "generator_seed": GENERATOR_SEED,
        "jitter": {
            "distribution": "torch.randint-0-or-1",
            "global_shape": list(GLOBAL_TARGET_SHAPE),
            "scale_hex": (0.0625).hex(),
        },
        "model": {
            "bias": tensor_record(model.bias),
            "type": "torch.nn.Linear",
            "weight": tensor_record(model.weight),
        },
        "optimizer": {
            "dampening_hex": (0.0).hex(),
            "learning_rate_hex": float(LEARNING_RATE).hex(),
            "momentum_hex": float(MOMENTUM).hex(),
            "nesterov": False,
            "type": "torch.optim.SGD",
            "weight_decay_hex": (0.0).hex(),
        },
        "state_contract_schema": STATE_CONTRACT_SCHEMA,
        "training": {
            "loss": "mean-squared-error-mean",
            "supported_world_sizes": [1, 2],
        },
    }


def state_contract_digest() -> str:
    return sha256_json(state_contract())
