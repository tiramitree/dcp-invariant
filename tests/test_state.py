from __future__ import annotations

import torch

from dcp_invariant.state import (
    BIAS_SHAPE,
    GLOBAL_BATCH_SHAPE,
    GLOBAL_TARGET_SHAPE,
    MODEL_SHAPE,
    build_generator,
    build_model,
    canonical_global_batches,
    normalize_for_digest,
    normalized_digest,
    state_contract_digest,
    tensor_record,
)


def test_registered_fixture_shapes_and_dtypes() -> None:
    model = build_model()
    assert tuple(model.weight.shape) == MODEL_SHAPE
    assert tuple(model.bias.shape) == BIAS_SHAPE
    assert model.weight.dtype is torch.float64
    batches = canonical_global_batches()
    assert len(batches) == 2
    assert all(tuple(inputs.shape) == GLOBAL_BATCH_SHAPE for inputs, _ in batches)
    assert all(tuple(targets.shape) == GLOBAL_TARGET_SHAPE for _, targets in batches)
    assert all(inputs.dtype is torch.float64 for inputs, _ in batches)
    assert all(targets.dtype is torch.float64 for _, targets in batches)


def test_fixture_builders_return_independent_identical_state() -> None:
    first_model = build_model()
    second_model = build_model()
    assert torch.equal(first_model.weight, second_model.weight)
    assert torch.equal(first_model.bias, second_model.bias)
    first_generator = build_generator()
    second_generator = build_generator()
    assert torch.equal(first_generator.get_state(), second_generator.get_state())
    first_batches = canonical_global_batches()
    first_batches[0][0][0, 0] = 99.0
    assert canonical_global_batches()[0][0][0, 0].item() == 1.0


def test_normalization_distinguishes_bool_and_integer() -> None:
    assert normalize_for_digest(True) != normalize_for_digest(1)
    assert normalized_digest(True) != normalized_digest(1)


def test_tensor_digest_binds_dtype_shape_and_bytes() -> None:
    base = torch.tensor([1, 2, 3, 4], dtype=torch.int64)
    reshaped = base.reshape(2, 2)
    as_float = base.to(torch.float64)
    changed = torch.tensor([1, 2, 3, 5], dtype=torch.int64)
    records = [tensor_record(value) for value in (base, reshaped, as_float, changed)]
    assert records[0]["data_sha256"] == records[1]["data_sha256"]
    assert records[0]["shape"] != records[1]["shape"]
    assert records[0]["dtype"] != records[2]["dtype"]
    assert records[0]["data_sha256"] != records[3]["data_sha256"]
    digests = {
        normalized_digest(value) for value in (base, reshaped, as_float, changed)
    }
    assert len(digests) == 4


def test_state_contract_digest_is_stable_sha256() -> None:
    digest = state_contract_digest()
    assert len(digest) == 64
    assert digest == state_contract_digest()
