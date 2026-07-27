from __future__ import annotations

import pytest

from dcp_invariant.canonical import (
    canonical_json,
    exact_json_equal,
    strict_json_loads,
)


def test_canonical_json_is_compact_and_sorted() -> None:
    assert canonical_json({"b": 1, "a": [True, None]}) == '{"a":[true,null],"b":1}'


@pytest.mark.parametrize(
    "payload",
    [
        '{"a":1,"a":2}',
        '{"a":NaN}',
        '{"a":Infinity}',
        '{"a":-Infinity}',
    ],
)
def test_strict_json_rejects_nonstandard_or_duplicate_values(payload: str) -> None:
    with pytest.raises(ValueError):
        strict_json_loads(payload)


def test_exact_json_equality_rejects_bool_integer_coercion() -> None:
    assert not exact_json_equal({"value": True}, {"value": 1})
