"""Unit tests for ``coerce_to_schema_type`` number handling.

These tests import only the pure conversion helper so they can run without
pulling in the full serving stack (and its heavy optional dependencies).
"""

import json

import pytest

from sglang.srt.function_call.utils import coerce_to_schema_type


LARGE_INTEGERS = [
    2**53 + 1,
    -(2**53 + 1),
    9007199254740993,
    1234567890123456789,
    2**63 - 1,
    -(2**63),
    0,
]

NUMBER_SCHEMA_TYPES = ["number", "double", ["number", "string"]]


@pytest.mark.parametrize("value", LARGE_INTEGERS)
@pytest.mark.parametrize("schema_type", NUMBER_SCHEMA_TYPES)
def test_large_integer_preserved(value, schema_type):
    result = coerce_to_schema_type(str(value), schema_type)
    assert isinstance(result, int), (value, schema_type, result)
    assert result == value, (value, schema_type, result)
    assert json.loads(json.dumps(result)) == value


@pytest.mark.parametrize("value", LARGE_INTEGERS)
def test_large_integer_preserved_as_int_input(value):
    result = coerce_to_schema_type(value, "number")
    assert isinstance(result, int)
    assert result == value


@pytest.mark.parametrize(
    "value,expected",
    [
        ("5.0", 5),
        ("1e3", 1000),
        ("3.14", 3.14),
        ("-2.5", -2.5),
        ("abc", "abc"),
    ],
)
def test_non_integer_number_semantics_unchanged(value, expected):
    result = coerce_to_schema_type(value, "number")
    assert result == expected
    if isinstance(expected, int):
        assert isinstance(result, int)


def test_non_finite_number_unchanged():
    assert coerce_to_schema_type("inf", "number") == float("inf")
    assert coerce_to_schema_type("-inf", "number") == float("-inf")


def test_bool_not_coerced_to_int():
    assert coerce_to_schema_type(True, "number") is True
    assert coerce_to_schema_type(False, "number") is False


def test_large_integer_nested_roundtrip():
    value = 2**63 - 1
    nested = {"items": [value, {"limit": value}]}
    assert json.loads(json.dumps(nested)) == nested
    assert coerce_to_schema_type(str(value), "number") == value


def test_large_negative_integer_nested_roundtrip():
    value = -(2**63)
    nested = {"items": [value, {"limit": value}]}
    assert json.loads(json.dumps(nested)) == nested
    assert coerce_to_schema_type(str(value), "number") == value


def test_integer_schema_still_parses_large_values():
    value = 2**63 - 1
    result = coerce_to_schema_type(str(value), "integer")
    assert isinstance(result, int)
    assert result == value


@pytest.mark.parametrize("value", [2**63 - 1, -(2**63), 2**53 + 1, -(2**53 + 1), 0])
@pytest.mark.parametrize("schema_type", NUMBER_SCHEMA_TYPES)
def test_int64_boundaries_preserved(value, schema_type):
    result = coerce_to_schema_type(str(value), schema_type)
    assert isinstance(result, int), (value, schema_type, result)
    assert result == value, (value, schema_type, result)
    assert json.loads(json.dumps(result)) == value


@pytest.mark.parametrize("value", [2**63 - 1, -(2**63)])
def test_large_integer_in_nested_structures(value):
    nested = {"items": [value, {"limit": value}], "limit": value}
    assert json.loads(json.dumps(nested)) == nested
    result = coerce_to_schema_type(str(value), "number")
    assert isinstance(result, int)
    assert result == value


def test_float_input_semantics_unchanged():
    assert coerce_to_schema_type(5.0, "number") == 5
    assert isinstance(coerce_to_schema_type(5.0, "number"), int)
    assert coerce_to_schema_type(3.14, "number") == 3.14
    assert coerce_to_schema_type(float("inf"), "number") == float("inf")


def test_string_schema_not_coerced_to_number():
    assert coerce_to_schema_type("123", "string") == "123"
    assert coerce_to_schema_type("123", "integer") == 123
