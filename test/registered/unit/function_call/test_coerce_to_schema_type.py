"""Unit tests for ``coerce_to_schema_type`` number handling.

These tests import only the pure conversion helper so they can run without
pulling in the full serving stack (and its heavy optional dependencies).
"""

import json
import sys

import pytest

from sglang.srt.function_call.utils import coerce_to_schema_type

# These tests only exercise the pure conversion helper, so they must not pull
# in the serving stack. They are also expected to run on the project's
# supported interpreters (3.10-3.12); a 3.14 interpreter breaks dependency
# resolution for the wider sglang package before pytest even starts.
pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10) or sys.version_info >= (3, 14),
    reason="sglang dependencies support Python 3.10-3.12; run on a supported interpreter",
)


LARGE_INTEGERS = [
    2**53 + 1,
    -(2**53 + 1),
    9007199254740993,
    1234567890123456789,
    2**63 - 1,
    -(2**63),
    0,
]

# Values that must never be routed through float: 2**53 + 1 is the first
# integer not exactly representable as an IEEE-754 double.
IEEE754_UNSAFE_INTEGERS = [
    2**53 + 1,
    -(2**53 + 1),
    9007199254740993,
    1234567890123456789,
    2**63 - 1,
    -(2**63),
]

NUMBER_SCHEMA_TYPES = ["number", "double", ["number", "string"]]


@pytest.mark.parametrize("value", LARGE_INTEGERS)
@pytest.mark.parametrize("schema_type", NUMBER_SCHEMA_TYPES)
def test_large_integer_preserved(value, schema_type):
    result = coerce_to_schema_type(str(value), schema_type)
    assert isinstance(result, int), (value, schema_type, result)
    assert result == value, (value, schema_type, result)
    assert json.loads(json.dumps(result)) == value


@pytest.mark.parametrize("value", IEEE754_UNSAFE_INTEGERS)
@pytest.mark.parametrize("schema_type", NUMBER_SCHEMA_TYPES)
def test_large_integer_not_rounded_through_float(value, schema_type):
    """Guard against the regression where ``number`` went through ``float``.

    ``float(value)`` would round these values, so asserting the exact int
    (and that it differs from the float round-trip) pins the int-first path.
    """
    result = coerce_to_schema_type(str(value), schema_type)
    assert isinstance(result, int), (value, schema_type, result)
    assert result == value, (value, schema_type, result)
    assert result != int(float(value)), (value, schema_type, result)


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
    # int64 边界与 2^53 边界不得经 float 中转。
    if value not in (0,):
        assert result != int(float(value)), (value, schema_type, result)


@pytest.mark.parametrize("value", [2**63 - 1, -(2**63)])
def test_large_integer_in_nested_structures(value):
    nested = {"items": [value, {"limit": value}], "limit": value}
    assert json.loads(json.dumps(nested)) == nested
    result = coerce_to_schema_type(str(value), "number")
    assert isinstance(result, int)
    assert result == value
    assert result != int(float(value))


@pytest.mark.parametrize("value", IEEE754_UNSAFE_INTEGERS)
def test_large_integer_in_nested_structures_exact(value):
    nested = {"items": [value, {"limit": value}], "limit": value}
    assert json.loads(json.dumps(nested)) == nested
    result = coerce_to_schema_type(str(value), "number")
    assert isinstance(result, int)
    assert result == value
    assert result != int(float(value))


def test_float_input_semantics_unchanged():
    assert coerce_to_schema_type(5.0, "number") == 5
    assert isinstance(coerce_to_schema_type(5.0, "number"), int)
    assert coerce_to_schema_type(3.14, "number") == 3.14
    assert coerce_to_schema_type(float("inf"), "number") == float("inf")


def test_string_schema_not_coerced_to_number():
    assert coerce_to_schema_type("123", "string") == "123"
    assert coerce_to_schema_type("123", "integer") == 123
