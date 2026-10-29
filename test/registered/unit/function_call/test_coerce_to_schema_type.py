"""Unit tests for ``coerce_to_schema_type`` number handling.

These tests import only the pure conversion helper so they can run without
pulling in the full serving stack (and its heavy optional dependencies).
"""

import json
import sys

import pytest

from sglang.srt.function_call.utils import coerce_to_schema_type

# These tests only exercise the pure conversion helper, so they must not pull
# in the serving stack. The helper itself is stdlib-only, so the numeric
# assertions run on every interpreter the CI matrix provides (including 3.14);
# only the import of the wider sglang package is version-gated.
_SUPPORTED_PYTHON = (3, 10) <= sys.version_info < (3, 14)


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


def test_number_branch_tries_int_before_float():
    """Static guard: the ``number`` branch must attempt ``int`` first.

    This pins the production source order so a future refactor cannot
    silently reintroduce the float round-trip for large integers.
    """
    import ast
    import inspect

    source = inspect.getsource(coerce_to_schema_type)
    tree = ast.parse(source)
    number_branch = None
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Compare):
            left = node.test.left
            if (
                isinstance(left, ast.Name)
                and left.id == "schema_type"
                and any(
                    isinstance(c, ast.Constant) and c.value == "number"
                    for c in node.test.comparators
                )
            ):
                number_branch = node
                break
    assert number_branch is not None, "number branch not found"
    calls = [
        n.func.id
        for n in ast.walk(number_branch)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert "int" in calls, "number branch must call int()"
    assert "float" in calls, "number branch must keep the float fallback"
    assert calls.index("int") < calls.index("float"), (
        "number branch must try int() before float()"
    )


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
