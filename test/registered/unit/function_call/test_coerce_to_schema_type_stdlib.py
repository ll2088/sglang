"""Standard-library-only regression tests for ``coerce_to_schema_type``.

These tests do not import ``sglang`` (and therefore do not require the full
serving dependency set to be installed), so they can run on any interpreter
that can import the pure helper module. They pin the int-first ``number``
path so that large integer tool arguments are never routed through ``float``.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[5]
UTILS_PATH = REPO_ROOT / "python" / "sglang" / "srt" / "function_call" / "utils.py"


pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 10),
    reason="sglang dependencies support Python 3.10+",
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

IEEE754_UNSAFE_INTEGERS = [
    2**53 + 1,
    -(2**53 + 1),
    9007199254740993,
    1234567890123456789,
    2**63 - 1,
    -(2**63),
]

NUMBER_SCHEMA_TYPES = ["number", "double", ["number", "string"]]


def _load_coerce():
    """Load ``coerce_to_schema_type`` without importing the sglang package.

    ``utils.py`` imports ``sglang.srt.entrypoints.openai.protocol`` at module
    scope, so a plain ``importlib`` load would still pull in the serving stack.
    Instead, extract the function source with ``ast`` and exec it in an
    isolated namespace. This keeps the test dependency-free while exercising
    the exact production source text.
    """
    import ast

    source = UTILS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "coerce_to_schema_type":
            module = ast.Module(body=[node], type_ignores=[])
            namespace = {}
            exec(compile(module, str(UTILS_PATH), "exec"), namespace)
            return namespace["coerce_to_schema_type"]
    raise AssertionError("coerce_to_schema_type not found in utils.py")


coerce_to_schema_type = _load_coerce()


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


@pytest.mark.parametrize("value", [2**63 - 1, -(2**63), 2**53 + 1, -(2**53 + 1), 0])
@pytest.mark.parametrize("schema_type", NUMBER_SCHEMA_TYPES)
def test_int64_boundaries_preserved(value, schema_type):
    result = coerce_to_schema_type(str(value), schema_type)
    assert isinstance(result, int), (value, schema_type, result)
    assert result == value, (value, schema_type, result)
    assert json.loads(json.dumps(result)) == value
    if value != 0:
        assert result != int(float(value)), (value, schema_type, result)


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
