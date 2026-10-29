"""Standard-library-only regression tests for ``coerce_to_schema_type``.

These tests do not import ``sglang`` (and therefore do not require the full
serving dependency set to be installed), so they can run on any interpreter
that can import the pure helper module. They pin the int-first ``number``
path so that large integer tool arguments are never routed through ``float``.
"""

import ast
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[5]
UTILS_PATH = REPO_ROOT / "python" / "sglang" / "srt" / "function_call" / "utils.py"


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


@pytest.mark.parametrize("value", [2**63 - 1, -(2**63), 2**53 + 1, -(2**53 + 1), 0])
@pytest.mark.parametrize("schema_type", NUMBER_SCHEMA_TYPES)
def test_int64_boundaries_preserved(value, schema_type):
    result = coerce_to_schema_type(str(value), schema_type)
    assert isinstance(result, int), (value, schema_type, result)
    assert result == value, (value, schema_type, result)
    assert json.loads(json.dumps(result)) == value
    if value != 0:
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


@pytest.mark.parametrize("value", [2**63 - 1, -(2**63), 2**53 + 1, -(2**53 + 1), 0])
def test_int64_boundaries_in_nested_structures(value):
    nested = {"items": [value, {"limit": value}], "limit": value}
    assert json.loads(json.dumps(nested)) == nested
    result = coerce_to_schema_type(str(value), "number")
    assert isinstance(result, int)
    assert result == value
    if value != 0:
        assert result != int(float(value))


def test_float_input_semantics_unchanged():
    assert coerce_to_schema_type(5.0, "number") == 5
    assert isinstance(coerce_to_schema_type(5.0, "number"), int)
    assert coerce_to_schema_type(3.14, "number") == 3.14
    assert coerce_to_schema_type(float("inf"), "number") == float("inf")


def test_string_schema_not_coerced_to_number():
    assert coerce_to_schema_type("123", "string") == "123"
    assert coerce_to_schema_type("123", "integer") == 123


def test_number_branch_tries_int_before_float():
    """Static guard on the production source text.

    The ``number`` branch must call ``int()`` before ``float()`` so that
    large integer literals never round-trip through IEEE-754.
    """
    source = UTILS_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "coerce_to_schema_type"
    )
    number_branch = None
    for node in ast.walk(fn):
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
