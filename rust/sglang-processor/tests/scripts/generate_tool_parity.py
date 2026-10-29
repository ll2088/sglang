"""Fill tests/fixtures/tool_parity/*.json with SGLang's tool-call parsing.

Each fixture names a `--tool-call-parser`, the request's tools and chunked
outputs; this records `FunctionCallParser`'s streaming steps (the last chunk
also flushed, as `serving_chat` does at finish) and its one-shot parse.

Run from rust/sglang-processor in a SGLang Python environment:
    python tests/scripts/generate_tool_parity.py
"""

import json
import sys
from pathlib import Path

# Record this checkout's SGLang, not whichever one is installed.
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "python"))

from fixture_json import dump

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/tool_parity"

# Large integers that exceed the IEEE-754 exact range. JSON Schema ``number``
# accepts integers, so both parsers must preserve them exactly instead of
# routing them through ``float``.
LARGE_INTEGER_VALUES = [
    2**53 + 1,
    -(2**53 + 1),
    9007199254740993,
    1234567890123456789,
    2**63 - 1,
    -(2**63),
    0,
]


def large_integer_tool():
    return {
        "type": "function",
        "function": {
            "name": "lookup",
            "description": "Look up a record by numeric limit.",
            "parameters": {
                "type": "object",
                "properties": {"limit": {"type": "number"}},
                "required": ["limit"],
            },
        },
    }


def large_integer_inputs():
    return [
        [
            '{"name": "lookup", "parameters": {"limit": %d}}' % value
            for value in LARGE_INTEGER_VALUES
        ]
    ]


def calls(items):
    return [[c.tool_index, c.name, c.parameters] for c in items]


def parse(parser_name, tools, chunks):
    streaming = FunctionCallParser(tools, parser_name)
    steps = []
    for i, chunk in enumerate(chunks):
        normal, found = streaming.parse_stream_chunk(chunk)
        if i == len(chunks) - 1:
            end_normal, end_found = streaming.parse_stream_end()
            normal, found = (normal or "") + end_normal, list(found) + end_found
        steps.append([normal or "", calls(found)])
    normal, found = FunctionCallParser(tools, parser_name).parse_non_stream(
        "".join(chunks)
    )
    return {"steps": steps, "unary": [normal, calls(found)]}


for path in sorted(FIXTURES.glob("*.json")):
    fixture = json.loads(path.read_text())
    tools = [Tool(**tool) for tool in fixture["tools"]]
    fixture["cases"] = [
        {"chunks": chunks, **parse(fixture["parser"], tools, chunks)}
        for chunks in fixture["inputs"]
    ]
    path.write_text(dump(fixture))

# Record a dedicated large-integer fixture for every parser that supports the
# generic JSON tool-call format, so the Rust parity test can assert exact
# integer preservation across languages.
for path in sorted(FIXTURES.glob("*.json")):
    fixture = json.loads(path.read_text())
    if fixture.get("parser") != "json":
        continue
    large_fixture = {
        "parser": fixture["parser"],
        "tools": [large_integer_tool()],
        "inputs": large_integer_inputs(),
    }
    tools = [Tool(**tool) for tool in large_fixture["tools"]]
    large_fixture["cases"] = [
        {"chunks": chunks, **parse(large_fixture["parser"], tools, chunks)}
        for chunks in large_fixture["inputs"]
    ]
    (FIXTURES / "large_integer_number.json").write_text(dump(large_fixture))
