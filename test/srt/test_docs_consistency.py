from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_docs_mention_protobuf_env():
    text = (ROOT / "docs" / "protobuf.md").read_text(encoding="utf-8")
    assert "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=upb" in text
    assert "call_soon_threadsafe" in text
    assert "zero-copy" in text or "零拷贝" in text
