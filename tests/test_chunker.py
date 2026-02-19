"""Tests for memboot.chunker."""

from __future__ import annotations

from pathlib import Path

import pytest

from memboot.chunker import (
    ChunkResult,
    _chunk_json,
    _chunk_markdown,
    _chunk_python,
    _chunk_window,
    _chunk_yaml,
    chunk_file,
)
from memboot.exceptions import ChunkError
from memboot.models import ChunkType, MembootConfig


@pytest.fixture
def config() -> MembootConfig:
    return MembootConfig()


class TestChunkFile:
    def test_dispatch_python(self, tmp_path: Path, config: MembootConfig):
        f = tmp_path / "test.py"
        f.write_text("def foo():\n    pass\n")
        chunks = chunk_file(f, config)
        assert len(chunks) >= 1
        assert any(c.chunk_type == ChunkType.FUNCTION for c in chunks)

    def test_dispatch_markdown(self, tmp_path: Path, config: MembootConfig):
        f = tmp_path / "test.md"
        f.write_text("# Header\n\nContent here.\n")
        chunks = chunk_file(f, config)
        assert len(chunks) >= 1
        assert any(c.chunk_type == ChunkType.MARKDOWN_SECTION for c in chunks)

    def test_dispatch_yaml(self, tmp_path: Path, config: MembootConfig):
        f = tmp_path / "test.yaml"
        f.write_text("key1: value1\nkey2: value2\n")
        chunks = chunk_file(f, config)
        assert len(chunks) >= 1
        assert any(c.chunk_type == ChunkType.YAML_KEY for c in chunks)

    def test_dispatch_json(self, tmp_path: Path, config: MembootConfig):
        f = tmp_path / "test.json"
        f.write_text('{"key": "value"}\n')
        chunks = chunk_file(f, config)
        assert len(chunks) >= 1
        assert any(c.chunk_type == ChunkType.JSON_KEY for c in chunks)

    def test_dispatch_txt_window(self, tmp_path: Path, config: MembootConfig):
        f = tmp_path / "test.txt"
        f.write_text("Some text content.\n")
        chunks = chunk_file(f, config)
        assert len(chunks) >= 1
        assert any(c.chunk_type == ChunkType.WINDOW for c in chunks)

    def test_empty_file(self, tmp_path: Path, config: MembootConfig):
        f = tmp_path / "empty.py"
        f.write_text("")
        chunks = chunk_file(f, config)
        assert chunks == []

    def test_whitespace_only_file(self, tmp_path: Path, config: MembootConfig):
        f = tmp_path / "blank.py"
        f.write_text("   \n\n  \n")
        chunks = chunk_file(f, config)
        assert chunks == []

    def test_unreadable_file(self, tmp_path: Path, config: MembootConfig):
        f = tmp_path / "nope.py"
        # File doesn't exist
        with pytest.raises(ChunkError, match="Cannot read"):
            chunk_file(f, config)


class TestChunkPython:
    def test_function_extraction(self, config: MembootConfig):
        code = 'def hello():\n    """Say hello."""\n    return "hello"\n'
        chunks = _chunk_python(code, config)
        assert len(chunks) >= 1
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(func_chunks) == 1
        assert func_chunks[0].metadata["name"] == "hello"

    def test_class_extraction(self, config: MembootConfig):
        code = "class Foo:\n    x = 1\n    y = 2\n"
        chunks = _chunk_python(code, config)
        class_chunks = [c for c in chunks if c.chunk_type == ChunkType.CLASS]
        assert len(class_chunks) == 1
        assert class_chunks[0].metadata["name"] == "Foo"

    def test_class_method_splitting(self, config: MembootConfig):
        # Create a class large enough to trigger method splitting
        methods = []
        for i in range(20):
            methods.append(f"    def method_{i}(self):\n" + "        pass\n" * 50)
        code = "class Big:\n" + "\n".join(methods)
        chunks = _chunk_python(code, config)
        method_chunks = [c for c in chunks if c.chunk_type == ChunkType.METHOD]
        assert len(method_chunks) > 0

    def test_module_level_code(self, config: MembootConfig):
        code = "import os\nimport sys\n\nX = 42\n\ndef foo():\n    pass\n"
        chunks = _chunk_python(code, config)
        module_chunks = [c for c in chunks if c.chunk_type == ChunkType.MODULE]
        assert len(module_chunks) >= 1

    def test_syntax_error_fallback(self, config: MembootConfig):
        code = "def broken(:\n    pass\n"
        chunks = _chunk_python(code, config)
        # Should fallback to window chunking
        assert len(chunks) >= 1
        assert any(c.chunk_type == ChunkType.WINDOW for c in chunks)

    def test_async_function(self, config: MembootConfig):
        code = "async def fetch():\n    return await get_data()\n"
        chunks = _chunk_python(code, config)
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(func_chunks) == 1
        assert func_chunks[0].metadata["name"] == "fetch"

    def test_line_numbers(self, config: MembootConfig):
        code = "\n\ndef hello():\n    pass\n"
        chunks = _chunk_python(code, config)
        func_chunks = [c for c in chunks if c.chunk_type == ChunkType.FUNCTION]
        assert len(func_chunks) == 1
        assert func_chunks[0].start_line == 3


class TestChunkMarkdown:
    def test_header_splitting(self, config: MembootConfig):
        md = "# Title\n\nIntro.\n\n## Section 1\n\nContent 1.\n\n## Section 2\n\nContent 2.\n"
        chunks = _chunk_markdown(md, config)
        assert len(chunks) == 3  # Title + Section 1 + Section 2

    def test_preamble_before_header(self, config: MembootConfig):
        md = "Some preamble.\n\n# Title\n\nContent.\n"
        chunks = _chunk_markdown(md, config)
        assert len(chunks) == 2
        assert chunks[0].metadata["header"] == "preamble"

    def test_no_headers_fallback(self, config: MembootConfig):
        md = "Just some plain text.\nNo headers here.\n"
        chunks = _chunk_markdown(md, config)
        assert len(chunks) >= 1
        assert any(c.chunk_type == ChunkType.WINDOW for c in chunks)

    def test_header_metadata(self, config: MembootConfig):
        md = "# My Title\n\nContent.\n"
        chunks = _chunk_markdown(md, config)
        assert chunks[0].metadata["header"] == "My Title"


class TestChunkYaml:
    def test_top_level_keys(self, config: MembootConfig):
        yml = "name: test\nversion: 1\nsettings:\n  debug: true\n"
        chunks = _chunk_yaml(yml, config)
        assert len(chunks) == 3
        keys = [c.metadata["key"] for c in chunks]
        assert "name" in keys
        assert "version" in keys
        assert "settings" in keys

    def test_invalid_yaml_fallback(self, config: MembootConfig):
        yml = "{{invalid: yaml: [}"
        chunks = _chunk_yaml(yml, config)
        assert len(chunks) >= 1
        assert any(c.chunk_type == ChunkType.WINDOW for c in chunks)

    def test_non_dict_yaml_fallback(self, config: MembootConfig):
        yml = "- item1\n- item2\n- item3\n"
        chunks = _chunk_yaml(yml, config)
        assert len(chunks) >= 1
        assert any(c.chunk_type == ChunkType.WINDOW for c in chunks)


class TestChunkJson:
    def test_dict_keys(self, config: MembootConfig):
        jsn = '{"key1": "val1", "key2": [1, 2]}'
        chunks = _chunk_json(jsn, config)
        assert len(chunks) == 2
        keys = [c.metadata["key"] for c in chunks]
        assert "key1" in keys
        assert "key2" in keys

    def test_non_dict_fallback(self, config: MembootConfig):
        jsn = "[1, 2, 3]"
        chunks = _chunk_json(jsn, config)
        assert len(chunks) >= 1
        assert any(c.chunk_type == ChunkType.WINDOW for c in chunks)

    def test_invalid_json_fallback(self, config: MembootConfig):
        jsn = "{broken json"
        chunks = _chunk_json(jsn, config)
        assert len(chunks) >= 1
        assert any(c.chunk_type == ChunkType.WINDOW for c in chunks)

    def test_empty_dict(self, config: MembootConfig):
        jsn = "{}"
        chunks = _chunk_json(jsn, config)
        # Empty dict → no keys → fallback to window
        assert len(chunks) >= 1


class TestChunkWindow:
    def test_basic_chunking(self, config: MembootConfig):
        text = "Hello world. " * 100
        chunks = _chunk_window(text, config)
        assert len(chunks) >= 1
        assert all(c.chunk_type == ChunkType.WINDOW for c in chunks)

    def test_overlap(self):
        config = MembootConfig(max_chunk_tokens=25, overlap_tokens=5)
        text = "word " * 200  # ~1000 chars, chunks of ~100 chars with 20 char overlap
        chunks = _chunk_window(text, config)
        assert len(chunks) > 1

    def test_single_chunk_small_text(self, config: MembootConfig):
        text = "Short text."
        chunks = _chunk_window(text, config)
        assert len(chunks) == 1

    def test_line_numbers(self, config: MembootConfig):
        text = "Line 1\nLine 2\nLine 3\n"
        chunks = _chunk_window(text, config)
        assert chunks[0].start_line >= 1


class TestChunkResult:
    def test_slots(self):
        cr = ChunkResult(
            content="test",
            chunk_type=ChunkType.WINDOW,
            start_line=1,
            end_line=5,
        )
        assert cr.content == "test"
        assert cr.metadata == {}

    def test_with_metadata(self):
        cr = ChunkResult(
            content="test",
            chunk_type=ChunkType.FUNCTION,
            start_line=1,
            end_line=5,
            metadata={"name": "foo"},
        )
        assert cr.metadata["name"] == "foo"
