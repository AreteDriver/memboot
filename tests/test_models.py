"""Tests for memboot.models."""

from __future__ import annotations

from memboot.models import (
    Chunk,
    ChunkType,
    MembootConfig,
    Memory,
    MemoryType,
    ProjectInfo,
    SearchResult,
)


class TestChunkType:
    def test_all_values(self):
        expected = {
            "function",
            "class",
            "method",
            "module",
            "markdown_section",
            "yaml_key",
            "json_key",
            "window",
        }
        assert {ct.value for ct in ChunkType} == expected

    def test_from_string(self):
        assert ChunkType("function") == ChunkType.FUNCTION
        assert ChunkType("window") == ChunkType.WINDOW


class TestMemoryType:
    def test_all_values(self):
        expected = {"decision", "note", "observation", "pattern"}
        assert {mt.value for mt in MemoryType} == expected

    def test_from_string(self):
        assert MemoryType("note") == MemoryType.NOTE
        assert MemoryType("decision") == MemoryType.DECISION


class TestChunk:
    def test_creation(self, sample_chunk):
        assert sample_chunk.id == "chunk-001"
        assert sample_chunk.content == "def foo(): pass"
        assert sample_chunk.source_file == "src/foo.py"
        assert sample_chunk.chunk_type == ChunkType.FUNCTION

    def test_defaults(self):
        chunk = Chunk(
            id="c1",
            content="hello",
            source_file="test.py",
            chunk_type=ChunkType.WINDOW,
        )
        assert chunk.start_line is None
        assert chunk.end_line is None
        assert chunk.embedding is None
        assert chunk.created_at  # auto-generated

    def test_created_at_auto(self):
        c1 = Chunk(id="a", content="x", source_file="f", chunk_type=ChunkType.WINDOW)
        c2 = Chunk(id="b", content="y", source_file="f", chunk_type=ChunkType.WINDOW)
        assert c1.created_at  # non-empty
        assert c2.created_at  # non-empty

    def test_with_embedding(self):
        chunk = Chunk(
            id="c1",
            content="test",
            source_file="t.py",
            chunk_type=ChunkType.FUNCTION,
            embedding=[0.1, 0.2, 0.3],
        )
        assert chunk.embedding == [0.1, 0.2, 0.3]


class TestMemory:
    def test_creation(self, sample_memory):
        assert sample_memory.id == "mem-001"
        assert sample_memory.memory_type == MemoryType.PATTERN
        assert sample_memory.tags == ["testing", "patterns"]

    def test_defaults(self):
        mem = Memory(
            id="m1",
            content="A note",
            memory_type=MemoryType.NOTE,
        )
        assert mem.tags == []
        assert mem.embedding is None
        assert mem.created_at

    def test_with_all_fields(self):
        mem = Memory(
            id="m2",
            content="Decision made",
            memory_type=MemoryType.DECISION,
            embedding=[0.5, 0.6],
            tags=["arch"],
            created_at="2025-01-01T00:00:00Z",
        )
        assert mem.created_at == "2025-01-01T00:00:00Z"
        assert mem.embedding == [0.5, 0.6]


class TestSearchResult:
    def test_creation(self):
        result = SearchResult(
            content="def foo(): pass",
            source="src/foo.py",
            score=0.95,
            chunk_type=ChunkType.FUNCTION,
            start_line=1,
            end_line=3,
        )
        assert result.score == 0.95
        assert result.chunk_type == ChunkType.FUNCTION

    def test_optional_fields(self):
        result = SearchResult(content="text", source="file.txt", score=0.5)
        assert result.chunk_type is None
        assert result.start_line is None
        assert result.end_line is None


class TestProjectInfo:
    def test_creation(self):
        info = ProjectInfo(
            project_path="/tmp/proj",
            project_hash="abc123",
            db_path="/tmp/db.sqlite",
            chunk_count=42,
        )
        assert info.chunk_count == 42

    def test_defaults(self):
        info = ProjectInfo(
            project_path="/tmp/proj",
            project_hash="abc123",
            db_path="/tmp/db.sqlite",
        )
        assert info.chunk_count == 0
        assert info.memory_count == 0
        assert info.last_indexed is None
        assert info.embedding_dim == 0
        assert info.embedding_backend == "tfidf"


class TestMembootConfig:
    def test_defaults(self):
        config = MembootConfig()
        assert config.max_chunk_tokens == 500
        assert config.overlap_tokens == 50
        assert config.embedding_backend == "tfidf"
        assert config.max_features == 512
        assert ".py" in config.file_extensions
        assert "__pycache__" in config.ignore_patterns

    def test_custom_values(self):
        config = MembootConfig(
            max_chunk_tokens=1000,
            overlap_tokens=100,
            embedding_backend="sentence-transformers",
            file_extensions=[".py", ".rs"],
        )
        assert config.max_chunk_tokens == 1000
        assert config.file_extensions == [".py", ".rs"]

    def test_default_extensions_list(self):
        config = MembootConfig()
        assert ".md" in config.file_extensions
        assert ".yaml" in config.file_extensions
        assert ".yml" in config.file_extensions
        assert ".json" in config.file_extensions
        assert ".txt" in config.file_extensions

    def test_default_ignore_patterns(self):
        config = MembootConfig()
        assert ".git" in config.ignore_patterns
        assert "node_modules" in config.ignore_patterns
        assert ".venv" in config.ignore_patterns
